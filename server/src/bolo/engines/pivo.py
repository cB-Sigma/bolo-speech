"""PiVo-1.4M (Empero Research, Apache-2.0) in process: four ONNX graphs on ONNX Runtime, no server, no PyTorch.

Two English voices (Pi, Vo) at 16 kHz from 1.4 M learned parameters: an acoustic model (``encoder.onnx`` predicts a
duration per phone, ``decoder.onnx`` 20 vocoder features per 10 ms frame, with a key/value cache across chunks) and a
FARGAN-derived vocoder (``conditioning.onnx`` once per frame, ``vocoder_step.onnx`` four times per frame for 40-sample
subframes, carrying its recurrent state), then de-emphasis. The text front end reads words in CMUdict (ARPAbet, the
model's ``pivo/resources/cmudict.dict``) after expanding numbers, money, ordinals and ISO dates. This module follows
PiVo's reference code (``pivo/text.py``, ``runtime.py``, ``phase.py``, ``audio.py`` at model revision
515d84fc3420da6ae138f1b81017fcbc24d622a8), with two changes for a server: a word CMUdict lacks is spelled letter by
letter instead of refusing the request (the reference's non-strict mode), and characters the model cannot say are
left out instead of refusing it. A text longer than the model takes (512 tokens, 3,000 frames) is cut after commas,
then between words.
"""

from __future__ import annotations

import json
import math
import re
import threading
import unicodedata
from collections.abc import Callable, Iterator
from datetime import date as calendar_date
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import FloatArray, LoadOptions, TTSModel, Voice
from bolo.engines.tiny_voca import shorter

CONSONANTS = [
    "B",
    "CH",
    "D",
    "DH",
    "F",
    "G",
    "HH",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
VOWELS = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
SPECIAL = ("<pad>", "<bos>", "<eos>", "<space>", ".", ",", "?", "!", ";", ":")
SYMBOLS = (*SPECIAL, *CONSONANTS, *(v + s for v in VOWELS for s in "012"))
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
MAX_TOKENS = 512
PREEMPHASIS = 0.85
_STATE_WIDTHS = (256, 160, 128, 128, 164)

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["zero", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORDINALS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
    "twelve": "twelfth",
}  # fmt: skip
_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_ABBREVIATIONS = {
    "Mr": "Mister", "Mrs": "Missus", "Dr": "Doctor", "Prof": "Professor", "St": "Saint", "vs": "versus",
    "etc": "et cetera",
}  # fmt: skip
_QUOTE_MAP: dict[str, str | int | None] = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": ",", "–": ",", "…": "..."}
_QUOTES = str.maketrans(_QUOTE_MAP)
_WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]")


def cardinal(number: int) -> str:
    """``number`` in English words (digit by digit from a quadrillion up)."""
    if number < 0:
        return "minus " + cardinal(-number)
    if number < 20:
        return _ONES[number]
    if number < 100:
        return _TENS[number // 10] + (" " + cardinal(number % 10) if number % 10 else "")
    if number >= 10**15:
        return " ".join(_ONES[int(d)] for d in str(number))
    for scale, label in ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if number >= scale:
            rest = number % scale
            return cardinal(number // scale) + " " + label + (" " + cardinal(rest) if rest else "")
    rest = number % 100
    return cardinal(number // 100) + " hundred" + (" " + cardinal(rest) if rest else "")


def ordinal(number: int) -> str:
    words = cardinal(number).split()
    last = words[-1]
    words[-1] = _ORDINALS.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return " ".join(words)


def _money(match: re.Match[str]) -> str:
    symbol, raw = match.groups()
    whole, _, fraction = raw.replace(",", "").partition(".")
    unit = {"$": "dollar", "£": "pound", "€": "euro"}[symbol]
    result = cardinal(int(whole)) + " " + unit + ("s" if int(whole) != 1 else "")
    cents = int(fraction.ljust(2, "0")) if fraction else 0
    if cents:
        small = {"£": (" penny", " pence")}.get(symbol, (" cent", " cents"))[0 if cents == 1 else 1]
        result += " and " + cardinal(cents) + small
    return result


def _iso_date(match: re.Match[str]) -> str:
    year, month, day = map(int, match.groups())
    try:
        calendar_date(year, month, day)
    except ValueError:
        return match.group(0).replace("-", " ")
    return f"{_MONTHS[month - 1]} {ordinal(day)}, {cardinal(year)}"


def _number(match: re.Match[str]) -> str:
    raw = match.group(0)
    sign = "minus " if raw.startswith("-") else ""
    whole, _, fraction = raw.lstrip("-").partition(".")
    spoken = cardinal(int(whole))
    if fraction:
        spoken += " point " + " ".join(_ONES[int(c)] for c in fraction)
    return sign + spoken


def normalize(text: str) -> str:
    """Numbers, money, ordinals, ISO dates and a few abbreviations in words; what the model cannot say removed."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", _iso_date, text)
    text = re.sub(r"(\d)/(\d)", r"\1 slash \2", text)
    for abbreviation, expansion in _ABBREVIATIONS.items():
        text = re.sub(rf"\b{abbreviation}\.(?=\s|$)", expansion, text, flags=re.IGNORECASE)
    text = re.sub(r"([$£€])(\d[\d,]*(?:\.\d{1,2})?)(?!\d)", _money, text)
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: ordinal(int(m[1])), text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: m[0].replace(",", ""), text)
    text = re.sub(r"-?(?:\d+\.\d+|\d+)", _number, text)
    text = text.replace("%", " percent ").replace("&", " and ").replace("+", " plus ")
    # a spoken acronym keeps its dictionary reading; other short capitals are spelled (CPU, GPU)
    text = re.sub(r"\bNASA\b", "nasa", text)
    text = re.sub(r"\b[A-Z]{2,5}\b", lambda m: " ".join(m[0]), text)
    text = text.replace("-", " ")
    # accents folded ("café"), then anything outside the model's letters and marks left out
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = re.sub(r"[^a-zA-Z\s'.,?!;:]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_lexicon(path: Path) -> dict[str, list[str]]:
    """CMUdict as word -> its first pronunciation (ARPAbet phones)."""
    lexicon: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("#", 1)[0].split()
        if len(parts) < 2:
            continue
        lexicon.setdefault(re.sub(r"\(\d+\)$", "", parts[0]).lower(), parts[1:])
    return lexicon


def tokens(text: str, lexicon: dict[str, list[str]]) -> list[int]:
    """Token ids of normalised ``text``: <bos>, phones with <space> between words and the marks, <eos>. A word
    CMUdict lacks is spelled letter by letter."""
    out = ["<bos>"]
    for item in _WORDS.findall(text):
        if item in SPECIAL:
            out.append(item)
            continue
        word = item.lower()
        phones = lexicon.get(word)
        if phones is None:
            phones = [p for c in word if c.isalpha() for p in lexicon.get(c, [])]
        if out[-1] not in ("<bos>", "<space>"):
            out.append("<space>")
        out.extend(p for p in phones if p in SYMBOL_TO_ID)
    out.append("<eos>")
    return [SYMBOL_TO_ID[s] for s in out]


def phase_features(durations: np.ndarray[Any, Any], max_token_frames: int) -> FloatArray:
    """Where each frame sits within its phone: [-1, 1], its sine and cosine, and the phone's length."""
    counts = np.repeat(durations, durations).astype(np.float32)
    starts = np.repeat(np.cumsum(durations) - durations, durations).astype(np.float32)
    fraction = (np.arange(len(counts), dtype=np.float32) - starts + 0.5) / counts
    return np.stack(
        (2 * fraction - 1, np.sin(math.pi * fraction), np.cos(math.pi * fraction),
         np.log1p(counts) / math.log1p(max_token_frames)),
        axis=-1,
    ).astype(np.float32)  # fmt: skip


def periods(features: FloatArray) -> np.ndarray[Any, Any]:
    """FARGAN's pitch period in samples from feature 18 (log2(256 / period) - 1.5)."""
    period: np.ndarray[Any, Any] = np.rint(256 / np.power(2.0, np.clip(features[..., 18], -10, 10) + 1.5))
    return period.clip(32, 255).astype(np.int64)


def deemphasis(audio: FloatArray, state: float) -> tuple[FloatArray, float]:
    out = np.empty_like(audio)
    for i, sample in enumerate(audio):
        state = float(sample) + PREEMPHASIS * state
        out[i] = state
    return out, state


class PiVoOnnx(TTSModel):
    """PiVo-1.4M on ONNX Runtime (see the module docstring)."""

    sample_rate = 16000

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from bolo.engines.lid_onnx import _options
        from bolo.ort_providers import create_session, providers_for

        onnx_dir = path / str(spec.get("onnx_dir", "onnx"))
        self._meta = json.loads((onnx_dir / "metadata.json").read_text(encoding="utf-8"))
        if self._meta.get("format") != "pivo-onnx-v1" or tuple(self._meta.get("symbols", ())) != SYMBOLS:
            raise ValueError("not a PiVo ONNX export this engine knows (format pivo-onnx-v1, its phone symbols)")
        self._config: dict[str, Any] = self._meta["config"]
        self.sample_rate = int(self._config.get("sample_rate", 16000))
        options = _options(max(1, min(4, opts.threads)))
        self._sessions = {
            name: create_session(str(onnx_dir / f"{name}.onnx"), options, providers_for("cpu"))
            for name in ("encoder", "decoder", "conditioning", "vocoder_step")
        }
        self._lexicon = read_lexicon(path / str(spec.get("lexicon", "pivo/resources/cmudict.dict")))
        self.voices = tuple(
            Voice(str(name), str(name), "en-US", "", int(index))
            for name, index in sorted(self._meta["voices"].items(), key=lambda kv: int(kv[1]))
        )
        self._lock = threading.Lock()

    def _durations(self, ids: list[int], voice: int, speed: float) -> tuple[FloatArray, np.ndarray[Any, Any]]:
        c = self._config
        token_array = np.array([ids], dtype=np.int64)
        encoded, log_durations = self._sessions["encoder"].run(
            None,
            {
                "tokens": token_array,
                "voice": np.array([voice], dtype=np.int64),
                "lengths": np.array([len(ids)], dtype=np.int64),
            },
        )
        top = int(c["max_token_frames"])
        durations = np.rint(np.expm1(log_durations.clip(0, np.log1p(top))) / speed).clip(0, top).astype(np.int64)
        durations = np.maximum(durations, (token_array >= len(SPECIAL)).astype(np.int64))
        durations[token_array <= 2] = 0
        return encoded, durations

    def _fits(self, text: str) -> bool:
        ids = tokens(text, self._lexicon)
        return len(ids) <= MAX_TOKENS

    def _pieces(self, text: str) -> list[str]:
        """``text`` (normalised) in pieces the model takes."""
        if self._fits(text):
            return [text]
        parts = shorter(text)
        if not parts:
            return [text]
        return [p for part in parts for p in self._pieces(part)]

    def _speak(self, text: str, voice: int, speed: float, cancelled: Callable[[], bool] | None) -> Iterator[FloatArray]:
        ids = tokens(text, self._lexicon)
        if not any(i >= len(SPECIAL) for i in ids):
            return
        c = self._config
        encoded, durations = self._durations(ids, voice, speed)
        if int(durations.sum()) > int(c["max_utterance_frames"]):
            parts = shorter(text)
            if not parts:
                raise ValueError(f"one word is too long to speak: {text[:40]!r}")
            for part in parts:
                yield from self._speak(part, voice, speed, cancelled)
            return
        expanded = np.repeat(encoded[0], durations[0], axis=0)[None]
        if c.get("phoneme_phase", False):
            phase = phase_features(durations[0], int(c["max_token_frames"]))[None]
            expanded = np.concatenate((expanded, phase), axis=-1)
        head = int(c["width"]) // int(c["heads"])
        cache = {n: np.zeros((1, int(c["kv_heads"]), 0, head), dtype=np.float32) for n in self._meta["cache_names"]}
        state = {
            n: np.zeros((1, w), dtype=np.float32)
            for n, w in zip(self._meta["step_state_names"], _STATE_WIDTHS, strict=True)
        }
        queue: list[FloatArray] = []
        previous: FloatArray | None = None
        emphasis = 0.0
        chunk = 16
        for position in range(0, expanded.shape[1], chunk):
            if cancelled is not None and cancelled():
                return
            result = self._sessions["decoder"].run(
                None,
                {"expanded": expanded[:, position : position + chunk], "position": np.array(position, dtype=np.int64),
                 **cache},
            )  # fmt: skip
            cache = dict(zip(self._meta["cache_names"], result[1:], strict=True))
            queue.extend(result[0][0])
            final = position + chunk >= expanded.shape[1]
            frames: list[FloatArray] = []
            while len(queue) >= 2 or (final and queue):
                current = queue.pop(0)
                following = queue[0] if queue else current
                previous = current if previous is None else previous
                context = np.stack((previous, current, following))[None].astype(np.float32)
                pitch = periods(context)
                (conditioning,) = self._sessions["conditioning"].run(None, {"features": context, "periods": pitch})
                frame: list[FloatArray] = []
                for sub in range(4):
                    out = self._sessions["vocoder_step"].run(
                        None,
                        {"conditioning": conditioning[:, 0, sub * 80 : (sub + 1) * 80], "period": pitch[:, 1], **state},
                    )
                    state = dict(zip(self._meta["step_state_names"], out[1:], strict=True))
                    frame.append(out[0][0])
                pcm, emphasis = deemphasis(np.concatenate(frame).astype(np.float32), emphasis)
                previous = current
                frames.append(pcm)
            if frames:
                yield np.concatenate(frames)

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[FloatArray]:
        chosen = self.find_voice(voice) or self.voices[0]
        speed = min(2.0, max(0.5, float(speed)))
        plain = normalize(text)
        if not plain:
            return
        with self._lock:
            for piece in self._pieces(plain):
                for audio in self._speak(piece, chosen.sid, speed, cancelled):
                    if not np.isfinite(audio).all():
                        raise RuntimeError("PiVo produced non-finite audio")
                    yield np.clip(audio, -1.0, 1.0).astype(np.float32)
