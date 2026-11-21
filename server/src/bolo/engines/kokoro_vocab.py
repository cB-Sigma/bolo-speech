"""Small ONNX voices that read Kokoro's phoneme vocabulary, in process (no server, no PyTorch).

Kokoro-82M's distilled and derived students take the teacher's phoneme ids (``config.json``: ``vocab``, symbol to
id, misaki's spelling of English) with a pad (id 0) at each end, and give 24 kHz audio. The phonemes come from
bolo's phonemiser layer (:mod:`bolo.text.phonemize`, profile ``kokoro``: uccharan's port of misaki), per voice:
American English for the ``a*`` voices, British for the ``b*`` ones.

* **Paradee-8M** (kind ``paradee_onnx``; Sahil Mahendrakar, Apache-2.0): 7.9 M parameters distilled from Kokoro-82M,
  four voices (af_heart, bf_isabella, am_puck, bm_fable) in one graph: ``input_ids`` [1, T], ``speed`` [1] and
  ``voice`` [1] (the voice's place in ``config.json``'s ``voices``) to ``waveform`` [1, samples].
* **Kitten TTS 2 nano, pico and mini** (kind ``kitten2_onnx``; KittenML, Apache-2.0): Kokoro's ids with an end token,
  a style row per input length from each voice's pack, one graph per language (nano, pico) or for every language
  (mini); see :class:`Kitten2Onnx`.
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import FloatArray, LoadOptions, TTSModel, Voice

#: Kokoro's graphs take at most 512 ids, the two pads included
KOKORO_MAX = 510
PEAK = 0.99

_ACCENTS = {"a": ("en-US", "American"), "b": ("en-GB", "British")}


def fit(phones: str, limit: int = KOKORO_MAX) -> list[str]:
    """Phonemes cut at spaces into pieces of at most ``limit`` symbols (a word longer than that is cut anyway)."""
    out: list[str] = []
    phones = phones.strip()
    while len(phones) > limit:
        cut = phones.rfind(" ", 0, limit)
        cut = cut if cut > 0 else limit
        out.append(phones[:cut].strip())
        phones = phones[cut:].strip()
    if phones:
        out.append(phones)
    return out


def ids(phones: str, vocab: dict[str, int]) -> list[int]:
    """Kokoro ids of a phoneme string with a pad at each end; symbols the vocabulary lacks are left out."""
    return [0, *(vocab[c] for c in phones if c in vocab), 0]


def int8_session(path: Path, file: str, threads: int, device: str | None) -> Any:
    """An ONNX Runtime session for one of these int8 graphs. With bolo's exact int8 products
    (``session.x64quantprecision``), ONNX Runtime 1.32's extended fusions fail on them ("Attempt to replace the
    existing tensor"), so only the basic ones run."""
    import onnxruntime as ort

    from bolo.engines.lid_onnx import _find, _options
    from bolo.ort_providers import create_session, providers_for

    options = _options(max(1, min(4, threads)))
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return create_session(str(_find(path, file)), options, providers_for(device))


def kokoro_voice(name: str, sid: int) -> Voice:
    """A Kokoro-style voice name (``af_heart``: American, female) as a :class:`Voice`."""
    language, accent = _ACCENTS.get(name[:1], ("en-US", "American"))
    gender = {"f": "f", "m": "m"}.get(name[1:2], "")
    label = name.split("_", 1)[-1].replace("_", " ").title()
    sex = {"f": "female", "m": "male"}.get(gender, "")
    return Voice(name, f"{label} ({accent}{', ' + sex if sex else ''})", language, gender, sid)


class ParadeeOnnx(TTSModel):
    """Paradee-8M on ONNX Runtime (see the module docstring)."""

    sample_rate = 24000
    accepts_phonemizer = True

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        self._session = int8_session(path, str(spec.get("model", "onnx/paradee_int8.onnx")), opts.threads, opts.device)
        config = json.loads((path / str(spec.get("config", "config.json"))).read_text(encoding="utf-8"))
        self._vocab: dict[str, int] = {str(k): int(v) for k, v in config["vocab"].items()}
        self.sample_rate = int(config.get("sample_rate", 24000))
        self.voices = tuple(kokoro_voice(str(v), i) for i, v in enumerate(config["voices"]))
        self._lock = threading.Lock()

    def phonemes(self, text: str, voice: Voice, phonemizer: str | None = None) -> list[str]:
        """The phoneme pieces of ``text`` in ``voice``'s accent, each short enough for the graph."""
        from bolo.text import phonemize

        language = (voice.language or "en-US").lower()
        ph, improved = phonemize.resolve("kokoro", language, phonemizer)
        sentences = ["".join(map(chr, cps)) for cps in ph.voice_codepoints(text, language, "kokoro", improved)]
        return [piece for s in sentences for piece in fit(s)]

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
        phonemizer: str | None = None,
    ) -> Iterator[FloatArray]:
        chosen = self.find_voice(voice) or self.voices[0]
        pieces = self.phonemes(text, chosen, phonemizer)
        if not pieces:
            return
        for phones in pieces:
            if cancelled is not None and cancelled():
                return
            with self._lock:
                (audio,) = self._session.run(
                    None,
                    {
                        "input_ids": np.array([ids(phones, self._vocab)], dtype=np.int64),
                        "speed": np.array([speed], dtype=np.float32),
                        "voice": np.array([chosen.sid], dtype=np.int64),
                    },
                )
            wave = np.asarray(audio, dtype=np.float32).reshape(-1)
            loudest = float(np.abs(wave).max()) if len(wave) else 0.0
            yield (wave * (PEAK / loudest) if loudest > PEAK else wave).astype(np.float32)


# -- Kitten TTS 2 nano / pico / mini ---------------------------------------------------------------------------------
#: Kitten's text front end (kittenml's kittentts2_tiny/frontend, Apache-2.0, at kittenml 0.9.8) in the parts bolo
#: needs: misaki's espeak-to-misaki table for languages read through espeak-ng, and the training lists' folds
#: (phonemize_v1.FOLD, phonemize_v2.FOLD_EXTRA, the Hindi and Arabic marks and the Hindi flap).
KITTEN_END = 10
KITTEN_MAX_PHONEMES = 350
KITTEN_GAP_S = 0.15
_E2M = sorted(
    {
        "a^ɪ": "I", "a^ʊ": "W", "d^z": "ʣ", "d^ʒ": "ʤ", "e^ɪ": "A", "o^ʊ": "O", "ə^ʊ": "Q", "s^s": "S",
        "t^s": "ʦ", "t^ʃ": "ʧ", "ɔ^ɪ": "Y",
    }.items()
)  # fmt: skip
_FOLD_V1 = {
    "̪": "", "̩": "", "̍": "", "̯": "", "ɭ": "l", "ɫ": "l", "ɵ": "ø",
    "ε": "ɛ", "\u0005": "", "\u0006": "", "\u0007": "", "\u0008": "", "{": "", "}": "", "[": "", "]": "",
    "¿": "", "¡": "", " ": " ",
}  # fmt: skip
_FOLD_V2_EXTRA = {
    " ": " ", "\t": " ", "«": "", "»": "", "‘": "", "’": "", "'": "", "~": "", "·": "", "#": "", "g": "ɡ",
}  # fmt: skip
#: Kitten's espeak-ng voice per language (phonemize_v1.ESPEAK_VOICE and v2's Korean)
KITTEN_ESPEAK_VOICE = {
    "es": "es", "fr": "fr-fr", "it": "it", "pt": "pt-br", "hi": "hi", "de": "de", "nl": "nl", "pl": "pl", "ru": "ru",
    "tr": "tr", "ar": "ar", "he": "he", "sv": "sv", "da": "da", "nb": "nb", "fi": "fi", "el": "el", "cs": "cs",
    "hu": "hu", "ro": "ro", "uk": "uk", "ms": "ms", "sw": "sw", "ko": "ko",
}  # fmt: skip
_TEXT_MAP = {"hi": {"।": ".", "॥": "."}, "ar": {"؟": "?"}}
_HI_FLAP = re.compile("[ड़ढ़]|[डढ]़")
_CJK_STRIP = frozenset("-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def espeak_to_misaki(phones: str) -> str:
    """espeak-ng IPA written with ``^`` ties into misaki's symbols, as misaki's EspeakG2P does (version 1.0)."""
    for old, new in _E2M:
        phones = phones.replace(old, new)
    return phones.replace("^", "").replace("-", "")


def kitten_fold(phones: str, language: str, version: str, text: str = "") -> str:
    """The folds Kitten's training lists went through (phonemizer ``v1``, or ``v2`` for the mini)."""
    table = {**_FOLD_V1, **_FOLD_V2_EXTRA} if version == "v2" else _FOLD_V1
    phones = phones.strip()
    for old, new in table.items():
        phones = phones.replace(old, new)
    base = language.split("-", maxsplit=1)[0]
    if version == "v2" and base == "hi":
        phones = phones.replace("r.h", "ɽʰ")
        phones = re.sub(r"r\.(?=[^\s.,?!;:…])", "ɽ", phones)
        if _HI_FLAP.search(text):
            phones = phones.replace("r.", "ɽ")
    if base == "zh" or (version == "v2" and base == "ja"):
        phones = "".join(c for c in phones if c not in _CJK_STRIP)
    return phones


def kitten_text(text: str, language: str, version: str) -> str:
    """Marks the v2 front end rewrites before phonemising (Hindi dandas, the Arabic question mark)."""
    if version != "v2":
        return text
    for old, new in _TEXT_MAP.get(language.split("-", maxsplit=1)[0], {}).items():
        text = text.replace(old, new)
    return text


def kitten_tokens(phones: str, vocab: dict[str, int]) -> list[int]:
    """[0] + phoneme ids (brackets read as commas) + the end token + [0], as Kitten was trained."""
    comma = vocab.get(",")
    out = [0]
    for c in phones:
        if c in vocab:
            out.append(vocab[c])
        elif c in "()" and comma is not None:
            out.append(comma)
    return [*out, KITTEN_END, 0]


def kitten_trim(wave: FloatArray, durations: np.ndarray[Any, Any], keep_start: bool) -> FloatArray:
    """Cut the end token's frames (and, after the first chunk, the start pause) and fade 5 ms in and out."""
    w = np.clip(wave.astype(np.float32).reshape(-1), -1, 1)
    durations = np.asarray(durations).reshape(-1)
    per_frame = max(1, round(len(w) / max(int(durations.sum()), 1)))
    lead = 0 if keep_start else max(0, int(durations[0]) - 1) * per_frame
    trail = max(0, int(durations[-1]) - 1) * per_frame
    if lead + trail + per_frame < len(w):
        w = w[lead : len(w) - trail] if trail else w[lead:]
    w = np.ascontiguousarray(w)
    fade = min(120, len(w) // 8)
    if fade:
        w[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        w[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return w


class Kitten2Onnx(TTSModel):
    """Kitten TTS 2 nano, pico and mini (KittenML, Apache-2.0) on ONNX Runtime.

    A repository's ``config.json`` names, per language, its graph (``languages.<lang>.model_file``: one per
    language for nano and pico, one for every language for the mini) and voice packs (``voices_<lang>.npz``: per
    voice, one 256-wide style row per input length). The graph takes Kokoro's phoneme ids (``vocab.json``),
    ``style`` [1, 256] and ``speed`` [1]. Phonemes come from bolo's phonemiser in the ``kokoro`` profile, which is
    misaki's spelling for English and misaki's espeak-ng conversion for the other languages, as Kitten's own front
    end does, then Kitten's folds.
    """

    sample_rate = 24000
    accepts_phonemizer = True

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        self._path = path
        self._config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        vocab_file = path / str(self._config.get("vocab", "vocab.json"))
        self._vocab: dict[str, int] = {str(k): int(v) for k, v in json.loads(vocab_file.read_text("utf-8")).items()}
        self._version = str(self._config.get("phonemizer", "v1"))
        self.sample_rate = int(self._config.get("sample_rate", 24000))
        self._threads = max(1, min(4, opts.threads))
        self._device = opts.device
        self._sessions: dict[str, Any] = {}
        self._packs: dict[str, dict[str, FloatArray]] = {}
        self._languages = {str(k): dict(v) for k, v in self._config["languages"].items()}
        wanted = {str(x) for x in spec.get("languages") or self._languages}
        voices: list[Voice] = []
        self._voice_language: dict[str, tuple[str, str]] = {}
        for language, entries in self._config["voices"].items():
            if language not in wanted or language not in self._languages:
                continue
            for entry in entries:
                vid = f"{language}-{entry['key']}"
                description = str(entry.get("description", ""))
                gender = "f" if "(female)" in description or "Female" in description else ""
                gender = gender or ("m" if "(male)" in description or "Male" in description else "")
                name = f"{entry.get('name', entry['key'])} ({language}{', ' + description if description else ''})"
                voices.append(Voice(vid, name, language, gender, len(voices)))
                self._voice_language[vid] = (language, str(entry["key"]))
        if not voices:
            raise ValueError("the Kitten TTS 2 repository has no voices in the languages asked for")
        self.voices = tuple(voices)
        self._lock = threading.Lock()

    def _session(self, language: str) -> Any:
        file = str(self._languages[language]["model_file"])
        if file not in self._sessions:
            self._sessions[file] = int8_session(self._path, file, self._threads, self._device)
        return self._sessions[file]

    def _pack(self, language: str, key: str) -> FloatArray:
        file = str(self._languages[language]["voices"])
        if file not in self._packs:
            with np.load(self._path / file) as packs:
                self._packs[file] = {k: packs[k].astype(np.float32) for k in packs.files}
        return self._packs[file][key]

    def phonemes(self, text: str, voice: Voice, phonemizer: str | None = None) -> list[str]:
        """The phoneme chunks of ``text`` in ``voice``'s language, folded as Kitten's training lists were."""
        from bolo.text import phonemize

        language = voice.language
        text = kitten_text(text, language, self._version)
        tag = {"en": "en-us", "pt": "pt-br"}.get(language, language)
        ph, improved = phonemize.resolve("kokoro", tag, phonemizer)
        if ph.component == phonemize.ESPEAK and language != "en":
            # misaki's espeak conversion reads ties (a^ɪ -> I); NFC undoes the NFD of the Piper convention
            voice_name = KITTEN_ESPEAK_VOICE.get(language, language)
            sentences = [
                espeak_to_misaki(unicodedata.normalize("NFC", s)) for s in ph.phonemize(text, voice_name, tie="^")
            ]
        else:
            sentences = ["".join(map(chr, cps)) for cps in ph.voice_codepoints(text, tag, "kokoro", improved)]
        out: list[str] = []
        for sentence in sentences:
            folded = kitten_fold(sentence, language, self._version, text)
            out.extend(fit(folded, KITTEN_MAX_PHONEMES))
        return [p for p in out if p]

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
        phonemizer: str | None = None,
    ) -> Iterator[FloatArray]:
        chosen = self.find_voice(voice) or self.voices[0]
        language, key = self._voice_language[chosen.id]
        pack = self._pack(language, key)
        gap = np.zeros(int(KITTEN_GAP_S * self.sample_rate), dtype=np.float32)
        for i, phones in enumerate(self.phonemes(text, chosen, phonemizer)):
            if cancelled is not None and cancelled():
                return
            tokens = kitten_tokens(phones, self._vocab)
            style = pack[min(len(tokens) - 1, pack.shape[0] - 1)][None]
            with self._lock:
                wave, durations = self._session(language).run(
                    None,
                    {
                        "input_ids": np.array([tokens], dtype=np.int64),
                        "style": style.astype(np.float32),
                        "speed": np.array([speed], dtype=np.float32),
                    },
                )
            audio = kitten_trim(np.asarray(wave), np.asarray(durations), keep_start=i == 0)
            yield audio if i == 0 else np.concatenate([gap, audio])
