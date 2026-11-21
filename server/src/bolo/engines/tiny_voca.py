"""tiny-voca (Muhammed Nazeem, Apache-2.0) in process: the authors' ONNX export on ONNX Runtime, no server.

A 4.5 M-parameter English voice (F2 female, M2 male) at 24 kHz. little-tts (TuringLogic, weights CC BY-NC 4.0) is
the same graph interface over the same phonemiser with other weights and voices (92 female, 6097 male); its
description file (``onnx/model.json``) is flat where tiny-voca's ``config.json`` keeps the model's settings under
``model`` (:func:`layout` reads both). Text becomes phonemes and stress values with dizione,
the authors' phonemiser (carried in :mod:`bolo.vendor.dizione`), over the data files of the model repository
(``phonemizer/``); the graph takes phoneme ids, stress, the voice, speed, pitch shift and pitch range and returns the
waveform. As the reference (``tiny_voca.infer.speak_text``) does: a text without a final mark gets a full stop, an
utterance longer than the model takes (``max_tokens``, less a margin) is cut after commas, then between words, and the
audio is scaled down when it peaks above 0.99.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import FloatArray, LoadOptions, TTSModel, Voice

PEAK = 0.99
_SPACE = re.compile("[\\s" + chr(0xFEFF) + "]+")
_TEXT_END = re.compile("[\\s" + chr(0xFEFF) + "]+$")
_CLAUSE_END = re.compile(r"(?<=[,;:—])\s+")
_CLOSING = "\"'”’)]»"


def finished(text: str) -> str:
    """``text`` with a full stop when it ends without a mark (without one the last word breaks off)."""
    body = _TEXT_END.sub("", text)
    if not body:
        return text
    last = _SPACE.split(body)[-1]
    if any(char.isalnum() for char in last):
        marked = last.rstrip(_CLOSING)[-1:] in tuple(".!?…,;:")
    else:
        marked = any(char in ".!?…" for char in last) or last in (",", ";", ":")
    return text if marked else body + "."


def shorter(text: str) -> list[str]:
    """``text`` in smaller pieces: after commas and the like, else in two between words; [] for one word."""
    pieces = _CLAUSE_END.split(text)
    if len(pieces) > 1:
        return pieces
    words = text.split(" ")
    if len(words) < 2:
        return []
    half = (len(words) + 1) // 2
    return [" ".join(words[:half]), " ".join(words[half:])]


def layout(config: dict[str, Any]) -> tuple[list[str], int]:
    """The voices and the most phoneme tokens the graph takes, from tiny-voca's ``config.json`` (settings under
    ``model``) or little-tts's flat ``onnx/model.json``. A symbol table (``vocab`` or ``symbols``) that is not
    dizione's means the graph was trained on other ids: refused rather than spoken as noise."""
    from bolo.vendor.dizione.symbols import SYMBOLS

    nested = config.get("model")
    model: dict[str, Any] = nested if isinstance(nested, dict) else {}
    speakers = [str(v) for v in config.get("voices") or model.get("speakers") or ()]
    if not speakers:
        raise ValueError("the model's description names no voices")
    table = config.get("vocab") or config.get("symbols")
    if table is not None and tuple(table) != tuple(SYMBOLS):
        raise ValueError("the model's phoneme symbols are not dizione's: another front end trained it")
    return speakers, int(config.get("max_tokens") or model.get("max_tokens") or 512)


_NAMES = {
    "F2": ("F2 (female)", "f"),
    "M2": ("M2 (male)", "m"),
    "92": ("92 (female)", "f"),
    "6097": ("6097 (male)", "m"),
}


class TinyVocaOnnx(TTSModel):
    """tiny-voca on ONNX Runtime (see the module docstring)."""

    sample_rate = 24000

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from bolo.engines.lid_onnx import _find, _options
        from bolo.ort_providers import create_session, providers_for
        from bolo.vendor.dizione.engine import Phonemizer

        threads = max(1, min(4, opts.threads))
        self._session = create_session(
            str(_find(path, str(spec.get("model", "onnx/model.onnx")))), _options(threads), providers_for(opts.device)
        )
        config = json.loads((path / str(spec.get("config", "config.json"))).read_text(encoding="utf-8"))
        self._speakers, limit = layout(config)
        self.sample_rate = int(config.get("sample_rate", 24000))
        self._limit = limit - 12
        self._phonemizer = Phonemizer(path / str(spec.get("phonemizer", "phonemizer")))
        self.voices = tuple(
            Voice(v, _NAMES.get(v, (v, ""))[0], "en-US", _NAMES.get(v, (v, ""))[1], i)
            for i, v in enumerate(self._speakers)
        )
        self._lock = threading.Lock()

    def _phonemes(self, text: str) -> Any:
        from bolo.vendor.dizione.symbols import tokenize

        return tokenize(self._phonemizer.pieces(text))

    def _utterance(self, text: str, voice: int, speed: float) -> list[FloatArray]:
        try:
            phones = self._phonemes(text)
        except ValueError:
            return []  # nothing to pronounce (another script, signs only)
        if len(phones.ids) > self._limit:
            pieces = shorter(text)
            if not pieces:
                raise ValueError(f"one word is too long to speak: {text[:40]!r}")
            return [w for part in pieces for w in self._utterance(part, voice, speed)]

        def one(value: float) -> FloatArray:
            return np.array([value], dtype=np.float32)

        with self._lock:
            audio, _durations = self._session.run(
                None,
                {
                    "tokens": np.array([phones.ids], dtype=np.int64),
                    "stress": np.array([phones.stress], dtype=np.int64),
                    "voice": np.array([voice], dtype=np.int64),
                    "speed": one(speed),
                    "pitch_shift": one(0.0),
                    "pitch_range": one(1.0),
                },
            )
        return [np.asarray(audio, dtype=np.float32).reshape(-1)]

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[FloatArray]:
        chosen = self.find_voice(voice)
        index = self._speakers.index(chosen.id) if chosen is not None else 0
        waves = self._utterance(finished(text.strip()), index, speed)
        if not waves:
            return
        wave = np.concatenate(waves)
        loudest = float(np.abs(wave).max()) if len(wave) else 0.0
        yield (wave * (PEAK / loudest) if loudest > PEAK else wave).astype(np.float32)
