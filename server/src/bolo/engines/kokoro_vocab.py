"""Small ONNX voices that read Kokoro's phoneme vocabulary, in process (no server, no PyTorch).

Kokoro-82M's distilled and derived students take the teacher's phoneme ids (``config.json``: ``vocab``, symbol to
id, misaki's spelling of English) with a pad (id 0) at each end, and give 24 kHz audio. The phonemes come from
bolo's phonemiser layer (:mod:`bolo.text.phonemize`, profile ``kokoro``: uccharan's port of misaki), per voice:
American English for the ``a*`` voices, British for the ``b*`` ones.

* **Paradee-8M** (kind ``paradee_onnx``; Sahil Mahendrakar, Apache-2.0): 7.9 M parameters distilled from Kokoro-82M,
  four voices (af_heart, bf_isabella, am_puck, bm_fable) in one graph: ``input_ids`` [1, T], ``speed`` [1] and
  ``voice`` [1] (the voice's place in ``config.json``'s ``voices``) to ``waveform`` [1, samples].
"""

from __future__ import annotations

import json
import threading
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
        import onnxruntime as ort

        from bolo.engines.lid_onnx import _find, _options
        from bolo.ort_providers import create_session, providers_for

        options = _options(max(1, min(4, opts.threads)))
        # With bolo's exact int8 products (session.x64quantprecision), ONNX Runtime 1.32's extended fusions fail on
        # this int8 graph ("Attempt to replace the existing tensor"); the basic ones load it.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self._session = create_session(
            str(_find(path, str(spec.get("model", "onnx/paradee_int8.onnx")))), options, providers_for(opts.device)
        )
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
