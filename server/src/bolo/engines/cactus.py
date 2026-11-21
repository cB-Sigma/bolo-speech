"""Cactus Compute's runtime (cactus-compute, Apache-2.0, https://github.com/cactus-compute/cactus): Whistle.

Whistle is a 16.9 MB speech-to-text model (English, German, French, Spanish, Italian, Dutch and Polish) stored
in Cactus's own ``.cact`` format, whose 2- and 4-bit "Cactus Quants" tensors only Cactus's engine reads. bolo
downloads the ``.cact`` file like any other model; the ``cactus-compute`` package loads it into the engine
(``libcactus``, a native library the package fetches on first use from its pinned, checksummed release in
Cactus-Compute/cactus on Hugging Face) and decodes in-process. The engine detects the language, gives word times
with probabilities and biases decoding towards given keywords (a request's ``hotwords``).

The package sends anonymous usage counts unless ``CACTUS_TELEMETRY=0``; bolo sets that before importing it (an
operator who wants the counts sent sets ``CACTUS_TELEMETRY=1`` themselves).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import (
    Engine,
    EngineStatus,
    LoadOptions,
    Model,
    Segment,
    STTModel,
    STTRequest,
    STTResult,
    Word,
)

INSTALL_HINT = "pip install 'bolo-speech[cactus]'"


def _import_cactus() -> Any:
    os.environ.setdefault("CACTUS_TELEMETRY", "0")
    import cactus

    return cactus


def _keywords(hotwords: str | None) -> list[str] | None:
    """A request's hotwords (phrases separated by commas, semicolons, slashes or new lines; a sherpa-onnx style
    score such as ``Zintrava :3`` is dropped) as Whistle's keyword list."""
    words = [re.sub(r"\s*:\s*[0-9.]+$", "", p).strip() for p in re.split(r"[,;/\n]+", hotwords or "")]
    return [w for w in words if w] or None


class WhistleSTT(STTModel):
    #: Whistle reads at most 30 s at a time (its config's max_audio_seconds); bolo cuts longer audio at pauses
    max_chunk_s = 30.0
    supports_word_timestamps = True
    request_fields = ("hotwords",)

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        cactus = _import_cactus()
        weights = sorted(path.glob("*.cact"))
        if not weights:
            raise FileNotFoundError(f"no .cact file in {path}")
        cactus.set_threads(max(int(opts.threads), 0))
        self._model = cactus.Whistle(weights=str(weights[0]))
        self.languages = tuple(str(x) for x in spec.get("languages") or ("en", "de", "fr", "es", "it", "nl", "pl"))
        self.default_language = str(spec.get("language", "")) or None

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        return [self._one(r) for r in requests]

    def _one(self, r: STTRequest) -> STTResult:
        lang = (r.language or self.default_language or "").split("-")[0].lower() or None
        if lang and lang not in self.languages:
            lang = None  # a language it does not know: let it detect (it answers in one of its own)
        out = self._model.transcribe(
            np.ascontiguousarray(r.samples, dtype=np.float32),
            language=lang,
            keywords=_keywords(r.hotwords),
            word_timestamps=bool(r.word_timestamps),
        )
        text = str(out.get("text") or "").strip()
        words = [
            Word(str(w["word"]).strip(), float(w["start"]), float(w["end"]), float(w.get("probability", 1.0)))
            for w in out.get("words") or []
            if str(w.get("word", "")).strip()
        ]
        duration = len(r.samples) / self.sample_rate
        language = str(out.get("language") or lang or "") or None
        return STTResult(text, language, [Segment(0.0, duration, text, words)] if text else [], words)


class CactusEngine(Engine):
    id = "cactus"
    name = "cactus-compute"
    license = "Apache-2.0"
    url = "https://github.com/cactus-compute/cactus"
    kinds = ("cactus-whistle",)

    def status(self) -> EngineStatus:
        try:
            cactus = _import_cactus()
            return EngineStatus(self.id, True, str(getattr(cactus, "__version__", "unknown")), None)
        except Exception as exc:
            return EngineStatus(self.id, False, None, f"{type(exc).__name__}: {exc}. Optional: {INSTALL_HINT}")

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind != "cactus-whistle":
            raise ValueError(f"cactus engine cannot load kind {kind!r}")
        return WhistleSTT(path, spec, opts)
