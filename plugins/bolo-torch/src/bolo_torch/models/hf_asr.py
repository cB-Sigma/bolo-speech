"""Speech-to-text models through a transformers ``automatic-speech-recognition`` pipeline (kind ``hf-asr-pipeline``).

A generic loader for checkpoints that transformers runs with its own pipeline, or with a pipeline class the model
repository ships (``custom_pipelines`` in ``config.json``, remote code). Remote code runs only after the plugin's
remote-code check (:mod:`bolo_torch.remote_code`: the operator's ``allow_remote_code`` for the model and the files
pinned by sha256 at the reviewed commit); a catalog entry without ``options.remote_code`` loads with
``trust_remote_code=False``.

Each request is one call of the pipeline on 16 kHz float32 samples (``{"array": ..., "sampling_rate": 16000}``),
greedy, with a token budget that grows with the audio (``tokens_per_s``, at most ``max_new_tokens``).

Catalog options:

* ``language_kwarg``: the pipeline's argument for the language (``language``), passed when a request names one;
  ``language_map`` turns bolo's codes into the model's (``{"ar": "ar"}``), and a code missing from a non-empty map
  is refused. Without ``language_kwarg`` the language is not passed.
* ``call_kwargs``: further plain arguments of every call (``{"code_switch": true}``).
* ``max_new_tokens`` (default 448), ``tokens_per_s`` (default 8), ``max_chunk_s`` (default 30).
* ``extra``: the plugin extra named when transformers is missing (default ``hf-asr``).
* ``result_language_key``: the key of the pipeline's result that holds the detected language, if it gives one.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from bolo.engines.base import Segment, STTModel, STTRequest, STTResult

from bolo_torch import remote_code
from bolo_torch.runtime import InvalidRequest, LoadContext, model_dir, require

SR = 16000
TASK = "automatic-speech-recognition"

Pipe = Callable[..., Any]


def _plain(value: Any, where: str) -> Any:
    """Catalog values passed to the pipeline must be plain data."""
    if isinstance(value, dict):
        return {str(k): _plain(v, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v, where) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise InvalidRequest(f"{where}: {type(value).__name__} is not a plain value")


class HfAsrPipeline(STTModel):
    """A transformers speech-recognition pipeline as a bolo speech-to-text model."""

    sample_rate = SR
    supports_translate = False
    supports_word_timestamps = False

    def __init__(
        self,
        pipe: Pipe,
        *,
        language_kwarg: str | None = None,
        language_map: dict[str, str] | None = None,
        call_kwargs: dict[str, Any] | None = None,
        max_new_tokens: int = 448,
        tokens_per_s: float = 8.0,
        max_chunk_s: float = 30.0,
        result_language_key: str | None = None,
    ) -> None:
        self.pipe = pipe
        self.language_kwarg = language_kwarg
        self.language_map = dict(language_map or {})
        #: the languages a request may name (empty: the language is not passed, or any code goes through)
        self.languages = tuple(self.language_map) if language_kwarg else ()
        self.call_kwargs = dict(call_kwargs or {})
        self.max_new_tokens = max_new_tokens
        self.tokens_per_s = tokens_per_s
        self.max_chunk_s = max_chunk_s
        self.result_language_key = result_language_key
        self._lock = threading.Lock()

    def budget(self, seconds: float) -> int:
        return max(16, min(self.max_new_tokens, int(seconds * self.tokens_per_s) + 32))

    def model_language(self, language: str | None) -> str | None:
        """The model's code for a request's language, or None (detect)."""
        code = (language or "").strip().lower().split("-", 1)[0]
        if not code or not self.language_kwarg:
            return None
        if self.language_map:
            if code not in self.language_map:
                raise InvalidRequest(
                    f"this model takes the languages {', '.join(sorted(self.language_map))}, not {language!r}"
                )
            return self.language_map[code]
        return code

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        out: list[STTResult] = []
        for r in requests:
            if r.task == "translate":
                raise InvalidRequest("this model transcribes only; it does not translate")
            audio = np.ascontiguousarray(r.samples, dtype=np.float32)
            duration = len(audio) / SR
            lang = self.model_language(r.language)
            if len(audio) < int(0.05 * SR):
                out.append(STTResult("", r.language))
                continue
            kwargs = {**self.call_kwargs, "max_new_tokens": self.budget(duration)}
            if lang is not None and self.language_kwarg:
                kwargs[self.language_kwarg] = lang
            with self._lock:
                result = self.pipe({"array": audio, "sampling_rate": SR}, **kwargs)
            if isinstance(result, list):
                result = result[0] if result else {}
            text = str((result or {}).get("text") or "").strip()
            detected = (result or {}).get(self.result_language_key) if self.result_language_key else None
            language = r.language or (str(detected) if detected else None)
            out.append(STTResult(text, language, [Segment(0.0, duration, text)] if text else []))
        return out


def load(ctx: LoadContext) -> HfAsrPipeline:
    path = model_dir(ctx.path, ctx.options)
    trust = remote_code.trust_remote_code(ctx, path) if ctx.options.get("remote_code") else False
    transformers = require("transformers", str(ctx.option("extra", "hf-asr")))
    pipe = transformers.pipeline(
        str(ctx.option("pipeline_task", TASK)),
        model=str(path),
        trust_remote_code=trust,
        dtype=ctx.dtype,
        device=ctx.device,
    )
    return HfAsrPipeline(
        pipe,
        language_kwarg=ctx.option("language_kwarg"),
        language_map={str(k): str(v) for k, v in dict(ctx.option("language_map", {})).items()},
        call_kwargs=_plain(dict(ctx.option("call_kwargs", {})), "call_kwargs"),
        max_new_tokens=int(ctx.option("max_new_tokens", 448)),
        tokens_per_s=float(ctx.option("tokens_per_s", 8.0)),
        max_chunk_s=float(ctx.option("max_chunk_s", 30.0)),
        result_language_key=ctx.option("result_language_key"),
    )
