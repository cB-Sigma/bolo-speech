"""Salesforce SOV-Transcribe-1: a Whisper large-v3 encoder in front of Gemma 4 26B A4B, through transformers.

The model is a speech-augmented language model for non-streaming speech recognition in English, French, German,
Spanish, Italian, Portuguese and Dutch. Its modelling and processing code is Python in its own Hugging Face
repository (``trust_remote_code``), so the plugin loads it only after the remote-code check
(:mod:`bolo_torch.remote_code`: the operator's ``allow_remote_code`` for this model, and the files pinned by sha256
at the reviewed commit). The repository's ``vllm_plugin/`` folder is not downloaded (the catalog excludes it).

A request is one user turn: an instruction from the card's prompt guide and the audio, through the processor's own
``apply_transcription_request`` (it pads the audio to whole 30 s windows, 375 audio tokens each, and renders the chat
template with reasoning off), then greedy ``generate``:

* no language: ``Transcribe the audio verbatim in the language's standard writing system, ...`` (the card's
  "automatic language" prompt);
* a language: ``Transcribe the <Language> audio into text, ensuring all punctuation marks are included.``;
* ``hotwords``: ``This audio may contain the following words or phrases: <terms>.`` in front (the card's keyword
  prompt).

The model gives no timestamps: bolo cuts long audio at pauses into pieces of at most ``max_chunk_s`` and the segments
are those pieces. No translation. A temperature above 0 samples instead of decoding greedily.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from bolo.engines.base import STTModel, STTRequest, STTResult

from bolo_torch import remote_code
from bolo_torch.runtime import InvalidRequest, LoadContext, MissingDependency, model_dir, require, split_kwargs

log = logging.getLogger("bolo_torch")

SR = 16000
LANGUAGES = ("en", "fr", "de", "es", "it", "pt", "nl")
NAMES = {"en": "English", "fr": "French", "de": "German", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
         "nl": "Dutch"}  # fmt: skip
AUTO = (
    "Transcribe the audio verbatim in the language's standard writing system, using punctuation and capitalization. "
    "Output only the transcript."
)
WITH_LANGUAGE = "Transcribe the {language} audio into text, ensuring all punctuation marks are included."
KEYWORDS = "This audio may contain the following words or phrases: {keywords}. "


def instruction(language: str | None, hotwords: str | None = None) -> str:
    """The user instruction of one request (the card's prompt guide)."""
    code = (language or "").strip().lower().split("-", 1)[0]
    if code and code not in NAMES:
        raise InvalidRequest(f"SOV-Transcribe-1 transcribes {', '.join(LANGUAGES)}, not {language!r}")
    text = WITH_LANGUAGE.format(language=NAMES[code]) if code else AUTO
    terms = ", ".join(t.strip() for t in (hotwords or "").replace("\n", ",").split(",") if t.strip())
    return KEYWORDS.format(keywords=terms) + text if terms else text


class Backend:
    """What the model wrapper needs from the network (a fake in tests, transformers in production)."""

    def generate(self, audio: Any, prompt: str, max_new_tokens: int, temperature: float) -> str:
        raise NotImplementedError


class TransformersBackend(Backend):
    def __init__(self, model: Any, processor: Any, device: str) -> None:
        self.model = model
        self.processor = processor
        self.device = device

    def generate(self, audio: Any, prompt: str, max_new_tokens: int, temperature: float) -> str:
        import torch

        inputs = self.processor.apply_transcription_request([audio], prompt=prompt)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        sampling: dict[str, Any] = {"do_sample": False}
        if temperature > 0:
            sampling = {"do_sample": True, "temperature": float(temperature)}
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
        new = out[:, inputs["input_ids"].shape[1] :]
        return str(self.processor.batch_decode(new, skip_special_tokens=True)[0])


class SovTranscribe(STTModel):
    """SOV-Transcribe-1 as a bolo speech-to-text model."""

    sample_rate = SR
    supports_translate = False
    supports_word_timestamps = False
    request_fields = ("hotwords", "temperature")
    languages = LANGUAGES
    placement_report: dict[str, Any] | None = None

    def __init__(
        self, backend: Backend, *, max_chunk_s: float = 30.0, tokens_per_s: float = 12.0, max_new_tokens: int = 8192
    ) -> None:
        self.backend = backend
        self.max_chunk_s = max_chunk_s
        self.tokens_per_s = tokens_per_s
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

    def budget(self, seconds: float) -> int:
        """Output tokens for ``seconds`` of audio: stops a model that repeats itself long before ``max_new_tokens``."""
        return max(16, min(self.max_new_tokens, int(seconds * self.tokens_per_s) + 32))

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        out: list[STTResult] = []
        for r in requests:
            if r.task == "translate":
                raise InvalidRequest("SOV-Transcribe-1 transcribes only; it does not translate")
            prompt = instruction(r.language, r.hotwords)
            audio = np.ascontiguousarray(r.samples, dtype=np.float32)
            if len(audio) < int(0.05 * SR):
                out.append(STTResult("", r.language))
                continue
            with self._lock:
                text = self.backend.generate(audio, prompt, self.budget(len(audio) / SR), float(r.temperature or 0.0))
            out.append(STTResult(text.strip(), r.language))
        return out


def load(ctx: LoadContext) -> SovTranscribe:
    path = model_dir(ctx.path, ctx.options)
    trust = remote_code.trust_remote_code(ctx, path)
    transformers = require("transformers", "sov-transcribe")
    model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
    if model_cls is None:
        raise MissingDependency(
            f"transformers {getattr(transformers, '__version__', '?')} has no AutoModelForMultimodalLM; "
            'SOV-Transcribe-1 needs transformers 5.19 to 5.22: pip install "bolo-torch[sov-transcribe]"'
        )
    processor = transformers.AutoProcessor.from_pretrained(str(path), trust_remote_code=trust)
    split = split_kwargs(ctx, model_cls, path, trust_remote_code=trust)
    model = model_cls.from_pretrained(str(path), trust_remote_code=trust, dtype=ctx.dtype, **split)
    if not split:
        model = model.to(ctx.device)
    model.eval()
    out = SovTranscribe(
        TransformersBackend(model, processor, ctx.device),
        max_chunk_s=float(ctx.option("max_chunk_s", 30.0)),
        tokens_per_s=float(ctx.option("tokens_per_s", 12.0)),
        max_new_tokens=int(ctx.option("max_new_tokens", 8192)),
    )
    if split:
        from bolo_torch.runtime import device_map_report

        out.placement_report = device_map_report(model)
    return out
