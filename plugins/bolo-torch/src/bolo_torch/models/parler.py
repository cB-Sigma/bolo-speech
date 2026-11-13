"""Parler-TTS (Indic Parler-TTS, Parler-TTS Tiny/Mini/Large): text in, voice described in words."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from bolo.engines.base import FloatArray, TTSModel

from bolo_torch.models import peak_safe, to_voices, voices_from_options
from bolo_torch.runtime import LoadContext, ModelGated, PluginError, hub_snapshot, model_dir, require
from bolo_torch.text import split_sentences

log = logging.getLogger("bolo_torch.parler")

DEFAULT_TEMPLATE = (
    "{name}'s voice is clear and natural, at a moderate pace, in a very close recording that almost has no "
    "background noise. Very clear audio."
)
GENERIC_DESCRIPTION = (
    "A speaker delivers clear, natural speech at a moderate pace, in a very close recording that almost has no "
    "background noise. Very clear audio."
)


class ParlerTTS(TTSModel):
    """Streams sentence by sentence; with ``stream_steps`` also inside a sentence (Parler's own streamer)."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        description_tokenizer: Any,
        *,
        sample_rate: int,
        device: str = "cpu",
        voices: list[dict[str, Any]] | None = None,
        template: str = DEFAULT_TEMPLATE,
        max_chars: int = 200,
        seed: int | None = None,
        stream_steps: int = 0,
        streamer_factory: Callable[..., Any] | None = None,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.description_tokenizer = description_tokenizer
        self.sample_rate = sample_rate
        self.device = device
        self._voice_entries = {str(v["id"]).lower(): v for v in voices or []}
        self.voices = to_voices(voices or [])
        self.template = template
        self.max_chars = max_chars
        self.seed = seed
        self.stream_steps = stream_steps
        self.streamer_factory = streamer_factory
        self.generate_kwargs = generate_kwargs or {}
        self._lock = threading.Lock()

    # -- voice --------------------------------------------------------------
    def describe(self, voice: str | None, speed: float = 1.0) -> str:
        """The description for a voice id or name, or ``voice`` itself when it is a description."""
        entry: dict[str, Any] | None = None
        if voice:
            key = voice.strip().lower()
            entry = self._voice_entries.get(key) or next(
                (v for v in self._voice_entries.values() if str(v.get("name", "")).lower() == key), None
            )
            if entry is None and len(voice.split()) >= 3:
                text = voice.strip()
                return text + _speed_hint(speed)
            if entry is None:
                raise PluginError(f"unknown voice {voice!r}", "invalid_request")
        elif self._voice_entries:
            entry = next(iter(self._voice_entries.values()))
        if entry is None:
            return GENERIC_DESCRIPTION + _speed_hint(speed)
        text = str(entry.get("description") or self.template.format(name=entry.get("name") or entry["id"]))
        return text + _speed_hint(speed)

    # -- synthesis ----------------------------------------------------------
    def synthesize(
        self, text: str, voice: str | None = None, speed: float = 1.0, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[FloatArray]:
        import torch

        description = self.describe(voice, speed)
        d = self.description_tokenizer(description, return_tensors="pt").to(self.device)
        for sentence in split_sentences(text, self.max_chars):
            if cancelled is not None and cancelled():
                return
            p = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            kwargs = {
                "input_ids": d.input_ids, "attention_mask": d.attention_mask,
                "prompt_input_ids": p.input_ids, "prompt_attention_mask": p.attention_mask,
                **self.generate_kwargs,
            }  # fmt: skip
            with self._lock:
                if self.seed is not None:
                    torch.manual_seed(self.seed)
                if self.stream_steps > 0 and self.streamer_factory is not None:
                    yield from self._streamed(kwargs, cancelled)
                    continue
                with torch.inference_mode():
                    generation = self.model.generate(**kwargs)
            audio = generation.to(torch.float32).cpu().numpy().squeeze()
            if audio.size:
                yield peak_safe(audio)

    def _streamed(self, kwargs: dict[str, Any], cancelled: Callable[[], bool] | None) -> Iterator[FloatArray]:
        import torch

        assert self.streamer_factory is not None
        streamer = self.streamer_factory(self.model, device=self.device, play_steps=self.stream_steps)
        errors: list[BaseException] = []

        def run() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**kwargs, streamer=streamer)
            except BaseException as exc:  # reported to the caller below
                errors.append(exc)
                streamer.end()

        worker = threading.Thread(target=run, daemon=True, name="parler-generate")
        worker.start()
        for chunk in streamer:
            if cancelled is not None and cancelled():
                break
            audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if audio.size:
                yield peak_safe(audio)
        worker.join()
        if errors:
            raise errors[0]

    def close(self) -> None:
        self.model = None


def _speed_hint(speed: float) -> str:
    if speed >= 1.2:
        return " The speaker talks fast."
    if speed <= 0.8:
        return " The speaker talks slowly."
    return ""


def load(ctx: LoadContext) -> ParlerTTS:
    import torch

    parler = require("parler_tts", "parler")
    transformers = require("transformers", "parler")
    root = model_dir(ctx.path, ctx.options)
    model = parler.ParlerTTSForConditionalGeneration.from_pretrained(str(root), torch_dtype=ctx.dtype)
    model = model.to(ctx.device).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(root))
    description_tokenizer = _description_tokenizer(ctx, model, tokenizer, transformers)
    stream_s = float(ctx.option("stream_seconds", 0) or 0)
    frame_rate = int(getattr(model.audio_encoder.config, "frame_rate", 86) or 86)
    streamer_factory = None
    if stream_s > 0:
        streamer_factory = require("parler_tts", "parler").ParlerTTSStreamer
    seed = ctx.option("seed")
    gen_kwargs: dict[str, Any] = {}
    if ctx.option("max_new_tokens"):
        gen_kwargs["max_new_tokens"] = int(ctx.option("max_new_tokens"))
    torch.set_grad_enabled(False)
    return ParlerTTS(
        model, tokenizer, description_tokenizer,
        sample_rate=int(model.config.sampling_rate), device=ctx.device,
        voices=voices_from_options(ctx.options), template=str(ctx.option("description_template", DEFAULT_TEMPLATE)),
        max_chars=int(ctx.option("max_chars", 200)), seed=None if seed is None else int(seed),
        stream_steps=int(stream_s * frame_rate) if stream_s > 0 else 0, streamer_factory=streamer_factory,
        generate_kwargs=gen_kwargs,
    )  # fmt: skip


def _description_tokenizer(ctx: LoadContext, model: Any, tokenizer: Any, transformers: Any) -> Any:
    """Indic Parler-TTS reads the description with its text encoder's tokenizer (a separate repository)."""
    name = ctx.option("description_tokenizer")
    if name == "same":
        return tokenizer
    if not name:
        name = getattr(getattr(model.config, "text_encoder", None), "_name_or_path", None)
    if not name:
        return tokenizer
    local = Path(str(name))
    if local.is_dir():
        return transformers.AutoTokenizer.from_pretrained(str(local))
    try:
        path = hub_snapshot(str(name), allow_patterns=["*.json", "*.model", "*.txt"])
    except ModelGated:
        raise
    except PluginError as exc:
        log.warning("description tokenizer %s not available (%s); using the prompt tokenizer", name, exc)
        return tokenizer
    return transformers.AutoTokenizer.from_pretrained(str(path))
