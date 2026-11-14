"""Indic-Speak (Bodhan AI): a Llama-3.2-3B speech model writing SNAC codes, decoded by a fine-tuned Vocos.

The repository is gated (Indic Open Model License v1.0) and ships its own ``inference.py`` with a ``TTS`` class;
its model card documents the call this loader makes::

    from inference import TTS
    tts = TTS("<model directory>")
    wav = tts(text, speaker="Amit", style="ANGER", temperature=0.6, top_p=0.9, top_k=50, seed=None)  # 24 kHz

The repository's ``inference.py`` is imported from the downloaded model directory (it is not a library on PyPI).
A voice is a speaker name the model saw in training (two per language, plus the English list): an unknown name
does not fail, it gives an averaged voice, so unknown names are refused here. ``options.style`` (or a voice id
``<speaker>:<STYLE>``) picks an expressive style.
"""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from bolo.engines.base import FloatArray, TTSModel

from bolo_torch.models import peak_safe, to_voices, voices_from_options
from bolo_torch.runtime import InvalidRequest, LoadContext, model_dir, require
from bolo_torch.text import split_sentences

SAMPLE_RATE = 24000

Generate = Callable[[str, str, str | None], Any]


class IndicSpeakTTS(TTSModel):
    def __init__(
        self,
        generate: Generate,
        voices: list[dict[str, Any]],
        *,
        style: str | None = None,
        max_chars: int = 200,
    ) -> None:
        if not voices:
            raise InvalidRequest("Indic-Speak needs its speaker names (catalog voices)")
        self.generate = generate
        self._speakers = {str(v["id"]).lower(): str(v.get("speaker") or v.get("name") or v["id"]) for v in voices}
        self.voices = to_voices(voices)
        self.sample_rate = SAMPLE_RATE
        self.style = style
        self.max_chars = max_chars
        self._lock = threading.Lock()

    def speaker(self, voice: str | None) -> tuple[str, str | None]:
        """``(speaker name, style)`` for a voice id (``amit`` or ``amit:anger``)."""
        if not voice:
            return next(iter(self._speakers.values())), self.style
        name, _, style = voice.strip().partition(":")
        speaker = self._speakers.get(name.lower())
        if speaker is None:
            raise InvalidRequest(f"unknown voice {voice!r}: Indic-Speak knows {', '.join(sorted(self._speakers))}")
        return speaker, (style.upper() or None) if style else self.style

    def synthesize(
        self, text: str, voice: str | None = None, speed: float = 1.0, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[FloatArray]:
        if speed != 1.0:
            raise InvalidRequest("Indic-Speak speaks at one speed only; send speed 1.0")
        speaker, style = self.speaker(voice)
        for sentence in split_sentences(text, self.max_chars):
            if cancelled is not None and cancelled():
                return
            with self._lock:
                audio = self.generate(sentence, speaker, style)
            if hasattr(audio, "detach"):
                audio = audio.detach().float().cpu().numpy()
            out = np.asarray(audio, dtype=np.float32).reshape(-1)
            if out.size:
                yield peak_safe(out)


def load(ctx: LoadContext) -> IndicSpeakTTS:
    require("snac", "indic-speak")
    require("transformers", "indic-speak")
    root = model_dir(ctx.path, ctx.options)
    if not (root / "inference.py").is_file():
        raise InvalidRequest(f"{root} does not look like the Indic-Speak repository (inference.py is missing)")
    if sys.path[:1] != [str(root)]:
        sys.path.insert(0, str(root))
    sys.modules.pop("inference", None)  # the repository's own module, not another one of that name
    tts = importlib.import_module("inference").TTS(str(root))
    sampling: dict[str, Any] = {
        k: ctx.options[k] for k in ("temperature", "top_p", "top_k", "max_new_tokens") if k in ctx.options
    }

    def generate(text: str, speaker: str, style: str | None) -> Any:
        kwargs: dict[str, Any] = {"speaker": speaker, **sampling}
        if style:
            kwargs["style"] = style
        return tts(text, **kwargs)

    return IndicSpeakTTS(
        generate, voices_from_options(ctx.options), style=ctx.option("style"),
        max_chars=int(ctx.option("max_chars", 200)),
    )  # fmt: skip
