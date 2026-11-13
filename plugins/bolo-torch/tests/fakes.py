"""Stand-ins for the model libraries: the plugin's logic runs without downloading anything."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from bolo_torch.models.parler import ParlerTTS
from bolo_torch.models.separation import SepFormer
from bolo_torch.runtime import LoadContext, ModelGated, gated_message


class Encoded:
    def __init__(self, ids: list[int]) -> None:
        self.input_ids = torch.tensor([ids])
        self.attention_mask = torch.ones_like(self.input_ids)

    def to(self, device: str) -> Encoded:
        return self


class FakeTokenizer:
    """One token per character; remembers what it was asked to encode."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, text: str, return_tensors: str = "pt") -> Encoded:
        self.seen.append(text)
        return Encoded([ord(c) % 1000 for c in text])

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.seen.append(text)
        return [ord(c) % 1000 for c in text]


class FakeParlerModel:
    """Audio length proportional to the prompt: 100 samples per prompt token."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.calls.append(kwargs)
        n = int(kwargs["prompt_input_ids"].shape[-1]) * 100
        return torch.linspace(-0.5, 0.5, n).unsqueeze(0)


def fake_parler(voices: list[dict[str, Any]] | None = None, **kwargs: Any) -> ParlerTTS:
    return ParlerTTS(FakeParlerModel(), FakeTokenizer(), FakeTokenizer(), sample_rate=44100, voices=voices, **kwargs)


def fake_separate_batch(mix: torch.Tensor) -> torch.Tensor:
    """Two 'sources': the mixture scaled by 0.6 and by -0.4, shape (batch, time, speakers)."""
    return torch.stack([mix * 0.6, mix * -0.4], dim=-1)


# loaders used by the fake plugin process (tests/fake_plugin.py)
def load_fake_parler(ctx: LoadContext) -> ParlerTTS:
    return fake_parler(voices=list(ctx.options.get("voices") or []))


def load_fake_sepformer(ctx: LoadContext) -> SepFormer:
    return SepFormer(fake_separate_batch, int(ctx.options.get("model_sample_rate", 8000)))


def load_gated(ctx: LoadContext) -> ParlerTTS:
    raise ModelGated(gated_message("example/gated-model"))


def tone(seconds: float, rate: int = 16000, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
