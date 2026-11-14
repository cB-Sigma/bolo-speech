"""Stand-ins for the model libraries: the plugin's logic runs without downloading anything."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from bolo_torch.models.dicow import Backend, DecodedSegment, DiCoW
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


# -- DiCoW ---------------------------------------------------------------------------------------------


class FakeDiCoWBackend(Backend):
    """Records the STNO masks it is given; "decodes" one segment per stretch where the target talks (alone or in
    overlap), with the text ``spk<i> <start>-<end>``. Masks are 30 s long for short input, as Whisper pads."""

    def __init__(self, use_enrollments: bool = False) -> None:
        self.calls: list[tuple[np.ndarray, str | None, str, list[int] | None]] = []
        self.use_enrollments = use_enrollments

    def mask_frames(self, num_samples: int) -> int:
        return max(1500, num_samples // 320)

    def decode(
        self, audio: np.ndarray, masks: np.ndarray, language: str | None, task: str, enrol_starts: list[int] | None
    ) -> list[list[DecodedSegment]]:
        self.calls.append((masks.copy(), language, task, enrol_starts))
        out: list[list[DecodedSegment]] = []
        for i, mask in enumerate(masks):
            on = np.concatenate([[0], ((mask[1] + mask[3]) > 0.5).astype(np.int8), [0]])
            edges = np.flatnonzero(np.diff(on))
            out.append(
                [
                    DecodedSegment(a / 50, b / 50, f" spk{i} {a / 50:.2f}-{b / 50:.2f}")
                    for a, b in zip(edges[0::2], edges[1::2])
                ]
            )
        return out


def load_fake_dicow(ctx: LoadContext) -> DiCoW:
    return DiCoW(FakeDiCoWBackend())
