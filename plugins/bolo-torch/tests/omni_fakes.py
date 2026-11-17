"""Stand-ins for transformers' Qwen-Omni model and processor (no torch needed)."""

from __future__ import annotations

from typing import Any

import numpy as np

from bolo_torch.models.omni import FAMILIES, OmniS2S
from bolo_torch.runtime import LoadContext


class Inputs(dict):  # type: ignore[type-arg]
    def to(self, device: str) -> Inputs:
        return self


class FakeProcessor:
    """The prompt is one token per message; remembers the template's messages and the audio."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.audio: list[np.ndarray] | None = None

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        assert kwargs == {"add_generation_prompt": True, "tokenize": False}
        self.messages = messages
        return "|".join(m["role"] for m in messages)

    def __call__(self, text: str, audio: list[np.ndarray] | None = None, **kwargs: Any) -> Inputs:
        self.audio = audio
        n = len(text.split("|"))
        out = Inputs(input_ids=np.zeros((1, n), dtype=np.int64))
        if audio:
            out["input_features"] = np.zeros((1, 128, 10), dtype=np.float32)
        return out

    def batch_decode(self, ids: np.ndarray, **kwargs: Any) -> list[str]:
        return [" ".join(f"w{int(i)}" for i in ids[0]) + " "]


class Sequences:
    def __init__(self, ids: np.ndarray) -> None:
        self.sequences = ids


class FakeOmni:
    """Answers three tokens after the prompt and, when asked, 0.5 s of audio at 24 kHz."""

    dtype = None

    def __init__(self, returns_dict: bool = False) -> None:
        self.returns_dict = returns_dict
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        prompt = kwargs["input_ids"]
        ids = np.concatenate([prompt, np.array([[7, 8, 9]])], axis=1)
        text: Any = Sequences(ids) if self.returns_dict else ids
        if not kwargs.get("return_audio"):
            return (text, None) if self.returns_dict else text
        return text, np.full((1, 12000), 2.0, dtype=np.float32)  # loud: the plugin scales it down


def omni(family: str = "qwen2.5-omni") -> tuple[OmniS2S, FakeOmni, FakeProcessor]:
    fam = FAMILIES[family]
    model, processor = FakeOmni(fam.returns_dict), FakeProcessor()
    return OmniS2S(model, processor, fam, max_new_tokens=256), model, processor


def load_fake_omni(ctx: LoadContext) -> OmniS2S:
    return omni(str(ctx.option("family", "qwen2.5-omni")))[0]
