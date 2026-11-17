"""Stand-ins for the model classes and inference code of the token-streaming speech-to-speech models.

Each fake has the entry points the plugin calls on the real object, with the same names and arguments
(transformers' ``generate`` with ``streamer`` and ``stopping_criteria``, Step-Audio 2's ``utils`` functions and
``Token2wav.stream``, GLM-4-Voice's ``AudioDecoder.token2wav``, Kimi-Audio's ``KimiAudio`` pieces, MiniCPM-o's
``streaming_prefill``/``streaming_generate``/``chat``). Torch is the CPU build of the test environment."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class FakeGenerator:
    """``generate(**kwargs, streamer=..., stopping_criteria=...)``: the prompt, then ``script`` token by token."""

    def __init__(self, script: list[int]) -> None:
        self.script = script
        self.calls: list[dict[str, Any]] = []
        self.produced = 0

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        streamer = kwargs["streamer"]
        prompt = kwargs["input_ids"]
        streamer.put(prompt)
        ids = prompt
        self.produced = 0
        for token in self.script:
            stops = kwargs.get("stopping_criteria") or []
            if any(bool(torch.as_tensor(c(ids, None)).all()) for c in stops):
                break
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
            streamer.put(torch.tensor([token]))
            self.produced += 1
        streamer.end()  # transformers' generate ends the streamer itself
        return ids


class StepTokenizer:
    """Step-Audio 2's tokenizer: one id per prompt piece plus one per ``<audio_patch>``."""

    pad_token_id = 151643
    words = {1: "Hello", 2: " there", 3: ".", 4: "function", 5: "\n", 6: "get_time", 7: '{"zone": "UTC"}'}
    specials = {151657: "<tool_call>", 151658: "</tool_call>", 151665: "<|EOT|>"}

    def __init__(self) -> None:
        self.pieces: list[str] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|EOT|>": 151665}[token]

    def __call__(self, text: str, return_tensors: str = "pt", padding: bool = True) -> dict[str, Any]:
        self.pieces.append(text)
        n = 1 + text.count("<audio_patch>")
        return {"input_ids": torch.zeros((1, n), dtype=torch.long)}

    def decode(self, ids: list[int]) -> str:
        return "".join(self.words.get(i, self.specials.get(i, "")) for i in ids)


class StepUtils:
    """The functions of Step-Audio2's ``utils.py`` the plugin uses."""

    def __init__(self) -> None:
        self.pieces: list[int] = []

    def log_mel_spectrogram(self, audio: torch.Tensor, n_mels: int = 128, padding: int = 479) -> torch.Tensor:
        assert n_mels == 128 and padding == 479 and isinstance(audio, torch.Tensor)
        self.pieces.append(int(audio.shape[0]))
        return torch.zeros((128, (audio.shape[0] + padding) // 160 + 2))

    def compute_token_num(self, n: int) -> int:
        return max(1, (n - 2) // 8)

    def padding_mels(self, mels: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        longest = max(m.shape[1] for m in mels)
        out = torch.zeros((len(mels), 128, longest))
        for i, m in enumerate(mels):
            out[i, :, : m.shape[1]] = m
        return out, torch.tensor([m.shape[1] - 2 for m in mels], dtype=torch.int32)


class StepToken2wav:
    """``Token2wav``: ``set_stream_cache(prompt_wav)`` then ``stream(tokens, prompt_wav, last_chunk)`` -> int16 PCM."""

    class Flow:
        pre_lookahead_len = 3

    def __init__(self) -> None:
        self.flow = self.Flow()
        self.calls: list[tuple[int, str, bool]] = []
        self.cached: list[str] = []

    def set_stream_cache(self, prompt_wav: str) -> None:
        self.cached.append(prompt_wav)

    def stream(self, tokens: list[int], prompt_wav: str, last_chunk: bool = False) -> bytes:
        assert all(0 <= t < 6561 for t in tokens)
        self.calls.append((len(tokens), prompt_wav, last_chunk))
        return (np.full(len(tokens) * 960, 1000, dtype="<i2")).tobytes()  # 25 Hz tokens at 24 kHz


def step_model(script: list[int]) -> tuple[Any, FakeGenerator, StepTokenizer, StepUtils, StepToken2wav]:
    from bolo_torch.models.step_audio import StepAudio2S2S

    llm, tok, utils, t2w = FakeGenerator(script), StepTokenizer(), StepUtils(), StepToken2wav()
    model = StepAudio2S2S(
        llm, tok, utils, t2w, {"female": "/code/assets/default_female.wav", "male": "/code/assets/default_male.wav"},
        generation_config=dict, device="cpu", max_new_tokens=512,
    )  # fmt: skip
    return model, llm, tok, utils, t2w
