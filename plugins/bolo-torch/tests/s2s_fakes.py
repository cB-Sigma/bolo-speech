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


class Encoded(dict):  # type: ignore[type-arg]
    """transformers' BatchEncoding / BatchFeature: a dict with ``.to``."""

    def to(self, device: Any = None, **kwargs: Any) -> Encoded:
        return self


class GlmTokenizer:
    """ChatGLM4Tokenizer: ``<|audio_N|>`` from 152353, ``<|user|>`` 151336; words 1-3."""

    audio_0 = 152353
    words = {1: "Sure", 2: ", here", 3: " it is."}

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|audio_0|>": self.audio_0, "<|user|>": 151336}[token]

    def __call__(self, texts: list[str], return_tensors: str = "pt") -> Encoded:
        self.prompts.extend(texts)
        return Encoded(input_ids=torch.zeros((1, 5), dtype=torch.long), attention_mask=torch.ones((1, 5)))

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        assert kwargs == {"spaces_between_special_tokens": False}
        return "".join(self.words.get(i, "") for i in ids)


class WhisperVQ:
    """``WhisperVQEncoder``: strides 1 and 2, pooling 4; one token per 8 feature frames."""

    class Conv:
        def __init__(self, stride: int) -> None:
            self.stride = (stride,)

    def __init__(self) -> None:
        self.config = type("C", (), {"pooling_kernel_size": 4})()
        self.conv1, self.conv2 = self.Conv(1), self.Conv(2)

    def __call__(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        n = attention_mask.shape[1] // 8
        return type("O", (), {"quantized_token_ids": torch.arange(n).repeat(attention_mask.shape[0], 1) + 10})()


class WhisperFeatures:
    """``WhisperFeatureExtractor``: 100 frames a second, padded to the longest and to ``pad_to_multiple_of``."""

    hop_length = 160

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, pieces: list[np.ndarray], **kwargs: Any) -> Encoded:
        self.calls.append(kwargs)
        frames = [len(p) // 160 for p in pieces]
        multiple = kwargs["pad_to_multiple_of"] // self.hop_length
        longest = -(-max(frames) // multiple) * multiple
        mask = torch.zeros((len(pieces), longest), dtype=torch.long)
        for i, f in enumerate(frames):
            mask[i, :f] = 1
        return Encoded(input_features=torch.zeros((len(pieces), 128, longest)), attention_mask=mask)


class GlmDecoder:
    """``flow_inference.AudioDecoder.token2wav``: 22050 / 12.5 samples per token; the mel has 80 bins."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def token2wav(self, token: torch.Tensor, uuid: str, prompt_token: torch.Tensor, prompt_feat: torch.Tensor,
                  finalize: bool = False) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
        self.calls.append({"n": int(token.shape[1]), "prompt_token": int(prompt_token.shape[1]),
                           "prompt_feat": int(prompt_feat.shape[1]), "finalize": finalize, "uuid": uuid})  # fmt: skip
        n = int(token.shape[1])
        return torch.full((1, n * 1764), 0.1), torch.zeros((1, 80, n * 2))
