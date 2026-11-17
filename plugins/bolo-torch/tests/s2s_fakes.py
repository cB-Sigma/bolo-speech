"""Stand-ins for the model classes and inference code of the token-streaming speech-to-speech models.

Each fake has the entry points the plugin calls on the real object, with the same names and arguments
(transformers' ``generate`` with ``streamer`` and ``stopping_criteria``, Step-Audio 2's ``utils`` functions and
``Token2wav.stream``, GLM-4-Voice's ``AudioDecoder.token2wav``, Kimi-Audio's ``KimiAudio`` pieces, MiniCPM-o's
``streaming_prefill``/``streaming_generate``/``chat``). Torch is the CPU build of the test environment."""

from __future__ import annotations

import time
from pathlib import Path
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
            time.sleep(0.002)  # a token takes time on a real model: a cancel can arrive mid-generation
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


class KimiContent:
    """``KimiAContent``: ``to_tensor()`` and ``continuous_feature``."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.continuous_feature = [torch.zeros((1, 4, 5120))]

    def to_tensor(self) -> tuple[torch.Tensor, ...]:
        ids = torch.zeros((1, self.n), dtype=torch.long)
        mask = torch.zeros((1, self.n), dtype=torch.bool)
        return ids, ids.clone(), mask, mask, mask


class KimiFake:
    """``KimiAudio``: ``alm``, ``prompt_manager``, ``detokenizer``, ``extra_tokens``, the delay and the token offset.
    Each step of ``alm.forward`` produces the next (text, speech) pair of ``script``."""

    offset = 1000
    words = {1: "Of", 2: " course", 3: "."}

    def __init__(self, script: list[tuple[int, int]], prompt: int = 20) -> None:
        import types

        self.script = script
        self.extra_tokens = types.SimpleNamespace(msg_end=7, media_end=15, kimia_text_blank=18, kimia_text_eos=19)
        self.kimia_text_audiodelaytokens = 2
        self.kimia_token_offset = self.offset
        self.forwards: list[dict[str, Any]] = []
        self.chats: list[list[dict[str, Any]]] = []
        self.detokenized: list[tuple[int, bool, int]] = []
        self.cleared = 0
        fake = self

        class PromptManager:
            class text_tokenizer:
                @staticmethod
                def decode(ids: list[int]) -> str:
                    return "".join(fake.words.get(i, "") for i in ids)

            @staticmethod
            def get_prompt(chats: list[dict[str, Any]], output_type: str = "text") -> KimiContent:
                for c in chats:
                    if c["message_type"] == "audio":
                        assert Path(c["content"]).is_file()
                fake.chats.append(chats)
                fake.output_type = output_type
                return KimiContent(prompt)

        class Alm:
            @staticmethod
            def forward(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, Any]:
                step = len(fake.forwards)
                fake.forwards.append(kwargs)
                return torch.tensor([step]), torch.tensor([step]), ("past", step)

        class Detokenizer:
            @staticmethod
            def clear_states() -> None:
                fake.cleared += 1

            @staticmethod
            def detokenize_streaming(
                tokens: torch.Tensor, is_final: bool = False, upsample_factor: int = 1
            ) -> torch.Tensor:
                fake.detokenized.append((int(tokens.shape[1]), is_final, upsample_factor))
                return torch.full((1, int(tokens.shape[1]) * 1920), 0.2)

        self.prompt_manager, self.alm, self.detokenizer = PromptManager(), Alm(), Detokenizer()

    def sampler(self, **kwargs: Any) -> Any:
        """``KimiASampler(**infer.py's sampling)``: the scripted tokens of each step."""
        fake = self
        fake.sampling = kwargs

        class Sampler:
            def sample_text_logits(self, logits: torch.Tensor, recent_tokens: Any = None) -> torch.Tensor:
                step = int(logits[0])
                return torch.tensor([fake.script[min(step, len(fake.script) - 1)][0]])

            def sample_audio_logits(self, logits: torch.Tensor, recent_tokens: Any = None) -> torch.Tensor:
                step = int(logits[0])
                return torch.tensor([fake.script[min(step, len(fake.script) - 1)][1]])

        return Sampler()


class MiniCPMoFake:
    """``MiniCPMO`` (remote code): ``get_sys_prompt``, ``reset_session``, ``streaming_prefill``,
    ``streaming_generate`` (``OmniOutput`` pieces) and ``chat``."""

    def __init__(self, pieces: int = 3) -> None:
        import types

        self.pieces = pieces
        self.prefilled: list[tuple[str, dict[str, Any]]] = []
        self.generated: list[dict[str, Any]] = []
        self.chats: list[dict[str, Any]] = []
        self.resets = 0
        self.out = types.SimpleNamespace

    def get_sys_prompt(self, ref_audio: Any = None, mode: str = "default", language: str = "zh") -> dict[str, Any]:
        assert mode == "audio_assistant" and language == "en" and isinstance(ref_audio, np.ndarray)
        return {"role": "user", "content": ["Use the voice in the audio prompt to synthesize new content.", ref_audio,
                                            "You are a helpful assistant with the above voice style."]}  # fmt: skip

    def reset_session(self) -> None:
        self.resets += 1

    def streaming_prefill(self, session_id: str, msgs: list[dict[str, Any]], tokenizer: Any, **kwargs: Any) -> None:
        assert len(msgs) == 1
        self.prefilled.append((session_id, msgs[0]))

    def streaming_generate(self, session_id: str, tokenizer: Any, **kwargs: Any) -> Any:
        self.generated.append({"session_id": session_id, **kwargs})
        for i in range(self.pieces):
            text = f"part{i} " + ("<|tts_eos|>" if i == self.pieces - 1 else "")
            yield self.out(text=text, audio_wav=np.full(24000 // 2, 0.3, dtype=np.float32), sampling_rate=24000)

    def chat(self, **kwargs: Any) -> Any:
        self.chats.append(kwargs)
        wav = np.full(24000, 3.0, dtype=np.float32) if kwargs["generate_audio"] else None
        return self.out(text="whole answer<|tts_eos|>", audio_wav=wav, sampling_rate=24000)
