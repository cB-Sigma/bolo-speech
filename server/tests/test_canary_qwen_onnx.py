"""Canary-Qwen 2.5B in ONNX Runtime (bolo.engines.canary_qwen_onnx), with stand-ins for the three graphs: the prompt
around the audio, the causal mask, the cache that grows by one position per token, and the byte-level BPE text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bolo.engines import canary_qwen_onnx as cq
from bolo.engines.base import LoadOptions, STTRequest

LAYERS, HEADS, DIM, HIDDEN = 2, 1, 4, 3
VOCAB = {"Hello": 10, "Ġworld": 11, ".": 12, "Ġcaf": 13, "Ã©": 14, "<|im_end|>": 99}
ANSWER = [10, 11, 13, 14, 12, 99]  # "Hello world café." then end of turn


class _Input:
    def __init__(self, name: str, type_: str = "tensor(float)") -> None:
        self.name, self.type = name, type_


class _Graph:
    def __init__(self, role: str, log: list[tuple[str, dict[str, Any]]]) -> None:
        self.role, self.log, self.step = role, log, 0

    def get_inputs(self) -> list[_Input]:
        return [_Input("inputs_embeds" if self.role == "decoder" else "x")]

    def run(self, _names: Any, feed: dict[str, Any]) -> list[Any]:
        self.log.append((self.role, feed))
        if self.role == "encoder":
            frames = feed["input_features"].shape[2]
            n = frames // 8
            return [np.ones((1, n + 2, HIDDEN), np.float32), np.asarray([n], np.int64)]  # 2 padding frames
        if self.role == "embed_tokens":
            ids = feed["input_ids"]
            return [np.repeat(ids[..., None].astype(np.float32), HIDDEN, axis=2)]
        s = feed["inputs_embeds"].shape[1]
        past = feed["past_key_values.0.key"].shape[2]
        logits = np.zeros((1, s, 100), np.float32)
        logits[0, -1, ANSWER[self.step]] = 1.0
        self.step += 1
        present = [np.zeros((1, HEADS, past + s, DIM), np.float32) for _ in range(2 * LAYERS)]
        return [logits, *present]


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    config = {
        "num_layers": LAYERS, "num_key_value_heads": HEADS, "head_dim": DIM, "eos_token_ids": [99],
        "prompt_prefix_ids": [1, 2], "prompt_suffix_ids": [3],
    }  # fmt: skip
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "vocab.json").write_text(json.dumps(VOCAB), encoding="utf-8")
    return tmp_path


def test_greedy_decoding_with_prompt_mask_and_cache(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[tuple[str, dict[str, Any]]] = []
    roles: list[str] = []

    def fake_session(path: str, _options: Any, _providers: Any, role: str | None = None) -> _Graph:
        roles.append(f"{role}:{Path(path).name}")
        return _Graph(str(role), log)

    monkeypatch.setattr(cq, "create_session", fake_session)
    model = cq.CanaryQwenOnnx(model_dir, {}, LoadOptions(threads=2))
    assert roles == ["encoder:encoder_int8.onnx", "embed_tokens:embed_tokens_int8.onnx", "decoder:decoder_int8.onnx"]

    (result,) = model.transcribe([STTRequest(np.random.default_rng(0).standard_normal(16000).astype(np.float32))])
    assert (result.text, result.language) == ("Hello world café.", "en")

    encoder_feed = next(f for r, f in log if r == "encoder")
    assert encoder_feed["input_features"].shape == (1, 128, 100)
    decoder_feeds = [f for r, f in log if r == "decoder"]
    first = decoder_feeds[0]
    audio_frames = 100 // 8
    s = 2 + audio_frames + 1  # prefix, audio without the encoder's padding frames, suffix
    assert first["inputs_embeds"].shape == (1, s, HIDDEN)
    assert first["attn_bias"].shape == (1, 1, s, s)
    assert first["attn_bias"][0, 0, 0, 1] < -1e30 and first["attn_bias"][0, 0, 1, 0] == 0  # causal
    assert first["position_ids"].tolist() == [list(range(s))]
    second = decoder_feeds[1]
    assert second["inputs_embeds"].shape == (1, 1, HIDDEN)
    assert second["position_ids"].tolist() == [[s]]
    assert second["attn_bias"].shape == (1, 1, 1, s + 1) and not second["attn_bias"].any()
    assert second["past_key_values.1.value"].shape == (1, HEADS, s, DIM)
    assert len(decoder_feeds) == len(ANSWER)  # stops at the end-of-turn token


def test_short_audio_and_feature_normalisation(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cq, "create_session", lambda *a, role=None, **k: _Graph(str(role), []))
    model = cq.CanaryQwenOnnx(model_dir, {}, LoadOptions())
    (empty,) = model.transcribe([STTRequest(np.zeros(100, np.float32))])
    assert empty.text == "" and empty.segments == []
    mel = np.random.default_rng(1).standard_normal((50, 4)) * 3 + 7
    norm = cq.normalize_per_feature(mel)
    assert np.allclose(norm.mean(axis=0), 0, atol=1e-5) and np.allclose(norm.std(axis=0, ddof=1), 1, atol=1e-3)
