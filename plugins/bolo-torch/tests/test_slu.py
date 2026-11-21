"""Spoken intent (task ``intent``): SpeechBrain SLU decoding with stand-in modules, the ``intent`` op, and the
encoder wrappers built from their configuration only."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from bolo.plugin.protocol import audio_bytes, read_message

from bolo_torch.models.slu import SpeechBrainSlu, config_only_encoders
from bolo_torch.server import TorchServer

torch = pytest.importorskip("torch")

SLURP = "{'scenario': 'alarm', 'action': 'set', 'entities': [{'type': 'time', 'filler': 'seven am'}]}"


class _Tokenizer:
    def decode_ids(self, ids: list[int]) -> str:
        assert ids == [5, 6, 7]
        return SLURP


def _model(seen: dict[str, Any]) -> Any:
    def hubert(wavs: Any) -> Any:
        seen["wavs"] = tuple(wavs.shape)
        return torch.zeros(1, 4, 8)

    def beam_searcher(encoded: Any, lens: Any) -> Any:
        seen["lens"] = lens.tolist()
        return [[5, 6, 7]], None, None, None

    return SimpleNamespace(
        mods=SimpleNamespace(hubert=hubert, beam_searcher=beam_searcher),
        hparams=SimpleNamespace(tokenizer=_Tokenizer()),
    )


def test_the_semantics_text_becomes_an_intent() -> None:
    seen: dict[str, Any] = {}
    model = SpeechBrainSlu(_model(seen), "cpu", 12.0)
    result = model.understand(np.zeros(3200, np.float32))
    assert seen == {"wavs": (1, 3200), "lens": [1.0]}
    assert (result.intent, result.scenario, result.action) == ("alarm_set", "alarm", "set")
    assert result.slots == [{"type": "time", "value": "seven am"}] and result.text == SLURP
    assert model.max_audio_s == 12.0


def test_server_answers_the_intent_op() -> None:
    class Engine:
        def load(self, *args: object) -> SpeechBrainSlu:
            return SpeechBrainSlu(_model({}), "cpu", 20.0)

    out = io.BytesIO()
    server = TorchServer(Engine(), io.BytesIO(), out)  # type: ignore[arg-type]
    server.handle("1", "load", {"kind": "speechbrain-slu", "task": "intent", "path": "/x", "options": {}}, b"")
    server.handle("2", "intent", {"handle": "h1"}, audio_bytes(np.zeros(1600, np.float32)))
    out.seek(0)
    load, _ = read_message(out)
    assert load["ok"] and load["info"]["max_s"] == 20.0
    reply, _ = read_message(out)
    assert reply["ok"] and reply["intent"]["intent"] == "alarm_set", json.dumps(reply)
    assert reply["intent"]["slots"] == [{"type": "time", "value": "seven am"}]


def test_encoders_are_built_from_their_configuration() -> None:
    """Inside the context the wrapper's network comes from its configuration (no weights are downloaded); after it,
    SpeechBrain's own loader is back."""
    pytest.importorskip("transformers")
    huggingface = pytest.importorskip("speechbrain.integrations.huggingface.huggingface")
    interface = huggingface.HFTransformersInterface
    original = interface._from_pretrained
    built: list[Any] = []

    class AutoClass:
        @staticmethod
        def from_config(config: Any) -> Any:
            built.append(config)
            return torch.nn.Linear(1, 1)

    wrapper = SimpleNamespace(auto_class=AutoClass, config={"hidden_size": 8})
    with config_only_encoders():
        interface._from_pretrained(wrapper, "facebook/hubert-base-ls960", "save", "cache", device="cpu")
    assert built == [{"hidden_size": 8}] and isinstance(wrapper.model, torch.nn.Linear)
    assert interface._from_pretrained is original
