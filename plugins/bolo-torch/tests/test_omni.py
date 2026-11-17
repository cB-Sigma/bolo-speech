"""Qwen-Omni speech-to-speech with fake transformers objects: the chat template's input, voices, text and audio out."""

from __future__ import annotations

import numpy as np
import pytest
from bolo.engines.base import ConversationRequest, ConversationTurn

from bolo_torch.models.omni import FAMILIES, QWEN25_SYSTEM, conversation
from bolo_torch.runtime import InvalidRequest
from omni_fakes import omni


def spoken(seconds: float) -> ConversationTurn:
    return ConversationTurn("user", audio=np.zeros(int(seconds * 16000), dtype=np.float32))


def test_conversation_follows_the_model_card() -> None:
    fam = FAMILIES["qwen2.5-omni"]
    request = ConversationRequest(
        [spoken(1.0), ConversationTurn("assistant", text="Hello."), spoken(0.5), ConversationTurn("user", text="Hi")],
        instructions="Be brief.",
    )
    messages, audios = conversation(fam, request)
    assert messages[0] == {"role": "system", "content": [{"type": "text", "text": QWEN25_SYSTEM + "\n\nBe brief."}]}
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert [p["type"] for p in messages[3]["content"]] == ["audio", "text"]  # a turn's audio and text together
    assert [len(a) for a in audios] == [16000, 8000]
    # Qwen3-Omni needs no fixed system prompt; a tool result becomes a user message
    three, _ = conversation(FAMILIES["qwen3-omni"], ConversationRequest([ConversationTurn("tool", text="42")]))
    assert three == [{"role": "user", "content": [{"type": "text", "text": "(result of a tool call: 42)"}]}]


@pytest.mark.parametrize("family", ["qwen2.5-omni", "qwen3-omni"])
def test_respond_gives_text_and_24k_audio(family: str) -> None:
    model, fake, processor = omni(family)
    chunks = list(model.respond(ConversationRequest([spoken(1.0)], voice="chelsie", max_output_tokens=1000)))
    assert len(chunks) == 1 and chunks[0].text == "w7 w8 w9"
    audio = chunks[0].audio
    assert audio is not None and len(audio) == 12000 and float(np.max(np.abs(audio))) <= 1.0
    call = fake.calls[0]
    assert call["speaker"] == "Chelsie" and call["return_audio"] is True and call["use_audio_in_video"] is False
    assert call["thinker_max_new_tokens"] == 256  # the catalog's limit caps the request's
    assert processor.audio is not None and len(processor.audio[0]) == 16000
    assert model.sample_rate == 24000 and [v.id for v in model.voices] == list(FAMILIES[family].voices)
    if family == "qwen3-omni":
        assert call["thinker_return_dict_in_generate"] is True


def test_text_only_voices_and_cancel() -> None:
    model, fake, _ = omni()
    chunks = list(model.respond(ConversationRequest([ConversationTurn("user", text="hi")], audio_out=False)))
    assert chunks[0].audio is None and chunks[0].text and "speaker" not in fake.calls[-1]
    assert model.voice(None) == "Chelsie" and model.voice("ETHAN") == "Ethan"
    with pytest.raises(InvalidRequest):
        list(model.respond(ConversationRequest([spoken(0.2)], voice="alloy")))
    assert list(model.respond(ConversationRequest([spoken(0.2)]), lambda: True)) == []
