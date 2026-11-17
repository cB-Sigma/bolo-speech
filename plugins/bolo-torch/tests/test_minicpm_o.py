"""MiniCPM-o 2.6 with a fake of its remote-code model: the card's audio-assistant prompt, the streaming API with the
conversation prefilled, chat mode and text-only answers."""

from __future__ import annotations

import numpy as np
import pytest
from bolo.engines.base import ConversationRequest, ConversationTurn

from bolo_torch.models.minicpm_o import MiniCPMoS2S
from bolo_torch.runtime import InvalidRequest
from s2s_fakes import MiniCPMoFake


def spoken(seconds: float) -> ConversationTurn:
    return ConversationTurn("user", audio=np.zeros(int(seconds * 16000), dtype=np.float32))


def minicpm(mode: str = "stream") -> tuple[MiniCPMoS2S, MiniCPMoFake]:
    fake = MiniCPMoFake()
    voices = {"female": np.ones(10, dtype=np.float32), "male": np.zeros(10, dtype=np.float32)}
    return MiniCPMoS2S(fake, object(), voices, mode=mode, max_new_tokens=256), fake


def test_streaming_prefills_the_conversation() -> None:
    model, fake = minicpm()
    request = ConversationRequest(
        [spoken(2.5), ConversationTurn("assistant", text="Hello."), ConversationTurn("user", text="And now?")],
        instructions="Be brief.", voice="Male", max_output_tokens=1000,
    )  # fmt: skip
    chunks = list(model.respond(request))
    assert "".join(c.text for c in chunks) == "part0 part1 part2 " and all(c.audio is not None for c in chunks)
    sessions = {s for s, _ in fake.prefilled}
    assert len(sessions) == 1 and fake.resets == 1 and fake.generate_calls[0]["session_id"] in sessions
    msgs = [m for _, m in fake.prefilled]
    assert msgs[0]["content"][-1] == "Be brief." and msgs[0]["content"][1] is model.references["male"]
    assert [len(m["content"][0]) for m in msgs[1:4]] == [16000, 16000, 8000]  # speech one second at a time
    assert msgs[4] == {"role": "assistant", "content": ["<|im_end|>\n<|im_start|>assistant\nHello.<|im_end|>\n"]}
    assert msgs[5] == {"role": "user", "content": ["And now?"]}
    assert fake.generate_calls[0]["max_new_tokens"] == 256 and fake.generate_calls[0]["generate_audio"] is True
    assert model.sample_rate == 24000 and [v.id for v in model.voices] == ["female", "male"]


def test_chat_mode_and_text_only() -> None:
    model, fake = minicpm("chat")
    chunks = list(model.respond(ConversationRequest([spoken(1.0)])))
    assert len(chunks) == 1 and chunks[0].text == "whole answer" and chunks[0].audio is not None
    assert float(np.max(np.abs(chunks[0].audio))) <= 1.0
    call = fake.chats[0]
    assert call["use_tts_template"] is True and call["generate_audio"] is True and call["sampling"] is True
    assert [m["role"] for m in call["msgs"]] == ["user", "user"] and isinstance(
        call["msgs"][1]["content"][0], np.ndarray
    )
    stream_model, stream_fake = minicpm()
    text_only = list(stream_model.respond(ConversationRequest([ConversationTurn("user", text="hi")], audio_out=False)))
    assert (
        text_only[0].audio is None and stream_fake.chats[0]["generate_audio"] is False and stream_fake.prefilled == []
    )


def test_voices_and_cancel() -> None:
    model, fake = minicpm()
    with pytest.raises(InvalidRequest):
        list(model.respond(ConversationRequest([spoken(0.2)], voice="alloy")))
    assert list(model.respond(ConversationRequest([spoken(0.2)]), lambda: True)) == []
    seen = []
    for chunk in model.respond(ConversationRequest([spoken(0.2)]), lambda: len(seen) >= 1):
        seen.append(chunk)
    assert len(seen) == 1
    with pytest.raises(InvalidRequest):
        MiniCPMoS2S(fake, object(), {"female": np.zeros(1)}, mode="duplex")


def test_load_needs_the_opt_in_and_the_assets(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import json
    import types

    from bolo_torch.models import minicpm_o
    from bolo_torch.remote_code import RemoteCodeRefused, pins
    from bolo_torch.runtime import LoadContext

    rev = "c" * 40
    root = tmp_path / "model"
    (root / "assets" / "input_examples").mkdir(parents=True)
    (root / "assets" / "chattts_tokenizer").mkdir()
    (root / "manifest.json").write_text(json.dumps({"source": {"revision": rev}}))
    for name in ("modeling_minicpmo.py", "processing_minicpmo.py"):
        (root / name).write_text("")
    for rel in minicpm_o.VOICES.values():
        (root / rel).write_bytes(b"wav")
    fake = MiniCPMoFake()
    calls: list[dict] = []
    fake.to = lambda device: fake  # type: ignore[attr-defined]
    fake.eval = lambda: fake  # type: ignore[attr-defined]
    fake.init_tts = lambda: calls.append({"init_tts": True})  # type: ignore[attr-defined]
    fake.tts = types.SimpleNamespace(float=lambda: calls.append({"tts_float": True}))  # type: ignore[attr-defined]

    def from_pretrained(path: str, **kw):  # type: ignore[no-untyped-def]
        calls.append(kw)
        return fake

    fake_tf = types.SimpleNamespace(
        AutoModel=types.SimpleNamespace(from_pretrained=from_pretrained),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path, **kw: calls.append(kw) or "tok"),
    )
    fake_librosa = types.SimpleNamespace(load=lambda path, sr, mono: (np.zeros(sr, dtype=np.float32), sr))
    monkeypatch.setattr(minicpm_o, "require", lambda module, extra: fake_librosa if module == "librosa" else fake_tf)
    ctx = LoadContext("minicpm-o", "speech-to-speech", root, {"remote_code": pins(root, rev)}, device="cpu")
    with pytest.raises(RemoteCodeRefused, match="allow_remote_code"):
        minicpm_o.load(ctx)
    ctx.options["allow_remote_code"] = True
    with pytest.raises(InvalidRequest, match=r"Vocos\.pt"):
        minicpm_o.load(ctx)
    (root / "assets" / "Vocos.pt").write_bytes(b"x")
    (root / "assets" / "chattts_tokenizer" / "tokenizer.json").write_text("{}")
    model = minicpm_o.load(ctx)
    assert (
        calls[0]["trust_remote_code"] is True
        and calls[0]["init_tts"] is True
        and calls[0]["attn_implementation"] == "sdpa"
    )
    assert (
        {"init_tts": True} in calls and {"tts_float": True} in calls and set(model.references) == set(minicpm_o.VOICES)
    )
