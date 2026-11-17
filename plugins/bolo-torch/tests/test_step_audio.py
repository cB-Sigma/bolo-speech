"""Step-Audio 2 mini with fakes of its remote-code model, tokenizer, utils.py and Token2wav: the prompt, streamed
text, tool calls and speech, voices and cancel."""

from __future__ import annotations

import numpy as np
import pytest
from bolo.engines.base import ConversationRequest, ConversationTurn

from bolo_torch.models.step_audio import AUDIO_OFFSET, messages, prompt_pieces
from bolo_torch.runtime import InvalidRequest
from s2s_fakes import StepUtils, step_model

TOOL = {"type": "function", "name": "get_time", "description": "the time", "parameters": {"type": "object"}}


def spoken(seconds: float) -> ConversationTurn:
    return ConversationTurn("user", audio=np.zeros(int(seconds * 16000), dtype=np.float32))


def test_messages_follow_the_examples() -> None:
    call = {"call_id": "c1", "name": "get_time", "arguments": "{}"}
    request = ConversationRequest(
        [spoken(1.0), ConversationTurn("assistant", text="One moment.", tool_calls=[call]),
         ConversationTurn("tool", text="12:00", call_id="c1")],
        instructions="Be brief.", tools=[TOOL],
    )  # fmt: skip
    msgs = messages(request, tools=True)
    assert [m["role"] for m in msgs] == ["system", "tool_json_schemas", "human", "assistant", "input", "assistant"]
    assert msgs[0]["content"] == "Be brief." and '"name": "get_time"' in msgs[1]["content"]
    assert msgs[1]["content"].startswith('[{"type": "function", "function": {')
    assert msgs[3]["content"] == "One moment.<tool_call>function\nget_time\n{}</tool_call>"
    assert msgs[-1] == {"role": "assistant", "content": "<tts_start>", "eot": False}
    text_only = messages(ConversationRequest([spoken(0.5)], audio_out=False), tools=True)
    assert text_only[0]["content"] == "You are a helpful assistant." and text_only[-1]["content"] is None
    # 60 s of speech: three pieces of at most 25 s, each with its placeholders
    utils = StepUtils()
    pieces, mels = prompt_pieces(messages(ConversationRequest([spoken(60.0)]), tools=False), utils)
    assert utils.pieces == [400000, 400000, 160000] and len(mels) == 3
    assert pieces[0] == "<|BOT|>system\nYou are a helpful assistant.<|EOT|>" and pieces[1] == "<|BOT|>human\n"
    assert pieces[2].startswith("<audio_start><audio_patch>") and pieces[-1] == "<|BOT|>assistant\n<tts_start>"


def test_streams_text_tool_calls_and_speech() -> None:
    speech = [AUDIO_OFFSET + 5] * 60 + [AUDIO_OFFSET + 6561]  # the last one is padding
    script = [1, 2, 3, 151657, 4, 5, 6, 5, 7, 151658, *speech, 151665]
    model, llm, tok, _, t2w = step_model(script)
    request = ConversationRequest([spoken(1.0)], tools=[TOOL], voice="Male", max_output_tokens=4000)
    chunks = list(model.respond(request))
    text = "".join(c.text for c in chunks)
    assert text == "Hello there."
    calls = [c.tool_call for c in chunks if c.tool_call]
    assert calls == [{"call_id": "call_0", "name": "get_time", "arguments": '{"zone": "UTC"}'}]
    audio = [c.audio for c in chunks if c.audio is not None]
    # 60 speech tokens: two pieces of 25 (+3 look-ahead), the rest at the end
    assert t2w.calls == [(28, "/code/assets/default_male.wav", False), (28, "/code/assets/default_male.wav", False),
                         (10, "/code/assets/default_male.wav", True)]  # fmt: skip
    assert t2w.cached == ["/code/assets/default_male.wav"] and len(audio) == 3
    assert all(a.dtype == np.float32 and abs(float(a[0]) - 1000 / 32768) < 1e-6 for a in audio)
    call = llm.calls[0]
    assert call["generation_config"]["max_new_tokens"] == 512 and call["generation_config"]["eos_token_id"] == 151665
    assert call["wavs"].shape[0] == 1 and int(call["wav_lens"][0]) > 0
    assert tok.pieces[1] == "<|BOT|>tool_json_schemas\n" + messages(request, True)[1]["content"] + "<|EOT|>"
    first_audio = next(i for i, c in enumerate(chunks) if c.audio is not None)
    assert first_audio > 0 and chunks[0].text  # text first, speech from the 28th speech token


def test_voices_text_only_and_cancel() -> None:
    model, llm, _, _, t2w = step_model([1, AUDIO_OFFSET + 1, 2, 151665])
    assert [v.id for v in model.voices] == ["female", "male"] and model.voice(None) == "female"
    with pytest.raises(InvalidRequest):
        model.voice("alloy")
    chunks = list(model.respond(ConversationRequest([ConversationTurn("user", text="hi")], audio_out=False)))
    assert "".join(c.text for c in chunks) == "Hello there" and all(c.audio is None for c in chunks)
    assert t2w.calls == [] and llm.calls[-1]["wavs"] is None
    assert list(model.respond(ConversationRequest([spoken(0.2)]), lambda: True)) == []
    # a cancel during generation stops it at the next token
    long_model, long_llm, *_ = step_model([1] * 500)
    seen = []
    for chunk in long_model.respond(ConversationRequest([spoken(0.2)]), lambda: len(seen) >= 3):
        seen.append(chunk)
    assert len(seen) == 3 and long_llm.produced < 500
    # without a GPU there is no token2wav: text only
    model.token2wav = None
    assert all(c.audio is None for c in model.respond(ConversationRequest([spoken(0.2)])))


def test_load_checks_remote_code_and_files_first(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``load``: remote code and flow.yaml checked before transformers is touched; the pinned code bundle; the
    model's own classes through transformers' Auto classes with trust_remote_code; text only off CUDA."""
    import hashlib
    import json
    import types

    from bolo_torch.models import step_audio
    from bolo_torch.remote_code import RemoteCodeRefused, pins
    from bolo_torch.runtime import LoadContext
    from s2s_fakes import FakeGenerator, StepTokenizer

    rev = "e" * 40
    root = tmp_path / "model"
    (root / "token2wav").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"source": {"revision": rev}}))
    (root / "modeling_step_audio_2.py").write_text("")
    (root / "config.json").write_text(json.dumps({"auto_map": {"AutoModelForCausalLM": "modeling_step_audio_2.M"}}))
    (root / "token2wav" / "flow.yaml").write_text("flow: 1\n")
    code = tmp_path / "code"
    code.mkdir()
    (code / "utils.py").write_text("def compute_token_num(n):\n    return n\n")
    calls: list[tuple[str, dict]] = []

    class Auto:
        def __init__(self, kind: str, make):  # type: ignore[no-untyped-def]
            self.kind, self.make = kind, make

        def from_pretrained(self, path: str, **kwargs):  # type: ignore[no-untyped-def]
            calls.append((self.kind, kwargs))
            return self.make()

    llm = FakeGenerator([])
    llm.config = types.SimpleNamespace(eos_token_id=0)  # type: ignore[attr-defined]
    llm.to = lambda device: llm  # type: ignore[attr-defined]
    llm.eval = lambda: llm  # type: ignore[attr-defined]
    fake_tf = types.SimpleNamespace(
        AutoTokenizer=Auto("tokenizer", StepTokenizer),
        AutoModelForCausalLM=Auto("model", lambda: llm),
        GenerationConfig=dict,
    )
    monkeypatch.setattr(step_audio, "require", lambda module, extra: fake_tf)
    monkeypatch.setattr(step_audio.upstream, "ensure", lambda name: code)
    options = {
        "remote_code": pins(root, rev),
        "sha256": {"token2wav/flow.yaml": hashlib.sha256(b"flow: 1\n").hexdigest()},
    }
    ctx = LoadContext("step-audio-2", "speech-to-speech", root, dict(options), device="cpu", model_id="step")
    with pytest.raises(RemoteCodeRefused, match="allow_remote_code"):
        step_audio.load(ctx)
    assert calls == []
    ctx.options["allow_remote_code"] = True
    model = step_audio.load(ctx)
    assert [k for k, _ in calls] == ["tokenizer", "model"] and all(kw["trust_remote_code"] for _, kw in calls)
    assert model.token2wav is None and model.utils.compute_token_num(7) == 7 and llm.config.eos_token_id == 151665
    (root / "token2wav" / "flow.yaml").write_text("flow: !new:os.system\n")
    with pytest.raises(RemoteCodeRefused, match=r"flow\.yaml"):
        step_audio.load(ctx)
