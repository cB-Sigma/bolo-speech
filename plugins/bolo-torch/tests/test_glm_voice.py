"""GLM-4-Voice with fakes of ChatGLM, its tokenizer, WhisperVQEncoder and AudioDecoder: the demo's prompt, interleaved
text and speech, the decoder's growing blocks."""

from __future__ import annotations

import numpy as np
import pytest
from bolo.engines.base import ConversationRequest, ConversationTurn

from bolo_torch.models.glm_voice import AUDIO_SYSTEM, TEXT_SYSTEM, GlmVoiceS2S, speech_tokens
from bolo_torch.runtime import InvalidRequest
from s2s_fakes import FakeGenerator, GlmDecoder, GlmTokenizer, WhisperFeatures, WhisperVQ

A = GlmTokenizer.audio_0


def glm(script: list[int]) -> tuple[GlmVoiceS2S, FakeGenerator, GlmTokenizer, GlmDecoder]:
    gen, tok, dec = FakeGenerator(script), GlmTokenizer(), GlmDecoder()
    return GlmVoiceS2S(gen, tok, WhisperVQ(), WhisperFeatures(), dec, device="cpu", max_new_tokens=300), gen, tok, dec


def spoken(seconds: float) -> ConversationTurn:
    return ConversationTurn("user", audio=np.zeros(int(seconds * 16000), dtype=np.float32))


def test_speech_tokens_in_30_s_windows() -> None:
    feats = WhisperFeatures()
    ids = speech_tokens(WhisperVQ(), feats, np.zeros(45 * 16000, dtype=np.float32), "cpu")
    assert feats.calls[0]["pad_to_multiple_of"] == 1 * 2 * 4 * 160 and feats.calls[0]["sampling_rate"] == 16000
    assert len(ids) == 3000 // 8 + -(-1500 // 8) and ids[:2] == [10, 11]
    assert speech_tokens(WhisperVQ(), feats, np.zeros(0, dtype=np.float32), "cpu") == []


def test_prompt_follows_the_demo() -> None:
    model, *_ = glm([])
    prompt = model.prompt(
        ConversationRequest(
            [spoken(1.0), ConversationTurn("assistant", text="Hi."), spoken(0.5)], instructions="Be kind."
        )
    )
    assert prompt.startswith(f"<|system|>\n{AUDIO_SYSTEM.rstrip()}\nBe kind.<|user|>\n<|begin_of_audio|><|audio_10|>")
    assert "<|end_of_audio|><|assistant|>streaming_transcription\nHi.<|user|>\n<|begin_of_audio|>" in prompt
    assert prompt.endswith("<|end_of_audio|><|assistant|>streaming_transcription\n")
    text = model.prompt(ConversationRequest([ConversationTurn("user", text="hello")]))
    assert text == f"<|system|>\n{TEXT_SYSTEM}<|user|>\nhello<|assistant|>streaming_transcription\n"


def test_streams_text_and_speech_in_growing_blocks() -> None:
    speech = [A + i % 7 for i in range(200)]
    script = [1, 2, *speech[:60], 3, *speech[60:], 151336, 1]  # nothing after <|user|> is used
    model, gen, tok, dec = glm(script)
    chunks = list(model.respond(ConversationRequest([spoken(1.0)])))
    assert "".join(c.text for c in chunks) == "Sure, here it is."
    # blocks of 25, 50, 100, then the rest when <|user|> ends the answer, each conditioned on the ones before
    assert [c["n"] for c in dec.calls] == [25, 50, 100, 25]
    assert [c["prompt_token"] for c in dec.calls] == [0, 25, 75, 175]
    assert [c["prompt_feat"] for c in dec.calls] == [0, 50, 150, 350] and dec.calls[-1]["finalize"] is True
    assert len({c["uuid"] for c in dec.calls}) == 1
    audio = [c.audio for c in chunks if c.audio is not None]
    assert len(audio) == 4 and sum(len(a) for a in audio) == 200 * 1764 and model.sample_rate == 22050
    assert gen.calls[0]["max_new_tokens"] == 300 and gen.calls[0]["temperature"] == 0.2
    assert tok.prompts[0].endswith("streaming_transcription\n")


def test_text_only_voice_and_cancel() -> None:
    model, _, _, dec = glm([1, A + 1, A + 2, 2, 151336])
    chunks = list(model.respond(ConversationRequest([ConversationTurn("user", text="x")], audio_out=False)))
    assert "".join(c.text for c in chunks) == "Sure, here" and dec.calls == []
    with pytest.raises(InvalidRequest):
        list(model.respond(ConversationRequest([spoken(0.2)], voice="alloy")))
    assert list(model.respond(ConversationRequest([spoken(0.2)]), lambda: True)) == []
    long_model, gen, *_ = glm([1] * 400)
    seen = []
    for chunk in long_model.respond(ConversationRequest([spoken(0.2)]), lambda: len(seen) >= 2):
        seen.append(chunk)
    assert len(seen) == 2 and gen.produced < 400


def test_load_wires_the_pieces(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import hashlib
    import json
    import sys
    import types

    from bolo_torch.models import glm_voice
    from bolo_torch.remote_code import RemoteCodeRefused, pins
    from bolo_torch.runtime import LoadContext

    rev = "3" * 40
    root = tmp_path / "model"
    (root / "glm-4-voice-decoder").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"source": {"revision": rev}}))
    (root / "modeling_chatglm.py").write_text("")
    (root / "glm-4-voice-decoder" / "config.yaml").write_text("flow: 1\n")
    code = tmp_path / "code"
    (code / "speech_tokenizer").mkdir(parents=True)
    (code / "speech_tokenizer" / "__init__.py").write_text("")
    (code / "speech_tokenizer" / "modeling_whisper.py").write_text(
        "class WhisperVQEncoder:\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, path):\n"
        "        o = cls(); o.path = path; return o\n"
        "    def eval(self):\n        return self\n"
        "    def to(self, device):\n        self.device = device; return self\n"
    )
    (code / "flow_inference.py").write_text(
        "class AudioDecoder:\n    def __init__(self, **kw):\n        self.kw = kw\n"
    )
    loaded: list[tuple[str, dict]] = []

    def auto(kind: str, make):  # type: ignore[no-untyped-def]
        def from_pretrained(path: str, **kw):  # type: ignore[no-untyped-def]
            loaded.append((kind, kw))
            return make()

        return types.SimpleNamespace(from_pretrained=from_pretrained)

    gen = FakeGenerator([])
    gen.eval = lambda: gen  # type: ignore[attr-defined]
    gen.to = lambda device: gen  # type: ignore[attr-defined]
    fake_tf = types.SimpleNamespace(
        AutoTokenizer=auto("tokenizer", GlmTokenizer), AutoModel=auto("model", lambda: gen),
        WhisperFeatureExtractor=auto("features", WhisperFeatures),
        BitsAndBytesConfig=lambda **kw: ("bnb", kw),
    )  # fmt: skip

    def ensure(name: str):  # type: ignore[no-untyped-def]
        assert name == "glm-4-voice"
        sys.path.insert(0, str(code))
        return code

    monkeypatch.setattr(glm_voice, "require", lambda module, extra: fake_tf)
    monkeypatch.setattr(glm_voice.upstream, "ensure", ensure)
    digest = hashlib.sha256(b"flow: 1\n").hexdigest()
    options = {"remote_code": pins(root, rev), "sha256": {"glm-4-voice-decoder/config.yaml": digest}}
    ctx = LoadContext("glm-4-voice", "speech-to-speech", root, dict(options), device="cpu", model_id="glm")
    try:
        with pytest.raises(RemoteCodeRefused):
            glm_voice.load(ctx)
        ctx.options["allow_remote_code"] = True
        model = glm_voice.load(ctx)
        assert [k for k, _ in loaded] == ["tokenizer", "model", "features"]
        assert loaded[0][1] == {"trust_remote_code": True} and loaded[1][1]["trust_remote_code"] is True
        assert model.whisper.path.endswith("glm-4-voice-tokenizer") and model.decoder.kw["device"] == "cpu"
        assert model.decoder.kw["config_path"].endswith("glm-4-voice-decoder/config.yaml")
        with pytest.raises(InvalidRequest, match="quantization"):
            glm_voice.load(
                LoadContext("glm-4-voice", "speech-to-speech", root, {**ctx.options, "quantization": "int3"})
            )
    finally:
        sys.path.remove(str(code))
        for name in ("flow_inference", "speech_tokenizer", "speech_tokenizer.modeling_whisper"):
            sys.modules.pop(name, None)
