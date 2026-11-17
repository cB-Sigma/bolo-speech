"""Kimi-Audio with a fake KimiAudio (language model, prompt manager, sampler, detokenizer): the conversation as
kimia_infer takes it, the step-by-step generation loop, text and 30-token speech chunks streamed."""

from __future__ import annotations

import numpy as np
import pytest
from bolo.engines.base import ConversationRequest, ConversationTurn

from bolo_torch.models.kimi_audio import KimiAudioS2S, chats
from bolo_torch.models.s2s_util import WavFiles
from bolo_torch.runtime import InvalidRequest, LoadContext
from s2s_fakes import KimiFake

OFF = KimiFake.offset
BLANK, EOS = 18, 19


def spoken(seconds: float) -> ConversationTurn:
    return ConversationTurn("user", audio=np.zeros(int(seconds * 16000), dtype=np.float32))


def kimi(script: list[tuple[int, int]], prompt: int = 20) -> tuple[KimiAudioS2S, KimiFake]:
    fake = KimiFake(script, prompt)
    return KimiAudioS2S(fake, fake.sampler, device="cpu"), fake


def test_chats_follow_kimia_infer() -> None:
    request = ConversationRequest(
        [spoken(1.0), ConversationTurn("assistant", text="Hello."), ConversationTurn("tool", text="42"),
         ConversationTurn("user", text="and?", audio=np.zeros(800, dtype=np.float32))],
        instructions="Be brief.",
    )  # fmt: skip
    with WavFiles() as wavs:
        out = chats(request, wavs)
        assert [(c["role"], c["message_type"]) for c in out] == [
            ("user", "text"), ("user", "audio"), ("assistant", "text"), ("user", "text"), ("user", "audio"),
            ("user", "text"),
        ]  # fmt: skip
        assert out[0]["content"] == "Be brief." and out[1]["content"].endswith(".wav")
        import soundfile as sf

        audio, rate = sf.read(out[1]["content"])
        assert rate == 16000 and len(audio) == 16000


def test_streams_text_then_speech_in_30_token_chunks() -> None:
    # text "Of course." then its end; speech starts after the delay of 2 steps; 70 speech tokens, then <|im_media_end|>
    text = [1, 2, 3, EOS] + [BLANK] * 80
    speech = [BLANK, BLANK] + [OFF + 5] * 70 + [15] + [BLANK] * 20
    model, fake = kimi(list(zip(text, speech)))
    chunks = list(model.respond(ConversationRequest([spoken(1.0)])))
    assert "".join(c.text for c in chunks) == "Of course."
    assert fake.detokenized == [(30, False, 4), (30, False, 4), (10, True, 4)] and fake.cleared == 1
    audio = [c.audio for c in chunks if c.audio is not None]
    assert sum(len(a) for a in audio) == 70 * 1920 and model.sample_rate == 24000
    assert len(fake.forwards) == 73 and fake.output_type == "both"  # stopped at the end of the speech stream
    first, second = fake.forwards[0], fake.forwards[1]
    assert first["position_ids"].shape == (1, 20) and first["whisper_input_feature"] is not None
    assert second["position_ids"].tolist() == [[20]] and second["whisper_input_feature"] is None
    assert second["past_key_values"] == ("past", 0) and fake.sampling["audio_temperature"] == 0.8


def test_text_only_limits_voice_and_cancel() -> None:
    model, fake = kimi([(1, OFF + 1), (2, OFF + 1), (EOS, OFF + 1)])
    chunks = list(model.respond(ConversationRequest([ConversationTurn("user", text="x")], audio_out=False)))
    assert "".join(c.text for c in chunks) == "Of course" and fake.detokenized == [] and fake.output_type == "text"
    assert len(fake.forwards) == 3
    long_model, long_fake = kimi([(BLANK, OFF + 1)] * 2000, prompt=1450)
    list(long_model.respond(ConversationRequest([spoken(0.5)])))
    assert len(long_fake.forwards) == 1500 - 1450  # KimiAudio.generate's 120 s of positions
    with pytest.raises(InvalidRequest):
        list(model.respond(ConversationRequest([spoken(0.2)], voice="alloy")))
    assert list(model.respond(ConversationRequest([spoken(0.2)]), lambda: True)) == []
    steps = 0

    def stop_after_five() -> bool:
        return len(long_fake.forwards) - steps >= 5

    steps = len(long_fake.forwards)
    list(long_model.respond(ConversationRequest([spoken(0.2)]), stop_after_five))
    assert len(long_fake.forwards) - steps == 5


def test_load_needs_cuda_and_the_opt_in(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from bolo_torch.models import kimi_audio

    with pytest.raises(InvalidRequest, match="NVIDIA GPU"):
        kimi_audio.load(LoadContext("kimi-audio", "speech-to-speech", tmp_path, {}, device="cpu"))


def test_the_speech_tokenizer_is_the_pinned_copy(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import types

    from bolo_torch.models.kimi_audio import _pinned_tokenizer

    made: list[str] = []
    module = types.SimpleNamespace(Glm4Tokenizer=lambda path: made.append(path) or path)
    module.Glm4Tokenizer = _pinned_tokenizer(module, tmp_path / "a")
    module.Glm4Tokenizer("THUDM/glm-4-voice-tokenizer")
    module.Glm4Tokenizer = _pinned_tokenizer(module, tmp_path / "b")  # a second load points at its own copy
    module.Glm4Tokenizer("THUDM/glm-4-voice-tokenizer")
    assert made == [str(tmp_path / "a"), str(tmp_path / "b")]
