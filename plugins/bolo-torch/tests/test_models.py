"""Model wrappers with fake backends: voices, sentence streaming, token mapping, separation shapes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from bolo_torch.models import orpheus
from bolo_torch.models.f5 import F5TTS, batch_result, collect_voices, normalize_ref_text
from bolo_torch.models.indic_speak import IndicSpeakTTS
from bolo_torch.models.separation import PixIT, SepFormer
from bolo_torch.runtime import InvalidRequest, PluginError
from fakes import FakeTokenizer, fake_parler, fake_separate_batch, tone

# -- Parler-TTS ---------------------------------------------------------------


def test_parler_voices_and_descriptions() -> None:
    model = fake_parler(voices=[{"id": "divya", "name": "Divya", "language": "hi-IN"}, {"id": "jon", "name": "Jon"}])
    assert [v.id for v in model.voices] == ["divya", "jon"]
    assert model.describe(None).startswith("Divya's voice")  # first voice is the default
    assert model.describe("JON").startswith("Jon's voice")
    free = "A calm male voice, slow and close to the microphone."
    assert model.describe(free) == free
    assert model.describe("jon", speed=1.5).endswith("The speaker talks fast.")
    assert model.describe("jon", speed=0.5).endswith("The speaker talks slowly.")
    with pytest.raises(PluginError):
        model.describe("nobody")
    custom = fake_parler(voices=[{"id": "x", "description": "Exactly this."}])
    assert custom.describe("x") == "Exactly this."
    assert fake_parler().describe(None).startswith("A speaker delivers")


def test_parler_streams_one_chunk_per_sentence() -> None:
    model = fake_parler(voices=[{"id": "jon", "name": "Jon"}])
    chunks = list(model.synthesize("Hello there. How are you today? नमस्ते। ठीक है।", "jon"))
    assert len(chunks) == 4
    assert all(c.dtype == np.float32 and c.ndim == 1 for c in chunks)
    assert model.tokenizer.seen == ["Hello there.", "How are you today?", "नमस्ते।", "ठीक है।"]
    assert model.description_tokenizer.seen == [model.describe("jon")]  # the description is encoded once


def test_parler_cancel_and_seed() -> None:
    model = fake_parler(seed=7)
    calls = iter([False, True])
    chunks = list(model.synthesize("One. Two. Three.", None, cancelled=lambda: next(calls, True)))
    assert len(chunks) == 1


# -- Orpheus / SNAC (Veena, Svara) ---------------------------------------------


def test_prompts_follow_the_model_cards() -> None:
    tok = FakeTokenizer()
    veena = orpheus.build_prompt("veena", tok, "नमस्ते", "kavya")
    assert veena[0] == orpheus.START_OF_HUMAN and veena[-3:] == [
        orpheus.END_OF_HUMAN,
        orpheus.START_OF_AI,
        orpheus.START_OF_SPEECH,
    ]
    assert tok.seen[-1] == "<spk_kavya> नमस्ते"
    svara = orpheus.build_prompt("svara", tok, "hello", "Hindi (Male)")
    assert svara[:3] == [orpheus.BOS, orpheus.START_OF_HUMAN, orpheus.SVARA_AUDIO]
    assert svara[-4:] == [orpheus.END_OF_HUMAN, orpheus.END_OF_TURN, orpheus.START_OF_AI, orpheus.START_OF_SPEECH]
    assert tok.seen[-1] == "Hindi (Male): hello"
    with pytest.raises(InvalidRequest):
        orpheus.build_prompt("other", tok, "x", "y")


def _frame_tokens(frame: list[int]) -> list[int]:
    return [orpheus.AUDIO_OFFSET + i * orpheus.CODEBOOK + c for i, c in enumerate(frame)]


def test_token_frames_and_snac_levels() -> None:
    frames = [[1, 2, 3, 4, 5, 6, 7], [10, 20, 30, 40, 50, 60, 70]]
    tokens = [orpheus.START_OF_SPEECH] + _frame_tokens(frames[0]) + _frame_tokens(frames[1]) + [orpheus.AUDIO_OFFSET]
    assert orpheus.frames_from_tokens(tokens) == frames  # the incomplete last frame is dropped
    assert orpheus.snac_levels(frames) == [[1, 10], [2, 5, 20, 50], [3, 4, 6, 7, 30, 40, 60, 70]]


def _orpheus_model(n_frames: int, **kwargs: Any) -> orpheus.OrpheusSNAC:
    def generate(prompt: list[int], streamer: Any, cancelled: Any, budget: int) -> None:
        streamer.put(torch.tensor(prompt))  # transformers streamers get the prompt first
        for f in range(n_frames):
            if cancelled():
                return
            for t in _frame_tokens([f % 4096] * 7):
                streamer.put(torch.tensor([t]))
        streamer.put(torch.tensor([orpheus.END_OF_SPEECH]))

    def decode(levels: list[list[int]]) -> np.ndarray:
        # 2048 samples per frame, each frame's samples equal to its level-0 code
        return np.repeat(np.asarray(levels[0], dtype=np.float32) / 4096, 2048)

    voices = [{"id": "kavya", "name": "Kavya"}, {"id": "hi_male", "name": "Hindi (Male)"}]
    return orpheus.OrpheusSNAC(generate, FakeTokenizer(), decode, voices=voices, **{"prompt_format": "veena", **kwargs})


def test_orpheus_streams_frames_in_order_without_gaps() -> None:
    model = _orpheus_model(11, stream_frames=3, context_frames=2)
    chunks = list(model.synthesize("Hello there.", "kavya"))
    assert len(chunks) >= 3  # streamed while generating, not all at the end
    audio = np.concatenate(chunks)
    assert len(audio) == 11 * 2048
    expected = np.repeat(np.arange(11, dtype=np.float32) / 4096, 2048)
    assert np.allclose(audio, expected)


def test_orpheus_speaker_names_per_format() -> None:
    assert _orpheus_model(1).speaker("kavya") == "kavya"
    assert _orpheus_model(1, prompt_format="svara").speaker("hi_male") == "Hindi (Male)"
    assert _orpheus_model(1).speaker(None) == "kavya"
    with pytest.raises(InvalidRequest):
        _orpheus_model(1).speaker("nobody")


def test_orpheus_cancel_stops_generation() -> None:
    model = _orpheus_model(500, stream_frames=2)
    seen = 0

    def cancelled() -> bool:
        return seen >= 2

    for _ in model.synthesize("A long sentence.", "kavya", cancelled=cancelled):
        seen += 1
    assert seen == 2


# -- IndicF5 --------------------------------------------------------------------


def test_f5_reference_voices(tmp_path: Path) -> None:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "a.wav").write_bytes(b"RIFF")
    voices_dir = tmp_path / "mine"
    voices_dir.mkdir()
    (voices_dir / "asha.wav").write_bytes(b"RIFF")
    (voices_dir / "asha.txt").write_text("नमस्ते", encoding="utf-8")
    (voices_dir / "lonely.wav").write_bytes(b"RIFF")  # no transcript: skipped
    options = {
        "voices": [
            {"id": "pan", "ref_audio": "prompts/a.wav", "ref_text": "ਸਤ"},
            {"id": "missing", "ref_audio": "x.wav", "ref_text": "x"},
        ],
        "voices_dir": str(voices_dir),
    }
    voices = collect_voices(tmp_path, options)
    assert [v["id"] for v in voices] == ["pan", "asha"]
    assert voices[0]["ref_audio"] == str(tmp_path / "prompts" / "a.wav")
    assert (
        normalize_ref_text("abc") == "abc. "
        and normalize_ref_text("abc।") == "abc। "
        and normalize_ref_text("a. ") == "a. "
    )

    calls: list[tuple[str, str, float]] = []

    def generate(text: str, ref: dict[str, Any], speed: float) -> np.ndarray:
        calls.append((text, ref["id"], speed))
        return (np.ones(240) * 16000).astype(np.int16)

    model = F5TTS(generate, voices)
    chunks = list(model.synthesize("पहला वाक्य। दूसरा वाक्य।", "asha", speed=1.2))
    assert [c[1] for c in calls] == ["asha", "asha"] and calls[0][2] == 1.2
    assert chunks[0].dtype == np.float32 and abs(float(chunks[0][0]) - 16000 / 32768) < 1e-6
    with pytest.raises(InvalidRequest):
        F5TTS(generate, [])


# -- separation -------------------------------------------------------------------


@pytest.mark.parametrize("model_rate", [8000, 16000])
def test_sepformer_tracks_are_16k_and_as_long_as_the_input(model_rate: int) -> None:
    model = SepFormer(fake_separate_batch, model_rate)
    assert model.mode == "blind" and model.sample_rate == 16000
    mix = tone(1.25)
    tracks = model.separate(mix)
    assert len(tracks) == 2 and all(len(t) == len(mix) and t.dtype == np.float32 for t in tracks)
    corr = float(np.corrcoef(tracks[0], mix)[0, 1])
    assert corr > 0.95 and float(np.corrcoef(tracks[1], mix)[0, 1]) < -0.95


class _Segment:
    def __init__(self, start: float, end: float) -> None:
        self.start, self.end = start, end


class _Diarization:
    def labels(self) -> list[str]:
        return ["SPEAKER_00", "SPEAKER_01"]

    def itertracks(self, yield_label: bool = False) -> list[tuple[_Segment, str, str]]:
        return [(_Segment(0.5, 1.0), "A", "SPEAKER_01"), (_Segment(0.0, 0.6), "B", "SPEAKER_00")]


class _Sources:
    def __init__(self, n: int) -> None:
        self.data = np.stack([np.full(n - 10, 0.5), np.full(n - 10, -2.0), np.zeros(n - 10)], axis=1)


def test_pixit_tracks_and_turns() -> None:
    def pipeline(file: dict[str, Any], **kwargs: Any) -> tuple[_Diarization, _Sources]:
        assert file["sample_rate"] == 16000 and file["waveform"].shape[0] == 1
        assert kwargs == {"num_speakers": 2}
        return _Diarization(), _Sources(file["waveform"].shape[-1])

    model = PixIT(pipeline, {"num_speakers": 2})
    mix = tone(1.0)
    tracks = model.separate(mix)
    assert len(tracks) == 2  # one per diarized speaker, the extra source column is ignored
    assert all(len(t) == len(mix) for t in tracks)
    assert float(np.max(np.abs(np.concatenate(tracks)))) <= 0.991  # scaled down instead of clipping
    assert model.last_turns == [(0.0, 0.6, 0), (0.5, 1.0, 1)]


# -- original F5-TTS checkpoints --------------------------------------------------


def test_f5_voices_from_bolo_samples(tmp_path: Path) -> None:
    options = {"voices": [{"id": "heart", "reference": "en-short"}, {"id": "nothing", "reference": "no-such-clip"}]}
    voices = collect_voices(tmp_path, options)
    assert [v["id"] for v in voices] == ["heart"]
    assert Path(voices[0]["ref_audio"]).is_file() and voices[0]["ref_text"].startswith("Good morning.")


def test_f5_batch_result_takes_both_forms() -> None:
    wave = np.zeros(4, dtype=np.float32)

    def generator() -> Any:
        yield wave, 24000, None

    assert batch_result((wave, 24000, None))[0] is wave
    assert batch_result(generator())[0] is wave


# -- Indic-Speak --------------------------------------------------------------------


def test_indic_speak_speakers_styles_and_sentences() -> None:
    calls: list[tuple[str, str, str | None]] = []

    def generate(text: str, speaker: str, style: str | None) -> np.ndarray:
        calls.append((text, speaker, style))
        return np.full(2400, 2.0, dtype=np.float64)  # loud: scaled down, never clipped

    voices = [
        {"id": "amit", "name": "Amit", "language": "hi-IN", "gender": "m"},
        {"id": "kavya", "name": "Kavya", "language": "hi-IN", "gender": "f"},
    ]
    model = IndicSpeakTTS(generate, voices, style=None)
    assert model.sample_rate == 24000 and [v.id for v in model.voices] == ["amit", "kavya"]
    chunks = list(model.synthesize("पहला वाक्य। दूसरा वाक्य।", "Kavya:anger"))
    assert calls == [("पहला वाक्य।", "Kavya", "ANGER"), ("दूसरा वाक्य।", "Kavya", "ANGER")]
    assert chunks[0].dtype == np.float32 and float(np.max(np.abs(chunks[0]))) == pytest.approx(1.0)
    list(model.synthesize("Hello."))
    assert calls[-1] == ("Hello.", "Amit", None)  # the first voice by default
    with pytest.raises(InvalidRequest, match="unknown voice"):
        list(model.synthesize("x", "nobody"))
    with pytest.raises(InvalidRequest, match="one speed"):
        list(model.synthesize("x", "amit", speed=1.5))
    stop = iter([False, True])
    calls.clear()
    list(model.synthesize("One. Two. Three.", "amit", cancelled=lambda: next(stop)))
    assert len(calls) == 1
    with pytest.raises(InvalidRequest):
        IndicSpeakTTS(generate, [])


def test_indic_speak_needs_the_repository(tmp_path: Path) -> None:
    pytest.importorskip("snac")
    pytest.importorskip("transformers")
    from bolo_torch.models.indic_speak import load
    from bolo_torch.runtime import LoadContext

    (tmp_path / "x").write_text("")
    with pytest.raises(InvalidRequest, match=r"inference\.py"):
        load(LoadContext("indic-speak", "tts", tmp_path, {}))
