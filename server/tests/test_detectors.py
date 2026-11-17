"""Models chosen by name: voice activity (frame-probability detectors, WebRTC, MarbleNet, the energy detector), end
of turn (silence only, models that read the transcript) and speaker embedders on their own."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import websockets

from bolo.catalog import Catalog
from bolo.config import Settings
from bolo.core import Bolo
from bolo.engines.base import FloatArray, VadParams
from bolo.engines.frame_vad import MARBLENET_MELS, FrameScorer, FrameVad, MarbleNetVAD
from conftest import LiveServer, speechlike


class Scripted(FrameScorer):
    """Probabilities from a list, one per 20 ms frame of audio received."""

    frame = 320

    def __init__(self, probs: list[float]) -> None:
        self.probs = probs
        self.received = 0
        self.done = 0

    def push(self, samples: FloatArray) -> list[float]:
        self.received += len(samples)
        ready = min(len(self.probs), self.received // self.frame)
        out, self.done = self.probs[self.done : ready], ready
        return out

    def finish(self) -> list[float]:
        out, self.done = self.probs[self.done :], len(self.probs)
        return out


def events(probs: list[float], params: VadParams, chunk: int = 320) -> list[tuple[str, int]]:
    vad = FrameVad(Scripted(probs), params)
    audio = np.zeros(len(probs) * 320, dtype=np.float32)
    out = []
    for off in range(0, len(audio), chunk):
        out += [(e.kind, e.sample // 320) for e in vad.accept(audio[off : off + chunk])]
    return out + [(e.kind, e.sample // 320) for e in vad.flush()]


def test_frame_probabilities_become_segments() -> None:
    params = VadParams(threshold=0.5, min_silence_s=0.1, min_speech_s=0.1, speech_pad_s=0.0)
    probs = [0.0] * 10 + [0.9] * 20 + [0.0] * 10 + [0.9] * 3 + [0.0] * 10
    # 20 frames of speech: one segment; three frames (60 ms) are shorter than min_speech_s and ignored
    assert events(probs, params) == [("speech_start", 10), ("speech_end", 30)]
    # the same with the audio in other pieces
    assert events(probs, params, chunk=4000) == [("speech_start", 10), ("speech_end", 30)]
    # once in speech, a frame stays speech down to threshold - 0.15
    dip = [0.0] * 5 + [0.9] * 10 + [0.4] * 10 + [0.0] * 10
    assert events(dip, params) == [("speech_start", 5), ("speech_end", 25)]
    # padding, and a segment still open at the end of the input
    padded = VadParams(threshold=0.5, min_silence_s=0.1, min_speech_s=0.1, speech_pad_s=0.04)
    assert events([0.0] * 10 + [0.9] * 10, padded) == [("speech_start", 8), ("speech_end", 20)]
    # max_speech_s cuts a long segment
    capped = VadParams(threshold=0.5, min_silence_s=0.1, min_speech_s=0.1, max_speech_s=0.4)
    assert events([0.9] * 30, capped)[:2] == [("speech_start", 0), ("speech_end", 20)]


def _fake_marblenet(lookahead_s: float) -> MarbleNetVAD:
    model = MarbleNetVAD.__new__(MarbleNetVAD)
    model.context_frames = 80
    model.lookahead_frames = int(lookahead_s * 100)
    model.min_frames = 5

    # a stand-in network: frame k is "speech" when its two feature frames are loud on average, smoothed over
    # +-3 frames, so streaming must give it context on both sides
    def probabilities(mel: FloatArray) -> FloatArray:
        if len(mel) % 2:
            mel = np.concatenate([mel, mel[-1:]])
        level = mel.reshape(-1, 2, MARBLENET_MELS).mean(axis=(1, 2))
        smooth = np.convolve(np.pad(level, 3, mode="edge"), np.ones(7) / 7, mode="valid")
        return np.clip((smooth + 17.0) / 15.0, 0.0, 1.0).astype(np.float32)

    model.probabilities = probabilities  # type: ignore[method-assign]
    return model


def test_marblenet_scores_a_stream_as_it_scores_a_whole_recording() -> None:
    from bolo.audio.features import nemo_log_mel

    audio = np.concatenate([np.zeros(16000, np.float32), speechlike(1.5), np.zeros(8000, np.float32), speechlike(1.0)])
    model = _fake_marblenet(lookahead_s=0.2)
    whole = model.probabilities(nemo_log_mel(audio, MARBLENET_MELS))
    for chunk in (320, 1000, 16000, len(audio)):
        scorer = model.create_detector().scorer
        got: list[float] = []
        for off in range(0, len(audio), chunk):
            got += scorer.push(audio[off : off + chunk])
        got += scorer.finish()
        assert len(got) == len(whole) == -(-len(audio) // 320)
        np.testing.assert_allclose(got, whole, atol=1e-5)
    # without look-ahead the first scores of a piece cannot see what follows
    scorer = _fake_marblenet(lookahead_s=0.0).create_detector().scorer
    early: list[float] = []
    for off in range(0, len(audio), 320):
        early += scorer.push(audio[off : off + 320])
    assert np.abs(np.array(early) - whole[: len(early)]).max() > 0.01


def test_webrtc_detector_finds_speech_and_not_silence() -> None:
    pytest.importorskip("webrtcvad")
    from bolo.engines.base import LoadOptions
    from bolo.engines.frame_vad import WebRtcVAD
    from bolo.streaming.vad import detect_segments

    model = WebRtcVAD({"mode": 2}, LoadOptions())
    assert detect_segments(model.create_detector(), np.zeros(32000, dtype=np.float32)) == []
    voice = Path(__file__).resolve().parent.parent / "src" / "bolo" / "samples" / "en-short.wav"
    import soundfile as sf

    speech, rate = sf.read(voice, dtype="float32")
    assert rate == 16000
    audio = np.concatenate([np.zeros(16000, np.float32), speech, np.zeros(16000, np.float32)])
    segments = detect_segments(model.create_detector(VadParams(min_silence_s=0.5)), audio)
    assert segments and 0.8 * 16000 < segments[0][0] < 1.3 * 16000
    with pytest.raises(ValueError, match="mode"):
        WebRtcVAD({"mode": 7}, LoadOptions())


def test_detector_entries_are_pinned_and_licensed() -> None:
    catalog = Catalog.load(None)
    vads = [s for s in catalog.all() if s.task == "vad"]
    assert {"silero-vad", "silero-vad-v6", "ten-vad", "marblenet-vad", "webrtc-vad", "energy-vad"} <= {
        s.id for s in vads
    }
    for spec in vads:
        assert spec.license and spec.license != "unknown"
        for f in spec.source.urls:
            assert len(str(f.get("sha256", ""))) == 64, f"{spec.id}: {f['path']} is not pinned by sha256"
    ten = catalog.resolve("ten-vad")
    assert ten is not None and not ten.commercial_use and "Agora" in ten.license and ten.license_url


async def test_the_energy_detector_can_be_chosen_by_name(core: Bolo) -> None:
    detector, name = await core.vad_detector(VadParams(), "energy-vad")
    assert name == "energy-vad"
    from bolo.streaming.vad import detect_segments

    assert len(detect_segments(detector, speechlike(1.0))) == 1


async def session_update(url: str, turn_detection: dict[str, Any]) -> dict[str, Any]:
    async with websockets.connect(url.replace("http", "ws") + "/v1/realtime?intent=transcription") as ws:
        json.loads(await ws.recv())
        session = {"type": "transcription", "audio": {"input": {"turn_detection": turn_detection}}}
        await ws.send(json.dumps({"type": "session.update", "session": session}))
        return dict(json.loads(await ws.recv()))


async def test_a_realtime_session_chooses_its_detector(settings: Settings) -> None:
    live = LiveServer(settings)
    live.start()
    try:
        ok = await session_update(live.url, {"type": "server_vad", "vad_model": "energy-vad"})
        assert ok["type"] == "session.updated"
        assert ok["session"]["audio"]["input"]["turn_detection"]["vad_model"] == "energy-vad"
        bad = await session_update(live.url, {"type": "server_vad", "vad_model": "no-such-detector"})
        assert bad["type"] == "error"
        assert bad["error"]["param"] == "session.audio.input.turn_detection.vad_model"
        wrong = await session_update(live.url, {"type": "server_vad", "vad_model": "fake-stt"})
        assert wrong["type"] == "error" and "vad" in wrong["error"]["message"]
    finally:
        live.stop()


# -- end of turn -------------------------------------------------------------------------------------
async def test_silence_only_turns_semantic_vad_into_server_vad(core: Bolo) -> None:
    core.settings.default_turn = "silence-only"
    assert core.turn_model() is None  # no model: a pause ends the turn
    chosen = core.turn_model("fake-turn")
    assert chosen is not None and chosen.id == "fake-turn"
    info = core.capabilities()["realtime"]["semantic_vad"]
    assert info["model"] == "silence-only" and info["available"] is False


async def test_a_text_turn_model_is_asked_about_the_transcript(core: Bolo) -> None:
    probability = await core.end_of_turn(speechlike(1.0), text="I would like to", model="fake-turn-text")
    assert probability is not None and probability < 0.5
    probability = await core.end_of_turn(
        speechlike(1.0), text="I would like a large coffee with oat milk, please.", model="fake-turn-text"
    )
    assert probability is not None and probability > 0.5
    # without a transcript a text model cannot judge: the pause decides
    assert await core.end_of_turn(speechlike(1.0), text=None, model="fake-turn-text") is None


async def test_a_realtime_session_chooses_its_end_of_turn_model(settings: Settings) -> None:
    live = LiveServer(settings)
    live.start()
    try:
        ok = await session_update(live.url, {"type": "semantic_vad", "turn_model": "fake-turn-text"})
        assert ok["type"] == "session.updated"
        assert ok["session"]["audio"]["input"]["turn_detection"]["turn_model"] == "fake-turn-text"
        bad = await session_update(live.url, {"type": "semantic_vad", "turn_model": "silero-vad"})
        assert bad["type"] == "error" and bad["error"]["param"] == "session.audio.input.turn_detection.turn_model"
    finally:
        live.stop()


async def test_semantic_vad_with_a_text_model_decodes_at_the_pause(settings: Settings) -> None:
    from test_voices_and_turns import turns

    live = LiveServer(settings)
    live.start()
    try:
        silence = np.zeros(int(24000 * 0.8), dtype=np.float32)
        audio = np.concatenate(
            [speechlike(2.0, 24000), silence, speechlike(2.0, 24000), np.zeros(24000 * 2, dtype=np.float32)]
        )
        events = await turns(live.url, {"type": "semantic_vad", "turn_model": "fake-turn-text"}, audio)
        types = [e["type"] for e in events]
        # the fake recogniser's text of the first two seconds does not end a sentence; four seconds' does
        assert types.count("input_audio_buffer.speech_stopped") == 1, types
        model = live.core.manager.loaded["fake-turn-text"].model
        assert model.texts and all(isinstance(t, str) and t for t in model.texts)  # type: ignore[attr-defined]
    finally:
        live.stop()


# -- speaker embedders -------------------------------------------------------------------------------
def test_speaker_embedders_are_catalog_entries_of_their_own() -> None:
    catalog = Catalog.load(None)
    embedders = {s.id: s for s in catalog.all() if s.task == "speaker-embedding"}
    assert {
        "eres2net-voxceleb", "campplus-voxceleb", "wespeaker-resnet34-voxceleb", "titanet-small", "ecapa-tdnn-voxceleb",
    } <= set(embedders)  # fmt: skip
    for spec in embedders.values():
        assert spec.source.urls and all(len(str(u.get("sha256", ""))) == 64 for u in spec.source.urls), spec.id
    # the standalone ERes2Net is the very file the default diarization pipeline embeds with
    pipeline = catalog.resolve("pyannote-3.0-eres2net")
    assert pipeline is not None
    shas = {u["sha256"] for u in pipeline.source.urls}
    assert embedders["eres2net-voxceleb"].source.urls[0]["sha256"] in shas


async def test_similarity_accepts_embedders_and_diarizers_by_name(core: Bolo) -> None:
    from bolo import similarity
    from bolo.errors import InvalidRequest

    assert similarity.embedder(core).id == "fake-diarization"  # the default diarization model's embedder
    assert similarity.embedder(core, "fake-speaker-embedding").task == "speaker-embedding"
    a = speechlike(1.0)
    vector = await similarity.embed(core, a, 16000, "fake-speaker-embedding")
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5
    result = await similarity.score(core, a, 16000, a, 16000, model="fake-speaker-embedding")
    assert result["embedding_model"] == "fake-speaker-embedding" and result["similarity"] > 0.99
    assert result["calibrated"] is False
    with pytest.raises(InvalidRequest, match="speaker-embedding or diarization"):
        similarity.embedder(core, "fake-stt")


def test_the_embedding_route_takes_any_embedder(live: LiveServer) -> None:
    import httpx

    from conftest import wav_file

    for model in ("fake-speaker-embedding", "fake-diarization", "", "pyannote/wespeaker-voxceleb-resnet34-LM"):
        r = httpx.post(
            live.url + "/v1/audio/speech/embedding",
            files={"file": wav_file(1.0)},
            data={"model": model} if model else {},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == (model if model.startswith("fake-") else "fake-diarization")
        assert body["data"][0]["embedding"]
    r = httpx.post(live.url + "/v1/audio/speech/embedding", files={"file": wav_file(1.0)}, data={"model": "fake-tts"})
    assert r.status_code == 400 and "speaker-embedding" in r.json()["error"]["message"]


def test_kaldi_filterbank_frames_and_bands() -> None:
    from bolo.audio.features import kaldi_fbank

    t = np.arange(16000) / 16000
    tone = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    feats = kaldi_fbank(tone)
    assert feats.shape == (98, 80) and feats.dtype == np.float32  # 1 + (16000 - 400) // 160 frames, no padding
    np.testing.assert_allclose(feats.mean(axis=0), 0.0, atol=1e-4)  # the mean over the recording is removed
    raw = kaldi_fbank(tone, mean_norm=False)
    assert int(np.argmax(raw[50])) < 15, "a 300 Hz tone sits in the low bands"
    # 16-bit sample values only move every band by the same amount
    np.testing.assert_allclose(kaldi_fbank(tone, scale=32768.0), feats, atol=1e-3)
    assert kaldi_fbank(tone[:300]).shape == (0, 80)
