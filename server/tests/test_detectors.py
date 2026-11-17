"""Voice-activity detectors chosen by name: frame-probability detectors, WebRTC, MarbleNet, the energy detector."""

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
