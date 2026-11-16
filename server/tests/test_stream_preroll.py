"""Live transcription with a streaming model: pre-roll, VAD-gated streams and per-utterance gain."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from bolo.catalog import ModelSpec
from bolo.core import Bolo
from bolo.engines.fake import FakeStreamingSTT
from bolo.streaming.session import SR, PreRoll, SessionConfig, STTSession, speech_gain
from conftest import speechlike


def add(core: Bolo, model_id: str, **options: object) -> None:
    core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": model_id, "task": "stt", "engine": "fake", "kind": "fake-stt-streaming", "family": "Fake",
                "languages": ["en"], "license": "Apache-2.0", "builtin": True, "options": options, "streaming": True,
            }
        )
    )  # fmt: skip


async def run_stream(
    core: Bolo, model: str, audio: np.ndarray, *, chunk_ms: int = 40, pace: float = 1.0
) -> tuple[list[dict], STTSession]:
    messages: list[dict] = []

    async def send(msg: dict) -> None:
        messages.append(msg)

    session = STTSession(core, send, SessionConfig(model=model))
    await session.start()
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    step = 16 * chunk_ms * 2
    for off in range(0, len(pcm), step):
        session.feed(pcm[off : off + step])
        await asyncio.sleep(chunk_ms / 1000 * pace)
    await session.stop()
    return messages, session


def test_pre_roll_keeps_the_newest_audio_with_its_position() -> None:
    pre = PreRoll(0.5)
    a = np.arange(6000, dtype=np.float32)
    pre.push(a, 6000)
    assert (pre.start, pre.end) == (0, 6000) and len(pre.buf) == 6000
    pre.push(np.arange(6000, 10000, dtype=np.float32), 10000)  # 0.625 s pushed: only the newest 0.5 s are kept
    assert (pre.start, pre.end) == (2000, 10000)
    assert pre.buf[0] == 2000 and pre.buf[-1] == 9999
    pre.push(np.arange(10000, 20000, dtype=np.float32), 20000)  # one push longer than the buffer
    assert (pre.start, pre.end) == (12000, 20000) and pre.buf[0] == 12000
    audio, start = pre.take(since=15000)
    assert start == 15000 and audio[0] == 15000 and len(audio) == 5000
    assert len(pre.buf) == 0 and pre.end == 20000  # emptied, still knows where it is
    audio, start = pre.take(since=0)
    assert len(audio) == 0 and start == 20000
    # a request for more than is held gets what there is
    pre.push(np.ones(100, dtype=np.float32), 20100)
    audio, start = pre.take(since=0)
    assert start == 20000 and len(audio) == 100


def test_speech_gain_lifts_quiet_speech_only() -> None:
    loud = speechlike(1.0)  # 0.3 amplitude: about -13 dBFS
    assert speech_gain(loud, -20.0) == 1.0
    quiet = loud * 10 ** (-20 / 20)  # about -33 dBFS
    gain = speech_gain(quiet, -20.0)
    assert 3.0 < gain < 6.0
    # pauses do not count: the same speech with a long silence in it gets the same gain
    paused = np.concatenate([quiet[:8000], np.zeros(24000, dtype=np.float32), quiet[8000:]])
    assert speech_gain(paused, -20.0) == pytest.approx(gain, rel=0.05)
    assert speech_gain(quiet * 1e-4, -20.0) == pytest.approx(10 ** (24 / 20))  # capped
    assert speech_gain(np.zeros(16000, dtype=np.float32), -20.0) == 1.0
    assert speech_gain(np.zeros(100, dtype=np.float32), -20.0) == 1.0


def utterances(*parts: tuple[str, float]) -> np.ndarray:
    pieces = []
    for kind, seconds in parts:
        if kind == "speech":
            pieces.append(speechlike(seconds))
        else:  # a quiet room, not digital silence
            pieces.append(np.random.default_rng(len(pieces)).standard_normal(int(seconds * SR)) * 3e-4)
    return np.concatenate(pieces).astype(np.float32)


async def test_a_gated_stream_hears_speech_and_its_pre_roll_only(core: Bolo) -> None:
    add(core, "t-gated", step_delay_s=0.005, preroll_s=0.4)
    entry = await core.manager.load("t-gated")
    model = entry.model
    assert isinstance(model, FakeStreamingSTT)
    audio = utterances(("room", 3.0), ("speech", 1.6), ("room", 2.5), ("speech", 1.2), ("room", 1.5))
    messages, session = await run_stream(core, "t-gated", audio, pace=0.1)
    vad = [m for m in messages if m["type"] == "vad"]
    finals = [m for m in messages if m["type"] == "final"]
    assert [v["event"] for v in vad] == ["speech_start", "speech_end"] * 2
    assert len(finals) == 2
    # each utterance begins where the VAD heard it, and its words are timed from the pre-roll before that
    for final, start in zip(finals, (vad[0], vad[2])):
        assert final["t0"] == pytest.approx(start["t"], abs=0.01)
        assert final["words"][0]["t0"] == pytest.approx(start["t"] - 0.4, abs=0.03)
    assert finals[0]["t1"] == pytest.approx(vad[1]["t"], abs=0.01)
    # the recogniser heard the speech and 0.4 s before each start, not the room in between: besides that, only the
    # silence the VAD waits for before it reports an end (0.5 s) and the tail padding that flushes the model (0.5 s)
    spoken = sum(v["t"] - u["t"] for u, v in zip(vad[0::2], vad[1::2]))
    assert spoken + 2 * (0.4 + 0.5) <= model.heard / SR <= spoken + 2 * (0.4 + 0.5 + 0.5) + 0.1
    assert model.heard < 0.7 * len(audio)
    assert session.trace is not None and session.trace.extra["utterances"] == 2


async def test_queued_audio_still_ends_one_utterance_where_the_next_begins(core: Bolo) -> None:
    """Audio sent much faster than real time arrives in large pieces with an end and a start in them."""
    add(core, "t-gated-q", step_delay_s=0.05, preroll_s=0.4)
    await core.manager.load("t-gated-q")
    audio = utterances(("room", 1.0), ("speech", 1.4), ("room", 1.2), ("speech", 1.4), ("room", 1.5))
    messages, _ = await run_stream(core, "t-gated-q", audio, chunk_ms=500, pace=0.01)
    vad = [m for m in messages if m["type"] == "vad"]
    finals = [m for m in messages if m["type"] == "final"]
    assert [v["event"] for v in vad] == ["speech_start", "speech_end"] * 2
    assert len(finals) == 2
    assert finals[0]["t1"] == pytest.approx(vad[1]["t"], abs=0.01) and finals[0]["t1"] < finals[1]["t0"]
    assert finals[1]["t0"] == pytest.approx(vad[2]["t"], abs=0.01)


async def test_quiet_speech_is_raised_for_the_recogniser(core: Bolo) -> None:
    add(core, "t-gain", step_delay_s=0.005, preroll_s=0.3, speech_dbfs=-20.0)
    entry = await core.manager.load("t-gain")
    model = entry.model
    assert isinstance(model, FakeStreamingSTT)
    quiet = utterances(("room", 1.0), ("speech", 1.5), ("room", 1.5)) * 10 ** (-20 / 20)
    messages, _ = await run_stream(core, "t-gain", quiet.astype(np.float32), pace=0.1)
    assert [m["type"] for m in messages].count("final") == 1
    peak_in = float(np.max(np.abs(quiet)))
    assert model.peak > 3 * peak_in  # what the VAD heard as about -33 dBFS reached the recogniser near -20
    assert model.peak <= 1.0


async def test_without_a_pre_roll_the_recogniser_hears_everything(core: Bolo) -> None:
    add(core, "t-open", step_delay_s=0.005)
    entry = await core.manager.load("t-open")
    model = entry.model
    assert isinstance(model, FakeStreamingSTT)
    audio = utterances(("room", 2.0), ("speech", 1.4), ("room", 1.5))
    messages, _ = await run_stream(core, "t-open", audio, pace=0.1)
    assert [m["type"] for m in messages].count("final") == 1
    assert model.heard >= len(audio)
