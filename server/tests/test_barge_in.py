"""Barge-in while the agent speaks, with synthetic audio: the agent's voice playing (and its echo in the microphone)
plus quieter user speech, as a browser's echo canceller and noise suppressor leave it during double talk."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from bolo.core import Bolo
from bolo.realtime import RealtimeError, RealtimeSession

SR = 16000
RNG = np.random.default_rng(11)


@pytest.fixture(autouse=True)
def _same_noise() -> None:
    global RNG
    RNG = np.random.default_rng(11)


def noise(seconds: float, db: float) -> np.ndarray:
    return (RNG.standard_normal(int(seconds * SR)) * 10 ** (db / 20)).astype(np.float32)


def voice(seconds: float, db: float, pitch: float = 180.0) -> np.ndarray:
    """Speech-like: a tone whose loudness rises and falls four times a second (by 8 dB: syllables, as the browser's
    automatic gain control leaves them)."""
    t = np.arange(int(seconds * SR)) / SR
    x = np.sin(2 * np.pi * pitch * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    return (x / np.sqrt(np.mean(x**2)) * 10 ** (db / 20)).astype(np.float32)


def speechlike(seconds: float, seed: int, db: float) -> np.ndarray:
    """Broadband syllables (an echo canceller needs a broadband reference to adapt)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = np.convolve(rng.standard_normal(n), np.hanning(9) / 4, mode="same")
    x *= np.sin(np.arange(n) / SR * 2 * np.pi * 3.0) > -0.3
    return (x / np.sqrt(np.mean(x**2)) * 10 ** (db / 20)).astype(np.float32)


def echo_of(far: np.ndarray, delay_s: float = 0.04, gain: float = 0.03) -> np.ndarray:
    """The agent's voice as the microphone hears it from the speakers: 40 ms late, with a short room tail."""
    d = int(delay_s * SR)
    ir = np.zeros(d + 800, dtype=np.float32)
    ir[d] = gain
    ir[d + 1 :] = np.random.default_rng(7).standard_normal(799) * np.exp(-np.arange(799) / 200) * gain / 15
    return np.convolve(far, ir)[: len(far)].astype(np.float32)


async def session(core: Bolo, events: list[dict[str, Any]], **turn_detection: Any) -> RealtimeSession:
    async def send(event: dict[str, Any]) -> None:
        events.append(event)

    s = RealtimeSession(core, send)
    s.configure(
        {"type": "realtime", "audio": {"input": {"transcription": {"model": "fake-stt"},
                                                 "turn_detection": {"type": "server_vad", **turn_detection}}}}
    )  # fmt: skip
    return s


async def feed(s: RealtimeSession, audio: np.ndarray) -> None:
    for i in range(0, len(audio), 320):  # 20 ms pieces, as a browser sends them
        await s.accept_audio(audio[i : i + 320])


def starts(events: list[dict[str, Any]]) -> list[int]:
    return [e["audio_start_ms"] for e in events if e["type"] == "input_audio_buffer.speech_started"]


async def agent_says(s: RealtimeSession, audio: np.ndarray) -> None:
    """What a response does with each piece of the answer: the client will play it from now on."""
    for i in range(0, len(audio), 4800):
        s._encode(audio[i : i + 4800], SR)


@pytest.mark.parametrize("barge_in_threshold", [0.3, 0.5])
async def test_quiet_user_speech_interrupts_the_agent(core: Bolo, barge_in_threshold: float) -> None:
    """Room noise at -55 dBFS; the agent speaks for 4 s; 1.5 s in, the user says something at -43 dBFS, as quiet as
    the browser's processing leaves a voice during double talk. At the ordinary threshold (0.5, which is what 0.5
    here means: no separate barge-in threshold) the detector does not hear it; at the barge-in threshold it does."""
    events: list[dict[str, Any]] = []
    s = await session(core, events, barge_in_threshold=barge_in_threshold)
    try:
        await feed(s, noise(1.0, -55))  # the floor
        await agent_says(s, voice(4.0, -20, pitch=240))
        assert s.agent_speaking
        user = np.concatenate([noise(1.5, -55), noise(1.5, -55) + voice(1.5, -43), noise(1.0, -55)])
        await feed(s, user)
        got = starts(events)
        if barge_in_threshold < 0.5:
            assert len(got) == 1 and 2000 <= got[0] <= 2600, got  # the user started at 2.5 s (less the padding)
            assert not s.agent_speaking  # the client stops playing on speech_started
            assert s.barge_ins == 1
            assert any(e["type"] == "input_audio_buffer.speech_stopped" for e in events)  # the turn ends too
        else:
            assert got == []  # what the browser saw 3 times in 4
        # the agent is quiet now: the same quiet voice is not a turn (the lower threshold applies only while it talks)
        events.clear()
        await feed(s, np.concatenate([noise(1.0, -55) + voice(1.0, -43), noise(1.0, -55)]))
        assert starts(events) == []
    finally:
        await s.close()


async def test_with_the_servers_echo_canceller(core: Bolo) -> None:
    """The browser's own canceller off, bolo's on (speex-aec, the reference is what the session sent): the agent's
    echo alone never interrupts it; the user talking over it does."""
    pytest.importorskip("pyaec")
    events: list[dict[str, Any]] = []
    s = await session(core, events)
    s.configure({"audio": {"input": {"echo_cancellation": "speex-aec"}}})
    try:
        await feed(s, noise(0.5, -60))
        far = speechlike(8.0, 3, -20)
        await agent_says(s, far)
        mic = echo_of(far) + noise(8.0, -60)
        await feed(s, mic[: 5 * SR])  # 5 s of echo only: the canceller adapts, nothing is heard
        assert starts(events) == [] and s.agent_speaking
        mic[5 * SR : int(6.5 * SR)] += speechlike(1.5, 9, -35)  # the user (SpeexDSP's suppressor takes a pure tone
        # for noise: a voice with a broadband spectrum)
        await feed(s, mic[5 * SR :])
        got = starts(events)
        assert len(got) == 1 and 5000 <= got[0] <= 5700, got  # mic time 5 s is session time 5.5 s (less the padding)
        info = s.session_object()["audio"]["input"]["echo_cancellation"]
        assert info["model"] == "speex-aec" and info["with_reference_s"] > 3  # until the barge-in stopped playback
    finally:
        await s.close()
    # the same echo without the canceller: the agent hears itself and interrupts its own answer
    events.clear()
    s = await session(core, events)
    try:
        await feed(s, noise(0.5, -60))
        await agent_says(s, far)
        await feed(s, (echo_of(far) + noise(8.0, -60))[: 2 * SR])
        assert starts(events)
    finally:
        await s.close()


async def test_barge_in_threshold_is_validated_and_reported(core: Bolo) -> None:
    events: list[dict[str, Any]] = []
    s = await session(core, events)
    assert s.session_object()["audio"]["input"]["turn_detection"]["barge_in_threshold"] == 0.3
    with pytest.raises(RealtimeError) as err:
        s.configure({"audio": {"input": {"turn_detection": {"type": "server_vad", "barge_in_threshold": 2}}}})
    assert err.value.param == "session.audio.input.turn_detection.barge_in_threshold"
    s.configure({"audio": {"input": {"turn_detection": {"type": "server_vad", "interrupt_response": False}}}})
    assert await s._ensure_barge() is None  # no interruption: nothing for it to do
    await s.close()
