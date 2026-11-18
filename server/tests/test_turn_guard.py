"""A turn a steady noise keeps open is ended anyway: noise_plateau_ms and max_turn_seconds (bolo.streaming.turn_guard)."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from bolo.core import Bolo
from bolo.realtime import RealtimeError, RealtimeSession
from bolo.streaming.turn_guard import TurnGuard

SR = 16000
RNG = np.random.default_rng(7)


def fan(seconds: float, db: float = -30.0) -> np.ndarray:
    """A fan: steady broadband noise at ``db`` dBFS (RMS). Its level per 20 ms frame barely moves."""
    x = RNG.standard_normal(int(seconds * SR))
    return (x * 10 ** (db / 20)).astype(np.float32)


def voice(seconds: float, db: float = -18.0) -> np.ndarray:
    """Speech-like: a tone whose loudness rises and falls four times a second, with short dips between syllables."""
    t = np.arange(int(seconds * SR)) / SR
    envelope = np.clip(np.sin(2 * np.pi * 4 * t), 0.05, None)
    x = np.sin(2 * np.pi * 200 * t) * envelope
    return (x / np.sqrt(np.mean(x**2)) * 10 ** (db / 20)).astype(np.float32)


async def run(session: RealtimeSession, audio: np.ndarray, chunk_s: float = 0.1) -> None:
    n = int(chunk_s * SR)
    for i in range(0, len(audio), n):
        await session.accept_audio(audio[i : i + n])


def turns(events: list[dict[str, Any]]) -> list[tuple[str, int]]:
    return [
        (e["type"].rsplit(".", 1)[1], e.get("audio_start_ms", e.get("audio_end_ms")))
        for e in events
        if e["type"] in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped")
    ]


async def new_session(core: Bolo, events: list[dict[str, Any]], turn_detection: dict[str, Any]) -> RealtimeSession:
    async def send(event: dict[str, Any]) -> None:
        events.append(event)

    session = RealtimeSession(core, send, transcription=True)
    session.configure(
        {"type": "transcription", "audio": {"input": {"transcription": {"model": "fake-stt"},
                                                      "turn_detection": {"type": "server_vad", **turn_detection}}}}
    )  # fmt: skip
    await session.start()
    return session


async def test_an_endless_fan_does_not_keep_the_turn_open(core: Bolo) -> None:
    """The laptop's case: the user speaks, then a fan keeps the detector in "speech" for good. Without the guard no
    speech_stopped ever comes and the agent never answers."""
    events: list[dict[str, Any]] = []
    session = await new_session(core, events, {})
    td = session.session_object()["audio"]["input"]["turn_detection"]
    assert td["max_turn_seconds"] == 30.0 and td["noise_plateau_ms"] == 2000.0  # the settings' defaults, reported
    try:
        await run(session, np.concatenate([fan(1.0, -75), voice(1.5), fan(20.0)]))
        got = turns(events)
        assert got[0] == ("speech_started", pytest.approx(700, abs=150))
        # ended where the steady noise began (about 2.5 s), found 2.5 s later: not after 30 s, and not never
        assert got[1][0] == "speech_stopped" and 2300 <= got[1][1] <= 3200
        assert any(e["type"] == "input_audio_buffer.committed" for e in events)
        assert len(got) == 2  # the fan alone starts no new turn
        # the user speaks again over the fan: a new turn starts on the rise and ends when only the fan is left
        await run(session, np.concatenate([voice(1.5, -12.0), fan(5.0)]))
        got = turns(events)
        assert [k for k, _ in got] == ["speech_started", "speech_stopped", "speech_started", "speech_stopped"]
        assert 22000 <= got[2][1] <= 22500 and 23700 <= got[3][1] <= 24500  # the voice: 22.5 s to 24 s
        for _ in range(200):  # both turns transcribed (and traced)
            done = "conversation.item.input_audio_transcription.completed"
            if sum(e["type"] == done for e in events) == 2:
                break
            await asyncio.sleep(0.02)
    finally:
        await session.close()
    trace = next(t for t in core.traces.recent() if t.extra.get("turn_end"))
    assert trace.extra["turn_end"] == "noise_plateau"


async def test_max_turn_seconds_commits_the_turn(core: Bolo) -> None:
    """With the plateau check off, the turn is committed after max_turn_seconds of "speech" whatever it is."""
    events: list[dict[str, Any]] = []
    session = await new_session(core, events, {"noise_plateau_ms": 0, "max_turn_seconds": 4})
    try:
        await run(session, np.concatenate([fan(0.5, -75), fan(9.0, -30)]))
        got = turns(events)
        start, stop = got[0][1], got[1][1]
        assert got[0][0] == "speech_started" and got[1][0] == "speech_stopped"
        assert 3900 <= stop - start <= 4300
        assert len(got) == 2  # still the same steady fan: no new turn on it
    finally:
        await session.close()
    with pytest.raises(RealtimeError) as err:
        session.configure({"audio": {"input": {"turn_detection": {"type": "server_vad", "max_turn_seconds": -1}}}})
    assert err.value.param == "session.audio.input.turn_detection.max_turn_seconds"


async def test_ordinary_turns_are_untouched(core: Bolo) -> None:
    """Speech with pauses, in a quiet room: the detector ends each turn itself, the guard does nothing."""
    events: list[dict[str, Any]] = []
    session = await new_session(core, events, {"silence_duration_ms": 400})
    try:
        quiet = fan(0.8, -75)
        await run(session, np.concatenate([quiet, voice(4.0), quiet, voice(2.0), quiet]))
        got = turns(events)
        assert [k for k, _ in got] == ["speech_started", "speech_stopped"] * 2
    finally:
        await session.close()
    assert not any(t.extra.get("turn_end") for t in core.traces.recent())


def test_the_guard_on_its_own() -> None:
    guard = TurnGuard(max_turn_s=0, plateau_s=1.0)
    assert guard.enabled and not TurnGuard(0, 0).enabled
    guard.feed(voice(2.0))
    assert guard.check(0, 2 * SR) is None  # a voice is never flat
    guard.feed(fan(2.0))
    reason, end = guard.check(0, 4 * SR) or ("", 0)
    assert reason == "noise_plateau" and end == 3 * SR and guard.armed
    assert guard.check(0, 4 * SR, plateau=False) is None  # the detector heard silence: its call
    assert guard.feed(fan(1.0)) is None  # the same noise: no new utterance
    onset = guard.feed(np.concatenate([fan(0.2), voice(0.5, -12.0)]))
    assert onset is not None and 5.15 * SR <= onset <= 5.3 * SR and not guard.armed
