"""Echo cancellation (bolo.echo): the cancellers on a synthetic echo, and a Realtime session that removes its own
voice from what it hears (audio.input.echo_cancellation)."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import numpy as np
import pytest
import websockets

from bolo.echo import EchoReference, SpeexAec, WebRtcAec3, parse_echo
from bolo.errors import InvalidRequest
from conftest import LiveServer

SR = 16000


def speechlike(seconds: float, seed: int) -> np.ndarray:
    """Noise shaped like speech: syllables of filtered noise with pauses (an echo canceller needs a broadband
    reference to adapt)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = rng.standard_normal(n).astype(np.float32)
    x = np.convolve(x, np.hanning(9) / 4, mode="same").astype(np.float32)
    env = (np.sin(np.arange(n) / SR * 2 * np.pi * 3.0) > -0.3).astype(np.float32)
    return 0.2 * x * env


def echo_of(far: np.ndarray, delay_s: float = 0.04, gain: float = 0.5) -> np.ndarray:
    d = int(delay_s * SR)
    rng = np.random.default_rng(7)
    ir = np.zeros(d + 800, dtype=np.float32)
    ir[d] = gain
    ir[d + 1 :] = rng.standard_normal(799) * np.exp(-np.arange(799) / 200) * 0.03
    return np.convolve(far, ir)[: len(far)].astype(np.float32)


def test_reference_and_options() -> None:
    ref = EchoReference()
    ref.push(np.ones(100, dtype=np.float32))
    ref.push(np.full(50, 2.0, dtype=np.float32))
    out = ref.take(120)
    assert out[:100].tolist() == [1.0] * 100 and out[100:].tolist() == [2.0] * 20
    assert ref.pending_s == pytest.approx(30 / SR)
    ref.clear()
    assert not ref.take(10).any()  # nothing playing: silence
    assert parse_echo(None) is None and parse_echo(False) is None and parse_echo({"enabled": False}) is None
    assert parse_echo(True).model is None  # type: ignore[union-attr]
    assert parse_echo("dtln-aec-128").model == "dtln-aec-128"  # type: ignore[union-attr]
    assert parse_echo({"model": "speex-aec"}).model == "speex-aec"  # type: ignore[union-attr]
    assert parse_echo({"model": "dtln-aec-128", "delay_ms": 120}).delay_ms == 120  # type: ignore[union-attr]
    with pytest.raises(InvalidRequest):
        parse_echo(3)
    with pytest.raises(InvalidRequest, match="delay_ms"):
        parse_echo({"delay_ms": 5000})


def test_delay_line_holds_the_reference_back() -> None:
    from bolo.echo import EchoCanceller, FakeEchoCanceller

    model = FakeEchoCanceller({"id": "fake-echo"})
    canceller = EchoCanceller(model, delay_ms=10)  # 160 samples
    canceller.far(np.ones(320, dtype=np.float32))
    mic = np.full(320, 0.5, dtype=np.float32)
    out = canceller.near(mic)  # the fake mutes a piece with any reference in it
    assert not out.any() and model.reference_s == pytest.approx(160 / SR)
    out = canceller.near(mic)  # the last 160 samples of reference come out of the line now
    assert model.reference_s == pytest.approx(320 / SR) and not out.any()
    assert canceller.near(mic).tolist() == mic.tolist()  # nothing left: the microphone goes through
    assert canceller.summary()["delay_ms"] == 10


@pytest.mark.parametrize("engine", ["speex", "webrtc"])
def test_cancellers_remove_an_echo_and_keep_the_near_end(engine: str) -> None:
    if engine == "speex":
        pytest.importorskip("pyaec")
        model: Any = SpeexAec({"id": "speex-aec"})
    else:
        pytest.importorskip("livekit")
        model = WebRtcAec3({"id": "webrtc-aec3"})
    far = speechlike(6.0, 1)
    mic = echo_of(far)
    stream = model.create_stream()
    out = np.concatenate([stream.process(mic[i : i + 320], far[i : i + 320]) for i in range(0, len(mic), 320)])
    assert len(out) == len(mic)
    tail = slice(3 * SR, 6 * SR)  # after it has adapted
    erle = 10 * np.log10(np.mean(mic[tail] ** 2) / (np.mean(out[tail] ** 2) + 1e-12))
    assert erle > 15, erle
    # no far end: the near end goes through
    near = speechlike(2.0, 2)
    quiet = model.create_stream()
    kept = np.concatenate(
        [quiet.process(near[i : i + 320], np.zeros(320, dtype=np.float32)) for i in range(0, len(near), 320)]
    )
    level = 10 * np.log10(np.mean(kept[SR:] ** 2) / np.mean(near[SR:] ** 2))
    assert level > -6, level


async def _events_until(ws: Any, last: str, timeout: float = 20.0) -> list[dict[str, Any]]:
    out = []
    async with asyncio.timeout(timeout):
        while True:
            ev = json.loads(await ws.recv())
            out.append(ev)
            if ev["type"] == last:
                return out


async def _converse(srv: LiveServer, echo: Any) -> tuple[list[str], dict[str, Any]]:
    """Speak one answer, then send it back as the microphone would hear it: what does the session hear?"""
    async with websockets.connect(srv.ws_url + "/v1/realtime") as ws:
        await _events_until(ws, "session.created")
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime", "audio": {
            "input": {"echo_cancellation": echo}, "output": {}}}}))  # fmt: skip
        updated = (await _events_until(ws, "session.updated"))[-1]
        silence = np.zeros(4800, dtype="<i2").tobytes()  # the microphone is open from the start
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(silence).decode()}))
        await ws.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello there"}]}}))  # fmt: skip
        await ws.send(json.dumps({"type": "response.create"}))
        events = await _events_until(ws, "response.done")
        pcm = b"".join(base64.b64decode(e["delta"]) for e in events if e["type"] == "response.output_audio.delta")
        assert len(pcm) > 24000  # half a second of the agent's voice at least
        for i in range(0, len(pcm), 960):  # 20 ms pieces, as a browser sends them
            piece = base64.b64encode(pcm[i : i + 960]).decode()
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": piece}))
        tail = base64.b64encode(np.zeros(24000, dtype="<i2").tobytes()).decode()
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": tail}))
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime"}}))
        heard = await _events_until(ws, "session.updated")
        return [e["type"] for e in heard], {"first": updated["session"], "last": heard[-1]["session"]}


async def test_session_does_not_hear_its_own_voice(agent: LiveServer) -> None:
    types, sessions = await _converse(agent, "fake-echo")
    assert sessions["first"]["audio"]["input"]["echo_cancellation"]["model"] == "fake-echo"
    info = sessions["last"]["audio"]["input"]["echo_cancellation"]
    assert info["with_reference_s"] > 0.4 and info["audio_s"] > info["with_reference_s"]
    assert "input_audio_buffer.speech_started" not in types  # the echo did not barge in
    canceller = agent.core.manager.loaded["fake-echo"].model
    assert canceller.reference_s > 0.4  # type: ignore[attr-defined]
    # without it, the agent's own voice is heard as the user speaking
    types, sessions = await _converse(agent, False)
    assert sessions["last"]["audio"]["input"]["echo_cancellation"] is None
    assert "input_audio_buffer.speech_started" in types


async def test_unknown_canceller_is_refused(agent: LiveServer) -> None:
    async with websockets.connect(agent.ws_url + "/v1/realtime") as ws:
        await _events_until(ws, "session.created")
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime", "audio": {
            "input": {"echo_cancellation": "no-such-model"}}}}))  # fmt: skip
        error = (await _events_until(ws, "error"))[-1]["error"]
        assert error["param"] == "session.audio.input.echo_cancellation"


async def test_session_with_speex(agent: LiveServer) -> None:
    """SpeexDSP adapts within the fake voice's three seconds of tones (WebRTC's AEC3 needs longer, and speech)."""
    pytest.importorskip("pyaec")
    types, sessions = await _converse(agent, "speex-aec")
    info = sessions["last"]["audio"]["input"]["echo_cancellation"]
    assert info["model"] == "speex-aec" and info["with_reference_s"] > 0.4 and "error" not in info
    assert "input_audio_buffer.speech_started" not in types
