from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import numpy as np
import pytest
import websockets

from bolo.audio import mulaw_encode
from conftest import LiveServer, speechlike


def pcm16(x: np.ndarray) -> bytes:
    return (x * 32767).astype("<i2").tobytes()


async def stream_audio(ws, payload: bytes, frame: int, delay: float = 0.02) -> None:  # type: ignore[no-untyped-def]
    for off in range(0, len(payload), frame):
        await ws.send(payload[off : off + frame])
        await asyncio.sleep(delay)


async def collect(ws) -> list[dict]:  # type: ignore[no-untyped-def]
    out = []
    async for message in ws:
        out.append(json.loads(message))
    return out


async def test_stt_stream_native(live: LiveServer) -> None:
    async with websockets.connect(live.ws_url + "/api/v1/stt/stream?model=fake-stt-streaming") as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready" and ready["model"] == "fake-stt-streaming"
        assert ready["sample_rate"] == 16000 and ready["request_id"].startswith("r_") and ready["mode"] == "native"
        audio = np.concatenate([speechlike(2.0), np.zeros(16000, dtype=np.float32)])
        await stream_audio(ws, pcm16(audio), 3200)
        await ws.send(json.dumps({"type": "stop"}))
        messages = await collect(ws)
    types = [m["type"] for m in messages]
    assert types[-1] == "done" and messages[-1]["request_id"] == ready["request_id"]
    partials = [m for m in messages if m["type"] == "partial"]
    finals = [m for m in messages if m["type"] == "final"]
    assert len(partials) >= 3 and len(finals) == 1
    assert set(partials[0]) == {"type", "text", "t0", "t1", "latency_ms", "stable_chars"}
    assert set(finals[0]) == {"type", "text", "t0", "t1", "latency_ms", "words", "language"}
    assert finals[0]["text"].startswith("the quick brown fox") and finals[0]["language"] == "en"
    assert finals[0]["words"][0] == {"w": "the", "t0": 0.0, "t1": 0.4}
    metrics = [m for m in messages if m["type"] == "metrics"]
    assert metrics and set(metrics[-1]) == {"type", "audio_seconds", "rtf", "queue_ms"}
    assert metrics[-1]["audio_seconds"] == pytest.approx(3.0, abs=0.01)
    vad = [m for m in messages if m["type"] == "vad"]
    assert [v["event"] for v in vad] == ["speech_start", "speech_end"] and vad[1]["t"] == pytest.approx(2.0, abs=0.2)
    trace = httpx.get(live.url + f"/api/v1/requests/{ready['request_id']}").json()
    assert trace["transport"] == "websocket" and trace["status"] == "ok" and trace["utterances"] == 1
    assert trace["first_partial_ms"] == partials[0]["latency_ms"] and trace["final_ms"] == finals[0]["latency_ms"]
    assert httpx.get(live.url + "/api/v1/stats").json()["active_streams"] == 0


async def test_stt_stream_default_model_is_the_realtime_default(live: LiveServer) -> None:
    async with websockets.connect(live.ws_url + "/api/v1/stt/stream") as ws:
        assert json.loads(await ws.recv())["model"] == "fake-stt-streaming"
        stats = httpx.get(live.url + "/api/v1/stats").json()
        assert stats["active_streams"] == 1


async def test_stt_stream_telephony_mulaw_8k(live: LiveServer) -> None:
    url = live.ws_url + "/api/v1/stt/stream?model=fake-stt&encoding=mulaw&sample_rate=8000"
    async with websockets.connect(url) as ws:
        ready = json.loads(await ws.recv())
        assert ready["encoding"] == "mulaw" and ready["sample_rate"] == 8000 and ready["mode"] == "pseudo"
        audio = np.concatenate([speechlike(2.0, 8000), np.zeros(8000, dtype=np.float32)])
        await stream_audio(ws, mulaw_encode(audio), 160, delay=0.004)  # 20 ms RTP-sized frames
        await ws.send(json.dumps({"type": "stop"}))
        messages = await collect(ws)
    finals = [m for m in messages if m["type"] == "final"]
    assert len(finals) == 1 and finals[0]["text"].startswith("the quick brown fox")
    assert finals[0]["t1"] == pytest.approx(2.2, abs=0.3)  # times are in seconds of the 8 kHz input stream
    assert [m for m in messages if m["type"] == "metrics"][-1]["audio_seconds"] == pytest.approx(3.0, abs=0.02)


async def test_stt_stream_flush_config_and_float_pcm(live: LiveServer) -> None:
    async with websockets.connect(live.ws_url + "/api/v1/stt/stream?model=fake-stt&vad=false&interim=false") as ws:
        assert json.loads(await ws.recv())["encoding"] == "pcm_s16le"
        await ws.send(json.dumps({"type": "config", "encoding": "pcm_f32le", "sample_rate": 48000, "language": "hi"}))
        ready = json.loads(await ws.recv())
        assert ready["encoding"] == "pcm_f32le" and ready["sample_rate"] == 48000 and ready["vad"] is None
        await ws.send(speechlike(1.2, 48000).astype("<f4").tobytes())
        await ws.send(json.dumps({"type": "flush"}))
        first = json.loads(await ws.recv())
        assert first["type"] == "final" and first["text"].startswith("the quick") and first["language"] == "hi"
        assert json.loads(await ws.recv())["type"] == "metrics"
        await ws.send(speechlike(0.8, 48000).astype("<f4").tobytes())
        await ws.send(json.dumps({"type": "ping"}))
        await ws.send(json.dumps({"type": "stop"}))
        rest = await collect(ws)
    types = [m["type"] for m in rest]
    assert "pong" in types and types[-1] == "done" and "partial" not in types
    second = next(m for m in rest if m["type"] == "final")
    assert second["text"].startswith("the") and second["t0"] == pytest.approx(1.2, abs=0.05)


async def test_stt_stream_errors(live: LiveServer) -> None:
    async with websockets.connect(live.ws_url + "/api/v1/stt/stream?model=missing") as ws:
        err = json.loads(await ws.recv())
        assert err == {"type": "error", "code": "model_not_found", "message": err["message"]}
    async with websockets.connect(live.ws_url + "/api/v1/stt/stream?encoding=g729") as ws:
        assert json.loads(await ws.recv())["code"] == "invalid_request"
    async with websockets.connect(live.ws_url + "/api/v1/stt/stream?model=fake-tts") as ws:
        assert json.loads(await ws.recv())["type"] == "error"
    async with websockets.connect(live.ws_url + "/api/v1/stt/stream?model=fake-stt") as ws:
        await ws.recv()
        await ws.send("not json")
        assert json.loads(await ws.recv())["code"] == "invalid_request"


async def test_stt_stream_client_disconnect_is_recorded_as_cancelled(live: LiveServer) -> None:
    async with websockets.connect(live.ws_url + "/api/v1/stt/stream?model=fake-stt-streaming") as ws:
        request_id = json.loads(await ws.recv())["request_id"]
        await ws.send(pcm16(speechlike(0.5)))
        await asyncio.sleep(0.1)
    for _ in range(50):
        r = httpx.get(live.url + f"/api/v1/requests/{request_id}")
        if r.status_code == 200:
            break
        await asyncio.sleep(0.05)
    assert r.json()["status"] == "cancelled"
    assert httpx.get(live.url + "/api/v1/stats").json()["active_streams"] == 0


async def test_stt_stream_send_to_a_gone_client_is_cancelled_not_an_error(core: Any) -> None:
    """The client can leave while a message is on its way to it (the WebSocket send fails before the disconnect is
    read): the stream is recorded as cancelled, as when the disconnect is read first."""
    from bolo.streaming.session import ClientDisconnected, SessionConfig, STTSession

    sent: list[dict[str, Any]] = []

    async def send(msg: dict[str, Any]) -> None:
        if msg["type"] not in ("ready",):
            raise ClientDisconnected("gone")
        sent.append(msg)

    session = STTSession(core, send, SessionConfig(model="fake-stt-streaming"))
    await session.start()
    session.feed(pcm16(speechlike(1.0)))
    async with asyncio.timeout(10):
        while session.trace is None or session.trace.status not in ("cancelled", "error", "ok"):
            await asyncio.sleep(0.02)
    assert session.trace.status == "cancelled" and session.trace.error == "client disconnected"
    assert core.active_streams == 0
    await session.close()


async def test_tts_stream_incremental_text(live: LiveServer) -> None:
    async with websockets.connect(live.ws_url + "/api/v1/tts/stream?model=fake-tts&voice=alpha") as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready" and ready["sample_rate"] == 24000 and ready["format"] == "pcm_s16le"
        for piece in ["Hello wor", "ld. And then some", " more words"]:
            await ws.send(json.dumps({"type": "text", "text": piece}))
        await ws.send(json.dumps({"type": "stop"}))
        chunks, audio, others = [], bytearray(), []
        async for message in ws:
            if isinstance(message, bytes):
                audio.extend(message)
            else:
                msg = json.loads(message)
                (chunks if msg["type"] == "chunk" else others).append(msg)
    assert [c["seq"] for c in chunks] == list(range(len(chunks)))
    assert set(chunks[0]) == {"type", "seq", "t0", "dur", "ttfa_ms", "text"}
    assert [c["text"] for c in chunks if c["text"]] == ["Hello world.", "And then some more words"]
    assert chunks[0]["ttfa_ms"] is not None and chunks[0]["ttfa_ms"] < 500 and chunks[1]["ttfa_ms"] is None
    total = sum(c["dur"] for c in chunks)
    assert chunks[-1]["t0"] + chunks[-1]["dur"] == pytest.approx(total, abs=0.01)
    assert len(audio) == pytest.approx(total * 24000 * 2, rel=0.01)
    assert [m["type"] for m in others] == ["metrics", "metrics", "done"]
    assert others[0]["rtf"] > 0 and others[0]["request_id"].startswith("r_")
    traces = httpx.get(live.url + "/api/v1/requests", params={"kind": "tts"}).json()["requests"]
    assert len(traces) == 2 and all(t["transport"] == "websocket" and t["ttfa_ms"] is not None for t in traces)


async def test_tts_stream_formats_and_errors(live: LiveServer) -> None:
    async with websockets.connect(live.ws_url + "/api/v1/tts/stream?model=fake-tts&format=mulaw") as ws:
        assert json.loads(await ws.recv())["sample_rate"] == 8000
        await ws.send(json.dumps({"type": "text", "text": "Hello there.", "flush": True}))
        await ws.send(json.dumps({"type": "stop"}))
        audio = b"".join([m async for m in ws if isinstance(m, bytes)])
    assert len(audio) == pytest.approx(0.72 * 8000, rel=0.02)
    async with websockets.connect(
        live.ws_url + "/api/v1/tts/stream?model=fake-tts&format=pcm_f32le&sample_rate=16000"
    ) as ws:
        assert json.loads(await ws.recv())["sample_rate"] == 16000
        await ws.send(json.dumps({"type": "text", "text": "Hello there."}))
        await ws.send(json.dumps({"type": "stop"}))
        audio = b"".join([m async for m in ws if isinstance(m, bytes)])
    assert len(audio) == pytest.approx(0.72 * 16000 * 4, rel=0.02)
    async with websockets.connect(live.ws_url + "/api/v1/tts/stream?model=fake-tts&voice=nobody") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "text", "text": "Hello there."}))
        await ws.send(json.dumps({"type": "flush"}))
        messages = [json.loads(m) async for m in ws if isinstance(m, str)]
    assert messages[-1]["type"] == "error" and "voice" in messages[-1]["message"]
    async with websockets.connect(live.ws_url + "/api/v1/tts/stream?model=fake-stt") as ws:
        assert json.loads(await ws.recv())["type"] == "error"


async def test_ws_auth(settings) -> None:  # type: ignore[no-untyped-def]
    settings.api_key = "sekrit"
    srv = LiveServer(settings)
    srv.start()
    try:
        async with websockets.connect(srv.ws_url + "/api/v1/stt/stream") as ws:
            assert json.loads(await ws.recv())["code"] == "invalid_api_key"
        async with websockets.connect(srv.ws_url + "/api/v1/stt/stream?api_key=sekrit") as ws:
            assert json.loads(await ws.recv())["type"] == "ready"
        async with websockets.connect(
            srv.ws_url + "/api/v1/tts/stream", additional_headers={"Authorization": "Bearer sekrit"}
        ) as ws:
            assert json.loads(await ws.recv())["type"] == "ready"
    finally:
        srv.stop()


def band_limited_mulaw(seconds: float = 2.5) -> bytes:
    """Speech-like audio as a phone line delivers it: 8 kHz, nothing below 300 Hz or above 3400 Hz, mu-law."""
    rate = 8000
    rng = np.random.default_rng(3)
    t = np.arange(int(seconds * rate)) / rate
    # a buzzy source with harmonics up to Nyquist plus noise, amplitude-modulated like syllables
    source = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 33)) + 0.3 * rng.standard_normal(len(t))
    source *= 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    spectrum = np.fft.rfft(source)
    freqs = np.fft.rfftfreq(len(source), 1 / rate)
    spectrum[(freqs < 300) | (freqs > 3400)] = 0  # brick-wall telephone band
    x = np.fft.irfft(spectrum, len(source))
    x = (0.25 * x / np.abs(x).max()).astype(np.float32)
    check = np.abs(np.fft.rfft(x))
    assert check[(freqs < 250) | (freqs > 3500)].max() < 1e-3 * check.max()
    return mulaw_encode(np.concatenate([np.zeros(4000, dtype=np.float32), x, np.zeros(8000, dtype=np.float32)]))


async def test_band_limited_telephone_audio_is_detected_and_finalised(live: LiveServer) -> None:
    payload = band_limited_mulaw()
    for model in ("fake-stt", "fake-stt-streaming"):
        async with websockets.connect(
            live.ws_url + f"/api/v1/stt/stream?model={model}&encoding=mulaw&sample_rate=8000"
        ) as ws:
            await ws.recv()
            await stream_audio(ws, payload, 160, delay=0)
            await ws.send(json.dumps({"type": "stop"}))
            messages = await collect(ws)
        vad = [m["event"] for m in messages if m["type"] == "vad"]
        finals = [m for m in messages if m["type"] == "final"]
        assert vad == ["speech_start", "speech_end"], model
        assert len(finals) == 1 and finals[0]["text"], model
        assert finals[0]["t0"] == pytest.approx(0.35, abs=0.3) and finals[0]["t1"] == pytest.approx(3.1, abs=0.6), model


async def test_endpointing_is_configurable_per_stream(live: LiveServer) -> None:
    # two bursts of speech separated by 0.4 s: one utterance with patient endpointing, two with fast
    burst = speechlike(1.0)
    audio = np.concatenate([burst, np.zeros(6400, dtype=np.float32), burst, np.zeros(16000, dtype=np.float32)])
    counts = {}
    for query in ("endpointing=fast", "endpoint_ms=800"):
        async with websockets.connect(live.ws_url + f"/api/v1/stt/stream?model=fake-stt&{query}") as ws:
            ready = json.loads(await ws.recv())
            await stream_audio(ws, pcm16(audio), 640, delay=0.002)
            await ws.send(json.dumps({"type": "stop"}))
            messages = await collect(ws)
        counts[query] = (ready["endpoint_ms"], len([m for m in messages if m["type"] == "final"]))
    assert counts == {"endpointing=fast": (250, 2), "endpoint_ms=800": (800, 1)}
    async with websockets.connect(live.ws_url + "/api/v1/stt/stream?endpoint_ms=5") as ws:
        assert json.loads(await ws.recv())["code"] == "invalid_request"
    async with websockets.connect(live.ws_url + "/api/v1/stt/stream?endpointing=psychic") as ws:
        assert json.loads(await ws.recv())["code"] == "invalid_request"


async def test_stream_trace_has_one_phase_per_name(live: LiveServer) -> None:
    audio = np.concatenate([speechlike(1.0), np.zeros(12000, dtype=np.float32)] * 3)
    async with websockets.connect(live.ws_url + "/api/v1/stt/stream?model=fake-stt") as ws:
        request_id = json.loads(await ws.recv())["request_id"]
        await stream_audio(ws, pcm16(audio), 1600, delay=0.01)
        await ws.send(json.dumps({"type": "stop"}))
        messages = await collect(ws)
    assert len([m for m in messages if m["type"] == "final"]) == 3
    trace = httpx.get(live.url + f"/api/v1/requests/{request_id}").json()
    names = [p["name"] for p in trace["phases"]]
    assert len(names) == len(set(names)) and {"vad", "decode", "finalize"} <= set(names)
    assert trace["utterances"] == 3
