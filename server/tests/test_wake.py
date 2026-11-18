"""Wake words (bolo.wake): POST /api/v1/wake, WS /api/v1/wake/stream, the Wyoming wake service and bolo wake, with
a fake model that "hears" its phrase in every loud burst."""

from __future__ import annotations

import asyncio
import io
import json
import threading

import httpx
import numpy as np
import pytest
import soundfile as sf
import websockets

from bolo.config import Settings
from bolo.errors import InvalidRequest
from bolo.wake import parse_keywords
from conftest import LiveServer

SR = 16000


def bursts(at: tuple[float, ...] = (1.0, 3.0), seconds: float = 4.5) -> np.ndarray:
    x = np.zeros(int(seconds * SR), dtype=np.float32)
    t = np.arange(int(0.4 * SR)) / SR
    for start in at:
        i = int(start * SR)
        x[i : i + len(t)] = 0.5 * np.sin(2 * np.pi * 440 * t)
    return x


def wav(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def live(settings: Settings) -> LiveServer:
    settings.default_wake_word = "fake-wake"
    srv = LiveServer(settings)
    srv.start()
    yield srv
    srv.stop()


def test_parse_keywords() -> None:
    assert parse_keywords(None) is None and parse_keywords("") is None
    assert parse_keywords("hey computer, ok bolo;\nwake up") == ["hey computer", "ok bolo", "wake up"]
    assert parse_keywords(["a", "b,c"]) == ["a", "b", "c"]
    with pytest.raises(InvalidRequest):
        parse_keywords([3])
    with pytest.raises(InvalidRequest):
        parse_keywords(["x"] * 51)


def test_wake_route(live: LiveServer) -> None:
    r = httpx.post(live.url + "/api/v1/wake", files={"file": ("a.wav", wav(bursts()), "audio/wav")}, timeout=30)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "fake-wake" and body["detected"] and body["keywords"] == ["hey fake"]
    times = [d["time"] for d in body["detections"]]
    assert len(times) == 2 and times[0] == pytest.approx(1.1, abs=0.15) and times[1] == pytest.approx(3.1, abs=0.15)
    r = httpx.post(
        live.url + "/api/v1/wake",
        files={"file": ("a.wav", wav(bursts((2.0,))), "audio/wav")},
        data={"keywords": "ok computer"},
        timeout=30,
    )
    assert [d["keyword"] for d in r.json()["detections"]] == ["ok computer"]
    quiet = httpx.post(live.url + "/api/v1/wake", files={"file": ("a.wav", wav(np.zeros(SR)), "audio/wav")})
    assert quiet.json()["detected"] is False
    bad = httpx.post(
        live.url + "/api/v1/wake", files={"file": ("a.wav", wav(np.zeros(SR)), "audio/wav")}, data={"model": "fake-stt"}
    )
    assert bad.status_code == 400
    catalog = httpx.get(live.url + "/api/v1/catalog", params={"task": "wake-word"}).json()["models"]
    fake = next(m for m in catalog if m["id"] == "fake-wake")
    assert fake["wake_phrases"] == ["hey fake"] and fake["custom_keywords"] is True


async def test_wake_stream(live: LiveServer) -> None:
    url = live.ws_url + "/api/v1/wake/stream?keywords=hey%20there&sample_rate=8000"
    async with websockets.connect(url) as ws:
        ready = json.loads(await ws.recv())
        assert ready == {"type": "ready", "model": "fake-wake", "keywords": ["hey there"]}
        x = bursts((0.5,), 1.5)[::2]  # 8 kHz, resampled by the server
        pcm = (x * 32767).astype("<i2").tobytes()
        for i in range(0, len(pcm), 640):
            await ws.send(pcm[i : i + 640])
        hit = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert hit["type"] == "wake" and hit["keyword"] == "hey there" and hit["time"] == pytest.approx(0.6, abs=0.15)
        await ws.send(json.dumps({"type": "stop"}))


async def test_wyoming_wake(settings: Settings) -> None:
    pytest.importorskip("wyoming")
    import socket

    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.client import AsyncTcpClient
    from wyoming.info import Describe, Info
    from wyoming.wake import Detect, Detection, NotDetected

    from bolo.core import Bolo
    from bolo.wyoming_server import start_wyoming

    settings.default_wake_word = "fake-wake"
    core = Bolo(settings)
    await core.start()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = await start_wyoming(core, "127.0.0.1", port)
    try:
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Describe().event())
            info = Info.from_event(await client.read_event())
            assert "fake-wake" in [m.name for p in info.wake for m in p.models]
            await client.write_event(Detect(names=["fake-wake:hey there"]).event())
            await client.write_event(AudioStart(rate=SR, width=2, channels=1).event())
            pcm = (bursts((0.5,), 1.5) * 32767).astype("<i2").tobytes()
            for i in range(0, len(pcm), 3200):
                await client.write_event(AudioChunk(rate=SR, width=2, channels=1, audio=pcm[i : i + 3200]).event())
            event = await asyncio.wait_for(client.read_event(), 10)
            assert Detection.is_type(event.type)
            hit = Detection.from_event(event)
            assert hit.name == "fake-wake:hey there" and hit.timestamp == pytest.approx(600, abs=150)
            await client.write_event(AudioStop().event())
            # a second listen that hears nothing ends with not-detected
            await client.write_event(Detect(names=["fake-wake"]).event())
            await client.write_event(AudioStart(rate=SR, width=2, channels=1).event())
            silence = np.zeros(SR, dtype="<i2").tobytes()
            await client.write_event(AudioChunk(rate=SR, width=2, channels=1, audio=silence).event())
            await client.write_event(AudioStop().event())
            assert NotDetected.is_type((await asyncio.wait_for(client.read_event(), 10)).type)
    finally:
        await server.stop()
        await core.stop()


def test_cli_wake(settings: Settings, tmp_path: object, capsys: pytest.CaptureFixture[str]) -> None:
    from pathlib import Path

    from bolo import cli

    path = Path(str(tmp_path)) / "a.wav"
    path.write_bytes(wav(bursts((1.0,), 2.0)))
    code = cli.main(
        ["wake", str(path), "--fake-engine", "--data-dir", str(settings.data_dir), "-m", "fake-wake", "-k", "hi bolo",
         "--json"]
    )  # fmt: skip
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert [d["keyword"] for d in body["detections"]] == ["hi bolo"]


def test_keyword_spotter_hears_a_wake_word_at_the_very_start() -> None:
    """sherpa-onnx's spotter missed a wake word said from the stream's first sample: each stream first hears a moment
    of silence, which the times it reports leave out."""
    from bolo import wake

    class Stream:
        def __init__(self) -> None:
            self.fed: list[np.ndarray] = []

        def accept_waveform(self, sr: int, x: np.ndarray) -> None:
            assert sr == SR
            self.fed.append(np.asarray(x))

    class Kws:
        def __init__(self) -> None:
            self.streams: list[Stream] = []
            self.ready = False

        def create_stream(self, spec: str | None = None) -> Stream:
            self.streams.append(Stream())
            return self.streams[-1]

        def is_ready(self, s: Stream) -> bool:
            ready, self.ready = self.ready, False
            return ready

        def decode_stream(self, s: Stream) -> None:
            pass

        def get_result(self, s: Stream) -> str:
            return "HEY_JARVIS"

        def reset_stream(self, s: Stream) -> None:
            pass

    model = wake.SherpaKeywordSpotter.__new__(wake.SherpaKeywordSpotter)
    model.kws = Kws()
    model._lock = threading.Lock()
    model._default_names = {"HEY_JARVIS": "HEY JARVIS"}
    model.default_keywords = ["HEY JARVIS"]
    detector = model.create_detector()
    stream = model.kws.streams[0]
    assert len(stream.fed) == 1 and len(stream.fed[0]) >= int(0.1 * SR) and not stream.fed[0].any()

    model.kws.ready = True
    hits = detector.accept(np.full(SR, 0.1, dtype=np.float32))
    assert [(h.keyword, h.time) for h in hits] == [("HEY JARVIS", 1.0)]  # the silence ahead is not counted
    assert len(stream.fed) == 2
