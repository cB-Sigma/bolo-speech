"""The cloud APIs added after the first five, against stand-ins written from each provider's public documentation
(tests/fake_cloud_apis.py): request and message shapes, authentication headers, live sockets, speaker labels and
error mapping. Nothing here talks to a real provider."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import numpy as np
import pytest
import websockets

from bolo.catalog import Catalog
from bolo.config import Settings
from conftest import LiveServer, speechlike, wav_bytes
from fake_cloud_apis import KEY, FakeCloudHTTP, FakeCloudSockets
from test_ws import pcm16, stream_audio


@pytest.fixture
def http() -> Iterator[FakeCloudHTTP]:
    server = FakeCloudHTTP().start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def sockets() -> Iterator[FakeCloudSockets]:
    server = FakeCloudSockets().start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def srv(settings: Settings, http: FakeCloudHTTP, sockets: FakeCloudSockets) -> Iterator[LiveServer]:
    settings.providers = {
        "assemblyai": {"base_url": http.url, "api_key": KEY, "options": {"realtime_url": sockets.url.replace("http", "ws") + "/v3/ws"}},
    }  # fmt: skip
    server = LiveServer(settings)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def transcribe(server: LiveServer, model: str, seconds: float = 2.0, **data: Any) -> httpx.Response:
    return httpx.post(
        f"{server.url}/v1/audio/transcriptions",
        data={"model": model, **data},
        files={"file": ("a.wav", wav_bytes(seconds), "audio/wav")},
        timeout=60,
    )


def trace_of(server: LiveServer, response: httpx.Response) -> dict[str, Any]:
    doc: dict[str, Any] = httpx.get(f"{server.url}/api/v1/requests/{response.headers['x-request-id']}").json()
    return doc


async def run_live(server: LiveServer, model: str, audio: np.ndarray, extra: str = "") -> list[dict[str, Any]]:
    async with websockets.connect(f"{server.ws_url}/api/v1/stt/stream?model={model}&vad=false{extra}") as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready" and ready["mode"] == "native", ready
        await stream_audio(ws, pcm16(audio), 3200, delay=0.02)
        await ws.send(json.dumps({"type": "stop"}))
        return [json.loads(m) async for m in ws]


async def eventually(check: Callable[[], bool], seconds: float = 5.0) -> bool:
    """A socket's last messages go out after the session's ``done``: wait for them."""
    for _ in range(int(seconds / 0.05)):
        if check():
            return True
        await asyncio.sleep(0.05)
    return check()


# -- the catalog ------------------------------------------------------------------------------------------------
NEW_ENTRIES = {
    "assemblyai-stt-3.7": ("assemblyai", "assemblyai-stt-3-7", False),
    "assemblyai-stt-3.7-realtime": ("assemblyai", "assemblyai-stt-3-7-realtime", True),
    "assemblyai-stt-3.7-realtime-agent": ("assemblyai", "assemblyai-stt-3-7-realtime-agent", True),
}


def test_new_cloud_entries_are_labelled_like_the_others() -> None:
    catalog = Catalog.load()
    for mid, (provider, model, streaming) in NEW_ENTRIES.items():
        m = catalog.resolve(mid)
        assert m is not None, mid
        assert m.cloud and m.engine == "remote" and m.validated == "untested", mid  # added, not yet run
        assert m.options["provider"] == provider and m.options["model"] == model and m.streaming is streaming, mid
        assert "leaves this server" in m.description and m.license_url.startswith("https://"), mid
        assert f"BOLO_PROVIDER_{provider.upper().replace('-', '_')}_API_KEY" in m.description, mid
        api = m.to_api()
        assert api["notice"] == "paid cloud API; audio leaves this server" and m.languages, mid


def test_the_catalog_names_the_key_variable(srv: LiveServer) -> None:
    listed = {m["id"]: m for m in httpx.get(f"{srv.url}/api/v1/catalog", timeout=60).json()["models"]}
    aai = listed["assemblyai-stt-3.7"]
    assert aai["installed"] is True and aai["cloud"] is True
    assert aai["provider"]["key_env"] == "BOLO_PROVIDER_ASSEMBLYAI_API_KEY" and aai["provider"]["configured"] is True
    assert listed["openai-whisper-1"]["provider"]["key_env"] == "BOLO_PROVIDER_OPENAI_API_KEY"
    assert listed["openai-whisper-1"]["provider"]["configured"] is False


# -- AssemblyAI ---------------------------------------------------------------------------------------------------
def test_assemblyai_recorded_audio(srv: LiveServer, http: FakeCloudHTTP) -> None:
    r = transcribe(
        srv, "assemblyai-stt-3.7", language="en", prompt="A call about bolo.", hotwords="Bolo, New Delhi",
        response_format="verbose_json", **{"timestamp_granularities[]": "word"},
    )  # fmt: skip
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["text"] == "Heard 2.0 seconds." and [w["word"] for w in doc["words"]] == ["Heard", "2.0", "seconds."]
    assert doc["language"] == "english" and doc["words"][1]["start"] == pytest.approx(0.666, abs=0.01)
    upload = http.last("/v2/upload", "POST")
    # the key goes in Authorization as it is (no Bearer); the WAV is uploaded raw
    assert upload["headers"]["Authorization"] == KEY and upload["headers"]["Content-Type"] == "application/octet-stream"
    assert upload["body"][:4] == b"RIFF"
    body = http.last("/v2/transcript", "POST")["json"]
    assert body == {
        "speech_models": ["assemblyai-stt-3-7"], "language_code": "en", "keyterms_prompt": ["Bolo", "New Delhi"],
        "prompt": "A call about bolo.", "audio_url": body["audio_url"],
    }  # fmt: skip
    assert KEY not in json.dumps(body)
    # queued, processing, completed: two polls; then the transcript is deleted at AssemblyAI
    polls = [q for q in http.requests if q["method"] == "GET" and q["path"].startswith("/v2/transcript/")]
    assert len(polls) == 3 and len(http.deleted) == 1
    trace = trace_of(srv, r)
    assert trace["provider"]["kind"] == "assemblyai" and trace["provider"]["requests"] == 5  # upload, submit, 3 looks
    # no language: AssemblyAI detects it
    r = transcribe(srv, "assemblyai-stt-3.7")
    assert r.status_code == 200 and http.last("/v2/transcript", "POST")["json"]["language_detection"] is True
    # the realtime entries send recorded audio (and Realtime turns) to the pre-recorded model of the same generation
    r = transcribe(srv, "assemblyai-stt-3.7-realtime-agent")
    assert r.status_code == 200 and http.last("/v2/transcript", "POST")["json"]["speech_models"] == [
        "assemblyai-stt-3-7"
    ]
    test = httpx.post(f"{srv.url}/api/v1/providers/assemblyai/test").json()
    assert test["ok"] and test["checked"] == "credentials"
    assert http.last("/v2/transcript", "GET")["query"] == {"limit": "1"}


def test_assemblyai_labels_speakers_itself(srv: LiveServer, http: FakeCloudHTTP) -> None:
    r = transcribe(srv, "assemblyai-stt-3.7", language="en", response_format="diarized_json")
    assert r.status_code == 200, r.text
    segments = r.json()["segments"]
    assert [(s["speaker"], s["text"]) for s in segments] == [("A", "Heard"), ("B", "2.0 seconds.")]
    assert http.last("/v2/transcript", "POST")["json"]["speaker_labels"] is True
    assert trace_of(srv, r)["diarization_model"] == "provider"
    # without diarization, nothing about speakers is asked for or answered
    r = transcribe(srv, "assemblyai-stt-3.7", language="en", response_format="verbose_json")
    assert "speaker_labels" not in http.last("/v2/transcript", "POST")["json"]
    assert all("speaker" not in s for s in r.json()["segments"])


def test_assemblyai_errors(srv: LiveServer, http: FakeCloudHTTP) -> None:
    http.assemblyai_fail = "Transcoding failed. File does not appear to contain audio."
    r = transcribe(srv, "assemblyai-stt-3.7", language="en")
    assert r.status_code == 502 and "does not appear to contain audio" in r.json()["error"]["message"]
    r = transcribe(srv, "assemblyai-stt-3.7", language="en", temperature="0.4")
    assert r.status_code == 400 and r.json()["error"]["param"] == "temperature"
    http.key = "another-key"
    r = transcribe(srv, "assemblyai-stt-3.7", language="en")
    assert r.status_code == 502 and r.json()["error"]["code"] == "provider_auth"
    assert KEY not in r.text


async def test_assemblyai_live_socket_with_speaker_labels(srv: LiveServer, sockets: FakeCloudSockets) -> None:
    gap = np.zeros(8000, dtype=np.float32)
    audio = np.concatenate([speechlike(1.5), gap, speechlike(1.0), gap])
    messages = await run_live(srv, "assemblyai-stt-3.7-realtime", audio, "&language=en&diarize=true")
    partials = [m["text"] for m in messages if m["type"] == "partial"]
    finals = [m for m in messages if m["type"] == "final"]
    assert partials and partials[0].startswith("hearing ")
    # two turns, each labelled by AssemblyAI (bolo ran no speaker model of its own)
    assert [f["text"] for f in finals] == ["Heard 1.5 seconds.", "Heard 1.0 seconds."]
    assert [f["speaker"] for f in finals] == ["A", "B"]
    assert messages[-1]["type"] == "done"
    path = sockets.paths[0]
    assert path.startswith("/v3/ws?speech_model=assemblyai-stt-3-7-realtime&sample_rate=16000&encoding=pcm_s16le")
    assert "language_codes=%5B%22en%22%5D" in path and "speaker_labels=true" in path and KEY not in path
    assert sockets.headers[0]["Authorization"] == KEY
    # every frame within AssemblyAI's 20 ms to 1 s; the session ends with Terminate
    assert sockets.frames[0] and all(640 <= n <= 32000 for n in sockets.frames[0])
    assert await eventually(lambda: sockets.messages[0][-1:] == [{"type": "Terminate"}])


async def test_assemblyai_live_socket_without_diarization(srv: LiveServer, sockets: FakeCloudSockets) -> None:
    audio = np.concatenate([speechlike(1.2), np.zeros(8000, dtype=np.float32)])
    messages = await run_live(srv, "assemblyai-stt-3.7-realtime-agent", audio)
    finals = [m for m in messages if m["type"] == "final"]
    assert [f["text"] for f in finals] == ["Heard 1.2 seconds."] and "speaker" not in finals[0]
    path = sockets.paths[0]
    assert "speech_model=assemblyai-stt-3-7-realtime-agent" in path and "speaker_labels" not in path
    assert "language_codes" not in path
