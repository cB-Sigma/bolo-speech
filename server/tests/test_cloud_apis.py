"""The cloud APIs added after the first five, against stand-ins written from each provider's public documentation
(tests/fake_cloud_apis.py): request and message shapes, authentication headers, live sockets, speaker labels and
error mapping. Nothing here talks to a real provider."""

from __future__ import annotations

import asyncio
import contextlib
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
from fake_cloud_apis import KEY, FakeCloudHTTP, FakeCloudSockets, FakeNovaStream, nova_chunk
from test_realtime import b64_pcm24
from test_s2s import TOOLS, connect, until, update
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
    ws_url = sockets.url.replace("http", "ws")
    settings.providers = {
        "assemblyai": {"base_url": http.url, "api_key": KEY, "options": {"realtime_url": ws_url + "/v3/ws"}},
        "elevenlabs": {"base_url": http.url, "api_key": KEY,
                       "options": {"realtime_url": ws_url + "/v1/speech-to-text/realtime"}},
        "azure-mai": {"base_url": http.url, "api_key": KEY,
                      "options": {"realtime_url": ws_url + "/mai/v1/realtime?intent=transcription",
                                  "tts_url": http.url + "/cognitiveservices/v1"}},
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
    "elevenlabs-scribe-v3": ("elevenlabs", "scribe_v3", False),
    "elevenlabs-scribe-v3-live": ("elevenlabs", "scribe_v3_live", True),
    "xai-grok-voice-3.0": ("xai", "grok-voice-3.0", True),
    "rime-prism": ("rime", "prism", True),
    "mai-transcribe-2-streaming": ("azure-mai", "MAI-Transcribe-2-Streaming", True),
    "mai-voice-2.1": ("azure-mai", "MAI-Voice-2.1", False),
    "mai-voice-2.1-flash": ("azure-mai", "MAI-Voice-2.1-Flash", False),
    "amazon-nova-2.5-sonic": ("bedrock", "amazon.nova-2-5-sonic", True),
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


# -- ElevenLabs Scribe v3 and Scribe v3 Live ---------------------------------------------------------------------------
def test_scribe_v3_labels_speakers_itself(srv: LiveServer, http: FakeCloudHTTP) -> None:
    r = transcribe(srv, "elevenlabs-scribe-v3", language="en", response_format="diarized_json")
    assert r.status_code == 200, r.text
    segments = r.json()["segments"]
    assert [(s["speaker"], s["text"]) for s in segments] == [("A", "Heard"), ("B", "2.0 seconds.")]
    form = http.last("/v1/speech-to-text", "POST")["form"]
    assert form["model_id"] == "scribe_v3" and form["diarize"] == "true" and form["language_code"] == "en"
    assert http.last("/v1/speech-to-text", "POST")["headers"]["xi-api-key"] == KEY
    assert trace_of(srv, r)["diarization_model"] == "provider"
    # without diarization nothing about speakers is asked for or answered
    r = transcribe(srv, "elevenlabs-scribe-v3", language="en", response_format="verbose_json")
    assert r.status_code == 200 and "diarize" not in http.last("/v1/speech-to-text", "POST")["form"]
    assert all("speaker" not in s for s in r.json()["segments"])
    # the live entry sends recorded audio (and Realtime turns) to scribe_v3
    r = transcribe(srv, "elevenlabs-scribe-v3-live", language="en")
    assert r.status_code == 200 and http.last("/v1/speech-to-text", "POST")["form"]["model_id"] == "scribe_v3"


async def test_scribe_v3_live_with_turn_prediction_and_speakers(srv: LiveServer, sockets: FakeCloudSockets) -> None:
    gap = np.zeros(8000, dtype=np.float32)
    audio = np.concatenate([speechlike(1.5), gap, speechlike(1.0), gap])
    messages = await run_live(srv, "elevenlabs-scribe-v3-live", audio, "&language=en&diarize=true")
    partials = [m["text"] for m in messages if m["type"] == "partial"]
    finals = [m for m in messages if m["type"] == "final"]
    assert partials and partials[0].startswith("hearing ")
    # each segment waited for its diarized transcript; the merge made the second speaker B
    assert [(f["text"], f["speaker"]) for f in finals] == [("Heard 1.5 seconds.", "A"), ("Heard 1.0 seconds.", "B")]
    path = sockets.paths[0]
    assert path.startswith("/v1/speech-to-text/realtime?model_id=scribe_v3_live&audio_format=pcm_16000")
    assert "commit_strategy=turn_prediction" in path and "include_timestamps=true" in path
    assert "language_code=en" in path and "diarize=true" in path and KEY not in path
    assert sockets.headers[0]["xi-api-key"] == KEY
    sent = sockets.messages[0]
    assert all(m["message_type"] == "input_audio_chunk" and m["sample_rate"] == 16000 for m in sent)
    assert sent[-1]["commit"] is True and not any(m["commit"] for m in sent[:-1])  # bolo commits at the end only


async def test_scribe_v3_live_without_speakers(srv: LiveServer, sockets: FakeCloudSockets) -> None:
    audio = np.concatenate([speechlike(1.2), np.zeros(8000, dtype=np.float32)])
    messages = await run_live(srv, "elevenlabs-scribe-v3-live", audio)
    finals = [m for m in messages if m["type"] == "final"]
    assert [f["text"] for f in finals] == ["Heard 1.2 seconds."] and "speaker" not in finals[0]
    assert "diarize" not in sockets.paths[0] and "language_code" not in sockets.paths[0]


# -- cloud speech-to-speech: xAI Grok Voice 3.0 and Rime Prism ------------------------------------------------------
@contextlib.contextmanager
def s2s_server(settings: Settings, dialect: str, model: str) -> Iterator[tuple[LiveServer, FakeCloudSockets]]:
    upstream = FakeCloudSockets(realtime=dialect).start()
    settings.providers = {dialect: {"base_url": upstream.url + "/v1", "api_key": KEY}}
    settings.s2s_model = model
    server = LiveServer(settings)
    server.start()
    try:
        yield server, upstream
    finally:
        server.stop()
        upstream.stop()


async def spoken_turn(server: LiveServer, voice: str) -> list[dict[str, Any]]:
    ws = await connect(server)
    try:
        updated = await update(
            ws,
            {
                "type": "realtime", "pipeline": "speech-to-speech", "instructions": "Be brief.", "tools": TOOLS,
                "audio": {"input": {"transcription": {"model": "fake-stt"},
                                    "turn_detection": {"type": "server_vad", "silence_duration_ms": 300}},
                          "output": {"voice": voice}},
            },
        )  # fmt: skip
        assert updated["type"] == "session.updated", updated
        for chunk in b64_pcm24(2.0):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
        return await until(ws, "response.done")
    finally:
        await ws.close()


async def tool_round(server: LiveServer) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ws = await connect(server)
    try:
        await update(
            ws, {"type": "realtime", "pipeline": "speech-to-speech", "tools": TOOLS, "output_modalities": ["text"]}
        )
        item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "the weather in Pune?"}]}
        await ws.send(json.dumps({"type": "conversation.item.create", "item": item}))
        await ws.send(json.dumps({"type": "response.create"}))
        first = await until(ws, "response.done")
        output = {"type": "function_call_output", "call_id": "call_7", "output": "sunny"}
        await ws.send(json.dumps({"type": "conversation.item.create", "item": output}))
        await ws.send(json.dumps({"type": "response.create"}))
        second = await until(ws, "response.done")
        return first, second
    finally:
        await ws.close()


def spoken_text(events: list[dict[str, Any]]) -> str:
    done = next(e for e in events if e["type"] == "response.done")["response"]
    assert done["status"] == "completed", done
    return str(done["output"][0]["content"][0]["transcript"])


async def test_xai_grok_voice_speech_to_speech(settings: Settings) -> None:
    with s2s_server(settings, "xai", "xai-grok-voice-3.0") as (server, upstream):
        caps = httpx.get(f"{server.url}/api/v1/info", timeout=30).json()["capabilities"]["realtime"]["pipelines"]
        assert caps["speech-to-speech"]["configured"] is True and caps["speech-to-speech"]["available"] is True
        events = await spoken_turn(server, "marin")
        transcript = spoken_text(events)
        # the stand-in measured what it was sent at the 16 kHz bolo declared: the turn's speech
        assert transcript.startswith("I heard ") and 2.0 <= float(transcript.split()[2]) <= 2.8, transcript
        assert any(e["type"] == "response.output_audio.delta" for e in events)
        assert upstream.paths == ["/v1/realtime?model=grok-voice-3.0"]
        assert upstream.headers[0]["Authorization"] == f"Bearer {KEY}"
        session = upstream.messages[0][0]["session"]
        # xAI's session shape: OpenAI's voice name is not xAI's, so the entry's default voice; bolo commits turns
        assert session["voice"] == "eve" and session["turn_detection"] == {"type": None}
        assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 16000}
        assert session["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
        assert session["tools"][0] == {
            "type": "function",
            "function": {k: v for k, v in TOOLS[0].items() if k != "type"},
        }
        kinds = [e["type"] for e in upstream.messages[0]]
        assert kinds.index("input_audio_buffer.commit") < kinds.index("response.create")
        create = next(e for e in upstream.messages[0] if e["type"] == "response.create")
        assert create["response"] == {"modalities": ["text", "audio"]}
        described = httpx.get(f"{server.url}/api/v1/info", timeout=30).json()["capabilities"]["realtime"]["pipelines"]
        assert described["speech-to-speech"]["backend"] == "xai-realtime" and KEY not in json.dumps(described)
        test = httpx.post(f"{server.url}/api/v1/providers/xai/test", timeout=30).json()
        assert test["ok"] and test["checked"] == "credentials" and "2 voices" in test["message"]


async def test_xai_text_turns_and_tool_calls(settings: Settings) -> None:
    with s2s_server(settings, "xai", "xai-grok-voice-3.0") as (server, upstream):
        first, second = await tool_round(server)
        call = next(e for e in first if e["type"] == "response.function_call_arguments.done")
        assert call["name"] == "get_weather" and call["call_id"] == "call_7"
        # one call, though xAI announces it twice (arguments.done and output_item.done)
        assert sum(e["type"] == "response.function_call_arguments.done" for e in first) == 1
        text = "".join(e["delta"] for e in second if e["type"] == "response.output_text.delta")
        assert text == "The tool said: sunny." and not any(e["type"] == "response.output_audio.delta" for e in second)
        creates = [e for e in upstream.messages[-1] if e["type"] == "response.create"]
        assert creates[0]["response"] == {"modalities": ["text"]}


async def test_rime_prism_speech_to_speech(settings: Settings) -> None:
    with s2s_server(settings, "rime", "rime-prism") as (server, upstream):
        events = await spoken_turn(server, "aurora")
        transcript = spoken_text(events)
        assert transcript.startswith("I heard ") and 2.0 <= float(transcript.split()[2]) <= 2.8, transcript
        assert any(e["type"] == "response.output_audio.delta" for e in events)
        # no query string, the key in the upgrade's Authorization header, and only fields Prism knows
        assert upstream.paths == ["/v1/realtime"] and upstream.headers[0]["Authorization"] == f"Bearer {KEY}"
        session = upstream.messages[0][0]["session"]
        assert session == {"turn_detection": None, "tools": [TOOLS[0]], "voice": "aurora", "instructions": "Be brief."}
        assert not any(e["type"] == "error" for e in events)
        create = next(e for e in upstream.messages[0] if e["type"] == "response.create")
        assert set(create) == {"type", "event_id"}  # turn detection off: a plain request answers the turn
        test = httpx.post(f"{server.url}/api/v1/providers/rime/test", timeout=30).json()
        assert test["ok"] and test["checked"] == "credentials"
        upstream.key = "another-key"
        test = httpx.post(f"{server.url}/api/v1/providers/rime/test", timeout=30).json()
        assert not test["ok"] and test["code"] == "provider_auth" and KEY not in json.dumps(test)


async def test_rime_prism_typed_turns_and_tool_continuations(settings: Settings) -> None:
    with s2s_server(settings, "rime", "rime-prism") as (server, upstream):
        first, second = await tool_round(server)
        call = next(e for e in first if e["type"] == "response.function_call_arguments.done")
        assert call["name"] == "get_weather" and call["call_id"] == "call_7"
        text = "".join(e["delta"] for e in second if e["type"] == "response.output_text.delta")
        # Prism always speaks; a text-only response leaves its audio out
        assert text == "The tool said: sunny." and not any(e["type"] == "response.output_audio.delta" for e in second)
        creates = [e for e in upstream.messages[-1] if e["type"] == "response.create"]
        assert creates[0]["response"] == {
            "prism_input_text": "the weather in Pune?",
            "metadata": {"prism_cause": "user_text"},
        }
        # the continuation waited for Prism's acknowledgement of the result and names the reply that made the call
        assert creates[1]["response"] == {
            "metadata": {"prism_cause": "tool_continuation", "prism_parent_response_id": "resp_1"}
        }
        # the typed text is never sent as a conversation item (Prism takes typed turns in response.create)
        assert not any(
            e["type"] == "conversation.item.create" and e["item"]["type"] == "message" for e in upstream.messages[-1]
        )


async def test_cloud_speech_to_speech_without_its_provider(settings: Settings) -> None:
    settings.providers = {}
    settings.s2s_model = "rime-prism"
    server = LiveServer(settings)
    server.start()
    try:
        caps = httpx.get(f"{server.url}/api/v1/info", timeout=30).json()["capabilities"]["realtime"]["pipelines"]
        s2s = caps["speech-to-speech"]
        assert s2s["configured"] is False and s2s["available"] is False
        assert "BOLO_PROVIDER_RIME_API_KEY" in s2s["reason"]
        r = httpx.post(f"{server.url}/v1/audio/transcriptions", data={"model": "rime-prism"},
                       files={"file": ("a.wav", wav_bytes(1.0), "audio/wav")}, timeout=30)  # fmt: skip
        assert r.status_code >= 400
    finally:
        server.stop()


# -- Microsoft MAI-Transcribe-2-Streaming and MAI-Voice-2.1 ----------------------------------------------------------
async def test_mai_transcribe_streaming_commits_at_pauses(srv: LiveServer, sockets: FakeCloudSockets) -> None:
    gap = np.zeros(8000, dtype=np.float32)
    audio = np.concatenate([speechlike(1.5), gap, speechlike(1.0), gap])
    messages = await run_live(srv, "mai-transcribe-2-streaming", audio, "&language=en")
    partials = [m["text"] for m in messages if m["type"] == "partial"]
    finals = [m["text"] for m in messages if m["type"] == "final"]
    assert partials and partials[0].startswith("hearing ")
    # MAI has no speech detection of its own: bolo committed at each pause, and each commit came back as a final
    assert finals == ["Heard 1.5 seconds.", "Heard 1.0 seconds."]
    assert sockets.paths[0] == "/mai/v1/realtime?intent=transcription" and sockets.headers[0]["api-key"] == KEY
    sent = sockets.messages[0]
    assert sent[0] == {
        "type": "session.update",
        "session": {"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": 16000},
            "transcription": {"model": "MAI-Transcribe-2-Streaming", "language": "en"},
            "turn_detection": None, "noise_reduction": None}}},
    }  # fmt: skip
    kinds = [m["type"] for m in sent]
    assert set(kinds[1:]) <= {"input_audio_buffer.append", "input_audio_buffer.commit"}
    assert kinds.count("input_audio_buffer.commit") >= 2


def test_mai_transcribe_recorded_audio_and_deployments(
    settings: Settings, srv: LiveServer, sockets: FakeCloudSockets
) -> None:
    r = transcribe(srv, "mai-transcribe-2-streaming", language="hi")
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "Heard 2.0 seconds."
    sent = sockets.messages[-1]
    assert sent[0]["session"]["audio"]["input"]["transcription"] == {
        "model": "MAI-Transcribe-2-Streaming",
        "language": "hi",
    }
    assert sent[-1] == {"type": "input_audio_buffer.commit"} and sent[1]["type"] == "input_audio_buffer.append"
    # options a socket recogniser has no use for are refused, not dropped silently
    r = transcribe(srv, "mai-transcribe-2-streaming", prompt="About bolo.")
    assert r.status_code == 400 and r.json()["error"]["param"] == "prompt"
    test = httpx.post(f"{srv.url}/api/v1/providers/azure-mai/test", timeout=30).json()
    assert test["ok"] and test["checked"] == "credentials"


def test_mai_needs_the_address_of_the_resource(settings: Settings) -> None:
    settings.providers = {"azure-mai": {"api_key": KEY, "options": {"region": "eastus"}}}
    server = LiveServer(settings)
    server.start()
    try:
        listed = {m["id"]: m for m in httpx.get(f"{server.url}/api/v1/catalog", timeout=60).json()["models"]}
        mai = listed["mai-transcribe-2-streaming"]
        assert mai["installed"] is False and mai["provider"]["configured"] is False
        assert mai["provider"]["key_env"] == "BOLO_PROVIDER_AZURE_MAI_API_KEY"
        r = transcribe(server, "mai-transcribe-2-streaming")
        assert r.status_code == 503 and r.json()["error"]["code"] == "provider_not_configured"
        assert "BOLO_PROVIDER_AZURE_MAI_BASE_URL" in r.json()["error"]["message"]
    finally:
        server.stop()


def test_mai_voice(srv: LiveServer, http: FakeCloudHTTP) -> None:
    import io

    import soundfile as sf

    r = httpx.post(
        f"{srv.url}/v1/audio/speech",
        json={"model": "mai-voice-2.1-flash", "input": "Hola, buenos días.", "voice": "es-MX-Valeria",
              "response_format": "wav"},
        timeout=60,
    )  # fmt: skip
    assert r.status_code == 200, r.text
    call = http.last("/cognitiveservices/v1", "POST")
    assert call["voice"] == "es-MX-Valeria:MAI-Voice-2.1-Flash" and call["lang"] == "es-MX"
    assert call["headers"]["Ocp-Apim-Subscription-Key"] == KEY
    assert call["headers"]["X-Microsoft-OutputFormat"] == "audio-24khz-160kbitrate-mono-mp3"
    with sf.SoundFile(io.BytesIO(r.content)) as f:
        seconds = len(f) / f.samplerate
    assert seconds == pytest.approx(0.05 * len(call["text"]), rel=0.1)  # the MP3 was decoded whole
    # the default voice, the long-form model, text that needs escaping in SSML
    r = httpx.post(f"{srv.url}/v1/audio/speech", json={"model": "mai-voice-2.1", "input": "Fish & chips <now>."},
                   timeout=60)  # fmt: skip
    assert r.status_code == 200, r.text
    call = http.last("/cognitiveservices/v1", "POST")
    assert call["voice"] == "en-US-Harper:MAI-Voice-2.1" and "&amp;" in call["text"] and "&lt;now&gt;" in call["text"]
    voices = {
        v["id"]
        for v in httpx.get(f"{srv.url}/v1/audio/voices", params={"model": "mai-voice-2.1"}, timeout=30).json()["voices"]
    }
    assert {"en-US-Harper", "hi-IN-Kavya", "es-MX-Valeria"} <= voices and len(voices) == 97
    r = httpx.post(
        f"{srv.url}/v1/audio/speech", json={"model": "mai-voice-2.1", "input": "Hi.", "speed": 1.5}, timeout=60
    )
    assert r.status_code == 400 and r.json()["error"]["param"] == "speed"


# -- Amazon Nova 2.5 Sonic (Bedrock, through AWS's SDK: a stand-in stream) ------------------------------------------
#: not real AWS credentials: an id and a secret of the documented form, for bolo to split and pass on
BEDROCK_KEY = "TESTACCESSKEYIDNOTREAL:test-secret-not-real-0b5e"
NOVA_SESSION = {
    "type": "realtime", "pipeline": "speech-to-speech", "instructions": "Answer in one sentence.",
    "audio": {"input": {"transcription": {"model": "fake-stt"},
                        "turn_detection": {"type": "server_vad", "silence_duration_ms": 300}},
              "output": {"voice": "tiffany"}},
}  # fmt: skip


@contextlib.contextmanager
def nova_server(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, stream: FakeNovaStream
) -> Iterator[tuple[LiveServer, list[Any]]]:
    from bolo.duplex import nova_sonic

    opened: list[Any] = []

    async def open_stream(backend: Any) -> tuple[FakeNovaStream, Any]:
        opened.append(backend)
        return stream, nova_chunk

    monkeypatch.setattr(nova_sonic.NovaSonicBackend, "_open_stream", open_stream)
    monkeypatch.setattr(nova_sonic, "sdk_missing", lambda: None)  # AWS's SDK is stood in for
    settings.providers = {"bedrock": {"api_key": BEDROCK_KEY, "options": {"region": "eu-north-1"}}}
    settings.s2s_model = "amazon-nova-2.5-sonic"
    settings.duplex_silence_ms = 400
    settings.duplex_jitter_ms = 100
    server = LiveServer(settings)
    server.start()
    try:
        yield server, opened
    finally:
        server.stop()


def nova_speech(speech_s: float, quiet_s: float) -> list[str]:
    import base64

    audio = np.concatenate([speechlike(speech_s, 24000), np.zeros(int(quiet_s * 24000), dtype=np.float32)])
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    return [base64.b64encode(pcm[i : i + 4800]).decode() for i in range(0, len(pcm), 4800)]


async def test_nova_sonic_session(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FakeNovaStream()
    with nova_server(settings, monkeypatch, stream) as (server, opened):
        info = httpx.get(f"{server.url}/api/v1/info", timeout=30).json()["capabilities"]["realtime"]["pipelines"]
        s2s = info["speech-to-speech"]
        assert s2s["full_duplex"] is True and s2s["available"] is True and s2s["configured"] is True
        ws = await connect(server)
        try:
            updated = await update(ws, NOVA_SESSION)
            assert updated["session"]["full_duplex"] is True
            for chunk in nova_speech(1.0, 0.6):
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
            # the client stops sending; bolo keeps feeding silence on the clock, so Nova hears the pause and answers
            events = await until(ws, "response.done", timeout=30)
        finally:
            await ws.close()
        assert await eventually(lambda: stream.closed)
        backend = opened[0]
    done = next(e for e in events if e["type"] == "response.done")["response"]
    transcript = done["output"][0]["content"][0]["transcript"]
    # the planned (speculative) text, once: the final text that repeats it after the audio is not doubled
    assert done["status"] == "completed" and transcript.startswith("I heard ") and transcript.count("I heard") == 1
    assert 0.9 <= float(transcript.split()[2]) <= 1.4, transcript
    assert any(e["type"] == "response.output_audio.delta" for e in events)
    # what went to Bedrock, in the documented order, with the credentials split out of the key
    names = [next(iter(e)) for e in stream.events]
    assert names[:6] == ["sessionStart", "promptStart", "contentStart", "textInput", "contentEnd", "contentStart"]
    prompt = stream.events[1]["promptStart"]["audioOutputConfiguration"]
    assert prompt["voiceId"] == "tiffany" and prompt["sampleRateHertz"] == 24000 and prompt["encoding"] == "base64"
    assert stream.events[3]["textInput"]["content"] == "Answer in one sentence."
    assert stream.events[2]["contentStart"]["role"] == "SYSTEM" and stream.events[5]["contentStart"]["type"] == "AUDIO"
    assert names[-3:] == ["contentEnd", "promptEnd", "sessionEnd"]
    assert stream.audio_seconds > 1.6  # what the client sent plus bolo's silence on the clock
    assert (backend.model_id, backend.region) == ("amazon.nova-2-5-sonic", "eu-north-1")
    assert backend.endpoint_uri == "https://bedrock-runtime.eu-north-1.amazonaws.com"
    assert backend.credentials == {
        "aws_access_key_id": "TESTACCESSKEYIDNOTREAL",
        "aws_secret_access_key": "test-secret-not-real-0b5e",
    }
    assert "test-secret" not in json.dumps(info)


async def test_nova_sonic_barge_in(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FakeNovaStream(word_delay_s=0.25)  # a long answer
    with nova_server(settings, monkeypatch, stream) as (server, _):
        ws = await connect(server)
        try:
            await update(ws, NOVA_SESSION)
            for chunk in nova_speech(1.0, 0.6):
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
            await until(ws, "response.output_audio.delta", timeout=30)
            for chunk in nova_speech(1.0, 0.0):  # the caller talks over it: Nova stops and says so
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
            events = await until(ws, "response.done", timeout=30)
        finally:
            await ws.close()
    done = events[-1]["response"]
    assert done["status"] == "cancelled" and done["status_details"]["reason"] == "turn_detected"
    assert stream.barge_ins == 1


async def test_nova_sonic_refusal_and_readiness(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FakeNovaStream(fail="Invalid input request, please fix your input and try again.")
    with nova_server(settings, monkeypatch, stream) as (server, _):
        ws = await connect(server)
        try:
            await update(ws, NOVA_SESSION)
            for chunk in nova_speech(0.5, 0.2):
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
            error = (await until(ws, "error", timeout=30))[-1]["error"]
        finally:
            await ws.close()
    assert "ValidationException" in error["message"] and "fix your input" in error["message"]
    # no credentials anywhere: not configured, and what to set is named
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    settings.providers = {}
    server = LiveServer(settings)
    server.start()
    try:
        s2s = httpx.get(f"{server.url}/api/v1/info", timeout=30).json()["capabilities"]["realtime"]["pipelines"]
        reason = s2s["speech-to-speech"]["reason"]
        assert s2s["speech-to-speech"]["configured"] is False
        assert "BOLO_PROVIDER_BEDROCK_API_KEY" in reason and "AWS_ACCESS_KEY_ID" in reason
    finally:
        server.stop()


def test_bedrock_settings_and_sdk_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from bolo.duplex.nova_sonic import bedrock_settings, sdk_missing
    from bolo.providers import ProviderConfig

    cfg = ProviderConfig.from_dict(
        "bedrock", {"api_key": "ID1:SECRET1:TOKEN1", "base_url": "https://bedrock-runtime.ap-northeast-1.amazonaws.com"}
    )
    assert bedrock_settings(cfg) == (
        "ap-northeast-1", "https://bedrock-runtime.ap-northeast-1.amazonaws.com",
        {"aws_access_key_id": "ID1", "aws_secret_access_key": "SECRET1", "aws_session_token": "TOKEN1"},
    )  # fmt: skip
    cfg = ProviderConfig.from_dict(
        "bedrock", {"api_key": "SECRET2", "options": {"access_key_id": "ID2", "region": "us-west-2"}}
    )
    assert bedrock_settings(cfg)[0] == "us-west-2"
    assert bedrock_settings(cfg)[2] == {"aws_access_key_id": "ID2", "aws_secret_access_key": "SECRET2"}
    # without a key, AWS's own variables make the provider usable (AWS's SDK reads them)
    bare = ProviderConfig.from_dict("bedrock", {})
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    assert bare.configured is False
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ID3")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET3")
    assert bare.configured is True
    monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime", None)  # not installed
    assert "bolo-speech[nova]" in (sdk_missing() or "")
