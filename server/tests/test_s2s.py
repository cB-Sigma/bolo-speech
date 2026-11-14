"""Realtime pipelines other than the cascade: speech-to-speech (a fake llama-server /v1/realtime) and audio-llm."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import websockets

from bolo.config import Settings
from bolo.s2s import AudioDelta, LlamaRealtimeBackend, ResponseEnd, S2SConfig, TextDelta, ToolCall, realtime_url
from conftest import LiveServer
from fake_realtime_server import FakeRealtimeServer
from test_realtime import b64_pcm24

HERE = Path(__file__).parent
TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]


@pytest.fixture
def upstream() -> Iterator[FakeRealtimeServer]:
    server = FakeRealtimeServer(api_key="up-secret").start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def s2s(settings: Settings, upstream: FakeRealtimeServer) -> Iterator[LiveServer]:
    settings.s2s_server_url = upstream.url
    settings.s2s_server_api_key = "up-secret"
    srv = LiveServer(settings)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


async def connect(server: LiveServer) -> Any:
    ws = await websockets.connect(server.ws_url + "/v1/realtime?model=gpt-realtime")
    created = json.loads(await ws.recv())
    assert created["type"] == "session.created"
    return ws


async def update(ws: Any, session: dict[str, Any]) -> dict[str, Any]:
    await ws.send(json.dumps({"type": "session.update", "session": session}))
    while True:
        event = json.loads(await ws.recv())
        if event["type"] in ("session.updated", "error"):
            return event  # type: ignore[no-any-return]


async def until(ws: Any, last: str, timeout: float = 20.0, count: int = 1) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async with asyncio.timeout(timeout):
        while True:
            event = json.loads(await ws.recv())
            events.append(event)
            if event["type"] == last:
                count -= 1
                if count == 0:
                    return events


def test_realtime_url() -> None:
    assert realtime_url("http://h:8080") == "ws://h:8080/v1/realtime"
    assert realtime_url("https://h/v1/") == "wss://h/v1/realtime"
    assert realtime_url("ws://h:1/v1/realtime") == "ws://h:1/v1/realtime"
    assert realtime_url("h:9") == "ws://h:9/v1/realtime"


async def test_speech_to_speech_turn(s2s: LiveServer, upstream: FakeRealtimeServer) -> None:
    ws = await connect(s2s)
    try:
        updated = await update(
            ws,
            {
                "type": "realtime", "pipeline": "speech-to-speech", "instructions": "Be brief.", "tools": TOOLS,
                "audio": {"input": {"transcription": {"model": "fake-stt"},
                                    "turn_detection": {"type": "server_vad", "silence_duration_ms": 300}},
                          "output": {"format": {"type": "audio/pcmu"}, "voice": "alloy"}},
            },
        )  # fmt: skip
        assert updated["type"] == "session.updated" and updated["session"]["pipeline"] == "speech-to-speech"
        for chunk in b64_pcm24(2.0):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
        events = await until(ws, "response.done")
        # the user transcript comes from bolo's own STT, alongside the response
        events += (
            []
            if any(e["type"].endswith("transcription.completed") for e in events)
            else await until(ws, "conversation.item.input_audio_transcription.completed")
        )
    finally:
        await ws.close()
    types = [e["type"] for e in events]
    order = [
        "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped", "input_audio_buffer.committed",
        "response.created", "response.output_item.added", "response.content_part.added",
        "response.output_audio_transcript.delta", "response.output_audio.delta", "response.output_audio.done",
        "response.output_audio_transcript.done", "response.output_item.done", "response.done",
    ]  # fmt: skip
    positions = [types.index(t) for t in order]
    assert positions == sorted(positions), types
    done = next(e for e in events if e["type"] == "response.done")["response"]
    transcript = done["output"][0]["content"][0]["transcript"]
    assert done["status"] == "completed" and transcript.startswith("I heard ") and transcript.endswith(" seconds.")
    heard = float(transcript.split()[2])
    assert 2.0 <= heard <= 2.8  # the turn's speech (and prefix padding), not the silence before or after
    assert done["usage"]["output_tokens"] == 4
    # 0.1 s of model audio per word, re-encoded to 8 kHz mu-law for the client
    mulaw = b"".join(base64.b64decode(e["delta"]) for e in events if e["type"] == "response.output_audio.delta")
    assert len(mulaw) == pytest.approx(4 * 0.1 * 8000, abs=8)
    completed = next(e for e in events if e["type"] == "conversation.item.input_audio_transcription.completed")
    assert completed["transcript"].startswith("the quick brown fox")
    # what went upstream: the session (16 kHz in, tools, no OpenAI voice name), the turn's audio, commit, response
    sent = upstream.events[-1]
    kinds = [e["type"] for e in sent]
    assert kinds[0] == "session.update"
    session = sent[0]["session"]
    assert session["instructions"] == "Be brief." and session["tools"][0]["name"] == "get_weather"
    assert session["audio"]["input"]["format"]["rate"] == 16000 and "output" not in session["audio"]
    assert kinds.index("input_audio_buffer.commit") < kinds.index("response.create")
    assert kinds[-2:] == ["input_audio_buffer.commit", "response.create"]
    trace = httpx.get(s2s.url + "/api/v1/requests", params={"kind": "agent"}).json()["requests"][0]
    assert trace["status"] == "ok" and trace["llm_model"] == "lfm3-audio-2b-llamacpp" and trace["ttfa_ms"] > 0


async def test_speech_to_speech_text_and_tool_calls(s2s: LiveServer, upstream: FakeRealtimeServer) -> None:
    ws = await connect(s2s)
    try:
        await update(
            ws, {"type": "realtime", "pipeline": "speech-to-speech", "tools": TOOLS, "output_modalities": ["text"]}
        )
        item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "the weather in Pune?"}]}
        await ws.send(json.dumps({"type": "conversation.item.create", "item": item}))
        await ws.send(json.dumps({"type": "response.create"}))
        first = await until(ws, "response.done")
        call = next(e for e in first if e["type"] == "response.function_call_arguments.done")
        assert call["name"] == "get_weather" and json.loads(call["arguments"]) == {"city": "Pune"}
        assert call["call_id"] == "call_0"
        output = {"type": "function_call_output", "call_id": "call_0", "output": "sunny"}
        await ws.send(json.dumps({"type": "conversation.item.create", "item": output}))
        await ws.send(json.dumps({"type": "response.create"}))
        second = await until(ws, "response.done")
    finally:
        await ws.close()
    text = "".join(e["delta"] for e in second if e["type"] == "response.output_text.delta")
    assert text == "The tool said: sunny."
    assert not any(e["type"] == "response.output_audio.delta" for e in second)
    sent = [e for e in upstream.events[-1] if e["type"] in ("conversation.item.create", "response.create")]
    assert sent[0]["item"]["content"][0]["text"] == "the weather in Pune?"
    assert sent[1]["response"]["output_modalities"] == ["text"]
    assert sent[2]["item"] == output


async def test_speech_to_speech_barge_in(s2s: LiveServer, upstream: FakeRealtimeServer) -> None:
    upstream.word_delay_s = 0.25
    ws = await connect(s2s)
    try:
        await update(
            ws,
            {"type": "realtime", "pipeline": "speech-to-speech",
             "audio": {"input": {"transcription": {"model": "fake-stt"},
                                 "turn_detection": {"type": "server_vad", "silence_duration_ms": 300}}}},
        )  # fmt: skip
        for chunk in b64_pcm24(1.5):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
        await until(ws, "response.output_audio.delta")
        for chunk in b64_pcm24(1.2):  # the user talks over the model
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
            await asyncio.sleep(0.02)
        events = await until(ws, "response.done")
        upstream.word_delay_s = 0.0
        second = await until(ws, "response.done")
    finally:
        await ws.close()
    cancelled = events[-1]["response"]
    assert cancelled["status"] == "cancelled" and cancelled["status_details"]["reason"] == "turn_detected"
    assert upstream.cancelled == 1
    assert second[-1]["response"]["status"] == "completed"
    # the second answer is about the second turn only: the rest of the cancelled one was dropped
    assert second[-1]["response"]["output"][0]["content"][0]["transcript"].startswith("I heard 1.")


async def test_pipeline_validation_and_unreachable_backend(settings: Settings) -> None:
    settings.s2s_server_url = "http://127.0.0.1:9"  # nothing listens there
    srv = LiveServer(settings)
    srv.start()
    try:
        ws = await connect(srv)
        try:
            bad = await update(ws, {"type": "realtime", "pipeline": "telepathy"})
            assert bad["type"] == "error" and bad["error"]["param"] == "session.pipeline"
            await update(ws, {"type": "realtime", "pipeline": "speech-to-speech"})
            item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            await ws.send(json.dumps({"type": "conversation.item.create", "item": item}))
            error = await until(ws, "error")
            assert error[-1]["error"]["code"] == "engine_unavailable"
            assert "not reachable" in error[-1]["error"]["message"]
        finally:
            await ws.close()
    finally:
        srv.stop()


async def test_backend_refuses_a_voice(upstream: FakeRealtimeServer) -> None:
    backend = LlamaRealtimeBackend(upstream.url, api_key="up-secret")
    with pytest.raises(Exception, match="unknown voice"):
        await backend.open(S2SConfig(voice="robot"))
    wrong = LlamaRealtimeBackend(upstream.url, api_key="nope")
    with pytest.raises(Exception, match="not reachable"):
        await wrong.open(S2SConfig())


def fake_llama_server(tmp_path: Path) -> Path:
    path = tmp_path / "bin" / "llama-server"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(HERE)!r})\n"
                    "from fake_realtime_server import main\nmain(sys.argv[1:])\n")  # fmt: skip
    path.chmod(0o755)
    return path


async def test_managed_server(tmp_path: Path) -> None:
    from bolo.core import Bolo

    settings = Settings(
        data_dir=tmp_path / "data", llama_server_bin=str(fake_llama_server(tmp_path)), s2s_parallel=3, threads=2
    )
    core = Bolo(settings)
    model = tmp_path / "model"
    model.mkdir()
    (model / "m.gguf").write_bytes(b"GGUF")
    (model / "mm.gguf").write_bytes(b"GGUF")
    process = core.s2s._process("lfm3-test", model, {"model_file": "m.gguf", "mmproj_file": "mm.gguf"})
    backend = LlamaRealtimeBackend(process=process, model="lfm3-test")
    try:
        conv = await backend.open(S2SConfig(instructions="hi"))
        assert process.running and process.api_key and "--parallel" in process.argv
        assert process.argv[process.argv.index("--parallel") + 1] == "3"
        await conv.add_text("hello")
        events = [e async for e in conv.respond(modalities=["audio"])]
        assert isinstance(events[-1], ResponseEnd) and events[-1].status == "completed"
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "You wrote: hello."
        audio = [e for e in events if isinstance(e, AudioDelta)]
        assert audio and audio[0].sample_rate == 24000 and not any(isinstance(e, ToolCall) for e in events)
        assert backend.describe()["managed"] is True
        await conv.close()
    finally:
        await backend.close()
    assert not process.running


async def test_audio_llm_pipeline(agent: LiveServer) -> None:
    ws = await connect(agent)
    try:
        await update(
            ws,
            {"type": "realtime", "pipeline": "audio-llm",
             "audio": {"input": {"transcription": {"model": "fake-stt"},
                                 "turn_detection": {"type": "server_vad", "silence_duration_ms": 300}}}},
        )  # fmt: skip
        for chunk in b64_pcm24(1.5):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
        events = await until(ws, "response.done")
        types = [e["type"] for e in events]
        if "conversation.item.input_audio_transcription.completed" not in types:
            events += await until(ws, "conversation.item.input_audio_transcription.completed")
        item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "and then?"}]}
        await ws.send(json.dumps({"type": "conversation.item.create", "item": item}))
        await ws.send(json.dumps({"type": "response.create"}))
        await until(ws, "response.done")
    finally:
        await ws.close()
    assert any(e["type"] == "response.output_audio.delta" for e in events)  # bolo's TTS speaks the answer
    first, second = agent.llm.requests[-2:]  # type: ignore[attr-defined]
    content = first["messages"][-1]["content"]
    assert isinstance(content, list) and content[0]["type"] == "input_audio"
    assert base64.b64decode(content[0]["input_audio"]["data"])[:4] == b"RIFF"
    assert content[0]["input_audio"]["format"] == "wav"
    # the next request still carries that turn as audio (it is the latest heard turn), then the text
    assert second["messages"][-1] == {"role": "user", "content": "and then?"}
    assert isinstance(second["messages"][-3]["content"], list)
