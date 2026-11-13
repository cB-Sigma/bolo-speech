"""The OpenAI Realtime API endpoint, with the official SDK's realtime client and a raw WebSocket."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx
import numpy as np
import openai
import pytest
import websockets

from bolo.audio import mulaw_decode, mulaw_encode
from bolo.config import Settings
from conftest import LiveServer, speechlike


def b64_pcm24(seconds: float, silence_after: float = 1.0) -> list[str]:
    """Speech-like audio then silence, as 100 ms chunks of 24 kHz PCM16 (the Realtime default format)."""
    audio = np.concatenate([speechlike(seconds, 24000), np.zeros(int(silence_after * 24000), dtype=np.float32)])
    pcm = (audio * 32767).astype("<i2").tobytes()
    return [base64.b64encode(pcm[i : i + 4800]).decode() for i in range(0, len(pcm), 4800)]


async def collect_until(conn: Any, last: str, timeout: float = 20.0) -> list[Any]:
    events = []
    async with asyncio.timeout(timeout):
        async for event in conn:
            events.append(event)
            if event.type == last:
                return events
    return events


def sdk(server: LiveServer) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=server.url + "/v1", api_key="x", max_retries=0)


async def test_transcription_session_with_the_sdk(live: LiveServer) -> None:
    async with sdk(live).realtime.connect(model="gpt-4o-transcribe") as conn:
        created = await conn.recv()
        assert created.type == "session.created" and created.session.type == "realtime"
        await conn.session.update(
            session={
                "type": "transcription",
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "transcription": {"model": "fake-stt", "language": "en"},
                                    "turn_detection": {"type": "server_vad", "silence_duration_ms": 300}}},
            }
        )  # fmt: skip
        updated = await conn.recv()
        assert updated.type == "session.updated" and updated.session.type == "transcription"
        assert updated.session.audio.input.transcription.model == "fake-stt"
        assert updated.session.audio.input.turn_detection.silence_duration_ms == 300
        for chunk in b64_pcm24(2.0):
            await conn.input_audio_buffer.append(audio=chunk)
        events = await collect_until(conn, "conversation.item.input_audio_transcription.completed")
    types = [e.type for e in events]
    assert types == [
        "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped", "input_audio_buffer.committed",
        "conversation.item.added", "conversation.item.done",
        "conversation.item.input_audio_transcription.delta", "conversation.item.input_audio_transcription.completed",
    ]  # fmt: skip
    started, stopped, committed, added, _done, delta, completed = events
    assert started.audio_start_ms < 300 and 1800 < stopped.audio_end_ms < 2600
    assert started.item_id == stopped.item_id == committed.item_id == added.item.id == completed.item_id
    assert added.item.role == "user" and added.item.content[0].type == "input_audio"
    assert completed.transcript.startswith("the quick brown fox") and delta.delta == completed.transcript
    assert completed.usage.type == "duration" and completed.usage.seconds == pytest.approx(2.3, abs=0.5)
    trace = httpx.get(live.url + "/api/v1/requests", params={"limit": 5}).json()["requests"]
    turn = next(t for t in trace if t["transport"] == "realtime" and t["kind"] == "stt" and t.get("stt_model"))
    assert turn["status"] == "ok" and turn["final_ms"] is not None


async def test_conversation_turn_with_the_sdk(agent: LiveServer) -> None:
    async with sdk(agent).realtime.connect(model="gpt-realtime") as conn:
        await conn.recv()
        await conn.session.update(
            session={
                "type": "realtime",
                "instructions": "Be brief.",
                "audio": {"input": {"transcription": {"model": "fake-stt"}}, "output": {"voice": "beta"}},
            }
        )
        updated = await conn.recv()
        assert updated.session.instructions == "Be brief." and updated.session.audio.output.voice == "beta"
        for chunk in b64_pcm24(2.0):
            await conn.input_audio_buffer.append(audio=chunk)
        events = await collect_until(conn, "response.done")
    types = [e.type for e in events]
    order = [
        "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped", "input_audio_buffer.committed",
        "conversation.item.input_audio_transcription.completed", "response.created", "response.output_item.added",
        "response.content_part.added", "response.output_audio_transcript.delta", "response.output_audio.delta",
        "response.output_audio.done", "response.output_audio_transcript.done", "response.content_part.done",
        "response.output_item.done", "response.done",
    ]  # fmt: skip
    positions = [types.index(t) for t in order]
    assert positions == sorted(positions), types
    by_type = {e.type: e for e in events}
    reply = "You said: the quick brown fox jumps. That is all I heard. Goodbye for now."
    assert by_type["response.output_audio_transcript.done"].transcript == reply
    assert "".join(e.delta for e in events if e.type == "response.output_audio_transcript.delta") == reply
    pcm = b"".join(base64.b64decode(e.delta) for e in events if e.type == "response.output_audio.delta")
    assert len(pcm) / 2 / 24000 == pytest.approx(len(reply) * 0.06, rel=0.15)  # 24 kHz PCM16 by default
    # audio starts while the model is still producing text
    assert types.index("response.output_audio.delta") < len(types) - 1 - types[::-1].index(
        "response.output_audio_transcript.delta"
    )
    done = by_type["response.done"].response
    assert done.status == "completed" and done.output[0].role == "assistant" and done.output[0].status == "completed"
    assert done.output[0].content[0].type == "output_audio" and done.output[0].content[0].transcript == reply
    assert done.usage.input_tokens == 7 and done.usage.output_tokens > 5
    assert by_type["response.created"].response.status == "in_progress"
    sent = agent.llm.requests[-1]  # type: ignore[attr-defined]
    assert sent["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "the quick brown fox jumps"},
    ]
    assert sent["model"] == "fake-llm" and sent["stream"] is True
    trace = httpx.get(agent.url + "/api/v1/requests", params={"kind": "agent"}).json()["requests"][0]
    assert trace["transport"] == "realtime" and trace["model"] == "voice-agent" and trace["status"] == "ok"
    assert [p["name"] for p in trace["phases"]] == ["decode", "llm_first_token", "synthesize_first", "synthesize_rest"]
    # end of speech -> transcript -> first LLM token -> first audio
    assert 0 < trace["final_ms"] < trace["first_token_ms"] < trace["ttfa_ms"] < trace["total_ms"]
    assert trace["first_token_ms"] - trace["final_ms"] == pytest.approx(
        50, abs=60
    )  # the fake model's first-token delay
    assert trace["llm_model"] == "fake-llm" and trace["stt_model"] == "fake-stt" and trace["audio_seconds_out"] > 3
    stats = {m["id"]: m for m in httpx.get(agent.url + "/api/v1/stats").json()["models"]}
    assert stats["voice-agent"]["task"] == "agent" and stats["voice-agent"]["latency_ms"]["ttfa"][
        "p50"
    ] == pytest.approx(trace["ttfa_ms"], abs=0.2)


async def test_barge_in_cancels_the_response(agent: LiveServer) -> None:
    agent.llm.token_delay_s = 0.15  # type: ignore[attr-defined]  a slow, long answer
    async with sdk(agent).realtime.connect(model="gpt-realtime") as conn:
        await conn.recv()
        await conn.session.update(
            session={
                "type": "realtime",
                "audio": {
                    "input": {
                        "transcription": {"model": "fake-stt"},
                        "turn_detection": {"type": "server_vad", "silence_duration_ms": 300},
                    }
                },
            }
        )
        await conn.recv()
        for chunk in b64_pcm24(1.5):
            await conn.input_audio_buffer.append(audio=chunk)
        first = await collect_until(conn, "response.output_audio.delta")
        assert first[-1].type == "response.output_audio.delta"
        for chunk in b64_pcm24(1.2):  # the user starts talking over the assistant
            await conn.input_audio_buffer.append(audio=chunk)
            await asyncio.sleep(0.02)
        events = await collect_until(conn, "response.done")
        agent.llm.token_delay_s = 0.0  # type: ignore[attr-defined]
        second = await collect_until(conn, "response.done")
    types = [e.type for e in events]
    assert "input_audio_buffer.speech_started" in types
    cancelled = events[-1].response
    assert cancelled.status == "cancelled" and cancelled.status_details.reason == "turn_detected"
    assert cancelled.output[0].status == "incomplete"
    assert types.index("input_audio_buffer.speech_started") < types.index("response.done")
    assert second[-1].response.status == "completed"  # the interrupting utterance gets its own answer
    statuses = [
        t["status"] for t in httpx.get(agent.url + "/api/v1/requests", params={"kind": "agent"}).json()["requests"]
    ]
    assert statuses == ["ok", "cancelled"]
    # the cancelled reply is in the history only as far as it got
    history = agent.llm.requests[-1]["messages"]  # type: ignore[attr-defined]
    partial = next(m for m in history if m["role"] == "assistant")["content"]
    assert partial and len(partial) < len("You said: the quick brown fox. That is all I heard. Goodbye for now.")


async def test_text_turns_tools_and_response_cancel(agent: LiveServer) -> None:
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    async with sdk(agent).realtime.connect(model="gpt-realtime") as conn:
        await conn.recv()
        await conn.session.update(
            session={"type": "realtime", "output_modalities": ["text"], "tools": tools, "tool_choice": "auto"}
        )
        updated = await conn.recv()
        assert updated.session.output_modalities == ["text"] and updated.session.tools[0].name == "get_weather"
        await conn.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What is the weather in Pune?"}],
            }
        )
        await conn.response.create()
        events = await collect_until(conn, "response.done")
        types = [e.type for e in events]
        assert types[:2] == ["conversation.item.added", "conversation.item.done"]
        assert "response.function_call_arguments.delta" in types and "response.output_text.delta" not in types
        call_done = next(e for e in events if e.type == "response.function_call_arguments.done")
        assert (
            call_done.name == "get_weather"
            and json.loads(call_done.arguments) == {"city": "Pune"}
            and call_done.call_id == "call_fake1"
        )
        item = events[-1].response.output[0]
        assert item.type == "function_call" and item.name == "get_weather" and item.status == "completed"
        sent = agent.llm.requests[-1]  # type: ignore[attr-defined]
        assert sent["tools"][0]["function"]["name"] == "get_weather" and sent["tool_choice"] == "auto"
        # the client runs the function and hands back the result
        await conn.conversation.item.create(
            item={"type": "function_call_output", "call_id": call_done.call_id, "output": "31 degrees and sunny"}
        )
        await conn.response.create()
        events = await collect_until(conn, "response.done")
        text = "".join(e.delta for e in events if e.type == "response.output_text.delta")
        assert text == "The tool said: 31 degrees and sunny."
        assert next(e for e in events if e.type == "response.output_text.done").text == text
        assert "response.output_audio.delta" not in [e.type for e in events]
        history = agent.llm.requests[-1]["messages"]  # type: ignore[attr-defined]
        assert [m["role"] for m in history] == ["system", "user", "assistant", "tool"]
        assert (
            history[2]["tool_calls"][0]["function"]["name"] == "get_weather"
            and history[3]["tool_call_id"] == "call_fake1"
        )
        # response.cancel
        agent.llm.token_delay_s = 0.2  # type: ignore[attr-defined]
        await conn.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Tell me a long story"}],
            }
        )
        await conn.response.create(response={"instructions": "Take your time.", "max_output_tokens": 500})
        await collect_until(conn, "response.output_text.delta")
        await conn.response.cancel()
        events = await collect_until(conn, "response.done")
        assert (
            events[-1].response.status == "cancelled"
            and events[-1].response.status_details.reason == "client_cancelled"
        )
        assert (
            agent.llm.requests[-1]["max_tokens"] == 500
            and agent.llm.requests[-1]["messages"][0]["content"] == "Take your time."
        )  # type: ignore[attr-defined]
        await conn.response.cancel()
        error = await conn.recv()
        assert error.type == "error" and error.error.code == "response_cancel_not_active"


async def raw(server: LiveServer, query: str = "model=gpt-realtime", **kw: Any) -> Any:
    return await websockets.connect(server.ws_url + "/v1/realtime?" + query, **kw)


async def recv_until(ws: Any, last: str, timeout: float = 20.0) -> list[dict[str, Any]]:
    out = []
    async with asyncio.timeout(timeout):
        while True:
            event = json.loads(await ws.recv())
            out.append(event)
            if event["type"] == last:
                return out


async def test_raw_websocket_g711_manual_commit_and_item_operations(agent: LiveServer) -> None:
    ws = await raw(agent)
    created = json.loads(await ws.recv())
    assert created["type"] == "session.created" and created["event_id"].startswith("event_")
    assert created["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert created["session"]["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime", "audio": {
        "input": {"format": {"type": "audio/pcmu"}, "turn_detection": None, "transcription": {"model": "fake-stt"}},
        "output": {"format": {"type": "audio/pcmu"}}}}}))  # fmt: skip
    updated = json.loads(await ws.recv())
    assert updated["session"]["audio"]["input"]["turn_detection"] is None and updated["session"]["audio"]["output"][
        "format"
    ] == {"type": "audio/pcmu"}
    ulaw = mulaw_encode(speechlike(2.0, 8000))
    for off in range(0, len(ulaw), 800):
        await ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(ulaw[off : off + 800]).decode()})
        )
    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
    events = await recv_until(ws, "conversation.item.input_audio_transcription.completed")
    assert [e["type"] for e in events][:1] == ["input_audio_buffer.committed"]
    assert "input_audio_buffer.speech_started" not in [e["type"] for e in events]  # no server VAD in manual mode
    assert events[-1]["transcript"].startswith("the quick brown fox")
    user_item = events[0]["item_id"]
    await ws.send(json.dumps({"type": "response.create"}))  # manual mode: the client asks for the answer
    events = await recv_until(ws, "response.done")
    audio = b"".join(base64.b64decode(e["delta"]) for e in events if e["type"] == "response.output_audio.delta")
    transcript = next(e for e in events if e["type"] == "response.output_audio_transcript.done")["transcript"]
    assert len(audio) / 8000 == pytest.approx(len(transcript) * 0.06, rel=0.15)  # 8 kHz mu-law out
    assert np.abs(mulaw_decode(audio)).max() > 0.05
    assistant = events[-1]["response"]["output"][0]["id"]
    # the client reports that only the first 1.5 s were played: the rest is forgotten
    await ws.send(
        json.dumps(
            {"type": "conversation.item.truncate", "item_id": assistant, "content_index": 0, "audio_end_ms": 1500}
        )
    )
    truncated = json.loads(await ws.recv())
    assert truncated == {
        "type": "conversation.item.truncated",
        "event_id": truncated["event_id"],
        "item_id": assistant,
        "content_index": 0,
        "audio_end_ms": 1500,
    }
    await ws.send(json.dumps({"type": "conversation.item.retrieve", "item_id": assistant}))
    retrieved = json.loads(await ws.recv())
    kept = retrieved["item"]["content"][0]["transcript"]
    assert retrieved["type"] == "conversation.item.retrieved"
    assert kept.startswith("You said: the quick brown fox") and kept.endswith(".") and "That is all" not in kept
    await ws.send(json.dumps({"type": "conversation.item.delete", "item_id": user_item}))
    deleted = json.loads(await ws.recv())
    assert deleted["type"] == "conversation.item.deleted" and deleted["item_id"] == user_item
    await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
    assert json.loads(await ws.recv())["type"] == "input_audio_buffer.cleared"
    # errors keep the connection open
    for bad, code in [
        ({"type": "input_audio_buffer.commit"}, "input_audio_buffer_commit_empty"),
        ({"type": "nope.event", "event_id": "client_1"}, "unknown_event"),
        ({"type": "conversation.item.delete", "item_id": "item_missing"}, "item_not_found"),
        ({"type": "input_audio_buffer.append", "audio": "@@@"}, "invalid_request"),
        (
            {"type": "session.update", "session": {"audio": {"input": {"format": {"type": "audio/flac"}}}}},
            "invalid_request",
        ),
        ({"type": "session.update", "session": {"tools": [{"type": "mcp", "server_label": "x"}]}}, "unsupported"),
    ]:
        await ws.send(json.dumps(bad))
        error = json.loads(await ws.recv())
        assert error["type"] == "error" and error["error"]["code"] == code, error
        assert set(error["error"]) == {"type", "code", "message", "param", "event_id"}
        if "event_id" in bad:
            assert error["error"]["event_id"] == "client_1"
    await ws.send("this is not json")
    assert json.loads(await ws.recv())["error"]["code"] == "invalid_json"
    await ws.close()
    await asyncio.sleep(0.2)
    assert httpx.get(agent.url + "/api/v1/stats").json()["active_streams"] == 0


async def test_conversation_without_a_language_model_fails_clearly(live: LiveServer) -> None:
    ws = await raw(live)
    await ws.recv()
    await ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            }
        )
    )
    await ws.send(json.dumps({"type": "response.create"}))
    events = await recv_until(ws, "response.done")
    error = next(e for e in events if e["type"] == "error")
    assert error["error"]["code"] == "llm_not_configured" and "llm_base_url" in error["error"]["message"]
    assert events[-1]["response"]["status"] == "failed"
    # a transcription session never needs one, and refuses responses
    await ws.close()
    ws = await raw(live, "intent=transcription")
    assert json.loads(await ws.recv())["session"]["type"] == "transcription"
    await ws.send(json.dumps({"type": "response.create"}))
    assert json.loads(await ws.recv())["error"]["code"] == "unsupported"
    await ws.close()


async def test_realtime_auth(settings: Settings) -> None:
    settings.api_key = "sekrit"
    srv = LiveServer(settings)
    srv.start()
    try:
        ws = await raw(srv)
        denied = json.loads(await ws.recv())
        assert denied["type"] == "error" and denied["error"]["code"] == "invalid_api_key"
        ws = await raw(srv, additional_headers={"Authorization": "Bearer sekrit"})  # what the SDKs send
        assert json.loads(await ws.recv())["type"] == "session.created"
        await ws.close()
        ws = await raw(srv, subprotocols=["realtime", "openai-insecure-api-key.sekrit"])  # the browser convention
        assert ws.subprotocol == "realtime" and json.loads(await ws.recv())["type"] == "session.created"
        await ws.close()
        async with openai.AsyncOpenAI(base_url=srv.url + "/v1", api_key="sekrit").realtime.connect(
            model="gpt-realtime"
        ) as conn:
            assert (await conn.recv()).type == "session.created"
    finally:
        srv.stop()
