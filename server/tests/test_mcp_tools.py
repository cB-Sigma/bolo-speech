"""MCP tools in Realtime sessions (bolo.mcp_tools): the model's calls go to an MCP server and its answer comes back."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import websockets

from bolo.config import Settings
from bolo.mcp_tools import McpServerSpec, configured_servers, parse_tool
from conftest import LiveServer
from fake_llm import FakeLLM
from fake_mcp_server import FakeMcpHttp

pytest.importorskip("mcp")

HERE = Path(__file__).parent


async def until(ws: Any, last: str, timeout: float = 20.0) -> list[dict[str, Any]]:
    events = []
    async with asyncio.timeout(timeout):
        while True:
            event = json.loads(await ws.recv())
            events.append(event)
            if event["type"] == last:
                return events


async def ask(ws: Any, text: str) -> None:
    await ws.send(json.dumps({"type": "conversation.item.create", "item": {
        "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}}))  # fmt: skip
    await ws.send(json.dumps({"type": "response.create"}))


def test_parse_tools() -> None:
    s = Settings(mcp_servers={"files": {"command": "x", "require_approval": "always", "allowed_tools": ["a", "b"]}})
    spec = parse_tool({"type": "mcp", "server_label": "files", "allowed_tools": ["b", "c"]}, s)
    assert spec.command == "x" and spec.allowed_tools == ["b"] and spec.require_approval == "always"
    assert "command" not in spec.public() and "server_url" not in spec.public()
    url = parse_tool(
        {"type": "mcp", "server_label": "x", "server_url": "https://h/mcp", "authorization": "tok",
         "require_approval": {"never": {"tool_names": ["a"]}}},
        s,
    )  # fmt: skip
    assert url.headers == {"Authorization": "Bearer tok"} and url.public()["headers"] == {"Authorization": "***"}
    assert not url.needs_approval("a") and url.needs_approval("b")
    with pytest.raises(ValueError, match="no MCP server 'nope'"):
        parse_tool({"type": "mcp", "server_label": "nope"}, s)
    with pytest.raises(ValueError, match="http"):
        parse_tool({"type": "mcp", "server_label": "x", "server_url": "file:///etc"}, s)
    by_url = {"type": "mcp", "server_label": "x", "server_url": "https://h/mcp"}
    s.host = "0.0.0.0"  # an open, non-loopback bind: no URLs from sessions unless requests need a key
    with pytest.raises(ValueError, match="realtime_mcp_urls"):
        parse_tool(by_url, s)
    assert parse_tool(by_url, s, authenticated=True).url == "https://h/mcp"
    s.realtime_mcp_urls = True
    assert parse_tool(by_url, s).url == "https://h/mcp"
    s.realtime_mcp_urls = False
    with pytest.raises(ValueError, match="realtime_mcp_urls"):
        parse_tool(by_url, s, authenticated=True)
    assert isinstance(configured_servers(s)["files"], McpServerSpec)
    assert Settings(mcp_servers={"f": {"url": "u", "headers": {"k": "v"}}}).public_dict()["mcp_servers"] == {
        "f": {"url": "u", "headers": {"k": "***"}}
    }


@pytest.fixture
def mcp_http() -> Any:
    server = FakeMcpHttp().start()
    try:
        yield server
    finally:
        server.stop()


async def test_mcp_tool_over_http_flows_like_a_function(agent: LiveServer, mcp_http: FakeMcpHttp) -> None:
    async with websockets.connect(agent.ws_url + "/v1/realtime") as ws:
        await until(ws, "session.created")
        tool = {"type": "mcp", "server_label": "weather", "server_url": mcp_http.url, "require_approval": "never",
                "headers": {"X-Token": "test-header"}, "allowed_tools": ["get_weather", "add"]}  # fmt: skip
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "output_modalities": ["text"], "tools": [tool]}}))  # fmt: skip
        events = await until(ws, "mcp_list_tools.completed")
        updated = next(e for e in events if e["type"] == "session.updated")
        assert updated["session"]["tools"][0]["headers"] == {"X-Token": "***"}
        listed = next(e for e in events if e["type"] == "conversation.item.added")["item"]
        assert listed["type"] == "mcp_list_tools" and listed["server_label"] == "weather"
        done = await until(ws, "conversation.item.done")
        assert [t["name"] for t in done[-1]["item"]["tools"]] == ["get_weather", "add"]  # "fail" is not allowed
        assert mcp_http.headers[0]["x-token"] == "test-header"

        await ask(ws, "What is the weather in Pune?")
        first = await until(ws, "response.done")
        types = [e["type"] for e in first]
        assert "response.mcp_call_arguments.done" in types and "response.mcp_call.completed" in types
        call = first[-1]["response"]["output"][0]
        assert call["type"] == "mcp_call" and call["name"] == "get_weather" and call["server_label"] == "weather"
        assert call["output"] == "31 degrees and sunny in Pune" and call["error"] is None
        assert mcp_http.calls == [{"name": "get_weather", "arguments": {"city": "Pune"}}]
        # the model hears the result and answers in a second response, without the client doing anything
        second = await until(ws, "response.done")
        text = "".join(e["delta"] for e in second if e["type"] == "response.output_text.delta")
        assert text == "The tool said: 31 degrees and sunny in Pune."
        offered = agent.llm.requests[0]["tools"]  # type: ignore[attr-defined]
        assert [t["function"]["name"] for t in offered] == ["get_weather", "add"]
        history = agent.llm.requests[-1]["messages"]  # type: ignore[attr-defined]
        assert [m["role"] for m in history] == ["system", "user", "assistant", "tool"]
        assert history[2]["tool_calls"][0]["id"] == history[3]["tool_call_id"] == "call_fake1"


async def test_configured_stdio_server_with_approval(settings: Settings, llm: FakeLLM) -> None:
    settings.llm_base_url, settings.llm_model = llm.url, "fake-llm"
    settings.mcp_servers = {
        "local": {"command": sys.executable, "args": [str(HERE / "fake_mcp_server.py")], "require_approval": "always"}
    }
    srv = LiveServer(settings)
    srv.start()
    try:
        async with websockets.connect(srv.ws_url + "/v1/realtime") as ws:
            await until(ws, "session.created")
            await ws.send(json.dumps({"type": "session.update", "session": {
                "type": "realtime", "output_modalities": ["text"],
                "tools": [{"type": "mcp", "server_label": "local"}]}}))  # fmt: skip
            await until(ws, "mcp_list_tools.completed")
            await ask(ws, "Weather in Pune please")
            events = await until(ws, "response.done")
            request = events[-1]["response"]["output"][0]
            assert request["type"] == "mcp_approval_request" and request["name"] == "get_weather"
            await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                "type": "mcp_approval_response", "approval_request_id": request["id"], "approve": True}}))  # fmt: skip
            events = await until(ws, "response.done")
            call = events[-1]["response"]["output"][0]
            assert call["type"] == "mcp_call" and call["approval_request_id"] == request["id"]
            assert call["output"] == "31 degrees and sunny in Pune"
            events = await until(ws, "response.done")
            text = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
            assert text == "The tool said: 31 degrees and sunny in Pune."
            # a declined call: the model is told so
            await ask(ws, "And the weather in Delhi?")
            events = await until(ws, "response.done")
            request = events[-1]["response"]["output"][0]
            await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                "type": "mcp_approval_response", "approval_request_id": request["id"], "approve": False}}))  # fmt: skip
            events = await until(ws, "response.done")
            text = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
            assert text == "The tool said: The user did not allow this tool call.."
            # an unknown server is refused when the session is updated
            await ws.send(json.dumps({"type": "session.update", "session": {
                "type": "realtime", "tools": [{"type": "mcp", "server_label": "elsewhere"}]}}))  # fmt: skip
            error = (await until(ws, "error"))[-1]
            assert "no MCP server 'elsewhere'" in error["error"]["message"]
    finally:
        srv.stop()


async def test_unreachable_server_reports_failure(agent: LiveServer) -> None:
    async with websockets.connect(agent.ws_url + "/v1/realtime") as ws:
        await until(ws, "session.created")
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime", "tools": [
            {"type": "mcp", "server_label": "gone", "server_url": "http://127.0.0.1:9/mcp"}]}}))  # fmt: skip
        events = await until(ws, "mcp_list_tools.failed")
        assert any(e["type"] == "mcp_list_tools.in_progress" for e in events)
        error = (await until(ws, "error"))[-1]
        assert error["error"]["code"] == "mcp_list_tools_failed" and "gone" in error["error"]["message"]
