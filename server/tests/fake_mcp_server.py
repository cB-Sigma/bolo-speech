"""A small MCP server for the tests, written against the protocol rather than an SDK (so it serves clients of
both mcp 1.x and 2.x): JSON-RPC over stdio (``python fake_mcp_server.py``) or over Streamable HTTP with plain
JSON answers (:class:`FakeMcpHttp`). Tools: ``get_weather`` (the fake language model calls it when asked about
the weather), ``add`` and ``fail``."""

from __future__ import annotations

import http.server
import json
import sys
import threading
from typing import Any

TOOLS = [
    {
        "name": "get_weather",
        "description": "Weather for a city",
        "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
    {
        "name": "add",
        "description": "Add two numbers",
        "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
    },
    {"name": "fail", "description": "Always fails", "inputSchema": {"type": "object", "properties": {}}},
]


def handle(message: dict[str, Any], calls: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """The answer to one JSON-RPC message (None for a notification)."""
    if "id" not in message:
        return None
    method = message.get("method")
    params = message.get("params") or {}
    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": params.get("protocolVersion") or "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-mcp", "version": "1.0"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "ping":
        result = {}
    elif method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if calls is not None:
            calls.append({"name": name, "arguments": args})
        if name == "get_weather":
            result = {"content": [{"type": "text", "text": f"31 degrees and sunny in {args.get('city')}"}]}
        elif name == "add":
            result = {"content": [{"type": "text", "text": str(args.get("a", 0) + args.get("b", 0))}]}
        else:
            result = {"content": [{"type": "text", "text": "this tool always fails"}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": f"no method {method}"}}
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


class FakeMcpHttp:
    """Streamable HTTP: each POST carries one JSON-RPC message and gets a JSON answer (202 for notifications)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                self.send_response(405)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_DELETE(self) -> None:
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self) -> None:
                owner.headers.append({k.lower(): v for k, v in self.headers.items()})
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                answer = handle(body, owner.calls)
                if answer is None:
                    self.send_response(202)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Mcp-Session-Id", "fake-session")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/mcp"

    def start(self) -> FakeMcpHttp:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        answer = handle(json.loads(line))
        if answer is not None:
            sys.stdout.write(json.dumps(answer) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
