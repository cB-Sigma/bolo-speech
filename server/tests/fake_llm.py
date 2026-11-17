"""A deterministic OpenAI-compatible chat-completions server for tests.

Replies are a pure function of the request:

* last message from a tool           -> "The tool said: <content>."
* tools offered and "weather" asked  -> a ``get_weather`` tool call
* otherwise                          -> "You said: <last user text>. That is all I heard. Goodbye for now."

Streaming sends one word per chunk with ``token_delay_s`` between chunks.

Run on its own (``python fake_llm.py --port 8090``) it serves until stopped: the browser WebRTC test
(ui/e2e-webrtc) points a real ``bolo serve`` at it, so no language model has to be downloaded.
"""

from __future__ import annotations

import http.server
import json
import threading
import time
from typing import Any


class FakeLLM:
    def __init__(self, token_delay_s: float = 0.01, first_token_delay_s: float = 0.05, port: int = 0) -> None:
        self.token_delay_s = token_delay_s
        self.first_token_delay_s = first_token_delay_s
        self.requests: list[dict[str, Any]] = []
        self.fail_with: int | None = None
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                pass

            def _json(self, status: int, doc: dict[str, Any]) -> None:
                body = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path.endswith("/models"):
                    self._json(200, {"object": "list", "data": [{"id": "fake-llm", "object": "model"}]})
                else:
                    self._json(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                body["_authorization"] = self.headers.get("Authorization")
                owner.requests.append(body)
                if owner.fail_with:
                    self._json(
                        owner.fail_with, {"error": {"message": "the fake model is unwell", "type": "server_error"}}
                    )
                    return
                text, tool = owner.reply(body)
                base = {"id": "chatcmpl-fake", "created": 1, "model": body.get("model") or "fake-llm"}
                usage = {
                    "prompt_tokens": 7,
                    "completion_tokens": len(text.split()),
                    "total_tokens": 7 + len(text.split()),
                }
                if not body.get("stream"):
                    message: dict[str, Any] = {"role": "assistant", "content": text or None}
                    if tool:
                        message["tool_calls"] = [tool]
                    choice = {"index": 0, "message": message, "finish_reason": "tool_calls" if tool else "stop"}
                    self._json(200, {**base, "object": "chat.completion", "choices": [choice], "usage": usage})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()

                def emit(delta: dict[str, Any], finish: str | None = None) -> None:
                    chunk = {
                        **base,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()

                try:
                    time.sleep(owner.first_token_delay_s)
                    emit({"role": "assistant", "content": ""})
                    if tool:
                        fn = tool["function"]
                        emit(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": tool["id"],
                                        "type": "function",
                                        "function": {"name": fn["name"], "arguments": ""},
                                    }
                                ]
                            }
                        )
                        half = len(fn["arguments"]) // 2
                        for piece in (fn["arguments"][:half], fn["arguments"][half:]):
                            emit({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]})
                    words = text.split(" ") if text else []
                    for i, word in enumerate(words):
                        emit({"content": word + ("" if i == len(words) - 1 else " ")})
                        time.sleep(owner.token_delay_s)
                    emit({}, "tool_calls" if tool else "stop")
                    if (body.get("stream_options") or {}).get("include_usage"):
                        self.wfile.write(
                            f"data: {json.dumps({**base, 'object': 'chat.completion.chunk', 'choices': [], 'usage': usage})}\n\n".encode()
                        )
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except ConnectionError:  # BrokenPipe, ConnectionReset, ConnectionAborted (Windows)
                    pass  # the client cancelled the response

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @staticmethod
    def reply(body: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        messages = body.get("messages") or []
        last = messages[-1] if messages else {}
        if last.get("role") == "tool":
            return f"The tool said: {last.get('content')}.", None
        user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        content = user.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        content = str(content or "")
        if body.get("tools") and "weather" in content.lower():
            return "", {
                "id": "call_fake1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Pune"}'},
            }
        return f"You said: {content}. That is all I heard. Goodbye for now.", None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def start(self) -> FakeLLM:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="a deterministic OpenAI-compatible chat server for tests")
    parser.add_argument("--port", type=int, default=8090)
    server = FakeLLM(port=parser.parse_args().port)
    print(f"fake language model at {server.url}", flush=True)
    server.server.serve_forever()
