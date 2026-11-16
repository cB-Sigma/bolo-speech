"""A stand-in for llama-server running an Orpheus-style GGUF: the HTTP surface bolo's orpheus-llamacpp kind uses.

Run in a thread (``FakeOrpheusServer(...).start()``) or as a process with llama-server's command line::

    fake_orpheus_server.py -m model.gguf --host 127.0.0.1 --port N --api-key K -t 2 -c 4096 -np 1

``POST /tokenize`` maps each character to one id (``1000 + ord(c) % 20000``). ``POST /completion`` (streamed as
server-sent events, ``tokens`` per event) answers a prompt with ``frames`` SNAC frames, where ``frames`` is half
the number of characters of the text after the speaker's colon (stripped), at most ``n_predict // 7``; the
codes of frame ``j`` are ``(j + 1) % 4096`` on level 0 and ``(7 j + i) % 4096`` elsewhere. Then it sends
END_OF_SPEECH and, unless ``stop_after_eos`` is set, keeps sending audio tokens that a correct client never reads.
The text "crash now" makes the process exit; "bad request" answers 400 as llama-server does. ``GET
/files/<name>`` serves the files given as ``files`` (a decoder download).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

AUDIO_OFFSET = 128266
END_OF_SPEECH = 128258


def frame_tokens(j: int) -> list[int]:
    codes = [(j + 1) % 4096] + [(7 * j + i) % 4096 for i in range(1, 7)]
    return [AUDIO_OFFSET + i * 4096 + c for i, c in enumerate(codes)]


class FakeOrpheusServer:
    def __init__(
        self,
        api_key: str | None = None,
        port: int = 0,
        files: dict[str, bytes] | None = None,
        stop_after_eos: bool = False,
        exit_on_crash: bool = False,
        tokens_per_event: int = 3,
        delay_s: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.files = files or {}
        self.stop_after_eos = stop_after_eos
        self.exit_on_crash = exit_on_crash
        self.tokens_per_event = tokens_per_event
        self.delay_s = delay_s
        self.requests: list[dict[str, Any]] = []
        self.sent_after_eos = 0
        self.disconnects = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _json(self, code: int, doc: Any) -> None:
                data = json.dumps(doc).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _authorised(self) -> bool:
                if outer.api_key and self.headers.get("Authorization") != f"Bearer {outer.api_key}":
                    self._json(
                        401, {"error": {"code": 401, "message": "Invalid API Key", "type": "authentication_error"}}
                    )
                    return False
                return True

            def do_GET(self) -> None:
                if self.path.startswith("/health"):
                    self._json(200, {"status": "ok"})
                elif self.path.startswith("/files/") and self.path[7:] in outer.files:
                    data = outer.files[self.path[7:]]
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._json(404, {"error": {"code": 404, "message": "File Not Found"}})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append({"path": self.path, "body": body})
                if not self._authorised():
                    return
                if self.path == "/tokenize":
                    text = str(body.get("content", ""))
                    if "crash now" in text and outer.exit_on_crash:
                        os._exit(3)
                    self._json(200, {"tokens": [1000 + ord(c) % 20000 for c in text]})
                elif self.path == "/completion":
                    self._completion(body)
                else:
                    self._json(404, {"error": {"code": 404, "message": "File Not Found"}})

            def _completion(self, body: dict[str, Any]) -> None:
                prompt = [int(t) for t in body.get("prompt") or []]
                text = "".join(chr(t - 1000) for t in prompt if 1000 <= t < 21000)
                if "bad request" in text:
                    self._json(400, {"error": {"code": 400, "message": "bad prompt", "type": "invalid_request_error"}})
                    return
                spoken = text.split(":", 1)[1].strip() if ":" in text else text
                frames = max(1, len(spoken) // 2)
                budget = int(body.get("n_predict", -1))
                if budget >= 0:
                    frames = min(frames, budget // 7)
                tokens = [t for j in range(frames) for t in frame_tokens(j)] + [END_OF_SPEECH]
                eos_at = len(tokens)
                if not outer.stop_after_eos:
                    tokens += frame_tokens(frames) * 40
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                step = outer.tokens_per_event
                try:
                    for i in range(0, len(tokens), step):
                        piece = tokens[i : i + step]
                        if i >= eos_at:
                            outer.sent_after_eos += len(piece)
                        self._chunk(f"data: {json.dumps({'content': '', 'tokens': piece, 'stop': False})}\n\n")
                        if outer.delay_s:
                            time.sleep(outer.delay_s)
                    self._chunk(f"data: {json.dumps({'content': '', 'tokens': [], 'stop': True})}\n\n")
                    self.wfile.write(b"0\r\n\r\n")
                except ConnectionError:  # BrokenPipe, ConnectionReset, ConnectionAborted (Windows)
                    outer.disconnects += 1

            def _chunk(self, text: str) -> None:
                data = text.encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.httpd.server_address[1]
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> FakeOrpheusServer:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def main(argv: list[str]) -> None:
    args: dict[str, str] = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("-") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            args[argv[i]] = argv[i + 1]
            i += 2
        else:
            args[argv[i]] = ""
            i += 1
    log = os.environ.get("FAKE_ARGV_LOG")
    if log:
        Path(log).write_text(json.dumps(argv), encoding="utf-8")
    server = FakeOrpheusServer(api_key=args.get("--api-key") or None, port=int(args["--port"]), exit_on_crash=True)
    server.httpd.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
