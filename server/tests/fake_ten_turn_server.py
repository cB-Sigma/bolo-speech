"""A stand-in for llama-server running TEN Turn Detection: ``POST /v1/chat/completions`` answers one token.

The "model" says ``unfinished`` when the user's text ends in a word that cannot end a sentence ("to", "the",
"a", "for", "and", "my"), ``wait`` when it asks the agent to stop talking, else ``finished``; with ``logprobs`` it
gives the three labels as the first token's top log-probabilities (0.9 for its answer, 0.05 for the others, split
"un" + "finished" the way a tokenizer may).
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

OPEN_ENDINGS = ("to", "the", "a", "for", "and", "my")


def answer(text: str) -> str:
    words = text.lower().rstrip(" .?!,").split()
    if "stop talking" in text.lower():
        return "wait"
    return "unfinished" if words and words[-1] in OPEN_ENDINGS else "finished"


class FakeTenTurnServer:
    def __init__(self, logprobs: bool = True) -> None:
        self.logprobs = logprobs
        self.requests: list[dict[str, Any]] = []
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

            def do_GET(self) -> None:
                self._json(200 if self.path.startswith("/health") else 404, {"status": "ok"})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                outer.requests.append(body)
                if self.path != "/v1/chat/completions":
                    self._json(404, {"error": {"code": 404, "message": "File Not Found"}})
                    return
                label = answer(str(body["messages"][-1]["content"]))
                token = "un" if label == "unfinished" else label
                choice: dict[str, Any] = {"index": 0, "message": {"role": "assistant", "content": token}}
                if outer.logprobs and body.get("logprobs"):
                    probs = {"finished": 0.05, "un": 0.05, "wait": 0.05}
                    probs[token] = 0.9
                    top = [{"token": t, "logprob": math.log(p)} for t, p in probs.items()]
                    choice["logprobs"] = {"content": [{"token": token, "logprob": math.log(0.9), "top_logprobs": top}]}
                self._json(200, {"choices": [choice]})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> FakeTenTurnServer:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
