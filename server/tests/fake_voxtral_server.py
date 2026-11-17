"""A stand-in for llama-server running a Voxtral GGUF with its audio encoder: what the voxtral-llamacpp kind uses.

Run in a thread (``FakeVoxtralServer(...).start()``) or as a process with llama-server's command line::

    fake_voxtral_server.py -m model.gguf -mm mmproj.gguf --host 127.0.0.1 --port N --api-key K -t 2 -c 4096

``GET /props`` names a random-looking media marker and the audio modality (none without ``-mm``). ``POST
/completion`` takes ``prompt: {prompt_string, multimodal_data}``, decodes the WAV and answers ``content``:
``"heard <seconds> s"`` plus `` lang <code>`` for a transcription prompt with a language, or ``"translated <seconds>
s"`` for the translation instruction. A prompt without the marker, or a marker count that differs from the number of
files, is refused with 400 as llama-server does. Audio whose first sample is exactly 0.5 makes the process exit.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MARKER = "<__media_fake0123456789__>"


class FakeVoxtralServer:
    def __init__(
        self, api_key: str | None = None, port: int = 0, audio: bool = True, exit_on_crash: bool = False
    ) -> None:
        self.api_key = api_key
        self.audio = audio
        self.exit_on_crash = exit_on_crash
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

            def _authorised(self) -> bool:
                if outer.api_key and self.headers.get("Authorization") != f"Bearer {outer.api_key}":
                    self._json(401, {"error": {"code": 401, "message": "Invalid API Key"}})
                    return False
                return True

            def do_GET(self) -> None:
                if self.path.startswith("/health"):
                    self._json(200, {"status": "ok"})
                elif self.path.startswith("/props"):
                    if not self._authorised():
                        return
                    self._json(200, {"media_marker": MARKER, "modalities": {"vision": False, "audio": outer.audio}})
                else:
                    self._json(404, {"error": {"code": 404, "message": "File Not Found"}})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append({"path": self.path, "body": body})
                if not self._authorised():
                    return
                if self.path != "/completion":
                    self._json(404, {"error": {"code": 404, "message": "File Not Found"}})
                    return
                prompt = body.get("prompt") or {}
                text = str(prompt.get("prompt_string", ""))
                files = list(prompt.get("multimodal_data") or [])
                if text.count(MARKER) != len(files) or not files:
                    self._json(400, {"error": {"code": 400, "message": "number of media markers does not match"}})
                    return
                with wave.open(io.BytesIO(base64.b64decode(files[0]))) as w:
                    frames = w.readframes(w.getnframes())
                    seconds = w.getnframes() / w.getframerate()
                    first = int.from_bytes(frames[:2], "little", signed=True) if frames else 0
                if first == 16383 and outer.exit_on_crash:
                    os._exit(3)
                if "[TRANSCRIBE]" in text:
                    content = f"heard {seconds:.2f} s"
                    m = re.search(r"lang:([a-z]+)\[TRANSCRIBE\]", text)
                    if m:
                        content += f" lang {m.group(1)}"
                elif "Translate" in text:
                    content = f"translated {seconds:.2f} s"
                else:
                    content = "?"
                self._json(200, {"content": " " + content, "stop": True, "stop_type": "eos"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> FakeVoxtralServer:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
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
    server = FakeVoxtralServer(
        api_key=args.get("--api-key") or None, port=int(args["--port"]), audio="-mm" in args, exit_on_crash=True
    )
    server.httpd.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
