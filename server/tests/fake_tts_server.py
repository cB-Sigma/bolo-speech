"""A stand-in for llama-server, audiocpp_server and the Tontaube server in tests: the HTTP surface the tts-server
engine uses.

Run in a thread (``FakeTTSServer(...).start()``) or as a process with the command line of the real binaries::

    fake_tts_server.py -m model.gguf -mm mmproj.gguf --host 127.0.0.1 --port N --api-key K -t 2   (llama-server)
    fake_tts_server.py --config server.json --no-ui                                              (audiocpp_server)
    fake_tts_server.py run --python 3.12 --frozen --directory D tontaube serve --host H --port N      (as uv: Tontaube)

Audio is a tone whose length follows the text (0.05 s per character) at ``rate``. The text "crash now" makes the
process exit (to test restarts); "bad request" answers 400 the way the real servers do; "cuda error now" prints
llama.cpp's CUDA error lines on the console (to test bolo.gpu_health) and answers as usual. With ``FAKE_ENV_FILE``
set, the process writes the environment it was started with to that file.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def tone(text: str, rate: int) -> np.ndarray:
    n = int(rate * 0.05 * max(1, len(text)))
    t = np.arange(n) / rate
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


class FakeTTSServer:
    def __init__(
        self,
        runtime: str = "llama",
        rate: int = 16000,
        voices: list[str] | None = None,
        api_key: str | None = None,
        port: int = 0,
        load_delay_s: float = 0.0,
        exit_on_crash: bool = False,
        model_id: str = "test-model",
    ) -> None:
        self.runtime = runtime
        self.rate = rate
        self.voices = voices or []
        self.api_key = api_key
        self.started = time.monotonic()
        self.load_delay_s = load_delay_s
        self.exit_on_crash = exit_on_crash
        self.model_id = model_id
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

            def _error(self, code: int, message: str) -> None:
                self._json(code, {"error": {"code": code, "message": message, "type": "invalid_request_error"}})

            def do_GET(self) -> None:
                if self.path.startswith("/readyz"):
                    if time.monotonic() - outer.started < outer.load_delay_s:
                        self._json(503, {"status": "loading"})
                    else:
                        self._json(200, {"status": "ready"})
                elif self.path.startswith("/controls"):
                    self._json(200, {"controls": [{"name": "rate", "bins": 5}]})
                elif self.path.startswith("/health"):
                    if time.monotonic() - outer.started < outer.load_delay_s:
                        self._error(503, "Loading model")
                    else:
                        self._json(200, {"status": "ok"})
                elif self.path.startswith("/props"):
                    self._json(200, {"voices": outer.voices, "modalities": {"audio": True}})
                elif self.path.startswith("/v1/audio/voices"):
                    self._json(200, {"voices": outer.voices})
                else:
                    self._error(404, "not found")

            def do_POST(self) -> None:
                if outer.runtime == "tontaube":
                    self._tontaube()
                    return
                if outer.api_key and self.headers.get("Authorization") != f"Bearer {outer.api_key}":
                    self._json(
                        401, {"error": {"code": 401, "message": "Invalid API Key", "type": "authentication_error"}}
                    )
                    return
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                outer.requests.append({"path": self.path, "body": body})
                text = str(body.get("input", ""))
                if text == "crash now":
                    if outer.exit_on_crash:
                        os._exit(3)
                    self.close_connection = True
                    self.connection.close()
                    return
                if text == "cuda error now":
                    print("CUDA error: unspecified launch failure", flush=True)
                    print("  current device: 0, in function ggml_cuda_compute_forward at ggml-cuda.cu:2519", flush=True)
                    text = "ok"
                if text == "bad request":
                    self._error(400, "voice 'nobody' is not a voice of this model")
                    return
                if outer.runtime == "llama" and self.path == "/tts":
                    self._llama(body, text)
                elif outer.runtime == "audiocpp" and self.path == "/v1/audio/speech":
                    if body.get("model") != outer.model_id:
                        self._error(404, f"unknown model '{body.get('model')}'")
                        return
                    self._audiocpp(body, text)
                else:
                    self._error(404, "not found")

            def _chunked(self, content_type: str, data: bytes, pieces: int = 4) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                step = max(1, len(data) // pieces)
                for i in range(0, len(data), step):
                    part = data[i : i + step]
                    self.wfile.write(b"%x\r\n" % len(part) + part + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")

            def _wav(self, x: np.ndarray) -> None:
                buf = io.BytesIO()
                sf.write(buf, x, outer.rate, format="WAV", subtype="PCM_16")
                data = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _llama(self, body: dict[str, Any], text: str) -> None:
                if body.get("voice") and outer.voices and body["voice"] not in outer.voices:
                    self._error(400, f"unknown voice '{body['voice']}'")
                    return
                if body.get("speaker_ref_b64"):
                    base64.b64decode(body["speaker_ref_b64"], validate=True)
                x = tone(text, outer.rate)
                fmt = body.get("response_format", "wav")
                if fmt == "pcm16":
                    ctype = f"audio/pcm;rate={outer.rate};encoding=signed-int;bits=16"
                    self._chunked(ctype, pcm16(x))
                elif fmt == "pcm":
                    ctype = f"audio/pcm;rate={outer.rate};encoding=float;bits=32"
                    self._chunked(ctype, x.astype("<f4").tobytes())
                else:
                    self._wav(x)

            def _tontaube(self) -> None:
                if outer.api_key and self.headers.get("x-api-key") != outer.api_key:
                    self._json(401, {"error": "unauthorized"})
                    return
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                outer.requests.append({"path": self.path, "body": body})
                if self.path != "/predict":
                    self._error(404, "not found")
                    return
                if body.get("tag") not in ("audiobook", "conversational", "agentic"):
                    self._json(422, {"detail": [{"msg": "unsupported tag"}]})
                    return
                if body.get("voice_audio_b64"):
                    base64.b64decode(body["voice_audio_b64"], validate=True)
                buf = io.BytesIO()
                sf.write(buf, tone(str(body.get("text", "")), 24000), 24000, format="WAV", subtype="PCM_16")
                self._json(200, {"audio_b64": base64.b64encode(buf.getvalue()).decode(), "mime_type": "audio/wav",
                                 "sample_rate": 24000})  # fmt: skip

            def _audiocpp(self, body: dict[str, Any], text: str) -> None:
                ref = body.get("voice_ref")
                if ref is not None and not (isinstance(ref, dict) and ref.get("type") == "base64"):
                    self._error(400, "voice_ref must be an object")
                    return
                x = tone(text, outer.rate)
                if body.get("stream_format") == "audio":
                    self._chunked("application/octet-stream", pcm16(x))
                else:
                    self._wav(x)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = int(self.httpd.server_address[1])
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> FakeTTSServer:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def main(argv: list[str]) -> None:
    args = dict(zip(argv[::2], argv[1::2])) if "--config" not in argv else {}
    if argv[:1] == ["run"]:  # as uv, starting the Tontaube server
        if not Path(os.environ.get("MODEL_PATH", ""), "cb0").is_dir():
            print("MODEL_PATH has no cb0/", flush=True)
            sys.exit(2)
        server = FakeTTSServer(
            "tontaube", rate=24000, port=int(argv[argv.index("--port") + 1]), api_key=os.environ.get("TTS_API_KEY"),
            load_delay_s=float(os.environ.get("FAKE_LOAD_DELAY", "0")),
        )  # fmt: skip
    elif "--config" in argv:
        config = json.loads(Path(argv[argv.index("--config") + 1]).read_text(encoding="utf-8"))
        model = config["models"][0]
        server = FakeTTSServer(
            "audiocpp", rate=int(os.environ.get("FAKE_RATE", "16000")), port=int(config["port"]),
            exit_on_crash=True, model_id=model["id"],
        )  # fmt: skip
    else:
        server = FakeTTSServer(
            "llama", rate=int(os.environ.get("FAKE_RATE", "16000")), api_key=args.get("--api-key"),
            port=int(args["--port"]), load_delay_s=float(os.environ.get("FAKE_LOAD_DELAY", "0")), exit_on_crash=True,
            voices=[v for v in os.environ.get("FAKE_VOICES", "").split(",") if v],
        )  # fmt: skip
    if os.environ.get("FAKE_ENV_FILE"):  # what the server was started with, for the test to read
        Path(os.environ["FAKE_ENV_FILE"]).write_text(json.dumps(dict(os.environ)), encoding="utf-8")
    print("ggml_cuda_init: found 1 CUDA devices:", flush=True)  # every start prints this: not an error
    print(f"fake server listening on {server.port} ({' '.join(argv)})", flush=True)
    server.httpd.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
