"""Stand-ins for the cloud speech APIs the ``remote`` engine talks to (CI has no network and no keys).

One HTTP server answers the routes of all four API families, each with its own authentication header:

=================  ==========================================================  ===========================
family             routes                                                      key header
=================  ==========================================================  ===========================
openai-compatible  ``GET /v1/models``, ``POST /v1/audio/transcriptions``,      ``Authorization: Bearer``
                   ``/v1/audio/translations``, ``/v1/audio/speech``
sarvam             ``POST /speech-to-text``, ``/speech-to-text-translate``,    ``api-subscription-key``
                   ``/text-to-speech``
elevenlabs         ``GET /v2/voices``, ``POST /v1/speech-to-text``,            ``xi-api-key``
                   ``/v1/text-to-speech/{voice}/stream``
deepgram           ``GET /v1/projects``, ``POST /v1/listen``                   ``Authorization: Token``
=================  ==========================================================  ===========================

Recognition answers ``"heard <seconds> seconds"`` (plus the language it was told); synthesis answers a tone of
0.05 s per character. ``fail`` holds HTTP statuses to answer the next requests with, ``delay_s`` makes every
answer slow (to test timeouts), and the text ``bad request`` is refused with 400. Every request is recorded in
``requests`` (method, path, query, headers, form fields or JSON body).

:class:`FakeLiveServer` is the WebSocket side: Deepgram's ``/v1/listen`` and Sarvam's ``/speech-to-text/ws``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
from email.parser import BytesParser
from email.policy import HTTP
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import numpy as np
import soundfile as sf
from websockets.asyncio.server import ServerConnection, serve
from websockets.http11 import Request, Response

#: not a real key: the stand-ins accept exactly this string
KEY = "test-key-not-real-7c1e9a2b44d0"
VOICES = [
    {"voice_id": "v_rachel_01", "name": "Rachel", "labels": {"gender": "female", "language": "en"}},
    {"voice_id": "v_arjun_02", "name": "Arjun", "labels": {"gender": "male", "language": "hi"}},
]


def tone(text: str, rate: int) -> np.ndarray:
    n = int(rate * 0.05 * max(1, len(text)))
    t = np.arange(n) / rate
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def wav(x: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def seconds_of(audio: bytes) -> float:
    try:
        with sf.SoundFile(io.BytesIO(audio)) as f:
            return float(len(f) / f.samplerate)
    except Exception:
        return max(0.0, (len(audio) - 44) / 32000.0)


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Form fields (a repeated name becomes a list) and files of a multipart body."""
    message = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    fields: dict[str, Any] = {}
    files: dict[str, bytes] = {}
    for part in message.iter_parts():
        name = str(part.get_param("name", header="content-disposition"))
        payload = part.get_payload(decode=True)
        data = payload if isinstance(payload, bytes) else b""
        if part.get_filename():
            files[name] = data
            fields[name + ".filename"] = part.get_filename()
            fields[name + ".content_type"] = part.get_content_type()
            continue
        value = data.decode()
        if name in fields:
            fields[name] = [*fields[name], value] if isinstance(fields[name], list) else [fields[name], value]
        else:
            fields[name] = value
    return fields, files


class FakeProviders:
    def __init__(self, key: str = KEY, port: int = 0) -> None:
        self.key = key
        self.requests: list[dict[str, Any]] = []
        #: statuses to answer the next requests with (popped from the front)
        self.fail: list[int] = []
        #: answer every request with this status
        self.fail_always: int | None = None
        self.delay_s = 0.0
        self.retry_after: str | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, code: int, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, doc: Any, headers: dict[str, str] | None = None) -> None:
                self._send(code, json.dumps(doc).encode(), "application/json", headers)

            def _chunked(self, content_type: str, pieces: list[bytes], pause_s: float = 0.0) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for piece in pieces:
                    self.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
                    self.wfile.flush()
                    if pause_s:
                        time.sleep(pause_s)
                self.wfile.write(b"0\r\n\r\n")

            # -- one error body per family, as the real services shape them ---------------------------
            def _error(self, family: str, code: int, message: str) -> None:
                headers = {"Retry-After": outer.retry_after} if outer.retry_after and code == 429 else None
                if family == "elevenlabs":
                    self._json(code, {"detail": {"status": "error", "message": message}}, headers)
                elif family == "deepgram":
                    self._json(code, {"err_code": "ERROR", "err_msg": message, "request_id": "r"}, headers)
                elif family == "sarvam":
                    self._json(code, {"error": {"message": message, "code": "invalid_request_error"}}, headers)
                else:
                    self._json(code, {"error": {"message": message, "type": "invalid_request_error"}}, headers)

            def _family(self, path: str) -> str:
                if path.startswith(("/v1/audio", "/v1/models")):
                    return "openai"
                if path.startswith(("/speech-to-text", "/text-to-speech")):
                    return "sarvam"
                if path.startswith(("/v1/text-to-speech", "/v1/speech-to-text", "/v2/voices")):
                    return "elevenlabs"
                return "deepgram"

            def _authorised(self, family: str) -> bool:
                h = self.headers
                return {
                    "openai": h.get("Authorization") == f"Bearer {outer.key}",
                    "sarvam": h.get("api-subscription-key") == outer.key,
                    "elevenlabs": h.get("xi-api-key") == outer.key,
                    "deepgram": h.get("Authorization") == f"Token {outer.key}",
                }[family]

            def _handle(self, method: str) -> None:
                url = urlsplit(self.path)
                path = url.path
                query = parse_qsl(url.query, keep_blank_values=True)
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                content_type = self.headers.get("Content-Type", "")
                record: dict[str, Any] = {
                    "method": method, "path": path, "query": query, "headers": dict(self.headers.items()),
                    "bytes": len(body),
                }  # fmt: skip
                fields: dict[str, Any] = {}
                files: dict[str, bytes] = {}
                doc: dict[str, Any] = {}
                if content_type.startswith("multipart/form-data"):
                    fields, files = parse_multipart(content_type, body)
                    record["form"] = fields
                elif content_type.startswith("application/json") and body:
                    doc = json.loads(body)
                    record["json"] = doc
                else:
                    record["body"] = body
                outer.requests.append(record)
                family = self._family(path)
                if outer.delay_s:
                    time.sleep(outer.delay_s)
                if not self._authorised(family):
                    # the real services name (part of) the key they were given: bolo must not pass that on
                    presented = self.headers.get("Authorization") or self.headers.get("xi-api-key") or ""
                    self._error(family, 403 if family == "sarvam" else 401, f"Incorrect API key provided: {presented}")
                    return
                status = outer.fail_always or (outer.fail.pop(0) if outer.fail else None)
                if status:
                    self._error(family, status, f"simulated failure {status} for key {outer.key}")
                    return
                audio = files.get("file", body if family == "deepgram" else b"")
                secs = seconds_of(audio) if audio else 0.0
                # ---- OpenAI-compatible ---------------------------------------------------------------
                if path == "/v1/models":
                    self._json(200, {"object": "list", "data": [{"id": "whisper-1"}, {"id": "gpt-4o-mini-tts"}]})
                elif path in ("/v1/audio/transcriptions", "/v1/audio/translations"):
                    language = fields.get("language")
                    text = f"heard {secs:.1f} seconds" + (f" in {language}" if language else "")
                    if path.endswith("translations"):
                        text = f"translated {secs:.1f} seconds"
                    if fields.get("prompt") == "bad request":
                        self._error(family, 400, "the prompt is not acceptable")
                    elif fields.get("stream") == "true":
                        words = text.split()
                        events = [
                            {"type": "transcript.text.delta", "delta": w if i == 0 else " " + w}
                            for i, w in enumerate(words)
                        ]
                        events.append({"type": "transcript.text.done", "text": text})
                        self._chunked("text/event-stream", [f"data: {json.dumps(e)}\n\n".encode() for e in events])
                    elif fields.get("response_format") == "verbose_json":
                        words = text.split()
                        step = secs / max(1, len(words))
                        out: dict[str, Any] = {
                            "task": "transcribe", "language": "hindi" if language == "hi" else "english",
                            "duration": secs, "text": text,
                            "segments": [{"id": 0, "start": 0.0, "end": secs, "text": text, "avg_logprob": -0.2,
                                          "no_speech_prob": 0.01}],
                        }  # fmt: skip
                        if fields.get("timestamp_granularities[]"):
                            out["words"] = [
                                {"word": w, "start": round(i * step, 3), "end": round((i + 1) * step, 3)}
                                for i, w in enumerate(words)
                            ]
                        self._json(200, out)
                    else:
                        self._json(200, {"text": text, "usage": {"type": "duration", "seconds": round(secs)}})
                elif path == "/v1/audio/speech":
                    if doc.get("input") == "bad request":
                        self._error(family, 400, "the input is not acceptable")
                        return
                    data = pcm16(tone(str(doc.get("input", "")), 24000))
                    step = 4800  # 0.1 s
                    self._chunked("audio/pcm", [data[i : i + step] for i in range(0, len(data), step)])
                # ---- Sarvam --------------------------------------------------------------------------
                elif path == "/speech-to-text":
                    code = fields.get("language_code", "unknown")
                    mode = fields.get("mode", "transcribe")
                    text = (
                        f"heard {secs:.1f} seconds in {code}"
                        if mode != "translate"
                        else f"translated {secs:.1f} seconds"
                    )
                    out = {
                        "request_id": "s1",
                        "transcript": text,
                        "language_code": "hi-IN" if code == "unknown" else code,
                    }
                    if fields.get("with_timestamps") == "true":
                        half = secs / 2
                        out["timestamps"] = {
                            "words": [" ".join(text.split()[:2]), " ".join(text.split()[2:])],
                            "start_time_seconds": [0.0, half],
                            "end_time_seconds": [half, secs],
                        }
                    self._json(200, out)
                elif path == "/speech-to-text-translate":
                    self._json(
                        200,
                        {"request_id": "s2", "transcript": f"translated {secs:.1f} seconds", "language_code": "hi-IN"},
                    )
                elif path == "/text-to-speech":
                    if not doc.get("text"):
                        self._error(family, 422, "text is required")
                        return
                    rate = int(doc.get("speech_sample_rate") or 22050)
                    data = wav(tone(str(doc["text"]), rate), rate)
                    self._json(200, {"request_id": "s3", "audios": [base64.b64encode(data).decode()]})
                # ---- ElevenLabs ----------------------------------------------------------------------
                elif path == "/v2/voices":
                    self._json(
                        200, {"voices": VOICES, "has_more": False, "total_count": len(VOICES), "next_page_token": None}
                    )
                elif path == "/v1/speech-to-text":
                    text = f"heard {secs:.1f} seconds"
                    words = []
                    step = secs / 3
                    for i, w in enumerate(text.split()):
                        if i:
                            words.append({"text": " ", "type": "spacing", "start": i * step, "end": i * step})
                        words.append({"text": w, "type": "word", "start": i * step, "end": (i + 1) * step})
                    self._json(200, {"language_code": fields.get("language_code") or "en", "language_probability": 0.99,
                                     "text": text, "words": words})  # fmt: skip
                elif path.startswith("/v1/text-to-speech/") and path.endswith("/stream"):
                    voice = path.split("/")[3]
                    if voice not in {v["voice_id"] for v in VOICES}:
                        self._error(family, 404, f"voice {voice} was not found")
                        return
                    fmt = dict(query).get("output_format", "mp3_44100_128")
                    rate = int(fmt.split("_")[1])
                    data = pcm16(tone(str(doc.get("text", "")), rate))
                    step = rate // 5
                    self._chunked("application/octet-stream", [data[i : i + step] for i in range(0, len(data), step)])
                # ---- Deepgram ------------------------------------------------------------------------
                elif path == "/v1/projects":
                    self._json(200, {"projects": [{"project_id": "p1", "name": "test"}]})
                elif path == "/v1/listen":
                    q = dict(query)
                    text = f"heard {secs:.1f} seconds"
                    step = secs / 3
                    words = [
                        {"word": w, "punctuated_word": w.capitalize() if i == 0 else w, "start": i * step,
                         "end": (i + 1) * step, "confidence": 0.9}
                        for i, w in enumerate(text.split())
                    ]  # fmt: skip
                    channel: dict[str, Any] = {
                        "alternatives": [{"transcript": text.capitalize(), "confidence": 0.9, "words": words}]
                    }
                    if q.get("detect_language") == "true":
                        channel["detected_language"] = "hi"
                    self._json(200, {
                        "metadata": {"request_id": "d1", "duration": secs, "channels": 1},
                        "results": {"channels": [channel],
                                    "utterances": [{"start": 0.0, "end": secs, "transcript": text.capitalize()}]},
                    })  # fmt: skip
                else:
                    self._error(family, 404, f"no route {path}")

            def do_GET(self) -> None:
                self._handle("GET")

            def do_POST(self) -> None:
                self._handle("POST")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.httpd.daemon_threads = True
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> FakeProviders:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def last(self, path: str) -> dict[str, Any]:
        return next(r for r in reversed(self.requests) if r["path"].startswith(path))

    def count(self, path: str) -> int:
        return sum(1 for r in self.requests if r["path"].startswith(path))


class FakeLiveServer:
    """Deepgram's live socket (``/v1/listen``) and Sarvam's (``/speech-to-text/ws``).

    Deepgram: an interim ``Results`` per 0.5 s of audio; after 0.4 s of silence (zeros) or ``Finalize`` a final
    one with ``speech_final``. Sarvam: one ``data`` message per ``flush`` (or after 0.4 s of silence)."""

    def __init__(self, key: str = KEY) -> None:
        self.key = key
        self.port = 0
        self.paths: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.control: list[str] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Future[None] | None = None
        self.thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> FakeLiveServer:
        self.thread.start()
        self._ready.wait(10)
        return self

    def stop(self) -> None:
        if self.loop is not None and self._stop is not None:
            self.loop.call_soon_threadsafe(self._stop.set_result, None)
        self.thread.join(5)

    def _check(self, connection: ServerConnection, request: Request) -> Response | None:
        self.paths.append(request.path)
        self.headers.append(dict(request.headers.raw_items()))
        deepgram = request.path.startswith("/v1/listen")
        ok = (
            request.headers.get("Authorization") == f"Token {self.key}"
            if deepgram
            else request.headers.get("Api-Subscription-Key") == self.key
        )
        if not ok:
            return connection.respond(HTTPStatus.UNAUTHORIZED, "Invalid credentials\n")
        return None

    async def _main(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._stop = self.loop.create_future()
        async with serve(self._session, "127.0.0.1", 0, process_request=self._check) as server:
            self.port = int(next(iter(server.sockets)).getsockname()[1])
            self._ready.set()
            await self._stop

    async def _session(self, ws: ServerConnection) -> None:
        assert ws.request is not None
        deepgram = ws.request.path.startswith("/v1/listen")
        speech = 0.0  # seconds of non-silent audio in the open utterance
        silence = 0.0
        total = 0.0
        reported = 0.0

        async def final(from_finalize: bool = False) -> None:
            nonlocal speech, silence, reported
            if speech <= 0:
                return
            text = f"heard {speech:.1f} seconds"
            if deepgram:
                words = [{"word": w, "start": total - speech, "end": total, "confidence": 0.9} for w in text.split()]
                await ws.send(json.dumps({
                    "type": "Results", "is_final": True, "speech_final": True, "from_finalize": from_finalize,
                    "start": total - speech, "duration": speech,
                    "channel": {"alternatives": [{"transcript": text, "words": words}]},
                }))  # fmt: skip
            else:
                await ws.send(json.dumps({"type": "events", "data": {"signal_type": "END_SPEECH"}}))
                await ws.send(json.dumps({
                    "type": "data",
                    "data": {"request_id": "x", "transcript": text, "language_code": "hi-IN",
                             "metrics": {"audio_duration": speech, "processing_latency": 0.01}},
                }))  # fmt: skip
            speech, silence, reported = 0.0, 0.0, 0.0

        async for message in ws:
            pcm = b""
            if isinstance(message, bytes):
                pcm = message
            else:
                doc = json.loads(message)
                if "audio" in doc:
                    pcm = base64.b64decode(doc["audio"]["data"])
                else:
                    kind = str(doc.get("type"))
                    self.control.append(kind)
                    if kind in ("Finalize", "flush"):
                        await final(from_finalize=True)
                    elif kind == "CloseStream":
                        await final()
                        break
                    continue
            x = np.frombuffer(pcm, dtype="<i2")
            secs = len(x) / 16000
            total += secs
            if len(x) and np.abs(x).max() > 300:
                speech += secs
                silence = 0.0
            else:
                silence += secs
            if deepgram and speech - reported >= 0.5:
                reported = speech
                await ws.send(json.dumps({
                    "type": "Results", "is_final": False, "speech_final": False,
                    "channel": {"alternatives": [{"transcript": f"hearing {speech:.1f}", "words": []}]},
                }))  # fmt: skip
            if silence >= 0.4:
                await final()
