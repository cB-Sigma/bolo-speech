"""Stand-ins for the cloud speech APIs the ``remote`` engine talks to (CI has no network and no keys).

One HTTP server answers the routes of all five API families, each with its own authentication header:

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
soniox             under ``/soniox``: ``GET /v1/models``, ``POST /v1/files``,  ``Authorization: Bearer``
                   ``POST /v1/transcriptions``, ``GET /v1/transcriptions/{id}``
                   (and ``/transcript``), ``DELETE`` of both
=================  ==========================================================  ===========================

Recognition answers ``"heard <seconds> seconds"`` (plus the language it was told); synthesis answers a tone of
0.05 s per character. ``fail`` holds HTTP statuses to answer the next requests with, ``delay_s`` makes every
answer slow (to test timeouts), and the text ``bad request`` is refused with 400. Every request is recorded in
``requests`` (method, path, query, headers, form fields or JSON body).

:class:`FakeLiveServer` is the WebSocket side: Deepgram's ``/v1/listen``, Sarvam's ``/speech-to-text/ws`` and
Soniox's ``/transcribe-websocket``.

Soniox's routes live under ``/soniox`` here (its ``/v1/models`` would otherwise meet OpenAI's): a test points the
provider's ``base_url`` at ``<url>/soniox``. A transcription is ``queued``, then ``processing``, then
``completed`` on the third look (``soniox_fail`` makes it end in ``error`` instead); its transcript is made of
pieces of words, as Soniox's tokens are.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
import uuid
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


def soniox_tokens(secs: float, config: dict[str, Any]) -> list[dict[str, Any]]:
    """``heard <secs> seconds.`` as Soniox's tokens: pieces of words, a leading space starting a word."""
    pieces = [["He", "ard"], [f" {secs:.1f}"], [" sec", "onds", "."]]
    step = secs / 3
    hints = config.get("language_hints") or []
    tokens: list[dict[str, Any]] = []
    for i, word in enumerate(pieces):
        t0 = i * step
        for j, text in enumerate(word):
            a = t0 + step * j / len(word)
            tok: dict[str, Any] = {
                "text": text, "start_ms": round(a * 1000), "end_ms": round((a + step / len(word)) * 1000),
                "confidence": 0.93,
            }  # fmt: skip
            if config.get("enable_speaker_diarization"):
                tok["speaker"] = "1" if i == 0 else "2"
            if config.get("enable_language_identification"):
                tok["language"] = hints[0] if hints else "hi"
            if config.get("translation"):
                tok["translation_status"] = "original"
            tokens.append(tok)
    if config.get("translation"):
        for text in ["Trans", "lated", f" {secs:.1f}", " seconds"]:
            tokens.append({"text": text, "confidence": 0.9, "translation_status": "translation", "language": "en",
                           "source_language": hints[0] if hints else "hi"})  # fmt: skip
    return tokens


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
        #: Soniox: uploaded files, transcriptions, and (error_type, message) to fail the next transcription with
        self.soniox_files: dict[str, bytes] = {}
        self.soniox_jobs: dict[str, dict[str, Any]] = {}
        self.soniox_fail: tuple[str, str] | None = None
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
                elif family == "soniox":
                    self._json(code, {"status_code": code, "error_type": "invalid_request" if code == 400 else "error",
                                      "message": message, "validation_errors": [], "request_id": "r",
                                      "more_info": "https://docs.soniox.com/api-reference/errors"}, headers)  # fmt: skip
                else:
                    self._json(code, {"error": {"message": message, "type": "invalid_request_error"}}, headers)

            def _family(self, path: str) -> str:
                if path.startswith("/soniox/"):
                    return "soniox"
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
                    "soniox": h.get("Authorization") == f"Bearer {outer.key}",
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
                if family == "soniox":
                    self._soniox(method, path[len("/soniox") :], doc, files)
                    return
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

            def _soniox(self, method: str, path: str, doc: dict[str, Any], files: dict[str, bytes]) -> None:
                parts = path.strip("/").split("/")
                if method == "GET" and path == "/v1/models":
                    self._json(200, {"models": [
                        {"id": "stt-async-v6", "name": "Speech-to-Text Async v6", "transcription_mode": "async",
                         "languages": [{"code": "en", "name": "English"}], "aliased_model_id": None},
                        {"id": "stt-rt-v6", "name": "Speech-to-Text Real-time v6", "transcription_mode": "real_time",
                         "languages": [{"code": "en", "name": "English"}], "aliased_model_id": None},
                    ]})  # fmt: skip
                elif method == "POST" and path == "/v1/files":
                    file_id = str(uuid.uuid4())
                    outer.soniox_files[file_id] = files.get("file", b"")
                    self._json(201, {"id": file_id, "filename": "audio.wav", "size": len(files.get("file", b"")),
                                     "created_at": "2026-11-17T00:00:00Z"})  # fmt: skip
                elif method == "POST" and path == "/v1/transcriptions":
                    if doc.get("file_id") not in outer.soniox_files:
                        self._error("soniox", 400, "file_id is not a file of this project")
                        return
                    if doc.get("model", "").startswith("stt-rt-"):
                        self._error("soniox", 400, f"model {doc['model']} is a real-time model")
                        return
                    job_id = str(uuid.uuid4())
                    outer.soniox_jobs[job_id] = {"config": doc, "looks": 0, "fail": outer.soniox_fail}
                    outer.soniox_fail = None
                    self._json(201, self._job(job_id))
                elif len(parts) >= 3 and parts[:2] == ["v1", "transcriptions"] and parts[2] in outer.soniox_jobs:
                    job_id = parts[2]
                    job = outer.soniox_jobs[job_id]
                    if method == "DELETE":
                        if job["looks"] < 2:
                            self._error("soniox", 409, "the transcription is still processing")
                            return
                        del outer.soniox_jobs[job_id]
                        self._send(204, b"", "application/json")
                    elif len(parts) == 4 and parts[3] == "transcript":
                        if self._job(job_id)["status"] != "completed":
                            self._error("soniox", 409, "invalid transcription state")
                            return
                        secs = seconds_of(outer.soniox_files.get(job["config"]["file_id"], b""))
                        tokens = soniox_tokens(secs, job["config"])
                        text = "".join(t["text"] for t in tokens if t.get("translation_status") != "translation")
                        self._json(200, {"id": job_id, "text": text, "tokens": tokens})
                    else:
                        job["looks"] += 1
                        self._json(200, self._job(job_id))
                elif method == "DELETE" and len(parts) == 3 and parts[:2] == ["v1", "files"]:
                    if outer.soniox_files.pop(parts[2], None) is None:
                        self._error("soniox", 404, "file not found")
                        return
                    self._send(204, b"", "application/json")
                else:
                    self._error("soniox", 404, f"no route {path}")

            def _job(self, job_id: str) -> dict[str, Any]:
                job = outer.soniox_jobs[job_id]
                config = job["config"]
                status = ("queued", "processing", "completed")[min(2, job["looks"])]
                error = job["fail"] if status == "completed" else None
                return {
                    "id": job_id, "status": "error" if error else status, "created_at": "2026-11-17T00:00:00Z",
                    "model": config.get("model"), "file_id": config.get("file_id"), "filename": "audio.wav",
                    "language_hints": config.get("language_hints"),
                    "enable_speaker_diarization": bool(config.get("enable_speaker_diarization")),
                    "enable_language_identification": bool(config.get("enable_language_identification")),
                    "error_type": error[0] if error else None, "error_message": error[1] if error else None,
                }  # fmt: skip

            def do_GET(self) -> None:
                self._handle("GET")

            def do_DELETE(self) -> None:
                self._handle("DELETE")

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

    def last(self, path: str, method: str | None = None) -> dict[str, Any]:
        return next(r for r in reversed(self.requests) if r["path"].startswith(path) and method in (None, r["method"]))

    def count(self, path: str) -> int:
        return sum(1 for r in self.requests if r["path"].startswith(path))


class FakeLiveServer:
    """Deepgram's live socket (``/v1/listen``), Sarvam's (``/speech-to-text/ws``) and Soniox's
    (``/transcribe-websocket``).

    Deepgram: an interim ``Results`` per 0.5 s of audio; after 0.4 s of silence (zeros) or ``Finalize`` a final
    one with ``speech_final``. Sarvam: one ``data`` message per ``flush`` (or after 0.4 s of silence). Soniox: a
    start message first (kept in ``configs``); per 0.5 s of audio the final token ``heard`` (once) and non-final
    tokens; after 0.4 s of silence the rest as final tokens and ``<end>``, after ``finalize`` the same with
    ``<fin>``; an empty text frame ends the stream with ``finished``."""

    def __init__(self, key: str = KEY) -> None:
        self.key = key
        self.port = 0
        self.paths: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.control: list[str] = []
        self.configs: list[dict[str, Any]] = []
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
        if request.path.startswith("/transcribe-websocket"):
            ok = request.headers.get("Authorization") == f"Bearer {self.key}"
        elif deepgram:
            ok = request.headers.get("Authorization") == f"Token {self.key}"
        else:
            ok = request.headers.get("Api-Subscription-Key") == self.key
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
        if ws.request.path.startswith("/transcribe-websocket"):
            await self._soniox(ws)
            return
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

    async def _soniox(self, ws: ServerConnection) -> None:
        first = await ws.recv()
        if not isinstance(first, str):
            await ws.send(json.dumps({"tokens": [], "error_code": 400, "error_type": "invalid_request",
                                      "error_message": "Start request expected."}))  # fmt: skip
            return
        self.configs.append(json.loads(first))
        speech = silence = total = reported = 0.0
        heard_sent = False  # the first word went out as a final token already

        def tokens(start: float, final: bool) -> list[dict[str, Any]]:
            words = ([] if heard_sent else ["heard"]) + [f" {speech:.1f}", " sec", "onds"]
            return [
                {"text": w, "start_ms": round(start * 1000), "end_ms": round(total * 1000), "confidence": 0.9,
                 "is_final": final}
                for w in words
            ]  # fmt: skip

        async def final(marker: str) -> None:
            nonlocal speech, silence, reported, heard_sent
            if speech > 0:
                done = tokens(total - speech, True) + [{"text": marker, "is_final": True}]
                await ws.send(json.dumps({"tokens": done, "final_audio_proc_ms": round(total * 1000),
                                          "total_audio_proc_ms": round(total * 1000)}))  # fmt: skip
            elif marker == "<fin>":
                await ws.send(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
            speech, silence, reported, heard_sent = 0.0, 0.0, 0.0, False

        async for message in ws:
            if isinstance(message, str):
                if message == "":
                    self.control.append("end")
                    await final("<end>")
                    await ws.send(
                        json.dumps(
                            {
                                "tokens": [],
                                "final_audio_proc_ms": round(total * 1000),
                                "total_audio_proc_ms": round(total * 1000),
                                "finished": True,
                            }
                        )
                    )
                    break
                kind = str(json.loads(message).get("type"))
                self.control.append(kind)
                if kind == "finalize":
                    await final("<fin>")
                continue
            x = np.frombuffer(message, dtype="<i2")
            secs = len(x) / 16000
            total += secs
            if len(x) and np.abs(x).max() > 300:
                speech += secs
                silence = 0.0
            else:
                silence += secs
            if speech - reported >= 0.5:
                reported = speech
                out = []
                if not heard_sent:
                    out.append({"text": "heard", "start_ms": round((total - speech) * 1000), "end_ms": 0,
                                "confidence": 0.9, "is_final": True})  # fmt: skip
                    heard_sent = True
                out += [{"text": " hear", "is_final": False}, {"text": "ing", "is_final": False}]
                await ws.send(json.dumps({"tokens": out, "final_audio_proc_ms": 0,
                                          "total_audio_proc_ms": round(total * 1000)}))  # fmt: skip
            if silence >= 0.4 and speech > 0:
                await final("<end>")
