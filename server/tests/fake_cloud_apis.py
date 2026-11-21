"""Stand-ins for the cloud speech APIs added after the first five (tests/fake_providers.py has those), written from
each provider's public documentation. CI has no network and no keys.

:class:`FakeCloudHTTP` answers the REST routes:

==========  ==================================================================  ==============================
family      routes                                                              key header
==========  ==================================================================  ==============================
assemblyai  ``POST /v2/upload``, ``POST /v2/transcript``,                       ``Authorization: <key>``
            ``GET`` / ``DELETE /v2/transcript/{id}``, ``GET /v2/transcript``
==========  ==================================================================  ==============================

An AssemblyAI transcript is ``queued``, then ``processing``, then ``completed`` on the third look
(``assemblyai_fail`` makes it end in ``error``). Its text is ``"Heard <seconds> seconds."``; with
``speaker_labels`` the first word is speaker ``A``'s and the rest ``B``'s.

:class:`FakeCloudSockets` is the WebSocket side, routed by path:

* ``/v3/ws``: AssemblyAI's Streaming API v3 (``Authorization: <key>``). ``Begin`` first; per 0.5 s of speech a
  partial ``Turn``; after 0.4 s of silence, or on ``ForceEndpoint``, the complete turn (``end_of_turn`` and
  ``turn_is_formatted``) with its words and, with ``speaker_labels=true``, ``speaker_label`` (``A`` for the first
  turn, ``B`` for the next, …); ``Terminate`` answers ``Termination`` and closes. A binary frame shorter than 20 ms
  or longer than 1 s is an ``Error`` (3005), as AssemblyAI documents.

Speech is any audio louder than silence (an absolute sample above 300). Every connection's path, headers and
client messages are recorded.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.http11 import Request, Response

#: not a real key: the stand-ins accept exactly this string
KEY = "test-key-not-real-5d2f8e61aa93"
SR = 16000


def seconds_of_wav(audio: bytes) -> float:
    """Seconds of a PCM16 WAV file (the 44-byte header bolo writes), else of raw PCM16 at 16 kHz."""
    if audio[:4] == b"RIFF":
        import io

        import soundfile as sf

        with sf.SoundFile(io.BytesIO(audio)) as f:
            return len(f) / f.samplerate
    return len(audio) / 2 / SR


class FakeCloudHTTP:
    def __init__(self, key: str = KEY) -> None:
        self.key = key
        self.requests: list[dict[str, Any]] = []
        self.uploads: dict[str, bytes] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []
        #: AssemblyAI: the error message to fail the next transcript with
        self.assemblyai_fail: str | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, code: int, body: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, doc: Any) -> None:
                self._send(code, json.dumps(doc).encode(), "application/json")

            def _handle(self, method: str) -> None:
                url = urlsplit(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                record: dict[str, Any] = {
                    "method": method, "path": url.path, "query": dict(parse_qsl(url.query)),
                    "headers": dict(self.headers.items()), "body": body,
                }  # fmt: skip
                if self.headers.get("Content-Type", "").startswith("application/json") and body:
                    record["json"] = json.loads(body)
                outer.requests.append(record)
                if url.path.startswith("/v2/"):
                    self._assemblyai(method, url.path, record)
                else:
                    self._json(404, {"error": "not found"})

            # ---- AssemblyAI --------------------------------------------------------------------------
            def _assemblyai(self, method: str, path: str, record: dict[str, Any]) -> None:
                if self.headers.get("Authorization") != outer.key:
                    self._json(401, {"error": "Authentication error, API token missing/invalid"})
                    return
                if path == "/v2/upload" and method == "POST":
                    if self.headers.get("Content-Type") != "application/octet-stream":
                        self._json(400, {"error": "upload the raw file as application/octet-stream"})
                        return
                    upload_id = uuid.uuid4().hex
                    outer.uploads[upload_id] = record["body"]
                    self._json(200, {"upload_url": f"https://cdn.assemblyai.com/upload/{upload_id}"})
                elif path == "/v2/transcript" and method == "POST":
                    doc = record.get("json") or {}
                    upload_id = str(doc.get("audio_url", "")).rsplit("/", 1)[-1]
                    if upload_id not in outer.uploads:
                        self._json(400, {"error": "Download error, unable to download audio_url"})
                        return
                    job_id = uuid.uuid4().hex
                    outer.jobs[job_id] = {
                        "request": doc,
                        "looks": 0,
                        "seconds": seconds_of_wav(outer.uploads[upload_id]),
                    }
                    self._json(200, {"id": job_id, "status": "queued", **{k: doc.get(k) for k in ("speech_models",)}})
                elif path == "/v2/transcript" and method == "GET":
                    self._json(200, {"page_details": {"limit": 1, "result_count": 0}, "transcripts": []})
                elif path.startswith("/v2/transcript/"):
                    job_id = path.rsplit("/", 1)[-1]
                    job = outer.jobs.get(job_id)
                    if job is None:
                        self._json(404, {"error": "Transcript lookup error, transcript id not found"})
                        return
                    if method == "DELETE":
                        outer.deleted.append(job_id)
                        self._json(200, {"id": job_id, "status": "completed", "text": "Deleted by user."})
                        return
                    job["looks"] += 1
                    if job["looks"] < 3:
                        self._json(200, {"id": job_id, "status": "queued" if job["looks"] == 1 else "processing"})
                    elif outer.assemblyai_fail:
                        message, outer.assemblyai_fail = outer.assemblyai_fail, None
                        self._json(200, {"id": job_id, "status": "error", "error": message})
                    else:
                        self._json(200, {"id": job_id, **assemblyai_transcript(job["seconds"], job["request"])})
                else:
                    self._json(404, {"error": "not found"})

            def do_GET(self) -> None:
                self._handle("GET")

            def do_POST(self) -> None:
                self._handle("POST")

            def do_DELETE(self) -> None:
                self._handle("DELETE")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self) -> FakeCloudHTTP:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def last(self, path: str, method: str | None = None) -> dict[str, Any]:
        return next(r for r in reversed(self.requests) if r["path"] == path and method in (None, r["method"]))


def assemblyai_transcript(seconds: float, request: dict[str, Any]) -> dict[str, Any]:
    """A completed AssemblyAI transcript of ``seconds`` of audio, shaped as GET /v2/transcript/{id} documents it."""
    text = f"Heard {seconds:.1f} seconds."
    tokens = text.split()
    step = int(seconds * 1000 / len(tokens))
    diarize = bool(request.get("speaker_labels"))
    words = [
        {"text": t, "start": i * step, "end": (i + 1) * step, "confidence": 0.9,
         "speaker": ("A" if i == 0 else "B") if diarize else None}
        for i, t in enumerate(tokens)
    ]  # fmt: skip
    out: dict[str, Any] = {
        "status": "completed", "text": text, "words": words, "language_code": request.get("language_code") or "en",
        "language_confidence": None if request.get("language_code") else 0.97,
        "speech_model_used": (request.get("speech_models") or ["universal-3-6-pro"])[0],
        "speaker_labels": diarize, "utterances": None,
    }  # fmt: skip
    if diarize:
        out["utterances"] = [
            {"speaker": "A", "start": 0, "end": step, "text": tokens[0], "confidence": 0.9, "words": words[:1]},
            {"speaker": "B", "start": step, "end": len(tokens) * step, "text": " ".join(tokens[1:]), "confidence": 0.9,
             "words": words[1:]},
        ]  # fmt: skip
    return out


class FakeCloudSockets:
    def __init__(self, key: str = KEY) -> None:
        self.key = key
        self.port = 0
        self.paths: list[str] = []
        self.headers: list[dict[str, str]] = []
        #: the JSON (text) messages each connection sent, in order
        self.messages: list[list[dict[str, Any]]] = []
        #: the length in bytes of each binary frame each connection sent
        self.frames: list[list[int]] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Future[None] | None = None
        self.thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> FakeCloudSockets:
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
        if request.path.startswith("/v3/ws"):
            ok = request.headers.get("Authorization") == self.key
        else:
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
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
        messages: list[dict[str, Any]] = []
        frames: list[int] = []
        self.messages.append(messages)
        self.frames.append(frames)
        if ws.request.path.startswith("/v3/ws"):
            await self._assemblyai(ws, dict(parse_qsl(urlsplit(ws.request.path).query)), messages, frames)

    # ---- AssemblyAI Streaming API v3 ---------------------------------------------------------------------------
    async def _assemblyai(
        self, ws: ServerConnection, query: dict[str, str], messages: list[dict[str, Any]], frames: list[int]
    ) -> None:
        rate = int(query.get("sample_rate") or SR)
        diarize = query.get("speaker_labels") == "true"
        await ws.send(json.dumps({"type": "Begin", "id": str(uuid.uuid4()), "expires_at": 1792857632,
                                  "configuration": {"model": query.get("speech_model", "universal-3-6-pro")}}))  # fmt: skip
        speech = silence = total = reported = 0.0
        turn = 0

        def words(final: bool) -> list[dict[str, Any]]:
            start = total - speech - silence
            text = f"Heard {speech:.1f} seconds." if final else f"hearing {speech:.1f}"
            out = []
            for i, w in enumerate(text.split()):
                word: dict[str, Any] = {"start": round((start + i * 0.3) * 1000), "end": round((start + i * 0.3 + 0.25) * 1000),
                                        "text": w, "confidence": 0.9, "word_is_final": final}  # fmt: skip
                if final and diarize:
                    word["speaker"] = chr(ord("A") + turn)
                out.append(word)
            return out

        async def end_turn() -> None:
            nonlocal speech, silence, reported, turn
            if speech <= 0:
                await ws.send(json.dumps({"type": "Turn", "turn_order": turn, "turn_is_formatted": True,
                                          "end_of_turn": True, "transcript": "", "words": [], "utterance": ""}))  # fmt: skip
                return
            msg: dict[str, Any] = {"type": "Turn", "turn_order": turn, "turn_is_formatted": True, "end_of_turn": True,
                                   "end_of_turn_confidence": 1.0, "transcript": f"Heard {speech:.1f} seconds.",
                                   "words": words(True), "utterance": f"Heard {speech:.1f} seconds."}  # fmt: skip
            if diarize:
                msg.update(speaker_label=chr(ord("A") + turn), speaker_confidence=0.97)
            await ws.send(json.dumps(msg))
            speech = silence = reported = 0.0
            turn += 1

        async for message in ws:
            if isinstance(message, str):
                doc = json.loads(message)
                messages.append(doc)
                if doc.get("type") == "ForceEndpoint":
                    await end_turn()
                elif doc.get("type") == "Terminate":
                    if speech > 0:
                        await end_turn()
                    await ws.send(json.dumps({"type": "Termination", "audio_duration_seconds": round(total),
                                              "session_duration_seconds": round(total) + 1}))  # fmt: skip
                    await ws.close(1000)
                    return
                continue
            frames.append(len(message))
            secs = len(message) / 2 / rate
            if not 0.02 <= secs <= 1.0:
                await ws.send(json.dumps({"type": "Error", "error_code": 3005,
                                          "error": "Input duration violation: audio chunks must be 20-1000 ms"}))  # fmt: skip
                await ws.close(1008)
                return
            x = np.frombuffer(message, dtype="<i2")
            total += secs
            if np.abs(x).max() > 300:
                if speech == 0:
                    await ws.send(json.dumps({"type": "SpeechStarted", "timestamp": round((total - secs) * 1000),
                                              "confidence": 0.9}))  # fmt: skip
                speech += secs
                silence = 0.0
            elif speech > 0:
                silence += secs
            if speech - reported >= 0.5:
                reported = speech
                await ws.send(json.dumps({"type": "Turn", "turn_order": turn, "turn_is_formatted": False,
                                          "end_of_turn": False, "transcript": f"hearing {speech:.1f}",
                                          "end_of_turn_confidence": 0.1, "words": words(False), "utterance": ""}))  # fmt: skip
            if silence >= 0.4 and speech > 0:
                await end_turn()
