"""Stand-ins for the cloud speech APIs added after the first five (tests/fake_providers.py has those), written from
each provider's public documentation. CI has no network and no keys.

:class:`FakeCloudHTTP` answers the REST routes:

==========  ==================================================================  ==============================
family      routes                                                              key header
==========  ==================================================================  ==============================
assemblyai  ``POST /v2/upload``, ``POST /v2/transcript``,                       ``Authorization: <key>``
            ``GET`` / ``DELETE /v2/transcript/{id}``, ``GET /v2/transcript``
elevenlabs  ``POST /v1/speech-to-text`` (Scribe, with ``diarize``)              ``xi-api-key``
azure       ``POST /cognitiveservices/v1`` (Azure Speech text to speech, SSML)  ``Ocp-Apim-Subscription-Key``
==========  ==================================================================  ==============================

Azure Speech answers a tone of 0.05 s per character of the SSML's text, as MP3 (``audio-24khz-160kbitrate-mono-mp3``)
or raw PCM (``raw-24khz-16bit-mono-pcm``); a voice name without a MAI model suffix, or an unknown voice, is a 400.

An AssemblyAI transcript is ``queued``, then ``processing``, then ``completed`` on the third look
(``assemblyai_fail`` makes it end in ``error``). Its text is ``"Heard <seconds> seconds."``; with
``speaker_labels`` the first word is speaker ``A``'s and the rest ``B``'s.

:class:`FakeCloudSockets` is the WebSocket side, routed by path:

* ``/v3/ws``: AssemblyAI's Streaming API v3 (``Authorization: <key>``). ``Begin`` first; per 0.5 s of speech a
  partial ``Turn``; after 0.4 s of silence, or on ``ForceEndpoint``, the complete turn (``end_of_turn`` and
  ``turn_is_formatted``) with its words and, with ``speaker_labels=true``, ``speaker_label`` (``A`` for the first
  turn, ``B`` for the next, …); ``Terminate`` answers ``Termination`` and closes. A binary frame shorter than 20 ms
  or longer than 1 s is an ``Error`` (3005), as AssemblyAI documents.

ElevenLabs' Scribe answers ``"Heard <seconds> seconds."``; with ``diarize=true`` the first word is ``speaker_0``'s and
the rest ``speaker_1``'s.

* ``/v1/speech-to-text/realtime``: ElevenLabs' realtime socket (``xi-api-key``). ``session_started`` first; JSON
  ``input_audio_chunk`` messages; per 0.5 s of speech a ``partial_transcript``; after 0.4 s of silence (the stand-in's
  turn prediction) or a chunk with ``commit: true``, ``committed_transcript`` and, with ``include_timestamps``,
  ``committed_transcript_with_timestamps``. With ``diarize=true`` the committed words carry draft speakers and a
  ``diarized_transcript`` follows each commit with the final ones: segment *n* is first told ``speaker_<2n>``, and from
  the second segment on a ``speakers_merged`` (``speaker_<2n>`` into ``speaker_<n>``) comes before its diarized
  transcript, so the final labels are ``speaker_0``, ``speaker_1``, ….

* ``/mai/v1/realtime?intent=transcription``: MAI-Transcribe-2-Streaming (``api-key``). ``session.created`` first;
  the ``session.update`` (``type: transcription``, a ``transcription.model``, ``turn_detection: null``) must come
  before any audio, and a later one is an ``error``. Per 0.5 s of speech an ``…intermediate``; on
  ``input_audio_buffer.commit``, ``input_audio_buffer.committed``, a ``…delta`` and ``…completed`` with
  ``"Heard <seconds> seconds."`` for the speech since the previous commit.
* ``/v1/realtime`` (``Authorization: Bearer``), in the dialect given as ``realtime``:

  - ``"xai"``: xAI's Voice Agent API. ``session.created`` and ``conversation.created`` first; ``session.update``
    with only the documented fields (``voice``, ``instructions``, ``turn_detection``, ``audio``, ``tools`` with
    nested ``function`` …; anything else is an ``error``); ``input_audio_buffer.commit`` only with
    ``turn_detection.type`` null; ``response.create`` with ``response.modalities``: audio as
    ``response.output_audio.delta`` (PCM at ``audio.output.format.rate``) with ``response.output_audio_transcript.delta``,
    text as ``response.output_text.delta``.
  - ``"rime"``: Rime's Prism. ``session.created`` first; a ``session.update`` with an unknown field or voice is
    refused (``invalid_client_event``, ``invalid_voice``); audio before an accepted update is ``session_not_initialized``;
    16 kHz audio; a commit of less than 100 ms is ``input_audio_buffer_commit_empty``; a tool result is acknowledged
    with ``conversation.item.created`` after 0.2 s; ``response.create`` follows Prism's rules (a plain request with
    turn detection off, ``prism_cause`` ``user_text`` with ``prism_input_text``, ``tool_continuation`` naming the
    parent reply, refused before every result was acknowledged); replies as ``response.text.delta`` and
    ``response.audio.delta`` (24 kHz).

  Replies: to a tool result "The tool said: <output>."; to text with "weather" and tools set, a ``get_weather``
  call; to text "You wrote: <text>."; to audio "I heard <seconds> seconds." (0.1 s of tone per word).

Speech is any audio louder than silence (an absolute sample above 300). Every connection's path, headers and
client messages are recorded.
"""

from __future__ import annotations

import asyncio
import base64
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

from fake_providers import parse_multipart

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
                elif url.path == "/v1/speech-to-text" and method == "POST":
                    self._scribe(record)
                elif url.path == "/cognitiveservices/v1" and method == "POST":
                    self._azure_tts(record)
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

            # ---- ElevenLabs Scribe ----------------------------------------------------------------------
            def _scribe(self, record: dict[str, Any]) -> None:
                if self.headers.get("xi-api-key") != outer.key:
                    self._json(401, {"detail": {"status": "invalid_api_key", "message": "Invalid API key"}})
                    return
                fields, files = parse_multipart(self.headers.get("Content-Type", ""), record["body"])
                record["form"] = fields
                secs = seconds_of_wav(files.get("file", b""))
                diarize = fields.get("diarize") == "true"
                tokens = f"Heard {secs:.1f} seconds.".split()
                step = secs / len(tokens)
                words: list[dict[str, Any]] = []
                for i, t in enumerate(tokens):
                    speaker = ("speaker_0" if i == 0 else "speaker_1") if diarize else None
                    if i:
                        words.append({"text": " ", "type": "spacing", "start": i * step, "end": i * step,
                                      "speaker_id": speaker, "logprob": 0.0})  # fmt: skip
                    words.append({"text": t, "type": "word", "start": i * step, "end": (i + 1) * step,
                                  "speaker_id": speaker, "logprob": -0.1})  # fmt: skip
                self._json(200, {"language_code": fields.get("language_code") or "eng", "language_probability": 0.98,
                                 "text": " ".join(tokens), "words": words, "transcription_id": uuid.uuid4().hex})  # fmt: skip

            # ---- Azure Speech text to speech (MAI-Voice) ---------------------------------------------------
            def _azure_tts(self, record: dict[str, Any]) -> None:
                import io
                import re

                import soundfile as sf

                if self.headers.get("Ocp-Apim-Subscription-Key") != outer.key:
                    self._send(401, b"", "text/plain")
                    return
                if self.headers.get("Content-Type") != "application/ssml+xml":
                    self._send(415, b"", "text/plain")
                    return
                ssml = record["body"].decode()
                match = re.search(r'<voice name="([^"]+)">(.*)</voice>', ssml, re.S)
                lang = re.search(r'xml:lang="([^"]+)"', ssml)
                if not match or not lang:
                    self._send(400, b"SSML parsing error", "text/plain")
                    return
                voice, text = match.group(1), match.group(2)
                record["voice"], record["lang"], record["text"] = voice, lang.group(1), text
                name, _, model = voice.partition(":")
                if model not in ("MAI-Voice-2.1", "MAI-Voice-2.1-Flash") or name.count("-") != 2:
                    self._send(400, f"Unsupported voice {voice}.".encode(), "text/plain")
                    return
                n = int(24000 * 0.05 * max(1, len(text)))
                x = (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / 24000)).astype(np.float32)
                fmt = self.headers.get("X-Microsoft-OutputFormat")
                if fmt == "raw-24khz-16bit-mono-pcm":
                    self._send(200, (x * 32767).astype("<i2").tobytes(), "audio/x-wav")
                elif fmt == "audio-24khz-160kbitrate-mono-mp3":
                    buf = io.BytesIO()
                    sf.write(buf, x, 24000, format="MP3")
                    self._send(200, buf.getvalue(), "audio/mpeg")
                else:
                    self._send(400, b"Unsupported output format", "text/plain")

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


RIME_VOICES = ("astra", "aurora", "brando", "celeste", "orion")
XAI_SESSION_FIELDS = {"model", "instructions", "reasoning", "voice", "turn_detection", "resumption", "audio", "tools",
                      "replace"}  # fmt: skip
RIME_SESSION_FIELDS = {"voice", "speed", "instructions", "tools", "turn_detection", "prism_client_caps", "truncation",
                       "prism_response_metadata", "prism_review", "modalities", "tool_choice", "input_audio_format"}  # fmt: skip


def tone_b64(rate: int, seconds: float = 0.1) -> str:
    t = np.arange(int(rate * seconds)) / rate
    return base64.b64encode((0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype("<i2").tobytes()).decode()


class FakeCloudSockets:
    def __init__(self, key: str = KEY, realtime: str = "xai") -> None:
        self.key = key
        #: the dialect of /v1/realtime: "xai" or "rime"
        self.realtime = realtime
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
        elif request.path.startswith("/v1/speech-to-text/realtime"):
            ok = request.headers.get("xi-api-key") == self.key
        elif request.path.startswith("/v1/realtime"):
            ok = request.headers.get("Authorization") == f"Bearer {self.key}"
        elif request.path.startswith("/mai/v1/realtime"):
            ok = request.headers.get("api-key") == self.key
        elif request.path.startswith("/v1/tts/voices"):  # xAI's REST route on the same host (bolo's key check)
            if request.headers.get("Authorization") != f"Bearer {self.key}":
                return connection.respond(HTTPStatus.UNAUTHORIZED, '{"error": "Incorrect API key provided"}')
            return connection.respond(HTTPStatus.OK, json.dumps({"voices": [{"voice_id": "eve", "name": "Eve"},
                                                                            {"voice_id": "ara", "name": "Ara"}]}))  # fmt: skip
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
        query = dict(parse_qsl(urlsplit(ws.request.path).query))
        if ws.request.path.startswith("/v3/ws"):
            await self._assemblyai(ws, query, messages, frames)
        elif ws.request.path.startswith("/v1/speech-to-text/realtime"):
            await self._scribe_live(ws, query, messages)
        elif ws.request.path.startswith("/v1/realtime"):
            await self._realtime(ws, messages)
        elif ws.request.path.startswith("/mai/v1/realtime"):
            await self._mai_transcribe(ws, messages)

    # ---- MAI-Transcribe-2-Streaming -------------------------------------------------------------------------
    async def _mai_transcribe(self, ws: ServerConnection, messages: list[dict[str, Any]]) -> None:
        prefix = "conversation.item.input_audio_transcription."
        await ws.send(json.dumps({"type": "session.created", "session": {"id": "sess_mai", "type": "transcription",
                                  "audio": {"input": {"turn_detection": {"type": "server_vad"}}}}}))  # fmt: skip
        configured = audio_started = False
        speech = reported = 0.0
        item = 0

        async def error(message: str) -> None:
            await ws.send(json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": message}}))

        async for raw in ws:
            event = json.loads(raw)
            kind = event.get("type")
            messages.append({k: v for k, v in event.items() if k != "audio"})
            if kind == "session.update":
                session = event.get("session") or {}
                inp = (session.get("audio") or {}).get("input") or {}
                if audio_started:
                    await error("settings can't change after the first audio append")
                elif session.get("type") != "transcription" or not (inp.get("transcription") or {}).get("model"):
                    await error("session.type must be transcription with audio.input.transcription.model")
                elif inp.get("turn_detection") is not None or inp.get("noise_reduction") is not None:
                    await error("only null is supported for turn_detection and noise_reduction")
                else:
                    configured = True
                    await ws.send(json.dumps({"type": "session.updated", "session": session}))
            elif kind == "input_audio_buffer.append":
                if not configured:
                    await error("configure the session before sending audio")
                    continue
                audio_started = True
                x = np.frombuffer(base64.b64decode(event["audio"]), dtype="<i2")
                secs = len(x) / SR
                if len(x) and np.abs(x).max() > 300:
                    speech += secs
                if speech - reported >= 0.5:
                    reported = speech
                    await ws.send(json.dumps({"type": prefix + "intermediate", "item_id": f"item_{item}",
                                              "content_index": 0, "intermediate": f"hearing {speech:.1f}"}))  # fmt: skip
            elif kind == "input_audio_buffer.commit":
                await ws.send(json.dumps({"type": "input_audio_buffer.committed"}))
                text = f"Heard {speech:.1f} seconds." if speech > 0 else ""
                if text:
                    await ws.send(json.dumps({"type": prefix + "delta", "item_id": f"item_{item}", "content_index": 0,
                                              "delta": text}))  # fmt: skip
                await ws.send(json.dumps({"type": prefix + "completed", "item_id": f"item_{item}", "content_index": 0,
                                          "transcript": text}))  # fmt: skip
                speech = reported = 0.0
                item += 1

    # ---- xAI's Voice Agent API and Rime's Prism -----------------------------------------------------------------
    async def _realtime(self, ws: ServerConnection, messages: list[dict[str, Any]]) -> None:
        rime = self.realtime == "rime"
        state: dict[str, Any] = {"session": None, "audio_s": 0.0, "text": None, "tool_output": None, "acked": True,
                                 "last_response": None, "responses": 0}  # fmt: skip
        out_rate = 24000

        async def send(event: dict[str, Any]) -> None:
            await ws.send(json.dumps({"event_id": f"evt_{uuid.uuid4().hex[:8]}", **event}))

        async def error(code: str, message: str, param: str | None = None) -> None:
            await send({"type": "error", "error": {"type": "invalid_request_error", "code": code, "message": message,
                                                   "param": param}})  # fmt: skip

        await send({"type": "session.created", "session": {"id": "sess_1", "object": "realtime.session",
                                                           "model": "prism" if rime else "grok-voice-3.0"}})  # fmt: skip
        if not rime:
            await send({"type": "conversation.created", "conversation": {"id": "conv_1"}})

        def in_rate() -> int:
            session = state["session"] or {}
            if rime:
                return SR
            return int(((session.get("audio") or {}).get("input") or {}).get("format", {}).get("rate") or 24000)

        async def reply(modalities: list[str]) -> None:
            state["responses"] += 1
            response_id = f"resp_{state['responses']}"
            state["last_response"] = response_id
            await send({"type": "response.created", "response": {"id": response_id, "status": "in_progress"}})
            session = state["session"] or {}
            tools = session.get("tools") or []
            if state["tool_output"] is None and state["text"] and "weather" in state["text"] and tools:
                call = {"type": "function_call", "id": "item_call", "object": "realtime.item", "call_id": "call_7",
                        "name": "get_weather", "arguments": '{"city": "Pune"}', "status": "completed"}  # fmt: skip
                await send({"type": "response.function_call_arguments.done", "response_id": response_id,
                            "item_id": "item_call", "output_index": 1, "call_id": "call_7", "name": "get_weather",
                            "arguments": call["arguments"]})  # fmt: skip
                await send({"type": "response.output_item.done", "response_id": response_id, "output_index": 1,
                            "item": call})  # fmt: skip
                await send({"type": "response.done", "response": {"id": response_id, "status": "completed",
                                                                  "output": [call], "usage": {"output_tokens": 5}}})  # fmt: skip
                state["text"] = None
                return
            if state["tool_output"] is not None:
                text = f"The tool said: {state['tool_output']}."
            elif state["text"]:
                text = f"You wrote: {state['text']}."
            else:
                text = f"I heard {state['audio_s']:.1f} seconds."
            state.update(text=None, tool_output=None, audio_s=0.0)
            base = {"response_id": response_id, "item_id": "item_msg", "output_index": 0, "content_index": 0}
            await send({"type": "response.output_item.added", "response_id": response_id, "output_index": 0,
                        "item": {"id": "item_msg", "type": "message", "role": "assistant"}})  # fmt: skip
            for i, word in enumerate(text.split()):
                delta = word if i == 0 else " " + word
                if rime:
                    await send({"type": "response.text.delta", **base, "delta": delta})
                    await send({"type": "response.audio.delta", **base, "delta": tone_b64(out_rate)})
                elif "audio" in modalities:
                    await send({"type": "response.output_audio_transcript.delta", **base, "delta": delta})
                    await send({"type": "response.output_audio.delta", **base, "delta": tone_b64(out_rate)})
                else:
                    await send({"type": "response.output_text.delta", **base, "delta": delta})
            if rime:
                await send({"type": "response.text.done", **base, "text": text})
                await send({"type": "response.audio.done", **base})
            await send({"type": "response.done", "response": {"id": response_id, "status": "completed",
                                                              "usage": {"output_tokens": len(text.split())}}})  # fmt: skip

        async for raw in ws:
            event = json.loads(raw)
            messages.append(event)
            kind = event.get("type")
            if kind == "session.update":
                session = event.get("session") or {}
                allowed = RIME_SESSION_FIELDS if rime else XAI_SESSION_FIELDS
                unknown = sorted(set(session) - allowed)
                if unknown:
                    await error("invalid_client_event", f"unknown field {unknown[0]}", unknown[0])
                    continue
                if rime and session.get("voice") not in (None, "", "default", *RIME_VOICES):
                    await error("invalid_voice", f"Prism has no voice {session['voice']!r}", "voice")
                    continue
                if not rime and any(
                    "function" not in t for t in session.get("tools") or [] if t.get("type") == "function"
                ):
                    await error("invalid_request_error", "a function tool needs `function`", "tools")
                    continue
                state["session"] = session
                await send({"type": "session.updated", "session": {"id": "sess_1", **session}})
            elif kind == "input_audio_buffer.append":
                if rime and state["session"] is None:
                    await error("session_not_initialized", "send session.update first")
                    continue
                x = np.frombuffer(base64.b64decode(event["audio"]), dtype="<i2")
                state["audio_s"] += len(x) / in_rate()
            elif kind == "input_audio_buffer.commit":
                session = state["session"] or {}
                if not rime and (session.get("turn_detection") or {}).get("type") is not None:
                    await error("invalid_request_error", "commit is only available with turn_detection type null")
                    continue
                if rime and state["audio_s"] < 0.1:
                    await error("input_audio_buffer_commit_empty", "less than 100 ms of audio to commit")
                    continue
                await send({"type": "input_audio_buffer.committed", "item_id": "item_user"})
            elif kind == "conversation.item.create":
                item = event.get("item") or {}
                if item.get("type") == "function_call_output":
                    state["tool_output"] = item.get("output")
                    if rime:
                        state["acked"] = False

                        async def ack(item_id: str = str(item.get("call_id"))) -> None:
                            await asyncio.sleep(0.2)
                            state["acked"] = True
                            await send({"type": "conversation.item.created", "item": {"id": f"item_{item_id}"}})

                        asyncio.get_running_loop().create_task(ack())
                    else:
                        await send({"type": "conversation.item.added", "item": item})
                elif item.get("type") == "message":
                    state["text"] = item["content"][0]["text"]
                    await send({"type": "conversation.item.added" if not rime else "conversation.item.created",
                                "item": item})  # fmt: skip
            elif kind == "response.create":
                response = event.get("response") or {}
                if rime:
                    metadata = response.get("metadata") or {}
                    cause = metadata.get("prism_cause")
                    extra = sorted(
                        set(response) - {"metadata", "prism_input_text", "prism_instruction", "prism_call_id"}
                    )
                    if extra:
                        await error("invalid_client_event", f"unknown field {extra[0]}", extra[0])
                        continue
                    if cause == "tool_continuation":
                        if metadata.get("prism_parent_response_id") != state["last_response"] or not state["acked"]:
                            await error("tool_continuation_not_ready", "wait for the parent reply and every result")
                            continue
                    elif cause == "user_text":
                        state["text"] = response.get("prism_input_text")
                        await send({"type": "conversation.item.created", "item": {"id": "item_typed"}})
                    elif cause is not None or (state["session"] or {}).get("turn_detection", {}) is not None:
                        await error("invalid_response_cause", "this response.create needs a prism_cause")
                        continue
                    await reply(["text", "audio"])
                else:
                    if set(response) - {"modalities", "instructions", "metadata"}:
                        await error("invalid_request_error", "unknown response field")
                        continue
                    await reply(list(response.get("modalities") or ["text", "audio"]))
            elif kind == "response.cancel":
                await send({"type": "response.done", "response": {"id": state["last_response"], "status": "cancelled"}})

    # ---- ElevenLabs realtime speech to text ------------------------------------------------------------------
    async def _scribe_live(self, ws: ServerConnection, query: dict[str, str], messages: list[dict[str, Any]]) -> None:
        diarize = query.get("diarize") == "true"
        timestamps = query.get("include_timestamps") == "true"
        await ws.send(json.dumps({"message_type": "session_started", "session_id": uuid.uuid4().hex,
                                  "config": {"sample_rate": SR, "audio_format": query.get("audio_format"),
                                             "model_id": query.get("model_id"),
                                             "commit_strategy": query.get("commit_strategy")}}))  # fmt: skip
        speech = silence = total = reported = 0.0
        segment = 0

        async def commit() -> None:
            nonlocal speech, silence, reported, segment
            if speech <= 0:
                return
            text = f"Heard {speech:.1f} seconds."
            start = total - speech - silence
            words = [
                {"text": t, "type": "word", "start": start + i * 0.3, "end": start + i * 0.3 + 0.25, "logprob": -0.1,
                 "speaker_id": f"speaker_{2 * segment}" if diarize else None}
                for i, t in enumerate(text.split())
            ]  # fmt: skip
            await ws.send(json.dumps({"message_type": "committed_transcript", "text": text}))
            if timestamps:
                await ws.send(json.dumps({"message_type": "committed_transcript_with_timestamps", "text": text,
                                          "language_code": query.get("language_code") or "eng", "words": words}))  # fmt: skip
            if diarize:
                if segment:
                    await ws.send(json.dumps({"message_type": "speakers_merged", "from_speaker_id": f"speaker_{2 * segment}",
                                              "to_speaker_id": f"speaker_{segment}"}))  # fmt: skip
                await ws.send(json.dumps({"message_type": "diarized_transcript", "segment_index": segment, "text": text,
                                          "words": words}))  # fmt: skip
            speech = silence = reported = 0.0
            segment += 1

        async for message in ws:
            doc = json.loads(message)
            messages.append({k: v for k, v in doc.items() if k != "audio_base_64"})
            if doc.get("message_type") != "input_audio_chunk":
                await ws.send(json.dumps({"message_type": "input_error", "error": "unknown message type"}))
                continue
            x = np.frombuffer(base64.b64decode(doc["audio_base_64"]), dtype="<i2")
            secs = len(x) / int(doc.get("sample_rate") or SR)
            total += secs
            if len(x) and np.abs(x).max() > 300:
                speech += secs
                silence = 0.0
            elif speech > 0:
                silence += secs
            if speech - reported >= 0.5:
                reported = speech
                await ws.send(json.dumps({"message_type": "partial_transcript", "text": f"hearing {speech:.1f}"}))
            if doc.get("commit") or (silence >= 0.4 and query.get("commit_strategy") in ("vad", "turn_prediction")):
                await commit()

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
