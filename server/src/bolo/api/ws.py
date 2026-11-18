"""WebSocket endpoints: realtime speech-to-text and streaming text-to-speech."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from bolo.api.common import check_auth, parse_bool
from bolo.api.openai import SpeechStream
from bolo.audio.encode import StreamEncoder, normalize_format
from bolo.audio.resample import StreamResampler
from bolo.core import Bolo
from bolo.denoise import parse_denoise
from bolo.errors import BoloError, InvalidRequest
from bolo.keepalive import request_keep_alive
from bolo.obs.trace import Trace
from bolo.streaming.session import ClientDisconnected, SessionConfig, STTSession
from bolo.text.romanised import parse_mode as parse_romanised
from bolo.text.segment import IncrementalSegmenter

router = APIRouter()

# Trailing silence (ms) that ends an utterance. Shorter answers sooner but splits speakers who pause.
ENDPOINT_PRESETS = {"fast": 250.0, "balanced": 500.0, "patient": 900.0}


async def _fail(ws: WebSocket, exc: BaseException) -> None:
    """Report an error as a JSON message, then close.

    Request errors (bad parameters, unknown model, …) close normally so
    clients can read the message; policy violations use 1008 and bugs 1011.
    """
    if isinstance(exc, BoloError):
        payload = {"type": "error", "code": exc.code, "message": exc.message}
        code = 1008 if exc.status in (401, 403) else 1000
    else:
        payload = {"type": "error", "code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}
        code = 1011
    with contextlib.suppress(Exception):
        await ws.send_text(json.dumps(payload))
    with contextlib.suppress(Exception):
        await ws.close(code=code)


def _stt_config(params: Any, base: SessionConfig | None = None) -> SessionConfig:
    cfg = base or SessionConfig()
    get = params.get
    encoding = (get("encoding") or cfg.encoding).lower()
    encoding = {"ulaw": "mulaw", "pcmu": "mulaw", "pcma": "alaw", "pcm": "pcm_s16le", "linear16": "pcm_s16le"}.get(
        encoding, encoding
    )
    default_rate = 8000 if encoding in ("mulaw", "alaw") and base is None else cfg.sample_rate
    try:
        rate = int(get("sample_rate") or default_rate)
    except (TypeError, ValueError) as exc:
        raise InvalidRequest("sample_rate must be an integer", param="sample_rate") from exc
    endpoint_ms = cfg.endpoint_ms
    preset = (get("endpointing") or "").lower()
    if preset:
        if preset not in ENDPOINT_PRESETS:
            raise InvalidRequest(f"endpointing must be one of {', '.join(ENDPOINT_PRESETS)}", param="endpointing")
        endpoint_ms = ENDPOINT_PRESETS[preset]
    if get("endpoint_ms"):
        try:
            endpoint_ms = float(get("endpoint_ms"))
        except (TypeError, ValueError) as exc:
            raise InvalidRequest("endpoint_ms must be a number", param="endpoint_ms") from exc
        if not 100 <= endpoint_ms <= 5000:
            raise InvalidRequest("endpoint_ms must be between 100 and 5000", param="endpoint_ms")
    try:
        max_speakers = int(get("max_speakers")) if get("max_speakers") else cfg.max_speakers
    except (TypeError, ValueError) as exc:
        raise InvalidRequest("max_speakers must be an integer", param="max_speakers") from exc
    denoise = cfg.denoise
    if get("denoise") is not None or get("denoise_strength") is not None:
        denoise = parse_denoise(get("denoise"), strength=get("denoise_strength")) or cfg.denoise
    return SessionConfig(
        denoise=denoise,
        diarize=parse_bool(get("diarize"), cfg.diarize),
        max_speakers=max_speakers,
        endpoint_ms=endpoint_ms,
        model=get("model") or cfg.model,
        language=get("language") or cfg.language,
        sample_rate=rate,
        encoding=encoding,
        interim=parse_bool(get("interim"), cfg.interim),
        vad=parse_bool(get("vad"), cfg.vad),
        transport="websocket",
        word_timestamps=parse_bool(get("words"), cfg.word_timestamps),
    )


async def _with_keep_alive(ws: WebSocket, tasks: tuple[str, ...], handler: Any) -> None:
    """A session's ``?keep_alive=`` (bolo extension, Ollama style): applied to the models of ``tasks`` the session
    used when it ends, so ``0`` unloads them then and not between utterances."""
    core: Bolo = ws.app.state.core
    try:
        ka = request_keep_alive(ws.query_params.get("keep_alive"), tasks, session=True)
    except InvalidRequest as exc:
        await ws.accept()
        await _fail(ws, exc)
        return
    try:
        await handler(ws)
    finally:
        if ka is not None:
            with contextlib.suppress(Exception):
                await core.manager.apply_keep_alive(ka.used, ka.seconds)


@router.websocket("/api/v1/stt/stream")
async def stt_stream(ws: WebSocket) -> None:
    await _with_keep_alive(ws, ("stt",), _stt_stream)


@router.websocket("/api/v1/tts/stream")
async def tts_stream(ws: WebSocket) -> None:
    await _with_keep_alive(ws, ("tts",), _tts_stream)


async def _stt_stream(ws: WebSocket) -> None:
    """Realtime transcription.

    Query: ``model``, ``language``, ``sample_rate`` (16000), ``encoding``
    (pcm_s16le | pcm_f32le | mulaw | alaw), ``interim`` (true), ``vad`` (true), ``denoise`` (true, false, a strength
    or a denoise model id; ``denoise_strength``), ``keep_alive``.
    Client sends binary audio frames and JSON ``config`` / ``flush`` / ``stop``.
    """
    core: Bolo = ws.app.state.core
    await ws.accept()
    send_lock = asyncio.Lock()

    async def send(msg: dict[str, Any]) -> None:
        async with send_lock:
            try:
                await ws.send_text(json.dumps(msg, ensure_ascii=False))
            except (WebSocketDisconnect, RuntimeError) as exc:  # RuntimeError: sending after the close
                raise ClientDisconnected(str(exc)) from exc

    session: STTSession | None = None
    try:
        check_auth(core, ws)
        cfg = _stt_config(ws.query_params)
        session = STTSession(core, send, cfg)
        await session.start()
        got_audio = False
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                await session.close()
                return
            data = message.get("bytes")
            if data is not None:
                if session.done:
                    break
                got_audio = True
                session.feed(data)
                continue
            text = message.get("text")
            if not text:
                continue
            try:
                control = json.loads(text)
                kind = control.get("type")
            except (json.JSONDecodeError, AttributeError) as exc:
                raise InvalidRequest("text frames must be JSON objects with a 'type'") from exc
            if kind == "config":
                if got_audio:
                    raise InvalidRequest("'config' must be sent before any audio")
                await session.close()
                cfg = _stt_config(control, cfg)
                session = STTSession(core, send, cfg)
                await session.start()
            elif kind == "flush":
                session.flush()
            elif kind == "stop":
                await session.stop()
                break
            elif kind == "ping":
                await send({"type": "pong"})
            else:
                raise InvalidRequest(f"unknown message type {kind!r}")
        with contextlib.suppress(Exception):
            await ws.close()
    except (WebSocketDisconnect, ClientDisconnected):
        if session:
            await session.close()
    except Exception as exc:
        if session:
            await session.close()
        await _fail(ws, exc)


_WS_TTS_FORMATS = {"pcm_s16le": "pcm", "pcm": "pcm", "pcm_f32le": "pcm_f32le"}


async def _tts_stream(ws: WebSocket) -> None:
    """Streaming synthesis.

    Query: ``model``, ``voice``, ``format`` (pcm_s16le | pcm_f32le | mulaw |
    alaw | wav | mp3 | opus | flac | aac), ``sample_rate``, ``speed``,
    ``phonemizer`` (auto | uccharan | uccharan:improved | espeak-ng; bolo extension),
    ``wordlists`` (comma-separated word-list names; empty applies none) and ``normalize``
    (uccharan | light | model | none), ``keep_alive``, ``instructions`` (direction for the delivery, for models
    that follow it), ``watermark`` (on | off | a watermark model; unset: the server's policy).
    Client sends JSON ``{"type":"text","text":…}`` (incrementally),
    ``flush`` and ``stop``; the server answers with binary audio frames and
    JSON ``ready`` / ``chunk`` / ``metrics`` / ``done`` / ``error``.
    """
    core: Bolo = ws.app.state.core
    await ws.accept()
    params = ws.query_params
    pieces: asyncio.Queue[tuple[str, float] | None] = asyncio.Queue()
    worker: asyncio.Task[None] | None = None
    try:
        check_auth(core, ws)
        spec = core.resolve(params.get("model"), "tts")
        fmt_raw = (params.get("format") or "pcm_s16le").lower()
        fmt = _WS_TTS_FORMATS.get(fmt_raw) or normalize_format(fmt_raw)
        try:
            requested_rate = int(params["sample_rate"]) if params.get("sample_rate") else None
            speed = float(params.get("speed") or 1.0)
        except ValueError as exc:
            raise InvalidRequest("sample_rate and speed must be numbers") from exc
        voice = params.get("voice") or None
        phonemizer = params.get("phonemizer") or None
        language = params.get("language") or None
        wordlists = [n for n in params["wordlists"].split(",") if n.strip()] if "wordlists" in params else None
        raw_norm = (params.get("normalize") or "").lower()
        normalize: bool | str = (
            True if raw_norm in ("", "true", "1") else False if raw_norm in ("false", "0", "off") else raw_norm
        )
        romanised = parse_romanised(params.get("romanised") or params.get("romanized"))
        instructions = params.get("instructions") or None
        watermark = params.get("watermark") or None
        async with core.manager.use(spec.id) as entry:
            model_rate = entry.model.sample_rate
            core.active_streams += 1
            try:
                f32 = fmt == "pcm_f32le"
                encoder = None if f32 else StreamEncoder(fmt, model_rate, requested_rate)
                out_rate = (requested_rate or model_rate) if f32 else encoder.sample_rate  # type: ignore[union-attr]
                f32_resampler = StreamResampler(model_rate, out_rate) if f32 else None
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "ready",
                            "model": spec.id,
                            "voice": voice,
                            "format": fmt_raw,
                            "sample_rate": out_rate,
                            "request_id": f"ws_{id(ws):x}",
                        }
                    )
                )
                state = {"seq": 0, "t": 0.0}

                async def run_pieces() -> None:
                    while True:
                        item = await pieces.get()
                        if item is None:
                            return
                        text, received = item
                        trace = Trace("tts", "websocket", spec.id)
                        stream = SpeechStream(
                            core,
                            trace,
                            encoder,
                            text=text,
                            model=spec.id,
                            voice=voice,
                            speed=speed,
                            language=language,
                            phonemizer=phonemizer,
                            wordlists=wordlists,
                            normalize=normalize,
                            romanised=romanised,
                            instructions=instructions,
                            watermark=watermark,
                        )
                        first = True
                        try:
                            async for chunk in stream.chunks():
                                if f32_resampler is not None:
                                    data = np.asarray(f32_resampler.process(chunk.samples), dtype="<f4").tobytes()
                                else:
                                    data = stream.encode(chunk)
                                dur = len(chunk.samples) / chunk.sample_rate
                                msg: dict[str, Any] = {
                                    "type": "chunk",
                                    "seq": state["seq"],
                                    "t0": round(state["t"], 3),
                                    "dur": round(dur, 3),
                                    "ttfa_ms": None,
                                    "text": chunk.text,
                                }
                                if first:
                                    ttfa = (time.perf_counter() - received) * 1000.0
                                    trace.mark_ttfa(trace.elapsed_ms())
                                    msg["ttfa_ms"] = round(ttfa, 1)
                                    first = False
                                await ws.send_text(json.dumps(msg, ensure_ascii=False))
                                if data:
                                    await ws.send_bytes(data)
                                state["seq"] += 1
                                state["t"] += dur
                        finally:
                            await stream.aclose()
                        await ws.send_text(
                            json.dumps(
                                {
                                    "type": "metrics",
                                    "rtf": round(trace.rtf, 4) if trace.rtf is not None else None,
                                    "audio_seconds": round(trace.audio_seconds or 0.0, 3),
                                    "ttfa_ms": round(trace.ttfa_ms, 1) if trace.ttfa_ms is not None else None,
                                    "request_id": trace.id,
                                }
                            )
                        )

                worker = asyncio.create_task(run_pieces())
                segmenter = IncrementalSegmenter()
                stopping = False
                while not stopping:
                    recv = asyncio.create_task(ws.receive())
                    done, _ = await asyncio.wait({recv, worker}, return_when=asyncio.FIRST_COMPLETED)
                    if worker in done:
                        recv.cancel()
                        worker.result()  # re-raise synthesis errors
                        break
                    message = recv.result()
                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect()
                    text = message.get("text")
                    if not text:
                        continue
                    try:
                        control = json.loads(text)
                        kind = control.get("type")
                    except (json.JSONDecodeError, AttributeError) as exc:
                        raise InvalidRequest("frames must be JSON objects with a 'type'") from exc
                    now = time.perf_counter()
                    if kind == "text":
                        for sentence in segmenter.push(str(control.get("text", ""))):
                            pieces.put_nowait((sentence, now))
                        if control.get("flush"):
                            for sentence in segmenter.flush():
                                pieces.put_nowait((sentence, now))
                    elif kind == "flush":
                        for sentence in segmenter.flush():
                            pieces.put_nowait((sentence, now))
                    elif kind == "stop":
                        for sentence in segmenter.flush():
                            pieces.put_nowait((sentence, now))
                        stopping = True
                    elif kind == "ping":
                        await ws.send_text('{"type":"pong"}')
                    else:
                        raise InvalidRequest(f"unknown message type {kind!r}")
                pieces.put_nowait(None)
                await worker
                if encoder is not None:
                    tail = encoder.finish()
                elif f32_resampler is not None:
                    tail = np.asarray(
                        f32_resampler.process(np.zeros(0, dtype=np.float32), last=True), dtype="<f4"
                    ).tobytes()
                else:
                    tail = b""
                if tail:
                    await ws.send_bytes(tail)
                await ws.send_text('{"type":"done"}')
            finally:
                core.active_streams -= 1
        with contextlib.suppress(Exception):
            await ws.close()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await _fail(ws, exc)
    finally:
        if worker is not None and not worker.done():
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await worker
