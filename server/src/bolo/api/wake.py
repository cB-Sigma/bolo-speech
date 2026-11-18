"""Wake words and keyword spotting: ``POST /api/v1/wake`` (a recording) and ``WS /api/v1/wake/stream`` (live).
docs/api.md, "Wake words"."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import numpy as np
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from bolo.api.common import check_auth, get_core, read_upload
from bolo.api.ws import _fail
from bolo.audio.resample import StreamResampler, resample
from bolo.core import Bolo
from bolo.errors import BoloError, InvalidRequest
from bolo.keepalive import request_keep_alive
from bolo.obs.trace import Trace
from bolo.wake import SR, WakeModel, detect, parse_keywords, wake_body

router = APIRouter()
#: the stream checks its key itself (a WebSocket has no request dependencies)
ws_router = APIRouter()


def _threshold(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    try:
        value = float(str(raw))
    except ValueError as exc:
        raise InvalidRequest("threshold must be a number between 0 and 1", param="threshold") from exc
    if not 0.0 < value < 1.0:
        raise InvalidRequest("threshold must be between 0 and 1", param="threshold")
    return value


def _model(core: Bolo, model_id: str, model: Any) -> WakeModel:
    if not isinstance(model, WakeModel):
        raise InvalidRequest(f"'{model_id}' is not a wake-word model", param="model")
    if not model.id:
        model.id = model_id
    return model


@router.post("/api/v1/wake", tags=["native"], summary="Wake words and keywords in a recording")
async def wake(request: Request) -> JSONResponse:
    """Extension. Which wake words or keywords a recording holds, and when each was heard (seconds from the start,
    at the end of the phrase). Multipart form: ``file``, ``model`` (a ``wake-word`` model; default
    ``default_wake_word``), ``keywords`` (phrases to look for, for the models that take them: the sherpa-onnx
    keyword spotters; several fields or one with commas; default: the model's own phrases), ``threshold`` (0 to 1;
    lower hears more and wakes falsely more) and ``keep_alive``."""
    core = get_core(request)
    form = await request.form()
    file = form.get("file")
    if file is None or isinstance(file, str):
        raise InvalidRequest("a multipart 'file' field with the audio is required", param="file")
    request_keep_alive(form.get("keep_alive"), ("wake-word",))
    spec = core.resolve(str(form.get("model") or "") or None, "wake-word")
    keywords = parse_keywords(form.getlist("keywords") or None)
    threshold = _threshold(form.get("threshold"))
    samples, rate = await read_upload(core, file)
    if rate != SR:
        samples = resample(samples, rate, SR)
    trace = Trace("wake-word", "http", spec.id)
    trace.audio_seconds = len(samples) / SR
    try:
        async with core.manager.use(spec.id) as entry:
            model = _model(core, spec.id, entry.model)
            if keywords and not model.custom_keywords:
                raise InvalidRequest(
                    f"'{spec.id}' hears {', '.join(model.default_keywords)} only; for phrases of your own use a "
                    "keyword spotter (kws-zipformer-gigaspeech, kws-zipformer-zh-en)",
                    param="keywords",
                )
            with trace.phase("wake"):
                found = await asyncio.to_thread(detect, model, samples, keywords, threshold)
            body = wake_body(spec.id, model, keywords or model.default_keywords, found, len(samples) / SR)
        trace.extra["wake"] = [d.keyword for d in found]
        trace.finish("ok")
    except Exception as exc:
        trace.finish("error", exc.message if isinstance(exc, BoloError) else f"{type(exc).__name__}: {exc}")
        raise
    finally:
        core.traces.record(trace)
    return JSONResponse(body, headers={"x-request-id": trace.id, "x-bolo-model": spec.id})


@ws_router.websocket("/api/v1/wake/stream")
async def wake_stream(ws: WebSocket) -> None:
    """Live wake-word detection. Query: ``model``, ``keywords`` (comma-separated), ``threshold``, ``sample_rate``
    (16000) and ``encoding`` (``pcm_s16le`` or ``pcm_f32le``). The client sends binary audio; the server sends
    ``{"type": "ready", "model", "keywords"}``, then ``{"type": "wake", "keyword", "time", "score"}`` for each
    detection. JSON ``{"type": "reset"}`` starts listening afresh, ``{"type": "stop"}`` ends the stream."""
    core: Bolo = ws.app.state.core
    await ws.accept()
    try:
        check_auth(core, ws, scope="stt")
        q = ws.query_params
        spec = core.resolve(q.get("model") or None, "wake-word")
        keywords = parse_keywords(q.get("keywords"))
        threshold = _threshold(q.get("threshold"))
        encoding = (q.get("encoding") or "pcm_s16le").lower()
        if encoding not in ("pcm_s16le", "pcm", "pcm_f32le"):
            raise InvalidRequest("encoding must be pcm_s16le or pcm_f32le", param="encoding")
        try:
            rate = int(q.get("sample_rate") or SR)
        except ValueError as exc:
            raise InvalidRequest("sample_rate must be an integer", param="sample_rate") from exc
        if not 8000 <= rate <= 48000:
            raise InvalidRequest("sample_rate must be between 8000 and 48000", param="sample_rate")
        async with core.manager.use(spec.id) as entry:
            model = _model(core, spec.id, entry.model)
            if keywords and not model.custom_keywords:
                raise InvalidRequest(f"'{spec.id}' takes no keywords of your own", param="keywords")
            detector = await asyncio.to_thread(model.create_detector, keywords, threshold)
            resampler = StreamResampler(rate, SR) if rate != SR else None
            await ws.send_text(
                json.dumps({"type": "ready", "model": spec.id, "keywords": detector.keywords}, ensure_ascii=False)
            )
            try:
                while True:
                    message = await ws.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    data = message.get("bytes")
                    if data is not None:
                        if encoding == "pcm_f32le":
                            samples = np.frombuffer(data[: len(data) // 4 * 4], dtype="<f4").astype(np.float32)
                        else:
                            samples = (
                                np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
                            )
                        if resampler is not None:
                            samples = resampler.process(samples)
                        for hit in await asyncio.to_thread(detector.accept, samples):
                            await ws.send_text(json.dumps({"type": "wake", **hit.to_dict()}, ensure_ascii=False))
                        continue
                    text = message.get("text")
                    if not text:
                        continue
                    try:
                        kind = json.loads(text).get("type")
                    except (json.JSONDecodeError, AttributeError) as exc:
                        raise InvalidRequest("text frames must be JSON objects with a 'type'") from exc
                    if kind == "reset":
                        detector.close()
                        detector = await asyncio.to_thread(model.create_detector, keywords, threshold)
                        await ws.send_text(json.dumps({"type": "reset"}))
                    elif kind == "stop":
                        for hit in await asyncio.to_thread(detector.finish):
                            await ws.send_text(json.dumps({"type": "wake", **hit.to_dict()}, ensure_ascii=False))
                        break
                    elif kind == "ping":
                        await ws.send_text(json.dumps({"type": "pong"}))
                    else:
                        raise InvalidRequest(f"unknown message type {kind!r}")
            finally:
                detector.close()
        with contextlib.suppress(Exception):
            await ws.close()
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await _fail(ws, exc)


__all__ = ["router", "ws_router"]
