"""Audio analysis routes: speech enhancement on its own (``/v1/audio/denoise``, with ElevenLabs' voice-isolation
route as an alias).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from bolo.api.common import get_core, parse_bool, read_upload
from bolo.audio.encode import FORMATS, StreamEncoder, encode_audio, normalize_format
from bolo.core import Bolo
from bolo.denoise import LiveDenoiser, enhance, mix
from bolo.engines.base import DenoiseModel, FloatArray
from bolo.errors import BoloError, InvalidRequest
from bolo.keepalive import request_keep_alive
from bolo.obs.trace import Trace

router = APIRouter()

#: seconds of input per piece when a denoised recording is streamed back
STREAM_PIECE_S = 0.5


def _upload(form: Any, *names: str) -> Any:
    for name in names:
        file = form.get(name)
        if file is not None and not isinstance(file, str):
            return file
    raise InvalidRequest(f"a multipart '{names[0]}' field with the audio is required", param=names[0])


def _strength(raw: Any) -> float:
    if raw in (None, ""):
        return 1.0
    try:
        value = float(str(raw))
    except ValueError as exc:
        raise InvalidRequest("strength must be a number between 0 and 1", param="strength") from exc
    if not 0.0 < value <= 1.0:
        raise InvalidRequest("strength must be above 0 and at most 1", param="strength")
    return value


def _rate(raw: Any) -> int | None:
    if raw in (None, ""):
        return None
    try:
        value = int(str(raw))
    except ValueError as exc:
        raise InvalidRequest("sample_rate must be a whole number of hertz", param="sample_rate") from exc
    if not 8000 <= value <= 48000:
        raise InvalidRequest("sample_rate must be between 8000 and 48000", param="sample_rate")
    return value


def _denoise_header(model: str, strength: float) -> str:
    return model if strength >= 1.0 else f"{model}; strength={strength:g}"


async def _pieces(
    core: Bolo, samples: FloatArray, rate: int, model_id: str, strength: float, trace: Trace
) -> AsyncGenerator[FloatArray, None]:
    """The enhanced recording piece by piece: models with a live form run on the stream as it is read (the first
    audio comes back after the first piece, not the whole recording); the others enhance the whole recording first."""
    async with core.manager.use(model_id) as entry:
        model = entry.model
        assert isinstance(model, DenoiseModel)
        trace.extra["denoise"] = {"model": model_id, "strength": strength, "sample_rate": int(model.sample_rate)}
        if model.live:
            live = await asyncio.to_thread(LiveDenoiser, model, rate, strength)
            step = max(1, int(STREAM_PIECE_S * rate))
            for i in range(0, len(samples), step):
                out = await asyncio.to_thread(live.process, samples[i : i + step])
                if len(out):
                    yield out
            tail = await asyncio.to_thread(live.finish)
            trace.extra["denoise"]["live"] = live.summary()
            if len(tail):
                yield tail
            return
    enhanced, _ = await enhance(core, samples, rate, model_id, trace)
    out = mix(samples, enhanced, strength)
    step = max(1, int(STREAM_PIECE_S * rate))
    for i in range(0, len(out), step):
        yield out[i : i + step]


@router.post("/v1/audio/denoise", tags=["native"], summary="Remove background noise from a recording")
async def denoise_audio(request: Request) -> Response:
    """Extension. Speech enhancement on its own: the recording comes back with the background noise removed and
    the voice kept (voice isolation). Multipart form:

    * ``file``: the audio (any format bolo decodes);
    * ``model``: any ``denoise`` model of the catalog (``gtcrn-simple``, ``dpdfnet2``, ``deepfilternet3``,
      ``rnnoise``, ``demucs-dns48``, ``elevenlabs-voice-isolator``, ...); default ``default_denoise``;
    * ``strength``: 0 to 1 (default 1): the enhanced audio mixed with the original at that weight;
    * ``response_format``: ``wav`` (default), ``mp3``, ``flac``, ``opus``, ``aac``, ``pcm`` (16-bit), ``mulaw``,
      ``alaw``; ``sample_rate``: the output's rate (default the input's);
    * ``stream``: ``true`` sends the audio back as it is made (chunked). Models with a live form (GTCRN, DPDFNet,
      DeepFilterNet 3, RNNoise) enhance the recording piece by piece as it is sent; the others enhance it whole
      first. A streamed WAV has no length in its header;
    * ``keep_alive``.

    The model runs at its own sample rate (16 or 48 kHz); the audio is resampled to it and back. Headers:
    ``x-bolo-denoise`` (model and strength), ``x-bolo-model``, ``x-sample-rate``. A model that cannot run is an
    error here (transcriptions skip it instead)."""
    core = get_core(request)
    form = await request.form()
    file = _upload(form, "file", "audio")
    fmt = normalize_format(str(form.get("response_format") or form.get("output_format") or "wav"))
    if fmt not in FORMATS:
        raise InvalidRequest(f"response_format must be one of {', '.join(FORMATS)}", param="response_format")
    strength = _strength(form.get("strength"))
    out_rate = _rate(form.get("sample_rate"))
    stream = parse_bool(form.get("stream"))
    request_keep_alive(form.get("keep_alive"), ("denoise",))
    spec = core.resolve(str(form.get("model") or "") or None, "denoise")
    samples, rate = await read_upload(core, file)
    return await _denoise_response(core, samples, rate, spec.id, strength, fmt, out_rate, stream)


@router.post("/v1/audio-isolation", tags=["native"], summary="Voice isolation (ElevenLabs' route)")
async def audio_isolation(request: Request) -> Response:
    """ElevenLabs' voice-isolation route: multipart ``audio``; the voice comes back as MP3 (``output_format``, or
    the fields of ``POST /v1/audio/denoise``, choose otherwise). The denoiser is ``model`` or ``default_denoise``."""
    return await _isolation(request, stream=False)


@router.post("/v1/audio-isolation/stream", tags=["native"], summary="Voice isolation, streamed (ElevenLabs' route)")
async def audio_isolation_stream(request: Request) -> Response:
    """As ``POST /v1/audio-isolation``, the audio sent back as it is made."""
    return await _isolation(request, stream=True)


async def _isolation(request: Request, stream: bool) -> Response:
    core = get_core(request)
    form = await request.form()
    file = _upload(form, "audio", "file")
    raw_fmt = str(form.get("response_format") or form.get("output_format") or "mp3")
    fmt = normalize_format(raw_fmt.split("_", maxsplit=1)[0])  # ElevenLabs' "mp3_44100_128" names the codec first
    if fmt not in FORMATS:
        raise InvalidRequest(f"output_format must be one of {', '.join(FORMATS)}", param="output_format")
    spec = core.resolve(str(form.get("model") or "") or None, "denoise")
    samples, rate = await read_upload(core, file)
    return await _denoise_response(core, samples, rate, spec.id, _strength(form.get("strength")), fmt, None, stream)


async def _denoise_response(
    core: Bolo,
    samples: FloatArray,
    rate: int,
    model_id: str,
    strength: float,
    fmt: str,
    out_rate: int | None,
    stream: bool,
) -> Response:
    trace = Trace("denoise", "http", model_id)
    trace.audio_seconds = len(samples) / rate if rate else 0.0
    headers = {
        "x-request-id": trace.id,
        "x-bolo-model": model_id,
        "x-bolo-denoise": _denoise_header(model_id, strength),
    }
    if not stream:
        try:
            with trace.phase("denoise"):
                enhanced, native = await enhance(core, samples, rate, model_id, trace)
            out = mix(samples, enhanced, strength)
            trace.extra["denoise"] = {"model": model_id, "strength": strength, "sample_rate": native}
            data = await asyncio.to_thread(encode_audio, out, rate, fmt, out_rate)
            trace.finish("ok")
        except Exception as exc:
            trace.finish("error", exc.message if isinstance(exc, BoloError) else f"{type(exc).__name__}: {exc}")
            raise
        finally:
            core.traces.record(trace)
        encoder = StreamEncoder(fmt, rate, out_rate)
        headers["x-sample-rate"] = str(encoder.sample_rate)
        return Response(data, media_type=encoder.media_type, headers=headers)

    encoder = StreamEncoder(fmt, rate, out_rate)
    headers["x-sample-rate"] = str(encoder.sample_rate)
    pieces = _pieces(core, samples, rate, model_id, strength, trace)
    try:  # the first piece before the response starts, so that a model that cannot load is an HTTP error
        first: FloatArray | None = await pieces.__anext__()
    except StopAsyncIteration:
        first = None
    except Exception as exc:
        trace.finish("error", exc.message if isinstance(exc, BoloError) else f"{type(exc).__name__}: {exc}")
        core.traces.record(trace)
        raise

    async def body() -> AsyncIterator[bytes]:
        status, error = "ok", None
        try:
            if first is not None:
                trace.mark_ttfa()
                data = await asyncio.to_thread(encoder.feed, first)
                if data:
                    yield data
            async for piece in pieces:
                data = await asyncio.to_thread(encoder.feed, piece)
                if data:
                    yield data
            tail = await asyncio.to_thread(encoder.finish)
            if tail:
                yield tail
        except asyncio.CancelledError:
            status, error = "cancelled", "client disconnected"
            raise
        except Exception as exc:
            status = "error"
            error = exc.message if isinstance(exc, BoloError) else f"{type(exc).__name__}: {exc}"
            raise
        finally:
            with contextlib.suppress(Exception):
                await pieces.aclose()
            trace.finish(status, error)
            core.traces.record(trace)

    return StreamingResponse(body(), media_type=encoder.media_type, headers=headers)


__all__ = ["router"]
