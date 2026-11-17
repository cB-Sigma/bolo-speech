"""Routes that mirror `speaches` so its clients keep working unchanged.

The shapes were derived from the speaches route definitions (MIT licence,
see NOTICE); no code was copied. Covered: model management
(``/v1/models/{id}`` POST/DELETE, ``/v1/registry``, ``/v1/audio/models``,
``/v1/audio/voices``), loaded-model control (``/api/ps``), ``/health`` and
voice-activity timestamps (``/v1/audio/speech/timestamps``) and speaker
embeddings (``/v1/audio/speech/embedding``).
``/v1/chat/completions`` audio and ``/v1/realtime`` live in their own modules;
diarization has its own route (``/api/v1/diarize``).
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from bolo import similarity
from bolo.api.common import get_core, read_upload
from bolo.api.openai import _model_object
from bolo.audio.resample import resample
from bolo.engines.base import VadParams
from bolo.errors import InvalidRequest, ModelNotFound
from bolo.obs.trace import Trace
from bolo.streaming.vad import detect_segments

router = APIRouter()

TASK_NAMES = {
    "automatic-speech-recognition": "stt",
    "text-to-speech": "tts",
    "voice-activity-detection": "vad",
    "speaker-diarization": "diarization",
}


@router.get("/v1/registry", tags=["speaches"], summary="List every model in the catalog")
async def registry(request: Request, task: str | None = None) -> JSONResponse:
    core = get_core(request)
    internal = TASK_NAMES.get(task or "", task)
    return JSONResponse({"object": "list", "data": [_model_object(m) for m in core.catalog_api(task=internal)]})


@router.get("/v1/audio/models", tags=["speaches"], summary="List installed text-to-speech models")
async def audio_models(request: Request) -> JSONResponse:
    core = get_core(request)
    return JSONResponse(
        {"object": "list", "models": [_model_object(m) for m in core.catalog_api(task="tts", installed=True)]}
    )


@router.get("/v1/audio/voices", tags=["speaches"], summary="List voices of installed text-to-speech models")
async def audio_voices(request: Request, model: str | None = None) -> JSONResponse:
    """The voices of the installed text-to-speech models; ``model`` (an id or alias): that model's voices only.
    ``blendable`` marks the voices that can be mixed into a blend (``af_bella(0.7)+af_sky(0.3)`` as ``voice``);
    saved blends are listed with their model, with ``blend`` set."""
    core = get_core(request)
    voices: list[dict[str, Any]] = []
    wanted: str | None = None
    if model:
        spec = core.catalog.resolve(model)
        if spec is None or spec.task != "tts":
            raise ModelNotFound(f"no text-to-speech model '{model}'", param="model")
        wanted = spec.id
    blends = await asyncio.to_thread(core.voices.blends)
    for m in core.catalog_api(task="tts", installed=None if wanted else True):
        if wanted is not None and m["id"] != wanted:
            continue
        blendable = "voice-blending" in m["features"]
        for v in _model_object(m)["voices"]:
            voices.append({**v, "model_id": m["id"], "blendable": blendable})
        # saved blends of this model's voices (bolo.voices), spoken by naming them like any voice
        for b in blends:
            if b.default_model == m["id"]:
                voices.append(
                    {"name": b.name, "id": b.id, "language": b.language or "", "gender": "", "model_id": m["id"],
                     "blendable": False, "blend": b.blend}
                )  # fmt: skip
    return JSONResponse({"object": "list", "voices": voices})


@router.post("/v1/models/{model_id:path}", tags=["speaches"], summary="Download a model (blocks until done)")
async def download_model(request: Request, model_id: str) -> Response:
    core = get_core(request)
    spec = core.manager.resolve(model_id)
    if core.manager.is_installed(spec):
        return PlainTextResponse(f"Model '{model_id}' already exists", status_code=201)
    job = await core.pull_and_wait(spec.id)
    if job.state != "done":
        status = 401 if spec.gated else 502
        return JSONResponse(
            {
                "detail": job.error,
                "error": {"message": job.error, "type": "server_error", "code": "download_failed", "param": None},
            },
            status_code=status,
        )
    return PlainTextResponse(f"Model '{model_id}' downloaded", status_code=200)


@router.delete("/v1/models/{model_id:path}", tags=["speaches"], summary="Delete an installed model")
async def delete_model(request: Request, model_id: str) -> JSONResponse:
    core = get_core(request)
    spec = core.catalog.resolve(model_id)
    if spec is None or not await core.remove(spec.id):
        raise ModelNotFound(f"Model '{model_id}' is not installed")
    return JSONResponse({"detail": f"Model '{model_id}' deleted"})


@router.get("/api/ps", tags=["speaches"], summary="List loaded models")
async def ps(request: Request) -> JSONResponse:
    return JSONResponse({"models": list(get_core(request).manager.loaded)})


@router.post("/api/ps/{model_id:path}", tags=["speaches"], summary="Load a model into memory")
async def ps_load(request: Request, model_id: str) -> JSONResponse:
    core = get_core(request)
    spec = core.catalog.resolve(model_id)
    if spec is None:
        return JSONResponse({"message": f"Model '{model_id}' not supported."}, status_code=404)
    if spec.id in core.manager.loaded:
        return JSONResponse({"message": f"Model '{model_id}' is already loaded."}, status_code=409)
    await core.manager.load(spec.id)
    return JSONResponse({"message": f"Model '{model_id}' loaded."}, status_code=201)


@router.delete("/api/ps/{model_id:path}", tags=["speaches"], summary="Unload a model from memory")
async def ps_unload(request: Request, model_id: str) -> JSONResponse:
    core = get_core(request)
    spec = core.catalog.resolve(model_id)
    if spec is None or spec.id not in core.manager.loaded:
        return JSONResponse({"message": f"Model {model_id} is not loaded."}, status_code=404)
    if core.manager.loaded[spec.id].active:
        return JSONResponse({"message": f"Model {model_id} is in use."}, status_code=409)
    await core.manager.unload(spec.id)
    return JSONResponse({"message": f"Model {model_id} unloaded."})


@router.post("/v1/audio/speech/timestamps", tags=["speaches"], summary="Voice-activity timestamps (milliseconds)")
async def speech_timestamps(request: Request) -> JSONResponse:
    """Multipart form: ``file`` and optional ``model``, ``threshold``, ``min_silence_duration_ms``,
    ``min_speech_duration_ms``, ``max_speech_duration_s``, ``speech_pad_ms``.
    Returns ``[{"start": ms, "end": ms}, …]``."""
    core = get_core(request)
    form = await request.form()
    file = form.get("file")
    if file is None or isinstance(file, str):
        raise InvalidRequest("a multipart 'file' field with the audio is required", param="file")

    def num(name: str, default: float) -> float:
        raw = form.get(name)
        try:
            return float(str(raw)) if raw not in (None, "") else default
        except ValueError as exc:
            raise InvalidRequest(f"{name} must be a number", param=name) from exc

    max_speech = num("max_speech_duration_s", 3600.0)
    params = VadParams(
        threshold=num("threshold", 0.5),
        min_silence_s=num("min_silence_duration_ms", 1000.0) / 1000.0,
        min_speech_s=max(num("min_speech_duration_ms", 0.0) / 1000.0, 0.03),
        max_speech_s=max_speech if max_speech != float("inf") else 3600.0,
        speech_pad_s=num("speech_pad_ms", 0.0) / 1000.0,
    )
    model = str(form.get("model") or "") or None
    if model and core.catalog.resolve(model) is None:
        model = None  # speaches clients send its fixed silero id
    samples, rate = await read_upload(core, file)
    trace = Trace("vad", "http", model or core.settings.default_vad)
    trace.audio_seconds = len(samples) / rate
    try:
        with trace.phase("resample"):
            samples16 = resample(samples, rate, 16000)
        detector, name = await core.vad_detector(params, model)
        trace.model = name
        with trace.phase("vad"):
            segments = await asyncio.to_thread(detect_segments, detector, samples16)
        trace.finish("ok")
    except Exception as exc:
        trace.finish("error", str(exc))
        raise
    finally:
        core.traces.record(trace)
    return JSONResponse([{"start": round(a / 16), "end": round(b / 16)} for a, b in segments])


@router.post("/v1/audio/speech/embedding", tags=["speaches"], summary="Speaker embedding of a recording")
async def speech_embedding(request: Request) -> JSONResponse:
    """Multipart form: ``file`` and optional ``model``. Returns the speaker embedding in the shape of OpenAI's
    embeddings response: ``{"object": "list", "data": [{"object": "embedding", "embedding": [...], "index": 0}],
    "model": ..., "usage": {"prompt_tokens": n, "total_tokens": n}}`` (``n``: samples at 16 kHz, as speaches counts).

    The embedder is the speaker-embedding model of a diarization pipeline, the one that scores cloned voices
    (setting ``voice_similarity_model``, else ``default_diarization``). A ``model`` that is not in bolo's catalog
    (speaches' ``pyannote/wespeaker-voxceleb-resnet34-LM``, for one) uses that default. The vector is unit length;
    the cosine of two of them is the speaker similarity. Clips longer than 30 s use their first 30 s."""
    core = get_core(request)
    form = await request.form()
    file = form.get("file")
    if file is None or isinstance(file, str):
        raise InvalidRequest("a multipart 'file' field with the audio is required", param="file")
    model = str(form.get("model") or "") or None
    if model and core.catalog.resolve(model) is None:
        model = None  # speaches clients send its own embedding model's id
    name = model or similarity.embedding_model(core)
    spec = core.resolve(name, "diarization")
    samples, rate = await read_upload(core, file)
    trace = Trace("embedding", "http", spec.id)
    trace.audio_seconds = len(samples) / rate if rate else 0.0
    try:
        with trace.phase("embed"):
            vector = await similarity.embed(core, samples, rate, spec.id)
        trace.finish("ok")
    except Exception as exc:
        trace.finish("error", str(exc))
        raise
    finally:
        core.traces.record(trace)
    used = min(
        len(samples) * similarity.EMBEDDING_RATE // max(rate, 1),
        int(similarity.MAX_EMBED_S * similarity.EMBEDDING_RATE),
    )
    return JSONResponse(
        {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [float(v) for v in vector], "index": 0}],
            "model": spec.id,
            "usage": {"prompt_tokens": used, "total_tokens": used},
        }
    )
