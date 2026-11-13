"""The native JSON API under ``/api/v1`` (used by the web console)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from bolo import samples
from bolo.api.common import get_core, parse_bool, sse
from bolo.bench import run_bench
from bolo.errors import BoloError, InvalidRequest, ModelNotFound
from bolo.obs.metrics import render_metrics

router = APIRouter(prefix="/api/v1")
root_router = APIRouter()


@router.get("/info", tags=["native"], summary="Server, engine and platform information")
async def info(request: Request) -> JSONResponse:
    return JSONResponse(get_core(request).info())


@router.get("/catalog", tags=["native"], summary="Model catalog with install and load state")
async def catalog(
    request: Request, task: str | None = None, q: str | None = None, installed: str | None = None
) -> JSONResponse:
    core = get_core(request)
    flag = parse_bool(installed) if installed not in (None, "") else None
    return JSONResponse({"models": core.catalog_api(task or None, q, flag)})


class PullRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(
        description="Catalog id or alias, `hf:<repo>` for any compatible Hugging Face model, or `piper:<voice>` "
        "(for example `piper:te_IN-venkatesh-medium`) for any voice of the Piper project."
    )
    kind: str | None = Field(default=None, description="Only for `hf:` pulls: loader kind when it cannot be detected.")
    task: str | None = None


@router.post("/models/pull", status_code=202, tags=["native"], summary="Start downloading a model")
async def pull(request: Request, body: PullRequest) -> JSONResponse:
    core = get_core(request)
    name = body.id
    if name.startswith("hf:"):
        from bolo.custom import register_hf_model

        spec = await asyncio.to_thread(register_hf_model, core, name[3:], body.kind, body.task)
        name = spec.id
    elif name.startswith("piper:"):
        from bolo.piper import register_piper_voice

        name = register_piper_voice(core, name[6:]).id
    elif name.startswith("file:"):
        raise InvalidRequest(
            "local directories are registered from the command line (bolo pull ./dir), not over HTTP", param="id"
        )
    job = core.start_pull(name)
    return JSONResponse({"job_id": job.id}, status_code=202)


@router.get("/jobs/{job_id}", tags=["native"], summary="Poll a background job")
async def job(request: Request, job_id: str) -> JSONResponse:
    j = get_core(request).jobs.get(job_id)
    if j is None:
        raise ModelNotFound(f"unknown job '{job_id}'")
    return JSONResponse(j.to_dict())


@router.get("/jobs", tags=["native"], summary="List background jobs")
async def jobs(request: Request) -> JSONResponse:
    return JSONResponse({"jobs": [j.to_dict() for j in reversed(get_core(request).jobs.all())]})


class ModelSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phonemizer: str | None = Field(
        default=None,
        description="Pin the phonemiser of a voice: auto, uccharan, uccharan:improved or espeak-ng; null removes the pin.",
    )


@router.get("/models/{model_id:path}/settings", tags=["native"], summary="Settings pinned for one model")
async def get_model_settings(request: Request, model_id: str) -> JSONResponse:
    core = get_core(request)
    spec = core.manager.resolve(model_id)
    return JSONResponse({"model": spec.id, "settings": core.store.model_settings(spec.id)})


@router.put("/models/{model_id:path}/settings", tags=["native"], summary="Pin settings for one model (phonemiser)")
async def put_model_settings(request: Request, model_id: str, body: ModelSettingsRequest) -> JSONResponse:
    core = get_core(request)
    spec = core.manager.resolve(model_id)
    values = body.model_dump(exclude_unset=True)
    if values.get("phonemizer"):
        core.components.validate_pin(spec, values["phonemizer"])
    settings = core.store.set_model_settings(spec.id, values)
    core.events.publish("model", {"id": spec.id, "event": "settings", "settings": settings})
    return JSONResponse({"model": spec.id, "settings": settings})


@router.post("/models/{model_id:path}/load", tags=["native"], summary="Load a model into memory")
async def load(request: Request, model_id: str) -> JSONResponse:
    core = get_core(request)
    core.require_components(core.manager.resolve(model_id))
    entry = await core.manager.load(model_id)
    return JSONResponse(
        {"id": entry.id, "loaded": True, "memory_bytes": entry.memory_bytes, "load_ms": round(entry.load_ms, 1)}
    )


@router.post("/models/{model_id:path}/unload", tags=["native"], summary="Unload a model from memory")
async def unload(request: Request, model_id: str) -> JSONResponse:
    core = get_core(request)
    spec = core.manager.resolve(model_id)
    await core.manager.unload(spec.id)
    return JSONResponse({"id": spec.id, "loaded": False})


@router.delete("/models/{model_id:path}", status_code=204, tags=["native"], summary="Delete an installed model")
async def delete(request: Request, model_id: str) -> Response:
    core = get_core(request)
    spec = core.manager.resolve(model_id)
    if spec.builtin:
        raise InvalidRequest(f"model '{spec.id}' is built in and cannot be deleted")
    await core.remove(spec.id)
    return Response(status_code=204)


@router.get("/models/{model_id:path}", tags=["native"], summary="One catalog entry")
async def show(request: Request, model_id: str) -> JSONResponse:
    core = get_core(request)
    return JSONResponse(core.model_api(core.manager.resolve(model_id)))


@router.post("/diarize", tags=["native"], summary="Speaker diarization: who spoke when")
async def diarize(request: Request) -> JSONResponse:
    """Multipart form: ``file`` and optional ``model``, ``num_speakers``, ``min_speakers``,
    ``max_speakers``, ``threshold`` (clustering distance; lower finds more speakers)."""
    from bolo.api.common import read_upload
    from bolo.api.openai import _int_field, known_speakers
    from bolo.diarize import DiarizeOptions, find_overlaps
    from bolo.obs.trace import Trace

    core = get_core(request)
    form = await request.form()
    file = form.get("file")
    if file is None or isinstance(file, str):
        raise InvalidRequest("a multipart 'file' field with the audio is required", param="file")
    try:
        threshold = float(str(form.get("threshold"))) if form.get("threshold") else None
        identify = float(str(form.get("identify_threshold"))) if form.get("identify_threshold") else None
    except ValueError as exc:
        raise InvalidRequest("threshold and identify_threshold must be numbers", param="threshold") from exc
    opts = DiarizeOptions(
        model=str(form.get("model") or "") or None,
        num_speakers=_int_field(form, "num_speakers"),
        min_speakers=_int_field(form, "min_speakers"),
        max_speakers=_int_field(form, "max_speakers"),
        threshold=threshold,
        known=await known_speakers(form),
        identify_threshold=identify,
    )
    spec = core.resolve(opts.model, "diarization")
    samples, rate = await read_upload(core, file)
    trace = Trace("diarization", "http", spec.id)
    turns = await core.diarize(samples, rate, opts, trace)
    speakers = sorted({t.speaker for t in turns})
    return JSONResponse(
        {
            "model": spec.id,
            "duration": round(len(samples) / rate, 3),
            "speakers": [opts.label(s) for s in speakers],
            "segments": [
                {"start": round(t.start, 3), "end": round(t.end, 3), "speaker": opts.label(t.speaker)} for t in turns
            ],
            # bolo extension: stretches where two or more speakers talk at once (segments above may overlap)
            "overlaps": [o.to_dict(opts.label) for o in find_overlaps(turns)],
            **(
                {
                    "known_speakers": [
                        {
                            "name": k.name,
                            "matched": k.name in opts.names.values(),
                            "similarity": opts.similarities.get(k.name),
                        }
                        for k in opts.known
                    ]
                }
                if opts.known
                else {}
            ),
            "request_id": trace.id,
        },
        headers={"x-request-id": trace.id},
    )


@router.get("/components", tags=["native"], summary="Optional components (uccharan, espeak-ng, engine plugins)")
async def components(request: Request) -> JSONResponse:
    return JSONResponse({"components": get_core(request).components.status()})


class InstallRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    accept_license: bool = False


@router.post(
    "/components/{component_id}/install", status_code=202, tags=["native"], summary="Install an optional component"
)
async def install_component(request: Request, component_id: str, body: InstallRequest) -> JSONResponse:
    """Installs the component into the running environment and activates it without a restart.
    Requires ``{"accept_license": true}``: eSpeak NG is GPL-3.0-or-later."""
    job = get_core(request).components.start_install(component_id, body.accept_license)
    return JSONResponse({"job_id": job.id}, status_code=202)


@router.get("/stats", tags=["native"], summary="Live server statistics")
async def stats(request: Request) -> JSONResponse:
    return JSONResponse(get_core(request).stats())


@router.get("/requests", tags=["native"], summary="Recent request traces, newest first")
async def requests(
    request: Request, limit: int = 100, kind: str | None = None, model: str | None = None
) -> JSONResponse:
    core = get_core(request)
    traces = core.traces.recent(max(1, min(limit, 1000)), kind or None, model or None)
    return JSONResponse({"requests": [t.to_dict() for t in traces]})


@router.get("/requests/{request_id}", tags=["native"], summary="One request trace")
async def request_trace(request: Request, request_id: str) -> JSONResponse:
    t = get_core(request).traces.get(request_id)
    if t is None:
        raise ModelNotFound(f"unknown request '{request_id}' (traces are kept in memory and rotate)")
    return JSONResponse(t.to_dict())


@router.get("/events", tags=["native"], summary="Server-Sent Events: stats, request, request_progress, job, model, log")
async def events(request: Request) -> StreamingResponse:
    core = get_core(request)

    async def gen() -> AsyncIterator[bytes]:
        yield b": connected\n\n"
        yield sse({"type": "stats", **core.stats()})
        sub = core.events.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(sub.__anext__(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield b": keep-alive\n\n"
                    continue
                yield sse(_flatten(event))
        finally:
            with contextlib.suppress(Exception):
                await sub.aclose()

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


def _flatten(event: dict[str, Any]) -> dict[str, Any]:
    """`request` and `job` events carry their object's fields at the top level."""
    for key in ("request", "job"):
        if event.get("type") == key and isinstance(event.get(key), dict):
            return {"type": key, **event[key]}
    return event


class BenchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: str = "stt"
    model: str | None = None
    concurrency: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    duration_s: float = 10.0
    sample: str | None = None
    mode: str = Field(
        default="batch", description="`batch` (back-to-back requests) or `realtime` (1x paced streams); STT only."
    )
    voice: str | None = None


@router.post("/bench", status_code=202, tags=["native"], summary="Start a concurrency sweep")
async def bench(request: Request, body: BenchRequest) -> JSONResponse:
    core = get_core(request)
    spec = core.resolve(body.model, body.kind if body.kind in ("stt", "tts") else "stt")
    if body.kind not in ("stt", "tts"):
        raise InvalidRequest("kind must be 'stt' or 'tts'", param="kind")
    j = core.jobs.create("bench", spec.id)

    async def run() -> None:
        j.state = "running"
        core.events.publish("job", {"job": j.to_dict()})
        try:
            j.result = await run_bench(
                core, kind=body.kind, model=spec.id, concurrency=body.concurrency, duration_s=body.duration_s,
                sample=body.sample, mode=body.mode, voice=body.voice, job=j,
            )  # fmt: skip
            j.state, j.progress = "done", 1.0
        except BoloError as exc:
            j.state, j.error = "error", exc.message
        except Exception as exc:
            j.state, j.error = "error", f"{type(exc).__name__}: {exc}"
        core.events.publish("job", {"job": j.to_dict()})

    task = asyncio.create_task(run(), name=f"bolo-bench-{j.id}")
    core._tasks.append(task)
    task.add_done_callback(lambda t: core._tasks.remove(t) if t in core._tasks else None)
    return JSONResponse({"job_id": j.id}, status_code=202)


@router.get("/samples", tags=["native"], summary="Built-in audio and text samples")
async def list_samples() -> JSONResponse:
    return JSONResponse({"samples": samples.listing()})


@router.get("/samples/{name}", tags=["native"], summary="Download a built-in audio sample")
async def get_sample(name: str) -> Response:
    return Response(samples.read_file(name), media_type="audio/wav", headers={"Cache-Control": "public, max-age=86400"})


@root_router.get("/metrics", tags=["operations"], summary="Prometheus metrics")
async def metrics(request: Request) -> Response:
    return Response(render_metrics(get_core(request)), media_type="text/plain; version=0.0.4; charset=utf-8")


@root_router.get("/healthz", tags=["operations"], summary="Liveness")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@root_router.get("/health", tags=["operations"], summary="Liveness (speaches-compatible name)")
async def health() -> JSONResponse:
    return JSONResponse({"message": "OK"})


@root_router.get("/readyz", tags=["operations"], summary="Readiness (preloaded models are loaded)")
async def readyz(request: Request) -> JSONResponse:
    core = get_core(request)
    missing = [
        m for m in core.settings.preload if (s := core.catalog.resolve(m)) is None or s.id not in core.manager.loaded
    ]
    if not core.ready or missing:
        return JSONResponse({"status": "starting", "loading": missing}, status_code=503)
    return JSONResponse({"status": "ready"})
