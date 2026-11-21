"""Pipelines (docs/pipelines.md): definitions at ``/api/v1/pipelines``, runs at ``/v1/pipelines/{id}/run``."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from bolo.api.common import get_core, parse_bool
from bolo.audio import decode_audio
from bolo.core import Bolo
from bolo.errors import BoloError, InvalidRequest
from bolo.mcp_tools import LOOPBACK
from bolo.pipelines import Audio, PipelineInvalid, parse_text, validate
from bolo.pipelines.definition import STEP_TYPES
from bolo.pipelines.runner import media_type
from bolo.pipelines.templates import all_templates

router = APIRouter()
TAG = "pipelines"


def remote_urls_allowed(core: Bolo) -> bool:
    """``pipeline_remote_urls``; unset: on a loopback bind, or when requests need an API key."""
    value = core.settings.pipeline_remote_urls
    if value is not None:
        return bool(value)
    return bool(core.settings.api_key or core.keys.all()) or core.settings.host in LOOPBACK


def _invalid(exc: PipelineInvalid) -> JSONResponse:
    body = {
        "error": {"message": exc.message, "type": exc.type, "param": exc.param, "code": "pipeline_invalid"},
        "detail": exc.message,
        "validation": exc.report.to_dict(),
    }
    return JSONResponse(body, status_code=400)


async def _body(request: Request) -> Any:
    raw = await request.body()
    if not raw.strip():
        raise InvalidRequest("send the pipeline as JSON or YAML in the body")
    return parse_text(raw, request.headers.get("content-type", ""))


@router.get("/api/v1/pipelines", tags=[TAG], summary="List pipelines")
async def list_pipelines(request: Request) -> JSONResponse:
    core = get_core(request)
    return JSONResponse({"object": "list", "data": core.pipelines.store.all()})


@router.post("/api/v1/pipelines", status_code=201, tags=[TAG], summary="Create a pipeline (JSON or YAML)")
async def create_pipeline(request: Request) -> Response:
    """The body is a pipeline definition (docs/pipelines.md), as JSON or YAML (``Content-Type: application/yaml``).
    The definition is validated first: a 400 lists every problem (``validation.errors``). 409 when the id is taken."""
    core = get_core(request)
    try:
        doc = core.pipelines.create(await _body(request))
    except PipelineInvalid as exc:
        return _invalid(exc)
    return JSONResponse(doc, status_code=201)


@router.post("/api/v1/pipelines/validate", tags=[TAG], summary="Validate a pipeline without saving it")
async def validate_pipeline(request: Request) -> JSONResponse:
    """Always 200 for a readable body: ``valid``, ``errors`` and ``warnings`` (each with the ``path`` of the field),
    and ``steps``: each step's resolved model, inputs (with the defaults they take) and outputs."""
    core = get_core(request)
    return JSONResponse(validate(await _body(request), core).to_dict())


@router.get("/api/v1/pipelines/step-types", tags=[TAG], summary="Step types with their inputs, options and outputs")
async def step_types(request: Request) -> JSONResponse:
    return JSONResponse({"object": "list", "data": [t.to_dict() for t in STEP_TYPES.values()]})


@router.get("/api/v1/pipelines/templates", tags=[TAG], summary="Ready-made pipelines")
async def templates(request: Request) -> JSONResponse:
    return JSONResponse({"object": "list", "data": all_templates()})


@router.get("/api/v1/pipelines/runs", tags=[TAG], summary="Pipeline run history, newest first")
async def list_runs(request: Request, pipeline: str | None = None, limit: int = 50) -> JSONResponse:
    core = get_core(request)
    limit = max(1, min(500, limit))
    active = [r.to_dict() for r in core.pipelines.active.values() if not pipeline or r.pipeline == pipeline]
    ids = {r["id"] for r in active}
    stored = [r for r in core.pipelines.store.runs(pipeline, limit) if r["id"] not in ids]
    runs = sorted(active, key=lambda r: r["created_at"], reverse=True) + stored
    return JSONResponse({"object": "list", "data": runs[:limit]})


@router.get("/api/v1/pipelines/runs/{run_id}", tags=[TAG], summary="One pipeline run")
async def get_run(request: Request, run_id: str) -> JSONResponse:
    return JSONResponse(get_core(request).pipelines.get_run(run_id))


@router.post("/api/v1/pipelines/runs/{run_id}/cancel", tags=[TAG], summary="Stop a queued or running pipeline run")
async def cancel_run(request: Request, run_id: str) -> JSONResponse:
    return JSONResponse(await get_core(request).pipelines.cancel(run_id))


@router.get("/api/v1/pipelines/runs/{run_id}/files/{name}", tags=[TAG], summary="A file a pipeline run wrote")
async def run_file(request: Request, run_id: str, name: str) -> Response:
    path = get_core(request).pipelines.store.file_path(run_id, name)
    return FileResponse(path, media_type=media_type(name), filename=name)


@router.get("/api/v1/pipelines/{pipeline_id}", tags=[TAG], summary="One pipeline (?version= for an older one)")
async def get_pipeline(request: Request, pipeline_id: str, version: int | None = None) -> JSONResponse:
    return JSONResponse(get_core(request).pipelines.definition(pipeline_id, version))


@router.get("/api/v1/pipelines/{pipeline_id}/versions", tags=[TAG], summary="A pipeline's versions")
async def get_versions(request: Request, pipeline_id: str) -> JSONResponse:
    core = get_core(request)
    versions = core.pipelines.store.versions(pipeline_id)
    if not versions:
        core.pipelines.definition(pipeline_id)  # 404
    data = []
    for v in versions:
        doc = core.pipelines.definition(pipeline_id, v)
        data.append({"version": v, "updated_at": doc.get("updated_at"), "steps": len(doc.get("steps") or [])})
    return JSONResponse({"object": "list", "data": data})


@router.put("/api/v1/pipelines/{pipeline_id}", tags=[TAG], summary="Save a new version of a pipeline")
async def update_pipeline(request: Request, pipeline_id: str) -> Response:
    core = get_core(request)
    try:
        doc = core.pipelines.update(pipeline_id, await _body(request))
    except PipelineInvalid as exc:
        return _invalid(exc)
    return JSONResponse(doc)


@router.delete("/api/v1/pipelines/{pipeline_id}", status_code=204, tags=[TAG], summary="Delete a pipeline")
async def delete_pipeline(request: Request, pipeline_id: str) -> Response:
    get_core(request).pipelines.store.delete(pipeline_id)
    return Response(status_code=204)


# -- running --------------------------------------------------------------------------------------------------
async def _fetch(core: Bolo, url: str) -> tuple[bytes, str]:
    if not url.startswith(("http://", "https://")):
        raise InvalidRequest("url must be an http:// or https:// address", param="url")
    if not remote_urls_allowed(core):
        raise InvalidRequest(
            "this server does not fetch recordings from URLs (pipeline_remote_urls; by default allowed on a loopback "
            "address or when requests need an API key)",
            param="url",
        )
    limit = core.settings.max_upload_mb * 1024 * 1024
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0), follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise InvalidRequest(f"the url answered HTTP {resp.status_code}", param="url")
                data = bytearray()
                async for chunk in resp.aiter_bytes():
                    data += chunk
                    if len(data) > limit:
                        raise InvalidRequest(f"the recording is larger than the {core.settings.max_upload_mb} MB limit",
                                             param="url")  # fmt: skip
                return bytes(data), resp.headers.get("content-type", "")
    except httpx.HTTPError as exc:
        raise InvalidRequest(f"the url could not be fetched: {type(exc).__name__}: {exc}", param="url") from exc


async def _run(request: Request, definition: dict[str, Any] | None, pipeline_id: str | None) -> Response:
    core = get_core(request)
    form = await request.form()
    if definition is None:
        raw = form.get("pipeline")
        if raw is None:
            raise InvalidRequest("send the pipeline to run in the 'pipeline' field (JSON or YAML)", param="pipeline")
        source = raw if isinstance(raw, str) else (await raw.read()).decode("utf-8")
        definition = parse_text(source)
        if not isinstance(definition, dict):
            raise InvalidRequest("a pipeline is an object with steps", param="pipeline")
    variables: dict[str, Any] = {}
    if form.get("vars"):
        try:
            variables = json.loads(str(form.get("vars")))
        except ValueError as exc:
            raise InvalidRequest("vars must be a JSON object", param="vars") from exc
        if not isinstance(variables, dict):
            raise InvalidRequest("vars must be a JSON object", param="vars")
    run_async = parse_bool(form.get("async"))
    callback = str(form.get("callback_url") or "").strip() or None
    if callback is not None:
        if not run_async:
            raise InvalidRequest("callback_url needs async=true", param="callback_url")
        if not callback.startswith(("http://", "https://")):
            raise InvalidRequest("callback_url must be an http:// or https:// address", param="callback_url")
        if not remote_urls_allowed(core):
            raise InvalidRequest("this server does not call back URLs a request names (pipeline_remote_urls)",
                                 param="callback_url")  # fmt: skip
    text = str(form.get("text")) if form.get("text") not in (None, "") else None
    file = form.get("file")
    filename: str | None = None
    data: bytes | None = None
    content_type = ""
    if file is not None and not isinstance(file, str):
        data = await file.read()
        if len(data) > core.settings.max_upload_mb * 1024 * 1024:
            raise InvalidRequest(f"the upload is larger than the {core.settings.max_upload_mb} MB limit", param="file")
        filename, content_type = file.filename, str(file.content_type or "")
    elif form.get("url"):
        url = str(form.get("url"))
        data, content_type = await _fetch(core, url)
        filename = url.rsplit("/", 1)[-1].split("?", maxsplit=1)[0] or None
    try:
        run = core.pipelines.runner.prepare(definition, text=text, filename=filename, variables=variables)
    except PipelineInvalid as exc:
        return _invalid(exc)
    if data is not None:
        samples, rate = await asyncio.to_thread(decode_audio, data, filename=filename, content_type=content_type)
        run.inputs["audio"] = Audio(samples, rate)
        run.inputs["duration"] = round(len(samples) / rate, 3) if rate else 0.0
    if run.inputs.get("audio") is None and text is None:
        raise InvalidRequest("give the run a recording (file or url) or a text", param="file")
    if run_async:
        core.pipelines.start(run, callback)
        return JSONResponse(run.to_dict(), status_code=202)
    try:
        await core.pipelines.run(run)
    except BoloError:
        raise
    return JSONResponse(run.to_dict())


@router.post("/v1/pipelines/run", tags=[TAG], summary="Run a pipeline sent with the request (not saved)")
async def run_adhoc(request: Request) -> Response:
    """Multipart: ``pipeline`` (the definition, JSON or YAML) and the fields of ``POST /v1/pipelines/{id}/run``."""
    return await _run(request, None, None)


@router.post("/v1/pipelines/{pipeline_id}/run", tags=[TAG], summary="Run a saved pipeline")
async def run_pipeline(request: Request, pipeline_id: str, version: int | None = None) -> Response:
    """Multipart: ``file`` (the recording) or ``url`` (fetched by the server), ``text`` (``input.text``), ``vars``
    (a JSON object), ``version`` (an older version), ``async`` (``true``: answer 202 at once with the queued run;
    poll ``GET /api/v1/pipelines/runs/{run_id}``) and ``callback_url`` (with ``async``: the finished run is POSTed
    there, signed with ``X-Bolo-Signature``). The answer is the run: each step's model, status, time in ms and
    outputs, and the pipeline's ``outputs``. A step that fails stops the run (``status: error``) and keeps every
    output made before it."""
    core = get_core(request)
    form = await request.form()
    wanted = version
    if form.get("version") not in (None, ""):
        try:
            wanted = int(str(form.get("version")))
        except ValueError as exc:
            raise InvalidRequest("version must be a whole number", param="version") from exc
    definition = core.pipelines.definition(pipeline_id, wanted)
    return await _run(request, definition, pipeline_id)


__all__ = ["router"]
