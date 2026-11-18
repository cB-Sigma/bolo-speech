"""Enrolled speakers: ``/v1/audio/speakers`` (enrol, list, change, delete), ``/verify`` and ``/identify``
(bolo.speakers)."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from bolo import speakers
from bolo.api.common import get_core, read_upload
from bolo.core import Bolo
from bolo.errors import InvalidRequest
from bolo.keepalive import request_keep_alive
from bolo.obs.trace import Trace
from bolo.voices import now_iso

router = APIRouter()


def _files(form: Any) -> list[Any]:
    files = [f for key in ("file", "file[]", "files", "files[]") for f in form.getlist(key) if not isinstance(f, str)]
    return files


def _threshold(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    try:
        value = float(str(raw))
    except ValueError as exc:
        raise InvalidRequest("threshold must be a number between -1 and 1", param="threshold") from exc
    if not -1.0 <= value <= 1.0:
        raise InvalidRequest("threshold must be between -1 and 1", param="threshold")
    return value


def _metadata(raw: Any) -> dict[str, Any] | None:
    if raw in (None, ""):
        return None
    try:
        value = json.loads(str(raw))
    except ValueError as exc:
        raise InvalidRequest("metadata must be a JSON object", param="metadata") from exc
    if not isinstance(value, dict):
        raise InvalidRequest("metadata must be a JSON object", param="metadata")
    return value


def _model(core: Bolo, form: Any) -> str:
    return speakers.embedder_id(core, str(form.get("model") or "") or None)


async def _with_voiceprint(core: Bolo, speaker: speakers.Speaker, model: str) -> dict[str, Any]:
    await speakers.voiceprint(core, core.speakers.get(speaker.id), model)
    return core.speakers.get(speaker.id).to_dict()


@router.post("/v1/audio/speakers", tags=["native"], summary="Enrol a speaker")
async def enrol_speaker(request: Request) -> JSONResponse:
    """Extension. Multipart form: ``name`` (required), one or more ``file`` fields (recordings of that person alone,
    1 to 30 s each; longer ones keep their first 30 s), optional ``id`` (lowercase letters, digits, ``-``, ``_``;
    default: from the name), ``metadata`` (a JSON object kept with the speaker) and ``model`` (a speaker embedder:
    its voiceprint is computed now, so that the first verification is fast; any embedder works later)."""
    core = get_core(request)
    form = await request.form()
    files = _files(form)
    if not files:
        raise InvalidRequest("at least one multipart 'file' with a recording of the speaker is required", param="file")
    if len(files) > speakers.MAX_CLIPS:
        raise InvalidRequest(f"a speaker has at most {speakers.MAX_CLIPS} recordings", param="file")
    model = _model(core, form) if form.get("model") else None
    clips = [speakers.prepare_clip(*(await read_upload(core, f))) for f in files]
    now = now_iso()
    speaker = core.speakers.create(
        str(form.get("name") or ""), str(form.get("id") or "") or None, _metadata(form.get("metadata")), now
    )
    try:
        for clip in clips:
            core.speakers.add_clip(speaker, clip, now)
        body = await _with_voiceprint(core, speaker, model) if model else core.speakers.get(speaker.id).to_dict()
    except Exception:
        core.speakers.delete(speaker.id)
        raise
    return JSONResponse(body, status_code=201)


@router.get("/v1/audio/speakers", tags=["native"], summary="List enrolled speakers")
async def list_speakers(request: Request) -> JSONResponse:
    core = get_core(request)
    model = speakers.embedder_id(core)
    threshold, calibrated = speakers.threshold_for(model)
    return JSONResponse(
        {
            "object": "list",
            "data": [s.to_dict() for s in core.speakers.all()],
            "embedding_model": model,
            "threshold": threshold,
            "calibrated": calibrated,
            "thresholds": speakers.THRESHOLDS,
        }
    )


async def _probe(request: Request, kind: str) -> tuple[Bolo, Any, Any, str, Trace]:
    core = get_core(request)
    form = await request.form()
    file = form.get("file")
    if file is None or isinstance(file, str):
        raise InvalidRequest("a multipart 'file' field with the audio is required", param="file")
    request_keep_alive(form.get("keep_alive"), ("speaker-embedding", "diarization"))
    model = _model(core, form)
    trace = Trace(kind, "http", model)
    return core, form, file, model, trace


@router.post("/v1/audio/speakers/verify", tags=["native"], summary="Is this recording the enrolled speaker?")
async def verify_speaker(request: Request) -> JSONResponse:
    """Extension. Multipart form: ``file``, ``speaker`` (an enrolled speaker's id or name), optional ``model`` (a
    speaker embedder; default ``voice_similarity_model``, else the default diarization
    model's embedder) and ``threshold`` (default: the embedder's measured threshold). ``match`` is true when the
    cosine similarity of the recording and the speaker's voiceprint is at or above the threshold."""
    core, form, file, model, trace = await _probe(request, "speaker-verify")
    ref = str(form.get("speaker") or "")
    if not ref:
        raise InvalidRequest("'speaker' (an enrolled speaker's id or name) is required", param="speaker")
    speaker = core.speakers.find(ref)
    threshold = _threshold(form.get("threshold"))
    samples, rate = await read_upload(core, file)
    trace.audio_seconds = len(samples) / rate if rate else 0.0
    try:
        with trace.phase("embed"):
            body = await speakers.verify(core, samples, rate, speaker, model, threshold)
        trace.extra["speaker"] = {"id": speaker.id, "similarity": body["similarity"], "match": body["match"]}
        trace.finish("ok")
    except Exception as exc:
        trace.finish("error", str(exc))
        raise
    finally:
        core.traces.record(trace)
    return JSONResponse(body, headers={"x-request-id": trace.id})


@router.post("/v1/audio/speakers/identify", tags=["native"], summary="Which enrolled speaker is this?")
async def identify_speaker(request: Request) -> JSONResponse:
    """Extension. Multipart form: ``file``, optional ``speakers`` (ids or names, comma-separated or repeated: only
    these are compared; default every enrolled speaker), ``model``, ``threshold`` and ``top_k`` (candidates listed,
    default 5). ``speaker`` is the closest enrolled speaker when their similarity reaches the threshold, else
    null (someone not enrolled)."""
    core, form, file, model, trace = await _probe(request, "speaker-identify")
    names = [n.strip() for raw in form.getlist("speakers") for n in str(raw).split(",") if n.strip()]
    candidates = [core.speakers.find(n) for n in names] if names else core.speakers.all()
    threshold = _threshold(form.get("threshold"))
    try:
        top_k = int(str(form.get("top_k") or 5))
    except ValueError as exc:
        raise InvalidRequest("top_k must be a whole number", param="top_k") from exc
    if not 1 <= top_k <= 100:
        raise InvalidRequest("top_k must be between 1 and 100", param="top_k")
    samples, rate = await read_upload(core, file)
    trace.audio_seconds = len(samples) / rate if rate else 0.0
    try:
        with trace.phase("embed"):
            body = await speakers.identify(core, samples, rate, candidates, model, threshold, top_k)
        best = body["speaker"]
        trace.extra["speaker"] = {"id": best["id"] if best else None, "similarity": body["similarity"]}
        trace.finish("ok")
    except Exception as exc:
        trace.finish("error", str(exc))
        raise
    finally:
        core.traces.record(trace)
    return JSONResponse(body, headers={"x-request-id": trace.id})


@router.get("/v1/audio/speakers/{speaker_id}", tags=["native"], summary="One enrolled speaker")
async def get_speaker(request: Request, speaker_id: str) -> JSONResponse:
    return JSONResponse(get_core(request).speakers.find(speaker_id).to_dict())


@router.post("/v1/audio/speakers/{speaker_id}", tags=["native"], summary="Rename a speaker or add recordings")
async def change_speaker(request: Request, speaker_id: str) -> JSONResponse:
    """Multipart form (or JSON): ``name``, ``metadata`` (replaces it), and more ``file`` recordings to add. Adding or
    removing a recording recomputes the speaker's voiceprints the next time they are used."""
    core = get_core(request)
    speaker = core.speakers.find(speaker_id)
    now = now_iso()
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
        except ValueError as exc:
            raise InvalidRequest("the request body must be JSON") from exc
        if not isinstance(body, dict):
            raise InvalidRequest("the request body must be a JSON object")
        changes = {k: body[k] for k in ("name", "metadata") if k in body}
        return JSONResponse(core.speakers.update(speaker.id, changes, now).to_dict())
    form = await request.form()
    changes = {}
    if form.get("name") is not None:
        changes["name"] = form.get("name")
    if form.get("metadata") is not None:
        changes["metadata"] = _metadata(form.get("metadata")) or {}
    clips = [speakers.prepare_clip(*(await read_upload(core, f))) for f in _files(form)]
    if len(speaker.clips()) + len(clips) > speakers.MAX_CLIPS:
        raise InvalidRequest(f"a speaker has at most {speakers.MAX_CLIPS} recordings", param="file")
    if changes:
        core.speakers.update(speaker.id, changes, now)
    for clip in clips:
        core.speakers.add_clip(speaker, clip, now)
    return JSONResponse(core.speakers.get(speaker.id).to_dict())


@router.delete("/v1/audio/speakers/{speaker_id}", tags=["native"], summary="Delete an enrolled speaker")
async def delete_speaker(request: Request, speaker_id: str) -> JSONResponse:
    core = get_core(request)
    speaker = core.speakers.find(speaker_id)
    core.speakers.delete(speaker.id)
    return JSONResponse({"id": speaker.id, "object": "speaker.deleted", "deleted": True})


@router.get("/v1/audio/speakers/{speaker_id}/clips/{clip_id}", tags=["native"], summary="One enrolment recording (WAV)")
async def speaker_clip(request: Request, speaker_id: str, clip_id: str) -> Response:
    core = get_core(request)
    speaker = core.speakers.find(speaker_id)
    return Response(core.speakers.clip_audio(speaker, clip_id), media_type="audio/wav")


@router.delete(
    "/v1/audio/speakers/{speaker_id}/clips/{clip_id}", tags=["native"], summary="Remove an enrolment recording"
)
async def delete_speaker_clip(request: Request, speaker_id: str, clip_id: str) -> JSONResponse:
    core = get_core(request)
    speaker = core.speakers.find(speaker_id)
    return JSONResponse(core.speakers.remove_clip(speaker.id, clip_id, now_iso()).to_dict())


__all__ = ["router"]
