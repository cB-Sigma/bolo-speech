"""Saved voices (the voice library) under ``/api/v1/voices``, and OpenAI's custom-voice routes.

Native routes: create (multipart), list, get, change, delete, the reference audio, a speaker-similarity score
for a model, export and import (.zip). OpenAI-compatible: ``POST /v1/audio/voices`` (``audio_sample``, ``name``,
``consent``), ``DELETE /v1/audio/voices/{id}`` and ``/v1/audio/voice_consents`` for recorded consent statements,
as in the OpenAI API reference (openai-python 3.x: ``client.audio.voices`` and ``client.audio.voice_consents``).
A saved voice is used by naming it in ``/v1/audio/speech``: ``voice="priya"`` or ``voice={"id": "priya"}``.

Saved blends (``POST /api/v1/voices/blends``): a name for a blend of a model's own voices
(``af_bella(0.7)+af_sky(0.3)``), used the same way. They need no recording and no consent, and are listed apart
from the recordings (``blends`` in ``GET /api/v1/voices``).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from bolo import similarity
from bolo.api.common import get_core, parse_bool, read_upload
from bolo.audio import decode_audio
from bolo.audio.encode import encode_audio
from bolo.core import Bolo
from bolo.engines.base import TTSModel, Voice
from bolo.engines.blend import Blend, model_blends, resolve_blend
from bolo.errors import BoloError, InvalidRequest
from bolo.voices import (
    CONSENT,
    MULTI_SPEAKER_SIMILARITY,
    SavedVoice,
    VoiceNotFound,
    clean_reference,
    consistency,
    now_iso,
    parse_consent,
    speaker_windows,
)

router = APIRouter()

_FILE_KEYS = ("file", "files", "files[]", "audio", "audio_sample")


def created_by(request: Request, name: Any = None) -> dict[str, Any]:
    """Who created a voice: whether an API key authorised it, the name the client gave, and its address."""
    core = get_core(request)
    return {
        "auth": "api_key" if core.settings.api_key else "none",
        "name": (str(name).strip()[:100] or None) if name else None,
        "client": request.client.host if request.client else None,
    }


def _files(form: Any) -> list[Any]:
    out = []
    for key in _FILE_KEYS:
        out += [f for f in form.getlist(key) if not isinstance(f, str)]
    return out


async def _analyse_speakers(core: Bolo, voice: SavedVoice) -> None:
    """Warn when the reference sounds like more than one person: windows compared with each other, by the
    speaker-embedding model, only when it is installed (never downloaded for this)."""
    try:
        spec = similarity.embedder(core)
        if not core.manager.is_installed(spec):
            return
        samples, rate = decode_audio(voice.reference_bytes())
        windows = speaker_windows(samples, rate)
        if len(windows) < 2:
            return
        vectors = [await similarity.embed(core, w, 16000) for w in windows]
    except BoloError:
        return
    low = consistency(vectors)
    if low is None:
        return
    warnings = []
    if low < MULTI_SPEAKER_SIMILARITY:
        warnings.append(
            "parts of the recording sound like different people (speaker similarity "
            f"{low:.2f}); use a recording of one person only"
        )
    core.voices.add_warnings(voice.id, warnings, speaker_consistency=round(low, 3))


async def _auto_transcript(core: Bolo, samples: Any, rate: int, language: str | None) -> str | None:
    """The reference transcribed by the default speech-to-text model, when it is installed."""
    try:
        spec = core.resolve(None, "stt")
        if not core.manager.is_installed(spec):
            return None
        result = await core.transcribe(samples, rate, language=language, record=False)
    except BoloError:
        return None
    text = result.text.strip()
    return text or None


async def create_voice(
    core: Bolo,
    request: Request,
    *,
    clips: list[tuple[Any, int]],
    name: str,
    consent: Any,
    voice_id: str | None = None,
    transcript: str | None = None,
    language: str | None = None,
    description: str | None = None,
    default_model: str | None = None,
    by_name: Any = None,
    transcribe: bool | None = None,
    source: str = "upload",
    consent_record: dict[str, Any] | None = None,
) -> SavedVoice:
    basis = parse_consent(consent)
    if default_model:
        spec = core.resolve(default_model, "tts")
        if "voice-cloning" not in spec.features:
            raise InvalidRequest(
                f"model '{spec.id}' cannot clone voices; choose one of: {', '.join(core.cloning_models()) or 'none'}",
                param="default_model",
            )
        default_model = spec.id
    cleaned = await asyncio.to_thread(clean_reference, clips)
    text = (transcript or "").strip() or None
    meta_extra: dict[str, Any] = {}
    if text is None and transcribe is not False:
        text = await _auto_transcript(core, cleaned.samples, cleaned.sample_rate, language)
        if text:
            meta_extra["transcript_source"] = "auto"
    voice = await asyncio.to_thread(
        lambda: core.voices.create(
            name=name, cleaned=cleaned, consent=basis, created_by=created_by(request, by_name), voice_id=voice_id,
            transcript=text, language=language, description=description, default_model=default_model,
            source=source, extra=meta_extra, consent_record=consent_record,
        )
    )  # fmt: skip
    await _analyse_speakers(core, voice)
    core.events.publish("voice", {"id": voice.id, "event": "created"})
    return core.voices.get(voice.id)


async def _score(core: Bolo, voice_id: str, model: str | None, text: str | None, audio: bool) -> dict[str, Any]:
    result = await similarity.score_voice(core, voice_id, model, text, keep_audio=audio)
    samples = result.pop("samples", None)
    rate = result.pop("sample_rate", None)
    stored = {k: v for k, v in result.items() if k not in ("voice", "text")}
    core.voices.record_score(voice_id, result["model"], stored)
    if audio and samples is not None and rate:
        result["audio"] = "data:audio/wav;base64," + base64.b64encode(encode_audio(samples, rate, "wav")).decode()
    return result


# -- native routes -----------------------------------------------------------------------------------------------


@router.get("/api/v1/voices", tags=["voices"], summary="List saved voices")
async def list_voices(request: Request) -> JSONResponse:
    core = get_core(request)
    voices = [v.to_dict() for v in await asyncio.to_thread(core.voices.all)]
    blends = [v.to_dict() for v in await asyncio.to_thread(core.voices.blends)]
    return JSONResponse(
        {
            "object": "list",
            "voices": voices,
            "blends": blends,
            "cloning_models": core.cloning_models(),
            "blending_models": core.blending_models(),
            "consent": CONSENT,
            "similarity": {
                "embedding_model": similarity.embedding_model(core),
                "same_speaker": similarity.SAME_SPEAKER,
                "similar": similarity.SIMILAR,
                "calibration": similarity.CALIBRATION,
            },
        }
    )


@router.post("/api/v1/voices", status_code=201, tags=["voices"], summary="Create a saved voice (multipart)")
async def post_voice(request: Request) -> JSONResponse:
    """Fields: ``file`` (one or more recordings of the same person; ``files[]`` also works), ``name``,
    ``consent`` (required: ``self`` = "this is my voice", ``permission`` = "I have the speaker's permission"),
    optional ``id``, ``transcript`` (else transcribed by the default STT model when installed; ``transcribe=false``
    skips that), ``language``, ``description``, ``default_model``, ``created_by`` (a name to record) and
    ``score_model`` (speak a test sentence with that model and return the speaker similarity; default: the
    voice's ``default_model``). The reference is cleaned (silence trimmed, loudness normalised, 24 kHz, 2 to
    60 s) and checked; ``warnings`` says when it is short, noisy, clipped or sounds like several people."""
    core = get_core(request)
    form = await request.form()
    consent = form.get("consent")
    parse_consent(consent if isinstance(consent, str) else None)  # before reading any audio
    files = _files(form)
    if not files:
        raise InvalidRequest("send the recording as a file part named 'file'", param="file")
    name = form.get("name")
    if not isinstance(name, str) or not name.strip():
        raise InvalidRequest("a voice needs a name", param="name")
    clips = [await read_upload(core, f) for f in files[:10]]

    def text(key: str) -> str | None:
        value = form.get(key)
        return value if isinstance(value, str) and value.strip() else None

    transcribe_field = text("transcribe")
    voice = await create_voice(
        core, request, clips=clips, name=name, consent=consent, voice_id=text("id"), transcript=text("transcript"),
        language=text("language"), description=text("description"), default_model=text("default_model"),
        by_name=text("created_by"), transcribe=None if transcribe_field is None else parse_bool(transcribe_field),
    )  # fmt: skip
    body = voice.to_dict()
    score_model = text("score_model") or voice.default_model
    if score_model:
        try:
            body["score"] = await _score(core, voice.id, score_model, None, parse_bool(text("score_audio")))
        except BoloError as exc:
            body["score"] = {"model": score_model, "error": exc.message}
    return JSONResponse(body, status_code=201)


@router.get("/api/v1/voices/export", tags=["voices"], summary="Export saved voices as a .zip")
async def export_voices(request: Request, ids: str | None = None) -> Response:
    core = get_core(request)
    names = [i.strip() for i in (ids or "").split(",") if i.strip()]
    data = await asyncio.to_thread(core.voices.export_zip, names or None)
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="bolo-voices.zip"'},
    )


@router.post("/api/v1/voices/import", tags=["voices"], summary="Import saved voices from a .zip export")
async def import_voices(request: Request) -> JSONResponse:
    """``file``: an export; ``consent`` (for voices whose metadata has no consent record); ``replace=true``
    overwrites voices of the same id (they are skipped otherwise)."""
    core = get_core(request)
    form = await request.form()
    files = _files(form)
    if not files:
        raise InvalidRequest("send the export as a file part named 'file'", param="file")
    data = await files[0].read()
    consent = form.get("consent")
    added = await asyncio.to_thread(
        lambda: core.voices.import_zip(
            data,
            imported_by=created_by(request, form.get("created_by")),
            consent=consent if isinstance(consent, str) and consent else None,
            replace=parse_bool(form.get("replace")),
        )
    )
    for vid in added:
        core.events.publish("voice", {"id": vid, "event": "created"})
    return JSONResponse({"imported": added})


async def check_blend(core: Bolo, blend: str, model: str | None) -> tuple[str, Blend]:
    """The model a blend is for (``model``, else the default text-to-speech model) and the blend checked against
    its voices: 400 for a model that cannot blend voices or a voice it does not have (naming the near matches)."""
    if not isinstance(blend, str) or not blend.strip():
        raise InvalidRequest("a blend names its voices: blend='af_bella(0.7)+af_sky(0.3)'", param="blend")
    spec = core.resolve(model, "tts")
    if not model_blends(spec.kind, spec.features):
        models = core.blending_models() or core.blending_models(installed_only=False)
        raise InvalidRequest(
            f"model '{spec.id}' cannot blend voices; models that can: {', '.join(models) or 'none'}",
            param="model" if model else "blend",
        )
    voices = [
        Voice(str(v["id"]), str(v.get("name", "")), str(v.get("language", "")), str(v.get("gender", "")),
              int(v.get("sid", i)))
        for i, v in enumerate(spec.voices)
    ]  # fmt: skip
    if not voices:  # a model whose voices come from its files (kokoro-onnx): ask the loaded model
        async with core.manager.use(spec.id) as entry:
            if isinstance(entry.model, TTSModel):
                voices = list(entry.model.voices)
    return spec.id, resolve_blend(blend, voices, spec.id)


class BlendCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="The blend's name; its id is made from it unless `id` is given.")
    blend: str = Field(
        description="Voices of the model joined by '+', with optional weights: af_bella(0.7)+af_sky(0.3)."
    )
    model: str | None = Field(
        default=None, description="A model that blends voices (Kokoro); default: the default TTS model."
    )
    id: str | None = None
    description: str | None = None
    created_by: str | None = None


@router.post("/api/v1/voices/blends", status_code=201, tags=["voices"], summary="Save a blend of a model's voices")
async def post_blend(request: Request, body: BlendCreate) -> JSONResponse:
    """Saves ``blend`` (voices of ``model`` joined by '+', each optionally weighted: ``af_bella(0.7)+af_sky(0.3)``;
    weights are normalised) under ``name``. Use it as ``voice`` in ``/v1/audio/speech``; without ``model`` there,
    the blend's model answers. A blend imitates no one: no recording and no consent statement are needed. The
    entry is marked ``"kind": "blend"`` and listed under ``blends``."""
    core = get_core(request)
    model_id, blend = await check_blend(core, body.blend, body.model)
    voice = await asyncio.to_thread(
        lambda: core.voices.create_blend(
            name=body.name, blend=blend.key, voices=blend.to_dict()["voices"], model=model_id,
            created_by=created_by(request, body.created_by), voice_id=body.id, language=blend.lead.language or None,
            description=body.description,
        )
    )  # fmt: skip
    core.events.publish("voice", {"id": voice.id, "event": "created"})
    return JSONResponse(voice.to_dict(), status_code=201)


@router.get("/api/v1/voices/{voice_id}", tags=["voices"], summary="One saved voice")
async def get_voice(request: Request, voice_id: str) -> JSONResponse:
    return JSONResponse(get_core(request).voices.get(voice_id).to_dict())


class VoicePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    description: str | None = None
    language: str | None = None
    transcript: str | None = None
    default_model: str | None = Field(
        default=None,
        description="A model that clones voices (for a blend: one that blends voices); empty string clears it.",
    )
    blend: str | None = Field(default=None, description="Saved blends only: new voices and weights.")


@router.patch("/api/v1/voices/{voice_id}", tags=["voices"], summary="Change a saved voice's details")
async def patch_voice(request: Request, voice_id: str, body: VoicePatch) -> JSONResponse:
    core = get_core(request)
    changes = body.model_dump(exclude_unset=True)
    current = core.voices.get(voice_id)
    if current.is_blend:
        if "default_model" in changes and not changes["default_model"]:
            raise InvalidRequest("a blend keeps the model it is for", param="default_model")
        if "blend" in changes or changes.get("default_model"):
            model_id, blend = await check_blend(
                core, changes.get("blend") or current.blend, changes.get("default_model") or current.default_model
            )
            changes.update(default_model=model_id, blend=blend.key, voices=blend.to_dict()["voices"])
        voice = core.voices.update(voice_id, changes)
        core.events.publish("voice", {"id": voice.id, "event": "updated"})
        return JSONResponse(voice.to_dict())
    if "blend" in changes:
        raise InvalidRequest(f"'{current.id}' is a recording, not a blend", param="blend")
    if changes.get("default_model"):
        spec = core.resolve(changes["default_model"], "tts")
        if "voice-cloning" not in spec.features:
            raise InvalidRequest(f"model '{spec.id}' cannot clone voices", param="default_model")
        changes["default_model"] = spec.id
    voice = core.voices.update(voice_id, changes)
    core.events.publish("voice", {"id": voice.id, "event": "updated"})
    return JSONResponse(voice.to_dict())


@router.delete("/api/v1/voices/{voice_id}", tags=["voices"], summary="Delete a saved voice")
async def delete_voice(request: Request, voice_id: str) -> JSONResponse:
    core = get_core(request)
    voice = core.voices.get(voice_id)
    core.voices.delete(voice.id)
    core.events.publish("voice", {"id": voice.id, "event": "deleted"})
    return JSONResponse({"id": voice.id, "object": "voice", "deleted": True})


@router.get("/api/v1/voices/{voice_id}/audio", tags=["voices"], summary="The cleaned reference recording (WAV)")
async def voice_audio(request: Request, voice_id: str) -> Response:
    voice = get_core(request).voices.get(voice_id)
    return Response(voice.reference_bytes(), media_type="audio/wav", headers={"Cache-Control": "no-cache"})


@router.get("/api/v1/voices/{voice_id}/score", tags=["voices"], summary="Speaker similarity of a model's clone")
async def voice_score(
    request: Request, voice_id: str, model: str | None = None, text: str | None = None, audio: str | None = None
) -> JSONResponse:
    """Speaks a test sentence (``text``, else one in the voice's language) in this voice with ``model`` (default:
    the voice's default model, else the default TTS model) and compares the speaker of the result with the
    reference: cosine similarity of speaker embeddings (``similarity``), the ``verdict`` from the calibrated
    thresholds, and the synthesis speed. ``audio=true`` also returns the clip as a data URL. The score is kept in
    the voice's ``scores``."""
    core = get_core(request)
    if core.voices.get(voice_id).is_blend:
        raise InvalidRequest(f"'{voice_id}' is a saved blend: it has no recording to compare with", param="voice_id")
    return JSONResponse(await _score(core, voice_id, model, text, parse_bool(audio)))


# -- OpenAI's custom voices -------------------------------------------------------------------------------------


def _consent_dir(core: Bolo) -> Path:
    return Path(core.settings.data_dir) / "voices" / ".consents"


def _consent_object(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": meta["id"], "object": "audio.voice_consent", "created_at": meta["created_at"], "name": meta["name"],
        "language": meta["language"],
    }  # fmt: skip


def _read_consent(core: Bolo, consent_id: str) -> dict[str, Any]:
    if not consent_id.startswith("cons_") or not consent_id[5:].isalnum():
        raise VoiceNotFound(f"no voice consent '{consent_id}'")
    path = _consent_dir(core) / consent_id / "consent.json"
    if not path.is_file():
        raise VoiceNotFound(f"no voice consent '{consent_id}'")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


@router.post("/v1/audio/voice_consents", tags=["openai"], summary="Upload a consent recording")
async def create_voice_consent(request: Request) -> JSONResponse:
    """OpenAI's consent recording: ``recording`` (the speaker reading a consent statement), ``name`` and
    ``language`` (BCP 47). Its id is then the ``consent`` of ``POST /v1/audio/voices``. bolo stores the
    recording with the voice library; it does not check what is said in it."""
    core = get_core(request)
    form = await request.form()
    rec = form.get("recording")
    name, language = form.get("name"), form.get("language")
    if rec is None or isinstance(rec, str):
        raise InvalidRequest("send the consent recording as a file part named 'recording'", param="recording")
    if not isinstance(name, str) or not name.strip():
        raise InvalidRequest("name is required", param="name")
    if not isinstance(language, str) or not language.strip():
        raise InvalidRequest("language is required (a BCP 47 tag such as en-US)", param="language")
    data = await rec.read()
    if len(data) > 10 * 1024 * 1024:
        raise InvalidRequest("the consent recording is limited to 10 MiB", param="recording")
    samples, rate = await asyncio.to_thread(decode_audio, data, filename=rec.filename, content_type=rec.content_type)
    if len(samples) < rate:
        raise InvalidRequest("the consent recording is shorter than a second", param="recording")
    cid = "cons_" + secrets.token_hex(12)
    path = _consent_dir(core) / cid
    path.mkdir(parents=True)
    (path / "recording.wav").write_bytes(encode_audio(samples, rate, "wav"))
    meta = {
        "id": cid, "created_at": int(time.time()), "name": name.strip(), "language": language.strip(),
        "by": created_by(request), "at": now_iso(),
    }  # fmt: skip
    (path / "consent.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return JSONResponse(_consent_object(meta))


@router.get("/v1/audio/voice_consents", tags=["openai"], summary="List consent recordings")
async def list_voice_consents(request: Request) -> JSONResponse:
    core = get_core(request)
    root = _consent_dir(core)
    items = []
    if root.is_dir():
        for p in sorted(root.iterdir()):
            with contextlib.suppress(OSError, ValueError, KeyError):
                items.append(_consent_object(json.loads((p / "consent.json").read_text(encoding="utf-8"))))
    items.sort(key=lambda m: m["created_at"], reverse=True)
    return JSONResponse({"object": "list", "data": items, "has_more": False})


@router.get("/v1/audio/voice_consents/{consent_id}", tags=["openai"], summary="One consent recording")
async def get_voice_consent(request: Request, consent_id: str) -> JSONResponse:
    return JSONResponse(_consent_object(_read_consent(get_core(request), consent_id)))


class ConsentPatch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str


@router.post("/v1/audio/voice_consents/{consent_id}", tags=["openai"], summary="Rename a consent recording")
async def update_voice_consent(request: Request, consent_id: str, body: ConsentPatch) -> JSONResponse:
    core = get_core(request)
    meta = _read_consent(core, consent_id)
    meta["name"] = body.name.strip() or meta["name"]
    (_consent_dir(core) / consent_id / "consent.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return JSONResponse(_consent_object(meta))


@router.delete("/v1/audio/voice_consents/{consent_id}", tags=["openai"], summary="Delete a consent recording")
async def delete_voice_consent(request: Request, consent_id: str) -> JSONResponse:
    core = get_core(request)
    _read_consent(core, consent_id)
    shutil.rmtree(_consent_dir(core) / consent_id)
    return JSONResponse({"id": consent_id, "object": "audio.voice_consent", "deleted": True})


def _voice_object(voice: SavedVoice) -> dict[str, Any]:
    created = voice.meta.get("created_at")
    try:
        import datetime as dt

        epoch = int(dt.datetime.fromisoformat(str(created).replace("Z", "+00:00")).timestamp())
    except ValueError:
        epoch = 0
    return {"id": voice.id, "object": "audio.voice", "created_at": epoch, "name": voice.name, "type": "audio_sample"}


@router.post("/v1/audio/voices", tags=["openai"], summary="Create a custom voice from an audio sample")
async def create_openai_voice(request: Request) -> JSONResponse:
    """OpenAI's shape: ``audio_sample`` (file), ``name`` and ``consent``: the id of a consent recording
    (``/v1/audio/voice_consents``), or bolo's statements ``self`` / ``permission``. Voices from a text prompt
    (``type: prompt``) are not supported. The voice's id is made from its name; use it as ``voice`` (or
    ``{"id": ...}``) in ``/v1/audio/speech`` with a model that clones voices."""
    core = get_core(request)
    form = await request.form()
    if form.get("type") == "prompt":
        raise InvalidRequest("bolo creates voices from audio samples only (type audio_sample)", param="type")
    consent = form.get("consent")
    if not isinstance(consent, str) or not consent.strip():
        raise InvalidRequest("consent is required: a consent recording id, or self / permission", param="consent")
    record: dict[str, Any] | None = None
    basis = consent
    if consent.startswith("cons_"):
        found = _read_consent(core, consent)
        basis = "permission"
        record = {
            "basis": "recording", "statement": "A recorded consent statement from the speaker.",
            "recording": found["id"],
        }  # fmt: skip
    else:
        parse_consent(consent)
    files = _files(form)
    if not files:
        raise InvalidRequest("send the sample as a file part named 'audio_sample'", param="audio_sample")
    name = form.get("name")
    if not isinstance(name, str) or not name.strip():
        raise InvalidRequest("name is required", param="name")
    clips = [await read_upload(core, files[0])]
    voice = await create_voice(
        core, request, clips=clips, name=name, consent=basis, consent_record=record, source="openai"
    )
    return JSONResponse(_voice_object(voice))


@router.delete("/v1/audio/voices/{voice_id}", tags=["openai"], summary="Delete a custom voice")
async def delete_openai_voice(request: Request, voice_id: str) -> JSONResponse:
    core = get_core(request)
    voice = core.voices.get(voice_id)
    core.voices.delete(voice.id)
    core.events.publish("voice", {"id": voice.id, "event": "deleted"})
    return JSONResponse({"id": voice.id, "object": "audio.voice", "deleted": True})
