"""Pronunciation word lists and per-voice text handling under ``/api/v1``.

Lists live in ``<data_dir>/wordlists/<name>.tsv`` (see :mod:`bolo.text.wordlists`).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from bolo.api.common import get_core
from bolo.errors import InvalidRequest
from bolo.text import wordlists as wl
from bolo.text.prepare import prepare, text_input

router = APIRouter(prefix="/api/v1")

_EXPORT_TYPES = {
    "tsv": "text/tab-separated-values; charset=utf-8",
    "pls": "application/pls+xml; charset=utf-8",
    "json": "application/json",
}


def _store(request: Request) -> wl.WordListStore:
    return wl.store_for(get_core(request).settings.data_dir)


@router.get("/wordlists", tags=["native"], summary="List pronunciation word lists")
async def list_wordlists(request: Request) -> JSONResponse:
    return JSONResponse({"wordlists": _store(request).all(), "available": wl.uccharan_module() is not None})


@router.get("/wordlists/{name}", tags=["native"], summary="One word list (JSON entries, TSV or PLS)")
async def get_wordlist(request: Request, name: str, format: str | None = None, lang: str | None = None) -> Response:
    store = _store(request)
    fmt = (format or "").lower()
    if not fmt:
        accept = request.headers.get("accept", "")
        fmt = "tsv" if "tab-separated" in accept else "pls" if "pls+xml" in accept else "json"
    if fmt not in _EXPORT_TYPES:
        raise InvalidRequest("format must be json, tsv or pls", param="format")
    if fmt == "json":
        body = store.summary(name)
        body["entries"] = store.entries(name)
        return JSONResponse(body)
    return Response(store.export(name, fmt, lang), media_type=_EXPORT_TYPES[fmt])


@router.put("/wordlists/{name}", tags=["native"], summary="Create or replace a word list (TSV, PLS or JSON body)")
async def put_wordlist(request: Request, name: str) -> JSONResponse:
    raw = await request.body()
    if len(raw) > wl.MAX_BYTES:
        raise InvalidRequest(f"a word list is limited to {wl.MAX_BYTES} bytes", param="body")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidRequest("a word list must be UTF-8 text", param="body") from exc
    info = _store(request).put(wl.check_name(name), text)
    get_core(request).events.publish("wordlist", {"name": name, "event": "updated"})
    return JSONResponse(info)


@router.delete("/wordlists/{name}", tags=["native"], summary="Delete a word list")
async def delete_wordlist(request: Request, name: str) -> JSONResponse:
    _store(request).delete(wl.check_name(name))
    get_core(request).events.publish("wordlist", {"name": name, "event": "deleted"})
    return JSONResponse({"deleted": name})


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(description="Text to render.")
    models: list[str] = Field(min_length=1, max_length=8, description="TTS model ids (up to eight).")
    language: str | None = Field(default=None, description="Language of the text (default: each model's).")
    voice: str | None = Field(default=None, description="Voice id, for models whose voices differ.")
    input_format: str = Field(default="text", description="text, ssml or markdown.")
    normalize: bool | str = True


def _phonemes(core: Any, spec: Any, voice: str | None, text: str) -> str | None:
    """The phonemes a phoneme voice would get (one line per sentence), when a phonemiser is available."""
    from bolo.text import phonemize

    target = core.components.phoneme_target(spec, voice)
    if target is None:
        return None
    profile, language = target
    try:
        ph, improved = phonemize.resolve(profile, language, core.components.pinned_phonemizer(spec))
        cps = ph.voice_codepoints(text, language, profile, improved)
    except Exception:  # no phonemiser, or its data is not available here: the text is still shown
        return None
    return "\n".join("".join(map(chr, s)) for s in cps)


@router.post("/wordlists/{name}/preview", tags=["native"], summary="What each voice would receive")
async def preview(request: Request, name: str, body: PreviewRequest) -> JSONResponse:
    """Render ``text`` with this word list (``-`` for none beyond the pinned and default lists)
    for each model: the capability record, the text or phonemes the voice would receive, and the
    entries applied."""
    core = get_core(request)
    lists = None if name == "-" else [wl.check_name(name)]
    out: list[dict[str, Any]] = []
    for model_id in body.models:
        spec = core.resolve(model_id, "tts")
        voice = body.voice or (str(spec.voices[0]["id"]) if spec.voices else None)
        lang = (
            body.language
            or (str(spec.voices[0].get("language") or "") if spec.voices else "")
            or (spec.languages[0] if spec.languages else "en")
        )
        cap = text_input(core, spec, voice)
        prepared = prepare(
            core, spec, voice, body.text, lang, normalize=body.normalize, wordlists=lists,
            input_format=body.input_format,
        )  # fmt: skip
        item: dict[str, Any] = {
            "model": spec.id,
            "voice": voice,
            "language": lang,
            "text_input": cap.to_dict(),
            "normalizer": prepared.normalizer,
            "text": prepared.text,
            "applied": prepared.applied,
        }
        if cap.reads == "phonemes":
            item["phonemes"] = _phonemes(core, spec, voice, prepared.text)
        out.append(item)
    return JSONResponse({"wordlist": name, "previews": out})


@router.get("/text-input/{model_id:path}", tags=["native"], summary="How a model's voices take text")
async def model_text_input(request: Request, model_id: str, voice: str | None = None) -> JSONResponse:
    core = get_core(request)
    spec = core.resolve(model_id, "tts")
    return JSONResponse({"model": spec.id, "voice": voice, "text_input": text_input(core, spec, voice).to_dict()})
