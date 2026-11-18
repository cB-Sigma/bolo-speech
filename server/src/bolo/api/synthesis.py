"""Speech synthesis beyond one voice: multi-speaker dialogues (``POST /v1/audio/dialogue``) and voice conversion
(``POST /v1/audio/voice-conversion``, with ElevenLabs' speech-to-speech route as an alias)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from bolo.api.common import get_core, read_upload
from bolo.audio.encode import FORMATS, StreamEncoder, encode_audio, normalize_format
from bolo.dialogue import MODES, Turn, parse_script, speak_dialogue
from bolo.errors import BoloError, InvalidRequest
from bolo.keepalive import request_keep_alive
from bolo.obs.trace import Trace

router = APIRouter()


class DialogueTurn(BaseModel):
    speaker: str = Field(max_length=40, description="Who speaks this turn (any name; the same name, the same voice).")
    text: str = Field(description="What they say.")
    instructions: str | None = Field(
        default=None, max_length=4096, description="Direction for this turn only (stitched dialogues)."
    )


class DialogueRequest(BaseModel):
    model: str | None = Field(default=None, description="Any text-to-speech model; default `default_tts`.")
    turns: list[DialogueTurn] | None = Field(default=None, description="The dialogue, turn by turn.")
    script: str | None = Field(
        default=None,
        description="Instead of `turns`: one `Name: text` line per turn (a line without a name continues the turn).",
    )
    voices: dict[str, str] = Field(
        default_factory=dict,
        description="Speaker name -> voice: a voice of the model, a saved voice, a blend (stitched), or one of the "
        "reference clips `en-short`, `en-long`, `hi-short` (one take). A speaker without one gets the model's "
        "default voice (stitched) or a reference clip (one take).",
    )
    mode: str = Field(
        default="auto",
        description="`auto`: in one take when the model speaks dialogues itself (`features` has `dialogue`) and "
        "every voice has a recording, stitched otherwise; `native`: in one take or an error; `stitched`: each turn "
        "on its own, joined with `pause_ms` between them.",
    )
    pause_ms: int = Field(default=300, description="Silence between turns of a stitched dialogue (0 to 5000).")
    language: str | None = None
    speed: float = 1.0
    instructions: str | None = Field(
        default=None, max_length=4096, description="Direction for the whole dialogue, for models that follow it."
    )
    watermark: bool | str | None = Field(
        default=None,
        description="As for `/v1/audio/speech`: on, off or a watermark model; unset: the server's policy (a one-take "
        "dialogue imitates recordings, so it counts as cloned speech).",
    )
    response_format: str = "wav"
    sample_rate: int | None = None
    keep_alive: str | int | float | None = None


@router.post("/v1/audio/dialogue", tags=["native"], summary="Speak a multi-speaker dialogue")
async def create_dialogue(request: Request, body: DialogueRequest) -> Response:
    """Extension. A dialogue (turns of speaker and text, a voice per speaker) as one recording.

    Models that speak a whole script themselves (`features` contains `dialogue`: VibeVoice on audio.cpp) do it
    in one take, with a recording of each voice (a saved voice or a reference clip); any other model speaks each
    turn on its own and the turns are joined. Headers: `x-bolo-dialogue` (`native` or `stitched`),
    `x-bolo-speakers` (the speakers in order), `x-bolo-model`, `x-bolo-segments` (stitched: JSON
    `[{turn, speaker, start, end}]`, in seconds)."""
    core = get_core(request)
    fmt = normalize_format(body.response_format)
    if fmt not in FORMATS:
        raise InvalidRequest(f"response_format must be one of {', '.join(FORMATS)}", param="response_format")
    if body.mode not in MODES:
        raise InvalidRequest(f"mode must be one of {', '.join(MODES)}", param="mode")
    if body.sample_rate is not None and not 8000 <= body.sample_rate <= 48000:
        raise InvalidRequest("sample_rate must be between 8000 and 48000", param="sample_rate")
    if body.turns and body.script:
        raise InvalidRequest("send turns or a script, not both", param="script")
    turns = (
        [Turn(t.speaker.strip(), t.text, t.instructions) for t in body.turns]
        if body.turns
        else parse_script(body.script or "")
    )
    request_keep_alive(body.keep_alive, ("tts",))
    trace = Trace("tts", "http", body.model or "")
    try:
        result = await speak_dialogue(
            core, turns, {k.strip(): v for k, v in body.voices.items()}, model=body.model, mode=body.mode,
            pause_ms=body.pause_ms, language=body.language, speed=body.speed, instructions=body.instructions,
            trace=trace, watermark=body.watermark,
        )  # fmt: skip
        trace.audio_seconds = len(result.samples) / result.sample_rate if result.sample_rate else 0.0
        data = await asyncio.to_thread(encode_audio, result.samples, result.sample_rate, fmt, body.sample_rate)
    except BoloError as exc:
        trace.finish("error", exc.message)
        if trace.model:
            core.traces.record(trace)
        raise
    core.complete_tts(trace)
    encoder = StreamEncoder(fmt, result.sample_rate, body.sample_rate)
    headers = {
        "x-request-id": trace.id,
        "x-bolo-model": result.model,
        "x-bolo-dialogue": result.mode,
        "x-bolo-speakers": ", ".join(result.speakers),
        "x-sample-rate": str(encoder.sample_rate),
    }
    if result.segments:
        headers["x-bolo-segments"] = json.dumps(result.segments, ensure_ascii=True, separators=(",", ":"))
    if trace.extra.get("instructions"):
        headers["x-bolo-instructions"] = str(trace.extra["instructions"])
    from bolo.core import fallback_headers

    if "x-bolo-watermark" in (wm := fallback_headers(trace)):
        headers["x-bolo-watermark"] = wm["x-bolo-watermark"]
    return Response(data, media_type=encoder.media_type, headers=headers)


def _form_file(form: Any, *names: str) -> Any:
    for name in names:
        value = form.get(name)
        if value is not None and not isinstance(value, str):
            return value
    return None


async def _voice_conversion(
    request: Request, form: Any, voice: str | None, model: str | None, fmt_raw: str
) -> Response:
    from bolo.voice_conversion import convert_voice

    core = get_core(request)
    file = _form_file(form, "file", "audio")
    if file is None:
        raise InvalidRequest("a multipart 'file' field with the speech to convert is required", param="file")
    fmt = normalize_format(fmt_raw)
    if fmt not in FORMATS:
        raise InvalidRequest(f"response_format must be one of {', '.join(FORMATS)}", param="response_format")
    out_rate: int | None = None
    if form.get("sample_rate"):
        try:
            out_rate = int(str(form.get("sample_rate")))
        except ValueError as exc:
            raise InvalidRequest("sample_rate must be a whole number of hertz", param="sample_rate") from exc
        if not 8000 <= out_rate <= 48000:
            raise InvalidRequest("sample_rate must be between 8000 and 48000", param="sample_rate")
    options: dict[str, Any] = {}
    if form.get("options"):
        try:
            options = json.loads(str(form.get("options")))
        except ValueError as exc:
            raise InvalidRequest("options must be a JSON object", param="options") from exc
        if not isinstance(options, dict):
            raise InvalidRequest("options must be a JSON object", param="options")
    ref_file = _form_file(form, "reference_audio")
    reference = await ref_file.read() if ref_file is not None else None
    request_keep_alive(form.get("keep_alive"), ("voice-conversion",))
    samples, rate = await read_upload(core, file)
    trace = Trace("voice-conversion", "http", model or "")
    try:
        result = await convert_voice(
            core, samples, rate, model=model, voice=voice, reference=reference, options=options, trace=trace
        )
        data = await asyncio.to_thread(encode_audio, result.samples, result.sample_rate, fmt, out_rate)
        trace.finish("ok")
    except BoloError as exc:
        trace.finish("error", exc.message)
        raise
    finally:
        if trace.model:
            core.traces.record(trace)
    encoder = StreamEncoder(fmt, result.sample_rate, out_rate)
    headers = {
        "x-request-id": trace.id,
        "x-bolo-model": result.model,
        "x-sample-rate": str(encoder.sample_rate),
        "x-bolo-target": ", ".join(f"{k}={v}" for k, v in result.target.items()),
    }
    return Response(data, media_type=encoder.media_type, headers=headers)


@router.post("/v1/audio/voice-conversion", tags=["native"], summary="Say the same words in another voice")
async def voice_conversion(request: Request) -> Response:
    """Extension. Multipart form: ``file`` (the speech to convert), ``voice`` (the target: a saved voice, a reference
    clip ``en-short`` / ``en-long`` / ``hi-short``, or a voice of the model for RVC), or ``reference_audio`` (a
    recording of the target voice, which wins over ``voice``); ``model`` (a ``voice-conversion`` model, default
    ``default_voice_conversion``), ``options`` (JSON: the model's request options, e.g. Seed-VC's
    ``num_inference_steps``), ``response_format`` (default ``wav``), ``sample_rate``, ``keep_alive``.

    Headers: ``x-bolo-model``, ``x-bolo-target`` (``saved=priya``, ``clip=en-short``, ``voice=default`` or
    ``reference=upload``), ``x-sample-rate``."""
    form = await request.form()
    return await _voice_conversion(
        request,
        form,
        str(form.get("voice") or "") or None,
        str(form.get("model") or "") or None,
        str(form.get("response_format") or "wav"),
    )


@router.post("/v1/speech-to-speech/{voice_id}", tags=["native"], summary="Voice conversion (ElevenLabs' route)")
async def speech_to_speech(request: Request, voice_id: str) -> Response:
    """ElevenLabs' speech-to-speech route: multipart ``audio``, the target ``voice_id`` in the path (a saved voice
    or reference clip, as ``voice`` of ``POST /v1/audio/voice-conversion``), ``model_id``; MP3 back unless
    ``output_format`` (``mp3_44100_128``, ``pcm_16000``, ...: the codec is the part before the first ``_``) says
    otherwise."""
    form = await request.form()
    raw = str(form.get("output_format") or form.get("response_format") or "mp3")
    return await _voice_conversion(
        request,
        form,
        voice_id,
        str(form.get("model_id") or form.get("model") or "") or None,
        raw.split("_", maxsplit=1)[0],
    )


@router.post("/v1/audio/watermark/detect", tags=["native"], summary="Is this recording watermarked?")
async def detect_watermark(request: Request) -> Response:
    """Extension. Multipart ``file`` (any decodable recording), optional ``model`` (a ``watermark`` model; default
    the server's ``watermark_model``; ``all``: every installed watermarker, the most certain answer first).

    Returns ``marked``, ``model`` (the watermarker whose mark was looked for), ``score`` (the share of the recording
    that holds the mark, 0 to 1; marked from 0.5), ``payload`` (the bits the mark carries, for models that carry
    some), ``payload_is_this_servers`` and ``segments`` (where the mark is, in seconds); with ``model=all``,
    ``results`` for each watermarker."""
    from fastapi.responses import JSONResponse

    from bolo.watermark import detect

    core = get_core(request)
    form = await request.form()
    file = _form_file(form, "file", "audio")
    if file is None:
        raise InvalidRequest("a multipart 'file' field with the recording is required", param="file")
    wanted = str(form.get("model") or "") or core.settings.watermark_model
    if wanted == "all":
        models = [m.id for m in core.catalog.all() if m.task == "watermark" and core.manager.is_installed(m)]
        if not models:
            raise InvalidRequest("no watermark model is installed (bolo pull audioseal-16bits)", param="model")
    else:
        models = [wanted]
    samples, rate = await read_upload(core, file)
    trace = Trace("watermark", "http", models[0])
    trace.audio_seconds = len(samples) / rate if rate else 0.0
    try:
        with trace.phase("detect"):
            results = [await detect(core, samples, rate, m) for m in models]
        trace.finish("ok")
    except BoloError as exc:
        trace.finish("error", exc.message)
        raise
    finally:
        core.traces.record(trace)
    results.sort(key=lambda r: (not r.marked, -r.score))
    best = results[0]
    body: dict[str, Any] = {
        "object": "audio.watermark",
        "duration": round(len(samples) / rate, 3) if rate else 0.0,
        **best.to_dict(),
    }
    if wanted == "all":
        body["results"] = [r.to_dict() for r in results]
    return JSONResponse(body, headers={"x-request-id": trace.id})


@router.post("/v1/audio/watermark", tags=["native"], summary="Add a watermark to a recording")
async def embed_watermark(request: Request) -> Response:
    """Extension. Multipart ``file``; ``model`` (a ``watermark`` model, default ``watermark_model``),
    ``response_format`` (default ``wav``), ``sample_rate``. The recording comes back with the server's mark
    (``watermark_payload``), at its own rate unless ``sample_rate`` says otherwise."""
    from bolo.watermark import marker

    core = get_core(request)
    form = await request.form()
    file = _form_file(form, "file", "audio")
    if file is None:
        raise InvalidRequest("a multipart 'file' field with the recording is required", param="file")
    fmt = normalize_format(str(form.get("response_format") or "wav"))
    if fmt not in FORMATS:
        raise InvalidRequest(f"response_format must be one of {', '.join(FORMATS)}", param="response_format")
    out_rate = int(str(form.get("sample_rate"))) if form.get("sample_rate") else None
    samples, rate = await read_upload(core, file)
    model = core.resolve(str(form.get("model") or "") or core.settings.watermark_model, "watermark").id
    async with marker(core, model) as mark:
        assert mark is not None
        if not mark.active:
            raise InvalidRequest(f"the watermark model '{model}' cannot run: {mark.reason}", param="model")
        marked = await mark.mark(samples, rate)
    data = await asyncio.to_thread(encode_audio, marked, rate, fmt, out_rate)
    encoder = StreamEncoder(fmt, rate, out_rate)
    return Response(
        data,
        media_type=encoder.media_type,
        headers={"x-bolo-watermark": model, "x-sample-rate": str(encoder.sample_rate)},
    )
