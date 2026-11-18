"""OpenAI-compatible audio endpoints.

Shapes follow the OpenAI API reference for ``/v1/audio/transcriptions``,
``/v1/audio/translations``, ``/v1/audio/speech`` and ``/v1/models``, and are
exercised with the official ``openai`` Python SDK in the test-suite.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import time
from collections.abc import AsyncGenerator, AsyncIterator, Coroutine
from typing import Any, Literal, TypeVar

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from bolo.api.common import get_core, parse_bool, read_upload, sse
from bolo.audio import decode_audio, resample
from bolo.audio.encode import StreamEncoder, normalize_format
from bolo.core import Bolo, SynthChunk, Transcript, fallback_headers
from bolo.denoise import parse_denoise
from bolo.diarize import DiarizeOptions, KnownSpeaker
from bolo.engines.base import FloatArray, Segment
from bolo.errors import BoloError, InvalidRequest, ModelNotFound, ModelNotInstalled
from bolo.keepalive import request_keep_alive
from bolo.lid import is_routing_request, route_headers
from bolo.lid import route as route_language
from bolo.obs.trace import Trace
from bolo.separate import SeparateOptions
from bolo.stt_options import OptionError
from bolo.stt_options import from_form as decode_options_from_form
from bolo.text.formats import to_srt, to_vtt

router = APIRouter()
T = TypeVar("T")

TRANSCRIPTION_FORMATS = ("json", "text", "srt", "vtt", "verbose_json", "diarized_json")

LANGUAGE_NAMES = {
    "en": "english", "hi": "hindi", "bn": "bengali", "ta": "tamil", "te": "telugu", "mr": "marathi",
    "gu": "gujarati", "kn": "kannada", "ml": "malayalam", "pa": "punjabi", "ur": "urdu", "ne": "nepali",
    "or": "odia", "as": "assamese", "sa": "sanskrit", "sd": "sindhi", "es": "spanish", "fr": "french",
    "de": "german", "it": "italian", "pt": "portuguese", "nl": "dutch", "ru": "russian", "uk": "ukrainian",
    "pl": "polish", "tr": "turkish", "ar": "arabic", "fa": "persian", "he": "hebrew", "zh": "chinese",
    "yue": "cantonese", "ja": "japanese", "ko": "korean", "vi": "vietnamese", "th": "thai", "id": "indonesian",
    "ms": "malay", "sv": "swedish", "da": "danish", "no": "norwegian", "fi": "finnish", "cs": "czech",
    "el": "greek", "hu": "hungarian", "ro": "romanian", "ca": "catalan", "sw": "swahili",
}  # fmt: skip


async def run_unless_disconnected(request: Request, coro: Coroutine[Any, Any, T]) -> T | None:
    """Run ``coro`` but cancel it if the HTTP client disconnects first (returns None then).

    A queued decode for a client that has gone is removed from the queue
    instead of wasting the engine's time.
    """
    task = asyncio.ensure_future(coro)

    async def watch() -> None:
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    watcher = asyncio.ensure_future(watch())
    try:
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            return task.result()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        return None
    finally:
        watcher.cancel()
        if not task.done():
            task.cancel()


def language_name(code: str | None) -> str:
    if not code:
        return "unknown"
    return LANGUAGE_NAMES.get(code.split("-")[0].lower(), code)


def _model_object(entry: dict[str, Any]) -> dict[str, Any]:
    task = {
        "stt": "automatic-speech-recognition",
        "tts": "text-to-speech",
        "vad": "voice-activity-detection",
        "diarization": "speaker-diarization",
    }.get(entry["task"], entry["task"])
    out = {
        "id": entry["id"],
        "object": "model",
        "created": 0,
        "owned_by": entry["engine"],
        # speaches-compatible extras
        "language": entry["languages"],
        "task": task,
        "aliases": entry["aliases"],
        "tier": entry["tier"],
        "license": entry["license"],
    }
    if entry["task"] == "tts":
        out["sample_rate"] = entry["sample_rate"]
        out["voices"] = [
            {
                "name": v["id"],
                "id": v["id"],
                "language": v["language"],
                "gender": {"f": "female", "m": "male"}.get(v["gender"], v["gender"]),
            }
            for v in entry["voices"]
        ]
    return out


@router.get("/v1/models", tags=["openai"], summary="List installed models")
async def list_models(request: Request, task: str | None = None) -> JSONResponse:
    core = get_core(request)
    data = [_model_object(m) for m in core.catalog_api(installed=True)]
    if task:
        data = [m for m in data if m["task"] == task]
    return JSONResponse({"object": "list", "data": data})


@router.get("/v1/models/{model_id:path}", tags=["openai"], summary="Retrieve a model")
async def get_model(request: Request, model_id: str) -> JSONResponse:
    core = get_core(request)
    spec = core.catalog.resolve(model_id)
    if spec is None or not core.manager.is_installed(spec):
        raise ModelNotFound(f"The model '{model_id}' does not exist or is not installed.", param="model")
    return JSONResponse(_model_object(core.model_api(spec)))


def _usage(seconds: float) -> dict[str, Any]:
    return {"type": "duration", "seconds": round(seconds)}


def _verbose(t: Transcript, task: str, word_level: bool, segment_level: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "task": task,
        "language": language_name(t.language),
        "duration": round(t.duration, 3),
        "text": t.text,
    }
    if t.language_probability is not None:
        out["language_probability"] = t.language_probability
    if word_level:
        speaker_of = {id(w): s.speaker for s in t.segments for w in s.words if s.speaker}
        out["words"] = [
            {
                "word": w.w,
                "start": round(w.t0, 3),
                "end": round(w.t1, 3),
                **({"probability": w.p} if w.p is not None else {}),
                **({"speaker": speaker_of[id(w)]} if id(w) in speaker_of else {}),
            }
            for w in t.words
        ]
    if segment_level:
        out["segments"] = [
            {
                "id": i,
                "seek": 0,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text,
                "tokens": [],
                "temperature": s.temperature if s.temperature is not None else 0.0,
                "avg_logprob": s.avg_logprob if s.avg_logprob is not None else 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": s.no_speech_prob if s.no_speech_prob is not None else 0.0,
                **({"speaker": s.speaker} if s.speaker else {}),
                **({"overlap": True} if s.overlap else {}),
                **({"separated": True} if s.separated else {}),
                **({"emotion": s.emotion} if s.emotion else {}),
                **({"events": s.events} if s.events else {}),
            }
            for i, s in enumerate(t.segments)
        ]
    out.update(paralinguistics(t.segments))
    out["usage"] = _usage(t.duration)
    if t.separation is not None:
        out["separation"] = t.separation
    return out


def paralinguistics(segments: list[Segment]) -> dict[str, Any]:
    """What a recogniser that tags emotions and sounds (SenseVoice) heard, over the whole recording: ``emotion`` (the
    one heard in the most speech, by seconds) and ``audio_events`` (each sound with the seconds of the segments it was
    heard in). Nothing when the recogniser does not tag."""
    emotions: dict[str, float] = {}
    events: dict[str, float] = {}
    for s in segments:
        length = max(0.0, s.end - s.start)
        if s.emotion:
            emotions[s.emotion] = emotions.get(s.emotion, 0.0) + length
        for e in s.events:
            events[e] = events.get(e, 0.0) + length
    out: dict[str, Any] = {}
    if emotions:
        out["emotion"] = max(emotions, key=lambda k: emotions[k])
        out["emotions"] = {k: round(v, 2) for k, v in sorted(emotions.items(), key=lambda kv: -kv[1])}
    if events:
        out["audio_events"] = {k: round(v, 2) for k, v in sorted(events.items(), key=lambda kv: -kv[1])}
    return out


def _diarized_segment(index: int, s: Segment) -> dict[str, Any]:
    return {
        "type": "transcript.text.segment",
        "id": f"seg_{index}",
        "start": round(s.start, 3),
        "end": round(s.end, 3),
        "text": s.text,
        "speaker": s.speaker or "A",
        **({"overlap": True} if s.overlap else {}),
        **({"separated": True} if s.separated else {}),
    }


def _diarized(t: Transcript) -> dict[str, Any]:
    """OpenAI's ``diarized_json`` shape (the response of ``gpt-4o-transcribe-diarize``)."""
    return {
        "task": "transcribe",
        "duration": round(t.duration, 3),
        "text": t.text,
        "segments": [_diarized_segment(i, s) for i, s in enumerate(t.segments)],
        "usage": _usage(t.duration),
        **({"separation": t.separation} if t.separation is not None else {}),
    }


def _render(t: Transcript, fmt: str, task: str, granularities: list[str], headers: dict[str, str]) -> Response:
    if t.separation is not None:
        status, reason = t.separation["status"], t.separation.get("reason")
        headers = {**headers, "x-bolo-separation": f"{status}; reason={reason}" if reason else status}
    if fmt == "diarized_json":
        return JSONResponse(_diarized(t), headers=headers)
    if fmt == "text":
        if any(s.speaker for s in t.segments):
            return PlainTextResponse("".join(f"{s.speaker}: {s.text}\n" for s in t.segments), headers=headers)
        return PlainTextResponse(t.text + "\n", headers=headers)
    if fmt == "srt":
        return PlainTextResponse(to_srt(t.segments), headers=headers)
    if fmt == "vtt":
        return PlainTextResponse(to_vtt(t.segments), headers=headers)
    if fmt == "verbose_json":
        return JSONResponse(_verbose(t, task, "word" in granularities, "segment" in granularities), headers=headers)
    body: dict[str, Any] = {"text": t.text, "usage": _usage(t.duration)}
    if t.separation is not None:
        body["separation"] = t.separation
    return JSONResponse(body, headers=headers)


def _int_field(form: Any, name: str) -> int | None:
    raw = form.get(name)
    if raw in (None, ""):
        return None
    try:
        return int(str(raw))
    except ValueError as exc:
        raise InvalidRequest(f"{name} must be an integer", param=name) from exc


async def known_speakers(form: Any) -> list[KnownSpeaker]:
    """``known_speaker_names[]`` and ``known_speaker_references[]`` (data URLs as in OpenAI's API, or file parts)."""
    names = [str(n).strip() for key in ("known_speaker_names[]", "known_speaker_names") for n in form.getlist(key)]
    refs = [r for key in ("known_speaker_references[]", "known_speaker_references") for r in form.getlist(key)]
    if not names and not refs:
        return []
    if len(names) != len(refs):
        raise InvalidRequest(
            f"known_speaker_names[] ({len(names)}) and known_speaker_references[] ({len(refs)}) must have the same "
            "number of entries",
            param="known_speaker_references[]",
        )
    if len(names) > 8 or any(not n for n in names) or len(set(names)) != len(names):
        raise InvalidRequest("give at most 8 known speakers, each with a distinct name", param="known_speaker_names[]")
    out: list[KnownSpeaker] = []
    for name, ref in zip(names, refs):
        try:
            if isinstance(ref, str):
                if not ref.startswith("data:") or "," not in ref:
                    raise ValueError("expected a data URL (data:audio/wav;base64,...) or a file part")
                header, payload = ref.split(",", 1)
                data = base64.b64decode(payload, validate=False)
                content_type, filename = header[5:].split(";")[0] or None, None
            else:
                data, content_type, filename = await ref.read(), ref.content_type, ref.filename
            samples, rate = await asyncio.to_thread(decode_audio, data, filename=filename, content_type=content_type)
        except (ValueError, BoloError) as exc:
            raise InvalidRequest(
                f"the reference for known speaker '{name}' could not be read: {exc}", param="known_speaker_references[]"
            ) from exc
        seconds = len(samples) / rate if rate else 0.0
        if not 1.0 <= seconds <= 60.0:
            raise InvalidRequest(
                f"the reference for known speaker '{name}' is {seconds:.1f} s long; use 2 to 10 s of that person "
                "speaking (1 to 60 s are accepted)",
                param="known_speaker_references[]",
            )
        out.append(KnownSpeaker(name, np.ascontiguousarray(resample(samples, rate, 16000), dtype=np.float32)))
    return out


def _float_field(form: Any, name: str) -> float | None:
    raw = form.get(name)
    if raw in (None, ""):
        return None
    try:
        return float(str(raw))
    except ValueError as exc:
        raise InvalidRequest(f"{name} must be a number", param=name) from exc


def separate_options(form: Any) -> SeparateOptions | None:
    """``separate_overlaps=true`` and its ``separation_*`` fields (bolo extension)."""
    if not parse_bool(form.get("separate_overlaps")):
        return None
    return SeparateOptions(
        model=str(form.get("separation_model") or "") or None,
        context_s=_float_field(form, "separation_context_s"),
        min_overlap_s=_float_field(form, "separation_min_overlap_s"),
        allow_narrowband=parse_bool(form.get("separation_allow_narrowband")),
        return_audio=parse_bool(form.get("separation_audio")),
    )


async def _diarize_options(form: Any, fmt: str, model: str | None, task: str) -> DiarizeOptions | None:
    """Diarization is on with ``diarize=true``, ``response_format=diarized_json`` or the OpenAI diarize model name."""
    wanted = parse_bool(form.get("diarize")) or fmt == "diarized_json" or (model or "").endswith("-diarize")
    if not wanted:
        return None
    if task != "transcribe":
        raise InvalidRequest("diarization is only available for transcriptions", param="diarize")
    try:
        threshold = float(str(form.get("diarization_threshold"))) if form.get("diarization_threshold") else None
        identify = float(str(form.get("identify_threshold"))) if form.get("identify_threshold") else None
    except ValueError as exc:
        raise InvalidRequest(
            "diarization_threshold and identify_threshold must be numbers", param="diarization_threshold"
        ) from exc
    return DiarizeOptions(
        model=str(form.get("diarization_model") or "") or None,
        num_speakers=_int_field(form, "num_speakers"),
        min_speakers=_int_field(form, "min_speakers"),
        max_speakers=_int_field(form, "max_speakers"),
        threshold=threshold,
        known=await known_speakers(form),
        identify_threshold=identify,
    )


async def _transcribe_endpoint(request: Request, task: str) -> Response:
    core = get_core(request)
    trace = Trace("stt", "http", "")
    form = await request.form()
    try:
        file = form.get("file")
        if file is None or isinstance(file, str):
            raise InvalidRequest("a multipart 'file' field with the audio is required", param="file")
        fmt = str(form.get("response_format") or "json")
        if fmt not in TRANSCRIPTION_FORMATS:
            raise InvalidRequest(
                f"response_format must be one of {', '.join(TRANSCRIPTION_FORMATS[:5])}", param="response_format"
            )
        granularities = [
            str(g) for g in (form.getlist("timestamp_granularities[]") or form.getlist("timestamp_granularities"))
        ]
        for g in granularities:
            if g not in ("word", "segment"):
                raise InvalidRequest(
                    "timestamp_granularities[] must be 'word' and/or 'segment'", param="timestamp_granularities[]"
                )
        granularities = granularities or ["segment"]
        stream = parse_bool(form.get("stream")) and task == "transcribe"
        try:
            temperature = float(str(form.get("temperature") or 0.0))
        except ValueError as exc:
            raise InvalidRequest("temperature must be a number", param="temperature") from exc
        model = str(form.get("model") or "") or None
        # extension (Ollama's keep_alive): how long the model stays loaded after this request
        request_keep_alive(form.get("keep_alive"), ("stt", "diarization", "separation", "denoise", "lid"))
        diarize = await _diarize_options(form, fmt, model, task)
        separate = separate_options(form)
        denoise = parse_denoise(form.get("denoise"), strength=form.get("denoise_strength"))
        language = str(form.get("language") or "") or None
        # extension: model "auto" without a language identifies the language and picks the recogniser (bolo.lid)
        routing = task == "transcribe" and is_routing_request(core, model, language)
        spec = core.resolve(None if routing else model, "stt")
        if denoise is not None and denoise.enabled and denoise.model:
            try:
                core.resolve(denoise.model, "denoise")  # unknown denoiser: fail before reading the upload
            except BoloError as exc:
                exc.param = "denoise"
                raise
        if diarize is not None:
            core.resolve(diarize.model, "diarization")  # unknown diarization model: fail before reading the upload
        if separate is not None:
            if diarize is None:
                raise InvalidRequest(
                    "separate_overlaps needs diarization: add diarize=true or response_format=diarized_json",
                    param="separate_overlaps",
                )
            core.resolve(separate.model, "separation")
        trace.model = spec.id
        samples, rate = await read_upload(core, file)
        trace.restart()
        if routing:
            decision = await route_language(core, samples, rate, str(model), trace)
            trace.extra["route"] = decision.to_dict()
            spec = core.resolve(decision.model, "stt")
            trace.model = spec.id
            language = decision.steer
        vad_raw = form.get("vad_filter")
        try:
            decode = decode_options_from_form(form)
        except OptionError as exc:
            raise InvalidRequest(str(exc), param=exc.name) from exc
        decode.pop("vad_filter", None)  # read as before, below
        kwargs: dict[str, Any] = {
            "model": spec.id,
            "language": language,
            "task": task,
            "prompt": str(form.get("prompt") or "") or None,
            "word_timestamps": "word" in granularities and fmt == "verbose_json",
            "temperature": temperature,
            "hotwords": str(form.get("hotwords") or "") or None,
            "vad_filter": parse_bool(vad_raw) if vad_raw not in (None, "") else None,
            "trace": trace,
            "diarize": diarize,
            "separate": separate,
            "options": decode,
            "denoise": denoise,
        }
    except BoloError as exc:
        trace.finish("error", exc.message)
        if trace.model:
            core.traces.record(trace)
        raise
    headers = {"x-request-id": trace.id}
    routed = trace.extra.get("route")
    if isinstance(routed, dict):
        headers.update(route_headers(routed))
    if not stream:
        transcript = await run_unless_disconnected(request, core.transcribe(samples, rate, **kwargs))
        if transcript is None:
            return Response(status_code=499)  # the client went away; nobody reads this
        headers["openai-processing-ms"] = str(round(trace.total_ms or 0))
        headers.update(fallback_headers(trace))
        return _render(transcript, fmt, task, granularities, headers)

    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    first = True

    counter = 0

    async def on_segment(seg: Segment) -> None:
        nonlocal first, counter
        delta = seg.text if first else " " + seg.text
        first = False
        if seg.speaker:  # diarized streams carry whole segments, as gpt-4o-transcribe-diarize does
            await queue.put(sse(_diarized_segment(counter, seg)))
            counter += 1
        else:
            await queue.put(sse({"type": "transcript.text.delta", "delta": delta}))

    async def run() -> None:
        try:
            t = await core.transcribe(samples, rate, on_segment=on_segment, **kwargs)
            done: dict[str, Any] = {"type": "transcript.text.done", "text": t.text, "usage": _usage(t.duration)}
            if t.separation is not None:
                done["separation"] = t.separation
            await queue.put(sse(done))
        except BoloError as exc:
            await queue.put(
                sse(
                    {
                        "type": "error",
                        "error": {"message": exc.message, "type": exc.type, "code": exc.code, "param": exc.param},
                    }
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(
                sse(
                    {
                        "type": "error",
                        "error": {"message": str(exc), "type": "server_error", "code": "internal_error", "param": None},
                    }
                )
            )
        finally:
            await queue.put(None)

    async def body() -> AsyncIterator[bytes]:
        task_ = asyncio.create_task(run())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not task_.done():
                task_.cancel()  # client went away: drop queued decode work
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task_

    return StreamingResponse(body(), media_type="text/event-stream", headers={**headers, "Cache-Control": "no-cache"})


@router.post("/v1/audio/transcriptions", tags=["openai"], summary="Transcribe audio")
async def create_transcription(request: Request) -> Response:
    """Multipart form: ``file``, ``model``, optional ``language``, ``prompt``,
    ``response_format`` (json, text, srt, vtt, verbose_json), ``temperature``,
    ``timestamp_granularities[]`` (word, segment) and ``stream``.

    Extensions: ``vad_filter`` (force or disable silence removal), ``vad_parameters`` (an object: ``threshold``,
    ``min_speech_duration_ms``, ``max_speech_duration_s``, ``min_silence_duration_ms``, ``speech_pad_ms``),
    ``chunk_length`` and ``hotwords``; Whisper decoding: ``condition_on_previous_text``, ``repetition_penalty``,
    ``no_repeat_ngram_size``, ``compression_ratio_threshold``, ``log_prob_threshold``, ``no_speech_threshold``,
    ``temperature_increment_on_fallback`` and ``hallucination_silence_threshold``; ``max_repeats`` (the repetition
    guard: a phrase repeated back to back more often than this is cut to one copy; 0 turns it off). Options a model
    cannot honour are listed in the ``x-bolo-ignored-options`` header. With diarization, ``separate_overlaps=true``
    transcribes overlapped speech from one separated track per speaker.
    ``denoise`` runs a speech-enhancement model first: ``true``, ``false``, a strength between 0 and 1 (the
    denoised audio mixed with the original) or a denoise model id; ``denoise_strength`` sets the strength
    separately. Unset: the model's pin, then the server's ``denoise`` setting (off by default). The response
    header ``x-bolo-denoise`` names what ran. Denoising can make clean audio worse.
    """
    return await _transcribe_endpoint(request, "transcribe")


@router.post("/v1/audio/translations", tags=["openai"], summary="Translate audio to English")
async def create_translation(request: Request) -> Response:
    """Same form as transcriptions; needs a multilingual Whisper model."""
    return await _transcribe_endpoint(request, "translate")


@router.post("/v1/audio/language", tags=["native"], summary="Identify the spoken language")
async def identify_language(request: Request) -> Response:
    """Extension. Multipart form: ``file``, optional ``model`` (a language-identification model; default
    ``default_lid``), ``top_k`` (how many languages to list, default 5) and ``keep_alive``.

    Returns the most likely language with its probability, the ``top_k`` most likely ones, and ``recogniser``:
    the speech-to-text model a transcription with ``model=auto`` would be sent to for this audio (routing, see
    ``POST /v1/audio/transcriptions``), whether it is installed, and the recogniser bolo recommends for the
    language when that one is not installed."""
    from bolo.lid import decide

    core = get_core(request)
    form = await request.form()
    file = form.get("file")
    if file is None or isinstance(file, str):
        raise InvalidRequest("a multipart 'file' field with the audio is required", param="file")
    top_k = _int_field(form, "top_k")
    if top_k is not None and not 1 <= top_k <= 500:
        raise InvalidRequest("top_k must be between 1 and 500", param="top_k")
    request_keep_alive(form.get("keep_alive"), ("lid",))
    spec = core.resolve(str(form.get("model") or "") or None, "lid")
    samples, rate = await read_upload(core, file)
    trace = Trace("lid", "http", spec.id)
    result = await run_unless_disconnected(request, core.identify_language(samples, rate, spec.id, trace))
    if result is None:
        return Response(status_code=499)  # the client went away; nobody reads this
    body = result.to_dict(top_k or 5)
    decision = decide(core, result)
    body["recogniser"] = {
        "model": decision.model,
        "reason": decision.reason,
        "language": decision.steer,
        "recommended": decision.recommended,
        "detail": decision.detail,
    }
    headers = {"x-request-id": trace.id, "x-bolo-model": spec.id}
    return JSONResponse(body, headers=headers)


@router.post("/v1/audio/alignments", tags=["native"], summary="Align a text to its audio (word timestamps)")
async def align_text(request: Request) -> Response:
    """Extension: forced alignment. Multipart form: ``file`` (the audio), ``text`` (what is said in it), optional
    ``model`` (an alignment model; default ``default_align``), ``language`` and ``keep_alive``.

    Returns every word of ``text`` (cut at whitespace, and at each character of scripts written without spaces)
    with ``start`` and ``end`` in seconds and ``score``, the mean probability the aligner gave the word's letters
    where it put them (0..1; ``null`` for a word with no letter the model knows, such as a lone dash, which gets
    the time where it stands). A text with more sounds than the audio has 20 ms frames is a 400 error."""
    core = get_core(request)
    form = await request.form()
    file = form.get("file")
    if file is None or isinstance(file, str):
        raise InvalidRequest("a multipart 'file' field with the audio is required", param="file")
    text = str(form.get("text") or "")
    if not text.strip():
        raise InvalidRequest("a 'text' field with the words spoken in the audio is required", param="text")
    request_keep_alive(form.get("keep_alive"), ("align",))
    spec = core.resolve(str(form.get("model") or "") or None, "align")
    language = str(form.get("language") or "") or None
    samples, rate = await read_upload(core, file)
    trace = Trace("align", "http", spec.id)
    result = await run_unless_disconnected(
        request, core.align(samples, rate, text, language=language, model=spec.id, trace=trace)
    )
    if result is None:
        return Response(status_code=499)  # the client went away; nobody reads this
    body = {
        "model": result.model,
        "text": result.text,
        "language": result.language,
        "duration": round(result.duration, 3),
        "words": [
            {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3), "score": w.score} for w in result.words
        ],
    }
    headers = {"x-request-id": trace.id, "x-bolo-model": spec.id}
    return JSONResponse(body, headers=headers)


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    input: str
    voice: str | dict[str, Any] | None = Field(
        default=None,
        description="A voice of the model, or a saved voice (/api/v1/voices) for a model that clones voices: by "
        'id (`"priya"`) or as OpenAI\'s custom-voice object (`{"id": "priya"}`, which prefers the saved voice). For a '
        'model that blends voices (Kokoro), a blend of its voices: `"af_bella+af_sky"` or `"af_bella(0.7)+af_sky(0.3)"` '
        "(weights normalised), or a saved blend's id.",
    )
    instructions: str | None = Field(
        default=None,
        max_length=4096,
        description="Direction for the delivery in plain words (style, emotion, accent, the voice itself), for models "
        "that follow it (`features` has `instructions`: Qwen3-TTS CustomVoice, CosyVoice3, FireRedTTS3 Instruct, "
        "Maya1, Parler-TTS, …). Other models speak as usual; `x-bolo-instructions` says `applied` or `ignored`.",
    )
    response_format: str = "mp3"
    speed: float = 1.0
    stream_format: str = "audio"
    sample_rate: int | None = Field(default=None, ge=8000, le=48000, description="Extension: resample the output.")
    language: str | None = Field(default=None, description="Extension: language hint for text normalisation.")
    detect_language: bool = Field(
        default=True,
        description="Extension: without `language`, guess the language of a multilingual model from the text and, "
        "unless a voice of the model is named, choose a voice that speaks it (`x-bolo-language` says what was "
        "decided). An explicit voice or language is never changed.",
    )
    keep_alive: str | float | None = Field(
        default=None,
        description="Extension (Ollama's keep_alive): how long the voice stays loaded after this request: seconds or "
        "a duration ('10m'); 0 unloads it when the request ends; -1 keeps it loaded.",
    )
    normalize: bool | Literal["uccharan", "light", "model", "none"] = Field(
        default=True,
        description="Extension: text normalisation. true: the voice's default (uccharan's normalise-only mode for voices "
        "that read text, when installed); `uccharan`, `light` (bolo's own), `model` (leave it to the model) or "
        "false/`none`.",
    )
    input_format: Literal["text", "ssml", "markdown"] = Field(
        default="text",
        description='Extension: `ssml` accepts <speak>, <sub alias>, <phoneme alphabet="ipa" ph>, <say-as>, <break>, '
        "<lang>, <p>, <s>. In every format, Kokoro-style [word](/ipa/) sets a word's pronunciation.",
    )
    wordlists: list[str] | None = Field(
        default=None,
        description="Extension: pronunciation word lists to apply, highest priority first (names from "
        "/api/v1/wordlists). Default: the lists pinned to the model, then `default`; [] applies none.",
    )
    pronunciations: list[dict[str, Any]] | None = Field(
        default=None,
        description="Extension: inline word-list entries [{word, ipa?, say?, lang?, match?}], applied before any list.",
    )
    romanised: bool | Literal["off", "auto", "on"] | None = Field(
        default=None,
        validation_alias=AliasChoices("romanised", "romanized"),
        description='Extension: write an Indian language typed in Latin letters ("mujhe kal office jana hai") in the '
        "voice's script before it is read. `auto`: when the voice speaks an Indian language and the text is mostly "
        "Latin script; `on` (or true): always for such a voice, 424 `component_missing` when uccharan's transliterator "
        "or the language's pack is not installed; `off` (or false). Default: the model's pinned setting, else the "
        "server's `romanised` setting.",
    )
    phonemizer: str | None = Field(
        default=None,
        description="Extension: who turns text into phonemes for voices that read phonemes: auto, uccharan "
        "(= uccharan:compatible), uccharan:improved or espeak-ng. Default: the voice's pinned choice, else the "
        "server's `phonemizer` setting.",
    )
    reference_audio: str | None = Field(
        default=None,
        description="Extension: a recording of the voice to imitate, base64 or a data URL (WAV, FLAC, MP3, …), for "
        "models that clone voices (`features` has `voice-cloning`). Up to 30 s are used.",
    )
    reference_text: str | None = Field(
        default=None, description="Extension: the transcript of `reference_audio` (some cloning models use it)."
    )
    options: dict[str, Any] | None = Field(
        default=None,
        description="Extension: model-specific request options, for example Tontaube's `style` (audiobook, "
        "conversational, agentic) and sliders (`rate`, `pause`, `pitch_range`, `pitch_level`, `legato`). A model "
        "answers 400 for an option it does not read.",
    )
    timestamps: bool = Field(
        default=False,
        description="Extension: word timestamps of the speech (captioned speech), found by aligning each sentence's "
        "text to its audio with the alignment model (`align_model`, default `default_align`). With "
        "`stream_format: sse`, a `speech.words` event follows the audio of each sentence; POST "
        "/v1/audio/speech/captioned returns the audio and all its words in one JSON answer.",
    )
    align_model: str | None = Field(
        default=None, description="Extension: the alignment model for `timestamps` (default `default_align`)."
    )


class Captions:
    """Word times of synthesised speech: the audio of each piece of text (from a chunk marked ``first_of_text``
    to the next) is aligned to that text (bolo.alignment), its times counted from the start of the whole answer.
    The text is the one the voice read (after normalisation: "25" is "twenty-five")."""

    def __init__(self, core: Bolo, model: str, language: str | None) -> None:
        self.core = core
        self.model = model
        self.language = language
        self.offset = 0.0  # seconds of audio before the piece being collected
        self.text = ""
        self.audio: list[FloatArray] = []
        self.rate = 0

    @staticmethod
    def check(core: Bolo, model: str | None) -> str:
        """The alignment model to use, or a 4xx before any audio is made when it cannot be used."""
        spec = core.resolve(model, "align")
        if not core.manager.is_installed(spec) and not core.settings.auto_pull:
            raise ModelNotInstalled(
                f"word timestamps of speech need the alignment model '{spec.id}', which is not installed; "
                f"run `bolo pull {spec.id}`",
                param="align_model",
            )
        return spec.id

    async def add(self, chunk: SynthChunk) -> list[dict[str, Any]]:
        """Collect a chunk; when it starts a new piece of text, the words of the piece before it."""
        words: list[dict[str, Any]] = []
        if chunk.first_of_text and self.audio:
            words = await self.flush()
        if not self.audio:
            self.text = chunk.text
        self.rate = chunk.sample_rate
        self.audio.append(np.asarray(chunk.samples, dtype=np.float32))
        return words

    async def flush(self) -> list[dict[str, Any]]:
        """The words of the piece collected so far."""
        if not self.audio:
            return []
        samples = np.concatenate(self.audio)
        text, offset = self.text, self.offset
        self.audio, self.text = [], ""
        self.offset += len(samples) / self.rate
        if not text.strip() or not len(samples):
            return []
        trace = Trace("align", "internal", self.model)
        result = await self.core.align(
            samples, self.rate, text, language=self.language, model=self.model, trace=trace, record=False
        )
        return [
            {"word": w.word, "start": round(w.start + offset, 3), "end": round(w.end + offset, 3)} for w in result.words
        ]


class SpeechStream:
    """Drives synthesis for one HTTP/WebSocket response and owns the trace bookkeeping."""

    def __init__(self, core: Bolo, trace: Trace, encoder: StreamEncoder | None, **kwargs: Any) -> None:
        self.core = core
        self.trace = trace
        self.encoder = encoder
        self._gen: AsyncGenerator[SynthChunk, None] = core.synthesize(trace=trace, **kwargs)
        self._encode_s = 0.0
        self._encode_started: float | None = None
        self._completed = False

    def encode(self, chunk: SynthChunk, last: bool = False) -> bytes:
        assert self.encoder is not None
        t = self.trace.now()
        if self._encode_started is None:
            self._encode_started = t
        data = self.encoder.feed(chunk.samples)
        self._encode_s += self.trace.now() - t
        return data

    def finish_encoding(self) -> bytes:
        assert self.encoder is not None
        t = self.trace.now()
        data = self.encoder.finish()
        self._encode_s += self.trace.now() - t
        return data

    async def chunks(self) -> AsyncIterator[SynthChunk]:
        try:
            async for chunk in self._gen:
                yield chunk
        except BoloError as exc:
            self.trace.finish("error", exc.message)
            raise
        except (asyncio.CancelledError, GeneratorExit):
            self.trace.finish("cancelled", "client disconnected")
            raise
        except Exception as exc:
            self.trace.finish("error", f"{type(exc).__name__}: {exc}")
            raise

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self._gen.aclose()
        self.complete()

    def complete(self) -> None:
        if self._completed:
            return
        self._completed = True
        if self._encode_started is not None:
            self.trace.add_phase("encode", self._encode_started, self._encode_started + self._encode_s)
        self.core.complete_tts(self.trace)


@router.post("/v1/audio/speech", tags=["openai"], summary="Synthesise speech")
async def create_speech(request: Request, body: SpeechRequest) -> Response:
    """Returns audio as it is synthesised (chunked transfer).

    ``response_format``: mp3 (default), opus, aac, flac, wav, pcm, plus the
    telephony extensions ``mulaw`` and ``alaw`` (8 kHz G.711).
    ``stream_format``: ``audio`` (default) or ``sse`` (``speech.audio.delta`` events).
    """
    core = get_core(request)
    trace = Trace("tts", "http", "")
    try:
        fmt = normalize_format(body.response_format)
        if body.stream_format not in ("audio", "sse"):
            raise InvalidRequest("stream_format must be 'audio' or 'sse'", param="stream_format")
        voice, custom_voice = speech_voice(body.voice)
        request_keep_alive(body.keep_alive, ("tts",))
        reference = decode_reference(body.reference_audio) if body.reference_audio else None
        model = body.model
        saved = core.voices.find(voice) if voice and reference is None and model is None else None
        if saved is not None and saved.default_model:
            model = saved.default_model  # the saved voice's own model (synthesize decides the same way)
        spec = core.resolve(model, "tts")
        trace.model = spec.id
        encoder = StreamEncoder(fmt, spec.sample_rate, body.sample_rate)
        if body.timestamps and body.stream_format != "sse":
            raise InvalidRequest(
                "timestamps need stream_format 'sse' (or POST /v1/audio/speech/captioned)", param="timestamps"
            )
        captions = Captions(core, Captions.check(core, body.align_model), body.language) if body.timestamps else None
    except BoloError as exc:
        trace.finish("error", exc.message)
        if trace.model:
            core.traces.record(trace)
        raise
    stream = SpeechStream(
        core, trace, encoder,
        text=body.input, model=spec.id, voice=voice, speed=body.speed, language=body.language, normalize=body.normalize,
        phonemizer=body.phonemizer, reference_audio=reference, reference_text=body.reference_text,
        wordlists=body.wordlists, pronunciations=body.pronunciations, input_format=body.input_format,
        romanised=body.romanised, custom_voice=custom_voice, options=body.options,
        detect_language=body.detect_language, instructions=body.instructions,
    )  # fmt: skip
    chunks = stream.chunks()
    try:
        # Pull the first chunk before committing to a 200 so errors keep their status code.
        first = await chunks.__anext__()
    except StopAsyncIteration:
        first = None
    except BaseException:
        await stream.aclose()
        raise
    if encoder.model_rate != (first.sample_rate if first else encoder.model_rate):
        # the catalog's sample rate was wrong for this model: rebuild the encoder with the real one
        assert first is not None
        encoder = stream.encoder = StreamEncoder(fmt, first.sample_rate, body.sample_rate)
    as_sse = body.stream_format == "sse"

    async def gen() -> AsyncIterator[bytes]:
        try:
            chunk = first
            while chunk is not None:
                if captions is not None:
                    words = await captions.add(chunk)  # the sentence before this one is complete
                    if words:
                        yield sse({"type": "speech.words", "words": words})
                data = stream.encode(chunk)
                if data:
                    trace.mark_ttfa()
                    yield (
                        sse({"type": "speech.audio.delta", "audio": base64.b64encode(data).decode()})
                        if as_sse
                        else data
                    )
                try:
                    chunk = await chunks.__anext__()
                except StopAsyncIteration:
                    chunk = None
            tail = stream.finish_encoding()
            if tail:
                trace.mark_ttfa()
                yield sse({"type": "speech.audio.delta", "audio": base64.b64encode(tail).decode()}) if as_sse else tail
            if captions is not None:
                try:
                    words = await captions.flush()
                except BoloError as exc:
                    yield sse({"type": "error", "error": {"message": exc.message, "type": exc.type, "code": exc.code}})
                    words = []
                if words:
                    yield sse({"type": "speech.words", "words": words})
            if as_sse:
                n_in = len(body.input)
                yield sse(
                    {
                        "type": "speech.audio.done",
                        "usage": {"input_tokens": n_in, "output_tokens": 0, "total_tokens": n_in},
                    }
                )
        except asyncio.CancelledError:
            trace.finish("cancelled", "client disconnected")
            raise
        finally:
            await stream.aclose()

    headers = {
        "x-request-id": trace.id,
        "x-sample-rate": str(encoder.sample_rate),
        "Cache-Control": "no-cache",
        # which model answered (a fallback is decided before the first audio, so it is known here)
        **{k: v for k, v in fallback_headers(trace).items() if k != "x-bolo-estimated-cost"},
    }
    media = "text/event-stream" if as_sse else encoder.media_type
    return StreamingResponse(gen(), media_type=media, headers=headers)


@router.post("/v1/audio/speech/captioned", tags=["native"], summary="Synthesise speech with word timestamps")
async def create_captioned_speech(request: Request, body: SpeechRequest) -> Response:
    """Extension (captioned speech): the same request as ``/v1/audio/speech``, answered when synthesis is done
    with one JSON object: ``audio`` (base64, in ``response_format``), ``sample_rate``, ``model``, ``aligner`` and
    ``words`` (``word``, ``start``, ``end`` in seconds of that audio). The words are the text the voice read,
    aligned to the audio sentence by sentence (``align_model``, default ``default_align``)."""
    core = get_core(request)
    trace = Trace("tts", "http", "")
    try:
        fmt = normalize_format(body.response_format)
        if fmt in ("pcm", "mulaw", "alaw"):
            raise InvalidRequest("captioned speech needs a format with a header (wav, mp3, opus, aac, flac)",
                                 param="response_format")  # fmt: skip
        voice, custom_voice = speech_voice(body.voice)
        request_keep_alive(body.keep_alive, ("tts", "align"))
        reference = decode_reference(body.reference_audio) if body.reference_audio else None
        model = body.model
        saved = core.voices.find(voice) if voice and reference is None and model is None else None
        if saved is not None and saved.default_model:
            model = saved.default_model
        spec = core.resolve(model, "tts")
        trace.model = spec.id
        aligner = Captions.check(core, body.align_model)
    except BoloError as exc:
        trace.finish("error", exc.message)
        if trace.model:
            core.traces.record(trace)
        raise
    stream = SpeechStream(
        core, trace, None,
        text=body.input, model=spec.id, voice=voice, speed=body.speed, language=body.language, normalize=body.normalize,
        phonemizer=body.phonemizer, reference_audio=reference, reference_text=body.reference_text,
        wordlists=body.wordlists, pronunciations=body.pronunciations, input_format=body.input_format,
        romanised=body.romanised, custom_voice=custom_voice, options=body.options,
        detect_language=body.detect_language,
    )  # fmt: skip
    captions = Captions(core, aligner, body.language)
    words: list[dict[str, Any]] = []
    audio: list[FloatArray] = []
    rate = spec.sample_rate

    async def run() -> None:
        nonlocal rate
        try:
            async for chunk in stream.chunks():
                words.extend(await captions.add(chunk))
                audio.append(np.asarray(chunk.samples, dtype=np.float32))
                rate = chunk.sample_rate
            words.extend(await captions.flush())
        finally:
            await stream.aclose()

    if await run_unless_disconnected(request, _done(run())) is None:
        return Response(status_code=499)
    samples = np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)
    encoder = StreamEncoder(fmt, rate, body.sample_rate)
    data = encoder.feed(samples) + encoder.finish()
    body_out = {
        "model": spec.id,
        "aligner": aligner,
        "format": fmt,
        "sample_rate": encoder.sample_rate,
        "duration": round(len(samples) / rate, 3) if rate else 0.0,
        "audio": base64.b64encode(data).decode(),
        "words": words,
    }
    headers = {"x-request-id": trace.id, **fallback_headers(trace)}
    return JSONResponse(body_out, headers=headers)


async def _done(coro: Coroutine[Any, Any, None]) -> bool:
    await coro
    return True


def speech_voice(value: str | dict[str, Any] | None) -> tuple[str | None, bool]:
    """``voice`` of a speech request: ``(name, custom)``; ``custom`` for OpenAI's ``{"id": ...}`` object."""
    if isinstance(value, dict):
        vid = value.get("id")
        if not isinstance(vid, str) or not vid.strip():
            raise InvalidRequest('a voice object needs an "id"', param="voice")
        return vid.strip(), True
    return value, False


def decode_reference(value: str) -> bytes:
    """``reference_audio`` of a speech request: plain base64 or a ``data:audio/...;base64,`` URL."""
    text = value.strip()
    if text.startswith("data:"):
        _, sep, text = text.partition(",")
        if not sep:
            raise InvalidRequest("reference_audio is a data URL without data", param="reference_audio")
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequest("reference_audio must be base64 or a base64 data URL", param="reference_audio") from exc


def now_ms() -> int:
    return int(time.time() * 1000)
