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
from bolo.diarize import DiarizeOptions, KnownSpeaker
from bolo.engines.base import Segment
from bolo.errors import BoloError, InvalidRequest, ModelNotFound
from bolo.keepalive import request_keep_alive
from bolo.obs.trace import Trace
from bolo.separate import SeparateOptions
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
    if word_level:
        speaker_of = {id(w): s.speaker for s in t.segments for w in s.words if s.speaker}
        out["words"] = [
            {
                "word": w.w,
                "start": round(w.t0, 3),
                "end": round(w.t1, 3),
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
                "temperature": 0.0,
                "avg_logprob": s.avg_logprob if s.avg_logprob is not None else 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": s.no_speech_prob if s.no_speech_prob is not None else 0.0,
                **({"speaker": s.speaker} if s.speaker else {}),
                **({"overlap": True} if s.overlap else {}),
                **({"separated": True} if s.separated else {}),
            }
            for i, s in enumerate(t.segments)
        ]
    out["usage"] = _usage(t.duration)
    if t.separation is not None:
        out["separation"] = t.separation
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
        request_keep_alive(form.get("keep_alive"), ("stt", "diarization", "separation"))
        diarize = await _diarize_options(form, fmt, model, task)
        separate = separate_options(form)
        spec = core.resolve(model, "stt")
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
        vad_raw = form.get("vad_filter")
        kwargs: dict[str, Any] = {
            "model": spec.id,
            "language": str(form.get("language") or "") or None,
            "task": task,
            "prompt": str(form.get("prompt") or "") or None,
            "word_timestamps": "word" in granularities and fmt == "verbose_json",
            "temperature": temperature,
            "hotwords": str(form.get("hotwords") or "") or None,
            "vad_filter": parse_bool(vad_raw) if vad_raw not in (None, "") else None,
            "trace": trace,
            "diarize": diarize,
            "separate": separate,
        }
    except BoloError as exc:
        trace.finish("error", exc.message)
        if trace.model:
            core.traces.record(trace)
        raise
    headers = {"x-request-id": trace.id}
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

    Extensions: ``vad_filter`` (force or disable silence removal) and ``hotwords``; with diarization,
    ``separate_overlaps=true`` transcribes overlapped speech from one separated track per speaker.
    """
    return await _transcribe_endpoint(request, "transcribe")


@router.post("/v1/audio/translations", tags=["openai"], summary="Translate audio to English")
async def create_translation(request: Request) -> Response:
    """Same form as transcriptions; needs a multilingual Whisper model."""
    return await _transcribe_endpoint(request, "translate")


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    input: str
    voice: str | dict[str, Any] | None = None
    instructions: str | None = Field(default=None, description="Accepted for compatibility; ignored.")
    response_format: str = "mp3"
    speed: float = 1.0
    stream_format: str = "audio"
    sample_rate: int | None = Field(default=None, ge=8000, le=48000, description="Extension: resample the output.")
    language: str | None = Field(default=None, description="Extension: language hint for text normalisation.")
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
        voice = body.voice.get("id") if isinstance(body.voice, dict) else body.voice
        request_keep_alive(body.keep_alive, ("tts",))
        reference = decode_reference(body.reference_audio) if body.reference_audio else None
        spec = core.resolve(body.model, "tts")
        trace.model = spec.id
        encoder = StreamEncoder(fmt, spec.sample_rate, body.sample_rate)
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
        romanised=body.romanised,
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
