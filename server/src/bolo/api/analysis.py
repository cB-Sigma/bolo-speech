"""Audio analysis routes: speech enhancement on its own (``/v1/audio/denoise``, with ElevenLabs' voice-isolation
route as an alias), sound-event tags (``/v1/audio/tags``), speech emotion (``/v1/audio/emotion``), decisions
(``/v1/audio/decisions``: typed questions about a recording or a text), and classification against the caller's
own classes (``/v1/audio/classify``: zero-shot with CLAP, or a classifier trained from examples with
``/v1/audio/classifiers``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

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


def _float(form: Any, name: str, default: float | None = None) -> float | None:
    raw = form.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(str(raw))
    except ValueError as exc:
        raise InvalidRequest(f"{name} must be a number", param=name) from exc


def _top_k(form: Any, default: int) -> int:
    raw = form.get("top_k")
    try:
        value = default if raw in (None, "") else int(str(raw))
    except ValueError as exc:
        raise InvalidRequest("top_k must be a whole number", param="top_k") from exc
    if not 1 <= value <= 1000:
        raise InvalidRequest("top_k must be between 1 and 1000", param="top_k")
    return value


async def _classify(request: Request, task: str) -> tuple[Any, Any]:
    from bolo.classify import classify

    core = get_core(request)
    form = await request.form()
    file = _upload(form, "file")
    request_keep_alive(form.get("keep_alive"), (task,))
    spec = core.resolve(str(form.get("model") or "") or None, task)
    window, hop = _float(form, "window"), _float(form, "hop")
    samples, rate = await read_upload(core, file)
    trace = Trace(task, "http", spec.id)
    trace.audio_seconds = len(samples) / rate if rate else 0.0
    try:
        result = await classify(core, samples, rate, spec.id, task, trace, window, hop)
        trace.extra[task] = result.scores[0][0] if result.scores else None
        trace.finish("ok")
    except Exception as exc:
        trace.finish("error", exc.message if isinstance(exc, BoloError) else f"{type(exc).__name__}: {exc}")
        raise
    finally:
        core.traces.record(trace)
    return (form, result), {"x-request-id": trace.id, "x-bolo-model": spec.id}


@router.post("/v1/audio/tags", tags=["native"], summary="Sound events and audio tags (music, speech, ...)")
async def audio_tags(request: Request) -> JSONResponse:
    """Extension. Which sounds a recording holds (AudioSet's 527 classes: speech, music and its genres and
    instruments, laughter, applause, dog, siren, engine, ...), each with its probability. Multipart form: ``file``,
    ``model`` (an ``audio-tagging`` model: ``ced-tiny``, ``ced-mini``, ``ced-small``, ``ced-base``,
    ``zipformer-audio-tagging``, ``zipformer-small-audio-tagging``; default ``default_audio_tagging``), ``top_k``
    (default 10), ``threshold`` (leave out tags below this probability, default 0.1), ``labels`` (only these
    classes, comma-separated: ``Speech,Music``), ``window`` and ``hop`` (seconds: tag the recording window by window
    too, for sound event detection over time) and ``keep_alive``. Classes are independent: probabilities do not
    sum to 1."""
    (form, result), headers = await _classify(request, "audio-tagging")
    from bolo.classify import tags_body

    labels = [x.strip() for raw in form.getlist("labels") for x in str(raw).split(",") if x.strip()] or None
    threshold = _float(form, "threshold", 0.1)
    assert threshold is not None
    return JSONResponse(tags_body(result, _top_k(form, 10), threshold, labels), headers=headers)


@router.post("/v1/audio/emotion", tags=["native"], summary="Speech emotion")
async def audio_emotion(request: Request) -> JSONResponse:
    """Extension. How the speaker sounds: ``emotion`` (the most likely class) with its probability and every
    class's probability. Multipart form: ``file``, ``model`` (an ``emotion`` model: ``emotion2vec-plus-base``,
    ``emotion2vec-plus-large``; default ``default_emotion``), ``top_k`` (classes listed, default all), ``window``
    and ``hop`` (seconds: also window by window) and ``keep_alive``. SenseVoice's own emotion tags come back with
    its transcriptions (``verbose_json``)."""
    (form, result), headers = await _classify(request, "emotion")
    from bolo.classify import emotion_body

    return JSONResponse(emotion_body(result, _top_k(form, 1000)), headers=headers)


@router.post(
    "/v1/audio/decisions", tags=["native"], summary="Typed questions about a recording or a text (decision models)"
)
async def audio_decisions(request: Request) -> JSONResponse:
    """Extension. Ask a decision model named questions about a recording, a text or both, and get typed answers:
    ``yes_no`` (true or false, with the probability of yes), ``choice`` (one of the ``options``, with each option's
    probability) or ``score`` (a level of ``levels``, lowest first, with the expected level). The model reads the
    state and the questions in one pass and answers from its distribution over the options: it writes no text, so an
    answer is always one of the options.

    JSON body or multipart form. ``questions`` (required): an object mapping an id to ``{"type": "yes_no" | "choice"
    | "score", "question": "...", "options": [...] | {name: description}, "levels": [...]}`` (in a form, as JSON
    text). ``file`` (multipart) or ``audio`` (JSON: base64 of an audio file): the recording, at most 30 s for
    ``d1-omni-600m-llamacpp``. ``text``: text about which the questions are asked (a transcript, a message, JSON),
    alone or with the recording. ``model`` (a ``decision`` model; default ``default_decision``) and ``keep_alive``.
    See docs/api.md, "Decisions"."""
    import asyncio
    import base64
    import binascii

    from bolo.api.common import read_upload
    from bolo.decide import body, decide, parse_questions

    core = get_core(request)
    samples: FloatArray | None = None
    rate: int | None = None
    if "application/json" in (request.headers.get("content-type") or ""):
        try:
            doc = await request.json()
        except ValueError as exc:
            raise InvalidRequest(f"the body is not valid JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise InvalidRequest("the body must be a JSON object")
        fields: dict[str, Any] = doc
        if doc.get("audio"):
            try:
                data = base64.b64decode(str(doc["audio"]).split(",", 1)[-1], validate=True)
            except (binascii.Error, ValueError) as exc:
                raise InvalidRequest("audio must be base64 (or a data: URL) of an audio file", param="audio") from exc
            if len(data) > core.settings.max_upload_mb * 1024 * 1024:
                raise InvalidRequest(f"the audio is larger than the {core.settings.max_upload_mb} MB limit",
                                     param="audio")  # fmt: skip
            from bolo.audio import decode_audio

            samples, rate = await asyncio.to_thread(decode_audio, data)
    else:
        form = await request.form()
        fields = {k: form.get(k) for k in ("questions", "text", "model", "keep_alive")}
        file = form.get("file")
        if file is not None and not isinstance(file, str):
            samples, rate = await read_upload(core, file)
    questions = parse_questions(fields.get("questions"))
    request_keep_alive(fields.get("keep_alive"), ("decision",))
    spec = core.resolve(str(fields.get("model") or "") or None, "decision")
    text = fields.get("text")
    if text is not None and not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)  # a JSON state is given to the model as JSON text
    trace = Trace("decision", "http", spec.id)
    trace.audio_seconds = len(samples) / rate if samples is not None and rate else 0.0
    try:
        result = await decide(core, spec.id, text, samples, rate, questions, trace)
        trace.finish("ok")
    except Exception as exc:
        trace.finish("error", exc.message if isinstance(exc, BoloError) else f"{type(exc).__name__}: {exc}")
        raise
    finally:
        core.traces.record(trace)
    return JSONResponse(body(result), headers={"x-request-id": trace.id, "x-bolo-model": spec.id})


@router.post(
    "/v1/audio/classify",
    tags=["native"],
    summary="Classify audio against your own labels (zero-shot) or a trained classifier",
)
async def audio_classify(request: Request) -> JSONResponse:
    """Extension. Score a recording against classes the caller names. Multipart form: ``file``; then either
    ``labels`` (the classes in plain words: a JSON array, one per line, comma-separated, or the field repeated;
    "angry voice", "hold music", "a dog barking") with ``model`` (an ``audio-classify`` CLAP model; default
    ``default_audio_classify``) and ``prompt`` (the sentence each label is put into, ``{}`` marking where; default
    the model's, "This is a sound of {}."; an empty prompt uses the labels as they are), or ``classifier`` (a
    classifier trained with ``POST /v1/audio/classifiers``; its id also works as ``model``). Also ``top_k`` (scores
    listed, default all), ``window`` and ``hop`` (seconds: classify window by window too) and ``keep_alive``.
    Zero-shot scores are a softmax over the labels (they sum to 1) with ``similarity``, the cosine between the
    audio and the label's sentence; a trained classifier gives its head's probabilities."""
    from bolo.classifiers import TASK, body, classify_request, parse_labels

    core = get_core(request)
    form = await request.form()
    file = _upload(form, "file")
    request_keep_alive(form.get("keep_alive"), (TASK,))
    labels = parse_labels(form.getlist("labels") + form.getlist("label"))
    classifier = str(form.get("classifier") or "").strip() or None
    model = str(form.get("model") or "").strip() or None
    prompt = form.get("prompt")
    window, hop = _float(form, "window"), _float(form, "hop")
    samples, rate = await read_upload(core, file)
    trace = Trace(TASK, "http", classifier or model or core.settings.default_audio_classify)
    trace.audio_seconds = len(samples) / rate if rate else 0.0
    try:
        result = await classify_request(
            core, samples, rate, trace, model=model, classifier=classifier, labels=labels,
            prompt=None if prompt is None else str(prompt), window=window, hop=hop,
        )  # fmt: skip
        trace.model = result.classifier or result.model
        trace.extra[TASK] = result.scores[0][0] if result.scores else None
        trace.finish("ok")
    except Exception as exc:
        trace.finish("error", exc.message if isinstance(exc, BoloError) else f"{type(exc).__name__}: {exc}")
        raise
    finally:
        core.traces.record(trace)
    headers = {"x-request-id": trace.id, "x-bolo-model": result.model}
    if result.classifier:
        headers["x-bolo-classifier"] = result.classifier
    top_k = _top_k(form, 1000) if form.get("top_k") not in (None, "") else None
    return JSONResponse(body(result, top_k), headers=headers)


@router.post("/v1/audio/classifiers", tags=["native"], summary="Train a classifier from labelled examples")
async def train_classifier(request: Request) -> JSONResponse:
    """Extension. Train an audio classifier from examples and save it as a local model. Multipart form: ``name``
    (its id), ``file`` fields (the example clips) each followed by its ``label`` field (the class; the labels pair
    with the files in order), ``encoder`` (the model whose embeddings it learns from: an ``audio-classify`` CLAP
    model, a ``speaker-embedding`` model or an ``audio-tagging`` model; default ``default_audio_classify``),
    ``head`` (``logistic``, the default, or ``mlp``), ``replace`` (true: overwrite a classifier with that name) and
    ``description``. Returns the classifier with its cross-validated accuracy and warnings (fewer than 10 examples
    in a class, unbalanced classes, low accuracy). A few dozen clips per class train in seconds on a CPU."""
    from bolo.audio import decode_audio
    from bolo.classifiers import train

    core = get_core(request)
    form = await request.form()
    name = str(form.get("name") or "").strip()
    if not name:
        raise InvalidRequest("a 'name' for the classifier is required", param="name")
    files = [f for f in form.getlist("file") if not isinstance(f, str)]
    labels = [str(x).strip() for x in form.getlist("label")]
    if not files:
        raise InvalidRequest("no examples: send 'file' fields, each with a 'label'", param="file")
    if len(labels) != len(files) or not all(labels):
        raise InvalidRequest(
            f"{len(files)} files and {len(labels)} labels: give one non-empty 'label' per 'file'", param="label"
        )
    limit = core.settings.max_upload_mb * 1024 * 1024
    examples = []
    total = 0
    for f, label in zip(files, labels):
        data = await f.read()
        total += len(data)
        if total > limit:
            raise InvalidRequest(
                f"the examples are larger than the {core.settings.max_upload_mb} MB limit", param="file"
            )
        try:
            samples, rate = await asyncio.to_thread(
                decode_audio, data, filename=f.filename, content_type=f.content_type
            )
        except BoloError as exc:
            raise InvalidRequest(f"{f.filename}: {exc.message}", param="file") from exc
        examples.append((label, samples, rate))
    result = await train(
        core, name, examples, encoder=str(form.get("encoder") or "").strip() or None,
        kind=str(form.get("head") or "logistic").strip().lower(), replace=parse_bool(form.get("replace")),
        description=str(form.get("description") or "").strip() or None,
    )  # fmt: skip
    return JSONResponse(result, status_code=201)


@router.get("/v1/audio/classifiers", tags=["native"], summary="List trained classifiers")
async def list_classifiers(request: Request) -> JSONResponse:
    """Extension. The classifiers trained with ``POST /v1/audio/classifiers``: classes, encoder, examples per class,
    cross-validated accuracy and warnings."""
    from bolo.classifiers import list_classifiers as listing

    return JSONResponse({"object": "list", "data": listing(get_core(request))})


@router.get("/v1/audio/classifiers/{classifier_id}", tags=["native"], summary="One trained classifier")
async def get_classifier(request: Request, classifier_id: str) -> JSONResponse:
    """Extension. One trained classifier (as in the list)."""
    from bolo.classifiers import describe

    return JSONResponse(describe(get_core(request), classifier_id))


@router.delete("/v1/audio/classifiers/{classifier_id}", tags=["native"], summary="Delete a trained classifier")
async def delete_classifier(request: Request, classifier_id: str) -> JSONResponse:
    """Extension. Delete a trained classifier (its files and its catalog entry)."""
    from bolo.classifiers import delete

    await delete(get_core(request), classifier_id)
    return JSONResponse({"id": classifier_id, "object": "audio.classifier", "deleted": True})


@router.post("/v1/audio/intent", tags=["native"], summary="Spoken intent and slots (spoken language understanding)")
async def audio_intent(request: Request) -> JSONResponse:
    """Extension. What the speaker of one short utterance wants, straight from the audio: ``intent``
    (``alarm_set``), ``scenario`` and ``action`` (SLURP's two halves of the intent, ``alarm`` and ``set``), ``slots``
    (``[{"type": "time", "value": "seven"}]``, as the model filled them) and ``text`` (the model's own output).
    Multipart form: ``file`` (one utterance, at most 30 s), ``model`` (an ``intent`` model: ``slu-slurp-hubert``;
    default ``default_intent``) and ``keep_alive``. See docs/api.md, "Spoken intent"."""
    from bolo.intent import body, understand

    core = get_core(request)
    form = await request.form()
    file = _upload(form, "file")
    request_keep_alive(form.get("keep_alive"), ("intent",))
    spec = core.resolve(str(form.get("model") or "") or None, "intent")
    samples, rate = await read_upload(core, file)
    trace = Trace("intent", "http", spec.id)
    trace.audio_seconds = len(samples) / rate if rate else 0.0
    try:
        result, duration = await understand(core, samples, rate, spec.id, trace)
        trace.extra["intent"] = result.intent
        trace.finish("ok")
    except Exception as exc:
        trace.finish("error", exc.message if isinstance(exc, BoloError) else f"{type(exc).__name__}: {exc}")
        raise
    finally:
        core.traces.record(trace)
    return JSONResponse(body(result, spec.id, duration), headers={"x-request-id": trace.id, "x-bolo-model": spec.id})


__all__ = ["router"]
