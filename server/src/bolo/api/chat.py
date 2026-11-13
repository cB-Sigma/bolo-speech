"""``/v1/chat/completions``: a speech front end for the operator's language model.

Follows the OpenAI reference for audio in chat completions: ``input_audio``
content parts on the way in, ``modalities: ["text", "audio"]`` with an
``audio`` object (``voice``, ``format``) on the way out. bolo transcribes the
audio parts, forwards an ordinary text request to the configured model, and
synthesises the reply when audio was asked for. Requests that involve no
audio are proxied unchanged, so a client can use one base URL for everything.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from bolo.agents import speak_stream
from bolo.api.common import get_core, sse
from bolo.audio import decode_audio
from bolo.audio.encode import StreamEncoder, encode_audio, normalize_format
from bolo.core import Bolo
from bolo.errors import BoloError, InvalidRequest
from bolo.obs.trace import Trace, preview

router = APIRouter()

#: extension keys bolo reads from the request body and does not forward
EXTENSIONS = ("transcription_model", "speech_model", "transcription_language")


def _audio_format(fmt: str) -> str:
    return "pcm" if fmt == "pcm16" else normalize_format(fmt)


async def transcribe_parts(core: Bolo, body: dict[str, Any], trace: Trace) -> tuple[int, float]:
    """Replace every ``input_audio`` part in ``body["messages"]`` with its transcript. Returns (count, audio seconds)."""
    count, seconds = 0, 0.0
    stt_model = body.get("transcription_model")
    language = body.get("transcription_language")
    for i, message in enumerate(body.get("messages") or []):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        new_parts: list[Any] = []
        for j, part in enumerate(content):
            if not (isinstance(part, dict) and part.get("type") == "input_audio"):
                new_parts.append(part)
                continue
            param = f"messages[{i}].content[{j}].input_audio"
            spec = part.get("input_audio")
            if not isinstance(spec, dict) or not isinstance(spec.get("data"), str):
                raise InvalidRequest("input_audio needs base64 'data' and a 'format'", param=param)
            try:
                raw = base64.b64decode(spec["data"], validate=True)
            except (binascii.Error, ValueError) as exc:
                raise InvalidRequest("input_audio.data is not valid base64", param=param + ".data") from exc
            fmt = str(spec.get("format") or "wav")
            samples, rate = await asyncio.to_thread(decode_audio, raw, filename=f"audio.{fmt}")
            sub = Trace("stt", "chat", "")
            t0 = trace.now()
            transcript = await core.transcribe(samples, rate, model=stt_model, language=language, trace=sub)
            trace.add_phase("decode", t0, trace.now())
            count += 1
            seconds += len(samples) / rate
            new_parts.append({"type": "text", "text": transcript.text})
        if all(isinstance(p, dict) and p.get("type") == "text" for p in new_parts):
            message["content"] = " ".join(
                p["text"] for p in new_parts
            ).strip()  # plain string: understood by every server
        else:
            message["content"] = new_parts
    return count, seconds


def has_audio_parts(body: dict[str, Any]) -> bool:
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "input_audio" for p in content):
            return True
    return False


@router.post(
    "/v1/chat/completions",
    tags=["openai"],
    summary="Chat completions with audio in and out (speech front end to your LLM)",
)
async def chat_completions(request: Request) -> Response:
    """Needs ``llm_base_url``. Audio parts are transcribed, the text request goes to the configured
    model, and the reply is synthesised when ``modalities`` contains ``"audio"``. Requests without
    audio are proxied unchanged. Extensions (not forwarded): ``transcription_model``,
    ``transcription_language``, ``speech_model``."""
    core = get_core(request)
    try:
        body = await request.json()
    except ValueError as exc:
        raise InvalidRequest("the request body must be JSON") from exc
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        raise InvalidRequest("'messages' is required", param="messages")
    llm = core.llm
    _ = llm.base_url  # raises a clear error when no language model is configured
    wants_audio = "audio" in (body.get("modalities") or [])
    audio_in = has_audio_parts(body)
    stream = bool(body.get("stream"))

    if not wants_audio and not audio_in:  # nothing for bolo to do: a transparent proxy
        if not body.get("model") and core.settings.llm_model:
            body["model"] = core.settings.llm_model
        if stream:
            upstream = llm.raw_stream(body)
            first = await upstream.__anext__()  # surface connection and HTTP errors as a proper status

            async def passthrough() -> AsyncIterator[bytes]:
                yield first
                async for chunk in upstream:
                    yield chunk

            return StreamingResponse(
                passthrough(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
            )
        resp = await llm.request(body)
        return Response(
            resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json")
        )

    trace = Trace("agent", "chat", "voice-agent")
    audio_cfg: dict[str, Any] = body["audio"] if isinstance(body.get("audio"), dict) else {}
    try:
        fmt = _audio_format(str(audio_cfg.get("format") or "wav")) if wants_audio else "wav"
        voice = audio_cfg.get("voice")
        if isinstance(voice, dict):
            voice = voice.get("id")
        if wants_audio and not audio_cfg:
            raise InvalidRequest(
                "'audio' (voice and format) is required when modalities contains 'audio'", param="audio"
            )
        tts_spec = core.resolve(body.get("speech_model"), "tts") if wants_audio else None
        n_audio, seconds_in = await transcribe_parts(core, body, trace)
        trace.audio_seconds = seconds_in or None
        trace.mark_final()
        forward = {k: v for k, v in body.items() if k not in ("modalities", "audio", *EXTENSIONS)}
        if core.settings.llm_model:
            forward["model"] = core.settings.llm_model  # audio model names mean nothing to a text model
        trace.extra.update(
            llm_model=forward.get("model"), tts_model=tts_spec.id if tts_spec else None, audio_inputs=n_audio
        )
        headers = {"x-request-id": trace.id}

        if not stream:
            t_llm = trace.now()
            doc = await llm.chat(forward)
            trace.add_phase("llm", t_llm, trace.now())
            if wants_audio and tts_spec is not None:
                for choice in doc.get("choices") or []:
                    message = choice.get("message") or {}
                    text = message.get("content")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    t_tts = trace.now()
                    samples, rate = await core.synthesize_all(text, model=tts_spec.id, voice=voice)
                    data = await asyncio.to_thread(encode_audio, samples, rate, fmt)
                    trace.add_phase("synthesize", t_tts, trace.now())
                    trace.mark_ttfa()
                    message["audio"] = {
                        "id": "audio_" + uuid.uuid4().hex[:24],
                        "data": base64.b64encode(data).decode(),
                        "expires_at": int(time.time()) + 3600,
                        "transcript": text,
                    }
                    message["content"] = None
                    trace.text_preview = preview(text)
                    trace.chars = len(text)
            trace.finish("ok")
            core.traces.record(trace)
            return JSONResponse(doc, headers=headers)
    except BoloError as exc:
        trace.finish("error", exc.message)
        core.traces.record(trace)
        raise
    except Exception as exc:
        trace.finish("error", f"{type(exc).__name__}: {exc}")
        core.traces.record(trace)
        raise

    # -- streaming --------------------------------------------------------
    chunks = llm.stream_chat(forward)
    try:
        first_chunk = await chunks.__anext__()
    except StopAsyncIteration:
        first_chunk = None
    except BoloError as exc:
        trace.finish("error", exc.message)
        core.traces.record(trace)
        raise
    audio_id = "audio_" + uuid.uuid4().hex[:24]
    template: dict[str, Any] = {}
    finish: dict[str, Any] = {"reason": None, "usage": None, "passthrough": []}
    t_llm = trace.now()

    async def text_deltas() -> AsyncIterator[str]:
        chunk = first_chunk
        seen_token = False
        while chunk is not None:
            template.update({k: chunk[k] for k in ("id", "created", "model", "system_fingerprint") if k in chunk})
            if chunk.get("usage"):
                finish["usage"] = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if choice.get("finish_reason"):
                    finish["reason"] = choice["finish_reason"]
                if delta.get("tool_calls") or delta.get("function_call"):
                    finish["passthrough"].append(chunk)
                text = delta.get("content")
                if isinstance(text, str) and text:
                    if not seen_token:
                        seen_token = True
                        trace.extra["first_token_ms"] = round(trace.elapsed_ms(), 1)
                        trace.add_phase("llm_first_token", t_llm, trace.now())
                    yield text
            try:
                chunk = await chunks.__anext__()
            except StopAsyncIteration:
                chunk = None

    def envelope(delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
        doc = {
            "id": template.get("id", "chatcmpl-" + trace.id),
            "object": "chat.completion.chunk",
            "created": template.get("created", int(time.time())),
            "model": template.get("model", forward.get("model") or ""),
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return sse(doc)

    async def body_stream() -> AsyncIterator[bytes]:
        status, error = "ok", None
        said: list[str] = []
        seconds_out = 0.0
        try:
            yield envelope(
                {"role": "assistant", "content": None} if wants_audio else {"role": "assistant", "content": ""}
            )
            if wants_audio and tts_spec is not None:
                encoder = StreamEncoder(fmt, tts_spec.sample_rate)
                async for kind, value in speak_stream(
                    core, text_deltas(), model=tts_spec.id, voice=voice, transport="chat"
                ):
                    if kind == "text":
                        said.append(value)
                        yield envelope({"audio": {"id": audio_id, "transcript": value}})
                    elif kind == "audio":
                        if encoder.model_rate != value.sample_rate:
                            encoder = StreamEncoder(fmt, value.sample_rate)
                        data = encoder.feed(value.samples)
                        seconds_out += len(value.samples) / value.sample_rate
                        if data:
                            trace.mark_ttfa()
                            yield envelope({"audio": {"id": audio_id, "data": base64.b64encode(data).decode()}})
                tail = encoder.finish()
                if tail:
                    yield envelope({"audio": {"id": audio_id, "data": base64.b64encode(tail).decode()}})
                yield envelope({"audio": {"id": audio_id, "expires_at": int(time.time()) + 3600}})
            else:
                async for text in text_deltas():
                    said.append(text)
                    yield envelope({"content": text})
            for chunk in finish["passthrough"]:  # tool calls from the model, forwarded as they were
                yield sse(chunk)
            yield envelope({}, finish["reason"] or "stop")
            if finish["usage"]:
                yield sse(
                    {
                        **{k: template[k] for k in template},
                        "object": "chat.completion.chunk",
                        "choices": [],
                        "usage": finish["usage"],
                    }
                )
            yield b"data: [DONE]\n\n"
        except asyncio.CancelledError:
            status, error = "cancelled", "client disconnected"
            raise
        except BoloError as exc:
            status, error = "error", exc.message
            yield sse({"error": {"message": exc.message, "type": exc.type, "code": exc.code, "param": exc.param}})
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
            yield sse({"error": {"message": error, "type": "server_error", "code": "internal_error", "param": None}})
        finally:
            text = "".join(said)
            trace.text_preview = preview(text) if text else None
            trace.chars = len(text) or None
            trace.extra["audio_seconds_out"] = round(seconds_out, 2)
            trace.finish(status, error)
            core.traces.record(trace)

    return StreamingResponse(
        body_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "x-request-id": trace.id}
    )


__all__ = ["router"]
