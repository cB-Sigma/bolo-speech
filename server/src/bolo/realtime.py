"""OpenAI Realtime API sessions over WebSocket (``/v1/realtime``).

Implemented against the generally-available event set as typed in the
official ``openai`` Python SDK 3.50.0 (``openai.types.realtime``): GA names
such as ``response.output_audio.delta`` and ``conversation.item.added``, the
``session`` object with ``type: "realtime" | "transcription"`` and nested
``audio.input`` / ``audio.output``.

Two kinds of session:

* **transcription**: audio in, ``conversation.item.input_audio_transcription.*`` out.
* **realtime** (conversation): audio in → speech-to-text → the operator's
  language model (any OpenAI-compatible chat-completions endpoint) → streaming
  text-to-speech out, with server-side turn detection, barge-in, text input,
  function calling and per-turn latency traces.

By default bolo is a cascade (STT → LLM → TTS): the model never hears tone of
voice, and what it says is only as good as the language model the operator
connected. Two opt-in *pipelines* (setting ``realtime_pipeline``, or the bolo
extension ``session.pipeline``) change what answers a conversation turn:

* ``audio-llm``: the user's audio goes to an audio-understanding language model
  (chat completions with ``input_audio``: Gemma 4, Ultravox on llama-server)
  in place of STT → LLM; bolo's TTS still speaks the answer.
* ``speech-to-speech``: a speech-to-speech model (:mod:`bolo.s2s`, first
  backend llama-server's ``/v1/realtime`` with LFM3-Audio) hears the turn and
  answers with speech and tool calls.

In every pipeline bolo keeps the session itself: authentication, codecs and
resampling, turn detection and barge-in, conversation items, transcripts of the
user (its own STT) and traces.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import difflib
import hmac
import json
import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.agents import speak_stream
from bolo.audio.codecs import alaw_decode, alaw_encode, float_to_pcm16, mulaw_decode, mulaw_encode, pcm16_to_float
from bolo.audio.encode import encode_audio
from bolo.audio.resample import StreamResampler, resample
from bolo.denoise import DenoiseOptions, parse_denoise
from bolo.engines.base import FloatArray, VadDetector, VadParams
from bolo.errors import BoloError, EngineUnavailable
from bolo.obs.trace import Trace, preview
from bolo.s2s import AudioDelta, ResponseEnd, S2SConfig, S2SConversation, TextDelta, ToolCall
from bolo.streaming.agreement import LocalAgreement
from bolo.stt_options import DEFAULT_MAX_REPEATS, RepeatGuard

if TYPE_CHECKING:
    from bolo.core import Bolo

log = logging.getLogger("bolo.realtime")

Send = Callable[[dict[str, Any]], Awaitable[None]]
SR = 16000
FORMATS = {"audio/pcm": ("pcm", 24000), "audio/pcmu": ("mulaw", 8000), "audio/pcma": ("alaw", 8000)}
MAX_BUFFER_S = 120.0
#: semantic_vad: the end-of-turn model is asked after this much silence, and believed above this probability
SEMANTIC_PAUSE_S = 0.3
SEMANTIC_THRESHOLD = 0.5
#: what answers a conversation turn (see the module docstring)
PIPELINES = ("cascade", "audio-llm", "speech-to-speech")
#: audio-llm: user turns whose audio is kept for the model (older ones go as their transcript)
KEEP_USER_AUDIO = 8
#: audio kept after the detected end of speech (so a turn's audio_end_ms lies this far after the last speech)
END_PADDING_S = 0.2


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


class RealtimeError(Exception):
    def __init__(
        self,
        message: str,
        code: str = "invalid_request",
        param: str | None = None,
        type_: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.message, self.code, self.param, self.type = message, code, param, type_


def _format(spec: Any, param: str) -> tuple[str, int, dict[str, Any]]:
    """Validate an ``audio.*.format`` object. Returns (codec, rate, normalised object)."""
    if spec is None:
        return "pcm", 24000, {"type": "audio/pcm", "rate": 24000}
    if isinstance(spec, str):  # beta-style names are accepted for convenience
        spec = {"type": {"pcm16": "audio/pcm", "g711_ulaw": "audio/pcmu", "g711_alaw": "audio/pcma"}.get(spec, spec)}
    kind = spec.get("type") if isinstance(spec, dict) else None
    if kind not in FORMATS:
        raise RealtimeError("format.type must be audio/pcm, audio/pcmu or audio/pcma", param=param)
    codec, rate = FORMATS[kind]
    if codec == "pcm" and spec.get("rate") not in (None, 24000):
        raise RealtimeError("audio/pcm is 24000 Hz, 16-bit, mono, little-endian", param=param + ".rate")
    return codec, rate, ({"type": kind, "rate": 24000} if codec == "pcm" else {"type": kind})


def remaining_delta(sent: str, final: str) -> str:
    """What still has to be sent so that the deltas add up to ``final``.

    Deltas already sent cannot be taken back. When the final decode literally starts with them, the rest of
    it is returned and the deltas concatenate to the transcript exactly. When it revised something that was
    already sent (usually punctuation or capitalisation), the words that follow them in the final text are
    returned; ``completed.transcript`` is then the authoritative text.
    """
    if not sent:
        return final
    if final.startswith(sent):
        return final[len(sent) :]
    words = final.split()
    old = [w.strip(".,;:!?\"'").lower() for w in sent.split()]
    new = [w.strip(".,;:!?\"'").lower() for w in words]
    blocks = [b for b in difflib.SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks() if b.size]
    if blocks:  # continue after the place in the final text where the words already sent end
        last = blocks[-1]
        cut = last.b + last.size + (len(old) - (last.a + last.size))
    else:
        cut = len(old)
    rest = words[cut:]
    return (" " + " ".join(rest)) if rest else ""


class RealtimeSession:
    def __init__(
        self,
        core: Bolo,
        send: Send,
        *,
        model: str | None = None,
        transcription: bool = False,
        transport: str = "realtime",
    ) -> None:
        self.core = core
        self._send_raw = send
        self.transport = transport  # "realtime" (WebSocket) or "webrtc"; shown on traces
        self.id = new_id("sess")
        self.conversation_id = new_id("conv")
        self.kind = "transcription" if transcription else "realtime"
        s = core.settings
        if s.realtime_pipeline not in PIPELINES:
            raise RealtimeError(
                f"the setting realtime_pipeline must be one of {', '.join(PIPELINES)}", code="server_misconfigured"
            )
        self.pipeline = s.realtime_pipeline
        self.model = model or s.llm_model or "bolo-realtime"
        self.instructions = s.llm_instructions
        self.output_modalities = ["audio"]
        self.tools: list[dict[str, Any]] = []
        self.tool_choice: Any = "auto"
        self.max_output_tokens: Any = "inf"
        self.voice: str | None = None
        self.speed = 1.0
        #: bolo extension, session.audio.output.phonemizer
        self.phonemizer: str | None = None
        #: bolo extension, session.audio.output.romanised: "off" | "auto" | "on" (None: the voice's default)
        self.romanised: str | None = None
        #: set by transports that configure a session from a named agent (telephony): the text-to-speech model
        #: and the language model's name for this session, and fields added to every turn's trace
        self.tts_model: str | None = None
        self.llm_model: str | None = None
        self.trace_extra: dict[str, Any] = {}
        self.in_codec, self.in_rate, self.in_format = _format(None, "")
        self.out_codec, self.out_rate, self.out_format = _format(None, "")
        self.transcription: dict[str, Any] = {"model": s.default_realtime_stt, "language": None, "prompt": None}
        #: OpenAI's session.audio.input.noise_reduction ({"type": "near_field" | "far_field"} or null) and bolo's
        #: session.audio.input.denoise; either turns on the denoiser before transcription (None: the defaults)
        self.noise_reduction: dict[str, Any] | None = None
        self.denoise: DenoiseOptions | None = None
        self.turn_detection: dict[str, Any] | None = {
            "type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300,
            "silence_duration_ms": int(s.endpoint_silence_ms), "create_response": True, "interrupt_response": True,
        }  # fmt: skip
        self.items: list[dict[str, Any]] = []
        self._item_audio: dict[
            str, list[tuple[float, str]]
        ] = {}  # assistant item -> [(audio ms where sentence starts, text)]
        # audio input state (16 kHz)
        self._resampler = StreamResampler(self.in_rate, SR)
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0  # absolute sample index of _buf[0]
        self._pos = 0
        self._detector: VadDetector | None = None
        self._vad_name: str | None = None
        self._speech_start: int | None = None
        self._pending_item: str | None = None
        self._commit_from = 0  # manual mode: start of the uncommitted audio
        # semantic_vad: a pause the end-of-turn model called unfinished (end of speech so far, deadline)
        self._held_end: int | None = None
        self._hold_until = 0
        self._turn_checks: list[dict[str, Any]] = []
        # incremental user transcript: item id -> text already sent as deltas
        self._partial_task: asyncio.Task[None] | None = None
        self._partial_sent: dict[str, str] = {}
        #: the repetition guard across this session's transcripts, for the transcription model it was made for
        self._guard: tuple[str, RepeatGuard] | None = None
        # work
        self._turns: asyncio.Queue[Callable[[], Awaitable[None]] | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._response: asyncio.Task[None] | None = None
        self._response_id: str | None = None
        self._cancel_reason = "client_cancelled"
        self._closed = False
        self._busy = False  # a turn (transcription, response) is running in the worker
        # speech-to-speech pipeline: the model's conversation (opened on first use), how far the audio of the
        # current turn has been sent to it, and whether anything was sent since the last commit
        self._s2s: S2SConversation | None = None
        self._s2s_lock = asyncio.Lock()
        self._s2s_sent = 0
        self._s2s_pending = False
        self._s2s_broken = False
        # audio-llm pipeline: the audio of recent user turns, by item id
        self._user_audio: dict[str, FloatArray] = {}

    # -- plumbing -----------------------------------------------------------
    async def send(self, type_: str, **fields: Any) -> None:
        if not self._closed:
            await self._send_raw({"type": type_, "event_id": new_id("event"), **fields})

    async def error(
        self,
        message: str,
        *,
        code: str = "invalid_request",
        param: str | None = None,
        type_: str = "invalid_request_error",
        event_id: str | None = None,
    ) -> None:
        await self.send(
            "error", error={"type": type_, "code": code, "message": message, "param": param, "event_id": event_id}
        )

    def session_object(self) -> dict[str, Any]:
        audio_in: dict[str, Any] = {
            "format": self.in_format,
            "noise_reduction": self.noise_reduction,
            "denoise": self.denoise.to_dict() if self.denoise is not None else None,  # bolo extension
            "transcription": {k: v for k, v in self.transcription.items() if v is not None},
            "turn_detection": self.turn_detection,
        }
        if self.kind == "transcription":
            return {
                "type": "transcription",
                "id": self.id,
                "object": "realtime.transcription_session",
                "audio": {"input": audio_in},
                "include": [],
            }
        return {
            "type": "realtime",
            "id": self.id,
            "object": "realtime.session",
            "model": self.model,
            "instructions": self.instructions,
            "output_modalities": self.output_modalities,
            "audio": {
                "input": audio_in,
                "output": {
                    "format": self.out_format,
                    "voice": self.voice,
                    "speed": self.speed,
                    "phonemizer": self.phonemizer,
                    "romanised": self.romanised,
                },
            },
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "max_output_tokens": self.max_output_tokens,
            "truncation": "auto",
            "pipeline": self.pipeline,  # bolo extension
        }

    async def start(self) -> None:
        self.core.active_streams += 1
        self._worker = asyncio.create_task(self._work(), name=f"bolo-realtime-{self.id}")
        await self.send("session.created", session=self.session_object())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.core.active_streams -= 1
        for task in (self._partial_task, self._response, self._worker):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        conv, self._s2s = self._s2s, None
        if conv is not None:
            with contextlib.suppress(Exception):
                await conv.close()

    async def _work(self) -> None:
        """Turns (commit → transcribe → respond) run one at a time, in order."""
        while True:
            job = await self._turns.get()
            if job is None:
                return
            self._busy = True
            try:
                await job()
            except asyncio.CancelledError:
                raise
            except BoloError as exc:
                await self.error(exc.message, code=exc.code, type_=exc.type)
            except Exception as exc:
                log.exception("realtime turn failed")
                await self.error(f"{type(exc).__name__}: {exc}", code="internal_error", type_="server_error")
            finally:
                self._busy = False

    # -- client events ------------------------------------------------------
    async def handle(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        event_id = event.get("event_id")
        handler = {
            "session.update": self._on_session_update,
            "input_audio_buffer.append": self._on_append,
            "input_audio_buffer.commit": self._on_commit,
            "input_audio_buffer.clear": self._on_clear,
            "output_audio_buffer.clear": self._on_output_clear,
            "conversation.item.create": self._on_item_create,
            "conversation.item.delete": self._on_item_delete,
            "conversation.item.retrieve": self._on_item_retrieve,
            "conversation.item.truncate": self._on_item_truncate,
            "response.create": self._on_response_create,
            "response.cancel": self._on_response_cancel,
        }.get(str(kind))
        if handler is None:
            await self.error(
                f"unknown or unsupported event type {kind!r}", code="unknown_event", param="type", event_id=event_id
            )
            return
        try:
            await handler(event)
        except RealtimeError as exc:
            await self.error(exc.message, code=exc.code, param=exc.param, type_=exc.type, event_id=event_id)
        except BoloError as exc:
            await self.error(exc.message, code=exc.code, param=exc.param, type_=exc.type, event_id=event_id)

    async def _on_session_update(self, event: dict[str, Any]) -> None:
        self.configure(event.get("session"))
        await self.send("session.updated", session=self.session_object())

    def configure(self, session: Any) -> None:
        """Apply a session configuration (the ``session`` object of ``session.update``)."""
        if not isinstance(session, dict):
            raise RealtimeError("session.update needs a 'session' object", param="session")
        kind = session.get("type", self.kind)
        if kind not in ("realtime", "transcription"):
            raise RealtimeError("session.type must be 'realtime' or 'transcription'", param="session.type")
        if kind != self.kind:
            if self.items or self._pos:
                raise RealtimeError(
                    "the session type cannot change after audio or items were sent", param="session.type"
                )
            self.kind = kind
        audio = session.get("audio") or {}
        audio_in = audio.get("input") or {}
        audio_out = audio.get("output") or {}
        if "format" in audio_in:
            codec, rate, fmt = _format(audio_in["format"], "session.audio.input.format")
            if (codec, rate) != (self.in_codec, self.in_rate):
                if self._pos:
                    raise RealtimeError(
                        "the input format cannot change after audio was sent", param="session.audio.input.format"
                    )
                self._resampler = StreamResampler(rate, SR)
            self.in_codec, self.in_rate, self.in_format = codec, rate, fmt
        if "format" in audio_out:
            self.out_codec, self.out_rate, self.out_format = _format(audio_out["format"], "session.audio.output.format")
        if "transcription" in audio_in:
            cfg = audio_in["transcription"] or {}
            model = cfg.get("model")
            if model:
                self.core.resolve(model, "stt")  # validates; OpenAI model names map to the default
            self.transcription = {
                "model": model or self.core.settings.default_realtime_stt,
                "language": cfg.get("language"),
                "prompt": cfg.get("prompt"),
            }
        if "noise_reduction" in audio_in or "denoise" in audio_in:
            self._configure_denoise(audio_in)
        if "turn_detection" in audio_in:
            td = audio_in["turn_detection"]
            if td is None:
                self.turn_detection = None
            elif isinstance(td, dict) and td.get("type") in ("server_vad", "semantic_vad"):
                merged = {**(self.turn_detection or {}), **{k: v for k, v in td.items() if v is not None}}
                # semantic_vad: the detector finds the pause, an end-of-turn model decides whether it ends
                # the turn (see _on_pause). Without that model it behaves exactly like server_vad.
                merged.setdefault("threshold", 0.5)
                merged.setdefault("prefix_padding_ms", 300)
                merged.setdefault("silence_duration_ms", int(self.core.settings.endpoint_silence_ms))
                merged.setdefault("create_response", True)
                merged.setdefault("interrupt_response", True)
                # bolo extension: the voice-activity detector of this session (default: the default_vad setting)
                if merged.get("vad_model") is not None:
                    try:
                        merged["vad_model"] = self.core.resolve(str(merged["vad_model"]), "vad").id
                    except BoloError as exc:
                        raise RealtimeError(exc.message, param="session.audio.input.turn_detection.vad_model") from exc
                self.turn_detection = merged
            else:
                raise RealtimeError(
                    "turn_detection.type must be server_vad, semantic_vad or null",
                    param="session.audio.input.turn_detection",
                )
            self._detector = None
            self._held_end = None
        if "voice" in audio_out:
            voice = audio_out["voice"]
            self.voice = voice.get("id") if isinstance(voice, dict) else voice
        if audio_out.get("speed") is not None:
            self.speed = float(audio_out["speed"])
        if "phonemizer" in audio_out:
            from bolo.text.phonemize import parse_choice

            value = audio_out["phonemizer"]
            try:
                parse_choice(value)
            except Exception as exc:
                raise RealtimeError(str(getattr(exc, "message", exc)), param="session.audio.output.phonemizer") from exc
            self.phonemizer = value or None
        if "romanised" in audio_out or "romanized" in audio_out:
            from bolo.text.romanised import parse_mode

            try:
                self.romanised = parse_mode(audio_out.get("romanised", audio_out.get("romanized")))
            except Exception as exc:
                raise RealtimeError(str(getattr(exc, "message", exc)), param="session.audio.output.romanised") from exc
        if self.kind == "realtime":
            if "pipeline" in session:
                pipeline = session["pipeline"] or self.core.settings.realtime_pipeline
                if pipeline not in PIPELINES:
                    raise RealtimeError(f"pipeline must be one of {', '.join(PIPELINES)}", param="session.pipeline")
                if pipeline != self.pipeline and (self.items or self._pos):
                    raise RealtimeError(
                        "the pipeline cannot change after audio or items were sent", param="session.pipeline"
                    )
                self.pipeline = pipeline
            if self._s2s is not None and any(k in session for k in ("instructions", "tools", "tool_choice")):
                log.info(
                    "session %s: instructions and tools reach the speech-to-speech model only before the first turn",
                    self.id,
                )
            if isinstance(session.get("instructions"), str):
                self.instructions = session["instructions"]
            if session.get("output_modalities"):
                mods = session["output_modalities"]
                if not isinstance(mods, list) or not set(mods) <= {"audio", "text"}:
                    raise RealtimeError(
                        "output_modalities must be ['audio'] or ['text']", param="session.output_modalities"
                    )
                self.output_modalities = mods
            if "tools" in session:
                tools = session["tools"] or []
                for tool in tools:
                    if not isinstance(tool, dict) or tool.get("type", "function") != "function":
                        raise RealtimeError(
                            "only function tools are supported (MCP tools are not)",
                            param="session.tools",
                            code="unsupported",
                        )
                self.tools = tools
            if "tool_choice" in session:
                self.tool_choice = session["tool_choice"]
            if "max_output_tokens" in session:
                self.max_output_tokens = session["max_output_tokens"]
            if session.get("model"):
                self.model = session["model"]

    def _configure_denoise(self, audio_in: dict[str, Any]) -> None:
        """``noise_reduction`` (OpenAI: any type turns the default denoiser on; null leaves the server's default) and
        ``denoise`` (bolo: true, false, a strength, a model id or {model, strength}), which wins when both are given."""
        if "noise_reduction" in audio_in:
            nr = audio_in["noise_reduction"]
            if nr is None:
                self.noise_reduction = None
                if "denoise" not in audio_in:
                    self.denoise = None
            elif isinstance(nr, dict) and nr.get("type") in ("near_field", "far_field"):
                self.noise_reduction = {"type": nr["type"]}
                if "denoise" not in audio_in:
                    self.denoise = DenoiseOptions()
            else:
                raise RealtimeError(
                    'noise_reduction must be null or {"type": "near_field" | "far_field"}',
                    param="session.audio.input.noise_reduction",
                )
        if "denoise" in audio_in:
            param = "session.audio.input.denoise"
            try:
                opts = parse_denoise(audio_in["denoise"], param=param)
                if opts is not None and opts.enabled and opts.model:
                    self.core.resolve(opts.model, "denoise")
            except BoloError as exc:
                raise RealtimeError(exc.message, param=param) from exc
            self.denoise = opts

    # -- audio input ----------------------------------------------------------
    def _decode(self, payload: bytes) -> FloatArray:
        if self.in_codec == "mulaw":
            return mulaw_decode(payload)
        if self.in_codec == "alaw":
            return alaw_decode(payload)
        return pcm16_to_float(payload)

    def _ms(self, sample: int) -> int:
        return int(sample * 1000 / SR)

    async def _ensure_detector(self) -> VadDetector | None:
        if self.turn_detection is None:
            return None
        if self._detector is None:
            td = self.turn_detection
            silence_s = max(0.1, float(td.get("silence_duration_ms", 500)) / 1000.0)
            params = VadParams(
                threshold=float(td.get("threshold", 0.5)),
                min_silence_s=min(silence_s, SEMANTIC_PAUSE_S) if self._semantic else silence_s,
                max_speech_s=self.core.settings.max_utterance_s * 3,
            )
            vad_model = td.get("vad_model")
            self._detector, self._vad_name = await self.core.vad_detector(
                params, str(vad_model) if vad_model else None, fallback=True
            )
        return self._detector

    async def _on_append(self, event: dict[str, Any]) -> None:
        audio = event.get("audio")
        if not isinstance(audio, str):
            raise RealtimeError("input_audio_buffer.append needs base64 'audio'", param="audio")
        try:
            payload = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RealtimeError("'audio' is not valid base64", param="audio") from exc
        await self.accept_audio(self._resampler.process(self._decode(payload)))

    async def accept_audio(self, samples: FloatArray) -> None:
        """Take input audio that is already 16 kHz mono (the WebRTC transport calls this directly)."""
        if not len(samples):
            return
        self._buf = np.concatenate([self._buf, samples]) if len(self._buf) else samples.copy()
        self._pos += len(samples)
        detector = await self._ensure_detector()
        if detector is None:
            if len(self._buf) > MAX_BUFFER_S * SR:
                raise RealtimeError(
                    "the input audio buffer is full; commit or clear it", code="input_audio_buffer_full"
                )
            if self._s2s_mode:
                await self._s2s_forward(self._commit_from)
            return
        prefix = int(float((self.turn_detection or {}).get("prefix_padding_ms", 300)) * SR / 1000)
        for ev in detector.accept(samples):
            if ev.kind == "speech_start":
                if self._held_end is not None and self._speech_start is not None:
                    self._held_end = None  # the speaker went on after a pause: still the same turn
                    continue
                self._speech_start = max(self._buf_start, ev.sample - prefix)
                self._pending_item = new_id("item")
                await self.send(
                    "input_audio_buffer.speech_started",
                    audio_start_ms=self._ms(self._speech_start),
                    item_id=self._pending_item,
                )
                if (self.turn_detection or {}).get("interrupt_response", True):
                    await self._cancel_response("turn_detected")
                await self._stop_partials()
                self._partial_task = asyncio.create_task(
                    self._partials(self._pending_item, self._speech_start), name="bolo-realtime-partials"
                )
                if self._s2s_mode:
                    self._s2s_broken = False
                    await self._s2s_forward(self._speech_start)
            elif ev.kind == "speech_end" and self._speech_start is not None:
                end = min(ev.sample + int(END_PADDING_S * SR), self._pos)
                if self._semantic and not await self._turn_complete(end):
                    self._held_end = end  # only a pause, says the model: keep the turn open for a while
                    self._hold_until = ev.sample + int(self._max_pause_s() * SR)
                    continue
                await self._end_turn(end)
        if self._held_end is not None and self._speech_start is not None and self._pos >= self._hold_until:
            if not detector.in_speech:  # the pause lasted long enough to be the end after all
                await self._end_turn(self._held_end)
            else:
                self._held_end = None
        if self._s2s_mode and self._speech_start is not None:
            await self._s2s_forward(self._speech_start)
        self._trim()

    # -- speech-to-speech pipeline ------------------------------------------------
    @property
    def _s2s_mode(self) -> bool:
        return self.kind == "realtime" and self.pipeline == "speech-to-speech"

    async def _s2s_conversation(self) -> S2SConversation:
        async with self._s2s_lock:
            if self._s2s is None:
                backend = await self.core.s2s.backend()
                config = S2SConfig(
                    instructions=self.instructions, tools=list(self.tools), tool_choice=self.tool_choice,
                    voice=self.voice, max_output_tokens=self.max_output_tokens,
                )  # fmt: skip
                self._s2s = await backend.open(config)
            return self._s2s

    async def _s2s_forward(self, start: int) -> None:
        """Send the model the audio of the current turn from ``start`` (or where the last send stopped) to now.

        A failure is reported once as an ``error`` event; nothing more is sent until the next turn starts."""
        begin = max(start, self._s2s_sent, self._buf_start)
        if self._s2s_broken or self._pos <= begin:
            return
        try:
            conv = await self._s2s_conversation()
            await conv.append_audio(self._slice(begin, self._pos))
        except BoloError as exc:
            self._s2s_broken = True
            await self.error(exc.message, code=exc.code, type_=exc.type)
            return
        self._s2s_sent = self._pos
        self._s2s_pending = True

    @property
    def _semantic(self) -> bool:
        return (self.turn_detection or {}).get("type") == "semantic_vad" and self.core.turn_model() is not None

    def _max_pause_s(self) -> float:
        """How long a pause the end-of-turn model called unfinished may last before the turn ends anyway."""
        eagerness = str((self.turn_detection or {}).get("eagerness", "auto"))
        return {"low": 6.0, "medium": 3.0, "auto": 3.0, "high": 1.5}.get(eagerness, 3.0)

    async def _turn_complete(self, end: int) -> bool:
        started = time.perf_counter()
        probability = await self.core.end_of_turn(self._slice(self._speech_start or 0, end))
        self._turn_checks.append(
            {"probability": None if probability is None else round(probability, 3),
             "ms": round((time.perf_counter() - started) * 1000.0, 1)}
        )  # fmt: skip
        return probability is None or probability >= SEMANTIC_THRESHOLD

    def _agent_trace(self) -> Trace:
        """A voice-agent trace that says which pipeline answers (or would answer) the turn."""
        trace = Trace("agent", self.transport, "voice-agent")
        trace.extra["pipeline"] = self.pipeline if self.kind == "realtime" else "transcription"
        trace.extra.update(self.trace_extra)
        return trace

    # -- state other transports read ----------------------------------------------
    @property
    def responding(self) -> bool:
        """A response is being generated, or a committed turn is waiting for its transcript and response."""
        return (self._response is not None and not self._response.done()) or not self._turns.empty() or self._busy

    async def interrupt(self, reason: str = "client_cancelled") -> None:
        """Stop the response in progress, if there is one (no error when there is none)."""
        await self._cancel_response(reason)

    @property
    def user_speaking(self) -> bool:
        return self._speech_start is not None

    @property
    def input_ms(self) -> int:
        """Milliseconds of input audio received so far."""
        return self._ms(self._pos)

    async def _end_turn(self, end: int) -> None:
        item_id = self._pending_item or new_id("item")
        await self._stop_partials()
        await self.send("input_audio_buffer.speech_stopped", audio_end_ms=self._ms(end), item_id=item_id)
        trace = self._agent_trace()
        if self._turn_checks:
            trace.extra["turn_detection"] = {"type": "semantic_vad", "checks": self._turn_checks}
            self._turn_checks = []
        await self._commit(self._slice(self._speech_start or 0, end), item_id, trace)
        self._speech_start, self._pending_item, self._held_end = None, None, None

    async def _stop_partials(self) -> None:
        task, self._partial_task = self._partial_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _partials(self, item_id: str, start: int) -> None:
        """While the user is speaking, re-decode the utterance so far and send the words that have settled.

        A word is sent once two consecutive decodes agree on it (LocalAgreement), so deltas only ever append.
        """
        s = self.core.settings
        cfg = self.transcription
        try:
            if not self.core.interim_decodes(self.core.resolve(cfg["model"], "stt")):
                return  # a cloud recogniser: every interim decode would be a billed request
        except BoloError:
            return  # the final decode reports the error
        agreement = LocalAgreement(s.agreement)
        sent_words = 0
        decoded_to = 0
        while not self._closed:
            await asyncio.sleep(s.partial_interval_ms / 1000.0)
            end = self._pos
            if end - start < int(0.4 * SR) or end == decoded_to or end - start > s.max_utterance_s * SR:
                continue
            decoded_to = end
            try:
                result = await self.core.transcribe(
                    self._slice(start, end), SR, model=cfg["model"], language=cfg.get("language"),
                    prompt=cfg.get("prompt"), vad_filter=False, trace=Trace("stt", self.transport, ""), record=False,
                    denoise=self.denoise,
                )  # fmt: skip
            except BoloError:
                continue  # the final decode reports the error
            agreement.update(result.text)
            # hold the newest word back: its punctuation and casing often change once more audio follows
            upto = min(len(agreement.committed), len(result.text.split()) - 1)
            fresh = agreement.committed[sent_words:upto]
            if not fresh:
                continue
            delta = " ".join(fresh) if sent_words == 0 else " " + " ".join(fresh)
            sent_words = upto
            self._partial_sent[item_id] = self._partial_sent.get(item_id, "") + delta
            await self.send(
                "conversation.item.input_audio_transcription.delta", item_id=item_id, content_index=0, delta=delta
            )

    def _slice(self, start: int, end: int) -> FloatArray:
        a = max(0, start - self._buf_start)
        piece: FloatArray = self._buf[a : max(a, end - self._buf_start)].copy()
        return piece

    def _trim(self) -> None:
        if self.turn_detection is None:
            return
        keep_from = self._speech_start if self._speech_start is not None else self._pos - 2 * SR
        drop = max(0, min(keep_from - self._buf_start, len(self._buf)))
        if drop > SR:
            self._buf = self._buf[drop:].copy()
            self._buf_start += drop

    async def _on_commit(self, event: dict[str, Any]) -> None:
        audio = self._slice(max(self._commit_from, self._buf_start), self._pos)
        if len(audio) < int(0.1 * SR):
            raise RealtimeError(
                "the input audio buffer has less than 100 ms of audio", code="input_audio_buffer_commit_empty"
            )
        self._buf, self._buf_start, self._commit_from = np.zeros(0, dtype=np.float32), self._pos, self._pos
        self._speech_start = None
        await self._commit(audio, new_id("item"), self._agent_trace(), auto_response=False)

    async def _on_clear(self, event: dict[str, Any]) -> None:
        self._buf, self._buf_start, self._commit_from = np.zeros(0, dtype=np.float32), self._pos, self._pos
        self._speech_start, self._pending_item = None, None
        await self._stop_partials()
        self._partial_sent.clear()
        await self.send("input_audio_buffer.cleared")

    async def _on_output_clear(self, event: dict[str, Any]) -> None:
        # WebRTC/SIP only in the reference; over WebSocket the client owns playback. Stop generating.
        await self._cancel_response("client_cancelled")

    async def _commit(self, audio: FloatArray, item_id: str, trace: Trace, auto_response: bool | None = None) -> None:
        previous = self.items[-1]["id"] if self.items else None
        item = {
            "id": item_id, "object": "realtime.item", "type": "message", "role": "user", "status": "completed",
            "content": [{"type": "input_audio", "transcript": None}],
        }  # fmt: skip
        self.items.append(item)
        if self.kind == "realtime" and self.pipeline == "audio-llm":
            self._user_audio[item_id] = audio
            for old in list(self._user_audio)[:-KEEP_USER_AUDIO]:
                del self._user_audio[old]
        if self._s2s_mode and self._s2s_pending and self._s2s is not None:
            # in order with the audio already sent: the next turn's audio must not join this one
            self._s2s_pending = False
            try:
                await self._s2s.commit()
            except BoloError as exc:
                await self.error(exc.message, code=exc.code, type_=exc.type)
        await self.send("input_audio_buffer.committed", item_id=item_id, previous_item_id=previous)
        await self.send("conversation.item.added", item=item, previous_item_id=previous)
        await self.send("conversation.item.done", item=item, previous_item_id=previous)
        respond = (
            auto_response
            if auto_response is not None
            else bool((self.turn_detection or {}).get("create_response", True))
        )
        trace.audio_seconds = len(audio) / SR

        async def turn() -> None:
            if self.kind == "realtime" and self.pipeline != "cascade":
                # the model hears the audio itself: the transcript (for the client and the history) runs alongside
                transcript: asyncio.Task[bool] | None = None
                if self.core.settings.realtime_transcribe_user:
                    transcript = asyncio.create_task(
                        self._transcribe(item, audio, trace), name="bolo-realtime-transcript"
                    )
                if respond:
                    await self._start_response({}, trace)
                ok = True
                if transcript is not None:
                    ok = bool(await transcript)
                if not respond:
                    trace.kind, trace.model = "stt", self.transcription["model"]
                    trace.finish("ok" if ok else "error")
                    self.core.traces.record(trace)
                return
            ok = await self._transcribe(item, audio, trace)
            if ok and respond and self.kind == "realtime":
                await self._start_response({}, trace)
            elif self.kind == "transcription" or not respond:
                trace.kind, trace.model = "stt", self.transcription["model"]
                trace.finish("ok" if ok else "error")
                if trace.audio_seconds and trace.total_ms is not None:
                    trace.rtf = trace.total_ms / 1000.0 / trace.audio_seconds
                self.core.traces.record(trace)

        await self._turns.put(turn)

    async def _transcribe(self, item: dict[str, Any], audio: FloatArray, trace: Trace) -> bool:
        cfg = self.transcription
        sub = Trace("stt", self.transport, "")
        t0 = trace.now()
        try:
            result = await self.core.transcribe(
                audio,
                SR,
                model=cfg["model"],
                language=cfg.get("language"),
                prompt=cfg.get("prompt"),
                trace=sub,
                denoise=self.denoise,
            )
        except BoloError as exc:
            await self.send(
                "conversation.item.input_audio_transcription.failed", item_id=item["id"], content_index=0,
                error={"type": "transcription_error", "code": exc.code, "message": exc.message, "param": None},
            )  # fmt: skip
            trace.error = exc.message
            return False
        trace.add_phase("decode", t0, trace.now())
        trace.mark_final()
        trace.language = result.language
        trace.extra["stt_model"] = result.model
        text = self._guarded(result.model, result.text)
        trace.extra["user_text"] = preview(text)
        item["content"][0]["transcript"] = text
        sent = self._partial_sent.pop(item["id"], "")
        trace.extra["transcript_deltas_before_final"] = bool(sent)
        rest = remaining_delta(sent, text)
        if rest:
            await self.send(
                "conversation.item.input_audio_transcription.delta", item_id=item["id"], content_index=0, delta=rest
            )
        await self.send(
            "conversation.item.input_audio_transcription.completed", item_id=item["id"], content_index=0, transcript=text,
            usage={"type": "duration", "seconds": round(len(audio) / SR, 2)},
        )  # fmt: skip
        return bool(text.strip())

    def _guarded(self, model: str, text: str) -> str:
        """``text`` through the session's repetition guard (bolo.stt_options.RepeatGuard): each utterance's transcript
        is already trimmed on its own; this drops a loop that runs on from one utterance into the next. The model's
        pinned ``max_repeats`` applies (0: off)."""
        if self._guard is None or self._guard[0] != model:
            spec = self.core.catalog.resolve(model)
            limit = (
                self.core.decode_options(spec).get("max_repeats", DEFAULT_MAX_REPEATS) if spec else DEFAULT_MAX_REPEATS
            )
            self._guard = (model, RepeatGuard(int(limit)))
        guarded, _ = self._guard[1].apply(text)
        return guarded

    # -- conversation items ---------------------------------------------------
    def _find(self, item_id: Any) -> dict[str, Any]:
        for item in self.items:
            if item["id"] == item_id:
                return item
        raise RealtimeError(f"no item with id {item_id!r}", code="item_not_found", param="item_id")

    async def _on_item_create(self, event: dict[str, Any]) -> None:
        item = event.get("item")
        if not isinstance(item, dict):
            raise RealtimeError("conversation.item.create needs an 'item'", param="item")
        kind = item.get("type")
        if kind == "message":
            if item.get("role") not in ("user", "assistant", "system") or not isinstance(item.get("content"), list):
                raise RealtimeError("a message item needs a role and a content list", param="item")
            for part in item["content"]:
                if part.get("type") == "input_audio" and part.get("audio") and not part.get("transcript"):
                    try:
                        raw = base64.b64decode(part["audio"], validate=True)
                    except (binascii.Error, ValueError) as exc:
                        raise RealtimeError("content audio is not valid base64", param="item.content") from exc
                    audio = resample(self._decode(raw), self.in_rate, SR)
                    result = await self.core.transcribe(
                        audio,
                        SR,
                        model=self.transcription["model"],
                        language=self.transcription.get("language"),
                        denoise=self.denoise,
                    )
                    part["transcript"] = result.text
                    part.pop("audio", None)
                elif part.get("type") == "input_image":
                    raise RealtimeError("image input is not supported", code="unsupported", param="item.content")
        elif kind == "function_call_output":
            if not isinstance(item.get("call_id"), str) or not isinstance(item.get("output"), str):
                raise RealtimeError("function_call_output needs 'call_id' and 'output' strings", param="item")
        elif kind == "function_call":
            if not isinstance(item.get("name"), str):
                raise RealtimeError("function_call needs a 'name'", param="item")
        else:
            raise RealtimeError(f"unsupported item type {kind!r}", code="unsupported", param="item.type")
        stored = {
            "id": item.get("id") or new_id("item"),
            "object": "realtime.item",
            "status": "completed",
            **{k: v for k, v in item.items() if k != "id"},
        }
        previous = self.items[-1]["id"] if self.items else None
        if self._s2s_mode:
            await self._s2s_item(stored)
        self.items.append(stored)
        await self.send("conversation.item.added", item=stored, previous_item_id=previous)
        await self.send("conversation.item.done", item=stored, previous_item_id=previous)

    async def _s2s_item(self, item: dict[str, Any]) -> None:
        """Give the speech-to-speech model a client-created item: user text, or the output of a tool call. Other
        items (assistant or system messages) stay in bolo's conversation only."""
        if item.get("type") == "function_call_output":
            conv = await self._s2s_conversation()
            await conv.add_tool_output(str(item["call_id"]), str(item["output"]))
        elif item.get("type") == "message" and item.get("role") == "user":
            texts = [p.get("text") or p.get("transcript") or "" for p in item.get("content", [])]
            text = " ".join(t for t in texts if t).strip()
            if text:
                conv = await self._s2s_conversation()
                await conv.add_text(text)

    async def _on_item_delete(self, event: dict[str, Any]) -> None:
        item = self._find(event.get("item_id"))
        self.items.remove(item)
        await self.send("conversation.item.deleted", item_id=item["id"])

    async def _on_item_retrieve(self, event: dict[str, Any]) -> None:
        await self.send("conversation.item.retrieved", item=self._find(event.get("item_id")))

    async def _on_item_truncate(self, event: dict[str, Any]) -> None:
        """The client says how much of an assistant message was actually played: forget the rest."""
        item = self._find(event.get("item_id"))
        if item.get("role") != "assistant":
            raise RealtimeError("only assistant message items can be truncated", param="item_id")
        try:
            audio_end_ms = int(event.get("audio_end_ms"))  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise RealtimeError("audio_end_ms must be an integer", param="audio_end_ms") from exc
        heard = [text for start_ms, text in self._item_audio.get(item["id"], []) if start_ms < audio_end_ms]
        for part in item.get("content", []):
            if part.get("type") == "output_audio":
                part["transcript"] = " ".join(heard)
        await self.send(
            "conversation.item.truncated",
            item_id=item["id"],
            content_index=int(event.get("content_index") or 0),
            audio_end_ms=audio_end_ms,
        )

    # -- responses --------------------------------------------------------------
    async def _on_response_create(self, event: dict[str, Any]) -> None:
        if self.kind != "realtime":
            raise RealtimeError("transcription sessions cannot create responses", code="unsupported")
        params = event.get("response") if isinstance(event.get("response"), dict) else {}
        trace = self._agent_trace()

        async def turn() -> None:
            await self._start_response(params or {}, trace)

        await self._turns.put(turn)

    async def _on_response_cancel(self, event: dict[str, Any]) -> None:
        if self._response is None or self._response.done():
            raise RealtimeError("there is no response in progress", code="response_cancel_not_active")
        await self._cancel_response("client_cancelled")

    async def _cancel_response(self, reason: str) -> None:
        task = self._response
        if task is None or task.done():
            return
        self._cancel_reason = reason
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _start_response(self, params: dict[str, Any], trace: Trace) -> None:
        """Run a response as its own task so that new audio (barge-in) can cancel it."""
        await self._cancel_response("client_cancelled")
        self._response = asyncio.create_task(self._respond(params, trace))
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(self._wait(self._response))

    @staticmethod
    async def _wait(task: asyncio.Task[None]) -> None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    def _audio_part(self, item_id: str) -> dict[str, Any]:
        wav = encode_audio(self._user_audio[item_id], SR, "wav")
        return {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}}

    def _messages(self, instructions: str | None, extra: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        items = [*self.items, *(extra or [])]
        # audio-llm: the latest user turn is sent as audio (and earlier ones whose transcript is not there)
        heard = [i["id"] for i in items if i.get("id") in self._user_audio] if self.pipeline == "audio-llm" else []
        for item in items:
            kind = item.get("type")
            if kind == "message":
                texts = [p.get("text") or p.get("transcript") or "" for p in item.get("content", [])]
                text = " ".join(t for t in texts if t).strip()
                if item.get("id") in heard and (item["id"] == heard[-1] or not text):
                    messages.append({"role": "user", "content": [self._audio_part(item["id"])]})
                elif text:
                    messages.append({"role": item["role"], "content": text})
            elif kind == "function_call":
                call = {
                    "id": item.get("call_id") or item["id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item.get("arguments") or "{}"},
                }
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            elif kind == "function_call_output":
                messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
        return messages

    def _encode(self, samples: FloatArray, rate: int) -> str:
        x = resample(samples, rate, self.out_rate)
        if self.out_codec == "mulaw":
            raw = mulaw_encode(x)
        elif self.out_codec == "alaw":
            raw = alaw_encode(x)
        else:
            raw = float_to_pcm16(x).tobytes()
        return base64.b64encode(raw).decode()

    async def _respond(self, params: dict[str, Any], trace: Trace) -> None:
        core = self.core
        response_id = new_id("resp")
        self._response_id = response_id
        modalities = params.get("output_modalities") or self.output_modalities
        audio_mode = "audio" in modalities
        instructions = params.get("instructions") or self.instructions
        tools = params.get("tools") if params.get("tools") is not None else self.tools
        voice = ((params.get("audio") or {}).get("output") or {}).get("voice") or self.voice
        response: dict[str, Any] = {
            "id": response_id, "object": "realtime.response", "status": "in_progress", "status_details": None, "output": [],
            "conversation_id": self.conversation_id, "output_modalities": modalities, "usage": None,
            "audio": {"output": {"format": self.out_format, "voice": voice}}, "metadata": params.get("metadata"),
        }  # fmt: skip
        await self.send("response.created", response=response)
        llm = core.audio_llm if self.pipeline == "audio-llm" else core.llm
        if self._s2s_mode:
            model_name = str(core.s2s.describe().get("model") or "speech-to-speech")
            body: dict[str, Any] = {"model": model_name}
        else:
            body = {
                "model": (
                    core.audio_llm.model if self.pipeline == "audio-llm" else self.llm_model or core.settings.llm_model
                )
                or self.model,
                "messages": self._messages(instructions, params.get("input")),
                "stream_options": {"include_usage": True},
            }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description") or "",
                        "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
                for t in tools
            ]
            choice = params.get("tool_choice", self.tool_choice)
            if choice in ("auto", "none", "required"):
                body["tool_choice"] = choice
            elif isinstance(choice, dict) and choice.get("name"):
                body["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        limit = params.get("max_output_tokens", self.max_output_tokens)
        if isinstance(limit, int):
            body["max_tokens"] = limit

        item: dict[str, Any] | None = None
        part: dict[str, Any] = {}
        calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        text_parts: list[str] = []
        audio_ms = 0.0
        first_token: float | None = None
        first_audio: float | None = None
        finish_reason = "stop"
        t_llm = trace.now()
        previous = self.items[-1]["id"] if self.items else None

        async def open_message() -> dict[str, Any]:
            nonlocal item, part
            if item is None:
                item = {
                    "id": new_id("item"),
                    "object": "realtime.item",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                }
                self.items.append(item)
                self._item_audio[item["id"]] = []
                await self.send("response.output_item.added", response_id=response_id, output_index=0, item=item)
                await self.send("conversation.item.added", item=item, previous_item_id=previous)
                part = {"type": "output_audio", "transcript": ""} if audio_mode else {"type": "output_text", "text": ""}
                item["content"].append(part)
                wire = {"type": "audio", "transcript": ""} if audio_mode else {"type": "text", "text": ""}
                await self.send(
                    "response.content_part.added",
                    response_id=response_id,
                    item_id=item["id"],
                    output_index=0,
                    content_index=0,
                    part=wire,
                )
            return item

        async def text_deltas() -> AsyncIterator[str]:
            nonlocal first_token, finish_reason
            async for chunk in llm.stream_chat(body):
                if chunk.get("usage"):
                    usage.update(chunk["usage"])
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason"):
                        finish_reason = str(choice["finish_reason"])
                    for call in delta.get("tool_calls") or []:
                        slot = calls.setdefault(int(call.get("index") or 0), {"id": None, "name": "", "arguments": ""})
                        fn = call.get("function") or {}
                        slot["id"] = call.get("id") or slot["id"]
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    text = delta.get("content")
                    if isinstance(text, str) and text:
                        if first_token is None:
                            first_token = trace.now()
                            trace.extra["first_token_ms"] = round(trace.elapsed_ms(), 1)
                        yield text

        status = "completed"
        details: dict[str, Any] | None = None

        async def speech_to_speech() -> None:
            """The response of the speech-to-speech model, as the same events the cascade sends."""
            nonlocal first_token, first_audio, audio_ms, finish_reason
            conv = await self._s2s_conversation()
            choice = params.get("tool_choice", self.tool_choice)
            stream = conv.respond(
                modalities=list(modalities), max_output_tokens=params.get("max_output_tokens"),
                tool_choice=choice if isinstance(choice, str) else None,
            )  # fmt: skip
            sentence, sentence_ms = "", 0.0
            try:
                async for ev in stream:
                    if isinstance(ev, (TextDelta, AudioDelta)) and first_token is None:
                        first_token = trace.now()
                        trace.extra["first_token_ms"] = round(trace.elapsed_ms(), 1)
                    if isinstance(ev, TextDelta):
                        msg = await open_message()
                        base = {"response_id": response_id, "item_id": msg["id"], "output_index": 0, "content_index": 0}
                        text_parts.append(ev.text)
                        if audio_mode:
                            if not sentence:
                                sentence_ms = audio_ms
                            sentence += ev.text
                            if sentence.rstrip().endswith((".", "!", "?", "।")):
                                self._item_audio[msg["id"]].append((sentence_ms, sentence.strip()))
                                sentence = ""
                            part["transcript"] += ev.text
                            await self.send("response.output_audio_transcript.delta", delta=ev.text, **base)
                        else:
                            part["text"] += ev.text
                            await self.send("response.output_text.delta", delta=ev.text, **base)
                    elif isinstance(ev, AudioDelta) and audio_mode and len(ev.samples):
                        msg = await open_message()
                        base = {"response_id": response_id, "item_id": msg["id"], "output_index": 0, "content_index": 0}
                        if first_audio is None:
                            first_audio = trace.now()
                            trace.mark_ttfa()
                        audio_ms += len(ev.samples) * 1000.0 / ev.sample_rate
                        await self.send(
                            "response.output_audio.delta", delta=self._encode(ev.samples, ev.sample_rate), **base
                        )
                    elif isinstance(ev, ToolCall):
                        calls[len(calls)] = {"id": ev.call_id or None, "name": ev.name, "arguments": ev.arguments}
                    elif isinstance(ev, ResponseEnd):
                        usage.update(
                            prompt_tokens=ev.usage.get("input_tokens", 0), completion_tokens=ev.usage.get("output_tokens", 0),
                            total_tokens=ev.usage.get("total_tokens")
                            or ev.usage.get("input_tokens", 0) + ev.usage.get("output_tokens", 0),
                        )  # fmt: skip
                        if ev.status == "failed":
                            raise EngineUnavailable(
                                f"the speech-to-speech model failed: {ev.error or 'no reason given'}"
                            )
                        if ev.status == "cancelled":
                            raise asyncio.CancelledError
                        if ev.status == "incomplete":
                            finish_reason = "length"
            finally:
                if item is not None and sentence.strip():
                    self._item_audio[item["id"]].append((sentence_ms, sentence.strip()))
                await stream.aclose()

        try:
            if self._s2s_mode:
                await speech_to_speech()
            elif audio_mode:
                tts_spec = core.resolve(self.tts_model, "tts") if self.tts_model else core.realtime_tts(voice)
                async for kind, value in speak_stream(
                    core,
                    text_deltas(),
                    model=tts_spec.id,
                    voice=voice,
                    speed=self.speed,
                    transport="telephony" if self.transport == "telephony" else "realtime",
                    phonemizer=self.phonemizer,
                    romanised=self.romanised,
                ):
                    msg = await open_message()
                    base = {"response_id": response_id, "item_id": msg["id"], "output_index": 0, "content_index": 0}
                    if kind == "text":
                        text_parts.append(value)
                        part["transcript"] += value
                        await self.send("response.output_audio_transcript.delta", delta=value, **base)
                    elif kind == "sentence":
                        self._item_audio[msg["id"]].append((audio_ms, value))
                    elif kind == "audio":
                        if first_audio is None:
                            first_audio = trace.now()
                            trace.mark_ttfa()
                        audio_ms += len(value.samples) * 1000.0 / value.sample_rate
                        await self.send(
                            "response.output_audio.delta", delta=self._encode(value.samples, value.sample_rate), **base
                        )
            else:
                async for text in text_deltas():
                    msg = await open_message()
                    text_parts.append(text)
                    part["text"] += text
                    await self.send(
                        "response.output_text.delta",
                        response_id=response_id,
                        item_id=msg["id"],
                        output_index=0,
                        content_index=0,
                        delta=text,
                    )
        except asyncio.CancelledError:
            status, details = "cancelled", {"type": "cancelled", "reason": self._cancel_reason}
        except BoloError as exc:
            status, details = "failed", {"type": "failed", "error": {"type": exc.type, "code": exc.code}}
            await self.error(exc.message, code=exc.code, type_=exc.type)
        except Exception as exc:
            log.exception("realtime response failed")
            status, details = "failed", {"type": "failed", "error": {"type": "server_error", "code": "internal_error"}}
            await self.error(f"{type(exc).__name__}: {exc}", code="internal_error", type_="server_error")

        end = trace.now()
        if first_token is not None:
            trace.add_phase("llm_first_token", t_llm, first_token)
            if first_audio is not None:
                trace.add_phase("synthesize_first", first_token, first_audio)
                trace.add_phase("synthesize_rest", first_audio, end)
            else:
                trace.add_phase("llm", first_token, end)
        full_text = "".join(text_parts)
        output: list[dict[str, Any]] = []
        if item is not None:
            base = {"response_id": response_id, "item_id": item["id"], "output_index": 0, "content_index": 0}
            item["status"] = "completed" if status == "completed" else "incomplete"
            if audio_mode:
                await self.send("response.output_audio.done", **base)
                await self.send("response.output_audio_transcript.done", transcript=full_text, **base)
                wire = {"type": "audio", "transcript": full_text}
            else:
                await self.send("response.output_text.done", text=full_text, **base)
                wire = {"type": "text", "text": full_text}
            await self.send("response.content_part.done", part=wire, **base)
            await self.send("response.output_item.done", response_id=response_id, output_index=0, item=item)
            await self.send("conversation.item.done", item=item, previous_item_id=previous)
            output.append(item)
        if status == "completed":
            for index in sorted(calls):  # function calls requested by the model
                call = calls[index]
                call_item = {
                    "id": new_id("item"), "object": "realtime.item", "type": "function_call", "status": "completed",
                    "name": call["name"], "call_id": call["id"] or new_id("call"), "arguments": call["arguments"] or "{}",
                }  # fmt: skip
                out_index = len(output)
                before = self.items[-1]["id"] if self.items else None
                self.items.append(call_item)
                opening = {**call_item, "status": "in_progress", "arguments": ""}
                await self.send(
                    "response.output_item.added", response_id=response_id, output_index=out_index, item=opening
                )
                await self.send("conversation.item.added", item=opening, previous_item_id=before)
                common = {
                    "response_id": response_id,
                    "item_id": call_item["id"],
                    "output_index": out_index,
                    "call_id": call_item["call_id"],
                }
                await self.send("response.function_call_arguments.delta", delta=call_item["arguments"], **common)
                await self.send(
                    "response.function_call_arguments.done",
                    arguments=call_item["arguments"],
                    name=call_item["name"],
                    **common,
                )
                await self.send(
                    "response.output_item.done", response_id=response_id, output_index=out_index, item=call_item
                )
                await self.send("conversation.item.done", item=call_item, previous_item_id=before)
                output.append(call_item)
            if finish_reason == "length":
                status, details = "incomplete", {"type": "incomplete", "reason": "max_output_tokens"}
        response.update(
            status=status, status_details=details, output=output,
            usage={
                "input_tokens": int(usage.get("prompt_tokens") or 0), "output_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            },
        )  # fmt: skip
        with contextlib.suppress(Exception):
            await self.send("response.done", response=response)
        trace.text_preview = preview(full_text) if full_text else None
        trace.chars = len(full_text) or None
        trace.extra.update(
            llm_model=body["model"],
            audio_seconds_out=round(audio_ms / 1000.0, 2),
            response_id=response_id,
            tool_calls=len(calls),
        )
        trace.finish(
            {"completed": "ok", "incomplete": "ok", "cancelled": "cancelled"}.get(status, "error"),
            None if status != "failed" else "response failed",
        )
        core.traces.record(trace)
        self._response_id = None


# -- ephemeral client secrets -------------------------------------------------
SECRET_PREFIX = "ek_"


def mint_client_secret(core: Bolo, session: dict[str, Any] | None, seconds: int = 600) -> dict[str, Any]:
    """Create a short-lived key that can only open Realtime sessions (``POST /v1/realtime/client_secrets``).

    The session configuration given here is applied to every session opened with the secret, before
    anything the client sends. Secrets live in memory: a restart invalidates them.
    """
    if not 10 <= seconds <= 7200:
        raise RealtimeError("expires_after.seconds must be between 10 and 7200", param="expires_after.seconds")
    config = dict(session or {})

    async def nowhere(_event: dict[str, Any]) -> None:
        return None

    probe = RealtimeSession(core, nowhere, transcription=config.get("type") == "transcription")
    probe.configure(config)  # validates; raises RealtimeError / BoloError on a bad configuration
    now = time.time()
    for value in [v for v, (expires, _) in core.client_secrets.items() if expires < now]:
        del core.client_secrets[value]
    value = SECRET_PREFIX + secrets.token_urlsafe(32)
    expires_at = int(now) + int(seconds)
    core.client_secrets[value] = (float(expires_at), config)
    return {"value": value, "expires_at": expires_at, "session": probe.session_object()}


def client_secret_session(core: Bolo, presented: str | None) -> dict[str, Any] | None:
    """The session configuration behind a valid, unexpired client secret; None for anything else."""
    if not presented or not presented.startswith(SECRET_PREFIX):
        return None
    for value, (expires, config) in list(core.client_secrets.items()):
        if hmac.compare_digest(value.encode(), presented.encode()):
            if expires < time.time():
                del core.client_secrets[value]
                return None
            return config
    return None


async def serve(
    core: Bolo,
    receive: Callable[[], Awaitable[str | bytes | None]],
    send: Send,
    *,
    model: str | None,
    transcription: bool,
    config: dict[str, Any] | None = None,
) -> None:
    """Drive one Realtime connection until the client goes away (``receive`` returns None).

    ``config`` is a session configuration to apply before ``session.created`` (from a client secret).
    """
    if config:
        transcription = transcription or config.get("type") == "transcription"
    session = RealtimeSession(core, send, model=model, transcription=transcription)
    if config:
        session.configure(config)
    await session.start()
    try:
        while True:
            message = await receive()
            if message is None:
                return
            try:
                event = json.loads(message)
                if not isinstance(event, dict):
                    raise ValueError("not an object")
            except ValueError:
                await session.error("events must be JSON objects", code="invalid_json")
                continue
            await session.handle(event)
    finally:
        await session.close()
