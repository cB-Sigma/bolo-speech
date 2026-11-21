"""The provider-neutral call session: one phone call bridged into one Realtime session.

A :class:`CallSession` sits between a provider's media stream (through its :class:`~bolo.telephony.adapters.Wire`)
and a :class:`~bolo.realtime.RealtimeSession`. The Realtime session does what it does for every transport: turn
detection, transcription, the pipeline (cascade, audio-llm or speech-to-speech), tools and traces. The call
session adds what only a phone call has:

* the caller's audio arrives in the provider's codec and rate; the agent's goes back in frames the provider
  plays in real time, so bolo keeps count of what has been played and what is still queued;
* **barge-in**: when the caller starts speaking, the audio queued at the provider is cleared and the assistant's
  message is truncated to what was heard;
* a recording-consent announcement, a greeting, "are you still there?" prompts, a longest call length;
* key presses (DTMF), hanging up, transfer to a person, voicemail on outbound calls;
* the call record (:mod:`bolo.telephony.records`): timeline, per-turn latency, transcript, outcome.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np

from bolo.audio.codecs import Framer, decode_frame, encode_frame
from bolo.audio.resample import StreamResampler
from bolo.engines.base import FloatArray
from bolo.errors import BoloError
from bolo.obs.trace import Trace, preview
from bolo.realtime import END_PADDING_S, SR, RealtimeError, RealtimeSession
from bolo.telephony.adapters import (
    Adapter,
    StreamAudio,
    StreamDtmf,
    StreamEvent,
    StreamMark,
    StreamStart,
    StreamStop,
)
from bolo.telephony.records import CallRecord, Turn

if TYPE_CHECKING:
    from bolo.telephony.config import AgentConfig
    from bolo.telephony.manager import PendingCall, TelephonyManager

log = logging.getLogger("bolo.telephony")

#: key presses are handed to the agent after this long without another one (or at once on "#")
DTMF_GAP_S = 1.5
#: leaving a voicemail: speak once the far end has been quiet this long, and wait at most this long for it
VOICEMAIL_QUIET_S = 1.2
VOICEMAIL_WAIT_S = 30.0
TOOL_TIMEOUT_S = 10.0
ENCODINGS = {"mulaw": "mulaw", "alaw": "alaw", "pcm16": "pcm_s16le"}

HANG_UP_TOOL = {
    "type": "function",
    "name": "hang_up",
    "description": "End the phone call. Use it after saying goodbye, or when the caller asks to end the call.",
    "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}},
}
TRANSFER_TOOL = {
    "type": "function",
    "name": "transfer_call",
    "description": "Transfer the caller to a person. Use it when the caller asks for a human or you cannot help.",
    "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}},
}


class _Utterance:
    """Something the agent says: where it started playing and how much of it has been sent."""

    def __init__(self, name: str, entry: dict[str, Any] | None = None, item_id: str | None = None) -> None:
        self.name = name
        self.entry = entry
        self.item_id = item_id
        self.turn: Turn | None = None
        self.start: float | None = None
        self.sent_ms = 0.0


class CallSession:
    def __init__(
        self,
        manager: TelephonyManager,
        adapter: Adapter,
        pending: PendingCall | None,
        send_text: Callable[[str], Awaitable[None]],
        close: Callable[[], Awaitable[None]],
    ) -> None:
        self.manager = manager
        self.core = manager.core
        self.adapter = adapter
        self.pending = pending
        self.wire = adapter.wire()
        self._send_text = send_text
        self._close_socket = close
        self.rt: RealtimeSession | None = None
        self.record: CallRecord | None = pending.record if pending else None
        self.agent: AgentConfig | None = pending.agent if pending else None
        self._in_resampler: StreamResampler | None = None
        self._out_resampler: StreamResampler | None = None
        self._framer = Framer(self.wire.frame_bytes)
        # playback as the provider is assumed to perform it: frames play in real time, in order
        self._play_end = 0.0
        self._protected_until = 0.0  # no barge-in before this (consent, voicemail message)
        self._sending_protected = False
        self._utt: _Utterance | None = None
        self._utterances = 0
        self._speech_lock = asyncio.Lock()
        self._lock_owner: str | None = None
        self._say_cancel = False
        # the turn being timed
        self._turn: Turn | None = None
        self._marks: dict[str, float] = {}
        self._next_kind = "prompt"
        self._pending_truncate: tuple[str, int] | None = None
        self._response_utt: _Utterance | None = None
        # housekeeping
        self._tasks: set[asyncio.Task[Any]] = set()
        self._digits = ""
        self._dtmf_task: asyncio.Task[Any] | None = None
        self._idle_since: float | None = None
        self._prompts_done = 0
        self._busy = 0  # tools running, digits pending, opening speech
        self._ending = False
        self._ended = False
        self._finalized = False
        self.started = False

    # -- plumbing -----------------------------------------------------------------------------------------
    def _spawn(self, coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(self._guard(coro), name=f"bolo-call-{name}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _guard(self, coro: Coroutine[Any, Any, Any]) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("telephony call task failed")
            if self.record is not None:
                self.record.event("error", message=f"{type(exc).__name__}: {exc}")

    async def _send(self, frame: str | None) -> None:
        if frame is None or self._finalized:
            return
        with contextlib.suppress(Exception):
            await self._send_text(frame)

    @property
    def playing(self) -> bool:
        return time.perf_counter() < self._play_end

    def _bytes_per_ms(self) -> float:
        return self.wire.rate * (2 if self.wire.encoding == "pcm16" else 1) / 1000.0

    # -- provider → bolo ------------------------------------------------------------------------------------
    async def handle_message(self, message: str) -> None:
        for event in self.wire.parse(message):
            await self._on_wire(event)

    async def handle_event(self, event: StreamEvent) -> None:
        """For transports that do not speak a provider's JSON (SIP): a stream event as it is."""
        await self._on_wire(event)

    async def _on_wire(self, event: StreamEvent) -> None:
        if isinstance(event, StreamStart):
            await self._start(event)
        elif self.rt is None or self._finalized:
            return
        elif isinstance(event, StreamAudio):
            await self._on_audio(event.payload)
        elif isinstance(event, StreamDtmf):
            await self._on_dtmf(event.digit)
        elif isinstance(event, StreamMark):
            utt = self._utt
            if utt is not None and event.name == utt.name:
                self._play_end = min(self._play_end, time.perf_counter())
        elif isinstance(event, StreamStop):
            await self.finalize("caller_hangup")

    async def _on_audio(self, payload: bytes) -> None:
        rt, resampler = self.rt, self._in_resampler
        if not payload or rt is None or resampler is None:
            return
        samples = resampler.process(decode_frame(payload, ENCODINGS[self.wire.encoding]))
        try:
            await rt.accept_audio(samples)
        except (RealtimeError, BoloError) as exc:
            if self.record is not None:
                self.record.event("error", message=exc.message)

    async def _start(self, event: StreamStart) -> None:
        if self.started:
            return
        manager = self.manager
        pending = self.pending or manager.claim_stream(self.adapter.name, event)
        if pending is None or pending.session not in (None, self):
            log.warning("%s stream %s: no agent answers this call", self.adapter.name, event.stream_id)
            await self._close_socket()
            return
        self.started = True
        self.pending, pending.session = pending, self
        record = self.record = pending.record
        agent = self.agent = pending.agent
        record.stream_id = event.stream_id or record.stream_id
        record.provider_call_id = record.provider_call_id or event.call_id
        record.from_number = record.from_number or event.from_number
        record.to_number = record.to_number or event.to_number
        record.begin()
        record.event("stream_started", encoding=self.wire.encoding, sample_rate=self.wire.rate)
        self._framer = Framer(self.wire.frame_bytes)
        self._in_resampler = StreamResampler(self.wire.rate, SR)
        rt = RealtimeSession(self.core, self._on_rt, transport="telephony")
        rt.trace_extra = {"call_id": record.id, "agent": agent.name, "telephony_provider": self.adapter.name}
        tools = [
            {"type": "function", **{k: v for k, v in t.items() if k not in ("url", "timeout_s")}} for t in agent.tools
        ]
        names = {t["name"] for t in agent.tools}
        builtin = [HANG_UP_TOOL]
        if agent.transfer_number and self.adapter.can_transfer:
            builtin.append(TRANSFER_TOOL)
        tools += [t for t in builtin if t["name"] not in names]
        audio_in: dict[str, Any] = {"transcription": {"model": agent.stt_model, "language": agent.language}}
        if agent.turn_detection:
            audio_in["turn_detection"] = agent.turn_detection
        audio_out: dict[str, Any] = {"voice": agent.voice}
        if agent.romanised:
            audio_out["romanised"] = agent.romanised
        if self.wire.rate == 8000 and self.wire.encoding in ("mulaw", "alaw"):
            audio_out["format"] = {"type": "audio/pcmu" if self.wire.encoding == "mulaw" else "audio/pcma"}
        config: dict[str, Any] = {"type": "realtime", "audio": {"input": audio_in, "output": audio_out}, "tools": tools}
        if agent.instructions:
            config["instructions"] = agent.instructions
        if agent.pipeline:
            config["pipeline"] = agent.pipeline
        if agent.pipeline_id:
            config["pipeline_id"] = agent.pipeline_id
        try:
            rt.configure(config)
        except (RealtimeError, BoloError) as exc:
            await self.end("error", f"agent '{agent.name}': {exc.message}")
            return
        rt.tts_model, rt.llm_model = agent.tts_model, agent.llm_model
        record.pipeline = rt.pipeline
        self.rt = rt
        await rt.start()
        manager.publish(record, "started")
        self._spawn(self._opening(), "opening")
        self._spawn(self._watchdog(), "watchdog")

    # -- bolo → provider ----------------------------------------------------------------------------------
    def _begin_utterance(self, entry: dict[str, Any] | None = None, item_id: str | None = None) -> _Utterance:
        self._utterances += 1
        self._utt = _Utterance(f"u{self._utterances}", entry, item_id)
        self._out_resampler = None
        return self._utt

    async def _emit(self, raw: bytes, utt: _Utterance) -> None:
        """Queue bytes in the wire's encoding; full frames go to the provider."""
        for frame in self._framer.push(raw):
            await self._send_frame(frame, utt)

    async def _send_frame(self, frame: bytes, utt: _Utterance) -> None:
        now = time.perf_counter()
        ms = len(frame) / self._bytes_per_ms()
        self._play_end = max(now, self._play_end)
        if utt.start is None:
            utt.start = self._play_end
        utt.sent_ms += ms
        self._play_end += ms / 1000.0
        if self._sending_protected:
            self._protected_until = self._play_end
        await self._send(self.wire.media(frame))

    async def _end_utterance(self, utt: _Utterance) -> None:
        pad = b"\x00" if self.wire.encoding == "pcm16" else b"\xff" if self.wire.encoding == "mulaw" else b"\xd5"
        tail = self._framer.flush(pad)
        if tail:
            await self._send_frame(tail, utt)
        if utt.start is not None:
            await self._send(self.wire.mark(utt.name))

    def _encode(self, samples: FloatArray, rate: int) -> bytes:
        if self._out_resampler is None or self._out_resampler.sr_in != rate:
            self._out_resampler = StreamResampler(rate, self.wire.rate)
        return encode_frame(self._out_resampler.process(samples), ENCODINGS[self.wire.encoding])

    async def _drain(self, limit_s: float = 60.0) -> None:
        """Wait until the provider has played what was sent."""
        deadline = time.perf_counter() + limit_s
        while self.playing and time.perf_counter() < deadline and not self._finalized:  # noqa: ASYNC110
            await asyncio.sleep(min(0.1, max(0.01, self._play_end - time.perf_counter())))
        await asyncio.sleep(0.15)  # the network and the provider's jitter buffer

    def _resolve_tts(self) -> str:
        agent = self.agent
        assert agent is not None
        spec = self.core.resolve(agent.tts_model, "tts") if agent.tts_model else self.core.realtime_tts(agent.voice)
        return spec.id

    async def say(self, text: str, *, kind: str = "prompt", interruptible: bool = True) -> bool:
        """Speak ``text`` with the agent's voice, outside the language model (greeting, prompts, announcements)."""
        rt, record, agent = self.rt, self.record, self.agent
        if rt is None or record is None or agent is None or self._finalized or not text.strip():
            return False
        async with self._speech_lock:
            self._say_cancel = False
            self._sending_protected = not interruptible
            entry = record.say("agent", text, kind=kind)
            record.event(kind, text=preview(text))
            utt = self._begin_utterance(entry)
            trace = Trace("tts", "telephony", "")
            gen = self.core.synthesize(
                text, model=self._resolve_tts(), voice=agent.voice, trace=trace,
                romanised=agent.romanised, romanised_strict=False,
            )  # fmt: skip
            try:
                async for chunk in gen:
                    trace.mark_ttfa()
                    if self._say_cancel or self._finalized:
                        break
                    await self._emit(self._encode(chunk.samples, chunk.sample_rate), utt)
                if not self._say_cancel:
                    await self._end_utterance(utt)
            except BoloError as exc:
                record.event("error", message=exc.message)
                return False
            finally:
                with contextlib.suppress(Exception):
                    await gen.aclose()
                self.core.complete_tts(trace)
                self._sending_protected = False
        if not self._finalized:  # the language model should know what was said
            item = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
            await rt.handle({"type": "conversation.item.create", "item": item})
        return True

    async def _opening(self) -> None:
        agent, pending = self.agent, self.pending
        assert agent is not None and pending is not None
        self._busy += 1
        try:
            if pending.mode == "voicemail":
                await self._leave_message(pending.voicemail_stage)
                return
            if agent.recording_consent:
                await self.say(agent.recording_consent, kind="consent", interruptible=False)
            if agent.greeting and pending.mode != "voicemail":
                await self.say(agent.greeting, kind="greeting")
        finally:
            self._busy -= 1

    # -- barge-in -----------------------------------------------------------------------------------------
    def _heard_ms(self, utt: _Utterance) -> int:
        if utt.start is None:
            return 0
        return int(max(0.0, min(utt.sent_ms, (time.perf_counter() - utt.start) * 1000.0)))

    async def _interrupt(self, cause: str) -> bool:
        """Stop what the agent is saying: clear the provider's queue and forget what was not heard."""
        now = time.perf_counter()
        if now < self._protected_until or self._sending_protected:
            return False
        utt = self._utt
        sending = self._speech_lock.locked()
        if not (self.playing or sending) or utt is None:
            return False
        heard = self._heard_ms(utt)
        self._say_cancel = True
        self._framer = Framer(self.wire.frame_bytes)
        self._play_end = now
        await self._send(self.wire.clear())
        if self.record is not None:
            self.record.event("barge_in", cause=cause, heard_ms=heard, cleared_ms=int(max(0.0, utt.sent_ms - heard)))
        if utt.entry is not None:
            utt.entry["interrupted"] = True
            utt.entry["heard_ms"] = heard
        if utt.turn is not None:
            utt.turn.interrupted = True
        if utt.item_id and self.rt is not None:
            if self._lock_owner is not None:  # still being generated: truncate once the response has stopped
                self._pending_truncate = (utt.item_id, heard)
            else:
                await self._truncate(utt, heard)
        return True

    async def _truncate(self, utt: _Utterance, heard_ms: int) -> None:
        rt = self.rt
        if rt is None or not utt.item_id:
            return
        await rt.handle(
            {"type": "conversation.item.truncate", "item_id": utt.item_id, "content_index": 0, "audio_end_ms": heard_ms}
        )
        item = next((i for i in rt.items if i.get("id") == utt.item_id), None)
        if item is not None and utt.entry is not None:
            heard = " ".join(str(p.get("transcript") or "") for p in item.get("content", [])).strip()
            utt.entry["full_text"] = utt.entry.get("full_text") or utt.entry["text"]
            utt.entry["text"] = heard
            utt.entry["interrupted"] = True

    # -- Realtime session → call ----------------------------------------------------------------------------
    async def _on_rt(self, event: dict[str, Any]) -> None:
        record, rt = self.record, self.rt
        if record is None or rt is None or self._finalized:
            return
        kind = event["type"]
        now = time.perf_counter()
        if kind == "response.output_audio.delta":
            await self._on_response_audio(event, now)
        elif kind == "input_audio_buffer.speech_started":
            self._prompts_done = 0
            record.event("user_speech_started", audio_ms=event.get("audio_start_ms"))
            await self._interrupt("speech")
        elif kind == "input_audio_buffer.speech_stopped":
            self._close_turn()
            turn = self._turn = Turn(len(record.turns), "speech", record.now_ms())
            record.turns.append(turn)
            # the detector's wait: audio received since the last speech (audio_end_ms includes the padding)
            waited = rt.input_ms - int(event.get("audio_end_ms") or 0) + END_PADDING_S * 1000.0
            turn.endpoint_ms = float(max(0.0, waited))
            self._marks = {"start": now}
            record.event("user_speech_stopped", audio_ms=event.get("audio_end_ms"))
        elif kind == "conversation.item.input_audio_transcription.completed":
            text = str(event.get("transcript") or "").strip()
            heard = self._turn
            if heard is not None and heard.kind == "speech" and heard.user_text is None:
                heard.user_text = text
                heard.stt_ms = (now - self._marks["start"]) * 1000.0
                self._marks["stt"] = now
                if not text:
                    heard.status = "no_speech"
            if text:
                record.say("caller", text)
        elif kind == "conversation.item.input_audio_transcription.failed":
            record.event("error", message=str((event.get("error") or {}).get("message") or "transcription failed"))
        elif kind == "response.created":
            current = self._turn
            if current is None or current.status != "in_progress" or "response" in self._marks:
                self._close_turn()
                fresh = self._turn = Turn(len(record.turns), self._next_kind, record.now_ms())
                record.turns.append(fresh)
                self._marks = {"start": now}
            self._next_kind = "prompt"
            self._marks["response"] = now
        elif kind in ("response.output_audio_transcript.delta", "response.output_text.delta"):
            if self._turn is not None and "token" not in self._marks:
                self._marks["token"] = now
                self._turn.llm_first_token_ms = (now - self._marks.get("stt", self._marks["start"])) * 1000.0
        elif kind == "response.output_audio.done":
            if self._response_utt is not None:
                await self._end_utterance(self._response_utt)
        elif kind == "response.done":
            await self._on_response_done(event.get("response") or {})
        elif kind == "error":
            record.event("error", message=str((event.get("error") or {}).get("message") or "error"))
        elif kind == "pipeline.action":
            self._spawn(self._pipeline_action(event), "pipeline-action")

    async def _on_response_audio(self, event: dict[str, Any], now: float) -> None:
        rt, turn = self.rt, self._turn
        assert rt is not None
        response_id = str(event.get("response_id") or "")
        if self._lock_owner != response_id:
            await self._speech_lock.acquire()  # whatever is being said finishes first
            self._lock_owner = response_id
            self._say_cancel = False
            self._response_utt = self._begin_utterance(item_id=str(event.get("item_id") or "") or None)
            self._response_utt.turn = turn
            if turn is not None and "audio" not in self._marks:
                self._marks["audio"] = now
                turn.tts_first_audio_ms = (now - self._marks.get("token", self._marks["start"])) * 1000.0
        utt = self._response_utt
        if utt is None or self._say_cancel:
            return
        raw = base64.b64decode(event["delta"])
        if rt.out_codec == "pcm":
            raw = self._encode(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, rt.out_rate)
        first = utt.start is None
        await self._emit(raw, utt)
        if first and utt.start is not None and turn is not None and turn.total_ms is None:
            sent = time.perf_counter()
            turn.send_ms = (sent - self._marks.get("audio", sent)) * 1000.0
            turn.total_ms = (sent - self._marks["start"]) * 1000.0
            if self.record is not None:
                self.record.event("agent_audio_started", turn=turn.index, response_ms=round(turn.response_ms or 0.0, 1))

    def _close_turn(self) -> None:
        turn, self._turn = self._turn, None
        if turn is not None and turn.status == "in_progress":
            turn.status = "no_response"

    async def _on_response_done(self, response: dict[str, Any]) -> None:
        record, rt = self.record, self.rt
        assert record is not None and rt is not None
        status = str(response.get("status") or "")
        utt, self._response_utt = self._response_utt, None
        if self._lock_owner is not None:
            self._lock_owner = None
            self._speech_lock.release()
        text = ""
        calls: list[dict[str, Any]] = []
        for item in response.get("output") or []:
            if item.get("type") == "message":
                parts = item.get("content") or []
                text = " ".join(str(p.get("transcript") or p.get("text") or "") for p in parts).strip()
            elif item.get("type") == "function_call":
                calls.append(item)
        entry: dict[str, Any] | None = None
        if text:
            entry = record.say("agent", text)
            if utt is not None:
                utt.entry = entry
        if self._pending_truncate is not None and utt is not None:
            _item, heard = self._pending_truncate
            self._pending_truncate = None
            await self._truncate(utt, heard)
        turn, self._turn = self._turn, None
        if turn is not None:
            turn.assistant_text = (entry or {}).get("text") if entry else None
            turn.interrupted = turn.interrupted or status == "cancelled" or bool(entry and entry.get("interrupted"))
            turn.status = status or "completed"
            turn.tool_calls = [str(c.get("name")) for c in calls]
        if status == "failed":
            record.event("response_failed")
        if calls and status == "completed" and not self._ending:
            self._spawn(self._run_tools(calls, spoke=bool(text)), "tools")

    # -- tools ------------------------------------------------------------------------------------------------
    async def _run_tools(self, calls: list[dict[str, Any]], spoke: bool) -> None:
        rt, record, agent = self.rt, self.record, self.agent
        assert rt is not None and record is not None and agent is not None
        self._busy += 1
        action: str | None = None
        follow = False
        try:
            for call in calls:
                name, arguments = str(call.get("name") or ""), str(call.get("arguments") or "{}")
                record.event("tool_call", name=name, arguments=preview(arguments, 300))
                if name == "hang_up":
                    output, action = "The call is ending.", action or "hang_up"
                elif name == "transfer_call":
                    if agent.transfer_number and self.adapter.can_transfer:
                        output, action = "The caller is being transferred.", "transfer"
                    else:
                        output, follow = "Transfers are not available on this line.", True
                else:
                    output, follow = await self._call_tool(name, arguments), True
                    record.event("tool_result", name=name, output=preview(output, 300))
                item = {"type": "function_call_output", "call_id": call.get("call_id"), "output": output}
                await rt.handle({"type": "conversation.item.create", "item": item})
            if action == "transfer":
                self._ending = True
                if agent.transfer_message:
                    await self.say(agent.transfer_message, kind="transfer_message", interruptible=False)
                await self._drain()
                await self.transfer(str(agent.transfer_number))
            elif action == "hang_up":
                self._ending = True
                if not spoke and agent.goodbye:
                    await self.say(agent.goodbye, kind="goodbye", interruptible=False)
                await self._drain()
                await self.end("agent_hangup")
            elif follow and not self._finalized:
                self._next_kind = "tool"
                await rt.handle({"type": "response.create"})
        finally:
            self._busy -= 1

    async def _call_tool(self, name: str, arguments: str) -> str:
        agent, record = self.agent, self.record
        assert agent is not None and record is not None
        tool = next((t for t in agent.tools if t.get("name") == name), None)
        if tool is None or not tool.get("url"):
            return f"The tool '{name}' is not available."
        try:
            parsed = json.loads(arguments or "{}")
        except ValueError:
            parsed = {}
        body = {
            "name": name,
            "arguments": parsed,
            "call": {
                "id": record.id, "provider": record.provider, "direction": record.direction, "agent": agent.name,
                "from": record.from_number, "to": record.to_number, "dtmf": record.dtmf,
            },
        }  # fmt: skip
        try:
            timeout = float(tool.get("timeout_s") or TOOL_TIMEOUT_S)
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                resp = await client.post(str(tool["url"]), json=body)
        except (httpx.HTTPError, ValueError) as exc:
            return f"The tool failed: {type(exc).__name__}"
        if resp.status_code >= 400:
            return f"The tool failed with HTTP {resp.status_code}."
        return resp.text[:4000]

    async def _pipeline_action(self, event: dict[str, Any]) -> None:
        """A live pipeline's transfer or hangup step (bolo.pipelines.live): after what was said has played."""
        record, agent = self.record, self.agent
        assert record is not None and agent is not None
        action = str(event.get("action") or "")
        record.event("pipeline_action", action=action, to=event.get("to"))
        if action == "transfer":
            number = event.get("to") or agent.transfer_number
            if not number or not self.adapter.can_transfer:
                record.event("error", message="the pipeline asked for a transfer, and this line cannot transfer"
                                              + ("" if number else " (no transfer_number)"))  # fmt: skip
                return
            self._ending = True
            await self._drain()
            await self.transfer(str(number))
        elif action == "hangup":
            self._ending = True
            await self._drain()
            await self.end("agent_hangup")

    # -- key presses --------------------------------------------------------------------------------------
    async def _on_dtmf(self, digit: str) -> None:
        record, agent, rt = self.record, self.agent, self.rt
        if not digit or record is None or agent is None or rt is None:
            return
        record.dtmf += digit
        record.event("dtmf", digit=digit)
        self._prompts_done = 0
        self._idle_since = None
        if agent.dtmf == "ignore" or self._ending:
            return
        self._digits += digit
        await self._interrupt("dtmf")
        await rt.interrupt()
        if self._dtmf_task is not None:
            self._dtmf_task.cancel()
        self._dtmf_task = self._spawn(self._deliver_digits(0.0 if digit == "#" else DTMF_GAP_S), "dtmf")

    async def _deliver_digits(self, wait_s: float) -> None:
        await asyncio.sleep(wait_s)
        self._dtmf_task = None
        rt, record = self.rt, self.record
        digits, self._digits = self._digits, ""
        if not digits or rt is None or record is None or self._finalized:
            return
        text = f"[The caller pressed {' '.join(digits)} on the phone keypad]"
        record.say("caller", text, kind="dtmf")
        item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
        await rt.handle({"type": "conversation.item.create", "item": item})
        self._next_kind = "dtmf"
        await rt.handle({"type": "response.create"})

    # -- silence, length, voicemail -----------------------------------------------------------------------
    async def _watchdog(self) -> None:
        agent, rt, record = self.agent, self.rt, self.record
        assert agent is not None and rt is not None and record is not None
        started = time.perf_counter()
        while not self._finalized:
            await asyncio.sleep(0.2)
            now = time.perf_counter()
            if self._ending:
                continue
            if agent.max_duration_s and now - started >= agent.max_duration_s:
                self._ending = True
                record.event("max_duration", seconds=agent.max_duration_s)
                await self._interrupt("max_duration")
                await rt.interrupt()
                if agent.max_duration_message:
                    await self.say(agent.max_duration_message, kind="max_duration_message", interruptible=False)
                    await self._drain()
                await self.end("max_duration")
                return
            idle = not (
                self.playing or rt.responding or rt.user_speaking or self._busy or self._digits
                or self._speech_lock.locked()
            )  # fmt: skip
            if not idle:
                self._idle_since = None
                continue
            if self._idle_since is None:
                self._idle_since = now
            if not agent.silence_timeout_s or now - self._idle_since < agent.silence_timeout_s:
                continue
            self._idle_since = None
            if self._prompts_done < agent.silence_prompts:
                self._prompts_done += 1
                await self.say(agent.silence_prompt, kind="silence_prompt")
            else:
                self._ending = True
                record.event("no_response", prompts=self._prompts_done)
                if agent.goodbye:
                    await self.say(agent.goodbye, kind="goodbye", interruptible=False)
                    await self._drain()
                await self.end("no_response")
                return

    async def on_machine(self, decision: str, stage: str) -> None:
        """An answering machine was detected while the stream is up (Plivo, Telnyx)."""
        if self._finalized or self._ending:
            return
        if decision == "hangup":
            self._ending = True
            await self.end("voicemail")
        elif decision == "message":
            self._spawn(self._leave_message(stage), "voicemail")

    async def _leave_message(self, stage: str) -> None:
        rt, agent, record = self.rt, self.agent, self.record
        assert rt is not None and agent is not None and record is not None
        self._ending = True
        self._say_cancel = True
        with contextlib.suppress(RealtimeError, BoloError):  # the machine is not a caller: nothing answers it
            rt.configure({"audio": {"input": {"turn_detection": {"type": "server_vad", "create_response": False, "interrupt_response": False}}}})  # fmt: skip
        await rt.interrupt()
        await self._send(self.wire.clear())
        self._play_end = time.perf_counter()
        if stage != "end":  # the greeting may still be playing: wait for it to go quiet
            deadline = time.perf_counter() + VOICEMAIL_WAIT_S
            quiet_since: float | None = None
            while time.perf_counter() < deadline and not self._finalized:
                await asyncio.sleep(0.1)
                if rt.user_speaking:
                    quiet_since = None
                elif quiet_since is None:
                    quiet_since = time.perf_counter()
                elif time.perf_counter() - quiet_since >= VOICEMAIL_QUIET_S:
                    break
        if agent.voicemail_message:
            await self.say(agent.voicemail_message, kind="voicemail_message", interruptible=False)
            await self._drain()
        await self.end("voicemail")

    # -- endings ------------------------------------------------------------------------------------------
    async def transfer(self, number: str) -> None:
        record, pending = self.record, self.pending
        assert record is not None and pending is not None
        record.transfer_to = number
        record.event("transfer", to=number)
        if self.adapter.ends_by == "api":
            try:
                await self.adapter.transfer(pending, number)
            except BoloError as exc:
                record.event("error", message=exc.message)
                await self.end("error", exc.message)
                return
        await self.finalize("transferred")
        await self._close_socket()

    async def end(self, outcome: str, error: str | None = None) -> None:
        """End the call from bolo's side: hang up at the provider, then close the stream."""
        if self._ended:
            return
        self._ended = True
        self._ending = True
        if self.adapter.ends_by == "api" and self.pending is not None and not self._finalized:
            try:
                await self.adapter.hangup(self.pending)
            except BoloError as exc:
                if self.record is not None:
                    self.record.event("error", message=exc.message)
        await self.finalize(outcome, error)
        await self._close_socket()

    async def finalize(self, outcome: str, error: str | None = None) -> None:
        """Close the record and the Realtime session. The first outcome stands."""
        if self._finalized:
            return
        self._finalized = True
        self._close_turn()
        record, rt = self.record, self.rt
        current = asyncio.current_task()
        for task in list(self._tasks):
            if task is not current and not task.done():
                task.cancel()
        if rt is not None:
            with contextlib.suppress(Exception):
                await rt.close()
        if self._speech_lock.locked() and self._lock_owner is not None:
            self._lock_owner = None
            self._speech_lock.release()
        if record is not None and self.pending is not None:
            self.manager.close_record(self.pending, outcome, error)
