"""A full-duplex model behind a Realtime (WebSocket or WebRTC) session.

The OpenAI Realtime API is built around turns: the user speaks, the server answers with a *response*. A full-duplex
model has no turns, so the bridge maps its continuous stream onto that event set without changing what the model
does:

**Audio in.** Everything the caller sends goes to the model as it arrives (resampled from bolo's 16 kHz to the
model's rate), whether or not anyone is speaking: the model hears silence, the caller and itself (through the
caller's microphone: use echo cancellation or headphones). A model steps once per frame of input, so if the caller
stops sending (a WebSocket client that only sends while someone talks, a network hiccup), the bridge feeds the
model silence once the input falls more than ``jitter_ms`` behind the wall clock; audio that arrives later is
still sent, at once. The model therefore keeps listening and speaking in real time whatever the client does.

**Audio and text out.** The model's output is cut into *responses*, one per stretch of speech: a response starts
when the model says a word (a text token) or its audio gets louder than about -40 dBFS, with the 160 ms before
that, and ends after ``silence_ms`` of output without either. Each response is the usual sequence
(``response.created``, ``response.output_item.added``, ``response.content_part.added``, ``response.output_audio.delta``
and ``response.output_audio_transcript.delta`` as the model produces them, the ``.done`` events,
``response.done``), so stock clients play and display it; over WebRTC the audio goes on the outgoing track. With
``output_modalities: ["text"]`` only the text is sent (``response.output_text.delta``).

**Barge-in.** The caller can talk over the model: the model hears it and decides itself whether to stop.
bolo's voice-activity detection still reports ``input_audio_buffer.speech_started`` / ``speech_stopped`` and
the user's transcript (an input item per utterance, ``realtime_transcribe_user``), but it does not cancel the
model's response. ``response.cancel`` (and ``output_audio_buffer.clear`` over WebRTC) does: that response ends
as ``cancelled`` and what the model says until its next pause is dropped. ``response.create`` and text input
are refused (the model answers speech, by itself).

**Closing.** The model's session is opened with the first audio and closed with the Realtime session, which gives
its slot back to the server (one conversation at a time on Kyutai's Python servers).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.audio.resample import StreamResampler
from bolo.duplex.base import DuplexAudio, DuplexConfig, DuplexEnd, DuplexSession, DuplexText
from bolo.engines.base import FloatArray
from bolo.errors import BoloError
from bolo.obs.trace import preview

if TYPE_CHECKING:
    from bolo.realtime import RealtimeSession

log = logging.getLogger("bolo.duplex")

SR = 16000
#: output louder than this (RMS over a 40 ms window) counts as the model speaking: about -40 dBFS
SPEECH_RMS = 0.01
PREROLL_S = 0.16
WINDOW_S = 0.04


class DuplexBridge:
    def __init__(self, session: RealtimeSession, *, silence_ms: int = 800, jitter_ms: int = 200) -> None:
        self.session = session
        self.silence_s = max(0.2, silence_ms / 1000.0)
        self.jitter_s = max(0.04, jitter_ms / 1000.0)
        self.model: DuplexSession | None = None
        self._open_lock = asyncio.Lock()
        self._resampler: StreamResampler | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._closed = False
        self.failed = False
        # input clock
        self._t0: float | None = None
        self.sent_samples = 0  # at the model's input rate, audio and padding
        self.padded_samples = 0
        self._push_lock = asyncio.Lock()
        # output segmentation, in samples of the model's output
        self._out_rate = 24000
        self._out_pos = 0
        self._last_active = -(10**9)
        self._preroll: list[FloatArray] = []
        self._window = np.zeros(0, dtype=np.float32)
        self._muted = False  # a cancelled response: drop output until the model pauses
        # the response in progress
        self._resp: dict[str, Any] | None = None
        self.responses = 0

    # -- lifecycle --------------------------------------------------------------------------------------------
    async def _ensure(self) -> DuplexSession | None:
        if self.model is not None or self._closed or self.failed:
            return self.model
        async with self._open_lock:
            if self.model is None and not self._closed and not self.failed:
                s = self.session
                backend = await s.core.s2s.duplex()
                config = DuplexConfig(instructions=s.instructions, voice=s.voice, options=dict(s.duplex_options))
                try:
                    self.model = await backend.open(config)
                except BoloError as exc:
                    self.failed = True
                    await s.error(exc.message, code=exc.code, type_=exc.type)
                    return None
                self._resampler = StreamResampler(SR, self.model.input_rate)
                self._t0 = time.monotonic()
                self._tasks = [
                    asyncio.create_task(self._read(), name=f"bolo-duplex-out-{s.id}"),
                    asyncio.create_task(self._clock(), name=f"bolo-duplex-clock-{s.id}"),
                ]
        return self.model

    async def start(self) -> None:
        """Open the model's session now (else it opens with the first audio)."""
        await self._ensure()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in self._tasks:
            if not task.done() and task is not asyncio.current_task():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        model, self.model = self.model, None
        if model is not None:
            with contextlib.suppress(Exception):
                await model.close()

    def describe(self) -> dict[str, Any]:
        rate = self.model.input_rate if self.model is not None else 24000
        return {
            "open": self.model is not None, "responses": self.responses,
            "input_seconds": round(self.sent_samples / rate, 2), "padded_seconds": round(self.padded_samples / rate, 2),
        }  # fmt: skip

    # -- audio in ---------------------------------------------------------------------------------------------
    async def push(self, samples16: FloatArray) -> None:
        model = await self._ensure()
        if model is None or self._resampler is None or not len(samples16):
            return
        x = self._resampler.process(np.asarray(samples16, dtype=np.float32))
        await self._send(model, x)

    async def _send(self, model: DuplexSession, x: FloatArray, padding: bool = False) -> None:
        if not len(x):
            return
        async with self._push_lock:
            try:
                await model.push(x)
            except BoloError as exc:
                if not self.failed:
                    self.failed = True
                    await self.session.error(exc.message, code=exc.code, type_=exc.type)
                return
            self.sent_samples += len(x)
            if padding:
                self.padded_samples += len(x)

    async def _clock(self) -> None:
        """Feed the model silence when the caller's audio falls behind the wall clock by more than the jitter."""
        model = self.model
        assert model is not None and self._t0 is not None
        rate = model.input_rate
        while not self._closed and not self.failed:
            await asyncio.sleep(0.04)
            due = int((time.monotonic() - self._t0 - self.jitter_s) * rate)
            behind = due - self.sent_samples
            if behind >= model.frame_samples // 2:
                await self._send(model, np.zeros(behind, dtype=np.float32), padding=True)

    # -- audio and text out -----------------------------------------------------------------------------------
    async def _read(self) -> None:
        model = self.model
        assert model is not None
        try:
            async for event in model.events():
                if isinstance(event, DuplexAudio):
                    await self._audio(event)
                elif isinstance(event, DuplexText):
                    await self._text(event.text)
                elif isinstance(event, DuplexEnd):
                    await self._finish("completed")
                    if event.reason != "closed" and not self._closed:
                        self.failed = True
                        message = (
                            f"the full-duplex model failed: {event.error}"
                            if event.reason == "failed"
                            else "the full-duplex model ended the conversation (it has a maximum length); start a new session"
                        )
                        await self.session.error(message, code="session_ended", type_="server_error")
                    return
        except asyncio.CancelledError:
            await self._finish("cancelled", "client_cancelled", record=True)
            raise
        except Exception:
            log.exception("full-duplex output failed")
            await self._finish("failed")

    async def _audio(self, ev: DuplexAudio) -> None:
        self._out_rate = ev.sample_rate
        x = ev.samples
        start = self._out_pos
        self._out_pos += len(x)
        window = int(WINDOW_S * ev.sample_rate)
        self._window = np.concatenate([self._window, x])
        loud = False
        while len(self._window) >= window:
            chunk, self._window = self._window[:window], self._window[window:]
            if float(np.sqrt(np.mean(chunk * chunk))) > SPEECH_RMS:
                loud = True
        if loud:
            self._last_active = self._out_pos
        quiet = (self._out_pos - self._last_active) / ev.sample_rate >= self.silence_s
        if self._muted:
            if quiet:
                self._muted = False
            return
        if self._resp is None:
            if loud:
                await self._begin(start)
                await self._emit_audio(x)
            else:
                self._preroll.append(x)
                keep = int(PREROLL_S * ev.sample_rate)
                while sum(len(p) for p in self._preroll) - len(self._preroll[0]) >= keep:
                    self._preroll.pop(0)
            return
        await self._emit_audio(x)
        if quiet:
            await self._finish("completed")

    async def _text(self, text: str) -> None:
        self._last_active = self._out_pos
        if self._muted:
            return
        if self._resp is None:
            await self._begin(self._out_pos)
        resp = self._resp
        assert resp is not None
        if not resp["text"]:
            text = text.lstrip()
            if not text:
                return
        resp["text"] += text
        resp["sentence"] += text
        if not resp["sentence_ms"]:
            resp["sentence_ms"] = resp["audio_ms"]
        s = self.session
        item = resp["item"]
        if resp["sentence"].rstrip().endswith((".", "!", "?")):
            s._item_audio[item["id"]].append((resp["sentence_ms"], resp["sentence"].strip()))
            resp["sentence"], resp["sentence_ms"] = "", 0.0
        base = {"response_id": resp["id"], "item_id": item["id"], "output_index": 0, "content_index": 0}
        if resp["audio_mode"]:
            resp["part"]["transcript"] += text
            await s.send("response.output_audio_transcript.delta", delta=text, **base)
        else:
            resp["part"]["text"] += text
            await s.send("response.output_text.delta", delta=text, **base)

    async def _emit_audio(self, x: FloatArray) -> None:
        resp = self._resp
        if resp is None or not len(x):
            return
        resp["audio_ms"] += len(x) * 1000.0 / self._out_rate
        if not resp["audio_mode"]:
            return
        s = self.session
        if resp["first_audio"] is None:
            resp["first_audio"] = resp["trace"].now()
            resp["trace"].mark_ttfa()
        base = {"response_id": resp["id"], "item_id": resp["item"]["id"], "output_index": 0, "content_index": 0}
        await s.send("response.output_audio.delta", delta=s._encode(x, self._out_rate), **base)

    async def _begin(self, start: int) -> None:
        from bolo.realtime import new_id

        s = self.session
        trace = s._agent_trace()
        trace.extra["full_duplex"] = True
        response_id = new_id("resp")
        audio_mode = "audio" in s.output_modalities
        voice = s.voice
        response: dict[str, Any] = {
            "id": response_id, "object": "realtime.response", "status": "in_progress", "status_details": None,
            "output": [], "conversation_id": s.conversation_id, "output_modalities": list(s.output_modalities),
            "usage": None, "audio": {"output": {"format": s.out_format, "voice": voice}}, "metadata": None,
        }  # fmt: skip
        previous = s.items[-1]["id"] if s.items else None
        content: list[dict[str, Any]] = []
        item: dict[str, Any] = {
            "id": new_id("item"), "object": "realtime.item", "type": "message", "role": "assistant",
            "status": "in_progress", "content": content,
        }  # fmt: skip
        part: dict[str, Any] = (
            {"type": "output_audio", "transcript": ""} if audio_mode else {"type": "output_text", "text": ""}
        )
        content.append(part)
        s.items.append(item)
        s._item_audio[item["id"]] = []
        s._response_id = response_id
        self._resp = {
            "id": response_id, "response": response, "item": item, "part": part, "audio_mode": audio_mode,
            "text": "", "sentence": "", "sentence_ms": 0.0, "audio_ms": 0.0, "trace": trace, "first_audio": None,
            "previous": previous, "since_user_ms": s.duplex_since_user_ms(),
        }  # fmt: skip
        self.responses += 1
        await s.send("response.created", response=response)
        await s.send(
            "response.output_item.added", response_id=response_id, output_index=0, item={**item, "content": []}
        )
        await s.send("conversation.item.added", item={**item, "content": []}, previous_item_id=previous)
        wire = {"type": "audio", "transcript": ""} if audio_mode else {"type": "text", "text": ""}
        await s.send(
            "response.content_part.added", response_id=response_id, item_id=item["id"], output_index=0, content_index=0,
            part=wire,
        )  # fmt: skip
        preroll, self._preroll = self._preroll, []
        for piece in preroll:
            await self._emit_audio(piece)

    async def _finish(self, status: str, reason: str | None = None, *, record: bool = True) -> None:
        resp, self._resp = self._resp, None
        self._preroll = []
        if resp is None:
            return
        s = self.session
        item = resp["item"]
        if resp["sentence"].strip():
            s._item_audio[item["id"]].append((resp["sentence_ms"], resp["sentence"].strip()))
        text = resp["text"]
        base = {"response_id": resp["id"], "item_id": item["id"], "output_index": 0, "content_index": 0}
        item["status"] = "completed" if status == "completed" else "incomplete"
        with contextlib.suppress(Exception):
            if resp["audio_mode"]:
                await s.send("response.output_audio.done", **base)
                await s.send("response.output_audio_transcript.done", transcript=text, **base)
                wire = {"type": "audio", "transcript": text}
            else:
                await s.send("response.output_text.done", text=text, **base)
                wire = {"type": "text", "text": text}
            await s.send("response.content_part.done", part=wire, **base)
            await s.send("response.output_item.done", response_id=resp["id"], output_index=0, item=item)
            await s.send("conversation.item.done", item=item, previous_item_id=resp["previous"])
            response = resp["response"]
            details = {"type": "cancelled", "reason": reason or "client_cancelled"} if status == "cancelled" else None
            response.update(
                status=status, status_details=details, output=[item],
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )  # fmt: skip
            await s.send("response.done", response=response)
        if s._response_id == resp["id"]:
            s._response_id = None
        if record:
            trace = resp["trace"]
            trace.text_preview = preview(text) if text else None
            trace.chars = len(text) or None
            trace.extra.update(
                llm_model=s.core.s2s.model_name(), audio_seconds_out=round(resp["audio_ms"] / 1000.0, 2),
                response_id=resp["id"], tool_calls=0,
            )  # fmt: skip
            if resp["since_user_ms"] is not None:
                trace.extra["started_after_user_ms"] = resp["since_user_ms"]
            trace.finish({"completed": "ok", "cancelled": "cancelled"}.get(status, "error"))
            s.core.traces.record(trace)

    # -- client actions ---------------------------------------------------------------------------------------
    @property
    def speaking(self) -> bool:
        return self._resp is not None

    async def cancel(self, reason: str = "client_cancelled") -> bool:
        """End the response in progress as cancelled and drop the model's output until it pauses."""
        if self._resp is None:
            return False
        self._muted = True
        await self._finish("cancelled", reason)
        return True
