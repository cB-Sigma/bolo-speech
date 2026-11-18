"""The WebRTC transport of the Realtime API (``POST /v1/realtime/calls``).

A browser (or any WebRTC client) sends an SDP offer with one audio track and a
data channel. The session is the same :class:`~bolo.realtime.RealtimeSession`
that backs the WebSocket endpoint; only the plumbing differs:

* the caller's microphone arrives as an Opus audio track instead of
  ``input_audio_buffer.append`` events;
* the reply is played on an outgoing audio track instead of being sent as
  ``response.output_audio.delta`` events (those are not sent over WebRTC);
* every other event travels as JSON over the data channel (``oai-events``),
  plus ``output_audio_buffer.started`` / ``stopped`` / ``cleared``, which
  describe what the outgoing track is doing.

This module needs `aiortc <https://github.com/aiortc/aiortc>`_ (BSD-3-Clause),
installed with the ``webrtc`` extra. It is imported only when a call is made.
No STUN or TURN server is contacted unless ``webrtc_ice_servers`` is set, so
out of the box calls work on the same machine or network only.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import AudioFrame
from av.audio.resampler import AudioResampler

from bolo.audio.codecs import alaw_decode, mulaw_decode, pcm16_to_float
from bolo.audio.resample import resample
from bolo.realtime import RealtimeSession, new_id

if TYPE_CHECKING:
    from bolo.core import Bolo

#: seconds a peer connection may take to close before it is let go
PC_CLOSE_TIMEOUT_S = 10.0

log = logging.getLogger("bolo.webrtc")

RATE = 48000  # what Opus carries
FRAME = RATE // 50  # 20 ms
CONNECT_TIMEOUT_S = 30.0


class OutputTrack(MediaStreamTrack):  # type: ignore[misc]
    """The assistant's voice: paced 20 ms frames, silence when there is nothing to say.

    Audio is tagged with the response it belongs to. ``on_started(response_id)`` is called when a response's audio
    starts going out and ``on_stopped(response_id)`` when it has all gone out: at the boundary in the stream when one
    response's audio follows another's without a gap (a tool call's answer after "let me check"), not when the next
    response's audio is queued."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._pcm = bytearray()  # 48 kHz mono int16 waiting to be played
        self._start: float | None = None
        self._timestamp = 0
        self.playing = False
        self.finished = True  # no more audio is coming for the current response
        self.on_started: Any = None
        self.on_stopped: Any = None
        #: the response whose audio is going out now, and where later responses' audio starts in the queue
        self.current: str | None = None
        self._marks: list[tuple[int, str | None]] = []  # (byte offset in _pcm, response id)
        self._tail: str | None = None  # the response of the last audio queued

    @property
    def queued_ms(self) -> int:
        return len(self._pcm) // 2 * 1000 // RATE

    def push(self, samples: np.ndarray, rate: int, response_id: str | None = None) -> None:
        if not len(samples):
            return
        audio = resample(samples, rate, RATE) if rate != RATE else samples
        if self.playing and response_id != self._tail:
            self._marks.append((len(self._pcm), response_id))  # this response starts after what is queued
        self._tail = response_id
        self._pcm += (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        self.finished = False
        if not self.playing:
            self.playing = True
            self.current = response_id
            if self.on_started:
                self.on_started(response_id)

    def clear(self) -> int:
        """Drop what has not been played yet (barge-in). Returns how many milliseconds were dropped."""
        dropped = self.queued_ms
        self._pcm.clear()
        self._marks.clear()
        was_playing, self.playing, self.finished = self.playing, False, True
        return dropped if was_playing else 0

    def _advance(self, n: int) -> None:
        """``n`` bytes went out: a response whose audio starts within them takes over (stopped, then started)."""
        while self._marks and self._marks[0][0] <= n:
            _, nxt = self._marks.pop(0)
            if self.on_stopped:
                self.on_stopped(self.current)
            self.current = nxt
            if self.on_started:
                self.on_started(nxt)
        self._marks = [(at - n, rid) for at, rid in self._marks]

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        if self._start is None:
            self._start = time.time()
        else:
            self._timestamp += FRAME
            wait = self._start + self._timestamp / RATE - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        chunk = bytes(self._pcm[: FRAME * 2])
        del self._pcm[: FRAME * 2]
        if self._marks:
            self._advance(len(chunk))
        if self.playing and not self._pcm and self.finished:
            self.playing = False
            if self.on_stopped:
                self.on_stopped(self.current)
        frame = AudioFrame(format="s16", layout="mono", samples=FRAME)
        frame.planes[0].update(chunk.ljust(FRAME * 2, b"\0"))
        frame.pts = self._timestamp
        frame.sample_rate = RATE
        frame.time_base = Fraction(1, RATE)
        return frame


class Call:
    """One WebRTC peer connection driving one Realtime session."""

    def __init__(self, core: Bolo, *, model: str | None, transcription: bool) -> None:
        self.core = core
        self.id = new_id("rtc")
        servers = [RTCIceServer(urls=url) for url in core.settings.webrtc_ice_servers]
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=servers))
        self.out = OutputTrack()
        self.session = RealtimeSession(core, self._send, model=model, transcription=transcription, transport="webrtc")
        self._channel: Any = None
        self._outbox: list[str] = []
        self._inbox: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []
        self._response_id: str | None = None
        self._closed = False
        self.out.on_started = lambda rid: self._event("output_audio_buffer.started", response_id=rid)
        self.out.on_stopped = lambda rid: self._event("output_audio_buffer.stopped", response_id=rid)

    # -- events to the client ---------------------------------------------------
    def _deliver(self, text: str) -> None:
        channel = self._channel
        if channel is not None and channel.readyState == "open":
            channel.send(text)
        elif len(self._outbox) < 1000:
            self._outbox.append(text)

    def _event(self, type_: str, **fields: Any) -> None:
        self._deliver(json.dumps({"type": type_, "event_id": new_id("event"), **fields}, ensure_ascii=False))

    async def _send(self, event: dict[str, Any]) -> None:
        """What the session calls for every server event."""
        kind = event.get("type")
        if kind == "response.output_audio.delta":  # audio goes out on the track, not as events
            self._response_id = event.get("response_id")
            payload = base64.b64decode(event.get("delta") or "")
            codec = self.session.out_codec
            samples = (
                mulaw_decode(payload)
                if codec == "mulaw"
                else alaw_decode(payload)
                if codec == "alaw"
                else pcm16_to_float(payload)
            )
            self.out.push(samples, self.session.out_rate, self._response_id)
            return
        if kind in ("response.output_audio.done", "response.done"):
            self.out.finished = True
        if kind == "response.done" and (event.get("response") or {}).get("status") == "cancelled":
            self._clear_output()
        self._deliver(json.dumps(event, ensure_ascii=False))
        if (
            kind == "input_audio_buffer.speech_started"
            and (self.session.turn_detection or {}).get("interrupt_response", True)
            and not self.session.full_duplex  # a full-duplex model hears the caller and decides itself
        ):
            self._clear_output()  # barge-in also silences audio that was generated but not played yet

    def _clear_output(self) -> None:
        playing = self.out.current
        if self.out.clear():
            self._event("output_audio_buffer.cleared", response_id=playing or self._response_id)

    # -- lifecycle --------------------------------------------------------------
    async def answer(self, offer_sdp: str, config: dict[str, Any] | None) -> str:
        pc = self.pc
        if config:
            self.session.configure(config)

        @pc.on("datachannel")  # type: ignore[untyped-decorator]
        def on_datachannel(channel: Any) -> None:
            self._channel = channel

            @channel.on("message")  # type: ignore[untyped-decorator]
            def on_message(message: Any) -> None:
                self._inbox.put_nowait(("event", message))

            def flush() -> None:
                pending, self._outbox = self._outbox, []
                for text in pending:
                    channel.send(text)

            if channel.readyState == "open":
                flush()
            else:
                channel.on("open", flush)

        @pc.on("track")  # type: ignore[untyped-decorator]
        def on_track(track: Any) -> None:
            if track.kind == "audio":
                self._tasks.append(asyncio.create_task(self._listen(track), name=f"bolo-webrtc-in-{self.id}"))

        @pc.on("connectionstatechange")  # type: ignore[untyped-decorator]
        async def on_state() -> None:
            if pc.connectionState in ("failed", "closed"):
                await self.close()

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
        if any(t.kind == "audio" for t in pc.getTransceivers()):
            pc.addTrack(self.out)
        await pc.setLocalDescription(await pc.createAnswer())
        self.core.calls[self.id] = self
        await self.session.start()
        self._tasks.append(asyncio.create_task(self._work(), name=f"bolo-webrtc-{self.id}"))
        self._tasks.append(asyncio.create_task(self._watchdog(), name=f"bolo-webrtc-watch-{self.id}"))
        return str(pc.localDescription.sdp)

    async def _watchdog(self) -> None:
        await asyncio.sleep(CONNECT_TIMEOUT_S)
        if self.pc.connectionState != "connected":
            log.info("WebRTC call %s never connected; closing it", self.id)
            asyncio.get_running_loop().create_task(self.close())

    async def _listen(self, track: Any) -> None:
        """Microphone audio: Opus frames from the peer, as 16 kHz mono for the session."""
        resampler = AudioResampler(format="s16", layout="mono", rate=16000)
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                return
            for out in resampler.resample(frame):
                samples = out.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
                self._inbox.put_nowait(("audio", samples))

    async def _work(self) -> None:
        """Audio and client events are handled one at a time, in arrival order."""
        session = self.session
        while True:
            item = await self._inbox.get()
            if item is None:
                return
            kind, payload = item
            try:
                if kind == "audio":
                    await session.accept_audio(payload)
                    continue
                try:
                    event = json.loads(payload)
                    if not isinstance(event, dict):
                        raise ValueError("not an object")
                except ValueError:
                    await session.error("events must be JSON objects", code="invalid_json")
                    continue
                if event.get("type") == "input_audio_buffer.append":
                    await session.error(
                        "over WebRTC, audio is sent on the audio track; input_audio_buffer.append is not used",
                        code="unsupported",
                    )
                    continue
                if event.get("type") == "output_audio_buffer.clear":
                    self._clear_output()
                await session.handle(event)
            except Exception:  # one bad event must not end the call
                log.exception("WebRTC call %s: error while handling %s", self.id, kind)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.core.calls.pop(self.id, None)
        self._inbox.put_nowait(None)
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        with contextlib.suppress(Exception):
            await self.session.close()
        self.out.stop()
        with contextlib.suppress(Exception):  # TimeoutError: a peer connection that does not close is let go
            await asyncio.wait_for(self.pc.close(), PC_CLOSE_TIMEOUT_S)


async def create_call(
    core: Bolo, offer_sdp: str, *, config: dict[str, Any] | None, model: str | None, transcription: bool
) -> tuple[str, str]:
    """Answer an SDP offer. Returns ``(call id, SDP answer)``."""
    call = Call(core, model=model, transcription=transcription or (config or {}).get("type") == "transcription")
    try:
        answer = await call.answer(offer_sdp, config)
    except BaseException:
        await call.close()
        raise
    return call.id, answer
