"""Realtime speech-to-text sessions.

One :class:`STTSession` backs one audio stream (a WebSocket connection, a
Wyoming connection, a benchmark worker). Audio is pushed in with
:meth:`STTSession.feed`, which never blocks; a single task per session
resamples it, runs voice-activity detection and drives decoding through the
model's scheduler queue, so concurrent sessions are decoded in batches.

Two decoding strategies share the message protocol:

* **native** – for streaming models (transducer/CTC with incremental
  encoders). New audio advances the recogniser a few frames; partials are
  append-only by construction.
* **pseudo-streaming** – for models that only decode whole utterances
  (Whisper, Moonshine, Parakeet, SenseVoice, …). While someone is speaking
  the growing utterance is re-decoded every ``partial_interval_ms`` and a
  LocalAgreement policy decides which words are stable; when the VAD sees
  the end of speech the whole utterance is decoded once more for the final.

``latency_ms`` on every partial and final is the time from the server
receiving the last audio frame that contributed to that text until the
message is handed to the transport.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.audio.codecs import decode_frame
from bolo.audio.resample import StreamResampler
from bolo.diarize import OnlineSpeakerLabeler, speaker_label
from bolo.engines.base import (
    FloatArray,
    StreamResult,
    STTModel,
    STTRequest,
    STTResult,
    STTStream,
    VadDetector,
    VadParams,
    Word,
)
from bolo.errors import BoloError, InvalidRequest, Overloaded
from bolo.obs.trace import Trace, preview
from bolo.streaming.agreement import LocalAgreement, PrefixStability

if TYPE_CHECKING:
    from bolo.core import Bolo
    from bolo.lifecycle import LoadedModel

log = logging.getLogger("bolo.stream")

Send = Callable[[dict[str, Any]], Awaitable[None]]

SR = 16000
PRE_ROLL_S = 1.5  # audio kept while idle, so an utterance can start before the VAD noticed it
ONSET_PAD_S = 0.25  # neural VADs report the start a little late, more so on narrow-band audio
MAX_BACKLOG_S = 30.0
TAIL_PAD_S = 0.5


@dataclass
class SessionConfig:
    model: str | None = None
    language: str | None = None
    sample_rate: int = 16000
    encoding: str = "pcm_s16le"
    interim: bool = True
    vad: bool = True
    transport: str = "websocket"
    word_timestamps: bool = True
    endpoint_ms: float | None = None  # trailing silence that ends an utterance; None: the server default
    diarize: bool = False  # label every final with a speaker (one speaker per utterance)
    max_speakers: int | None = None


class STTSession:
    def __init__(self, core: Bolo, send: Send, config: SessionConfig) -> None:
        self.core = core
        self.cfg = config
        self._send = send
        self._in: asyncio.Queue[tuple[str, Any, float]] = asyncio.Queue()
        self._backlog_bytes = 0
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._failed: BaseException | None = None
        self.trace: Trace | None = None
        self.spec = core.resolve(config.model or "realtime", "stt")
        if config.sample_rate < 4000 or config.sample_rate > 192000:
            raise InvalidRequest("sample_rate must be between 4000 and 192000", param="sample_rate")
        try:
            decode_frame(b"", config.encoding)
        except ValueError as exc:
            raise InvalidRequest(str(exc), param="encoding") from exc
        self._bytes_per_s = config.sample_rate * {"pcm_s16le": 2, "pcm_f32le": 4}.get(config.encoding, 1)
        self.utterances = 0
        self.finals: list[dict[str, Any]] = []
        self._decode_ms = 0.0
        self._queue_ms: list[float] = []
        self._diarizer: LoadedModel | None = None
        self._labeler: OnlineSpeakerLabeler | None = None
        self._embedded: list[int | None] = []  # per final: index of its embedding in the labeler, if it has one

    # -- public API -------------------------------------------------------
    async def start(self) -> None:
        self.trace = Trace("stt", self.cfg.transport, self.spec.id)
        self._task = asyncio.create_task(self._run(), name=f"bolo-stt-{self.trace.id}")
        ready = asyncio.create_task(self._ready.wait())
        done, _ = await asyncio.wait({ready, self._task}, return_when=asyncio.FIRST_COMPLETED)
        if ready not in done:
            ready.cancel()
            exc = self._task.exception() if not self._task.cancelled() else None
            if exc:
                raise exc

    def feed(self, data: bytes) -> None:
        """Queue raw audio in the session's encoding. Never blocks."""
        if not data:
            return
        self._backlog_bytes += len(data)
        self._in.put_nowait(("audio", data, time.perf_counter()))

    def flush(self) -> None:
        """Finalise whatever has been said so far without closing the stream."""
        self._in.put_nowait(("flush", None, time.perf_counter()))

    async def stop(self) -> None:
        """End of input: emit the last final, metrics and ``done``; wait until sent."""
        self._in.put_nowait(("stop", None, time.perf_counter()))
        if self._task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def close(self) -> None:
        """Abort (client went away)."""
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    @property
    def done(self) -> bool:
        return self._task is not None and self._task.done()

    # -- main loop --------------------------------------------------------
    async def _run(self) -> None:
        trace = self.trace
        assert trace is not None
        core = self.core
        core.active_streams += 1
        status, error = "ok", None
        pos = 0
        try:
            async with contextlib.AsyncExitStack() as stack:
                entry = await stack.enter_async_context(core.manager.use(self.spec.id))
                stt = entry.model
                assert isinstance(stt, STTModel)
                detector: VadDetector | None = None
                vad_name = None
                endpoint_ms = self.cfg.endpoint_ms or core.settings.endpoint_silence_ms
                if self.cfg.vad:
                    params = VadParams(min_silence_s=endpoint_ms / 1000.0, max_speech_s=core.settings.max_utterance_s)
                    detector, vad_name = await core.vad_detector(params, fallback=True)
                if self.cfg.diarize:
                    await stack.enter_async_context(self._speakers(core))
                proc: _Processor
                if stt.streaming:
                    proc = _NativeProcessor(self, entry, stt, detector)
                    # a stream that holds a connection (a cloud provider's live socket) is closed with the session
                    release = getattr(proc.stream, "close", None)
                    if release is not None:
                        stack.callback(release)
                else:
                    proc = _PseudoProcessor(self, entry, stt, detector)
                resampler = StreamResampler(self.cfg.sample_rate, SR)
                await self._send(
                    {
                        "type": "ready",
                        "model": self.spec.id,
                        "sample_rate": self.cfg.sample_rate,
                        "request_id": trace.id,
                        "encoding": self.cfg.encoding,
                        "mode": "native" if stt.streaming else "pseudo",
                        "vad": vad_name,
                        "endpoint_ms": round(endpoint_ms),
                        "diarization": self._diarizer.id if self._diarizer else None,
                    }
                )
                trace.restart()
                self._ready.set()
                stopping = False
                while not stopping:
                    kind, payload, recv_t = await self._in.get()
                    raw: list[bytes] = []
                    flush = False
                    while True:
                        if kind == "audio":
                            raw.append(payload)
                            last_recv = recv_t
                        elif kind == "flush":
                            flush = True
                        elif kind == "stop":
                            stopping = True
                        if stopping or self._in.empty():
                            break
                        kind, payload, recv_t = self._in.get_nowait()
                    if raw:
                        data = b"".join(raw)
                        self._backlog_bytes -= len(data)
                        if self._backlog_bytes > MAX_BACKLOG_S * self._bytes_per_s:
                            raise Overloaded("audio is arriving faster than it can be transcribed; closing the stream")
                        t_res = trace.now()
                        samples = resampler.process(decode_frame(data, self.cfg.encoding))
                        trace.add_phase("resample", t_res, trace.now())
                        if len(samples):
                            pos += len(samples)
                            await proc.audio(samples, last_recv)
                    if flush or stopping:
                        if stopping:
                            tail = resampler.process(np.zeros(0, dtype=np.float32), last=True)
                            if len(tail):
                                pos += len(tail)
                                await proc.audio(tail, time.perf_counter())
                        await proc.finalize(time.perf_counter(), reason="stop" if stopping else "flush")
                await self._send_metrics(pos)
                await self._revise_speakers()
                await self._send({"type": "done", "request_id": trace.id})
        except asyncio.CancelledError:
            status, error = "cancelled", "client disconnected"
            raise
        except BoloError as exc:
            status, error = "error", exc.message
            self._failed = exc
            if self._ready.is_set():
                with contextlib.suppress(Exception):
                    await self._send({"type": "error", "code": exc.code, "message": exc.message})
            else:
                raise
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
            self._failed = exc
            log.exception("stream %s failed", trace.id)
            if self._ready.is_set():
                with contextlib.suppress(Exception):
                    await self._send({"type": "error", "code": "internal_error", "message": error})
            else:
                raise
        finally:
            core.active_streams -= 1
            trace.audio_seconds = pos / SR
            if self.finals:
                trace.final_ms = self.finals[-1]["latency_ms"]
                trace.language = self.finals[-1].get("language")
                text = " ".join(f["text"] for f in self.finals)
                trace.text_preview = preview(text)
                trace.chars = len(text)
            trace.extra["utterances"] = self.utterances
            trace.compact()
            trace.finish(status, error)
            if trace.audio_seconds:
                trace.rtf = self._decode_ms / 1000.0 / trace.audio_seconds
            core.traces.record(trace, observe=False)
            if trace.rtf is not None and status == "ok":
                core.traces.observe_latency(self.spec.id, "stt", rtf=trace.rtf)

    # -- speaker labels ----------------------------------------------------
    @contextlib.asynccontextmanager
    async def _speakers(self, core: Bolo) -> AsyncIterator[None]:
        spec = core.resolve(None, "diarization")
        async with core.manager.use(spec.id) as entry:
            self._diarizer = entry
            self._labeler = OnlineSpeakerLabeler(max_speakers=self.cfg.max_speakers)
            try:
                yield
            finally:
                self._diarizer = None

    async def speaker_for(self, samples: FloatArray) -> str | None:
        """Label an utterance with a speaker (None when diarization is off or the audio is too short)."""
        if self._diarizer is None or self._labeler is None:
            return None
        trace = self.trace
        assert trace is not None
        if len(samples) < int(0.5 * SR):  # too little speech for a reliable embedding: keep the last speaker
            self._embedded.append(None)
            return speaker_label(self._labeler.last) if self._labeler.last is not None else None
        start = trace.now()
        embedding, _info = await self._diarizer.queue.submit("embed", samples)
        trace.add_phase("diarize", start, trace.now())
        self._embedded.append(len(self._labeler.embeddings))
        return speaker_label(self._labeler.assign(embedding))

    async def _revise_speakers(self) -> None:
        """At the end of the stream, cluster all utterances together and report the labels that change.

        Sent as one ``speakers`` message before ``done``. Labels given earlier in ``final`` messages are
        not re-sent; ``utterances`` lists every final (1-based ``utterance`` number, as in the SSE progress
        events) with its revised ``speaker`` and, when it changed, the label it had before in ``was``.
        """
        labeler = self._labeler
        if labeler is None or not self.finals or len(self._embedded) != len(self.finals):
            return
        revised = labeler.revise()
        rows: list[dict[str, Any]] = []
        changed = 0
        previous: str | None = None
        for number, (final, index) in enumerate(zip(self.finals, self._embedded), start=1):
            label = speaker_label(revised[index]) if index is not None else previous
            row: dict[str, Any] = {"utterance": number, "t0": final["t0"], "t1": final["t1"], "speaker": label}
            if label != final.get("speaker"):
                row["was"] = final.get("speaker")
                changed += 1
            rows.append(row)
            previous = label
        speakers = sorted({r["speaker"] for r in rows if r["speaker"]})
        await self._send({"type": "speakers", "revised": changed, "speakers": speakers, "utterances": rows})
        if self.trace is not None:
            self.trace.extra["speakers"] = len(speakers)
            self.trace.extra["speaker_labels_revised"] = changed

    # -- helpers used by processors ----------------------------------------
    def note_job(self, queue_ms: float, run_ms: float, batch: int, started: float, finished: float) -> None:
        trace = self.trace
        assert trace is not None
        self._decode_ms += run_ms
        self._queue_ms.append(queue_ms)
        trace.note_batch(batch)
        if queue_ms >= 0.5:
            trace.add_phase("queue", started - queue_ms / 1000.0, started)
        trace.add_phase("decode", started, finished)
        self.core.traces.observe_latency(self.spec.id, "stt", queue_ms=queue_ms)

    async def send_vad(self, event: str, sample: int) -> None:
        await self._send({"type": "vad", "event": event, "t": round(sample / SR, 3)})

    async def send_partial(
        self, text: str, t0: float, t1: float, recv_t: float, stable_chars: int, first: bool
    ) -> None:
        latency = (time.perf_counter() - recv_t) * 1000.0
        await self._send(
            {
                "type": "partial",
                "text": text,
                "t0": round(t0, 3),
                "t1": round(t1, 3),
                "latency_ms": round(latency, 1),
                "stable_chars": stable_chars,
            }
        )
        trace = self.trace
        assert trace is not None
        if first:
            trace.mark_first_partial(latency)
            self.core.traces.observe_latency(self.spec.id, "stt", first_partial_ms=latency)

    async def send_final(
        self,
        text: str,
        t0: float,
        t1: float,
        recv_t: float,
        words: list[Word],
        language: str | None,
        audio: FloatArray | None = None,
    ) -> None:
        trace = self.trace
        assert trace is not None
        speaker = await self.speaker_for(audio) if audio is not None else None
        t_fin = trace.now()
        latency = (time.perf_counter() - recv_t) * 1000.0
        msg: dict[str, Any] = {
            "type": "final",
            "text": text,
            "t0": round(t0, 3),
            "t1": round(t1, 3),
            "latency_ms": round(latency, 1),
            "words": [w.to_dict() for w in words],
            "language": language
            or self.cfg.language
            or (self.spec.languages[0] if len(self.spec.languages) == 1 else None),
        }
        if self._diarizer is not None:
            msg["speaker"] = speaker
        await self._send(msg)
        trace.add_phase("finalize", t_fin, trace.now())
        self.utterances += 1
        self.finals.append(msg)
        self.core.traces.observe_latency(self.spec.id, "stt", final_ms=latency)
        self.core.events.publish(
            "request_progress",
            {
                "id": trace.id,
                "phase": "final",
                "utterance": self.utterances,
                "model": self.spec.id,
                "first_partial_ms": trace.first_partial_ms,
                "final_ms": round(latency, 1),
                "t0": msg["t0"],
                "t1": msg["t1"],
                "text_preview": preview(text),
            },
        )
        await self._send_metrics(int(t1 * SR))

    async def _send_metrics(self, pos: int) -> None:
        seconds = pos / SR
        rtf = self._decode_ms / 1000.0 / seconds if seconds > 0 else None
        queue = sum(self._queue_ms) / len(self._queue_ms) if self._queue_ms else 0.0
        await self._send(
            {
                "type": "metrics",
                "audio_seconds": round(seconds, 3),
                "rtf": round(rtf, 4) if rtf is not None else None,
                "queue_ms": round(queue, 1),
            }
        )


class _Processor:
    def __init__(self, session: STTSession, entry: LoadedModel, stt: STTModel, detector: VadDetector | None) -> None:
        self.s = session
        self.entry = entry
        self.stt = stt
        self.detector = detector
        self.pos = 0  # samples seen (16 kHz)
        self.first_partial_pending = True

    async def audio(self, samples: FloatArray, recv_t: float) -> None:
        raise NotImplementedError

    async def finalize(self, recv_t: float, reason: str) -> None:
        raise NotImplementedError

    def run_vad(self, samples: FloatArray) -> list[Any]:
        if self.detector is None:
            return []
        trace = self.s.trace
        assert trace is not None
        t = trace.now()
        events = self.detector.accept(samples)
        trace.add_phase("vad", t, trace.now())
        return events


class _NativeProcessor(_Processor):
    """Incremental decoding with a streaming model."""

    def __init__(self, session: STTSession, entry: LoadedModel, stt: STTModel, detector: VadDetector | None) -> None:
        super().__init__(session, entry, stt, detector)
        self.stream: STTStream = stt.create_stream(session.cfg.language)
        self.base = 0  # stream position where the current utterance began
        self.speech_start: int | None = None  # where the VAD heard speech begin in this utterance
        self.speech_end: int | None = None  # where the VAD heard it end (the final's t1, however late it was seen)
        self.kept: list[FloatArray] = []  # the utterance's audio, kept only when speakers are labelled
        self.stability = PrefixStability()
        self.last_text = ""

    def _t0(self) -> float:
        start = self.speech_start if self.speech_start is not None and self.speech_start >= self.base else self.base
        return start / SR

    async def _step(self) -> tuple[StreamResult, bool]:
        (result, endpoint), info = await self.entry.queue.submit("stream", self.stream)
        self.s.note_job(info.queue_ms, info.run_ms, info.batch_size, info.started, info.finished)
        return result, bool(endpoint)

    async def audio(self, samples: FloatArray, recv_t: float) -> None:
        # Audio that queued up while the model was busy arrives here in one piece, possibly with one utterance's end
        # and the next one's start in it. Fed whole, the two would come out as one final: the piece is cut where
        # the VAD heard the next utterance start, and the part before it ends the first utterance.
        first = self.pos  # the stream position of samples[0]
        done = 0
        ended = False
        for ev in self.run_vad(samples):
            await self.s.send_vad(ev.kind, ev.sample)
            if ev.kind == "speech_end":
                ended = True
                self.speech_end = ev.sample
            elif ev.kind == "speech_start":
                if ended:
                    cut = min(len(samples), max(done, ev.sample - first))
                    await self._feed(samples[done:cut], recv_t, vad_end=True)
                    done, ended = cut, False
                self.speech_start = max(self.base, ev.sample)
        if ended or done == 0 or done < len(samples):
            await self._feed(samples[done:], recv_t, vad_end=ended)

    def _accept(self, samples: FloatArray) -> None:
        self.stream.accept(samples)
        self.pos += len(samples)
        if self.s.cfg.diarize:
            self.kept.append(samples)

    async def _feed(self, samples: FloatArray, recv_t: float, vad_end: bool) -> None:
        self._accept(samples)
        result, endpoint = await self._step()
        text = result.text.strip()
        if text and text != self.last_text and self.s.cfg.interim and not (endpoint or vad_end):
            self.last_text = text
            stable = self.stability.update(text)
            await self.s.send_partial(text, self._t0(), self.pos / SR, recv_t, stable, self.first_partial_pending)
            self.first_partial_pending = False
        if (endpoint or vad_end) and (text or endpoint):
            await self._finish_utterance(recv_t, pad=vad_end and not endpoint)

    async def _finish_utterance(self, recv_t: float, pad: bool) -> None:
        if pad:
            # flush the model's look-ahead so the last word is not held back
            self.stream.accept(np.zeros(int(TAIL_PAD_S * SR), dtype=np.float32))
            result, _ = await self._step()
        else:
            result = self.stream.result()
        text = result.text.strip()
        if text:
            base = self.base / SR
            words = [Word(w.w, w.t0 + base, w.t1 + base) for w in result.words] if self.s.cfg.word_timestamps else []
            audio = None
            if self.s.cfg.diarize and self.kept:
                audio = np.concatenate(self.kept)
                if self.speech_start is not None:  # drop the silence before the speaker started
                    audio = audio[max(0, self.speech_start - self.base) :]
            end = (
                self.speech_end if self.speech_end is not None and self.base < self.speech_end < self.pos else self.pos
            )
            await self.s.send_final(text, self._t0(), end / SR, recv_t, words, None, audio)
        self.kept = []
        self.stream.reset()
        self.base = self.pos
        self.speech_start = None
        self.speech_end = None
        self.last_text = ""
        self.stability.reset()
        self.first_partial_pending = True

    async def finalize(self, recv_t: float, reason: str) -> None:
        if reason == "stop":
            self.stream.accept(np.zeros(int(TAIL_PAD_S * SR), dtype=np.float32))
            self.stream.finish()
            await self._step()
            await self._finish_utterance(recv_t, pad=False)
        else:
            await self._finish_utterance(recv_t, pad=True)
        if self.detector is not None:
            for ev in self.detector.flush():
                await self.s.send_vad(ev.kind, ev.sample)


class _PseudoProcessor(_Processor):
    """VAD-segmented re-decoding with LocalAgreement for non-streaming models."""

    def __init__(self, session: STTSession, entry: LoadedModel, stt: STTModel, detector: VadDetector | None) -> None:
        super().__init__(session, entry, stt, detector)
        settings = session.core.settings
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_start = 0  # absolute sample index of buf[0]
        self.utt_start: int | None = None if detector is not None else 0
        self.agreement = LocalAgreement(settings.agreement)
        #: repeated decodes of the growing utterance (a cloud model gets the final decode only: each is billed)
        self.interim_ok = session.core.interim_decodes(entry.spec)
        self.interval = settings.partial_interval_ms / 1000.0
        self.max_utt = int(min(settings.max_utterance_s, stt.max_chunk_s) * SR)
        self.last_partial_at = 0.0
        self.last_partial_pos = 0
        self.last_decode_s = 0.0
        self.language: str | None = None

    def _slice(self, start: int, end: int) -> FloatArray:
        a = max(0, start - self.buf_start)
        b = max(a, end - self.buf_start)
        return self.buf[a:b]

    def _trim(self) -> None:
        keep_from = self.utt_start if self.utt_start is not None else self.pos - int(PRE_ROLL_S * SR)
        drop = max(0, min(keep_from - self.buf_start, len(self.buf)))
        if drop > SR:  # avoid re-allocating on every frame
            self.buf = self.buf[drop:].copy()
            self.buf_start += drop

    async def _decode(self, start: int, end: int, words: bool) -> STTResult:
        req = STTRequest(self._slice(start, end).copy(), self.s.cfg.language, "transcribe", None, words)
        result, info = await self.entry.queue.submit("batch", req)
        assert isinstance(result, STTResult)
        self.s.note_job(info.queue_ms, info.run_ms, info.batch_size, info.started, info.finished)
        self.last_decode_s = info.run_ms / 1000.0
        self.language = result.language or self.language
        return result

    async def audio(self, samples: FloatArray, recv_t: float) -> None:
        self.buf = np.concatenate([self.buf, samples]) if len(self.buf) else samples.copy()
        self.pos += len(samples)
        for ev in self.run_vad(samples):
            if ev.kind == "speech_start":
                start = max(self.buf_start, min(ev.sample, self.pos) - int(ONSET_PAD_S * SR))
                self.utt_start = start
                self.last_partial_pos = start
                self.last_partial_at = 0.0
                await self.s.send_vad("speech_start", ev.sample)
            elif ev.kind == "speech_end":
                await self.s.send_vad("speech_end", ev.sample)
                if self.utt_start is not None:
                    await self._final(min(ev.sample + int(0.2 * SR), self.pos), recv_t)
                    self.utt_start = None
        if self.utt_start is not None:
            length = self.pos - self.utt_start
            if length >= self.max_utt:
                await self._final(self.pos, recv_t)
                self.utt_start = self.pos if self.detector is None or self.detector.in_speech else None
            elif self.s.cfg.interim and self.interim_ok:
                await self._maybe_partial(recv_t)
        self._trim()

    async def _maybe_partial(self, recv_t: float) -> None:
        assert self.utt_start is not None
        now = time.perf_counter()
        interval = max(self.interval, 1.5 * self.last_decode_s)
        if now - self.last_partial_at < interval:
            return
        if self.pos - self.last_partial_pos < int(0.2 * SR) or self.pos - self.utt_start < int(0.4 * SR):
            return
        self.last_partial_at = now
        self.last_partial_pos = self.pos
        end = self.pos
        result = await self._decode(self.utt_start, end, words=False)
        hypothesis = result.text.strip()
        if not hypothesis:
            return
        text, stable = self.agreement.update(hypothesis)
        await self.s.send_partial(text, self.utt_start / SR, end / SR, recv_t, stable, self.first_partial_pending)
        self.first_partial_pending = False

    async def _final(self, end: int, recv_t: float) -> None:
        assert self.utt_start is not None
        start = self.utt_start
        self.agreement.reset()
        self.first_partial_pending = True
        if end - start < int(0.1 * SR):
            return
        result = await self._decode(start, end, words=self.s.cfg.word_timestamps and self.stt.supports_word_timestamps)
        text = result.text.strip()
        if not text:
            return
        t0 = start / SR
        words = [
            Word(w.w, w.t0 + t0, w.t1 + t0) for w in (result.words or [w for s in result.segments for w in s.words])
        ]
        audio = self._slice(start, end).copy() if self.s.cfg.diarize else None
        await self.s.send_final(text, t0, end / SR, recv_t, words, result.language, audio)

    async def finalize(self, recv_t: float, reason: str) -> None:
        if self.detector is not None:
            for ev in self.detector.flush():
                if ev.kind == "speech_end":
                    await self.s.send_vad("speech_end", ev.sample)
        if self.utt_start is not None:
            await self._final(self.pos, recv_t)
            self.utt_start = None if self.detector is not None else self.pos
        self._trim()
