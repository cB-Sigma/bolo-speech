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
import difflib
import logging
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from bolo.audio.codecs import decode_frame
from bolo.audio.resample import StreamResampler
from bolo.denoise import DenoiseOptions, LiveDenoiser
from bolo.diarize import OnlineSpeakerLabeler, speaker_label
from bolo.engines.base import (
    DenoiseModel,
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
from bolo.hotwords import Spelling
from bolo.obs.trace import Trace, preview
from bolo.streaming.agreement import LocalAgreement, PrefixStability
from bolo.stt_options import DEFAULT_MAX_REPEATS, ENGINE_OPTIONS, RepeatGuard

if TYPE_CHECKING:
    from bolo.core import Bolo
    from bolo.lifecycle import LoadedModel

log = logging.getLogger("bolo.stream")


class ClientDisconnected(Exception):
    """Raised by a session's ``send`` when the client has gone: the stream ends as cancelled, not as an error."""


Send = Callable[[dict[str, Any]], Awaitable[None]]

SR = 16000
PRE_ROLL_S = 1.5  # audio kept while idle, so an utterance can start before the VAD noticed it
ONSET_PAD_S = 0.25  # neural VADs report the start a little late, more so on narrow-band audio
MAX_BACKLOG_S = 30.0
TAIL_PAD_S = 0.5
VAD_LAG_S = 2.0  # how far back a VAD may place a start it reports (the energy fallback's patience, and some)
MAX_GAIN_DB = 24.0  # most a streaming model's quiet speech is raised by
FRAME = 320  # 20 ms
# A VAD reports a start once it has heard enough confident speech; a quiet first word, or one followed by a breath,
# is not that, and the start lands on the next word (Silero on simulated microphone takes: up to 0.8 s late). The
# speech before it is found again in the audio held back: louder than the room by ONSET_MARGIN_DB, through pauses
# shorter than ONSET_GAP_S, at most ONSET_LOOKBACK_S before the reported start, and starting with a stretch that has
# ONSET_VOICED_S of voiced frames in it (a burst of noise, a click or a fricative's hiss on its own is not a word).
ONSET_LOOKBACK_S = 1.2
ONSET_GAP_S = 0.4
# 9 dB missed quiet words only 6-10 dB above the room: on simulated microphone takes with a quiet first word, the
# 20M Zipformer had it right in 43 of 96 with 6 dB, 37 with 9
ONSET_MARGIN_DB = 6.0
ONSET_FLOOR_DB = -70.0  # quieter than this is never taken for speech
DIGITAL_SILENCE_DB = -90.0  # quieter frames count as this level: a file's silent lead-in is a very quiet room
ONSET_VOICED_S = 0.04  # voiced frames a stretch of sound before the reported start needs to be taken for speech
VOICED_R = 0.5  # normalised autocorrelation at the pitch period that makes a frame voiced
PENDING_S = 0.5  # how long a fresh stream waits for a vowel after the VAD reports a start
LEAD_BACK_S = 0.1  # how far before the start the VAD reported the speech's first vowel is looked for
CHUNK_LEAD_S = 0.05  # where in an encoder chunk a first word is kept best (see aligned_preroll)
BURST_GAP_S = 0.1  # sound with no vowel in it that ends this long before the speech is a burst, not its first word
# Quiet input is raised for the VAD (only): see InputLevel
VAD_SPEECH_DBFS = -30.0  # the loudest recent frame is raised to this
VAD_ROOM_DBFS = -65.0  # the room is raised no higher than this: a microphone's own noise is not raised at all
VAD_PEAK_WINDOW_S = 10.0
VAD_ROOM_WINDOW_S = 30.0


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
    #: ``denoise`` of the stream (bolo.denoise); None: the model's pin, its catalog option and the server setting
    denoise: DenoiseOptions | None = None
    #: names and terms to favour (the model gets them when it takes hotwords), and whether the finals' near-misses
    #: of them are spelt their way (bolo.hotwords)
    hotwords: str | None = None
    hotword_spelling: bool = True


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
        #: decoding options pinned for the model (or in its catalog entry): a live session has no request form, so
        #: these are how its Whisper options and the repetition guard are set
        self.decode_options = core.decode_options(self.spec)
        self.guard = RepeatGuard(int(self.decode_options.get("max_repeats", DEFAULT_MAX_REPEATS)))
        self.spelling = Spelling.of(config.hotwords, config.hotword_spelling)
        #: the denoiser the stream runs before recognition (bolo.denoise.LiveDenoiser), or None
        self.denoise_plan = core.denoise_plan(self.spec, config.denoise)
        self._live: LiveDenoiser | None = None
        self._denoise_info: dict[str, Any] | None = None

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
                if self.denoise_plan is not None:
                    await self._start_denoiser(stack)
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
                        "denoise": self._denoise_info,
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
                        samples = await self._denoised(samples)
                        if len(samples):
                            pos += len(samples)
                            await proc.audio(samples, last_recv)
                    if flush or stopping:
                        if stopping:
                            tail = await self._denoised(resampler.process(np.zeros(0, dtype=np.float32), last=True))
                            if self._live is not None:
                                tail = np.concatenate([tail, self._live.finish()])
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
        except ClientDisconnected:  # the client left while a message was on its way to it
            status, error = "cancelled", "client disconnected"
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
            if self._denoise_info is not None:
                trace.extra["denoise"] = {**self._denoise_info, **(self._live.summary() if self._live else {})}
            trace.compact()
            trace.finish(status, error)
            if trace.audio_seconds:
                trace.rtf = self._decode_ms / 1000.0 / trace.audio_seconds
            core.traces.record(trace, observe=False)
            if trace.rtf is not None and status == "ok":
                core.traces.observe_latency(self.spec.id, "stt", rtf=trace.rtf)

    # -- denoising ---------------------------------------------------------
    async def _start_denoiser(self, stack: contextlib.AsyncExitStack) -> None:
        """Load the stream's denoiser. One that cannot run live (a cloud API) or cannot load is skipped, said in
        ``ready.denoise``; the stream goes on with the original audio, as a transcription request does."""
        plan = self.denoise_plan
        assert plan is not None and plan.model is not None
        info: dict[str, Any] = {"model": plan.model, "strength": plan.strength, "status": "applied"}
        self._denoise_info = info
        try:
            entry = await stack.enter_async_context(self.core.manager.use(plan.model))
            model = entry.model
            assert isinstance(model, DenoiseModel)
            if not model.live:
                raise InvalidRequest(f"'{plan.model}' enhances whole recordings only, not live streams")
            self._live = await asyncio.to_thread(LiveDenoiser, model, SR, plan.strength)
            info.update(sample_rate=model.sample_rate, stateful=self._live.stream.stateful)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            detail = exc.message if isinstance(exc, BoloError) else f"{type(exc).__name__}: {exc}"
            info.update(status="skipped", detail=detail)
            log.warning("live denoising with '%s' skipped, the original audio is used: %s", plan.model, detail)

    async def _denoised(self, samples: FloatArray) -> FloatArray:
        live = self._live
        if live is None or not len(samples):
            return samples
        trace = self.trace
        assert trace is not None
        t0 = trace.now()
        out = await asyncio.to_thread(live.process, samples)
        trace.add_phase("denoise", t0, trace.now())
        return out

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
        text, words = self.guard.apply(text, words)
        if self.guard.removed:
            trace.extra["repeats_trimmed"] = self.guard.removed
        if not text.strip():  # nothing but a repeat of what was already sent: no final, as for an empty decode
            log.info("%s: the repetition guard dropped a final that only repeated earlier ones", self.spec.id)
            return
        spelt: list[dict[str, Any]] = []
        if self.spelling is not None:  # the hotwords their way, whatever the engine made of them
            text, words, spelt = self.spelling.text(text, words)
            if spelt:
                trace.extra.setdefault("hotword_spelling", []).extend(spelt)
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
        if spelt:
            msg["hotword_spelling"] = spelt
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
                **({"denoise": self._live.summary()} if self._live is not None else {}),
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
        self.input_level = InputLevel()

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
        self.input_level.push(samples)
        gain = self.input_level.vad_gain()
        if gain != 1.0:
            samples = np.clip(samples * gain, -1.0, 1.0).astype(np.float32)
        events = self.detector.accept(samples)
        trace.add_phase("vad", t, trace.now())
        return events


def frame_rms(samples: FloatArray) -> FloatArray:
    """The RMS level of each whole 20 ms frame of ``samples``."""
    n = len(samples) // FRAME
    frames = np.asarray(samples[: n * FRAME], dtype=np.float64).reshape(n, FRAME)
    rms: FloatArray = np.sqrt(np.mean(np.square(frames), axis=1))
    return rms


class InputLevel:
    """The levels of the input's recent 20 ms frames, and the gain the VAD hears it with.

    Silero hears speech below about -50 dBFS poorly or not at all: a clip 40 dB down after digital silence (speech
    near -55 dBFS) gave no start, or one seconds late, and the recogniser lost the start or everything. The VAD (only
    the VAD; the recogniser has its own gain) is given quiet input raised so its loudest recent frame reaches
    ``VAD_SPEECH_DBFS``, and never so far that the room (the quietest tenth of the frames, digital silence counted as
    ``DIGITAL_SILENCE_DB``) goes above ``VAD_ROOM_DBFS``, so by 25 dB at most: a microphone's own noise, near -60
    dBFS, is not raised at all, and the VAD does not start taking a quiet room for speech. Input at an ordinary level
    is not touched either. At -40 dB over a room at -60 dBFS (speech about 6 dB above the room) nothing is raised:
    there the speech is at the noise floor of a real microphone, and neither Silero nor the recognisers find it."""

    def __init__(self) -> None:
        self.peak: deque[float] = deque(maxlen=int(VAD_PEAK_WINDOW_S * SR) // FRAME)
        self.room: deque[float] = deque(maxlen=int(VAD_ROOM_WINDOW_S * SR) // FRAME)
        self.rest = np.zeros(0, dtype=np.float32)

    def push(self, samples: FloatArray) -> None:
        x = np.concatenate([self.rest, samples]) if len(self.rest) else samples
        n = len(x) // FRAME
        self.rest = np.array(x[n * FRAME :], dtype=np.float32)
        db = np.maximum(20.0 * np.log10(frame_rms(x[: n * FRAME]) + 1e-9), DIGITAL_SILENCE_DB)
        self.peak.extend(db.tolist())
        self.room.extend(db.tolist())

    def vad_gain(self) -> float:
        if len(self.room) < 5:  # too little heard to tell the room from speech
            return 1.0
        room = float(np.percentile(np.fromiter(self.room, dtype=np.float64), 10))
        db = min(VAD_SPEECH_DBFS - max(self.peak), VAD_ROOM_DBFS - room)
        return float(10.0 ** (db / 20.0)) if db > 0.0 else 1.0


def speech_level(samples: FloatArray) -> float:
    """The level of the speech in ``samples`` (RMS, 0 for less than a frame): the 90th percentile of the 20 ms frame
    levels, so pauses do not count."""
    rms = frame_rms(samples)
    return float(np.percentile(rms, 90)) if len(rms) else 0.0


def room_level(samples: FloatArray) -> float:
    """The level of the room in ``samples`` (dBFS): the quietest tenth of its 20 ms frames, digital silence counted as
    ``DIGITAL_SILENCE_DB`` (a file's silent lead-in, a microphone that has just opened); ``DIGITAL_SILENCE_DB`` for
    less than a frame."""
    rms = frame_rms(samples)
    if len(rms) == 0:
        return DIGITAL_SILENCE_DB
    return float(np.percentile(np.maximum(20.0 * np.log10(rms + 1e-9), DIGITAL_SILENCE_DB), 10))


def sounding(samples: FloatArray, room_db: float) -> FloatArray:
    """The 20 ms frames of ``samples`` louder than the room (``room_db``) by ``ONSET_MARGIN_DB``, joined; nothing when
    there is less than 0.1 s of them. What a gain is judged on: right after the VAD reports a start, much of what it
    heard can be silence (a click or a burst it took for speech, then a pause), and the 90th percentile of that made
    a 24 dB gain that clipped the speech that followed ("ular arrive" from the 20M Zipformer for en-short)."""
    n = len(samples) // FRAME
    frames = np.asarray(samples[: n * FRAME], dtype=np.float32).reshape(n, FRAME)
    loud = 20.0 * np.log10(frame_rms(samples) + 1e-9) > max(room_db + ONSET_MARGIN_DB, ONSET_FLOOR_DB)
    if int(np.sum(loud)) * FRAME < int(0.1 * SR):
        return np.zeros(0, dtype=np.float32)
    out: FloatArray = frames[loud].reshape(-1)
    return out


def speech_gain(samples: FloatArray, target_dbfs: float, max_gain_db: float = MAX_GAIN_DB) -> float:
    """The gain that brings speech in ``samples`` up to ``target_dbfs``: 1 for speech already that loud, at most
    ``max_gain_db``."""
    level = speech_level(samples)
    if level <= 1e-6:
        return 1.0
    return float(min(max(10.0 ** (target_dbfs / 20.0) / level, 1.0), 10.0 ** (max_gain_db / 20.0)))


def voiced(samples: FloatArray) -> NDArray[np.bool_]:
    """Whether each whole 20 ms frame of ``samples`` is voiced: periodic at a speaking pitch (60-400 Hz), judged on
    40 ms from its start (from 20 ms before it for the last frame). A vowel is; room noise, a click or a burst of noise
    is not. Under 40 ms of audio, nothing is."""
    n = len(samples) // FRAME
    out = np.zeros(n, dtype=bool)
    x = np.asarray(samples, dtype=np.float64)
    lo, hi = SR // 400, SR // 60
    for k in range(n):
        a = max(0, min(k * FRAME, n * FRAME - 2 * FRAME))
        w = x[a : a + 2 * FRAME]
        w = w - w.mean()
        if len(w) <= hi + FRAME // 2 or not np.any(w):
            continue
        spec = np.fft.rfft(w, 2 * len(w))
        ac = np.fft.irfft(spec * np.conj(spec))[: hi + 1]
        if ac[0] <= 0:
            continue
        # unbiased: the overlap shrinks with the lag
        r = ac[lo : hi + 1] / ac[0] * len(w) / (len(w) - np.arange(lo, hi + 1))
        out[k] = float(np.max(r)) > VOICED_R
    return out


def speech_onset(audio: FloatArray, at: int) -> int:
    """Where the speech a VAD reports starting at ``audio[at]`` begins: earlier when the VAD missed its first word.

    Steps back from ``at`` a 20 ms frame at a time while frames are louder than the room by ``ONSET_MARGIN_DB``,
    across pauses shorter than ``ONSET_GAP_S``, at most ``ONSET_LOOKBACK_S``. The room is the quietest tenth of the
    frames held before ``at``, digital silence counted as such, so a file's silent lead-in, or a recording quiet
    throughout, is not taken for a room as loud as its speech. The sound found is then trimmed from its start to the
    first stretch with voiced frames in it (``ONSET_VOICED_S``): a burst of noise, a click, a breath or the hiss of
    a fricative on its own is not a word the VAD missed, and raising it made the recogniser write a letter ("s",
    "o") or a number before the first word."""
    at = min(at, len(audio))
    if at <= 0:
        return at
    first = at % FRAME  # frames end at ``at``
    rms = frame_rms(audio[first:at])
    if len(rms) < 5:  # too little held to tell the room from speech
        return at
    db = np.maximum(20.0 * np.log10(rms + 1e-9), DIGITAL_SILENCE_DB)
    threshold = max(room_level(audio[first:at]) + ONSET_MARGIN_DB, ONSET_FLOOR_DB)
    lowest = max(0, len(db) - int(ONSET_LOOKBACK_S * SR) // FRAME)
    stretches: list[list[int]] = []  # [first, last] frames of each stretch of sound, the latest first
    quiet = 0
    for k in range(len(db) - 1, lowest - 1, -1):
        if db[k] > threshold:
            if quiet or not stretches:
                stretches.append([k, k])
            stretches[-1][0] = k
            quiet = 0
        else:
            quiet += FRAME
            if quiet >= ONSET_GAP_S * SR:
                break
    need = int(ONSET_VOICED_S * SR) // FRAME
    while stretches:
        a, b = stretches[-1]
        seg = audio[first + a * FRAME : first + (b + 1) * FRAME]
        if int(np.sum(voiced(seg))) >= need:
            return first + a * FRAME
        stretches.pop()
    return at


def lift_onset(audio: FloatArray, a: int, b: int, heard: FloatArray) -> FloatArray:
    """``audio`` with the speech in ``audio[a:b]`` (what the VAD missed) raised to the level of ``heard`` (the speech
    it heard): one gain, at most ``MAX_GAIN_DB``, never below 1. A quiet first word is otherwise left far below the rest
    of the utterance, where the recogniser makes it into a fragment ("t can you") or drops it.

    The level of ``audio[a:b]`` is that of its voiced frames, its vowels: a part with none is left alone, and a hiss
    or a breath before a vowel that is already loud is not raised to the vowel's level ("16. Speech servers")."""
    b = min(b, len(audio))
    if b - a < FRAME or len(heard) < int(0.1 * SR):
        return audio
    part = audio[a:b]
    vowels = voiced(part)
    if not np.any(vowels):
        return audio
    level = float(np.max(frame_rms(part)[vowels]))
    gain = min(10.0 ** (MAX_GAIN_DB / 20.0), speech_level(heard) / max(level, 1e-9))
    if gain <= 10.0 ** (2.0 / 20.0):  # as loud already: one frame against the 90th percentile of many
        return audio
    out = np.array(audio, dtype=np.float32)
    out[a:b] = np.clip(out[a:b] * gain, -1.0, 1.0)
    return out


# One letter on its own where an utterance starts, other than the words "a" and "I" ("'s" included): what a small
# streaming transducer makes of the first sound of a fresh stream when it hears it twice ("o good morning",
# "s speech servers", "t after early nightfall").
STRAY_LETTER = re.compile(r"^'?[b-hj-z]$", re.IGNORECASE)


def hush_before(audio: FloatArray, end: int, room_db: float) -> FloatArray:
    """``audio`` with each burst of sound before ``audio[end]``, where the speech begins, brought down to the room
    (``room_db``): a stretch of 20 ms frames (counted back from ``end``) louder than the room by ``ONSET_MARGIN_DB``
    that ends ``BURST_GAP_S`` or more before the speech and has fewer than ``ONSET_VOICED_S`` of voiced frames in it;
    after digital silence (``DIGITAL_SILENCE_DB``) it becomes digital silence. A quiet word (voiced) and the hiss that
    starts the first word (right before it) are left alone. This is for a click or a burst of noise that the VAD did
    not take for speech but a pre-roll lengthened to an encoder chunk reached back over: the 20M Zipformer heard an
    80 ms click 0.3 s before en-short and wrote "ular good morning"; at the room's level it is gone ("good morning"),
    while brought down only to -90 dBFS after digital silence it still gave "ul"."""
    end = min(end, len(audio))
    n = end // FRAME
    if n == 0:
        return audio
    a = end - n * FRAME
    db = 20.0 * np.log10(frame_rms(audio[a:end]) + 1e-9)
    loud = db > max(room_db + ONSET_MARGIN_DB, ONSET_FLOOR_DB)
    last = n - int(BURST_GAP_S * SR) // FRAME  # stretches ending before this frame are far enough from the speech
    if last <= 0 or not np.any(loud[:last]):
        return audio
    vowels = voiced(audio[a:end])
    need = int(ONSET_VOICED_S * SR) // FRAME
    out = np.array(audio, dtype=np.float32)
    frames = out[a:end].reshape(n, FRAME)
    k = 0
    while k < n:
        if not loud[k]:
            k += 1
            continue
        j = k
        while j < n and loud[j]:
            j += 1
        if j <= last and int(np.sum(vowels[k:j])) < need:  # [k, j): a burst, not a word
            if room_db <= DIGITAL_SILENCE_DB:
                frames[k:j] = 0.0
            else:
                frames[k:j] *= (10.0 ** ((room_db - db[k:j]) / 20.0)).astype(np.float32)[:, None]
        k = j
    return out


def without_stray_letter(text: str, words: list[Word]) -> tuple[str, list[Word]]:
    """``text`` and its ``words`` without a stray letter at the start (:data:`STRAY_LETTER`)."""
    head, _, rest = text.partition(" ")
    if not STRAY_LETTER.match(head):
        return text, words
    if words and STRAY_LETTER.match(words[0].w.strip()):
        words = words[1:]
    return rest.strip(), words


def speech_lead(audio: FloatArray, onset: int, room_db: float) -> int | None:
    """Where the sound of the speech that starts at or after ``audio[onset]`` begins: the first voiced frames
    (``ONSET_VOICED_S``) louder than the room (``room_db``) by ``ONSET_MARGIN_DB``, from ``LEAD_BACK_S`` before
    ``onset`` on, and the sound right before them with no pause (the hiss of an "s"); None when ``audio`` has no
    such frames yet."""
    a0 = max(0, onset - int(LEAD_BACK_S * SR))
    seg = audio[a0:]
    db = 20.0 * np.log10(frame_rms(seg) + 1e-9)
    loud = db > max(room_db + ONSET_MARGIN_DB, ONSET_FLOOR_DB)
    ok = voiced(seg) & loud
    need = int(ONSET_VOICED_S * SR) // FRAME
    if len(ok) < need:
        return None
    hits = np.nonzero(np.convolve(ok.astype(np.int32), np.ones(need, dtype=np.int32), mode="valid") >= need)[0]
    if len(hits) == 0:
        return None
    k = int(hits[0])
    while k > 0 and loud[k - 1]:
        k -= 1
    return a0 + k * FRAME


def aligned_preroll(n: int, primed_s: float, chunk_s: float) -> int:
    """The shortest pre-roll of at least ``n`` samples that puts the speech after it ``CHUNK_LEAD_S`` into one of the
    recogniser's encoder chunks (``chunk_s`` long, counted from the ``primed_s`` it hears when a stream opens);
    ``n`` when the chunk is not known."""
    if chunk_s <= 0:
        return n
    c = round(chunk_s * SR)
    return n + (round((CHUNK_LEAD_S - primed_s) * SR) - n) % c


def better_start(main: StreamResult, alt: StreamResult, shift_s: float) -> StreamResult:
    """Of an utterance decoded twice, ``main`` from a pre-roll aligned to the encoder's chunks and ``alt`` from the
    pre-roll asked for: ``main`` when it has more words before the two first agree (a stray letter at the start not
    counted; with nothing in common, more words in all), else ``alt``, its word times moved by ``shift_s`` (where its
    stream began, less where the main one did).

    The 20M Zipformer keeps or loses a first word by where in its 0.32 s encoder chunks the word starts, and which
    place keeps it differs from word to word: "Good morning" was kept 0.05 s into a chunk and lost 0.2 s in ("in could
    you tell me"), "Speech servers" the other way round ("servers are judged"). Where the two first agree ("could
    you", "servers are") the one with more before it has the first word; what they make of the rest of a long
    utterance differs by a word here and there and says nothing about its start."""
    m = without_stray_letter(main.text.strip(), [])[0].lower().split()
    a = without_stray_letter(alt.text.strip(), [])[0].lower().split()
    blocks = [b for b in difflib.SequenceMatcher(None, m, a, autojunk=False).get_matching_blocks() if b.size]
    agree = next((b for b in blocks if b.size >= 2), blocks[0] if blocks else None)
    if (agree.a > agree.b) if agree is not None else (len(m) > len(a)):
        return main
    return StreamResult(alt.text, [Word(w.w, w.t0 + shift_s, w.t1 + shift_s, w.p) for w in alt.words])


class PreRoll:
    """The newest ``seconds`` of audio, with the stream position where it ends."""

    def __init__(self, seconds: float) -> None:
        self.size = max(0, int(seconds * SR))
        self.buf = np.zeros(0, dtype=np.float32)
        self.end = 0  # stream position just after the last sample held

    def push(self, samples: FloatArray, end: int) -> None:
        """Hold ``samples``, the audio that ends at stream position ``end``."""
        self.end = end
        if len(samples) >= self.size:
            self.buf = np.array(samples[len(samples) - self.size :], dtype=np.float32)
        elif len(samples):
            self.buf = np.concatenate([self.buf[max(0, len(self.buf) + len(samples) - self.size) :], samples])

    @property
    def start(self) -> int:
        return self.end - len(self.buf)

    def take(self, since: int = 0) -> tuple[FloatArray, int]:
        """Empty the buffer: the audio held from stream position ``since`` on, and where that audio starts."""
        start = max(since, self.start)
        audio = self.buf[start - self.start :]
        self.buf = np.zeros(0, dtype=np.float32)
        return audio, start


class _NativeProcessor(_Processor):
    """Incremental decoding with a streaming model.

    With a VAD and a model that asks for it (``stream_preroll_s``), the recogniser hears only speech: idle audio is
    held back, the newest ``stream_preroll_s`` of it, and when the VAD hears speech start a fresh stream gets that
    pre-roll and then the live audio, until the VAD hears speech end. A streaming encoder gives its first words
    away when it starts on silence it has been fed for a long time, or when its own endpointing resets it mid-word
    (the 20M Zipformer turned "Speech servers" into "ers", and quiet microphone speech into nothing at all).
    Without a VAD, or for a model that wants every sample (a cloud provider's live socket), the recogniser hears
    everything and its own endpointing ends utterances.
    """

    def __init__(self, session: STTSession, entry: LoadedModel, stt: STTModel, detector: VadDetector | None) -> None:
        super().__init__(session, entry, stt, detector)
        self.stream: STTStream = stt.create_stream(session.cfg.language)
        self.gated = detector is not None and stt.stream_preroll_s > 0
        # a model whose encoder chunk is known decodes each utterance twice: from a pre-roll aligned to a chunk (the
        # main stream) and from the pre-roll asked for (this one); the result whose start is fuller is taken
        # (better_start). The 20M Zipformer, the one model that gives its chunk, is small enough for the cost.
        self.alt: STTStream | None = None
        if self.gated and stt.stream_chunk_s > 0:
            self.alt = stt.create_stream(session.cfg.language)
        self.alt_base = 0  # stream position where the second stream began
        self.preroll_n = int(stt.stream_preroll_s * SR)
        # a VAD reports a start after it (a neural one by a few hundred milliseconds, the energy fallback by up to its
        # patience), so more than the pre-roll is held; and more again while a stream waits for a vowel (PENDING_S),
        # with room for a pre-roll lengthened to an encoder chunk
        self.preroll = PreRoll(stt.stream_preroll_s + VAD_LAG_S + PENDING_S + 0.5) if self.gated else None
        self.pending: int | None = None  # where the VAD heard speech start, while the stream waits for a vowel
        self.listening = not self.gated  # the recogniser is hearing the audio
        self.target_dbfs = stt.stream_speech_dbfs if self.gated else None
        self.gain = 1.0  # applied to what the recogniser hears in this utterance
        self.fed = 0  # stream position just after the last sample the recogniser heard
        self.base = 0  # stream position where the recogniser's current stream began
        self.speech_start: int | None = None  # where the VAD heard speech begin in this utterance
        self.speech_end: int | None = None  # where the VAD heard it end (the final's t1, however late it was seen)
        self.kept: list[FloatArray] = []  # the utterance's audio, kept only when speakers are labelled
        self.stability = PrefixStability()
        self.last_text = ""

    def _t0(self) -> float:
        start = self.speech_start if self.speech_start is not None and self.speech_start >= self.base else self.base
        return start / SR

    async def _step(self) -> tuple[StreamResult, bool]:
        if self.alt is None:
            (result, endpoint), info = await self.entry.queue.submit("stream", self.stream)
        else:
            main, other = await asyncio.gather(
                self.entry.queue.submit("stream", self.stream), self.entry.queue.submit("stream", self.alt)
            )
            (result, endpoint), info = main
            second: StreamResult = other[0][0]
            result = better_start(result, second, (self.alt_base - self.base) / SR)
        self.s.note_job(info.queue_ms, info.run_ms, info.batch_size, info.started, info.finished)
        return result, bool(endpoint)

    def _result(self) -> StreamResult:
        result = self.stream.result()
        if self.alt is not None:
            result = better_start(result, self.alt.result(), (self.alt_base - self.base) / SR)
        return result

    async def audio(self, samples: FloatArray, recv_t: float) -> None:
        # Audio that queued up while the model was busy arrives here in one piece, possibly with one utterance's end
        # and the next one's start in it. Fed whole, the two would come out as one final: the piece is cut where
        # the VAD heard the next utterance start, and the part before it ends the first utterance.
        first = self.pos  # the stream position of samples[0]
        self.pos += len(samples)
        done = 0
        ended = False
        for ev in self.run_vad(samples):
            await self.s.send_vad(ev.kind, ev.sample)
            cut = min(len(samples), max(done, ev.sample - first))
            if ev.kind == "speech_end":
                self.speech_end = ev.sample
                if self.gated:
                    if not self.listening:
                        self._hold(samples[done:cut], first + cut)
                        done = cut
                        if self.pending is not None:  # what the VAD heard, voiced or not, is decoded as before
                            self._listen(force=True)
                    if self.listening:  # the recogniser hears up to where the speech ended, then finishes
                        await self._feed(samples[done:cut], recv_t, vad_end=True)
                        self.listening = False
                    done = cut
                else:
                    ended = True
            elif ev.kind == "speech_start":
                if self.gated:
                    if not self.listening and self.pending is None:
                        self._hold(samples[done:cut], first + cut)
                        done = cut
                        self.pending = ev.sample
                elif ended:
                    await self._feed(samples[done:cut], recv_t, vad_end=True)
                    done, ended = cut, False
                self.speech_start = max(self.base, ev.sample)
        if not self.listening:
            self._hold(samples[done:], self.pos)
            if self.pending is not None:
                self._listen(force=False)
                if self.listening:
                    await self._feed(samples[:0], recv_t, vad_end=False)
        elif ended or done == 0 or done < len(samples):
            await self._feed(samples[done:], recv_t, vad_end=ended)

    def _hold(self, samples: FloatArray, end: int) -> None:
        assert self.preroll is not None
        self.preroll.push(samples, end)

    def _listen(self, force: bool) -> None:
        """The VAD heard speech start at ``self.pending``: a fresh stream hears the pre-roll before the speech.

        The speech may have begun before the reported start (a quiet first word the VAD missed): the pre-roll is
        counted from where it began, and that word is raised to the level of the rest. The VAD may also have reported
        the start on a burst of noise or a click before the speech: the stream opens once the audio held has the
        speech's voiced frames in it (:func:`speech_lead`), ``PENDING_S`` after the reported start without them, or
        (``force``) when the VAD hears the speech end. The pre-roll is made just long enough that the speech falls
        where the recogniser's encoder chunks keep a first word (:func:`aligned_preroll`), and the sound before the
        speech in it is brought down to the room (:func:`hush_before`). What was heard after the reported start sets
        the utterance's gain."""
        pre = self.preroll
        assert pre is not None and self.pending is not None
        speech_at = min(self.pending, pre.end)
        at = speech_at - pre.start
        room = room_level(pre.buf[: max(0, at)])
        onset = speech_onset(pre.buf, at)
        lead = speech_lead(pre.buf, onset, room)
        if lead is None:
            if not force and pre.end - speech_at < int(PENDING_S * SR):
                return  # nothing voiced yet: wait for it
            lead = onset
        self.pending = None
        # the pre-roll is counted from the quiet word found before the reported start, else from the speech's sound
        first = pre.start + (min(onset, lead) if onset < at else lead)
        lead += pre.start  # stream positions from here on
        since = lead - aligned_preroll(lead - first + self.preroll_n, self.stt.stream_primed_s, self.stt.stream_chunk_s)
        # the second stream's as before: from the reported start or the quiet word, no further back than what is held
        alt_since = max(pre.start, pre.start + onset - self.preroll_n)
        lowest = min(since, alt_since) if self.alt is not None else since
        audio, start = pre.take(since=lowest)
        if start > lowest:  # less held than that: digital silence before it, as a fresh stream's own priming is
            audio = np.concatenate([np.zeros(start - lowest, dtype=np.float32), audio])
        heard = sounding(audio[speech_at - lowest :], room)
        audio = lift_onset(audio, first - lowest, speech_at - lowest, heard)
        audio = hush_before(audio, first - lowest, room)
        if self.target_dbfs is not None:
            self.gain = speech_gain(heard, self.target_dbfs)
        self.stream.restart()
        self.base = self.fed = since
        self.listening = True
        alt_audio = None
        if self.alt is not None:
            self.alt.restart()
            self.alt_base = alt_since
            alt_audio = audio[alt_since - lowest :]
        self._accept(audio[since - lowest :], alt_audio)

    def _accept(self, samples: FloatArray, alt: FloatArray | None = None) -> None:
        """The recognisers hear ``samples`` (the second stream ``alt`` instead, when given), with the gain."""
        if self.alt is not None:
            alt = samples if alt is None else alt
            if len(alt):
                self.alt.accept(np.clip(alt * self.gain, -1.0, 1.0).astype(np.float32) if self.gain != 1.0 else alt)
        if len(samples) == 0:
            return
        if self.gain != 1.0:
            samples = np.clip(samples * self.gain, -1.0, 1.0).astype(np.float32)
        self.stream.accept(samples)
        self.fed += len(samples)
        if self.s.cfg.diarize:
            self.kept.append(samples)

    async def _feed(self, samples: FloatArray, recv_t: float, vad_end: bool) -> None:
        self._accept(samples)
        result, endpoint = await self._step()
        if self.gated:
            endpoint = False  # the VAD ends utterances; the model's endpointing would cut one short mid-word
        text = result.text.strip()
        if self.gated:
            text, _ = without_stray_letter(text, [])
        if text and text != self.last_text and self.s.cfg.interim and not (endpoint or vad_end):
            self.last_text = text
            stable = self.stability.update(text)
            await self.s.send_partial(text, self._t0(), self.fed / SR, recv_t, stable, self.first_partial_pending)
            self.first_partial_pending = False
        if (endpoint or vad_end) and (text or endpoint or self.gated):
            await self._finish_utterance(recv_t, pad=vad_end and not endpoint)

    async def _finish_utterance(self, recv_t: float, pad: bool) -> None:
        if pad:
            # flush the model's look-ahead so the last word is not held back
            self._flush()
            result, _ = await self._step()
        else:
            result = self._result()
        text = result.text.strip()
        base = self.base / SR
        words = [Word(w.w, w.t0 + base, w.t1 + base, w.p) for w in result.words] if self.s.cfg.word_timestamps else []
        if self.gated:
            text, words = without_stray_letter(text, words)
        if text:
            audio = None
            if self.s.cfg.diarize and self.kept:
                audio = np.concatenate(self.kept)
                if self.speech_start is not None:  # drop the silence before the speaker started
                    audio = audio[max(0, self.speech_start - self.base) :]
            end = (
                self.speech_end if self.speech_end is not None and self.base < self.speech_end < self.fed else self.fed
            )
            await self.s.send_final(text, self._t0(), end / SR, recv_t, words, None, audio)
        self.kept = []
        self.stream.reset()
        if self.alt is not None:
            self.alt.reset()
        self.base = self.fed
        self.speech_start = None
        self.speech_end = None
        self.last_text = ""
        self.stability.reset()
        self.first_partial_pending = True

    def _flush(self) -> None:
        """Padding after the speech, so the model's look-ahead does not hold its last word back."""
        pad = np.zeros(int(TAIL_PAD_S * SR), dtype=np.float32)
        self.stream.accept(pad)
        if self.alt is not None:
            self.alt.accept(pad)

    async def finalize(self, recv_t: float, reason: str) -> None:
        if self.pending is not None and not self.listening:
            self._listen(force=True)
        if reason == "stop":
            if self.listening:
                self._flush()
                self.stream.finish()
                if self.alt is not None:
                    self.alt.finish()
                await self._step()
                await self._finish_utterance(recv_t, pad=False)
        elif self.listening:
            await self._finish_utterance(recv_t, pad=True)
        if self.detector is not None:
            for ev in self.detector.flush():
                await self.s.send_vad(ev.kind, ev.sample)
        if self.gated:
            self.listening = False


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
        self.final_end = 0  # where the audio of the last final ended: the next utterance starts no earlier

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
        options = {
            k: v for k, v in self.s.decode_options.items() if k in ENGINE_OPTIONS and k in self.stt.decode_options
        }
        req = STTRequest(
            self._slice(start, end).copy(),
            self.s.cfg.language,
            "transcribe",
            None,
            words,
            options=options,
            hotwords=self.s.cfg.hotwords if "hotwords" in self.stt.request_fields else None,
        )
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
                # a quiet first word the VAD missed is found again before its start, and raised to the level of the
                # speech it heard (Moonshine left it out: "Can you tell me" for "So can you tell me")
                at = min(ev.sample, self.pos)
                floor = max(self.buf_start, self.final_end)
                onset = floor + speech_onset(self._slice(floor, at), at - floor) if at > floor else at
                if onset < at:
                    a, b = onset - self.buf_start, at - self.buf_start
                    heard = sounding(self._slice(at, self.pos), room_level(self._slice(floor, at)))
                    self.buf[a:b] = lift_onset(self.buf[a:b], 0, b - a, heard)
                # padded at both ends, two utterances close together would share audio, and a word on the seam
                # would be in both finals
                start = max(self.buf_start, self.final_end, onset - int(ONSET_PAD_S * SR))
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
        self.final_end = end
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
            Word(w.w, w.t0 + t0, w.t1 + t0, w.p)
            for w in (result.words or [w for s in result.segments for w in s.words])
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
