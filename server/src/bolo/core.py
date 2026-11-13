"""The server core: one object that owns every subsystem and implements the
operations (transcribe, synthesise, pull, …) that the protocol front ends call.

HTTP routes, WebSockets, the Wyoming server, the CLI and the benchmark all go
through this class, so they share scheduling, tracing and model lifecycle.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psutil

from bolo import __version__
from bolo.audio.resample import resample
from bolo.catalog import Catalog, ModelSpec
from bolo.components import ComponentManager
from bolo.config import Settings
from bolo.diarize import DiarizeOptions, merge_turns, renumber, speaker_label
from bolo.engines import EngineRegistry
from bolo.engines.base import (
    FloatArray,
    Segment,
    SpeakerTurn,
    STTModel,
    STTRequest,
    STTResult,
    TTSModel,
    VadDetector,
    VADModel,
    VadParams,
    Voice,
    Word,
)
from bolo.errors import (
    BoloError,
    ComponentMissing,
    EngineUnavailable,
    GatedModel,
    InvalidRequest,
    ModelNotInstalled,
    SynthesisFailed,
)
from bolo.jobs import Job, JobStore
from bolo.lifecycle import LoadedModel, ModelManager
from bolo.llm import LLMClient
from bolo.obs.events import EventBus
from bolo.obs.trace import Trace, TraceStore, preview
from bolo.store import HubClient, ModelStore
from bolo.streaming.vad import EnergyVad, FallbackVad, detect_segments, merge_segments
from bolo.text.normalize import normalize_for_tts
from bolo.text.phonemize import INSTALL_HINT
from bolo.text.segment import chunk_text

log = logging.getLogger("bolo")

OPENAI_STT_ALIASES = (
    "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "gpt-4o-transcribe-diarize", "gpt-transcribe",
    "gpt-live-transcribe",
)  # fmt: skip
OPENAI_TTS_ALIASES = ("tts-1", "tts-1-hd", "gpt-4o-mini-tts")
OPENAI_VOICES = (
    "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse", "marin", "cedar",
)  # fmt: skip


@dataclass
class Transcript:
    text: str
    language: str | None
    duration: float
    segments: list[Segment] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    model: str = ""


@dataclass
class SynthChunk:
    samples: FloatArray
    sample_rate: int
    seq: int
    text: str
    first_of_text: bool


class Bolo:
    def __init__(self, settings: Settings | None = None, engines: EngineRegistry | None = None) -> None:
        self.settings = settings or Settings()
        s = self.settings
        self.started = time.monotonic()
        self.events = EventBus()
        self.traces = TraceStore(s.trace_buffer)
        self.jobs = JobStore()
        self.engines = engines or EngineRegistry(include_fake=s.fake_engine)
        self.catalog = Catalog.load(s.data_dir, include_fake=s.fake_engine)
        token = s.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.store = ModelStore(s.data_dir, HubClient(s.hf_endpoint, token))
        self.manager = ModelManager(s, self.catalog, self.store, self.engines, self.events, self._auto_pull)
        self.components = ComponentManager(self)
        self.llm = LLMClient(s)
        self.active_streams = 0
        self._tasks: list[asyncio.Task[Any]] = []
        self._pull_tasks: dict[str, asyncio.Task[Any]] = {}
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)
        self._cpu_percent = 0.0
        self._cpu_at = 0.0
        self.ready = False
        self.traces.listeners.append(self._on_trace)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        self.events.bind(asyncio.get_running_loop())
        self._tasks.append(asyncio.create_task(self._stats_ticker(), name="bolo-stats"))
        self.ready = True
        if self.settings.preload:
            self._tasks.append(asyncio.create_task(self._preload(), name="bolo-preload"))

    async def _preload(self) -> None:
        for name in self.settings.preload:
            try:
                await self.manager.load(name)
            except Exception as exc:
                self.log("warning", f"could not preload '{name}': {exc}")

    async def stop(self) -> None:
        self.ready = False
        for t in [*self._tasks, *self._pull_tasks.values()]:
            t.cancel()
        for t in [*self._tasks, *self._pull_tasks.values()]:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks.clear()
        await self.manager.close()

    def log(self, level: str, message: str) -> None:
        getattr(log, level if level in ("debug", "info", "warning", "error") else "info")(message)
        self.events.publish_threadsafe("log", {"level": level, "message": message})

    def _on_trace(self, trace: Trace) -> None:
        self.events.publish_threadsafe("request", {"request": trace.to_dict()})

    async def _stats_ticker(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            if self.events.subscribers:
                self.events.publish("stats", self.stats())

    # ------------------------------------------------------------------
    # information
    # ------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine().lower())
        system = {"darwin": "macos"}.get(platform.system().lower(), platform.system().lower())
        s = self.settings
        return {
            "name": "bolo",
            "version": __version__,
            "engines": [e.to_dict() for e in self.engines.statuses()],
            "device": s.device,
            "cpu_count": os.cpu_count() or 1,
            "platform": f"{system}-{machine}",
            "python": platform.python_version(),
            "auth_required": bool(s.api_key),
            "data_dir": str(s.data_dir),
            "defaults": {"stt": s.default_stt, "realtime_stt": s.default_realtime_stt, "tts": s.default_tts, "vad": s.default_vad},
            "wyoming": {"enabled": s.wyoming, "host": s.wyoming_host, "port": s.wyoming_port},
            "components": self.components.status(),
            "capabilities": self.capabilities(),
            "scheduler": {
                "workers": s.workers, "threads": s.resolved_threads(), "max_batch": s.max_batch,
                "batch_wait_ms": s.batch_wait_ms, "max_queue": s.max_queue,
            },
        }  # fmt: skip

    def capabilities(self) -> dict[str, Any]:
        """What this server can do beyond plain STT/TTS, for clients that want to adapt their UI."""
        s = self.settings
        diar = self.catalog.resolve(s.default_diarization)
        diar_ready = bool(diar and self.manager.is_installed(diar))
        llm = self.llm.describe()
        return {
            "diarization": {
                "available": bool(diar) and (diar_ready or s.auto_pull),
                "installed": diar_ready,
                "default_model": diar.id if diar else None,
                "http": {"endpoint": "/api/v1/diarize", "transcriptions": ["diarize", "diarized_json"]},
                "websocket": "diarize=true (one speaker per utterance, labelled online)",
            },
            "realtime": {
                "endpoint": "/v1/realtime",
                "transcription_sessions": True,
                "conversation_sessions": llm["configured"],
                "audio_formats": ["audio/pcm (24 kHz)", "audio/pcmu", "audio/pcma"],
                "reference": "OpenAI Realtime API (GA event names), as typed in openai-python 3.50.0",
            },
            "chat_completions": {
                "endpoint": "/v1/chat/completions",
                "available": llm["configured"],
                "audio": llm["configured"],
            },
            "llm": {"configured": llm["configured"], "base_url": llm["base_url"], "model": llm["model"]},
        }

    def model_api(self, spec: ModelSpec) -> dict[str, Any]:
        job = self.jobs.active_pull(spec.id)
        pulling = {"job_id": job.id, "progress": round(job.progress, 4)} if job else None
        installed = self.manager.is_installed(spec)
        size = self.store.size_on_disk(spec.id) if installed and not spec.builtin else None
        return spec.to_api(
            installed=installed,
            loaded=spec.id in self.manager.loaded,
            pulling=pulling,
            size_on_disk=size,
            available=all(self.components.is_installed(c) for c in spec.requires),
        )

    def catalog_api(
        self, task: str | None = None, q: str | None = None, installed: bool | None = None
    ) -> list[dict[str, Any]]:
        out = []
        for spec in self.catalog.search(task or None, q):
            entry = self.model_api(spec)
            if installed is not None and entry["installed"] != installed:
                continue
            out.append(entry)
        return out

    def cpu_percent(self) -> float:
        now = time.monotonic()
        if now - self._cpu_at >= 0.5:
            self._cpu_at = now
            with contextlib.suppress(Exception):
                self._cpu_percent = self._proc.cpu_percent(None) / (os.cpu_count() or 1)
        return round(self._cpu_percent, 1)

    def stats(self) -> dict[str, Any]:
        models: list[dict[str, Any]] = []
        ids = list(dict.fromkeys([*self.manager.loaded, *self.traces.models]))
        for model_id in ids:
            entry = self.manager.loaded.get(model_id)
            st = self.traces.models.get(model_id)
            spec = self.catalog.resolve(model_id)
            task = entry.spec.task if entry else (st.task if st else (spec.task if spec else "stt"))
            ttl = self.manager.ttl_for(spec) if spec else None
            item: dict[str, Any] = {
                "id": model_id,
                "task": task,
                "loaded": entry is not None,
                "memory_bytes": entry.memory_bytes if entry else 0,
                "idle_s": round(entry.idle_s(), 1) if entry else None,
                "ttl_s": ttl,
                "active": entry.active if entry else 0,
                "requests": st.requests if st else 0,
                "errors": st.errors if st else 0,
                "latency_ms": st.latency_summary() if st else {"first_partial": None, "final": None, "ttfa": None},
                "rtf": st.rtf_summary() if st else None,
            }
            if entry:
                q = entry.queue
                item["queue"] = {
                    "depth": q.depth,
                    "inflight": q.inflight,
                    "workers": q.workers,
                    "batches": q.stats.batches,
                    "mean_batch": round(q.stats.mean_batch, 2),
                    "max_batch": q.stats.max_batch_seen,
                    "rejected": q.stats.rejected,
                    "queue_ms": q.stats.queue_ms.summary(),
                }
            models.append(item)
        mem = self._proc.memory_info()
        return {
            "uptime_s": round(time.monotonic() - self.started, 1),
            "active_streams": self.active_streams,
            "queue_depth": sum(m.queue.depth for m in self.manager.loaded.values()),
            "system": {"cpu_percent": self.cpu_percent(), "rss_bytes": mem.rss, "threads": self._proc.num_threads()},
            "totals": {
                "requests": self.traces.total_requests,
                "errors": self.traces.total_errors,
                "audio_seconds_in": round(self.traces.audio_seconds_in, 2),
                "audio_seconds_out": round(self.traces.audio_seconds_out, 2),
            },
            "models": models,
        }

    # ------------------------------------------------------------------
    # model resolution, pulling
    # ------------------------------------------------------------------
    def resolve(self, name: str | None, task: str) -> ModelSpec:
        s = self.settings
        if not name or name in ("default", "auto"):
            defaults = {
                "stt": s.default_stt,
                "tts": s.default_tts,
                "vad": s.default_vad,
                "diarization": s.default_diarization,
            }
            name = defaults[task]
        elif name == "realtime":
            name = s.default_realtime_stt
        if self.catalog.resolve(name) is None:
            if task == "stt" and name in OPENAI_STT_ALIASES:
                name = s.default_stt
            elif task == "tts" and name in OPENAI_TTS_ALIASES:
                name = s.default_tts
        spec = self.manager.resolve(name)
        if spec.task != task:
            raise InvalidRequest(
                f"model '{spec.id}' is a {spec.task} model and cannot be used for {task}", param="model"
            )
        return spec

    def _pull_blocking(self, spec: ModelSpec, job: Job) -> None:
        def progress(done: int, total: int) -> None:
            job.bytes_done, job.bytes_total = done, total
            job.progress = min(1.0, done / total) if total else 0.0
            now = time.monotonic()
            if now - progress.last >= 0.25 or done == total:  # type: ignore[attr-defined]
                progress.last = now  # type: ignore[attr-defined]
                self.events.publish_threadsafe("job", {"job": job.to_dict()})

        progress.last = 0.0  # type: ignore[attr-defined]
        self.store.pull(spec.id, spec.source, progress=progress, cancelled=lambda: job.cancel_requested)

    async def _run_pull(self, spec: ModelSpec, job: Job) -> None:
        job.state = "running"
        self.events.publish("job", {"job": job.to_dict()})
        try:
            await asyncio.get_running_loop().run_in_executor(None, self._pull_blocking, spec, job)
            job.state, job.progress = "done", 1.0
            self.events.publish("model", {"id": spec.id, "event": "installed"})
            self.log("info", f"installed {spec.id}")
        except asyncio.CancelledError:
            job.cancel_requested = True
            job.state, job.error = "error", "cancelled"
            raise
        except Exception as exc:
            job.state = "error"
            job.error = exc.message if isinstance(exc, BoloError) else f"{type(exc).__name__}: {exc}"
            self.log("error", f"pull of {spec.id} failed: {job.error}")
        finally:
            self.events.publish("job", {"job": job.to_dict()})

    def start_pull(self, name: str) -> Job:
        """Start (or join) a background download of a catalog model."""
        spec = self.manager.resolve(name)
        existing = self.jobs.active_pull(spec.id)
        if existing:
            return existing
        job = self.jobs.create("pull", spec.id)
        if spec.builtin or self.store.is_installed(spec.id):
            job.state, job.progress = "done", 1.0
            return job
        if spec.gated and not self.store.hub.token:
            job.state = "error"
            job.error = (
                f"'{spec.id}' is a gated model. Accept its terms at {spec.source_url} and set BOLO_HF_TOKEN "
                "(or HF_TOKEN) to a Hugging Face token with access, then pull again."
            )
            return job
        task = asyncio.create_task(self._run_pull(spec, job), name=f"bolo-pull-{spec.id}")
        self._pull_tasks[job.id] = task
        task.add_done_callback(lambda _t: self._pull_tasks.pop(job.id, None))
        return job

    async def pull_and_wait(self, name: str) -> Job:
        job = self.start_pull(name)
        task = self._pull_tasks.get(job.id)
        if task is not None:
            await asyncio.shield(task)
        return job

    async def _auto_pull(self, spec: ModelSpec) -> None:
        if not self.settings.auto_pull:
            raise ModelNotInstalled(
                f"model '{spec.id}' is not installed and auto_pull is off; run `bolo pull {spec.id}` "
                f"or POST /api/v1/models/pull"
            )
        if spec.gated and not self.store.hub.token:
            raise GatedModel(
                f"model '{spec.id}' is gated and is never downloaded automatically. Accept its terms at "
                f"{spec.source_url}, set BOLO_HF_TOKEN and run `bolo pull {spec.id}`."
            )
        self.log("info", f"model '{spec.id}' is not installed; pulling it now")
        job = await self.pull_and_wait(spec.id)
        if job.state != "done":
            raise ModelNotInstalled(f"could not pull model '{spec.id}': {job.error}")

    async def remove(self, name: str) -> bool:
        spec = self.manager.resolve(name)
        await self.manager.unload(spec.id, reason="removed")
        removed = await asyncio.get_running_loop().run_in_executor(None, self.store.remove, spec.id)
        if spec.custom:
            self.catalog.remove_custom(spec.id)
        if removed:
            self.events.publish("model", {"id": spec.id, "event": "removed"})
        return removed

    # ------------------------------------------------------------------
    # VAD
    # ------------------------------------------------------------------
    async def vad_detector(
        self, params: VadParams | None = None, model: str | None = None, fallback: bool = False
    ) -> tuple[VadDetector, str]:
        """A fresh detector and the name of what backs it.

        Uses the configured neural VAD when it is installed (or can be pulled);
        otherwise falls back to the built-in energy detector.
        """
        params = params or VadParams(
            min_silence_s=self.settings.endpoint_silence_ms / 1000.0, max_speech_s=self.settings.max_utterance_s
        )
        try:
            spec = self.resolve(model, "vad")
            usable = self.engines.status(spec.engine).available and (
                self.manager.is_installed(spec) or (self.settings.auto_pull and not spec.gated)
            )
            if usable:
                entry = await self.manager.load(spec.id)
                entry.last_used = time.monotonic()
                assert isinstance(entry.model, VADModel)
                detector = entry.model.create_detector(params)
                return (FallbackVad(detector, params) if fallback else detector), spec.id
        except BoloError as exc:
            if model:
                raise
            log.debug("neural VAD unavailable (%s); using the energy detector", exc.message)
        return EnergyVad(params), "energy"

    # ------------------------------------------------------------------
    # speech to text
    # ------------------------------------------------------------------
    async def transcribe(
        self,
        samples: FloatArray,
        sample_rate: int,
        *,
        model: str | None = None,
        language: str | None = None,
        task: str = "transcribe",
        prompt: str | None = None,
        word_timestamps: bool = False,
        temperature: float = 0.0,
        hotwords: str | None = None,
        vad_filter: bool | None = None,
        trace: Trace | None = None,
        on_segment: Callable[[Segment], Awaitable[None]] | None = None,
        diarize: DiarizeOptions | None = None,
    ) -> Transcript:
        """Transcribe a complete recording. Long audio is split at pauses and decoded in batches."""
        spec = self.resolve(model, "stt")
        trace = trace or Trace("stt", "internal", spec.id)
        trace.model = spec.id
        trace.audio_seconds = len(samples) / sample_rate if sample_rate else 0.0
        duration = trace.audio_seconds
        try:
            async with self.manager.use(spec.id) as entry:
                stt = entry.model
                assert isinstance(stt, STTModel)
                if task == "translate" and not stt.supports_translate:
                    raise InvalidRequest(
                        f"model '{spec.id}' cannot translate; use a multilingual Whisper model", param="model"
                    )
                with trace.phase("resample"):
                    if sample_rate != stt.sample_rate:
                        if len(samples) > sample_rate * 30:
                            samples = await asyncio.to_thread(resample, samples, sample_rate, stt.sample_rate)
                        else:
                            samples = resample(samples, sample_rate, stt.sample_rate)
                    samples = np.ascontiguousarray(samples, dtype=np.float32)
                sr = stt.sample_rate
                use_vad = vad_filter if vad_filter is not None else duration > stt.max_chunk_s
                chunks: list[tuple[int, int]]
                speakers: list[str] | None = None
                if diarize is not None:
                    # Cut at the speaker turns and transcribe each turn: the speaker of every
                    # piece of text is then known without needing word timestamps.
                    turns = merge_turns(await self._speaker_turns(samples, diarize, trace), max_len_s=stt.max_chunk_s)
                    chunks = [(int(t.start * sr), min(len(samples), int(t.end * sr))) for t in turns]
                    speakers = [speaker_label(t.speaker) for t in turns]
                elif use_vad:
                    with trace.phase("vad"):
                        detector, _ = await self.vad_detector(
                            VadParams(min_silence_s=0.3, max_speech_s=stt.max_chunk_s)
                        )
                        speech = await asyncio.to_thread(detect_segments, detector, samples)
                        chunks = merge_segments(speech, len(samples), sample_rate=sr, max_chunk_s=stt.max_chunk_s)
                elif len(samples) < int(0.05 * sr):
                    chunks = []
                else:
                    chunks = [(0, len(samples))]
                transcript = await self._decode_chunks(
                    entry, stt, samples, chunks, trace,
                    language=language, task=task, prompt=prompt, word_timestamps=word_timestamps,
                    temperature=temperature, hotwords=hotwords, on_segment=on_segment, speakers=speakers,
                )  # fmt: skip
                transcript.duration = duration
                transcript.model = spec.id
                trace.mark_final()
                trace.language = transcript.language
                trace.text_preview = preview(transcript.text)
                trace.chars = len(transcript.text)
                trace.finish("ok")
                if duration > 0 and trace.total_ms is not None:
                    trace.rtf = trace.total_ms / 1000.0 / duration
                return transcript
        except asyncio.CancelledError:
            trace.finish("cancelled", "client disconnected")
            raise
        except BoloError as exc:
            trace.finish("error", exc.message)
            raise
        except Exception as exc:
            trace.finish("error", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.traces.record(trace)

    async def _decode_chunks(
        self,
        entry: LoadedModel,
        stt: STTModel,
        samples: FloatArray,
        chunks: list[tuple[int, int]],
        trace: Trace,
        *,
        language: str | None,
        task: str,
        prompt: str | None,
        word_timestamps: bool,
        temperature: float,
        hotwords: str | None,
        on_segment: Callable[[Segment], Awaitable[None]] | None,
        speakers: list[str] | None = None,
    ) -> Transcript:
        sr = stt.sample_rate
        submitted = trace.now()
        pending = [
            asyncio.ensure_future(
                entry.queue.submit(
                    "batch",
                    STTRequest(samples[a:b], language, task, prompt, word_timestamps, temperature, hotwords),
                )
            )
            for a, b in chunks
        ]
        segments: list[Segment] = []
        words: list[Word] = []
        detected: str | None = language
        first_start: float | None = None
        last_finish = submitted
        try:
            for index, ((a, _b), fut) in enumerate(zip(chunks, pending)):
                result, info = await fut
                assert isinstance(result, STTResult)
                trace.note_batch(info.batch_size)
                if first_start is None or info.started < first_start:
                    first_start = info.started
                last_finish = max(last_finish, info.finished)
                offset = a / sr
                detected = detected or result.language
                chunk_segments = result.segments or (
                    [Segment(0.0, (_b - a) / sr, result.text, result.words)] if result.text.strip() else []
                )
                for seg in chunk_segments:
                    shifted = Segment(
                        seg.start + offset,
                        seg.end + offset,
                        seg.text.strip(),
                        [Word(w.w, w.t0 + offset, w.t1 + offset) for w in seg.words],
                        seg.avg_logprob,
                        seg.no_speech_prob,
                        speakers[index] if speakers else None,
                    )
                    if not shifted.text:
                        continue
                    segments.append(shifted)
                    words.extend(shifted.words)
                    if on_segment is not None:
                        trace.mark_first_partial()
                        await on_segment(shifted)
                if not result.segments:
                    pass
                elif not words and result.words:
                    words.extend(Word(w.w, w.t0 + offset, w.t1 + offset) for w in result.words)
        except BaseException:
            for fut in pending:
                fut.cancel()
            raise
        if chunks:
            start = first_start if first_start is not None else submitted
            trace.add_phase("queue", submitted, start)
            trace.add_phase("decode", start, last_finish)
            self.traces.observe_latency(entry.id, "stt", queue_ms=(start - submitted) * 1000.0)
        with trace.phase("finalize"):
            text = " ".join(s.text for s in segments).strip()
        return Transcript(text, detected, 0.0, segments, words)

    # ------------------------------------------------------------------
    # speaker diarization
    # ------------------------------------------------------------------
    async def _speaker_turns(self, samples16: FloatArray, opts: DiarizeOptions, trace: Trace) -> list[SpeakerTurn]:
        """Run the diarization model on 16 kHz audio (adds a ``diarize`` phase to ``trace``)."""
        for name in ("num_speakers", "min_speakers", "max_speakers"):
            value = getattr(opts, name)
            if value is not None and not 1 <= value <= 32:
                raise InvalidRequest(f"{name} must be between 1 and 32", param=name)
        if opts.min_speakers and opts.max_speakers and opts.min_speakers > opts.max_speakers:
            raise InvalidRequest("min_speakers must not exceed max_speakers", param="min_speakers")
        spec = self.resolve(opts.model, "diarization")
        if len(samples16) < 1600:
            return []
        start = trace.now()
        async with self.manager.use(spec.id) as entry:
            turns, _ = await entry.queue.submit("diarize", (samples16, opts.num_speakers, opts.threshold))
            found = len({t.speaker for t in turns})
            forced = None
            if not opts.num_speakers:
                if opts.max_speakers and found > opts.max_speakers:
                    forced = opts.max_speakers
                elif opts.min_speakers and found < opts.min_speakers:
                    forced = opts.min_speakers
            if forced:
                turns, _ = await entry.queue.submit("diarize", (samples16, forced, opts.threshold))
        trace.add_phase("diarize", start, trace.now())
        trace.extra["diarization_model"] = spec.id
        trace.extra["speakers"] = len({t.speaker for t in turns})
        return renumber(turns)

    async def diarize(
        self,
        samples: FloatArray,
        sample_rate: int,
        opts: DiarizeOptions | None = None,
        trace: Trace | None = None,
    ) -> list[SpeakerTurn]:
        """Who spoke when. Returns speaker turns sorted by time, speakers numbered by first appearance."""
        opts = opts or DiarizeOptions()
        spec = self.resolve(opts.model, "diarization")
        trace = trace or Trace("diarization", "internal", spec.id)
        trace.model = spec.id
        trace.audio_seconds = len(samples) / sample_rate if sample_rate else 0.0
        try:
            with trace.phase("resample"):
                samples16 = await asyncio.to_thread(resample, samples, sample_rate, 16000)
            turns = await self._speaker_turns(np.ascontiguousarray(samples16, dtype=np.float32), opts, trace)
            trace.mark_final()
            trace.finish("ok")
            if trace.audio_seconds and trace.total_ms is not None:
                trace.rtf = trace.total_ms / 1000.0 / trace.audio_seconds
            return turns
        except asyncio.CancelledError:
            trace.finish("cancelled", "client disconnected")
            raise
        except BoloError as exc:
            trace.finish("error", exc.message)
            raise
        except Exception as exc:
            trace.finish("error", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.traces.record(trace)

    # ------------------------------------------------------------------
    # text to speech
    # ------------------------------------------------------------------
    def pick_voice(self, tts: TTSModel, voice: str | None, spec: ModelSpec) -> Voice | None:
        found = tts.find_voice(voice)
        if found is not None or not tts.voices:
            return found
        assert voice is not None
        if voice.lower() in OPENAI_VOICES:
            # Stock OpenAI clients send OpenAI voice names; map them onto the model's voices.
            return tts.voices[OPENAI_VOICES.index(voice.lower()) % len(tts.voices)]
        names = ", ".join(v.id for v in tts.voices[:12]) + (", …" if len(tts.voices) > 12 else "")
        raise InvalidRequest(f"voice '{voice}' is not available for model '{spec.id}'. Voices: {names}", param="voice")

    def require_components(self, spec: ModelSpec) -> None:
        """Fail fast when a model needs an optional component that is not installed."""
        for component in spec.requires:
            if not self.components.is_installed(component):
                raise ComponentMissing(f"model '{spec.id}' cannot be used yet: " + INSTALL_HINT)

    async def synthesize(
        self,
        text: str,
        *,
        model: str | None = None,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
        normalize: bool = True,
        trace: Trace | None = None,
    ) -> AsyncGenerator[SynthChunk, None]:
        """Synthesise ``text`` sentence by sentence, yielding audio as soon as it exists.

        This generator records the ``queue``, ``normalize``, ``synthesize_first``
        and ``synthesize_rest`` phases. A caller that passes its own ``trace``
        owns the ``encode`` phase and ``ttfa`` and must call
        :meth:`complete_tts` when it has delivered the audio.
        """
        own_trace = trace is None
        if trace is None:
            trace = Trace("tts", "internal", model or self.settings.default_tts)
        spec = self.resolve(model, "tts")
        trace.model = spec.id
        trace.chars = len(text)
        trace.text_preview = preview(text)
        total_samples = 0
        sample_rate = spec.sample_rate
        try:
            if not text.strip():
                raise InvalidRequest("the input text is empty", param="input")
            if not 0.25 <= speed <= 4.0:
                raise InvalidRequest("speed must be between 0.25 and 4.0", param="speed")
            self.require_components(spec)
            async with self.manager.use(spec.id) as entry:
                tts = entry.model
                assert isinstance(tts, TTSModel)
                sample_rate = tts.sample_rate
                chosen = self.pick_voice(tts, voice, spec)
                lang = language or (
                    chosen.language if chosen and chosen.language else (spec.languages[0] if spec.languages else "en")
                )
                trace.language = lang
                with trace.phase("normalize"):
                    clean = normalize_for_tts(text, lang) if normalize else text
                    pieces = chunk_text(clean) or [clean]
                loop = asyncio.get_running_loop()
                stop = threading.Event()  # set when the consumer goes away; the engine polls it
                seq = 0
                first_audio_at: float | None = None
                synth_started: float | None = None
                for piece in pieces:
                    out: asyncio.Queue[FloatArray | None] = asyncio.Queue()

                    def emit(chunk: FloatArray, _out: asyncio.Queue[FloatArray | None] = out) -> None:
                        loop.call_soon_threadsafe(_out.put_nowait, chunk)

                    submitted = trace.now()
                    job = asyncio.ensure_future(
                        entry.queue.submit("tts", (piece, chosen.id if chosen else voice, speed, emit, stop.is_set))
                    )
                    job.add_done_callback(lambda _f, _out=out: _out.put_nowait(None))  # type: ignore[misc]
                    first_of_text = True
                    try:
                        while True:
                            chunk = await out.get()
                            if chunk is None:
                                break
                            if first_audio_at is None:
                                first_audio_at = trace.now()
                            total_samples += len(chunk)
                            yield SynthChunk(chunk, sample_rate, seq, piece if first_of_text else "", first_of_text)
                            seq += 1
                            first_of_text = False
                        _n, info = await job
                    except BaseException:
                        stop.set()
                        job.cancel()
                        raise
                    if synth_started is None:
                        synth_started = info.started
                        trace.add_phase("queue", submitted, info.started)
                        self.traces.observe_latency(entry.id, "tts", queue_ms=info.queue_ms)
                    trace.note_batch(info.batch_size)
                end = trace.now()
                if synth_started is not None:
                    split = first_audio_at if first_audio_at is not None else end
                    trace.add_phase("synthesize_first", synth_started, split)
                    trace.add_phase("synthesize_rest", split, end)
            trace.audio_seconds = total_samples / sample_rate if sample_rate else 0.0
            if total_samples == 0:
                raise EngineUnavailable(f"model '{spec.id}' produced no audio for this text")
        except (asyncio.CancelledError, GeneratorExit):
            trace.audio_seconds = total_samples / sample_rate if sample_rate else 0.0
            trace.finish("cancelled", "client disconnected")
            raise
        except BoloError as exc:
            trace.finish("error", exc.message)
            raise
        except Exception as exc:
            trace.finish("error", f"{type(exc).__name__}: {exc}")
            raise SynthesisFailed(f"synthesis failed in model '{spec.id}': {exc}") from exc
        finally:
            if own_trace:
                self.complete_tts(trace)

    def complete_tts(self, trace: Trace) -> None:
        """Finish and store a synthesis trace (callers that passed their own trace call this)."""
        trace.finish("ok")
        if trace.status == "ok" and trace.audio_seconds and trace.total_ms is not None:
            trace.rtf = trace.total_ms / 1000.0 / trace.audio_seconds
        self.traces.record(trace)

    async def synthesize_all(self, text: str, **kwargs: Any) -> tuple[FloatArray, int]:
        parts: list[FloatArray] = []
        rate = 0
        async for chunk in self.synthesize(text, **kwargs):
            parts.append(chunk.samples)
            rate = chunk.sample_rate
        if not parts:
            return np.zeros(0, dtype=np.float32), rate or 16000
        return np.concatenate(parts), rate
