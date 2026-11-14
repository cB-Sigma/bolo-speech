"""The server core: one object that owns every subsystem and implements the
operations (transcribe, synthesise, pull, …) that the protocol front ends call.

HTTP routes, WebSockets, the Wyoming server, the CLI and the benchmark all go
through this class, so they share scheduling, tracing and model lifecycle.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
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
from bolo.audio.decode import decode_audio
from bolo.audio.encode import encode_audio
from bolo.audio.resample import resample
from bolo.catalog import Catalog, ModelSpec
from bolo.components import ComponentManager
from bolo.config import Settings
from bolo.diarize import (
    CONDITIONING,
    DiarizeOptions,
    conditioning_activity,
    consolidate_speakers,
    find_overlaps,
    label_speakers,
    match_known,
    merge_same_speaker,
    merge_turns,
    remap_activity,
    renumber,
)
from bolo.engines import EngineRegistry
from bolo.engines.base import (
    FloatArray,
    Segment,
    SpeakerRequest,
    SpeakerTurn,
    STTModel,
    STTRequest,
    STTResult,
    SynthesisExtras,
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
from bolo.llm import AudioLLMClient, LLMClient
from bolo.obs.events import EventBus
from bolo.obs.trace import Trace, TraceStore, preview
from bolo.s2s import S2SManager
from bolo.separate import SeparateOptions, SeparationResult, check_request, is_narrowband, plan_chunks, split_long
from bolo.separate import separate as run_separation
from bolo.store import HubClient, ModelStore
from bolo.streaming.vad import EnergyVad, FallbackVad, detect_segments, merge_segments
from bolo.text import phonemize
from bolo.text.phonemize import strip_overrides
from bolo.text.prepare import prepare as prepare_text
from bolo.text.segment import chunk_text

log = logging.getLogger("bolo")

#: cosine similarity above which a diarized speaker is taken to be a known speaker
IDENTIFY_THRESHOLD = 0.5

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
    #: what ``separate_overlaps`` did (the response's ``separation`` object), when it was asked for
    separation: dict[str, Any] | None = None


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
        self.engines = engines or EngineRegistry(include_fake=s.fake_engine, settings=s)
        self.catalog = Catalog.load(s.data_dir, include_fake=s.fake_engine)
        token = s.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.store = ModelStore(s.data_dir, HubClient(s.hf_endpoint, token))
        self.manager = ModelManager(s, self.catalog, self.store, self.engines, self.events, self._auto_pull)
        self.components = ComponentManager(self)
        phonemize.set_preference(s.phonemizer)
        self.llm = LLMClient(s)
        self.audio_llm = AudioLLMClient(s)
        #: the backend of the Realtime speech-to-speech pipeline (built on first use)
        self.s2s = S2SManager(self)
        #: ephemeral Realtime keys: value -> (expiry as epoch seconds, session configuration)
        self.client_secrets: dict[str, tuple[float, dict[str, Any]]] = {}
        #: open WebRTC calls by id (objects from bolo.webrtc; closed in stop())
        self.calls: dict[str, Any] = {}
        self.active_streams = 0
        #: requests that asked for separate_overlaps, by outcome, and seconds of audio separated
        self.separation_counts: dict[str, int] = {"applied": 0, "partial": 0, "skipped": 0}
        self.separation_audio_s = 0.0
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
        for call in list(self.calls.values()):
            with contextlib.suppress(Exception):
                await call.close()
        with contextlib.suppress(Exception):
            await self.s2s.close()
        await self.manager.close()
        await asyncio.get_running_loop().run_in_executor(None, self.engines.shutdown)

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
            "defaults": {
                "stt": s.default_stt, "realtime_stt": s.default_realtime_stt, "tts": s.default_tts,
                "realtime_tts": s.default_realtime_tts, "vad": s.default_vad,
            },
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
        webrtc = importlib.util.find_spec("aiortc") is not None
        turn = self.catalog.resolve(s.default_turn)
        return {
            "diarization": {
                "available": bool(diar) and (diar_ready or s.auto_pull),
                "installed": diar_ready,
                "default_model": diar.id if diar else None,
                "http": {"endpoint": "/api/v1/diarize", "transcriptions": ["diarize", "diarized_json"]},
                "websocket": "diarize=true (one speaker per utterance, labelled online)",
                "word_timestamps": True,
                "known_speakers": True,
                "revised_labels_at_end_of_stream": True,
                "speaker_conditioned_models": [
                    m.id
                    for m in self.catalog.all()
                    if m.task == "stt" and "speaker-conditioned" in m.features and m.status != "pending"
                ],
            },
            "separation": self._separation_capability(),
            "realtime": {
                "endpoint": "/v1/realtime",
                "transcription_sessions": True,
                "conversation_sessions": llm["configured"],
                "audio_formats": ["audio/pcm (24 kHz)", "audio/pcmu", "audio/pcma"],
                "reference": "OpenAI Realtime API (GA event names), as typed in openai-python 3.50.0",
                "transports": ["websocket", "webrtc"] if webrtc else ["websocket"],
                "webrtc": {
                    "endpoint": "/v1/realtime/calls",
                    "available": webrtc,
                    "install": None if webrtc else "pip install 'bolo-speech[webrtc]'",
                    "ice_servers": list(s.webrtc_ice_servers),
                },
                "client_secrets": "/v1/realtime/client_secrets",
                "transcript_deltas": True,
                "turn_detection": ["server_vad", "semantic_vad"],
                "semantic_vad": {
                    "model": turn.id if turn else None,
                    "available": self.turn_model() is not None,
                    "installed": bool(turn and self.manager.is_installed(turn)),
                },
                "sip": False,
                "pipeline": s.realtime_pipeline,
                "pipelines": {
                    "cascade": {"available": llm["configured"]},
                    "audio-llm": self.audio_llm.describe(),
                    "speech-to-speech": self.s2s.describe(),
                },
            },
            "chat_completions": {
                "endpoint": "/v1/chat/completions",
                "available": llm["configured"],
                "audio": llm["configured"],
            },
            "llm": {"configured": llm["configured"], "base_url": llm["base_url"], "model": llm["model"]},
        }

    def _separation_capability(self) -> dict[str, Any]:
        s = self.settings
        spec = self.catalog.resolve(s.default_separation)
        installed = bool(spec and self.manager.is_installed(spec))
        engine_ok = bool(spec and self.engines.status(spec.engine).available)
        return {
            "available": bool(spec) and engine_ok and (installed or s.auto_pull),
            "installed": installed,
            "default_model": spec.id if spec else None,
            "http": {
                "transcriptions": "separate_overlaps=true (with diarization)",
                "diarize": "separate_overlaps=true",
            },
            "default_on": False,
            "languages": list(spec.languages) if spec else [],
            "min_sample_rate": 16000,
            "context_s": s.separation_context_s,
            "min_overlap_s": s.separation_min_overlap_s,
        }

    def note_separation(self, result: SeparationResult) -> None:
        self.separation_counts[result.status] = self.separation_counts.get(result.status, 0) + 1
        self.separation_audio_s += result.separated_audio_s

    def model_api(self, spec: ModelSpec) -> dict[str, Any]:
        requires, phonemizers, available = self.components.requirements(spec)
        job = self.jobs.active_pull(spec.id)
        pulling = {"job_id": job.id, "progress": round(job.progress, 4)} if job else None
        installed = self.manager.is_installed(spec)
        size = self.store.size_on_disk(spec.id) if installed and not spec.builtin else None
        return spec.to_api(
            installed=installed,
            loaded=spec.id in self.manager.loaded,
            pulling=pulling,
            size_on_disk=size,
            available=available,
            requires=requires,
            phonemizers=phonemizers,
            voice_phonemizers=self.components.voice_phonemizers(spec),
            settings=self.store.model_settings(spec.id),
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
            "separation": {
                "requests": dict(self.separation_counts),
                "audio_seconds": round(self.separation_audio_s, 2),
            },
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
                "turn": s.default_turn,
                "separation": s.default_separation,
            }
            name = defaults[task]
        elif name == "realtime":
            name = s.default_realtime_stt if task == "stt" else s.default_realtime_tts
        if self.catalog.resolve(name) is None:
            if task == "stt" and name in OPENAI_STT_ALIASES:
                name = s.default_stt
            elif task == "tts" and name in OPENAI_TTS_ALIASES:
                # OpenAI's tts-1 is the low-latency model, tts-1-hd and gpt-4o-mini-tts the quality ones
                name = s.default_realtime_tts if name == "tts-1" else s.default_tts
        spec = self.manager.resolve(name)
        if spec.task != task:
            raise InvalidRequest(
                f"model '{spec.id}' is a {spec.task} model and cannot be used for {task}", param="model"
            )
        return spec

    def realtime_tts(self, voice: str | None = None) -> ModelSpec:
        """The voice for latency-sensitive speech (``default_realtime_tts``), unless ``voice`` names a voice that
        only the quality model (``default_tts``) has: then that model, so a chosen voice is never refused."""
        fast = self.resolve("realtime", "tts")
        if not voice or voice.lower() in OPENAI_VOICES:
            return fast
        known = {str(v["id"]).lower() for v in fast.voices}
        if voice.lower() in known:
            return fast
        quality = self.resolve(None, "tts")
        if voice.lower() in {str(v["id"]).lower() for v in quality.voices}:
            return quality
        return fast

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
        if not self.store.is_installed(spec.id):
            spec.require_published()
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
        spec.require_published()
        if spec.gated and not self.store.hub.token:
            raise GatedModel(
                f"model '{spec.id}' is gated and is never downloaded automatically. Accept its terms at "
                f"{spec.source_url}, set BOLO_HF_TOKEN and run `bolo pull {spec.id}`."
            )
        if not self.settings.auto_pull:
            raise ModelNotInstalled(
                f"model '{spec.id}' is not installed and auto_pull is off; run `bolo pull {spec.id}` "
                f"or POST /api/v1/models/pull"
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

    def turn_model(self) -> ModelSpec | None:
        """The end-of-turn model, if it can be used now (installed, or allowed to be pulled on first use)."""
        spec = self.catalog.resolve(self.settings.default_turn)
        if spec is None or spec.task != "turn" or not self.engines.status(spec.engine).available:
            return None
        if self.manager.is_installed(spec) or (self.settings.auto_pull and not spec.gated):
            return spec
        return None

    async def end_of_turn(self, samples16: FloatArray) -> float | None:
        """Probability that the utterance (16 kHz) is a finished turn, or None without an end-of-turn model."""
        spec = self.turn_model()
        if spec is None:
            return None
        try:
            async with self.manager.use(spec.id) as entry:
                probability, _ = await entry.queue.submit("turn", samples16)
                return float(probability)
        except BoloError as exc:
            log.warning("end-of-turn model unavailable (%s); falling back to silence only", exc.message)
            return None

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
        separate: SeparateOptions | None = None,
        record: bool = True,
    ) -> Transcript:
        """Transcribe a complete recording. Long audio is split at pauses and decoded in batches.

        ``record=False`` keeps the trace out of the request log (interim decodes of a live utterance).
        """
        spec = self.resolve(model, "stt")
        trace = trace or Trace("stt", "internal", spec.id)
        trace.model = spec.id
        trace.audio_seconds = len(samples) / sample_rate if sample_rate else 0.0
        duration = trace.audio_seconds
        separation: SeparationResult | None = None
        conditioned = "speaker-conditioned" in spec.features
        try:
            if separate is not None:
                if diarize is None:
                    raise InvalidRequest(
                        "separate_overlaps needs diarization (diarize=true)", param="separate_overlaps"
                    )
                if conditioned:
                    raise _conditioned_separation(spec.id)
                check_request(sample_rate, separate)
                self.resolve(separate.model, "separation")
            if diarize is not None and conditioned and not diarize.model and spec.options.get("diarization_model"):
                # a speaker-conditioned model brings the diarizer it works best with, unless the request names one
                diarize.model = str(spec.options["diarization_model"])
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
                if diarize is not None and stt.speaker_conditioned:
                    if separate is not None:
                        raise _conditioned_separation(spec.id)
                    mode = str(spec.options.get("conditioning", "turns"))
                    if mode not in CONDITIONING:
                        raise InvalidRequest(f"model '{spec.id}' has an unknown conditioning option {mode!r}")
                    transcript = await self._decode_speakers(
                        entry, stt, samples, diarize, trace, mode,
                        language=language, task=task, prompt=prompt, word_timestamps=word_timestamps,
                        temperature=temperature, on_segment=on_segment,
                    )  # fmt: skip
                else:
                    use_vad = vad_filter if vad_filter is not None else duration > stt.max_chunk_s
                    chunks: list[tuple[int, int]]
                    speakers: list[str] | None = None
                    overlapped: list[bool] | None = None
                    sources: list[FloatArray | None] | None = None
                    if diarize is not None:
                        # Cut at the speaker turns and transcribe each turn: the speaker of every
                        # piece of text is then known without needing word timestamps. Stretches where two
                        # people talk at once are marked on the segments (keep_overlap: see DiarizeOptions).
                        raw = await self._speaker_turns(samples, diarize, trace)
                        overlaps = find_overlaps(raw)
                        diarize.overlaps = overlaps
                        turns = merge_turns(raw, max_len_s=stt.max_chunk_s, keep_overlap=diarize.keep_overlap)
                        chunks = [(int(t.start * sr), min(len(samples), int(t.end * sr))) for t in turns]
                        speakers = [diarize.label(t.speaker) for t in turns]
                        overlapped = [any(o.start < t.end and o.end > t.start for o in overlaps) for t in turns]
                        if overlaps:
                            trace.extra["overlapped_speech_s"] = round(sum(o.end - o.start for o in overlaps), 2)
                        if separate is not None:
                            separation, plan = await self._separated_plan(
                                samples, sr, raw, overlaps, turns, separate, trace,
                                narrowband=sample_rate < 16000, max_chunk_s=stt.max_chunk_s,
                                diarization_model=diarize.model,
                            )  # fmt: skip
                            if plan is not None:
                                chunks = [(int(c.start * sr), min(len(samples), int(c.end * sr))) for c in plan]
                                speakers = [diarize.label(c.speaker) for c in plan]
                                overlapped = [c.overlap for c in plan]
                                sources = [self._track_audio(separation, c, sr) for c in plan]
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
                        overlapped=overlapped, sources=sources,
                    )  # fmt: skip
                transcript.duration = duration
                transcript.model = spec.id
                if separation is not None and diarize is not None:
                    transcript.separation = separation.to_dict(diarize.label, duration)
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
            if record:
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
        overlapped: list[bool] | None = None,
        sources: list[FloatArray | None] | None = None,
    ) -> Transcript:
        """Decode ``chunks`` (sample ranges of ``samples``). ``sources[i]``, when given, is the audio to decode
        for chunk ``i`` instead of the mixture (a separated track); it is placed at the chunk's start."""
        sr = stt.sample_rate

        def audio(index: int, a: int, b: int) -> FloatArray:
            source = sources[index] if sources else None
            return source if source is not None else samples[a:b]

        submitted = trace.now()
        pending = [
            asyncio.ensure_future(
                entry.queue.submit(
                    "batch",
                    STTRequest(audio(i, a, b), language, task, prompt, word_timestamps, temperature, hotwords),
                )
            )
            for i, (a, b) in enumerate(chunks)
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
                        bool(overlapped and overlapped[index]),
                        bool(sources and sources[index] is not None),
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

    async def _decode_speakers(
        self,
        entry: LoadedModel,
        stt: STTModel,
        samples: FloatArray,
        diarize: DiarizeOptions,
        trace: Trace,
        mode: str,
        *,
        language: str | None,
        task: str,
        prompt: str | None,
        word_timestamps: bool,
        temperature: float,
        on_segment: Callable[[Segment], Awaitable[None]] | None,
    ) -> Transcript:
        """Diarization-conditioned recognition: diarize, then decode the whole recording once per speaker.

        The recogniser hears the full mixture and is told per frame who talks (``conditioning_activity``), so
        overlapping speech is transcribed for each speaker instead of being cut at turn boundaries."""
        sr = stt.sample_rate
        samples16 = samples if sr == 16000 else resample(samples, sr, 16000)
        turns = await self._speaker_turns(samples16, diarize, trace)
        overlaps = find_overlaps(turns)
        diarize.overlaps = overlaps
        if overlaps:
            trace.extra["overlapped_speech_s"] = round(sum(o.end - o.start for o in overlaps), 2)
        duration = len(samples) / sr
        if not turns:
            return Transcript("", language, 0.0)
        activity, used = conditioning_activity(turns, diarize.activity, duration, mode)
        trace.extra["conditioning"] = used
        request = SpeakerRequest(samples, activity, language, task, prompt, word_timestamps, temperature)
        submitted = trace.now()
        results, info = await entry.queue.submit("speakers", request)
        trace.note_batch(info.batch_size)
        trace.add_phase("queue", submitted, info.started)
        trace.add_phase("decode", info.started, info.finished)
        self.traces.observe_latency(entry.id, "stt", queue_ms=(info.started - submitted) * 1000.0)
        segments: list[Segment] = []
        for k, result in enumerate(results):
            label = diarize.label(k)
            pieces = result.segments or ([Segment(0.0, duration, result.text)] if result.text.strip() else [])
            for seg in pieces:
                text = seg.text.strip()
                if not text:
                    continue
                overlap = any(o.start < seg.end and o.end > seg.start and k in o.speakers for o in overlaps)
                segments.append(
                    Segment(
                        seg.start, seg.end, text, list(seg.words), seg.avg_logprob, seg.no_speech_prob, label, overlap
                    )
                )
        segments.sort(key=lambda seg: (seg.start, seg.end))
        if on_segment is not None:
            for seg in segments:
                trace.mark_first_partial()
                await on_segment(seg)
        with trace.phase("finalize"):
            text = " ".join(seg.text for seg in segments).strip()
            words = sorted((w for seg in segments for w in seg.words), key=lambda w: w.t0)
        detected = language or next((r.language for r in results if r.language), None)
        return Transcript(text, detected, 0.0, segments, words)

    async def _separated_plan(
        self,
        samples: FloatArray,
        sr: int,
        raw: list[SpeakerTurn],
        overlaps: list[Any],
        turns: list[SpeakerTurn],
        opts: SeparateOptions,
        trace: Trace,
        *,
        narrowband: bool,
        max_chunk_s: float,
        diarization_model: str | None = None,
    ) -> tuple[SeparationResult, list[Any] | None]:
        """Run overlap separation; return its result and the chunks to decode (None: decode as without it)."""
        samples16 = samples if sr == 16000 else resample(samples, sr, 16000)
        narrowband = narrowband or is_narrowband(samples16)
        result = await run_separation(
            self, samples16, raw, overlaps, opts, trace, narrowband=narrowband, diarization_model=diarization_model
        )
        opts.result = result
        if not result.tracks:
            return result, None
        plan = split_long(plan_chunks(turns, raw, result.windows, overlaps), max_chunk_s)
        return result, plan

    @staticmethod
    def _track_audio(result: SeparationResult, chunk: Any, sr: int) -> FloatArray | None:
        """The part of a separated track that ``chunk`` covers, at the recogniser's rate (None: the mixture)."""
        if chunk.track is None:
            return None
        window = result.windows[chunk.track[0]]
        track = result.tracks[chunk.track]
        a = max(0, round((chunk.start - window.start) * 16000))
        b = min(len(track), round((chunk.end - window.start) * 16000))
        piece = np.ascontiguousarray(track[a:b], dtype=np.float32)
        return piece if sr == 16000 else resample(piece, 16000, sr)

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
            result, _ = await entry.queue.submit("diarize_detailed", (samples16, opts.num_speakers, opts.threshold))
            turns, activity = result.turns, result.activity
            found = len({t.speaker for t in turns})
            forced = None
            if not opts.num_speakers:
                if opts.max_speakers and found > opts.max_speakers:
                    forced = opts.max_speakers
                elif opts.min_speakers and found < opts.min_speakers and activity is None:
                    # an end-to-end model (activity given) cannot be asked for more speakers than it hears
                    forced = opts.min_speakers
            if forced:
                result, _ = await entry.queue.submit("diarize_detailed", (samples16, forced, opts.threshold))
                turns, activity = result.turns, result.activity
            order: dict[int, int] = {}
            for t in sorted(turns, key=lambda t: t.start):
                order.setdefault(t.speaker, len(order))
            if activity is not None and any(a != b for a, b in order.items()):
                activity = remap_activity(activity, order)
            turns = renumber(turns)
            trace.add_phase("diarize", start, trace.now())
            voices: dict[int, FloatArray] | None = None
            s = self.settings
            model_default = bool(getattr(entry.model, "consolidate_default", True))
            wanted = opts.consolidate if opts.consolidate is not None else model_default
            repair = (
                wanted
                and not opts.num_speakers
                and (s.diarization_min_speaker_s > 0 or s.diarization_merge_similarity > 0)
            )
            if repair and len({t.speaker for t in turns}) > max(1, opts.min_speakers or 1):
                with trace.phase("consolidate"):
                    voices = await self._voice_prints(entry, samples16, turns)
                    mapping = consolidate_speakers(
                        turns,
                        voices,
                        min_speech_s=s.diarization_min_speaker_s,
                        merge_similarity=s.diarization_merge_similarity if s.diarization_merge_similarity > 0 else 2.0,
                        min_speakers=opts.min_speakers or 1,
                    )
                    if any(a != b for a, b in mapping.items()):
                        trace.extra["speakers_before_consolidation"] = len(mapping)
                        merged = merge_same_speaker([SpeakerTurn(t.start, t.end, mapping[t.speaker]) for t in turns])
                        if activity is not None:
                            first: dict[int, int] = {}
                            for t in merged:
                                first.setdefault(t.speaker, len(first))
                            activity = remap_activity(activity, {a: first[b] for a, b in mapping.items()})
                        turns = renumber(merged)
                        voices = None  # the clusters changed
            named: dict[int, str] = {}
            if opts.known and turns:
                with trace.phase("identify"):
                    named = await self._identify(entry, samples16, turns, opts, voices)
        opts.names = label_speakers(sorted({t.speaker for t in turns}), named)
        opts.activity = activity
        trace.extra["diarization_model"] = spec.id
        trace.extra["speakers"] = len({t.speaker for t in turns})
        if opts.known:
            trace.extra["known_speakers"] = {k.name: opts.similarities.get(k.name) for k in opts.known}
        return turns

    async def _voice_prints(
        self, entry: LoadedModel, samples16: FloatArray, turns: list[SpeakerTurn]
    ) -> dict[int, FloatArray]:
        """One speaker embedding per diarized speaker, from that speaker's longest turns."""
        sr = 16000
        clusters: dict[int, FloatArray] = {}
        for speaker in sorted({t.speaker for t in turns}):
            parts: list[FloatArray] = []
            total = 0.0
            for t in sorted((t for t in turns if t.speaker == speaker), key=lambda t: t.start - t.end):
                parts.append(samples16[int(t.start * sr) : int(t.end * sr)])
                total += t.end - t.start
                if total >= 30.0:  # the longest turns, up to half a minute, are plenty for a voice print
                    break
            audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            if len(audio) >= int(0.5 * sr):
                clusters[speaker], _ = await entry.queue.submit("embed", audio)
        return clusters

    async def _identify(
        self,
        entry: LoadedModel,
        samples16: FloatArray,
        turns: list[SpeakerTurn],
        opts: DiarizeOptions,
        voices: dict[int, FloatArray] | None = None,
    ) -> dict[int, str]:
        """Compare each speaker's voice with the reference recordings of the known speakers."""
        clusters = voices if voices is not None else await self._voice_prints(entry, samples16, turns)
        references: list[tuple[str, FloatArray]] = []
        for known in opts.known:
            vector, _ = await entry.queue.submit("embed", known.samples)
            references.append((known.name, vector))
        threshold = opts.identify_threshold if opts.identify_threshold is not None else IDENTIFY_THRESHOLD
        named, opts.similarities = match_known(clusters, references, threshold)
        return named

    async def diarize(
        self,
        samples: FloatArray,
        sample_rate: int,
        opts: DiarizeOptions | None = None,
        trace: Trace | None = None,
        separate: SeparateOptions | None = None,
    ) -> list[SpeakerTurn]:
        """Who spoke when. Returns speaker turns sorted by time, speakers numbered by first appearance.

        With ``separate``, overlapped stretches are also separated into one track per speaker
        (``separate.result``)."""
        opts = opts or DiarizeOptions()
        spec = self.resolve(opts.model, "diarization")
        trace = trace or Trace("diarization", "internal", spec.id)
        trace.model = spec.id
        trace.audio_seconds = len(samples) / sample_rate if sample_rate else 0.0
        try:
            if separate is not None:
                check_request(sample_rate, separate)
                self.resolve(separate.model, "separation")
            with trace.phase("resample"):
                samples16 = await asyncio.to_thread(resample, samples, sample_rate, 16000)
            samples16 = np.ascontiguousarray(samples16, dtype=np.float32)
            turns = await self._speaker_turns(samples16, opts, trace)
            if separate is not None:
                overlaps = find_overlaps(turns)
                narrowband = sample_rate < 16000 or is_narrowband(samples16)
                separate.result = await run_separation(
                    self, samples16, turns, overlaps, separate, trace, narrowband=narrowband,
                    diarization_model=opts.model,
                )  # fmt: skip
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

    def require_components(self, spec: ModelSpec, phonemizer: str | None = None) -> None:
        """Fail fast when a model needs an optional component that is not installed."""
        reason = self.components.missing_reason(spec, phonemizer)
        if reason:
            raise ComponentMissing(f"model '{spec.id}' cannot be used yet: " + reason)

    async def synthesize(
        self,
        text: str,
        *,
        model: str | None = None,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
        normalize: bool | str = True,
        trace: Trace | None = None,
        phonemizer: str | None = None,
        reference_audio: bytes | None = None,
        reference_text: str | None = None,
        wordlists: list[str] | None = None,
        pronunciations: list[dict[str, Any]] | None = None,
        input_format: str = "text",
    ) -> AsyncGenerator[SynthChunk, None]:
        """Synthesise ``text`` sentence by sentence, yielding audio as soon as it exists.

        This generator records the ``queue``, ``normalize``, ``synthesize_first``
        and ``synthesize_rest`` phases. ``reference_audio`` (any decodable recording) asks a model that clones
        voices to speak in that voice; other models answer 400. A caller that passes its own ``trace``
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
            self.require_components(spec, phonemizer)
            async with self.manager.use(spec.id) as entry:
                tts = entry.model
                assert isinstance(tts, TTSModel)
                sample_rate = tts.sample_rate
                chosen = self.pick_voice(tts, voice, spec)
                phonemizer = self.components.check_phonemizer(spec, chosen.id if chosen else voice, phonemizer)
                lang = language or (
                    chosen.language if chosen and chosen.language else (spec.languages[0] if spec.languages else "en")
                )
                trace.language = lang
                extras: SynthesisExtras | None = None
                if reference_audio is not None and not tts.supports_cloning:
                    raise InvalidRequest(
                        f"model '{spec.id}' cannot imitate a voice from reference audio", param="reference_audio"
                    )
                if tts.accepts_extras:
                    extras = SynthesisExtras(
                        language=language or (chosen.language if chosen and chosen.language else None),
                        reference_audio=reference_wav(reference_audio) if reference_audio is not None else None,
                        reference_text=reference_text,
                    )
                with trace.phase("normalize"):
                    # normaliser, word lists, SSML and pronunciation overrides for this voice (bolo.text.prepare)
                    prepared = prepare_text(
                        self, spec, chosen.id if chosen else voice, text, lang, normalize=normalize,
                        wordlists=wordlists, pronunciations=pronunciations, input_format=input_format,
                    )  # fmt: skip
                    trace.extra.update(prepared.trace_extra())
                    clean = prepared.text
                    clean = tts.fit_text(clean, lang)  # closed character sets: spell digits out, or refuse
                    # a slow voice gets a shorter first chunk (catalog option first_chunk_chars), so audio starts early
                    first = int(spec.options.get("first_chunk_chars", 120))
                    pieces = chunk_text(clean, first_max_chars=first) or [clean]
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
                        entry.queue.submit(
                            "tts",
                            (piece, chosen.id if chosen else voice, speed, emit, stop.is_set, phonemizer, extras),
                        )
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
                            shown = strip_overrides(piece) if first_of_text else ""
                            yield SynthChunk(chunk, sample_rate, seq, shown, first_of_text)
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


REFERENCE_MAX_S = 30.0


def reference_wav(data: bytes) -> bytes:
    """A voice reference as the servers take it: mono PCM16 WAV, at most ``REFERENCE_MAX_S`` seconds."""
    if not data:
        raise InvalidRequest("the reference audio is empty", param="reference_audio")
    try:
        samples, rate = decode_audio(data)
    except BoloError as exc:
        raise InvalidRequest(
            f"the reference audio could not be decoded: {exc.message}", param="reference_audio"
        ) from exc
    if len(samples) < rate * 0.5:
        raise InvalidRequest("the reference audio is shorter than half a second", param="reference_audio")
    samples = samples[: int(rate * REFERENCE_MAX_S)]
    return encode_audio(samples, rate, "wav")


def _conditioned_separation(model_id: str) -> InvalidRequest:
    return InvalidRequest(
        f"model '{model_id}' transcribes overlapping speech for each speaker itself; separate_overlaps cannot be "
        "combined with it",
        param="separate_overlaps",
    )
