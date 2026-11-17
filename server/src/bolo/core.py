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
import shutil
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from bolo import __version__, logs, ort_providers
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
from bolo.engines.blend import Blend, looks_like_blend, model_blends, resolve_blend, unknown_voice_message
from bolo.errors import (
    BoloError,
    ComponentMissing,
    EngineUnavailable,
    GatedModel,
    InvalidRequest,
    ModelNotInstalled,
    Overloaded,
    SynthesisFailed,
)
from bolo.jobs import Job, JobStore
from bolo.lifecycle import LoadedModel, ModelManager
from bolo.llm import AudioLLMClient, LLMClient
from bolo.obs import otel
from bolo.obs.events import EventBus
from bolo.obs.trace import Trace, TraceStore, preview
from bolo.providers import KINDS as PROVIDER_KINDS
from bolo.providers import TERMS, ProviderRegistry, Usage
from bolo.recommend import ALL_LEVELS as FIT_LEVELS
from bolo.recommend import TASKS as RECOMMEND_TASKS
from bolo.recommend import USES as RECOMMEND_USES
from bolo.recommend import Advisor
from bolo.s2s import S2SManager
from bolo.separate import SeparateOptions, SeparationResult, check_request, is_narrowband, plan_chunks, split_long
from bolo.separate import separate as run_separation
from bolo.store import HubClient, ModelStore
from bolo.streaming.vad import EnergyVad, FallbackVad, detect_segments, merge_segments
from bolo.telephony.manager import TelephonyManager
from bolo.text import langdetect, packs, phonemize
from bolo.text.phonemize import strip_overrides
from bolo.text.prepare import prepare as prepare_text
from bolo.text.romanised import status as romanised_status
from bolo.text.segment import chunk_text
from bolo.text.wordlists import MAX_LISTS_PER_REQUEST
from bolo.voices import SavedVoice, VoiceStore

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
        for spec in self.catalog.all():  # presets (`bolo cp`) use their model's files
            if spec.preset_of:
                self.store.links[spec.id] = spec.preset_of
        self.manager = ModelManager(s, self.catalog, self.store, self.engines, self.events, self._auto_pull)
        #: cloud providers (settings and <data_dir>/providers.json), their counters and price tables
        self.providers = ProviderRegistry(s)
        self.manager.providers = self.providers
        remote = self.engines.get("remote")
        if remote is not None and hasattr(remote, "attach"):
            remote.attach(self.providers)
        self._fallbacks = parse_fallbacks(s.fallbacks)
        self._fallback_warned: set[tuple[str, str]] = set()
        self.components = ComponentManager(self)
        #: saved voices (the voice library): cleaned references for the models that clone voices (bolo.voices)
        self.voices = VoiceStore(s.data_dir)
        phonemize.set_preference(s.phonemizer)
        # uccharan's language packs: downloaded by bolo, on first use, as a pull job (bolo.text.packs)
        packs.configure(s.uccharan_auto_download, reporter=self)
        self.llm = LLMClient(s)
        self.audio_llm = AudioLLMClient(s)
        #: the backend of the Realtime speech-to-speech pipeline (built on first use)
        self.s2s = S2SManager(self)
        #: ephemeral Realtime keys: value -> (expiry as epoch seconds, session configuration)
        self.client_secrets: dict[str, tuple[float, dict[str, Any]]] = {}
        #: open WebRTC calls by id (objects from bolo.webrtc; closed in stop())
        self.calls: dict[str, Any] = {}
        #: phone calls: providers, agents, numbers, call records (bolo.telephony)
        self.telephony = TelephonyManager(self)
        #: what this machine is and which models suit it; it recommends and warns, it never restricts
        self.advisor = Advisor(self)
        self.manager.on_loaded = self._on_model_loaded
        self.sip: Any = None  # the inbound SIP endpoint, when the setting sip is on (bolo.telephony.sip)
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
        #: OpenTelemetry export of the request traces (bolo.obs.otel), when switched on and installed
        self.otel = otel.setup(s.otel, self._device_of)
        if self.otel is not None:
            self.traces.listeners.append(self.otel.export)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        self.events.bind(asyncio.get_running_loop())
        self._tasks.append(asyncio.create_task(self._stats_ticker(), name="bolo-stats"))
        self.ready = True
        # look at the machine off the event loop, so the first catalog request does not wait for it
        self.advisor.refresh_soon()
        self._tasks.append(asyncio.create_task(self._check_ort_cuda(), name="bolo-ort-cuda"))
        if self.settings.preload:
            self._tasks.append(asyncio.create_task(self._preload(), name="bolo-preload"))

    async def _check_ort_cuda(self) -> None:
        """Say once, at start, when ``device`` wants CUDA on a machine with an NVIDIA GPU but the installed ONNX
        Runtime has no CUDA provider (``uv tool upgrade`` put plain onnxruntime back over onnxruntime-gpu)."""
        with contextlib.suppress(Exception):
            message = await asyncio.get_running_loop().run_in_executor(None, self.ort_cuda_warning)
            if message:
                self.log("warning", message)

    def ort_cuda_warning(self, nvidia_gpu: bool | None = None, providers: list[str] | None = None) -> str | None:
        """The warning :meth:`_check_ort_cuda` logs (bolo.ort_providers.cuda_dropped), or None."""
        name = str(self.settings.device or "cpu").strip().lower().partition(":")[0]
        if name not in ("cuda", "gpu", "auto"):
            return None
        packages = ort_providers.installed_ort_packages()
        if not packages:
            return None
        if nvidia_gpu is None:
            nvidia_gpu = shutil.which("nvidia-smi") is not None
        if providers is None:
            providers = ort_providers.available_providers()
        return ort_providers.cuda_dropped(self.settings.device, nvidia_gpu, packages, providers)

    def _on_model_loaded(self, spec: ModelSpec) -> None:
        """Say once, in the log, when a model that was just loaded is unlikely to suit this machine."""
        with contextlib.suppress(Exception):
            self.advisor.warn_once(spec)

    def note_fit(self, trace: Trace, spec: ModelSpec) -> None:
        """Record on the request's trace that its model is expected to be slow or not to fit here."""
        found = self.advisor.warnings(spec, wait=False)  # from what is known: a request never waits for this
        if found:
            trace.extra["warnings"] = found

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
            await self.telephony.close()
        with contextlib.suppress(Exception):
            await self.s2s.close()
        await self.manager.close()
        if self.otel is not None:  # send the spans still queued, without waiting long for a collector that is gone
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.to_thread(self.otel.shutdown), 15)
        await asyncio.get_running_loop().run_in_executor(None, self.engines.shutdown)

    def pack_download(self, pack: Any, reason: str, run: Any) -> None:
        """Run one uccharan pack download as a ``pull`` job (bolo.text.packs calls this from a worker thread)."""
        from bolo.text.packs import human_size

        job = self.jobs.create("pull", pack.component)
        job.state, job.bytes_total = "running", pack.size
        self.events.publish_threadsafe("job", {"job": job.to_dict()})
        self.log(
            "info",
            f"downloading uccharan language pack '{pack.name}' ({human_size(pack.size)}, {pack.license}) for {reason}; "
            "uccharan_auto_download = false turns this off",
        )

        def progress(n: int) -> None:
            job.bytes_done += n
            job.progress = min(0.99, job.bytes_done / max(1, pack.size))

        try:
            run(progress)
        except BoloError as exc:
            job.state, job.error = "error", exc.message
            self.events.publish_threadsafe("job", {"job": job.to_dict()})
            self.log("error", f"uccharan language pack '{pack.name}' was not downloaded: {exc.message}")
            raise
        job.state, job.progress = "done", 1.0
        self.events.publish_threadsafe("job", {"job": job.to_dict()})
        self.events.publish_threadsafe("component", {"id": pack.component, "event": "installed"})
        self.log("info", f"uccharan language pack '{pack.name}' is installed (SHA-256 checked)")

    def ensure_language_packs(
        self, spec: ModelSpec, voice_id: str | None, language: str | None, text: str, phonemizer: str | None = None
    ) -> None:
        """Before a voice speaks: the uccharan packs its phonemes (424 when one cannot be had) or its text
        normalisation (best effort) need. Blocking: called in a worker thread."""
        if not packs.table():
            return
        target = self.components.phoneme_target(spec, voice_id)
        if target is not None:
            profile, voice_language = target
            try:
                ph, _improved = phonemize.resolve(profile, voice_language, phonemizer)
            except BoloError:
                return  # no phonemiser: require_components / the engine say so
            tag = phonemize.uccharan_language(profile, voice_language)
            if ph.component == phonemize.UCCHARAN and tag is not None:
                packs.ensure(tag, profile, text)
            return
        from bolo.text.prepare import normalizes_numbers, uccharan_language

        tag = uccharan_language(language)
        if tag is not None and normalizes_numbers(tag):
            packs.ensure(tag, "espeak-ipa", text, required=False)

    def log(self, level: str, message: str) -> None:
        getattr(log, level if level in ("debug", "info", "warning", "error") else "info")(message)
        self.events.publish_threadsafe("log", {"level": level, "message": message})

    def _device_of(self, model: str) -> str | None:
        """Where a loaded model runs (``cpu``, ``cuda:0``, ``remote`` ...), or None when it is not loaded."""
        entry = self.manager.loaded.get(model)
        return str(entry.placement.device) if entry is not None else None

    def _on_trace(self, trace: Trace) -> None:
        self.events.publish_threadsafe("request", {"request": trace.to_dict()})
        entry = self.manager.loaded.get(trace.model)
        if entry is not None and trace.status == "ok" and self.manager.on_gpu(entry):
            # far slower than this model's recent median: is the GPU still there? (bolo.gpu_health)
            self.manager.gpu.note_rtf(trace.model, trace.rtf)

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
            # models that are not installed are downloaded on first use
            "auto_pull": s.auto_pull,
            "providers": [
                {"name": p.name, "kind": p.kind, "configured": p.configured} for p in self.providers.all()
            ],
            "defaults": {
                "stt": s.default_stt, "realtime_stt": s.default_realtime_stt, "tts": s.default_tts,
                "realtime_tts": s.default_realtime_tts, "vad": s.default_vad,
            },
            "wyoming": {"enabled": s.wyoming, "host": s.wyoming_host, "port": s.wyoming_port},
            "components": self.components.status(),
            "capabilities": self.capabilities(),
            "scheduler": {
                "workers": s.workers, "num_parallel": s.parallel(), "threads": s.resolved_threads(),
                "max_batch": s.max_batch, "batch_wait_ms": s.batch_wait_ms, "max_queue": s.max_queue,
            },
            # the file this server also writes its log to (null: standard error only); `bolo logs` reads it
            "log_file": str(logs.ACTIVE_LOG_FILE) if logs.ACTIVE_LOG_FILE else None,
            "health": self.health(),
        }  # fmt: skip

    def health(self) -> dict[str, Any]:
        """Health details (``/healthz``, ``GET /api/v1/info``): "degraded" while the GPU is lost (bolo.gpu_health)."""
        gpu = self.manager.gpu.health
        return {
            "status": "degraded" if gpu.lost else "ok",
            "gpu": {**gpu.to_dict(), "interval_s": self.manager.gpu.interval_s, "checks": self.manager.gpu.checks},
        }

    def capabilities(self) -> dict[str, Any]:
        """What this server can do beyond plain STT/TTS, for clients that want to adapt their UI."""
        s = self.settings
        diar = self.catalog.resolve(s.default_diarization)
        diar_ready = bool(diar and self.manager.is_installed(diar))
        llm = self.llm.describe()
        webrtc = importlib.util.find_spec("aiortc") is not None
        turn = self.catalog.resolve(s.default_turn)
        audio_llm = self.audio_llm.describe()
        s2s = self.s2s.describe()
        uccharan = wordlist_support()
        return {
            # pronunciation word lists (docs/api.md, "Pronunciation word lists"); they need uccharan
            "wordlists": {
                "available": uccharan,
                "endpoint": "/api/v1/wordlists",
                "install": None if uccharan else "pip install 'bolo-speech[uccharan]'",
                "request_fields": ["wordlists", "pronunciations"],
                "formats": ["tsv", "pls", "json"],
                "pin_per_model": True,
                "preview": "/api/v1/wordlists/{name}/preview",
                "max_lists_per_request": MAX_LISTS_PER_REQUEST,
            },
            # romanised text for text-to-speech (docs/api.md, "Romanised text"): what the installed uccharan can do
            "romanised": {
                **romanised_status(),
                "default": s.romanised,
                "request_field": "romanised",
                "pin_per_model": True,
            },
            # cloud providers and models (docs/providers.md)
            "providers": {
                "endpoint": "/api/v1/providers",
                "kinds": list(PROVIDER_KINDS),
                "editable": self.providers.edit_blocked_reason() is None,
                "edit_blocked_reason": self.providers.edit_blocked_reason(),
                "configured": [p.name for p in self.providers.all() if p.configured],
                "adhoc_models": "<provider>:<model>",
                "fallback": True,
                "cost_estimates": any(p.prices for p in self.providers.all()),
            },
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
            "voices": {
                "endpoint": "/api/v1/voices",
                "openai": ["/v1/audio/voices", "/v1/audio/voice_consents"],
                "consent": ["self", "permission"],
                "cloning_models": self.cloning_models(),
                "similarity_model": self.settings.voice_similarity_model or self.settings.default_diarization,
            },
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
                # per pipeline: `configured` (its settings are in place) and `available` (a session can use it now)
                "pipelines": {
                    "cascade": {"configured": llm["configured"], "available": llm["configured"]},
                    "audio-llm": {**audio_llm, "available": bool(audio_llm["configured"])},
                    "speech-to-speech": s2s,
                },
            },
            "chat_completions": {
                "endpoint": "/v1/chat/completions",
                "available": llm["configured"],
                "audio": llm["configured"],
            },
            "llm": {"configured": llm["configured"], "base_url": llm["base_url"], "model": llm["model"]},
            # phone calls (docs/telephony.md)
            "telephony": self.telephony.capability(),
            # hardware detection and model recommendations (docs/recommendations.md): advice, never a restriction
            "recommendations": {
                "hardware": "/api/v1/hardware",
                "endpoint": "/api/v1/recommendations",
                "setup": "/api/v1/recommendations/setup",
                "benchmark": "/api/v1/hardware/benchmark",
                "tasks": list(RECOMMEND_TASKS),
                "uses": list(RECOMMEND_USES),
                "fit_levels": list(FIT_LEVELS),
                "warnings": s.hardware_warnings,
                "restricts": False,
            },
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
        if spec.cloud:
            size, pulling = None, None
        manifest = self.store.manifest(spec.id) if installed and not spec.builtin and not spec.cloud else None
        entry = spec.to_api(
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
        entry["fallback"] = self.fallback_ids(spec)
        # when the files were downloaded (or registered): `bolo list`'s MODIFIED
        stamp = (manifest or {}).get("installed_at")
        entry["modified_at"] = _iso(float(stamp)) if isinstance(stamp, (int, float)) else None
        # how well the model suits this machine (bolo.recommend); advice only, never a restriction
        try:
            entry["fit"] = self.advisor.fit(spec, stale_ok=True).to_dict()
        except Exception:  # pragma: no cover - a fit is never worth failing a catalog request for
            entry["fit"] = None
        if spec.cloud:
            provider = self.providers.find(
                str(spec.options.get("provider") or "") or None, str(spec.options.get("provider_kind") or "") or None
            )
            name = provider.name if provider else str(spec.options.get("provider") or "")
            remote_model = str(spec.options.get("model") or "")
            entry["provider"] = {
                "name": name,
                "kind": provider.kind if provider else str(spec.options.get("provider_kind") or ""),
                "model": remote_model,
                "configured": bool(provider and provider.configured),
                "price": provider.price_for(remote_model) if provider else None,
            }
            entry["adhoc"] = spec.adhoc
            # voices read from the provider's account when the model was loaded (ElevenLabs) are listed from then on
            live = self.manager.loaded.get(spec.id)
            if live is not None and isinstance(live.model, TTSModel) and len(live.model.voices) > len(spec.voices):
                entry["voices"] = [v.to_dict() for v in live.model.voices]
        else:
            entry["provider"] = None
        return entry

    def catalog_api(
        self, task: str | None = None, q: str | None = None, installed: bool | None = None
    ) -> list[dict[str, Any]]:
        out = []
        # entries whose files are not published yet are listed too: `status` is "pending" for them
        for spec in self.catalog.search(task or None, q, include_pending=True):
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
            ttl = self.manager.ttl_for(spec, entry) if spec else None
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
                item.update(self._placement_fields(entry))
                q = entry.queue
                item["queue"] = {
                    "depth": q.depth,
                    "inflight": q.inflight,
                    "workers": q.workers,
                    "num_parallel": q.workers,
                    "max_queue": q.max_queue,
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
            "system": {
                "cpu_percent": self.cpu_percent(),
                "rss_bytes": mem.rss,
                "threads": self._proc.num_threads(),
                **self._machine(),
            },
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
            # phone calls: in progress, total, and by provider and outcome
            "telephony": self.telephony.stats(),
            # cloud providers: requests, errors, time, bytes and estimated cost per provider and model
            "providers": self.providers.stats(),
            "fallbacks": [
                {"requested": a, "answered": b, "count": n} for (a, b), n in self.providers.fallbacks.items()
            ],
        }

    def _machine(self) -> dict[str, Any]:
        """RAM and accelerators, for stats and ps."""
        out: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            vm = psutil.virtual_memory()
            out.update(ram_total_bytes=int(vm.total), ram_available_bytes=int(vm.available))
        gpus: list[dict[str, Any]] = []
        with contextlib.suppress(Exception):
            gpus = [d.to_dict() for d in self.manager.accel.devices()]
        allowed = set(self.manager.allowed_gpus() or [])
        by_gpu = self.manager.gpu_use_by_gpu()
        for g in gpus:
            if g.get("backend") == "cuda":
                g["allowed"] = g["index"] in allowed  # the gpus setting
            g["models_vram_bytes"] = by_gpu.get(int(g["index"]), 0)  # what bolo's loaded models hold on it
        out["gpus"] = gpus
        out["gpu_spread"] = self.settings.gpu_spread
        out["accelerator_source"] = self.manager.accel.source or None
        out["models_vram_bytes"] = self.manager.gpu_in_use()
        out["models_ram_bytes"] = self.manager.memory_in_use()
        out["gpu_health"] = self.manager.gpu.health.to_dict()
        return out

    def _placement_fields(self, entry: Any) -> dict[str, Any]:
        expires_in = None if entry.stopping else self.manager.expires_in(entry)
        ttl = self.manager.ttl_for(entry.spec, entry)
        return {
            "size_vram_bytes": entry.vram_bytes,
            "ram_bytes": entry.ram_bytes,
            "vram_measured": entry.vram_measured,
            "vram_source": entry.vram_source or None,
            "processor": entry.processor(),
            "device": entry.placement.device,
            "placement": entry.placement.to_dict(),
            "keep_alive_s": ttl,
            "expires_in_s": None if expires_in is None else round(expires_in, 1),
            "expires_at": None if expires_in is None else _iso(time.time() + expires_in),
            "stopping": entry.stopping,
            # on a GPU that was lost (bolo.gpu_health): why; null otherwise
            "degraded": entry.degraded,
        }

    def ps(self) -> dict[str, Any]:
        """Loaded models, Ollama style (``GET /api/v1/ps``, ``bolo ps``)."""
        from bolo.keepalive import format_keep_alive, humanize_until

        models: list[dict[str, Any]] = []
        entries = [*self.manager.loaded.values(), *self.manager.stopping.values()]
        for entry in entries:
            fields = self._placement_fields(entry)
            st = self.traces.models.get(entry.id)
            p = entry.placement
            models.append({
                "name": entry.id, "model": entry.id, "task": entry.spec.task,
                "size": entry.memory_bytes, "size_vram": entry.vram_bytes, "size_ram": entry.ram_bytes,
                "processor": fields["processor"], "device": p.device, "split": p.split, "gpus": list(p.gpus),
                "queued": entry.queue.depth, "running": entry.queue.inflight, "num_parallel": entry.queue.workers,
                "max_queue": entry.queue.max_queue,
                "expires_at": fields["expires_at"], "expires_in_s": fields["expires_in_s"],
                "until": humanize_until(fields["expires_in_s"], entry.stopping),
                "keep_alive": format_keep_alive(fields["keep_alive_s"]), "keep_alive_s": fields["keep_alive_s"],
                "active": entry.active, "requests": entry.requests, "idle_s": round(entry.idle_s(), 1),
                "load_ms": round(entry.load_ms, 1), "vram_measured": entry.vram_measured,
                "vram_source": entry.vram_source or None,
                "latency_ms": st.latency_summary() if st else None, "rtf": st.rtf_summary() if st else None,
                "placement": fields["placement"], "degraded": entry.degraded,
                "details": {"engine": entry.spec.engine, "kind": entry.spec.kind, "family": entry.spec.family},
            })  # fmt: skip
        mem = self._proc.memory_info()
        s = self.settings
        out: dict[str, Any] = {
            "models": models,
            "system": {
                "cpu_percent": self.cpu_percent(),
                "rss_bytes": mem.rss,
                "device": s.device,
                **self._machine(),
            },
            "budgets": {
                "memory_budget_bytes": s.memory_budget_mb * 1024 * 1024 or None,
                "gpu_memory_budget_bytes": s.gpu_memory_budget_mb * 1024 * 1024 or None,
                "gpu_reserve_bytes": s.gpu_reserve_mb * 1024 * 1024,
                "gpus": list(s.gpus) or None,
                "gpu_spread": s.gpu_spread,
                "max_loaded_models": s.max_loaded_models or None,
                "keep_alive": format_keep_alive(self.manager.default_keep_alive()),
            },
        }
        if self.s2s.placement is not None:
            out["speech_to_speech"] = {"model": s.s2s_model, "placement": self.s2s.placement.to_dict()}
        return out

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
        if self.catalog.resolve(name) is None:
            self._adhoc(name, task)
        spec = self.manager.resolve(name)
        if spec.task != task:
            raise InvalidRequest(
                f"model '{spec.id}' is a {spec.task} model and cannot be used for {task}", param="model"
            )
        return spec

    @staticmethod
    def interim_decodes(spec: ModelSpec) -> bool:
        """Whether a live utterance may be decoded again and again while it grows (pseudo-streaming partials).

        Each such decode of a cloud model is a billed request, so cloud models only get the final decode unless
        the model option ``interim_decodes`` asks for more. (A provider's live socket is not affected.)"""
        return not spec.cloud or bool(spec.options.get("interim_decodes"))

    def _adhoc(self, name: str, task: str) -> ModelSpec | None:
        """``<provider>:<model>``: any model of a configured provider, without a catalog entry."""
        prefix, sep, model = name.partition(":")
        provider = self.providers.get(prefix) if sep and model else None
        if provider is None or task not in ("stt", "tts"):
            return None
        options: dict[str, Any] = {"provider": provider.name, "model": model, "workers": 4}
        per_model = provider.options.get("models")
        if isinstance(per_model, dict) and isinstance(per_model.get(model), dict):
            options.update(per_model[model])
        terms = TERMS[provider.kind] if provider.is_default_host else provider.base_url
        spec = ModelSpec(
            id=f"{provider.name}:{model}",
            task=task,
            engine="remote",
            kind=f"remote-{task}",
            family=f"{provider.name} (cloud)",
            languages=[],
            sample_rate=int(options.get("sample_rate") or (16000 if task == "stt" else 24000)),
            license=f"terms of {provider.name} (cloud API)",
            license_url=terms,
            cloud=True,
            adhoc=True,
            features=["telephony"],
            description=f"'{model}' at the provider '{provider.name}' ({provider.kind}), asked for by name. "
            "Cloud API: audio or text leaves this server.",
            source_url=provider.base_url,
            options=options,
        )
        return self.catalog.add_adhoc(spec)

    # ------------------------------------------------------------------
    # fallback
    # ------------------------------------------------------------------
    def fallback_ids(self, spec: ModelSpec) -> list[str]:
        """The model's ``fallback`` list: pinned through the API, else the ``fallbacks`` setting, else the catalog."""
        pinned = self.store.model_settings(spec.id).get("fallback")
        chosen = pinned if pinned is not None else self._fallbacks.get(spec.id)
        if chosen is None:
            for alias in spec.aliases:
                if alias in self._fallbacks:
                    chosen = self._fallbacks[alias]
                    break
        if chosen is None:
            chosen = spec.options.get("fallback") or []
        return [str(x) for x in ([chosen] if isinstance(chosen, str) else chosen)]

    def fallback_chain(self, spec: ModelSpec, task: str) -> list[ModelSpec]:
        """The models that answer instead of ``spec`` when it is unavailable, in order (misconfigured names are
        skipped with a warning, never an error for the request)."""
        out: list[ModelSpec] = []
        for name in self.fallback_ids(spec):
            try:
                candidate = self.resolve(name, task)
            except BoloError as exc:
                if (spec.id, name) not in self._fallback_warned:  # once, not on every request
                    self._fallback_warned.add((spec.id, name))
                    log.warning("fallback '%s' of '%s' is not usable: %s", name, spec.id, exc.message)
                continue
            if candidate.id != spec.id and all(candidate.id != c.id for c in out):
                out.append(candidate)
        return out

    def _note_fallback(
        self, trace: Trace, requested: ModelSpec, failed: ModelSpec, nxt: ModelSpec, exc: BoloError
    ) -> None:
        record = trace.extra.setdefault("fallback", {"requested": requested.id, "answered": nxt.id, "attempts": []})
        record["answered"] = nxt.id
        record["attempts"].append({"model": failed.id, "code": exc.code, "message": exc.message[:300]})
        self.providers.note_fallback(requested.id, nxt.id)
        self.log("warning", f"model '{failed.id}' failed ({exc.code}: {exc.message}); falling back to '{nxt.id}'")

    @staticmethod
    def _note_usage(trace: Trace, usage: Usage) -> None:
        """Put what a request cost at a cloud provider on its trace."""
        if not usage.requests:
            return
        trace.extra["provider"] = usage.to_dict()
        if usage.cost is not None:
            trace.extra["cost"] = usage.cost

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
            self.advisor.invalidate()
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
        if not self.store.is_installed(spec.id) and not spec.cloud:
            spec.require_published()
        existing = self.jobs.active_pull(spec.id)
        if existing:
            return existing
        job = self.jobs.create("pull", spec.id)
        if not (spec.builtin or spec.cloud or self.store.is_installed(spec.id)):
            job.warnings = self.advisor.warnings(spec, pulling=True, wait=False)
            for warning in job.warnings:
                self.log("warning", str(warning["message"]))
        if spec.builtin or self.store.is_installed(spec.id):
            job.state, job.progress = "done", 1.0
            return job
        if spec.cloud:  # nothing to download: the provider has to be configured
            try:
                self._require_provider(spec)
                job.state, job.progress = "done", 1.0
            except BoloError as exc:
                job.state, job.error = "error", exc.message
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

    def _require_provider(self, spec: ModelSpec) -> None:
        self.providers.require(
            str(spec.options.get("provider") or "") or None,
            str(spec.options.get("provider_kind") or "") or None,
            model=spec.id,
        )

    async def _auto_pull(self, spec: ModelSpec) -> None:
        if spec.cloud:
            self._require_provider(spec)
            return
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
        if spec.preset_of:  # a preset: its entry and settings go, the model's files stay
            self.catalog.remove_custom(spec.id)
            self.store.links.pop(spec.id, None)
            self.store.set_model_settings(spec.id, dict.fromkeys(self.store.model_settings(spec.id)))
            self.events.publish("model", {"id": spec.id, "event": "removed"})
            return True
        removed = await asyncio.get_running_loop().run_in_executor(None, self.store.remove, spec.id)
        if spec.custom:
            self.catalog.remove_custom(spec.id)
        if removed:
            self.advisor.invalidate()
            self.events.publish("model", {"id": spec.id, "event": "removed"})
        return removed

    def copy_model(self, source: str, destination: str) -> ModelSpec:
        """``bolo cp`` / ``POST /api/v1/models/copy``: a preset, i.e. a new catalog entry ``destination`` that uses
        ``source``'s files and starts with its settings (catalog options and pinned settings), which can then be
        changed on their own. Kept in ``<data_dir>/custom.yaml``; removing it leaves the files alone."""
        from bolo.catalog import CatalogError

        src = self.manager.resolve(source)
        if src.cloud or src.adhoc:
            raise InvalidRequest(f"'{src.id}' is a cloud model; copy it by adding a catalog entry", param="source")
        dst = str(destination or "").strip()
        if self.catalog.resolve(dst) is not None:
            raise InvalidRequest(f"a model called '{dst}' already exists", param="destination")
        base = src.preset_of or src.id
        entry = src.to_dict()
        entry.update(id=dst, aliases=[], preset_of=base, builtin=src.builtin)
        entry["description"] = f"Preset of {base}. " + (src.description or "")
        try:
            spec = ModelSpec.from_dict(entry)
            self.catalog.add_custom(spec)
        except CatalogError as exc:
            raise InvalidRequest(str(exc), param="destination") from exc
        self.store.links[spec.id] = base
        pins = self.store.model_settings(src.id)
        if pins:
            self.store.set_model_settings(spec.id, pins)
        self.events.publish("model", {"id": spec.id, "event": "copied", "source": src.id})
        return spec

    def show_api(self, spec: ModelSpec, license_text: bool = False) -> dict[str, Any]:
        """Everything about one model (``bolo show``, ``GET /api/v1/models/{id}/show``): the catalog entry, its files
        and sizes, the settings that apply to it (and where each comes from), and how fast it is."""
        from bolo.llama_args import resolve_tuning

        entry = self.model_api(spec)
        entry["kind"] = spec.kind
        entry["source"] = spec.source.to_dict()
        entry["options"] = {k: v for k, v in spec.options.items() if k not in ("hardware",)}
        manifest = self.store.manifest(spec.id) if not spec.cloud else None
        entry["path"] = str(self.store.model_dir(spec.id)) if manifest else None
        entry["files"] = [
            {"path": f.get("path"), "size": f.get("size"), "sha256": f.get("sha256")}
            for f in (manifest or {}).get("files", [])
        ]
        entry["presets"] = sorted(s.id for s in self.catalog.all() if s.preset_of == spec.id)
        pins = self.store.model_settings(spec.id)
        effective: dict[str, Any] = {
            "device": self.manager.requested_device(spec),
            "keep_alive_s": self.manager.ttl_for(spec, self.manager.loaded.get(spec.id)),
            "num_parallel": self.manager.num_parallel(spec),
            "max_queue": self.manager.max_queue(spec),
        }
        if spec.engine == "tts-server" and str(spec.options.get("runtime") or "llama-server") == "llama-server":
            effective["llama"] = resolve_tuning(self.settings, spec.options, pins)
        entry["effective"] = effective
        st = self.traces.models.get(spec.id)
        live = self.manager.loaded.get(spec.id)
        entry["speed"] = {
            # the catalog's measurements (and the machine they were taken on)
            "catalog": spec.measured or None,
            # what this server measured since it started: real-time factor and latency
            "server": {"requests": st.requests, "rtf": st.rtf_summary(), "latency_ms": st.latency_summary()}
            if st
            else None,
            "expected_rtf": (entry.get("fit") or {}).get("expected_rtf"),
            "rtf_basis": (entry.get("fit") or {}).get("rtf_basis"),
        }
        entry["loaded_placement"] = live.placement.to_dict() if live is not None else None
        if license_text:
            entry["license_text"] = self.license_text(spec)
        return entry

    def license_text(self, spec: ModelSpec) -> str | None:
        """The licence file shipped with the model's files (LICENSE, LICENCE, COPYING, ...), when there is one."""
        if spec.cloud or not self.store.is_installed(spec.id):
            return None
        root = self.store.model_dir(spec.id)
        for f in (self.store.manifest(spec.id) or {}).get("files", []):
            name = Path(str(f.get("path", ""))).name.lower()
            if name.startswith(("license", "licence", "copying", "notice")) and (root / str(f["path"])).is_file():
                with contextlib.suppress(OSError):
                    return (root / str(f["path"])).read_text(encoding="utf-8", errors="replace")
        return None

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
                entry.last_used = time.perf_counter()
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
        except Exception as exc:  # a runtime error in the model: the session goes on, ending turns on silence
            log.warning("end-of-turn model failed (%s: %s); falling back to silence only", type(exc).__name__, exc)
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
        When the model has a ``fallback`` list and is unavailable (its engine, an overloaded queue, a cloud
        provider's 5xx or timeout), the next model of the list answers; the trace's ``fallback`` says so.
        """
        spec = self.resolve(model, "stt")
        trace = trace or Trace("stt", "internal", spec.id)
        chain = [spec, *self.fallback_chain(spec, "stt")]
        for i, candidate in enumerate(chain):
            more = i + 1 < len(chain)
            try:
                return await self._transcribe_once(
                    samples, sample_rate, candidate, trace, more,
                    language=language, task=task, prompt=prompt, word_timestamps=word_timestamps,
                    temperature=temperature, hotwords=hotwords, vad_filter=vad_filter, on_segment=on_segment,
                    diarize=diarize, separate=separate, record=record,
                )  # fmt: skip
            except BoloError as exc:
                if not (more and fallback_worthy(exc)) or trace.status != "running":
                    raise
                self._note_fallback(trace, spec, candidate, chain[i + 1], exc)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _transcribe_once(
        self,
        samples: FloatArray,
        sample_rate: int,
        spec: ModelSpec,
        trace: Trace,
        more: bool,
        *,
        language: str | None,
        task: str,
        prompt: str | None,
        word_timestamps: bool,
        temperature: float,
        hotwords: str | None,
        vad_filter: bool | None,
        on_segment: Callable[[Segment], Awaitable[None]] | None,
        diarize: DiarizeOptions | None,
        separate: SeparateOptions | None,
        record: bool,
    ) -> Transcript:
        """One model's attempt. With ``more`` (a fallback is left), an availability error leaves the trace open."""
        handed_on = False
        trace.model = spec.id
        self.note_fit(trace, spec)
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
                    # a recogniser that labels speakers itself (Soniox) is asked to, unless the request needs
                    # bolo's diarizer: a named diarization model, known speakers or overlap separation
                    native = (
                        diarize is not None
                        and stt.native_diarization
                        and separate is None
                        and not diarize.model
                        and not diarize.known
                    )
                    if native:
                        trace.extra["diarization_model"] = "provider"
                    if diarize is not None and not native:
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
                        overlapped=overlapped, sources=sources, diarize=native,
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
            if more and fallback_worthy(exc):
                handed_on = True  # the next model of the fallback list takes the request (and the trace)
                raise
            trace.finish("error", exc.message)
            raise
        except Exception as exc:
            trace.finish("error", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if record and not handed_on:
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
        diarize: bool = False,
    ) -> Transcript:
        """Decode ``chunks`` (sample ranges of ``samples``). ``sources[i]``, when given, is the audio to decode
        for chunk ``i`` instead of the mixture (a separated track); it is placed at the chunk's start. ``diarize``
        asks a model with native diarization for speaker labels (kept from its segments when ``speakers`` is
        not given)."""
        sr = stt.sample_rate

        def audio(index: int, a: int, b: int) -> FloatArray:
            source = sources[index] if sources else None
            return source if source is not None else samples[a:b]

        submitted = trace.now()
        pending = [
            asyncio.ensure_future(
                entry.queue.submit(
                    "batch",
                    STTRequest(audio(i, a, b), language, task, prompt, word_timestamps, temperature, hotwords, diarize),
                )
            )
            for i, (a, b) in enumerate(chunks)
        ]
        segments: list[Segment] = []
        words: list[Word] = []
        detected: str | None = language
        first_start: float | None = None
        last_finish = submitted
        usage = Usage()
        try:
            for index, ((a, _b), fut) in enumerate(zip(chunks, pending)):
                result, info = await fut
                assert isinstance(result, STTResult)
                if isinstance(result.usage, Usage):
                    usage.add(result.usage)
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
                        speakers[index] if speakers else seg.speaker,
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
        self._note_usage(trace, usage)
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
        raise InvalidRequest(unknown_voice_message(voice, tts.voices, spec.id), param="voice")

    def blending_models(self, installed_only: bool = True) -> list[str]:
        """Text-to-speech models whose voices can be blended (``features`` has ``voice-blending``)."""
        out = []
        for spec in self.catalog.all():
            if spec.task != "tts" or not model_blends(spec.kind, spec.features) or spec.status == "pending":
                continue
            if not installed_only or self.manager.is_installed(spec):
                out.append(spec.id)
        return out

    def no_blending_message(self, spec: ModelSpec, voice: str) -> str:
        models = self.blending_models() or self.blending_models(installed_only=False)
        return (
            f"'{voice}' is a voice blend, and model '{spec.id}' cannot blend voices. Models that can: "
            f"{', '.join(models) or 'none'}."
        )

    def choose_language(
        self,
        tts: TTSModel,
        spec: ModelSpec,
        text: str,
        voice: str | None,
        *,
        blend: Blend | None = None,
        cloning: bool = False,
    ) -> tuple[dict[str, Any], str | None, str | None]:
        """For a request without a language: ``(note, voice, language)``, the language guessed from ``text`` and
        what follows from it. A model with one language is left alone (``source=model``). Unless the request
        named a voice of the model (an OpenAI voice name mapped onto the model's voices does not count), the first
        voice that speaks the language is chosen (``voice`` in the note); a voice that speaks no particular
        language keeps the voice and tells the model the language. An unsure guess, or a language the model does
        not offer, keeps the current default (``source=default``); a named voice of another language is kept and
        the note says so (``source=voice``)."""
        voices = tts.voices
        voice_langs = [langdetect.base(v.language) for v in voices if v.language]
        offered = list(dict.fromkeys([langdetect.base(x) for x in spec.languages] + voice_langs))
        if len(offered) <= 1:
            return {"source": "model"}, voice, None
        found = langdetect.detect(text)
        note: dict[str, Any] = {"detected": found.language or "unsure", "script": found.script}
        if found.sure:
            note["confidence"] = round(found.confidence, 2)
        elif found.reason:
            note["reason"] = found.reason
        if not found.sure:
            return {**note, "source": "default"}, voice, None
        wanted = [found.language, *found.alternatives]
        named = blend is not None or (
            bool(voice) and (tts.find_voice(voice) is not None or str(voice).lower() not in OPENAI_VOICES)
        )
        if named or cloning:
            current = blend.lead if blend is not None else (tts.find_voice(voice) if voice else None)
            speaks = langdetect.base(current.language) if current is not None and current.language else ""
            if speaks:
                if speaks not in wanted:
                    note["mismatch"] = True
                return {**note, "source": "voice"}, voice, None
            match = next((x for x in wanted if x in offered), None)
            if match is None:
                return {**note, "source": "default", "offered": False}, voice, None
            return {**note, "source": "detected"}, voice, match
        for lang in wanted:
            fits = [v for v in voices if langdetect.base(v.language) == lang]
            if fits:
                index = OPENAI_VOICES.index(str(voice).lower()) if voice else 0
                pick = fits[index % len(fits)]
                if lang != found.language:
                    note["matched"] = lang
                return {**note, "source": "detected", "voice": pick.id}, pick.id, None
        match = next((x for x in wanted if x in offered), None)
        if match is not None and not voice_langs:
            return {**note, "source": "detected"}, voice, match
        return {**note, "source": "default", "offered": False}, voice, None

    def cloning_models(self, installed_only: bool = True) -> list[str]:
        """Text-to-speech models that clone voices (``features`` has ``voice-cloning``), installed ones by
        default (a model on an operator-run server counts as installed)."""
        out = []
        for spec in self.catalog.all():
            if spec.task != "tts" or "voice-cloning" not in spec.features or spec.status == "pending":
                continue
            if not installed_only or self.manager.is_installed(spec):
                out.append(spec.id)
        return out

    def no_cloning_message(self, spec: ModelSpec, voice_id: str) -> str:
        installed = self.cloning_models()
        where = (
            f"Installed models that clone voices: {', '.join(installed)}"
            if installed
            else "No model that clones voices is installed; pull one (for example `bolo pull pocket-tts-en-llamacpp`, "
            "or see `bolo list --all` for models with voice-cloning)"
        )
        return f"'{voice_id}' is a saved voice, and model '{spec.id}' cannot clone voices. {where}."

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
        romanised: bool | str | None = None,
        romanised_strict: bool = True,
        custom_voice: bool = False,
        options: dict[str, Any] | None = None,
        detect_language: bool = False,
    ) -> AsyncGenerator[SynthChunk, None]:
        """Synthesise ``text`` sentence by sentence, yielding audio as soon as it exists.

        A ``voice`` that names a saved voice (:mod:`bolo.voices`) and is not one of the model's own voices is
        spoken from the saved reference by a model that clones voices; ``custom_voice`` (OpenAI's
        ``voice: {"id": ...}``) prefers the saved voice even over a model voice of that name. Without ``model``
        the saved voice's default model answers. ``options`` are model-specific request options
        (``SynthesisExtras.options``).

        A ``voice`` written as a blend (``af_bella(0.7)+af_sky(0.3)``, :mod:`bolo.engines.blend`), or naming a
        saved blend, is spoken by a model that blends voices (Kokoro); its voices are checked first (400 naming
        the near matches.

        ``detect_language``: when the request names no language, the language of a multilingual model is guessed
        from the text (:mod:`bolo.text.langdetect`) and, unless a voice of the model was named, a voice that speaks
        it is chosen; an explicit voice or language is never changed. ``trace.extra["language"]`` records the
        decision (the ``x-bolo-language`` header).

        ``romanised`` (``off`` / ``auto`` / ``on``; None: the voice's default) writes an Indian language typed in
        Latin letters in the voice's script first (bolo.text.romanised). With ``romanised_strict`` false (sessions,
        agents) a conversion that cannot happen is a notice in the trace, never an error.

        This generator records the ``queue``, ``normalize``, ``synthesize_first``
        and ``synthesize_rest`` phases. ``reference_audio`` (any decodable recording) asks a model that clones
        voices to speak in that voice; other models answer 400. A caller that passes its own ``trace``
        owns the ``encode`` phase and ``ttfa`` and must call
        :meth:`complete_tts` when it has delivered the audio.
        """
        own_trace = trace is None
        if trace is None:
            trace = Trace("tts", "internal", model or self.settings.default_tts)
        saved = self.voices.find(voice) if reference_audio is None and voice else None
        if saved is not None and model is None and saved.default_model:
            model = saved.default_model
        spec = self.resolve(model, "tts")
        blend_name: str | None = None
        if saved is not None and saved.is_blend:
            if custom_voice or not any(str(v.get("id", "")).lower() == str(voice).lower() for v in spec.voices):
                blend_name, voice = saved.id, saved.blend
            saved = None  # a blend is spoken by the model's own voices, never cloned
        if (
            saved is not None
            and not custom_voice
            and any(str(v.get("id", "")).lower() == str(voice).lower() for v in spec.voices)
        ):
            saved = None  # the model has a voice of that name: it wins over the library (send {"id": ...} for that)
        chain = [spec, *self.fallback_chain(spec, "tts")]
        if saved is not None or reference_audio is not None:
            # a cloned voice must stay the same person: only models that clone voices can stand in
            chain = [spec, *(c for c in chain[1:] if "voice-cloning" in c.features)]
        try:
            for i, candidate in enumerate(chain):
                more = i + 1 < len(chain)
                started = False
                attempt = self._synthesize_once(
                    text, candidate, trace, more, own_voice=i == 0,
                    voice=voice, speed=speed, language=language, normalize=normalize, phonemizer=phonemizer,
                    reference_audio=reference_audio, reference_text=reference_text, wordlists=wordlists,
                    pronunciations=pronunciations, input_format=input_format, romanised=romanised,
                    romanised_strict=romanised_strict, saved=saved, custom_voice=custom_voice, options=options,
                    blend_name=blend_name, detect_language=detect_language,
                )  # fmt: skip
                try:
                    async for chunk in attempt:
                        started = True
                        yield chunk
                    return
                except BoloError as exc:
                    # once audio was delivered the answer cannot change voice: no fallback after the first chunk
                    if started or not more or not fallback_worthy(exc) or trace.status != "running":
                        raise
                    self._note_fallback(trace, spec, candidate, chain[i + 1], exc)
                finally:
                    # a consumer that stopped listening closes this generator: the attempt has to hear of it
                    # now (it stops the engine and marks the trace), not when it is garbage-collected
                    await attempt.aclose()
        finally:
            if own_trace:
                self.complete_tts(trace)

    async def _synthesize_once(
        self,
        text: str,
        spec: ModelSpec,
        trace: Trace,
        more: bool,
        *,
        own_voice: bool,
        voice: str | None,
        speed: float,
        language: str | None,
        normalize: bool | str,
        phonemizer: str | None,
        reference_audio: bytes | None,
        reference_text: str | None,
        wordlists: list[str] | None,
        pronunciations: list[dict[str, Any]] | None,
        input_format: str,
        romanised: bool | str | None,
        romanised_strict: bool,
        saved: SavedVoice | None = None,
        custom_voice: bool = False,
        options: dict[str, Any] | None = None,
        blend_name: str | None = None,
        detect_language: bool = False,
    ) -> AsyncGenerator[SynthChunk, None]:
        """One model's attempt at :meth:`synthesize`. ``own_voice`` is false for a fallback model: a voice it
        does not have is replaced by its default instead of refused."""
        trace.model = spec.id
        self.note_fit(trace, spec)
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
                if saved is not None and not custom_voice and tts.find_voice(voice) is not None:
                    saved = None  # a voice the server reported, of the same name
                if saved is not None:
                    if not tts.supports_cloning:
                        raise InvalidRequest(self.no_cloning_message(spec, saved.id), param="voice")
                    reference_audio = saved.reference_bytes()
                    reference_text = reference_text or saved.transcript
                    language = language or saved.language
                    voice = None
                    consent = saved.meta.get("consent") or {}
                    trace.extra["voice"] = {"saved": saved.id, "cloned": True, "consent": consent.get("basis")}
                elif reference_audio is not None:
                    trace.extra["voice"] = {"saved": None, "cloned": True}
                blend: Blend | None = None
                if voice and looks_like_blend(voice) and tts.find_voice(voice) is None:
                    try:
                        if not tts.supports_blending:
                            raise InvalidRequest(self.no_blending_message(spec, blend_name or voice), param="voice")
                        blend = resolve_blend(voice, tts.voices, spec.id)
                    except InvalidRequest:
                        if own_voice:
                            raise
                        voice = None  # a fallback model speaks with its default voice
                    else:
                        trace.extra["voice"] = {"saved": blend_name, "cloned": False, **blend.to_dict()}
                language_note: dict[str, Any] | None = None
                if language:
                    language_note = {"source": "request"}
                elif detect_language:
                    language_note, voice, language = self.choose_language(
                        tts, spec, text, voice, blend=blend, cloning=reference_audio is not None
                    )
                try:
                    chosen = blend.lead if blend is not None else self.pick_voice(tts, voice, spec)
                except InvalidRequest:
                    if own_voice:
                        raise
                    voice, chosen = None, tts.find_voice(None)
                    phonemizer = None  # the pin was for the requested model's voice
                engine_voice = blend.key if blend is not None else (chosen.id if chosen else voice)
                phonemizer = self.components.check_phonemizer(spec, chosen.id if chosen else voice, phonemizer)
                lang = language or (
                    chosen.language if chosen and chosen.language else (spec.languages[0] if spec.languages else "en")
                )
                trace.language = lang
                if language_note is not None:
                    trace.extra["language"] = {"language": lang, **language_note}
                extras: SynthesisExtras | None = None
                if reference_audio is not None and not tts.supports_cloning:
                    raise InvalidRequest(
                        f"model '{spec.id}' cannot imitate a voice from reference audio", param="reference_audio"
                    )
                if options:
                    accepted = tts.request_options if tts.accepts_extras else ()
                    unknown = sorted(k for k in options if "*" not in accepted and k not in accepted)
                    if unknown:
                        known = f" (it reads: {', '.join(accepted)})" if accepted else ""
                        raise InvalidRequest(
                            f"model '{spec.id}' does not take the option(s) {', '.join(unknown)}{known}",
                            param="options",
                        )
                if tts.accepts_extras:
                    extras = SynthesisExtras(
                        language=language or (chosen.language if chosen and chosen.language else None),
                        reference_audio=reference_wav(reference_audio) if reference_audio is not None else None,
                        reference_text=reference_text,
                        usage=Usage() if spec.cloud else None,
                        options=dict(options or {}),
                    )
                with trace.phase("normalize"):
                    # uccharan's language packs for this voice, downloaded on first use (bolo.text.packs)
                    if packs.table():
                        await asyncio.to_thread(
                            self.ensure_language_packs, spec, chosen.id if chosen else voice, lang, text, phonemizer
                        )
                    # normaliser, word lists, SSML and pronunciation overrides for this voice (bolo.text.prepare)
                    prepared = prepare_text(
                        self, spec, chosen.id if chosen else voice, text, lang, normalize=normalize,
                        wordlists=wordlists, pronunciations=pronunciations, input_format=input_format,
                        romanised=romanised, romanised_strict=romanised_strict,
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
                            (piece, engine_voice, speed, emit, stop.is_set, phonemizer, extras),
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
                if extras is not None and isinstance(extras.usage, Usage):
                    self._note_usage(trace, extras.usage)
            trace.audio_seconds = total_samples / sample_rate if sample_rate else 0.0
            if total_samples == 0:
                raise EngineUnavailable(f"model '{spec.id}' produced no audio for this text")
        except (asyncio.CancelledError, GeneratorExit):
            trace.audio_seconds = total_samples / sample_rate if sample_rate else 0.0
            trace.finish("cancelled", "client disconnected")
            raise
        except BoloError as exc:
            if more and total_samples == 0 and fallback_worthy(exc):
                raise  # the next model of the fallback list takes the request (and the trace)
            trace.finish("error", exc.message)
            raise
        except Exception as exc:
            trace.finish("error", f"{type(exc).__name__}: {exc}")
            raise SynthesisFailed(f"synthesis failed in model '{spec.id}': {exc}") from exc

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


def _iso(epoch: float) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def fallback_worthy(exc: BoloError) -> bool:
    """Errors that mean "this model cannot answer right now" (not "the request is wrong"): an unavailable
    engine, a full queue, a cloud provider's 5xx, timeout, rate limit or refused key."""
    return isinstance(exc, (EngineUnavailable, Overloaded))


def parse_fallbacks(items: list[str]) -> dict[str, list[str]]:
    """The ``fallbacks`` setting: ``"<model>=<fallback>[,<fallback>...]"`` per entry."""
    out: dict[str, list[str]] = {}
    for item in items:
        model, sep, rest = str(item).partition("=")
        names = [n.strip() for n in rest.replace("|", ",").split(",") if n.strip()]
        if not sep or not model.strip() or not names:
            raise ValueError(f"fallbacks: expected '<model>=<fallback>[,<fallback>...]', got {item!r}")
        out[model.strip()] = names
    return out


def wordlist_support() -> bool:
    from bolo.text.wordlists import uccharan_module

    return uccharan_module() is not None


def fallback_headers(trace: Trace) -> dict[str, str]:
    """Response headers that say which model answered: ``x-bolo-model`` always, ``x-bolo-fallback`` when it was
    not the one asked for, ``x-bolo-estimated-cost`` when the provider's price table knows the model."""
    headers = {"x-bolo-model": trace.model}
    cloned = trace.extra.get("voice")
    if isinstance(cloned, dict) and cloned.get("cloned"):
        headers["x-bolo-cloned"] = "true"
    if isinstance(cloned, dict) and cloned.get("saved"):
        headers["x-bolo-voice"] = str(cloned["saved"])
    if isinstance(cloned, dict) and cloned.get("blend"):
        headers["x-bolo-blend"] = str(cloned["blend"])
    language = language_header(trace)
    if language:
        headers["x-bolo-language"] = language
    record = trace.extra.get("fallback")
    if isinstance(record, dict) and record.get("attempts"):
        first = record["attempts"][0]
        headers["x-bolo-fallback"] = f"{record['requested']} -> {trace.model}; reason={first.get('code')}"
    cost = trace.extra.get("cost")
    if isinstance(cost, dict):
        headers["x-bolo-estimated-cost"] = f"{cost.get('amount')} {cost.get('currency')}"
    return headers


def language_header(trace: Trace) -> str | None:
    """``x-bolo-language``: the language spoken and how it was decided, ``hi-IN; source=detected; detected=hi;
    script=Devanagari; voice=hf_alpha``. ``source`` is ``request`` (the request's ``language``), ``detected``
    (guessed from the text and applied), ``voice`` (a named voice decided; ``mismatch`` when it speaks another
    language than the text), ``default`` (the guess was unsure, ``detected=unsure``, or names a language the
    model does not offer, ``offered=false``) or ``model`` (a model with one language)."""
    note = trace.extra.get("language")
    if not isinstance(note, dict):
        return None
    parts = [str(note.get("language") or trace.language or ""), f"source={note.get('source')}"]
    for key in ("detected", "script", "matched", "voice"):
        if note.get(key):
            parts.append(f"{key}={note[key]}")
    if note.get("mismatch"):
        parts.append("mismatch=true")
    if note.get("offered") is False:
        parts.append("offered=false")
    return "; ".join(parts)


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
