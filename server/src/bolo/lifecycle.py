"""Model lifecycle: lazy loading, idle unloading, memory budget and LRU eviction."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import psutil

from bolo.catalog import Catalog, ModelSpec
from bolo.config import Settings
from bolo.engines import EngineRegistry
from bolo.engines.base import (
    DiarizationModel,
    LoadOptions,
    Model,
    SeparationModel,
    STTModel,
    STTStream,
    TTSModel,
    TurnModel,
)
from bolo.errors import EngineUnavailable, ModelNotFound, ModelNotInstalled, Overloaded
from bolo.obs.events import EventBus
from bolo.scheduler import Handler, ModelQueue
from bolo.store import ModelStore

log = logging.getLogger("bolo.lifecycle")


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: Model
    queue: ModelQueue
    memory_bytes: int
    load_ms: float
    loaded_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    active: int = 0
    requests: int = 0

    @property
    def id(self) -> str:
        return self.spec.id

    def idle_s(self) -> float:
        return 0.0 if self.active else time.monotonic() - self.last_used


def _tts_handler(model: TTSModel) -> Handler:
    def run(payloads: list[Any]) -> list[Any]:
        out: list[Any] = []
        for text, voice, speed, emit, cancelled, *extra in payloads:
            n = 0
            phonemizer = extra[0] if extra else None
            kwargs = {"phonemizer": phonemizer} if phonemizer and model.accepts_phonemizer else {}
            for chunk in model.synthesize(text, voice, speed, cancelled, **kwargs):
                emit(chunk)
                n += len(chunk)
                if cancelled():
                    break
            out.append(n)
        return out

    return run


def _stream_handler(model: STTModel) -> Handler:
    def run(streams: list[Any]) -> list[Any]:
        typed: list[STTStream] = list(streams)
        model.decode_streams(typed)
        return [(s.result(), s.is_endpoint()) for s in typed]

    return run


class ModelManager:
    def __init__(
        self,
        settings: Settings,
        catalog: Catalog,
        store: ModelStore,
        engines: EngineRegistry,
        events: EventBus,
        ensure_installed: Callable[[ModelSpec], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.store = store
        self.engines = engines
        self.events = events
        self.loaded: dict[str, LoadedModel] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._reaper: asyncio.Task[None] | None = None
        self._ensure_installed = ensure_installed
        self.evictions = 0

    # -- helpers ----------------------------------------------------------
    def resolve(self, name: str) -> ModelSpec:
        spec = self.catalog.resolve(name)
        if spec is None:
            raise ModelNotFound(
                f"The model '{name}' does not exist. Use GET /v1/registry (or `bolo list --all`) to see the catalog."
            )
        return spec

    def is_installed(self, spec: ModelSpec) -> bool:
        return spec.builtin or self.store.is_installed(spec.id)

    def ttl_for(self, spec: ModelSpec) -> float:
        s = self.settings
        ttls = {
            "stt": s.stt_ttl_s,
            "tts": s.tts_ttl_s,
            "vad": s.vad_ttl_s,
            "diarization": s.diarization_ttl_s,
            "separation": s.diarization_ttl_s,
        }
        return ttls.get(spec.task, s.stt_ttl_s)

    def memory_in_use(self) -> int:
        return sum(m.memory_bytes for m in self.loaded.values())

    # -- loading ----------------------------------------------------------
    def _build_queue(self, spec: ModelSpec, model: Model) -> ModelQueue:
        s = self.settings
        handlers: dict[str, Handler] = {}
        max_batch: dict[str, int] = {}
        if isinstance(model, STTModel):
            handlers["batch"] = model.transcribe
            max_batch["batch"] = max(1, min(model.max_batch, s.max_batch))
            if model.streaming:
                handlers["stream"] = _stream_handler(model)
                max_batch["stream"] = 1 if s.max_batch <= 1 else s.max_batch * 4
        elif isinstance(model, TTSModel):
            handlers["tts"] = _tts_handler(model)
            max_batch["tts"] = 1
        elif isinstance(model, DiarizationModel):
            diarizer = model
            handlers["diarize"] = lambda items: [diarizer.diarize(*item) for item in items]
            handlers["diarize_detailed"] = lambda items: [diarizer.diarize_detailed(*item) for item in items]
            handlers["embed"] = lambda items: [diarizer.embed(item) for item in items]
            max_batch["diarize"] = 1
            max_batch["diarize_detailed"] = 1
            max_batch["embed"] = 8
        elif isinstance(model, SeparationModel):
            separator = model
            handlers["enrol"] = lambda items: [_enrol(separator, item) for item in items]
            handlers["extract"] = lambda items: [separator.extract(*item) for item in items]
            handlers["separate"] = lambda items: [separator.separate(item) for item in items]
            max_batch["enrol"] = 1
            max_batch["extract"] = 1
            max_batch["separate"] = 1
        elif isinstance(model, TurnModel):
            turn = model
            handlers["turn"] = lambda items: [turn.probability(item) for item in items]
            max_batch["turn"] = 1
        return ModelQueue(
            spec.id,
            handlers,
            workers=int(spec.options.get("workers", s.workers)),
            max_batch=max_batch,
            max_wait_ms=s.batch_wait_ms,
            max_queue=s.max_queue,
            queue_timeout_s=s.queue_timeout_s,
        )

    def _load_blocking(self, spec: ModelSpec) -> tuple[Model, int, float]:
        engine = self.engines.get(spec.engine)
        if engine is None:
            raise EngineUnavailable(f"model '{spec.id}' needs the unknown engine '{spec.engine}'")
        status = engine.status()
        if not status.available:
            raise EngineUnavailable(
                f"model '{spec.id}' needs the '{spec.engine}' engine, which is not available: {status.reason}"
            )
        self._probe(spec)
        proc = psutil.Process()
        before = proc.memory_info().rss
        t0 = time.perf_counter()
        opts = LoadOptions(
            threads=int(spec.options.get("threads", self.settings.resolved_threads())),
            device=self.settings.device,
            options={
                "endpoint_silence_s": self.settings.endpoint_silence_ms / 1000.0,
                "max_utterance_s": self.settings.max_utterance_s,
            },
        )
        model_opts = dict(spec.options)
        if spec.voices:
            model_opts.setdefault("voices", spec.voices)
        if len(spec.languages) == 1:
            model_opts.setdefault("language", spec.languages[0])
        if spec.engine.startswith("plugin:"):
            model_opts.setdefault("model_id", spec.id)
        model = engine.load(spec.kind, spec.task, self.store.model_dir(spec.id), model_opts, opts)
        if isinstance(model, TTSModel):
            model.warm_up()
        load_ms = (time.perf_counter() - t0) * 1000.0
        grown = max(0, proc.memory_info().rss - before)
        # An estimate. RSS growth during the load reads low, sometimes zero, when the allocator reuses memory
        # that an unloaded model gave back, so it is never taken to be less than the model's files on disk.
        memory = max(grown, self.store.size_on_disk(spec.id))
        return model, memory, load_ms

    def _probe(self, spec: ModelSpec) -> None:
        """First load of an installed model happens in a child process (see ``bolo._probe``)."""
        if spec.builtin or not self.settings.probe_models or spec.engine.startswith("plugin:"):
            return  # plugins run in their own process already
        manifest = self.store.manifest(spec.id)
        engine_version = self.engines.status(spec.engine).version
        if manifest is None or manifest.get("probed") == engine_version:
            return
        cmd = [
            sys.executable,
            "-m",
            "bolo._probe",
            str(self.settings.data_dir),
            spec.id,
            str(self.settings.resolved_threads()),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        except subprocess.TimeoutExpired as exc:
            raise EngineUnavailable(f"model '{spec.id}' did not load within 10 minutes in the probe process") from exc
        if result.returncode != 0:
            tail = (result.stderr or result.stdout).decode(errors="replace").strip().splitlines()[-6:]
            raise EngineUnavailable(
                f"model '{spec.id}' cannot be loaded by {spec.engine} {engine_version or ''} "
                f"(probe exited with {result.returncode}): " + " | ".join(tail)
            )
        manifest["probed"] = engine_version
        path = self.store.model_dir(spec.id) / "manifest.json"
        with contextlib.suppress(OSError):
            path.write_text(json.dumps(manifest, indent=2))

    async def _make_room(self, spec: ModelSpec) -> None:
        s = self.settings
        budget = s.memory_budget_mb * 1024 * 1024
        need = int((self.store.size_on_disk(spec.id) or spec.size_bytes) * 1.2)

        def over() -> bool:
            too_many = bool(s.max_loaded_models) and len(self.loaded) >= s.max_loaded_models
            too_big = bool(budget) and self.memory_in_use() + need > budget
            return too_many or too_big

        while over():
            idle = sorted((m for m in self.loaded.values() if m.active == 0), key=lambda m: m.last_used)
            if not idle:
                if not self.loaded:
                    log.warning("model %s alone exceeds the memory budget; loading it anyway", spec.id)
                    return
                raise Overloaded(
                    f"cannot load '{spec.id}': the memory budget is exhausted and every loaded model is in use",
                    retry_after_s=2.0,
                )
            victim = idle[0]
            log.info("evicting %s to make room for %s", victim.id, spec.id)
            self.evictions += 1
            await self.unload(victim.id, reason="evicted")

    async def load(self, name: str) -> LoadedModel:
        spec = self.resolve(name)
        existing = self.loaded.get(spec.id)
        if existing is not None:
            return existing
        lock = self._locks.setdefault(spec.id, asyncio.Lock())
        async with lock:
            existing = self.loaded.get(spec.id)
            if existing is not None:
                return existing
            status = self.engines.status(spec.engine)
            if not status.available:  # say so before downloading anything
                raise EngineUnavailable(
                    f"model '{spec.id}' needs the '{spec.engine}' engine, which is not available: {status.reason}"
                )
            if not self.is_installed(spec):
                if self._ensure_installed is None:
                    raise ModelNotInstalled(f"model '{spec.id}' is not installed; run `bolo pull {spec.id}`")
                await self._ensure_installed(spec)
            await self._make_room(spec)
            loop = asyncio.get_running_loop()
            model, memory, load_ms = await loop.run_in_executor(None, self._load_blocking, spec)
            entry = LoadedModel(spec, model, self._build_queue(spec, model), memory, load_ms)
            self.loaded[spec.id] = entry
            log.info("loaded %s in %.0f ms (~%.0f MB)", spec.id, load_ms, memory / 1e6)
            self.events.publish("model", {"id": spec.id, "event": "loaded"})
            self._ensure_reaper()
            return entry

    async def unload(self, name: str, reason: str = "requested") -> bool:
        spec = self.catalog.resolve(name)
        if spec is None or spec.id not in self.loaded:
            return False
        entry = self.loaded.pop(spec.id)
        await entry.queue.close()
        with contextlib.suppress(Exception):
            entry.model.close()
        log.info("unloaded %s (%s)", spec.id, reason)
        self.events.publish("model", {"id": spec.id, "event": "unloaded", "reason": reason})
        return True

    @contextlib.asynccontextmanager
    async def use(self, name: str) -> AsyncIterator[LoadedModel]:
        """Borrow a model: loads it if needed and keeps it from being unloaded."""
        entry = await self.load(name)
        entry.active += 1
        entry.requests += 1
        entry.last_used = time.monotonic()
        try:
            yield entry
        finally:
            entry.active -= 1
            entry.last_used = time.monotonic()

    # -- idle unloading ---------------------------------------------------
    def _ensure_reaper(self) -> None:
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.create_task(self._reap_loop(), name="bolo-reaper")

    async def reap_once(self) -> list[str]:
        gone: list[str] = []
        for entry in list(self.loaded.values()):
            ttl = self.ttl_for(entry.spec)
            if ttl >= 0 and entry.active == 0 and entry.idle_s() >= ttl:
                await self.unload(entry.id, reason="idle")
                gone.append(entry.id)
        return gone

    async def _reap_loop(self) -> None:
        while self.loaded:
            ttls = [t for t in (self.ttl_for(m.spec) for m in self.loaded.values()) if t >= 0]
            await asyncio.sleep(max(0.05, min([1.0, *[t / 4 for t in ttls]])))
            with contextlib.suppress(Exception):
                await self.reap_once()

    async def close(self) -> None:
        if self._reaper:
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reaper
        for model_id in list(self.loaded):
            await self.unload(model_id, reason="shutdown")


def _enrol(separator: SeparationModel, item: tuple[str, Any]) -> bool:
    separator.enrol(*item)
    return True
