"""Model lifecycle: lazy loading, keep-alive unloading, memory budgets, LRU eviction and GPU/CPU placement.

Loading a model first places it (:mod:`bolo.placement`): on the accelerator when it fits in the free accelerator
memory minus ``gpu_reserve_mb`` and in ``gpu_memory_budget_mb``, after unloading idle GPU-resident models (least
recently used first) when that makes it fit; otherwise split (llama.cpp layers, ONNX Runtime graphs, a PyTorch
plugin's device map) or on the CPU, with a notice. Then it makes room in RAM (``memory_budget_mb``,
``max_loaded_models``), loads, and measures what the load took: the growth of this process and the processes it
started (RSS), and of the accelerator memory they hold (:mod:`bolo.accel`).
"""

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

from bolo.accel import Accelerators, our_pids
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
from bolo.keepalive import REQUEST_KEEP_ALIVE, parse_keep_alive
from bolo.llama_args import has_gpu_flags
from bolo.obs.events import EventBus
from bolo.placement import (
    MB,
    LayerInfo,
    Placement,
    best_gpu,
    family,
    gguf_block_count,
    operator_layers,
    plan,
    plan_spread,
    processor,
    split_mode,
)
from bolo.providers import ProviderRegistry
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
    #: of ``memory_bytes``, the part in accelerator memory
    vram_bytes: int = 0
    #: ``vram_bytes`` was measured (NVML, nvidia-smi, the plugin's report), not taken from the plan
    vram_measured: bool = False
    placement: Placement = field(default_factory=Placement)
    #: set by a request's or a load's ``keep_alive`` (seconds; -1: forever); None: the settings decide
    keep_alive_s: float | None = None
    stopping: bool = False

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def ram_bytes(self) -> int:
        return max(0, self.memory_bytes - self.vram_bytes)

    def processor(self) -> str:
        """Where the weights are, as Ollama's PROCESSOR column says it. A model placed whole on one side reads
        "100% GPU" / "100% CPU" (the runtime's own host memory of a GPU model is in SIZE, not here); a split model
        reads by bytes: measured when the accelerator memory was measured, else as planned."""
        p = self.placement
        if p.device in ("remote", "cloud"):
            return p.device
        if p.gpu_bytes <= 0 or self.vram_bytes <= 0:
            return "100% CPU"
        if p.cpu_bytes <= 0:
            return "100% GPU"
        if self.vram_measured:
            return processor(self.vram_bytes, self.memory_bytes)
        return p.processor()

    def idle_s(self) -> float:
        return 0.0 if self.active else time.monotonic() - self.last_used


def _tts_handler(model: TTSModel) -> Handler:
    def run(payloads: list[Any]) -> list[Any]:
        out: list[Any] = []
        for text, voice, speed, emit, cancelled, *extra in payloads:
            n = 0
            phonemizer = extra[0] if extra else None
            extras = extra[1] if len(extra) > 1 else None
            kwargs: dict[str, Any] = {"phonemizer": phonemizer} if phonemizer and model.accepts_phonemizer else {}
            if extras is not None and model.accepts_extras:
                kwargs["extras"] = extras
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
        accel: Accelerators | None = None,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.store = store
        self.engines = engines
        self.events = events
        self.loaded: dict[str, LoadedModel] = {}
        #: models being unloaded (shown as "Stopping..." by ps)
        self.stopping: dict[str, LoadedModel] = {}
        #: accelerator memory (NVML, nvidia-smi, unified memory); tests pass a fake
        self.accel = accel or Accelerators()
        self.gpu_evictions = 0
        self._locks: dict[str, asyncio.Lock] = {}
        self._reaper: asyncio.Task[None] | None = None
        self._ensure_installed = ensure_installed
        self.evictions = 0
        #: set by the core: who the cloud providers are (a cloud model is installed when its provider is configured)
        self.providers: ProviderRegistry | None = None
        #: called with the spec after a model was loaded (the core logs a hardware notice; it cannot stop the load)
        self.on_loaded: Callable[[ModelSpec], None] | None = None

    # -- helpers ----------------------------------------------------------
    def resolve(self, name: str) -> ModelSpec:
        spec = self.catalog.resolve(name)
        if spec is None:
            raise ModelNotFound(
                f"The model '{name}' does not exist. Use GET /v1/registry (or `bolo list --all`) to see the catalog."
            )
        return spec

    def is_installed(self, spec: ModelSpec) -> bool:
        """Files on disk, or nothing to download: built in, or served by an operator-run TTS server."""
        from bolo.tts_runtimes import remote_url

        if spec.cloud:
            provider = (
                self.providers.find(
                    str(spec.options.get("provider") or "") or None,
                    str(spec.options.get("provider_kind") or "") or None,
                )
                if self.providers is not None
                else None
            )
            return provider is not None and provider.configured
        return spec.builtin or self.store.is_installed(spec.id) or remote_url(self.settings, spec) is not None

    def ttl_for(self, spec: ModelSpec, entry: LoadedModel | None = None) -> float:
        """Seconds a model stays loaded once idle (-1: forever). The first that is set wins: the keep-alive of the
        last request or load that gave one, the model's pinned ``keep_alive`` setting, the catalog entry's
        ``options.keep_alive``, the per-task TTL (``stt_ttl_s``, ...), the ``keep_alive`` setting."""
        if entry is not None and entry.keep_alive_s is not None:
            return entry.keep_alive_s
        for raw in (self.store.model_settings(spec.id).get("keep_alive"), spec.options.get("keep_alive")):
            try:
                value = parse_keep_alive(raw)
            except ValueError:
                value = None  # a damaged pin must not fail requests
            if value is not None:
                return value
        s = self.settings
        ttls = {
            "stt": s.stt_ttl_s,
            "tts": s.tts_ttl_s,
            "vad": s.vad_ttl_s,
            "diarization": s.diarization_ttl_s,
            "separation": s.diarization_ttl_s,
        }
        ttl = ttls.get(spec.task)
        if ttl is not None:
            return -1.0 if ttl < 0 else float(ttl)
        return self.default_keep_alive()

    def default_keep_alive(self) -> float:
        try:
            value = parse_keep_alive(self.settings.keep_alive)
        except ValueError:
            value = None
        return 300.0 if value is None else value

    def expires_in(self, entry: LoadedModel) -> float | None:
        """Seconds until an idle unload (counted from now while the model is in use); None: never."""
        ttl = self.ttl_for(entry.spec, entry)
        if ttl < 0:
            return None
        return max(0.0, ttl - entry.idle_s())

    def memory_in_use(self) -> int:
        """RAM held by loaded models (accelerator memory is counted apart: :meth:`gpu_in_use`)."""
        return sum(m.ram_bytes for m in self.loaded.values())

    def gpu_in_use(self) -> int:
        return sum(m.vram_bytes for m in self.loaded.values())

    # -- placement ----------------------------------------------------------
    def requested_device(self, spec: ModelSpec) -> str:
        """``device`` pinned for the model (``PUT /api/v1/models/{id}/settings``), else the catalog entry's
        ``options.device``, else the ``device`` setting."""
        pinned = self.store.model_settings(spec.id).get("device") or spec.options.get("device")
        return str(pinned or self.settings.device or "cpu").strip().lower()

    def resolve_device(self, requested: str) -> str:
        """``auto`` -> the accelerator found (``cuda`` when there are NVIDIA GPUs: one is picked per model; else
        ``coreml``) or ``cpu``; anything else as it is."""
        if requested in ("", "auto"):
            found = self.accel.default_device()
            return "cuda" if found and found.startswith("cuda") else (found or "cpu")
        return requested

    def allowed_gpus(self) -> list[int] | None:
        """NVIDIA GPUs bolo may use, as CUDA numbers them here: the ``gpus`` setting within the visible GPUs (all
        of them when it is empty); None when no NVIDIA GPU can be read."""
        found = [d.index for d in self.accel.cuda_devices()]
        if not found:
            return None
        wanted = set(self.settings.gpus)
        return [i for i in found if i in wanted] if wanted else found

    def gpu_available(self, device: str, freed: int = 0, freed_total: int | None = None) -> int | None:
        """Bytes a new model may take on ``device``: free memory minus the reserve, capped by what is left of
        ``gpu_memory_budget_mb``; None when neither is known. ``freed``: bytes that unloading idle models would
        give back on this device (``freed_total`` on all of them, for the budget)."""
        s = self.settings
        found = self.accel.device(device, fresh=True)
        free = None
        if found is not None and found.free_bytes is not None:
            free = found.free_bytes - s.gpu_reserve_mb * MB + freed
        if s.gpu_memory_budget_mb > 0:
            left = s.gpu_memory_budget_mb * MB - self.gpu_in_use() + (freed if freed_total is None else freed_total)
            free = left if free is None else min(free, left)
        return None if free is None else max(0, free)

    def gpu_budget_left(self, freed_total: int = 0) -> int | None:
        s = self.settings
        if s.gpu_memory_budget_mb <= 0:
            return None
        return max(0, s.gpu_memory_budget_mb * MB - self.gpu_in_use() + freed_total)

    def estimate_bytes(self, spec: ModelSpec) -> int:
        need = int((self.store.size_on_disk(spec.id) or spec.size_bytes) * 1.2)
        hw = spec.options.get("hardware")
        if isinstance(hw, dict) and hw.get("vram_gb"):
            with contextlib.suppress(TypeError, ValueError):
                need = max(need, int(float(hw["vram_gb"]) * 1024**3))
        return need

    def _layers(self, path: Any, options: dict[str, Any]) -> LayerInfo | None:
        from pathlib import Path

        if not options.get("model_file"):
            return None
        model_file = Path(path) / str(options["model_file"])
        blocks = gguf_block_count(model_file)
        if blocks is None or blocks <= 0:
            blocks = int(options.get("block_count") or 0)
        if blocks <= 0 or not model_file.is_file():
            return None
        extra = 0
        if options.get("mmproj_file"):
            mm = Path(path) / str(options["mmproj_file"])
            extra = mm.stat().st_size if mm.is_file() else 0
        return LayerInfo(blocks, model_file.stat().st_size, extra)

    def _llama_args(self, spec: ModelSpec) -> list[str]:
        return [str(a) for a in spec.options.get("server_args") or []] + list(self.settings.tts_server_args)

    def _plan(self, spec: ModelSpec, available: int | None, device: str | None = None) -> Placement:
        from bolo.tts_runtimes import remote_url

        requested = self.requested_device(spec)
        if spec.cloud or spec.engine == "remote":
            return Placement(device="cloud", requested=requested)
        if remote_url(self.settings, spec) is not None:
            return Placement(device="remote", requested=requested)
        device = device or self.resolve_device(requested)
        need = self.estimate_bytes(spec)
        split = split_mode(spec.engine, spec.kind, spec.options)
        path = self.store.model_dir(spec.id)
        layers = graphs = None
        op_ngl = None
        if split == "layers":
            layers = self._layers(path, spec.options)
            op_ngl = operator_layers(self._llama_args(spec))
        elif split == "graphs":
            engine = self.engines.get(spec.engine)
            with contextlib.suppress(Exception):
                graphs = engine.graphs(spec.kind, path, spec.options) if engine is not None else None
        _, _, idx = device.partition(":")
        return self._finish(plan(
            model_id=spec.id, engine_id=spec.engine, requested=requested, device=device, split=split,
            need_bytes=need, available_bytes=available, layers=layers, graphs=graphs, operator_ngl=op_ngl,
            device_index=int(idx) if idx.isdigit() else 0, onnx_split=self.onnx_split_allowed(spec),
        ))  # fmt: skip

    @staticmethod
    def _finish(p: Placement) -> Placement:
        """Fill in which GPU a model placed on one device uses."""
        fam = family(p.device)
        if p.gpus or p.gpu_bytes <= 0 or fam not in ("cuda", "metal"):
            return p
        _, _, idx = p.device.partition(":")
        if fam == "cuda" and not idx.isdigit():
            return p  # no GPU readings: the runtime's default GPU, number unknown
        index = int(idx) if idx.isdigit() else 0
        p.gpus, p.gpu_bytes_by_gpu = [index], {index: p.gpu_bytes}
        return p

    def _on_cpu(self, spec: ModelSpec, requested: str, notice: str) -> Placement:
        return Placement(device="cpu", requested=requested, cpu_bytes=self.estimate_bytes(spec), notice=notice)

    def _place_gpus(self, spec: ModelSpec, device: str, freed: dict[int, int]) -> Placement:
        """One placement decision, as if the bytes in ``freed`` (GPU -> bytes) had been given back.

        Several NVIDIA GPUs: a model that names no GPU goes on the allowed one with the most memory available; a
        llama.cpp model is spread over several when it does not fit whole on that one (or always, with
        ``gpu_spread``)."""
        requested = self.requested_device(spec)
        total_freed = sum(freed.values())
        allowed = self.allowed_gpus() if family(device) == "cuda" else None
        if allowed is None:
            return self._plan(spec, self.gpu_available(device, total_freed), device)
        if not allowed:
            return self._on_cpu(
                spec,
                requested,
                f"none of the GPUs in gpus = {self.settings.gpus} is visible: '{spec.id}' runs on the CPU",
            )
        _, _, idx = device.partition(":")
        candidates = allowed
        if idx.isdigit():
            if int(idx) not in allowed:
                return self._on_cpu(
                    spec,
                    requested,
                    f"GPU {idx} is not one bolo may use (gpus = {self.settings.gpus}): '{spec.id}' runs on the CPU",
                )
            candidates = [int(idx)]
        if spec.engine == "sherpa-onnx":
            # sherpa-onnx builds its CUDA sessions itself and always uses GPU 0
            candidates = [c for c in candidates if c == 0]
            if not candidates:
                return self._on_cpu(
                    spec,
                    requested,
                    f"sherpa-onnx can only use GPU 0, which is not in gpus: '{spec.id}' runs on the CPU",
                )
        avail = {i: self.gpu_available(f"cuda:{i}", freed.get(i, 0), total_freed) for i in candidates}
        best = best_gpu(avail)
        single = self._plan(spec, avail[best], f"cuda:{best}")
        if len(candidates) < 2 or None in avail.values() or single.split != "layers" or single.operator_layers:
            return single
        if has_gpu_flags(self._llama_args(spec)):
            return single  # the operator chose the GPUs (-sm, -mg, -ts, --device)
        layers = self._layers(self.store.model_dir(spec.id), spec.options)
        if layers is None:
            return single
        if not self.settings.gpu_spread and (single.gpu_layers or 0) >= layers.total_layers:
            return single  # fits whole on one GPU
        room = {i: int(a or 0) for i, a in avail.items()}
        budget = self.gpu_budget_left(total_freed)
        if budget is not None and sum(room.values()) > budget:  # gpu_memory_budget_mb counts every GPU together
            total = sum(room.values()) or 1
            room = {i: int(a * budget / total) for i, a in room.items()}
        spread = plan_spread(model_id=spec.id, requested=requested, layers=layers, available=room)
        if self.settings.gpu_spread or (spread.gpu_layers or 0) > (single.gpu_layers or 0):
            return spread
        return single

    def _vram_by_gpu(self, models: list[LoadedModel]) -> dict[int, int]:
        """Accelerator bytes the given models hold, per GPU."""
        out: dict[int, int] = {}
        for m in models:
            p = m.placement
            planned = sum(p.gpu_bytes_by_gpu.values())
            if planned > 0:
                for i, b in p.gpu_bytes_by_gpu.items():
                    out[i] = out.get(i, 0) + int(m.vram_bytes * b / planned)
            else:
                _, _, idx = p.device.partition(":")
                i = p.gpus[0] if p.gpus else (int(idx) if idx.isdigit() else 0)
                out[i] = out.get(i, 0) + m.vram_bytes
        return out

    def gpu_use_by_gpu(self) -> dict[int, int]:
        """Accelerator bytes held by loaded models, per GPU (``bolo ps``, the console)."""
        return self._vram_by_gpu([m for m in self.loaded.values() if m.vram_bytes > 0])

    def onnx_split_allowed(self, spec: ModelSpec) -> bool:
        """``onnx_split = "auto"``, the ``onnx`` package installed, and a model whose graphs bolo can cut."""
        from bolo.onnx_split import available
        from bolo.placement import GRAPH_KINDS, SPLITTABLE_KINDS

        if str(self.settings.onnx_split).lower() != "auto" or spec.engine != "onnxruntime":
            return False
        return spec.kind in (*SPLITTABLE_KINDS, *GRAPH_KINDS) and available()

    async def place(self, spec: ModelSpec) -> Placement:
        """Decide where a model goes, unloading idle GPU-resident models (LRU) when that lets it fit whole."""
        requested = self.requested_device(spec)
        device = self.resolve_device(requested)
        if family(device) == "cpu" or spec.cloud or spec.engine == "remote":
            return self._plan(spec, None)
        while True:
            p = self._place_gpus(spec, device, {})
            whole = p.device not in ("cpu",) and p.cpu_bytes == 0
            if whole or p.operator_layers or not p.measured or p.device in ("remote", "cloud"):
                return p
            idle = sorted(
                (m for m in self.loaded.values() if m.active == 0 and m.vram_bytes > 0), key=lambda m: m.last_used
            )
            # evict only when unloading every idle GPU-resident model would make this one fit whole
            if not idle or self._place_gpus(spec, device, self._vram_by_gpu(idle)).cpu_bytes > 0:
                return p
            victim = idle[0]
            log.info("unloading %s from %s to make room for %s", victim.id, device, spec.id)
            self.gpu_evictions += 1
            self.evictions += 1
            await self.unload(victim.id, reason="evicted")

    def place_llama_server(self, name: str, path: Any, options: dict[str, Any], extra: list[str]) -> Placement:
        """Layers of a llama-server bolo starts outside the model manager (the speech-to-speech server)."""
        requested = str(self.settings.device or "cpu").strip().lower()
        device = self.resolve_device(requested)
        layers = self._layers(path, options)
        size = (layers.file_bytes + layers.extra_bytes) if layers else 0

        def one(dev: str) -> Placement:
            return self._finish(plan(
                model_id=name, engine_id="tts-server", requested=requested, device=dev,
                split="layers" if layers else "whole", need_bytes=size,
                available_bytes=self.gpu_available(dev) if family(dev) != "cpu" else None, layers=layers,
                operator_ngl=operator_layers(extra),
            ))  # fmt: skip

        allowed = self.allowed_gpus() if family(device) == "cuda" else None
        _, _, idx = device.partition(":")
        candidates = [int(idx)] if idx.isdigit() else (allowed or [])
        if allowed is not None and allowed and all(c in allowed for c in candidates):
            avail = {i: self.gpu_available(f"cuda:{i}") for i in candidates}
            best = best_gpu(avail)
            p = one(f"cuda:{best}")
            device = f"cuda:{best}"
            if (
                layers is not None and len(candidates) > 1 and None not in avail.values()
                and operator_layers(extra) is None and not has_gpu_flags(extra)
                and (self.settings.gpu_spread or (p.gpu_layers or 0) < layers.total_layers)
            ):  # fmt: skip
                spread = plan_spread(
                    model_id=name, requested=requested, layers=layers,
                    available={i: int(a or 0) for i, a in avail.items()},
                )  # fmt: skip
                if self.settings.gpu_spread or (spread.gpu_layers or 0) > (p.gpu_layers or 0):
                    p = spread
        elif allowed is not None:
            p = Placement(device="cpu", requested=requested, cpu_bytes=size,
                          notice=f"no GPU bolo may use (gpus = {self.settings.gpus}): '{name}' runs on the CPU")  # fmt: skip
        else:
            p = one(device)
        if layers is None and family(device) != "cpu" and p.device == "cpu" and not p.notice:
            p.device = device  # layer count unknown: all layers, as before (-ngl 99)
        if p.notice:
            log.warning("%s", p.notice)
        return p

    # -- loading ----------------------------------------------------------
    def _build_queue(self, spec: ModelSpec, model: Model) -> ModelQueue:
        s = self.settings
        handlers: dict[str, Handler] = {}
        max_batch: dict[str, int] = {}
        if isinstance(model, STTModel):
            handlers["batch"] = model.transcribe
            max_batch["batch"] = max(1, min(model.max_batch, s.max_batch))
            if model.speaker_conditioned:
                stt = model
                handlers["speakers"] = lambda items: [stt.transcribe_speakers(item) for item in items]
                max_batch["speakers"] = 1
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
            workers=self.num_parallel(spec),
            max_batch=max_batch,
            max_wait_ms=s.batch_wait_ms,
            max_queue=self.max_queue(spec),
            queue_timeout_s=s.queue_timeout_s,
        )

    def _pinned_int(self, spec: ModelSpec, keys: tuple[str, ...]) -> int | None:
        """The first of ``keys`` set as a pin, then as a catalog option, as a whole number."""
        pins = self.store.model_settings(spec.id)
        for source in (pins, spec.options):
            for key in keys:
                raw = source.get(key)
                if raw in (None, ""):
                    continue
                with contextlib.suppress(TypeError, ValueError):
                    return int(raw)
        return None

    def num_parallel(self, spec: ModelSpec) -> int:
        """Requests the model works on at once (Ollama's num_parallel): a pin, the catalog's ``num_parallel`` (or its
        older name ``workers``), else the ``num_parallel`` setting (``workers`` when that is 0)."""
        value = self._pinned_int(spec, ("num_parallel", "workers"))
        return max(1, value if value is not None and value > 0 else self.settings.parallel())

    def max_queue(self, spec: ModelSpec) -> int:
        """Requests that may wait for the model before 429: a pin, the catalog's ``max_queue``, else the setting."""
        value = self._pinned_int(spec, ("max_queue",))
        return max(0, value if value is not None and value >= 0 else self.settings.max_queue)

    def _rss(self) -> int:
        """Resident memory of this process and the processes it started (servers, plugins)."""
        total = 0
        for pid in our_pids():
            with contextlib.suppress(Exception):
                total += psutil.Process(pid).memory_info().rss
        return total

    def _vram(self, place: Placement) -> tuple[int | None, bool]:
        """(accelerator memory held now on the placement's GPUs, by this server where that can be told else on the
        whole device; unified)."""
        device = place.device
        if family(device) in ("cpu", "remote", "cloud"):
            return None, False
        names = [f"cuda:{i}" for i in place.gpus] if family(device) == "cuda" and len(place.gpus) > 1 else [device]
        total = 0
        for name in names:
            found = self.accel.device(name, fresh=True)
            if found is None:
                return None, False
            if found.unified:
                return None, True
            held = found.process_bytes if found.process_bytes is not None else found.used_bytes
            if held is None:
                return None, False
            total += held
        return total, False

    def _load_blocking(
        self, spec: ModelSpec, placement: Placement | None = None
    ) -> tuple[Model, int, float, int, bool]:
        engine = self.engines.get(spec.engine)
        if engine is None:
            raise EngineUnavailable(f"model '{spec.id}' needs the unknown engine '{spec.engine}'")
        status = engine.status()
        if not status.available:
            raise EngineUnavailable(
                f"model '{spec.id}' needs the '{spec.engine}' engine, which is not available: {status.reason}"
            )
        self._probe(spec)
        place = placement or Placement(device=self.settings.device)
        device = place.device if place.device not in ("remote", "cloud") else self.settings.device
        before = self._rss()
        vram_before, unified = self._vram(place)
        t0 = time.perf_counter()
        opts = LoadOptions(
            threads=int(spec.options.get("threads", self.settings.resolved_threads())),
            device=device,
            options={
                "endpoint_silence_s": self.settings.endpoint_silence_ms / 1000.0,
                "max_utterance_s": self.settings.max_utterance_s,
            },
            gpu_layers=place.gpu_layers if place.split == "layers" and not place.operator_layers else None,
            graph_devices=dict(place.graphs),
            max_memory=place.max_memory,
            graph_split=dict(place.graph_split),
            gpus=list(place.gpus) if family(place.device) == "cuda" else [],
            tensor_split=list(place.tensor_split) if place.tensor_split else None,
            main_gpu=place.main_gpu,
            visible_gpus=max(1, len(self.accel.cuda_devices())),
            parallel=self.num_parallel(spec),
        )
        model_opts = dict(spec.options)
        pins = self.store.model_settings(spec.id)
        from bolo.llama_args import TUNING_KEYS

        for key in TUNING_KEYS:  # llama.cpp speed settings pinned for the model win over the catalog's
            if pins.get(key) not in (None, ""):
                model_opts[key] = pins[key]
        if spec.voices:
            model_opts.setdefault("voices", spec.voices)
        if len(spec.languages) == 1:
            model_opts.setdefault("language", spec.languages[0])
        if spec.engine.startswith("plugin:") or spec.engine in ("tts-server", "remote"):
            model_opts.setdefault("model_id", spec.id)
        if spec.engine == "remote":
            model_opts.setdefault("sample_rate", spec.sample_rate)
        if spec.engine == "tts-server":
            model_opts.setdefault("model_aliases", list(spec.aliases))
            model_opts.setdefault("sample_rate", spec.sample_rate)
        model = engine.load(spec.kind, spec.task, self.store.model_dir(spec.id), model_opts, opts)
        if isinstance(model, TTSModel):
            model.warm_up()
        load_ms = (time.perf_counter() - t0) * 1000.0
        grown = max(0, self._rss() - before)
        vram_after, _ = self._vram(place)
        measured = vram_before is not None and vram_after is not None
        if measured:
            vram = max(0, int(vram_after or 0) - int(vram_before or 0))
        else:  # unified memory (it is in the RSS growth already) or nothing to ask: what was planned
            vram = place.gpu_bytes if place.device not in ("cpu", "remote", "cloud") else 0
        report = getattr(model, "placement_report", None)
        if isinstance(report, dict) and report.get("gpu_bytes") is not None:
            # a plugin that split the model says how (accelerate's device map); more precise than a delta
            with contextlib.suppress(TypeError, ValueError):
                vram, measured = int(report["gpu_bytes"]), True
                place.gpu_bytes, place.cpu_bytes = vram, int(report.get("cpu_bytes") or 0)
        # An estimate. RSS growth during the load reads low, sometimes zero, when the allocator reuses memory
        # that an unloaded model gave back, so it is never taken to be less than the model's files on disk.
        ram = grown if unified else grown + vram
        memory = max(ram, self.store.size_on_disk(spec.id), vram)
        if isinstance(report, dict) and report.get("gpu_bytes") is not None:
            memory = max(memory, place.gpu_bytes + place.cpu_bytes)
        return model, memory, load_ms, min(vram, memory), measured

    def _probe(self, spec: ModelSpec) -> None:
        """First load of an installed model happens in a child process (see ``bolo._probe``)."""
        if spec.builtin or not self.settings.probe_models or spec.engine.startswith("plugin:"):
            return  # plugins run in their own process already
        if spec.engine in ("tts-server", "remote"):
            return  # the model runs in a server process (or at a provider); loading it there is the check
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

    async def _make_room(self, spec: ModelSpec, need: int | None = None) -> None:
        s = self.settings
        budget = s.memory_budget_mb * 1024 * 1024
        if need is None:
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
            placement = await self.place(spec)
            if placement.notice:
                log.warning("%s", placement.notice)
            need_ram = int((self.store.size_on_disk(spec.id) or spec.size_bytes) * 1.2)
            if placement.gpu_bytes and (placement.gpu_bytes + placement.cpu_bytes):
                need_ram = int(need_ram * placement.cpu_bytes / (placement.gpu_bytes + placement.cpu_bytes))
            await self._make_room(spec, need_ram)
            loop = asyncio.get_running_loop()
            model, memory, load_ms, vram, measured = await loop.run_in_executor(
                None, self._load_blocking, spec, placement
            )
            entry = LoadedModel(spec, model, self._build_queue(spec, model), memory, load_ms)
            entry.vram_bytes, entry.vram_measured, entry.placement = vram, measured, placement
            self.loaded[spec.id] = entry
            log.info(
                "loaded %s in %.0f ms (~%.0f MB, %s on %s)",
                spec.id, load_ms, memory / 1e6, entry.processor(), placement.device,
            )  # fmt: skip
            event: dict[str, Any] = {"id": spec.id, "event": "loaded", "processor": entry.processor()}
            if placement.notice:
                event["notice"] = placement.notice
            self.events.publish("model", event)
            if self.on_loaded is not None:
                with contextlib.suppress(Exception):
                    self.on_loaded(spec)
            self._ensure_reaper()
            return entry

    async def unload(self, name: str, reason: str = "requested") -> bool:
        spec = self.catalog.resolve(name)
        if spec is None or spec.id not in self.loaded:
            return False
        entry = self.loaded.pop(spec.id)
        entry.stopping = True
        self.stopping[spec.id] = entry
        try:
            await entry.queue.close()
            with contextlib.suppress(Exception):
                entry.model.close()
        finally:
            self.stopping.pop(spec.id, None)
        log.info("unloaded %s (%s)", spec.id, reason)
        self.events.publish("model", {"id": spec.id, "event": "unloaded", "reason": reason})
        return True

    @contextlib.asynccontextmanager
    async def use(self, name: str) -> AsyncIterator[LoadedModel]:
        """Borrow a model: loads it if needed and keeps it from being unloaded.

        The request's ``keep_alive`` (:data:`bolo.keepalive.REQUEST_KEEP_ALIVE`), when it has one and it applies to
        the model's task, becomes the model's keep-alive; ``0`` unloads the model when the last request using it
        ends (for sessions: when the session ends, see :meth:`apply_keep_alive`)."""
        entry = await self.load(name)
        ka = REQUEST_KEEP_ALIVE.get()
        if ka is not None and ka.applies(entry.spec.task):
            ka.used.add(entry.id)
            if not ka.session or ka.seconds != 0:
                entry.keep_alive_s = ka.seconds
        entry.active += 1
        entry.requests += 1
        entry.last_used = time.monotonic()
        try:
            yield entry
        finally:
            entry.active -= 1
            entry.last_used = time.monotonic()
            if entry.active == 0 and entry.keep_alive_s == 0 and self.loaded.get(entry.id) is entry:
                with contextlib.suppress(Exception):
                    await self.unload(entry.id, reason="keep_alive 0")

    async def apply_keep_alive(self, ids: Any, seconds: float) -> None:
        """At the end of a session: give the models it used this keep-alive (``0`` unloads the idle ones now)."""
        for model_id in list(ids):
            entry = self.loaded.get(model_id)
            if entry is None:
                continue
            entry.keep_alive_s = seconds
            entry.last_used = time.monotonic()
            if seconds == 0 and entry.active == 0:
                with contextlib.suppress(Exception):
                    await self.unload(model_id, reason="keep_alive 0")

    def set_keep_alive(self, entry: LoadedModel, seconds: float | None) -> None:
        """A load's ``keep_alive`` (``bolo load``, the console's "Keep loaded"); restarts the countdown."""
        if seconds is None:
            return
        entry.keep_alive_s = seconds
        entry.last_used = time.monotonic()
        self._ensure_reaper()

    # -- idle unloading ---------------------------------------------------
    def _ensure_reaper(self) -> None:
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.create_task(self._reap_loop(), name="bolo-reaper")

    async def reap_once(self) -> list[str]:
        gone: list[str] = []
        for entry in list(self.loaded.values()):
            ttl = self.ttl_for(entry.spec, entry)
            if ttl >= 0 and entry.active == 0 and entry.idle_s() >= ttl:
                await self.unload(entry.id, reason="idle")
                gone.append(entry.id)
        return gone

    async def _reap_loop(self) -> None:
        while self.loaded:
            ttls = [t for t in (self.ttl_for(m.spec, m) for m in self.loaded.values()) if t >= 0]
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
        self.accel.close()


def _enrol(separator: SeparationModel, item: tuple[str, Any]) -> bool:
    separator.enrol(*item)
    return True
