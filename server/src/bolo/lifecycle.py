"""Model lifecycle: lazy loading, keep-alive unloading, memory budgets, LRU eviction and GPU/CPU placement.

Loading a model first places it (:mod:`bolo.placement`): on the accelerator when it fits in the free accelerator
memory minus ``gpu_reserve_mb`` and in ``gpu_memory_budget_mb``, after unloading idle GPU-resident models (least
recently used first) when that makes it fit; otherwise split (llama.cpp layers, ONNX Runtime graphs, a PyTorch
plugin's device map) or on the CPU, with a notice. Then it makes room in RAM (``memory_budget_mb``,
``max_loaded_models``), loads, and measures what the load took: the growth of this process and the processes it
started (RSS), and of the accelerator memory they hold (:mod:`bolo.accel`).

Where the driver keeps no per-process figure (Windows under WDDM), a load is measured by this process's own use
as DXGI reports it (runtimes in bolo's process) or else by the growth of the device's used memory, with such loads
serialised so that two of them are not counted together; when neither grows the accelerator share is the plan.
Either way a model placed on the accelerator reads as on it (``processor``): memory that cannot be measured is
"unknown" (``vram_measured`` false, ``vram_source`` "plan"), never a reason to say "100% CPU".
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import psutil

from bolo import cuda_libs, gpu_health, ort_providers, remote_code
from bolo.accel import Accelerators, our_pids
from bolo.catalog import Catalog, ModelSpec
from bolo.config import Settings
from bolo.engines import EngineRegistry
from bolo.engines.base import (
    AudioClassifierModel,
    AlignModel,
    DenoiseModel,
    DiarizationModel,
    LidModel,
    LoadOptions,
    Model,
    SeparationModel,
    SpeakerEmbeddingModel,
    STTModel,
    STTStream,
    TTSModel,
    TurnModel,
    VoiceConversionModel,
)
from bolo.errors import EngineUnavailable, ModelNotFound, ModelNotInstalled, Overloaded
from bolo.keepalive import REQUEST_KEEP_ALIVE, parse_keep_alive
from bolo.llama_args import has_gpu_flags
from bolo.obs.events import EventBus
from bolo.placement import (
    CPU_ONLY_ON,
    MB,
    LayerInfo,
    Placement,
    best_gpu,
    family,
    gguf_block_count,
    operator_layers,
    partly_on_cpu,
    plan,
    plan_spread,
    processor,
    runtime_label,
    split_mode,
)
from bolo.providers import ProviderRegistry
from bolo.scheduler import Handler, ModelQueue
from bolo.store import ModelStore

log = logging.getLogger("bolo.lifecycle")

#: device families that are a GPU, which bolo.gpu_health watches and a lost GPU takes away
GPU_FAMILIES = frozenset({"cuda", "vulkan", "dml", "rocm", "hip"})


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: Model
    queue: ModelQueue
    memory_bytes: int
    load_ms: float
    loaded_at: float = field(default_factory=time.monotonic)
    #: ``time.perf_counter()`` of the last use: the least recently used model is unloaded first, and Windows'
    #: ``time.monotonic()`` ticks only every 15.6 ms, too coarse to order uses close together
    last_used: float = field(default_factory=time.perf_counter)
    active: int = 0
    requests: int = 0
    #: of ``memory_bytes``, the part in accelerator memory
    vram_bytes: int = 0
    #: ``vram_bytes`` was measured (NVML, nvidia-smi, DXGI, the device's growth, the plugin's report), not the plan
    vram_measured: bool = False
    #: where ``vram_bytes`` comes from: "process" (per-process figures: NVML, nvidia-smi, DXGI for DirectML),
    #: "dxgi" (this process's use under WDDM), "device" (the device's used memory, before and after), "plugin",
    #: "unified" (Apple Silicon: the plan), "plan" (could not be measured) or "" (on the CPU)
    vram_source: str = ""
    placement: Placement = field(default_factory=Placement)
    #: set by a request's or a load's ``keep_alive`` (seconds; -1: forever); None: the settings decide
    keep_alive_s: float | None = None
    stopping: bool = False
    #: the GPU the model is on was lost (bolo.gpu_health): why; the model is slow or broken until it is moved
    degraded: str | None = None

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def ram_bytes(self) -> int:
        return max(0, self.memory_bytes - self.vram_bytes)

    def processor(self) -> str:
        """Where the weights are, as Ollama's PROCESSOR column says it. A model placed whole on one side reads
        "100% GPU" / "100% CPU" (the runtime's own host memory of a GPU model is in SIZE, not here); a split model
        reads by bytes: measured when the accelerator memory was measured, else as planned. A model placed on the
        accelerator reads as on it when its memory could not be measured (Windows under WDDM, unified memory);
        only a measurement that found nothing there (the runtime fell back to the CPU) makes it "100% CPU"."""
        p = self.placement
        if p.device in ("remote", "cloud"):
            return p.device
        if family(p.device) == "cpu" or (p.gpu_bytes <= 0 and p.cpu_bytes > 0):
            return "100% CPU"
        if self.vram_measured and self.vram_bytes <= 0:
            return "100% CPU"
        if p.cpu_bytes <= 0:
            return "100% GPU"
        if self.vram_measured:
            return processor(self.vram_bytes, self.memory_bytes)
        return p.processor()

    @property
    def vram_unknown(self) -> bool:
        """On an accelerator whose memory use could not be measured (not unified memory, where that is known)."""
        return self.vram_source == "plan"

    @property
    def busy(self) -> bool:
        """In use: a request holds it (``active``), or its queue has work running or waiting. A request whose client
        disconnected no longer holds the model, but the engine may still be decoding it; the model must not be
        unloaded under that work."""
        queue = self.queue
        return self.active > 0 or bool(queue is not None and queue.busy)

    def idle_s(self) -> float:
        """Seconds since the model was last in use (since its last request ended or its last job finished)."""
        if self.busy:
            return 0.0
        done = self.queue.last_done if self.queue is not None else 0.0
        return time.perf_counter() - max(self.last_used, done)


@dataclass
class VramReading:
    """Accelerator memory on a model's GPUs at one moment (bolo.accel), each figure None when unknown."""

    #: held by this server and the processes it started, from per-process figures (None under WDDM)
    process: int | None = None
    #: this process's own use as DXGI reports it (Windows)
    own: int | None = None
    #: the devices' used memory, whoever holds it
    device: int | None = None
    unified: bool = False
    #: the GPUs were found
    ok: bool = False


def _add(total: int | None, value: int | None) -> int | None:
    return None if total is None or value is None else total + value


def attribute_vram(
    before: VramReading, after: VramReading, in_process: bool, place: Placement
) -> tuple[int, bool, str]:
    """What a load put in accelerator memory: (bytes, measured, source). Per-process figures when the driver
    keeps them; else this process's own DXGI use when the runtime is in bolo's process and it grew; else the
    devices' used memory when it grew; else the plan (not measured)."""
    planned = place.gpu_bytes if family(place.device) not in ("cpu", "remote", "cloud") else 0
    if not before.ok or not after.ok:
        return planned, False, "plan" if planned else ""
    if before.unified:
        return planned, False, "unified"
    if before.process is not None and after.process is not None:
        return max(0, after.process - before.process), True, "process"
    if in_process and before.own is not None and after.own is not None and after.own > before.own:
        return after.own - before.own, True, "dxgi"
    if before.device is not None and after.device is not None and after.device > before.device:
        return after.device - before.device, True, "device"
    return planned, False, "plan"


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


#: how long a model moving off a lost GPU waits for the requests using it
MOVE_WAIT_S = 30.0


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
        self.accel = accel or Accelerators(directml=str(settings.device).lower().startswith(("dml", "directml")))
        self.gpu_evictions = 0
        self._locks: dict[str, asyncio.Lock] = {}
        #: loads measured by the device's used memory go one at a time (two at once would be counted together)
        self._device_measure = threading.Lock()
        self._reaper: asyncio.Task[None] | None = None
        self._ensure_installed = ensure_installed
        self.evictions = 0
        #: set by the core: who the cloud providers are (a cloud model is installed when its provider is configured)
        self.providers: ProviderRegistry | None = None
        #: called with the spec after a model was loaded (the core logs a hardware notice; it cannot stop the load)
        self.on_loaded: Callable[[ModelSpec], None] | None = None
        #: is the GPU still there? Checked while models are on one (bolo.gpu_health)
        self.gpu = gpu_health.GpuMonitor(
            interval_s=float(settings.gpu_health_interval_s),
            has_gpu_models=self.has_gpu_models,
            on_lost=self.on_gpu_lost,
            fallback=self.gpu_lost_fallback,
        )

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
        if spec.builtin:
            # nothing to download; a detector from an optional package (WebRTC VAD) is there once the package is
            return spec.engine in ("fake", "builtin") or self.engines.status(spec.engine).available
        return self.store.is_installed(spec.id) or remote_url(self.settings, spec) is not None

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
            "speaker-embedding": s.diarization_ttl_s,
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

    # -- a lost GPU (bolo.gpu_health) --------------------------------------
    @property
    def gpu_lost_fallback(self) -> str:
        """``gpu_lost_fallback``: "cpu" (load the models again on the CPU) or "fail" (their requests fail)."""
        return "fail" if str(getattr(self.settings, "gpu_lost_fallback", "cpu")).lower() == "fail" else "cpu"

    @staticmethod
    def on_gpu(entry: LoadedModel) -> bool:
        return family(entry.placement.device) in GPU_FAMILIES

    def has_gpu_models(self) -> bool:
        return any(self.on_gpu(e) for e in self.loaded.values())

    def _gpu_lost_notice(self, model_id: str) -> str:
        health = self.gpu.health
        reason = health.reason or "the driver lost it"
        return f"the GPU was lost ({reason}): '{model_id}' runs on the CPU; {gpu_health.REBOOT} the GPU"

    async def on_gpu_lost(self, health: gpu_health.GpuHealth) -> None:
        """The GPU is gone: mark the models on it as degraded, say so, and (``gpu_lost_fallback = "cpu"``) load
        them again on the CPU, one task per model; a request in flight on the GPU is let finish first."""
        victims = [e for e in self.loaded.values() if self.on_gpu(e)]
        for e in victims:
            e.degraded = f"the GPU was lost ({health.reason or 'the driver lost it'}); {gpu_health.REBOOT}"
            e.placement.notice = f"{e.placement.notice}; {e.degraded}" if e.placement.notice else e.degraded
            self.events.publish("model", {"id": e.id, "event": "degraded", "notice": e.degraded})
        health.degraded = [e.id for e in victims]
        self.events.publish("gpu", health.to_dict())
        if not victims:
            return
        if self.gpu_lost_fallback == "fail":
            log.error(
                "gpu_lost_fallback = fail: requests to %s fail until bolo is restarted after a reboot",
                ", ".join(e.id for e in victims),
            )
            return
        log.warning("loading %s again on the CPU", ", ".join(e.id for e in victims))
        await asyncio.gather(*(self._move_to_cpu(e, health) for e in victims))

    async def _move_to_cpu(self, entry: LoadedModel, health: gpu_health.GpuHealth) -> None:
        keep = entry.keep_alive_s
        try:
            # let the requests using it finish first (on a GPU that is gone they may fail; one stuck in the driver
            # is given up on after MOVE_WAIT_S, and the queue then fails it rather than leave it waiting)
            deadline = time.monotonic() + MOVE_WAIT_S
            while entry.busy and time.monotonic() < deadline:  # noqa: ASYNC110 (a deadline)
                await asyncio.sleep(0.05)
            if self.loaded.get(entry.id) is entry:
                await self.unload(entry.id, reason="GPU lost")
            moved = await self.load(entry.id)
            if keep is not None:
                moved.keep_alive_s = keep
            health.moved.append(entry.id)
            log.warning("%s now runs on %s (the GPU was lost)", entry.id, moved.placement.device)
        except Exception as exc:
            log.error("could not load %s again on the CPU after the GPU was lost: %s", entry.id, exc)
        finally:
            with contextlib.suppress(ValueError):
                health.degraded.remove(entry.id)
            self.events.publish("gpu", health.to_dict())

    # -- placement ----------------------------------------------------------
    def requested_device(self, spec: ModelSpec) -> str:
        """``device`` pinned for the model (``PUT /api/v1/models/{id}/settings``), else the catalog entry's
        ``options.device``, else the ``device`` setting."""
        pinned = self.store.model_settings(spec.id).get("device") or spec.options.get("device")
        return str(pinned or self.settings.device or "cpu").strip().lower()

    def resolve_device(self, requested: str, engine: str | None = None) -> str:
        """``auto`` -> the accelerator found (``cuda`` when there are NVIDIA GPUs: one is picked per model; else
        ``coreml``) or ``cpu``; anything else as it is.

        For the ``onnxruntime`` engine ``auto`` is what the installed ONNX Runtime can reach
        (:func:`bolo.ort_providers.auto_device`): with onnxruntime-directml that is ``dml`` (Windows' high-performance
        adapter), not ``cuda`` it has no provider for."""
        if requested in ("", "auto"):
            found = self.accel.default_device()
            if engine == "onnxruntime":
                device = ort_providers.auto_device(found)
                if device == "dml" and getattr(self.accel, "directml", True) is False:
                    self.accel.directml = True  # read DirectML adapters' memory (DXGI) for placement and ps
                return device
            return "cuda" if found and found.startswith("cuda") else (found or "cpu")
        return requested

    def device_for(self, spec: ModelSpec, requested: str | None = None) -> tuple[str, str | None]:
        """The device ``spec`` is placed on (``requested`` resolved, :meth:`resolve_device`), and why it is the CPU
        when the model cannot use the accelerator asked for although its engine generally can: a model DirectML
        cannot run (:data:`bolo.placement.CPU_ONLY_ON`), a CPU build of sherpa-onnx with ``cuda``."""
        requested = self.requested_device(spec) if requested is None else requested
        device = self.resolve_device(requested, spec.engine)
        fam = family(device)
        reason = CPU_ONLY_ON.get((spec.engine, spec.kind), {}).get(fam)
        if reason:
            return "cpu", f"'{spec.id}' runs on the CPU, not {device}: {reason}"
        if spec.engine == "sherpa-onnx" and fam == "cuda" and ort_providers.sherpa_gpu_build() is False:
            return "cpu", ort_providers.sherpa_cpu_notice(spec.id)
        return device, None

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
        why = None
        if device is None:
            device, why = self.device_for(spec, requested)
        if why:
            p = plan(
                model_id=spec.id, engine_id=spec.engine, requested=requested, device="cpu", split="whole",
                need_bytes=self.estimate_bytes(spec), available_bytes=None,
            )  # fmt: skip
            p.notice = why
            return p
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
        p = self._finish(plan(
            model_id=spec.id, engine_id=spec.engine, requested=requested, device=device, split=split,
            need_bytes=need, available_bytes=available, layers=layers, graphs=graphs, operator_ngl=op_ngl,
            device_index=int(idx) if idx.isdigit() else 0, onnx_split=self.onnx_split_allowed(spec),
        ))  # fmt: skip
        part = partly_on_cpu(spec.engine, spec.kind, p.device) if p.gpu_bytes > 0 else None
        if part:  # "100% GPU" counts the weights: say what runs on the CPU all the same
            part = f"'{spec.id}' on {p.device}: {part}"
            p.notice = f"{p.notice}; {part}" if p.notice else part
        return p

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
                i = p.gpus[0] if p.gpus else self.accel.index_of(p.device)
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
        device, _ = self.device_for(spec, requested)
        if family(device) == "cpu" or spec.cloud or spec.engine == "remote":
            return self._plan(spec, None)
        if self.gpu.lost and family(device) in GPU_FAMILIES:
            if self.gpu_lost_fallback == "fail":
                raise EngineUnavailable(
                    f"model '{spec.id}' is placed on {device}, but {self.gpu.health.notice()} "
                    '(set gpu_lost_fallback = "cpu" to run it on the CPU)'
                )
            return self._on_cpu(spec, requested, self._gpu_lost_notice(spec.id))
        while True:
            p = self._place_gpus(spec, device, {})
            whole = p.device not in ("cpu",) and p.cpu_bytes == 0
            if whole or p.operator_layers or not p.measured or p.device in ("remote", "cloud"):
                return p
            idle = sorted(
                (m for m in self.loaded.values() if not m.busy and m.vram_bytes > 0), key=lambda m: m.last_used
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
        if self.gpu.lost and family(device) in GPU_FAMILIES:
            p = Placement(device="cpu", requested=requested, cpu_bytes=size, notice=self._gpu_lost_notice(name))
            log.warning("%s", p.notice)
            return p

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
        elif isinstance(model, SpeakerEmbeddingModel):
            embedder = model
            handlers["embed"] = lambda items: [embedder.embed(item) for item in items]
            max_batch["embed"] = 8
        elif isinstance(model, SeparationModel):
            separator = model
            handlers["enrol"] = lambda items: [_enrol(separator, item) for item in items]
            handlers["extract"] = lambda items: [separator.extract(*item) for item in items]
            handlers["separate"] = lambda items: [separator.separate(item) for item in items]
            max_batch["enrol"] = 1
            max_batch["extract"] = 1
            max_batch["separate"] = 1
        elif isinstance(model, DenoiseModel):
            denoiser = model
            handlers["denoise"] = lambda items: [denoiser.denoise(item) for item in items]
            max_batch["denoise"] = 1
        elif isinstance(model, AudioClassifierModel):
            classifier = model
            handlers["classify"] = lambda items: [classifier.classify(item) for item in items]
            max_batch["classify"] = 1
        elif isinstance(model, VoiceConversionModel):
            converter = model
            handlers["convert"] = lambda items: [converter.convert(*item) for item in items]
            max_batch["convert"] = 1
        elif isinstance(model, LidModel):
            lid = model
            handlers["lid"] = lambda items: [lid.identify(item) for item in items]
            max_batch["lid"] = 1
        elif isinstance(model, AlignModel):
            aligner = model
            handlers["align"] = lambda items: [aligner.align(*item) for item in items]
            max_batch["align"] = 1
        elif isinstance(model, TurnModel):
            turn = model
            handlers["turn"] = lambda items: [turn.probability(item) for item in items]
            handlers["turn_text"] = lambda items: [turn.probability_text(item) for item in items]
            max_batch["turn"] = 1
            max_batch["turn_text"] = 1
        return ModelQueue(
            spec.id,
            handlers,
            workers=self.num_parallel(spec),
            max_batch=max_batch,
            max_wait_ms=s.batch_wait_ms,
            max_queue=self.max_queue(spec),
            queue_timeout_s=self.queue_timeout(spec),
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

    def queue_timeout(self, spec: ModelSpec) -> float:
        """Seconds a request may wait for the model before 429: the catalog's ``queue_timeout_s`` (0: no limit; for
        slow models whose long recordings queue many pieces at once), else the setting."""
        value = self._pinned_int(spec, ("queue_timeout_s",))
        return float(value) if value is not None and value >= 0 else self.settings.queue_timeout_s

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

    def _vram(self, place: Placement) -> VramReading:
        """Accelerator memory on the placement's GPUs now: held by this server where the driver tells it per
        process, this process's own use (DXGI), and the devices' used memory."""
        device = place.device
        if family(device) in ("cpu", "remote", "cloud"):
            return VramReading()
        names = [f"cuda:{i}" for i in place.gpus] if family(device) == "cuda" and len(place.gpus) > 1 else [device]
        reading = VramReading(process=0, own=0, device=0, ok=True)
        for name in names:
            found = self.accel.device(name, fresh=True)
            if found is None:
                return VramReading()
            if found.unified:
                return VramReading(unified=True, ok=True)
            reading.process = _add(reading.process, found.process_bytes)
            reading.own = _add(reading.own, found.own_bytes)
            reading.device = _add(reading.device, found.used_bytes)
        return reading

    def _load_blocking(
        self, spec: ModelSpec, placement: Placement | None = None
    ) -> tuple[Model, int, float, int, bool, str]:
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
        vram_before = self._vram(place)
        serial = contextlib.ExitStack()
        if vram_before.ok and vram_before.process is None and not vram_before.unified:
            # no per-process figure: the device's growth may be what is measured, one load at a time
            serial.enter_context(self._device_measure)
            vram_before = self._vram(place)
        try:
            model, load_ms, vram_after = self._load_measured(spec, engine, place, device)
        finally:
            serial.close()
        grown = max(0, self._rss() - before)
        in_process = not (spec.engine.startswith("plugin:") or spec.engine in ("tts-server", "remote"))
        vram, measured, source = attribute_vram(vram_before, vram_after, in_process, place)
        report = getattr(model, "placement_report", None)
        if isinstance(report, dict) and report.get("gpu_bytes") is not None:
            # a plugin that split the model says how (accelerate's device map); more precise than a delta
            with contextlib.suppress(TypeError, ValueError):
                vram, measured, source = int(report["gpu_bytes"]), True, "plugin"
                place.gpu_bytes, place.cpu_bytes = vram, int(report.get("cpu_bytes") or 0)
        # An estimate. RSS growth during the load reads low, sometimes zero, when the allocator reuses memory
        # that an unloaded model gave back, so it is never taken to be less than the model's files on disk.
        ram = grown if vram_before.unified else grown + vram
        memory = max(ram, self.store.size_on_disk(spec.id), vram)
        if isinstance(report, dict) and report.get("gpu_bytes") is not None:
            memory = max(memory, place.gpu_bytes + place.cpu_bytes)
        return model, memory, load_ms, min(vram, memory), measured, source

    def _load_measured(
        self, spec: ModelSpec, engine: Any, place: Placement, device: str
    ) -> tuple[Model, float, VramReading]:
        """Load the model; (model, load time in ms, accelerator memory right after)."""
        t0 = time.perf_counter()
        opts = LoadOptions(
            threads=int(spec.options.get("threads", self.settings.resolved_threads())),
            device=device,
            options={
                "endpoint_silence_s": self.settings.endpoint_silence_ms / 1000.0,
                "max_utterance_s": self.settings.max_utterance_s,
                # the GPU was lost: managed servers start their CPU build with every GPU hidden
                **({"gpu_lost": True} if self.gpu.lost and family(device) == "cpu" else {}),
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
        # remote code runs only on the operator's opt-in for this model, never on a catalog entry's say-so
        if spec.engine.startswith("plugin:"):
            model_opts["allow_remote_code"] = remote_code.allowed(pins)
            if pins.get("quantization"):  # a plugin model that can load quantized (GLM-4-Voice: int4)
                model_opts["quantization"] = str(pins["quantization"])
        if remote_code.pin(spec) is not None:
            log.warning("%s", remote_code.notice(spec, pins))
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
            model_opts.setdefault("languages", list(spec.languages))
        with cuda_libs.watch() as watched:  # an accelerator the runtime did not start (bolo.cuda_libs)
            model = engine.load(spec.kind, spec.task, self.store.model_dir(spec.id), model_opts, opts)
            if isinstance(model, TTSModel):
                model.warm_up()
        if watched.fallbacks:
            place.fell_back([f.to_dict() for f in watched.fallbacks], watched.notice(), watched.whole)
        # a model that knows where it runs says so (MLX without Metal runs on the CPU)
        actual = getattr(model, "device", None)
        place.runtime = runtime_label(
            spec.engine, actual if spec.engine == "mlx" and isinstance(actual, str) else place.device, spec.options
        )
        load_ms = (time.perf_counter() - t0) * 1000.0
        return model, load_ms, self._vram(place)

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
            path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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
            idle = sorted((m for m in self.loaded.values() if not m.busy), key=lambda m: m.last_used)
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
            model, memory, load_ms, vram, measured, source = await loop.run_in_executor(
                None, self._load_blocking, spec, placement
            )
            entry = LoadedModel(spec, model, self._build_queue(spec, model), memory, load_ms)
            entry.vram_bytes, entry.vram_measured, entry.placement = vram, measured, placement
            entry.vram_source = source
            with contextlib.suppress(AttributeError):  # a model the runtime may move to the CPU says so here
                model.on_fallback = lambda fallback, e=entry: self._fell_back(e, fallback)
            self.loaded[spec.id] = entry
            log.info(
                "loaded %s in %.0f ms (~%.0f MB, %s on %s%s)",
                spec.id, load_ms, memory / 1e6, entry.processor(), placement.device,
                ", accelerator memory not measured" if entry.vram_unknown else "",
            )  # fmt: skip
            event: dict[str, Any] = {"id": spec.id, "event": "loaded", "processor": entry.processor()}
            if placement.notice:
                event["notice"] = placement.notice
            self.events.publish("model", event)
            if self.on_loaded is not None:
                with contextlib.suppress(Exception):
                    self.on_loaded(spec)
            self._ensure_reaper()
            if self.on_gpu(entry) or self.gpu.pending:
                self.gpu.ensure_running()
            return entry

    def _fell_back(self, entry: LoadedModel, fallback: dict[str, Any]) -> None:
        """A loaded model's runtime gave up its accelerator while serving and now runs on the CPU (the model said so
        through ``on_fallback``, from a worker thread): its placement, ``bolo ps`` and the console say the CPU."""
        entry.placement.fell_back(
            [*(entry.placement.provider_fallback or []), fallback], str(fallback.get("message") or ""), True
        )
        entry.vram_bytes = 0
        event = {"id": entry.id, "event": "fell_back", "processor": entry.processor(), "notice": entry.placement.notice}
        self.events.publish_threadsafe("model", event)

    async def unload(self, name: str, reason: str = "requested") -> bool:
        spec = self.catalog.resolve(name)
        if spec is None or spec.id not in self.loaded:
            return False
        entry = self.loaded.pop(spec.id)
        self.gpu.forget(spec.id)
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
        if entry.degraded and self.gpu_lost_fallback == "fail":
            raise EngineUnavailable(f"model '{entry.id}': {entry.degraded} (gpu_lost_fallback = fail)")
        ka = REQUEST_KEEP_ALIVE.get()
        if ka is not None and ka.applies(entry.spec.task):
            ka.used.add(entry.id)
            if not ka.session or ka.seconds != 0:
                entry.keep_alive_s = ka.seconds
        entry.active += 1
        entry.requests += 1
        entry.last_used = time.perf_counter()
        try:
            yield entry
        finally:
            entry.active -= 1
            entry.last_used = time.perf_counter()
            if not entry.busy and entry.keep_alive_s == 0 and self.loaded.get(entry.id) is entry:
                with contextlib.suppress(Exception):
                    await self.unload(entry.id, reason="keep_alive 0")

    async def apply_keep_alive(self, ids: Any, seconds: float) -> None:
        """At the end of a session: give the models it used this keep-alive (``0`` unloads the idle ones now)."""
        for model_id in list(ids):
            entry = self.loaded.get(model_id)
            if entry is None:
                continue
            entry.keep_alive_s = seconds
            entry.last_used = time.perf_counter()
            if seconds == 0 and not entry.busy:
                with contextlib.suppress(Exception):
                    await self.unload(model_id, reason="keep_alive 0")

    def set_keep_alive(self, entry: LoadedModel, seconds: float | None) -> None:
        """A load's ``keep_alive`` (``bolo load``, the console's "Keep loaded"); restarts the countdown."""
        if seconds is None:
            return
        entry.keep_alive_s = seconds
        entry.last_used = time.perf_counter()
        self._ensure_reaper()

    # -- idle unloading ---------------------------------------------------
    def _ensure_reaper(self) -> None:
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.create_task(self._reap_loop(), name="bolo-reaper")

    async def reap_once(self) -> list[str]:
        gone: list[str] = []
        for entry in list(self.loaded.values()):
            ttl = self.ttl_for(entry.spec, entry)
            if ttl >= 0 and not entry.busy and entry.idle_s() >= ttl:
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
        await self.gpu.close()
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
