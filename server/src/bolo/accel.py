"""Accelerator memory as the scheduler sees it: total, free and used memory per GPU, and how much of it this
server (and the processes it started) holds.

* NVIDIA: NVML through ``pynvml`` (the ``nvidia-ml-py`` package, optional), else ``nvidia-smi``. Both report
  per-process usage, so the memory of llama-server, audio.cpp and plugin processes bolo started is counted too.
  Except on Windows under the WDDM driver model (every GeForce card, and most others on Windows): there the driver
  does not account memory per process, NVML reports ``NVML_VALUE_NOT_AVAILABLE`` (or 0) and nvidia-smi "N/A".
  Such a figure is unknown, never 0: ``process_bytes`` is None, ``driver_model`` says "WDDM", and on Windows
  ``own_bytes`` carries what DXGI says this process uses on the matching adapter (``IDXGIAdapter3::
  QueryVideoMemoryInfo``, the local segment's ``CurrentUsage``). bolo.lifecycle then measures a load by this
  process's DXGI usage (in-process runtimes) or by the device's used memory, loads serialised.
* AMD on Linux (``device = "rocm:N"``): the amdgpu driver's counters in sysfs (``/sys/class/drm/card*/device``:
  ``mem_info_vram_total``, ``mem_info_vram_used``, ``gpu_busy_percent``, ``product_name``), numbered as HIP numbers
  them (PCI order, then ``HIP_VISIBLE_DEVICES`` / ``ROCR_VISIBLE_DEVICES``). The driver keeps no per-process figure
  there, so ``process_bytes`` is None. An APU (Radeon 780M and the like: a small VRAM carve-out, the rest is the
  machine's memory) is ``unified``. Read without ROCm installed; whether a runtime can use the GPU is bolo doctor's
  business (bolo.hardware). Not run on a real AMD GPU in bolo's CI (docs/rocm.md).
* Apple Silicon: the GPU shares the machine's memory. "Free" is the memory the system calls available and
  per-process GPU memory cannot be told apart from ordinary memory; the figures say so (``unified: true``,
  ``process_bytes: null``).
* Windows, DirectML (``device = "dml:N"``, read only when ``device`` is ``dml``): every hardware graphics adapter DXGI lists, numbered as DirectML numbers
  them. ``dml`` with no number is Windows' high-performance adapter (bolo.winhw.preferred_index), the one
  bolo.ort_providers runs it on, not adapter 0 (on a hybrid laptop, the integrated GPU). "Free" is the local video memory budget Windows gives this process minus what it uses
  (``IDXGIAdapter3::QueryVideoMemoryInfo``); ``process_bytes`` is this process's own use, which is where DirectML
  runs (ONNX Runtime in bolo's process). Integrated graphics report no free memory (it is the machine's).
* Anything else: nothing is reported, and the scheduler treats free accelerator memory as unknown.

NVIDIA devices are numbered as CUDA numbers them in this process: when ``CUDA_VISIBLE_DEVICES`` is set (indices or
``GPU-<uuid>`` prefixes), only the GPUs it lists are reported, in its order, as 0, 1, ...; ``physical_index`` keeps
the number NVML and nvidia-smi use. The ``gpus`` setting and ``cuda:N`` devices use the CUDA numbers.

Everything here is best effort and never raises; tests pass a fake NVML module or command runner. None of it has
run on a real GPU in bolo's CI (docs/architecture.md, "Placement").
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

log = logging.getLogger("bolo.accel")

Runner = Callable[[list[str]], "str | None"]


@dataclass
class DeviceMemory:
    index: int
    vendor: str  # nvidia | apple | amd | intel | ... (DirectML adapters)
    name: str | None = None
    total_bytes: int | None = None
    free_bytes: int | None = None
    used_bytes: int | None = None
    #: the GPU shares the machine's memory (Apple Silicon): "used" and "free" are the machine's
    unified: bool = False
    #: "cuda", "rocm", "metal" or "dml": the ``device`` family that reaches it
    backend: str = "cuda"
    #: percent of time the GPU was busy over the last sample period, when the driver says
    utilization: float | None = None
    #: accelerator memory held by this server and the processes it started; None when it cannot be told
    process_bytes: int | None = None
    source: str = ""
    #: the number NVML and nvidia-smi give the GPU (differs from ``index`` under CUDA_VISIBLE_DEVICES)
    physical_index: int | None = None
    uuid: str | None = None
    #: Windows: "WDDM", "TCC" or "MCDM" (NVML's driver model); None elsewhere or when not told
    driver_model: str | None = None
    #: this process's own use of the GPU as DXGI reports it (Windows; not the processes it started)
    own_bytes: int | None = None
    #: the PCI bus id NVML or nvidia-smi gives ("00000000:01:00.0"), used to find the DXGI adapter
    pci_bus_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "vendor": self.vendor,
            "name": self.name,
            "backend": self.backend,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "used_bytes": self.used_bytes,
            "unified": self.unified,
            "utilization": self.utilization,
            "process_bytes": self.process_bytes,
            "source": self.source,
            "physical_index": self.physical_index if self.physical_index is not None else self.index,
            "uuid": self.uuid,
            "driver_model": self.driver_model,
            "own_bytes": self.own_bytes,
        }


def _family(device: str) -> str:
    """``cuda:1`` -> ``cuda``; ``mps``/``metal``/``coreml`` -> ``metal``; ``hip``/``migraphx`` -> ``rocm``."""
    name = (device or "cpu").strip().lower().partition(":")[0]
    if name in ("cuda", "gpu"):
        return "cuda"
    if name in ("rocm", "hip", "migraphx"):
        return "rocm"
    if name in ("coreml", "mps", "metal"):
        return "metal"
    if name == "directml":
        return "dml"
    return name


def _index(device: str) -> int:
    _, _, idx = (device or "").partition(":")
    return int(idx) if idx.isdigit() else 0


def our_pids() -> set[int]:
    """This process and every process it started (llama-server, audio.cpp, plugins, probes)."""
    pids = {os.getpid()}
    with contextlib.suppress(Exception):
        pids.update(p.pid for p in psutil.Process().children(recursive=True))
    return pids


def _run(argv: list[str]) -> str | None:
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=6.0, check=False)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return proc.stdout.decode(errors="replace") if proc.returncode == 0 else None


def _text(x: Any) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


class Accelerators:
    """Reads accelerator memory, at most every ``max_age_s`` (nvidia-smi is a process start)."""

    def __init__(
        self,
        nvml: Any | None = None,
        runner: Runner | None = None,
        system: str | None = None,
        machine: str | None = None,
        pids: Callable[[], set[int]] | None = None,
        virtual_memory: Callable[[], Any] | None = None,
        max_age_s: float = 2.0,
        which: Callable[[str], str | None] | None = None,
        environ: dict[str, str] | None = None,
        adapters: Callable[[], list[Any] | None] | None = None,
        directml: bool = False,
        cuda_luids: Callable[[], dict[tuple[int, int, int, int], int] | None] | None = None,
        drm_root: str = "/sys/class/drm",
    ) -> None:
        self._nvml = nvml
        self._nvml_tried = nvml is not None
        self._nvml_ready = False
        self._runner = runner or _run
        self._which = which or shutil.which
        self._system = (system or platform.system()).lower()
        self._machine = (machine or platform.machine()).lower()
        self._pids = pids or our_pids
        self._vm = virtual_memory or psutil.virtual_memory
        self._adapters = adapters
        #: Windows: the CUDA driver's LUID per PCI address, to find the DXGI adapter an NVIDIA GPU is
        self._cuda_luids = cuda_luids
        #: DirectML's adapter for ``dml`` with no number, from the last DXGI reading
        self._dml_default: int | None = None
        #: also list DirectML devices (Windows; on when ``device`` is ``dml``, or when ``adapters`` is given)
        self.directml = directml or adapters is not None
        self.max_age_s = max_age_s
        self._cache: tuple[float, list[DeviceMemory]] | None = None
        env = os.environ if environ is None else environ
        #: CUDA_VISIBLE_DEVICES as a list (None: not set, every GPU is visible)
        self.cuda_visible: list[str] | None = (
            [x.strip() for x in env["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
            if "CUDA_VISIBLE_DEVICES" in env
            else None
        )
        #: HIP_VISIBLE_DEVICES (else ROCR_VISIBLE_DEVICES) as a list of indices (None: every AMD GPU is visible)
        hip = env.get("HIP_VISIBLE_DEVICES", env.get("ROCR_VISIBLE_DEVICES"))
        self.hip_visible: list[str] | None = (
            [x.strip() for x in hip.split(",") if x.strip()] if hip is not None else None
        )
        self._drm_root = drm_root
        self._lock = threading.Lock()
        #: where the last reading came from: "nvml", "nvidia-smi", "unified", "dxgi", "sysfs", or "" (nothing to ask)
        self.source = ""

    # -- public ---------------------------------------------------------------------------------------------
    def devices(self, fresh: bool = False) -> list[DeviceMemory]:
        with self._lock:
            now = time.monotonic()
            if not fresh and self._cache is not None and now - self._cache[0] < self.max_age_s:
                return list(self._cache[1])
            found = self._read()
            self._cache = (now, found)
            return list(found)

    def device(self, device: str, fresh: bool = False) -> DeviceMemory | None:
        """The accelerator a ``device`` value names (``cuda:1``, ``coreml``, ``dml``, ...), or None when unknown."""
        family = _family(device)
        found = self.devices(fresh=fresh)
        wanted = self.index_of(device)
        for d in found:
            if d.backend == family and d.index == wanted:
                return d
        return None

    def index_of(self, device: str) -> int:
        """The device number ``device`` means: its ``:N``, else 0, except ``dml`` with no number, which is the
        high-performance adapter (the discrete GPU of a hybrid laptop, as bolo.ort_providers uses it)."""
        _, _, idx = (device or "").partition(":")
        if not idx.isdigit() and _family(device) == "dml" and self._dml_default is not None:
            return self._dml_default
        return _index(device)

    def default_device(self) -> str | None:
        """What ``device = "auto"`` means here: ``cuda:0`` (an NVIDIA GPU first), ``rocm:0`` (an AMD GPU),
        ``coreml`` (Apple Silicon), or None (no accelerator found)."""
        found = self.devices()
        for backend, name in (("cuda", "cuda:{}"), ("rocm", "rocm:{}"), ("metal", "coreml")):
            d = next((d for d in found if d.backend == backend), None)
            if d is not None:
                return name.format(d.index)
        return None

    def close(self) -> None:
        if self._nvml_ready and self._nvml is not None:
            with contextlib.suppress(Exception):
                self._nvml.nvmlShutdown()
        self._nvml_ready = False

    # -- sources --------------------------------------------------------------------------------------------
    def cuda_devices(self) -> list[DeviceMemory]:
        """The NVIDIA GPUs this process can use, numbered as CUDA numbers them."""
        return [d for d in self.devices() if d.backend == "cuda"]

    def _visible(self, found: list[DeviceMemory]) -> list[DeviceMemory]:
        """Apply CUDA_VISIBLE_DEVICES: keep the GPUs it names, in its order, renumbered from 0."""
        for d in found:
            d.physical_index = d.index
        if self.cuda_visible is None:
            return found
        out: list[DeviceMemory] = []
        for entry in self.cuda_visible:
            if entry.startswith("-"):
                break  # CUDA stops at the first invalid entry ("-1" hides every GPU)
            match = next(
                (
                    d
                    for d in found
                    if (entry.isdigit() and d.index == int(entry))
                    or (d.uuid and not entry.isdigit() and d.uuid.lower().startswith(entry.lower()))
                ),
                None,
            )
            if match is None:
                break
            if match not in out:
                out.append(match)
        for i, d in enumerate(out):
            d.index = i
        return out

    def _read(self) -> list[DeviceMemory]:
        windows = self._system == "windows"
        want_dml = self.directml and (windows or self._adapters is not None)
        nvidia = self._from_nvml()
        if nvidia is not None:
            self.source = "nvml"
        else:
            nvidia = self._from_smi()
            self.source = "nvidia-smi" if nvidia is not None else ""
        # Windows under WDDM: the driver keeps no per-process figure; DXGI says what this process uses
        need_own = windows and any(d.process_bytes is None for d in nvidia or [])
        adapters = self._dxgi_list() if want_dml or need_own else []
        if need_own and adapters:
            self._attach_own(nvidia or [], adapters)
        dml = self._from_dxgi(adapters) if want_dml else []
        amd = self._from_amd_sysfs() if self._system == "linux" else []
        if nvidia is not None:
            return self._visible(nvidia) + amd + dml
        if amd:
            self.source = "sysfs"
            return amd
        if self._system == "darwin" and self._machine in ("arm64", "aarch64"):
            self.source = "unified"
            return [self._apple()]
        self.source = "dxgi" if dml else ""
        return dml

    def _dxgi_list(self) -> list[Any]:
        if self._adapters is None:
            from bolo import winhw

            self._adapters = winhw.dxgi_adapters
        try:
            return list(self._adapters() or [])
        except Exception as exc:
            log.debug("DXGI query failed: %s", exc)
            return []

    def _attach_own(self, nvidia: list[DeviceMemory], adapters: list[Any]) -> None:
        """Give each NVIDIA GPU this process's own use of it as DXGI reports it (``own_bytes``), through the
        adapter with the same LUID (the CUDA driver says which LUID a PCI address has), else the only NVIDIA
        adapter with the same name or memory."""
        luids: dict[tuple[int, int, int, int], int] | None = None
        if any(d.pci_bus_id for d in nvidia):
            if self._cuda_luids is None:
                from bolo import winhw

                self._cuda_luids = winhw.cuda_luids
            with contextlib.suppress(Exception):
                luids = self._cuda_luids()
        cards = [a for a in adapters if getattr(a, "vendor", "") == "nvidia" and not getattr(a, "software", False)]
        for d in nvidia:
            a = match_adapter(d, cards, luids or {}, len(nvidia))
            if a is not None and getattr(a, "usage_bytes", None) is not None:
                d.own_bytes = int(a.usage_bytes)

    def _from_dxgi(self, adapters: list[Any]) -> list[DeviceMemory]:
        """DirectML devices (Windows): the hardware adapters DXGI lists, with this process's memory budget."""
        from bolo.winhw import is_integrated, preferred_index

        self._dml_default = preferred_index(adapters)
        out: list[DeviceMemory] = []
        for a in adapters:
            integrated = is_integrated(a.vendor, a.name, a.dedicated_bytes)
            free = None
            if a.budget_bytes is not None and not integrated:
                free = max(0, a.budget_bytes - (a.usage_bytes or 0))
            out.append(
                DeviceMemory(
                    index=a.index,
                    vendor=a.vendor,
                    name=a.name,
                    total_bytes=a.dedicated_bytes or None,
                    free_bytes=free,
                    unified=integrated,
                    backend="dml",
                    process_bytes=a.usage_bytes,
                    source="dxgi",
                )
            )
        return out

    def _load_nvml(self) -> Any | None:
        if not self._nvml_tried:
            self._nvml_tried = True
            try:
                import pynvml  # type: ignore[import-not-found,import-untyped]

                self._nvml = pynvml
            except Exception:
                self._nvml = None
        if self._nvml is None:
            return None
        if not self._nvml_ready:
            try:
                self._nvml.nvmlInit()
            except Exception as exc:
                log.debug("NVML is not usable: %s", exc)
                self._nvml = None
                return None
            self._nvml_ready = True
        return self._nvml

    def _from_nvml(self) -> list[DeviceMemory] | None:
        nv = self._load_nvml()
        if nv is None:
            return None
        pids = self._pids()
        out: list[DeviceMemory] = []
        try:
            for i in range(int(nv.nvmlDeviceGetCount())):
                handle = nv.nvmlDeviceGetHandleByIndex(i)
                mem = nv.nvmlDeviceGetMemoryInfo(handle)
                util: float | None = None
                with contextlib.suppress(Exception):
                    util = float(nv.nvmlDeviceGetUtilizationRates(handle).gpu)
                model: str | None = None
                with contextlib.suppress(Exception):
                    current = nv.nvmlDeviceGetDriverModel(handle)
                    if isinstance(current, (list, tuple)):  # pynvml answers [current, pending]
                        current = current[0]
                    model = DRIVER_MODELS.get(int(current))
                uuid: str | None = None
                with contextlib.suppress(Exception):
                    uuid = _text(nv.nvmlDeviceGetUUID(handle))
                bus: str | None = None
                with contextlib.suppress(Exception):
                    bus = _text(nv.nvmlDeviceGetPciInfo(handle).busId) or None
                out.append(
                    DeviceMemory(
                        index=i,
                        vendor="nvidia",
                        name=_text(nv.nvmlDeviceGetName(handle)),
                        total_bytes=int(mem.total),
                        free_bytes=int(mem.free),
                        used_bytes=int(mem.used),
                        utilization=util,
                        process_bytes=_nvml_process_bytes(nv, handle, pids, model),
                        source="nvml",
                        uuid=uuid,
                        driver_model=model,
                        pci_bus_id=bus,
                    )
                )
        except Exception as exc:
            log.debug("NVML query failed: %s", exc)
            return out or None
        return out

    def _from_smi(self) -> list[DeviceMemory] | None:
        exe = self._which("nvidia-smi")
        if not exe:
            return None
        fields = "index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu"
        windows = self._system == "windows"
        if windows:  # driver_model is a Windows field; pci.bus_id finds the DXGI adapter
            fields += ",driver_model.current,pci.bus_id"
        out = self._runner([exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
        if out is None:
            return None
        mib = 1024 * 1024
        devices: list[DeviceMemory] = []
        uuids: dict[str, DeviceMemory] = {}
        for line in out.splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) < 7 or not cells[0].isdigit():
                continue
            model = cells[7].upper() if windows and len(cells) > 7 else None
            model = model if model in DRIVER_MODELS.values() else None
            d = DeviceMemory(
                index=int(cells[0]),
                vendor="nvidia",
                name=cells[2] or None,
                total_bytes=_mib(cells[3]),
                free_bytes=_mib(cells[4]),
                used_bytes=_mib(cells[5]),
                utilization=_float(cells[6]),
                source="nvidia-smi",
                uuid=cells[1] or None,
                driver_model=model,
                pci_bus_id=(cells[8] or None) if windows and len(cells) > 8 else None,
            )
            devices.append(d)
            uuids[cells[1]] = d
        apps = self._runner([exe, "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"])
        if apps is not None:
            pids = self._pids()
            held: dict[str, list[int | None]] = {}
            for line in apps.splitlines():
                cells = [c.strip() for c in line.split(",")]
                if len(cells) < 3 or not cells[0].isdigit() or int(cells[0]) not in pids or cells[1] not in uuids:
                    continue
                used = _float(cells[2])  # "[N/A]" under WDDM
                held.setdefault(cells[1], []).append(int(used * mib) if used is not None else None)
            for key, d in uuids.items():
                d.process_bytes = _process_total(held.get(key, []), d.driver_model)
        return devices

    def _from_amd_sysfs(self) -> list[DeviceMemory]:
        """AMD GPUs the amdgpu driver lists, in PCI order (as HIP numbers them), with ``HIP_VISIBLE_DEVICES``
        applied; [] when there are none or sysfs cannot be read."""
        root = Path(self._drm_root)
        try:
            cards = sorted(
                (c for c in root.iterdir() if c.name.startswith("card") and c.name[4:].isdigit()),
                key=lambda c: int(c.name[4:]),
            )
        except OSError:
            return []
        found: list[tuple[str, DeviceMemory]] = []
        for card in cards:
            base = card / "device"
            if _read(base / "vendor").lower() != "0x1002":
                continue
            total = _int(_read(base / "mem_info_vram_total"))
            if total is None:  # not the amdgpu driver (or an old one): nothing to measure
                continue
            used = _int(_read(base / "mem_info_vram_used"))
            name = _read(base / "product_name") or f"AMD GPU ({_read(base / 'device')})"
            busy = _int(_read(base / "gpu_busy_percent"))
            try:
                bus = base.resolve().name
            except OSError:
                bus = ""
            d = DeviceMemory(
                index=len(found),
                vendor="amd",
                name=name,
                total_bytes=total,
                free_bytes=total - used if used is not None else None,
                used_bytes=used,
                backend="rocm",
                utilization=float(busy) if busy is not None else None,
                source="sysfs",
                pci_bus_id=bus if ":" in bus else None,
            )
            if total < APU_VRAM_BYTES:  # an APU's carve-out: the GPU works from the machine's memory
                d.unified = True
                with contextlib.suppress(Exception):
                    vm = self._vm()
                    d.total_bytes, d.free_bytes = int(vm.total), int(vm.available)
                    d.used_bytes = d.total_bytes - d.free_bytes
            found.append((bus, d))
        devices = [d for _, d in found]
        for d in devices:
            d.physical_index = d.index
        if self.hip_visible is None:
            return devices
        out: list[DeviceMemory] = []
        for entry in self.hip_visible:
            if not entry.isdigit() or int(entry) >= len(devices):
                break
            if devices[int(entry)] not in out:
                out.append(devices[int(entry)])
        for i, d in enumerate(out):
            d.index = i
        return out

    def _apple(self) -> DeviceMemory:
        vm = None
        with contextlib.suppress(Exception):
            vm = self._vm()
        total = int(vm.total) if vm is not None else None
        free = int(vm.available) if vm is not None else None
        return DeviceMemory(
            index=0,
            vendor="apple",
            name="Apple GPU (unified memory)",
            total_bytes=total,
            free_bytes=free,
            used_bytes=total - free if total is not None and free is not None else None,
            unified=True,
            backend="metal",
            source="unified",
        )


#: an AMD GPU with less VRAM than this is an APU's carve-out (512 MB to 1 GB by default), not a graphics card's memory
APU_VRAM_BYTES = 2 * 1024**3


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _int(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


def _float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _mib(text: str) -> int | None:
    value = _float(text)
    return int(value * 1024 * 1024) if value is not None else None


#: NVML's nvmlDriverModel_t: WDDM (a display driver; Windows does the memory accounting), WDM (TCC), MCDM
DRIVER_MODELS = {0: "WDDM", 1: "TCC", 2: "MCDM"}
#: driver models under which the driver keeps no per-process memory figures (NVML: N/A or 0; nvidia-smi: N/A)
_NO_PROCESS_MEMORY = frozenset({"WDDM", "MCDM"})
#: NVML_VALUE_NOT_AVAILABLE as an unsigned long long: what usedGpuMemory holds when the driver cannot tell
NVML_VALUE_NOT_AVAILABLE = 2**64 - 1


def _nvml_used(value: Any, driver_model: str | None) -> int | None:
    """One process's ``usedGpuMemory``: None when NVML could not tell (``NVML_VALUE_NOT_AVAILABLE``, None, which
    pynvml turns it into, or 0 under WDDM, where the driver keeps no per-process figure)."""
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    if n < 0 or n >= NVML_VALUE_NOT_AVAILABLE:
        return None
    if n == 0 and driver_model in _NO_PROCESS_MEMORY:
        return None
    return n


def _process_total(values: list[int | None], driver_model: str | None) -> int | None:
    """What this server's processes hold on one GPU, from their per-process figures: None when any of them is
    unknown, and under WDDM unless one of them is a real figure (the driver lists processes but not their memory,
    so even "none listed" says nothing)."""
    if any(v is None for v in values):
        return None
    if driver_model in _NO_PROCESS_MEMORY and not any(values):
        return None
    return sum(v or 0 for v in values)


def _nvml_process_bytes(nv: Any, handle: Any, pids: set[int], driver_model: str | None) -> int | None:
    """Per-process memory of this server's processes on one GPU: compute processes (CUDA) and graphics processes
    (under WDDM a CUDA process may be listed as a graphics one); None when it cannot be told."""
    try:
        procs = list(nv.nvmlDeviceGetComputeRunningProcesses(handle))
    except Exception:
        return None
    with contextlib.suppress(Exception):
        procs += list(nv.nvmlDeviceGetGraphicsRunningProcesses(handle))
    by_pid: dict[int, list[int | None]] = {}
    for p in procs:
        with contextlib.suppress(Exception):
            if int(p.pid) in pids:
                by_pid.setdefault(int(p.pid), []).append(_nvml_used(getattr(p, "usedGpuMemory", None), driver_model))
    # a process listed twice (compute and graphics) counts once: its larger known figure
    values = [max((v for v in vs if v is not None), default=None) for vs in by_pid.values()]
    return _process_total(values, driver_model)


def bus_key(text: str | None) -> tuple[int, int, int, int] | None:
    """A PCI address as (domain, bus, device, function): NVML writes "00000000:01:00.0", CUDA "0000:01:00.0"."""
    m = re.search(r"([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-9a-fA-F]+)", text or "")
    if m is None:
        return None
    domain, bus, dev, fn = (int(x, 16) for x in m.groups())
    return domain, bus, dev, fn


def match_adapter(d: DeviceMemory, cards: list[Any], luids: dict[tuple[int, int, int, int], int], gpus: int) -> Any:
    """The DXGI adapter (bolo.winhw.Adapter) an NVIDIA GPU is, or None when it cannot be told for sure: by LUID
    when the CUDA driver gave one for the GPU's PCI address, else the only NVIDIA adapter with its name, else the
    only one with its memory (within 10%), else the only NVIDIA adapter when there is one GPU."""
    key = bus_key(d.pci_bus_id)
    if key is not None and key in luids:
        return next((a for a in cards if getattr(a, "luid", 0) == luids[key]), None)
    name = " ".join((d.name or "").split()).lower()
    same = [a for a in cards if " ".join(str(a.name).split()).lower() == name] if name else []
    if len(same) == 1:
        return same[0]
    total = d.total_bytes or 0
    close = [a for a in cards if total and a.dedicated_bytes and abs(total - a.dedicated_bytes) <= 0.1 * total]
    if len(close) == 1:
        return close[0]
    return cards[0] if len(cards) == 1 and gpus == 1 else None
