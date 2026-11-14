"""Accelerator memory as the scheduler sees it: total, free and used memory per GPU, and how much of it this
server (and the processes it started) holds.

* NVIDIA: NVML through ``pynvml`` (the ``nvidia-ml-py`` package, optional), else ``nvidia-smi``. Both report
  per-process usage, so the memory of llama-server, audio.cpp and plugin processes bolo started is counted too.
* Apple Silicon: the GPU shares the machine's memory. "Free" is the memory the system calls available and
  per-process GPU memory cannot be told apart from ordinary memory; the figures say so (``unified: true``,
  ``process_bytes: null``).
* Anything else: nothing is reported, and the scheduler treats free accelerator memory as unknown.

Everything here is best effort and never raises; tests pass a fake NVML module or command runner. None of it has
run on a real GPU in bolo's CI (docs/architecture.md, "Placement").
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psutil

log = logging.getLogger("bolo.accel")

Runner = Callable[[list[str]], "str | None"]


@dataclass
class DeviceMemory:
    index: int
    vendor: str  # nvidia | apple
    name: str | None = None
    total_bytes: int | None = None
    free_bytes: int | None = None
    used_bytes: int | None = None
    #: the GPU shares the machine's memory (Apple Silicon): "used" and "free" are the machine's
    unified: bool = False
    #: "cuda" or "metal": the ``device`` family that reaches it
    backend: str = "cuda"
    #: percent of time the GPU was busy over the last sample period, when the driver says
    utilization: float | None = None
    #: accelerator memory held by this server and the processes it started; None when it cannot be told
    process_bytes: int | None = None
    source: str = ""

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
        }


def _family(device: str) -> str:
    """``cuda:1`` -> ``cuda``; ``mps``/``metal``/``coreml`` -> ``metal``."""
    name = (device or "cpu").strip().lower().partition(":")[0]
    if name in ("cuda", "gpu"):
        return "cuda"
    if name in ("coreml", "mps", "metal"):
        return "metal"
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
        self.max_age_s = max_age_s
        self._cache: tuple[float, list[DeviceMemory]] | None = None
        self._lock = threading.Lock()
        #: where the last reading came from: "nvml", "nvidia-smi", "unified", or "" (nothing to ask)
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
        """The accelerator a ``device`` value names (``cuda:1``, ``coreml``, ...), or None when unknown."""
        family = _family(device)
        wanted = _index(device)
        for d in self.devices(fresh=fresh):
            if d.backend == family and d.index == wanted:
                return d
        return None

    def default_device(self) -> str | None:
        """What ``device = "auto"`` means here: ``cuda:0``, ``coreml``, or None (no accelerator found)."""
        for d in self.devices():
            if d.backend == "cuda":
                return f"cuda:{d.index}"
            if d.backend == "metal":
                return "coreml"
        return None

    def close(self) -> None:
        if self._nvml_ready and self._nvml is not None:
            with contextlib.suppress(Exception):
                self._nvml.nvmlShutdown()
        self._nvml_ready = False

    # -- sources --------------------------------------------------------------------------------------------
    def _read(self) -> list[DeviceMemory]:
        found = self._from_nvml()
        if found is not None:
            self.source = "nvml"
            return found
        found = self._from_smi()
        if found is not None:
            self.source = "nvidia-smi"
            return found
        if self._system == "darwin" and self._machine in ("arm64", "aarch64"):
            self.source = "unified"
            return [self._apple()]
        self.source = ""
        return []

    def _load_nvml(self) -> Any | None:
        if not self._nvml_tried:
            self._nvml_tried = True
            try:
                import pynvml  # type: ignore[import-not-found]

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
                ours: int | None = None
                with contextlib.suppress(Exception):
                    procs = nv.nvmlDeviceGetComputeRunningProcesses(handle)
                    ours = sum(int(p.usedGpuMemory or 0) for p in procs if int(p.pid) in pids)
                out.append(
                    DeviceMemory(
                        index=i,
                        vendor="nvidia",
                        name=_text(nv.nvmlDeviceGetName(handle)),
                        total_bytes=int(mem.total),
                        free_bytes=int(mem.free),
                        used_bytes=int(mem.used),
                        utilization=util,
                        process_bytes=ours,
                        source="nvml",
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
        out = self._runner(
            [
                exe,
                "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        if out is None:
            return None
        mib = 1024 * 1024
        devices: list[DeviceMemory] = []
        uuids: dict[str, DeviceMemory] = {}
        for line in out.splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) < 7 or not cells[0].isdigit():
                continue
            d = DeviceMemory(
                index=int(cells[0]),
                vendor="nvidia",
                name=cells[2] or None,
                total_bytes=_mib(cells[3]),
                free_bytes=_mib(cells[4]),
                used_bytes=_mib(cells[5]),
                utilization=_float(cells[6]),
                source="nvidia-smi",
            )
            devices.append(d)
            uuids[cells[1]] = d
        apps = self._runner([exe, "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"])
        if apps is not None:
            pids = self._pids()
            for d in devices:
                d.process_bytes = 0
            for line in apps.splitlines():
                cells = [c.strip() for c in line.split(",")]
                if len(cells) < 3 or not cells[0].isdigit():
                    continue
                d_or_none = uuids.get(cells[1])
                used = _float(cells[2])
                if d_or_none is not None and int(cells[0]) in pids and used is not None:
                    d_or_none.process_bytes = (d_or_none.process_bytes or 0) + int(used * mib)
        return devices

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


def _float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _mib(text: str) -> int | None:
    value = _float(text)
    return int(value * 1024 * 1024) if value is not None else None
