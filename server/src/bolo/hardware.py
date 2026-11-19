"""What this machine is: CPU, memory, disk, GPUs, the runtimes that can use them, and container limits.

Everything here is best effort. Detection never raises: a value that cannot be read is ``None``, which the API
returns as ``null`` and the command line prints as "unknown". Nothing in bolo refuses to do something because of
what is found here; it feeds recommendations and warnings only (``bolo.recommend``, docs/recommendations.md).

The outside world is read through a :class:`Host`, so tests describe a machine with a few fake files and commands.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import functools
import importlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bolo.config import Settings

BENCHMARK_FILE = "hardware-benchmark.json"
BENCHMARK_VERSION = 1

#: x86 flags in /proc/cpuinfo -> the name reported
_X86_FLAGS = {
    "sse4_2": "sse4.2",
    "avx": "avx",
    "avx2": "avx2",
    "fma": "fma",
    "f16c": "f16c",
    "avx_vnni": "avx_vnni",
    "avx512f": "avx512",
    "avx512_vnni": "avx512_vnni",
    "avx512_bf16": "avx512_bf16",
    "avx512_fp16": "avx512_fp16",
    "amx_tile": "amx",
    "amx_int8": "amx",
    "amx_bf16": "amx",
}
#: ARM "Features" in /proc/cpuinfo -> the name reported
_ARM_FLAGS = {
    "asimd": "neon",
    "neon": "neon",
    "asimddp": "dotprod",
    "asimdhp": "fp16",
    "i8mm": "i8mm",
    "bf16": "bf16",
    "sve": "sve",
    "sve2": "sve2",
}
_ARM_PARTS = {
    "0xd03": "Cortex-A53",
    "0xd04": "Cortex-A35",
    "0xd05": "Cortex-A55",
    "0xd07": "Cortex-A57",
    "0xd08": "Cortex-A72",
    "0xd09": "Cortex-A73",
    "0xd0a": "Cortex-A75",
    "0xd0b": "Cortex-A76",
    "0xd0c": "Neoverse-N1",
    "0xd0d": "Cortex-A77",
    "0xd40": "Neoverse-V1",
    "0xd41": "Cortex-A78",
    "0xd44": "Cortex-X1",
    "0xd49": "Neoverse-N2",
    "0xd4f": "Neoverse-V2",
}
_PCI_VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}
_HYPERVISOR_WORDS = (
    "kvm", "qemu", "vmware", "virtualbox", "xen", "microsoft corporation", "amazon ec2", "google", "bochs",
    "parallels", "firecracker", "openstack", "digitalocean", "hetzner",
)  # fmt: skip
_ORT_GPU_PROVIDERS = (
    "CUDAExecutionProvider", "TensorrtExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider",
    "ROCMExecutionProvider", "MIGraphXExecutionProvider", "OpenVINOExecutionProvider",
)  # fmt: skip


class ProbeCache:
    """The answers of the slow probes (``nvidia-smi``, ``llama-server --version`` and ``--list-devices``, the torch
    plugin's ``doctor``, CTranslate2's CUDA device count), kept for ``ttl_s`` so that asking for the hardware again
    (``GET /api/v1/hardware?refresh=1``) does not run them every time: on a Windows laptop they took about 3 s per
    request. :meth:`clear` forgets them (a component was installed, or ``refresh=force``)."""

    def __init__(self, ttl_s: float = 60.0, clock: Any = time.monotonic) -> None:
        self.ttl_s = ttl_s
        self._clock = clock
        self._entries: dict[tuple[Any, ...], tuple[float, Any]] = {}
        self._lock = threading.Lock()
        #: how many probes really ran (tests)
        self.misses = 0

    def get(self, key: tuple[Any, ...], compute: Any) -> Any:
        now = self._clock()
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None and now - hit[0] < self.ttl_s:
                return hit[1]
        value = compute()  # not under the lock: a probe can take seconds
        with self._lock:
            self._entries[key] = (self._clock(), value)
            self.misses += 1
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def _stamp(path: str | os.PathLike[str] | None) -> tuple[Any, ...]:
    """What identifies a binary for :class:`ProbeCache`: its path, size and modification time (a new build at the
    same path is probed afresh)."""
    if not path:
        return (None,)
    try:
        st = Path(path).stat()
    except OSError:
        return (str(path),)
    return (str(path), st.st_size, st.st_mtime_ns)


class Host:
    """The machine as detection sees it. Tests subclass this; every method may return None for "cannot tell"."""

    #: the slow probes' answers, when detection may reuse them (:class:`ProbeCache`); None asks every time
    cache: ProbeCache | None = None

    def __init__(
        self, root: str | os.PathLike[str] = "/", env: dict[str, str] | None = None, cache: ProbeCache | None = None
    ) -> None:
        self.root = Path(root)
        self.env = dict(os.environ) if env is None else dict(env)
        self.cache = cache

    def cached(self, key: tuple[Any, ...], compute: Any) -> Any:
        """``compute()``, or its answer from :attr:`cache` while that is fresh."""
        return compute() if self.cache is None else self.cache.get(key, compute)

    def system(self) -> str:
        return platform.system().lower()

    def machine(self) -> str:
        return platform.machine().lower()

    def processor(self) -> str | None:
        return platform.processor() or None

    def _path(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def read(self, path: str) -> str | None:
        try:
            return self._path(path).read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            return None

    def exists(self, path: str) -> bool:
        try:
            return self._path(path).exists()
        except OSError:
            return False

    def listdir(self, path: str) -> list[str]:
        try:
            return sorted(p.name for p in self._path(path).iterdir())
        except OSError:
            return []

    def run(self, argv: list[str], timeout: float = 4.0) -> str | None:
        """The command's standard output, or None when it is missing, fails or takes too long."""
        try:
            proc = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.decode(errors="replace")

    def probe(
        self, argv: list[str], timeout: float, env: dict[str, str] | None = None
    ) -> tuple[str | None, str | None]:
        """``(output, None)`` of a command that asks a runtime about the machine (``llama-server
        --list-devices``), or ``(None, why)``: it is missing, failed, or did not answer within ``timeout`` (then
        it is killed, and not waited for beyond a few seconds: a process stuck in a driver may not die at once)."""
        try:
            proc = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env={**os.environ, **env} if env else None,
            )  # fmt: skip
        except (OSError, ValueError) as exc:
            return None, f"{type(exc).__name__}: {exc}"
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.communicate(timeout=5.0)
            return None, f"did not answer within {timeout:.0f} s"
        if proc.returncode != 0:
            tail = (err or out or b"").decode(errors="replace").strip().splitlines()[-1:]
            return None, f"exit code {proc.returncode}" + (f": {tail[0][:200]}" if tail else "")
        return out.decode(errors="replace"), None

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def module(self, name: str) -> Any | None:
        try:
            return importlib.import_module(name)
        except Exception:
            return None

    def logical_cores(self) -> int | None:
        return os.cpu_count()

    def windows_cpu(self) -> tuple[str | None, list[str] | None]:
        """Windows: the CPU's name (registry) and instruction sets (IsProcessorFeaturePresent)."""
        from bolo import winhw

        return winhw.cpu_name(), winhw.cpu_features(self.machine())

    def windows_adapters(self) -> list[Any] | None:
        """Windows: the hardware graphics adapters DXGI lists (bolo.winhw.Adapter), None when it cannot be asked."""
        from bolo import winhw

        return winhw.dxgi_adapters()

    def physical_cores(self) -> int | None:
        try:
            import psutil

            return psutil.cpu_count(logical=False)
        except Exception:
            return None

    def affinity(self) -> int | None:
        """How many CPUs this process may run on (taskset, cpuset), where the system says."""
        if sys.platform in ("win32", "darwin"):  # no sched_getaffinity
            return None
        try:
            return len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            return None

    def memory(self) -> tuple[int | None, int | None]:
        """``(total, available)`` bytes from the operating system."""
        try:
            import psutil

            vm = psutil.virtual_memory()
            return int(vm.total), int(vm.available)
        except Exception:
            return None, None

    def disk(self, path: Path) -> tuple[int | None, int | None]:
        """``(total, free)`` bytes of the file system that holds ``path`` (or its nearest existing parent)."""
        probe = path
        for _ in range(64):
            if probe.exists() or probe.parent == probe:
                break
            probe = probe.parent
        try:
            usage = shutil.disk_usage(probe)
        except OSError:
            return None, None
        return int(usage.total), int(usage.free)


@dataclass
class Cpu:
    model: str | None = None
    vendor: str | None = None
    architecture: str | None = None
    physical_cores: int | None = None
    logical_cores: int | None = None
    #: cores this process can really use: the smallest of the logical cores, the CPU affinity and the cgroup quota
    usable_cores: float | None = None
    #: instruction sets that matter for inference speed, e.g. ["avx2", "avx512", "avx512_vnni", "amx"] or ["neon"]
    features: list[str] | None = None
    #: the operating system's processor identifier (``platform.processor()``; on Windows "Intel64 Family 6 Model
    #: 154 Stepping 3, GenuineIntel"): steadier than ``model``, whose source changed in bolo 0.6 on Windows.
    #: Only for recognising a stored benchmark; not in :meth:`to_dict`
    processor_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "vendor": self.vendor,
            "architecture": self.architecture,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "usable_cores": self.usable_cores,
            "features": self.features,
        }


@dataclass
class Memory:
    total_bytes: int | None = None
    available_bytes: int | None = None
    #: a cgroup memory limit below the machine's memory (a container), else None
    limit_bytes: int | None = None

    @property
    def usable_bytes(self) -> int | None:
        """What a model can count on at most: the machine's memory, or the container's limit when that is lower."""
        if self.total_bytes is None:
            return self.limit_bytes
        return min(self.total_bytes, self.limit_bytes) if self.limit_bytes else self.total_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "limit_bytes": self.limit_bytes,
            "usable_bytes": self.usable_bytes,
        }


@dataclass
class Disk:
    path: str | None = None
    total_bytes: int | None = None
    free_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "total_bytes": self.total_bytes, "free_bytes": self.free_bytes}


@dataclass
class Gpu:
    vendor: str  # nvidia | apple | amd | intel | unknown
    name: str | None = None
    vram_total_bytes: int | None = None
    vram_free_bytes: int | None = None
    driver: str | None = None
    #: the GPU shares the machine's memory (Apple Silicon, integrated graphics)
    unified_memory: bool = False
    #: how inference runtimes reach it: "cuda", "rocm" (an AMD GPU with ROCm installed), "metal", "dml" (DirectML),
    #: or None when no runtime bolo uses can
    backend: str | None = None
    source: str = ""  # nvidia-smi | pynvml | sysfs | platform | dxgi
    #: integrated graphics in a PC (Intel UHD/Iris, an AMD APU's Radeon, <= 512 MB dedicated): never the best GPU
    integrated: bool = False
    #: Windows: DirectML's ``device_id`` for this card (``device = "dml:N"``), also for an NVIDIA card bolo reaches
    #: through CUDA
    dml_index: int | None = None
    #: Windows: the adapter ``device = "dml"`` (no number) runs on, Windows' high-performance GPU
    dml_preferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "name": self.name,
            "vram_total_bytes": self.vram_total_bytes,
            "vram_free_bytes": self.vram_free_bytes,
            "driver": self.driver,
            "unified_memory": self.unified_memory,
            "backend": self.backend,
            "source": self.source,
            "integrated": self.integrated,
            "dml_index": self.dml_index,
            "dml_preferred": self.dml_preferred,
        }


@dataclass
class Container:
    #: True in a container, False when nothing says so, None when it cannot be told (not Linux)
    detected: bool | None = None
    kind: str | None = None  # docker | podman | kubernetes | lxc | container
    virtual_machine: bool | None = None
    hypervisor: str | None = None
    #: CPUs' worth of time the cgroup allows (2.0 = two cores), None without a quota
    cpu_quota: float | None = None
    memory_limit_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "kind": self.kind,
            "virtual_machine": self.virtual_machine,
            "hypervisor": self.hypervisor,
            "cpu_quota": self.cpu_quota,
            "memory_limit_bytes": self.memory_limit_bytes,
        }


@dataclass
class Hardware:
    system: str | None = None  # linux | macos | windows
    cpu: Cpu = field(default_factory=Cpu)
    memory: Memory = field(default_factory=Memory)
    disk: Disk = field(default_factory=Disk)
    gpus: list[Gpu] = field(default_factory=list)
    #: False when GPU detection itself could not run; an empty ``gpus`` then means "unknown", not "none"
    gpus_known: bool = True
    #: runtime name -> what it can use here (see ``detect_runtimes``)
    runtimes: dict[str, dict[str, Any]] = field(default_factory=dict)
    container: Container = field(default_factory=Container)
    #: the ``device`` setting and the inference threads bolo uses per worker on this machine
    device: str = "cpu"
    #: False when the ``device`` setting was left at its default (bolo.config.Settings.device_explicit)
    device_explicit: bool = True
    threads: int | None = None
    notes: list[str] = field(default_factory=list)
    detected_at: str = ""
    #: True when slow probes ran too (the torch plugin's device, server binaries' versions)
    deep: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "cpu": self.cpu.to_dict(),
            "memory": self.memory.to_dict(),
            "disk": self.disk.to_dict(),
            "gpus": [g.to_dict() for g in self.gpus],
            "gpus_known": self.gpus_known,
            "runtimes": self.runtimes,
            "container": self.container.to_dict(),
            "device": self.device,
            "threads": self.threads,
            "notes": list(self.notes),
            "detected_at": self.detected_at,
            "deep": self.deep,
        }

    def best_gpu(self, backend: str | None = None) -> Gpu | None:
        """The GPU with the most memory that a runtime can reach (optionally through one backend). Integrated
        graphics never count: on a hybrid laptop the discrete card is the one to use, and an integrated GPU alone
        is rarely faster than the CPU for bolo's models."""
        usable = [g for g in self.gpus if g.backend and not g.integrated and (backend is None or g.backend == backend)]
        return max(usable, key=lambda g: g.vram_total_bytes or 0) if usable else None

    def summary(self) -> str:
        """One line for people: "4 cores (Intel Xeon, AVX-512), 15.6 GB RAM, no GPU"."""
        cpu = self.cpu
        cores = _num(cpu.usable_cores if cpu.usable_cores is not None else cpu.logical_cores)
        isa = _isa_label(cpu.features)
        what = ", ".join(x for x in (cpu.model or cpu.architecture, isa) if x)
        parts = [f"{cores} core{'' if cores == '1' else 's'}" + (f" ({what})" if what else "")]
        usable = self.memory.usable_bytes
        parts.append(f"{gigabytes(usable)} RAM" if usable else "RAM unknown")
        if self.gpus:
            for g in self.gpus:
                mem = "shared memory" if g.unified_memory else (gigabytes(g.vram_total_bytes) or "VRAM unknown")
                parts.append(f"{g.name or g.vendor + ' GPU'} ({mem})")
        else:
            parts.append("no GPU found" if self.gpus_known else "GPU unknown")
        if self.container.cpu_quota is not None:
            parts.append(f"CPU quota {_num(self.container.cpu_quota)}")
        return ", ".join(parts)


def gigabytes(n: float | None) -> str:
    """ "15.6 GB"; "" for unknown."""
    if n is None:
        return ""
    gb = n / 1024**3
    return f"{gb:.0f} GB" if gb >= 100 else f"{gb:.1f} GB"


def _num(x: float | None) -> str:
    if x is None:
        return "unknown"
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def _isa_label(features: list[str] | None) -> str:
    if not features:
        return ""
    for key, label in (("amx", "AMX"), ("avx512", "AVX-512"), ("avx2", "AVX2"), ("avx", "AVX"), ("sve", "SVE")):
        if key in features:
            return label
    return "NEON" if "neon" in features else ""


def _int(text: str | None) -> int | None:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def _uniq(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


# -- CPU --------------------------------------------------------------------------------------------------------
def _arch(machine: str) -> str | None:
    m = machine.lower()
    if not m:
        return None
    if m in ("x86_64", "amd64", "x64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "arm64"
    return m


def _cpuinfo(text: str) -> list[dict[str, str]]:
    """/proc/cpuinfo as one dict per processor block (keys lowercased)."""
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if current:
                blocks.append(current)
                current = {}
            continue
        key, sep, value = line.partition(":")
        if sep:
            current[key.strip().lower()] = value.strip()
    if current:
        blocks.append(current)
    return blocks


def _linux_cpu(host: Host, cpu: Cpu) -> bool:
    """Fill ``cpu`` from /proc/cpuinfo; True when the CPU says it runs under a hypervisor."""
    text = host.read("/proc/cpuinfo")
    if not text:
        return False
    blocks = _cpuinfo(text)
    if not blocks:
        return False
    first = blocks[0]
    merged: dict[str, str] = {}
    for block in blocks:
        for key, value in block.items():
            merged.setdefault(key, value)
    cpu.model = merged.get("model name") or merged.get("model") or merged.get("hardware") or merged.get("cpu model")
    if not cpu.model:
        model = host.read("/sys/firmware/devicetree/base/model")
        part = _ARM_PARTS.get(merged.get("cpu part", "").lower())
        cpu.model = (model or "").strip("\x00\n ") or part
    cpu.vendor = merged.get("vendor_id") or ({"0x41": "ARM", "0x61": "Apple"}.get(merged.get("cpu implementer", "")))
    flags = (first.get("flags") or merged.get("flags") or "").split()
    arm = (first.get("features") or merged.get("features") or "").split()
    feats: list[str] = [_X86_FLAGS[f] for f in flags if f in _X86_FLAGS]
    feats += [_ARM_FLAGS[f] for f in arm if f in _ARM_FLAGS]
    if flags or arm:
        cpu.features = _uniq(feats)
    cores = {(b.get("physical id", "0"), b["core id"]) for b in blocks if "core id" in b}
    if cores:
        cpu.physical_cores = len(cores)
    if cpu.logical_cores is None:
        cpu.logical_cores = sum(1 for b in blocks if "processor" in b) or None
    return "hypervisor" in flags


def _sysctl(host: Host, name: str) -> str | None:
    out = host.run(["sysctl", "-n", name])
    return out.strip() if out and out.strip() else None


def _mac_cpu(host: Host, cpu: Cpu) -> None:
    cpu.model = _sysctl(host, "machdep.cpu.brand_string")
    cpu.physical_cores = _int(_sysctl(host, "hw.physicalcpu")) or cpu.physical_cores
    cpu.logical_cores = _int(_sysctl(host, "hw.logicalcpu")) or cpu.logical_cores
    if cpu.architecture == "arm64":
        cpu.vendor = "Apple"
        cpu.features = ["neon", "dotprod", "fp16"]
        return
    text = " ".join(
        x for x in (_sysctl(host, "machdep.cpu.features"), _sysctl(host, "machdep.cpu.leaf7_features")) if x
    ).lower()
    if text:
        names = {
            "sse4.2": "sse4.2", "avx1.0": "avx", "avx2": "avx2", "fma": "fma", "avx512f": "avx512",
            "avx512vnni": "avx512_vnni",
        }  # fmt: skip
        cpu.features = [label for token, label in names.items() if token in text.split()]


def detect_cpu(host: Host) -> tuple[Cpu, bool]:
    """The CPU, and whether it reports a hypervisor."""
    cpu = Cpu(architecture=_arch(host.machine()), logical_cores=host.logical_cores())
    hypervisor = False
    system = host.system()
    if system == "linux":
        hypervisor = _linux_cpu(host, cpu)
    elif system == "darwin":
        _mac_cpu(host, cpu)
    elif system == "windows":
        name, features = host.windows_cpu()
        cpu.model = name or cpu.model
        if features is not None:
            cpu.features = features
        if name:
            lowered = name.lower()
            cpu.vendor = "GenuineIntel" if "intel" in lowered else "AuthenticAMD" if "amd" in lowered else None
    processor = host.processor()
    cpu.processor_id = processor.strip() if processor else None
    if cpu.model is None:
        cpu.model = processor
    if cpu.physical_cores is None:
        cpu.physical_cores = host.physical_cores()
    if cpu.features is None and cpu.architecture == "arm64":
        cpu.features = ["neon"]  # every 64-bit ARM core has it
    return cpu, hypervisor


def int8_may_saturate(cpu: Cpu) -> bool:
    """Whether ONNX Runtime may compute int8 matrix products wrongly here: an x86 CPU without AVX-512 VNNI, where
    it multiplies with AVX2's saturating instructions (an x86 CPU whose features cannot be read counts as one).

    Seen with sherpa-onnx's int8 Whisper turbo decoder: on an AMD EPYC 7763 (AVX2, no VNNI) it gave Telugu,
    Bengali and Tamil text that turned to nonsense, while the same files gave the right text on Intel Xeons with
    AVX-512 VNNI and on ARM."""
    return cpu.architecture == "x86_64" and "avx512_vnni" not in (cpu.features or [])


@functools.cache
def this_cpu() -> Cpu:
    """This machine's CPU (read once)."""
    try:
        return detect_cpu(Host())[0]
    except Exception:
        return Cpu(architecture=_arch(platform.machine()))


# -- memory, cgroups, containers ----------------------------------------------------------------------------------
def _meminfo(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        fields = rest.split()
        if sep and fields:
            value = _int(fields[0])
            if value is not None:
                out[key.strip()] = value * (1024 if len(fields) > 1 and fields[1].lower() == "kb" else 1)
    return out


def detect_memory(host: Host) -> Memory:
    mem = Memory()
    text = host.read("/proc/meminfo") if host.system() == "linux" else None
    if text:
        info = _meminfo(text)
        mem.total_bytes = info.get("MemTotal")
        mem.available_bytes = info.get("MemAvailable")
        if mem.available_bytes is None and "MemFree" in info:
            mem.available_bytes = info["MemFree"] + info.get("Buffers", 0) + info.get("Cached", 0)
    if mem.total_bytes is None and host.system() == "darwin":
        mem.total_bytes = _int(_sysctl(host, "hw.memsize"))
    if mem.total_bytes is None or mem.available_bytes is None:
        total, available = host.memory()
        mem.total_bytes = mem.total_bytes if mem.total_bytes is not None else total
        mem.available_bytes = mem.available_bytes if mem.available_bytes is not None else available
    return mem


def _cgroup_cpu_quota(host: Host) -> float | None:
    text = host.read("/sys/fs/cgroup/cpu.max")  # cgroup v2: "<quota|max> <period>"
    if text:
        parts = text.split()
        if len(parts) >= 2 and parts[0] != "max":
            quota, period = _int(parts[0]), _int(parts[1])
            if quota and period and quota > 0:
                return round(quota / period, 2)
        return None
    for base in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):  # cgroup v1
        quota = _int(host.read(f"{base}/cpu.cfs_quota_us"))
        period = _int(host.read(f"{base}/cpu.cfs_period_us"))
        if quota and period and quota > 0:
            return round(quota / period, 2)
    return None


def _cgroup_memory_limit(host: Host) -> int | None:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        text = host.read(path)
        if text is None:
            continue
        limit = _int(text)
        # "max", or cgroup v1's "no limit" (a number near 2**63), is no limit
        if limit is not None and 0 < limit < 1 << 60:
            return limit
    return None


def detect_container(host: Host, hypervisor_flag: bool = False) -> Container:
    box = Container()
    if host.system() != "linux":
        return box
    kind: str | None = None
    cgroup = (host.read("/proc/1/cgroup") or "") + (host.read("/proc/self/cgroup") or "")
    if host.env.get("KUBERNETES_SERVICE_HOST") or "kubepods" in cgroup:
        kind = "kubernetes"
    elif host.exists("/.dockerenv") or "docker" in cgroup:
        kind = "docker"
    elif host.exists("/run/.containerenv") or host.env.get("container") == "podman":
        kind = "podman"
    elif "lxc" in cgroup or host.env.get("container") == "lxc":
        kind = "lxc"
    elif host.env.get("container") or "containerd" in cgroup:
        kind = "container"
    box.detected = kind is not None
    box.kind = kind
    box.cpu_quota = _cgroup_cpu_quota(host)
    box.memory_limit_bytes = _cgroup_memory_limit(host)
    vendor = " ".join(
        (host.read(p) or "").strip() for p in ("/sys/class/dmi/id/sys_vendor", "/sys/class/dmi/id/product_name")
    ).strip()
    named = next((w for w in _HYPERVISOR_WORDS if w in vendor.lower()), None)
    if hypervisor_flag or named:
        box.virtual_machine = True
        box.hypervisor = vendor or None
    elif vendor:
        box.virtual_machine = False
    return box


# -- GPUs -------------------------------------------------------------------------------------------------------
def _nvidia_smi(host: Host) -> list[Gpu] | None:
    """NVIDIA GPUs from nvidia-smi; None when the tool is not there or does not answer."""
    exe = host.which("nvidia-smi")
    if not exe:
        return None
    argv = [exe, "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"]
    out = host.cached(("nvidia-smi", *_stamp(exe)), lambda: host.run(argv, timeout=6.0))
    if out is None:
        return None
    gpus: list[Gpu] = []
    for line in out.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 4 or not cells[0]:
            continue
        total, free = _int(cells[1]), _int(cells[2])
        gpus.append(
            Gpu(
                vendor="nvidia",
                name=cells[0],
                vram_total_bytes=total * 1024**2 if total is not None else None,
                vram_free_bytes=free * 1024**2 if free is not None else None,
                driver=cells[3] or None,
                backend="cuda",
                source="nvidia-smi",
            )
        )
    return gpus


def _pynvml(host: Host) -> list[Gpu] | None:
    nv = host.module("pynvml")
    if nv is None:
        return None
    try:
        nv.nvmlInit()
    except Exception:
        return None
    gpus: list[Gpu] = []
    try:
        driver = nv.nvmlSystemGetDriverVersion()
        for i in range(int(nv.nvmlDeviceGetCount())):
            handle = nv.nvmlDeviceGetHandleByIndex(i)
            name = nv.nvmlDeviceGetName(handle)
            mem = nv.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                Gpu(
                    vendor="nvidia",
                    name=name.decode() if isinstance(name, bytes) else str(name),
                    vram_total_bytes=int(mem.total),
                    vram_free_bytes=int(mem.free),
                    driver=driver.decode() if isinstance(driver, bytes) else str(driver),
                    backend="cuda",
                    source="pynvml",
                )
            )
    except Exception:
        return gpus or None
    finally:
        with contextlib.suppress(Exception):
            nv.nvmlShutdown()
    return gpus


#: ROCm's version file, and the programs a ROCm installation puts on PATH
_ROCM_VERSION_FILES = ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev")
_ROCM_PROGRAMS = ("rocminfo", "amd-smi", "rocm-smi", "hipconfig")


def rocm_runtime(host: Host) -> dict[str, Any]:
    """Whether ROCm (AMD's GPU compute stack, Linux) is installed: ``{"installed", "version", "path"}``. Found from
    ``/opt/rocm/.info/version`` or one of its programs on PATH; ROCM_PATH (or HIP_PATH) names another location."""
    roots = [r for r in (host.env.get("ROCM_PATH"), host.env.get("HIP_PATH")) if r]
    files = [f"{r.rstrip('/')}/.info/version" for r in roots] + list(_ROCM_VERSION_FILES)
    for f in files:
        text = host.read(f)
        if text and text.strip():
            return {"installed": True, "version": text.strip().splitlines()[0], "path": f.rsplit("/.info/", 1)[0]}
    for name in _ROCM_PROGRAMS:
        found = host.which(name)
        if found:
            return {"installed": True, "version": None, "path": found}
    return {"installed": False, "version": None, "path": None}


#: an AMD GPU with less VRAM than this is an APU's carve-out; its GPU works from the machine's memory
_APU_VRAM_BYTES = 2 * 1024**3


def _sysfs_gpus(host: Host, have_nvidia: bool) -> list[Gpu]:
    """Graphics cards the kernel lists (/sys/class/drm): AMD with its VRAM counters (reached through ROCm when it
    is installed), Intel, NVIDIA without tools."""
    gpus: list[Gpu] = []
    rocm: bool | None = None
    for card in host.listdir("/sys/class/drm"):
        if not card.startswith("card") or not card[4:].isdigit():
            continue
        base = f"/sys/class/drm/{card}/device"
        vendor_id = (host.read(f"{base}/vendor") or "").strip().lower()
        vendor = _PCI_VENDORS.get(vendor_id)
        if vendor is None or (vendor == "nvidia" and have_nvidia):
            continue
        device = (host.read(f"{base}/device") or "").strip()
        total = _int(host.read(f"{base}/mem_info_vram_total"))
        used = _int(host.read(f"{base}/mem_info_vram_used"))
        label = {"amd": "AMD graphics", "intel": "Intel graphics", "nvidia": "NVIDIA GPU (no driver tools found)"}
        product = (host.read(f"{base}/product_name") or "").strip() if vendor == "amd" else ""
        backend = None
        if vendor == "amd" and total is not None:
            if rocm is None:
                rocm = bool(rocm_runtime(host)["installed"])
            # ROCm builds of ONNX Runtime (MIGraphX), llama.cpp (HIP) and PyTorch use it; without ROCm only a
            # Vulkan build of llama-server can (device = "vulkan")
            backend = "rocm" if rocm else None
        gpus.append(
            Gpu(
                vendor=vendor,
                name=product or f"{label[vendor]}{f' ({device})' if device else ''}",
                vram_total_bytes=total,
                vram_free_bytes=total - used if total is not None and used is not None else None,
                unified_memory=vendor == "intel" or (vendor == "amd" and total is not None and total < _APU_VRAM_BYTES),
                backend=backend,
                source="sysfs",
            )
        )
    return gpus


def detect_gpus(host: Host, cpu: Cpu, memory: Memory) -> tuple[list[Gpu], bool]:
    """``(gpus, known)``: ``known`` is False when nothing could be asked at all."""
    system = host.system()
    gpus: list[Gpu] = []
    nvidia = _nvidia_smi(host)
    if nvidia is None:
        nvidia = _pynvml(host)
    gpus += nvidia or []
    if system == "linux":
        gpus += _sysfs_gpus(host, have_nvidia=bool(nvidia))
        return gpus, True
    if system == "darwin":
        if cpu.architecture == "arm64":
            chip = cpu.model or "Apple Silicon"
            gpus.append(
                Gpu(
                    vendor="apple",
                    name=f"{chip} GPU",
                    vram_total_bytes=memory.total_bytes,
                    vram_free_bytes=memory.available_bytes,
                    unified_memory=True,
                    backend="metal",
                    source="platform",
                )
            )
        return gpus, True
    if system == "windows":
        adapters = host.windows_adapters()
        if adapters is None:
            return gpus, nvidia is not None
        gpus += _dxgi_gpus(host, adapters, nvidia or [])
        return gpus, True
    return gpus, nvidia is not None


def _dxgi_gpus(host: Host, adapters: list[Any], nvidia: list[Gpu]) -> list[Gpu]:
    """Windows graphics adapters as DXGI lists them: what DirectML (``device = "dml:N"``) can use when the
    DirectML build of ONNX Runtime is installed. One entry per card: an NVIDIA card already read through
    NVML/nvidia-smi is not listed twice, it gets its DirectML number (``dml_index``) instead."""
    from bolo.winhw import is_integrated, preferred_index

    preferred = preferred_index(adapters)
    ort = host.module("onnxruntime")
    dml = False
    if ort is not None:
        with contextlib.suppress(Exception):
            dml = "DmlExecutionProvider" in ort.get_available_providers()
    gpus: list[Gpu] = []
    unmatched = [g for g in nvidia if g.dml_index is None]
    for a in adapters:
        if a.vendor == "nvidia" and nvidia:
            match = _same_card(a, unmatched)
            if match is not None:
                unmatched.remove(match)
                match.dml_index, match.dml_preferred = a.index, a.index == preferred
                continue
        integrated = is_integrated(a.vendor, a.name, a.dedicated_bytes)
        gpus.append(
            Gpu(
                vendor=a.vendor if a.vendor in ("nvidia", "amd", "intel") else "unknown",
                name=f"{a.name} (dml:{a.index})",
                vram_total_bytes=a.dedicated_bytes or None,
                vram_free_bytes=(
                    max(0, a.budget_bytes - (a.usage_bytes or 0))
                    if a.budget_bytes is not None and not integrated
                    else None
                ),
                unified_memory=integrated,
                backend="dml" if dml else None,
                source="dxgi",
                integrated=integrated,
                dml_index=a.index,
                dml_preferred=a.index == preferred,
            )
        )
    return gpus


def _same_card(adapter: Any, nvidia: list[Gpu]) -> Gpu | None:
    """The NVIDIA GPU (from NVML/nvidia-smi) that a DXGI adapter is: the same name, else memory within 10%, else
    the first one left (both list NVIDIA cards in PCI order)."""
    name = " ".join(str(adapter.name).split()).lower()
    for g in nvidia:
        if g.name and " ".join(g.name.split()).lower() == name:
            return g
    for g in nvidia:
        total = g.vram_total_bytes
        if total and adapter.dedicated_bytes and abs(total - adapter.dedicated_bytes) <= 0.1 * total:
            return g
    return nvidia[0] if nvidia else None


# -- runtimes ---------------------------------------------------------------------------------------------------
def _sherpa_gpu(mod: Any) -> bool | None:
    """Whether the sherpa-onnx module is a CUDA build (None: cannot tell)."""
    from bolo.ort_providers import sherpa_build

    try:
        where = getattr(mod, "__file__", None)
        return sherpa_build(Path(where).parent if where else None, [_version(mod) or ""])
    except Exception:
        return None


def _version(mod: Any) -> str | None:
    version = getattr(mod, "__version__", None)
    return str(version) if version else None


def _server_runtime(host: Host, settings: Settings | None, runtime: str, deep: bool) -> dict[str, Any]:
    from bolo import tts_runtimes

    info = tts_runtimes.RUNTIME_INFO[runtime]
    path: Path | None = None
    source: str | None = None
    if settings is not None:
        with contextlib.suppress(Exception):
            path, source = tts_runtimes.find_binary(settings, runtime)
    else:
        found = host.which(info.binary)
        path, source = (Path(found), "path") if found else (None, None)
    out: dict[str, Any] = {
        "installed": path is not None,
        "path": str(path) if path else None,
        "source": source,
        "version": None,
        # whether this build can use a GPU: True, False, or None for "cannot tell"
        "gpu": None,
        "gpu_devices": None,
    }
    if path is None:
        return out
    if source == "bolo":
        # the release bolo installs: CPU builds on Linux, Metal on Apple Silicon, and on Windows the CPU, CUDA or
        # Vulkan build, each in its own folder (bolo.tts_runtimes)
        out["version"] = info.version
        builds = tts_runtimes.installed_flavours(settings, runtime) if settings is not None else {}
        out["flavour"] = next((f for f, p in builds.items() if p == path), None)
        metal = host.system() == "darwin" and _arch(host.machine()) == "arm64"
        out["gpu"] = metal or out["flavour"] in ("cuda", "vulkan", "rocm")
    if deep:
        version = host.cached(("version", *_stamp(path)), lambda: host.run([str(path), "--version"], timeout=15.0))
        if version and version.strip():
            out["version"] = version.strip().splitlines()[0][:80]
        if runtime == tts_runtimes.LLAMA:
            flavour = out.get("flavour") or tts_runtimes.flavour_of(settings, runtime, path)
            env = tts_runtimes.child_env(flavour) if host.system() == "windows" else {}
            devices, why = host.cached(
                ("list-devices", *_stamp(path), tuple(sorted((env or {}).items()))),
                lambda: host.probe(
                    [str(path), "--list-devices"], timeout=list_devices_timeout(host.system()), env=env or None
                ),
            )
            if why is not None:
                hint = ""
                if flavour == "vulkan":
                    hint = (
                        "; a Vulkan build that hangs here: on a laptop with AMD and NVIDIA graphics set "
                        "DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1=1 (bolo sets it for the servers it starts when "
                        "it sees an AMD adapter), or use the CPU or CUDA build"
                    )
                out["probe_error"] = f"`{path.name} --list-devices` {why}{hint}"
            if devices is not None:
                names = [ln.strip() for ln in devices.splitlines() if _DEVICE_LINE.match(ln)]
                out["gpu_devices"] = names
                out["gpu"] = bool(names)
    return out


#: a device line of ``llama-server --list-devices`` (``  Vulkan0: NVIDIA GeForce RTX 3060 (6144 MiB, 5120 MiB free)``)
_DEVICE_LINE = re.compile(r"^\s*(CUDA|Vulkan|Metal|MTL|SYCL|ROCm|HIP|OpenCL|BLAS|RPC)\d*\s*:", re.IGNORECASE)

#: how long ``llama-server --list-devices`` may take (it starts every GPU backend of the build)
LIST_DEVICES_TIMEOUT_S = 20.0
#: macOS: Metal loads (and compiles) its shader library the first time a process uses it, and the first
#: ``--list-devices`` took 11 s and more there
LIST_DEVICES_TIMEOUT_MACOS_S = 60.0


def list_devices_timeout(system: str) -> float:
    """How long ``--list-devices`` may take on ``system`` (``Host.system()``: "darwin", "linux", "windows")."""
    return LIST_DEVICES_TIMEOUT_MACOS_S if system == "darwin" else LIST_DEVICES_TIMEOUT_S


def _torch_plugin(host: Host, settings: Settings | None, deep: bool) -> dict[str, Any]:
    command: list[str] | None = None
    if settings is not None:
        with contextlib.suppress(Exception):
            from bolo.plugin.host import discover_entry_points, parse_plugin_setting

            found = discover_entry_points() if settings.plugin_entry_points else {}
            for item in settings.plugins:
                name, cmd = parse_plugin_setting(item)
                found[name] = cmd
            command = found.get("torch")
    cli = host.which("bolo-torch")
    out: dict[str, Any] = {
        "installed": bool(command or cli),
        "version": None,
        "torch": None,
        # the device the plugin picks by itself ("auto"): cuda, mps or cpu; None until it was asked
        "device": None,
        "cuda": None,
        # Apple's Metal (MPS) and AMD's ROCm (a ROCm build of torch, which serves torch.cuda) as the plugin sees them
        "mps": None,
        "rocm": None,
        "probed": False,
    }
    forced = host.env.get("BOLO_TORCH_DEVICE")
    if forced:
        out["device"] = forced.strip().lower()
    if deep and cli:
        text = host.cached(("bolo-torch doctor", *_stamp(cli)), lambda: host.run([cli, "doctor"], timeout=45.0))
        try:
            doc = json.loads(text) if text else None
        except ValueError:
            doc = None
        if isinstance(doc, dict):
            out.update(
                version=doc.get("bolo_torch"),
                torch=doc.get("torch"),
                device=out["device"] or doc.get("auto_device"),
                cuda=bool(doc.get("cuda_available")),
                mps=bool(doc.get("mps_available")),
                rocm=bool(doc.get("hip_build")) if "hip_build" in doc else None,
                probed=True,
            )
    return out


def detect_runtimes(host: Host, settings: Settings | None = None, deep: bool = False) -> dict[str, dict[str, Any]]:
    """Which inference runtimes are here and what each can use. ``gpu`` is True, False, or None for "cannot tell"."""
    out: dict[str, dict[str, Any]] = {}

    ort = host.module("onnxruntime")
    providers: list[str] | None = None
    if ort is not None:
        try:
            providers = [str(p) for p in ort.get_available_providers()]
        except Exception:
            providers = None
    out["onnxruntime"] = {
        "installed": ort is not None,
        "version": _version(ort) if ort is not None else None,
        "providers": providers,
        "gpu": (any(p in _ORT_GPU_PROVIDERS for p in providers) if providers is not None else None)
        if ort is not None
        else False,
    }

    sherpa = host.module("sherpa_onnx")
    out["sherpa-onnx"] = {
        "installed": sherpa is not None,
        "version": _version(sherpa) if sherpa is not None else None,
        # it carries its own ONNX Runtime; the wheel on PyPI is the CPU build, its CUDA build has ONNX Runtime's
        # CUDA provider next to it (bolo.ort_providers.sherpa_build)
        "gpu": _sherpa_gpu(sherpa) if sherpa is not None else False,
    }

    ct2 = host.module("ctranslate2")
    cuda_devices: int | None = None
    if ct2 is not None:
        try:
            cuda_devices = int(host.cached(("ctranslate2 cuda",), ct2.get_cuda_device_count))
        except Exception:
            cuda_devices = None
    out["ctranslate2"] = {
        "installed": ct2 is not None,
        "version": _version(ct2) if ct2 is not None else None,
        "cuda_devices": cuda_devices,
        "gpu": (cuda_devices > 0 if cuda_devices is not None else None) if ct2 is not None else False,
    }

    try:
        out["plugin:torch"] = _torch_plugin(host, settings, deep)
    except Exception:
        out["plugin:torch"] = {"installed": None}
    for runtime in ("llama-server", "audio.cpp"):
        try:
            out[runtime] = _server_runtime(host, settings, runtime, deep)
        except Exception:
            out[runtime] = {"installed": None, "gpu": None}
    try:
        out["whisper.cpp"] = _whispercpp(host)
    except Exception:
        out["whisper.cpp"] = {"installed": None, "gpu": None}
    try:
        out["mlx"] = _mlx(host)
    except Exception:
        out["mlx"] = {"installed": None, "gpu": None}
    if host.system() == "linux":
        with contextlib.suppress(Exception):
            out["rocm"] = rocm_runtime(host)
    return out


def _whispercpp(host: Host) -> dict[str, Any]:
    """pywhispercpp and the GPU back ends its whisper.cpp was built with (Metal in the macOS arm64 wheels)."""
    from bolo.engines.whispercpp import families_of

    pw = host.module("_pywhispercpp")
    if pw is None:
        return {"installed": False, "version": None, "gpu": False, "backends": []}
    info = ""
    with contextlib.suppress(Exception):
        info = str(pw.whisper_print_system_info())
    backends = sorted(families_of(info))
    return {"installed": True, "version": _dist_version("pywhispercpp"), "gpu": bool(backends), "backends": backends}


def _mlx(host: Host) -> dict[str, Any]:
    """MLX (Apple's array framework) and whether it reaches the GPU through Metal; mlx-audio, the speech models on it
    (``bolo-speech[mlx]``)."""
    mx = host.module("mlx.core")
    if mx is None:
        return {"installed": False, "version": None, "gpu": False, "metal": False, "mlx_audio": None}
    metal: bool | None = None
    with contextlib.suppress(Exception):
        metal = bool(mx.metal.is_available())
    device = None
    with contextlib.suppress(Exception):
        device = str(mx.default_device())
    return {
        "installed": True,
        "version": _version(mx) or _dist_version("mlx"),
        "gpu": metal,
        "metal": metal,
        "device": device,
        "mlx_audio": _dist_version("mlx-audio"),
    }


def _dist_version(name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(name)
    except Exception:
        return None


# -- everything -------------------------------------------------------------------------------------------------
def detect(
    settings: Settings | None = None, host: Host | None = None, deep: bool = False, cache: ProbeCache | None = None
) -> Hardware:
    """Describe this machine. Never raises; what cannot be read stays None.

    ``deep`` also runs the slow probes (the torch plugin's ``doctor``, the server binaries' ``--version``); the
    server skips them at start-up and runs them when asked (``GET /api/v1/hardware?refresh=true``). ``cache``: the
    answers of slow probes that may be reused (:class:`ProbeCache`; the server keeps one).
    """
    host = host or Host(cache=cache)
    hw = Hardware(detected_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), deep=deep)
    with contextlib.suppress(Exception):
        hw.system = {"darwin": "macos"}.get(host.system(), host.system()) or None
    hypervisor = False
    try:
        hw.cpu, hypervisor = detect_cpu(host)
    except Exception:
        hw.notes.append("the CPU could not be read")
    try:
        hw.memory = detect_memory(host)
    except Exception:
        hw.notes.append("memory could not be read")
    try:
        hw.container = detect_container(host, hypervisor)
    except Exception:
        hw.notes.append("container limits could not be read")
    limit = hw.container.memory_limit_bytes
    if limit and (hw.memory.total_bytes is None or limit < hw.memory.total_bytes):
        hw.memory.limit_bytes = limit
    cores: list[float] = [float(c) for c in (hw.cpu.logical_cores,) if c]
    with contextlib.suppress(Exception):
        affinity = host.affinity()
        if affinity:
            cores.append(float(affinity))
    if hw.container.cpu_quota:
        cores.append(hw.container.cpu_quota)
    hw.cpu.usable_cores = min(cores) if cores else None
    try:
        data_dir = Path(settings.data_dir) if settings is not None else None
        if data_dir is not None:
            total, free = host.disk(data_dir)
            hw.disk = Disk(str(data_dir), total, free)
    except Exception:
        hw.notes.append("the data directory's disk could not be read")
    try:
        hw.gpus, hw.gpus_known = detect_gpus(host, hw.cpu, hw.memory)
    except Exception:
        hw.gpus_known = False
        hw.notes.append("GPUs could not be read")
    try:
        hw.runtimes = detect_runtimes(host, settings, deep)
    except Exception:
        hw.notes.append("runtimes could not be read")
    if settings is not None:
        hw.device = settings.device
        hw.device_explicit = bool(getattr(settings, "device_explicit", True))
        with contextlib.suppress(Exception):
            hw.threads = settings.resolved_threads()
    quota = hw.container.cpu_quota
    if quota is not None and hw.threads is not None and hw.threads > math.ceil(quota):
        hw.notes.append(
            f"the CPU quota allows about {_num(quota)} core(s) but bolo uses {hw.threads} inference threads; "
            f"`threads = {max(1, math.floor(quota))}` avoids throttling"
        )
    amd = [g for g in hw.gpus if g.vendor == "amd" and g.source == "sysfs" and g.vram_total_bytes is not None]
    if amd and all(g.backend is None for g in amd):
        hw.notes.append(
            f"an AMD GPU was found ({amd[0].name}) but ROCm is not installed, so only a Vulkan build of "
            'llama-server can use it (device = "vulkan"); with ROCm, device = "rocm" runs ONNX Runtime '
            "(onnxruntime-migraphx), llama.cpp (its HIP build) and the torch plugin (a ROCm build of PyTorch) on it: "
            "docs/rocm.md"
        )
    if hw.system == "windows":
        for g in hw.gpus:
            if g.vendor == "amd":
                hw.notes.append(windows_rocm_note(g))
    best = hw.best_gpu()
    if best is not None and hw.device == "cpu" and best.backend == "metal" and not hw.device_explicit:
        hw.notes.append(
            "Apple silicon GPU: MLX, whisper.cpp and llama.cpp models use Metal; the other built-in engines run on "
            'the CPU until the `device` setting asks for the GPU (device = "coreml"), and device = "cpu" keeps '
            "everything on the CPU; the torch plugin picks its own device"
        )
    elif best is not None and hw.device == "cpu":
        example = device_for(best)
        hw.notes.append(
            'a GPU was found but the `device` setting is "cpu": the built-in engines run on the CPU until it is '
            f'set (for example device = "{example}"); the torch plugin picks its own device'
        )
    name, _, index = hw.device.strip().lower().partition(":")
    if best is not None and name in ("dml", "directml") and index.isdigit():
        pinned = next((g for g in hw.gpus if g.dml_index == int(index)), None)
        if pinned is not None and pinned.integrated and pinned is not best:
            hw.notes.append(
                f'device = "{hw.device}" is integrated graphics ({pinned.name}); {best.name} is faster: '
                f'device = "{device_for(best)}"'
            )
    return hw


#: Radeon cards AMD's HIP runtime for Windows covers (RDNA 2 to 4: RX 6000, 7000 and 9000, Radeon PRO W6000/W7000),
#: which llama.cpp's Windows ROCm build of llama-server is for
_WINDOWS_HIP = re.compile(r"Radeon\s*(?:\(TM\)\s*)?(?:RX\s*(?:6[4-9]|7[6-9]|9[06])\d\d|PRO\s*W[67]\d\d\d)", re.I)


def windows_rocm_note(gpu: Gpu) -> str:
    """What ROCm means on Windows for an AMD GPU: only llama-server's ROCm (HIP) build, and only on the Radeon cards
    AMD's HIP runtime for Windows covers; everything else in bolo reaches the GPU through DirectML (or Vulkan)."""
    name = (gpu.name or "AMD GPU").split(" (dml:")[0]
    if _WINDOWS_HIP.search(name):
        return (
            f"ROCm on Windows ({name}): only llama-server's ROCm (HIP) build uses it (bolo components install "
            "llama-server --flavour rocm; it brings its own HIP runtime and needs a recent Adrenalin driver; not run "
            "by bolo's maintainers); ONNX Runtime engines use DirectML (device = \"dml\"), and llama-server's Vulkan "
            "build works too: docs/rocm.md"
        )
    return (
        f'ROCm: not available on Windows for this GPU ({name}); bolo uses DirectML (device = "dml") or '
        'llama-server\'s Vulkan build (device = "vulkan") for it: docs/rocm.md'
    )


def device_for(gpu: Gpu) -> str:
    """The ``device`` value that reaches ``gpu``: ``cuda`` (NVIDIA through CUDA), ``rocm`` (AMD through ROCm),
    ``coreml`` (Apple), ``dml`` when it is the adapter ``dml`` picks (Windows' high-performance GPU), else ``dml:N``."""
    if gpu.backend == "metal":
        return "coreml"
    if gpu.backend == "rocm":
        return "rocm"
    if gpu.backend != "dml":
        return "cuda"
    return "dml" if gpu.dml_preferred or gpu.dml_index is None else f"dml:{gpu.dml_index}"


# -- the micro-benchmark ----------------------------------------------------------------------------------------
#: best time per iteration, in milliseconds, on the reference machine (the catalog's: 4 vCPUs of an Intel Xeon
#: at 2.1 GHz with AVX-512 and AMX): the onnxruntime workload with 4 threads (onnxruntime 1.29), the numpy
#: workload on one core. Taken with this code on 2026-11-14 as the best of about twenty runs, because the machine
#: was shared, as it was for the catalog's own measurements; see docs/recommendations.md
REFERENCE_MS = {"onnxruntime": 2.9, "numpy": 6.1}
#: threads the reference timing of the onnxruntime workload used (the catalog's measurements used 4)
REFERENCE_THREADS = 4


def _pb_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _pb(field_no: int, value: bytes | str | int) -> bytes:
    """One protobuf field: a varint for an int, length-delimited otherwise."""
    if isinstance(value, int):
        return _pb_varint(field_no << 3) + _pb_varint(value)
    data = value.encode() if isinstance(value, str) else value
    return _pb_varint(field_no << 3 | 2) + _pb_varint(len(data)) + data


def _onnx_value(name: str, shape: tuple[int, ...]) -> bytes:
    dims = b"".join(_pb(1, _pb(1, d)) for d in shape)
    tensor = _pb(1, 1) + _pb(2, dims)  # elem_type FLOAT, shape
    return _pb(1, name) + _pb(2, _pb(1, tensor))


def benchmark_model() -> bytes:
    """A tiny fixed ONNX graph, written out by hand (no ``onnx`` package): a 3x3 convolution over 64 channels
    with a ReLU on a 96x96 map, and a 512x512 matrix product. Weights are inputs, so the file is a few hundred bytes."""
    pads = _pb(1, "pads") + b"".join(_pb(8, 1) for _ in range(4)) + _pb(20, 7)  # INTS
    conv = _pb(1, "x") + _pb(1, "w") + _pb(2, "c") + _pb(4, "Conv") + _pb(5, pads)
    relu = _pb(1, "c") + _pb(2, "y") + _pb(4, "Relu")
    matmul = _pb(1, "a") + _pb(1, "a") + _pb(2, "z") + _pb(4, "MatMul")
    graph = (
        _pb(1, conv) + _pb(1, relu) + _pb(1, matmul) + _pb(2, "bolo-benchmark")
        + _pb(11, _onnx_value("x", (1, 64, 96, 96))) + _pb(11, _onnx_value("w", (64, 64, 3, 3)))
        + _pb(11, _onnx_value("a", (512, 512)))
        + _pb(12, _onnx_value("y", (1, 64, 96, 96))) + _pb(12, _onnx_value("z", (512, 512)))
    )  # fmt: skip
    return _pb(1, 8) + _pb(8, _pb(1, "") + _pb(2, 13)) + _pb(7, graph)  # ir_version, opset 13, graph


def _time_loop(step: Any, budget_s: float) -> tuple[float, int]:
    """Best time of one ``step()`` in milliseconds over about ``budget_s`` seconds, and how many were timed."""
    for _ in range(3):
        step()
    best = math.inf
    runs = 0
    deadline = time.perf_counter() + budget_s
    while runs < 5 or time.perf_counter() < deadline:
        t0 = time.perf_counter()
        step()
        best = min(best, time.perf_counter() - t0)
        runs += 1
        if runs >= 2000:
            break
    return best * 1000.0, runs


def _bench_onnxruntime(ort: Any, threads: int, budget_s: float) -> tuple[float, int]:
    import numpy as np

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(benchmark_model(), sess_options=options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(7)
    feed = {
        "x": rng.standard_normal((1, 64, 96, 96), dtype=np.float32),
        "w": rng.standard_normal((64, 64, 3, 3), dtype=np.float32) * 0.05,
        "a": rng.standard_normal((512, 512), dtype=np.float32) * 0.05,
    }
    return _time_loop(lambda: session.run(None, feed), budget_s)


def _bench_numpy(budget_s: float) -> tuple[float, int]:
    """One core's worth of arithmetic that numpy does without a BLAS thread pool: element-wise float32 maths."""
    import numpy as np

    rng = np.random.default_rng(7)
    x = rng.standard_normal((64, 4096), dtype=np.float32)
    w = rng.standard_normal((64, 4096), dtype=np.float32)
    out = np.empty_like(x)

    def step() -> None:
        for _ in range(24):
            np.multiply(x, w, out=out)
            np.add(out, x, out=out)
            np.maximum(out, 0.0, out=out)
            np.sqrt(out, out=out)
        float(out.sum())

    return _time_loop(step, budget_s)


def run_benchmark(
    hardware: Hardware | None = None, threads: int | None = None, budget_s: float = 3.0
) -> dict[str, Any]:
    """Time a tiny fixed workload and compare it with the reference machine. Takes a few seconds.

    With onnxruntime installed, the workload is a small convolution and matrix product run with the number of
    threads bolo uses here, so the result covers both the speed of a core and how many there are. Without it, a
    single-threaded numpy workload gives the speed of one core and the thread count is applied by rule.
    """
    hw = hardware or detect()
    threads = max(1, int(threads or hw.threads or min(4, int(hw.cpu.usable_cores or hw.cpu.logical_cores or 1))))
    result: dict[str, Any] = {
        "version": BENCHMARK_VERSION,
        "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cpu_model": hw.cpu.model,
        "processor_id": hw.cpu.processor_id,
        "logical_cores": hw.cpu.logical_cores,
        "usable_cores": hw.cpu.usable_cores,
        "threads": threads,
    }
    ort = None
    with contextlib.suppress(Exception):
        ort = importlib.import_module("onnxruntime")
    if ort is not None:
        try:
            ms, runs = _bench_onnxruntime(ort, threads, budget_s)
            result.update(workload="onnxruntime", ms=round(ms, 4), runs=runs, reference_ms=REFERENCE_MS["onnxruntime"])
            result["speed"] = round(REFERENCE_MS["onnxruntime"] / ms, 3)
            return result
        except Exception as exc:
            result["onnxruntime_error"] = f"{type(exc).__name__}: {exc}"[:200]
    ms, runs = _bench_numpy(budget_s)
    result.update(workload="numpy", ms=round(ms, 4), runs=runs, reference_ms=REFERENCE_MS["numpy"])
    # one core measured; the number of threads is applied by the same rule the heuristic uses
    result["single_core_speed"] = round(REFERENCE_MS["numpy"] / ms, 3)
    result["speed"] = round(result["single_core_speed"] * thread_scale(threads), 3)
    return result


def thread_scale(threads: float) -> float:
    """Speed with ``threads`` inference threads relative to the reference's four: sub-linear, flat above four
    (bolo never uses more than four threads per worker unless ``threads`` is set)."""
    return float(min(1.0, (max(0.25, threads) / REFERENCE_THREADS) ** 0.8))


def save_benchmark(data_dir: Path, result: dict[str, Any]) -> Path:
    path = Path(data_dir) / BENCHMARK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


#: ``platform.processor()`` values that name an architecture, not a processor
_VAGUE_PROCESSOR_IDS = {"", "x86_64", "amd64", "i386", "i686", "x86", "arm", "arm64", "aarch64", "arm64e"}


def _same_cpu(doc: dict[str, Any], cpu: Cpu) -> bool:
    """Whether a stored benchmark was taken on this CPU: the same name, or the same processor identifier. The
    identifier lets a benchmark survive a change in how the name is read (bolo before 0.6 stored Windows's
    processor identifier as the name; it now reads the marketing name from the registry)."""

    def norm(value: Any) -> str:
        return " ".join(str(value or "").split()).lower()

    if norm(doc.get("cpu_model")) == norm(cpu.model) and cpu.model:
        return True
    ours = norm(cpu.processor_id)
    if ours in _VAGUE_PROCESSOR_IDS:
        return False
    return ours in (norm(doc.get("processor_id")), norm(doc.get("cpu_model")))


def benchmark_status(data_dir: Path, hardware: Hardware | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """``(benchmark, why_not)``: the stored benchmark when it applies to this machine; otherwise None and, when a
    benchmark is stored but not used, why (for ``bolo doctor`` and ``bolo recommend``; None when there is none)."""
    path = Path(data_dir) / BENCHMARK_FILE
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError):
        return None, f"the stored benchmark ({path}) cannot be read; run `bolo recommend --benchmark` again"
    if not isinstance(doc, dict) or doc.get("version") != BENCHMARK_VERSION:
        return None, "the stored benchmark is from another version of bolo; run `bolo recommend --benchmark` again"
    speed = doc.get("speed")
    if not isinstance(speed, (int, float)) or not speed > 0:
        return None, f"the stored benchmark ({path}) has no speed; run `bolo recommend --benchmark` again"
    if hardware is not None:
        if not _same_cpu(doc, hardware.cpu):
            # the data directory moved to another machine, or the CPU's name is read differently now
            return None, (
                f"the stored benchmark was taken on another CPU ({doc.get('cpu_model') or 'unknown'}, here "
                f"{hardware.cpu.model or 'unknown'}); run `bolo recommend --benchmark` again"
            )
        if (
            doc.get("logical_cores") != hardware.cpu.logical_cores
            or doc.get("usable_cores") != hardware.cpu.usable_cores
        ):
            return None, (
                f"the stored benchmark was taken with other cores ({_num(doc.get('logical_cores'))} logical, "
                f"{_num(doc.get('usable_cores'))} usable; here {_num(hardware.cpu.logical_cores)} and "
                f"{_num(hardware.cpu.usable_cores)}); run `bolo recommend --benchmark` again"
            )
    return doc, None


def load_benchmark(data_dir: Path, hardware: Hardware | None = None) -> dict[str, Any] | None:
    """The stored benchmark, or None when there is none, it cannot be read, or it is from another machine."""
    return benchmark_status(data_dir, hardware)[0]
