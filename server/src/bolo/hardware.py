"""What this machine is: CPU, memory, disk, GPUs, the runtimes that can use them, and container limits.

Everything here is best effort. Detection never raises: a value that cannot be read is ``None``, which the API
returns as ``null`` and the command line prints as "unknown". Nothing in bolo refuses to do something because of
what is found here; it feeds recommendations and warnings only (``bolo.recommend``, docs/recommendations.md).

The outside world is read through a :class:`Host`, so tests describe a machine with a few fake files and commands.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
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


class Host:
    """The machine as detection sees it. Tests subclass this; every method may return None for "cannot tell"."""

    def __init__(self, root: str | os.PathLike[str] = "/", env: dict[str, str] | None = None) -> None:
        self.root = Path(root)
        self.env = dict(os.environ) if env is None else dict(env)

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
    #: how inference runtimes reach it: "cuda", "metal", "dml" (DirectML), or None when no runtime bolo uses can
    backend: str | None = None
    source: str = ""  # nvidia-smi | pynvml | sysfs | platform | dxgi

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
        """The GPU with the most memory that a runtime can reach (optionally through one backend)."""
        usable = [g for g in self.gpus if g.backend and (backend is None or g.backend == backend)]
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
        names = {"sse4.2": "sse4.2", "avx1.0": "avx", "avx2": "avx2", "fma": "fma", "avx512f": "avx512"}
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
    if cpu.model is None:
        cpu.model = host.processor()
    if cpu.physical_cores is None:
        cpu.physical_cores = host.physical_cores()
    if cpu.features is None and cpu.architecture == "arm64":
        cpu.features = ["neon"]  # every 64-bit ARM core has it
    return cpu, hypervisor


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
    out = host.run(
        [exe, "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
        timeout=6.0,
    )
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


def _sysfs_gpus(host: Host, have_nvidia: bool) -> list[Gpu]:
    """Graphics cards the kernel lists (/sys/class/drm): AMD with its VRAM counters, Intel, NVIDIA without tools."""
    gpus: list[Gpu] = []
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
        gpus.append(
            Gpu(
                vendor=vendor,
                name=f"{label[vendor]}{f' ({device})' if device else ''}",
                vram_total_bytes=total,
                vram_free_bytes=total - used if total is not None and used is not None else None,
                unified_memory=vendor == "intel",
                # no runtime bolo starts uses these out of the box (Vulkan and ROCm builds exist; not detected)
                backend=None,
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
        gpus += _dxgi_gpus(host, adapters, have_nvidia=bool(nvidia))
        return gpus, True
    return gpus, nvidia is not None


def _dxgi_gpus(host: Host, adapters: list[Any], have_nvidia: bool) -> list[Gpu]:
    """Windows graphics adapters as DXGI lists them: what DirectML (``device = "dml:N"``) can use when the
    DirectML build of ONNX Runtime is installed. NVIDIA cards already read through NVML/nvidia-smi are not listed
    twice."""
    ort = host.module("onnxruntime")
    dml = False
    if ort is not None:
        with contextlib.suppress(Exception):
            dml = "DmlExecutionProvider" in ort.get_available_providers()
    gpus: list[Gpu] = []
    for a in adapters:
        if a.vendor == "nvidia" and have_nvidia:
            continue
        integrated = a.vendor == "intel" or a.dedicated_bytes <= 512 * 1024 * 1024
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
            )
        )
    return gpus


# -- runtimes ---------------------------------------------------------------------------------------------------
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
        # the release bolo installs: CPU builds on Linux, Metal on Apple Silicon (bolo.tts_runtimes)
        out["version"] = info.version
        out["gpu"] = host.system() == "darwin" and _arch(host.machine()) == "arm64"
    if deep:
        version = host.run([str(path), "--version"], timeout=15.0)
        if version and version.strip():
            out["version"] = version.strip().splitlines()[0][:80]
        if runtime == tts_runtimes.LLAMA:
            devices = host.run([str(path), "--list-devices"], timeout=15.0)
            if devices is not None:
                names = [
                    ln.strip()
                    for ln in devices.splitlines()
                    if any(k in ln for k in ("CUDA", "Vulkan", "Metal", "SYCL", "ROCm", "OpenCL"))
                ]
                out["gpu_devices"] = names
                out["gpu"] = bool(names)
    return out


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
        "probed": False,
    }
    forced = host.env.get("BOLO_TORCH_DEVICE")
    if forced:
        out["device"] = forced.strip().lower()
    if deep and cli:
        text = host.run([cli, "doctor"], timeout=45.0)
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
        # it carries its own ONNX Runtime; the wheel on PyPI is the CPU build, a CUDA wheel exists (not told apart)
        "gpu": None if sherpa is not None else False,
    }

    ct2 = host.module("ctranslate2")
    cuda_devices: int | None = None
    if ct2 is not None:
        try:
            cuda_devices = int(ct2.get_cuda_device_count())
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
    return out


# -- everything -------------------------------------------------------------------------------------------------
def detect(settings: Settings | None = None, host: Host | None = None, deep: bool = False) -> Hardware:
    """Describe this machine. Never raises; what cannot be read stays None.

    ``deep`` also runs the slow probes (the torch plugin's ``doctor``, the server binaries' ``--version``); the
    server skips them at start-up and runs them when asked (``GET /api/v1/hardware?refresh=true``).
    """
    host = host or Host()
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
        with contextlib.suppress(Exception):
            hw.threads = settings.resolved_threads()
    quota = hw.container.cpu_quota
    if quota is not None and hw.threads is not None and hw.threads > math.ceil(quota):
        hw.notes.append(
            f"the CPU quota allows about {_num(quota)} core(s) but bolo uses {hw.threads} inference threads; "
            f"`threads = {max(1, math.floor(quota))}` avoids throttling"
        )
    best = hw.best_gpu()
    if best is not None and hw.device == "cpu":
        example = "dml" if best.backend == "dml" else "cuda"
        hw.notes.append(
            'a GPU was found but the `device` setting is "cpu": the built-in engines run on the CPU until it is '
            f'set (for example device = "{example}"); the torch plugin picks its own device'
        )
    return hw


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


def load_benchmark(data_dir: Path, hardware: Hardware | None = None) -> dict[str, Any] | None:
    """The stored benchmark, or None when there is none, it cannot be read, or it is from another machine."""
    try:
        doc = json.loads((Path(data_dir) / BENCHMARK_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("version") != BENCHMARK_VERSION:
        return None
    speed = doc.get("speed")
    if not isinstance(speed, (int, float)) or not speed > 0:
        return None
    if hardware is not None and (
        doc.get("cpu_model") != hardware.cpu.model
        or doc.get("logical_cores") != hardware.cpu.logical_cores
        or doc.get("usable_cores") != hardware.cpu.usable_cores
    ):
        return None  # the data directory moved to another machine, or the container's CPU limit changed
    return doc
