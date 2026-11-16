"""Hardware detection against described machines: fake system files and commands, never the real host."""

from __future__ import annotations

import json
import platform
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from bolo import hardware as hw
from bolo.config import Settings
from fakebin import printing_program

GB = 1024**3
MB = 1024 * 1024

X86_AVX2 = """processor\t: {n}
vendor_id\t: GenuineIntel
model name\t: Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz
physical id\t: 0
core id\t\t: {core}
cpu cores\t: 4
flags\t\t: fpu vme sse sse2 ssse3 sse4_1 sse4_2 fma f16c avx avx2 bmi2{extra}
"""
X86_AVX512 = """processor\t: {n}
vendor_id\t: AuthenticAMD
model name\t: AMD Ryzen 9 7950X 16-Core Processor
physical id\t: 0
core id\t\t: {core}
flags\t\t: fpu sse4_2 fma avx avx2 avx512f avx512dq avx512_vnni avx512_bf16
"""
ARM_PI = """processor\t: {n}
BogoMIPS\t: 108.00
Features\t: fp asimd evtstrm crc32 cpuid
CPU implementer\t: 0x41
CPU part\t: 0xd08
"""
ARM_GRAVITON = """processor\t: {n}
BogoMIPS\t: 2100.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp asimddp sve i8mm bf16
CPU implementer\t: 0x41
CPU part\t: 0xd40
"""
MEMINFO = "MemTotal:       {total} kB\nMemFree:         1000000 kB\nMemAvailable:   {available} kB\n"


def cpuinfo(template: str, logical: int, physical: int | None = None, extra: str = "") -> str:
    physical = physical or logical
    return "\n".join(template.format(n=i, core=i % physical, extra=extra) for i in range(logical))


class FakeHost(hw.Host):
    """A machine made of a few files and command outputs."""

    def __init__(
        self,
        system: str = "linux",
        machine: str = "x86_64",
        files: dict[str, str] | None = None,
        commands: dict[str, str] | None = None,
        binaries: dict[str, str] | None = None,
        modules: dict[str, Any] | None = None,
        cores: int | None = None,
        affinity: int | None = None,
        memory: tuple[int | None, int | None] = (None, None),
        env: dict[str, str] | None = None,
        windows_cpu: tuple[str | None, list[str] | None] = (None, None),
        adapters: list[Any] | None = None,
        processor: str | None = None,
    ) -> None:
        super().__init__("/nonexistent-root", env or {})
        self._windows_cpu, self._adapters, self._processor = windows_cpu, adapters, processor
        self._system, self._machine = system, machine
        self.files = files or {}
        self.commands = commands or {}
        self.binaries = binaries or {}
        self.modules = modules or {}
        self._cores, self._affinity, self._memory = cores, affinity, memory
        self.ran: list[list[str]] = []
        self.probed: list[tuple[list[str], dict[str, str] | None]] = []
        #: commands (their first two words) that never answer
        self.hangs: set[str] = set()

    def system(self) -> str:
        return self._system

    def machine(self) -> str:
        return self._machine

    def processor(self) -> str | None:
        return self._processor

    def read(self, path: str) -> str | None:
        return self.files.get(path)

    def exists(self, path: str) -> bool:
        return path in self.files

    def listdir(self, path: str) -> list[str]:
        prefix = path.rstrip("/") + "/"
        return sorted({p[len(prefix) :].split("/")[0] for p in self.files if p.startswith(prefix)})

    def run(self, argv: list[str], timeout: float = 4.0) -> str | None:
        self.ran.append(argv)
        return self.commands.get(" ".join(argv[:3])) or self.commands.get(" ".join(argv[:2]))

    def probe(
        self, argv: list[str], timeout: float, env: dict[str, str] | None = None
    ) -> tuple[str | None, str | None]:
        self.probed.append((argv, env))
        key = " ".join(argv[:2])
        if key in self.hangs:
            return None, f"did not answer within {timeout:.0f} s"
        out = self.run(argv, timeout)
        return (out, None) if out is not None else (None, "exit code 1")

    def which(self, name: str) -> str | None:
        return self.binaries.get(name)

    def module(self, name: str) -> Any | None:
        return self.modules.get(name)

    def logical_cores(self) -> int | None:
        return self._cores

    def physical_cores(self) -> int | None:
        return None

    def affinity(self) -> int | None:
        return self._affinity

    def memory(self) -> tuple[int | None, int | None]:
        return self._memory

    def disk(self, path: Path) -> tuple[int | None, int | None]:
        return 500 * GB, 120 * GB

    def windows_cpu(self) -> tuple[str | None, list[str] | None]:
        return self._windows_cpu

    def windows_adapters(self) -> list[Any] | None:
        return self._adapters


def settings(tmp_path: Path, **kw: Any) -> Settings:
    return Settings(**{"data_dir": tmp_path / "data", "plugin_entry_points": False, "threads": 4, **kw})


def laptop() -> FakeHost:
    return FakeHost(
        files={
            "/proc/cpuinfo": cpuinfo(X86_AVX2, 8, 4),
            "/proc/meminfo": MEMINFO.format(total=16 * 1024 * 1024, available=9 * 1024 * 1024),
            "/sys/class/dmi/id/sys_vendor": "LENOVO\n",
            "/sys/class/drm/card0/device/vendor": "0x8086\n",
            "/sys/class/drm/card0/device/device": "0x5917\n",
        },
        cores=8,
        affinity=8,
    )


def test_linux_x86_without_nvidia(tmp_path: Path) -> None:
    found = hw.detect(settings(tmp_path), laptop())
    cpu = found.cpu
    assert found.system == "linux" and cpu.architecture == "x86_64"
    assert cpu.model == "Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz" and cpu.vendor == "GenuineIntel"
    assert (cpu.physical_cores, cpu.logical_cores, cpu.usable_cores) == (4, 8, 8.0)
    assert cpu.features is not None and "avx2" in cpu.features and "avx512" not in cpu.features
    assert found.memory.total_bytes == 16 * GB and found.memory.available_bytes == 9 * GB
    assert found.memory.limit_bytes is None and found.memory.usable_bytes == 16 * GB
    assert found.disk.free_bytes == 120 * GB and found.disk.path == str(tmp_path / "data")
    # the integrated graphics are listed, but no runtime bolo starts can use them
    assert [(g.vendor, g.backend, g.unified_memory) for g in found.gpus] == [("intel", None, True)]
    assert found.best_gpu() is None and found.gpus_known
    box = found.container
    assert box.detected is False and box.virtual_machine is False and box.cpu_quota is None
    assert found.threads == 4 and found.device == "cpu"
    doc = found.to_dict()
    assert json.loads(json.dumps(doc)) == doc  # everything is plain JSON
    assert "8 cores" in found.summary() and "AVX2" in found.summary() and "16.0 GB RAM" in found.summary()


def test_linux_x86_with_nvidia(tmp_path: Path) -> None:
    host = FakeHost(
        files={
            "/proc/cpuinfo": cpuinfo(X86_AVX512, 32, 16),
            "/proc/meminfo": MEMINFO.format(total=64 * 1024 * 1024, available=60 * 1024 * 1024),
            "/sys/class/drm/card0/device/vendor": "0x10de\n",
        },
        binaries={"nvidia-smi": "/usr/bin/nvidia-smi"},
        commands={
            "/usr/bin/nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits": (
                "NVIDIA GeForce RTX 4090, 24564, 23010, 550.54.14\nNVIDIA GeForce RTX 3060, 12288, 100, 550.54.14\n"
            )
        },
        cores=32,
    )
    found = hw.detect(settings(tmp_path, device="cuda"), host)
    assert found.cpu.features is not None and {"avx512", "avx512_vnni", "avx512_bf16"} <= set(found.cpu.features)
    assert found.cpu.physical_cores == 16
    assert [g.name for g in found.gpus] == ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3060"]  # sysfs adds none
    first = found.gpus[0]
    assert first.vram_total_bytes == 24564 * 1024**2 and first.vram_free_bytes == 23010 * 1024**2
    assert first.driver == "550.54.14" and first.backend == "cuda" and first.source == "nvidia-smi"
    best = found.best_gpu("cuda")
    assert best is not None and best.name == "NVIDIA GeForce RTX 4090" and found.best_gpu("metal") is None
    assert not any("device" in note for note in found.notes)  # the device setting already uses the GPU
    assert "RTX 4090 (24.0 GB)" in found.summary()


def test_gpu_present_but_device_cpu_is_noted(tmp_path: Path) -> None:
    host = FakeHost(
        files={"/proc/cpuinfo": cpuinfo(X86_AVX2, 8)},
        binaries={"nvidia-smi": "/usr/bin/nvidia-smi"},
        commands={"/usr/bin/nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits": "Tesla T4, 15360, 15000, 535.1\n"},
        cores=8,
    )  # fmt: skip
    found = hw.detect(settings(tmp_path), host)
    assert found.best_gpu() is not None and any('"cpu"' in note for note in found.notes)


def test_nvidia_without_tools_and_amd_from_sysfs(tmp_path: Path) -> None:
    host = FakeHost(
        files={
            "/proc/cpuinfo": cpuinfo(X86_AVX2, 4),
            "/sys/class/drm/card0/device/vendor": "0x10de\n",
            "/sys/class/drm/card1/device/vendor": "0x1002\n",
            "/sys/class/drm/card1/device/mem_info_vram_total": str(8 * GB),
            "/sys/class/drm/card1/device/mem_info_vram_used": str(1 * GB),
            "/sys/class/drm/card1-HDMI-A-1/status": "connected",
        },
        cores=4,
    )
    gpus = hw.detect(settings(tmp_path), host).gpus
    assert [g.vendor for g in gpus] == ["nvidia", "amd"]
    assert gpus[0].vram_total_bytes is None and "no driver tools" in (gpus[0].name or "")
    assert gpus[1].vram_total_bytes == 8 * GB and gpus[1].vram_free_bytes == 7 * GB and gpus[1].backend is None


def test_pynvml_is_used_when_nvidia_smi_is_missing(tmp_path: Path) -> None:
    mem = types.SimpleNamespace(total=8 * GB, free=6 * GB)
    nv = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlSystemGetDriverVersion=lambda: b"545.23",
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetName=lambda h: b"NVIDIA RTX A2000",
        nvmlDeviceGetMemoryInfo=lambda h: mem,
    )
    host = FakeHost(files={"/proc/cpuinfo": cpuinfo(X86_AVX2, 4)}, modules={"pynvml": nv}, cores=4)
    gpus = hw.detect(settings(tmp_path), host).gpus
    assert len(gpus) == 1 and gpus[0].source == "pynvml" and gpus[0].name == "NVIDIA RTX A2000"
    assert gpus[0].driver == "545.23" and gpus[0].vram_free_bytes == 6 * GB


def test_arm_boards(tmp_path: Path) -> None:
    pi = FakeHost(
        machine="aarch64",
        files={
            "/proc/cpuinfo": cpuinfo(ARM_PI, 4) + "\nModel\t\t: Raspberry Pi 4 Model B Rev 1.4\n",
            "/proc/meminfo": MEMINFO.format(total=4 * 1024 * 1024, available=3 * 1024 * 1024),
        },
        cores=4,
    )
    found = hw.detect(settings(tmp_path), pi)
    assert found.cpu.architecture == "arm64" and found.cpu.model == "Raspberry Pi 4 Model B Rev 1.4"
    assert found.cpu.features == ["neon"] and found.cpu.vendor == "ARM"
    assert found.cpu.physical_cores is None  # the kernel does not say, and it is reported as unknown
    assert found.gpus == [] and found.gpus_known and "NEON" in found.summary()

    graviton = FakeHost(machine="aarch64", files={"/proc/cpuinfo": cpuinfo(ARM_GRAVITON, 16)}, cores=16)
    cpu = hw.detect(settings(tmp_path), graviton).cpu
    assert cpu.model == "Neoverse-V1" and cpu.features is not None
    assert {"neon", "dotprod", "sve", "i8mm", "bf16"} <= set(cpu.features)


def test_macos_apple_silicon(tmp_path: Path) -> None:
    host = FakeHost(
        system="darwin",
        machine="arm64",
        commands={
            "sysctl -n machdep.cpu.brand_string": "Apple M2 Pro\n",
            "sysctl -n hw.physicalcpu": "10\n",
            "sysctl -n hw.logicalcpu": "10\n",
            "sysctl -n hw.memsize": f"{32 * GB}\n",
        },
        cores=10,
        memory=(32 * GB, 20 * GB),
    )
    found = hw.detect(settings(tmp_path), host)
    assert found.system == "macos" and found.cpu.model == "Apple M2 Pro" and found.cpu.vendor == "Apple"
    assert (found.cpu.physical_cores, found.cpu.logical_cores) == (10, 10) and "neon" in (found.cpu.features or [])
    assert found.memory.total_bytes == 32 * GB and found.memory.available_bytes == 20 * GB
    gpu = found.best_gpu("metal")
    assert gpu is not None and gpu.unified_memory and gpu.vram_total_bytes == 32 * GB and gpu.vendor == "apple"
    assert found.container.detected is None  # only Linux can say
    assert "shared memory" in found.summary()


def test_macos_intel_reads_avx2(tmp_path: Path) -> None:
    host = FakeHost(
        system="darwin",
        machine="x86_64",
        commands={
            "sysctl -n machdep.cpu.brand_string": "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz",
            "sysctl -n machdep.cpu.features": "FPU SSE4.2 AVX1.0 FMA",
            "sysctl -n machdep.cpu.leaf7_features": "AVX2 BMI2",
        },
        cores=12,
        memory=(16 * GB, 8 * GB),
    )
    found = hw.detect(settings(tmp_path), host)
    assert found.cpu.features == ["sse4.2", "avx", "avx2", "fma"] and found.gpus == []


def test_cgroup_limited_container(tmp_path: Path) -> None:
    host = FakeHost(
        files={
            "/proc/cpuinfo": cpuinfo(X86_AVX512, 64, 32, extra="").replace("avx512_bf16", "avx512_bf16 hypervisor"),
            "/proc/meminfo": MEMINFO.format(total=256 * 1024 * 1024, available=200 * 1024 * 1024),
            "/.dockerenv": "",
            "/proc/1/cgroup": "0::/docker/abc\n",
            "/sys/fs/cgroup/cpu.max": "150000 100000\n",
            "/sys/fs/cgroup/memory.max": f"{2 * GB}\n",
            "/sys/class/dmi/id/sys_vendor": "Amazon EC2\n",
        },
        cores=64,
        affinity=64,
    )
    found = hw.detect(settings(tmp_path), host)
    box = found.container
    assert box.detected is True and box.kind == "docker" and box.cpu_quota == 1.5
    assert box.memory_limit_bytes == 2 * GB and box.virtual_machine is True and box.hypervisor == "Amazon EC2"
    assert found.cpu.logical_cores == 64 and found.cpu.usable_cores == 1.5
    assert found.memory.total_bytes == 256 * GB and found.memory.limit_bytes == 2 * GB
    assert found.memory.usable_bytes == 2 * GB
    # bolo would start four threads on one and a half cores: the report says so
    assert any("threads = 1" in note for note in found.notes)
    assert "CPU quota 1.5" in found.summary() and "2.0 GB RAM" in found.summary()


def test_cgroup_v1_kubernetes_and_unlimited(tmp_path: Path) -> None:
    v1 = FakeHost(
        files={
            "/proc/cpuinfo": cpuinfo(X86_AVX2, 16),
            "/proc/1/cgroup": "11:cpu:/kubepods/burstable/pod1\n",
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "400000\n",
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n",
        },
        cores=16,
        affinity=2,
    )
    found = hw.detect(settings(tmp_path), v1)
    assert found.container.kind == "kubernetes" and found.container.cpu_quota == 4.0
    assert found.container.memory_limit_bytes is None  # the "no limit" number is not a limit
    assert found.cpu.usable_cores == 2.0  # the affinity is tighter than the quota

    free = FakeHost(
        files={"/proc/cpuinfo": cpuinfo(X86_AVX2, 4), "/sys/fs/cgroup/cpu.max": "max 100000\n", "/sys/fs/cgroup/memory.max": "max\n"},
        cores=4,
    )  # fmt: skip
    box = hw.detect(settings(tmp_path), free).container
    assert box.cpu_quota is None and box.memory_limit_bytes is None and box.detected is False


def test_runtimes(tmp_path: Path) -> None:
    ort = types.SimpleNamespace(
        __version__="1.29.0", get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    ct2 = types.SimpleNamespace(__version__="4.5.0", get_cuda_device_count=lambda: 0)
    doctor = {"bolo_torch": "0.1.0", "torch": "2.11.0+cu124", "cuda_available": True, "auto_device": "cuda"}
    host = FakeHost(
        files={"/proc/cpuinfo": cpuinfo(X86_AVX2, 8)},
        modules={"onnxruntime": ort, "ctranslate2": ct2},
        binaries={"bolo-torch": "/venv/bin/bolo-torch"},
        commands={"/venv/bin/bolo-torch doctor": json.dumps(doctor)},
        cores=8,
    )
    shallow = hw.detect(settings(tmp_path), host).runtimes
    assert shallow["onnxruntime"] == {
        "installed": True,
        "version": "1.29.0",
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "gpu": True,
    }
    assert shallow["ctranslate2"]["cuda_devices"] == 0 and shallow["ctranslate2"]["gpu"] is False
    assert shallow["sherpa-onnx"] == {"installed": False, "version": None, "gpu": False}
    torch = shallow["plugin:torch"]
    assert torch["installed"] is True and torch["device"] is None and torch["probed"] is False
    assert not any(argv[-1] == "doctor" for argv in host.ran)  # the slow probe waits to be asked for
    assert shallow["llama-server"]["installed"] is False and shallow["audio.cpp"]["installed"] is False

    deep = hw.detect(settings(tmp_path), host, deep=True).runtimes["plugin:torch"]
    assert (
        deep["probed"] is True and deep["device"] == "cuda" and deep["cuda"] is True and deep["torch"] == "2.11.0+cu124"
    )


def test_server_binary_from_settings(tmp_path: Path) -> None:
    binary = printing_program(tmp_path / "llama-server")
    host = FakeHost(
        files={"/proc/cpuinfo": cpuinfo(X86_AVX2, 8)},
        commands={f"{binary} --version": "version: 6123 (abc)\n", f"{binary} --list-devices": "Available devices:\n  CUDA0: RTX 4090\n"},
        cores=8,
    )  # fmt: skip
    found = hw.detect(settings(tmp_path, llama_server_bin=str(binary)), host, deep=True).runtimes["llama-server"]
    assert found["installed"] is True and found["source"] == "settings" and found["version"] == "version: 6123 (abc)"
    assert found["gpu"] is True and found["gpu_devices"] == ["CUDA0: RTX 4090"]


def test_slow_probes_are_reused_for_a_minute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /api/v1/hardware?refresh=1 took about 3 s per call on a Windows laptop: nvidia-smi, --version and
    --list-devices ran every time. Their answers are reused for the cache's TTL, per binary."""
    binary = printing_program(tmp_path / "llama-server")
    now = [0.0]
    cache = hw.ProbeCache(60.0, clock=lambda: now[0])
    host = FakeHost(
        files={"/proc/cpuinfo": cpuinfo(X86_AVX2, 8)},
        commands={
            f"{binary} --version": "version: 6123 (abc)\n", f"{binary} --list-devices": "  CUDA0: RTX 3060\n",
            "/usr/bin/nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version": "RTX 3060, 6144, 5000, 621\n",
        },
        binaries={"nvidia-smi": "/usr/bin/nvidia-smi"},
        cores=8,
    )  # fmt: skip
    host.cache = cache
    conf = settings(tmp_path, llama_server_bin=str(binary))
    first = hw.detect(conf, host, deep=True)
    assert len(host.ran) == 3 and first.gpus[0].name == "RTX 3060"  # nvidia-smi, --version, --list-devices
    again = hw.detect(conf, host, deep=True)
    assert len(host.ran) == 3 and again.runtimes["llama-server"]["gpu_devices"] == ["CUDA0: RTX 3060"]
    now[0] = 61.0  # stale: asked again
    hw.detect(conf, host, deep=True)
    assert len(host.ran) == 6
    cache.clear()  # a component was installed
    hw.detect(conf, host, deep=True)
    assert len(host.ran) == 9
    host.cache = None  # no cache: every time, as before
    hw.detect(conf, host, deep=True)
    assert len(host.ran) == 12


def test_the_hardware_route_reuses_slow_probes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The server's view: ?refresh=1 detects again with the probes' answers from the last minute, ?refresh=force runs
    them again, and an installed component makes them run again."""
    from fastapi.testclient import TestClient

    from bolo.api.app import create_app

    calls: list[str] = []

    def slow(self: hw.Host, argv: list[str], timeout: float = 4.0) -> str | None:
        calls.append(" ".join(argv[1:2]))
        return "RTX 3060, 6144, 5000, 621\n" if "nvidia-smi" in argv[0] else None

    monkeypatch.setattr(hw.Host, "run", slow)
    monkeypatch.setattr(hw.Host, "which", lambda self, name: "/x/nvidia-smi" if name == "nvidia-smi" else None)
    app = create_app(settings(tmp_path))
    with TestClient(app) as client:
        core = app.state.core
        deadline = time.monotonic() + 10
        while core.advisor._refreshing and time.monotonic() < deadline:  # the start-up detection, in the background
            time.sleep(0.01)
        assert client.get("/api/v1/hardware?refresh=1").json()["gpus"][0]["name"] == "RTX 3060"
        seen = len(calls)
        assert seen >= 1
        for _ in range(3):
            assert client.get("/api/v1/hardware?refresh=1").status_code == 200
            assert client.get("/api/v1/hardware").status_code == 200
        assert len(calls) == seen
        assert client.get("/api/v1/hardware?refresh=force").status_code == 200
        assert len(calls) == seen + 1
        core.advisor.probes.clear()  # what an installed component does (Advisor.components_changed)
        assert client.get("/api/v1/hardware?refresh=1").status_code == 200
        assert len(calls) == seen + 2


def test_doctor_names_the_gpus_a_server_build_found() -> None:
    """bolo doctor prints the devices llama-server --list-devices found, and a GPU build that found none says so
    instead of passing for a CPU build (a Vulkan build on a laptop listed no devices and read "CPU build")."""
    from bolo.cli import _server_build

    vk = {"installed": True, "flavour": "vulkan", "gpu": True, "gpu_devices": ["Vulkan0: NVIDIA GeForce RTX 3060"]}
    assert _server_build(vk) == "GPU build: Vulkan0: NVIDIA GeForce RTX 3060"
    assert _server_build({**vk, "gpu": False, "gpu_devices": []}) == "vulkan build, but it found no GPU"
    assert _server_build({"flavour": "cpu", "gpu": False, "gpu_devices": []}) == "CPU build"
    assert _server_build({"gpu": None, "gpu_devices": None}) == "GPU support unknown"
    assert hw._DEVICE_LINE.match("  Vulkan0: AMD Radeon(TM) Graphics (512 MiB, 400 MiB free)")
    assert not hw._DEVICE_LINE.match("ggml_vulkan: Found 2 Vulkan devices:")
    assert not hw._DEVICE_LINE.match("Available devices:")


def test_detection_never_raises(tmp_path: Path) -> None:
    class Broken(FakeHost):
        def read(self, path: str) -> str | None:
            raise RuntimeError("no /proc here")

        def run(self, argv: list[str], timeout: float = 4.0) -> str | None:
            raise RuntimeError("no commands")

        def which(self, name: str) -> str | None:
            raise RuntimeError("no PATH")

        def module(self, name: str) -> Any | None:
            raise RuntimeError("no imports")

        def logical_cores(self) -> int | None:
            raise RuntimeError("no cores")

        def affinity(self) -> int | None:
            raise RuntimeError("no affinity")

        def disk(self, path: Path) -> tuple[int | None, int | None]:
            raise RuntimeError("no disk")

    found = hw.detect(settings(tmp_path), Broken())
    assert found.cpu.model is None and found.cpu.usable_cores is None and found.memory.total_bytes is None
    assert found.gpus == [] and found.gpus_known is False and found.notes
    assert "RAM unknown" in found.summary() and "GPU unknown" in found.summary()
    assert json.dumps(found.to_dict())

    empty = hw.detect(None, FakeHost(system="windows", machine="AMD64"))
    assert empty.cpu.architecture == "x86_64" and empty.cpu.features is None and empty.disk.path is None
    assert empty.gpus_known is False and empty.memory.total_bytes is None


def test_windows_cpu_features_and_directml_adapters(tmp_path: Path) -> None:
    from bolo.winhw import Adapter

    class Ort:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    adapters = [
        Adapter(0, "NVIDIA GeForce RTX 3060 Laptop GPU", 0x10DE, 6 * GB, 8 * GB, False, 5 * GB, 1 * GB),
        Adapter(1, "AMD Radeon(TM) Graphics", 0x1002, 512 * 1024 * 1024, 8 * GB, False, 400 * 1024 * 1024, 0),
    ]
    host = FakeHost(
        system="windows", machine="AMD64", cores=16, memory=(16 * GB, 9 * GB), modules={"onnxruntime": Ort()},
        windows_cpu=("AMD Ryzen 7 5800H with Radeon Graphics", ["sse4.2", "avx", "avx2"]), adapters=adapters,
    )  # fmt: skip
    found = hw.detect(settings(tmp_path), host)
    assert found.cpu.model == "AMD Ryzen 7 5800H with Radeon Graphics" and found.cpu.vendor == "AuthenticAMD"
    assert found.cpu.features == ["sse4.2", "avx", "avx2"] and "AVX2" in found.summary()
    assert found.gpus_known
    nvidia, amd = found.gpus
    assert nvidia.name == "NVIDIA GeForce RTX 3060 Laptop GPU (dml:0)" and nvidia.backend == "dml"
    assert nvidia.vram_total_bytes == 6 * GB and nvidia.vram_free_bytes == 4 * GB and nvidia.source == "dxgi"
    assert amd.vendor == "amd" and amd.backend == "dml" and amd.unified_memory and amd.vram_free_bytes is None
    assert found.best_gpu() is nvidia and any('device = "dml"' in n for n in found.notes)
    # without the DirectML build of ONNX Runtime the adapters are listed but nothing bolo runs can use them
    plain = hw.detect(settings(tmp_path), FakeHost(system="windows", machine="AMD64", adapters=adapters[1:]))
    assert plain.gpus[0].backend is None and plain.best_gpu() is None and plain.gpus_known


class _DmlOrt:
    @staticmethod
    def get_available_providers() -> list[str]:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]


_SMI = "C:/Windows/System32/nvidia-smi.exe"
_SMI_QUERY = f"{_SMI} --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits"


def _windows(adapters: list[Any], nvidia_smi: str | None = None) -> FakeHost:
    return FakeHost(
        system="windows", machine="AMD64", cores=16, memory=(16 * GB, 9 * GB), modules={"onnxruntime": _DmlOrt()},
        binaries={"nvidia-smi": _SMI} if nvidia_smi else {}, commands={_SMI_QUERY: nvidia_smi} if nvidia_smi else {},
        adapters=adapters,
    )  # fmt: skip


def test_hybrid_laptop_integrated_first_then_nvidia(tmp_path: Path) -> None:
    """The owner's laptop: DXGI lists the display's integrated Radeon first and the RTX 3060 second. One entry
    per card: the NVIDIA card (CUDA through nvidia-smi) carries its DirectML number, and it is the best GPU."""
    from bolo.winhw import Adapter, preferred_index

    adapters = [
        Adapter(0, "AMD Radeon(TM) Graphics", 0x1002, 512 * MB, 8 * GB, False, 400 * MB, 0),
        Adapter(1, "NVIDIA GeForce RTX 3060 Laptop GPU", 0x10DE, 6 * GB, 8 * GB, False, 5 * GB, 0),
    ]
    assert preferred_index(adapters) == 1  # no IDXGIFactory6 answer: the most dedicated memory
    smi = "NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 5900, 560.94\n"
    found = hw.detect(settings(tmp_path), _windows(adapters, smi))
    nvidia, amd = found.gpus
    assert (nvidia.name, nvidia.backend, nvidia.source) == ("NVIDIA GeForce RTX 3060 Laptop GPU", "cuda", "nvidia-smi")
    assert nvidia.dml_index == 1 and nvidia.dml_preferred and not nvidia.integrated
    assert amd.name == "AMD Radeon(TM) Graphics (dml:0)" and amd.integrated and not amd.dml_preferred
    assert found.best_gpu() is nvidia and found.best_gpu("dml") is None
    assert any('device = "cuda"' in n for n in found.notes) and not any("Radeon" in n for n in found.notes)
    assert found.to_dict()["gpus"][0]["dml_index"] == 1
    # without nvidia-smi or NVML both come from DXGI; "dml" (no number) is the NVIDIA card
    dxgi_only = hw.detect(settings(tmp_path), _windows(adapters))
    amd, nvidia = dxgi_only.gpus
    assert nvidia.backend == "dml" and nvidia.dml_preferred and dxgi_only.best_gpu() is nvidia
    assert any('device = "dml"' in n for n in dxgi_only.notes)
    # Windows' own high-performance choice wins over the memory rule
    adapters[0].high_performance = True
    assert preferred_index(adapters) == 0
    marked = hw.detect(settings(tmp_path), _windows(adapters))
    assert marked.best_gpu() is marked.gpus[1] and any('device = "dml:1"' in n for n in marked.notes)
    adapters[0].high_performance = False
    # a device pinned to the integrated adapter is pointed at the discrete one
    pinned = hw.detect(settings(tmp_path, device="dml:0"), _windows(adapters, smi))
    assert any('device = "dml:0" is integrated graphics' in n and 'device = "cuda"' in n for n in pinned.notes)


def test_nvidia_only_and_amd_discrete_only(tmp_path: Path) -> None:
    from bolo.winhw import Adapter, preferred_index

    nvidia_only = [Adapter(0, "NVIDIA GeForce RTX 4070", 0x10DE, 12 * GB, 16 * GB, False, 11 * GB, 0)]
    found = hw.detect(settings(tmp_path), _windows(nvidia_only, "NVIDIA GeForce RTX 4070, 12282, 11800, 560.94\n"))
    (gpu,) = found.gpus
    assert gpu.backend == "cuda" and gpu.dml_index == 0 and found.best_gpu() is gpu
    assert preferred_index(nvidia_only) == 0
    amd_only = [Adapter(0, "AMD Radeon RX 6700 XT", 0x1002, 12 * GB, 16 * GB, False, 11 * GB, 0)]
    found = hw.detect(settings(tmp_path), _windows(amd_only))
    (gpu,) = found.gpus
    assert gpu.backend == "dml" and not gpu.integrated and found.best_gpu() is gpu
    assert any('device = "dml"' in n for n in found.notes)


def test_intel_integrated_and_amd_discrete(tmp_path: Path) -> None:
    from bolo.winhw import Adapter, preferred_index

    adapters = [
        Adapter(0, "Intel(R) UHD Graphics", 0x8086, 128 * MB, 8 * GB, False, 3 * GB, 0),
        Adapter(1, "AMD Radeon RX 6600M", 0x1002, 8 * GB, 8 * GB, False, 7 * GB, 0),
    ]
    assert preferred_index(adapters) == 1
    found = hw.detect(settings(tmp_path), _windows(adapters))
    intel, amd = found.gpus
    assert intel.integrated and not amd.integrated and amd.dml_preferred
    assert found.best_gpu() is amd and any('device = "dml"' in n for n in found.notes)


def test_integrated_graphics_never_become_the_best_gpu(tmp_path: Path) -> None:
    from bolo.winhw import Adapter, is_integrated

    # an APU whose BIOS carves out 2 GB still counts as integrated
    apu = [Adapter(0, "AMD Radeon(TM) Graphics", 0x1002, 2 * GB, 8 * GB, False, 2 * GB, 0)]
    found = hw.detect(settings(tmp_path), _windows(apu))
    assert found.gpus[0].integrated and found.best_gpu() is None and not any("device =" in n for n in found.notes)
    assert is_integrated("amd", "AMD Radeon 780M Graphics", 4 * GB)
    assert is_integrated("intel", "Intel(R) Arc(TM) Graphics", 2 * GB)  # Meteor Lake's integrated Arc
    assert not is_integrated("intel", "Intel(R) Arc(TM) A770 Graphics", 16 * GB)
    assert not is_integrated("amd", "AMD Radeon RX 7600M XT", 8 * GB)
    assert is_integrated("nvidia", "NVIDIA GeForce MX110", 256 * MB)


@pytest.mark.skipif(sys.platform != "win32", reason="asks Windows itself (IsProcessorFeaturePresent, DXGI)")
def test_windows_probes_on_windows() -> None:
    from bolo import winhw

    features = winhw.cpu_features(platform.machine())
    assert features is not None and ("sse4.2" in features or "neon" in features)
    assert winhw.cpu_name()
    adapters = winhw.dxgi_adapters(include_software=True)
    assert adapters, "every Windows lists at least the Microsoft Basic Render Driver"
    assert all(a.name and a.index >= 0 for a in adapters)
    assert any(a.software for a in adapters) or any(a.dedicated_bytes for a in adapters)
    assert all(not a.software for a in winhw.dxgi_adapters() or [])
    hardware = [a.index for a in adapters if not a.software]
    assert winhw.preferred_index(adapters) in [*hardware, None] and winhw.default_dml_index() in [*hardware, None]
    assert sum(a.high_performance for a in adapters) <= 1 and all(a.luid for a in adapters)
    found = hw.detect(None)
    assert found.cpu.features and found.gpus_known


def test_real_host_detects_without_raising(tmp_path: Path) -> None:
    found = hw.detect(settings(tmp_path))
    assert found.cpu.logical_cores and found.memory.total_bytes and found.disk.free_bytes
    assert set(found.runtimes) == {
        "onnxruntime",
        "sherpa-onnx",
        "ctranslate2",
        "plugin:torch",
        "llama-server",
        "audio.cpp",
    }


def test_benchmark_model_and_storage(tmp_path: Path) -> None:
    model = hw.benchmark_model()
    assert model[:2] == b"\x08\x08" and b"Conv" in model and b"MatMul" in model and len(model) < 1024
    machine = hw.detect(settings(tmp_path), laptop())
    result = hw.run_benchmark(machine, budget_s=0.05)
    assert result["workload"] in ("onnxruntime", "numpy") and result["ms"] > 0 and result["speed"] > 0
    assert result["threads"] == 4 and result["cpu_model"] == machine.cpu.model and result["at"].endswith("Z")
    path = hw.save_benchmark(tmp_path / "data", result)
    assert path.name == "hardware-benchmark.json"
    assert hw.load_benchmark(tmp_path / "data", machine) == result
    # another machine's benchmark (the data directory was copied) is not used
    other = hw.detect(settings(tmp_path), FakeHost(files={"/proc/cpuinfo": cpuinfo(X86_AVX512, 16)}, cores=16))
    assert hw.load_benchmark(tmp_path / "data", other) is None
    path.write_text("not json")
    assert hw.load_benchmark(tmp_path / "data", machine) is None
    assert hw.load_benchmark(tmp_path / "nowhere", machine) is None


def test_a_benchmark_survives_the_windows_cpu_name_change(tmp_path: Path) -> None:
    """bolo before 0.6 stored Windows's processor identifier as the CPU's name; 0.6 reads the marketing name. A
    benchmark from then still counts (same processor identifier and cores); one from another CPU is not used, and
    `bolo doctor` / `bolo recommend` say why instead of silently estimating."""
    ident = "AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD"

    def windows(name: str | None, processor: str = ident, cores: int = 16) -> hw.Hardware:
        host = FakeHost(system="windows", machine="AMD64", cores=cores, processor=processor,
                        windows_cpu=(name, ["sse4.2", "avx", "avx2"]))  # fmt: skip
        return hw.detect(settings(tmp_path), host)

    machine = windows("AMD Ryzen 7 5800H with Radeon Graphics")
    assert machine.cpu.model == "AMD Ryzen 7 5800H with Radeon Graphics" and machine.cpu.processor_id == ident
    assert "processor_id" not in machine.cpu.to_dict()
    data = tmp_path / "data"
    old = {"version": hw.BENCHMARK_VERSION, "at": "2026-11-01T10:00:00Z", "cpu_model": ident, "threads": 4,
           "logical_cores": 16, "usable_cores": 16.0, "workload": "onnxruntime", "ms": 2.0, "speed": 1.7}  # fmt: skip
    hw.save_benchmark(data, old)
    assert machine.cpu.usable_cores == 16.0
    assert hw.benchmark_status(data, machine) == (old, None)
    # a new benchmark records both, and still matches if the name is read differently again
    new = hw.run_benchmark(machine, budget_s=0.05)
    assert new["processor_id"] == ident
    hw.save_benchmark(data, new)
    assert hw.load_benchmark(data, windows("AMD Ryzen 7 5800H")) == new
    # another CPU: not used, and the reason names it
    doc, why = hw.benchmark_status(data, windows("Intel Core i7-12700H", "Intel64 Family 6 Model 154 Stepping 3"))
    assert doc is None and why and "another CPU (AMD Ryzen 7 5800H with Radeon Graphics" in why
    assert "bolo recommend --benchmark" in why
    doc, why = hw.benchmark_status(data, windows("AMD Ryzen 7 5800H with Radeon Graphics", cores=8))
    assert doc is None and why and "other cores" in why
    # a processor identifier that only names the architecture (Linux) is not enough
    hw.save_benchmark(data, {**old, "cpu_model": "x86_64"})
    assert hw.load_benchmark(data, windows("Some CPU", processor="x86_64")) is None
    assert hw.benchmark_status(tmp_path / "nowhere", machine) == (None, None)
    # the advisor says so in the speed note (`bolo doctor`'s speed line)
    from bolo.core import Bolo

    advisor = Bolo(settings(tmp_path)).advisor
    advisor.use_hardware(windows("Intel Core i7-12700H", "Intel64 Family 6 Model 154 Stepping 3"))
    speed = advisor.speed()
    assert speed.source == "heuristic" and "another CPU" in speed.note and "bolo recommend --benchmark" in speed.note


@pytest.mark.parametrize(("threads", "expected"), [(4, 1.0), (8, 1.0), (2, 0.574), (1, 0.330)])
def test_thread_scale(threads: int, expected: float) -> None:
    assert hw.thread_scale(threads) == pytest.approx(expected, abs=0.001)
