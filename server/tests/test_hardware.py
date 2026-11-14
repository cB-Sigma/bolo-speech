"""Hardware detection against described machines: fake system files and commands, never the real host."""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from bolo import hardware as hw
from bolo.config import Settings

GB = 1024**3

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
    ) -> None:
        super().__init__("/nonexistent-root", env or {})
        self._system, self._machine = system, machine
        self.files = files or {}
        self.commands = commands or {}
        self.binaries = binaries or {}
        self.modules = modules or {}
        self._cores, self._affinity, self._memory = cores, affinity, memory
        self.ran: list[list[str]] = []

    def system(self) -> str:
        return self._system

    def machine(self) -> str:
        return self._machine

    def processor(self) -> str | None:
        return None

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
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    host = FakeHost(
        files={"/proc/cpuinfo": cpuinfo(X86_AVX2, 8)},
        commands={f"{binary} --version": "version: 6123 (abc)\n", f"{binary} --list-devices": "Available devices:\n  CUDA0: RTX 4090\n"},
        cores=8,
    )  # fmt: skip
    found = hw.detect(settings(tmp_path, llama_server_bin=str(binary)), host, deep=True).runtimes["llama-server"]
    assert found["installed"] is True and found["source"] == "settings" and found["version"] == "version: 6123 (abc)"
    assert found["gpu"] is True and found["gpu_devices"] == ["CUDA0: RTX 4090"]


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


@pytest.mark.parametrize(("threads", "expected"), [(4, 1.0), (8, 1.0), (2, 0.574), (1, 0.330)])
def test_thread_scale(threads: int, expected: float) -> None:
    assert hw.thread_scale(threads) == pytest.approx(expected, abs=0.001)
