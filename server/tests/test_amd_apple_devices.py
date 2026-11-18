"""AMD GPUs (ROCm, MIGraphX) and Apple silicon (CoreML, Metal, MPS, MLX): which device each engine gets.

None of this runs on a real AMD GPU in bolo's CI: the selection logic is checked here against fake sysfs trees,
provider lists and hardware descriptions (docs/rocm.md). Apple silicon runs for real in the macOS CI job.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bolo import hardware as hw
from bolo import ort_providers, placement, tts_runtimes
from bolo.accel import Accelerators
from bolo.catalog import ModelSpec
from bolo.engines.whispercpp import families_of
from bolo.hardware import Container, Cpu, Disk, Gpu, Hardware, Memory
from bolo.ort_providers import CPU, auto_device, providers_for, sherpa_provider
from bolo.recommend import gpu_for
from test_hardware import FakeHost

GB = 1024**3
MIGRAPHX = "MIGraphXExecutionProvider"
ROCM = "ROCMExecutionProvider"
CUDA = "CUDAExecutionProvider"
COREML = "CoreMLExecutionProvider"


# -- ONNX Runtime ------------------------------------------------------------------------------------------------
def test_rocm_asks_for_migraphx_then_the_rocm_provider() -> None:
    assert providers_for("rocm", [MIGRAPHX, ROCM, CPU]) == [MIGRAPHX, CPU]
    assert providers_for("rocm", [ROCM, CPU]) == [ROCM, CPU]
    assert providers_for("hip:1", [MIGRAPHX, CPU]) == [(MIGRAPHX, {"device_id": 1}), CPU]
    assert providers_for("migraphx", [MIGRAPHX, CPU]) == [MIGRAPHX, CPU]


def test_rocm_without_an_amd_build_runs_on_the_cpu_and_says_how(caplog: pytest.LogCaptureFixture) -> None:
    ort_providers._warned.clear()
    assert providers_for("rocm", [CPU]) == [CPU]
    assert "onnxruntime-migraphx" in caplog.text


def test_auto_takes_the_amd_provider_only_with_an_amd_gpu() -> None:
    assert providers_for("auto", [MIGRAPHX, CPU]) == [MIGRAPHX, CPU]
    assert auto_device("rocm:0", [MIGRAPHX, CPU]) == "rocm"
    assert auto_device("rocm:0", [ROCM, CPU]) == "rocm"
    assert auto_device(None, [MIGRAPHX, CPU]) == "cpu"  # onnxruntime-migraphx on a machine without an AMD GPU
    assert auto_device("cuda:0", [CUDA, MIGRAPHX, CPU]) == "cuda"
    assert auto_device("coreml", [COREML, CPU]) == "coreml"


def test_doctor_names_the_amd_build() -> None:
    problems = ort_providers.ort_problems("rocm", {"onnxruntime": "1.27.0"}, [CPU])
    assert any("onnxruntime-migraphx" in p for p in problems)
    assert ort_providers.ort_problems("rocm", {"onnxruntime-migraphx": "1.27.1"}, [MIGRAPHX, CPU]) == []
    assert "AMD MIGraphX build" in ort_providers.no_cuda_provider({"onnxruntime-migraphx": "1.27.1"})


def test_sherpa_onnx_has_no_rocm_build() -> None:
    assert sherpa_provider("rocm") == "cpu"
    assert sherpa_provider("coreml") == sherpa_provider("metal") == sherpa_provider("mps") == "coreml"


# -- accelerator memory: AMD through sysfs -------------------------------------------------------------------------
def _card(root: Path, n: int, vendor: str, total: int | None = None, used: int = 0, name: str = "") -> None:
    base = root / f"card{n}" / "device"
    base.mkdir(parents=True)
    (base / "vendor").write_text(vendor + "\n")
    (base / "device").write_text("0x744c\n")
    if total is not None:
        (base / "mem_info_vram_total").write_text(f"{total}\n")
        (base / "mem_info_vram_used").write_text(f"{used}\n")
        (base / "gpu_busy_percent").write_text("7\n")
    if name:
        (base / "product_name").write_text(name + "\n")


def _vm() -> Any:
    return SimpleNamespace(total=64 * GB, available=40 * GB)


def test_amd_gpus_are_read_from_sysfs(tmp_path: Path) -> None:
    _card(tmp_path, 0, "0x8086")  # Intel graphics: not an AMD GPU
    _card(tmp_path, 1, "0x1002", 24 * GB, 4 * GB, "AMD Radeon RX 7900 XTX")
    _card(tmp_path, 2, "0x1002", 512 * 1024 * 1024, 100, "")  # an APU's carve-out
    _card(tmp_path, 3, "0x1002")  # no amdgpu counters: skipped
    acc = Accelerators(
        nvml=None, which=lambda _: None, system="linux", drm_root=str(tmp_path), virtual_memory=_vm, environ={}
    )
    found = acc.devices()
    assert [(d.index, d.vendor, d.backend) for d in found] == [(0, "amd", "rocm"), (1, "amd", "rocm")]
    big, apu = found
    assert big.name == "AMD Radeon RX 7900 XTX" and big.free_bytes == 20 * GB and big.utilization == 7.0
    assert big.process_bytes is None and not big.unified
    assert apu.unified and apu.free_bytes == 40 * GB
    assert acc.default_device() == "rocm:0" and acc.source == "sysfs"
    assert acc.device("rocm:1") is apu and acc.device("hip:1") is apu


def test_hip_visible_devices_renumbers(tmp_path: Path) -> None:
    _card(tmp_path, 0, "0x1002", 16 * GB, 0, "first")
    _card(tmp_path, 1, "0x1002", 16 * GB, 0, "second")
    acc = Accelerators(
        nvml=None, which=lambda _: None, system="linux", drm_root=str(tmp_path), environ={"HIP_VISIBLE_DEVICES": "1"}
    )
    found = acc.devices()
    assert [(d.index, d.name, d.physical_index) for d in found] == [(0, "second", 1)]


def test_no_amd_gpu_on_other_systems(tmp_path: Path) -> None:
    _card(tmp_path, 0, "0x1002", 16 * GB, 0, "x")
    vm = SimpleNamespace(total=16 * GB, available=8 * GB)
    acc = Accelerators(
        nvml=None,
        which=lambda _: None,
        system="Darwin",
        machine="arm64",
        drm_root=str(tmp_path),
        virtual_memory=lambda: vm,
    )
    assert [d.backend for d in acc.devices()] == ["metal"] and acc.default_device() == "coreml"


def test_families() -> None:
    assert placement.family("hip:1") == placement.family("migraphx") == "rocm"
    assert placement.family("mps") == placement.family("coreml") == placement.family("mlx") == "metal"
    assert "rocm" in placement.engine_families("plugin:torch") and "metal" in placement.engine_families("mlx")


# -- whisper.cpp -------------------------------------------------------------------------------------------------
def test_whispercpp_build_is_read_from_its_system_information() -> None:
    assert families_of("WHISPER : COREML = 0 | OPENVINO = 0 | MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 |") == {"metal"}
    assert families_of("CUDA : ARCHS = 890 | USE_GRAPHS = 1 | CPU : AVX2 = 1 |") == {"cuda"}
    assert families_of("HIP : NO_VMM = 1 | CPU : AVX2 = 1") == {"rocm"}
    assert families_of("WHISPER : COREML = 0 | OPENVINO = 0 | CPU : SSE3 = 1 | AVX2 = 1 |") == frozenset()


# -- llama.cpp builds ----------------------------------------------------------------------------------------------
def test_rocm_device_asks_for_the_hip_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts_runtimes, "platform_key", lambda: "linux-x86_64")
    assert "rocm" in tts_runtimes.pinned_flavours(tts_runtimes.LLAMA)
    assert tts_runtimes.flavour_for(tts_runtimes.LLAMA, "rocm") == "rocm"
    assert tts_runtimes.flavour_for(tts_runtimes.LLAMA, "hip:1") == "rocm"
    release = tts_runtimes.release_for(tts_runtimes.LLAMA, "rocm")
    assert release is not None and "ubuntu-rocm" in release.url and len(release.sha256) == 64
    assert tts_runtimes._preference(tts_runtimes.LLAMA, "auto")[:2] == ["cuda", "rocm"]
    monkeypatch.setattr(tts_runtimes, "platform_key", lambda: "darwin-arm64")
    assert tts_runtimes.flavour_for(tts_runtimes.LLAMA, "rocm") == "cpu"  # no HIP build for macOS
    note = tts_runtimes.missing_build_note(tts_runtimes.LLAMA, "rocm")
    assert note is not None and "ROCm" in note


def test_hip_libraries_say_rocm(tmp_path: Path) -> None:
    (tmp_path / "libggml-hip.so").write_text("")
    assert tts_runtimes.detect_flavour(tmp_path) == "rocm"


# -- bolo doctor ---------------------------------------------------------------------------------------------------
def _amd_host(rocm: bool) -> FakeHost:
    files = {
        "/sys/class/drm/card0/device/vendor": "0x1002\n",
        "/sys/class/drm/card0/device/device": "0x744c\n",
        "/sys/class/drm/card0/device/mem_info_vram_total": f"{24 * GB}\n",
        "/sys/class/drm/card0/device/mem_info_vram_used": f"{GB}\n",
        "/sys/class/drm/card0/device/product_name": "AMD Radeon RX 7900 XTX\n",
    }
    if rocm:
        files["/opt/rocm/.info/version"] = "6.4.1-120\n"
    return FakeHost(files=files)


def test_doctor_finds_an_amd_gpu_and_rocm() -> None:
    host = _amd_host(rocm=True)
    assert hw.rocm_runtime(host) == {"installed": True, "version": "6.4.1-120", "path": "/opt/rocm"}
    gpus, known = hw.detect_gpus(host, Cpu(), Memory())
    assert known and [(g.vendor, g.name, g.backend) for g in gpus] == [("amd", "AMD Radeon RX 7900 XTX", "rocm")]
    assert hw.device_for(gpus[0]) == "rocm"
    found = hw.detect(None, host)
    assert found.runtimes["rocm"]["installed"] and not any("ROCm is not installed" in n for n in found.notes)


def test_doctor_says_rocm_is_missing() -> None:
    host = _amd_host(rocm=False)
    gpus, _ = hw.detect_gpus(host, Cpu(), Memory())
    assert gpus[0].backend is None
    found = hw.detect(None, host)
    assert any("ROCm is not installed" in n and "docs/rocm.md" in n for n in found.notes)


def test_doctor_reports_mlx_and_whispercpp() -> None:
    metal = SimpleNamespace(is_available=lambda: True)
    mx = SimpleNamespace(__version__="0.32.4", metal=metal, default_device=lambda: "Device(gpu, 0)")
    pw = SimpleNamespace(whisper_print_system_info=lambda: "MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 |")
    host = FakeHost(system="darwin", machine="arm64", modules={"mlx.core": mx, "_pywhispercpp": pw})
    rt = hw.detect_runtimes(host)
    assert rt["mlx"]["installed"] and rt["mlx"]["metal"] and rt["mlx"]["version"] == "0.32.4"
    assert rt["whisper.cpp"]["backends"] == ["metal"] and rt["whisper.cpp"]["gpu"]
    assert "rocm" not in rt


# -- bolo recommend ------------------------------------------------------------------------------------------------
def _spec(engine: str, kind: str = "whisper", **kw: Any) -> ModelSpec:
    return ModelSpec.from_dict(
        {"id": "m", "task": "stt", "engine": engine, "kind": kind, "family": "F", "languages": ["en"],
         "license": "MIT", "size_bytes": 10**8, "source": {"repo": "example/m"}, **kw}
    )  # fmt: skip


def _machine(gpu: Gpu, device: str, runtimes: dict[str, Any]) -> Hardware:
    return Hardware(
        system="macos" if gpu.backend == "metal" else "linux",
        cpu=Cpu("CPU", "Test", "arm64", 8, 8, 8, []),
        memory=Memory(32 * GB, 20 * GB, None),
        disk=Disk("/data", 500 * GB, 100 * GB),
        gpus=[gpu],
        runtimes=runtimes,
        container=Container(detected=False),
        device=device,
        threads=8,
    )


def test_recommend_counts_apple_silicon_with_auto() -> None:
    apple = Gpu("apple", "Apple M3 GPU", 32 * GB, 20 * GB, unified_memory=True, backend="metal", source="platform")
    rt = {"onnxruntime": {"providers": [COREML, CPU]}, "mlx": {"metal": True}, "whisper.cpp": {"backends": ["metal"]}}
    on = _machine(apple, "auto", rt)
    assert gpu_for(_spec("sherpa-onnx"), on) == (apple, None)
    assert gpu_for(_spec("onnxruntime", "smart_turn"), on) == (apple, None)
    assert gpu_for(_spec("mlx", "mlx-audio-stt"), on) == (apple, None)
    assert gpu_for(_spec("whisper.cpp", "whispercpp"), on) == (apple, None)
    assert gpu_for(_spec("faster-whisper", "faster-whisper"), on)[0] is None
    off = _machine(apple, "cpu", rt)
    assert gpu_for(_spec("sherpa-onnx"), off)[0] is None
    no_coreml = _machine(apple, "auto", {"onnxruntime": {"providers": [CPU]}})
    assert gpu_for(_spec("onnxruntime", "smart_turn"), no_coreml) == (None, "this ONNX Runtime has no CoreML provider")


def test_recommend_counts_an_amd_gpu_with_rocm() -> None:
    amd = Gpu("amd", "AMD Radeon RX 7900 XTX", 24 * GB, 23 * GB, backend="rocm", source="sysfs")
    rt = {"onnxruntime": {"providers": [MIGRAPHX, CPU]}, "llama-server": {"gpu": True}}
    on = _machine(amd, "auto", rt)
    assert gpu_for(_spec("onnxruntime", "smart_turn"), on) == (amd, None)
    assert gpu_for(_spec("tts-server", "tts-server"), on) == (amd, None)
    assert gpu_for(_spec("sherpa-onnx"), on) == (None, "sherpa-onnx has no ROCm build")
    assert gpu_for(_spec("mlx", "mlx-audio-stt"), on)[0] is None
