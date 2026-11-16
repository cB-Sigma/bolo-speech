"""CUDA's libraries for ONNX Runtime and CTranslate2: made loadable, checked, and a fallback said out loud.

No GPU here (CI has none): ONNX Runtime and faster-whisper are stand-ins that fail the way they fail on a Windows
laptop whose CUDA DLLs are not found (RTX 3060 Laptop, onnxruntime-gpu 1.32.0, ctranslate2 4.9.0).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bolo import cuda_libs, ort_providers
from bolo.cuda_libs import ProviderFallback

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"

#: what onnxruntime-gpu 1.32.0 printed on the laptop
ORT_LOG = (
    "2026-11-14 10:12:01.5 [E:onnxruntime:Default, provider_bridge_ort.cc:2251 onnxruntime::TryGetProviderInfo_CUDA] "
    "D:\\a\\_work\\1\\s\\onnxruntime\\core\\session\\provider_bridge_ort.cc:1844 onnxruntime::ProviderLibrary::Get "
    '[ONNXRuntimeError] : 1 : FAIL : Error loading "C:\\venv\\Lib\\site-packages\\onnxruntime\\capi\\'
    'onnxruntime_providers_cuda.dll" which depends on "cublasLt64_13.dll" which is missing. (Error 126: "The '
    'specified module could not be found.")\n'
    "2026-11-14 10:12:01.5 [W:onnxruntime:Default, onnxruntime_pybind_state.cc:1037 "
    "onnxruntime::python::CreateExecutionProviderInstance] Failed to create CUDAExecutionProvider.\n"
)
CT2_ERROR = "Library cublas64_12.dll is not found or cannot be loaded"


@pytest.fixture(autouse=True)
def fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_libs, "_ort_prepared", False)
    monkeypatch.setattr(cuda_libs, "_ct2_prepared", False)
    monkeypatch.setattr(cuda_libs, "_dirs_added", set())
    ort_providers._warned.clear()


def test_missing_libraries_from_what_the_runtimes_say() -> None:
    assert cuda_libs.missing_libraries(ORT_LOG) == ["cublasLt64_13.dll"]
    assert cuda_libs.missing_libraries(CT2_ERROR) == ["cublas64_12.dll"]
    linux = "libcublasLt.so.13: cannot open shared object file: No such file or directory"
    assert cuda_libs.missing_libraries(linux) == ["libcublasLt.so.13"]
    itself = 'Error loading "C:\\x\\onnxruntime_providers_cuda.dll" (Error 126)'
    assert cuda_libs.missing_libraries(itself) == ["onnxruntime_providers_cuda.dll"]
    assert cuda_libs.missing_libraries("") == []
    assert "cublasLt64_13.dll" in cuda_libs.ort_cuda_libraries(13, windows=True)
    assert "cufft64_12.dll" in cuda_libs.ort_cuda_libraries(13, windows=True)
    assert "libcublas.so.12" in cuda_libs.ort_cuda_libraries(12, windows=False)
    assert cuda_libs.ort_cuda_libraries(None) == []
    assert cuda_libs.ct2_cuda_libraries(windows=True)[0] == "cublas64_12.dll"
    assert cuda_libs.unloadable(["a.dll", "b.dll"], load=lambda n: n == "a.dll") == ["b.dll"]


def nvidia_tree(root: Path) -> Path:
    """site-packages/nvidia as NVIDIA's wheels lay it out on Windows (cu13 for CUDA 13) and on Linux."""
    nvidia = root / "nvidia"
    for folder, name in (
        ("cu13/bin/x86_64", "cublasLt64_13.dll"), ("cublas/bin", "cublas64_12.dll"),
        ("cudnn/bin", "cudnn64_9.dll"), ("cuda_nvrtc/bin", "nvrtc64_120_0.dll"), ("cublas/lib", "libcublas.so.12"),
    ):  # fmt: skip
        (nvidia / folder).mkdir(parents=True, exist_ok=True)
        (nvidia / folder / name).write_bytes(b"x")
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)  # empty: not added
    return nvidia


def test_the_bin_folders_of_nvidias_wheels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nvidia = nvidia_tree(tmp_path)
    monkeypatch.setattr("sys.path", [str(tmp_path), "", str(tmp_path / "nothing")])
    assert cuda_libs.nvidia_roots() == [nvidia]
    windows = cuda_libs.nvidia_lib_dirs(windows=True)
    assert windows == [
        nvidia / "cu13/bin/x86_64",
        nvidia / "cublas/bin",
        nvidia / "cuda_nvrtc/bin",
        nvidia / "cudnn/bin",
    ]
    assert cuda_libs.nvidia_lib_dirs(windows=False) == [nvidia / "cublas/lib"]


def test_windows_dll_folders_are_added_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nvidia_tree(tmp_path)
    monkeypatch.setattr("sys.path", [str(tmp_path)])
    monkeypatch.setattr(cuda_libs, "_windows", lambda: True)
    added: list[str] = []
    monkeypatch.setattr(os, "add_dll_directory", lambda p: added.append(p) or p, raising=False)
    monkeypatch.setenv("PATH", "windows-system")
    folders = cuda_libs.add_dll_directories()
    assert len(folders) == 4 and added == [str(f) for f in folders]
    path = os.environ["PATH"].split(os.pathsep)
    assert path[-1] == "windows-system" and str(tmp_path / "nvidia" / "cudnn" / "bin") in path
    assert cuda_libs.add_dll_directories() == [] and len(added) == 4  # once
    monkeypatch.setattr(cuda_libs, "_windows", lambda: False)
    monkeypatch.setattr(cuda_libs, "_dirs_added", set())
    assert cuda_libs.add_dll_directories() == []


def test_preload_dlls_is_called_once_before_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import onnxruntime as ort

    calls: list[int] = []

    def preload() -> None:
        calls.append(1)
        print("Failed to load cudnn64_9.dll")  # what preload_dlls prints goes to the debug log, not stdout

    monkeypatch.setattr(ort, "preload_dlls", preload, raising=False)
    cuda_libs.prepare_ort_cuda()
    cuda_libs.prepare_ort_cuda()
    assert calls == [1]


def test_native_stderr_is_caught_and_passed_on(capfd: pytest.CaptureFixture[str]) -> None:
    with cuda_libs.native_stderr() as said:
        os.write(2, b"from native code\n")
    assert said == ["from native code\n"]
    assert "from native code" in capfd.readouterr().err


class FakeSession:
    """onnxruntime.InferenceSession as the laptop's onnxruntime-gpu built it: CUDA asked for, the CPU given."""

    gives: list[str] = [CPU]

    def __init__(self, model: Any, sess_options: Any = None, providers: Any = None, **_: Any) -> None:
        self.asked = providers
        if any(ort_providers._name(p) == CUDA for p in providers) and CUDA not in self.gives:
            os.write(2, ORT_LOG.encode())

    def get_providers(self) -> list[str]:
        return list(self.gives)


def test_a_cuda_session_on_the_cpu_is_a_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import onnxruntime as ort

    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    prepared: list[int] = []
    monkeypatch.setattr(cuda_libs, "prepare_ort_cuda", lambda: prepared.append(1))
    with caplog.at_level(logging.WARNING, logger="bolo.ort_providers"), cuda_libs.watch() as watched:
        session = ort_providers.create_session("m.onnx", None, [(CUDA, {"device_id": 0}), CPU], role="encoder")
        ort_providers.create_session("m.onnx", None, [CUDA, CPU])
    assert isinstance(session, FakeSession) and prepared == [1, 1]
    assert watched.whole and len(watched.fallbacks) == 2
    fallback = watched.fallbacks[0]
    assert fallback.requested == CUDA and fallback.missing == ["cublasLt64_13.dll"] and fallback.role == "encoder"
    message = fallback.message()
    assert "ONNX Runtime could not start CUDA for its encoder (cublasLt64_13.dll is missing)" in message
    assert "runs on the CPU" in message and "onnxruntime-gpu[cuda,cudnn]" in message
    assert fallback.to_dict()["missing"] == ["cublasLt64_13.dll"]
    assert sum("cublasLt64_13.dll" in r.getMessage() for r in caplog.records) == 1  # logged once
    # the CPU alone: nothing to check; CUDA kept: counted as accelerated
    with cuda_libs.watch() as cpu_only:
        ort_providers.create_session("m.onnx", None, [CPU])
    assert not cpu_only.fallbacks and cpu_only.accelerated == 0
    monkeypatch.setattr(FakeSession, "gives", [CUDA, CPU])
    with cuda_libs.watch() as kept:
        ort_providers.create_session("m.onnx", None, [CUDA, CPU])
    assert kept.accelerated == 1 and not kept.fallbacks and not kept.whole


def test_no_missing_dll_in_the_log_probes_the_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    import onnxruntime as ort

    class Quiet(FakeSession):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(ort, "InferenceSession", Quiet)
    monkeypatch.setattr(cuda_libs, "prepare_ort_cuda", lambda: None)
    monkeypatch.setattr(cuda_libs, "ort_cuda_major", lambda: 13)
    monkeypatch.setattr(cuda_libs, "unloadable", lambda names: [n for n in names if "cudnn" in n])
    with cuda_libs.watch() as watched:
        ort_providers.create_session("m.onnx", None, [CUDA, CPU])
    assert watched.fallbacks[0].missing == [n for n in cuda_libs.ort_cuda_libraries(13) if "cudnn" in n]


def test_a_build_without_the_provider_is_recorded_too() -> None:
    with cuda_libs.watch() as watched:
        assert ort_providers.providers_for("cuda", [CPU]) == [CPU]
    (fallback,) = watched.fallbacks
    assert fallback.requested == CUDA and "no CUDAExecutionProvider" in fallback.message()
    with cuda_libs.watch() as auto:
        ort_providers.providers_for("auto", [CPU])  # auto asks for nothing in particular
    assert not auto.fallbacks


# -- faster-whisper ---------------------------------------------------------------------------------------------
class FakeWhisper:
    """faster_whisper.WhisperModel on a machine where CUDA 12's cuBLAS is not found: it loads, and the first
    matrix product fails."""

    made: list[tuple[str, str]] = []
    error = CT2_ERROR

    def __init__(self, path: str, device: str, compute_type: str, **_: Any) -> None:
        self.made.append((device, compute_type))
        self.device = device
        self.model = type("M", (), {"is_multilingual": True})()

    def transcribe(self, audio: Any, **_: Any) -> Any:
        if self.device == "cuda":
            raise RuntimeError(self.error)
        return iter([]), type("Info", (), {"language": "en"})()


def test_faster_whisper_falls_back_to_the_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fw = pytest.importorskip("faster_whisper")
    from bolo.engines.base import LoadOptions
    from bolo.engines.faster_whisper import FasterWhisperSTT

    monkeypatch.setattr(fw, "WhisperModel", FakeWhisper)
    monkeypatch.setattr(fw, "BatchedInferencePipeline", lambda model: model)
    monkeypatch.setattr(cuda_libs, "prepare_ct2_cuda", lambda: None)
    FakeWhisper.made = []
    with cuda_libs.watch() as watched:
        stt = FasterWhisperSTT(tmp_path, {}, LoadOptions(device="cuda:0"))
    assert stt.device == "cpu" and FakeWhisper.made == [("cuda", "float16"), ("cpu", "int8")]
    (fallback,) = watched.fallbacks
    assert fallback.runtime == "ctranslate2" and fallback.missing == ["cublas64_12.dll"] and watched.whole
    assert "CTranslate2 could not start CUDA (cublas64_12.dll is missing)" in fallback.message()
    assert "nvidia-cublas-cu12" in fallback.message()
    # a CUDA error of the model itself (out of memory) is not a missing library: it is raised
    monkeypatch.setattr(FakeWhisper, "error", "CUDA failed with error out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        FasterWhisperSTT(tmp_path, {}, LoadOptions(device="cuda"))
    FakeWhisper.made = []
    cpu = FasterWhisperSTT(tmp_path, {}, LoadOptions(device="cpu"))
    assert cpu.device == "cpu" and FakeWhisper.made == [("cpu", "int8")]


# -- placement and bolo doctor --------------------------------------------------------------------------------------
def test_placement_moves_to_the_cpu() -> None:
    from bolo.placement import Placement

    p = Placement(
        device="cuda:0", requested="cuda", gpu_bytes=300, cpu_bytes=10, gpus=[0], graphs={"encoder": "cuda:0"}
    )
    fb = ProviderFallback("onnxruntime", CUDA, missing=["cublasLt64_13.dll"])
    p.fell_back([fb.to_dict()], fb.message(), whole=True)
    assert p.device == "cpu" and p.gpu_bytes == 0 and p.cpu_bytes == 310 and p.gpus == []
    assert p.graphs == {"encoder": "cpu"} and p.processor() == "100% CPU"
    out = p.to_dict()
    assert out["provider_fallback"][0]["missing"] == ["cublasLt64_13.dll"] and "cublasLt64_13.dll" in out["notice"]
    partial = Placement(device="cuda:0", gpu_bytes=300, notice="split")
    partial.fell_back([fb.to_dict()], "one graph on the CPU", whole=False)
    assert partial.device == "cuda:0" and partial.notice == "split; one graph on the CPU"


def test_doctor_lines() -> None:
    assert cuda_libs.doctor_wants_cuda("cuda:1", False) and cuda_libs.doctor_wants_cuda("auto", True)
    assert not cuda_libs.doctor_wants_cuda("auto", False) and not cuda_libs.doctor_wants_cuda("dml", True)
    lines = cuda_libs.doctor_lines(
        ort_check=lambda: (False, "the CUDA provider does not start (cublasLt64_13.dll missing): pip ..."),
        ct2_check=lambda: (True, "ctranslate2 sees 1 CUDA device(s)"),
        have_ort=True,
        have_ct2=True,
    )
    assert lines[0] == "cuda:" and lines[1].startswith("  -- onnxruntime") and "cublasLt64_13.dll" in lines[1]
    assert lines[2].startswith("  ok faster-whisper") and cuda_libs.DOCS in lines[3]
    assert cuda_libs.doctor_lines(have_ort=False, have_ct2=False)[1].strip().startswith("neither")


def test_doctor_checks_on_this_machine() -> None:
    """Here (CI, no GPU) both checks say what is missing rather than fail."""
    ok, text = cuda_libs.check_ort_cuda()
    assert not ok and text
    pytest.importorskip("ctranslate2")
    ok, text = cuda_libs.check_ct2_cuda()
    assert not ok and "ctranslate2" in text


def test_doctor_tests_cuda_when_device_asks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from bolo import cli
    from bolo.config import Settings

    monkeypatch.setattr(cuda_libs, "doctor_lines", lambda: ["cuda:", "  -- onnxruntime  cublasLt64_13.dll missing"])
    cli._doctor_cuda(Settings(device="cuda"))
    assert "cublasLt64_13.dll missing" in capsys.readouterr().out
    cli._doctor_cuda(Settings(device="cpu"))
    assert capsys.readouterr().out == ""


def test_benchmark_graph_feeds_match_the_doctor() -> None:
    """The doctor runs the hardware benchmark's graph on CUDA with these inputs; on the CPU they run."""
    import onnxruntime as ort

    from bolo.hardware import benchmark_model

    session = ort.InferenceSession(benchmark_model(), providers=[CPU])
    out = session.run(None, {
        "x": np.zeros((1, 64, 96, 96), np.float32), "w": np.zeros((64, 64, 3, 3), np.float32),
        "a": np.zeros((512, 512), np.float32),
    })  # fmt: skip
    assert len(out) == 2
