"""Where ONNX Runtime and sherpa-onnx models really run: the end-of-turn model on the CPU with DirectML (and on the CPU
after a runtime failure on an accelerator), ``device = "auto"`` with onnxruntime-directml, and sherpa-onnx's CPU
build with ``device = "cuda"``. Everything here is faked: CI has no GPU (and runs on Windows too)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bolo import cuda_libs, ort_providers
from bolo.accel import DeviceMemory
from bolo.catalog import ModelSpec
from bolo.config import Settings
from bolo.core import Bolo
from bolo.engines.base import LoadOptions
from bolo.lifecycle import LoadedModel, ModelManager
from bolo.ort_providers import CPU, auto_device, providers_for, sherpa_build, sherpa_provider
from bolo.placement import Placement

CUDA = "CUDAExecutionProvider"
DML = "DmlExecutionProvider"
MB = 1024 * 1024
SHERPA_SAYS = (
    "Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON. Available providers: CPUExecutionProvider, . Fallback to cpu!\n"
)


def spec(model_id: str, engine: str, kind: str, task: str = "stt") -> ModelSpec:
    return ModelSpec.from_dict({
        "id": model_id, "task": task, "engine": engine, "kind": kind, "family": "Test", "languages": ["en"],
        "license": "MIT", "builtin": True, "size_bytes": 8 * MB,
    })  # fmt: skip


class FakeAccel:
    """bolo.accel.Accelerators with one NVIDIA GPU (or none) and nothing measured."""

    def __init__(self, nvidia: bool = True) -> None:
        self.nvidia = nvidia
        self.directml = False

    def default_device(self) -> str | None:
        return "cuda:0" if self.nvidia else None

    def devices(self, fresh: bool = False) -> list[DeviceMemory]:
        return []

    def device(self, device: str, fresh: bool = False) -> DeviceMemory | None:
        return None

    def cuda_devices(self) -> list[DeviceMemory]:
        return []

    def index_of(self, device: str) -> int:
        return 0

    def close(self) -> None:
        pass


def manager(settings: Settings, device: str, accel: FakeAccel | None = None) -> ModelManager:
    settings.device = device
    core = Bolo(settings)
    core.manager.accel = accel or FakeAccel()  # type: ignore[assignment]
    return core.manager


# -- 1. the end-of-turn model and DirectML --------------------------------------------------------------------------
class FakeSession:
    """onnxruntime.InferenceSession: on an accelerator its run fails as DirectML's does for Smart Turn."""

    made: list[list[str]] = []

    def __init__(self, model: Any, sess_options: Any = None, providers: Any = None, **_: Any) -> None:
        self.names = [ort_providers._name(p) for p in providers or [CPU]]
        self.made.append(self.names)

    def get_providers(self) -> list[str]:
        return list(self.names)

    def get_inputs(self) -> list[Any]:
        return [SimpleNamespace(name="input_features")]

    def run(self, outputs: Any, feeds: dict[str, Any]) -> list[Any]:
        if self.names[0] != CPU:
            raise RuntimeError(
                "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Reshape node. "
                "Name:'node_view_1' Status Message: MLOperatorAuthorImpl.cpp(2597)\\onnxruntime::"
                "MLOperatorKernelContext::AllocateTemporaryData 8007023E"
            )
        return [np.array([[0.75]], dtype=np.float32)]


@pytest.fixture
def turn_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import onnxruntime as ort

    FakeSession.made = []
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    monkeypatch.setattr(cuda_libs, "prepare_ort_cuda", lambda: None)
    (tmp_path / "smart-turn-v3.2-cpu.onnx").write_bytes(b"")
    return tmp_path


def test_the_turn_model_runs_on_the_cpu_with_dml(
    turn_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from bolo.engines import onnx_turn

    monkeypatch.setattr(ort_providers, "available_providers", lambda: [DML, CPU])
    monkeypatch.setattr(onnx_turn, "_said", set())
    with caplog.at_level(logging.WARNING, logger="bolo.engines.onnx_turn"):
        first = onnx_turn.SmartTurn(turn_dir, {}, LoadOptions(device="dml"))
        onnx_turn.SmartTurn(turn_dir, {}, LoadOptions(device="dml:1"))
    assert FakeSession.made == [[CPU], [CPU]] and first.accelerator is None
    assert first.probability(np.zeros(16000, dtype=np.float32)) == pytest.approx(0.75)
    said = [r.getMessage() for r in caplog.records]
    assert len(said) == 1 and "DirectML" in said[0] and "node_view_1" in said[0]  # once


async def test_the_turn_model_is_placed_on_the_cpu_with_dml(settings: Settings) -> None:
    m = manager(settings, "dml")
    turn = spec("smart-turn-v3", "onnxruntime", "smart_turn", task="turn")
    assert m.device_for(turn)[0] == "cpu"
    p = await m.place(turn)
    assert p.device == "cpu" and p.gpu_bytes == 0 and p.processor() == "100% CPU"
    assert "DirectML" in (p.notice or "") and "smart-turn-v3" in (p.notice or "")
    # other ONNX Runtime models still use DirectML
    other = spec("nemo", "onnxruntime", "nemotron_diarization", task="diarization")
    assert (await m.place(other)).device == "dml"


async def test_nemotron_on_dml_says_its_gelu_nodes_run_on_the_cpu(settings: Settings) -> None:
    # DirectML runs the int8 graph's MatMulNBits (8-bit) but has no kernel for the ONNX Gelu (opset 20): the weights
    # are all on the GPU ("100% GPU" by bytes) and the notice says what runs on the CPU all the same
    nemo = spec("nemo", "onnxruntime", "nemotron_diarization", task="diarization")
    p = await manager(settings, "dml").place(nemo)
    assert p.device == "dml" and p.processor() == "100% GPU"
    assert "Gelu" in (p.notice or "") and "CPU" in (p.notice or "") and "'nemo' on dml" in (p.notice or "")
    assert "Gelu" in ((await manager(settings, "dml:1").place(nemo)).notice or "")
    # on CUDA every node has a kernel; other ONNX Runtime models on DirectML say nothing
    assert (await manager(settings, "cuda:0").place(nemo)).notice is None
    other = spec("dicow", "onnxruntime", "dicow_onnx")
    assert "Gelu" not in ((await manager(settings, "dml").place(other)).notice or "")


def test_a_runtime_failure_on_an_accelerator_moves_the_turn_model_to_the_cpu_once(
    turn_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bolo.engines import onnx_turn

    monkeypatch.setattr(ort_providers, "available_providers", lambda: [CUDA, CPU])
    monkeypatch.setattr(onnx_turn, "_said", set())
    turn = onnx_turn.SmartTurn(turn_dir, {}, LoadOptions(device="cuda:0"))
    assert turn.accelerator == CUDA
    told: list[dict[str, Any]] = []
    turn.on_fallback = told.append
    audio = np.zeros(16000, dtype=np.float32)
    assert turn.probability(audio) == pytest.approx(0.75)  # answered, from the CPU
    assert turn.probability(audio) == pytest.approx(0.75)
    assert FakeSession.made == [[CUDA, CPU], [CPU]] and turn.accelerator is None
    (fallback,) = told
    assert fallback["requested"] == CUDA and fallback["running"] == CPU and "node_view_1" in fallback["message"]

    # a failure on the CPU itself is the model's own error
    def broken(self: FakeSession, outputs: Any, feeds: Any) -> Any:
        raise RuntimeError("bad input")

    monkeypatch.setattr(FakeSession, "run", broken)
    with pytest.raises(RuntimeError, match="bad input"):
        turn.probability(audio)


def test_the_manager_moves_a_model_that_fell_back_while_serving(settings: Settings) -> None:
    m = manager(settings, "dml")
    model = SimpleNamespace(on_fallback=None)
    entry = LoadedModel(spec("t", "onnxruntime", "smart_turn", task="turn"), model, None, 8 * MB, 1.0)  # type: ignore[arg-type]
    entry.placement = Placement(device="dml", requested="dml", gpu_bytes=8 * MB)
    entry.vram_bytes = 8 * MB
    fb = cuda_libs.ProviderFallback("onnxruntime", DML, CPU, [], "8007023E").to_dict()
    m._fell_back(entry, {**fb, "message": "failed on Dml while running; it runs on the CPU from now on"})
    p = entry.placement
    assert p.device == "cpu" and entry.processor() == "100% CPU" and entry.vram_bytes == 0
    assert "while running" in (p.notice or "") and p.to_dict()["provider_fallback"][0]["requested"] == DML


async def test_a_loaded_model_gets_its_fallback_hook(core: Bolo) -> None:
    entry = await core.manager.load("fake-vad")
    assert callable(entry.model.on_fallback)
    entry.model.on_fallback({"runtime": "onnxruntime", "requested": CUDA, "message": "fell back while running"})
    assert entry.placement.device == "cpu" and "fell back while running" in (entry.placement.notice or "")


async def test_end_of_turn_survives_a_model_error(core: Bolo, monkeypatch: pytest.MonkeyPatch) -> None:
    turn = spec("broken-turn", "fake", "fake-turn", task="turn")
    monkeypatch.setattr(core, "turn_model", lambda name=None: turn)

    class Boom:
        def __init__(self, *_: Any) -> None:
            pass

        async def __aenter__(self) -> Any:
            raise RuntimeError("Reshape node 'node_view_1' 8007023E")

        async def __aexit__(self, *_: Any) -> None:
            return None

    monkeypatch.setattr(core.manager, "use", Boom)
    assert await core.end_of_turn(np.zeros(1600, dtype=np.float32)) is None


# -- 2. device = "auto" and the providers this ONNX Runtime has ------------------------------------------------------
def test_auto_device_follows_the_installed_onnx_runtime() -> None:
    assert auto_device("cuda:0", [DML, CPU]) == "dml"  # onnxruntime-directml on a laptop with an NVIDIA GPU
    assert auto_device(None, [DML, CPU]) == "dml"  # an AMD or Intel GPU only
    assert auto_device("cuda:0", [CUDA, CPU]) == "cuda"
    assert auto_device("cuda:0", [CUDA, DML, CPU]) == "cuda"
    assert auto_device(None, [CUDA, CPU]) == "cpu"  # onnxruntime-gpu, no NVIDIA GPU
    assert auto_device("cuda:0", [CPU]) == "cpu"
    assert auto_device("coreml", ["CoreMLExecutionProvider", CPU]) == "coreml"


async def test_auto_with_onnxruntime_directml_places_onnx_models_on_directml(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ort_providers, "available_providers", lambda: [DML, CPU])
    monkeypatch.setattr(ort_providers, "default_dml_index", lambda: 1)  # the RTX card, not the Radeon iGPU
    accel = FakeAccel(nvidia=True)
    m = manager(settings, "auto", accel)
    nemo = spec("nemo", "onnxruntime", "nemotron_diarization", task="diarization")
    p = await m.place(nemo)
    assert p.device == "dml" and p.requested == "auto" and accel.directml  # DXGI's figures are read from now on
    assert providers_for(p.device, [DML, CPU]) == [(DML, {"device_id": 1}), CPU]
    # engines with CUDA of their own still get it
    assert m.device_for(spec("fw", "faster-whisper", "whisper"))[0] == "cuda"
    assert m.device_for(spec("sh", "sherpa-onnx", "whisper"))[0] == "cuda"
    # the end-of-turn model: DirectML cannot run it
    turn = spec("smart-turn-v3", "onnxruntime", "smart_turn", task="turn")
    assert m.device_for(turn)[0] == "cpu"
    # onnxruntime-gpu: CUDA as before; plain onnxruntime: the CPU, not a CUDA placement that falls back
    monkeypatch.setattr(ort_providers, "available_providers", lambda: [CUDA, CPU])
    assert m.device_for(nemo)[0] == "cuda"
    monkeypatch.setattr(ort_providers, "available_providers", lambda: [CPU])
    assert (await m.place(nemo)).device == "cpu"
    assert m.device_for(spec("fw", "faster-whisper", "whisper"))[0] == "cuda"


def test_cuda_with_onnxruntime_directml_says_how_to_use_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(ort_providers, "_warned", set())
    with caplog.at_level(logging.WARNING, logger="bolo.ort_providers"), cuda_libs.watch() as watched:
        assert providers_for("cuda:0", [DML, CPU]) == [CPU]
    (message,) = [r.getMessage() for r in caplog.records]
    assert "onnxruntime-directml" in message and 'device = "dml"' in message
    assert "in place of onnxruntime" not in message
    assert "onnxruntime-directml" in watched.fallbacks[0].recipe
    with caplog.at_level(logging.WARNING, logger="bolo.ort_providers"):
        providers_for("cuda:1", [CPU])
    assert "onnxruntime-gpu" in caplog.records[-1].getMessage()  # plain onnxruntime: get onnxruntime-gpu
    (problem,) = ort_providers.ort_problems("cuda", {"onnxruntime-directml": "1.24.4"}, [DML, CPU])
    assert "onnxruntime-directml" in problem and 'device = "dml"' in problem


# -- 3. sherpa-onnx's CPU build -------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("files", "versions", "gpu"),
    [
        (["lib/onnxruntime.dll", "lib/_sherpa_onnx.cp313-win_amd64.pyd"], ["2.0.1"], False),  # PyPI, Windows
        (["lib/libonnxruntime.so", "lib/libsherpa-onnx-c-api.so"], ["2.0.1"], False),  # PyPI, Linux
        (["lib/onnxruntime.dll", "lib/onnxruntime_providers_cuda.dll"], ["1.12.0"], True),
        (["lib/libonnxruntime.so", "lib/libonnxruntime_providers_cuda.so"], ["1.12.0"], True),
        (["lib/onnxruntime.dll"], ["1.12.0+cuda"], True),
        ([], ["2.0.1"], None),  # no ONNX Runtime library to judge by
    ],
)
def test_sherpa_build_is_told_from_its_files(
    tmp_path: Path, files: list[str], versions: list[str], gpu: bool | None
) -> None:
    package = tmp_path / "sherpa_onnx"
    package.mkdir()
    for name in files:
        (package / name).parent.mkdir(parents=True, exist_ok=True)
        (package / name).write_bytes(b"")
    assert sherpa_build(package, versions) is gpu
    assert sherpa_build(None, ["1.12.0+cuda12.cudnn9"]) is True and sherpa_build(None, ["2.0.1"]) is None


async def test_a_cpu_build_of_sherpa_onnx_places_its_models_on_the_cpu(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    whisper = spec("whisper-base", "sherpa-onnx", "whisper")
    m = manager(settings, "cuda")
    monkeypatch.setattr(ort_providers, "sherpa_gpu_build", lambda: False)
    p = await m.place(whisper)
    assert p.device == "cpu" and p.gpu_bytes == 0 and p.processor() == "100% CPU"
    assert "CPU build" in (p.notice or "") and "whisper-base" in (p.notice or "")
    assert p.to_dict()["device"] == "cpu"
    # the engine asked directly: the CPU too, said once
    monkeypatch.setattr(ort_providers, "_warned", set())
    with caplog.at_level(logging.WARNING, logger="bolo.ort_providers"):
        assert sherpa_provider("cuda:0") == "cpu" and sherpa_provider("cuda") == "cpu"
    assert len(caplog.records) == 1 and "CPU build" in caplog.records[0].getMessage()
    # a GPU build, or one that cannot be told: CUDA as asked
    for build in (True, None):
        monkeypatch.setattr(ort_providers, "sherpa_gpu_build", lambda b=build: b)
        assert (await m.place(whisper)).device.startswith("cuda") and sherpa_provider("cuda") == "cuda"


def test_sherpa_saying_it_fell_back_is_a_provider_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A build that could not be told apart: sherpa-onnx's own words at load time move the model to the CPU."""
    from bolo.engines.sherpa import SherpaOnnxEngine

    def load(self: Any, kind: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Any:
        os.write(2, SHERPA_SAYS.encode())
        return SimpleNamespace(kind=kind)

    monkeypatch.setattr(SherpaOnnxEngine, "_load", load)
    with cuda_libs.watch() as watched:
        SherpaOnnxEngine().load("whisper", "stt", tmp_path, {}, LoadOptions(device="cuda:0"))
    (fallback,) = watched.fallbacks
    assert watched.whole and fallback.runtime == "sherpa-onnx" and fallback.requested == "cuda"
    assert "sherpa-onnx could not start CUDA" in fallback.message() and "runs on the CPU" in fallback.message()
    placement = Placement(device="cuda:0", requested="cuda", gpu_bytes=382 * MB, gpus=[0])
    placement.fell_back([f.to_dict() for f in watched.fallbacks], watched.notice(), watched.whole)
    assert placement.device == "cpu" and placement.processor() == "100% CPU"

    def quiet(self: Any, kind: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Any:
        return SimpleNamespace(kind=kind)

    monkeypatch.setattr(SherpaOnnxEngine, "_load", quiet)
    with cuda_libs.watch() as fine:
        SherpaOnnxEngine().load("whisper", "stt", tmp_path, {}, LoadOptions(device="cuda:0"))
    assert not fine.fallbacks and fine.accelerated == 1


def test_hardware_says_which_sherpa_build_is_installed(tmp_path: Path) -> None:
    from bolo import hardware

    lib = tmp_path / "sherpa_onnx" / "lib"
    lib.mkdir(parents=True)
    (lib / "onnxruntime.dll").write_bytes(b"")
    module = SimpleNamespace(__file__=str(tmp_path / "sherpa_onnx" / "__init__.py"), __version__="2.0.1")
    assert hardware._sherpa_gpu(module) is False
    (lib / "onnxruntime_providers_cuda.dll").write_bytes(b"")
    assert hardware._sherpa_gpu(module) is True
    assert hardware._sherpa_gpu(SimpleNamespace(__version__="2.0.1")) is None


# -- Apple silicon: engines built for its GPU use Metal when no device is set ---------------------------------------
class FakeAppleAccel(FakeAccel):
    """An Apple silicon GPU (Metal), as bolo.accel reports one on a Mac."""

    def __init__(self) -> None:
        super().__init__(nvidia=False)

    def default_device(self) -> str | None:
        return "coreml"

    def devices(self, fresh: bool = False) -> list[DeviceMemory]:
        return [DeviceMemory(0, "apple", "Apple M4 Pro", 24 << 30, 12 << 30, unified=True, backend="metal")]


def apple_manager(settings: Settings, accel: FakeAccel | None = None) -> ModelManager:
    core = Bolo(settings)
    core.manager.accel = accel or FakeAppleAccel()  # type: ignore[assignment]
    return core.manager


def duplex(build: str) -> ModelSpec:
    return ModelSpec.from_dict({
        "id": f"moshi-{build}", "task": "speech-to-speech", "engine": "duplex", "kind": "duplex", "family": "Test",
        "languages": ["en"], "license": "MIT", "builtin": True, "size_bytes": 8 * MB, "options": {"build": build},
    })  # fmt: skip


def test_metal_first_engines_use_metal_on_apple_silicon_by_default(settings: Settings) -> None:
    assert settings.device == "cpu" and settings.device_explicit is False
    m = apple_manager(settings)
    for gpu_first in (spec("w", "mlx", "whisper"), spec("c", "whisper.cpp", "whisper"),
                      spec("l", "tts-server", "tts-server", task="tts"), duplex("mlx")):  # fmt: skip
        assert m.requested_device(gpu_first) == "coreml", gpu_first.id
        assert m.device_for(gpu_first)[0] == "coreml"
    # the engines whose Mac route is the CPU (or untested CoreML) stay there
    for cpu_first in (spec("s", "sherpa-onnx", "whisper"), spec("f", "faster-whisper", "whisper"),
                      spec("o", "onnxruntime", "smart_turn", task="turn"), duplex("pytorch")):  # fmt: skip
        assert m.requested_device(cpu_first) == "cpu", cpu_first.id


def test_an_explicit_cpu_keeps_metal_first_engines_on_the_cpu(settings: Settings) -> None:
    settings.device_explicit = True  # device = "cpu" written in the config, the environment or a flag
    m = apple_manager(settings)
    assert m.requested_device(spec("w", "mlx", "whisper")) == "cpu"
    # and so does a model pinned to the CPU, with the setting left alone
    settings.device_explicit = False
    m = apple_manager(settings)
    m.store.set_model_settings("pinned", {"device": "cpu"})
    assert m.requested_device(spec("pinned", "mlx", "whisper")) == "cpu"


def test_no_metal_no_change(settings: Settings) -> None:
    m = apple_manager(settings, FakeAccel(nvidia=False))  # an Intel Mac, or Linux: no Metal device
    assert m.requested_device(spec("w", "mlx", "whisper")) == "cpu"
    m = apple_manager(settings, FakeAccel(nvidia=True))  # an NVIDIA GPU: the default stays the CPU, as before
    assert m.requested_device(spec("c", "whisper.cpp", "whisper")) == "cpu"


def test_load_settings_knows_whether_device_was_set(tmp_path: Path) -> None:
    from bolo.config import load_settings

    env = {"BOLO_DATA_DIR": str(tmp_path)}
    assert load_settings(env=env).device_explicit is False
    assert load_settings(env={**env, "BOLO_DEVICE": "cpu"}).device_explicit is True
    assert load_settings(overrides={"device": "cpu"}, env=env).device_explicit is True
    config = tmp_path / "bolo.toml"
    config.write_text('device = "cpu"\n')
    assert load_settings(config, env=env).device_explicit is True
    config.write_text("device_explicit = true\n")  # an internal flag, not a setting
    with pytest.raises(ValueError, match="unknown setting"):
        load_settings(config, env=env)


def test_moonshine_v2_runs_on_the_cpu_with_coreml(settings: Settings) -> None:
    settings.device, settings.device_explicit = "coreml", True
    m = apple_manager(settings)
    device, reason = m.device_for(spec("moonshine-v2-base-en", "sherpa-onnx", "moonshine_v2"))
    assert device == "cpu" and reason is not None and "Unable to compute the prediction" in reason
    # Moonshine v1 and the other sherpa-onnx models still use CoreML when asked to
    assert m.device_for(spec("moonshine-base-en", "sherpa-onnx", "moonshine"))[0] == "coreml"
    assert m.device_for(spec("zipformer", "sherpa-onnx", "transducer"))[0] == "coreml"


def test_a_catalog_entry_can_keep_itself_off_a_device_family(settings: Settings) -> None:
    """Soprano 1.1 on audio.cpp segfaults on Metal (an M4 Pro): its entry says so and it runs on the CPU there."""
    m = apple_manager(settings)  # device left unset: tts-server models go on Metal
    entry = ModelSpec.from_dict({
        "id": "soprano-audiocpp", "task": "tts", "engine": "tts-server", "kind": "tts-server", "family": "Test",
        "languages": ["en"], "license": "MIT", "builtin": True, "size_bytes": 8 * MB,
        "options": {"runtime": "audio.cpp", "cpu_only_on": {"metal": "audio.cpp segfaults on Metal"}},
    })  # fmt: skip
    device, reason = m.device_for(entry)
    assert device == "cpu" and reason is not None and "segfaults on Metal" in reason
    assert m.device_for(spec("l", "tts-server", "tts-server", task="tts"))[0] == "coreml"  # the others still go there


def test_the_catalog_keeps_soprano_on_audiocpp_off_metal() -> None:
    from bolo.catalog import Catalog

    entry = Catalog.load().resolve("soprano-1.1-80m-audiocpp")
    assert entry is not None and "segmentation fault" in entry.options["cpu_only_on"]["metal"]
