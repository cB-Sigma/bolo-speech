"""The ``device`` setting reaches the engines that run ONNX graphs themselves."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from bolo import ort_providers
from bolo.ort_providers import CPU, providers_for, sherpa_provider

CUDA = "CUDAExecutionProvider"
COREML = "CoreMLExecutionProvider"


def test_cpu_is_the_default() -> None:
    assert providers_for(None, [CUDA, CPU]) == [CPU]
    assert providers_for("cpu", [CUDA, CPU]) == [CPU]
    assert providers_for("", [CUDA, CPU]) == [CPU]


def test_an_accelerator_comes_first_and_the_cpu_after_it() -> None:
    assert providers_for("cuda", [CUDA, CPU]) == [CUDA, CPU]
    assert providers_for("CUDA:1", [CUDA, CPU]) == [(CUDA, {"device_id": 1}), CPU]
    for name in ("coreml", "mps", "metal"):
        assert providers_for(name, [COREML, CPU]) == [COREML, CPU]


def test_auto_takes_the_first_accelerator_the_build_has() -> None:
    assert providers_for("auto", [COREML, CPU]) == [COREML, CPU]
    assert providers_for("auto", [COREML, CUDA, CPU]) == [CUDA, CPU]
    assert providers_for("auto", [CPU]) == [CPU]


def test_a_device_the_build_cannot_serve_falls_back_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    ort_providers._warned.clear()
    with caplog.at_level(logging.WARNING, logger="bolo.ort_providers"):
        assert providers_for("cuda", [CPU]) == [CPU]
        assert providers_for("cuda", [CPU]) == [CPU]
        assert providers_for("quantum", [CPU]) == [CPU]
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert "onnxruntime-gpu" in messages[0]


def test_sherpa_provider_names() -> None:
    assert sherpa_provider("cuda:0") == "cuda"
    assert sherpa_provider("mps") == "coreml"
    assert sherpa_provider("cpu") == "cpu"
    assert sherpa_provider("auto") == "cpu"


def test_engines_hand_the_device_to_onnx_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each engine that builds its own session asks for the configured device, not the CPU."""
    import onnxruntime as ort

    seen: list[Any] = []

    class Built(Exception):
        pass

    class FakeSession:
        def __init__(self, path: str, *args: Any, providers: Any = None, **kwargs: Any) -> None:
            seen.append(providers)
            raise Built  # stop here: the rest of each constructor needs a real graph

    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    monkeypatch.setattr(ort_providers, "available_providers", lambda: [CUDA, CPU])
    model = tmp_path / "m.onnx"
    model.write_bytes(b"")

    from bolo.engines import dicow_onnx, nemotron, orpheus_llamacpp

    with pytest.raises(Built):
        nemotron._OnnxStep(str(model), 1, "cuda")
    with pytest.raises(Built):
        orpheus_llamacpp.open_session(model, 1, "cuda")
    with pytest.raises(Built):
        dicow_onnx.OnnxNetwork(model, model, 1, "cuda")
    assert seen == [[CUDA, CPU]] * 3


def test_doctor_warns_about_onnxruntime_packages() -> None:
    from bolo.ort_providers import installed_ort_packages, ort_problems, sherpa_provider

    assert installed_ort_packages({"onnxruntime": "1.32.0", "onnxruntime-gpu": None}) == {"onnxruntime": "1.32.0"}
    both = {"onnxruntime": "1.32.0", "onnxruntime-directml": "1.24.4"}
    (problem,) = ort_problems("cpu", both, ["CPUExecutionProvider"])
    assert "several ONNX Runtime packages" in problem and "onnxruntime-directml 1.24.4" in problem
    problems = ort_problems("dml:1", {"onnxruntime": "1.32.0"}, ["CPUExecutionProvider"])
    assert len(problems) == 1 and "pip install onnxruntime-directml" in problems[0]
    assert (
        ort_problems("dml", {"onnxruntime-directml": "1.24.4"}, ["DmlExecutionProvider", "CPUExecutionProvider"]) == []
    )
    assert "onnxruntime-gpu" in ort_problems("cuda", {"onnxruntime": "1.32.0"}, ["CPUExecutionProvider"])[0]
    assert ort_problems("cpu", {}, None) == []
    assert sherpa_provider("dml:0") == "cpu" and sherpa_provider("cuda:1") == "cuda"


def test_dml_without_a_number_runs_on_the_high_performance_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """DirectML's own default is DXGI's adapter 0, the integrated GPU of a hybrid laptop: ``dml`` asks for
    Windows' high-performance adapter instead, and ``dml:N`` pins one."""
    dml = "DmlExecutionProvider"
    monkeypatch.setattr(ort_providers, "default_dml_index", lambda: 1)
    assert providers_for("dml", [dml, CPU]) == [(dml, {"device_id": 1}), CPU]
    assert providers_for("directml", [dml, CPU]) == [(dml, {"device_id": 1}), CPU]
    assert providers_for("auto", [dml, CPU]) == [(dml, {"device_id": 1}), CPU]
    assert providers_for("dml:0", [dml, CPU]) == [(dml, {"device_id": 0}), CPU]
    monkeypatch.setattr(ort_providers, "default_dml_index", lambda: None)  # not Windows, or no adapter
    assert providers_for("dml", [dml, CPU]) == [dml, CPU]


def test_every_engine_session_asks_for_exact_int8_products(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """On x86 CPUs with AVX2 but without VNNI, ONNX Runtime's uint8 x int8 kernel saturates (the end-of-turn model's
    probabilities moved by up to 0.57, DiCoW lost all its text): every session bolo opens itself carries
    ``session.x64quantprecision``, whichever engine builds its options."""
    ort = pytest.importorskip("onnxruntime")

    seen: list[str] = []

    class Built(Exception):
        pass

    class FakeSession:
        def __init__(self, path: str, sess_options: Any = None, providers: Any = None, **kwargs: Any) -> None:
            seen.append(sess_options.get_session_config_entry(ort_providers.X64_QUANT_PRECISION))
            raise Built  # stop here: the rest of each constructor needs a real graph

    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    model = tmp_path / "m.onnx"
    model.write_bytes(b"")
    onnx_dir = tmp_path / "lfm" / "onnx"
    onnx_dir.mkdir(parents=True)
    for name in ("decoder_q4.onnx", "audio_encoder.onnx", "codec_decoder.onnx", "depth_decoder.onnx"):
        (onnx_dir / name).write_bytes(b"")

    from bolo import s2s_onnx
    from bolo.engines import dicow_onnx, nemotron, onnx_turn, orpheus_llamacpp
    from bolo.engines.base import LoadOptions

    builds = [
        lambda: nemotron._OnnxStep(str(model), 1),
        lambda: orpheus_llamacpp.open_session(model, 1),
        lambda: dicow_onnx.OnnxNetwork(model, model, 1),
        lambda: onnx_turn.SmartTurn(tmp_path, {}, LoadOptions()),
        lambda: s2s_onnx.OnnxGraphs(tmp_path / "lfm"),
        lambda: ort_providers.create_session(str(model)),  # no options given
        lambda: ort_providers.create_session(str(model), ort.SessionOptions()),  # options made elsewhere
    ]
    for build in builds:
        with pytest.raises(Built):
            build()
    assert seen == ["1"] * len(builds)


DML = "DmlExecutionProvider"


def test_a_build_without_cuda_is_named_by_its_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """`bolo doctor`'s CUDA section names the package that is installed (onnxruntime-directml here), not "the CPU
    build" and `pip uninstall -y onnxruntime` whatever it is."""
    from bolo import cuda_libs
    from bolo.ort_providers import no_cuda_provider

    dml = {"onnxruntime-directml": "1.24.4"}
    text = no_cuda_provider(dml)
    assert (
        "onnxruntime-directml 1.24.4 (the DirectML build)" in text and "pip uninstall -y onnxruntime-directml" in text
    )
    assert "CPU build" not in text and 'device = "dml"' in text
    text = no_cuda_provider({"onnxruntime": "1.32.0"})
    assert "onnxruntime 1.32.0 (the CPU build)" in text and "pip uninstall -y onnxruntime," in text
    text = no_cuda_provider({"onnxruntime": "1.32.0", "onnxruntime-gpu": "1.32.0"})
    assert "pip uninstall -y onnxruntime onnxruntime-gpu," in text and "overwrite each other" in text
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(ort, "get_available_providers", lambda: [DML, CPU])
    ok, text = cuda_libs.check_ort_cuda(dml)
    assert not ok and "onnxruntime-directml" in text and "CPU build" not in text
    lines = cuda_libs.doctor_lines(ort_check=lambda: cuda_libs.check_ort_cuda(dml), have_ort=True, have_ct2=False)
    assert lines[1].startswith("  -- onnxruntime") and "pip uninstall -y onnxruntime-directml" in lines[1]


@pytest.mark.parametrize(
    ("device", "nvidia", "packages", "providers", "warned"),
    [
        ("cuda", True, {"onnxruntime": "1.32.0"}, [CPU], True),  # uv tool upgrade put plain onnxruntime back
        ("auto", True, {"onnxruntime": "1.32.0"}, [CPU], True),
        ("cuda:1", False, {"onnxruntime": "1.32.0"}, [CPU], True),  # asked for CUDA: no GPU check needed
        ("auto", False, {"onnxruntime": "1.32.0"}, [CPU], False),  # no NVIDIA GPU: nothing lost
        ("auto", True, {"onnxruntime-directml": "1.24.4"}, [DML, CPU], False),  # auto runs on DirectML
        ("cuda", True, {"onnxruntime-directml": "1.24.4"}, [DML, CPU], True),
        ("cuda", True, {"onnxruntime-gpu": "1.32.0"}, [CUDA, CPU], False),
        ("cpu", True, {"onnxruntime": "1.32.0"}, [CPU], False),
        ("cuda", True, {}, None, False),  # no ONNX Runtime at all
    ],
)
def test_cuda_dropped_warning(
    device: str, nvidia: bool, packages: dict[str, str], providers: list[str] | None, warned: bool
) -> None:
    message = ort_providers.cuda_dropped(device, nvidia, packages, providers)
    assert bool(message) is warned
    if message:
        assert "\n" not in message and "NVIDIA GPU" in message and "pip install" in message
        assert next(iter(packages)) in message and "uv tool upgrade" in message


def test_doctor_and_server_warn_once_when_cuda_was_dropped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, caplog: Any
) -> None:
    import asyncio
    import shutil

    from bolo import cli
    from bolo.config import Settings
    from bolo.core import Bolo

    monkeypatch.setattr(ort_providers, "installed_ort_packages", lambda: {"onnxruntime": "1.32.0"})
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: "/x/nvidia-smi" if name == "nvidia-smi" else None)
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(ort, "get_available_providers", lambda: [CPU])
    cli._doctor_onnxruntime(Settings(device="auto"))
    out = capsys.readouterr().out
    assert out.count("WARNING:") == 1 and "onnxruntime 1.32.0 (the CPU build)" in out
    cli._doctor_onnxruntime(Settings(device="cuda"))  # one line, not the same thing twice
    out = capsys.readouterr().out
    assert out.count("no CUDA provider") == 1
    core = Bolo(Settings(data_dir=tmp_path, device="auto"))
    assert core.ort_cuda_warning(nvidia_gpu=False, providers=[CPU]) is None

    async def start() -> None:
        await core.start()
        for _ in range(200):
            if any("no CUDA provider" in r.getMessage() for r in caplog.records):
                break
            await asyncio.sleep(0.01)
        await core.stop()

    with caplog.at_level(logging.WARNING, logger="bolo.core"):
        asyncio.run(start())
    assert sum("no CUDA provider" in r.getMessage() for r in caplog.records) == 1


def test_exact_int8_is_set_once() -> None:
    """session_options() sets the option and create_session sets it on the options it is given: the second set made
    ONNX Runtime print "Config with key [session.x64quantprecision] already exists" for every session."""
    pytest.importorskip("onnxruntime")
    from bolo.ort_providers import X64_QUANT_PRECISION, exact_int8, session_options

    calls: list[str] = []
    opts = session_options()
    real_add = opts.add_session_config_entry

    class Watch:
        def get_session_config_entry(self, key: str) -> str:
            return str(opts.get_session_config_entry(key))

        def add_session_config_entry(self, key: str, value: str) -> None:
            calls.append(key)
            real_add(key, value)

    exact_int8(Watch())
    assert calls == [] and opts.get_session_config_entry(X64_QUANT_PRECISION) == "1"
