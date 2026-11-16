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
