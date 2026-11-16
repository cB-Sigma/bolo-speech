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
