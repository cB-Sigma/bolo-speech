"""Device, dtype, threads, downloads and sentence chunking."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

import bolo_torch
from bolo_torch import runtime
from bolo_torch.runtime import InvalidRequest, MissingDependency, ModelGated, PluginError
from bolo_torch.text import split_sentences


def test_entry_point_is_light_and_returns_the_command() -> None:
    assert bolo_torch.plugin() == [sys.executable, "-m", "bolo_torch"]


def test_device_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOLO_TORCH_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert runtime.pick_device("cpu") == "cpu"
    assert runtime.pick_device("auto") in ("cpu", "mps")
    assert runtime.pick_device("cpu", {"device": "auto"}) in ("cpu", "mps")  # the model option wins over bolo's
    with pytest.raises(InvalidRequest, match="no CUDA device"):
        runtime.pick_device("cuda")
    with pytest.raises(InvalidRequest, match="unknown device"):
        runtime.pick_device("tpu")
    monkeypatch.setenv("BOLO_TORCH_DEVICE", "cpu")
    assert runtime.pick_device("cuda") == "cpu"  # the environment wins over everything
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("BOLO_TORCH_DEVICE", "auto")
    assert runtime.pick_device("cpu") == "cuda"


def test_bolo_device_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """bolo resolves ``auto`` to ``coreml`` on Apple silicon and ``rocm`` next to an AMD GPU: MPS and ROCm torch."""
    monkeypatch.delenv("BOLO_TORCH_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert runtime.pick_device("coreml") == runtime.pick_device("metal") == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert runtime.pick_device("coreml") == "cpu"  # given by bolo: the CPU, with a warning
    with pytest.raises(InvalidRequest, match="Metal"):
        runtime.pick_device("cpu", {"device": "coreml"})  # asked for by name: an error
    monkeypatch.setattr(torch.version, "hip", None)
    assert runtime.pick_device("rocm") == "cpu"  # a CUDA or CPU build of torch next to an AMD GPU
    with pytest.raises(InvalidRequest, match="ROCm build"):
        runtime.pick_device("cpu", {"device": "rocm"})
    monkeypatch.setattr(torch.version, "hip", "6.4.43482")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert runtime.pick_device("rocm") == "cuda" and runtime.pick_device("hip:1") == "cuda:1"


def test_dtype_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOLO_TORCH_DTYPE", raising=False)
    assert runtime.pick_dtype("cpu") == torch.float32
    assert runtime.pick_dtype("mps") == torch.float16
    assert runtime.pick_dtype("cpu", {"dtype": "bfloat16"}) == torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert runtime.pick_dtype("cuda") == torch.float16
    with pytest.raises(InvalidRequest):
        runtime.pick_dtype("cpu", {"dtype": "int3"})
    monkeypatch.setenv("BOLO_TORCH_DTYPE", "float32")
    assert runtime.pick_dtype("cuda") == torch.float32


def test_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    before = torch.get_num_threads()
    try:
        monkeypatch.delenv("BOLO_TORCH_THREADS", raising=False)
        assert runtime.set_threads(1) == 1
        monkeypatch.setenv("BOLO_TORCH_THREADS", "2")
        assert runtime.set_threads(1) == 2
    finally:
        torch.set_num_threads(before)


def test_gated_and_failed_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import huggingface_hub
    from huggingface_hub import utils

    def gated(*args: Any, **kwargs: Any) -> str:
        raise utils.GatedRepoError("Access to model x is restricted", response=_Response(403))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", gated)
    monkeypatch.setenv("HF_TOKEN", "hf_example")
    with pytest.raises(ModelGated) as info:
        runtime.hub_snapshot("ai4bharat/indic-parler-tts")
    assert info.value.code == "model_gated"
    assert "accept its conditions" in str(info.value) and "a token is set" in str(info.value)

    def other(*args: Any, **kwargs: Any) -> str:
        raise utils.HfHubHTTPError("server error", response=_Response(500))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", other)
    with pytest.raises(PluginError) as failed:
        runtime.hub_snapshot("a/b")
    assert failed.value.code == "download_failed"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path))
    assert runtime.model_dir(tmp_path / "missing", {"repo": "a/b"}) == tmp_path
    (tmp_path / "f").write_text("x")
    assert runtime.model_dir(tmp_path, {}) == tmp_path
    with pytest.raises(InvalidRequest):
        runtime.model_dir(tmp_path / "missing", {})


def test_missing_extra_names_the_install_command() -> None:
    with pytest.raises(MissingDependency, match=r'pip install "bolo-torch\[pyannote\]"') as info:
        runtime.require("not_a_real_module_xyz", "pyannote")
    assert info.value.code == "component_missing"


class _Response:
    def __init__(self, status: int) -> None:
        self.status_code = status
        self.headers: dict[str, str] = {}
        self.request = None
        self.url = "https://huggingface.co"

    def json(self) -> dict[str, Any]:
        return {}


def test_split_sentences() -> None:
    assert split_sentences("") == []
    assert split_sentences("One. Two!  Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("पहला वाक्य। दूसरा वाक्य॥ तीसरा") == ["पहला वाक्य।", "दूसरा वाक्य॥", "तीसरा"]
    assert split_sentences('He said "hi." Then left.') == ['He said "hi."', "Then left."]
    long = ", ".join(["word " * 8] * 10)
    parts = split_sentences(long, max_chars=60)
    assert all(len(p) <= 60 for p in parts) and " ".join(parts).split() == long.split()
    assert split_sentences("x" * 30 + " " + "y" * 30, max_chars=40) == ["x" * 30, "y" * 30]
