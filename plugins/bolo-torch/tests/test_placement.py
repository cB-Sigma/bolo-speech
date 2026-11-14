"""GPU/CPU splits of PyTorch models: the arguments for accelerate and the report sent back to bolo (fakes only;
no GPU in CI)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bolo_torch import runtime
from bolo_torch.runtime import LoadContext


def ctx(device: str, max_memory: dict[str, int] | None) -> LoadContext:
    return LoadContext("orpheus-snac", "tts", Path(), {}, device=device, max_memory=max_memory)


def test_split_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == "accelerate" else real(name))
    assert runtime.split_kwargs(ctx("cuda", None)) == {}  # nothing asked: whole model, then .to(device)
    assert runtime.split_kwargs(ctx("cpu", {"0": 1, "cpu": 2})) == {}
    assert runtime.split_kwargs(ctx("cuda", {"0": 4_000_000_000, "cpu": 8_000_000_000})) == {
        "device_map": "auto",
        "max_memory": {0: 4_000_000_000, "cpu": 8_000_000_000},  # accelerate wants GPU indices as integers
    }
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)
    assert runtime.split_kwargs(ctx("cuda", {"0": 1, "cpu": 2})) == {}  # without accelerate: whole on the GPU


def param(n: int, size: int, device: str) -> Any:
    return SimpleNamespace(numel=lambda: n, element_size=lambda: size, device=SimpleNamespace(type=device))


def test_device_map_report() -> None:
    model = SimpleNamespace(
        parameters=lambda: [param(1000, 2, "cuda"), param(500, 2, "cuda"), param(250, 4, "cpu")],
        hf_device_map={"model.layers.0": 0, "model.layers.1": "cpu"},
    )
    assert runtime.device_map_report(model) == {
        "gpu_bytes": 3000,
        "cpu_bytes": 1000,
        "device_map": {"model.layers.0": "0", "model.layers.1": "cpu"},
    }
    assert runtime.device_map_report(object()) is None
