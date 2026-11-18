"""A failed load gives its GPU memory back before the plugin answers (bolo bug 17)."""

from __future__ import annotations

import gc
import io
import weakref
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from bolo.engines.base import Model

from bolo_torch import runtime
from bolo_torch.engine import TorchEngine
from bolo_torch.runtime import LoadContext
from bolo_torch.server import TorchServer


class _Weights:
    """Stands in for a half-loaded model: a reference cycle (as modules and accelerate's hooks make), which only the
    garbage collector frees."""

    def __init__(self) -> None:
        self.me = self


class FakeCuda:
    """torch.cuda on a machine with a GPU, recording what the plugin asks of it."""

    def __init__(self, alive: Any) -> None:
        self.alive = alive
        self.calls: list[str] = []
        self.weights_alive_at_empty_cache: bool | None = None

    def is_initialized(self) -> bool:
        return True

    def synchronize(self) -> None:
        self.calls.append("synchronize")

    def empty_cache(self) -> None:
        self.calls.append("empty_cache")
        self.weights_alive_at_empty_cache = self.alive() is not None

    def ipc_collect(self) -> None:
        self.calls.append("ipc_collect")

    def device_count(self) -> int:
        return 1

    def memory_allocated(self, i: int) -> int:
        return 0

    def memory_reserved(self, i: int) -> int:
        return 2 * 1024**2


@pytest.fixture
def no_automatic_gc() -> Iterator[None]:
    """The collector runs only when asked: without the plugin's own gc.collect() the weights would stay."""
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def test_a_failed_load_frees_its_weights_and_the_cuda_cache(
    monkeypatch: pytest.MonkeyPatch, no_automatic_gc: None
) -> None:
    refs: list[Any] = []

    def failing_loader(ctx: LoadContext) -> Model:
        weights = _Weights()
        refs.append(weakref.ref(weights))
        raise RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")

    fake = FakeCuda(lambda: refs[0]() if refs else None)
    monkeypatch.setattr(torch, "cuda", fake)
    sent: list[dict[str, Any]] = []
    server = TorchServer(TorchEngine({"big": failing_loader}), io.BytesIO(), io.BytesIO())
    monkeypatch.setattr(server, "send", lambda message, payload=None: sent.append(message))
    message = {"kind": "big", "task": "tts", "path": str(Path("/nonexistent")), "threads": 1, "device": "cpu"}
    server.handle("7", "load", message, b"")

    assert len(sent) == 1 and sent[0]["id"] == "7" and sent[0]["ok"] is False
    error = sent[0]["error"]
    assert error["code"] == "RuntimeError" and "out of memory" in error["message"]
    # the weights were collected before torch was asked to give the cached blocks back, and what is left is reported
    assert fake.weights_alive_at_empty_cache is False and refs[0]() is None
    assert fake.calls == ["synchronize", "empty_cache", "ipc_collect"]
    assert error["gpu"] == {"initialized": True, "allocated_bytes": 0, "reserved_bytes": 2 * 1024**2}


def test_a_load_that_never_started_cuda_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU torch, or CUDA never initialised: nothing on the GPU, so bolo has no reason to restart the plugin."""
    assert runtime.release_memory() == {"initialized": False}  # CI's CPU build of torch

    sent: list[dict[str, Any]] = []

    def gated(ctx: LoadContext) -> Model:
        raise runtime.ModelGated("gated")

    server = TorchServer(TorchEngine({"gated": gated}), io.BytesIO(), io.BytesIO())
    monkeypatch.setattr(server, "send", lambda message, payload=None: sent.append(message))
    server.handle("1", "load", {"kind": "gated", "task": "tts", "path": "/x", "threads": 1}, b"")
    assert sent[0]["error"] == {"code": "model_gated", "message": "gated", "gpu": {"initialized": False}}
    server.handle("2", "unknown", {}, b"")  # other ops still fail as before, without the clean-up
    assert sent[1]["error"]["code"] == "unknown_op" and "gpu" not in sent[1]["error"]
