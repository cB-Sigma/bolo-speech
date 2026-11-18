"""GPU/CPU splits of PyTorch models: the arguments for accelerate and the report sent back to bolo (fakes only;
no GPU in CI)."""

from __future__ import annotations

import contextlib
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


def _fake_models() -> Any:
    """A tiny transformers model whose class, like Step-Audio 2's remote code, declares no ``_no_split_modules``:
    an embedding, a stack of residual blocks (which break when split inside) and a head; and one with no stack."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from torch import nn

    class StackConfig(transformers.PretrainedConfig):  # type: ignore[misc]
        model_type = "bolo-test-stack"

        def __init__(self, hidden: int = 64, layers: int = 6, stacked: bool = True, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.hidden, self.layers, self.stacked = hidden, layers, stacked

    class Block(nn.Module):  # type: ignore[misc]
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.up = nn.Linear(hidden, hidden)
            self.down = nn.Linear(hidden, hidden)

        def forward(self, x: Any) -> Any:
            return x + self.down(torch.relu(self.up(x)))

    class StackModel(transformers.PreTrainedModel):  # type: ignore[misc]
        config_class = StackConfig
        base_model_prefix = "stack"

        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.embed = nn.Linear(config.hidden, config.hidden)
            if config.stacked:
                self.layers = nn.ModuleList([Block(config.hidden) for _ in range(config.layers)])
            else:
                self.body = nn.Linear(config.hidden, config.hidden * config.layers)
            self.head = nn.Linear(config.hidden, 4)
            self.post_init()

        def _init_weights(self, module: Any) -> None:
            pass

        def forward(self, x: Any) -> Any:
            x = self.embed(x)
            for layer in getattr(self, "layers", []):
                x = layer(x)
            return self.head(x)

    with contextlib.suppress(ValueError):  # registered by an earlier test
        transformers.AutoConfig.register("bolo-test-stack", StackConfig)
        transformers.AutoModel.register(StackConfig, StackModel)
    return SimpleNamespace(torch=torch, transformers=transformers, Config=StackConfig, Model=StackModel)


def test_a_model_without_no_split_modules_is_split_by_its_repeated_layers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Step-Audio 2 mini on a 6 GB GPU: bolo asks for a GPU/CPU split, and transformers refuses device_map="auto"
    for a class without ``_no_split_modules``. The plugin finds the repeated layers itself and hands transformers
    an explicit device map that keeps each one whole."""
    fake = _fake_models()
    torch, transformers = fake.torch, fake.transformers
    reference = fake.Model(fake.Config())
    for p in reference.parameters():
        torch.nn.init.normal_(p, std=0.1)
    reference.save_pretrained(tmp_path)
    block = 2 * (64 * 64 + 64) * 4  # bytes of one layer in float32
    load = ctx("cuda", {"0": 3 * block, "cpu": 1 << 30})
    load.dtype = torch.float32
    load.model_id = "fake-stack"
    if int(transformers.__version__.split(".")[0]) < 5:  # what bolo-torch 0.1.6 ran into (Step-Audio: 4.49)
        with pytest.raises(ValueError, match="_no_split_modules"):
            transformers.AutoModel.from_pretrained(str(tmp_path), **runtime.split_kwargs(load))
    caplog.set_level("INFO", logger="bolo_torch")
    kwargs = runtime.split_kwargs(load, transformers.AutoModel, tmp_path)
    device_map = kwargs["device_map"]
    assert set(kwargs) == {"device_map"} and isinstance(device_map, dict)
    assert not any(name.startswith("layers.") and name.count(".") > 1 for name in device_map)  # no layer cut up
    assert {0, "cpu"} <= set(device_map.values())  # some layers on the GPU, the rest on the CPU
    assert "keeping each Block on one device" in caplog.text
    # the explicit map passes transformers' check; on a machine without a GPU, load "the GPU's" share on the CPU
    on_cpu = {name: "cpu" if where == 0 else where for name, where in device_map.items()}
    model = transformers.AutoModel.from_pretrained(str(tmp_path), device_map=on_cpu)
    x = torch.randn(2, 64)
    with torch.no_grad():
        assert torch.allclose(model(x), reference(x), atol=1e-5)
    assert runtime.repeated_blocks(model) == ["Block"]


def test_a_model_that_cannot_be_split_says_what_it_needs_before_loading(tmp_path: Path) -> None:
    fake = _fake_models()
    fake.Model(fake.Config(stacked=False)).save_pretrained(tmp_path)
    load = ctx("cuda", {"0": 10_000, "cpu": 1 << 30})
    load.dtype = fake.torch.float16
    load.model_id = "fake-monolith"
    with pytest.raises(runtime.PluginError, match=r"needs about 0\.0 GB on one device") as info:
        runtime.split_kwargs(load, fake.transformers.AutoModel, tmp_path)
    assert info.value.code == "out_of_memory" and "fake-monolith cannot be split" in str(info.value)
    # a class that names its layers is left to transformers' own device_map="auto"
    declared = SimpleNamespace(_no_split_modules=["Block"])
    assert runtime.split_kwargs(load, declared, tmp_path)["device_map"] == "auto"
