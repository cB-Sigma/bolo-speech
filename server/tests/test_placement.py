"""GPU/CPU placement, accelerator memory, keep-alive and `bolo ps`, all with fake NVML and made-up memory figures.

Nothing here touches a real GPU (CI has none); docs/architecture.md lists what was not tried on hardware.
"""

from __future__ import annotations

import asyncio
import os
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from bolo import cli
from bolo.accel import Accelerators
from bolo.catalog import ModelSpec
from bolo.config import Settings, load_settings
from bolo.core import Bolo
from bolo.engines.base import LoadOptions
from bolo.engines.tts_server import gpu_layer_args
from bolo.keepalive import REQUEST_KEEP_ALIVE, RequestKeepAlive, format_keep_alive, humanize_until, parse_keep_alive
from bolo.placement import (
    LayerInfo,
    assign_graphs,
    gguf_block_count,
    layer_split,
    offload_layers,
    operator_layers,
    plan,
    processor,
    split_mode,
)
from conftest import LiveServer, wav_file

MB = 1024 * 1024
GB = 1024 * MB


# -- keep-alive ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        (None, None), ("", None), (300, 300.0), ("300", 300.0), (2.5, 2.5), ("5m", 300.0), ("1h30m", 5400.0),
        ("300ms", 0.3), ("1.5m", 90.0), ("0", 0.0), (0, 0.0), (-1, -1.0), ("-1", -1.0), ("-5m", -1.0),
        ("forever", -1.0), ("24h", 86400.0),
    ],
)  # fmt: skip
def test_parse_keep_alive(raw: Any, seconds: float | None) -> None:
    assert parse_keep_alive(raw) == seconds


@pytest.mark.parametrize("raw", ["5 minutes", "m5", "5x", "abc", "1h-3m", True])
def test_parse_keep_alive_rejects(raw: Any) -> None:
    with pytest.raises(ValueError):
        parse_keep_alive(raw)


def test_keep_alive_formats() -> None:
    assert [format_keep_alive(x) for x in (None, -1.0, 0.0, 0.3, 90.0, 300.0, 5400.0)] == [
        "default", "forever", "0s", "300ms", "1m30s", "5m", "1h30m",
    ]  # fmt: skip
    assert humanize_until(None) == "Forever"
    assert humanize_until(12.0, stopping=True) == "Stopping..."
    assert humanize_until(0.4) == "Less than a second from now"
    assert humanize_until(1.0) == "1 second from now"
    assert humanize_until(42.0) == "42 seconds from now"
    assert humanize_until(75.0) == "About a minute from now"
    assert humanize_until(4 * 60 + 59) == "4 minutes from now"
    assert humanize_until(3700.0) == "About an hour from now"
    assert humanize_until(5 * 3600.0) == "5 hours from now"
    assert humanize_until(2 * 86400.0) == "2 days from now"


def test_keep_alive_setting_and_task_ttls(tmp_path: Path) -> None:
    env = {"BOLO_DATA_DIR": str(tmp_path), "BOLO_KEEP_ALIVE": "10m"}
    s = load_settings(env=env)
    assert s.keep_alive == "10m" and s.stt_ttl_s is None and s.vad_ttl_s == -1.0
    with pytest.raises(ValueError, match="keep_alive"):
        load_settings(env={**env, "BOLO_KEEP_ALIVE": "soon"})
    # Ollama's own variable is not read: bolo often runs next to an Ollama whose setting is meant for Ollama
    assert load_settings(env={"BOLO_DATA_DIR": str(tmp_path), "OLLAMA_KEEP_ALIVE": "-1"}).keep_alive == "5m"


# -- the pure placement arithmetic ------------------------------------------------------------------------------
def test_processor_strings() -> None:
    assert processor(0, 100) == "100% CPU"
    assert processor(100, 100) == "100% GPU"
    assert processor(65, 100) == "35%/65% CPU/GPU"
    assert processor(1, 1000) == "99%/1% CPU/GPU"  # never rounds a split to 100%
    assert processor(999, 1000) == "1%/99% CPU/GPU"
    assert processor(5, 0) == "100% CPU"


def write_gguf(path: Path, arch: str = "llama", blocks: int = 28, padding: int = 0) -> Path:
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    kvs = [
        s("general.architecture") + struct.pack("<I", 8) + s(arch),
        s("general.name") + struct.pack("<I", 8) + s("test model"),
        # an array of strings and one of int32 before the block count, as tokenizer metadata comes in real files
        s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) + s("a") + s("b"),
        s("tokenizer.ggml.token_type") + struct.pack("<IIQ", 9, 5, 3) + struct.pack("<iii", 1, 1, 1),
        s(f"{arch}.context_length") + struct.pack("<I", 4) + struct.pack("<I", 4096),
        s(f"{arch}.block_count") + struct.pack("<I", 4) + struct.pack("<I", blocks),
    ]
    header = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    path.write_bytes(header + b"".join(kvs) + b"\0" * padding)
    return path


def test_gguf_block_count(tmp_path: Path) -> None:
    assert gguf_block_count(write_gguf(tmp_path / "a.gguf", "qwen3", 36)) == 36
    assert gguf_block_count(write_gguf(tmp_path / "b.gguf", "llama", 16)) == 16
    (tmp_path / "c.gguf").write_bytes(b"not a gguf")
    assert gguf_block_count(tmp_path / "c.gguf") is None
    (tmp_path / "d.gguf").write_bytes(b"GGUF\x03\x00\x00\x00")  # truncated
    assert gguf_block_count(tmp_path / "d.gguf") is None
    assert gguf_block_count(tmp_path / "missing.gguf") is None


def test_operator_layers() -> None:
    assert operator_layers([]) is None
    assert operator_layers(["-c", "4096"]) is None
    assert operator_layers(["-ngl", "20"]) == 20
    assert operator_layers(["--n-gpu-layers=12"]) == 12
    assert operator_layers(["--gpu-layers", "auto"]) == 999
    # bolo leaves an operator's -ngl alone, passes its own count otherwise, and keeps the old default
    assert gpu_layer_args("cuda", 10, ["-ngl", "5"]) == []
    assert gpu_layer_args("cuda", 10, []) == ["-ngl", "10"]
    assert gpu_layer_args("cpu", 0, []) == ["-ngl", "0"]
    assert gpu_layer_args("cuda", None, []) == ["-ngl", "99"]
    assert gpu_layer_args("cpu", None, []) == []


def test_offload_layer_arithmetic() -> None:
    # 28 blocks + the output layer = 29 layers, 2.9 GB: 100 MB per layer; 10% (290 MB) for the KV cache and buffers
    layers = LayerInfo(block_count=28, file_bytes=2900 * MB)
    assert layers.total_layers == 29 and layers.per_layer == 100 * MB
    assert offload_layers(layers, 8 * GB) == 29  # all of it
    assert offload_layers(layers, 290 * MB + 1550 * MB) == 15  # 1550 MB / 100 MB
    assert offload_layers(layers, 290 * MB + 99 * MB) == 0
    assert offload_layers(layers, 100 * MB) == 0
    with_projector = LayerInfo(block_count=28, file_bytes=2900 * MB, extra_bytes=600 * MB)
    assert offload_layers(with_projector, 290 * MB + 600 * MB + 1000 * MB) == 10  # the projector goes first
    assert layer_split(layers, 0) == (0, 2900 * MB)
    assert layer_split(layers, 29) == (2900 * MB, 0)
    assert layer_split(layers, 99) == (2900 * MB, 0)
    gpu, cpu = layer_split(layers, 10)
    assert (gpu, cpu) == (1000 * MB, 1900 * MB)
    gpu, cpu = layer_split(with_projector, 10)
    assert (gpu, cpu) == (1600 * MB, 1900 * MB)
    assert processor(*layer_split(layers, 10)[:1], 2900 * MB) == "66%/34% CPU/GPU"


def test_assign_graphs() -> None:
    graphs = {"encoder": 600 * MB, "decoder": 200 * MB, "codec": 50 * MB}
    assert assign_graphs(graphs, 2 * GB) == {"encoder": True, "decoder": True, "codec": True}
    # 1.2x headroom: the encoder (720) fits in 800, the decoder (240) no longer does, the codec (60) still does
    assert assign_graphs(graphs, 800 * MB) == {"encoder": True, "decoder": False, "codec": True}
    assert assign_graphs(graphs, 300 * MB) == {"encoder": False, "decoder": True, "codec": True}
    assert assign_graphs(graphs, 0) == {"encoder": False, "decoder": False, "codec": False}


def test_split_modes() -> None:
    assert split_mode("tts-server", "tts-server", {"runtime": "llama-server"}) == "layers"
    assert split_mode("tts-server", "orpheus-llamacpp", {}) == "layers"
    assert split_mode("tts-server", "tts-server", {"runtime": "audiocpp"}) == "whole"
    assert split_mode("onnxruntime", "dicow_onnx", {}) == "graphs"
    assert split_mode("onnxruntime", "lfm3_audio_onnx", {}) == "graphs"
    assert split_mode("onnxruntime", "nemotron_diarization", {}) == "whole"
    assert split_mode("sherpa-onnx", "whisper", {}) == "whole"  # sherpa-onnx owns its sessions: one provider
    assert split_mode("faster-whisper", "faster-whisper", {}) == "whole"  # CTranslate2: one device per model
    assert split_mode("plugin:torch", "orpheus-snac", {}) == "device_map"


def place(**kw: Any) -> Any:
    args: dict[str, Any] = {
        "model_id": "m", "engine_id": "sherpa-onnx", "requested": "cuda", "device": "cuda", "split": "whole",
        "need_bytes": 500 * MB, "available_bytes": 2 * GB,
    }  # fmt: skip
    args.update(kw)
    return plan(**args)


def test_plan_whole_models() -> None:
    p = place()
    assert (p.device, p.gpu_bytes, p.cpu_bytes, p.notice, p.processor()) == ("cuda", 500 * MB, 0, None, "100% GPU")
    p = place(available_bytes=400 * MB)
    assert p.device == "cpu" and p.processor() == "100% CPU" and "runs on the CPU" in (p.notice or "")
    p = place(engine_id="faster-whisper", requested="coreml", device="coreml")
    assert p.device == "cpu" and "cannot use coreml" in (p.notice or "")  # CTranslate2 has no Metal backend
    p = place(engine_id="kokoro-onnx")
    assert p.device == "cpu" and "cannot use cuda" in (p.notice or "")
    p = place(available_bytes=None)  # free memory unknown (ROCm, DirectML, a driver without tools): as before
    assert p.device == "cuda" and p.gpu_bytes == 500 * MB and not p.measured
    p = place(requested="cpu", device="cpu")
    assert p.device == "cpu" and p.notice is None
    p = place(requested="cuda", device="cpu")  # asked for an accelerator this machine does not have
    assert p.device == "cpu" and "not available" in (p.notice or "")


def test_plan_layers_graphs_and_device_map() -> None:
    layers = LayerInfo(block_count=28, file_bytes=2900 * MB)
    p = place(engine_id="tts-server", split="layers", layers=layers, need_bytes=3 * GB, available_bytes=1840 * MB)
    assert (p.device, p.gpu_layers, p.total_layers) == ("cuda", 15, 29)
    assert p.processor() == "48%/52% CPU/GPU" and "15 of 29 layers" in (p.notice or "")
    p = place(engine_id="tts-server", split="layers", layers=layers, available_bytes=100 * MB)
    assert p.device == "cpu" and p.gpu_layers == 0 and p.processor() == "100% CPU"
    p = place(engine_id="tts-server", split="layers", layers=layers, available_bytes=100 * MB, operator_ngl=99)
    assert p.operator_layers and p.gpu_layers == 29 and p.notice is None  # the operator's -ngl is not second-guessed

    graphs = {"encoder": 600 * MB, "decoder": 200 * MB}
    p = place(engine_id="onnxruntime", split="graphs", graphs=graphs, need_bytes=960 * MB, available_bytes=800 * MB)
    assert p.graphs == {"encoder": "cuda", "decoder": "cpu"} and p.device == "cuda"
    assert p.gpu_bytes == 720 * MB and p.processor() == "25%/75% CPU/GPU" and "decoder" in (p.notice or "")
    opts = LoadOptions(device=p.device, graph_devices=p.graphs)
    assert opts.device_for("encoder") == "cuda" and opts.device_for("decoder") == "cpu"
    assert opts.device_for("codec") == "cuda"

    p = place(engine_id="plugin:torch", split="device_map", need_bytes=6 * GB, available_bytes=4 * GB, device="cuda:1")
    assert p.device == "cuda:1" and p.max_memory is not None and p.max_memory["0"] == 4 * GB
    assert p.processor() == "33%/67% CPU/GPU"


# -- accelerator memory -------------------------------------------------------------------------------------------
class FakeNvml:
    """The few pynvml calls bolo makes, over made-up figures."""

    def __init__(self, total: int = 12 * GB, used: int = 1 * GB, ours: int = 0, name: str = "NVIDIA GeForce RTX 3060"):
        self.total, self.used, self.ours, self.name = total, used, ours, name
        self.inits = 0

    def nvmlInit(self) -> None:
        self.inits += 1

    def nvmlShutdown(self) -> None:
        pass

    def nvmlDeviceGetCount(self) -> int:
        return 1

    def nvmlDeviceGetHandleByIndex(self, i: int) -> int:
        return i

    def nvmlDeviceGetName(self, h: int) -> bytes:
        return self.name.encode()

    def nvmlDeviceGetMemoryInfo(self, h: int) -> Any:
        used = self.used + self.ours
        return SimpleNamespace(total=self.total, used=used, free=self.total - used)

    def nvmlDeviceGetUtilizationRates(self, h: int) -> Any:
        return SimpleNamespace(gpu=37)

    def nvmlDeviceGetComputeRunningProcesses(self, h: int) -> list[Any]:
        return [
            SimpleNamespace(pid=os.getpid(), usedGpuMemory=self.ours),
            SimpleNamespace(pid=1, usedGpuMemory=self.used),
        ]


def test_nvml_readings() -> None:
    nv = FakeNvml(ours=300 * MB)
    acc = Accelerators(nvml=nv, which=lambda _: None)
    (d,) = acc.devices()
    assert (d.name, d.total_bytes, d.used_bytes, d.free_bytes) == (
        "NVIDIA GeForce RTX 3060",
        12 * GB,
        1324 * MB,
        12 * GB - 1324 * MB,
    )
    assert d.process_bytes == 300 * MB and d.utilization == 37 and acc.source == "nvml" and not d.unified
    assert acc.device("cuda:0") is d or acc.device("cuda:0") == d
    assert acc.device("cuda:1") is None and acc.device("coreml") is None
    assert acc.default_device() == "cuda:0"
    nv.ours = 0
    assert acc.devices()[0].process_bytes == 300 * MB  # cached for max_age_s
    assert acc.devices(fresh=True)[0].process_bytes == 0
    assert nv.inits == 1  # NVML is initialised once


def test_directml_adapters_on_windows() -> None:
    """With device = "dml" on Windows the DXGI adapters are DirectML devices: free memory is this process's budget
    minus its use (integrated graphics: none, it is the machine's memory)."""
    from bolo.winhw import Adapter

    adapters = [
        Adapter(0, "NVIDIA GeForce RTX 3060 Laptop GPU", 0x10DE, 6 * GB, 8 * GB, False, 5 * GB, 1 * GB),
        Adapter(1, "Intel(R) UHD Graphics", 0x8086, 128 * MB, 8 * GB, False, 3 * GB, 0),
    ]
    acc = Accelerators(nvml=None, which=lambda _: None, system="windows", adapters=lambda: adapters, max_age_s=0)
    acc._nvml_tried = True
    rtx, uhd = acc.devices()
    assert (rtx.backend, rtx.index, rtx.total_bytes, rtx.free_bytes, rtx.process_bytes) == (
        "dml",
        0,
        6 * GB,
        4 * GB,
        GB,
    )
    assert uhd.unified and uhd.free_bytes is None and uhd.vendor == "intel" and acc.source == "dxgi"
    assert acc.device("dml:1") is not None and acc.device("dml") == rtx and acc.device("directml:0") == rtx
    assert acc.default_device() is None and acc.cuda_devices() == []  # "auto" never picks DirectML
    # without device = "dml" nothing asks DXGI
    plain = Accelerators(nvml=None, which=lambda _: None, system="windows")
    plain._nvml_tried = True
    assert plain.devices() == []


def test_nvidia_smi_readings() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str]) -> str | None:
        calls.append(argv)
        if any(a.startswith("--query-gpu") for a in argv):
            return "0, GPU-aaa, NVIDIA GeForce RTX 3060, 12288, 9000, 3288, 12\n1, GPU-bbb, Tesla T4, 15360, 15000, 360, 0\n"
        return f"{os.getpid()}, GPU-aaa, 2048\n4242, GPU-aaa, 1000\n"

    acc = Accelerators(nvml=None, runner=run, which=lambda name: "/usr/bin/" + name, system="linux")
    acc._nvml_tried = True  # no pynvml
    devices = acc.devices()
    assert [d.name for d in devices] == ["NVIDIA GeForce RTX 3060", "Tesla T4"] and acc.source == "nvidia-smi"
    assert devices[0].total_bytes == 12 * GB and devices[0].used_bytes == 3288 * MB
    assert devices[0].process_bytes == 2 * GB and devices[1].process_bytes == 0
    assert devices[0].utilization == 12.0
    assert acc.device("cuda:1") is not None


def test_apple_unified_memory_and_nothing() -> None:
    vm = SimpleNamespace(total=16 * GB, available=10 * GB)
    acc = Accelerators(nvml=None, which=lambda _: None, system="Darwin", machine="arm64", virtual_memory=lambda: vm)
    acc._nvml_tried = True
    (d,) = acc.devices()
    assert d.unified and d.backend == "metal" and d.process_bytes is None  # not told apart from ordinary memory
    assert (d.total_bytes, d.free_bytes, d.used_bytes) == (16 * GB, 10 * GB, 6 * GB)
    assert acc.device("coreml") is not None and acc.device("mps") is not None and acc.default_device() == "coreml"
    none = Accelerators(nvml=None, which=lambda _: None, system="linux", machine="x86_64")
    none._nvml_tried = True
    assert none.devices() == [] and none.default_device() is None and none.source == ""


# -- the lifecycle manager with a fake GPU ------------------------------------------------------------------------
def add(core: Bolo, model_id: str, task: str = "stt", size_mb: int = 100, **options: Any) -> None:
    kind = {"stt": "fake-stt", "tts": "fake-tts", "vad": "fake-vad"}[task]
    core.catalog.add(
        ModelSpec.from_dict({
            "id": model_id, "task": task, "engine": "fake", "kind": kind, "family": "Fake", "languages": ["en"],
            "license": "Apache-2.0", "builtin": True, "size_bytes": size_mb * MB, "options": options,
        })
    )  # fmt: skip


@pytest.fixture
def gpu(monkeypatch: pytest.MonkeyPatch) -> FakeNvml:
    """A fake 12 GB card: a fake model placed on it takes its declared size of accelerator memory until closed."""
    from bolo.engines.fake import FakeEngine

    nv = FakeNvml(total=12 * GB, used=1 * GB)
    real = FakeEngine.load

    def load(self: FakeEngine, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Any:
        model = real(self, kind, task, path, spec, opts)
        if opts.device != "cpu":
            took = int(spec.get("gpu_mb", 100)) * MB
            nv.ours += took
            close = model.close

            def closed() -> None:
                nv.ours -= took
                close()

            model.close = closed  # type: ignore[method-assign]
        model.seen_device = opts.device  # type: ignore[attr-defined]
        return model

    monkeypatch.setattr(FakeEngine, "load", load)
    return nv


async def gpu_core(settings: Settings, nv: FakeNvml) -> Bolo:
    settings.device = "cuda"
    core = Bolo(settings)
    core.manager.accel = Accelerators(nvml=nv, which=lambda _: None, max_age_s=0)
    await core.start()
    return core


async def test_models_go_on_the_gpu_while_they_fit(settings: Settings, gpu: FakeNvml) -> None:
    settings.gpu_reserve_mb = 512
    core = await gpu_core(settings, gpu)
    try:
        add(core, "a")
        a = await core.manager.load("a")
        assert a.model.seen_device == "cuda:0"  # type: ignore[attr-defined]  # "cuda": the GPU with the most free memory
        assert a.vram_bytes == 100 * MB and a.vram_measured and a.processor() == "100% GPU"
        assert core.manager.gpu_in_use() == 100 * MB
        # 12 GB - 1 GB used by others - 100 MB - 512 MB reserve: about 10.4 GB left; 11 GB does not fit
        add(core, "big", size_mb=11 * 1024, gpu_mb=0)
        big = await core.manager.load("big")
        assert big.model.seen_device == "cpu" and big.processor() == "100% CPU"  # type: ignore[attr-defined]
        assert "runs on the CPU" in (big.placement.notice or "")
        add(core, "pinned")
        core.store.set_model_settings("pinned", {"device": "cpu"})  # a per-model pin wins over `device`
        pinned = await core.manager.load("pinned")
        assert pinned.model.seen_device == "cpu" and pinned.processor() == "100% CPU"  # type: ignore[attr-defined]
        add(core, "catalog-pinned", device="cpu")
        assert (await core.manager.load("catalog-pinned")).placement.device == "cpu"

        stats = core.stats()
        (g,) = stats["system"]["gpus"]
        assert g["name"] == "NVIDIA GeForce RTX 3060" and g["process_bytes"] == 100 * MB and g["utilization"] == 37
        row = next(m for m in stats["models"] if m["id"] == "a")
        assert row["processor"] == "100% GPU" and row["size_vram_bytes"] == 100 * MB and row["device"] == "cuda:0"

        await core.manager.unload("a")
        assert gpu.ours == 0 and core.manager.gpu_in_use() == 0
    finally:
        await core.stop()


async def test_gpu_budget_evicts_idle_gpu_models_then_uses_the_cpu(settings: Settings, gpu: FakeNvml) -> None:
    settings.gpu_memory_budget_mb = 250  # each fake model needs 120 MB (100 MB x 1.2) and takes 100 MB
    core = await gpu_core(settings, gpu)
    try:
        for name in ("a", "b", "c", "d"):
            add(core, name)
        a = await core.manager.load("a")
        await asyncio.sleep(0.01)
        b = await core.manager.load("b")
        assert a.processor() == b.processor() == "100% GPU"
        c = await core.manager.load("c")  # 200 MB in use, 50 left: the least recently used idle GPU model goes
        assert "a" not in core.manager.loaded and c.processor() == "100% GPU" and core.manager.gpu_evictions == 1
        async with core.manager.use("b"), core.manager.use("c"):
            d = await core.manager.load("d")  # both GPU models are busy: d goes on the CPU, nothing is refused
            assert d.processor() == "100% CPU" and "runs on the CPU" in (d.placement.notice or "")
        assert {"b", "c", "d"} <= set(core.manager.loaded)
    finally:
        await core.stop()


async def test_free_vram_evictions(settings: Settings, gpu: FakeNvml) -> None:
    gpu.used = 12 * GB - 512 * MB - 300 * MB  # 300 MB free after the reserve
    core = await gpu_core(settings, gpu)
    try:
        add(core, "a", gpu_mb=200)
        add(core, "b", gpu_mb=200)
        add(core, "huge", size_mb=4096)
        assert (await core.manager.load("a")).processor() == "100% GPU"
        b = await core.manager.load("b")  # 100 MB free: unloading idle "a" lets "b" fit whole
        assert b.processor() == "100% GPU" and "a" not in core.manager.loaded
        huge = await core.manager.load("huge")  # would not fit even after evicting "b": "b" stays, huge on the CPU
        assert huge.processor() == "100% CPU" and "b" in core.manager.loaded
    finally:
        await core.stop()


async def test_split_reported_from_a_plugin_and_unified_memory(settings: Settings, gpu: FakeNvml) -> None:
    from bolo.engines.fake import FakeEngine

    core = await gpu_core(settings, gpu)
    try:
        add(core, "split", size_mb=1000)
        real = FakeEngine.load

        def reporting(self: FakeEngine, *args: Any) -> Any:
            model = real(self, *args)
            model.placement_report = {"gpu_bytes": 300 * MB, "cpu_bytes": 700 * MB}  # type: ignore[attr-defined]
            return model

        FakeEngine.load = reporting  # type: ignore[method-assign]
        try:
            entry = await core.manager.load("split")
        finally:
            FakeEngine.load = real  # type: ignore[method-assign]
        assert entry.vram_bytes == 300 * MB and entry.processor() == "70%/30% CPU/GPU"
    finally:
        await core.stop()

    vm = SimpleNamespace(total=16 * GB, available=12 * GB)
    settings.device = "mps"
    mac = Bolo(settings)
    mac.manager.accel = Accelerators(nvml=None, which=lambda _: None, system="darwin", machine="arm64",
                                     virtual_memory=lambda: vm, max_age_s=0)  # fmt: skip
    mac.manager.accel._nvml_tried = True
    await mac.start()
    try:
        add(mac, "m")
        m = await mac.manager.load("m")
        # unified memory: the GPU share is the plan (it cannot be measured apart from the RSS); said so
        assert m.processor() == "100% GPU" and not m.vram_measured and m.placement.device == "mps"
    finally:
        await mac.stop()


# -- keep-alive in the lifecycle manager ---------------------------------------------------------------------------
async def test_ttl_precedence(core: Bolo) -> None:
    s = core.settings
    add(core, "s")
    add(core, "v", task="vad")
    spec, vad = core.manager.resolve("s"), core.manager.resolve("v")
    assert core.manager.ttl_for(spec) == 300.0  # keep_alive "5m"
    s.keep_alive = "10m"
    assert core.manager.ttl_for(spec) == 600.0 and core.manager.ttl_for(vad) == -1.0  # vad_ttl_s = -1
    s.stt_ttl_s = 30
    assert core.manager.ttl_for(spec) == 30.0  # the per-task TTL overrides the global keep_alive
    core.store.set_model_settings("s", {"keep_alive": "1m"})
    assert core.manager.ttl_for(spec) == 60.0  # a per-model pin overrides both
    entry = await core.manager.load("s")
    core.manager.set_keep_alive(entry, -1.0)
    assert core.manager.ttl_for(spec, entry) == -1.0 and core.manager.expires_in(entry) is None


async def test_request_keep_alive(core: Bolo) -> None:
    add(core, "s")
    add(core, "v", task="vad")
    token = REQUEST_KEEP_ALIVE.set(RequestKeepAlive(0.0, frozenset({"stt"})))
    try:
        async with core.manager.use("s"), core.manager.use("v"):
            assert {"s", "v"} <= set(core.manager.loaded)
        # keep_alive 0 unloads the request's model when it is done; the VAD it used on the side stays
        assert "s" not in core.manager.loaded and "v" in core.manager.loaded
    finally:
        REQUEST_KEEP_ALIVE.reset(token)

    token = REQUEST_KEEP_ALIVE.set(RequestKeepAlive(-1.0, frozenset({"stt"})))
    try:
        async with core.manager.use("s"):
            pass
    finally:
        REQUEST_KEEP_ALIVE.reset(token)
    entry = core.manager.loaded["s"]
    assert entry.keep_alive_s == -1.0 and core.manager.expires_in(entry) is None
    async with core.manager.use("s"):  # a request without keep_alive leaves the earlier one in place
        pass
    assert entry.keep_alive_s == -1.0

    # a session: 0 does not unload between utterances, only when the session ends
    ka = RequestKeepAlive(0.0, frozenset({"stt"}), session=True)
    token = REQUEST_KEEP_ALIVE.set(ka)
    try:
        for _ in range(3):
            async with core.manager.use("s"):
                pass
            assert "s" in core.manager.loaded
    finally:
        REQUEST_KEEP_ALIVE.reset(token)
    await core.manager.apply_keep_alive(ka.used, ka.seconds)
    assert "s" not in core.manager.loaded


async def test_keep_alive_expiry_unloads(core: Bolo) -> None:
    add(core, "s")
    entry = await core.manager.load("s")
    core.manager.set_keep_alive(entry, 0.2)
    left = core.manager.expires_in(entry)
    assert left is not None and 0.1 < left <= 0.2
    for _ in range(50):
        if "s" not in core.manager.loaded:
            break
        await asyncio.sleep(0.05)
    assert "s" not in core.manager.loaded


# -- routes, ps and the command line ----------------------------------------------------------------------------------
@pytest.fixture
def live(settings: Settings) -> Any:
    srv = LiveServer(settings)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def test_routes(live: LiveServer) -> None:
    url = live.url
    assert httpx.get(url + "/api/v1/ps").json()["models"] == []
    r = httpx.post(url + "/api/v1/models/fake-stt/load", json={"keep_alive": "-1"})
    assert r.status_code == 200
    body = r.json()
    assert body["processor"] == "100% CPU" and body["device"] == "cpu" and body["keep_alive_s"] == -1.0
    doc = httpx.get(url + "/api/v1/ps").json()
    (m,) = doc["models"]
    assert (m["name"], m["model"], m["task"], m["processor"], m["until"], m["expires_at"]) == (
        "fake-stt", "fake-stt", "stt", "100% CPU", "Forever", None,
    )  # fmt: skip
    assert m["size"] >= 0 and m["size_vram"] == 0 and m["keep_alive"] == "forever"
    assert {"cpu_percent", "rss_bytes", "gpus", "ram_total_bytes"} <= set(doc["system"])
    assert doc["budgets"]["keep_alive"] == "5m"
    assert httpx.post(url + "/api/v1/models/fake-stt/load?keep_alive=10m").json()["keep_alive_s"] == 600.0
    m = httpx.get(url + "/api/v1/ps").json()["models"][0]
    assert m["until"] in ("10 minutes from now", "9 minutes from now") and m["expires_at"].endswith("Z")
    assert httpx.post(url + "/api/v1/models/fake-stt/load", json={"keep_alive": "soon"}).status_code == 400
    assert httpx.post(url + "/api/v1/models/fake-stt/unload").json()["unloaded"] is True
    assert httpx.post(url + "/api/v1/models/fake-stt/unload").json()["unloaded"] is False
    # load with keep_alive 0: loaded and unloaded at once (Ollama's way to unload)
    assert httpx.post(url + "/api/v1/models/fake-stt/load", json={"keep_alive": 0}).json()["loaded"] is False

    # a transcription with keep_alive 0 leaves nothing loaded; OpenAI clients that do not send it are unaffected
    files = {"file": wav_file(1.0)}
    r = httpx.post(url + "/v1/audio/transcriptions", files=files, data={"model": "fake-stt", "keep_alive": "0"})
    assert r.status_code == 200 and "fake-stt" not in live.core.manager.loaded
    r = httpx.post(url + "/v1/audio/transcriptions", files={"file": wav_file(1.0)}, data={"model": "fake-stt"})
    assert r.status_code == 200 and "fake-stt" in live.core.manager.loaded
    r = httpx.post(url + "/v1/audio/transcriptions", files={"file": wav_file(1.0)}, data={"keep_alive": "x"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "keep_alive"
    # speech: a JSON field
    r = httpx.post(url + "/v1/audio/speech", json={"model": "fake-tts", "input": "Hi.", "keep_alive": -1})
    assert r.status_code == 200
    assert live.core.manager.loaded["fake-tts"].keep_alive_s == -1.0
    r = httpx.post(url + "/v1/audio/speech", json={"model": "fake-tts", "input": "Hi.", "keep_alive": 0})
    assert r.status_code == 200 and "fake-tts" not in live.core.manager.loaded

    # per-model pins through the settings route
    r = httpx.put(url + "/api/v1/models/fake-stt/settings", json={"device": "CPU", "keep_alive": "2m"})
    assert r.json()["settings"] == {"device": "cpu", "keep_alive": "2m"}
    assert httpx.put(url + "/api/v1/models/fake-stt/settings", json={"keep_alive": "x"}).status_code == 400

    # the speaches-compatible /api/ps keeps its own shape (a list of ids): no Ollama alias there
    assert httpx.get(url + "/api/ps").json() == {"models": ["fake-stt"]}


def test_websocket_keep_alive(live: LiveServer) -> None:
    import json

    from websockets.sync.client import connect

    with connect(live.ws_url + "/api/v1/tts/stream?model=fake-tts&keep_alive=0") as ws:
        json.loads(ws.recv())  # ready
        ws.send(json.dumps({"type": "text", "text": "One. Two.", "flush": True}))
        ws.send(json.dumps({"type": "stop"}))
        while True:
            msg = ws.recv()
            if isinstance(msg, str) and json.loads(msg).get("type") in ("done", "error"):
                break
    for _ in range(40):
        if "fake-tts" not in live.core.manager.loaded:
            break
        import time

        time.sleep(0.05)
    assert "fake-tts" not in live.core.manager.loaded  # unloaded when the session ended


def test_cli_load_ps_stop(live: LiveServer, capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--url", live.url]
    assert cli.main(["load", "fake-stt", "--keep-alive", "-1", *base]) == 0
    out = capsys.readouterr().out
    assert "fake-stt" in out and "100% CPU on cpu" in out and "keep alive forever" in out
    assert cli.main(["run", "fake-tts", *base]) == 0
    capsys.readouterr()
    assert cli.main(["ps", *base]) == 0
    out = capsys.readouterr().out
    assert "MODEL" in out and "PROCESSOR" in out and "UNTIL" in out and "Forever" in out and "ACTIVE" not in out
    assert cli.main(["ps", "-v", *base]) == 0
    assert "ACTIVE" in capsys.readouterr().out
    assert cli.main(["stop", "fake-stt", "fake-tts", *base]) == 0
    assert "unloaded fake-stt" in capsys.readouterr().out
    assert cli.main(["stop", "fake-stt", *base]) == 0
    assert "was not loaded" in capsys.readouterr().out
    assert cli.main(["load", "fake-stt", "--keep-alive", "soon", *base]) == 1


def test_format_ps() -> None:
    doc = {
        "system": {
            "cpu_percent": 12.5, "rss_bytes": 1536 * MB, "ram_total_bytes": 32 * GB, "ram_available_bytes": 20 * GB,
            "device": "cuda",
            "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3060", "used_bytes": int(3.1 * GB), "total_bytes": 12 * GB,
                      "process_bytes": 2 * GB, "utilization": 41.0, "unified": False}],
        },
        "models": [
            {"name": "whisper-large-v3-turbo", "task": "stt", "size": int(1.6 * GB), "processor": "100% GPU",
             "device": "cuda", "until": "4 minutes from now", "active": 1, "requests": 12, "placement": {}},
            {"name": "orpheus-3b-llamacpp", "task": "tts", "size": 3 * GB, "processor": "35%/65% CPU/GPU",
             "device": "cuda", "until": "Forever", "placement": {"notice": "19 of 29 layers on cuda"}},
            {"name": "silero-vad", "task": "vad", "size": 2 * MB, "processor": "100% CPU", "device": "cpu",
             "until": "Stopping..."},
        ],
    }  # fmt: skip
    text = cli.format_ps(doc)
    lines = text.splitlines()
    assert lines[0].startswith("cpu 12.5%  rss 1.5 GB  ram 12.0/32.0 GB  gpu0 NVIDIA GeForce RTX 3060 vram 3.1/12.0 GB")
    assert "(bolo 2.0 GB) 41% busy" in lines[0]
    assert lines[1].split() == ["MODEL", "TASK", "SIZE", "PROCESSOR", "DEVICE", "GPU", "UNTIL"]
    assert "35%/65% CPU/GPU" in text and "4 minutes from now" in text and "Stopping..." in text
    verbose = cli.format_ps(doc, verbose=True)
    assert "ACTIVE" in verbose and "note: orpheus-3b-llamacpp: 19 of 29 layers on cuda" in verbose
    assert verbose.splitlines()[1].split()[7:10] == ["ACTIVE", "QUEUED", "PARALLEL"]
    mac = {"system": {"cpu_percent": 1, "rss_bytes": 1, "gpus": [{"index": 0, "name": "Apple GPU (unified memory)",
           "unified": True, "used_bytes": 6 * GB, "total_bytes": 16 * GB}]}, "models": []}  # fmt: skip
    assert "unified memory 6.0/16.0 GB" in cli.format_ps(mac) and "no models loaded" in cli.format_ps(mac)


async def test_llama_layers_through_the_manager(core: Bolo, tmp_path: Path) -> None:
    """The scheduler reads the GGUF's layer count and gives the engine an -ngl that fits the free memory."""
    s = core.settings
    s.device, s.gpu_reserve_mb = "cuda", 0
    core.manager.accel = Accelerators(nvml=FakeNvml(total=2 * GB, used=0), which=lambda _: None, max_age_s=0)
    spec = ModelSpec.from_dict({
        "id": "llm-voice", "task": "tts", "engine": "tts-server", "kind": "tts-server", "family": "X",
        "languages": ["en"], "license": "MIT", "options": {"runtime": "llama-server", "model_file": "m.gguf"},
        "source": {"repo": "x/y"},
    })  # fmt: skip
    core.catalog.add(spec)
    model_dir = core.store.model_dir(spec.id)
    model_dir.mkdir(parents=True, exist_ok=True)
    write_gguf(model_dir / "m.gguf", blocks=28, padding=2900 * MB // 1000)  # a small file: all layers fit
    p = core.manager._plan(spec, core.manager.gpu_available("cuda"))
    assert p.split == "layers" and p.total_layers == 29 and p.gpu_layers == 29 and p.processor() == "100% GPU"
    p = core.manager._plan(spec, 0)
    assert p.gpu_layers == 0 and p.device == "cpu"
    s.tts_server_args = ["-ngl", "4"]
    p = core.manager._plan(spec, 0)
    assert p.operator_layers and p.gpu_layers == 4


def test_graphs_get_their_own_providers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """DiCoW's encoder and decoder, and LFM3-Audio's four graphs, each get the provider of their placement."""
    ort = pytest.importorskip("onnxruntime")
    from bolo import ort_providers, s2s_onnx
    from bolo.engines import dicow_onnx
    from bolo.engines.onnx_turn import OnnxRuntimeEngine

    cuda, cpu = "CUDAExecutionProvider", "CPUExecutionProvider"
    seen: list[Any] = []

    class Built(Exception):
        pass

    class FakeSession:
        def __init__(self, path: str, *args: Any, providers: Any = None, **kwargs: Any) -> None:
            seen.append((Path(path).name, providers))
            if len(seen) >= stop_after:
                raise Built

    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    monkeypatch.setattr(ort_providers, "available_providers", lambda: [cuda, cpu])
    for name in ("encoder.onnx", "decoder.onnx"):
        (tmp_path / name).write_bytes(b"x" * (600 if name.startswith("enc") else 200))
    stop_after = 2
    with pytest.raises(Built):
        dicow_onnx.OnnxNetwork(tmp_path / "encoder.onnx", tmp_path / "decoder.onnx", 1, "cuda", decoder_device="cpu")
    assert seen == [("encoder.onnx", [cuda, cpu]), ("decoder.onnx", [cpu])]
    assert OnnxRuntimeEngine().graphs("dicow_onnx", tmp_path, {}) == {"encoder": 600, "decoder": 200}

    onnx = tmp_path / "lfm" / "onnx"
    onnx.mkdir(parents=True)
    for name in ("decoder_q4.onnx", "audio_encoder.onnx", "depth_decoder.onnx", "codec_decoder.onnx"):
        (onnx / name).write_bytes(b"x" * 10)
    (onnx / "decoder_q4.onnx_data").write_bytes(b"x" * 90)
    seen.clear()
    stop_after = 4
    with pytest.raises(Built):
        s2s_onnx.OnnxGraphs(
            tmp_path / "lfm",
            providers=[cuda, cpu],
            graph_providers={"depth_decoder": [cpu], "codec_decoder": [cpu]},
        )
    assert [p for _, p in seen] == [[cuda, cpu], [cuda, cpu], [cpu], [cpu]]
    sizes = OnnxRuntimeEngine().graphs("lfm3_audio_onnx", tmp_path / "lfm", {})
    assert sizes == {"decoder": 100, "encoder": 10, "depth_decoder": 10, "codec_decoder": 10}
