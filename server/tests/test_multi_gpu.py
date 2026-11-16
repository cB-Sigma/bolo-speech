"""Several GPUs (``gpus``, ``gpu_spread``, CUDA_VISIBLE_DEVICES), llama-server arguments (GPUs, flash attention, KV
cache type, context and batch size, parallel slots) and ``num_parallel`` / ``max_queue``.

All of it runs against a fake NVML with made-up memory figures and llama-server command lines that are only
built, never run: none of it has been tried on a machine with two GPUs (docs/architecture.md, "Placement").
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bolo import cli
from bolo.accel import Accelerators
from bolo.catalog import ModelSpec
from bolo.config import Settings, load_settings
from bolo.core import Bolo
from bolo.engines.base import LoadOptions
from bolo.llama_args import check_tuning, gpu_args, parallel_args, resolve_tuning, tuning_args
from bolo.placement import LayerInfo, best_gpu, plan_spread, spread_layers
from fakebin import printing_program

MB = 1024 * 1024
GB = 1024 * MB


class MultiNvml:
    """pynvml over several made-up GPUs: (name, total, used by others) each; ``ours`` is what bolo holds."""

    def __init__(self, cards: list[tuple[str, int, int]]):
        self.cards = cards
        self.ours = [0] * len(cards)

    def nvmlInit(self) -> None:
        pass

    def nvmlShutdown(self) -> None:
        pass

    def nvmlDeviceGetCount(self) -> int:
        return len(self.cards)

    def nvmlDeviceGetHandleByIndex(self, i: int) -> int:
        return i

    def nvmlDeviceGetName(self, h: int) -> str:
        return self.cards[h][0]

    def nvmlDeviceGetUUID(self, h: int) -> str:
        return f"GPU-{h}{h}{h}-uuid"

    def nvmlDeviceGetMemoryInfo(self, h: int) -> Any:
        _, total, used = self.cards[h]
        used += self.ours[h]
        return SimpleNamespace(total=total, used=used, free=total - used)

    def nvmlDeviceGetComputeRunningProcesses(self, h: int) -> list[Any]:
        return [SimpleNamespace(pid=os.getpid(), usedGpuMemory=self.ours[h])]


def accel(nv: MultiNvml, environ: dict[str, str] | None = None) -> Accelerators:
    return Accelerators(nvml=nv, which=lambda _: None, max_age_s=0, environ=environ or {})


# -- CUDA_VISIBLE_DEVICES and the gpus setting ---------------------------------------------------------------------
def test_cuda_visible_devices_renumbers_like_cuda() -> None:
    nv = MultiNvml([("A", 8 * GB, 0), ("B", 12 * GB, 0), ("C", 24 * GB, 0)])
    assert [(d.index, d.name) for d in accel(nv).devices()] == [(0, "A"), (1, "B"), (2, "C")]
    seen = accel(nv, {"CUDA_VISIBLE_DEVICES": "2,0"}).devices()
    assert [(d.index, d.physical_index, d.name) for d in seen] == [(0, 2, "C"), (1, 0, "A")]
    assert seen[0].to_dict()["physical_index"] == 2 and seen[0].uuid == "GPU-222-uuid"
    by_uuid = accel(nv, {"CUDA_VISIBLE_DEVICES": "GPU-111"}).devices()
    assert [(d.index, d.name) for d in by_uuid] == [(0, "B")]
    assert accel(nv, {"CUDA_VISIBLE_DEVICES": "-1"}).devices() == []
    assert accel(nv, {"CUDA_VISIBLE_DEVICES": "1,7,2"}).devices()[-1].name == "B"  # CUDA stops at a bad entry


def test_gpus_setting(tmp_path: Path) -> None:
    assert load_settings(env={"BOLO_GPUS": "0, 2", "BOLO_GPU_SPREAD": "1"}).gpus == [0, 2]
    assert load_settings(env={"BOLO_GPUS": "[1]"}).gpu_spread is False
    cfg = tmp_path / "bolo.toml"
    cfg.write_text("gpus = [0, 2]\n[gpu]\nspread = true\n")
    s = load_settings(cfg, env={})
    assert s.gpus == [0, 2] and s.gpu_spread is True
    with pytest.raises(ValueError, match="gpus"):
        load_settings(env={"BOLO_GPUS": "zero"})
    with pytest.raises(ValueError, match="gpus"):
        load_settings(env={"BOLO_GPUS": "-1"})


# -- pure arithmetic -------------------------------------------------------------------------------------------------
def test_best_gpu_and_spread_layers() -> None:
    assert best_gpu({0: 4 * GB, 1: 9 * GB, 2: 9 * GB}) == 1  # most available; the lower number on a tie
    assert best_gpu({0: None, 1: 1}) == 1
    layers = LayerInfo(block_count=39, file_bytes=8 * GB)  # 40 layers of 204.8 MB
    # 6 GB + 4 GB: the 10% overhead (819 MB) is counted on GPU 0, then 40 layers fit in 9.2 GB, shared 5.2 : 4
    per = spread_layers(layers, {0: 6 * GB, 1: 4 * GB})
    assert sum(per.values()) == 40 and per[0] > per[1] > 0
    # too small together: as many as fit, never more than the two hold
    per = spread_layers(layers, {0: 3 * GB, 1: 2 * GB})
    assert 0 < sum(per.values()) < 40 and sum(per.values()) * layers.per_layer <= 5 * GB
    assert spread_layers(layers, {0: 0, 1: 0}) == {}

    p = plan_spread(model_id="m", requested="cuda", layers=layers, available={0: 6 * GB, 1: 4 * GB})
    assert p.device == "cuda:0" and p.gpus == [0, 1] and p.main_gpu == 0 and p.processor() == "100% GPU"
    assert p.tensor_split is not None and abs(sum(p.tensor_split) - 1) < 0.01 and p.notice is None
    assert sum(p.gpu_bytes_by_gpu.values()) == p.gpu_bytes and p.to_dict()["gpus"] == [0, 1]


def test_gpu_args() -> None:
    # one GPU on a one-GPU machine: only -ngl, as before
    assert gpu_args("cuda:0", 20, [], gpus=[0], visible=1) == ["-ngl", "20"]
    # one of several GPUs: llama-server would otherwise spread the layers over all of them
    assert gpu_args("cuda:1", 20, [], gpus=[1], visible=2) == ["-ngl", "20", "--split-mode", "none", "--main-gpu", "1"]
    assert gpu_args("cuda:0", 41, [], gpus=[0, 2], tensor_split=[0.6, 0.4], main_gpu=0, visible=3) == [
        "-ngl", "41", "--split-mode", "layer", "--tensor-split", "0.6,0,0.4", "--main-gpu", "0",
    ]  # fmt: skip
    # the operator's choice wins
    assert gpu_args("cuda:1", 20, ["-ts", "1,1"], gpus=[1], visible=2) == ["-ngl", "20"]
    assert gpu_args("cuda:1", 20, ["--main-gpu=1", "-ngl", "5"], gpus=[1], visible=2) == []
    assert gpu_args("cpu", 0, [], gpus=[], visible=2) == ["-ngl", "0"]
    assert parallel_args(1, []) == [] and parallel_args(3, []) == ["--parallel", "3"]
    assert parallel_args(1, [], always=True) == ["--parallel", "1"] and parallel_args(4, ["-np", "2"]) == []


# -- llama.cpp speed settings ----------------------------------------------------------------------------------------
def test_tuning_precedence_and_arguments() -> None:
    s = Settings(llama_flash_attention="on", llama_kv_cache_type="q8_0", llama_context_size=4096)
    assert resolve_tuning(s, {}) == {"flash_attention": "on", "kv_cache_type": "q8_0", "context_size": 4096}
    # a catalog option beats the setting, a pin beats both; 0 and "" mean unset
    t = resolve_tuning(s, {"kv_cache_type": "f16", "batch_size": 512}, {"kv_cache_type": "Q4_0", "context_size": 0})
    assert t == {"flash_attention": "on", "kv_cache_type": "q4_0", "context_size": 4096, "batch_size": 512}
    assert resolve_tuning(Settings(), {"flash_attention": False}) == {"flash_attention": "off"}
    assert tuning_args(t, []) == [
        "-fa", "on", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "-c", "4096", "-b", "512",
    ]  # fmt: skip
    # an operator's explicit argument wins, whatever its spelling
    assert tuning_args(t, ["--flash-attn", "off", "-ctk", "f16", "--ctx-size=2048", "-b", "64"]) == []
    # a quantized V cache needs flash attention: bolo turns it on unless told otherwise
    assert tuning_args({"kv_cache_type": "q8_0"}, []) == [
        "-fa",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
    ]
    assert tuning_args({"kv_cache_type": "f16"}, []) == ["--cache-type-k", "f16", "--cache-type-v", "f16"]
    assert tuning_args({}, []) == []


def test_tuning_settings_are_checked() -> None:
    s = load_settings(env={"BOLO_LLAMA_KV_CACHE_TYPE": "q8_0", "BOLO_LLAMA_FLASH_ATTENTION": "true"})
    assert s.llama_kv_cache_type == "q8_0" and resolve_tuning(s, {})["flash_attention"] == "on"
    with pytest.raises(ValueError, match="llama_kv_cache_type"):
        load_settings(env={"BOLO_LLAMA_KV_CACHE_TYPE": "q3"})
    with pytest.raises(ValueError, match="llama_flash_attention"):
        load_settings(env={"BOLO_LLAMA_FLASH_ATTENTION": "fast"})
    with pytest.raises(ValueError, match="context_size"):
        check_tuning({"context_size": "big"})


def write_gguf(path: Path, blocks: int, size: int) -> Path:
    """A GGUF header with ``llama.block_count`` and padding up to ``size`` bytes (a sparse file)."""

    def text(t: str) -> bytes:
        return struct.pack("<Q", len(t)) + t.encode()

    body = b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
    body += text("general.architecture") + struct.pack("<I", 8) + text("llama")
    body += text("llama.block_count") + struct.pack("<I", 4) + struct.pack("<I", blocks)
    with path.open("wb") as fh:
        fh.write(body)
        fh.truncate(size)
    return path


def llama_spec(core: Bolo, model_id: str, size: int, **options: Any) -> ModelSpec:
    spec = ModelSpec.from_dict({
        "id": model_id, "task": "tts", "engine": "tts-server", "kind": "tts-server", "family": "X",
        "languages": ["en"], "license": "MIT", "source": {"repo": "x/y"},
        "options": {"runtime": "llama-server", "model_file": "m.gguf", **options},
    })  # fmt: skip
    core.catalog.add(spec)
    model_dir = core.store.model_dir(spec.id)
    model_dir.mkdir(parents=True, exist_ok=True)
    write_gguf(model_dir / "m.gguf", blocks=39, size=size)
    (model_dir / "manifest.json").write_text(json.dumps({"complete": True, "files": [], "size_bytes": size}))
    return spec


def test_the_tts_server_command_line(tmp_path: Path) -> None:
    """What bolo would start: the GPUs, parallel slots and tuning arguments, the operator's arguments last."""
    from bolo.engines.tts_server import TTSServerEngine

    binary = printing_program(tmp_path / "llama-server")
    engine = TTSServerEngine()
    engine.configure(Settings(data_dir=tmp_path, llama_server_bin=str(binary), llama_kv_cache_type="q8_0",
                              tts_server_args=["-c", "1024"]))  # fmt: skip
    (tmp_path / "m.gguf").write_bytes(b"GGUF")
    opts = LoadOptions(threads=2, device="cuda:1", gpu_layers=40, gpus=[1], visible_gpus=2, parallel=3)
    process = engine._process("llama-server", "voice", tmp_path, {"model_file": "m.gguf", "batch_size": 256}, opts)
    argv = process.argv
    assert argv[argv.index("-ngl") + 1] == "40" and argv[argv.index("--main-gpu") + 1] == "1"
    assert argv[argv.index("--split-mode") + 1] == "none" and argv[argv.index("--parallel") + 1] == "3"
    assert argv[argv.index("--cache-type-v") + 1] == "q8_0" and argv[argv.index("-fa") + 1] == "on"
    assert argv[argv.index("-b") + 1] == "256"
    assert argv.count("-c") == 1 and argv[-2:] == ["-c", "1024"]  # the operator's context size, not a second one


# -- the scheduler with two fake GPUs --------------------------------------------------------------------------------
def fake_model(core: Bolo, model_id: str, size_mb: int = 100, **options: Any) -> None:
    core.catalog.add(
        ModelSpec.from_dict({
            "id": model_id, "task": "stt", "engine": "fake", "kind": "fake-stt", "family": "Fake",
            "languages": ["en"], "license": "Apache-2.0", "builtin": True, "size_bytes": size_mb * MB,
            "options": options,
        })
    )  # fmt: skip


async def two_gpu_core(settings: Settings, nv: MultiNvml) -> Bolo:
    settings.device, settings.gpu_reserve_mb = "cuda", 0
    core = Bolo(settings)
    core.manager.accel = accel(nv)
    await core.start()
    return core


async def test_each_model_goes_on_the_gpu_with_the_most_free_memory(settings: Settings) -> None:
    nv = MultiNvml([("small", 8 * GB, 2 * GB), ("big", 24 * GB, 2 * GB)])
    core = await two_gpu_core(settings, nv)
    try:
        fake_model(core, "a")
        a = await core.manager.load("a")
        assert a.placement.device == "cuda:1" and a.placement.gpus == [1]
        # GPU 1 filling up: once GPU 0 has more free memory, the next model goes there
        nv.cards[1] = ("big", 24 * GB, 20 * GB)
        fake_model(core, "b")
        b = await core.manager.load("b")
        assert b.placement.device == "cuda:0" and b.placement.gpus == [0]
        ps = core.ps()
        assert {m["name"]: m["gpus"] for m in ps["models"]} == {"a": [1], "b": [0]}
        assert [g["allowed"] for g in ps["system"]["gpus"]] == [True, True]
        # a pin to one GPU is honoured; a GPU outside `gpus` is refused (the CPU, with a notice)
        fake_model(core, "pinned", device="cuda:1")
        assert (await core.manager.load("pinned")).placement.device == "cuda:1"
        core.settings.gpus = [0]
        fake_model(core, "outside", device="cuda:1")
        outside = await core.manager.load("outside")
        assert outside.placement.device == "cpu" and "not one bolo may use" in (outside.placement.notice or "")
        fake_model(core, "c")
        assert (await core.manager.load("c")).placement.device == "cuda:0"
        assert [g["allowed"] for g in core.ps()["system"]["gpus"]] == [True, False]
        text = cli.format_ps(core.ps())
        assert "(not used: gpus setting)" in text and text.splitlines()[1].split()[5] == "GPU"
    finally:
        await core.stop()


@pytest.fixture
def fake_gpu_engine(monkeypatch: pytest.MonkeyPatch) -> list[LoadOptions]:
    """The tts-server engine replaced by one that records the LoadOptions it was given."""
    from bolo.engines.fake import FakeEngine
    from bolo.engines.tts_server import TTSServerEngine

    seen: list[LoadOptions] = []

    def load(self: Any, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Any:
        seen.append(opts)
        return FakeEngine().load("fake-tts", "tts", path, {}, opts)

    monkeypatch.setattr(TTSServerEngine, "load", load)
    return seen


async def test_a_llama_model_too_big_for_one_gpu_is_spread(
    settings: Settings, fake_gpu_engine: list[LoadOptions]
) -> None:
    nv = MultiNvml([("A", 6 * GB, 0), ("B", 5 * GB, 0)])
    core = await two_gpu_core(settings, nv)
    try:
        llama_spec(core, "voice", size=8 * GB)  # 40 layers of 204.8 MB, plus 10% for the KV cache and buffers
        p = await core.manager.place(core.manager.resolve("voice"))
        assert p.gpus == [0, 1] and p.gpu_layers == 40 and p.tensor_split and p.main_gpu == 0
        assert p.processor() == "100% GPU"
        await core.manager.load("voice")
        opts = fake_gpu_engine[-1]
        assert opts.gpus == [0, 1] and opts.visible_gpus == 2 and opts.tensor_split == p.tensor_split
        args = gpu_args(opts.device, opts.gpu_layers, [], gpus=opts.gpus, tensor_split=opts.tensor_split,
                        main_gpu=opts.main_gpu, visible=opts.visible_gpus)  # fmt: skip
        assert "--tensor-split" in args and args[args.index("--main-gpu") + 1] == "0"
        assert core.ps()["models"][0]["gpus"] == [0, 1]
        # an operator's -ts means the operator chose: one GPU, as many layers as fit there
        core.settings.tts_server_args = ["-ts", "1,0"]
        assert core.manager._place_gpus(core.manager.resolve("voice"), "cuda", {}).gpus == [0]
    finally:
        await core.stop()


async def test_a_model_that_fits_one_gpu_stays_there_unless_gpu_spread(settings: Settings) -> None:
    nv = MultiNvml([("A", 12 * GB, 0), ("B", 12 * GB, 0)])
    core = await two_gpu_core(settings, nv)
    try:
        spec = llama_spec(core, "voice", size=2 * GB)
        p = core.manager._place_gpus(spec, "cuda", {})
        assert p.gpus == [0] and p.tensor_split is None and p.processor() == "100% GPU"
        core.settings.gpu_spread = True  # OLLAMA_SCHED_SPREAD: every llama.cpp model over every allowed GPU
        p = core.manager._place_gpus(spec, "cuda", {})
        assert p.gpus == [0, 1] and p.tensor_split == [0.5, 0.5]
        core.settings.gpus = [1]
        assert core.manager._place_gpus(spec, "cuda", {}).gpus == [1]
        # models that cannot be split over GPUs are not spread: one GPU each
        fake_model(core, "whole")
        core.settings.gpus = []
        assert core.manager._place_gpus(core.manager.resolve("whole"), "cuda", {}).gpus in ([0], [1])
    finally:
        await core.stop()


async def test_the_gpu_budget_counts_every_gpu(settings: Settings) -> None:
    nv = MultiNvml([("A", 6 * GB, 0), ("B", 6 * GB, 0)])
    settings.gpu_memory_budget_mb = 4 * 1024
    core = await two_gpu_core(settings, nv)
    try:
        spec = llama_spec(core, "voice", size=8 * GB)
        p = core.manager._place_gpus(spec, "cuda", {})
        assert p.gpus and sum(p.gpu_bytes_by_gpu.values()) <= 4 * GB and p.cpu_bytes > 0
    finally:
        await core.stop()


async def test_sherpa_onnx_only_uses_gpu_0(settings: Settings) -> None:
    nv = MultiNvml([("A", 4 * GB, 0), ("B", 24 * GB, 0)])
    core = await two_gpu_core(settings, nv)
    try:
        spec = core.manager.resolve("zipformer-en-20m-streaming")
        assert spec.engine == "sherpa-onnx"
        assert core.manager._place_gpus(spec, "cuda", {}).device in ("cuda:0", "cpu")
        core.settings.gpus = [1]
        p = core.manager._place_gpus(spec, "cuda", {})
        assert p.device == "cpu" and "GPU 0" in (p.notice or "")
    finally:
        await core.stop()


# -- num_parallel and max_queue --------------------------------------------------------------------------------------
async def test_num_parallel_and_max_queue(settings: Settings) -> None:
    settings.num_parallel, settings.max_queue = 3, 10
    core = Bolo(settings)
    await core.start()
    try:
        assert settings.parallel() == 3 and core.info()["scheduler"]["num_parallel"] == 3
        entry = await core.manager.load("fake-stt")
        assert entry.queue.workers == 3 and entry.queue.max_queue == 10
        core.store.set_model_settings("fake-tts", {"num_parallel": 2, "max_queue": 4})
        tts = await core.manager.load("fake-tts")
        assert tts.queue.workers == 2 and tts.queue.max_queue == 4
        row = next(m for m in core.ps()["models"] if m["name"] == "fake-tts")
        assert (row["num_parallel"], row["max_queue"], row["queued"], row["running"]) == (2, 4, 0, 0)
        stats = next(m for m in core.stats()["models"] if m["id"] == "fake-tts")
        assert stats["queue"]["num_parallel"] == 2 and stats["queue"]["max_queue"] == 4
        fake_model(core, "old-style", workers=5)  # the catalog's older name for it
        assert core.manager.num_parallel(core.manager.resolve("old-style")) == 5
    finally:
        await core.stop()
    assert Settings(workers=2).parallel() == 2 and Settings(workers=2, num_parallel=4).parallel() == 4
    assert load_settings(env={"BOLO_NUM_PARALLEL": "4", "BOLO_MAX_QUEUE": "8"}).max_queue == 8
    with pytest.raises(ValueError, match="num_parallel"):
        load_settings(env={"BOLO_NUM_PARALLEL": "-1"})
