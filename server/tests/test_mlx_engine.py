"""The MLX engine (bolo.engines.mlx) without MLX: catalog entries, word joining, phoneme cutting, device choice, and
what bolo ps and the console call the runtime of a model. The models themselves run in the macOS CI job (on Apple
silicon) and were run on the CPU with MLX's Linux wheels (docs/apple-silicon.md)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bolo import cuda_libs, placement
from bolo.catalog import ModelSpec, packaged_entries
from bolo.engines import BUILTIN_ENGINES
from bolo.engines import mlx as mlx_engine
from bolo.engines.base import LoadOptions, Word


def test_registered_and_in_the_catalog() -> None:
    assert BUILTIN_ENGINES["mlx"] == "bolo.engines.mlx:MlxEngine"
    entries = [s for s in map(ModelSpec.from_dict, packaged_entries()) if s.engine == "mlx"]
    kinds = {s.kind for s in entries}
    assert {"mlx-whisper", "parakeet-mlx", "kokoro"} <= kinds
    assert kinds <= set(mlx_engine.MlxEngine.kinds)
    for s in entries:
        assert s.source.repo and len(s.source.revision) == 40, s.id
        assert "bolo-speech[mlx]" in s.description, s.id
    kokoro = next(s for s in entries if s.kind == "kokoro")
    # only the voices bolo can phonemise (no Japanese or Chinese front end)
    assert kokoro.voices and all(str(v["id"])[0] in "abefhip" for v in kokoro.voices)
    assert "voice-blending" in kokoro.to_api()["features"]


def test_parakeet_tokens_become_words() -> None:
    tok = lambda text, start, end: SimpleNamespace(text=text, start=start, end=end)  # noqa: E731
    words = mlx_engine._words(
        [tok(" Good", 0.0, 0.2), tok(" mor", 0.3, 0.4), tok("ning", 0.4, 0.6), tok(".", 0.6, 0.7)]
    )
    assert words == [Word("Good", 0.0, 0.2), Word("morning.", 0.3, 0.7)]
    assert mlx_engine._words([tok("Hi", 0.0, 0.1)]) == [Word("Hi", 0.0, 0.1)]


def test_long_phonemes_are_cut_at_spaces() -> None:
    assert mlx_engine._fit("ab cd ef", 5) == ["ab", "cd ef"]
    assert mlx_engine._fit("abcdefgh", 3) == ["abc", "def", "gh"]
    assert mlx_engine._fit("short", 510) == ["short"]


def test_gpu_only_when_placed_there_and_metal_is_there(monkeypatch: pytest.MonkeyPatch) -> None:
    assert mlx_engine.on_gpu("coreml") and mlx_engine.on_gpu("mlx") and mlx_engine.on_gpu("mps")
    assert not mlx_engine.on_gpu("cpu") and not mlx_engine.on_gpu("cuda") and not mlx_engine.on_gpu("rocm")
    monkeypatch.setattr(mlx_engine, "metal_available", lambda: True)
    assert mlx_engine._placed(LoadOptions(device="coreml"))
    assert not mlx_engine._placed(LoadOptions(device="cpu"))
    # placed on the GPU where MLX has no Metal (its Linux CPU wheels): the load reports a fallback to the CPU
    monkeypatch.setattr(mlx_engine, "metal_available", lambda: False)
    with cuda_libs.watch() as watched:
        assert not mlx_engine._placed(LoadOptions(device="coreml"))
    assert watched.whole and "MLX could not start Metal" in (watched.notice() or "")


def test_placement_families() -> None:
    assert placement.engine_families("mlx") == frozenset({"metal"})


@pytest.mark.parametrize(
    ("engine", "device", "options", "label"),
    [
        ("sherpa-onnx", "coreml", {}, "Core ML (sherpa-onnx)"),
        ("sherpa-onnx", "cuda:0", {}, "CUDA (sherpa-onnx)"),
        ("sherpa-onnx", "cpu", {}, "CPU"),
        ("faster-whisper", "cuda:1", {}, "CUDA (CTranslate2)"),
        ("mlx", "coreml", {}, "Metal (MLX)"),
        ("mlx", "cpu", {}, "CPU"),
        ("whisper.cpp", "metal", {}, "Metal (whisper.cpp)"),
        ("tts-server", "rocm:0", {}, "ROCm (llama.cpp)"),
        ("tts-server", "vulkan", {"runtime": "audio.cpp"}, "Vulkan (audio.cpp)"),
        ("plugin:torch", "coreml", {}, "MPS (PyTorch)"),
        ("plugin:torch", "rocm", {}, "ROCm (PyTorch)"),
        ("plugin:torch", "cuda:0", {}, "CUDA (PyTorch)"),
        ("remote", "cpu", {}, "remote"),
    ],
)
def test_runtime_labels(engine: str, device: str, options: dict[str, str], label: str) -> None:
    assert placement.runtime_label(engine, device, options) == label


def test_onnxruntime_label_names_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from bolo import ort_providers

    have = ["MIGraphXExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setattr(ort_providers, "available_providers", lambda: have)
    assert placement.runtime_label("onnxruntime", "rocm") == "MIGraphX (ONNX Runtime)"
    assert placement.runtime_label("onnxruntime", "coreml") == "Core ML (ONNX Runtime)"
    monkeypatch.setattr(ort_providers, "available_providers", lambda: ["CPUExecutionProvider"])
    ort_providers._warned.clear()
    assert placement.runtime_label("onnxruntime", "cuda") == "CPU"


def test_placement_carries_the_runtime() -> None:
    p = placement.Placement(device="coreml", runtime="Core ML (sherpa-onnx)")
    assert p.to_dict()["runtime"] == "Core ML (sherpa-onnx)"
