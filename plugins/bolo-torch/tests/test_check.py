"""``bolo-torch check``: imports, code bundles and the tiny build of a model's architecture (no weights)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bolo_torch import check
from bolo_torch.cli import main


def test_imports_say_what_is_missing(capsys: pytest.CaptureFixture[str]) -> None:
    ok = check.check_imports("demucs-denoiser", None)
    assert ok["status"] == "ok"
    missing = check.check_imports("pyannote-diarization", "pyannote")
    assert missing["status"] in ("ok", "fail")
    if missing["status"] == "fail":
        assert 'pip install "bolo-torch[pyannote]"' in missing["detail"]
    assert check.check_imports("sravaani", None)["status"] == "fail"
    assert check.check_imports("no-such-kind", None)["status"] == "fail"
    assert main(["check", json.dumps({"kind": "demucs-denoiser"})]) == 0
    assert json.loads(capsys.readouterr().out)["steps"][0]["name"] == "imports"


def test_a_kind_that_is_not_a_transformers_model_is_skipped_not_passed(tmp_path: Path) -> None:
    (step,) = check.check_tiny({"kind": "sepformer", "config_dir": str(tmp_path)})
    assert step["status"] == "skip"


def test_orpheus_built_tiny_from_its_full_size_config_loads_with_a_split_and_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Llama 3B-sized config (as Orpheus' config.json says it), built with 2 heads and 2 layers: saved, planned
    for a GPU/CPU split, loaded in bfloat16 and run once. Nothing of the real size is ever made. Without a GPU
    whatever the host has: the plan's GPU share is loaded on the CPU."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    config = transformers.LlamaConfig(
        hidden_size=3072, intermediate_size=8192, num_hidden_layers=28, num_attention_heads=24,
        num_key_value_heads=8, vocab_size=156940, max_position_embeddings=131072, architectures=["LlamaForCausalLM"],
    )  # fmt: skip
    config.save_pretrained(tmp_path)
    steps = check.check_tiny({"kind": "orpheus-snac", "config_dir": str(tmp_path), "model": "veena", "task": "tts"})
    by = {s["name"]: s for s in steps}
    assert by["tiny-build"]["status"] == "ok" and "hidden_size 3072->256" in by["tiny-build"]["detail"]
    assert by["tiny-split"]["status"] == "ok"
    assert by["tiny-load"]["status"] == "ok", by["tiny-load"]
    assert by["tiny-request"]["status"] == "ok" and "156940" in by["tiny-request"]["detail"]


def test_shrink_keeps_the_head_size_and_reaches_sub_configurations() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
                                      num_hidden_layers=32, intermediate_size=14336)  # fmt: skip
    changed = check.shrink(config)
    assert config.hidden_size == 256 and config.num_attention_heads == 2 and config.num_key_value_heads == 2
    assert config.num_hidden_layers == 2 and config.intermediate_size == 1024 and changed


def test_remote_code_is_built_only_after_opting_in(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (step,) = check.check_tiny(
        {"kind": "step-audio-2", "config_dir": str(tmp_path), "options": {"remote_code": {"revision": "x"}}}
    )
    assert step["status"] == "skip" and "allow_remote_code" in step["detail"]
