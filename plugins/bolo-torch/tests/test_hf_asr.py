"""The generic transformers speech-recognition loader (kind hf-asr-pipeline) with a tiny fake pipeline, and the
Jais-ASR catalog entry that uses it."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from bolo.catalog import Catalog
from bolo.engines.base import STTRequest

from bolo_torch.engine import LOADERS, TorchEngine
from bolo_torch.models import hf_asr
from bolo_torch.remote_code import RemoteCodeRefused
from bolo_torch.runtime import InvalidRequest, LoadContext


class FakePipe:
    def __init__(self, answer: Any = None) -> None:
        self.calls: list[tuple[int, int, dict[str, Any]]] = []
        self.answer = answer

    def __call__(self, inputs: dict[str, Any], **kwargs: Any) -> Any:
        assert inputs["array"].dtype == np.float32
        self.calls.append((len(inputs["array"]), inputs["sampling_rate"], kwargs))
        if self.answer is not None:
            return self.answer
        return {"text": f"  heard {len(inputs['array']) / 16000:.1f} s ", "language": "EG"}


def test_requests_become_pipeline_calls() -> None:
    pipe = FakePipe()
    model = hf_asr.HfAsrPipeline(
        pipe, language_kwarg="language", language_map={"ar": "ar", "en": "en"}, call_kwargs={"code_switch": True},
        max_new_tokens=100, tokens_per_s=6.0, result_language_key="language",
    )  # fmt: skip
    assert model.languages == ("ar", "en") and not model.supports_translate
    out = model.transcribe([
        STTRequest(np.zeros(32000, np.float64), language="ar-EG"),
        STTRequest(np.zeros(100, np.float32)),  # shorter than 50 ms: not sent
        STTRequest(np.zeros(16000 * 30, np.float32)),
    ])  # fmt: skip
    assert [r.text for r in out] == ["heard 2.0 s", "", "heard 30.0 s"]
    assert out[0].language == "ar-EG" and out[2].language == "EG"  # detected, when the request named none
    assert out[2].segments[0].end == 30.0
    assert pipe.calls[0] == (32000, 16000, {"code_switch": True, "max_new_tokens": 44, "language": "ar"})
    assert pipe.calls[1][2] == {"code_switch": True, "max_new_tokens": 100}  # budget capped, no language
    with pytest.raises(InvalidRequest, match="not 'de'"):
        model.transcribe([STTRequest(np.zeros(16000, np.float32), language="de")])
    with pytest.raises(InvalidRequest, match="does not translate"):
        model.transcribe([STTRequest(np.zeros(16000, np.float32), task="translate")])


def test_language_not_passed_without_a_kwarg_and_list_results() -> None:
    pipe = FakePipe(answer=[{"text": "hello"}])
    model = hf_asr.HfAsrPipeline(pipe)
    (r,) = model.transcribe([STTRequest(np.zeros(16000, np.float32), language="fr")])
    assert (r.text, r.language) == ("hello", "fr") and "language" not in pipe.calls[0][2]
    assert model.languages == ()


def test_load_builds_the_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built: dict[str, Any] = {}

    def pipeline(task: str, **kwargs: Any) -> FakePipe:
        built.update(task=task, **kwargs)
        return FakePipe()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=pipeline))
    (tmp_path / "config.json").write_text("{}")
    options = {"language_kwarg": "language", "language_map": {"ar": "ar"}, "max_new_tokens": 64,
               "call_kwargs": {"diacritics": False}}  # fmt: skip
    model = hf_asr.load(LoadContext("hf-asr-pipeline", "stt", tmp_path, options, device="cpu", dtype="bf16"))
    assert built == {"task": "automatic-speech-recognition", "model": str(tmp_path), "trust_remote_code": False,
                     "dtype": "bf16", "device": "cpu"}  # fmt: skip
    assert model.max_new_tokens == 64 and model.call_kwargs == {"diacritics": False}
    with pytest.raises(InvalidRequest, match="plain value"):
        hf_asr.load(LoadContext("hf-asr-pipeline", "stt", tmp_path, {"call_kwargs": {"x": object()}}))


def test_remote_code_needs_the_opt_in(tmp_path: Path) -> None:
    assert LOADERS["hf-asr-pipeline"] == ("bolo_torch.models.hf_asr", "load")
    assert "hf-asr-pipeline" in TorchEngine().kinds
    (tmp_path / "manifest.json").write_text(json.dumps({"source": {"revision": "c" * 40}}))
    ctx = LoadContext("hf-asr-pipeline", "stt", tmp_path, {"remote_code": {"revision": "c" * 40,
                                                                           "files": {"x.py": "0" * 64}}})  # fmt: skip
    with pytest.raises(RemoteCodeRefused, match="allow_remote_code=true"):
        hf_asr.load(ctx)  # refused before transformers is imported


def test_jais_catalog_entry() -> None:
    spec = Catalog.load().resolve("jais-asr")
    assert spec is not None and spec.kind == "hf-asr-pipeline" and spec.engine == "plugin:torch"
    assert spec.license == "Apache-2.0" and spec.commercial_use and not spec.gated
    assert spec.source.revision == spec.options["remote_code"]["revision"]
    assert "vllm_plugin/*" in spec.source.exclude
    assert set(spec.options["language_map"]) == set(spec.languages)
    assert spec.options["not_yet_run"] is True and spec.validated == "untested"
