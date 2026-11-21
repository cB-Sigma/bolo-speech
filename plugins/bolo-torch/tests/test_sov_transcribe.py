"""SOV-Transcribe-1 (Whisper encoder + Gemma 4 26B A4B) with a fake network: prompts, budget, the catalog entry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from bolo.catalog import Catalog
from bolo.engines.base import STTRequest

from bolo_torch.engine import LOADERS, NOT_IMPLEMENTED, TorchEngine
from bolo_torch.models import sov_transcribe as sov
from bolo_torch.remote_code import RemoteCodeRefused
from bolo_torch.runtime import InvalidRequest, LoadContext


class FakeBackend(sov.Backend):
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, int, float]] = []

    def generate(self, audio: Any, prompt: str, max_new_tokens: int, temperature: float) -> str:
        self.calls.append((len(audio), prompt, max_new_tokens, temperature))
        return f" heard {len(audio) / 16000:.1f} s \n"


def test_prompts_follow_the_model_card() -> None:
    assert sov.instruction(None) == sov.AUTO
    assert (
        sov.instruction("fr-FR")
        == "Transcribe the French audio into text, ensuring all punctuation marks are included."
    )
    keyed = sov.instruction("en", "Kubernetes, Abhinav\n bolo ,")
    assert keyed == (
        "This audio may contain the following words or phrases: Kubernetes, Abhinav, bolo. "
        "Transcribe the English audio into text, ensuring all punctuation marks are included."
    )
    with pytest.raises(InvalidRequest, match="not 'hi'"):
        sov.instruction("hi")


def test_transcription_with_a_fake_network() -> None:
    backend = FakeBackend()
    model = sov.SovTranscribe(backend, max_new_tokens=100)
    assert model.languages == ("en", "fr", "de", "es", "it", "pt", "nl") and not model.supports_translate
    out = model.transcribe(
        [
            STTRequest(np.zeros(32000, np.float32), language="de"),
            STTRequest(np.zeros(100, np.float32)),  # shorter than 50 ms: nothing is sent
            STTRequest(np.zeros(16000 * 20, np.float32), hotwords="bolo", temperature=0.4),
        ]
    )
    assert [r.text for r in out] == ["heard 2.0 s", "", "heard 20.0 s"] and out[0].language == "de"
    assert len(backend.calls) == 2
    assert backend.calls[0][1].startswith("Transcribe the German audio") and backend.calls[0][2:] == (56, 0.0)
    assert backend.calls[1][1].startswith("This audio may contain") and backend.calls[1][2:] == (100, 0.4)
    with pytest.raises(InvalidRequest, match="does not translate"):
        model.transcribe([STTRequest(np.zeros(16000, np.float32), task="translate")])


def test_the_kind_is_registered_and_refuses_unpinned_code(tmp_path: Path) -> None:
    assert LOADERS["sov-transcribe"] == ("bolo_torch.models.sov_transcribe", "load")
    assert "sov-transcribe" in TorchEngine().kinds and "sov-transcribe" not in NOT_IMPLEMENTED
    (tmp_path / "manifest.json").write_text(json.dumps({"source": {"revision": "b" * 40}}))
    ctx = LoadContext("sov-transcribe", "stt", tmp_path, {"remote_code": {"revision": "b" * 40,
                                                                          "files": {"x.py": "0" * 64}}})  # fmt: skip
    with pytest.raises(RemoteCodeRefused, match="allow_remote_code=true"):
        sov.load(ctx)  # no opt-in: refused before transformers is imported


def test_catalog_entry() -> None:
    spec = Catalog.load().resolve("sov-transcribe-1")
    assert spec is not None and spec.kind == "sov-transcribe" and spec.task == "stt" and spec.engine == "plugin:torch"
    assert spec.license == "Apache-2.0" and spec.commercial_use
    assert spec.source.repo == "salesforce-sovereign-intelligence/SOV-Transcribe-1"
    assert len(spec.source.revision) == 40 and spec.source.revision == spec.options["remote_code"]["revision"]
    pinned = set(spec.options["remote_code"]["files"])
    assert pinned == {
        "config.json", "processor_config.json", "configuration_sovereignai_transcribe.py",
        "modeling_sovereignai_transcribe.py", "processing_sovereignai_transcribe.py",
    }  # fmt: skip
    assert "vllm_plugin/*" in spec.source.exclude  # its Python files are not reviewed: never downloaded
    assert spec.options.get("not_yet_run") is True
