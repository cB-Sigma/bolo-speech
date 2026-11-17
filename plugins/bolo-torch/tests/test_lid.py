"""Language identifiers (task ``lid``): LoRA merging, label files, and the classifier wrappers with tiny modules."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from bolo.plugin.protocol import audio_bytes, read_message

from bolo_torch.models.lid import HFAudioLid, SpeechBrainLid, label_code, merge_lora, read_label_encoder
from bolo_torch.server import TorchServer

torch = pytest.importorskip("torch")


def test_merge_lora_folds_the_product_into_the_weight() -> None:
    w = np.zeros((3, 2))
    a = np.ones((1, 2))
    b = np.arange(3.0).reshape(3, 1)
    out = merge_lora({"fc.weight": w, "fc.lora_A": a, "fc.lora_B": b, "fc.bias": np.ones(3)}, 0.5)
    assert set(out) == {"fc.weight", "fc.bias"}
    assert np.allclose(out["fc.weight"], 0.5 * b @ a)


def test_label_files(tmp_path: Path) -> None:
    f = tmp_path / "language_encoder.txt"
    f.write_text("'marathi' => 0\n'hindi' => 1\n'english' => 2\n================\n'starting_index' => 0\n")
    assert read_label_encoder(f) == ["mr", "hi", "en"]
    assert [label_code(x) for x in ("Hindi", "konkani", "bodo", "santali", "hi")] == ["hi", "kok", "brx", "sat", "hi"]


class _Classifier(torch.nn.Module):  # type: ignore[misc]
    """Says the second label whatever it hears."""

    def forward(self, input_values: object) -> object:
        class Out:
            logits = torch.tensor([[0.0, 5.0, 0.0]])

        return Out()


def _extractor(x: np.ndarray, sampling_rate: int, return_tensors: str) -> dict[str, object]:
    return {"input_values": torch.from_numpy(x)[None]}


def test_hf_classifier_ranks_its_labels() -> None:
    model = HFAudioLid(_Classifier(), _extractor, ["ta", "hi", "or"], "cpu", 30.0)
    found = model.identify(np.zeros(16000, np.float32))
    assert found[0][0] == "hi" and abs(sum(p for _, p in found) - 1.0) < 1e-6 and model.task == "lid"


def test_speechbrain_scores_become_probabilities() -> None:
    modules = {
        "features": lambda wav: wav[:, :, None],
        "norm": lambda feats, lens: feats,
        "embedding": lambda feats, lens: feats.mean(dim=1),
        "classifier": lambda emb: torch.tensor([[[0.9, 0.1]]]),
    }
    model = SpeechBrainLid(modules, ["hi", "en"], "cpu", scale=30.0, max_s=30.0)
    found = model.identify(np.zeros(800, np.float32))  # shorter than 0.1 s: padded
    assert found[0][0] == "hi" and found[0][1] > 0.99


def test_server_answers_the_lid_op() -> None:
    class Engine:
        def load(self, *args: object) -> HFAudioLid:
            return HFAudioLid(_Classifier(), _extractor, ["ta", "hi", "or"], "cpu", 30.0)

    out = io.BytesIO()
    server = TorchServer(Engine(), io.BytesIO(), out)  # type: ignore[arg-type]
    server.handle("1", "load", {"kind": "k", "task": "lid", "path": "/x", "options": {}}, b"")
    server.handle("2", "lid", {"handle": "h1"}, audio_bytes(np.zeros(1600, np.float32)))
    out.seek(0)
    load, _ = read_message(out)
    assert load["ok"] and load["info"]["labels"] == ["ta", "hi", "or"] and load["info"]["max_s"] == 30.0
    reply, _ = read_message(out)
    assert reply["ok"] and reply["languages"][0][0] == "hi", json.dumps(reply)
