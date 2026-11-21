"""Reverb ASR from a TorchScript file (kind reverb-asr), with a tiny scripted model that has WeNet's export
interface: loading, the encoder call, greedy CTC and the BPE units; and the catalog entries of Rev's models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from bolo.catalog import Catalog
from bolo.engines.base import STTRequest

from bolo_torch.engine import LOADERS, TorchEngine
from bolo_torch.models import reverb_asr as rv
from bolo_torch.runtime import InvalidRequest, LoadContext

torch = pytest.importorskip("torch")

UNITS = "<blank> 0\n<unk> 1\n▁hello 2\n▁wor 3\nld 4\n. 5\n<sos/eos> 6\n"
PATH = [0, 2, 2, 0, 3, 4, 4, 0, 5, 6]  # "▁hello ▁wor ld ." with repeats, blanks and an end symbol


class TinyReverb(torch.nn.Module):  # type: ignore[misc,name-defined]
    """Encoder output: one frame per twenty feature frames; CTC: a fixed best path."""

    def __init__(self) -> None:
        super().__init__()
        self.path = torch.tensor(PATH)
        self.last_cat = torch.zeros(2)

    @torch.jit.export  # type: ignore[misc]
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor,
        cnn_cache: torch.Tensor,
        cat_embs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.last_cat = cat_embs
        frames = xs.size(1) // 20
        return torch.zeros(1, frames, 8), att_cache, cnn_cache

    @torch.jit.export  # type: ignore[misc]
    def ctc_activation(self, xs: torch.Tensor) -> torch.Tensor:
        n = xs.size(1)
        idx = self.path[torch.arange(n) % self.path.numel()]
        return torch.nn.functional.one_hot(idx, 7).float().unsqueeze(0).log_softmax(-1)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return xs


def _model_dir(tmp_path: Path) -> Path:
    torch.jit.script(TinyReverb()).save(str(tmp_path / rv.JIT_FILE))
    (tmp_path / rv.UNITS_FILE).write_text(UNITS, encoding="utf-8")
    (tmp_path / "config.yaml").write_text("dataset_conf:\n  fbank_conf:\n    num_mel_bins: 80\n", encoding="utf-8")
    return tmp_path


def test_pieces_and_greedy_ctc(tmp_path: Path) -> None:
    (tmp_path / "u.txt").write_text(UNITS, encoding="utf-8")
    units = rv.read_units(tmp_path / "u.txt")
    assert units[2] == "▁hello" and units[6] == "<sos/eos>"
    log_probs = np.eye(7)[PATH]
    assert rv.ctc_greedy(log_probs) == [2, 3, 4, 5, 6]
    assert rv.pieces_to_text([2, 3, 4, 5, 6], units) == "hello world."
    (tmp_path / "config.yaml").write_text("dataset_conf:\n  fbank_conf:\n    num_mel_bins: 64\n    frame_shift: 10\n")
    assert rv.read_fbank_conf(tmp_path)["num_mel_bins"] == 64


def test_load_and_transcribe_a_scripted_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def fake_features(conf: dict[str, float]) -> Any:
        assert conf["num_mel_bins"] == 80

        def features(audio: np.ndarray) -> Any:
            seen.append(len(audio))
            return torch.zeros(1, len(audio) // 160, 80)

        return features

    monkeypatch.setattr(rv, "kaldi_features", fake_features)
    root = _model_dir(tmp_path)
    model = rv.load(LoadContext("reverb-asr", "stt", root, {"verbatimicity": 0.25}))
    (r,) = model.transcribe([STTRequest(np.zeros(16000 * 2, np.float32))])  # 200 feature frames -> 10 encoder frames
    assert (r.text, r.language) == ("hello world.", "en") and seen == [32000]
    assert r.segments[0].end == 2.0
    assert model.module.last_cat.tolist() == [0.25, 0.75]
    (quiet,) = model.transcribe([STTRequest(np.zeros(100, np.float32))])
    assert quiet.text == ""
    with pytest.raises(InvalidRequest, match="English only"):
        model.transcribe([STTRequest(np.zeros(16000, np.float32), language="de")])
    with pytest.raises(InvalidRequest, match="verbatimicity"):
        rv.ReverbAsr(model.module, {}, lambda a: a, verbatimicity=2.0)


def test_kaldi_features_shape() -> None:
    pytest.importorskip("torchaudio")
    feats = rv.kaldi_features({"num_mel_bins": 80, "frame_length": 25, "frame_shift": 10})(
        np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    )
    assert tuple(feats.shape) == (1, 98, 80)


def test_kind_and_catalog_entries() -> None:
    assert LOADERS["reverb-asr"] == ("bolo_torch.models.reverb_asr", "load")
    assert "reverb-asr" in TorchEngine().kinds
    catalog = Catalog.load()
    asr = catalog.resolve("reverb-asr")
    assert asr is not None and asr.kind == "reverb-asr" and asr.engine == "plugin:torch" and asr.languages == ["en"]
    assert asr.gated and not asr.commercial_use and asr.validated == "cpu" and "not_yet_run" not in asr.options
    jit = [u for u in asr.source.urls if u["path"] == rv.JIT_FILE]
    assert jit and len(jit[0]["sha256"]) == 64 and asr.source.revision in jit[0]["url"]
    v2 = catalog.resolve("reverb-diarization-v2-torch")
    assert v2 is not None and v2.kind == "pyannote-diarization" and v2.gated and not v2.commercial_use
    assert "not_yet_run" not in v2.options and v2.options["repo"] == "Revai/reverb-diarization-v2"
