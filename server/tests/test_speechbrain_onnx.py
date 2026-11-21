"""SpeechBrain SepFormer and MetricGAN+ in ONNX Runtime (bolo.engines.speechbrain_onnx), with stand-in sessions:
lengths, levels, windows, the STFT round trip, and the catalog entries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from bolo.catalog import Catalog
from bolo.engines import speechbrain_onnx as sb
from bolo.engines.base import LoadOptions


class Sepformer:
    """A SepFormer stand-in: ``sources`` = scaled copies of the mixture, as long as the decoder would give."""

    def __init__(self, gains: list[float]) -> None:
        self.gains = gains
        self.lengths: list[int] = []

    def run(self, _: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        x = feeds["mixture"][0]
        self.lengths.append(len(x))
        assert (len(x) - sb.KERNEL) % sb.STRIDE == 0  # whole strides only, as SpeechBrain's encoder takes them
        return [np.stack([g * x for g in self.gains], axis=-1)[None]]


class Mask:
    def __init__(self, value: float) -> None:
        self.value = value

    def run(self, _: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [np.full_like(feeds["features"], self.value)]


def _tone(n: int, f: float = 300.0, rate: int = 16000) -> np.ndarray:
    return (0.3 * np.sin(2 * np.pi * f * np.arange(n) / rate)).astype(np.float32)


def test_sepformer_takes_whole_strides_and_pads_back() -> None:
    session = Sepformer([1.0, 2.0])
    out = sb.run_sepformer(session, _tone(4003))
    assert session.lengths == [16 + (4003 - 16) // 8 * 8] and out.shape == (4003, 2)
    assert np.all(out[session.lengths[0] :] == 0)  # SpeechBrain pads the tracks with zeros
    assert sb.run_sepformer(Sepformer([1.0]), _tone(5)).shape == (5, 1)  # shorter than the kernel


def _separator(gains: list[float], rate: int, speakers: int) -> sb.SepFormerSeparation:
    model = sb.SepFormerSeparation.__new__(sb.SepFormerSeparation)
    model._session, model.model_rate, model.num_speakers = Sepformer(gains), rate, speakers
    import threading

    model._lock = threading.Lock()
    return model


@pytest.mark.parametrize("rate", [8000, 16000])
def test_separated_tracks_come_back_at_16k_and_at_the_mixture_level(rate: int) -> None:
    x = _tone(16000)
    tracks = _separator([5.0, -3.0, 0.5], rate, 3).separate(x)
    assert len(tracks) == 3 and all(len(t) == len(x) and t.dtype == np.float32 for t in tracks)
    # SepFormer's output level is arbitrary: each track is fitted to the mixture (a negative copy is turned over)
    for t in tracks:
        assert np.allclose(t[2000:-2000], x[2000:-2000], atol=0.02)


def test_denoiser_windows_are_cross_faded() -> None:
    model = sb.SepFormerDenoiser.__new__(sb.SepFormerDenoiser)
    import threading

    model._session, model.sample_rate, model._lock = Sepformer([7.0]), 16000, threading.Lock()
    model.window, model.overlap = 16000, 4000
    x = _tone(50000)
    y = model.denoise(x)
    assert len(y) == len(x) and np.allclose(y[:49990], x[:49990], atol=1e-3)
    assert len(model._session.lengths) == 4  # 1 s windows with a quarter second of overlap
    assert model.denoise(np.zeros(0, np.float32)).size == 0


def test_stft_round_trip_and_masks() -> None:
    x = _tone(16001, 440.0) + _tone(16001, 1250.0) * 0.5
    assert sb.stft_features(x).shape == (1 + len(x) // sb.HOP, 257)
    same = sb.metricgan_enhance(Mask(1.0), x)  # mask 1 on the log magnitude: the input back
    assert same.shape == x.shape and np.max(np.abs(same - x)) < 1e-4
    quieter = sb.metricgan_enhance(Mask(0.5), x)
    assert float(np.sqrt(np.mean(quieter**2))) < 0.5 * float(np.sqrt(np.mean(x**2)))


def test_catalog_entries() -> None:
    catalog = Catalog.load(None)
    for name, speakers in (("sepformer-wsj02mix", 2), ("sepformer-wsj03mix", 3)):
        spec = catalog.resolve(name)
        assert spec is not None and spec.engine == "onnxruntime" and spec.kind == "sepformer_onnx"
        assert spec.task == "separation" and spec.options["speakers"] == speakers
        assert spec.license == "Apache-2.0" and "LDC" in spec.description  # WSJ0 is LDC-licensed data
        assert spec.source.repo == f"codebucketsolutions/{name}-onnx" and len(spec.source.revision) == 40


def test_engine_loads_by_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from bolo.engines.onnx_turn import OnnxRuntimeEngine

    monkeypatch.setattr(sb, "_session", lambda *a: Sepformer([1.0]))
    opts = LoadOptions(device="cpu", threads=1)
    engine = OnnxRuntimeEngine()
    sep = engine.load("sepformer_onnx", "separation", tmp_path, {"model_sample_rate": 8000, "speakers": 3}, opts)
    den = engine.load("sepformer_onnx", "denoise", tmp_path, {"model_sample_rate": 16000}, opts)
    gan = engine.load("metricgan_onnx", "denoise", tmp_path, {}, opts)
    assert isinstance(sep, sb.SepFormerSeparation) and sep.mode == "blind" and sep.num_speakers == 3
    assert isinstance(den, sb.SepFormerDenoiser) and den.sample_rate == 16000
    assert isinstance(gan, sb.MetricGanPlus) and gan.sample_rate == 16000


def test_ctc_greedy_and_pieces() -> None:
    logits = np.full((7, 5), -9.0, np.float32)
    for t, i in enumerate([3, 3, 0, 3, 4, 0, 0]):
        logits[t, i] = 0.0
    assert sb.ctc_greedy(logits) == [3, 3, 4]  # a blank between repeats keeps both
    pieces = ["<unk>", "<s>", "</s>", "▁a", "b"]
    assert sb.detokenize([3, 3, 4], pieces) == "a ab"
    assert sb.detokenize([1, 3, 0, 4, 2], pieces) == "a ⁇ b"  # control pieces drop, unknown is shown


def test_ctc_recogniser(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from bolo.engines.base import STTRequest

    (tmp_path / "tokens.txt").write_text("<unk> 0\n<s> 1\n</s> 2\n▁mu 3\nraho 4\n", encoding="utf-8")

    class Graph:
        def run(self, _: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert feeds["wav"].shape[1] >= 400
            out = np.full((1, 4, 5), -9.0, np.float32)
            for t, i in enumerate([3, 0, 4, 4]):
                out[0, t, i] = 0.0
            return [out]

    monkeypatch.setattr(sb, "_session", lambda *a: Graph())
    model = sb.SpeechBrainCtc(tmp_path, {"language": "rw"}, LoadOptions(device="cpu", threads=1))
    r = model.transcribe([STTRequest(np.zeros(100, np.float32)), STTRequest(np.zeros(0, np.float32))])
    assert r[0].text == "muraho" and r[0].language == "rw" and r[1].text == ""


def test_recogniser_and_denoiser_entries() -> None:
    catalog = Catalog.load(None)
    langs = {"wav2vec2-dvoice-fongbe": "fon", "wav2vec2-dvoice-wolof": "wo", "wav2vec2-dvoice-darija": "ary",
             "wav2vec2-dvoice-amharic": "am", "wav2vec2-commonvoice-14-rw": "rw"}  # fmt: skip
    for name, lang in langs.items():
        spec = catalog.resolve(name)
        assert spec is not None and spec.task == "stt" and spec.kind == "speechbrain_ctc" and spec.languages == [lang]
        assert spec.options["language"] == lang and spec.license == "Apache-2.0" and spec.validated == "cpu"
        assert set(spec.source.include) == {"asr.onnx", "tokens.txt", "speechbrain_onnx.json"}
    for name, kind in (("sepformer-dns4-16k", "sepformer_onnx"), ("metricgan-plus", "metricgan_onnx")):
        spec = catalog.resolve(name)
        assert spec is not None and spec.task == "denoise" and spec.kind == kind and spec.sample_rate == 16000
