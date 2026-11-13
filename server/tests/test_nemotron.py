"""Nemotron-3-Diarization backend: NeMo front-end, chunk loop, speaker cache and post-processing.

The model itself is not downloaded here (``test_real.py`` runs it); a small stand-in for the ONNX step
checks the chunking, the streaming path and the plumbing into bolo's diarization pipeline.
"""

from __future__ import annotations

import fnmatch
import threading
from pathlib import Path

import numpy as np
import pytest

from bolo.audio.features import NemoLogMel, nemo_log_mel
from bolo.core import Bolo
from bolo.diarize import DiarizeOptions, activity_to_turns, find_overlaps, limit_speakers, remap_activity
from bolo.engines.base import DiarizationResult, LoadOptions, SpeakerActivity, SpeakerTurn
from bolo.engines.nemotron import (
    DEFAULT_POSTPROCESSING,
    HIDDEN,
    NUM_SPEAKERS,
    OFFLINE,
    ChunkConfig,
    NemotronDiarization,
    SpeakerCache,
)


def golden_signal() -> np.ndarray:
    n = 16000 + 37
    t = np.arange(n) / 16000
    x = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * (200 + 1500 * t) * t)).astype(np.float32)
    x[:800] = 0.0
    return x


# Rows 0, 3, 10, 50, 99 (every 16th mel band) of NemotronAsrStreamingFeatureExtractor (transformers 5.23,
# the Nemotron-3-Diarization processor config: pre-emphasis 0.97, n_fft 512, win 400, hop 160, 128 mels) for
# golden_signal(). The reference returns one more, masked, frame at the end.
GOLDEN_ROWS = {
    0: [-16.6355] * 8,
    3: [-16.6355] * 8,
    10: [-16.4527, -1.3244, -14.612, -16.6142, -16.635, -16.6355, -16.6355, -16.6355],
    50: [-16.4903, -1.2943, -15.0879, -16.2684, -0.0384, -16.624, -16.6354, -16.6355],
    99: [-16.5356, -1.2942, -15.1014, -16.6207, -16.6342, -16.4928, -16.1143, -16.6354],
}
GOLDEN_MEAN = -14.122668


def test_nemo_front_end_matches_the_reference_feature_extractor() -> None:
    features = nemo_log_mel(golden_signal())
    assert features.shape == (100, 128) and features.dtype == np.float32
    for row, values in GOLDEN_ROWS.items():
        np.testing.assert_allclose(features[row, ::16], values, atol=2e-3)
    assert abs(float(features.mean()) - GOLDEN_MEAN) < 1e-3


@pytest.mark.parametrize("piece", [1, 159, 160, 500, 4003])
def test_nemo_front_end_gives_the_same_frames_in_pieces(piece: int) -> None:
    rng = np.random.default_rng(1)
    audio = (0.1 * rng.standard_normal(16000 * 2 + 77)).astype(np.float32)
    whole = nemo_log_mel(audio)
    stream = NemoLogMel()
    parts = [stream.push(audio[i : i + piece]) for i in range(0, len(audio), piece)]
    parts.append(stream.finish())
    pieces = np.concatenate(parts)
    assert pieces.shape == whole.shape == (len(audio) // 160, 128)
    np.testing.assert_allclose(pieces, whole, atol=1e-4)


def test_nemo_front_end_on_empty_and_tiny_audio() -> None:
    assert nemo_log_mel(np.zeros(0, np.float32)).shape == (0, 128)
    assert nemo_log_mel(np.zeros(100, np.float32)).shape == (0, 128)
    assert nemo_log_mel(np.zeros(320, np.float32)).shape == (2, 128)


# -- post-processing -------------------------------------------------------------------------------------


def probs_with(spans: dict[int, list[tuple[int, int, float]]], frames: int = 1000) -> SpeakerActivity:
    p = np.zeros((frames, NUM_SPEAKERS), np.float32)
    for k, items in spans.items():
        for a, b, v in items:
            p[a:b, k] = v
    return SpeakerActivity(p, 0.01)


def test_activity_to_turns_thresholds_pads_and_numbers_speakers_by_arrival() -> None:
    # channel 3 talks first, channel 1 later and over the end of channel 3
    activity = probs_with({3: [(100, 300, 0.9)], 1: [(250, 500, 0.8)], 5: [(600, 603, 0.9)]})
    turns, kept = activity_to_turns(activity, threshold=0.5, min_duration_on=0.1)
    assert [(t.start, t.end, t.speaker) for t in turns] == [(1.0, 3.0, 0), (2.5, 5.0, 1)]
    assert kept.probs.shape == (1000, 2)
    assert kept.probs[150, 0] == pytest.approx(0.9) and kept.probs[400, 1] == pytest.approx(0.8)
    overlaps = find_overlaps(turns)
    assert len(overlaps) == 1 and overlaps[0].speakers == [0, 1]
    padded, _ = activity_to_turns(activity, threshold=0.5, pad_onset=0.2, pad_offset=0.1, min_duration_on=0.1)
    assert (padded[0].start, padded[0].end) == pytest.approx((0.8, 3.1))


def test_activity_to_turns_hysteresis_gaps_and_quiet_speakers() -> None:
    # 0.45 dips stay inside the turn with offset 0.4 but end it with offset 0.5
    activity = probs_with({0: [(100, 200, 0.9), (200, 220, 0.45), (220, 300, 0.9), (400, 450, 0.9)]})
    turns, _ = activity_to_turns(activity, threshold=0.5, offset=0.4)
    assert [(t.start, t.end) for t in turns] == [(1.0, 3.0), (4.0, 4.5)]
    turns, _ = activity_to_turns(activity, threshold=0.5)
    assert len(turns) == 3
    joined, _ = activity_to_turns(activity, threshold=0.5, offset=0.4, min_duration_off=1.0)
    assert [(t.start, t.end) for t in joined] == [(1.0, 4.5)]
    assert activity_to_turns(activity, threshold=0.5, min_speaker_s=5.0)[0] == []
    lower, _ = activity_to_turns(activity, threshold=0.4)
    assert [(t.start, t.end) for t in lower] == [(1.0, 3.0), (4.0, 4.5)]


def test_limit_speakers_and_remap_activity() -> None:
    activity = probs_with({0: [(0, 400, 0.9)], 1: [(400, 800, 0.9)], 2: [(800, 850, 0.9)], 3: [(900, 950, 0.7)]})
    activity.probs[800:850, 1] = 0.3  # the model's second guess for speaker 2 is speaker 1
    activity.probs[900:950, 0] = 0.2
    turns, act = activity_to_turns(activity)
    assert len({t.speaker for t in turns}) == 4
    limited, merged = limit_speakers(turns, act, 2)
    assert {t.speaker for t in limited} == {0, 1}
    assert [(t.start, t.end, t.speaker) for t in limited] == [(0.0, 4.0, 0), (4.0, 8.5, 1), (9.0, 9.5, 0)]
    assert merged.probs.shape == (1000, 2) and merged.probs[820, 1] == pytest.approx(0.9)
    remapped = remap_activity(act, {0: 1, 1: 0, 2: 0, 3: 1})
    assert remapped.probs[100, 1] == pytest.approx(0.9) and remapped.probs[820, 0] == pytest.approx(0.9)


# -- speaker cache ---------------------------------------------------------------------------------------


def test_speaker_cache_moves_frames_and_compresses_by_speaker() -> None:
    config = ChunkConfig(chunk_length=50, right_context=0, fifo_length=40, update_period=30, cache_length=40)
    cache = SpeakerCache(config)
    rng = np.random.default_rng(0)
    silence = np.full(HIDDEN, 7.0, np.float32)
    for step in range(6):
        n_ctx = len(cache.embeds) + len(cache.fifo)
        chunk = rng.standard_normal((50, HIDDEN)).astype(np.float32)
        chunk[:, 0] = step * 100 + np.arange(50)  # remember where a frame came from
        probs = np.zeros((n_ctx + 50, NUM_SPEAKERS), np.float32)
        probs[n_ctx : n_ctx + 25, 0] = 0.9  # first half of each chunk: speaker 0, second half: speaker 1
        probs[n_ctx + 25 :, 1] = 0.9
        probs[:n_ctx, 0] = 0.9
        cache.update(chunk, probs, silence)
        assert len(cache.fifo) <= config.fifo_length
        assert len(cache.embeds) <= config.cache_length
    assert cache.compressed and len(cache.embeds) == config.cache_length
    # frames grouped by speaker: speaker 0's block first, each block in time order, a silence slot per speaker
    origin = cache.embeds[:, 0]
    is_silence = np.all(cache.embeds == 7.0, axis=1)
    assert is_silence.sum() >= 1
    speakers = np.argmax(cache.probs, axis=1)
    real = ~is_silence
    assert np.all(np.diff(speakers[real]) >= 0)
    for k in (0, 1):
        sel = real & (speakers == k) & (cache.probs.max(axis=1) > 0.5)
        assert np.all(np.diff(origin[sel]) > 0)


# -- the chunk loop, with a stand-in for the ONNX step -----------------------------------------------------


class FakeStep:
    """Speaker 0 where the chunk's log-mel energy is high, speaker 1 where it is middling; records the calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def __call__(
        self, features: np.ndarray, cached: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_enc = -(-len(features) // 8)
        assert len(mask) == len(cached) + n_enc
        self.calls.append((len(features), len(cached), int(mask.sum())))
        energy = np.concatenate([features.mean(axis=1), np.full(n_enc * 8 - len(features), -20.0)])
        logits = np.full(((len(cached) + n_enc) * 8, NUM_SPEAKERS), -8.0, np.float32)
        own = logits[len(cached) * 8 :]
        own[energy > -3.0, 0] = 8.0
        own[(energy > -9.0) & (energy <= -3.0), 1] = 8.0
        stacked = np.zeros((n_enc * 8, features.shape[1]), np.float32)
        stacked[: len(features)] = features
        embeds = stacked.reshape(n_enc, -1)[:, :HIDDEN]
        return logits, np.ascontiguousarray(embeds), np.zeros(HIDDEN, np.float32)


def fake_model(spec: dict[str, object] | None = None) -> tuple[NemotronDiarization, FakeStep]:
    model = NemotronDiarization.__new__(NemotronDiarization)
    step = FakeStep()
    model._run = step  # type: ignore[assignment]
    model._lock = threading.Lock()
    model.post = {**DEFAULT_POSTPROCESSING, **{k: float(v) for k, v in (spec or {}).items()}}  # type: ignore[arg-type]
    model.min_speaker_s = 0.0
    model._embedding_file = None
    model._extractor = None
    model._threads = 1
    return model, step


def two_voices(seconds: float = 70.0) -> tuple[np.ndarray, list[tuple[float, float, int]]]:
    """Loud and quiet noise bursts: 'speaker' 0 and 1 by level, with silence between."""
    rng = np.random.default_rng(3)
    audio = np.zeros(int(seconds * 16000), np.float32)
    truth = []
    t, k = 0.5, 0
    while t + 2.0 < seconds:
        a, b = int(t * 16000), int((t + 1.5) * 16000)
        audio[a:b] = (0.5 if k == 0 else 0.02) * rng.standard_normal(b - a)
        truth.append((t, t + 1.5, k))
        t += 2.0
        k = 1 - k
    return audio, truth


def test_offline_loop_chunks_long_audio_and_finds_both_voices() -> None:
    model, step = fake_model({"min_duration_on": 0.1})  # the stand-in flickers at the edges of a burst
    audio, truth = two_voices(70.0)  # longer than one 27.2 s offline chunk
    probs = model.speaker_probabilities(audio)
    assert probs.shape == (len(audio) // 160 + 1, NUM_SPEAKERS)
    assert len(step.calls) == 3  # 0-27.2 s, 27.2-54.4 s, 54.4-70 s
    assert step.calls[0][0] == (OFFLINE.chunk_length + OFFLINE.right_context) * 8
    assert step.calls[0][1] == 0 and step.calls[1][1] > 0  # later chunks see the speaker cache and FIFO
    result = model.diarize_detailed(audio)
    assert isinstance(result, DiarizationResult) and result.activity is not None
    assert len({t.speaker for t in result.turns}) == 2
    first = result.turns[0]
    assert first.speaker == 0 and abs(first.start - truth[0][0]) < 0.05 and abs(first.end - truth[0][1]) < 0.05
    assert result.activity.probs.shape[1] == 2
    # num_speakers folds the quieter voice into the other one
    assert {t.speaker for t in model.diarize(audio, num_speakers=1)} == {0}


@pytest.mark.parametrize("seconds", [0.0, 0.05, 0.3, 2.71])
def test_offline_loop_on_short_audio(seconds: float) -> None:
    model, _ = fake_model()
    audio = np.zeros(int(seconds * 16000), np.float32)
    probs = model.speaker_probabilities(audio)
    assert probs.shape == (len(audio) // 160 + 1, NUM_SPEAKERS)
    assert model.diarize(audio) == []


def test_streaming_gives_every_frame_once_and_agrees_with_the_turns() -> None:
    model, step = fake_model()
    audio, _ = two_voices(12.0)
    stream = model.stream("low_latency")
    assert stream.latency_s == pytest.approx(1.04)
    out = [stream.accept(audio[i : i + 1234]) for i in range(0, len(audio), 1234)]
    out.append(stream.finish())
    probs = np.concatenate(out)
    assert probs.shape == (len(audio) // 160, NUM_SPEAKERS)
    assert stream.frames_out == len(probs)
    assert all(call[0] == 13 * 8 for call in step.calls[:-1])  # 9 chunk + 4 look-ahead encoder frames
    turns, _ = activity_to_turns(SpeakerActivity(probs, 0.01), min_duration_on=0.1)
    assert len({t.speaker for t in turns}) == 2
    with pytest.raises(ValueError):
        model.stream("instant")


def test_threshold_from_the_catalog_and_per_request() -> None:
    model, _ = fake_model({"threshold": 0.9999})  # the stand-in's probabilities are sigmoid(8) = 0.99966
    audio, _ = two_voices(8.0)
    assert model.diarize(audio) == []
    assert model.diarize(audio, threshold=0.5) != []


def test_missing_onnx_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        NemotronDiarization(tmp_path, {}, LoadOptions())


def test_catalog_entry_states_licence_and_source() -> None:
    from bolo.catalog import Catalog

    spec = Catalog.load().resolve("nemotron-3-diarization")
    assert spec is not None and spec.task == "diarization" and spec.engine == "onnxruntime"
    assert spec.license == "OpenMDW-1.1" and spec.commercial_use
    assert spec.source.repo == "onnx-community/Nemotron-3-Diarization-ONNX"
    assert len(spec.source.revision) == 40
    assert "onnx/model_quantized.onnx_data" in spec.source.include


# -- plumbing into bolo's pipeline ------------------------------------------------------------------------


async def test_core_passes_frame_probabilities_through_and_renumbers_them(core: Bolo) -> None:
    from conftest import two_speakers

    audio, rate = two_speakers(), 16000
    probs = np.zeros((len(audio) // 160 + 1, 3), np.float32)
    probs[:, 2] = 0.25  # column 2 is the turn that comes first
    probs[:, 0] = 0.75

    def detailed(samples: np.ndarray, num: int | None = None, thr: float | None = None) -> DiarizationResult:
        return DiarizationResult(
            [SpeakerTurn(0.1, 1.0, 2), SpeakerTurn(1.2, 2.0, 0)], SpeakerActivity(probs.copy(), 0.01)
        )

    async with core.manager.use("fake-diarization") as entry:
        entry.model.diarize_detailed = detailed  # type: ignore[method-assign]
        opts = DiarizeOptions(model="fake-diarization", consolidate=False)
        turns = await core.diarize(audio, rate, opts)
    assert [t.speaker for t in turns] == [0, 1]
    assert opts.activity is not None and opts.activity.probs.shape == (len(probs), 2)
    assert opts.activity.probs[5, 0] == pytest.approx(0.25) and opts.activity.probs[5, 1] == pytest.approx(0.75)
    assert opts.label(0) == "A"


def test_hugging_face_and_local_directories_are_recognised() -> None:
    from bolo.custom import choose_files, detect_kind
    from bolo.store import RemoteFile

    paths = ["config.json", "onnx/model.onnx", "onnx/model.onnx_data", "onnx/model_quantized.onnx"]
    assert detect_kind("onnx-community/Nemotron-3-Diarization-ONNX", paths) == "nemotron_diarization"
    assert detect_kind("someone/other-onnx-model", paths) is None
    files = [RemoteFile(p, "") for p in [*paths, "onnx/model_quantized.onnx_data", "onnx/model_q4.onnx"]]
    exclude = choose_files("nemotron_diarization", files)
    kept = [f.path for f in files if not any(fnmatch.fnmatch(f.path, pattern) for pattern in exclude)]
    assert kept == ["config.json", "onnx/model_quantized.onnx", "onnx/model_quantized.onnx_data"]
