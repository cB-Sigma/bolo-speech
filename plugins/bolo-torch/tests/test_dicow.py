"""DiCoW (diarization-conditioned Whisper) with a fake network: STNO masks, per-speaker decoding, the plugin op."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from bolo.catalog import Catalog
from bolo.engines.base import LoadOptions, SpeakerActivity, SpeakerRequest, STTModel, STTRequest
from bolo.plugin.host import PluginConfig, PluginEngine

from bolo_torch.engine import LOADERS
from bolo_torch.models import dicow
from bolo_torch.runtime import InvalidRequest, LoadContext, MissingDependency
from fakes import FakeDiCoWBackend

FAKE_PLUGIN = [sys.executable, str(Path(__file__).parent / "fake_plugin.py")]


def reference_stno(diar_mask: torch.Tensor, s_index: int) -> torch.Tensor:
    """``DiCoWPipeline.get_stno_mask`` from BUTSpeechFIT/DiCoW ``pipeline.py`` (Apache-2.0), verbatim."""
    non_target_mask = torch.ones((diar_mask.shape[0],), dtype=torch.bool)
    non_target_mask[s_index] = False
    sil_frames = (1 - diar_mask).prod(axis=0)
    anyone_else = (1 - diar_mask[non_target_mask]).prod(axis=0)
    target_spk = diar_mask[s_index] * anyone_else
    non_target_spk = (1 - diar_mask[s_index]) * (1 - anyone_else)
    overlapping_speech = diar_mask[s_index] - target_spk
    return torch.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], axis=0)


def reference_diarization_mask(per_speaker: list[list[tuple[float, float]]], length: int) -> torch.Tensor:
    """``DiCoWPipeline.get_diarization_mask`` (same source): turns -> 0/1 at 50 Hz."""
    mask = torch.zeros(len(per_speaker), length)
    for i, spans in enumerate(per_speaker):
        for start, end in spans:
            mask[i, round(start * 50) : round(end * 50)] = 1
    return mask


# -- masks --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("speakers", [1, 2, 3, 5])
def test_stno_masks_match_the_reference_formula(speakers: int) -> None:
    rng = np.random.default_rng(speakers)
    hard = (rng.random((speakers, 400)) > 0.6).astype(np.float32)
    soft = rng.random((speakers, 400)).astype(np.float32)
    for activity in (hard, soft):
        ours = dicow.stno_masks(activity)
        ref = torch.stack([reference_stno(torch.from_numpy(activity).double(), s) for s in range(speakers)])
        assert ours.shape == (speakers, 4, 400)
        np.testing.assert_allclose(ours, ref.numpy(), atol=1e-6)
        np.testing.assert_allclose(ours.sum(axis=1), 1.0, atol=1e-6)  # the four classes split every frame
    one_hot = dicow.stno_masks(hard)
    assert set(np.unique(one_hot)) <= {0.0, 1.0} and (one_hot.sum(axis=1) == 1).all()


def test_turns_on_a_20ms_grid_give_the_reference_masks_exactly() -> None:
    from bolo.diarize import conditioning_activity
    from bolo.engines.base import SpeakerTurn

    # 10 ms diarizer times put many edges on a .5 tie at 50 Hz (1.23 * 50 = 61.5): rounded as the reference does
    turns = [
        SpeakerTurn(0.013, 1.23, 0),
        SpeakerTurn(1.01, 4.47, 1),
        SpeakerTurn(2.07, 3.011, 0),
        SpeakerTurn(4.49, 4.83, 0),
    ]
    activity, kind = conditioning_activity(turns, None, 5.0, "turns")
    assert kind == "turns"
    ours = dicow.stno_masks(dicow.activity_at_mask_rate(activity.probs, activity.frame_s, 1500))
    per_speaker = [[(t.start, t.end) for t in turns if t.speaker == k] for k in range(2)]
    diar = reference_diarization_mask(per_speaker, 1500)
    ref = torch.stack([reference_stno(diar, k) for k in range(2)])
    np.testing.assert_array_equal(ours, ref.numpy())


def test_activity_resampling_and_enrolment_window() -> None:
    probs = np.zeros((300, 2), np.float32)  # 10 ms frames, 3 s
    probs[100:200, 0] = 1.0
    probs[150:300, 1] = 0.5
    at50 = dicow.activity_at_mask_rate(probs, 0.01, 200)  # 4 s of mask frames: the last second is silent
    assert at50.shape == (2, 200)
    assert at50[0, 50:100].all() and not at50[0, :50].any() and not at50[0, 100:].any()
    assert np.allclose(at50[1, 75:150], 0.5) and not at50[1, 150:].any()
    with pytest.raises(InvalidRequest):
        dicow.activity_at_mask_rate(np.zeros(5, np.float32), 0.01, 10)
    target = np.zeros(4000, np.float32)
    target[2500:3200] = 1
    start = dicow.enrolment_window(target)
    assert start <= 2500 and start + 1500 >= 3200
    assert dicow.enrolment_window(np.ones(100, np.float32)) == 0


# -- the model wrapper ----------------------------------------------------------------------------------


def test_one_result_per_speaker_with_absolute_times() -> None:
    backend = FakeDiCoWBackend()
    model = dicow.DiCoW(backend)
    assert model.speaker_conditioned and model.sample_rate == 16000 and "hi" in model.languages
    probs = np.zeros((500, 3), np.float32)  # 5 s at 10 ms; speaker 2 says (almost) nothing
    probs[0:200, 0] = 1.0
    probs[150:400, 1] = 1.0
    probs[450:455, 2] = 1.0
    audio = np.zeros(5 * 16000, np.float32)
    results = model.transcribe_speakers(SpeakerRequest(audio, SpeakerActivity(probs), "en", "transcribe"))
    assert len(results) == 3
    masks, language, task, starts = backend.calls[0]
    assert masks.shape == (2, 4, 1500) and (language, task, starts) == ("en", "transcribe", None)  # 2 is skipped
    # target alone vs overlap, seen from speaker 0: 0-1.5 s alone, 1.5-2 s overlapped with speaker 1
    assert masks[0, 1, :75].all() and masks[0, 3, 75:100].all() and masks[0, 2, 100:200].all()
    assert masks[0, 0, 250:].all()  # padding to 30 s is silence
    assert [(s.start, s.end) for s in results[0].segments] == [(0.0, 2.0)]
    assert [(s.start, s.end) for s in results[1].segments] == [(1.5, 4.0)]
    assert results[1].text == "spk1 1.50-4.00" and results[2].text == "" and results[2].segments == []
    assert results[0].language == "en"


def test_plain_transcription_treats_the_input_as_one_target_speaker() -> None:
    backend = FakeDiCoWBackend()
    model = dicow.DiCoW(backend)
    out = model.transcribe([STTRequest(np.zeros(16000 * 2, np.float32)), STTRequest(np.zeros(8000, np.float32))])
    assert len(out) == 2 and len(backend.calls) == 2
    masks = backend.calls[0][0]
    assert masks.shape[0] == 1 and masks[0, 1, :100].all() and masks[0, 0, 100:].all()
    assert out[0].segments[0].start == 0.0 and out[0].segments[0].end == pytest.approx(2.0)


def test_enrolment_windows_are_passed_when_the_model_uses_them() -> None:
    backend = FakeDiCoWBackend(use_enrollments=True)
    model = dicow.DiCoW(backend)
    probs = np.zeros((6000, 2), np.float32)  # 60 s
    probs[4000:5000, 0] = 1.0
    probs[:1000, 1] = 1.0
    model.transcribe_speakers(SpeakerRequest(np.zeros(60 * 16000, np.float32), SpeakerActivity(probs)))
    starts = backend.calls[0][3]
    assert starts is not None and starts[1] == 0 and starts[0] <= 2000 and starts[0] + 1500 >= 2500


def test_segments_are_clamped_and_empty_text_dropped() -> None:
    result = dicow._result(
        [
            dicow.DecodedSegment(-0.5, 1.0, " hi "),
            dicow.DecodedSegment(2.0, 99.0, "there"),
            dicow.DecodedSegment(3, 4, " "),
        ],
        "en",
        5.0,
    )
    assert [(s.start, s.end, s.text) for s in result.segments] == [(0.0, 1.0, "hi"), (2.0, 5.0, "there")]
    assert result.text == "hi there"
    assert dicow.parse_timestamped("<|0.00|> Hello.<|1.20|><|1.40|> Again<|2.00|>") == [
        dicow.DecodedSegment(0.0, 1.2, " Hello."),
        dicow.DecodedSegment(1.4, 2.0, " Again"),
    ]


def test_lower_upper_mapping_and_bf16_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Tok:
        def get_vocab(self) -> dict[str, int]:
            return {"Ġhello": 1, "ĠHello": 2, "World": 3, "world": 4, "x": 5, "": 6}

    assert dicow.lower_upper_mapping(Tok()) == {1: 2, 4: 3}
    monkeypatch.setattr(torch._C, "_cpu", object(), raising=False)  # no torch probes: the CPU flags decide
    flags = tmp_path / "cpuinfo"
    flags.write_text("processor : 0\nflags : fpu sse avx512f avx512_bf16\n")
    assert dicow.cpu_has_fast_bf16(str(flags))
    flags.write_text("processor : 0\nflags : fpu sse amx_bf16 amx_tile\n")
    assert dicow.cpu_has_fast_bf16(str(flags))
    flags.write_text("processor : 0\nflags : fpu sse avx512f\n")
    assert not dicow.cpu_has_fast_bf16(str(flags))
    assert not dicow.cpu_has_fast_bf16(str(tmp_path / "missing"))


def test_model_code_is_imported_as_a_package_without_writing_bytecode(tmp_path: Path) -> None:
    root = tmp_path / "dicow-v3.3"  # a dot in the name, like bolo's model directory
    root.mkdir()
    (root / "modeling_dicow.py").write_text(
        "from .generation import VALUE\nclass DiCoWForConditionalGeneration:\n    value = VALUE\n"
    )
    (root / "generation.py").write_text("from .decoding import VALUE\n")  # nested: what transformers 4.55 misses
    (root / "decoding.py").write_text("VALUE = 42\n")
    module = dicow.import_model_code(root)
    assert module.DiCoWForConditionalGeneration.value == 42
    assert not (root / "__pycache__").exists()
    assert dicow.import_model_code(root) is module  # cached by content hash
    (tmp_path / "empty").mkdir()
    with pytest.raises(InvalidRequest, match="modeling_dicow"):
        dicow.import_model_code(tmp_path / "empty")


def test_loader_explains_the_missing_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)  # as if it were not installed
    with pytest.raises(MissingDependency, match=r"bolo-torch\[dicow\]"):
        dicow.load(LoadContext("dicow", "stt", tmp_path, {}))


# -- through bolo's plugin host --------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine() -> Iterator[PluginEngine]:
    eng = PluginEngine(PluginConfig("torch", FAKE_PLUGIN, idle_s=-1))
    yield eng
    eng.shutdown()


def test_speakers_op_through_the_host(engine: PluginEngine, tmp_path: Path) -> None:
    model = engine.load("dicow", "stt", tmp_path, {}, LoadOptions(threads=1))
    assert isinstance(model, STTModel) and model.speaker_conditioned and model.supports_translate
    probs = np.zeros((150, 2), np.float32)  # 3 s at 20 ms
    probs[:60, 0] = 1.0
    probs[50:150, 1] = 1.0
    results = model.transcribe_speakers(
        SpeakerRequest(np.zeros(3 * 16000, np.float32), SpeakerActivity(probs, 0.02), "en")
    )
    assert [r.text for r in results] == ["spk0 0.00-1.20", "spk1 1.00-3.00"]
    assert results[1].segments[0].start == 1.0 and results[1].segments[0].end == 3.0
    plain = model.transcribe([STTRequest(np.zeros(16000, np.float32))])
    assert plain[0].text == "spk0 0.00-1.00"
    model.close()
    assert engine.process.crashes == 0


def test_catalog_entry() -> None:
    spec = Catalog.load().resolve("dicow")
    assert spec is not None and spec.id == "dicow-v3.3" and spec.kind in LOADERS and spec.task == "stt"
    assert spec.license == "CC-BY-4.0" and "speaker-conditioned" in spec.features
    assert spec.options.get("diarization_model") == "nemotron-3-diarization"
    assert spec.source.repo == "BUT-FIT/DiCoW_v3_3" and len(spec.source.revision) == 40


def test_core_onnx_engine_builds_the_same_masks() -> None:
    """bolo's torch-free DiCoW (bolo.engines.dicow_onnx) conditions the model exactly as this plugin does."""
    onnx_engine = pytest.importorskip("bolo.engines.dicow_onnx")
    rng = np.random.default_rng(7)
    for speakers in (1, 2, 4):
        probs = (rng.random((900, speakers)) > 0.5).astype(np.float32) * rng.random((900, speakers)).astype(np.float32)
        for frame_s, n in ((0.01, 1500), (0.02, 2250), (0.08, 700)):
            ours = dicow.stno_masks(dicow.activity_at_mask_rate(probs, frame_s, n))
            core = onnx_engine.stno_masks(onnx_engine.activity_at_mask_rate(probs, frame_s, n))
            np.testing.assert_array_equal(ours, core)
    assert onnx_engine.WHISPER_LANGUAGES == dicow.WHISPER_LANGUAGES
