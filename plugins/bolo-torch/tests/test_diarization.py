"""pyannote speaker diarization pipelines (3.1, community-1) with a fake pipeline, and the configuration guard
in front of pyannote's ``Pipeline.from_pretrained``."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bolo_torch.models import diarization
from bolo_torch.models.diarization import PyannoteDiarization, annotation_turns
from bolo_torch.pyannote_config import UnsafeConfig, check_config, safe_config
from bolo_torch.runtime import LoadContext
from fakes import FakeAnnotation, FakePipeline, fake_embedding, tone

# the configurations of the pinned revisions (catalog: pyannote-speaker-diarization-3.1 at 84fd259,
# pyannote-speaker-diarization-community-1 at 3533c8c, pixit-ami at 9486b10), byte for byte
CONFIG_31 = """version: 3.1.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: pyannote/wespeaker-voxceleb-resnet34-LM
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: pyannote/segmentation-3.0
    segmentation_batch_size: 32

params:
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
  segmentation:
    min_duration_off: 0.0
"""
CONFIG_COMMUNITY_1 = """dependencies: \n  pyannote.audio: 4.0.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: VBxClustering
    segmentation: $model/segmentation
    segmentation_batch_size: 32
    embedding: $model/embedding
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    plda: $model/plda
    \nparams:
  clustering:
    threshold: 0.6
    Fa: 0.07
    Fb: 0.8
  segmentation:
    min_duration_off: 0.0
"""
CONFIG_PIXIT = """version: 3.3.0

pipeline:
  name: pyannote.audio.pipelines.SpeechSeparation
  params:
    segmentation: pyannote/separation-ami-1.0
    segmentation_step: 0.1
    embedding: speechbrain/spkrec-ecapa-voxceleb@5c0be3875fda05e81f3c004ed8c7c06be308de1e
    embedding_exclude_overlap: False
    clustering: AgglomerativeClustering
    embedding_batch_size: 8
    segmentation_batch_size: 8

params:
  segmentation:
    min_duration_off: 0.0
    threshold: 0.82
  clustering:
    method: centroid
    min_cluster_size: 15
    threshold: 0.68
  separation:
    leakage_removal: true
    asr_collar: 0.32
"""
SHA_31 = "04ad9cd59a93c3a7c754200ecc9e1c4ba87bf1657ef8a4debf7555e711daeeda"
SHA_COMMUNITY_1 = "5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be"
SHA_PIXIT = "141e143ae24f691e4e9d0b8c4dc5f57e25a9d4333995d8c57c50095c50dcfa63"
PIXIT_REV = "4d38e95cfd067c894b8b60b00761831fb01e4a8c"


def model_folder(tmp_path: Path, config: str, subfolders: tuple[str, ...] = ()) -> Path:
    root = tmp_path / "model"
    root.mkdir(parents=True)
    (root / "config.yaml").write_text(config, encoding="utf-8")
    for sub in subfolders:
        (root / sub).mkdir(parents=True)
        (root / sub / "pytorch_model.bin").write_bytes(b"weights")
    return root


# -- the model -------------------------------------------------------------------------------------------


def test_turns_in_bolo_format_and_speaker_options() -> None:
    pipeline = FakePipeline([(2.0, 3.0, "SPEAKER_01"), (0.5, 1.5, "SPEAKER_00"), (1.4, 2.1, "SPEAKER_01")])
    model = PyannoteDiarization(pipeline, fake_embedding, {"min_speakers": 2, "max_speakers": 4})
    assert model.task == "diarization" and model.sample_rate == 16000 and model.consolidate_default is False
    audio = tone(3.0)
    result = model.diarize_detailed(audio)
    # sorted by start, speakers numbered in order of first appearance (SPEAKER_00 talks first)
    assert [(t.start, t.end, t.speaker) for t in result.turns] == [(0.5, 1.5, 0), (1.4, 2.1, 1), (2.0, 3.0, 1)]
    assert result.activity is None
    file, kwargs = pipeline.calls[-1]
    assert file["sample_rate"] == 16000 and tuple(file["waveform"].shape) == (1, len(audio))
    assert kwargs == {"min_speakers": 2, "max_speakers": 4}  # the catalog's options
    model.diarize(audio, num_speakers=3, threshold=0.2)
    assert pipeline.calls[-1][1] == {"num_speakers": 3}  # a request's num_speakers wins; threshold is not used
    assert model.diarize(np.zeros(0, np.float32)) == []


def test_exclusive_diarization_and_pyannote_3_output() -> None:
    regular = [(0.0, 2.0, "A"), (1.5, 3.0, "B")]
    exclusive = [(0.0, 1.5, "A"), (1.5, 3.0, "B")]
    pipeline = FakePipeline(regular, exclusive=exclusive)
    assert [(t.start, t.end) for t in PyannoteDiarization(pipeline, None).diarize(tone(3.0))] == [
        (0.0, 2.0),
        (1.5, 3.0),
    ]
    excl = PyannoteDiarization(pipeline, None, exclusive=True)
    assert [(t.start, t.end) for t in excl.diarize(tone(3.0))] == [(0.0, 1.5), (1.5, 3.0)]
    # pyannote.audio 3.x returns the Annotation itself (or a tuple with embeddings)
    old = FakePipeline(regular, legacy=True)
    assert len(PyannoteDiarization(old, None).diarize(tone(3.0))) == 2
    assert annotation_turns(FakeAnnotation([(1.0, 1.0, "A")])) == []  # empty turns are dropped


def test_embedding_is_unit_length_and_padded() -> None:
    model = PyannoteDiarization(FakePipeline([]), fake_embedding)
    vec = model.embed(tone(1.0))
    assert vec.dtype == np.float32 and vec.shape == (4,) and abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6
    short = model.embed(tone(0.01))  # shorter than the embedding model's minimum: padded
    assert fake_embedding.seen[-1] >= fake_embedding.min_num_samples and short.shape == (4,)
    with pytest.raises(Exception, match="no speaker embedding"):
        PyannoteDiarization(FakePipeline([]), None).embed(tone(1.0))


# -- the configuration guard -----------------------------------------------------------------------------


def test_the_pinned_configurations_pass(tmp_path: Path) -> None:
    for text, digest in ((CONFIG_31, SHA_31), (CONFIG_COMMUNITY_1, SHA_COMMUNITY_1), (CONFIG_PIXIT, SHA_PIXIT)):
        assert hashlib.sha256(text.encode()).hexdigest() == digest

    root = model_folder(tmp_path, CONFIG_31, ("segmentation", "embedding"))
    config = safe_config(root, hashes={"config.yaml": SHA_31})
    params = config["pipeline"]["params"]
    assert config["pipeline"]["name"] == "pyannote.audio.pipelines.SpeakerDiarization"
    # the sub-models bolo downloaded with the pipeline, not the Hub's latest revision
    assert params["segmentation"] == {"checkpoint": str(root), "subfolder": "segmentation"}
    assert params["embedding"] == {"checkpoint": str(root), "subfolder": "embedding"}
    assert params["clustering"] == "AgglomerativeClustering" and config["params"]["clustering"]["method"] == "centroid"

    root = model_folder(tmp_path / "c1", CONFIG_COMMUNITY_1, ("segmentation", "embedding", "plda"))
    params = safe_config(root, hashes={"config.yaml": SHA_COMMUNITY_1})["pipeline"]["params"]
    assert params["plda"] == {"checkpoint": str(root), "subfolder": "plda"} and params["clustering"] == "VBxClustering"

    root = model_folder(tmp_path / "pixit", CONFIG_PIXIT)  # downloaded before the sub-model was in the catalog
    params = safe_config(root, revisions={"pyannote/separation-ami-1.0": PIXIT_REV})["pipeline"]["params"]
    assert params["segmentation"] == {"checkpoint": "pyannote/separation-ami-1.0", "revision": PIXIT_REV}
    assert params["embedding"] == "speechbrain/spkrec-ecapa-voxceleb@5c0be3875fda05e81f3c004ed8c7c06be308de1e"


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.replace("pyannote.audio.pipelines.SpeakerDiarization", "os.system"),
        lambda c: c.replace("pyannote.audio.pipelines.SpeakerDiarization", "evil.pipelines.SpeakerDiarization"),
        lambda c: c.replace("pyannote.audio.pipelines.SpeakerDiarization", "SpeakerDiarization"),
        lambda c: c.replace("AgglomerativeClustering", "subprocess.Popen"),
        lambda c: c + "preprocessors:\n  audio:\n    name: os.system\n    params:\n      command: id\n",
        lambda c: c.replace("    method: centroid", "    method:\n      _target_: builtins.exec"),
        lambda c: c.replace("    method: centroid", "    method: builtins.eval"),
        lambda c: c.replace("embedding_batch_size: 32", "embedding_batch_size: 32\n    hook: os.system"),
        lambda c: c.replace("pyannote/segmentation-3.0", "{checkpoint: /etc/passwd}"),
        lambda c: c.replace("pyannote/segmentation-3.0", "$model/../../etc"),
        lambda c: c.replace("pyannote/segmentation-3.0", "someone/segmentation@main"),
        lambda c: c.replace("pyannote/segmentation-3.0", "someone/other-model"),  # not downloaded, not pinned
        lambda c: "!!python/object/apply:os.system ['id']\n",
    ],
)
def test_configurations_naming_other_code_are_refused(tmp_path: Path, change: Any) -> None:
    root = model_folder(tmp_path, change(CONFIG_31), ("embedding",))
    with pytest.raises(UnsafeConfig):
        safe_config(root)


def test_changed_files_are_refused(tmp_path: Path) -> None:
    root = model_folder(tmp_path, CONFIG_31.replace("0.7045654963945799", "0.5"), ("segmentation", "embedding"))
    with pytest.raises(UnsafeConfig, match="does not match"):
        safe_config(root, hashes={"config.yaml": SHA_31})
    root = model_folder(tmp_path / "b", CONFIG_31, ("segmentation", "embedding"))
    with pytest.raises(UnsafeConfig, match="does not match"):
        safe_config(root, hashes={"config.yaml": SHA_31, "segmentation/pytorch_model.bin": "0" * 64})
    with pytest.raises(UnsafeConfig, match="not pinned to the catalog"):
        check_config(
            {"pipeline": {"name": "pyannote.audio.pipelines.SpeechSeparation",
                          "params": {"segmentation": f"pyannote/separation-ami-1.0@{'1' * 40}"}}},
            tmp_path, revisions={"pyannote/separation-ami-1.0": PIXIT_REV},
        )  # fmt: skip


def test_a_refused_configuration_never_reaches_pyannote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_pyannote() -> Any:
        raise AssertionError("pyannote was imported for a refused configuration")

    monkeypatch.setattr(diarization, "import_pyannote", no_pyannote)
    root = model_folder(tmp_path, CONFIG_31.replace("pyannote.audio.pipelines.SpeakerDiarization", "os.system"))
    ctx = LoadContext("pyannote-diarization", "diarization", root, {"repo": "pyannote/speaker-diarization-3.1"})
    with pytest.raises(UnsafeConfig, match=r"os\.system"):
        diarization.load(ctx)
    # the same guard in front of PixIT
    from bolo_torch.models.separation import load_pixit

    bad = model_folder(tmp_path / "pixit", CONFIG_PIXIT.replace("pyannote/separation-ami-1.0", "someone/separation"))
    with pytest.raises(UnsafeConfig, match="not pinned"):
        load_pixit(LoadContext("pixit", "separation", bad, {"repo": "pyannote/speech-separation-ami-1.0"}))
    # an empty model folder: the plugin downloads only at a pinned commit
    empty = tmp_path / "empty"
    empty.mkdir()
    ctx = LoadContext("pyannote-diarization", "diarization", empty, {"repo": "pyannote/speaker-diarization-3.1"})
    with pytest.raises(UnsafeConfig, match="commit"):
        diarization.load(ctx)


def test_load_builds_the_pipeline_from_the_checked_dictionary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakePipelineClass:
        @staticmethod
        def from_pretrained(checkpoint: Any, token: Any = None) -> FakePipeline:
            seen["checkpoint"] = checkpoint
            return FakePipeline([(0.0, 1.0, "A")])

    class SpeakerDiarization:
        def __init__(self, segmentation: Any = None, embedding: Any = None, plda: Any = None) -> None: ...

    class FakeAudio:
        Pipeline = FakePipelineClass

        class pipelines:
            pass

    FakeAudio.pipelines.SpeakerDiarization = SpeakerDiarization  # type: ignore[attr-defined]
    monkeypatch.setattr(diarization, "import_pyannote", lambda: FakeAudio)
    root = model_folder(tmp_path, CONFIG_31, ("segmentation", "embedding", "plda"))
    options = {"repo": "pyannote/speaker-diarization-3.1", "sha256": {"config.yaml": SHA_31}, "max_speakers": 3}
    model = diarization.load(LoadContext("pyannote-diarization", "diarization", root, options))
    config = seen["checkpoint"]
    assert isinstance(config, dict)  # never the file: pyannote gets the checked copy
    # pyannote.audio 4 would fetch community-1's PLDA from its latest revision: it gets the pinned copy instead
    assert config["pipeline"]["params"]["plda"] == {"checkpoint": str(root), "subfolder": "plda"}
    assert model.speakers == {"max_speakers": 3}
    assert [t.speaker for t in model.diarize(tone(1.0))] == [0]
