"""DiariZen: the install check (the authors' commit), and the load and call path with a fake of the authors'
``DiariZenPipeline`` that mirrors what the real one takes and returns (``diarizen_hub`` and ``embedding_model`` in,
a file-like WAV per call, a pyannote ``Annotation`` out, ``_embedding`` and ``to``)."""

from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from bolo_torch.engine import LOADERS, NOT_IMPLEMENTED, TorchEngine
from bolo_torch.models import diarizen
from bolo_torch.models.diarization import PyannoteDiarization
from bolo_torch.runtime import InvalidRequest, LoadContext, MissingDependency
from fakes import FakeAnnotation, fake_embedding, tone


def direct_url(commit: str | None) -> str:
    url = "https://github.com/BUTSpeechFIT/DiariZen"
    info: dict[str, Any] = {"url": url}
    if commit is not None:
        info["vcs_info"] = {"vcs": "git", "commit_id": commit, "requested_revision": commit}
    else:
        info["archive_info"] = {}
    return json.dumps(info)


def reader(commits: dict[str, str | None]) -> Any:
    def read(dist: str) -> str | None:
        if dist not in commits:
            from importlib import metadata

            raise metadata.PackageNotFoundError(dist)
        return direct_url(commits[dist])

    return read


def test_install_must_be_the_pinned_commit() -> None:
    good = {"diarizen": diarizen.COMMIT, "pyannote.audio": diarizen.COMMIT}
    diarizen.check_install(reader(good))  # nothing raised
    assert diarizen.installed_commit("diarizen", reader(good)) == diarizen.COMMIT
    for bad in (
        {"diarizen": diarizen.COMMIT, "pyannote.audio": "f" * 40},  # another commit of the fork
        {"diarizen": diarizen.COMMIT, "pyannote.audio": None},  # pyannote.audio from PyPI or an archive
        {"pyannote.audio": diarizen.COMMIT},  # diarizen not installed
    ):
        with pytest.raises(MissingDependency) as err:
            diarizen.check_install(reader(bad))
        assert diarizen.INSTALL in str(err.value) and "environment of their own" in str(err.value)
    assert diarizen.INSTALL.count(diarizen.COMMIT) == 2 and "#subdirectory=pyannote-audio" in diarizen.INSTALL


class FakeDiariZenPipeline:
    """``diarizen.pipelines.inference.DiariZenPipeline``: built from the model folder and an embedding checkpoint,
    called with a path or file object of audio (it runs ``torchaudio.load`` on it), returns an Annotation."""

    instances: list[FakeDiariZenPipeline] = []

    def __init__(self, diarizen_hub: Path, embedding_model: Any, config_parse: Any = None, rttm_out_dir: Any = None):
        assert isinstance(diarizen_hub, Path) and (diarizen_hub / "config.toml").is_file()
        self.hub = diarizen_hub
        self.embedding_model = embedding_model
        self._embedding = fake_embedding
        self.device: Any = None
        self.heard: list[tuple[np.ndarray, int]] = []
        FakeDiariZenPipeline.instances.append(self)

    def to(self, device: Any) -> FakeDiariZenPipeline:
        self.device = device
        return self

    def __call__(self, in_wav: Any, sess_name: str | None = None) -> FakeAnnotation:
        assert isinstance(in_wav, (str, io.BytesIO))
        audio, rate = sf.read(in_wav, dtype="float32")
        self.heard.append((audio, rate))
        half = len(audio) / rate / 2
        return FakeAnnotation([(half, 2 * half, "3"), (0.0, half, "1")])


def model_folder(tmp_path: Path) -> Path:
    root = tmp_path / "diarizen"
    (root / "plda").mkdir(parents=True)
    for name in diarizen.MODEL_FILES:
        (root / name).write_bytes(b"x")
    return root


@pytest.fixture
def fake_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    inference = types.ModuleType("diarizen.pipelines.inference")
    inference.DiariZenPipeline = FakeDiariZenPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "diarizen.pipelines.inference", inference)
    monkeypatch.setattr(diarizen, "check_install", lambda read=None: seen.setdefault("checked", True))
    monkeypatch.setattr(diarizen, "safe_globals", lambda: [])

    def snapshot(repo: str, revision: str | None = None, allow: list[str] | None = None, **_: Any) -> Path:
        seen["embedding"] = (repo, revision, allow)
        folder = tmp_path / "hub" / repo.replace("/", "--")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    monkeypatch.setattr(diarizen, "hub_snapshot", snapshot)
    FakeDiariZenPipeline.instances.clear()
    return seen


def test_load_runs_the_authors_pipeline(tmp_path: Path, fake_install: dict[str, Any]) -> None:
    root = model_folder(tmp_path)
    model = diarizen.load(LoadContext("diarizen", "diarization", root, {"model_id": "diarizen"}))
    assert isinstance(model, PyannoteDiarization) and model.sample_rate == 16000
    assert fake_install["checked"] is True
    # the WeSpeaker checkpoint at the pinned revision, given as a dictionary (a path naming "wespeaker" would send
    # pyannote to its ONNX loader)
    repo, revision, allow = fake_install["embedding"]
    assert (repo, revision) == (diarizen.EMBEDDING_REPO, diarizen.EMBEDDING_REVISION) and "pytorch_model.bin" in allow
    pipeline = FakeDiariZenPipeline.instances[-1]
    assert pipeline.hub == root.resolve() and str(pipeline.device) == "cpu"
    assert pipeline.embedding_model == {
        "checkpoint": str(tmp_path / "hub" / repo.replace("/", "--") / "pytorch_model.bin")
    }
    audio = tone(2.0)
    turns = model.diarize(audio, num_speakers=3)  # VBx picks the count: the request's is not passed on
    assert [(t.start, t.end, t.speaker) for t in turns] == [(0.0, 1.0, 0), (1.0, 2.0, 1)]
    heard, rate = pipeline.heard[-1]
    assert rate == 16000 and np.array_equal(heard, audio)  # a float WAV: the samples arrive unchanged
    vec = model.embed(tone(1.0))
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6


def test_a_folder_without_the_model_files_is_refused(tmp_path: Path, fake_install: dict[str, Any]) -> None:
    root = model_folder(tmp_path)
    (root / "plda" / "plda.npz").unlink()
    with pytest.raises(InvalidRequest, match=r"plda/plda.npz"):
        diarizen.load(LoadContext("diarizen", "diarization", root, {}))


def test_other_sample_rates_are_refused() -> None:
    runner = diarizen.DiariZenRunner(FakeDiariZenPipeline.__new__(FakeDiariZenPipeline))
    with pytest.raises(InvalidRequest, match="16000"):
        runner({"waveform": np.zeros((1, 10), np.float32), "sample_rate": 8000})


def test_the_kind_is_implemented() -> None:
    assert LOADERS["diarizen"] == ("bolo_torch.models.diarizen", "load") and "diarizen" not in NOT_IMPLEMENTED
    assert "diarizen" in TorchEngine().kinds
