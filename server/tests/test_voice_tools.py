"""tools/voices: Hugging Face publishing (with a stand-in HfApi) and the catalog update that reads it back."""

from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "voices"
pytestmark = pytest.mark.skipif(not TOOLS.is_dir(), reason="needs the repository checkout (tools/voices)")


def load(name: str) -> ModuleType:
    sys.path.insert(0, str(TOOLS))
    try:
        spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(TOOLS))


class FakeHfApi:
    def __init__(self) -> None:
        self.repos: list[tuple[str, dict[str, Any]]] = []
        self.uploads: list[dict[str, Any]] = []

    def create_repo(self, repo_id: str, **kw: Any) -> None:
        self.repos.append((repo_id, kw))

    def upload_folder(self, **kw: Any) -> SimpleNamespace:
        folder = Path(kw["folder_path"])
        kw["files"] = {
            str(p.relative_to(folder)): p.read_text(encoding="utf-8") for p in folder.rglob("*") if p.is_file()
        }
        self.uploads.append(kw)
        return SimpleNamespace(oid="ab" * 20)


def make_dist(tmp: Path) -> Path:
    dist = tmp / "dist"
    staged = tmp / "staged" / "mms-tts-pan"
    staged.mkdir(parents=True)
    for name, text in {
        "model.onnx": "x",
        "tokens.txt": "a 1\n",
        "voice.json": "{}",
        "LICENSE": "CC",
        "ATTRIBUTION": "Meta AI",
    }.items():
        (staged / name).write_text(text, encoding="utf-8")
    dist.mkdir()
    with tarfile.open(dist / "mms-tts-pan.tar.gz", "w:gz") as tar:
        tar.add(staged, arcname="mms-tts-pan")
    index = {
        "release": "voices-v1",
        "voices": [
            {
                "id": "mms-tts-pan", "language": "pa", "language_name": "Punjabi", "sample_rate": 16000,
                "source": {"repo": "facebook/mms-tts", "revision": "c" * 40, "path": "models/pan"},
                "license": {"weights": "CC-BY-NC-4.0", "commercial_use": False}, "file": "mms-tts-pan.tar.gz",
                "size": 123, "sha256": "d" * 64, "unpacked_size": 456,
            }
        ],
    }  # fmt: skip
    (dist / "voices.json").write_text(json.dumps(index), encoding="utf-8")
    return dist


def test_publish_to_hub_and_read_back_into_the_catalog(tmp_path: Path) -> None:
    publish_hf = load("publish_hf")
    update_catalog = load("update_catalog")
    dist = make_dist(tmp_path)
    measured = tmp_path / "measured.json"
    measured.write_text(json.dumps({"mms-tts-pan": {"round-trip WER": "0.194"}}), encoding="utf-8")
    api = FakeHfApi()
    index = publish_hf.publish(dist, "codebucketsolutions", api, None, measured)
    assert api.repos == [
        ("codebucketsolutions/bolo-mms-tts-punjabi", {"repo_type": "model", "exist_ok": True, "private": False})
    ]
    card = api.uploads[0]["files"]["README.md"]
    assert card.startswith("---\nlicense: cc-by-nc-4.0\n") and "non-commercial use only" in card
    assert "models/pan" in card and "Meta AI" in card and "| round-trip WER | 0.194 |" in card
    assert "bolo pull mms-tts-pan" in card and "text-to-speech" in card
    hub = index["voices"][0]["hf"]
    assert hub["repo"] == "codebucketsolutions/bolo-mms-tts-punjabi" and hub["revision"] == "ab" * 20
    assert "model.onnx" in hub["files"] and "README.md" not in hub["files"]  # the card is not a model file
    assert json.loads((dist / "voices.json").read_text(encoding="utf-8"))["voices"][0]["hf"] == hub

    catalog = tmp_path / "tts-mms.yaml"
    catalog.write_text(
        "# header\n"
        + yaml.safe_dump(
            [{"id": "mms-tts-pan", "status": "pending", "source": {"urls": []}}, {"id": "other", "status": "pending"}]
        ),
        encoding="utf-8",
    )
    done, pending = update_catalog.update_file(catalog, {v["id"]: v for v in index["voices"]}, "https://example.org/r/")
    assert done == ["mms-tts-pan"] and pending == ["other"]
    entry = yaml.safe_load(catalog.read_text(encoding="utf-8"))[0]
    assert "status" not in entry and entry["size_bytes"] == 456
    assert (
        entry["source"]["repo"] == "codebucketsolutions/bolo-mms-tts-punjabi"
        and entry["source"]["revision"] == "ab" * 20
    )
    assert entry["source"]["fallback"][0]["url"] == "https://example.org/r/mms-tts-pan.tar.gz"
    assert catalog.read_text(encoding="utf-8").startswith("# header")


def test_no_token_skips_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    publish_hf = load("publish_hf")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["publish_hf.py", "--dist", str(make_dist(tmp_path))])
    assert publish_hf.main() == 0 and "nothing uploaded" in capsys.readouterr().out
    assert publish_hf.repo_name({"id": "syspin-mai-female", "language_name": "Maithili", "gender": "female"}) == (
        "bolo-syspin-maithili-female"
    )


def test_fallback_source_is_used_when_the_hub_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bolo.errors import DownloadError, GatedModel
    from bolo.store import ModelStore, Source

    store = ModelStore(tmp_path)
    calls: list[Source] = []
    real_pull = ModelStore.pull

    def plan(self: ModelStore, source: Source) -> list[Any]:
        calls.append(source)
        if source.repo == "down/repo":
            raise DownloadError("HTTP 503")
        if source.repo == "gated/repo":
            raise GatedModel("denied")
        return []

    monkeypatch.setattr(ModelStore, "plan", plan)
    manifest = real_pull(store, "m", Source(repo="down/repo", revision="e" * 40, fallback=[{"path": "a", "url": "u"}]))
    assert manifest["complete"] and calls[-1].urls == [{"path": "a", "url": "u"}] and calls[-1].repo is None
    with pytest.raises(GatedModel):  # access problems are reported, not papered over
        real_pull(store, "g", Source(repo="gated/repo", revision="e" * 40, fallback=[{"path": "a", "url": "u"}]))
    assert Source.from_dict(Source(repo="r", fallback=[{"path": "a"}]).to_dict()).fallback == [{"path": "a"}]
