"""The remote-code opt-in (pinned commit, sha256 of every file, the operator's per-model setting) and the pinned
GitHub inference-code bundles."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from bolo_torch import remote_code, upstream
from bolo_torch.remote_code import RemoteCodeRefused, check, check_pinned_files, pins
from bolo_torch.runtime import LoadContext, MissingDependency, PluginError

REV = "a" * 40


def model_folder(tmp_path: Path, revision: str = REV) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"source": {"repo": "org/m", "revision": revision}}))
    (root / "modeling_m.py").write_bytes(b"X = 1\n")
    (root / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "modeling_m.M"}}))
    (root / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 10}))
    (root / "model.safetensors").write_bytes(b"weights")
    return root


def test_pins_and_the_opt_in(tmp_path: Path) -> None:
    root = model_folder(tmp_path)
    pin = pins(root, REV)
    assert pin == {"revision": REV, "files": {"config.json": pin["files"]["config.json"],
                                              "modeling_m.py": hashlib.sha256(b"X = 1\n").hexdigest()}}  # fmt: skip
    with pytest.raises(RemoteCodeRefused, match="allow_remote_code=true"):
        check(root, {"remote_code": pin}, "m")  # off by default
    with pytest.raises(RemoteCodeRefused, match="pins none"):
        check(root, {"allow_remote_code": True}, "m")
    assert check(root, {"remote_code": pin, "allow_remote_code": True}, "m") == pin
    ctx = LoadContext("k", "speech-to-speech", root, {"remote_code": pin, "allow_remote_code": True}, model_id="m")
    assert remote_code.trust_remote_code(ctx, root) is True


@pytest.mark.parametrize("change", ["revision", "edited", "unlisted", "foreign", "unpinned-map", "missing", "truthy"])
def test_anything_else_is_refused(tmp_path: Path, change: str) -> None:
    root = model_folder(tmp_path)
    pin = pins(root, REV)
    options = {"remote_code": pin, "allow_remote_code": True}
    if change == "revision":
        (root / "manifest.json").write_text(json.dumps({"source": {"revision": "b" * 40}}))
    elif change == "edited":
        (root / "modeling_m.py").write_text("import os\n")
    elif change == "unlisted":
        (root / "sub").mkdir()
        (root / "sub" / "extra.py").write_text("")
    elif change == "foreign":
        (root / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "evil/repo--modeling.M"}}))
        pin["files"]["config.json"] = hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
    elif change == "unpinned-map":
        (root / "tokenizer_config.json").write_text(json.dumps({"auto_map": {"AutoTokenizer": ["modeling_m.T", None]}}))
    elif change == "missing":
        (root / "modeling_m.py").unlink()
    elif change == "truthy":
        options["allow_remote_code"] = "yes"  # only a real true opts in
    with pytest.raises(RemoteCodeRefused):
        check(root, options, "m")


def test_revision_of_a_plugin_download(tmp_path: Path) -> None:
    root = model_folder(tmp_path)
    pin = pins(root, REV)
    (root / "manifest.json").unlink()
    with pytest.raises(RemoteCodeRefused, match="revision unknown"):
        check(root, {"remote_code": pin, "allow_remote_code": True})
    assert check(root, {"remote_code": pin, "allow_remote_code": True, "revision": REV})


def test_pinned_configuration_files(tmp_path: Path) -> None:
    (tmp_path / "flow.yaml").write_bytes(b"flow: !new:x.Y\n")
    digest = hashlib.sha256(b"flow: !new:x.Y\n").hexdigest()
    check_pinned_files(tmp_path, {"sha256": {"flow.yaml": digest}}, ["flow.yaml"])
    with pytest.raises(RemoteCodeRefused, match="pins no sha256"):
        check_pinned_files(tmp_path, {}, ["flow.yaml"])
    (tmp_path / "flow.yaml").write_bytes(b"flow: !new:os.system\n")
    with pytest.raises(RemoteCodeRefused, match="does not match"):
        check_pinned_files(tmp_path, {"sha256": {"flow.yaml": digest}}, ["flow.yaml"])


def test_bundled_pins_are_complete() -> None:
    found = upstream.bundles()
    assert {"step-audio-2", "glm-4-voice", "kimi-audio", "spark-tts"} <= set(found)
    for bundle in found.values():
        for part in bundle.parts:
            assert len(part.commit) == 40 and part.license and part.license_url.startswith("https://github.com/")
            assert all(len(sha) == 64 for sha in part.files.values())
    assert "token2wav.py" in found["step-audio-2"].files()
    assert "third_party/Matcha-TTS/matcha/models/components/flow_matching.py" in found["glm-4-voice"].files()
    assert "kimia_infer/models/tokenizer/glm4/speech_tokenizer/modeling_whisper.py" in found["kimi-audio"].files()


def small_bundle() -> tuple[upstream.Bundle, dict[str, bytes]]:
    files = {"pkg/__init__.py": b"VALUE = 42\n", "LICENSE": b"MIT\n"}
    bundle = upstream.parse(
        {
            "name": "tiny", "extra": "tiny", "paths": [""],
            "parts": [{"repo": "o/r", "commit": "c" * 40, "license": "MIT", "license_url": "https://github.com/o/r",
                       "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}],
        }
    )  # fmt: skip
    served = {f"https://raw.githubusercontent.com/o/r/{'c' * 40}/{k}": v for k, v in files.items()}
    return bundle, served


def test_bundle_install_verify_and_ensure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLO_TORCH_CODE_DIR", str(tmp_path))
    bundle, served = small_bundle()
    monkeypatch.setattr(upstream, "bundles", lambda: {"tiny": bundle})
    monkeypatch.setenv("BOLO_TORCH_CODE_AUTO_INSTALL", "0")
    with pytest.raises(MissingDependency, match="bolo-torch code install tiny"):
        upstream.ensure("tiny")
    monkeypatch.delenv("BOLO_TORCH_CODE_AUTO_INSTALL")
    bad = dict(served)
    bad[next(iter(bad))] = b"tampered"
    with pytest.raises(PluginError, match="is not the pinned"):
        upstream.install(bundle, fetch=bad.__getitem__)
    assert not upstream.bundle_dir(bundle).exists()
    root = upstream.ensure("tiny", fetch=served.__getitem__)
    assert root == tmp_path / "tiny" / bundle.digest and str(root) in sys.path
    assert upstream.verify(bundle) == []
    (root / "pkg" / "__init__.py").write_text("VALUE = 0\n")
    with pytest.raises(PluginError, match="does not match its pins"):
        upstream.ensure("tiny")
    sys.path.remove(str(root))
    assert upstream.describe()[0]["installed"] is True
