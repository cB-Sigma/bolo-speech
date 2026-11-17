"""Hugging Face remote code (``trust_remote_code=True``), as an explicit, pinned opt-in per model.

Some models exist only as Python files in their own Hugging Face repository; transformers imports and runs them when
``trust_remote_code=True`` is passed. That code runs with the full rights of the plugin's process. The plugin passes
``trust_remote_code=True`` only through :func:`trust_remote_code`, and only when all of this holds:

* the operator opted in for this model: the option ``allow_remote_code`` is true. bolo sets it from the per-model
  setting (``bolo models config <id> --set allow_remote_code=true``) and overrides whatever a catalog entry says, so
  a catalog entry cannot opt in by itself. Off by default.
* the catalog pins the code: ``remote_code.revision`` is the reviewed commit of the repository and
  ``remote_code.files`` the sha256 of every ``.py`` file and of every JSON file with an ``auto_map``.
* the model folder is that commit: bolo's download manifest (or the ``revision`` option, for a download by the
  plugin) names exactly that revision; any other revision is refused.
* every listed file is present with its sha256, the folder holds no ``.py`` file that is not listed, and no
  ``auto_map`` names a class in another repository (``other/repo--module.Class``), which transformers would fetch.

The check runs right before the model library is imported and loads the model, and it is logged as a warning every
time. :func:`pins` builds the ``remote_code`` block of a catalog entry from a folder (``bolo-torch pin-remote-code``).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from bolo_torch.pyannote_config import sha256_file
from bolo_torch.runtime import LoadContext, PluginError

log = logging.getLogger("bolo_torch")

COMMIT = re.compile(r"^[0-9a-f]{40}$")
#: JSON files transformers reads an ``auto_map`` from
AUTO_MAP_FILES = (
    "config.json", "tokenizer_config.json", "preprocessor_config.json", "processor_config.json",
    "feature_extractor_config.json", "image_processor_config.json", "video_preprocessor_config.json",
)  # fmt: skip
SKIP_DIRS = frozenset({".git", ".cache", "__pycache__"})
WARNING = (
    "remote code: {model} imports Python files from its own model repository ({files}) at commit {revision}. That "
    "code runs inside the bolo-torch process with its rights (files, network, other models' data). It was pinned "
    "after a review and each file matched its sha256 just now; it runs because the operator set "
    "allow_remote_code = true for this model."
)


class RemoteCodeRefused(PluginError):
    """A model needs remote code and the conditions above do not hold."""

    code = "remote_code_refused"


def _pin(options: dict[str, Any]) -> dict[str, Any] | None:
    raw = options.get("remote_code")
    return raw if isinstance(raw, dict) else None


def folder_revision(root: Path) -> str | None:
    """The revision the folder was downloaded at: bolo's ``manifest.json`` (``source.revision``), else None."""
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    source = data.get("source") if isinstance(data, dict) else None
    revision = source.get("revision") if isinstance(source, dict) else None
    return str(revision) if revision else None


def code_files(root: Path) -> list[str]:
    """Every ``.py`` file under ``root`` (relative POSIX paths)."""
    out = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        out.append(rel.as_posix())
    return sorted(out)


def auto_map_files(root: Path) -> dict[str, dict[str, Any]]:
    """The JSON files of ``root`` that hold an ``auto_map``, with it."""
    found: dict[str, dict[str, Any]] = {}
    for name in AUTO_MAP_FILES:
        path = root / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RemoteCodeRefused(f"{name} in {root} is not valid JSON: {exc}") from exc
        if isinstance(data, dict) and isinstance(data.get("auto_map"), dict):
            found[name] = data["auto_map"]
    return found


def _foreign(auto_map: dict[str, Any]) -> list[str]:
    """``auto_map`` targets that name another repository (``repo--module.Class``)."""
    out: list[str] = []
    for value in auto_map.values():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str) and "--" in item:
                out.append(item)
    return out


def check(root: Path, options: dict[str, Any], model: str = "") -> dict[str, Any]:
    """Verify the remote code of the folder ``root`` against the pin; returns the pin. Raises
    :class:`RemoteCodeRefused` (with what to do) when anything does not hold."""
    pin = _pin(options)
    name = model or root.name
    if pin is None:
        raise RemoteCodeRefused(
            f"{name} needs remote code (Python files from its model repository) and its catalog entry pins none. "
            "Add options.remote_code (revision and the sha256 of each file: bolo-torch pin-remote-code <folder> "
            "--revision <commit>) after reading the files, then opt in with allow_remote_code = true."
        )
    revision = str(pin.get("revision") or "")
    files = pin.get("files")
    if not COMMIT.match(revision) or not isinstance(files, dict) or not files:
        raise RemoteCodeRefused(f"{name}: options.remote_code needs a 40-character revision and the files' sha256")
    if options.get("allow_remote_code") is not True:
        listed = ", ".join(sorted(str(f) for f in files if str(f).endswith(".py")))
        raise RemoteCodeRefused(
            f"{name} runs remote code: {listed} from its model repository at commit {revision[:12]}, imported into "
            "the plugin's process with its rights. bolo does not run it unless you opt in for this model: read the "
            f"files, then run `bolo models config {name} --set allow_remote_code=true` (or PUT "
            '/api/v1/models/<id>/settings {"allow_remote_code": true}) and load it again.'
        )
    have = folder_revision(root) or (str(options["revision"]) if options.get("revision") else None)
    if have != revision:
        raise RemoteCodeRefused(
            f"{name}: the model files are at revision {have or 'unknown'}, and remote code runs only at the reviewed "
            f"commit {revision}. Download the pinned revision again (bolo rm, then bolo pull)."
        )
    for rel, expected in files.items():
        path = root / str(rel)
        if not path.is_file():
            raise RemoteCodeRefused(f"{name}: {rel} is missing from {root} (pull the model again)")
        got = sha256_file(path)
        if got != str(expected).lower():
            raise RemoteCodeRefused(
                f"{name}: {rel} does not match the reviewed file (sha256 {got}, pinned {expected}); refusing to run it"
            )
    unlisted = [f for f in code_files(root) if f not in files]
    if unlisted:
        raise RemoteCodeRefused(f"{name}: Python files that were not reviewed are in {root}: {', '.join(unlisted)}")
    for json_name, auto_map in auto_map_files(root).items():
        if json_name not in files:
            raise RemoteCodeRefused(f"{name}: {json_name} maps classes (auto_map) and is not pinned by sha256")
        foreign = _foreign(auto_map)
        if foreign:
            raise RemoteCodeRefused(
                f"{name}: {json_name} maps classes to another repository ({', '.join(foreign)}); refusing to fetch it"
            )
    return pin


def trust_remote_code(ctx: LoadContext, root: Path) -> bool:
    """``True`` for ``from_pretrained(..., trust_remote_code=...)`` once :func:`check` passed; logs the warning."""
    pin = check(root, ctx.options, ctx.model_id)
    py = ", ".join(sorted(str(f) for f in pin["files"] if str(f).endswith(".py")))
    log.warning(WARNING.format(model=ctx.model_id or root.name, files=py, revision=str(pin["revision"])[:12]))
    return True


def pins(root: Path, revision: str) -> dict[str, Any]:
    """The ``options.remote_code`` block for the model folder ``root`` (downloaded at ``revision``)."""
    if not COMMIT.match(revision):
        raise ValueError("revision must be a 40-character commit")
    files = {rel: sha256_file(root / rel) for rel in code_files(root)}
    for json_name in auto_map_files(root):
        files[json_name] = sha256_file(root / json_name)
    return {"revision": revision, "files": dict(sorted(files.items()))}


def check_pinned_files(root: Path, options: dict[str, Any], names: list[str], model: str = "") -> None:
    """Files a loader hands to code that builds objects from them (hyperpyyaml configurations, which name classes
    to instantiate) must match the catalog's ``sha256`` option before they are read."""
    hashes = {str(k): str(v).lower() for k, v in dict(options.get("sha256") or {}).items()}
    for rel in names:
        path = root / rel
        expected = hashes.get(rel)
        if expected is None:
            raise RemoteCodeRefused(f"{model or root.name}: the catalog pins no sha256 for {rel}; refusing to read it")
        if not path.is_file():
            raise RemoteCodeRefused(f"{model or root.name}: {rel} is missing from {root} (pull the model again)")
        got = sha256_file(path)
        if got != expected:
            raise RemoteCodeRefused(
                f"{model or root.name}: {rel} does not match the catalog (sha256 {got}, pinned {expected})"
            )
