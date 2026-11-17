"""A guard in front of ``pyannote.audio``'s ``Pipeline.from_pretrained``.

pyannote builds a pipeline from its ``config.yaml``: it imports and instantiates the class the file names
(``pipeline.name``), imports any ``preprocessors`` it lists, and downloads the sub-models it names from the
Hugging Face Hub, by default from their latest revision. A changed or malicious repository could therefore make
the plugin run any installed callable, or load unpinned weights (pyannote unpickles model checkpoints with
``weights_only=False``). So the plugin never hands pyannote a configuration file. It reads ``config.yaml`` itself
(``yaml.safe_load``), checks the file's sha256 when the catalog gives one, accepts only the pyannote pipeline classes
and clustering names below, only plain values in the parameters, and only sub-models that are either files bolo
downloaded (pinned revision, verified sha256) or Hub repositories pinned to a commit; and it passes pyannote the
checked dictionary, with every sub-model resolved to a local folder or a pinned revision. Anything else is refused
with :class:`UnsafeConfig` before pyannote is imported.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from bolo_torch.runtime import PluginError

#: pipeline classes pyannote may build (``pipeline.name``); short names would resolve in pyannote.pipeline.blocks
PIPELINES = frozenset(
    {
        "pyannote.audio.pipelines.SpeakerDiarization",
        "pyannote.audio.pipelines.speaker_diarization.SpeakerDiarization",
        "pyannote.audio.pipelines.SpeechSeparation",
        "pyannote.audio.pipelines.speech_separation.SpeechSeparation",
    }
)
#: values of ``pipeline.params.clustering`` (members of pyannote's own clustering enum)
CLUSTERING = frozenset({"AgglomerativeClustering", "VBxClustering"})
#: ``pipeline.params`` that name a sub-model
MODEL_KEYS = ("segmentation", "embedding", "plda")
#: other ``pipeline.params``: plain numbers, booleans or short strings
PLAIN_KEYS = frozenset(
    {
        "clustering", "segmentation_step", "segmentation_batch_size", "embedding_batch_size",
        "embedding_exclude_overlap", "legacy", "der_variant",
    }
)  # fmt: skip
TOP_KEYS = frozenset({"version", "dependencies", "pipeline", "params", "freeze"})
COMMIT = re.compile(r"^[0-9a-f]{40}$")
REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
DOTTED = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+(:\w+)?$")
SUBFOLDER = re.compile(r"^[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*$")


class UnsafeConfig(PluginError):
    """A pipeline configuration the plugin will not hand to pyannote."""

    code = "invalid_request"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_files(root: Path, hashes: dict[str, str]) -> None:
    """Check the sha256 of each listed file that is present (a missing sub-model falls back to its pinned Hub
    revision, if it has one; the configuration itself must be present)."""
    for rel, expected in hashes.items():
        path = root / rel
        if not path.is_file():
            if rel == "config.yaml":
                raise UnsafeConfig(f"{root} has no config.yaml")
            continue
        got = sha256_file(path)
        if got != str(expected).lower():
            raise UnsafeConfig(f"{rel} in {root} does not match the catalog: sha256 {got}, expected {expected}")


def _plain(value: Any, where: str) -> None:
    """Parameters must be plain data: numbers, booleans, null, short strings, and dicts or lists of them."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise UnsafeConfig(f"{where}: keys must be strings")
            if key in ("name", "_target_", "class", "module"):
                raise UnsafeConfig(f"{where}.{key}: the configuration may not name classes or callables here")
            _plain(item, f"{where}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _plain(item, f"{where}[{i}]")
    elif isinstance(value, str):
        if len(value) > 200 or DOTTED.match(value.strip()):
            raise UnsafeConfig(f"{where}: {value!r} looks like a dotted path, not a parameter value")
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise UnsafeConfig(f"{where}: unexpected value of type {type(value).__name__}")


def model_ref(key: str, value: Any, root: Path, revisions: dict[str, str]) -> str | dict[str, Any]:
    """A sub-model reference as something pinned: a folder of ``root`` or a Hub repository at a commit."""
    if not isinstance(value, str):
        raise UnsafeConfig(f"pipeline.params.{key}: expected a model reference, got {type(value).__name__}")
    if value.startswith("$model/"):
        sub = value[len("$model/") :]
        if not SUBFOLDER.match(sub):
            raise UnsafeConfig(f"pipeline.params.{key}: {value!r} is not a plain sub-folder")
        if not (root / sub).is_dir():
            raise UnsafeConfig(f"pipeline.params.{key}: {root / sub} is missing (download the model again)")
        return {"checkpoint": str(root), "subfolder": sub}
    repo, _, revision = value.partition("@")
    if not REPO.match(repo):
        raise UnsafeConfig(f"pipeline.params.{key}: {value!r} is not a Hugging Face repository id")
    if (root / key).is_dir() and any((root / key).iterdir()):
        return {"checkpoint": str(root), "subfolder": key}  # bolo downloaded it (pinned, sha256 verified)
    pinned = revisions.get(repo)
    if revision:
        if not COMMIT.match(revision) or (pinned and pinned != revision):
            raise UnsafeConfig(f"pipeline.params.{key}: {value!r} is not pinned to the catalog's commit")
        if repo.startswith("speechbrain/"):
            return value  # pyannote's SpeechBrain loader takes repo@revision
        return {"checkpoint": repo, "revision": revision}
    if pinned and COMMIT.match(pinned):
        return {"checkpoint": repo, "revision": pinned}
    raise UnsafeConfig(
        f"pipeline.params.{key}: {repo} is not pinned (neither downloaded with the model nor given a commit in the "
        "catalog option 'revisions')"
    )


def safe_config(
    root: Path,
    *,
    revisions: dict[str, str] | None = None,
    hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The checked pipeline configuration of the model folder ``root``, ready for ``Pipeline.from_pretrained``.

    ``revisions`` pins Hub sub-models (repository -> commit); ``hashes`` are sha256 of files under ``root``.
    """
    import yaml

    revisions = dict(revisions or {})
    if hashes:
        verify_files(root, hashes)
    path = root / "config.yaml"
    if not path.is_file():
        raise UnsafeConfig(f"{root} has no config.yaml")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise UnsafeConfig(f"config.yaml is not plain YAML: {exc}") from exc
    return check_config(config, root, revisions=revisions)


def check_config(config: Any, root: Path, *, revisions: dict[str, str] | None = None) -> dict[str, Any]:
    """Validate a parsed configuration and resolve its sub-models (see :func:`safe_config`)."""
    revisions = dict(revisions or {})
    if not isinstance(config, dict):
        raise UnsafeConfig("config.yaml must be a mapping")
    extra = set(config) - TOP_KEYS
    if extra:
        raise UnsafeConfig(f"config.yaml has sections the plugin does not accept: {', '.join(sorted(map(str, extra)))}")
    pipeline = config.get("pipeline")
    if not isinstance(pipeline, dict) or set(pipeline) - {"name", "params"}:
        raise UnsafeConfig("config.yaml: 'pipeline' must hold only 'name' and 'params'")
    name = pipeline.get("name")
    if name not in PIPELINES:
        raise UnsafeConfig(
            f"config.yaml names the pipeline class {name!r}; the plugin builds only {', '.join(sorted(PIPELINES))}"
        )
    params = pipeline.get("params") or {}
    if not isinstance(params, dict):
        raise UnsafeConfig("config.yaml: 'pipeline.params' must be a mapping")
    out_params: dict[str, Any] = {}
    for key, value in params.items():
        if key in MODEL_KEYS:
            out_params[key] = model_ref(key, value, root, revisions)
        elif key in PLAIN_KEYS:
            if key == "clustering" and value not in CLUSTERING:
                raise UnsafeConfig(
                    f"pipeline.params.clustering: {value!r} is not one of {', '.join(sorted(CLUSTERING))}"
                )
            _plain(value, f"pipeline.params.{key}")
            out_params[key] = value
        else:
            raise UnsafeConfig(f"pipeline.params.{key} is not a parameter the plugin accepts")
    deps = config.get("dependencies") or {}
    if not isinstance(deps, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in deps.items()):
        raise UnsafeConfig("config.yaml: 'dependencies' must map package names to versions")
    for section in ("params", "freeze"):
        if section in config:
            _plain(config[section], section)
    out: dict[str, Any] = {"pipeline": {"name": name, "params": out_params}}
    for key in ("version", "dependencies", "params", "freeze"):
        if key in config:
            out[key] = config[key]
    return out
