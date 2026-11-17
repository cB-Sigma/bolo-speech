"""Inference code that is not a package: files of a model's GitHub repository, pinned by commit and sha256.

Some models run only with code from their authors' repository (a speech tokenizer, a flow-matching decoder, a
prompt format), which is not published as a package bolo-torch could depend on. Each such *bundle* is described in
``bolo_torch/upstream/<name>.json``: the repositories (one or more, for example a git submodule), the commit of
each, its licence, and the sha256 of every file used. :func:`install` downloads exactly those files from
``raw.githubusercontent.com`` at the pinned commits, checks each sha256, and writes the bundle into the code folder:

* ``BOLO_TORCH_CODE_DIR``, else ``BOLO_TORCH_CACHE/code``, else ``~/.cache/bolo-torch/code``
  (``%LOCALAPPDATA%\\bolo-torch\\code`` on Windows).

:func:`ensure` is what a loader calls: it installs the bundle when it is missing (``BOLO_TORCH_CODE_AUTO_INSTALL=0``
turns that off), checks every file's sha256 again (a changed file is refused), logs the licence, and puts the
bundle's import roots on ``sys.path``. ``bolo-torch code list | install <name> | verify <name>`` does the same by
hand.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from bolo_torch.pyannote_config import sha256_file
from bolo_torch.runtime import MissingDependency, PluginError

log = logging.getLogger("bolo_torch")

RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
_lock = threading.Lock()


@dataclass(frozen=True)
class Part:
    repo: str
    commit: str
    license: str
    license_url: str
    #: where the repository's files go inside the bundle ("" for the root)
    dest: str
    files: dict[str, str]


@dataclass(frozen=True)
class Bundle:
    name: str
    about: str
    parts: tuple[Part, ...]
    #: folders of the bundle put on sys.path
    paths: tuple[str, ...]
    #: the pip extra with the libraries the code imports
    extra: str

    @property
    def digest(self) -> str:
        """Changes whenever a pin changes, so a new pin never mixes with files of an old one."""
        h = hashlib.sha256()
        for part in self.parts:
            h.update(f"{part.repo}@{part.commit}:{part.dest}\n".encode())
            for rel, sha in sorted(part.files.items()):
                h.update(f"{rel} {sha}\n".encode())
        return h.hexdigest()[:16]

    def licences(self) -> str:
        return "; ".join(f"{p.repo}@{p.commit[:12]}: {p.license} ({p.license_url})" for p in self.parts)

    def files(self) -> dict[str, str]:
        """bundle-relative path -> sha256"""
        out: dict[str, str] = {}
        for part in self.parts:
            for rel, sha in part.files.items():
                out[_join(part.dest, rel)] = sha
        return out


def _join(dest: str, rel: str) -> str:
    path = PurePosixPath(dest) / rel if dest else PurePosixPath(rel)
    if path.is_absolute() or ".." in path.parts:
        raise PluginError(f"bundle path {path} leaves the bundle", "invalid_request")
    return path.as_posix()


def bundles() -> dict[str, Bundle]:
    out: dict[str, Bundle] = {}
    for entry in resources.files("bolo_torch").joinpath("upstream").iterdir():
        if entry.name.endswith(".json"):
            b = parse(json.loads(entry.read_text(encoding="utf-8")))
            out[b.name] = b
    return out


def parse(data: dict[str, Any]) -> Bundle:
    parts = tuple(
        Part(
            str(p["repo"]), str(p["commit"]), str(p["license"]), str(p["license_url"]), str(p.get("dest", "")),
            {str(k): str(v) for k, v in p["files"].items()},
        )
        for p in data["parts"]
    )  # fmt: skip
    return Bundle(str(data["name"]), str(data.get("about", "")), parts, tuple(data.get("paths", [""])), data["extra"])


def code_root() -> Path:
    env = os.environ.get("BOLO_TORCH_CODE_DIR")
    if env:
        return Path(env)
    cache = os.environ.get("BOLO_TORCH_CACHE")
    if cache:
        return Path(cache) / "code"
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "bolo-torch" / "code"
    return Path.home() / ".cache" / "bolo-torch" / "code"


def bundle_dir(bundle: Bundle) -> Path:
    return code_root() / bundle.name / bundle.digest


def get(name: str) -> Bundle:
    found = bundles()
    if name not in found:
        raise PluginError(f"no code bundle {name!r} (bundles: {', '.join(sorted(found))})", "invalid_request")
    return found[name]


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "bolo-torch"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data: bytes = resp.read()
    return data


def install(bundle: Bundle, fetch: Any = None, force: bool = False) -> Path:
    """Download the bundle's files at their pinned commits into :func:`bundle_dir` (sha256 checked)."""
    fetch = fetch or _fetch
    target = bundle_dir(bundle)
    with _lock:
        if (target / ".complete").is_file() and not force:
            return target
        tmp = target.with_name(target.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        log.info("installing the %s inference code (%s) into %s", bundle.name, bundle.licences(), target)
        try:
            for part in bundle.parts:
                for rel, expected in sorted(part.files.items()):
                    url = RAW.format(repo=part.repo, commit=part.commit, path=quote(rel))
                    try:
                        data = fetch(url)
                    except Exception as exc:
                        raise PluginError(f"download of {url} failed: {exc}", "download_failed") from exc
                    got = hashlib.sha256(data).hexdigest()
                    if got != expected:
                        raise PluginError(
                            f"{part.repo}@{part.commit[:12]}/{rel}: sha256 {got} is not the pinned {expected}",
                            "download_failed",
                        )
                    path = tmp / _join(part.dest, rel)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
            meta = {"name": bundle.name, "parts": [{"repo": p.repo, "commit": p.commit, "license": p.license}
                                                    for p in bundle.parts]}  # fmt: skip
            (tmp / ".complete").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            shutil.rmtree(target, ignore_errors=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp.rename(target)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    return target


def verify(bundle: Bundle, root: Path | None = None) -> list[str]:
    """Problems with an installed bundle (missing or changed files); empty when it matches its pins."""
    root = root or bundle_dir(bundle)
    if not (root / ".complete").is_file():
        return [f"{bundle.name} is not installed in {root}"]
    problems = []
    for rel, expected in bundle.files().items():
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel} is missing")
        elif sha256_file(path) != expected:
            problems.append(f"{rel} was changed (sha256 does not match the pin)")
    return problems


def ensure(name: str, *, fetch: Any = None) -> Path:
    """The installed, verified bundle ``name`` on ``sys.path`` (installed first when missing and allowed)."""
    bundle = get(name)
    root = bundle_dir(bundle)
    if not (root / ".complete").is_file():
        if os.environ.get("BOLO_TORCH_CODE_AUTO_INSTALL", "1").strip().lower() in ("0", "false", "no", "off"):
            raise MissingDependency(
                f"the {name} inference code is not installed: run `bolo-torch code install {name}` "
                f"(downloads {bundle.licences()})"
            )
        install(bundle, fetch=fetch)
    problems = verify(bundle, root)
    if problems:
        raise PluginError(
            f"the {name} inference code in {root} does not match its pins ({'; '.join(problems[:5])}); "
            f"reinstall it with `bolo-torch code install {name} --force`",
            "component_missing",
        )
    log.info("%s inference code: %s", name, bundle.licences())
    for rel in bundle.paths:
        path = str(root / rel) if rel else str(root)
        if path not in sys.path:
            sys.path.insert(0, path)
    return root


def describe() -> list[dict[str, Any]]:
    out = []
    for b in sorted(bundles().values(), key=lambda b: b.name):
        root = bundle_dir(b)
        out.append(
            {
                "name": b.name, "about": b.about, "extra": f'pip install "bolo-torch[{b.extra}]"',
                "installed": (root / ".complete").is_file(), "path": str(root), "files": len(b.files()),
                "sources": [{"repo": p.repo, "commit": p.commit, "license": p.license, "license_url": p.license_url}
                            for p in b.parts],
            }
        )  # fmt: skip
    return out
