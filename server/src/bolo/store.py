"""Model store: resumable, checksum-verified downloads into a content-addressed cache.

Layout under the data directory::

    blobs/sha256-<hex>             one file per unique content
    blobs/partial/<key>.part       interrupted downloads (resumed with HTTP Range)
    models/<id>/manifest.json      what the model consists of and where it came from
    models/<id>/<relative path>    hard links (or symlinks/copies) to blobs

Models are fetched from Hugging Face model repositories (the listing API
supplies each file's size and, for LFS files, its SHA-256) or from plain
URLs with an optional checksum in the catalog.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import platform
import shutil
import stat
import sys
import tarfile
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import httpx

from bolo import logs
from bolo.errors import DownloadError, FilesInUse, GatedModel, ModelNotFound
from bolo.stt_options import DECODE_SETTINGS

log = logging.getLogger("bolo.store")


# -- removal that waits out Windows sharing violations ---------------------------------------------------------
#: Windows errors that mean another program has the file open for a moment (ERROR_ACCESS_DENIED,
#: ERROR_SHARING_VIOLATION): an antivirus scanner looking at files that were just written, an indexer, an editor
_SHARING_ERRORS = (5, 32)
#: how long :func:`remove_tree` waits between tries (about 3 s in all)
REMOVE_BACKOFF_S = (0.1, 0.2, 0.4, 0.8, 1.5)


def sharing_violation(exc: BaseException) -> bool:
    """A Windows "in use by another process" error (WinError 32, or 5 for a file being deleted or scanned)."""
    return isinstance(exc, OSError) and getattr(exc, "winerror", None) in _SHARING_ERRORS


def _remove_model_dir(path: Path) -> None:
    """The manifest first, so that a folder only partly deleted is no longer taken for an installed model."""
    (path / "manifest.json").unlink(missing_ok=True)
    shutil.rmtree(path)


def remove_tree(
    path: Path,
    remover: Callable[[Path], None] | None = None,
    sleep: Callable[[float], None] | None = None,
    backoff: tuple[float, ...] = REMOVE_BACKOFF_S,
) -> None:
    """Delete a folder, trying again with a growing pause while Windows says a file in it is in use by another
    process (antivirus scanners hold newly written files for a moment). Raises :class:`FilesInUse`, naming the
    folder and the file, when it is still in use after the last try; other errors as they are."""
    remove = remover or _remove_model_dir
    wait = sleep or time.sleep
    for pause in (*backoff, None):
        try:
            remove(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if not sharing_violation(exc):
                raise
            if pause is None:
                culprit = getattr(exc, "filename", None) or path
                raise FilesInUse(
                    f"a file is in use by another program (WinError {getattr(exc, 'winerror', '?')}: an antivirus "
                    f"scan or an open editor), so the model's folder could not be deleted: {culprit}. What is left "
                    f"of {path} can be deleted by hand once it is closed, or with `bolo rm` again"
                ) from exc
            log.debug("deleting %s: %s in use, trying again in %.1f s", path, exc.filename, pause)
            wait(pause)


# -- folder links (a local model folder registered in the store) ------------------------------------------------
def is_dir_link(path: Path) -> bool:
    """A symbolic link or, on Windows, a directory junction (which ``Path.is_symlink`` does not report)."""
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)  # Python 3.12+
    if isjunction is not None:
        return bool(isjunction(path))
    try:
        tag = getattr(os.lstat(path), "st_reparse_tag", 0)
    except OSError:
        return False
    return bool(tag) and tag == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", -1)


def remove_dir_link(path: Path) -> None:
    """Remove a folder link (symlink or junction), never what it points to."""
    try:
        path.unlink()
    except OSError:
        path.rmdir()  # a junction or a Windows directory symlink


def _create_junction(target: Path, link: Path) -> None:
    """A Windows directory junction: unlike a directory symlink it needs neither Developer Mode nor administrator
    rights (local folders only)."""
    try:
        import _winapi

        _winapi.CreateJunction(str(target), str(link))  # type: ignore[attr-defined,unused-ignore]
    except (ImportError, AttributeError):
        import subprocess

        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True, timeout=30
        )


def link_dir(target: Path, link: Path) -> str:
    """Make ``link`` show the folder ``target``: a directory symlink, else on Windows a junction, else a copy (with a
    notice). Returns which: ``"symlink"``, ``"junction"`` or ``"copy"``.

    On Windows a directory symlink needs Developer Mode or administrator rights (WinError 1314 otherwise)."""
    reason: BaseException
    try:
        link.symlink_to(target, target_is_directory=True)
        return "symlink"
    except OSError as exc:
        reason = exc
    if os.name == "nt":
        try:
            _create_junction(target, link)
            return "junction"
        except Exception as exc:
            reason = exc
    log.warning(
        "could not link %s to %s (%s); copying the folder instead (it takes the disk space twice, and later changes "
        "to %s are not seen)",
        link, target, reason, target,
    )  # fmt: skip
    shutil.copytree(target, link, symlinks=False)
    return "copy"


ProgressFn = Callable[[int, int], None]
CHUNK = 1 << 20


@dataclass
class RemoteFile:
    path: str
    url: str
    size: int | None = None
    sha256: str | None = None
    git_oid: str | None = None  # git blob SHA-1 for small non-LFS files
    extract: bool = False  # a .tar.gz / .tgz / .tar.bz2 / .tar / .zip archive to unpack into the model directory


@dataclass
class Source:
    """Where a model's files come from."""

    repo: str | None = None
    revision: str = "main"
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    subdir: str | None = None  # only this folder of the repo, stripped from local paths
    urls: list[dict[str, Any]] = field(default_factory=list)
    local: str | None = None  # a directory on this machine (``bolo pull ./dir``); nothing is downloaded
    convert: str | None = None  # a step run once after the download ("piper": see bolo.piper)
    #: plain-URL files (for example a release archive) used when the repository cannot be reached
    fallback: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Source:
        return cls(
            repo=d.get("repo"),
            revision=str(d.get("revision", "main")),
            include=list(d.get("include", [])),
            exclude=list(d.get("exclude", [])),
            subdir=d.get("subdir"),
            urls=list(d.get("urls", [])),
            local=d.get("local"),
            convert=d.get("convert"),
            fallback=list(d.get("fallback", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.repo:
            out["repo"] = self.repo
            out["revision"] = self.revision
        for key in ("include", "exclude", "subdir", "urls", "local", "convert", "fallback"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        return out


def _safe_member(name: str) -> str | None:
    """A relative POSIX path that stays inside the destination, or None for anything else."""
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or any(part in ("..", "") for part in path.parts) or not path.parts:
        return None
    if ":" in path.parts[0]:  # a Windows drive
        return None
    return str(path)


def extract_archive(archive: Path, dest: Path, name: str) -> list[str]:
    """Unpack a tar (optionally gzip- or bzip2-compressed) or zip archive into ``dest``; return the relative paths written.

    Only regular files and directories are accepted: links, devices and any path that is absolute or
    climbs out of ``dest`` make the whole archive invalid. A single top-level folder is stripped, so
    ``voice/model.onnx`` and ``model.onnx`` both end up as ``dest/model.onnx``.
    """
    members: list[tuple[str, Any]] = []
    lower = name.lower()
    opened: Any
    if lower.endswith((".zip", ".whl")):  # a Python wheel is a zip archive
        opened = zipfile.ZipFile(archive)
        for info in opened.infolist():
            if info.is_dir():
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise DownloadError(f"{name}: archive member '{info.filename}' is a link; refusing to unpack")
            members.append((info.filename, info))
    elif lower.endswith((".tar.gz", ".tgz", ".tar", ".tar.bz2", ".tbz2")):
        try:
            opened = tarfile.open(archive, "r:*")  # noqa: SIM115 - closed below
        except tarfile.TarError as exc:
            raise DownloadError(f"{name} is not a readable tar archive: {exc}") from exc
        for member in opened:
            if member.isdir():
                continue
            if not member.isfile():
                raise DownloadError(f"{name}: archive member '{member.name}' is not a regular file; refusing to unpack")
            members.append((member.name, member))
    else:
        raise DownloadError(f"{name}: unsupported archive type (use .tar.gz, .tgz, .tar.bz2, .tar, .zip or .whl)")
    try:
        cleaned: list[tuple[str, Any]] = []
        for raw, member in members:
            rel = _safe_member(raw.removeprefix("./"))
            if rel is None:
                raise DownloadError(f"{name}: archive member '{raw}' would be written outside the model directory")
            cleaned.append((rel, member))
        if not cleaned:
            raise DownloadError(f"{name}: the archive is empty")
        tops = {rel.split("/", 1)[0] for rel, _ in cleaned}
        strip = len(tops) == 1 and all("/" in rel for rel, _ in cleaned)
        written: list[str] = []
        root = dest.resolve()
        for rel, member in cleaned:
            rel = rel.split("/", 1)[1] if strip else rel
            target = (dest / rel).resolve()
            if root != target and root not in target.parents:
                raise DownloadError(f"{name}: archive member '{rel}' would be written outside the model directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = opened.open(member) if isinstance(opened, zipfile.ZipFile) else opened.extractfile(member)
            if source is None:
                raise DownloadError(f"{name}: could not read archive member '{rel}'")
            with source, target.open("wb") as out:
                shutil.copyfileobj(source, out, 1 << 20)
            written.append(rel)
        return sorted(written)
    finally:
        opened.close()


def platform_tag() -> str:
    """This machine as a download's ``platform`` names it: ``linux-x86_64``, ``linux-aarch64``, ``darwin-arm64``,
    ``win32-amd64``..."""
    machine = platform.machine().lower() or "unknown"
    machine = {"x86_64": "x86_64", "amd64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(machine, machine)
    system = "linux" if sys.platform.startswith("linux") else sys.platform
    return f"{system}-{machine}"


def platform_matches(wanted: Any, tag: str | None = None) -> bool:
    """Whether a file meant for ``wanted`` (a tag or a list of tags; ``darwin`` alone is any Mac) is for this machine."""
    tag = tag or platform_tag()
    system = tag.split("-", 1)[0]
    tags = [wanted] if isinstance(wanted, str) else list(wanted or [])
    return any(str(t) in (tag, system) for t in tags)


def safe_id(model_id: str) -> str:
    return model_id.replace("/", "--").replace(":", "--")


def git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def match_patterns(path: str, include: list[str], exclude: list[str]) -> bool:
    if include and not any(fnmatch.fnmatch(path, pat) for pat in include):
        return False
    return not any(fnmatch.fnmatch(path, pat) for pat in exclude)


class HubClient:
    """Minimal Hugging Face Hub client (listing and file URLs only)."""

    def __init__(
        self, endpoint: str = "https://huggingface.co", token: str | None = None, timeout: float = 30.0
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.timeout = timeout

    def headers(self) -> dict[str, str]:
        h = {"User-Agent": "bolo"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def client(self) -> httpx.Client:
        return httpx.Client(headers=self.headers(), timeout=self.timeout, follow_redirects=True)

    def _check(self, resp: httpx.Response, repo: str) -> None:
        if resp.status_code in (401, 403):
            raise GatedModel(
                f"Hugging Face denied access to '{repo}' (HTTP {resp.status_code}). The repository is gated or "
                "private: accept its terms on huggingface.co and set BOLO_HF_TOKEN (or HF_TOKEN) to a token "
                "with access."
            )
        if resp.status_code == 404:
            raise ModelNotFound(f"Hugging Face repository or revision not found: '{repo}'")
        if resp.status_code >= 400:
            raise DownloadError(f"Hugging Face returned HTTP {resp.status_code} for '{repo}'")

    def repo_info(self, repo: str) -> dict[str, Any]:
        with self.client() as c:
            resp = c.get(f"{self.endpoint}/api/models/{repo}")
            self._check(resp, repo)
            info: dict[str, Any] = resp.json()
            return info

    def list_files(self, repo: str, revision: str = "main") -> list[RemoteFile]:
        files: list[RemoteFile] = []
        url: str | None = f"{self.endpoint}/api/models/{repo}/tree/{quote(revision, safe='')}?recursive=true&limit=1000"
        with self.client() as c:
            while url:
                try:
                    resp = c.get(url)
                except httpx.HTTPError as exc:
                    raise DownloadError(f"could not reach Hugging Face to list '{repo}': {exc}") from exc
                self._check(resp, repo)
                for item in resp.json():
                    if item.get("type") != "file":
                        continue
                    lfs = item.get("lfs") or {}
                    files.append(
                        RemoteFile(
                            path=item["path"],
                            url=f"{self.endpoint}/{repo}/resolve/{quote(revision, safe='')}/{quote(item['path'])}",
                            size=int(lfs.get("size", item.get("size", 0))),
                            sha256=lfs.get("oid"),
                            git_oid=None if lfs else item.get("oid"),
                        )
                    )
                url = resp.links.get("next", {}).get("url")
        return files


def _file_stamp(path: Path) -> tuple[int, int, int] | None:
    """What a cached copy of ``path`` is checked against: its mtime, size and file id (``None`` if it is missing).
    The mtime alone can stay the same across a rewrite on a coarse clock; the size and file id (a new file for
    every ``os.replace``) catch most such rewrites by another process, and our own writes refresh the cache."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


class ModelStore:
    def __init__(self, data_dir: Path, hub: HubClient | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.blobs = self.data_dir / "blobs"
        self.partial = self.blobs / "partial"
        self.models = self.data_dir / "models"
        self.hub = hub or HubClient()
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        #: presets (``bolo cp``): preset id -> the model whose files it uses
        self.links: dict[str, str] = {}
        #: what deletes a model's folder (tests); None: the manifest, then the folder (:func:`remove_tree`)
        self.remover: Callable[[Path], None] | None = None
        #: held while a model's folder is deleted, and by every manifest read. A read from another thread (the
        #: hardware advisor's background refresh reads every manifest right after a request, so right after a load
        #: that failed with ComponentMissing) had the file open at the moment of the deletion: Windows refused it
        #: (WinError 32 on manifest.json), where a new process removed the folder fine
        self._removing = threading.RLock()
        #: model-settings.json as last read or written, keyed by the file's stamp (see ``_file_stamp``)
        self._settings_cache: tuple[tuple[int, int, int] | None, dict[str, Any]] | None = None

    # -- per-model settings chosen by the user (model-settings.json) ------
    MODEL_SETTINGS = (
        "phonemizer", "text_normalizer", "wordlists", "fallback", "romanised", "device", "keep_alive",
        "num_parallel", "max_queue", "flash_attention", "kv_cache_type", "context_size", "batch_size", "denoise",
        "allow_remote_code", "quantization", "punctuation",
        *DECODE_SETTINGS,
    )  # fmt: skip

    def model_settings(self, model_id: str | None = None) -> dict[str, Any]:
        """Settings pinned for one model (or, without an id, for all of them)."""
        path = self.data_dir / "model-settings.json"
        stamp = _file_stamp(path)
        cached = self._settings_cache
        if cached is not None and cached[0] == stamp:
            data = cached[1]
        else:
            try:
                data = json.loads(path.read_text(encoding="utf-8")) if stamp is not None else {}
            except (OSError, json.JSONDecodeError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            self._settings_cache = (stamp, data)
        if model_id is None:
            return {k: dict(v) if isinstance(v, dict) else v for k, v in data.items()}
        entry = data.get(model_id)
        return dict(entry) if isinstance(entry, dict) else {}

    def set_model_settings(self, model_id: str, values: dict[str, Any]) -> dict[str, Any]:
        """Merge ``values`` into a model's pinned settings (``None`` removes a key); returns the result."""
        with self._guard:
            data = self.model_settings()
            entry = dict(data.get(model_id) or {})
            for key, value in values.items():
                if key not in self.MODEL_SETTINGS:
                    raise KeyError(key)
                if value is None:
                    entry.pop(key, None)
                else:
                    entry[key] = value
            if entry:
                data[model_id] = entry
            else:
                data.pop(model_id, None)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            path = self.data_dir / "model-settings.json"
            tmp = self.data_dir / "model-settings.json.tmp"
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(path)
            # Our own write replaces the cache outright. Going by the file's stamp alone is not enough: on Windows
            # a file's mtime comes from a coarse clock (about 16 ms), so a write right after a read can leave the
            # file with the stamp the reader cached, and the next read would get the old settings back.
            self._settings_cache = (_file_stamp(path), data)
            return dict(entry)

    # -- queries ----------------------------------------------------------
    def model_dir(self, model_id: str) -> Path:
        return self.models / safe_id(self.links.get(model_id, model_id))

    def _read_manifest(self, path: Path) -> dict[str, Any] | None:
        """A manifest, read whole and closed at once, never while a model's folder is being deleted
        (:attr:`_removing`), and on Windows opened so that a deletion is not refused because of it."""
        with self._removing:
            if not path.is_file():
                return None
            try:
                with logs.open_shared(path, binary=True) as fh:
                    raw = fh.read()
                data = json.loads(raw.decode("utf-8"))
            except (OSError, ValueError):
                return None
        return data if isinstance(data, dict) else None

    def manifest(self, model_id: str) -> dict[str, Any] | None:
        return self._read_manifest(self.model_dir(model_id) / "manifest.json")

    def is_installed(self, model_id: str) -> bool:
        m = self.manifest(model_id)
        return bool(m and m.get("complete"))

    def installed(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.models.is_dir():
            return out
        for d in sorted(self.models.iterdir()):
            data = self._read_manifest(d / "manifest.json")
            if data is not None and data.get("complete"):
                out.append(data)
        return out

    def size_on_disk(self, model_id: str) -> int:
        m = self.manifest(model_id)
        return sum(int(f.get("size", 0)) for f in m.get("files", [])) if m else 0

    # -- planning ---------------------------------------------------------
    def plan(self, source: Source) -> list[RemoteFile]:
        """Resolve a source to the concrete list of files to fetch."""
        files: list[RemoteFile] = []
        if source.repo:
            prefix = source.subdir.strip("/") + "/" if source.subdir else ""
            for f in self.hub.list_files(source.repo, source.revision):
                if prefix:
                    if not f.path.startswith(prefix):
                        continue
                    rel = f.path[len(prefix) :]
                else:
                    rel = f.path
                if rel.startswith(".git") or not match_patterns(rel, source.include, source.exclude):
                    continue
                files.append(RemoteFile(rel, f.url, f.size, f.sha256, f.git_oid))
        skipped = []
        for u in source.urls:
            if u.get("platform") and not platform_matches(u["platform"]):
                skipped.append(u["platform"])  # a file built for another platform (a native library)
                continue
            files.append(
                RemoteFile(u["path"], u["url"], u.get("size"), u.get("sha256"), extract=bool(u.get("extract")))
            )
        if not files and skipped:
            published = ", ".join(sorted({t for s in skipped for t in ([s] if isinstance(s, str) else s)}))
            raise DownloadError(f"this model's files are published for {published}, not for {platform_tag()}")
        if not files:
            raise DownloadError("the model source matched no files")
        return files

    # -- download ---------------------------------------------------------
    def _lock(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def _blob_path(self, sha256: str) -> Path:
        return self.blobs / f"sha256-{sha256}"

    def fetch_blob(
        self,
        remote: RemoteFile,
        on_bytes: Callable[[int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        retries: int = 4,
    ) -> tuple[Path, str]:
        """Download one file (resuming a partial one) and return ``(blob path, sha256)``."""
        self.partial.mkdir(parents=True, exist_ok=True)
        if remote.sha256:
            blob = self._blob_path(remote.sha256)
            if blob.is_file() and (remote.size is None or blob.stat().st_size == remote.size):
                if on_bytes:
                    on_bytes(blob.stat().st_size)
                return blob, remote.sha256
        key = hashlib.sha256(f"{remote.url}|{remote.sha256 or remote.git_oid or ''}".encode()).hexdigest()[:32]
        part = self.partial / f"{key}.part"
        with self._lock(key):
            last_error: Exception | None = None
            for attempt in range(retries):
                try:
                    self._download(remote, part, on_bytes, cancelled)
                    break
                except (httpx.HTTPError, OSError) as exc:
                    last_error = exc
                    time.sleep(min(2.0**attempt, 8.0))
            else:
                raise DownloadError(f"download of {remote.path} failed after {retries} attempts: {last_error}")
            digest = sha256_file(part)
            if remote.sha256 and digest != remote.sha256:
                part.unlink(missing_ok=True)
                raise DownloadError(
                    f"checksum mismatch for {remote.path}: expected sha256 {remote.sha256}, got {digest}"
                )
            if remote.git_oid and not remote.sha256 and git_blob_sha1(part) != remote.git_oid:
                part.unlink(missing_ok=True)
                raise DownloadError(f"checksum mismatch for {remote.path}: git object id differs from the listing")
            if remote.size is not None and part.stat().st_size != remote.size:
                part.unlink(missing_ok=True)
                raise DownloadError(f"size mismatch for {remote.path}")
            blob = self._blob_path(digest)
            if blob.exists():
                part.unlink()
            else:
                part.replace(blob)
            return blob, digest

    def _download(
        self,
        remote: RemoteFile,
        part: Path,
        on_bytes: Callable[[int], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        have = part.stat().st_size if part.exists() else 0
        if remote.size is not None and have == remote.size and have > 0:
            if on_bytes:
                on_bytes(have)
            return
        if remote.size is not None and have > remote.size:
            part.unlink()
            have = 0
        headers = self.hub.headers()
        host = httpx.URL(remote.url).host
        hub_host = httpx.URL(self.hub.endpoint).host
        if host not in (hub_host, "huggingface.co") and not host.endswith((".huggingface.co", ".hf.co")):
            headers.pop("Authorization", None)  # the Hugging Face token is for Hugging Face only
        if have:
            headers["Range"] = f"bytes={have}-"
        with httpx.Client(timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True) as client:
            with client.stream("GET", remote.url, headers=headers) as resp:
                if resp.status_code in (401, 403):
                    raise GatedModel(
                        f"access denied (HTTP {resp.status_code}) while downloading {remote.path}; the repository "
                        "is gated or private. Set BOLO_HF_TOKEN to a token with access."
                    )
                if resp.status_code == 416:  # range not satisfiable: start over
                    part.unlink(missing_ok=True)
                    raise httpx.TransportError("stale partial file; restarting")
                if resp.status_code >= 400:
                    raise DownloadError(f"HTTP {resp.status_code} while downloading {remote.url}")
                resumed = resp.status_code == 206
                if have and not resumed:
                    have = 0
                if on_bytes and have:
                    on_bytes(have)
                with part.open("ab" if resumed else "wb") as fh:
                    for chunk in resp.iter_bytes(1 << 16):
                        if cancelled and cancelled():
                            raise DownloadError("download cancelled")
                        fh.write(chunk)
                        if on_bytes:
                            on_bytes(len(chunk))

    def _place(self, blob: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            os.link(blob, dest)
        except OSError:
            try:
                dest.symlink_to(blob)
            except OSError:
                shutil.copy2(blob, dest)

    def pull(
        self,
        model_id: str,
        source: Source,
        progress: ProgressFn | None = None,
        cancelled: Callable[[], bool] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fetch every file of a model and write its manifest. Safe to re-run. A preset fetches its model's files."""
        model_id = self.links.get(model_id, model_id)
        if source.local:
            path = Path(source.local)
            if not path.is_dir():
                raise DownloadError(f"the model directory {source.local} does not exist any more")
            mdir = self.model_dir(model_id)
            if is_dir_link(mdir) and mdir.resolve() != path.resolve():
                remove_dir_link(mdir)
            registered = self.register_local(model_id, path, meta)
            if progress:
                progress(registered["size_bytes"], registered["size_bytes"])
            return registered
        if source.repo and source.fallback:
            try:
                return self.pull(model_id, replace(source, fallback=[]), progress, cancelled, meta)
            except GatedModel:
                raise
            except (DownloadError, httpx.HTTPError) as exc:
                log.warning("%s: %s; using the fallback source", model_id, exc)
                return self.pull(model_id, Source(urls=source.fallback), progress, cancelled, meta)
        with self._lock("model:" + model_id):
            files = self.plan(source)
            total = sum(f.size or 0 for f in files)
            done = 0
            lock = threading.Lock()

            def bump(n: int) -> None:
                nonlocal done
                with lock:
                    done += n
                    if progress:
                        progress(done, total)

            if progress:
                progress(0, total)
            mdir = self.model_dir(model_id)
            mdir.mkdir(parents=True, exist_ok=True)
            entries: list[dict[str, Any]] = []
            for f in files:
                blob, digest = self.fetch_blob(f, on_bytes=bump, cancelled=cancelled)
                if f.extract:
                    # the unpacked files are what the model consists of; the archive itself is not kept
                    for rel in extract_archive(blob, mdir, f.path):
                        entries.append({"path": rel, "size": (mdir / rel).stat().st_size})
                    meta = {**(meta or {}), "archive": {"url": f.url, "sha256": digest, "size": blob.stat().st_size}}
                    blob.unlink(missing_ok=True)
                    continue
                self._place(blob, mdir / f.path)
                entries.append({"path": f.path, "size": blob.stat().st_size, "sha256": digest})
            if source.convert:
                if source.convert != "piper":
                    raise DownloadError(f"unknown conversion step '{source.convert}'")
                from bolo.piper import convert_piper_dir

                converted = convert_piper_dir(mdir)
                meta = {**(meta or {}), "piper": converted}
                kept = {e["path"]: e for e in entries}
                entries = []
                for p in sorted(mdir.rglob("*")):
                    rel = str(p.relative_to(mdir))
                    if p.is_file() and rel not in ("manifest.json", "manifest.json.tmp"):
                        entries.append(kept.get(rel) or {"path": rel, "size": p.stat().st_size})
            manifest: dict[str, Any] = {
                "id": model_id,
                "source": source.to_dict(),
                "files": entries,
                "size_bytes": sum(e["size"] for e in entries),
                "installed_at": time.time(),
                "complete": True,
            }
            if meta:
                manifest["meta"] = meta
            tmp = mdir / "manifest.json.tmp"
            tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            tmp.replace(mdir / "manifest.json")
            return manifest

    def register_local(self, model_id: str, path: Path, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """Register a model directory that already exists on disk (no download)."""
        mdir = self.model_dir(model_id)
        mdir.parent.mkdir(parents=True, exist_ok=True)
        if mdir.resolve() != path.resolve():
            if mdir.exists() or is_dir_link(mdir):
                raise DownloadError(f"model '{model_id}' already exists in the store")
            if link_dir(path.resolve(), mdir) == "junction":
                log.info("%s: linked %s to %s with a directory junction", model_id, mdir, path)
        entries: list[dict[str, Any]] = [
            {"path": str(p.relative_to(path)), "size": p.stat().st_size}
            for p in sorted(path.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        ]
        manifest: dict[str, Any] = {
            "id": model_id,
            "source": {"local": str(path)},
            "files": entries,
            "size_bytes": sum(e["size"] for e in entries),
            "installed_at": time.time(),
            "complete": True,
            "meta": meta or {},
        }
        (mdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    # -- removal ----------------------------------------------------------
    def remove(self, model_id: str) -> bool:
        if model_id in self.links:
            return False  # a preset has no files of its own
        mdir = self.model_dir(model_id)
        if not (mdir.exists() or is_dir_link(mdir)):
            return False
        with self._removing:
            if is_dir_link(mdir):
                target = mdir.resolve()
                remove_dir_link(mdir)
                (target / "manifest.json").unlink(missing_ok=True)
            else:
                remove_tree(mdir, self.remover)
        self.gc()
        return True

    def gc(self) -> int:
        """Delete blobs no model links to any more. Returns bytes freed."""
        if not self.blobs.is_dir():
            return 0
        referenced: set[str] = set()
        for m in self.installed():
            referenced.update(f["sha256"] for f in m.get("files", []) if "sha256" in f)
        freed = 0
        for blob in self.blobs.glob("sha256-*"):
            digest = blob.name.removeprefix("sha256-")
            if digest not in referenced:
                try:
                    size = blob.stat().st_size
                    blob.unlink()
                except OSError as exc:  # in use (Windows) or gone: the next gc tries again
                    log.debug("could not delete the blob %s: %s", blob.name, exc)
                    continue
                freed += size
        return freed
