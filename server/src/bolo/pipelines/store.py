"""Saved pipelines (with every version) and the history of runs, in the data directory:

* ``<data_dir>/pipelines/<id>/<version>.json``: one file per version;
* ``<data_dir>/pipelines/runs/<run_id>/run.json``, and the files the run wrote next to it.

The server and the CLI use the same folders; writes go through a lock file and an atomic replace.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from bolo.errors import BoloError
from bolo.locks import FileLock, replace_file
from bolo.pipelines.definition import PIPELINE_ID_RE, RESERVED_IDS, slug


class PipelineNotFound(BoloError):
    status = 404
    code = "pipeline_not_found"
    type = "invalid_request_error"


class PipelineExists(BoloError):
    status = 409
    code = "pipeline_exists"
    type = "invalid_request_error"


class RunNotFound(BoloError):
    status = 404
    code = "run_not_found"
    type = "invalid_request_error"


def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        replace_file(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp).unlink()
        raise


class PipelineStore:
    def __init__(self, data_dir: Path | str, runs_keep: int = 200) -> None:
        self.root = Path(data_dir) / "pipelines"
        self.runs_root = self.root / "runs"
        self.runs_keep = runs_keep

    def _lock(self) -> FileLock:
        return FileLock(self.root / ".lock", timeout=30.0)

    # -- pipelines -------------------------------------------------------------------------------
    def _dir(self, pid: str) -> Path:
        if not PIPELINE_ID_RE.match(pid) or pid in RESERVED_IDS:
            raise PipelineNotFound(f"no pipeline '{pid}'")
        return self.root / pid

    def versions(self, pid: str) -> list[int]:
        d = self._dir(pid)
        if not d.is_dir():
            return []
        return sorted(int(p.stem) for p in d.glob("*.json") if p.stem.isdigit())

    def get(self, pid: str, version: int | None = None) -> dict[str, Any]:
        found = self.versions(pid)
        if not found:
            raise PipelineNotFound(f"no pipeline '{pid}'")
        v = found[-1] if version is None else version
        if v not in found:
            raise PipelineNotFound(f"pipeline '{pid}' has no version {v} (versions: {', '.join(map(str, found))})")
        doc: dict[str, Any] = json.loads((self._dir(pid) / f"{v}.json").read_text(encoding="utf-8"))
        return doc

    def exists(self, pid: str) -> bool:
        try:
            return bool(self.versions(pid))
        except PipelineNotFound:
            return False

    def all(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        out = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and d.name != "runs" and PIPELINE_ID_RE.match(d.name):
                with contextlib.suppress(PipelineNotFound, OSError, ValueError):
                    out.append(self.get(d.name))
        return out

    def new_id(self, definition: dict[str, Any]) -> str:
        pid = definition.get("id") or slug(str(definition.get("name") or "pipeline"))
        if pid in RESERVED_IDS:
            pid = f"{pid}-pipeline"
        return str(pid)

    def save(self, definition: dict[str, Any], *, create: bool) -> dict[str, Any]:
        """Store ``definition`` (already validated) as a new version; ``create``: refuse an existing id."""
        pid = self.new_id(definition)
        with self._lock():
            versions = self.versions(pid)
            if create and versions:
                raise PipelineExists(f"a pipeline '{pid}' exists already: update it (PUT) or choose another id")
            if not create and not versions:
                raise PipelineNotFound(f"no pipeline '{pid}'")
            now = time.time()
            doc = {k: v for k, v in definition.items() if k not in ("version", "created_at", "updated_at")}
            doc["id"] = pid
            if versions:
                previous = self.get(pid)
                same = {k: v for k, v in previous.items() if k not in ("version", "created_at", "updated_at")}
                if same == doc:
                    return previous
                doc["created_at"] = previous.get("created_at") or _iso(now)
            else:
                doc["created_at"] = _iso(now)
            doc["updated_at"] = _iso(now)
            doc["version"] = (versions[-1] if versions else 0) + 1
            write_json(self._dir(pid) / f"{doc['version']}.json", doc)
            return doc

    def delete(self, pid: str) -> None:
        with self._lock():
            if not self.versions(pid):
                raise PipelineNotFound(f"no pipeline '{pid}'")
            shutil.rmtree(self._dir(pid), ignore_errors=True)

    # -- runs ------------------------------------------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        if not run_id.startswith("run_") or not run_id[4:].isalnum():
            raise RunNotFound(f"no run '{run_id}'")
        return self.runs_root / run_id

    def save_run(self, record: dict[str, Any]) -> None:
        write_json(self.run_dir(record["id"]) / "run.json", record)

    def get_run(self, run_id: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / "run.json"
        try:
            doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise RunNotFound(f"no run '{run_id}'") from None
        return doc

    def runs(self, pipeline: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if not self.runs_root.is_dir():
            return []
        dirs = sorted(self.runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        out: list[dict[str, Any]] = []
        for d in dirs:
            if len(out) >= limit:
                break
            try:
                doc = json.loads((d / "run.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if pipeline and doc.get("pipeline") != pipeline:
                continue
            out.append(doc)
        return out

    def file_path(self, run_id: str, name: str) -> Path:
        from bolo.pipelines.files import check_name

        try:
            check_name(name)
        except BoloError:
            raise RunNotFound(f"run '{run_id}' has no file '{name}'") from None
        path = self.run_dir(run_id) / name
        if not path.is_file():
            raise RunNotFound(f"run '{run_id}' has no file '{name}'")
        return path

    def prune(self) -> None:
        """Keep the newest ``runs_keep`` runs."""
        if not self.runs_root.is_dir():
            return
        dirs = sorted(
            (d for d in self.runs_root.iterdir() if d.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for old in dirs[max(0, self.runs_keep) :]:
            shutil.rmtree(old, ignore_errors=True)
