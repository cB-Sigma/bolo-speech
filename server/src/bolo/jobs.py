"""Background jobs (model pulls, benchmarks) with pollable state."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

_ids = itertools.count(1)


@dataclass
class Job:
    kind: str
    model: str
    id: str = field(default_factory=lambda: f"j_{next(_ids)}")
    state: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    bytes_done: int = 0
    bytes_total: int = 0
    error: str | None = None
    result: dict[str, Any] | None = None
    created: float = field(default_factory=time.time)
    cancel_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "model": self.model,
            "state": self.state,
            "progress": round(self.progress, 4),
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "error": self.error,
        }
        if self.result is not None:
            out["result"] = self.result
        return out


class JobStore:
    def __init__(self, keep: int = 200) -> None:
        self._jobs: dict[str, Job] = {}
        self._keep = keep

    def create(self, kind: str, model: str) -> Job:
        job = Job(kind, model)
        self._jobs[job.id] = job
        if len(self._jobs) > self._keep:
            finished = [j for j in self._jobs.values() if j.state in ("done", "error")]
            for old in finished[: len(self._jobs) - self._keep]:
                self._jobs.pop(old.id, None)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return list(self._jobs.values())

    def active_pull(self, model: str) -> Job | None:
        for job in self._jobs.values():
            if job.kind == "pull" and job.model == model and job.state in ("queued", "running"):
                return job
        return None
