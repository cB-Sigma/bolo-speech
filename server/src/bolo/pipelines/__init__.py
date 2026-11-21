"""Pipelines: chains of bolo's operations defined as data, validated, stored with versions and run synchronously
or as jobs with a signed callback (docs/pipelines.md)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any

from bolo.errors import InvalidRequest
from bolo.pipelines.definition import STEP_TYPES, PipelineInvalid, Report, parse_text, validate
from bolo.pipelines.runner import Audio, Run, Runner
from bolo.pipelines.store import PipelineNotFound, PipelineStore, RunNotFound

if TYPE_CHECKING:
    from bolo.core import Bolo

log = logging.getLogger("bolo.pipelines")

__all__ = [
    "STEP_TYPES", "Audio", "PipelineInvalid", "PipelineNotFound", "PipelineService", "PipelineStore", "Report", "Run",
    "RunNotFound", "Runner", "parse_text", "validate",
]  # fmt: skip


class PipelineService:
    """What the HTTP API and the triggers use: the store, the runner, and the runs in progress."""

    def __init__(self, core: Bolo) -> None:
        self.core = core
        self.store = PipelineStore(core.settings.data_dir, core.settings.pipeline_runs_keep)
        self.runner = Runner(core, self.store)
        self.active: dict[str, Run] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._slots: asyncio.Semaphore | None = None
        from bolo.pipelines.triggers import Triggers

        self.triggers = Triggers(core)

    def _semaphore(self) -> asyncio.Semaphore:
        if self._slots is None:
            self._slots = asyncio.Semaphore(max(1, self.core.settings.pipeline_max_jobs))
        return self._slots

    def definition(self, pid: str, version: int | None = None) -> dict[str, Any]:
        return self.store.get(pid, version)

    def create(self, definition: Any) -> dict[str, Any]:
        validate(definition, self.core).raise_if_invalid()
        return self.store.save(definition, create=True)

    def update(self, pid: str, definition: Any) -> dict[str, Any]:
        if not isinstance(definition, dict):
            raise InvalidRequest("a pipeline is an object with steps")
        if definition.get("id") not in (None, pid):
            raise InvalidRequest(f"the body's id '{definition['id']}' is not '{pid}'", param="id")
        definition = {**definition, "id": pid}
        validate(definition, self.core).raise_if_invalid()
        return self.store.save(definition, create=False)

    def get_run(self, run_id: str) -> dict[str, Any]:
        run = self.active.get(run_id)
        if run is not None:
            return run.to_dict()
        return self.store.get_run(run_id)

    async def run(self, run: Run) -> Run:
        """Run now and wait (synchronous runs)."""
        self.active[run.id] = run
        try:
            return await self.runner.execute(run)
        finally:
            self.active.pop(run.id, None)

    def start(self, run: Run, callback_url: str | None = None) -> Run:
        """Queue the run as a job (``GET /api/v1/jobs/{id}``) and return at once."""
        job = self.core.jobs.create("pipeline", run.pipeline or "pipeline")
        run.job_id = job.id
        job.result = {"run_id": run.id}
        self.active[run.id] = run
        if self.store is not None:
            with contextlib.suppress(OSError):
                self.store.save_run(run.to_dict())

        def progress(r: Run) -> None:
            total = max(1, len(r.report.steps) if r.report else 1)
            done = sum(1 for s in r.steps if s.get("depth", 0) == 0 and s["status"] not in ("running",))
            job.progress = min(0.99, done / total)
            self.core.events.publish("job", {"job": job.to_dict()})

        run.on_step = progress

        async def go() -> None:
            try:
                async with self._semaphore():
                    job.state = "running"
                    self.core.events.publish("job", {"job": job.to_dict()})
                    await self.runner.execute(run)
                job.state = "done" if run.status == "ok" else "error"
                job.error = (run.error or {}).get("message") if run.status != "ok" else None
                job.progress = 1.0
            except asyncio.CancelledError:
                job.state, job.error = "error", "cancelled"
                raise
            finally:
                job.result = {"run_id": run.id, "status": run.status}
                self.core.events.publish("job", {"job": job.to_dict()})
                self.active.pop(run.id, None)
                self._tasks.pop(run.id, None)
            if callback_url:
                await self._callback(run, callback_url)

        self._tasks[run.id] = asyncio.create_task(go(), name=f"bolo-pipeline-{run.id}")
        return run

    async def _callback(self, run: Run, url: str) -> None:
        from bolo.pipelines import webhooks

        event = "pipeline.run.completed" if run.status == "ok" else "pipeline.run.failed"
        body = json.dumps(run.to_dict(), ensure_ascii=False).encode()
        try:
            attempts = await webhooks.deliver(self.core.settings, url, body, event=event, run_id=run.id)
        except Exception as exc:  # never let a callback take the server down
            attempts = [{"error": f"{type(exc).__name__}: {exc}"}]
        ok = any(a.get("status", 500) < 400 for a in attempts)
        run.callback = {"url": url, "delivered": ok, "attempts": attempts}
        if not ok:
            log.warning("pipeline run %s: the callback to %s failed", run.id, url)
        with contextlib.suppress(OSError):
            self.store.save_run(run.to_dict())

    async def cancel(self, run_id: str) -> dict[str, Any]:
        task = self._tasks.get(run_id)
        if task is None:
            doc = self.store.get_run(run_id)  # 404 when unknown
            return doc
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        return self.store.get_run(run_id)

    def start_background(self) -> None:
        """Watch the folders of folder triggers (called when the server starts)."""
        self.triggers.start()

    async def close(self) -> None:
        await self.triggers.close()
        for task in list(self._tasks.values()):
            task.cancel()
        for task in list(self._tasks.values()):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
