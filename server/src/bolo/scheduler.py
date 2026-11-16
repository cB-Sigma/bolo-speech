"""Per-model work queues with dynamic micro-batching.

Every loaded model owns one :class:`ModelQueue`. Requests and streaming
decode steps are submitted as jobs; a small pool of worker threads drains
the queue. Batching is *natural*: while a worker is busy, jobs pile up, and
the next free worker takes as many compatible jobs as the engine can decode
in one call (``max_batch``). An idle server therefore adds no latency, and
a busy one trades a little waiting for much higher throughput where the
engine has a batched decode. ``max_wait_ms`` optionally lingers for more
jobs, which only helps on hardware where a batch costs little more than a
single item.

The queue is bounded: when it is full, ``submit`` raises
:class:`~bolo.errors.Overloaded` immediately (HTTP 429 with ``Retry-After``)
instead of letting latency grow without limit. Jobs whose caller has gone
away (client disconnect) are dropped before they reach the engine.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from bolo.errors import Overloaded
from bolo.obs.sketch import RollingQuantiles

Handler = Callable[[list[Any]], list[Any]]


@dataclass
class JobInfo:
    queue_ms: float
    run_ms: float
    batch_size: int
    started: float  # clock reading when the engine call began
    finished: float


@dataclass
class _Job:
    kind: str
    payload: Any
    future: asyncio.Future[tuple[Any, JobInfo]]
    enqueued: float
    cancelled: bool = False


@dataclass
class QueueStats:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    cancelled: int = 0
    batches: int = 0
    batched_jobs: int = 0
    max_batch_seen: int = 0
    batch_sizes: dict[int, int] = field(default_factory=dict)
    queue_ms: RollingQuantiles = field(default_factory=RollingQuantiles)
    run_ms: RollingQuantiles = field(default_factory=RollingQuantiles)

    @property
    def mean_batch(self) -> float:
        return self.batched_jobs / self.batches if self.batches else 0.0


class ModelQueue:
    def __init__(
        self,
        name: str,
        handlers: dict[str, Handler],
        *,
        workers: int = 1,
        max_batch: dict[str, int] | int = 1,
        max_wait_ms: float = 0.0,
        max_queue: int = 64,
        queue_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.name = name
        self._handlers = handlers
        self._workers_n = max(1, workers)
        self._max_batch = max_batch
        self._max_wait = max_wait_ms / 1000.0
        self._max_queue = max_queue
        self._queue_timeout = queue_timeout_s
        self._clock = clock
        self._pending: deque[_Job] = deque()
        self._cond: asyncio.Condition | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._executor = ThreadPoolExecutor(max_workers=self._workers_n, thread_name_prefix=f"bolo-{name}")
        self._closed = False
        self.inflight = 0
        self.stats = QueueStats()

    # -- introspection ----------------------------------------------------
    @property
    def depth(self) -> int:
        return len(self._pending)

    @property
    def workers(self) -> int:
        return self._workers_n

    def handles(self, kind: str) -> bool:
        """True when this model's queue runs jobs of ``kind`` (``embed``: a diarizer with a speaker embedder)."""
        return kind in self._handlers

    @property
    def max_queue(self) -> int:
        return self._max_queue

    def batch_limit(self, kind: str) -> int:
        if isinstance(self._max_batch, int):
            return max(1, self._max_batch)
        return max(1, self._max_batch.get(kind, 1))

    # -- lifecycle --------------------------------------------------------
    def _ensure_started(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
            self._tasks = [
                asyncio.create_task(self._worker(), name=f"bolo-queue-{self.name}-{i}") for i in range(self._workers_n)
            ]
        return self._cond

    async def close(self) -> None:
        self._closed = True
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        for job in self._pending:
            if not job.future.done():
                job.future.set_exception(Overloaded(f"model '{self.name}' was unloaded"))
        self._pending.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- submission -------------------------------------------------------
    async def submit(self, kind: str, payload: Any) -> tuple[Any, JobInfo]:
        """Queue one job and wait for its result. Raises ``Overloaded`` when full."""
        if self._closed:
            raise Overloaded(f"model '{self.name}' is shutting down")
        if kind not in self._handlers:
            raise ValueError(f"queue '{self.name}' has no handler for {kind!r}")
        cond = self._ensure_started()
        if len(self._pending) >= self._max_queue:
            self.stats.rejected += 1
            run = self.stats.run_ms.quantile(0.5) or 100.0
            retry = max(0.1, len(self._pending) * run / 1000.0 / self._workers_n)
            raise Overloaded(
                f"model '{self.name}' is overloaded: {len(self._pending)} requests are already queued",
                retry_after_s=round(min(retry, 30.0), 1),
            )
        loop = asyncio.get_running_loop()
        job = _Job(kind, payload, loop.create_future(), self._clock())
        self._pending.append(job)
        self.stats.submitted += 1
        async with cond:
            cond.notify()
        try:
            return await job.future
        except asyncio.CancelledError:
            job.cancelled = True
            with contextlib.suppress(ValueError):
                self._pending.remove(job)
            self.stats.cancelled += 1
            if not job.future.done():
                job.future.cancel()
            raise

    # -- workers ----------------------------------------------------------
    def _take_same_kind(self, batch: list[_Job], limit: int) -> None:
        if len(batch) >= limit or not self._pending:
            return
        kind = batch[0].kind
        keep: deque[_Job] = deque()
        while self._pending and len(batch) < limit:
            job = self._pending.popleft()
            if job.cancelled:
                continue
            if job.kind == kind:
                batch.append(job)
            else:
                keep.append(job)
        keep.extend(self._pending)
        self._pending = keep

    async def _next_batch(self) -> list[_Job]:
        cond = self._cond
        assert cond is not None
        async with cond:
            while not self._pending:
                await cond.wait()
            first = self._pending.popleft()
        if first.cancelled:
            return []
        batch = [first]
        limit = self.batch_limit(first.kind)
        self._take_same_kind(batch, limit)
        if self._max_wait > 0 and len(batch) < limit:
            deadline = self._clock() + self._max_wait
            while len(batch) < limit:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    break
                async with cond:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(cond.wait(), timeout=remaining)
                self._take_same_kind(batch, limit)
        return batch

    def _expire(self, batch: list[_Job]) -> list[_Job]:
        if self._queue_timeout <= 0:
            return batch
        now = self._clock()
        live: list[_Job] = []
        for job in batch:
            if now - job.enqueued > self._queue_timeout:
                self.stats.rejected += 1
                if not job.future.done():
                    job.future.set_exception(
                        Overloaded(f"request waited more than {self._queue_timeout:.0f}s in the queue of '{self.name}'")
                    )
            else:
                live.append(job)
        return live

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = [j for j in await self._next_batch() if not j.cancelled and not j.future.done()]
            batch = self._expire(batch)
            if not batch:
                continue
            await self._run(loop, batch)

    async def _run(self, loop: asyncio.AbstractEventLoop, batch: list[_Job]) -> None:
        handler = self._handlers[batch[0].kind]
        started = self._clock()
        self.inflight += len(batch)
        try:
            try:
                results = await loop.run_in_executor(self._executor, handler, [j.payload for j in batch])
                if len(results) != len(batch):
                    raise RuntimeError(f"engine returned {len(results)} results for a batch of {len(batch)}")
                errors: list[BaseException | None] = [None] * len(batch)
            except Exception as exc:
                if len(batch) == 1:
                    results, errors = [None], [exc]
                else:
                    # One bad request must not fail its batch-mates: retry them one by one.
                    results, errors = [], []
                    for job in batch:
                        try:
                            res = await loop.run_in_executor(self._executor, handler, [job.payload])
                            results.append(res[0])
                            errors.append(None)
                        except Exception as single_exc:
                            results.append(None)
                            errors.append(single_exc)
            finished = self._clock()
        finally:
            self.inflight -= len(batch)
        st = self.stats
        st.batches += 1
        st.batched_jobs += len(batch)
        st.max_batch_seen = max(st.max_batch_seen, len(batch))
        st.batch_sizes[len(batch)] = st.batch_sizes.get(len(batch), 0) + 1
        run_ms = (finished - started) * 1000.0
        st.run_ms.add(run_ms)
        for job, result, error in zip(batch, results, errors):
            queue_ms = (started - job.enqueued) * 1000.0
            st.queue_ms.add(queue_ms)
            if job.future.done():
                continue
            if error is not None:
                st.failed += 1
                job.future.set_exception(error)
            else:
                st.completed += 1
                job.future.set_result((result, JobInfo(queue_ms, run_ms, len(batch), started, finished)))
