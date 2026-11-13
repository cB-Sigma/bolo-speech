from __future__ import annotations

import asyncio
import threading
import time

import pytest

from bolo.errors import Overloaded
from bolo.scheduler import ModelQueue


class Recorder:
    """A handler with a fixed cost per call that records every batch it is given."""

    def __init__(self, delay: float = 0.05, fail_on: object = None) -> None:
        self.delay = delay
        self.fail_on = fail_on
        self.batches: list[list[object]] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def __call__(self, items: list[object]) -> list[object]:
        with self._lock:
            self.batches.append(list(items))
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            time.sleep(self.delay)
            if self.fail_on is not None and self.fail_on in items:
                raise RuntimeError(f"bad item {self.fail_on}")
            return [f"ok:{i}" for i in items]
        finally:
            with self._lock:
                self.concurrent -= 1


async def test_idle_queue_runs_immediately_without_batching_delay() -> None:
    rec = Recorder(0.03)
    q = ModelQueue("m", {"batch": rec}, max_batch=8)
    result, info = await q.submit("batch", 1)
    assert result == "ok:1"
    assert info.batch_size == 1
    assert info.queue_ms < 15
    assert info.run_ms == pytest.approx(30, abs=15)
    await q.close()


async def test_jobs_that_arrive_while_busy_are_batched_together() -> None:
    rec = Recorder(0.08)
    q = ModelQueue("m", {"batch": rec}, max_batch=4, workers=1)
    first = asyncio.ensure_future(q.submit("batch", 0))
    await asyncio.sleep(0.02)  # the worker is now busy with job 0
    rest = [asyncio.ensure_future(q.submit("batch", i)) for i in range(1, 7)]
    results = await asyncio.gather(first, *rest)
    assert [r[0] for r in results] == [f"ok:{i}" for i in range(7)]
    assert [len(b) for b in rec.batches] == [1, 4, 2]  # natural batching, capped at max_batch
    assert rec.batches[1] == [1, 2, 3, 4]  # FIFO order is preserved
    assert results[1][1].batch_size == 4
    # queue time is measured from submission to the start of the engine call
    assert results[1][1].queue_ms == pytest.approx(60, abs=25)
    assert results[5][1].queue_ms == pytest.approx(140, abs=35)
    st = q.stats
    assert st.batches == 3 and st.batched_jobs == 7 and st.max_batch_seen == 4
    assert st.mean_batch == pytest.approx(7 / 3)
    await q.close()


async def test_max_batch_one_never_batches() -> None:
    rec = Recorder(0.02)
    q = ModelQueue("m", {"tts": rec}, max_batch=1)
    await asyncio.gather(*[q.submit("tts", i) for i in range(5)])
    assert all(len(b) == 1 for b in rec.batches) and len(rec.batches) == 5
    await q.close()


async def test_kinds_are_not_mixed_in_a_batch() -> None:
    a, b = Recorder(0.05), Recorder(0.05)
    q = ModelQueue("m", {"batch": a, "stream": b}, max_batch={"batch": 8, "stream": 8})
    first = asyncio.ensure_future(q.submit("batch", "a0"))
    await asyncio.sleep(0.01)
    futs = [
        asyncio.ensure_future(q.submit(k, v))
        for k, v in [("stream", "s1"), ("batch", "a1"), ("stream", "s2"), ("batch", "a2")]
    ]
    await asyncio.gather(first, *futs)
    assert a.batches == [["a0"], ["a1", "a2"]]
    assert b.batches == [["s1", "s2"]]
    assert q.batch_limit("batch") == 8 and q.batch_limit("other") == 1
    with pytest.raises(ValueError, match="no handler"):
        await q.submit("nope", 1)
    await q.close()


async def test_linger_waits_for_a_fuller_batch() -> None:
    rec = Recorder(0.01)
    q = ModelQueue("m", {"batch": rec}, max_batch=3, max_wait_ms=80)
    t0 = time.perf_counter()
    futs = [asyncio.ensure_future(q.submit("batch", 0))]
    await asyncio.sleep(0.02)
    futs.append(asyncio.ensure_future(q.submit("batch", 1)))
    await asyncio.sleep(0.02)
    futs.append(asyncio.ensure_future(q.submit("batch", 2)))
    await asyncio.gather(*futs)
    assert [len(b) for b in rec.batches] == [3]  # the batch filled before the 80 ms linger ended
    assert time.perf_counter() - t0 < 0.075
    await q.submit("batch", 9)  # a lone job waits out the linger
    assert q.stats.queue_ms.quantile(1.0) == pytest.approx(80, abs=25)
    await q.close()


async def test_bounded_queue_rejects_with_retry_hint() -> None:
    rec = Recorder(0.2)
    q = ModelQueue("m", {"batch": rec}, max_batch=1, max_queue=2)
    running = asyncio.ensure_future(q.submit("batch", 0))
    await asyncio.sleep(0.02)
    waiting = [asyncio.ensure_future(q.submit("batch", i)) for i in (1, 2)]
    await asyncio.sleep(0)
    assert q.depth == 2 and q.inflight == 1
    with pytest.raises(Overloaded) as exc:
        await q.submit("batch", 3)
    assert exc.value.status == 429 and exc.value.retry_after_s > 0
    assert q.stats.rejected == 1
    await asyncio.gather(running, *waiting)  # accepted work still completes
    assert q.depth == 0
    await q.close()


async def test_cancelled_jobs_never_reach_the_engine() -> None:
    rec = Recorder(0.1)
    q = ModelQueue("m", {"batch": rec}, max_batch=1)
    running = asyncio.ensure_future(q.submit("batch", "keep"))
    await asyncio.sleep(0.02)
    doomed = asyncio.ensure_future(q.submit("batch", "drop"))
    later = asyncio.ensure_future(q.submit("batch", "after"))
    await asyncio.sleep(0.02)
    doomed.cancel()  # the client went away while its job was queued
    with pytest.raises(asyncio.CancelledError):
        await doomed
    await asyncio.gather(running, later)
    assert rec.batches == [["keep"], ["after"]]
    assert q.stats.cancelled == 1
    await q.close()


async def test_one_bad_request_does_not_fail_its_batch_mates() -> None:
    rec = Recorder(0.05, fail_on="bad")
    q = ModelQueue("m", {"batch": rec}, max_batch=4)
    first = asyncio.ensure_future(q.submit("batch", "a"))
    await asyncio.sleep(0.01)
    futs = [asyncio.ensure_future(q.submit("batch", x)) for x in ("b", "bad", "c")]
    results = await asyncio.gather(first, *futs, return_exceptions=True)
    assert results[0][0] == "ok:a" and results[1][0] == "ok:b" and results[3][0] == "ok:c"
    assert isinstance(results[2], RuntimeError)
    assert q.stats.failed == 1 and q.stats.completed == 3
    await q.close()


async def test_workers_run_in_parallel_up_to_the_pool_size() -> None:
    rec = Recorder(0.1)
    q = ModelQueue("m", {"tts": rec}, max_batch=1, workers=3)
    t0 = time.perf_counter()
    await asyncio.gather(*[q.submit("tts", i) for i in range(6)])
    elapsed = time.perf_counter() - t0
    assert rec.max_concurrent == 3
    assert 0.18 < elapsed < 0.4  # two rounds of three, not six in series
    await q.close()


async def test_queue_timeout_fails_stale_jobs() -> None:
    rec = Recorder(0.15)
    q = ModelQueue("m", {"batch": rec}, max_batch=1, queue_timeout_s=0.1)
    running = asyncio.ensure_future(q.submit("batch", 0))
    await asyncio.sleep(0.01)
    stale = asyncio.ensure_future(q.submit("batch", 1))
    await running
    with pytest.raises(Overloaded, match="waited more than"):
        await stale
    assert rec.batches == [[0]]
    await q.close()


async def test_close_fails_pending_jobs() -> None:
    rec = Recorder(0.3)
    q = ModelQueue("m", {"batch": rec}, max_batch=1)
    running = asyncio.ensure_future(q.submit("batch", 0))
    await asyncio.sleep(0.02)
    pending = asyncio.ensure_future(q.submit("batch", 1))
    await asyncio.sleep(0.01)
    await q.close()
    with pytest.raises(Overloaded, match="unloaded"):
        await pending
    running.cancel()
    with pytest.raises(Overloaded):
        await q.submit("batch", 2)
