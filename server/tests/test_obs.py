from __future__ import annotations

import random

import numpy as np
import pytest

from bolo.obs.events import EventBus
from bolo.obs.sketch import RollingQuantiles, quantile, summarize
from bolo.obs.trace import Trace, TraceStore, preview


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_quantiles_match_numpy() -> None:
    rng = random.Random(7)
    values = [rng.lognormvariate(4, 0.6) for _ in range(1500)]
    rq = RollingQuantiles(max_samples=5000)
    for v in values:
        rq.add(v)
    for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0):
        assert rq.quantile(q) == pytest.approx(float(np.quantile(values, q)), rel=1e-9)
    summary = rq.summary()
    assert summary is not None and set(summary) == {"p50", "p95", "p99"}
    assert summary["p95"] == pytest.approx(float(np.quantile(values, 0.95)), abs=0.05)
    assert quantile([], 0.5) is None and quantile([3.0], 0.99) == 3.0
    assert summarize([]) is None
    assert summarize([10, 20, 30, 40, 50]) == {"p50": 30.0, "p95": 48.0}


def test_rolling_window_forgets_old_samples() -> None:
    clock = FakeClock()
    rq = RollingQuantiles(max_samples=10, window_s=60, clock=clock)
    for v in range(100):
        rq.add(float(v))
    assert len(rq) == 10  # bounded by max_samples: only the newest remain
    assert rq.quantile(0.0) == 90.0
    clock.advance(30)
    rq.add(1000.0)
    clock.advance(40)  # the first ten are now older than the window
    assert len(rq) == 1 and rq.quantile(0.5) == 1000.0
    clock.advance(100)
    assert rq.summary() is None
    assert rq.count == 101
    rq.add(float("nan"))
    assert rq.count == 101


def test_trace_phases_use_the_injected_clock() -> None:
    clock = FakeClock()
    trace = Trace("stt", "http", "m", clock=clock)
    clock.advance(0.003)
    with trace.phase("resample"):
        clock.advance(0.012)
    start = clock()
    clock.advance(0.210)
    trace.add_phase("decode", start, clock())
    trace.mark_first_partial()
    clock.advance(0.020)
    trace.mark_final()
    trace.audio_seconds = 4.2
    trace.finish("ok")
    d = trace.to_dict()
    assert [(p["name"], p["start_ms"], p["dur_ms"]) for p in d["phases"]] == [
        ("resample", 3.0, 12.0),
        ("decode", 15.0, 210.0),
    ]
    assert d["first_partial_ms"] == 225.0 and d["final_ms"] == 245.0 and d["total_ms"] == 245.0
    assert d["status"] == "ok" and d["id"].startswith("r_")
    assert d["started_at"].endswith("Z") and "T" in d["started_at"]
    trace.finish("error", "late")  # finishing twice keeps the first outcome
    assert trace.status == "ok"
    trace.mark_first_partial(999)  # first partial is only recorded once
    assert trace.first_partial_ms == pytest.approx(225.0)


def test_trace_restart_and_phase_bound() -> None:
    clock = FakeClock()
    trace = Trace("stt", "websocket", "m", clock=clock)
    clock.advance(5)
    trace.restart()
    clock.advance(0.5)
    assert trace.elapsed_ms() == pytest.approx(500)
    for _ in range(500):
        s = clock()
        clock.advance(0.001)
        trace.add_phase("decode", s, clock())
    assert len(trace.phases) <= 96
    assert sum(p.dur_ms for p in trace.phases) == pytest.approx(500, rel=0.01)  # folded, not dropped


def test_trace_store_stats_and_filters() -> None:
    store = TraceStore(capacity=3)
    seen: list[str] = []
    store.listeners.append(lambda t: seen.append(t.id))
    for i in range(5):
        t = Trace("stt" if i % 2 == 0 else "tts", "http", "a" if i < 4 else "b")
        t.audio_seconds = 2.0
        t.first_partial_ms, t.final_ms, t.rtf = 100.0 + i, 200.0 + i, 0.1
        if t.kind == "tts":
            t.ttfa_ms = 50.0
        t.finish("ok" if i != 3 else "error", None if i != 3 else "boom")
        store.record(t)
    assert len(seen) == 5 and store.total_requests == 5 and store.total_errors == 1
    assert store.audio_seconds_in == 6.0 and store.audio_seconds_out == 4.0
    recent = store.recent()
    assert len(recent) == 3 and recent[0].model == "b"  # newest first, ring buffer of 3
    assert [t.kind for t in store.recent(kind="tts")] == ["tts"]
    assert store.get(recent[0].id) is recent[0] and store.get("nope") is None
    a = store.models["a"]
    assert a.requests == 4 and a.errors == 1
    assert a.final.summary() is not None and len(a.final) == 3  # the failed request is not in the latency sketch
    assert a.latency_summary()["ttfa"] == {"p50": 50.0, "p95": 50.0, "p99": 50.0}
    assert preview("a  b\n c") == "a b c" and len(preview("x" * 200)) == 80


async def test_event_bus_fanout_and_slow_subscriber() -> None:
    import asyncio

    bus = EventBus(max_queue=3)
    bus.bind(asyncio.get_running_loop())
    fast, slow = bus.subscribe(), bus.subscribe()
    first_fast = asyncio.ensure_future(fast.__anext__())
    first_slow = asyncio.ensure_future(slow.__anext__())
    await asyncio.sleep(0)
    assert bus.subscribers == 2
    bus.publish("log", {"n": 0})
    assert (await first_fast) == {"type": "log", "n": 0}
    await first_slow
    for n in range(1, 8):
        bus.publish("log", {"n": n})
    # the slow subscriber kept only the newest three; publishers never blocked
    got = [(await slow.__anext__())["n"] for _ in range(3)]
    assert got == [5, 6, 7]
    await fast.aclose()
    await slow.aclose()
    assert bus.subscribers == 0
    await asyncio.to_thread(bus.publish_threadsafe, "log", {"n": 99})  # no subscribers: harmless
