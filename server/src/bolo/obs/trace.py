"""Per-request traces with phase timings, and the rolling statistics built on them.

All durations come from one monotonic clock (``time.perf_counter`` by
default, injectable for tests). A trace is cheap: a handful of floats per
phase. The store keeps the most recent traces in memory only.
"""

from __future__ import annotations

import datetime as dt
import itertools
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from bolo.obs.sketch import RollingQuantiles

STT_PHASES = ("queue", "resample", "vad", "decode", "finalize")
TTS_PHASES = ("queue", "normalize", "synthesize_first", "synthesize_rest", "encode")
MAX_PHASES = 96

_ids = itertools.count(1)


def _now_iso() -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


@dataclass
class Phase:
    name: str
    start_ms: float
    dur_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "start_ms": round(self.start_ms, 2), "dur_ms": round(self.dur_ms, 2)}


@dataclass
class Trace:
    kind: str  # "stt" | "tts" | "vad"
    transport: str  # "http" | "websocket" | "wyoming" | "cli" | "bench"
    model: str
    clock: Callable[[], float] = time.perf_counter
    id: str = field(default_factory=lambda: f"r_{next(_ids):06d}")
    started_at: str = field(default_factory=_now_iso)
    #: wall-clock start in nanoseconds since the epoch (where OpenTelemetry spans are placed)
    started_ns: int = field(default_factory=time.time_ns)
    status: str = "running"
    error: str | None = None
    audio_seconds: float | None = None
    chars: int | None = None
    language: str | None = None
    phases: list[Phase] = field(default_factory=list)
    first_partial_ms: float | None = None
    final_ms: float | None = None
    ttfa_ms: float | None = None
    total_ms: float | None = None
    rtf: float | None = None
    batch_size: int | None = None
    text_preview: str | None = None
    compute_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self._t0 = self.clock()

    # -- timing -----------------------------------------------------------
    def now(self) -> float:
        return self.clock()

    def elapsed_ms(self) -> float:
        return (self.clock() - self._t0) * 1000.0

    def restart(self) -> None:
        """Move the origin to now (e.g. once an upload has been fully received)."""
        self._t0 = self.clock()
        self.started_at = _now_iso()
        self.started_ns = time.time_ns()

    def add_phase(self, name: str, start: float, end: float) -> None:
        """Record a phase from absolute clock readings ``start`` and ``end``."""
        start_ms = (start - self._t0) * 1000.0
        dur_ms = max(0.0, (end - start) * 1000.0)
        if self.phases and self.phases[-1].name == name and len(self.phases) >= MAX_PHASES:
            last = self.phases[-1]
            last.dur_ms = start_ms + dur_ms - last.start_ms
            return
        if len(self.phases) >= MAX_PHASES:
            # Keep the trace bounded on very long streams: fold into the last phase of that name.
            for p in reversed(self.phases):
                if p.name == name:
                    p.dur_ms += dur_ms
                    return
            return
        self.phases.append(Phase(name, start_ms, dur_ms))

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = self.clock()
        try:
            yield
        finally:
            self.add_phase(name, start, self.clock())

    def compact(self) -> None:
        """Merge repeated phases into one per name (start of the first, summed duration).

        A realtime stream decodes many times; its trace reports where the time went in total.
        """
        merged: dict[str, Phase] = {}
        for p in self.phases:
            if p.name in merged:
                merged[p.name].dur_ms += p.dur_ms
            else:
                merged[p.name] = Phase(p.name, p.start_ms, p.dur_ms)
        self.phases = sorted(merged.values(), key=lambda p: p.start_ms)

    def mark_first_partial(self, value_ms: float | None = None) -> None:
        if self.first_partial_ms is None:
            self.first_partial_ms = self.elapsed_ms() if value_ms is None else value_ms

    def mark_final(self, value_ms: float | None = None) -> None:
        self.final_ms = self.elapsed_ms() if value_ms is None else value_ms

    def mark_ttfa(self, value_ms: float | None = None) -> None:
        if self.ttfa_ms is None:
            self.ttfa_ms = self.elapsed_ms() if value_ms is None else value_ms

    def note_batch(self, size: int) -> None:
        self.batch_size = max(self.batch_size or 0, size)

    def finish(self, status: str = "ok", error: str | None = None) -> None:
        if self.status != "running":
            return
        self.status = status
        self.error = error
        self.total_ms = self.elapsed_ms()

    def to_dict(self) -> dict[str, Any]:
        def r(v: float | None, digits: int = 1) -> float | None:
            return None if v is None else round(v, digits)

        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "transport": self.transport,
            "model": self.model,
            "started_at": self.started_at,
            "status": self.status,
            "error": self.error,
            "audio_seconds": r(self.audio_seconds, 3),
            "chars": self.chars,
            "language": self.language,
            "phases": [p.to_dict() for p in self.phases],
            "first_partial_ms": r(self.first_partial_ms),
            "final_ms": r(self.final_ms),
            "ttfa_ms": r(self.ttfa_ms),
            "total_ms": r(self.total_ms),
            "rtf": r(self.rtf, 4),
            "batch_size": self.batch_size,
            "text_preview": self.text_preview,
        }
        out.update(self.extra)
        return out


def preview(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ModelStats:
    def __init__(self, model: str, task: str) -> None:
        self.model = model
        self.task = task
        self.requests = 0
        self.errors = 0
        self.first_partial = RollingQuantiles()
        self.final = RollingQuantiles()
        self.ttfa = RollingQuantiles()
        self.rtf = RollingQuantiles()
        self.queue = RollingQuantiles()

    def latency_summary(self) -> dict[str, Any]:
        return {
            "first_partial": self.first_partial.summary(),
            "final": self.final.summary(),
            "ttfa": self.ttfa.summary(),
        }

    def rtf_summary(self) -> dict[str, float] | None:
        return self.rtf.summary(qs=(0.5, 0.95), digits=4)


class TraceStore:
    """Ring buffer of finished traces plus per-model rolling statistics."""

    def __init__(self, capacity: int = 1000) -> None:
        self._traces: deque[Trace] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.models: dict[str, ModelStats] = {}
        self.total_requests = 0
        self.total_errors = 0
        self.audio_seconds_in = 0.0
        self.audio_seconds_out = 0.0
        self.listeners: list[Callable[[Trace], None]] = []

    def stats_for(self, model: str, task: str) -> ModelStats:
        with self._lock:
            st = self.models.get(model)
            if st is None:
                st = self.models[model] = ModelStats(model, task)
            return st

    def observe_latency(
        self,
        model: str,
        task: str,
        *,
        first_partial_ms: float | None = None,
        final_ms: float | None = None,
        ttfa_ms: float | None = None,
        rtf: float | None = None,
        queue_ms: float | None = None,
    ) -> None:
        """Feed the rolling sketches directly (streams report per utterance)."""
        st = self.stats_for(model, task)
        if first_partial_ms is not None:
            st.first_partial.add(first_partial_ms)
        if final_ms is not None:
            st.final.add(final_ms)
        if ttfa_ms is not None:
            st.ttfa.add(ttfa_ms)
        if rtf is not None:
            st.rtf.add(rtf)
        if queue_ms is not None:
            st.queue.add(queue_ms)

    def record(self, trace: Trace, *, observe: bool = True) -> None:
        """Store a finished trace. ``observe=False`` when sketches were fed per utterance."""
        if trace.status == "running":
            trace.finish()
        st = self.stats_for(trace.model, trace.kind)
        with self._lock:
            self._traces.append(trace)
            self.total_requests += 1
            st.requests += 1
            if trace.status != "ok":
                self.total_errors += 1
                st.errors += 1
            if trace.audio_seconds:
                if trace.kind == "tts":
                    self.audio_seconds_out += trace.audio_seconds
                else:
                    self.audio_seconds_in += trace.audio_seconds
        if observe and trace.status == "ok":
            self.observe_latency(
                trace.model,
                trace.kind,
                first_partial_ms=trace.first_partial_ms,
                final_ms=trace.final_ms,
                ttfa_ms=trace.ttfa_ms,
                rtf=trace.rtf,
            )
        for listener in list(self.listeners):
            listener(trace)

    def recent(self, limit: int = 100, kind: str | None = None, model: str | None = None) -> list[Trace]:
        with self._lock:
            items = list(self._traces)
        out: list[Trace] = []
        for t in reversed(items):
            if kind and t.kind != kind:
                continue
            if model and t.model != model:
                continue
            out.append(t)
            if len(out) >= limit:
                break
        return out

    def get(self, trace_id: str) -> Trace | None:
        with self._lock:
            for t in self._traces:
                if t.id == trace_id:
                    return t
        return None
