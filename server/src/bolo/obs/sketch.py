"""Rolling-window quantile sketch.

Latency dashboards want "what is p95 right now", not "since the process
started". ``RollingQuantiles`` keeps the most recent ``max_samples``
observations no older than ``window_s`` and answers quantile queries exactly
over that window. The window is small (a few thousand floats per model and
metric), so an exact answer is cheaper and easier to trust than an
approximate sketch.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable


class RollingQuantiles:
    def __init__(
        self,
        max_samples: int = 2048,
        window_s: float = 900.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._samples: deque[tuple[float, float]] = deque(maxlen=max_samples)
        self._window_s = window_s
        self._clock = clock
        self._lock = threading.Lock()
        self.count = 0
        self.total = 0.0

    def add(self, value: float) -> None:
        if math.isnan(value):
            return
        with self._lock:
            self._samples.append((self._clock(), float(value)))
            self.count += 1
            self.total += float(value)

    def _live(self) -> list[float]:
        cutoff = self._clock() - self._window_s
        with self._lock:
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()
            return sorted(v for _, v in self._samples)

    def __len__(self) -> int:
        return len(self._live())

    def quantile(self, q: float) -> float | None:
        return quantile(self._live(), q)

    def summary(self, qs: Iterable[float] = (0.5, 0.95, 0.99), digits: int = 1) -> dict[str, float] | None:
        values = self._live()
        if not values:
            return None
        out: dict[str, float] = {}
        for q in qs:
            v = quantile(values, q)
            assert v is not None
            out[f"p{round(q * 100):d}"] = round(v, digits)
        return out


def quantile(sorted_values: list[float], q: float) -> float | None:
    """Linear-interpolated quantile of an already sorted list (numpy 'linear')."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = min(max(q, 0.0), 1.0) * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def summarize(values: Iterable[float], qs: Iterable[float] = (0.5, 0.95), digits: int = 1) -> dict[str, float] | None:
    ordered = sorted(values)
    if not ordered:
        return None
    out: dict[str, float] = {}
    for q in qs:
        v = quantile(ordered, q)
        assert v is not None
        out[f"p{round(q * 100):d}"] = round(v, digits)
    return out
