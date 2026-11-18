"""Audio classification over a recording: sound-event tags (task ``audio-tagging``) and speech emotion (task
``emotion``), for the whole recording and, with ``window``, window by window (sound event detection over time).

A classifier hears at most 30 s at once. A longer recording without ``window`` is heard in 10 s windows and the
results are combined: for tags, each class keeps its highest probability in any window (a dog that barks once is
in the recording); for emotion, the probabilities are averaged over the windows that hold sound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.audio.resample import resample
from bolo.engines.base import FloatArray
from bolo.errors import InvalidRequest

if TYPE_CHECKING:
    from bolo.core import Bolo
    from bolo.obs.trace import Trace

RATE = 16000
#: the longest stretch a classifier hears at once
MAX_S = 30.0
#: windows a long recording is heard in when no ``window`` is asked for
LONG_WINDOW_S = 10.0
#: at most this many windows per request
MAX_WINDOWS = 2000


@dataclass
class Window:
    start: float
    end: float
    scores: list[tuple[str, float]]


@dataclass
class Classification:
    model: str
    task: str
    duration: float
    scores: list[tuple[str, float]]
    windows: list[Window] = field(default_factory=list)


def _windows(n: int, window_s: float, hop_s: float) -> list[tuple[int, int]]:
    size, hop = int(window_s * RATE), max(1, int(hop_s * RATE))
    if n <= size:
        return [(0, n)]
    out = [(a, min(n, a + size)) for a in range(0, n - size // 4, hop)]
    if len(out) > MAX_WINDOWS:
        raise InvalidRequest(
            f"{len(out)} windows: at most {MAX_WINDOWS} per request; use a longer window or hop", param="window"
        )
    return out


def _combine(task: str, results: list[list[tuple[str, float]]], loud: list[bool]) -> list[tuple[str, float]]:
    if len(results) == 1:
        return results[0]
    labels = [label for label, _ in results[0]]
    table = np.array([[dict(r).get(label, 0.0) for label in labels] for r in results])
    if task == "audio-tagging":
        combined = table.max(axis=0)
    else:
        keep = np.array(loud) if any(loud) else np.ones(len(results), dtype=bool)
        combined = table[keep].mean(axis=0)
    out = [(label, float(v)) for label, v in zip(labels, combined)]
    out.sort(key=lambda t: -t[1])
    return out


async def classify(
    core: Bolo,
    samples: FloatArray,
    rate: int,
    model: str,
    task: str,
    trace: Trace,
    window: float | None = None,
    hop: float | None = None,
) -> Classification:
    """Classify ``samples`` with the ``task`` model ``model`` (resolved), whole and, with ``window``, by window."""
    x = resample(np.asarray(samples, dtype=np.float32), rate, RATE) if rate != RATE else samples
    duration = len(x) / RATE
    if window is not None and not 0.5 <= window <= MAX_S:
        raise InvalidRequest(f"window must be between 0.5 and {MAX_S:g} seconds", param="window")
    if hop is not None and (window is None or not 0.1 <= hop <= window):
        raise InvalidRequest("hop must be between 0.1 s and the window", param="hop")
    if len(x) < RATE // 10:
        raise InvalidRequest("the recording is shorter than 0.1 s", param="file")
    async with core.manager.use(model) as entry:
        if not entry.queue.handles("classify"):
            raise InvalidRequest(f"model '{model}' does not classify audio", param="model")

        async def run(spans: list[tuple[int, int]]) -> list[list[tuple[str, float]]]:
            out = []
            for a, b in spans:
                scores, _ = await entry.queue.submit("classify", np.ascontiguousarray(x[a:b], dtype=np.float32))
                out.append([(str(label), float(score)) for label, score in scores])
            return out

        with trace.phase("classify"):
            whole_spans = _windows(len(x), LONG_WINDOW_S if duration > MAX_S else MAX_S, LONG_WINDOW_S)
            spans = _windows(len(x), window, hop or window) if window is not None else []
            if spans == whole_spans:
                per = await run(spans)
                whole = per
            else:
                whole = await run(whole_spans)
                per = await run(spans) if spans else []
    loud = [float(np.max(np.abs(x[a:b]))) > 1e-3 for a, b in whole_spans]
    result = Classification(model, task, duration, _combine(task, whole, loud))
    result.windows = [Window(a / RATE, b / RATE, s) for (a, b), s in zip(spans, per)]
    return result


def _pick(
    scores: list[tuple[str, float]], top_k: int, threshold: float, labels: list[str] | None
) -> list[dict[str, Any]]:
    if labels:
        wanted = {label.lower() for label in labels}
        scores = [s for s in scores if s[0].lower() in wanted]
    return [{"label": label, "score": round(score, 4)} for label, score in scores[:top_k] if score >= threshold]


def tags_body(result: Classification, top_k: int, threshold: float, labels: list[str] | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "object": "audio.tags",
        "model": result.model,
        "duration": round(result.duration, 3),
        "tags": _pick(result.scores, top_k, threshold, labels),
    }
    if result.windows:
        body["segments"] = [
            {"start": round(w.start, 3), "end": round(w.end, 3), "tags": _pick(w.scores, top_k, threshold, labels)}
            for w in result.windows
        ]
    return body


def emotion_body(result: Classification, top_k: int) -> dict[str, Any]:
    def one(scores: list[tuple[str, float]]) -> dict[str, Any]:
        label, score = scores[0] if scores else (None, 0.0)
        return {
            "emotion": label,
            "score": round(score, 4),
            "emotions": [{"label": lab, "score": round(s, 4)} for lab, s in scores[:top_k]],
        }

    body: dict[str, Any] = {
        "object": "audio.emotion",
        "model": result.model,
        "duration": round(result.duration, 3),
        **one(result.scores),
    }
    if result.windows:
        body["segments"] = [
            {"start": round(w.start, 3), "end": round(w.end, 3), **one(w.scores)} for w in result.windows
        ]
    return body
