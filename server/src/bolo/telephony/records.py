"""Per-call records: what happened on a phone call, when, and how quickly the agent answered.

A record is the call-level companion of the per-turn request traces (:mod:`bolo.obs.trace`): one per call,
kept in a ring buffer in memory like the traces, with

* a **timeline** of events (answered, greeting, user speech, barge-in, key presses, tool calls, transfer, ...)
  in milliseconds since the media stream started;
* **turns** with a latency breakdown from the end of the caller's speech to the first audio sent back:
  ``endpoint_ms`` (silence the turn detector waited for), ``stt_ms``, ``llm_first_token_ms``,
  ``tts_first_audio_ms``, ``send_ms`` and their sum ``response_ms``;
* the **transcript** with speakers, the **outcome** and, when a language model is configured, a **summary**.
"""

from __future__ import annotations

import datetime as dt
import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

OUTCOMES = (
    "caller_hangup", "agent_hangup", "transferred", "max_duration", "no_response", "voicemail", "no_answer",
    "busy", "failed", "canceled", "error", "decision_end",
)  # fmt: skip
#: the pieces of a turn's latency, in the order they happen
LATENCY_PARTS = ("endpoint_ms", "stt_ms", "llm_first_token_ms", "tts_first_audio_ms", "send_ms")
MAX_TIMELINE = 2000

_ids = itertools.count(1)


def _now_iso() -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


@dataclass
class Turn:
    index: int
    #: what started it: "speech" (the caller spoke), "dtmf", "tool" (the answer after a tool call) or "prompt"
    kind: str = "speech"
    started_ms: float = 0.0
    user_text: str | None = None
    assistant_text: str | None = None
    interrupted: bool = False
    status: str = "in_progress"
    tool_calls: list[str] = field(default_factory=list)
    endpoint_ms: float | None = None
    stt_ms: float | None = None
    llm_first_token_ms: float | None = None
    tts_first_audio_ms: float | None = None
    send_ms: float | None = None
    #: end of the caller's speech (as detected) to the first audio frame handed to the provider
    total_ms: float | None = None

    @property
    def response_ms(self) -> float | None:
        """What the caller waits: the detector's silence window plus everything after it."""
        if self.total_ms is None:
            return None
        return self.total_ms + (self.endpoint_ms or 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "started_ms": _r(self.started_ms),
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
            "interrupted": self.interrupted,
            "status": self.status,
            "tool_calls": self.tool_calls,
            "latency": {
                "endpoint_ms": _r(self.endpoint_ms),
                "stt_ms": _r(self.stt_ms),
                "llm_first_token_ms": _r(self.llm_first_token_ms),
                "tts_first_audio_ms": _r(self.tts_first_audio_ms),
                "send_ms": _r(self.send_ms),
                "total_ms": _r(self.total_ms),
                "response_ms": _r(self.response_ms),
            },
        }


@dataclass
class CallRecord:
    provider: str
    direction: str = "inbound"  # or "outbound"
    agent: str | None = None
    from_number: str | None = None
    to_number: str | None = None
    provider_call_id: str | None = None
    stream_id: str | None = None
    id: str = field(default_factory=lambda: f"call_{next(_ids):06d}_{int(time.time()) % 100000:05d}")
    created_at: str = field(default_factory=_now_iso)
    started_at: str | None = None  # the media stream started (the call was answered)
    ended_at: str | None = None
    status: str = "ringing"  # ringing | in_progress | ended
    outcome: str | None = None
    error: str | None = None
    answered_by: str | None = None  # outbound with machine detection: "human", "machine", "unknown", ...
    transfer_to: str | None = None
    duration_s: float | None = None
    pipeline: str | None = None
    timeline: list[dict[str, Any]] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    dtmf: str = ""
    summary: str | None = None
    summary_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _t0: float | None = field(default=None, repr=False)

    # -- time ---------------------------------------------------------------------------------------------
    def begin(self) -> None:
        if self._t0 is None:
            self._t0 = time.perf_counter()
            self.started_at = _now_iso()
            self.status = "in_progress"

    def now_ms(self) -> float:
        return 0.0 if self._t0 is None else (time.perf_counter() - self._t0) * 1000.0

    def event(self, type_: str, **fields_: Any) -> None:
        if len(self.timeline) < MAX_TIMELINE:
            self.timeline.append({"t_ms": _r(self.now_ms()), "type": type_, **fields_})

    def say(self, speaker: str, text: str, **fields_: Any) -> dict[str, Any]:
        """Add a line to the transcript. ``speaker`` is "caller" or "agent"."""
        entry = {"t_ms": _r(self.now_ms()), "speaker": speaker, "text": text, **fields_}
        self.transcript.append(entry)
        return entry

    def end(self, outcome: str, error: str | None = None) -> bool:
        """Close the record. Returns False when it was closed already (the first outcome stands)."""
        if self.status == "ended":
            return False
        self.status = "ended"
        self.outcome = outcome
        self.error = error or self.error
        self.ended_at = _now_iso()
        if self._t0 is not None:
            self.duration_s = round(self.now_ms() / 1000.0, 2)
        self.event("ended", outcome=outcome, **({"error": error} if error else {}))
        return True

    # -- views --------------------------------------------------------------------------------------------
    def latency_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"turns": len(self.turns)}
        for name in (*LATENCY_PARTS, "total_ms", "response_ms"):
            values = sorted(v for t in self.turns if (v := getattr(t, name)) is not None)
            if values:
                out[name] = {
                    "p50": _r(values[(len(values) - 1) // 2]),
                    "max": _r(values[-1]),
                    "mean": _r(sum(values) / len(values)),
                }
        return out

    def summary_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "direction": self.direction,
            "agent": self.agent,
            "from": self.from_number,
            "to": self.to_number,
            "provider_call_id": self.provider_call_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "outcome": self.outcome,
            "error": self.error,
            "answered_by": self.answered_by,
            "duration_s": self.duration_s,
            "pipeline": self.pipeline,
            "turns": len(self.turns),
            "interruptions": sum(1 for e in self.timeline if e["type"] == "barge_in"),
            "latency": self.latency_summary(),
            "summary": self.summary,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary_dict(),
            "stream_id": self.stream_id,
            "transfer_to": self.transfer_to,
            "dtmf": self.dtmf,
            "summary_error": self.summary_error,
            "metadata": self.metadata,
            "timeline": list(self.timeline),
            "turns": [t.to_dict() for t in self.turns],
            "transcript": list(self.transcript),
        }

    def transcript_text(self) -> str:
        names = {"caller": "Caller", "agent": "Agent"}
        return "\n".join(f"{names.get(str(e['speaker']), e['speaker'])}: {e['text']}" for e in self.transcript)


class CallStore:
    """Ring buffer of calls (live and finished), newest last, plus counters for the statistics."""

    def __init__(self, capacity: int = 200) -> None:
        self._calls: deque[CallRecord] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.total = 0
        self.by_outcome: dict[tuple[str, str], int] = {}

    def add(self, record: CallRecord) -> None:
        with self._lock:
            self._calls.append(record)
            self.total += 1

    def count(self, record: CallRecord) -> None:
        """Count a call that just ended."""
        with self._lock:
            key = (record.provider, record.outcome or "unknown")
            self.by_outcome[key] = self.by_outcome.get(key, 0) + 1

    def get(self, call_id: str) -> CallRecord | None:
        with self._lock:
            return next((c for c in self._calls if c.id == call_id), None)

    def find(self, provider: str, provider_call_id: str | None) -> CallRecord | None:
        if not provider_call_id:
            return None
        with self._lock:
            return next(
                (c for c in reversed(self._calls) if c.provider == provider and c.provider_call_id == provider_call_id),
                None,
            )

    def recent(
        self, limit: int = 50, provider: str | None = None, agent: str | None = None, status: str | None = None
    ) -> list[CallRecord]:
        with self._lock:
            items = list(self._calls)
        out: list[CallRecord] = []
        for c in reversed(items):
            if provider and c.provider != provider:
                continue
            if agent and c.agent != agent:
                continue
            if status and c.status != status:
                continue
            out.append(c)
            if len(out) >= limit:
                break
        return out

    def active(self) -> int:
        with self._lock:
            return sum(1 for c in self._calls if c.status == "in_progress")
