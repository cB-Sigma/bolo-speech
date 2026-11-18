"""Watermarks on synthetic and cloned speech, and their detection.

Speech gets an inaudible watermark (task ``watermark`` models: AudioSeal by default) when the request asks for one
(``watermark``: ``true`` / ``"on"``, or a watermark model's id) or, without that, when the server's ``watermark``
setting says so: ``cloned`` (the default: speech in a cloned voice, a saved voice or ``reference_audio``), ``all``
or ``off``. A watermarker that cannot run (not installed with ``auto_pull`` off, its engine missing) never stops
the speech: it is spoken unmarked and the response says why (``x-bolo-watermark: unavailable; ...``).

The mark is made at the watermarker's rate (16 kHz for AudioSeal): the audio is resampled to it, the mark computed,
and the difference resampled back and added, so speech at any rate keeps its own bandwidth. Detection resamples
to the detector's rate and reports the share of the recording that is marked, where, and the payload.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.audio.resample import resample
from bolo.engines.base import FloatArray, WatermarkDetection
from bolo.errors import BoloError, InvalidRequest

if TYPE_CHECKING:
    from bolo.core import Bolo
    from bolo.lifecycle import LoadedModel

log = logging.getLogger("bolo.watermark")

POLICIES = ("cloned", "all", "off")
#: a recording counts as marked when at least this share of it is
MARKED_SHARE = 0.5
#: frames of the audio before a chunk given to the watermarker as context
CONTEXT_HOPS = 10
#: shortest marked stretch reported as a segment
MIN_SEGMENT_S = 0.2


def choose(core: Bolo, requested: Any, cloned: bool) -> str | None:
    """The watermark model for a synthesis, or None: the request's ``watermark`` (bool, "on"/"off", a model id),
    else the server's ``watermark`` policy."""
    s = core.settings
    if requested is None or requested == "":
        policy = s.watermark if s.watermark in POLICIES else "cloned"
        on = policy == "all" or (policy == "cloned" and cloned)
        return s.watermark_model if on else None
    if isinstance(requested, bool):
        return s.watermark_model if requested else None
    value = str(requested).strip()
    if value.lower() in ("on", "true", "1", "yes", "default"):
        return s.watermark_model
    if value.lower() in ("off", "false", "0", "no", "none"):
        return None
    spec = core.catalog.resolve(value)
    if spec is None or spec.task != "watermark":
        known = ", ".join(m.id for m in core.catalog.all() if m.task == "watermark")
        raise InvalidRequest(
            f"watermark must be on, off or a watermark model ({known}); {value!r} is none of these", param="watermark"
        )
    return spec.id


@dataclass
class Marker:
    """A watermarker borrowed for one synthesis: marks chunk after chunk."""

    core: Bolo
    entry: LoadedModel | None
    model: str
    payload: str
    reason: str | None = None
    marked_s: float = field(default=0.0)
    _position: int = field(default=0)
    _history: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    @property
    def active(self) -> bool:
        return self.entry is not None

    def header(self) -> str:
        if self.entry is not None:
            return self.model
        return f"unavailable; {self.model}: {self.reason}"

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {"model": self.model, "applied": self.entry is not None}
        if self.entry is not None:
            out["payload"] = self.payload
        else:
            out["reason"] = self.reason
        return out

    async def mark(self, samples: FloatArray, rate: int) -> FloatArray:
        """Mark one chunk of a stream (chunks in order). The mark is computed on the chunk with the audio before it
        as context, starting on the model's frame grid counted from the start of the stream: AudioSeal's detector
        finds a mark made in pieces only when the pieces keep that alignment."""
        if self.entry is None or len(samples) == 0:
            return samples
        model = self.entry.model
        model_rate = model.sample_rate
        hop = int(getattr(model, "hop", 1) or 1)
        x = np.asarray(samples, dtype=np.float32)
        low = resample(x, rate, model_rate) if rate != model_rate else x
        offset = self._position % hop
        k = offset + CONTEXT_HOPS * hop if len(self._history) >= offset + CONTEXT_HOPS * hop else offset
        k = min(k, len(self._history))
        context = np.concatenate([self._history[len(self._history) - k :], low]) if k else low
        marked, _info = await self.entry.queue.submit("watermark", (context, self.payload))
        diff = (np.asarray(marked, dtype=np.float32) - context)[k:]
        self._history = np.concatenate([self._history, low])[-(CONTEXT_HOPS + 2) * hop :]
        self._position += len(low)
        if rate != model_rate:
            diff = resample(diff, model_rate, rate)
        n = min(len(diff), len(x))
        out = x.copy()
        out[:n] += diff[:n]
        self.marked_s += len(x) / rate
        return np.asarray(np.clip(out, -1.0, 1.0), dtype=np.float32)


@contextlib.asynccontextmanager
async def marker(core: Bolo, model: str | None) -> AsyncIterator[Marker | None]:
    """Borrow the watermarker ``model`` (None: no watermark). One that cannot run gives an inactive marker that
    says why, and the speech goes out unmarked."""
    if model is None:
        yield None
        return
    payload = core.settings.watermark_payload
    async with contextlib.AsyncExitStack() as stack:
        entry: LoadedModel | None = None
        reason: str | None = None
        spec_id = model
        try:
            spec = core.resolve(model, "watermark")
            spec_id = spec.id
            core.require_components(spec)
            entry = await stack.enter_async_context(core.manager.use(spec.id))
        except BoloError as exc:
            reason = exc.message
        except Exception as exc:  # an engine that fails to load must not stop the speech
            reason = f"{type(exc).__name__}: {exc}"
        if entry is None:
            log.warning("speech is not watermarked: %s: %s", spec_id, reason)
        yield Marker(core, entry, spec_id, payload, reason=reason)


@dataclass
class Detection:
    model: str
    marked: bool
    score: float
    payload: str | None
    #: the payload is the one this server marks with
    ours: bool
    segments: list[dict[str, float]]
    duration: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "marked": self.marked,
            "score": round(self.score, 4),
            "payload": self.payload if self.marked else None,
            "payload_is_this_servers": self.ours if self.marked else False,
            "segments": self.segments,
        }


def _segments(found: WatermarkDetection) -> list[dict[str, float]]:
    flags = found.frame_probs > 0.5
    out: list[dict[str, float]] = []
    start: int | None = None
    for i, on in enumerate([*flags.tolist(), False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            a, b = start * found.frame_s, i * found.frame_s
            if b - a >= MIN_SEGMENT_S:
                if out and a - out[-1]["end"] < MIN_SEGMENT_S:
                    out[-1]["end"] = round(b, 3)
                else:
                    out.append({"start": round(a, 3), "end": round(b, 3)})
            start = None
    return out


async def detect(core: Bolo, samples: FloatArray, rate: int, model: str) -> Detection:
    """Look for ``model``'s watermark in ``samples`` (mono float32 at ``rate``)."""
    spec = core.resolve(model, "watermark")
    core.require_components(spec)
    async with core.manager.use(spec.id) as entry:
        model_rate = entry.model.sample_rate
        x = np.asarray(samples, dtype=np.float32)
        x = resample(x, rate, model_rate) if rate != model_rate else x
        if len(x) < model_rate // 10:
            raise InvalidRequest("the recording is shorter than 0.1 s", param="file")
        found, _info = await entry.queue.submit("detect_watermark", x)
    marked = found.score >= MARKED_SHARE
    payload = found.payload
    return Detection(
        spec.id, marked, found.score, payload, bool(marked and payload == core.settings.watermark_payload),
        _segments(found) if marked or found.score > 0.05 else [], len(x) / model_rate,
    )  # fmt: skip
