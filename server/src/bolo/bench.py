"""In-process load generator behind ``POST /api/v1/bench`` and ``bolo bench``.

Two modes for speech-to-text:

* ``batch`` (default): every virtual client transcribes the sample
  back-to-back as fast as the server allows. ``throughput_x_realtime`` is
  audio seconds transcribed per wall-clock second.
* ``realtime``: every virtual client streams the sample at 1x speed through
  a realtime session, as a microphone would. ``first_partial_ms`` and
  ``final_ms`` are then the streaming latencies defined in the API docs,
  and throughput cannot exceed the number of clients.

Text-to-speech has one mode: clients synthesise the sample text
back-to-back; ``ttfa_ms`` is time to the first audio chunk.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo import samples as builtin_samples
from bolo.audio.resample import resample
from bolo.engines.base import FloatArray, Segment
from bolo.errors import BoloError, InvalidRequest
from bolo.jobs import Job
from bolo.obs.sketch import summarize
from bolo.obs.trace import Trace
from bolo.streaming.session import SessionConfig, STTSession

if TYPE_CHECKING:
    from bolo.core import Bolo


class _Collector:
    def __init__(self) -> None:
        self.ok = 0
        self.errors = 0
        self.first_partial: list[float] = []
        self.final: list[float] = []
        self.ttfa: list[float] = []
        self.rtf: list[float] = []
        self.audio_seconds = 0.0
        self.last_error: str | None = None

    def point(self, concurrency: int, elapsed: float) -> dict[str, Any]:
        out: dict[str, Any] = {
            "concurrency": concurrency,
            "ok": self.ok,
            "errors": self.errors,
            "first_partial_ms": summarize(self.first_partial),
            "final_ms": summarize(self.final),
            "ttfa_ms": summarize(self.ttfa),
            "rtf": summarize(self.rtf, digits=4),
            "throughput_x_realtime": round(self.audio_seconds / elapsed, 2) if elapsed > 0 else 0.0,
            "elapsed_s": round(elapsed, 2),
        }
        if self.last_error:
            out["last_error"] = self.last_error
        return out


async def _stt_batch_client(
    core: Bolo, model: str, audio: FloatArray, rate: int, deadline: float, col: _Collector
) -> None:
    while time.monotonic() < deadline:
        trace = Trace("stt", "bench", model)

        async def on_segment(_seg: Segment) -> None:
            return None

        try:
            await core.transcribe(audio, rate, model=model, trace=trace, on_segment=on_segment)
        except BoloError as exc:
            col.errors += 1
            col.last_error = exc.message
            await asyncio.sleep(0.05)
            continue
        col.ok += 1
        col.audio_seconds += len(audio) / rate
        if trace.first_partial_ms is not None:
            col.first_partial.append(trace.first_partial_ms)
        if trace.final_ms is not None:
            col.final.append(trace.final_ms)
        if trace.rtf is not None:
            col.rtf.append(trace.rtf)


async def _stt_realtime_client(
    core: Bolo, model: str, audio16: FloatArray, deadline: float, col: _Collector, frame_ms: int = 100
) -> None:
    pcm = (np.clip(audio16, -1, 1) * 32767).astype("<i2").tobytes()
    step = 16000 * frame_ms // 1000 * 2
    silence = b"\x00" * step
    while time.monotonic() < deadline:
        got_partial = False
        seen_final = False

        async def send(msg: dict[str, Any]) -> None:
            nonlocal got_partial, seen_final
            if msg["type"] == "partial" and not got_partial:
                got_partial = True
                col.first_partial.append(msg["latency_ms"])
            elif msg["type"] == "final":
                seen_final = True
                col.final.append(msg["latency_ms"])
                got_partial = False
            elif msg["type"] == "error":
                col.last_error = msg.get("message")

        session = STTSession(core, send, SessionConfig(model=model, transport="bench"))
        try:
            await session.start()
            started = time.monotonic()
            n = 0
            for off in range(0, len(pcm), step):
                session.feed(pcm[off : off + step])
                n += 1
                await asyncio.sleep(max(0.0, started + n * frame_ms / 1000.0 - time.monotonic()))
            for _ in range(1000 // frame_ms):  # one second of trailing silence so the endpoint fires
                session.feed(silence)
                n += 1
                await asyncio.sleep(max(0.0, started + n * frame_ms / 1000.0 - time.monotonic()))
            await session.stop()
        except BoloError as exc:
            col.errors += 1
            col.last_error = exc.message
            await session.close()
            await asyncio.sleep(0.05)
            continue
        if seen_final and session.trace and session.trace.status == "ok":
            col.ok += 1
            col.audio_seconds += len(audio16) / 16000
            if session.trace.rtf is not None:
                col.rtf.append(session.trace.rtf)
        else:
            col.errors += 1


async def _tts_client(core: Bolo, model: str, text: str, voice: str | None, deadline: float, col: _Collector) -> None:
    while time.monotonic() < deadline:
        trace = Trace("tts", "bench", model)
        status = "ok"
        try:
            async for _chunk in core.synthesize(text, model=model, voice=voice, trace=trace):
                trace.mark_ttfa()
        except BoloError as exc:
            status = "error"
            trace.finish("error", exc.message)
            col.errors += 1
            col.last_error = exc.message
            await asyncio.sleep(0.05)
        finally:
            core.complete_tts(trace)
        if status == "ok":
            col.ok += 1
            col.audio_seconds += trace.audio_seconds or 0.0
            if trace.ttfa_ms is not None:
                col.ttfa.append(trace.ttfa_ms)
            if trace.total_ms is not None:
                col.final.append(trace.total_ms)
            if trace.rtf is not None:
                col.rtf.append(trace.rtf)


async def run_bench(
    core: Bolo,
    *,
    kind: str,
    model: str | None,
    concurrency: list[int],
    duration_s: float = 10.0,
    sample: str | None = None,
    mode: str = "batch",
    voice: str | None = None,
    job: Job | None = None,
) -> dict[str, Any]:
    if kind not in ("stt", "tts"):
        raise InvalidRequest("kind must be 'stt' or 'tts'", param="kind")
    if mode not in ("batch", "realtime"):
        raise InvalidRequest("mode must be 'batch' or 'realtime'", param="mode")
    if not concurrency or any(c < 1 or c > 256 for c in concurrency):
        raise InvalidRequest("concurrency must be a list of integers between 1 and 256", param="concurrency")
    if not 0.2 <= duration_s <= 600:
        raise InvalidRequest("duration_s must be between 0.2 and 600", param="duration_s")
    spec = core.resolve(model, kind)
    audio: FloatArray | None = None
    rate = 16000
    text = ""
    if kind == "stt":
        audio, rate, meta = builtin_samples.load_audio(sample or builtin_samples.DEFAULT_AUDIO)
        sample_id = meta["id"]
    else:
        sample_id = (sample or "builtin:en-text-long").removeprefix("builtin:")
        text = builtin_samples.load_text(sample_id)
        if not text:
            raise InvalidRequest(f"sample '{sample_id}' has no text", param="sample")
    await core.manager.load(spec.id)  # keep model loading out of the measurements
    points: list[dict[str, Any]] = []
    for i, c in enumerate(concurrency):
        col = _Collector()
        entry = core.manager.loaded.get(spec.id)
        batches0 = entry.queue.stats.batches if entry else 0
        jobs0 = entry.queue.stats.batched_jobs if entry else 0
        started = time.monotonic()
        deadline = started + duration_s
        if kind == "stt":
            assert audio is not None
            if mode == "realtime":
                audio16 = resample(audio, rate, 16000)
                clients = [_stt_realtime_client(core, spec.id, audio16, deadline, col) for _ in range(c)]
            else:
                clients = [_stt_batch_client(core, spec.id, audio, rate, deadline, col) for _ in range(c)]
        else:
            clients = [_tts_client(core, spec.id, text, voice, deadline, col) for _ in range(c)]
        await asyncio.gather(*clients)
        point = col.point(c, time.monotonic() - started)
        entry = core.manager.loaded.get(spec.id)
        if entry is not None and entry.queue.stats.batches > batches0:
            st = entry.queue.stats
            point["mean_batch"] = round((st.batched_jobs - jobs0) / (st.batches - batches0), 2)
        points.append(point)
        if job is not None:
            job.progress = (i + 1) / len(concurrency)
            job.result = {"points": list(points)}
            core.events.publish("job", {"job": job.to_dict()})
    return {
        "kind": kind,
        "model": spec.id,
        "mode": mode if kind == "stt" else "batch",
        "sample": sample_id,
        "duration_s": duration_s,
        "points": points,
    }
