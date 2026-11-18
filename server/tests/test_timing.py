"""The latency numbers the console shows must be right.

These tests drive the real pipeline with the fake engine, whose delays are
known, and check every reported duration against them. Where a sleep can
overrun (a loaded machine; macOS runners oversleep by 80 ms and more), the
check is against what the fake engine's calls really took (its ``spans``).
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from bolo.catalog import ModelSpec
from bolo.core import Bolo
from bolo.engines.base import Segment
from bolo.obs.trace import Trace
from bolo.streaming.session import SessionConfig, STTSession
from conftest import speechlike

TOL = 60.0  # ms of scheduling slack tolerated on a busy, shared CI machine


def add(core: Bolo, model_id: str, kind: str, task: str = "stt", **options: object) -> None:
    core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": model_id, "task": task, "engine": "fake", "kind": kind, "family": "Fake", "languages": ["en"],
                "license": "Apache-2.0", "builtin": True, "options": options, "sample_rate": 24000 if task == "tts" else 16000,
                "streaming": kind == "fake-stt-streaming",
            }
        )
    )  # fmt: skip


def phase(trace: Trace, name: str) -> float:
    return sum(p.dur_ms for p in trace.phases if p.name == name)


def ms(seconds: float) -> float:
    return seconds * 1000


async def test_stt_decode_phase_matches_engine_delay(core: Bolo) -> None:
    add(core, "t-stt", "fake-stt", base_delay_s=0.400, delay_per_audio_s=0.0)
    entry = await core.manager.load("t-stt")
    started = time.perf_counter()
    trace = Trace("stt", "http", "")
    result = await core.transcribe(speechlike(2.0), 16000, model="t-stt", trace=trace)
    assert result.text == "the quick brown fox jumps"
    ((start, end),) = entry.model.spans  # type: ignore[attr-defined]  # the 400 ms decode, as long as it really took
    assert phase(trace, "decode") == pytest.approx(ms(end - start), abs=TOL)
    assert phase(trace, "queue") < 25
    assert trace.final_ms == pytest.approx(ms(end - started), abs=TOL + 10)
    assert trace.total_ms == pytest.approx(trace.final_ms, abs=5)
    assert trace.rtf == pytest.approx(trace.total_ms / 1000 / 2.0)
    assert trace.first_partial_ms is None and trace.batch_size == 1
    assert trace.status == "ok" and trace.audio_seconds == 2.0
    names = [p.name for p in trace.phases]
    assert names == ["resample", "queue", "decode", "finalize"]
    starts = [p.start_ms for p in trace.phases]
    assert starts == sorted(starts)


async def test_stt_queue_phase_measures_waiting_behind_another_request(core: Bolo) -> None:
    add(core, "t-q", "fake-stt", base_delay_s=0.400, delay_per_audio_s=0.0, max_batch=1)
    entry = await core.manager.load("t-q")
    t1 = Trace("stt", "http", "")
    first = asyncio.ensure_future(core.transcribe(speechlike(1.0), 16000, model="t-q", trace=t1))
    await asyncio.sleep(0.1)
    arrived = time.perf_counter()
    t2 = Trace("stt", "http", "")
    await core.transcribe(speechlike(1.0), 16000, model="t-q", trace=t2)
    await first
    (_, end1), (start2, end2) = entry.model.spans  # type: ignore[attr-defined]
    assert phase(t1, "queue") < 25
    # arrived about 100 ms into a 400 ms decode: it waited for the rest of it, then took its own
    assert start2 >= end1
    assert phase(t2, "queue") == pytest.approx(ms(start2 - arrived), abs=TOL)
    assert phase(t2, "decode") == pytest.approx(ms(end2 - start2), abs=TOL)
    assert t2.total_ms == pytest.approx(ms(end2 - arrived), abs=TOL + 20)
    assert core.stats()["models"][0]["queue"]["queue_ms"]["p95"] > 200


async def test_stt_batched_requests_share_one_engine_call(core: Bolo) -> None:
    add(core, "t-b", "fake-stt", base_delay_s=0.300, delay_per_audio_s=0.0, max_batch=8)
    entry = await core.manager.load("t-b")
    blocker = asyncio.ensure_future(core.transcribe(speechlike(1.0), 16000, model="t-b"))
    await asyncio.sleep(0.03)
    traces = [Trace("stt", "http", "") for _ in range(4)]
    t0 = time.perf_counter()
    await asyncio.gather(*[core.transcribe(speechlike(1.0), 16000, model="t-b", trace=t) for t in traces])
    elapsed = (time.perf_counter() - t0) * 1000
    await blocker
    assert entry.model.calls == [1, 4]  # type: ignore[attr-defined]
    assert all(t.batch_size == 4 for t in traces)
    _, (start, end) = entry.model.spans  # type: ignore[attr-defined]
    # one batched call, not four serial ones: the rest of the blocker's call (about 270 ms), then one call (300 ms)
    assert elapsed == pytest.approx(ms(end - t0), abs=TOL + 30)
    for t in traces:
        assert phase(t, "decode") == pytest.approx(ms(end - start), abs=TOL)


async def test_long_audio_is_segmented_and_first_segment_time_is_reported(core: Bolo) -> None:
    add(core, "t-long", "fake-stt", base_delay_s=0.250, delay_per_audio_s=0.0, max_batch=1, max_chunk_s=5.0)
    entry = await core.manager.load("t-long")
    audio = speechlike(12.0, gap_at=(3.5, 8.0))
    seen: list[tuple[float, Segment]] = []
    trace = Trace("stt", "http", "")

    async def on_segment(seg: Segment) -> None:
        seen.append((trace.elapsed_ms(), seg))

    result = await core.transcribe(audio, 16000, model="t-long", trace=trace, on_segment=on_segment)
    assert len(result.segments) == 3  # split at the two pauses
    assert [round(s.start) for s in result.segments] == [0, 4, 9]
    assert phase(trace, "vad") > 0
    assert trace.first_partial_ms == pytest.approx(seen[0][0], abs=5)
    calls = [ms(end - start) for start, end in entry.model.spans]  # type: ignore[attr-defined]  # 250 ms asked
    assert len(calls) == 3
    assert seen[1][0] - seen[0][0] == pytest.approx(calls[1], abs=TOL)  # one engine call per segment (max_batch=1)
    assert phase(trace, "decode") == pytest.approx(sum(calls), abs=TOL + 40)
    assert trace.final_ms > trace.first_partial_ms


async def test_tts_time_to_first_audio_and_phases(core: Bolo) -> None:
    add(core, "t-tts", "fake-tts", task="tts", first_delay_s=0.400, delay_per_char_s=0.0, pieces=2)
    entry = await core.manager.load("t-tts")
    started = time.perf_counter()
    trace = Trace("tts", "http", "")
    arrivals: list[float] = []
    async for _chunk in core.synthesize("Hello there. General greetings to you.", model="t-tts", trace=trace):
        trace.mark_ttfa()
        arrivals.append(trace.elapsed_ms())
    core.complete_tts(trace)
    # two sentences x two pieces; each piece costs 200 ms (what each really took: the fake's piece_spans)
    assert len(arrivals) == 4
    pieces = entry.model.piece_spans  # type: ignore[attr-defined]
    assert len(pieces) == 4
    assert trace.ttfa_ms == pytest.approx(ms(pieces[0][1] - started), abs=TOL)
    assert phase(trace, "synthesize_first") == pytest.approx(ms(pieces[0][1] - pieces[0][0]), abs=TOL)
    assert phase(trace, "synthesize_rest") == pytest.approx(ms(pieces[-1][1] - pieces[0][1]), abs=TOL + 30)
    assert trace.total_ms == pytest.approx(ms(pieces[-1][1] - started), abs=TOL + 40)
    assert [p.name for p in trace.phases] == ["normalize", "queue", "synthesize_first", "synthesize_rest"]
    assert trace.audio_seconds == pytest.approx(len("Hello there. General greetings to you.") * 0.06, rel=0.1)
    assert trace.rtf == pytest.approx(trace.total_ms / 1000 / trace.audio_seconds)
    stats = core.traces.models["t-tts"]
    assert stats.ttfa.quantile(0.5) == pytest.approx(trace.ttfa_ms)


async def run_stream(
    core: Bolo, model: str, audio: np.ndarray, *, chunk_ms: int = 40, pace: float = 1.0
) -> tuple[list[dict], STTSession]:
    messages: list[dict] = []

    async def send(msg: dict) -> None:
        msg["_at"] = time.perf_counter()
        messages.append(msg)

    session = STTSession(core, send, SessionConfig(model=model))
    await session.start()
    pcm = (audio * 32767).astype("<i2").tobytes()
    step = 16 * chunk_ms * 2
    for off in range(0, len(pcm), step):
        session.feed(pcm[off : off + step])
        await asyncio.sleep(chunk_ms / 1000 * pace)
    await session.stop()
    return messages, session


async def test_native_stream_latency_equals_decode_step(core: Bolo) -> None:
    add(core, "t-rt", "fake-stt-streaming", step_delay_s=0.200, endpoint_silence_s=0.4)
    entry = await core.manager.load("t-rt")
    audio = np.concatenate([speechlike(1.6), np.zeros(16000, dtype=np.float32)])
    messages, session = await run_stream(core, "t-rt", audio)
    partials = [m for m in messages if m["type"] == "partial"]
    finals = [m for m in messages if m["type"] == "final"]
    assert len(partials) >= 2 and len(finals) == 1
    steps = [ms(end - start) for start, end in entry.model.stream_spans]  # type: ignore[attr-defined]
    for m in [*partials, *finals]:  # receipt of the last audio -> message sent: one decode step (200 ms asked)
        assert min(steps) - 1 <= m["latency_ms"] <= max(steps) + TOL
    texts = [m["text"] for m in partials]
    assert all(b.startswith(a) for a, b in zip(texts, texts[1:]))  # append-only
    assert finals[0]["text"].startswith(texts[-1])
    trace = session.trace
    assert trace is not None and trace.status == "ok"
    assert trace.first_partial_ms == pytest.approx(partials[0]["latency_ms"], abs=0.2)
    assert trace.final_ms == pytest.approx(finals[0]["latency_ms"], abs=0.2)
    assert trace.extra["utterances"] == 1
    decode = phase(trace, "decode")
    assert trace.rtf == pytest.approx(decode / 1000 / trace.audio_seconds, rel=0.05)
    st = core.traces.models["t-rt"]
    assert st.first_partial.quantile(0.5) == pytest.approx(partials[0]["latency_ms"], abs=0.2)
    assert st.final.quantile(0.5) == pytest.approx(finals[0]["latency_ms"], abs=0.2)


async def test_pseudo_stream_partials_and_final_latency(core: Bolo) -> None:
    add(core, "t-ps", "fake-stt", base_delay_s=0.200, delay_per_audio_s=0.0)
    entry = await core.manager.load("t-ps")
    audio = np.concatenate([speechlike(2.4), np.zeros(16000, dtype=np.float32)])
    messages, session = await run_stream(core, "t-ps", audio)
    kinds = [m["type"] for m in messages]
    assert kinds[0] == "ready" and messages[0]["mode"] == "pseudo"
    assert kinds[-1] == "done" and "metrics" in kinds
    vad = [m["event"] for m in messages if m["type"] == "vad"]
    assert vad == ["speech_start", "speech_end"]
    partials = [m for m in messages if m["type"] == "partial"]
    finals = [m for m in messages if m["type"] == "final"]
    assert len(partials) >= 3 and len(finals) == 1
    decodes = [ms(end - start) for start, end in entry.model.spans]  # type: ignore[attr-defined]
    for m in [*partials, *finals]:  # one re-decode of the utterance: the engine's 200 ms
        assert min(decodes) - 1 <= m["latency_ms"] <= max(decodes) + TOL
    gaps = [b["_at"] - a["_at"] for a, b in zip(partials, partials[1:])]
    assert min(gaps) >= 0.30  # partial_interval_ms = 350 by default
    # LocalAgreement: the stable prefix only ever grows
    stable = [m["text"][: m["stable_chars"]] for m in partials]
    assert all(b.startswith(a) for a, b in zip(stable, stable[1:]))
    assert stable[-1] != ""
    assert finals[0]["words"] and finals[0]["words"][0]["w"] == "the"
    assert finals[0]["t0"] < 0.2 and finals[0]["t1"] == pytest.approx(2.6, abs=0.3)
    assert session.trace is not None and session.trace.audio_seconds == pytest.approx(3.4, abs=0.05)


async def test_concurrent_streams_are_decoded_in_batches(core: Bolo) -> None:
    add(core, "t-many", "fake-stt-streaming", step_delay_s=0.100)
    entry = await core.manager.load("t-many")
    audio = np.concatenate([speechlike(1.2), np.zeros(12800, dtype=np.float32)])
    results = await asyncio.gather(*[run_stream(core, "t-many", audio) for _ in range(6)])
    steps = [ms(end - start) for start, end in entry.model.stream_spans]  # type: ignore[attr-defined]
    calls = entry.model.stream_calls  # type: ignore[attr-defined]
    assert max(calls) >= 4  # six streams advance in shared engine calls
    assert entry.queue.stats.mean_batch > 2
    for messages, _session in results:
        finals = [m for m in messages if m["type"] == "final"]
        assert len(finals) == 1
        # even with six streams the wait is about one or two engine steps (100 ms asked), not six
        assert finals[0]["latency_ms"] < 3 * max(steps) + TOL
    assert core.active_streams == 0
