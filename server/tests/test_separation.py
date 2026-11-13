"""Overlapped-speech separation (``separate_overlaps``), with the fake diarizer, separator and recogniser."""

from __future__ import annotations

import asyncio
import base64
import io
import sys

import httpx
import numpy as np
import pytest
import soundfile as sf

from bolo.audio.encode import encode_audio
from bolo.catalog import ModelSpec
from bolo.config import Settings
from bolo.core import Bolo
from bolo.diarize import DiarizeOptions, Overlap
from bolo.engines.base import LoadOptions, SpeakerTurn
from bolo.engines.fake import FakeSeparation
from bolo.errors import InvalidRequest
from bolo.obs.trace import Trace
from bolo.plugin.host import PluginConfig, PluginEngine, PluginSeparation
from bolo.separate import (
    Chunk,
    SeparateOptions,
    Window,
    choose_sample,
    clean_speech,
    is_narrowband,
    plan_chunks,
    plan_windows,
    quiet_point,
    split_long,
    subtract,
)
from conftest import LiveServer

SR = 16000
A_HZ, B_HZ = 300.0, 500.0


def tone(freq: float, start: float, end: float, total: float, amp: float = 0.3) -> np.ndarray:
    out = np.zeros(int(total * SR), dtype=np.float32)
    t = np.arange(int((end - start) * SR)) / SR
    out[int(start * SR) : int(start * SR) + len(t)] = amp * np.sin(2 * np.pi * freq * t)
    return out


def meeting(b_first: float = 4.5, b_first_end: float = 8.5) -> np.ndarray:
    """A alone 0-4 s, B alone, A 9-14 s, B 12-16 s: a 2 s overlap from 12 to 14 s."""
    total = 16.5
    return (
        tone(A_HZ, 0.0, 4.0, total)
        + tone(B_HZ, b_first, b_first_end, total)
        + tone(A_HZ, 9.0, 14.0, total)
        + tone(B_HZ, 12.0, 16.0, total)
    )


def wav(audio: np.ndarray, rate: int = SR) -> io.BytesIO:
    f = io.BytesIO(encode_audio(audio, rate, "wav"))
    f.name = "talk.wav"
    return f


def add_fake(core: Bolo, model_id: str, **options: object) -> None:
    core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": model_id,
                "task": "separation",
                "engine": "fake",
                "kind": "fake-separation",
                "family": "Fake",
                "languages": [],
                "license": "Apache-2.0",
                "builtin": True,
                "options": dict(options),
            }
        )
    )


@pytest.fixture
def sep_settings(settings: Settings) -> Settings:
    settings.default_diarization = "fake-diarization-overlap"
    settings.default_separation = "fake-separation"
    return settings


@pytest.fixture
async def sep_core(sep_settings: Settings):  # type: ignore[no-untyped-def]
    c = Bolo(sep_settings)
    await c.start()
    try:
        yield c
    finally:
        await c.stop()


@pytest.fixture
def sep_live(sep_settings: Settings):  # type: ignore[no-untyped-def]
    server = LiveServer(sep_settings)
    server.start()
    try:
        yield server
    finally:
        server.stop()


# ----------------------------------------------------------------------
# planning helpers
# ----------------------------------------------------------------------
def test_interval_helpers_and_voice_samples() -> None:
    assert subtract([(0, 10)], [(2, 3), (5, 6)]) == [(0, 2), (3, 5), (6, 10)]
    assert subtract([(0, 2)], [(-1, 5)]) == []
    turns = [SpeakerTurn(0, 4, 0), SpeakerTurn(4.5, 8.5, 1), SpeakerTurn(9, 14, 0), SpeakerTurn(12, 16, 1)]
    clean = clean_speech(turns)
    assert clean[0] == [(0, 4), (9, 11.75)]
    assert clean[1] == [(4.5, 8.5), (14.25, 16)]
    # the longest stretches first, cut at the maximum; too little in total gives nothing
    assert choose_sample(clean[0], 3.0, 5.0) == [(0, 4), (9, 10.0)]
    assert choose_sample([(0, 1.0), (2, 3.5)], 3.0, 10.0) == []
    assert choose_sample([(0, 0.3), (1, 4.2)], 3.0, 10.0) == [(1, 4.2)]


def test_windows_are_widened_snapped_and_joined() -> None:
    overlaps = [Overlap(10, 11, [0, 1]), Overlap(13, 14, [1, 2]), Overlap(30, 30.3, [0, 1])]
    windows = plan_windows(overlaps, 40.0, context_s=2.0, min_overlap_s=0.5)
    assert len(windows) == 1  # 8-13 and 11-16 touch; the 0.3 s graze is ignored
    assert (windows[0].start, windows[0].end, windows[0].speakers) == (8.0, 16.0, [0, 1, 2])
    edge = plan_windows([Overlap(0.5, 2.0, [0, 1])], 3.0, context_s=2.0, min_overlap_s=0.5)[0]
    assert (edge.start, edge.end) == (0.0, 3.0)
    # edges move to a pause next to the nominal point
    audio = np.full(SR * 4, 0.3, dtype=np.float32)
    audio[int(1.2 * SR) : int(1.3 * SR)] = 0.0
    assert quiet_point(audio, 1.0) == pytest.approx(1.2, abs=0.02)
    snapped = plan_windows([Overlap(2.5, 3.5, [0, 1])], 4.0, context_s=1.4, min_overlap_s=0.5, samples=audio)[0]
    assert snapped.start == pytest.approx(1.2, abs=0.02)


def test_chunks_take_separated_speakers_from_their_tracks() -> None:
    raw = [SpeakerTurn(0, 4, 0), SpeakerTurn(4.5, 8.5, 1), SpeakerTurn(9, 14, 0), SpeakerTurn(12, 16, 1)]
    merged = [SpeakerTurn(0, 4, 0), SpeakerTurn(4.5, 8.5, 1), SpeakerTurn(9, 14, 0), SpeakerTurn(14, 16, 1)]
    overlaps = [Overlap(12, 14, [0, 1])]
    w = Window(10, 16, overlaps, [0, 1], separated=[0, 1])
    chunks = plan_chunks(merged, raw, [w], overlaps)
    assert [(c.start, c.end, c.speaker, c.track) for c in chunks] == [
        (0, 4, 0, None),
        (4.5, 8.5, 1, None),
        (9, 10, 0, None),
        (10, 14, 0, (0, 0)),
        (12, 16, 1, (0, 1)),  # B's clipped turn gets its overlapped start back
    ]
    # only A could be separated: B keeps its (clipped) turn from the mixture, marked as overlapped
    w.separated = [0]
    chunks = plan_chunks(merged, raw, [w], overlaps)
    assert [(c.start, c.end, c.speaker, c.track, c.overlap) for c in chunks][-2:] == [
        (10, 14, 0, (0, 0), True),
        (14, 16, 1, None, False),
    ]
    parts = split_long([Chunk(0, 70, 0, (0, 0), True)], 30.0)
    assert [(c.start, c.end) for c in parts] == [
        (0, pytest.approx(70 / 3)),
        (pytest.approx(70 / 3), pytest.approx(140 / 3)),
        (pytest.approx(140 / 3), 70),
    ]


def test_narrowband_detection() -> None:
    rng = np.random.default_rng(0)
    wide = rng.standard_normal(SR * 2).astype(np.float32) * 0.1
    spectrum = np.fft.rfft(wide)
    spectrum[np.fft.rfftfreq(len(wide), 1 / SR) > 3600] = 0
    narrow = np.fft.irfft(spectrum, len(wide)).astype(np.float32)
    assert not is_narrowband(wide) and is_narrowband(narrow) and is_narrowband(wide[:8000], 8000)


# ----------------------------------------------------------------------
# the pipeline (core)
# ----------------------------------------------------------------------
async def test_transcription_separates_the_overlap(sep_core: Bolo) -> None:
    trace = Trace("stt", "internal", "")
    opts = SeparateOptions()
    t = await sep_core.transcribe(meeting(), SR, diarize=DiarizeOptions(), separate=opts, trace=trace)
    report = t.separation
    assert report is not None and report["status"] == "applied" and report["reason"] is None
    assert report["model"] == "fake-separation" and report["rtf"] is not None
    assert [s["speaker"] for s in report["speakers"]] == ["A", "B"]
    assert all(s["sample_s"] >= 3.0 for s in report["speakers"])
    (window,) = report["windows"]
    assert window["separated"] == ["A", "B"] and window["skipped"] == []
    assert window["start"] < 12.0 and window["end"] > 14.0
    separated = [s for s in t.segments if s.separated]
    assert {s.speaker for s in separated} == {"A", "B"} and all(s.overlap for s in separated)
    b_track = next(s for s in separated if s.speaker == "B")
    assert b_track.start == pytest.approx(12.0, abs=0.3)  # B's words under A are transcribed now
    assert all(not s.separated for s in t.segments if s.end <= 9.5)
    assert "separate" in [p.name for p in trace.phases]
    assert trace.extra["separation"] == "applied" and trace.extra["separation_windows"] == 1
    model = sep_core.manager.loaded["fake-separation"].model
    assert isinstance(model, FakeSeparation) and model.voices == {}  # extractors released
    assert sep_core.stats()["separation"]["requests"]["applied"] == 1


async def test_separation_skips_and_partial_results(sep_core: Bolo) -> None:
    # no overlap at all
    calm = meeting()[: int(8.6 * SR)]
    t = await sep_core.transcribe(calm, SR, diarize=DiarizeOptions(), separate=SeparateOptions())
    assert t.separation is not None and (t.separation["status"], t.separation["reason"]) == ("skipped", "no_overlap")
    assert not any(s.separated for s in t.segments)
    # B says only 2.75 s alone in all: A is separated, B is not
    t = await sep_core.transcribe(meeting(4.5, 5.5), SR, diarize=DiarizeOptions(), separate=SeparateOptions())
    rep = t.separation
    assert rep is not None and rep["status"] == "partial" and rep["reason"] == "no_voice_sample"
    b = next(s for s in rep["speakers"] if s["speaker"] == "B")
    assert b["sample_s"] is None and b["skipped"].startswith("not_enough_clean_speech")
    assert rep["windows"][0]["skipped"] == [{"speaker": "B", "reason": "not_enough_clean_speech"}]
    assert {s.speaker for s in t.segments if s.separated} == {"A"}
    # overlaps shorter than the minimum are left alone
    t = await sep_core.transcribe(meeting(), SR, diarize=DiarizeOptions(), separate=SeparateOptions(min_overlap_s=3.0))
    assert t.separation is not None and t.separation["reason"] == "no_overlap"


async def test_separation_failure_timeout_and_cancellation(sep_core: Bolo) -> None:
    add_fake(sep_core, "fake-separation-broken", fail=True)
    t = await sep_core.transcribe(
        meeting(), SR, diarize=DiarizeOptions(), separate=SeparateOptions(model="fake-separation-broken")
    )
    assert t.separation is not None and (t.separation["status"], t.separation["reason"]) == ("skipped", "error")
    assert t.segments and not any(s.separated for s in t.segments)  # the mixture is transcribed as before

    add_fake(sep_core, "fake-separation-slow", delay_s=0.6)
    sep_core.settings.separation_timeout_s = 1.0
    t = await sep_core.transcribe(
        meeting(), SR, diarize=DiarizeOptions(), separate=SeparateOptions(model="fake-separation-slow")
    )
    assert t.separation is not None and t.separation["reason"] == "timeout"
    assert t.separation["status"] in ("partial", "skipped")
    sep_core.settings.separation_timeout_s = 300.0

    trace = Trace("stt", "internal", "")
    task = asyncio.create_task(
        sep_core.transcribe(
            meeting(), SR, diarize=DiarizeOptions(), separate=SeparateOptions(model="fake-separation-slow"), trace=trace
        )
    )
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.8)  # the extraction already running finishes in its worker thread
    slow = sep_core.manager.loaded["fake-separation-slow"].model
    assert isinstance(slow, FakeSeparation) and slow.voices == {}
    assert trace.status == "cancelled"


async def test_requests_that_are_refused(sep_core: Bolo) -> None:
    with pytest.raises(InvalidRequest, match="needs diarization"):
        await sep_core.transcribe(meeting(), SR, separate=SeparateOptions())
    eight = meeting()[::2].copy()
    with pytest.raises(InvalidRequest, match="16 kHz or more"):
        await sep_core.transcribe(eight, 8000, diarize=DiarizeOptions(), separate=SeparateOptions())
    t = await sep_core.transcribe(
        eight, 8000, diarize=DiarizeOptions(), separate=SeparateOptions(allow_narrowband=True)
    )
    assert t.separation is not None and t.separation["narrowband"] is True
    with pytest.raises(InvalidRequest, match="between 0 and 10"):
        await sep_core.transcribe(meeting(), SR, diarize=DiarizeOptions(), separate=SeparateOptions(context_s=20))
    with pytest.raises(InvalidRequest, match="cannot be used for separation"):
        await sep_core.transcribe(meeting(), SR, diarize=DiarizeOptions(), separate=SeparateOptions(model="fake-stt"))


async def test_blind_separator_tracks_are_matched_to_speakers(sep_core: Bolo) -> None:
    opts = SeparateOptions(model="fake-separation-blind")
    t = await sep_core.transcribe(meeting(), SR, diarize=DiarizeOptions(), separate=opts)
    assert t.separation is not None and t.separation["status"] == "applied", t.separation
    assert opts.result is not None
    for (_, speaker), track in opts.result.tracks.items():
        spectrum = np.abs(np.fft.rfft(track))
        peak = np.argmax(spectrum) * SR / len(track)
        assert abs(peak - (A_HZ if speaker == 0 else B_HZ)) < 20  # each speaker got their own tone


async def test_diarize_with_separation(sep_core: Bolo) -> None:
    opts = SeparateOptions(return_audio=True)
    dopts = DiarizeOptions()
    turns = await sep_core.diarize(meeting(), SR, dopts, separate=opts)
    assert len({t.speaker for t in turns}) == 2
    assert opts.result is not None and opts.result.status == "applied"
    doc = opts.result.to_dict(dopts.label, 16.5, audio=True)
    url = doc["windows"][0]["tracks"]["B"]
    audio, rate = sf.read(io.BytesIO(base64.b64decode(url.split(",", 1)[1])), dtype="float32")
    assert rate == SR and len(audio) == pytest.approx(
        (doc["windows"][0]["end"] - doc["windows"][0]["start"]) * SR, abs=2
    )


# ----------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------
def test_http_transcriptions_and_diarize(sep_live: LiveServer) -> None:
    url = sep_live.url
    r = httpx.post(
        url + "/v1/audio/transcriptions",
        files={"file": wav(meeting())},
        data={"model": "fake-stt", "response_format": "verbose_json", "diarize": "true", "separate_overlaps": "true"},
        timeout=60,
    )
    assert r.status_code == 200, r.text
    doc = r.json()
    assert r.headers["x-bolo-separation"] == "applied" and doc["separation"]["status"] == "applied"
    assert any(s.get("separated") for s in doc["segments"]) and all(
        s.get("overlap") for s in doc["segments"] if s.get("separated")
    )
    r = httpx.post(
        url + "/v1/audio/transcriptions",
        files={"file": wav(meeting())},
        data={"model": "fake-stt", "response_format": "diarized_json", "separate_overlaps": "true"},
        timeout=60,
    )
    assert r.status_code == 200 and r.json()["separation"]["windows"][0]["separated"] == ["A", "B"]
    assert any(s.get("separated") for s in r.json()["segments"])
    r = httpx.post(
        url + "/v1/audio/transcriptions",
        files={"file": wav(meeting()[: int(8.6 * SR)])},
        data={"model": "fake-stt", "response_format": "text", "diarize": "true", "separate_overlaps": "true"},
        timeout=60,
    )
    assert r.status_code == 200 and r.headers["x-bolo-separation"] == "skipped; reason=no_overlap"
    # refused requests
    r = httpx.post(
        url + "/v1/audio/transcriptions",
        files={"file": wav(meeting())},
        data={"model": "fake-stt", "separate_overlaps": "true"},
    )
    assert r.status_code == 400 and r.json()["error"]["param"] == "separate_overlaps"
    eight = meeting()[::2].copy()
    r = httpx.post(
        url + "/v1/audio/transcriptions",
        files={"file": wav(eight, 8000)},
        data={"model": "fake-stt", "diarize": "true", "separate_overlaps": "true"},
    )
    assert r.status_code == 400 and "16 kHz" in r.json()["error"]["message"]
    r = httpx.post(
        url + "/v1/audio/transcriptions",
        files={"file": wav(eight, 8000)},
        data={
            "model": "fake-stt",
            "diarize": "true",
            "separate_overlaps": "true",
            "response_format": "json",
            "separation_allow_narrowband": "true",
        },
        timeout=60,
    )
    assert r.status_code == 200 and r.json()["separation"]["narrowband"] is True
    r = httpx.post(
        url + "/v1/audio/transcriptions",
        files={"file": wav(meeting())},
        data={"model": "fake-stt", "diarize": "true", "separate_overlaps": "true", "separation_model": "nope"},
    )
    assert r.status_code == 404
    r = httpx.post(
        url + "/v1/audio/transcriptions",
        files={"file": wav(meeting())},
        data={"model": "fake-stt", "diarize": "true", "separate_overlaps": "true", "separation_context_s": "x"},
    )
    assert r.status_code == 400
    # native diarization endpoint, with the tracks
    r = httpx.post(
        url + "/api/v1/diarize",
        files={"file": wav(meeting())},
        data={"separate_overlaps": "true", "separation_audio": "true"},
        timeout=60,
    )
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["separation"]["status"] == "applied" and set(doc["separation"]["windows"][0]["tracks"]) == {"A", "B"}
    assert r.headers["x-bolo-separation"] == "applied"
    plain = httpx.post(url + "/api/v1/diarize", files={"file": wav(meeting())}, timeout=60).json()
    assert "separation" not in plain
    trace = httpx.get(url + f"/api/v1/requests/{doc['request_id']}").json()
    assert "separate" in [p["name"] for p in trace["phases"]] and trace["separation"] == "applied"
    info = httpx.get(url + "/api/v1/info").json()["capabilities"]["separation"]
    assert info["default_model"] == "fake-separation" and info["default_on"] is False and info["available"] is True
    metrics = httpx.get(url + "/metrics").text
    assert 'bolo_separation_requests_total{status="applied"}' in metrics
    assert "bolo_separation_audio_seconds_total" in metrics


def test_streamed_transcription_carries_separated_segments(sep_live: LiveServer) -> None:
    import json

    with httpx.stream(
        "POST",
        sep_live.url + "/v1/audio/transcriptions",
        files={"file": wav(meeting())},
        data={"model": "fake-stt", "diarize": "true", "separate_overlaps": "true", "stream": "true"},
        timeout=60,
    ) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: {")]
    segments = [e for e in events if e["type"] == "transcript.text.segment"]
    assert any(e.get("separated") for e in segments)
    assert events[-1]["type"] == "transcript.text.done" and events[-1]["separation"]["status"] == "applied"


# ----------------------------------------------------------------------
# plugins
# ----------------------------------------------------------------------
def test_separation_through_a_plugin(tmp_path) -> None:  # type: ignore[no-untyped-def]
    engine = PluginEngine(PluginConfig("ref", [sys.executable, "-m", "bolo.plugin.reference"]))
    try:
        mixture = meeting()[int(10 * SR) : int(16 * SR)]
        blind = engine.load("fake-separation", "separation", tmp_path, {"mode": "blind"}, LoadOptions())
        assert isinstance(blind, PluginSeparation) and blind.mode == "blind"
        tracks = blind.separate(mixture)
        assert len(tracks) == 2 and all(len(t) == len(mixture) for t in tracks)
        target = engine.load("fake-separation", "separation", tmp_path, {}, LoadOptions())
        assert target.mode == "target" and target.min_sample_s == 3.0
        target.enrol("k", tone(B_HZ, 0, 3.5, 3.5))
        out = target.extract("k", mixture)
        assert len(out) == len(mixture)
        assert abs(np.argmax(np.abs(np.fft.rfft(out))) * SR / len(out) - B_HZ) < 20
        target.release("k")
    finally:
        engine.shutdown()


def test_blind_tracks_are_assigned_by_total_score() -> None:
    from bolo.separate import assign_tracks, solo_energy_share

    # greedy on similarity alone would give track 0 to speaker 0 (0.60) and leave speaker 1 with track 1 (0.20);
    # the best total is the swap, which the solo energy confirms
    sims = {(0, 0): 0.60, (0, 1): 0.55, (1, 0): 0.58, (1, 1): 0.20}
    share = {(0, 0): 0.1, (0, 1): 0.9, (1, 0): 0.9, (1, 1): 0.1}
    assert assign_tracks(sims, share, 2, [0, 1]) == {0: 1, 1: 0}
    # a speaker whose voice matches no track (cosine <= 0) gets none
    assert assign_tracks({(0, 0): 0.5, (0, 1): -0.1}, {}, 1, [0, 1]) == {0: 0}
    # energy share: speaker 0 talks alone in 0-2 s of the window, speaker 1 in 4-6 s
    window = Window(10.0, 16.0, [Overlap(12.0, 14.0, [0, 1])], [0, 1])
    raw = [SpeakerTurn(5.0, 14.0, 0), SpeakerTurn(12.0, 20.0, 1)]
    t0 = tone(A_HZ, 0.0, 4.0, 6.0)  # speaker 0's voice, and silence from 4 s
    t1 = tone(B_HZ, 2.0, 6.0, 6.0)
    share = solo_energy_share([t1, t0], raw, window, [0, 1])
    assert share[(1, 0)] == pytest.approx(1.0) and share[(0, 1)] == pytest.approx(1.0) and share[(0, 0)] == 0.0


async def test_blind_tracks_keep_their_place_in_time(sep_core: Bolo) -> None:
    """Known tracks, returned in the wrong order, reach the right speakers sample for sample."""
    audio = meeting()
    source_a = tone(A_HZ, 0.0, 4.0, 16.5) + tone(A_HZ, 9.0, 14.0, 16.5)
    source_b = audio - source_a
    add_fake(sep_core, "oracle-blind", mode="blind")
    entry = await sep_core.manager.load("oracle-blind")

    def oracle(items: list[np.ndarray]) -> list[list[np.ndarray]]:
        out = []
        for mixture in items:
            start = next(i for i in range(len(audio)) if np.array_equal(audio[i : i + len(mixture)], mixture))
            out.append([source_b[start : start + len(mixture)], source_a[start : start + len(mixture)]])
        return out

    entry.queue._handlers["separate"] = oracle
    opts = SeparateOptions(model="oracle-blind")
    await sep_core.diarize(audio, SR, DiarizeOptions(), separate=opts)
    assert opts.result is not None and opts.result.status == "applied"
    (window,) = opts.result.windows
    a = int(window.start * SR)
    for speaker, source in ((0, source_a), (1, source_b)):
        track = opts.result.tracks[(0, speaker)]
        assert np.allclose(track, source[a : a + len(track)], atol=1e-6)
