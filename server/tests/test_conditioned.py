"""Diarization-conditioned recognition: a speaker-conditioned STT model decodes the whole recording once per
speaker, told who talks when (fake diarizer, fake recogniser, and the reference plugin out of process)."""

from __future__ import annotations

import io
import sys

import httpx
import numpy as np
import pytest

from bolo.audio.encode import encode_audio
from bolo.catalog import ModelSpec
from bolo.config import Settings
from bolo.core import Bolo
from bolo.diarize import DiarizeOptions, conditioning_activity, turns_to_activity
from bolo.engines.base import LoadOptions, SpeakerActivity, SpeakerRequest, SpeakerTurn
from bolo.engines.fake import FakeSpeakerSTT
from bolo.errors import InvalidRequest
from bolo.plugin import protocol
from bolo.plugin.host import PluginConfig, PluginEngine, PluginSTT
from bolo.separate import SeparateOptions
from conftest import LiveServer

SR = 16000
A_HZ, B_HZ = 300.0, 500.0


def tone(freq: float, start: float, end: float, total: float, amp: float = 0.3) -> np.ndarray:
    out = np.zeros(int(total * SR), dtype=np.float32)
    t = np.arange(int((end - start) * SR)) / SR
    out[int(start * SR) : int(start * SR) + len(t)] = amp * np.sin(2 * np.pi * freq * t)
    return out


def meeting() -> np.ndarray:
    """A alone 0-4 s, B alone 4.5-8.5 s, A 9-14 s, B 12-16 s: a 2 s overlap from 12 to 14 s."""
    total = 16.5
    return (
        tone(A_HZ, 0.0, 4.0, total)
        + tone(B_HZ, 4.5, 8.5, total)
        + tone(A_HZ, 9.0, 14.0, total)
        + tone(B_HZ, 12.0, 16.0, total)
    )


def add_conditioned(core: Bolo, model_id: str = "fake-dicow", engine: str = "fake", **options: object) -> None:
    core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": model_id, "task": "stt", "engine": engine, "kind": "fake-stt-speakers", "family": "Fake",
                "languages": ["en"], "license": "Apache-2.0", "builtin": True,
                "features": ["timestamps", "speaker-conditioned"], "options": dict(options),
            }
        )
    )  # fmt: skip


@pytest.fixture
def cond_settings(settings: Settings) -> Settings:
    settings.default_diarization = "fake-diarization-overlap"
    return settings


@pytest.fixture
async def core(cond_settings: Settings):  # type: ignore[no-untyped-def]
    c = Bolo(cond_settings)
    add_conditioned(c)
    await c.start()
    try:
        yield c
    finally:
        await c.stop()


def fake_model(core: Bolo, model_id: str = "fake-dicow") -> FakeSpeakerSTT:
    model = core.manager.loaded[model_id].model
    assert isinstance(model, FakeSpeakerSTT)
    return model


# ----------------------------------------------------------------------
# activity helpers
# ----------------------------------------------------------------------
def test_turns_to_activity_and_the_choice_of_conditioning() -> None:
    turns = [SpeakerTurn(0.0, 1.0, 0), SpeakerTurn(0.5, 2.0, 1), SpeakerTurn(2.5, 3.0, 0)]
    act = turns_to_activity(turns, 3.2)
    assert act.probs.shape == (320, 2) and act.frame_s == 0.01
    assert act.probs[:100, 0].all() and not act.probs[100:250, 0].any() and act.probs[250:300, 0].all()
    assert act.probs[50:200, 1].all() and act.probs[:50, 1].sum() == 0 and act.probs[200:, 1].sum() == 0
    assert set(np.unique(act.probs)) <= {0.0, 1.0}
    assert turns_to_activity([], 1.0).probs.shape == (100, 0)
    soft = SpeakerActivity(np.full((320, 2), 0.7, np.float32), 0.01)
    assert conditioning_activity(turns, soft, 3.2, "turns")[1] == "turns"
    chosen, kind = conditioning_activity(turns, soft, 3.2, "probabilities")
    assert kind == "probabilities" and chosen is soft
    # without probabilities (a clustering diarizer), or with columns that do not match the speakers: the turns
    assert conditioning_activity(turns, None, 3.2, "probabilities")[1] == "turns"
    three = SpeakerActivity(np.zeros((320, 3), np.float32), 0.01)
    assert conditioning_activity(turns, three, 3.2, "probabilities")[1] == "turns"
    with pytest.raises(ValueError):
        conditioning_activity(turns, None, 3.2, "magic")


def test_speakers_payload_roundtrip() -> None:
    audio = np.linspace(-1, 1, 11, dtype=np.float32)
    probs = np.arange(12, dtype=np.float32).reshape(6, 2) / 12
    payload = protocol.speaker_payload(audio, probs)
    got_audio, got_probs = protocol.split_speaker_payload(payload, 11, 6, 2)
    assert np.array_equal(got_audio, audio) and np.array_equal(got_probs, probs)
    with pytest.raises(protocol.ProtocolError):
        protocol.split_speaker_payload(payload, 11, 6, 3)


# ----------------------------------------------------------------------
# through the core
# ----------------------------------------------------------------------
async def test_one_decode_per_speaker_over_the_whole_recording(core: Bolo) -> None:
    opts = DiarizeOptions()
    t = await core.transcribe(meeting(), SR, model="fake-dicow", diarize=opts)
    model = fake_model(core)
    assert len(model.requests) == 1 and model.calls == []  # no turn-by-turn decoding
    request = model.requests[0]
    assert len(request.samples) == len(meeting()) and request.activity.num_speakers == 2
    assert set(np.unique(request.activity.probs)) <= {0.0, 1.0}  # conditioned on the turns by default
    # speaker A's column covers both of A's turns, including the overlap where B talks too
    act = request.activity
    assert act.frame_s == 0.02  # the 50 Hz of a Whisper encoder's input, as DiCoW's reference pipeline uses
    assert (
        act.probs[act.frame(1.0), 0] == 1 and act.probs[act.frame(13.0), 0] == 1 and act.probs[act.frame(6.0), 0] == 0
    )
    assert [s.speaker for s in t.segments] == ["A", "B", "A", "B"]
    assert [s.start for s in t.segments] == sorted(s.start for s in t.segments)
    assert all(s.text.startswith("speaker0" if s.speaker == "A" else "speaker1") for s in t.segments)
    # both speakers' overlapping turns are transcribed in full, and marked as overlapped
    late = [s for s in t.segments if s.start > 8.9]
    assert [s.speaker for s in late] == ["A", "B"] and late[0].end > 13.5 and late[1].start < 12.5
    assert all(s.overlap for s in late) and not any(s.overlap for s in t.segments if s.end < 8.9)
    assert t.text == " ".join(s.text for s in t.segments) and t.model == "fake-dicow"
    trace = core.traces.recent(1)[0]
    assert trace.extra["conditioning"] == "turns" and trace.extra["speakers"] == 2
    assert "decode" in {p.name for p in trace.phases} and "diarize" in {p.name for p in trace.phases}


async def test_without_diarization_it_is_an_ordinary_recogniser(core: Bolo) -> None:
    t = await core.transcribe(meeting(), SR, model="fake-dicow", vad_filter=False)
    model = fake_model(core)
    assert model.requests == [] and model.calls == [1] and t.text and all(s.speaker is None for s in t.segments)


async def test_probabilities_conditioning_uses_the_diarizer_probabilities(cond_settings: Settings) -> None:
    c = Bolo(cond_settings)
    add_conditioned(c, conditioning="probabilities")
    c.catalog.add(
        ModelSpec.from_dict(
            {
                "id": "fake-diarization-soft", "task": "diarization", "engine": "fake", "kind": "fake-diarization",
                "family": "Fake", "languages": [], "license": "Apache-2.0", "builtin": True,
                "options": {"multi_pitch": True, "activity": True},
            }
        )
    )  # fmt: skip
    await c.start()
    try:
        t = await c.transcribe(meeting(), SR, model="fake-dicow", diarize=DiarizeOptions(model="fake-diarization-soft"))
        request = fake_model(c).requests[0]
        assert np.allclose(np.unique(request.activity.probs), [0.05, 0.9])
        assert [s.speaker for s in t.segments] == ["A", "B", "A", "B"]
        assert c.traces.recent(1)[0].extra["conditioning"] == "probabilities"
        # the clustering-style fake diarizer gives no probabilities: the turns are used instead
        await c.transcribe(meeting(), SR, model="fake-dicow", diarize=DiarizeOptions())
        assert set(np.unique(fake_model(c).requests[-1].activity.probs)) <= {0.0, 1.0}
        assert c.traces.recent(1)[0].extra["conditioning"] == "turns"
    finally:
        await c.stop()


async def test_preferred_diarizer_and_refused_combinations(cond_settings: Settings) -> None:
    c = Bolo(cond_settings)
    add_conditioned(c, "fake-dicow-pref", diarization_model="fake-diarization-overlap")
    add_conditioned(c, "fake-dicow-bad", conditioning="magic")
    await c.start()
    try:
        c.settings.default_diarization = "fake-diarization"  # the clustering fake: hears one tone at a time
        opts = DiarizeOptions()
        await c.transcribe(meeting(), SR, model="fake-dicow-pref", diarize=opts)
        assert opts.model == "fake-diarization-overlap" and c.traces.recent(1)[0].extra["diarization_model"] == (
            "fake-diarization-overlap"
        )
        named = DiarizeOptions(model="fake-diarization")
        await c.transcribe(meeting(), SR, model="fake-dicow-pref", diarize=named)
        assert c.traces.recent(1)[0].extra["diarization_model"] == "fake-diarization"  # the request wins
        with pytest.raises(InvalidRequest, match="separate_overlaps") as err:
            await c.transcribe(
                meeting(), SR, model="fake-dicow-pref", diarize=DiarizeOptions(), separate=SeparateOptions()
            )
        assert err.value.param == "separate_overlaps"
        with pytest.raises(InvalidRequest, match="conditioning"):
            await c.transcribe(meeting(), SR, model="fake-dicow-bad", diarize=DiarizeOptions())
        # silence: no speakers, no decode
        t = await c.transcribe(np.zeros(SR * 2, np.float32), SR, model="fake-dicow-pref", diarize=DiarizeOptions())
        assert t.text == "" and t.segments == []
        caps = c.capabilities()["diarization"]["speaker_conditioned_models"]
        assert {"fake-dicow-pref", "fake-dicow-bad"} <= set(caps)
    finally:
        await c.stop()


def test_diarized_json_over_http(cond_settings: Settings) -> None:
    server = LiveServer(cond_settings)
    add_conditioned(server.core)
    server.start()
    try:
        f = io.BytesIO(encode_audio(meeting(), SR, "wav"))
        f.name = "talk.wav"
        r = httpx.post(
            server.url + "/v1/audio/transcriptions",
            files={"file": f},
            data={"model": "fake-dicow", "response_format": "diarized_json"},
            timeout=60,
        )
        assert r.status_code == 200, r.text
        segs = r.json()["segments"]
        assert [s["speaker"] for s in segs] == ["A", "B", "A", "B"] and segs[2].get("overlap") is True
        f.seek(0)
        r = httpx.post(
            server.url + "/v1/audio/transcriptions",
            files={"file": f},
            data={"model": "fake-dicow", "diarize": "true", "separate_overlaps": "true"},
            timeout=60,
        )
        assert r.status_code == 400 and r.json()["error"]["param"] == "separate_overlaps"
    finally:
        server.stop()


# ----------------------------------------------------------------------
# out of process: the plugin protocol's "speakers" op
# ----------------------------------------------------------------------
def test_speakers_through_the_reference_plugin(tmp_path) -> None:  # type: ignore[no-untyped-def]
    engine = PluginEngine(PluginConfig("ref", [sys.executable, "-m", "bolo.plugin.reference"], idle_s=-1))
    try:
        model = engine.load("fake-stt-speakers", "stt", tmp_path, {}, LoadOptions())
        assert isinstance(model, PluginSTT) and model.speaker_conditioned
        probs = np.zeros((300, 2), np.float32)
        probs[:150, 0] = 1.0
        probs[100:280, 1] = 0.8
        results = model.transcribe_speakers(SpeakerRequest(np.zeros(3 * SR, np.float32), SpeakerActivity(probs)))
        assert len(results) == 2
        assert [(s.start, s.end) for s in results[0].segments] == [(0.0, 1.5)]
        assert [(round(s.start, 2), round(s.end, 2)) for s in results[1].segments] == [(1.0, 2.8)]
        assert results[1].text.startswith("speaker1")
        plain = engine.load("fake-stt", "stt", tmp_path, {}, LoadOptions())
        assert isinstance(plain, PluginSTT) and not plain.speaker_conditioned
        with pytest.raises(InvalidRequest):
            plain.transcribe_speakers(SpeakerRequest(np.zeros(SR, np.float32), SpeakerActivity(probs)))
        assert engine.process.crashes == 0
    finally:
        engine.shutdown()


async def test_plugin_model_through_the_core(cond_settings: Settings) -> None:
    cond_settings.plugins = [f"ref={sys.executable} -m bolo.plugin.reference"]
    cond_settings.plugin_entry_points = False
    c = Bolo(cond_settings)
    add_conditioned(c, "plug-dicow", engine="plugin:ref")
    await c.start()
    try:
        t = await c.transcribe(meeting(), SR, model="plug-dicow", diarize=DiarizeOptions())
        assert [s.speaker for s in t.segments] == ["A", "B", "A", "B"]
        assert t.segments[2].overlap and t.segments[2].text.startswith("speaker0")
    finally:
        await c.stop()
