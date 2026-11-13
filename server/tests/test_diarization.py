from __future__ import annotations

import base64
import io
import json

import httpx
import numpy as np
import openai
import pytest
import websockets

from bolo.audio.encode import encode_audio
from bolo.core import Bolo
from bolo.diarize import (
    DiarizeOptions,
    OnlineSpeakerLabeler,
    cluster_embeddings,
    diarization_error_rate,
    label_speakers,
    match_known,
    merge_turns,
    renumber,
    speaker_label,
)
from bolo.engines.base import SpeakerTurn
from bolo.errors import InvalidRequest
from bolo.obs.trace import Trace
from conftest import LiveServer, two_speakers


def wav(audio: np.ndarray) -> io.BytesIO:
    f = io.BytesIO(encode_audio(audio, 16000, "wav"))
    f.name = "talk.wav"
    return f


def test_labels_merge_and_renumber() -> None:
    assert [speaker_label(i) for i in (0, 1, 25, 26)] == ["A", "B", "Z", "S27"]
    turns = [
        SpeakerTurn(5.0, 6.0, 7),
        SpeakerTurn(0.0, 1.0, 3),
        SpeakerTurn(1.2, 2.0, 3),
        SpeakerTurn(2.1, 2.2, 7),
        SpeakerTurn(2.3, 4.0, 3),
    ]
    ordered = renumber(turns)
    assert [t.speaker for t in ordered] == [0, 0, 1, 0, 1]  # numbered by first appearance
    merged = merge_turns(ordered)
    assert [(t.start, t.end, t.speaker) for t in merged] == [
        (0.0, 4.0, 0),
        (5.0, 6.0, 1),
    ]  # the 0.1 s blip is dropped, its neighbours joined
    long = merge_turns([SpeakerTurn(0, 20, 0), SpeakerTurn(20.5, 40, 0)], max_len_s=30)
    assert len(long) == 2  # never merged beyond what one decode can take


def test_diarization_error_rate() -> None:
    ref = [SpeakerTurn(0, 10, 0), SpeakerTurn(10, 20, 1)]
    assert (
        diarization_error_rate(ref, [SpeakerTurn(0, 10, 5), SpeakerTurn(10, 20, 2)])["der"] == 0.0
    )  # labels are arbitrary
    half_wrong = diarization_error_rate(ref, [SpeakerTurn(0, 20, 0)])
    assert half_wrong["confusion"] == pytest.approx(0.5, abs=0.01) and half_wrong["der"] == pytest.approx(0.5, abs=0.01)
    missed = diarization_error_rate(ref, [SpeakerTurn(0, 10, 0), SpeakerTurn(10, 15, 1)])
    assert missed["missed"] == pytest.approx(0.25, abs=0.01) and missed["confusion"] == 0.0
    extra = diarization_error_rate([SpeakerTurn(0, 10, 0)], [SpeakerTurn(0, 12, 0)])
    assert extra["false_alarm"] == pytest.approx(0.2, abs=0.01)
    shifted = [SpeakerTurn(0, 10.2, 0), SpeakerTurn(10.2, 20, 1)]
    assert (
        diarization_error_rate(ref, shifted)["der"] > 0
        and diarization_error_rate(ref, shifted, collar_s=0.25)["der"] == 0.0
    )
    assert diarization_error_rate([], [])["der"] == 0.0


def test_online_labeler() -> None:
    a, b = np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)
    near_a = np.array([0.95, 0.3], dtype=np.float32)
    labeler = OnlineSpeakerLabeler(threshold=0.5)
    assert [labeler.assign(e) for e in (a, b, near_a, b, a)] == [0, 1, 0, 1, 0]
    assert labeler.num_speakers == 2
    capped = OnlineSpeakerLabeler(threshold=0.99, max_speakers=1)
    assert [capped.assign(e) for e in (a, b)] == [0, 0]


async def test_core_diarize_and_speaker_count_limits(core: Bolo) -> None:
    audio = two_speakers(4)
    trace = Trace("diarization", "http", "")
    turns = await core.diarize(audio, 16000, trace=trace)
    assert [t.speaker for t in turns] == [0, 1, 0, 1]
    assert turns[0].start == pytest.approx(0.0, abs=0.15) and turns[0].end == pytest.approx(1.2, abs=0.2)
    assert [p.name for p in trace.phases] == ["resample", "diarize"] and trace.extra["speakers"] == 2
    assert trace.status == "ok" and trace.rtf is not None
    model = core.manager.loaded["fake-diarization"].model
    one = await core.diarize(audio, 16000, DiarizeOptions(max_speakers=1))
    assert {t.speaker for t in one} == {0}
    assert model.calls[-2:] == [(None, None), (1, None)]  # type: ignore[attr-defined]  re-clustered with the cap
    await core.diarize(audio, 16000, DiarizeOptions(num_speakers=2, threshold=0.7))
    assert model.calls[-1] == (2, 0.7)  # type: ignore[attr-defined]
    with pytest.raises(InvalidRequest, match="between 1 and 32"):
        await core.diarize(audio, 16000, DiarizeOptions(num_speakers=0))
    with pytest.raises(InvalidRequest, match="must not exceed"):
        await core.diarize(audio, 16000, DiarizeOptions(min_speakers=3, max_speakers=2))


def test_transcription_with_speakers_verbose_and_subtitles(live: LiveServer) -> None:
    client = openai.OpenAI(base_url=live.url + "/v1", api_key="x", max_retries=0)
    audio = two_speakers(4)
    raw = httpx.post(
        live.url + "/v1/audio/transcriptions",
        files={"file": wav(audio)},
        data={"model": "fake-stt", "response_format": "verbose_json", "diarize": "true"},
    )
    assert raw.status_code == 200, raw.text
    doc = raw.json()
    assert [s["speaker"] for s in doc["segments"]] == ["A", "B", "A", "B"]
    assert [round(s["start"]) for s in doc["segments"]] == [0, 2, 4, 5]
    assert doc["text"] == " ".join(s["text"] for s in doc["segments"])
    srt = client.audio.transcriptions.create(
        model="fake-stt", file=wav(audio), response_format="srt", extra_body={"diarize": "true"}
    )
    assert "[A] the quick brown" in srt and "[B] the quick brown" in srt
    vtt = client.audio.transcriptions.create(
        model="fake-stt", file=wav(audio), response_format="vtt", extra_body={"diarize": "true"}
    )
    assert "<v A>the quick brown" in vtt and "<v B>" in vtt
    text = client.audio.transcriptions.create(
        model="fake-stt", file=wav(audio), response_format="text", extra_body={"diarize": "true"}
    )
    assert text.splitlines()[0].startswith("A: ") and text.splitlines()[1].startswith("B: ")
    trace = httpx.get(live.url + "/api/v1/requests", params={"kind": "stt", "limit": 1}).json()["requests"][0]
    assert "diarize" in [p["name"] for p in trace["phases"]] and trace["speakers"] == 2
    assert trace["diarization_model"] == "fake-diarization"
    plain = httpx.post(
        live.url + "/v1/audio/transcriptions",
        files={"file": wav(audio)},
        data={"model": "fake-stt", "response_format": "verbose_json"},
    )
    assert "speaker" not in plain.json()["segments"][0]


def test_openai_diarized_json_shape_through_the_sdk(live: LiveServer) -> None:
    client = openai.OpenAI(base_url=live.url + "/v1", api_key="x", max_retries=0)
    result = client.audio.transcriptions.create(
        model="gpt-4o-transcribe-diarize",
        file=wav(two_speakers(4)),
        response_format="diarized_json",
        chunking_strategy="auto",
    )
    assert type(result).__name__ == "TranscriptionDiarized"
    assert result.task == "transcribe" and result.duration == pytest.approx(7.2, abs=0.1)
    assert [s.speaker for s in result.segments] == ["A", "B", "A", "B"]
    assert all(s.type == "transcript.text.segment" and s.id and s.end > s.start and s.text for s in result.segments)
    assert result.usage is not None and result.usage.type == "duration"
    # the model name alone switches diarization on
    by_name = httpx.post(
        live.url + "/v1/audio/transcriptions",
        files={"file": wav(two_speakers(2))},
        data={"model": "gpt-4o-transcribe-diarize", "response_format": "verbose_json"},
    )
    assert [s["speaker"] for s in by_name.json()["segments"]] == ["A", "B"]
    forced = httpx.post(
        live.url + "/v1/audio/transcriptions",
        files={"file": wav(two_speakers(4))},
        data={"model": "fake-stt", "response_format": "diarized_json", "max_speakers": "1"},
    )
    assert {s["speaker"] for s in forced.json()["segments"]} == {"A"}


def test_diarized_streaming_emits_segment_events(live: LiveServer) -> None:
    client = openai.OpenAI(base_url=live.url + "/v1", api_key="x", max_retries=0)
    events = list(
        client.audio.transcriptions.create(
            model="fake-stt", file=wav(two_speakers(4)), response_format="diarized_json", stream=True
        )
    )
    segments = [e for e in events if e.type == "transcript.text.segment"]
    assert [s.speaker for s in segments] == ["A", "B", "A", "B"] and events[-1].type == "transcript.text.done"
    assert not [e for e in events if e.type == "transcript.text.delta"]


def tone(freq: float, seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(seconds * 16000)) / 16000
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def data_url(audio: np.ndarray) -> str:
    return "data:audio/wav;base64," + base64.b64encode(encode_audio(audio, 16000, "wav")).decode()


def test_words_carry_speakers_when_diarization_and_word_timestamps_are_combined(live: LiveServer) -> None:
    r = httpx.post(
        live.url + "/v1/audio/transcriptions",
        files={"file": wav(two_speakers(4))},
        data={"model": "fake-stt", "response_format": "verbose_json", "diarize": "true",
              "timestamp_granularities[]": ["word", "segment"]},
    )  # fmt: skip
    assert r.status_code == 200, r.text
    doc = r.json()
    assert [s["speaker"] for s in doc["segments"]] == ["A", "B", "A", "B"]
    assert len(doc["words"]) >= 8 and {w["speaker"] for w in doc["words"]} == {"A", "B"}
    for word in doc["words"]:  # every word lies inside a segment of its own speaker
        middle = (word["start"] + word["end"]) / 2
        (segment,) = [s for s in doc["segments"] if s["start"] - 0.05 <= middle <= s["end"] + 0.05]
        assert segment["speaker"] == word["speaker"]
    assert " ".join(w["word"] for w in doc["words"]) == doc["text"]


def test_known_speakers_are_recognised_by_voice(live: LiveServer) -> None:
    client = openai.OpenAI(base_url=live.url + "/v1", api_key="x", max_retries=0)
    # the fake diarizer's "voices" are pitches: the 420 Hz speaker is the agent, the 150 Hz one the caller
    result = client.audio.transcriptions.create(
        model="gpt-4o-transcribe-diarize", file=wav(two_speakers(4)), response_format="diarized_json",
        chunking_strategy="auto", known_speaker_names=["agent", "caller"],
        known_speaker_references=[data_url(tone(420)), data_url(tone(150))],
    )  # fmt: skip
    assert [s.speaker for s in result.segments] == ["caller", "agent", "caller", "agent"]
    # only one of the two is known: the other keeps a letter
    r = httpx.post(
        live.url + "/api/v1/diarize", files={"file": wav(two_speakers(4))},
        data={"known_speaker_names[]": "agent", "known_speaker_references[]": data_url(tone(420))},
    )  # fmt: skip
    doc = r.json()
    assert r.status_code == 200 and doc["speakers"] == ["A", "agent"]
    assert [s["speaker"] for s in doc["segments"]] == ["A", "agent", "A", "agent"]
    assert doc["known_speakers"] == [{"name": "agent", "matched": True, "similarity": pytest.approx(1.0, abs=0.01)}]
    trace = httpx.get(live.url + f"/api/v1/requests/{doc['request_id']}").json()
    assert "identify" in [p["name"] for p in trace["phases"]] and trace["known_speakers"] == {"agent": 1.0}
    # a reference that is not similar enough to anyone names nobody; a file part works like a data URL
    r = httpx.post(
        live.url + "/api/v1/diarize",
        files={"file": wav(two_speakers(4)), "known_speaker_references[]": ("ref.wav", wav(tone(300)).getvalue())},
        data={"known_speaker_names[]": "stranger", "identify_threshold": "0.95"},
    )
    doc = r.json()
    assert r.status_code == 200, r.text
    assert doc["speakers"] == ["A", "B"] and doc["known_speakers"][0]["matched"] is False
    assert 0 < doc["known_speakers"][0]["similarity"] < 0.95


def test_clustering_and_label_helpers() -> None:
    def unit(degrees: float) -> np.ndarray:
        return np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees))], dtype=np.float32)

    assert cluster_embeddings([], 0.5) == []
    assert cluster_embeddings([unit(0), unit(80), unit(5), unit(85)], 0.5) == [0, 1, 0, 1]
    assert cluster_embeddings([unit(0), unit(80), unit(5), unit(85)], 0.5, max_speakers=1) == [0, 0, 0, 0]
    names, best = match_known({0: unit(0), 1: unit(80)}, [("ann", unit(78)), ("bob", unit(40))], 0.9)
    assert names == {1: "ann"} and best == {"ann": pytest.approx(0.999, abs=0.01), "bob": pytest.approx(0.77, abs=0.01)}
    # one name per cluster and one cluster per name, best pair first
    names, _ = match_known({0: unit(0), 1: unit(10)}, [("ann", unit(2)), ("bob", unit(4))], 0.5)
    assert names == {0: "ann", 1: "bob"}
    assert label_speakers([0, 1, 2], {1: "ann"}) == {0: "A", 1: "ann", 2: "B"}
    assert label_speakers([0, 1], {0: "A"}) == {0: "A", 1: "B"}  # a known speaker may be called "A"

    # online, the second utterance opens a new speaker; with the whole stream known it joins the first
    labeler = OnlineSpeakerLabeler(threshold=0.45)
    assert [labeler.assign(unit(d)) for d in (0, 65, 32)] == [0, 1, 0]
    assert labeler.revise() == [0, 0, 0]


def test_diarization_request_errors(live: LiveServer) -> None:
    def post(**data: str) -> httpx.Response:
        return httpx.post(
            live.url + "/v1/audio/transcriptions",
            files={"file": wav(two_speakers(2))},
            data={"model": "fake-stt", **data},
        )

    assert post(diarize="true", num_speakers="many").json()["error"]["param"] == "num_speakers"
    assert post(diarize="true", num_speakers="99").status_code == 400
    assert post(diarize="true", diarization_model="nope").status_code == 404
    # known speakers: names and references must pair up, and a reference must be audio
    r = post(diarize="true", **{"known_speaker_names[]": "agent"})
    assert r.status_code == 400 and "same number" in r.json()["error"]["message"]
    r = post(diarize="true", **{"known_speaker_names[]": "agent", "known_speaker_references[]": "not audio"})
    assert r.status_code == 400 and "could not be read" in r.json()["error"]["message"]
    r = post(
        diarize="true", **{"known_speaker_names[]": "agent", "known_speaker_references[]": data_url(tone(420, 0.3))}
    )
    assert r.status_code == 400 and "0.3 s long" in r.json()["error"]["message"]
    assert (
        httpx.post(
            live.url + "/v1/audio/translations",
            files={"file": wav(two_speakers(2))},
            data={"model": "fake-stt", "diarize": "true"},
        ).status_code
        == 400
    )


def test_native_diarize_endpoint(live: LiveServer) -> None:
    r = httpx.post(live.url + "/api/v1/diarize", files={"file": wav(two_speakers(4))})
    assert r.status_code == 200, r.text
    doc = r.json()
    assert (
        doc["model"] == "fake-diarization"
        and doc["speakers"] == ["A", "B"]
        and doc["duration"] == pytest.approx(7.2, abs=0.1)
    )
    assert [s["speaker"] for s in doc["segments"]] == ["A", "B", "A", "B"] and set(doc["segments"][0]) == {
        "start",
        "end",
        "speaker",
    }
    trace = httpx.get(live.url + f"/api/v1/requests/{doc['request_id']}").json()
    assert (
        trace["kind"] == "diarization" and trace["status"] == "ok" and "diarize" in [p["name"] for p in trace["phases"]]
    )
    assert httpx.post(
        live.url + "/api/v1/diarize", files={"file": wav(two_speakers(2))}, data={"num_speakers": "1"}
    ).json()["speakers"] == ["A"]
    assert httpx.post(live.url + "/api/v1/diarize", data={}).status_code == 400
    assert (
        httpx.post(
            live.url + "/api/v1/diarize", files={"file": wav(two_speakers(2))}, data={"model": "fake-stt"}
        ).status_code
        == 400
    )
    cat = httpx.get(live.url + "/api/v1/catalog", params={"task": "diarization"}).json()["models"]
    assert "fake-diarization" in [m["id"] for m in cat] and len(cat) >= 3
    caps = httpx.get(live.url + "/api/v1/info").json()["capabilities"]
    assert caps["diarization"]["available"] is True and caps["diarization"]["default_model"] == "fake-diarization"


@pytest.mark.parametrize("model", ["fake-stt", "fake-stt-streaming"])
async def test_websocket_finals_carry_speaker_labels(live: LiveServer, model: str) -> None:
    pcm = (two_speakers(4, gap=0.9) * 32767).astype("<i2").tobytes()
    async with websockets.connect(live.ws_url + f"/api/v1/stt/stream?model={model}&diarize=true") as ws:
        ready = json.loads(await ws.recv())
        assert ready["diarization"] == "fake-diarization"
        for off in range(0, len(pcm), 3200):
            await ws.send(pcm[off : off + 3200])
            import asyncio

            await asyncio.sleep(0.01)
        await ws.send(json.dumps({"type": "stop"}))
        messages = [json.loads(x) async for x in ws]
        finals = [m for m in messages if m["type"] == "final"]
    assert [f["speaker"] for f in finals] == ["A", "B", "A", "B"]
    # before "done": the labels once more, after clustering the whole stream (nothing changes here)
    assert [m["type"] for m in messages[-2:]] == ["speakers", "done"]
    summary = messages[-2]
    assert summary["revised"] == 0 and summary["speakers"] == ["A", "B"]
    assert [u["speaker"] for u in summary["utterances"]] == ["A", "B", "A", "B"]
    assert [u["utterance"] for u in summary["utterances"]] == [1, 2, 3, 4] and "was" not in summary["utterances"][0]
    trace = httpx.get(live.url + f"/api/v1/requests/{ready['request_id']}").json()
    assert "diarize" in [p["name"] for p in trace["phases"]]
    async with websockets.connect(live.ws_url + f"/api/v1/stt/stream?model={model}") as ws:
        assert json.loads(await ws.recv())["diarization"] is None


def test_spurious_speakers_are_folded_into_real_ones() -> None:
    from bolo.diarize import consolidate_speakers, merge_same_speaker

    def unit(*v: float) -> np.ndarray:
        a = np.array(v, dtype=np.float32)
        return a / np.linalg.norm(a)

    # speaker 0 and 1 talk for a minute each; 2 is a two-second blip that sounds like 1; 3 is 0 on another microphone
    turns = [SpeakerTurn(0, 60, 0), SpeakerTurn(60, 120, 1), SpeakerTurn(120, 122, 2), SpeakerTurn(122, 150, 3)]
    voices = {0: unit(1, 0, 0), 1: unit(0, 1, 0), 2: unit(0.2, 1, 0.3), 3: unit(1, 0.1, 0.6)}
    assert consolidate_speakers(turns, voices) == {0: 0, 3: 0, 1: 1, 2: 1}
    # each repair can be switched off, and min_speakers is a floor
    assert consolidate_speakers(turns, voices, min_speech_s=0, merge_similarity=2.0) == {0: 0, 1: 1, 2: 2, 3: 3}
    assert len(set(consolidate_speakers(turns, voices, min_speakers=3).values())) == 3
    # without a voice print (too little audio) the blip joins the speaker next to it in time
    assert consolidate_speakers(turns, {k: v for k, v in voices.items() if k != 2}, merge_similarity=2.0)[2] in (1, 3)
    # on a short recording a short answer is still a speaker: the floor is a tenth of the speech
    short = [SpeakerTurn(0, 8, 0), SpeakerTurn(8, 10, 1)]
    assert consolidate_speakers(short, {0: unit(1, 0), 1: unit(0, 1)}) == {0: 0, 1: 1}
    merged = merge_same_speaker(
        [SpeakerTurn(0, 5, 0), SpeakerTurn(5, 7, 0), SpeakerTurn(7, 9, 1), SpeakerTurn(9.5, 10, 1)]
    )
    assert [(t.start, t.end, t.speaker) for t in merged] == [(0, 7, 0), (7, 9, 1), (9.5, 10, 1)]
