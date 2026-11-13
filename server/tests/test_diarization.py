from __future__ import annotations

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
    diarization_error_rate,
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


def test_diarization_request_errors(live: LiveServer) -> None:
    def post(**data: str) -> httpx.Response:
        return httpx.post(
            live.url + "/v1/audio/transcriptions",
            files={"file": wav(two_speakers(2))},
            data={"model": "fake-stt", **data},
        )

    assert post(diarize="true", num_speakers="many").json()["error"]["param"] == "num_speakers"
    assert post(diarize="true", num_speakers="99").status_code == 400
    r = post(diarize="true", response_format="verbose_json", **{"timestamp_granularities[]": "word"})
    assert r.status_code == 400 and "word timestamps" in r.json()["error"]["message"]
    assert post(diarize="true", diarization_model="nope").status_code == 404
    assert post(diarize="true", **{"known_speaker_names[]": "agent"}).status_code == 400
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
        finals = [m for m in [json.loads(x) async for x in ws] if m["type"] == "final"]
    assert [f["speaker"] for f in finals] == ["A", "B", "A", "B"]
    trace = httpx.get(live.url + f"/api/v1/requests/{ready['request_id']}").json()
    assert "diarize" in [p["name"] for p in trace["phases"]]
    async with websockets.connect(live.ws_url + f"/api/v1/stt/stream?model={model}") as ws:
        assert json.loads(await ws.recv())["diarization"] is None
