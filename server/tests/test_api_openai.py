"""The OpenAI-compatible endpoints, exercised through the official ``openai`` SDK."""

from __future__ import annotations

import base64
import io

import httpx
import numpy as np
import openai
import pytest

from bolo.audio import decode_audio, mulaw_decode
from conftest import LiveServer, wav_bytes, wav_file


@pytest.fixture
def client(live: LiveServer) -> openai.OpenAI:
    return openai.OpenAI(base_url=live.url + "/v1", api_key="not-needed", max_retries=0)


def test_models_list_and_retrieve(client: openai.OpenAI) -> None:
    models = client.models.list()
    ids = [m.id for m in models.data]
    assert {"fake-stt", "fake-stt-streaming", "fake-tts", "fake-vad"} <= set(ids)
    m = client.models.retrieve("fake-tts")
    assert m.id == "fake-tts" and m.object == "model" and m.owned_by == "fake"
    with pytest.raises(openai.NotFoundError) as exc:
        client.models.retrieve("nope")
    assert exc.value.body["code"] == "model_not_found"  # type: ignore[index]


def test_transcription_json(client: openai.OpenAI) -> None:
    result = client.audio.transcriptions.create(model="whisper-1", file=wav_file(2.0))
    assert result.text == "the quick brown fox jumps"
    assert result.usage is not None and result.usage.type == "duration" and result.usage.seconds == 2


def test_transcription_text_srt_vtt(client: openai.OpenAI) -> None:
    text = client.audio.transcriptions.create(model="fake-stt", file=wav_file(2.0), response_format="text")
    assert text == "the quick brown fox jumps\n"
    srt = client.audio.transcriptions.create(model="fake-stt", file=wav_file(2.0), response_format="srt")
    assert srt == "1\n00:00:00,000 --> 00:00:02,000\nthe quick brown fox jumps\n"
    vtt = client.audio.transcriptions.create(model="fake-stt", file=wav_file(2.0), response_format="vtt")
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n")


def test_transcription_verbose_json_with_word_timestamps(client: openai.OpenAI) -> None:
    result = client.audio.transcriptions.create(
        model="fake-stt",
        file=wav_file(2.0),
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
        language="en",
    )
    assert result.text == "the quick brown fox jumps"
    assert result.language == "english" and result.duration == 2.0
    assert result.words is not None and [w.word for w in result.words] == result.text.split()
    assert result.words[0].start == 0.0 and result.words[0].end == pytest.approx(0.4)
    assert result.segments is not None and len(result.segments) == 1
    seg = result.segments[0]
    assert (seg.id, seg.start, seg.end, seg.text) == (0, 0.0, 2.0, result.text)
    only_segments = client.audio.transcriptions.create(
        model="fake-stt", file=wav_file(1.0), response_format="verbose_json"
    )
    assert only_segments.words is None and only_segments.segments


def test_transcription_streaming_events(client: openai.OpenAI) -> None:
    stream = client.audio.transcriptions.create(model="fake-stt", file=wav_file(70.0, gap_at=(20.0, 45.0)), stream=True)
    events = list(stream)
    deltas = [e for e in events if e.type == "transcript.text.delta"]
    done = [e for e in events if e.type == "transcript.text.done"]
    assert len(deltas) == 3 and len(done) == 1 and events[-1] is done[0]
    assert "".join(d.delta for d in deltas) == done[0].text
    assert len(done[0].text.split()) > 100


def test_translation(client: openai.OpenAI) -> None:
    result = client.audio.translations.create(model="fake-stt", file=wav_file(2.0))
    assert result.text == "the quick brown fox jumps"


def test_transcription_errors_use_the_openai_envelope(client: openai.OpenAI, live: LiveServer) -> None:
    with pytest.raises(openai.NotFoundError) as nf:
        client.audio.transcriptions.create(model="no-such-model", file=wav_file(1.0))
    assert nf.value.status_code == 404
    assert nf.value.body["type"] == "invalid_request_error" and nf.value.body["code"] == "model_not_found"  # type: ignore[index]
    with pytest.raises(openai.BadRequestError) as bad:
        client.audio.transcriptions.create(model="fake-stt", file=("x.wav", b"garbage-not-audio" * 20))
    assert bad.value.body["code"] == "audio_decode_error"  # type: ignore[index]
    with pytest.raises(openai.BadRequestError) as fmt:
        client.audio.transcriptions.create(model="fake-stt", file=wav_file(1.0), response_format="yaml")  # type: ignore[arg-type]
    assert fmt.value.body["param"] == "response_format"  # type: ignore[index]
    with pytest.raises(openai.BadRequestError):
        client.audio.transcriptions.create(model="fake-tts", file=wav_file(1.0))
    raw = httpx.post(live.url + "/v1/audio/transcriptions", data={"model": "fake-stt"})
    assert raw.status_code == 400
    body = raw.json()
    assert set(body["error"]) == {"message", "type", "param", "code"} and body["error"]["param"] == "file"
    assert body["detail"] == body["error"]["message"]


def test_speech_default_mp3_and_wav(client: openai.OpenAI) -> None:
    response = client.audio.speech.create(model="tts-1", voice="alloy", input="Hello there. This is a test.")
    assert response.response.headers["content-type"] == "audio/mpeg"
    assert response.response.headers["x-request-id"].startswith("r_")
    samples, rate = decode_audio(response.content, filename="a.mp3")
    assert rate == 24000 and len(samples) / rate == pytest.approx(len("Hello there. This is a test.") * 0.06, rel=0.15)
    wav = client.audio.speech.create(model="fake-tts", voice="beta", input="Hello there.", response_format="wav")
    assert wav.content[:4] == b"RIFF"
    samples, rate = decode_audio(wav.content, filename="a.wav")
    assert rate == 24000 and len(samples) == pytest.approx(0.72 * 24000, rel=0.02)


@pytest.mark.parametrize(
    ("fmt", "media"), [("flac", "audio/flac"), ("opus", "audio/ogg; codecs=opus"), ("pcm", "audio/pcm; rate=24000")]
)
def test_speech_formats(client: openai.OpenAI, fmt: str, media: str) -> None:
    response = client.audio.speech.create(model="fake-tts", voice="alpha", input="Hello there.", response_format=fmt)  # type: ignore[arg-type]
    assert response.response.headers["content-type"] == media
    if fmt == "pcm":
        assert len(response.content) == pytest.approx(0.72 * 24000 * 2, rel=0.02)
    else:
        samples, rate = decode_audio(response.content, filename=f"a.{fmt}")
        assert len(samples) / rate == pytest.approx(0.72, rel=0.1)


def test_speech_streams_audio_before_synthesis_finishes(client: openai.OpenAI, live: LiveServer) -> None:
    import time

    text = "One sentence here. " * 8
    # load the voice and the text pipeline first: the first request's set-up (0.2 s here) is not what is measured
    client.audio.speech.create(model="fake-tts", voice="alpha", input="Warm up.", response_format="pcm")
    t0 = time.perf_counter()
    first_at = None
    total = 0
    with client.audio.speech.with_streaming_response.create(
        model="fake-tts", voice="alpha", input=text, response_format="pcm"
    ) as response:
        for chunk in response.iter_bytes():
            if first_at is None and chunk:
                first_at = time.perf_counter() - t0
            total += len(chunk)
    elapsed = time.perf_counter() - t0
    assert first_at is not None and first_at < elapsed / 3  # the first audio arrives long before the last
    assert total == pytest.approx(len(text.strip()) * 0.06 * 24000 * 2, rel=0.1)
    trace = httpx.get(live.url + "/api/v1/requests", params={"kind": "tts", "limit": 1}).json()["requests"][0]
    assert trace["status"] == "ok" and trace["transport"] == "http"
    assert trace["ttfa_ms"] is not None and trace["ttfa_ms"] < trace["total_ms"] / 3
    assert [p["name"] for p in trace["phases"]] == [
        "normalize",
        "queue",
        "synthesize_first",
        "synthesize_rest",
        "encode",
    ]


def test_speech_sse_stream_format(live: LiveServer) -> None:
    with httpx.stream(
        "POST",
        live.url + "/v1/audio/speech",
        json={
            "model": "fake-tts",
            "voice": "alpha",
            "input": "Hello there.",
            "response_format": "pcm",
            "stream_format": "sse",
        },
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        import json

        events = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
    assert [e["type"] for e in events][-1] == "speech.audio.done"
    audio = b"".join(base64.b64decode(e["audio"]) for e in events if e["type"] == "speech.audio.delta")
    assert len(audio) == pytest.approx(0.72 * 24000 * 2, rel=0.02)
    assert set(events[-1]["usage"]) == {"input_tokens", "output_tokens", "total_tokens"}


def test_speech_telephony_and_sample_rate_extensions(live: LiveServer) -> None:
    r = httpx.post(
        live.url + "/v1/audio/speech",
        json={"model": "fake-tts", "voice": "alpha", "input": "Hello there.", "response_format": "mulaw"},
    )
    assert r.status_code == 200 and r.headers["content-type"] == "audio/basic" and r.headers["x-sample-rate"] == "8000"
    assert len(r.content) == pytest.approx(0.72 * 8000, rel=0.02)
    decoded = mulaw_decode(r.content)
    assert np.abs(decoded).max() > 0.1
    r = httpx.post(
        live.url + "/v1/audio/speech", json={"input": "Hello there.", "response_format": "wav", "sample_rate": 16000}
    )
    _samples, rate = decode_audio(r.content, filename="a.wav")
    assert rate == 16000


def test_speech_errors(client: openai.OpenAI, live: LiveServer) -> None:
    with pytest.raises(openai.BadRequestError) as bad_voice:
        client.audio.speech.create(model="fake-tts", voice="nobody", input="Hello.")
    assert bad_voice.value.body["param"] == "voice"  # type: ignore[index]
    with pytest.raises(openai.NotFoundError):
        client.audio.speech.create(model="missing", voice="alloy", input="Hello.")
    with pytest.raises(openai.BadRequestError):
        client.audio.speech.create(model="fake-tts", voice="alpha", input="Hello.", response_format="wma")  # type: ignore[arg-type]
    with pytest.raises(openai.BadRequestError):
        client.audio.speech.create(model="fake-tts", voice="alpha", input="  ")
    r = httpx.post(live.url + "/v1/audio/speech", json={"model": "fake-tts"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "input"
    errors = [t for t in httpx.get(live.url + "/api/v1/requests").json()["requests"] if t["status"] == "error"]
    assert len(errors) >= 2 and all(e["error"] for e in errors)


def test_upload_formats_and_raw_telephony(live: LiveServer) -> None:
    from bolo.audio import encode_audio, mulaw_encode
    from conftest import speechlike

    audio = speechlike(2.0, 8000)
    for name, payload, ctype in [
        ("call.ulaw", mulaw_encode(audio), "audio/basic"),
        ("a.flac", encode_audio(audio, 8000, "flac"), "audio/flac"),
        ("a.mp3", encode_audio(audio, 8000, "mp3"), "audio/mpeg"),
        ("a.ogg", encode_audio(audio, 8000, "opus"), "audio/ogg"),
    ]:
        r = httpx.post(
            live.url + "/v1/audio/transcriptions",
            files={"file": (name, io.BytesIO(payload), ctype)},
            data={"model": "fake-stt"},
        )
        assert r.status_code == 200, (name, r.text)
        assert len(r.json()["text"].split()) in (4, 5), name
    trace = httpx.get(live.url + "/api/v1/requests", params={"limit": 1}).json()["requests"][0]
    assert trace["audio_seconds"] == pytest.approx(2.0, abs=0.1)


def test_auth_when_api_key_is_set(settings) -> None:  # type: ignore[no-untyped-def]
    settings.api_key = "sekrit"
    srv = LiveServer(settings)
    srv.start()
    try:
        assert httpx.get(srv.url + "/healthz").status_code == 200
        assert httpx.get(srv.url + "/metrics").status_code == 200
        r = httpx.get(srv.url + "/v1/models")
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        assert r.json()["error"]["code"] == "invalid_api_key"
        assert httpx.get(srv.url + "/api/v1/info").status_code == 401
        assert httpx.get(srv.url + "/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert httpx.get(srv.url + "/v1/models", headers={"Authorization": "Bearer sekrit"}).status_code == 200
        assert httpx.get(srv.url + "/api/v1/info", headers={"X-API-Key": "sekrit"}).json()["auth_required"] is True
        assert httpx.get(srv.url + "/api/v1/stats", params={"api_key": "sekrit"}).status_code == 200
        ok = openai.OpenAI(base_url=srv.url + "/v1", api_key="sekrit", max_retries=0)
        assert ok.audio.transcriptions.create(model="fake-stt", file=io.BytesIO(wav_bytes(1.0))).text
        with pytest.raises(openai.AuthenticationError):
            openai.OpenAI(base_url=srv.url + "/v1", api_key="nope", max_retries=0).models.list()
    finally:
        srv.stop()


def test_disconnected_client_does_not_occupy_the_engine(live: LiveServer) -> None:
    """A queued transcription whose client hung up is dropped before it reaches the engine."""
    import socket
    import time

    from conftest import wav_bytes

    httpx.post(live.url + "/api/v1/models/fake-stt/load")
    model = live.core.manager.loaded["fake-stt"].model
    model.base_delay_s = 0.6  # type: ignore[attr-defined]
    model.max_batch = 1  # type: ignore[attr-defined]
    live.core.manager.loaded["fake-stt"].queue._max_batch = {"batch": 1}
    body = wav_bytes(1.0)
    boundary = "xxBOUNDARYxx"
    payload = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\nfake-stt\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + body
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = (
        f"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: x\r\nContent-Type: multipart/form-data; boundary={boundary}\r\n"
        f"Content-Length: {len(payload)}\r\n\r\n"
    ).encode() + payload

    def fire() -> socket.socket:
        s = socket.create_connection(("127.0.0.1", live.port))
        s.sendall(request)
        return s

    first = fire()  # occupies the single worker for 0.6 s
    time.sleep(0.15)
    doomed = fire()  # queued behind it
    time.sleep(0.15)
    doomed.close()  # hangs up while still queued
    assert b"200 OK" in first.recv(65536)
    first.close()
    time.sleep(0.5)
    assert model.calls == [1]  # type: ignore[attr-defined]  only the first request was ever decoded
    statuses = sorted(t["status"] for t in httpx.get(live.url + "/api/v1/requests").json()["requests"])
    assert statuses == ["cancelled", "ok"]
