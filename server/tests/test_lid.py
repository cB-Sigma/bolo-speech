"""Spoken-language identification (``POST /v1/audio/language``) and language routing (``model=auto``), with the
fake language identifier (pitch picks the language) and fake recognisers."""

from __future__ import annotations

import contextlib
import io
import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pytest

from bolo.audio.encode import encode_audio
from bolo.config import Settings, load_settings
from bolo.engines.lid_onnx import fbank60, read_labels
from bolo.lid import Route, parse_routes, route_headers
from bolo.text.iso639 import name, normalize, same_language
from conftest import LiveServer

SR = 16000
TONES = {"en": 220.0, "hi": 440.0, "ta": 660.0, "fr": 880.0}


def tone(language: str, seconds: float = 2.0, sr: int = SR) -> tuple[str, io.BytesIO, str]:
    t = np.arange(int(seconds * sr)) / sr
    x = (0.3 * np.sin(2 * np.pi * TONES[language] * t)).astype(np.float32)
    return ("clip.wav", io.BytesIO(encode_audio(x, sr, "wav")), "audio/wav")


@pytest.fixture
def lid_settings(settings: Settings) -> Settings:
    settings.default_lid = "fake-lid"
    return settings


@pytest.fixture
def lid_live(lid_settings: Settings):  # type: ignore[no-untyped-def]
    server = LiveServer(lid_settings)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@contextlib.contextmanager
def serve(settings: Settings) -> Iterator[LiveServer]:
    server = LiveServer(settings)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def transcribe(live: LiveServer, spoken: str, **data: str) -> httpx.Response:
    return httpx.post(live.url + "/v1/audio/transcriptions", files={"file": tone(spoken)}, data=data, timeout=30)


def test_codes() -> None:
    assert normalize("hin") == "hi" and normalize("hi-IN") == "hi" and normalize("iw") == "he"
    assert normalize("npi") == "ne" and normalize("ory") == "or" and normalize("cmn") == "zh"
    assert normalize("bho") == "bho" and normalize("kas") == "ks"
    assert same_language("jw", "jv") and same_language("hi", "hi-IN") and same_language("gom", "kok")
    assert not same_language("hi", "ur")
    assert name("hi") == "Hindi" and name("tam") == "Tamil" and name("mai") == "Maithili" and name("xx") == "xx"


def test_labels_and_features(tmp_path: Path) -> None:
    (tmp_path / "lang_map.json").write_text(json.dumps({"1": {"iso": "iw"}, "0": {"iso": "ab"}}))
    assert read_labels(tmp_path / "lang_map.json") == ["ab", "he"]
    (tmp_path / "config.json").write_text(json.dumps({"labels": ["en", "jw"]}))
    assert read_labels(tmp_path / "config.json") == ["en", "jv"]
    (tmp_path / "id2label.json").write_text(json.dumps({"id2label": {"0": "eng", "1": "hin", "2": "ory"}}))
    assert read_labels(tmp_path / "id2label.json") == ["en", "hi", "or"]
    (tmp_path / "labels.txt").write_text("vi: Vietnamese\nhi: Hindi\n")
    assert read_labels(tmp_path / "labels.txt") == ["vi", "hi"]
    feats = fbank60(np.zeros(SR, dtype=np.float32))
    assert feats.shape == (1, 101, 60) and feats.dtype == np.float32
    loud = fbank60((0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32))
    assert float(loud.max() - loud.min()) <= 80.0 + 1e-3  # floored 80 dB under the loudest value


def test_routes_setting() -> None:
    assert parse_routes(["ta=indicconformer-ta-community", "HI-IN = whisper-base", "bad"]) == {
        "ta": "indicconformer-ta-community",
        "hi": "whisper-base",
    }
    assert load_settings(env={"BOLO_LANGUAGE_ROUTES": "ta=a,hi=b"}).language_routes == ["ta=a", "hi=b"]
    with pytest.raises(ValueError, match="language_routes"):
        load_settings(env={"BOLO_LANGUAGE_ROUTES": "ta"})
    with pytest.raises(ValueError, match="lid_min_probability"):
        load_settings(env={"BOLO_LID_MIN_PROBABILITY": "2"})


def test_route_headers() -> None:
    r = Route("auto", "m", "recommended", lid="fake-lid", language="ta", probability=0.9123, steer="ta")
    assert route_headers(r.to_dict()) == {
        "x-bolo-route": "auto -> m; reason=recommended; language=ta; probability=0.912; lid=fake-lid",
        "x-bolo-language": "ta; source=detected; probability=0.912; applied=true",
    }
    assert route_headers({"nonsense": 1}) == {}


def test_identify_endpoint(lid_live: LiveServer) -> None:
    r = httpx.post(lid_live.url + "/v1/audio/language", files={"file": tone("hi")}, data={"top_k": "2"}, timeout=30)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "fake-lid" and body["language"] == "hi" and body["name"] == "Hindi"
    assert body["probability"] == pytest.approx(0.9)
    assert body["languages"][0]["language"] == "hi" and len(body["languages"]) == 2
    assert body["duration"] == pytest.approx(2.0, abs=0.01)
    rec = body["recogniser"]
    assert (rec["model"], rec["reason"], rec["language"], rec["detail"]) == ("fake-stt", "recommended", "hi", None)
    # (rec["recommended"] names a real Hindi recogniser once the background advice is ready: timing-dependent)
    assert r.headers["x-bolo-model"] == "fake-lid"
    trace = httpx.get(lid_live.url + "/api/v1/requests/" + r.headers["x-request-id"]).json()
    assert trace["kind"] == "lid" and trace["model"] == "fake-lid" and trace["status"] == "ok"
    # Tamil: the one-language recogniser
    r = httpx.post(lid_live.url + "/v1/audio/language", files={"file": tone("ta")}, data={"model": "fake-lid"})
    assert r.json()["language"] == "ta" and r.json()["recogniser"]["model"] == "fake-stt-ta"
    # errors: not a lid model, bad top_k, no file
    bad = httpx.post(lid_live.url + "/v1/audio/language", files={"file": tone("ta")}, data={"model": "fake-stt"})
    assert bad.status_code == 400 and bad.json()["error"]["param"] == "model"
    bad = httpx.post(lid_live.url + "/v1/audio/language", files={"file": tone("ta")}, data={"top_k": "0"})
    assert bad.status_code == 400 and bad.json()["error"]["param"] == "top_k"
    assert httpx.post(lid_live.url + "/v1/audio/language", data={"model": "fake-lid"}).status_code == 400


def test_info_capability(lid_live: LiveServer) -> None:
    info = httpx.get(lid_live.url + "/api/v1/info").json()
    lid = info["capabilities"]["lid"]
    assert lid["default_model"] == "fake-lid" and lid["available"] and lid["endpoint"] == "/v1/audio/language"
    assert lid["routing"]["enabled"] and lid["routing"]["model_names"] == ["auto"]


def test_routing_picks_a_recogniser_for_the_language(lid_live: LiveServer) -> None:
    r = transcribe(lid_live, "ta", model="auto", response_format="verbose_json")
    assert r.status_code == 200, r.text
    assert r.headers["x-bolo-model"] == "fake-stt-ta"
    assert r.headers["x-bolo-route"].startswith("auto -> fake-stt-ta; reason=recommended; language=ta;")
    assert r.headers["x-bolo-language"] == "ta; source=detected; probability=0.9; applied=true"
    trace = httpx.get(lid_live.url + "/api/v1/requests/" + r.headers["x-request-id"]).json()
    assert trace["model"] == "fake-stt-ta" and trace["route"]["reason"] == "recommended"
    assert "lid" in [p["name"] for p in trace["phases"]]
    # Hindi: the (installed, two-language) default recogniser speaks it
    r = transcribe(lid_live, "hi", model="auto")
    assert r.headers["x-bolo-model"] == "fake-stt" and "reason=recommended; language=hi" in r.headers["x-bolo-route"]
    # French: no installed recogniser speaks it, the default answers, not told the language
    r = transcribe(lid_live, "fr", model="auto")
    assert r.status_code == 200 and r.headers["x-bolo-model"] == "fake-stt"
    assert "reason=default; language=fr" in r.headers["x-bolo-route"]
    assert r.headers["x-bolo-language"].endswith("applied=false")


def test_routing_never_overrides_the_request(lid_live: LiveServer) -> None:
    r = transcribe(lid_live, "ta", model="fake-stt")  # a model named: kept
    assert r.headers["x-bolo-model"] == "fake-stt" and "x-bolo-route" not in r.headers
    r = transcribe(lid_live, "ta", model="auto", language="hi")  # a language named: no routing
    assert r.headers["x-bolo-model"] == "fake-stt" and "x-bolo-route" not in r.headers
    r = transcribe(lid_live, "ta")  # no model: the default, as before
    assert r.headers["x-bolo-model"] == "fake-stt" and "x-bolo-route" not in r.headers
    r = httpx.post(lid_live.url + "/v1/audio/translations", files={"file": tone("ta")}, data={"model": "auto"})
    assert r.status_code == 200 and "x-bolo-route" not in r.headers  # translations are not routed


def test_routing_stream_carries_the_headers(lid_live: LiveServer) -> None:
    with httpx.stream(
        "POST", lid_live.url + "/v1/audio/transcriptions", files={"file": tone("ta")},
        data={"model": "auto", "stream": "true"}, timeout=30,
    ) as r:  # fmt: skip
        assert r.status_code == 200
        assert r.headers["x-bolo-route"].startswith("auto -> fake-stt-ta;")
        assert "transcript.text.done" in "".join(r.iter_text())


def test_routing_table_and_threshold(lid_settings: Settings) -> None:
    lid_settings.language_routes = ["hi=fake-stt-ta", "ta=no-such-model"]
    with serve(lid_settings) as live:
        r = transcribe(live, "hi", model="auto")
        assert r.headers["x-bolo-model"] == "fake-stt-ta" and "reason=route" in r.headers["x-bolo-route"]
        assert r.headers["x-bolo-language"].endswith("applied=false")  # the routed model does not speak Hindi
        r = transcribe(live, "ta", model="auto")  # a route that cannot be used: the best installed recogniser
        assert r.headers["x-bolo-model"] == "fake-stt-ta" and "reason=recommended" in r.headers["x-bolo-route"]
    lid_settings.language_routes = []
    lid_settings.lid_min_probability = 0.95
    with serve(lid_settings) as live:
        r = transcribe(live, "ta", model="auto")
        assert r.headers["x-bolo-model"] == "fake-stt" and "reason=unsure" in r.headers["x-bolo-route"]


def test_routing_without_a_language_identifier(lid_settings: Settings) -> None:
    lid_settings.default_lid = "no-such-lid"
    with serve(lid_settings) as live:
        r = transcribe(live, "ta", model="auto")
        assert r.status_code == 200 and r.headers["x-bolo-model"] == "fake-stt"
        assert r.headers["x-bolo-route"].startswith("auto -> fake-stt; reason=lid-failed")
    lid_settings.default_lid = "fake-lid"
    lid_settings.language_routing = False
    with serve(lid_settings) as live:
        r = transcribe(live, "ta", model="auto")
        assert r.headers["x-bolo-model"] == "fake-stt" and "x-bolo-route" not in r.headers
