"""Punctuation restoration after recognition (``punctuation``), with the fake recogniser and punctuation model."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from bolo.errors import InvalidRequest
from bolo.punctuation import PunctuationOptions, parse_punctuation
from conftest import LiveServer, wav_bytes


def _post(live: LiveServer, **fields: Any) -> httpx.Response:
    data = {"model": "fake-stt", "response_format": "verbose_json", **{k: str(v) for k, v in fields.items()}}
    return httpx.post(
        live.url + "/v1/audio/transcriptions",
        data=data,
        files={"file": ("a.wav", wav_bytes(3.0), "audio/wav")},
        timeout=30,
    )


def test_parse() -> None:
    assert parse_punctuation(None) is None and parse_punctuation(" ") is None
    assert parse_punctuation(True) == PunctuationOptions("on")
    assert parse_punctuation("off") == PunctuationOptions("off")
    assert parse_punctuation("AUTO") == PunctuationOptions("auto")
    assert parse_punctuation("punct-en") == PunctuationOptions("on", "punct-en")
    with pytest.raises(InvalidRequest):
        parse_punctuation(3)


def test_punctuation_on_request(live: LiveServer) -> None:
    plain = _post(live)
    assert plain.status_code == 200 and "x-bolo-punctuation" not in plain.headers  # off by default
    r = _post(live, punctuation="on", timestamp_granularities="word")
    assert r.status_code == 200, r.text
    assert r.headers["x-bolo-punctuation"] == "fake-punctuation"
    body = r.json()
    assert body["text"].endswith(".") and body["text"][0].isupper()
    segmented = _post(live, punctuation="on").json()
    assert " ".join(seg["text"] for seg in segmented["segments"]) == segmented["text"] == body["text"]
    assert len(body["text"].split()) == len(plain.json()["text"].split())  # the same words
    words = [w["word"] for w in body["words"]]
    assert words == body["text"].split()  # each word takes its punctuation, timings kept
    trace = httpx.get(live.url + "/api/v1/requests/" + r.headers["x-request-id"]).json()
    assert "punctuation" in [p["name"] for p in trace["phases"]]


def test_auto_leaves_recognisers_that_punctuate_alone(live: LiveServer) -> None:
    assert "x-bolo-punctuation" in _post(live, punctuation="auto").headers  # fake-stt writes no punctuation
    spec = live.core.catalog.resolve("fake-stt")
    assert spec is not None
    spec.features.append("punctuation")
    try:
        assert "x-bolo-punctuation" not in _post(live, punctuation="auto").headers
        assert _post(live, punctuation="on").headers["x-bolo-punctuation"] == "fake-punctuation"
    finally:
        spec.features.remove("punctuation")


def test_setting_pin_and_language(live: LiveServer) -> None:
    live.core.settings.punctuation = "on"
    try:
        assert _post(live).headers["x-bolo-punctuation"] == "fake-punctuation"
        assert "x-bolo-punctuation" not in _post(live, punctuation="off").headers  # the request decides
        pin = httpx.put(live.url + "/api/v1/models/fake-stt/settings", json={"punctuation": "off"}, timeout=10)
        assert pin.status_code == 200, pin.text
        assert "x-bolo-punctuation" not in _post(live).headers  # the pin before the setting
    finally:
        live.core.settings.punctuation = "off"
        httpx.put(live.url + "/api/v1/models/fake-stt/settings", json={"punctuation": None}, timeout=10)
    # a language the punctuation model does not know: kept as it was, and said so
    r = _post(live, punctuation="on", language="fr")
    assert r.status_code == 200 and r.headers["x-bolo-punctuation"] == "skipped; model=fake-punctuation"
    assert not r.json()["text"].endswith(".")


def test_unknown_or_wrong_model_is_refused(live: LiveServer) -> None:
    r = _post(live, punctuation="no-such-model")
    assert r.status_code in (400, 404) and r.json()["error"]["param"] == "punctuation"
    r = _post(live, punctuation="fake-tts")
    assert r.status_code == 400 and r.json()["error"]["param"] == "punctuation"


def test_a_failing_model_keeps_the_transcript(live: LiveServer) -> None:
    assert _post(live, punctuation="on").status_code == 200  # loads it
    model = live.core.manager.loaded["fake-punctuation"].model
    model.fail = True  # type: ignore[attr-defined]
    try:
        r = _post(live, punctuation="on")
        assert r.status_code == 200 and r.headers["x-bolo-punctuation"].startswith("skipped")
    finally:
        model.fail = False  # type: ignore[attr-defined]
