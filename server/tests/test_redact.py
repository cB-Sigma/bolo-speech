"""Redaction of personal information in transcripts (``redact``), with the fake recogniser (whose words include
"fox" and "dog") and the fake pii model (fox and dog are PERSON)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from bolo.engines.base import PiiSpan, Word
from bolo.errors import InvalidRequest
from bolo.redact import RedactOptions, mask, parse_entities, parse_redact
from conftest import LiveServer, wav_bytes


def _post(live: LiveServer, route: str = "transcriptions", **fields: Any) -> httpx.Response:
    data = {"model": "fake-stt", "response_format": "verbose_json", **{k: str(v) for k, v in fields.items()}}
    return httpx.post(
        f"{live.url}/v1/audio/{route}", data=data, files={"file": ("a.wav", wav_bytes(4.0), "audio/wav")}, timeout=30
    )


def test_parse() -> None:
    assert parse_redact(None) is None and parse_redact(" ") is None
    assert parse_redact(True) == RedactOptions() and parse_redact("ON") == RedactOptions()
    assert parse_redact("off") == RedactOptions(False)
    assert parse_redact("pii-gliner-multi") == RedactOptions(True, "pii-gliner-multi")
    assert parse_entities("person, phone number;EMAIL_ADDRESS") == ("PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS")
    with pytest.raises(InvalidRequest):
        parse_entities("PERSON, <script>")


def test_mask_merges_the_words_and_says_where() -> None:
    text = "call Anna Lee on 555 1234 today"
    words = [Word(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(text.split())]
    spans = [PiiSpan("PERSON", 5, 13, 0.95), PiiSpan("PHONE_NUMBER", 17, 25, 0.8)]
    out, new_words, records = mask(text, spans, words)
    assert out == "call [PERSON] on [PHONE_NUMBER] today"
    assert [w.w for w in new_words] == ["call", "[PERSON]", "on", "[PHONE_NUMBER]", "today"]
    assert (new_words[1].t0, new_words[1].t1) == (0.5, 1.4)  # the time "Anna Lee" was said
    assert records[0] == {"type": "PERSON", "start": 5, "end": 13, "score": 0.95, "time_start": 0.5, "time_end": 1.4}
    assert out[records[1]["start"] : records[1]["end"]] == "[PHONE_NUMBER]"


def test_redacted_and_audited(live: LiveServer) -> None:
    plain = _post(live).json()
    assert "fox" in plain["text"] and "x-bolo-redact" not in _post(live).headers
    r = _post(live, redact="true", timestamp_granularities="word")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "fox" not in body["text"] and "dog" not in body["text"] and "[PERSON]" in body["text"]
    assert r.headers["x-bolo-redact"] == "fake-pii; PERSON=2"
    assert [x["type"] for x in body["redactions"]] == ["PERSON", "PERSON"]
    for x in body["redactions"]:
        assert body["text"][x["start"] : x["end"]] == "[PERSON]" and x["time_end"] > x["time_start"]
    assert "fox" not in " ".join(w["word"] for w in body["words"])
    trace = httpx.get(live.url + "/api/v1/requests/" + r.headers["x-request-id"]).json()
    assert "fox" not in str(trace)  # nothing unredacted in the request log
    r = _post(live, redact="fake-pii", redact_entities="PHONE_NUMBER")
    assert r.headers["x-bolo-redact"] == "fake-pii; none" and "fox" in r.json()["text"]
    r = _post(live, redact="on", redact_entities="EMAIL_ADDRESS")
    assert r.status_code == 400 and r.json()["error"]["param"] == "redact_entities"


def test_setting_pin_errors_and_streams(live: LiveServer) -> None:
    live.core.settings.redact = "on"
    try:
        assert _post(live).headers["x-bolo-redact"] == "fake-pii; PERSON=2"
        assert "x-bolo-redact" not in _post(live, redact="off").headers
        with httpx.stream(
            "POST", live.url + "/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(4.0), "audio/wav")},
            data={"model": "fake-stt", "stream": "true"}, timeout=30,
        ) as s:  # fmt: skip
            streamed = "".join(s.iter_text())
        assert "fox" not in streamed and "[PERSON]" in streamed  # no unredacted partial segment
    finally:
        live.core.settings.redact = "off"
    r = _post(live, redact="no-such-model")
    assert r.status_code in (400, 404) and r.json()["error"]["param"] == "redact"
    assert _post(live, redact="on").status_code == 200
    model = live.core.manager.loaded["fake-pii"].model
    model.fail = True  # type: ignore[attr-defined]
    try:
        r = _post(live, redact="on")
        assert r.status_code == 503 and "fox" not in r.text  # never the unredacted text
    finally:
        model.fail = False  # type: ignore[attr-defined]
