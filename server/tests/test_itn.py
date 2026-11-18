"""Inverse text normalisation after recognition (``itn``): parsing, word timings, choosing the model, and the HTTP
option with the fake recogniser and the fake itn model; the real grammars when wetext is installed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from bolo import itn
from bolo.engines.base import Word
from bolo.errors import InvalidRequest
from conftest import LiveServer, wav_bytes


@pytest.fixture
def words_with_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    from bolo.engines import fake

    monkeypatch.setattr(fake, "VOCAB", ["we", "paid", "twenty", "seven", "dollars", "today"])


def _post(live: LiveServer, **fields: Any) -> httpx.Response:
    data = {"model": "fake-stt", "response_format": "verbose_json", **{k: str(v) for k, v in fields.items()}}
    return httpx.post(
        live.url + "/v1/audio/transcriptions",
        data=data,
        files={"file": ("a.wav", wav_bytes(2.4), "audio/wav")},
        timeout=30,
    )


def test_parse() -> None:
    assert itn.parse_itn(None) is None and itn.parse_itn("") is None
    assert itn.parse_itn(True) == itn.ItnOptions() and itn.parse_itn("on") == itn.ItnOptions()
    assert itn.parse_itn("OFF") == itn.ItnOptions(False)
    assert itn.parse_itn("itn-zh-number") == itn.ItnOptions(True, "itn-zh-number")
    with pytest.raises(InvalidRequest):
        itn.parse_itn(3)


def test_words_follow_the_written_form() -> None:
    words = [Word("i", 0.0, 0.2), Word("paid", 0.2, 0.5), Word("twenty", 0.5, 0.9), Word("five", 0.9, 1.2),
             Word("dollars", 1.2, 1.7), Word("today", 1.7, 2.0)]  # fmt: skip
    out = itn.rewrite_words(words, "i paid $25 today")
    assert [w.w for w in out] == ["i", "paid", "$25", "today"]
    assert (out[2].t0, out[2].t1) == (0.5, 1.7)  # the time of the words it was said in
    assert (out[3].t0, out[3].t1) == (1.7, 2.0)
    two = itn.rewrite_words(words[2:4], "20 5")
    assert [(w.w, w.t0, w.t1) for w in two] == [("20", 0.5, 0.85), ("5", 0.85, 1.2)]
    assert itn.rewrite_words([], "x") == []


def test_header() -> None:
    assert itn.header({"status": "ok", "model": "itn-en-wetext", "language": "en"}) == "itn-en-wetext"
    skipped = {"status": "skipped", "model": "itn-en-wetext", "language": "ta"}
    assert itn.header(skipped) == "skipped; model=itn-en-wetext; language=ta"
    assert itn.header(None) is None


@pytest.mark.usefixtures("words_with_numbers")
def test_numbers_written_as_figures(live: LiveServer) -> None:
    plain = _post(live, language="en", timestamp_granularities="word").json()
    assert plain["text"] == "we paid twenty seven dollars today"
    assert "x-bolo-itn" not in _post(live, language="en").headers  # off by default
    r = _post(live, itn="true", language="en", timestamp_granularities="word")
    assert r.status_code == 200, r.text
    assert r.headers["x-bolo-itn"] == "fake-itn"
    body = r.json()
    assert body["text"] == "we paid $27 today"
    assert [w["word"] for w in body["words"]] == body["text"].split()
    said = plain["words"]  # "twenty seven dollars" was said from the start of "twenty" to the end of "dollars"
    figure = body["words"][2]
    assert (figure["start"], figure["end"]) == (said[2]["start"], said[4]["end"])
    assert _post(live, itn="fake-itn", language="en").headers["x-bolo-itn"] == "fake-itn"  # named
    trace = httpx.get(live.url + "/api/v1/requests/" + r.headers["x-request-id"]).json()
    assert "itn" in [p["name"] for p in trace["phases"]]


@pytest.mark.usefixtures("words_with_numbers")
def test_setting_pin_language_and_errors(live: LiveServer) -> None:
    live.core.settings.itn = "on"
    try:
        assert _post(live, language="en").headers["x-bolo-itn"] == "fake-itn"
        assert "x-bolo-itn" not in _post(live, language="en", itn="off").headers  # the request decides
        pin = httpx.put(live.url + "/api/v1/models/fake-stt/settings", json={"itn": "off"}, timeout=10)
        assert pin.status_code == 200, pin.text
        assert "x-bolo-itn" not in _post(live, language="en").headers  # the pin before the setting
    finally:
        live.core.settings.itn = "off"
        httpx.put(live.url + "/api/v1/models/fake-stt/settings", json={"itn": None}, timeout=10)
    r = _post(live, itn="on", language="hi")  # no itn model knows Hindi here: kept as it was, and said so
    assert r.status_code == 200 and r.headers["x-bolo-itn"] == "skipped; model=fake-itn; language=hi"
    assert "twenty seven" in r.json()["text"]
    r = _post(live, itn="no-such-model")
    assert r.status_code in (400, 404) and r.json()["error"]["param"] == "itn"
    r = _post(live, itn="fake-tts")
    assert r.status_code == 400 and r.json()["error"]["param"] == "itn"
    model_entry = None
    assert _post(live, itn="on", language="en").status_code == 200
    model_entry = live.core.manager.loaded["fake-itn"].model
    model_entry.fail = True  # type: ignore[attr-defined]
    try:
        r = _post(live, itn="on", language="en")
        assert r.status_code == 200 and r.headers["x-bolo-itn"].startswith("skipped")
    finally:
        model_entry.fail = False  # type: ignore[attr-defined]


def test_wetext_grammars_from_the_model_directory(tmp_path: Path) -> None:
    """The real English grammars, taken from an installed wetext as a pulled model would hold them."""
    wetext = pytest.importorskip("wetext")
    import shutil

    from bolo.engines.text_models import WetextItn

    source = Path(wetext.__file__).parent / "fsts" / "en" / "itn"
    target = tmp_path / "fsts" / "en" / "itn"
    shutil.copytree(source, target)
    model = WetextItn(tmp_path, {"language": "en"})
    assert model.normalize("i paid twenty five dollars on march third") == "i paid $25 on march 3"


def test_english_clock_times_before_a_time_of_day() -> None:
    from bolo.engines.text_models import english_clock_times

    # wetext 0.2.1 writes "ten thirty in the morning" as "1030 in the morning" (it takes a clock time only before
    # "a m" / "p m"); zipformer's transcript of an English recording, through itn-en-wetext, on an M4 Pro
    assert (
        english_clock_times("the meeting starts at 1030 in the morning") == "the meeting starts at 10:30 in the morning"
    )
    assert english_clock_times("at 730 in the evening, 1145 at night") == "at 7:30 in the evening, 11:45 at night"
    # not a clock time: left alone
    for text in (
        "1999 in the morning",
        "it costs 1030",
        "2400 in the morning",
        "0930 in the morning",
        "1260 in the morning",
    ):
        assert english_clock_times(text) == text
