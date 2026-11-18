"""Hotword spelling (bolo.hotwords): a transcript's near-misses of the request's hotwords written the hotwords' way,
on any engine, for files, live streams and Realtime sessions."""

from __future__ import annotations

import io
import json

import numpy as np
import openai
import pytest
import websockets

from bolo.engines.base import Segment, Word
from bolo.hotwords import Spelling, fold, matches, phrases, spell_text, spell_words
from conftest import LiveServer, speechlike, wav_bytes
from test_ws import collect, pcm16, stream_audio


def test_the_laptops_example() -> None:
    """whispercpp-small, given hotwords "Kshitij Wadhwani Zentriva", wrote "Kshitij Waadhwaani at Zentriva Labs"."""
    spelling = Spelling.of("Kshitij Wadhwani Zentriva")
    assert spelling is not None
    text, _, changes = spelling.text("Kshitij Waadhwaani at Zentriva Labs")
    assert text == "Kshitij Wadhwani at Zentriva Labs"
    assert changes == [{"from": "Waadhwaani", "to": "Wadhwani"}]
    assert phrases("Kshitij Wadhwani Zentriva") == [
        "Kshitij Wadhwani Zentriva", "Kshitij Wadhwani", "Wadhwani Zentriva", "Kshitij", "Wadhwani", "Zentriva"
    ]  # fmt: skip
    assert fold("Waadhwaani") == fold("Wadhwani") == fold("Vadhvani")


@pytest.mark.parametrize(
    "heard", ["wanted", "Wadhwa", "Washington", "windows", "Want", "wonder", "sentry", "central", "kitchen", "shitty"]
)
def test_real_different_words_stay(heard: str) -> None:
    assert not any(matches(heard, h) for h in phrases("Kshitij Wadhwani, Zentriva"))
    spelling = Spelling.of("Kshitij Wadhwani, Zentriva")
    assert spelling is not None
    sentence = f"We {heard} the report on time."
    assert spelling.text(sentence)[0] == sentence


def test_punctuation_capitals_runs_and_timestamps() -> None:
    spelling = Spelling.of("Wadhwani; Zentriva Labs")
    assert spelling is not None
    text, _, changes = spelling.text('"Waadhwaani," said ZENTRIVAA LAABS.')
    assert text == '"Wadhwani," said ZENTRIVA LABS.'
    assert [c["to"] for c in changes] == ["Wadhwani", "ZENTRIVA LABS"]
    words = [Word(" Mr", 0.0, 0.3, 0.9), Word(" Waadhwaani.", 0.3, 1.0, 0.4)]
    _, new, _ = spelling.text("Mr Waadhwaani.", words)
    assert new == [Word(" Mr", 0.0, 0.3, 0.9), Word(" Wadhwani.", 0.3, 1.0, 0.4)]
    # a two-word hotword heard as two words: each keeps its time
    _, new, _ = spelling.text("at Zentriva Laabs", [Word(" at", 0, 1), Word(" Zentriva", 1, 2), Word(" Laabs", 2, 3)])
    assert [(w.w, w.t0, w.t1) for w in new] == [(" at", 0, 1), (" Zentriva", 1, 2), (" Labs", 2, 3)]
    # already right: nothing to report
    assert spelling.text("Wadhwani at Zentriva Labs")[2] == []
    segment, changed = spelling.segment(Segment(4.0, 6.0, "Waadhwaani here"))
    assert segment.text == "Wadhwani here" and changed == [{"from": "Waadhwaani", "to": "Wadhwani", "start": 4.0}]
    assert Spelling.of("Wadhwani", enabled=False) is None and Spelling.of("  ,  ") is None


def test_file_transcription_spells_hotwords_and_says_so(live: LiveServer) -> None:
    """The fake recogniser says "... keep time for travellers on the old road"; hotwords "Travelers" (the
    American spelling) is what the transcript should say, and verbose_json lists the change."""
    client = openai.OpenAI(base_url=live.url + "/v1", api_key="x", max_retries=0)
    audio = io.BytesIO(wav_bytes(11.0))
    r = client.audio.transcriptions.create(
        model="fake-stt", file=audio, response_format="verbose_json", timestamp_granularities=["word"],
        extra_body={"hotwords": "Travelers"},
    )  # fmt: skip
    doc = r.model_dump()
    assert "for Travelers" in doc["text"] and "travellers" not in doc["text"]
    assert doc["hotword_spelling"] == [{"from": "travellers", "to": "Travelers", "start": 0.0}]
    assert any(w["word"].strip() == "Travelers" for w in doc["words"])
    off = client.audio.transcriptions.create(
        model="fake-stt", file=io.BytesIO(wav_bytes(11.0)), response_format="verbose_json",
        extra_body={"hotwords": "Travelers", "hotword_spelling": "false"},
    ).model_dump()  # fmt: skip
    assert "travellers" in off["text"] and "hotword_spelling" not in off
    plain = client.audio.transcriptions.create(
        model="fake-stt", file=io.BytesIO(wav_bytes(11.0)), response_format="verbose_json"
    ).model_dump()
    assert "hotword_spelling" not in plain  # no hotwords: nothing to report


async def test_live_stream_finals_spell_hotwords(live: LiveServer) -> None:
    url = live.ws_url + "/api/v1/stt/stream?model=fake-stt-streaming&hotwords=Kwik,Brawn"
    async with websockets.connect(url) as ws:
        json.loads(await ws.recv())
        audio = np.concatenate([speechlike(2.0), np.zeros(16000, dtype=np.float32)])
        await stream_audio(ws, pcm16(audio), 3200)
        await ws.send(json.dumps({"type": "stop"}))
        messages = await collect(ws)
    (final,) = [m for m in messages if m["type"] == "final"]
    assert final["text"].startswith("the quick Brawn fox")  # "brown" is near "Brawn"; "quick" is not near "Kwik"
    assert final["hotword_spelling"] == [{"from": "brown", "to": "Brawn"}]
    assert [w["w"] for w in final["words"]][:3] == ["the", "quick", "Brawn"]


def test_two_names_heard_as_one_word_are_split() -> None:
    from bolo.engines.base import Word

    hotwords = phrases("Kshitij Wadhwani Zentriva")
    assert "Kshitij Wadhwani" in hotwords and "Wadhwani Zentriva" in hotwords
    # whisper-tiny-mlx and parakeet on an M4 Pro: the first and last names run together
    text, changes = spell_text("a call with Kshetjwadwani from Zentrivar on Monday.", hotwords)
    assert text == "a call with Kshitij Wadhwani from Zentriva on Monday."
    assert changes == [{"from": "Kshetjwadwani", "to": "Kshitij Wadhwani"}, {"from": "Zentrivar", "to": "Zentriva"}]
    words = [Word(" with", 0.0, 0.2, 0.9), Word(" Kshetjwadwani", 0.2, 1.0, 0.6), Word(" from", 1.0, 1.2, 0.9)]
    out = spell_words(words, hotwords)
    assert [w.w for w in out] == [" with", " Kshitij", " Wadhwani", " from"]
    assert (out[1].t0, out[1].t1, out[2].t0, out[2].t1) == pytest.approx((0.2, 0.6, 0.6, 1.0))
    # a run of one name and an unrelated word stays
    assert spell_text("He wanted the water at noon.", hotwords)[1] == []
