"""Forced alignment: the Viterbi aligner, POST /v1/audio/alignments, word times for recognisers without them,
word times of synthesised speech, and word confidence."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import numpy as np
import pytest

from bolo.alignment import AlignmentError, align_words, char_tokenizer, log_softmax, split_words, viterbi
from bolo.config import Settings
from bolo.engines.ctc_align import read_vocab
from bolo.engines.sherpa import tokens_to_words
from conftest import LiveServer, wav_bytes


@pytest.fixture
def settings(settings: Settings) -> Settings:
    settings.default_align = "fake-align"
    return settings


# ----------------------------------------------------------------------
# the aligner
# ----------------------------------------------------------------------
def _emissions(spoken: list[tuple[int, int]], vocab: int = 6, frames: int = 40) -> np.ndarray:
    """Frame scores where token ``t`` is heard on frames ``[a, b)`` for each ``(t, a)`` ... blank elsewhere."""
    logits = np.full((frames, vocab), -5.0, dtype=np.float32)
    logits[:, 0] = 5.0
    for token, frame in spoken:
        logits[frame, token] = 8.0
    return log_softmax(logits)


def test_viterbi_puts_each_token_where_it_is_heard() -> None:
    lp = _emissions([(2, 5), (3, 6), (4, 20), (2, 30), (2, 33)])
    path = viterbi(lp, [2, 3, 4, 2, 2], blank=0)
    tokens = {int(s) // 2: t for t, s in enumerate(path) if s % 2}
    assert tokens == {0: 5, 1: 6, 2: 20, 3: 30, 4: 33}  # a repeated token needs a blank between its copies


def test_words_get_the_frames_of_their_letters() -> None:
    vocab = {"<blank>": 0, " ": 1, "a": 2, "b": 3, "c": 4}
    tokenize = char_tokenizer(vocab)
    lp = _emissions([(2, 5), (3, 6), (1, 10), (4, 20), (2, 21)])
    words = ["Ab,", "—", "ca!"]
    out = align_words(lp, 0.02, words, [tokenize(w) for w in words], blank=0, separator=1, offset=1.0)
    assert [w.word for w in out] == words
    assert (out[0].start, out[0].end) == pytest.approx((1.10, 1.14))
    assert out[1].start == out[1].end == pytest.approx(1.14) and out[1].score is None  # nothing to hear
    assert (out[2].start, out[2].end) == pytest.approx((1.40, 1.44))
    assert out[0].score is not None and 0.9 < out[0].score <= 1.0


def test_a_text_longer_than_the_audio_is_refused() -> None:
    with pytest.raises(AlignmentError):
        viterbi(_emissions([], frames=3), [2, 3, 4, 2], blank=0)


def test_words_of_scripts_without_spaces_are_characters() -> None:
    assert split_words("Hello, world!") == ["Hello,", "world!"]
    assert split_words("今日は 天気") == ["今", "日", "は", "天", "気"]
    assert split_words("谢谢。OK") == ["谢", "谢。", "OK"]


def test_tokenizer_drops_what_the_vocabulary_lacks() -> None:
    tokenize = char_tokenizer({"a": 1, "e": 2, "n": 3})
    assert tokenize("Àne!") == [1, 3, 2]  # NFKC, lower case; the accent comes off when nothing else is left
    assert tokenize("?") == []


def test_vocabularies_of_both_forms(tmp_path: Any) -> None:
    (tmp_path / "tokens.txt").write_text("<s> 0\n  1\na 2\n", encoding="utf-8")
    assert read_vocab(tmp_path / "tokens.txt") == {"<s>": 0, " ": 1, "a": 2}
    (tmp_path / "vocab.json").write_text(json.dumps({"<blank>": 0, "a": 4}), encoding="utf-8")
    assert read_vocab(tmp_path / "vocab.json") == {"<blank>": 0, "a": 4}


def test_token_log_probs_give_word_confidence() -> None:
    words = tokens_to_words(["▁he", "llo", "▁you"], [0.0, 0.2, 0.5], 1.0, [np.log(0.9), np.log(0.7), np.log(0.5)])
    assert [w.w for w in words] == ["hello", "you"]
    assert words[0].p == pytest.approx(0.8) and words[1].p == pytest.approx(0.5)
    assert tokens_to_words(["▁a"], [0.0], 1.0)[0].p is None


# ----------------------------------------------------------------------
# over HTTP
# ----------------------------------------------------------------------
def _align(live: LiveServer, text: str, **fields: str) -> httpx.Response:
    return httpx.post(
        live.url + "/v1/audio/alignments",
        data={"text": text, **fields},
        files={"file": ("a.wav", wav_bytes(2.0), "audio/wav")},
        timeout=30,
    )


def test_alignment_endpoint(live: LiveServer) -> None:
    r = _align(live, "hello there world")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "fake-align" and r.headers["x-bolo-model"] == "fake-align"
    assert [w["word"] for w in body["words"]] == ["hello", "there", "world"]
    starts = [w["start"] for w in body["words"]]
    assert starts == sorted(starts) and body["words"][-1]["end"] <= body["duration"] + 1e-6
    assert _align(live, "  ").status_code == 400
    wrong = _align(live, "hello", model="fake-stt")
    assert wrong.status_code == 400 and "align" in wrong.json()["error"]["message"]


def _transcribe(live: LiveServer, granularities: list[str]) -> httpx.Response:
    return httpx.post(
        live.url + "/v1/audio/transcriptions",
        data={"model": "fake-stt", "response_format": "verbose_json", "timestamp_granularities[]": granularities},
        files={"file": ("a.wav", wav_bytes(2.0), "audio/wav")},
        timeout=30,
    )


def test_a_recogniser_without_word_times_gets_them_from_the_aligner(live: LiveServer) -> None:
    assert _transcribe(live, ["segment"]).status_code == 200  # loads the model
    stt = live.core.manager.loaded["fake-stt"].model
    stt.supports_word_timestamps = False  # type: ignore[misc]
    plain = _transcribe(live, ["segment"]).json()
    for granularities in (["word"], ["segment", "word"]):
        r = _transcribe(live, granularities)
        assert r.status_code == 200, r.text
        assert "x-bolo-ignored-options" not in r.headers
        assert r.headers["x-bolo-word-aligner"] == "fake-align"
        body = r.json()
        assert body["text"] == plain["text"]
        assert " ".join(w["word"] for w in body["words"]) == body["text"]
        assert all(0.0 <= w["start"] <= w["end"] <= body["duration"] for w in body["words"])
    aligner = live.core.manager.loaded["fake-align"].model
    aligner.fail = True  # type: ignore[attr-defined]
    r = _transcribe(live, ["word"])
    assert r.status_code == 200 and r.headers["x-bolo-ignored-options"] == "timestamp_granularities"
    assert "x-bolo-word-aligner" not in r.headers
    aligner.fail = False  # type: ignore[attr-defined]
    live.core.settings.align_words = False
    r = _transcribe(live, ["word"])
    assert r.headers["x-bolo-ignored-options"] == "timestamp_granularities"


def test_word_probability_in_verbose_json(live: LiveServer) -> None:
    from bolo.engines.base import Word

    assert Word("a", 0.0, 1.0, 0.91234).to_dict() == {"w": "a", "t0": 0.0, "t1": 1.0, "p": 0.9123}
    assert "p" not in Word("a", 0.0, 1.0).to_dict()
    body = _transcribe(live, ["word"]).json()
    assert body["words"] and all("probability" not in w for w in body["words"])  # the fake model gives none


def test_captioned_speech(live: LiveServer) -> None:
    text = "Hello there. How are you today?"
    r = httpx.post(
        live.url + "/v1/audio/speech/captioned",
        json={"model": "fake-tts", "input": text, "response_format": "wav"},
        timeout=30,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["aligner"] == "fake-align" and body["format"] == "wav"
    assert base64.b64decode(body["audio"])[:4] == b"RIFF"
    words = body["words"]
    assert [w["word"].strip(".?") for w in words] == ["Hello", "there", "How", "are", "you", "today"]
    starts = [w["start"] for w in words]
    assert starts == sorted(starts) and words[-1]["end"] <= body["duration"] + 1e-3
    aligner = live.core.manager.loaded["fake-align"].model
    assert aligner.calls[-2:] == ["Hello there.", "How are you today?"]  # type: ignore[attr-defined]

    events: list[dict[str, Any]] = []
    with httpx.stream(
        "POST",
        live.url + "/v1/audio/speech",
        json={"model": "fake-tts", "input": text, "stream_format": "sse", "timestamps": True},
        timeout=30,
    ) as stream:
        assert stream.status_code == 200
        for line in stream.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    kinds = [e["type"] for e in events]
    assert kinds.count("speech.words") == 2 and kinds[-1] == "speech.audio.done"
    streamed = [w for e in events if e["type"] == "speech.words" for w in e["words"]]
    assert streamed == words
    r = httpx.post(live.url + "/v1/audio/speech", json={"model": "fake-tts", "input": text, "timestamps": True})
    assert r.status_code == 400 and r.json()["error"]["param"] == "timestamps"


def test_captioned_speech_needs_an_installed_aligner(live: LiveServer) -> None:
    live.core.settings.default_align = "mms-300m-forced-aligner"
    r = httpx.post(live.url + "/v1/audio/speech/captioned", json={"model": "fake-tts", "input": "Hello."}, timeout=30)
    assert r.status_code == 404 and r.json()["error"]["param"] == "align_model"
