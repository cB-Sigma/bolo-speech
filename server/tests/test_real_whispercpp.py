"""whisper.cpp (pywhispercpp) with a real model on CPU: the decoding options. Opt-in: ``BOLO_TEST_REAL=1 pytest -m real``.

Pulls the catalog's ``whispercpp-tiny`` (``ggml-tiny.bin``, 75 MB, pinned by sha256) into ``BOLO_DATA_DIR``.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf

from bolo import samples
from bolo.config import Settings
from bolo.engines.whispercpp import _PER_REQUEST
from conftest import LiveServer

pytestmark = [pytest.mark.real, pytest.mark.timeout(900)]
pytest.importorskip("pywhispercpp")

MODEL = "whispercpp-tiny"


def _data_dir() -> Path:
    return Path(os.environ.get("BOLO_DATA_DIR", Path.home() / ".cache" / "bolo"))


@pytest.fixture(scope="module")
def live() -> Iterator[LiveServer]:
    srv = LiveServer(Settings(data_dir=_data_dir(), auto_pull=True, device="cpu"))
    srv.start()
    try:
        if not srv.core.manager.is_installed(srv.core.catalog.resolve(MODEL)):  # type: ignore[arg-type]
            job = srv.call(srv.core.pull_and_wait(MODEL))
            assert job.state == "done", job  # type: ignore[attr-defined]
        yield srv
    finally:
        srv.stop()


def _padded_clip() -> bytes:
    """The bundled en-short sample with 10 s of near-silence before it and 15 s after (28 s)."""
    audio, rate, _ = samples.load_audio("en-short")
    rng = np.random.default_rng(0)
    quiet = lambda s: (rng.standard_normal(int(s * rate)) * 1e-4).astype(np.float32)  # noqa: E731
    buf = io.BytesIO()
    sf.write(buf, np.concatenate([quiet(10.0), audio, quiet(15.0)]), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


CLIP = _padded_clip()


def _post(live: LiveServer, **fields: str) -> httpx.Response:
    r = httpx.post(
        live.url + "/v1/audio/transcriptions",
        data={"model": MODEL, "response_format": "verbose_json", **fields},
        files={"file": ("a.wav", CLIP, "audio/wav")},
        timeout=600,
    )
    assert r.status_code == 200, r.text
    return r


def _params(live: LiveServer) -> dict[str, Any]:
    """The whisper.cpp parameters of the loaded model as the last request left them."""
    return live.core.manager.loaded[MODEL].model.model.get_params()  # type: ignore[attr-defined,no-any-return]


def test_whispercpp_options_reach_whisper_cpp_and_do_not_carry_over(live: LiveServer) -> None:
    base = _post(live, vad_filter="false").json()
    assert "order will arrive" in base["text"].lower()
    defaults = _params(live)
    assert defaults["no_context"] is True and defaults["language"] in (
        "auto",
        "",
    )  # detect (the binding reads "auto" back as "")
    r = _post(
        live, vad_filter="false", condition_on_previous_text="true", log_prob_threshold="-0.1",
        no_speech_threshold="0.01", temperature_increment_on_fallback="0.4", language="en",
    )  # fmt: skip
    # thresholds this strict may drop text; what is checked is that whisper.cpp was given them
    assert "x-bolo-ignored-options" not in r.headers
    p = _params(live)
    assert p["no_context"] is False
    assert (p["logprob_thold"], p["no_speech_thold"], p["temperature_inc"]) == pytest.approx((-0.1, 0.01, 0.4))
    # a prompt and a language given to one request are gone for the next one (pywhispercpp keeps parameters)
    _post(live, vad_filter="false", prompt="Order status call.", language="de")
    assert _params(live)["initial_prompt"] == "Order status call." and _params(live)["language"] == "de"
    again = _post(live, vad_filter="false").json()
    after = _params(live)
    assert {k: after[k] for k in _PER_REQUEST} == {k: defaults[k] for k in _PER_REQUEST}
    assert again["text"] == base["text"]


def test_whispercpp_bolo_side_options_and_ignored_ones(live: LiveServer) -> None:
    whole = _post(live, vad_filter="false").json()
    assert whole["segments"][0]["start"] == 0.0  # the model is given the silence too
    for fields in ({"vad_filter": "true"}, {"vad_parameters": '{"threshold": 0.5, "speech_pad_ms": 200}'},
                   {"chunk_length": "8"}):  # fmt: skip
        doc = _post(live, **fields).json()
        assert doc["segments"][0]["start"] > 9.0, fields  # silence removed before decoding
        assert "order will arrive" in doc["text"].lower()
    unsupported = {
        "repetition_penalty": "1.2", "no_repeat_ngram_size": "3", "compression_ratio_threshold": "2.0",
        "hallucination_silence_threshold": "2",
    }  # fmt: skip
    r = _post(live, **unsupported)
    assert r.headers["x-bolo-ignored-options"] == ", ".join(sorted(unsupported))
    guarded = _post(live, vad_filter="false", max_repeats="0")
    assert guarded.json()["text"] == whole["text"]  # nothing to trim on this clip either way


def test_whispercpp_detects_the_language_of_a_multilingual_model(live: LiveServer) -> None:
    audio, rate, _ = samples.load_audio("hi-short")
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
    r = httpx.post(
        live.url + "/v1/audio/transcriptions",
        data={"model": MODEL, "response_format": "verbose_json"},
        files={"file": ("hi.wav", buf.getvalue(), "audio/wav")},
        timeout=600,
    )
    assert r.status_code == 200, r.text
    assert r.json()["language"] == "hindi"
    told = _post(live, language="en").json()
    assert told["language"] == "english"


def test_whispercpp_word_timestamps(live: LiveServer) -> None:
    """Word timings from whisper.cpp's token timestamps; asking for them does not change the text."""
    plain = _post(live, vad_filter="false").json()
    for granularities in ("word", "segment"):
        r = httpx.post(
            live.url + "/v1/audio/transcriptions",
            data={"model": MODEL, "response_format": "verbose_json", "vad_filter": "false",
                  "timestamp_granularities[]": ["segment", "word"] if granularities == "segment" else ["word"]},
            files={"file": ("a.wav", CLIP, "audio/wav")},
            timeout=600,
        )  # fmt: skip
        assert r.status_code == 200, r.text
        assert "x-bolo-ignored-options" not in r.headers
        doc = r.json()
        assert doc["text"] == plain["text"]
        words = doc["words"]
        assert " ".join(w["word"] for w in words) == doc["text"]
        starts = [w["start"] for w in words]
        assert starts == sorted(starts) and all(0.0 <= w["start"] <= w["end"] <= doc["duration"] for w in words)
        assert 9.0 < words[0]["start"] < 12.0  # the speech starts after 10 s of near-silence
    assert _params(live)["token_timestamps"] is True
    _post(live, vad_filter="false")
    assert _params(live)["token_timestamps"] is False  # not carried over to the next request
