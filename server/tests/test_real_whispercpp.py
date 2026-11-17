"""whisper.cpp (pywhispercpp) with a real model on CPU: the decoding options. Opt-in: ``BOLO_TEST_REAL=1 pytest -m real``.

Downloads ``ggml-tiny.bin`` (75 MB) from ggerganov/whisper.cpp into ``BOLO_DATA_DIR``.
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
    data_dir = _data_dir()
    folder = data_dir / "validation" / "whispercpp-tiny"
    model_file = folder / "ggml-tiny.bin"
    if not model_file.exists():
        folder.mkdir(parents=True, exist_ok=True)
        url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
            r.raise_for_status()
            with model_file.open("wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
    srv = LiveServer(Settings(data_dir=data_dir, auto_pull=True, device="cpu"))
    from bolo.custom import register_local_model

    spec = srv.core.catalog.resolve(MODEL) or register_local_model(
        srv.core, str(folder), "whispercpp", "stt", name=MODEL
    )
    srv.start()
    try:
        if not srv.core.manager.is_installed(spec):
            srv.call(srv.core.pull_and_wait(spec.id))
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
    assert defaults["no_context"] is True and defaults["language"] == "en"
    r = _post(
        live, vad_filter="false", condition_on_previous_text="true", log_prob_threshold="-0.1",
        no_speech_threshold="0.01", temperature_increment_on_fallback="0.4",
    )  # fmt: skip
    assert "x-bolo-ignored-options" not in r.headers and "order will arrive" in r.json()["text"].lower()
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
