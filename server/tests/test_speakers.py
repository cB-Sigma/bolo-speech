"""Enrolled speakers: enrol, verify, identify (``/v1/audio/speakers``), with the fake embedder (the dominant pitch
is the voice: 100 Hz and 500 Hz are different people)."""

from __future__ import annotations

import io

import httpx
import numpy as np

from bolo import speakers
from bolo.audio.encode import encode_audio
from conftest import LiveServer

SR = 16000
MODEL = "fake-speaker-embedding"


def voice(hz: float, seconds: float = 2.0) -> tuple[str, io.BytesIO, str]:
    t = np.arange(int(seconds * SR)) / SR
    x = (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    return ("clip.wav", io.BytesIO(encode_audio(x, SR, "wav")), "audio/wav")


def test_threshold_for() -> None:
    assert speakers.threshold_for("not-measured") == (speakers.DEFAULT_THRESHOLD, False)
    for model, value in speakers.THRESHOLDS.items():
        assert speakers.threshold_for(model) == (value, True)
    for pipeline, embedder in speakers.SAME_FILE.items():
        assert speakers.threshold_for(pipeline) == speakers.threshold_for(embedder)


def test_enrol_verify_identify(live: LiveServer) -> None:
    url = live.url + "/v1/audio/speakers"
    r = httpx.post(
        url, data={"name": "Asha Rao", "model": MODEL, "metadata": '{"team": "support"}'},
        files=[("file", voice(100)), ("file", voice(100, 3.0))], timeout=30,
    )  # fmt: skip
    assert r.status_code == 201, r.text
    asha = r.json()
    assert asha["id"] == "asha-rao" and len(asha["clips"]) == 2 and asha["seconds"] == 5.0
    assert asha["embedding_models"] == [MODEL] and asha["metadata"] == {"team": "support"}
    r = httpx.post(url, data={"name": "Ben", "id": "ben"}, files={"file": voice(500)}, timeout=30)
    assert r.status_code == 201 and r.json()["embedding_models"] == []  # computed on first use
    assert httpx.post(url, data={"name": "Ben", "id": "ben"}, files={"file": voice(500)}).status_code == 409
    assert httpx.post(url, data={"name": "X", "id": "verify"}, files={"file": voice(500)}).status_code == 400
    r = httpx.post(url, data={"name": "Short"}, files={"file": voice(100, 0.5)})
    assert r.status_code == 400 and r.json()["error"]["param"] == "file"
    assert httpx.post(url, data={"name": "Nobody"}).status_code == 400

    listing = httpx.get(url).json()
    assert [s["id"] for s in listing["data"]] == ["asha-rao", "ben"] and "threshold" in listing

    verify = url + "/verify"
    r = httpx.post(verify, data={"speaker": "Asha Rao", "model": MODEL}, files={"file": voice(100)}, timeout=30)
    body = r.json()
    assert r.status_code == 200 and body["match"] is True and body["similarity"] > 0.99
    assert body["speaker"] == {"id": "asha-rao", "name": "Asha Rao"} and body["calibrated"] is False
    r = httpx.post(verify, data={"speaker": "ben", "model": MODEL}, files={"file": voice(100)}, timeout=30)
    assert r.json()["match"] is False and r.json()["similarity"] < 0.1
    r = httpx.post(verify, data={"speaker": "ben", "model": MODEL, "threshold": "-0.5"}, files={"file": voice(100)})
    assert r.json()["match"] is True and r.json()["threshold"] == -0.5
    r = httpx.post(verify, data={"speaker": "carol", "model": MODEL}, files={"file": voice(100)})
    assert r.status_code == 404 and r.json()["error"]["code"] == "speaker_not_found"

    identify = url + "/identify"
    r = httpx.post(identify, data={"model": MODEL}, files={"file": voice(500)}, timeout=30)
    body = r.json()
    assert body["speaker"] == {"id": "ben", "name": "Ben"} and [c["id"] for c in body["candidates"]] == [
        "ben",
        "asha-rao",
    ]
    assert httpx.get(url + "/ben").json()["embedding_models"] == [MODEL]  # the voiceprint was kept
    r = httpx.post(identify, data={"model": MODEL, "speakers": "asha-rao"}, files={"file": voice(500)}, timeout=30)
    assert r.json()["speaker"] is None and len(r.json()["candidates"]) == 1  # nobody enrolled sounds like this
    r = httpx.post(identify, data={"model": MODEL, "top_k": "1"}, files={"file": voice(100)}, timeout=30)
    assert r.json()["speaker"]["id"] == "asha-rao" and len(r.json()["candidates"]) == 1

    # more recordings, a new name, a recording removed; the voiceprint is recomputed
    r = httpx.post(url + "/ben", data={"name": "Ben K."}, files={"file": voice(500, 1.5)}, timeout=30)
    ben = r.json()
    assert ben["name"] == "Ben K." and len(ben["clips"]) == 2 and ben["embedding_models"] == []
    clip = ben["clips"][0]["id"]
    audio = httpx.get(url + f"/ben/clips/{clip}")
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
    assert len(httpx.delete(url + f"/ben/clips/{clip}").json()["clips"]) == 1
    r = httpx.post(url + "/ben", json={"metadata": {"role": "agent"}})
    assert r.json()["metadata"] == {"role": "agent"}
    assert httpx.delete(url + "/ben").json()["deleted"] is True
    assert httpx.get(url + "/ben").status_code == 404
    assert [s["id"] for s in httpx.get(url).json()["data"]] == ["asha-rao"]
