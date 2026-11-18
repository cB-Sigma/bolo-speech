"""Sound-event tags (``/v1/audio/tags``), speech emotion (``/v1/audio/emotion``) and SenseVoice's emotion and event
tags, with the fake models."""

from __future__ import annotations

import io

import httpx
import numpy as np

from bolo.api.openai import paralinguistics
from bolo.audio.encode import encode_audio
from bolo.engines.base import Segment
from bolo.engines.sherpa import sense_voice_tags
from conftest import LiveServer

SR = 16000


def clip(parts: list[tuple[float, float]]) -> tuple[str, io.BytesIO, str]:
    """(hz, seconds) pieces one after the other; 0 Hz is silence."""
    out = []
    for hz, seconds in parts:
        t = np.arange(int(seconds * SR)) / SR
        out.append((0.3 * np.sin(2 * np.pi * hz * t)) if hz else np.zeros(len(t)))
    x = np.concatenate(out).astype(np.float32)
    return ("clip.wav", io.BytesIO(encode_audio(x, SR, "wav")), "audio/wav")


def test_sense_voice_tags() -> None:
    assert sense_voice_tags("<|HAPPY|>", "<|Speech|>") == ("happy", ["speech"])
    assert sense_voice_tags("<|EMO_UNKNOWN|>", "<|BGM|><|Laughter|>") == (None, ["music", "laughter"])
    assert sense_voice_tags("", "") == (None, [])


def test_paralinguistics() -> None:
    segs = [
        Segment(0.0, 2.0, "a", emotion="happy", events=["speech"]),
        Segment(2.0, 5.0, "b", emotion="neutral", events=["speech", "laughter"]),
        Segment(5.0, 6.0, "c"),
    ]
    out = paralinguistics(segs)
    assert out["emotion"] == "neutral" and out["emotions"] == {"neutral": 3.0, "happy": 2.0}
    assert out["audio_events"] == {"speech": 5.0, "laughter": 3.0}
    assert paralinguistics([Segment(0.0, 1.0, "x")]) == {}


def test_tags_route(live: LiveServer) -> None:
    url = live.url + "/v1/audio/tags"
    r = httpx.post(url, files={"file": clip([(200, 2.0)])}, data={"model": "fake-tagging"}, timeout=30)
    body = r.json()
    assert r.status_code == 200 and r.headers["x-bolo-model"] == "fake-tagging"
    assert body["tags"] == [{"label": "Music", "score": 0.9}] and body["duration"] == 2.0  # 0.05 is under 0.1
    r = httpx.post(
        url, files={"file": clip([(200, 2.0)])}, data={"model": "fake-tagging", "threshold": "0", "top_k": "2"},
        timeout=30,
    )  # fmt: skip
    assert [t["label"] for t in r.json()["tags"]] == ["Music", "Speech"]
    r = httpx.post(
        url, files={"file": clip([(200, 2.0)])}, data={"model": "fake-tagging", "labels": "speech", "threshold": "0"},
        timeout=30,
    )  # fmt: skip
    assert r.json()["tags"] == [{"label": "Speech", "score": 0.05}]
    # sound event detection over time: window by window
    r = httpx.post(
        url, files={"file": clip([(200, 2.0), (0, 2.0), (600, 2.0)])}, data={"model": "fake-tagging", "window": "2"},
        timeout=30,
    )  # fmt: skip
    segs = r.json()["segments"]
    assert [(s["start"], s["end"], s["tags"][0]["label"]) for s in segs] == [
        (0.0, 2.0, "Music"),
        (2.0, 4.0, "Silence"),
        (4.0, 6.0, "Speech"),
    ]
    for data, param in (({"model": "fake-emotion"}, "model"), ({"model": "fake-tagging", "window": "0.1"}, "window")):
        r = httpx.post(url, files={"file": clip([(200, 2.0)])}, data=data, timeout=30)
        assert r.status_code == 400 and r.json()["error"]["param"] == param


def test_long_recording_is_heard_in_windows(live: LiveServer) -> None:
    """Over 30 s without a window: tags keep each class's highest probability in any 10 s window."""
    r = httpx.post(
        live.url + "/v1/audio/tags", files={"file": clip([(200, 25.0), (600, 10.0)])},
        data={"model": "fake-tagging", "top_k": "2"}, timeout=30,
    )  # fmt: skip
    assert {t["label"] for t in r.json()["tags"]} == {"Speech", "Music"}
    assert "segments" not in r.json()


def test_emotion_route(live: LiveServer) -> None:
    url = live.url + "/v1/audio/emotion"
    r = httpx.post(url, files={"file": clip([(500, 2.0)])}, data={"model": "fake-emotion"}, timeout=30)
    body = r.json()
    assert body["emotion"] == "happy" and body["score"] == 0.8 and len(body["emotions"]) == 4
    r = httpx.post(
        url, files={"file": clip([(100, 2.0), (500, 2.0)])}, data={"model": "fake-emotion", "window": "2", "top_k": "1"},
        timeout=30,
    )  # fmt: skip
    body = r.json()
    assert [s["emotion"] for s in body["segments"]] == ["neutral", "happy"] and len(body["emotions"]) == 1
