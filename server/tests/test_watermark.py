"""Watermarks on speech (bolo.watermark): the policy, marking chunk by chunk, detection, and never refusing speech."""

from __future__ import annotations

import base64
import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from bolo.api.app import create_app
from bolo.config import Settings
from conftest import wav_bytes
from test_voices import add_clone_model, tone_wav


def _client(settings: Settings) -> TestClient:
    add_clone_model(settings.data_dir)
    settings.watermark_model = "fake-watermark"
    return TestClient(create_app(settings))


def _detect(client: TestClient, audio: bytes, **data: str) -> dict:
    r = client.post("/v1/audio/watermark/detect", files={"file": ("a.wav", audio)}, data=data)
    assert r.status_code == 200, r.text
    return r.json()


def test_cloned_speech_is_marked_by_default_and_plain_speech_is_not(settings: Settings) -> None:
    with _client(settings) as client:
        ref = "data:audio/wav;base64," + base64.b64encode(tone_wav()).decode()
        body = {"model": "fake-tts-clone", "input": "Hello there. How are you?", "response_format": "wav"}
        r = client.post("/v1/audio/speech", json={**body, "reference_audio": ref})
        assert r.status_code == 200, r.text
        assert r.headers["x-bolo-watermark"] == "fake-watermark"
        found = _detect(client, r.content, model="fake-watermark")
        assert found["marked"] and found["score"] > 0.9 and found["payload_is_this_servers"]
        assert found["payload"] == settings.watermark_payload and found["segments"][0]["start"] == 0.0

        r = client.post("/v1/audio/speech", json=body)
        assert "x-bolo-watermark" not in r.headers
        assert not _detect(client, r.content, model="fake-watermark")["marked"]
        # asked for: marked; asked not to: a cloned voice goes out unmarked
        r = client.post("/v1/audio/speech", json={**body, "watermark": True})
        assert r.headers["x-bolo-watermark"] == "fake-watermark"
        r = client.post("/v1/audio/speech", json={**body, "reference_audio": ref, "watermark": "off"})
        assert "x-bolo-watermark" not in r.headers
        r = client.post("/v1/audio/speech", json={**body, "watermark": "nonsense"})
        assert r.status_code == 400 and r.json()["error"]["param"] == "watermark"


def test_a_watermarker_that_cannot_run_never_stops_the_speech(settings: Settings) -> None:
    settings.watermark = "all"
    with _client(settings) as client:
        settings.watermark_model = "audioseal-16bits"  # in the catalog, not installed, auto_pull off
        r = client.post("/v1/audio/speech", json={"input": "Hello.", "response_format": "wav"})
        assert r.status_code == 200
        assert r.headers["x-bolo-watermark"].startswith("unavailable; audioseal-16bits: ")


def test_marks_made_chunk_by_chunk_keep_the_grid() -> None:
    """A stream marked in chunks of any length is found again as a whole (the marker keeps the model's grid)."""
    import asyncio

    from bolo.engines.fake import FakeWatermark
    from bolo.watermark import Marker

    model = FakeWatermark()

    class Entry:
        def __init__(self) -> None:
            self.model = model

            class Queue:
                async def submit(self, op: str, item: tuple) -> tuple:
                    return model.embed(*item), None

            self.queue = Queue()

    mark = Marker(None, Entry(), "fake-watermark", "1" * 16)  # type: ignore[arg-type]
    x = np.zeros(24000 * 2, dtype=np.float32)
    pieces = [x[a : a + 3001] for a in range(0, len(x), 3001)]
    out = np.concatenate([asyncio.run(mark.mark(p, 24000)) for p in pieces])
    from bolo.audio.resample import resample

    found = model.detect(resample(out, 24000, 16000))
    assert found.score > 0.95 and len(out) == len(x)


def test_embed_and_detect_routes(settings: Settings) -> None:
    with _client(settings) as client:
        plain = wav_bytes(2.0)
        r = client.post("/v1/audio/watermark", files={"file": ("a.wav", plain)}, data={"model": "fake-watermark"})
        assert r.status_code == 200 and r.headers["x-bolo-watermark"] == "fake-watermark"
        audio, rate = sf.read(io.BytesIO(r.content))
        assert rate == 16000 and len(audio) == 32000
        assert _detect(client, r.content, model="fake-watermark")["marked"]
        assert not _detect(client, plain, model="fake-watermark")["marked"]
        r = client.post("/v1/audio/watermark/detect", files={"file": ("a.wav", plain)}, data={"model": "fake-tts"})
        assert r.status_code == 400
