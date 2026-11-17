"""A model with work in flight is never unloaded as idle, and a disconnected client's decode is stopped where the
engine can stop it (its result discarded where it cannot)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from bolo.config import Settings
from conftest import LiveServer, wav_bytes


@pytest.fixture
def slow(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """A server whose fake-stt takes 3 s per decode and is unloaded 0.2 s after it is idle."""
    settings.stt_ttl_s = 0.2

    def start(abortable: bool) -> LiveServer:
        srv = LiveServer(settings)
        spec = srv.core.catalog.resolve("fake-stt")
        assert spec is not None
        monkeypatch.setitem(spec.options, "base_delay_s", 3.0)
        monkeypatch.setitem(spec.options, "abortable", abortable)
        srv.start()
        servers.append(srv)
        return srv

    servers: list[LiveServer] = []
    yield start
    for srv in servers:
        srv.stop()


def _disconnect_mid_decode(live: LiveServer) -> float:
    """Send a request and hang up 0.5 s later, while the engine is decoding; returns when that was."""
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(
            live.url + "/v1/audio/transcriptions",
            data={"model": "fake-stt"},
            files={"file": ("a.wav", wav_bytes(1.0), "audio/wav")},
            timeout=httpx.Timeout(10.0, read=0.5),
        )
    return time.monotonic()


def _wait(condition: Any, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def test_a_decode_that_cannot_stop_keeps_its_model_loaded_until_it_ends(slow: Any) -> None:
    live = slow(abortable=False)
    hung_up = _disconnect_mid_decode(live)
    entry = live.core.manager.loaded["fake-stt"]
    assert _wait(lambda: entry.active == 0, 1.0)  # the server noticed the hang-up: nobody holds the model now
    assert entry.queue.inflight == 1  # but the engine is still at it
    assert entry.busy and entry.idle_s() == 0.0
    # the idle limit is 0.2 s: before the fix the model was unloaded here, under the running decode
    while entry.queue.inflight:
        assert "fake-stt" in live.core.manager.loaded, "unloaded while a decode was running"
        time.sleep(0.02)
    finished = time.monotonic()
    assert finished - hung_up > 1.5  # it really ran on (3 s in all)
    assert entry.model.aborted == 0  # type: ignore[attr-defined]
    # its result went nowhere; the model is idle from when the decode ended, and then unloaded
    assert _wait(lambda: "fake-stt" not in live.core.manager.loaded, 5.0)
    trace = live.core.traces.recent(1, "stt", None)[0]
    assert trace.status == "cancelled"


def test_a_decode_that_can_stop_is_stopped_when_the_client_goes(slow: Any) -> None:
    live = slow(abortable=True)
    hung_up = _disconnect_mid_decode(live)
    entry = live.core.manager.loaded.get("fake-stt")
    model = entry.model if entry else None
    assert _wait(lambda: model is not None and model.aborted == 1, 2.0)  # type: ignore[attr-defined]
    assert time.monotonic() - hung_up < 2.0  # well before the 3 s the decode would have taken
    assert _wait(lambda: "fake-stt" not in live.core.manager.loaded, 5.0)


def test_a_new_request_is_served_after_a_cancelled_one(slow: Any) -> None:
    live = slow(abortable=True)
    _disconnect_mid_decode(live)
    r = httpx.post(
        live.url + "/v1/audio/transcriptions",
        data={"model": "fake-stt"},
        files={"file": ("a.wav", wav_bytes(1.0), "audio/wav")},
        timeout=30,
    )
    assert r.status_code == 200 and r.json()["text"].startswith("the quick")
