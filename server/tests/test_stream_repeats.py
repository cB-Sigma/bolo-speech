"""The repetition guard on live transcripts (WebSocket STT and Realtime), across utterances, with fake recognisers
that loop."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx
import numpy as np
import pytest
import websockets

from bolo.engines.base import Word
from bolo.stt_options import RepeatGuard
from conftest import LiveServer, speechlike


def pcm16(x: np.ndarray) -> bytes:
    return (x * 32767).astype("<i2").tobytes()


async def collect(ws: Any) -> list[dict[str, Any]]:
    return [json.loads(m) async for m in ws]


@pytest.fixture
def loop(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Make a fake model loop (its catalog options are read when it loads; the catalog's built-in entries are shared
    between tests, so the change is undone afterwards)."""

    def apply(live: LiveServer, model: str, **options: Any) -> None:
        spec = live.core.catalog.resolve(model)
        assert spec is not None
        for key, value in options.items():
            monkeypatch.setitem(spec.options, key, value)

    return apply


def pin(live: LiveServer, model: str, **values: Any) -> None:
    r = httpx.put(live.url + f"/api/v1/models/{model}/settings", json=values)
    assert r.status_code == 200, r.text


async def utterances(live: LiveServer, model: str, count: int, seconds: float) -> tuple[str, list[dict[str, Any]]]:
    """``count`` utterances, each ended with a flush; returns the request id and every message."""
    url = live.ws_url + f"/api/v1/stt/stream?model={model}&vad=false&interim=false"
    async with websockets.connect(url) as ws:
        ready = json.loads(await ws.recv())
        for _ in range(count):
            await ws.send(pcm16(speechlike(seconds)))
            await ws.send(pcm16(np.zeros(1600, dtype=np.float32)))
            await ws.send(json.dumps({"type": "flush"}))
            await asyncio.sleep(0.2)
        await ws.send(json.dumps({"type": "stop"}))
        return ready["request_id"], await collect(ws)


def finals(messages: list[dict[str, Any]]) -> list[str]:
    return [m["text"] for m in messages if m["type"] == "final"]


# ----------------------------------------------------------------------
# the guard itself
# ----------------------------------------------------------------------


def test_the_guard_never_changes_what_was_sent_and_drops_copies_after_the_limit() -> None:
    guard = RepeatGuard(max_repeats=4)
    out = [guard.apply(t)[0] for t in ["Hello there.", *["Thank you."] * 6, "Thank you. Bye."]]
    # four copies went out before anything looked like a loop; they stay; the copies after them do not
    assert out == ["Hello there.", *["Thank you."] * 4, "", "", "Bye."]
    assert guard.removed == 6


def test_the_guard_sees_a_loop_that_runs_on_from_one_final_into_the_next() -> None:
    guard = RepeatGuard()
    assert guard.apply("so we went the the")[0] == "so we went the the"
    words = [Word("the", i * 0.2, i * 0.2 + 0.1) for i in range(6)] + [Word("end", 1.2, 1.4)]
    text, kept = guard.apply("the the the the the the end", words)
    assert text == "end" and [w.w for w in kept] == ["end"]
    assert guard.apply("the end")[0] == "the end"  # an ordinary repeat later on is left alone
    loop_inside_one = RepeatGuard().apply("hello " + "you know " * 10 + "right")[0]
    assert loop_inside_one == "hello you know right"


def test_a_guard_set_to_zero_does_nothing() -> None:
    guard = RepeatGuard(max_repeats=0)
    assert [guard.apply("the the the the the the")[0] for _ in range(3)] == ["the the the the the the"] * 3


# ----------------------------------------------------------------------
# WebSocket STT
# ----------------------------------------------------------------------


async def test_native_streaming_loop_across_utterances_is_cut(live: LiveServer, loop: Any) -> None:
    loop(live, "fake-stt-streaming", loop_word="the")
    # 1.2 s: two or three "the" per utterance; the third utterance makes it more than four in a row
    request_id, messages = await utterances(live, "fake-stt-streaming", 3, 1.2)
    sent = finals(messages)
    assert messages[-1]["type"] == "done" and sent and set(" ".join(sent).split()) == {"the"}
    assert len(" ".join(sent).split()) <= 4 < 3 * len(sent[0].split())
    trace = httpx.get(live.url + f"/api/v1/requests/{request_id}").json()
    assert trace["repeats_trimmed"] >= 1 and trace["utterances"] == len(sent)
    # a longer loop inside one utterance is cut there too
    _, messages = await utterances(live, "fake-stt-streaming", 2, 3.0)
    assert finals(messages) == ["the"]


async def test_a_pinned_max_repeats_applies_to_live_sessions(live: LiveServer, loop: Any) -> None:
    loop(live, "fake-stt-streaming", loop_word="the")
    pin(live, "fake-stt-streaming", max_repeats=0)
    _, messages = await utterances(live, "fake-stt-streaming", 3, 3.0)
    sent = finals(messages)
    assert len(sent) == 3 and all(len(t.split()) > 4 for t in sent)  # off: the loop goes out as it is
    pin(live, "fake-stt-streaming", max_repeats=20)
    _, messages = await utterances(live, "fake-stt-streaming", 8, 3.0)
    words = len(" ".join(finals(messages)).split())
    assert 4 < words <= 20 < 8 * len(sent[0].split())  # up to twenty in a row go out, then no more


async def test_pseudo_streaming_finals_are_guarded_and_get_the_pinned_options(live: LiveServer, loop: Any) -> None:
    loop(live, "fake-stt", loop_on="*")  # every decode: its words, then "the" forty times
    pin(live, "fake-stt", repetition_penalty=1.2)
    _, messages = await utterances(live, "fake-stt", 2, 1.2)
    assert finals(messages) == ["the quick brown the"] * 2
    seen = live.core.manager.loaded["fake-stt"].model.seen_options  # type: ignore[attr-defined]
    assert seen and all(o == {"repetition_penalty": 1.2} for o in seen)


# ----------------------------------------------------------------------
# Realtime
# ----------------------------------------------------------------------


async def _realtime_items(live: LiveServer, count: int) -> list[str]:
    ws = await websockets.connect(live.ws_url + "/v1/realtime?intent=transcription")
    try:
        assert json.loads(await ws.recv())["type"] == "session.created"
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "transcription", "audio": {
            "input": {"turn_detection": None, "transcription": {"model": "fake-stt"}}}}}))  # fmt: skip
        assert json.loads(await ws.recv())["type"] == "session.updated"
        out = []
        for _ in range(count):
            pcm = pcm16(speechlike(0.4, 24000))
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            async with asyncio.timeout(20):
                while True:
                    event = json.loads(await ws.recv())
                    if event["type"] == "conversation.item.input_audio_transcription.completed":
                        out.append(event["transcript"])
                        break
        return out
    finally:
        await ws.close()


async def test_realtime_transcripts_are_guarded_across_items(live: LiveServer, loop: Any) -> None:
    loop(live, "fake-stt", loop_on="*")
    # 0.4 s: one word, then the loop; each item is cut to "the" on its own, and the fifth "the" in a row is a loop
    assert await _realtime_items(live, 6) == ["the"] * 4 + ["", ""]
    pin(live, "fake-stt", max_repeats=0)
    [unguarded] = await _realtime_items(live, 1)
    assert set(unguarded.split()) == {"the"} and len(unguarded.split()) >= 40
