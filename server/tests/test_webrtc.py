"""The WebRTC transport of the Realtime API and ephemeral client secrets, with a real aiortc peer."""

from __future__ import annotations

import asyncio
import json
import time
from fractions import Fraction
from typing import Any

import httpx
import numpy as np
import openai
import pytest
import websockets

from bolo.config import Settings
from conftest import LiveServer, speechlike
from fake_llm import FakeLLM

aiortc = pytest.importorskip("aiortc", reason="the WebRTC transport needs the 'webrtc' extra")
av = pytest.importorskip("av")

RATE = 48000


class Microphone(aiortc.MediaStreamTrack):  # type: ignore[misc, name-defined]
    """Plays an array at real-time pace, then silence, like a live microphone."""

    kind = "audio"

    def __init__(self, audio: np.ndarray) -> None:
        super().__init__()
        self._pcm = (audio * 32767).astype("<i2").tobytes()
        self._n = 0
        self._start: float | None = None

    async def recv(self) -> Any:
        if self._start is None:
            self._start = time.time()
        wait = self._start + self._n * 0.02 - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        chunk = self._pcm[self._n * 1920 : (self._n + 1) * 1920].ljust(1920, b"\0")
        frame = av.AudioFrame(format="s16", layout="mono", samples=960)
        frame.planes[0].update(chunk)
        frame.pts, frame.sample_rate, frame.time_base = self._n * 960, RATE, Fraction(1, RATE)
        self._n += 1
        return frame


class Peer:
    """A browser stand-in: one audio track out, one in, and the ``oai-events`` data channel."""

    def __init__(self, speech_s: float = 2.0) -> None:
        self.pc = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=[]))
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.heard: list[np.ndarray] = []
        self.channel = self.pc.createDataChannel("oai-events")
        self.channel.on("message", lambda m: self.events.put_nowait(json.loads(m)))
        self.pc.addTrack(Microphone(speechlike(speech_s, RATE)))
        self._listener: asyncio.Task[None] | None = None

        @self.pc.on("track")
        def on_track(track: Any) -> None:
            self._listener = asyncio.ensure_future(self._listen(track))

    async def _listen(self, track: Any) -> None:
        while True:
            try:
                frame = await track.recv()
            except Exception:
                return
            self.heard.append(frame.to_ndarray().reshape(-1).astype(np.float32) / 32768.0)

    async def offer(self) -> str:
        await self.pc.setLocalDescription(await self.pc.createOffer())
        return str(self.pc.localDescription.sdp)

    async def accept(self, answer: str) -> None:
        await self.pc.setRemoteDescription(aiortc.RTCSessionDescription(sdp=answer, type="answer"))

    async def until(self, last: str, timeout: float = 30.0) -> list[dict[str, Any]]:
        out = []
        async with asyncio.timeout(timeout):
            while True:
                event = await self.events.get()
                out.append(event)
                if event["type"] == last:
                    return out

    async def close(self) -> None:
        if self._listener:
            self._listener.cancel()
        await asyncio.wait_for(self.pc.close(), 20)  # a failing test, not a hung suite


async def post_offer(url: str, sdp: str, **kw: Any) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        return await client.post(
            url, content=sdp, headers={"Content-Type": "application/sdp", **kw.pop("headers", {})}, **kw
        )


async def test_voice_agent_turn_over_webrtc(agent: LiveServer) -> None:
    peer = Peer()
    try:
        r = await post_offer(agent.url + "/v1/realtime/calls?model=gpt-realtime", await peer.offer())
        assert r.status_code == 201, r.text
        assert r.headers["content-type"].startswith("application/sdp") and "m=audio" in r.text
        call_path = r.headers["location"]
        assert call_path.startswith("/v1/realtime/calls/rtc_")
        await peer.accept(r.text)
        created = await peer.events.get()
        assert created["type"] == "session.created" and created["session"]["type"] == "realtime"
        peer.channel.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "audio": {"input": {"transcription": {"model": "fake-stt"}}}}}))  # fmt: skip
        events = await peer.until("output_audio_buffer.stopped")
        types = [e["type"] for e in events]
        for expected in (
            "session.updated", "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
            "conversation.item.input_audio_transcription.completed", "response.created",
            "response.output_audio_transcript.delta", "output_audio_buffer.started", "response.done",
        ):  # fmt: skip
            assert expected in types, expected
        # over WebRTC the audio travels on the track, never as events
        assert "response.output_audio.delta" not in types
        assert types.index("output_audio_buffer.started") < types.index("output_audio_buffer.stopped")
        user = next(e for e in events if e["type"] == "conversation.item.input_audio_transcription.completed")
        assert user["transcript"].startswith("the quick brown fox")
        done = next(e for e in events if e["type"] == "response.done")
        assert done["response"]["status"] == "completed"
        said = done["response"]["output"][0]["content"][0]["transcript"]
        assert said.startswith("You said: the quick brown fox")
        # the reply was really played to us: a few seconds of sound on the incoming track
        heard = np.concatenate(peer.heard)
        loud = float(np.mean(np.abs(heard) > 0.01)) * len(heard) / RATE
        assert loud > 2.0, f"only {loud:.1f} s of audible reply"
        trace = httpx.get(agent.url + "/api/v1/requests", params={"kind": "agent"}).json()["requests"][0]
        assert trace["transport"] == "webrtc" and trace["status"] == "ok" and trace["ttfa_ms"] > 0
        assert len(agent.core.calls) == 1
        assert httpx.post(agent.url + call_path + "/hangup").status_code == 200
        assert len(agent.core.calls) == 0
        assert httpx.post(agent.url + call_path + "/hangup").status_code == 404
    finally:
        await peer.close()


async def test_barge_in_clears_the_outgoing_audio(agent: LiveServer) -> None:
    # 1.5 s of speech, a pause long enough for the reply to start, then the user speaks again
    audio = np.concatenate([speechlike(1.5, RATE), np.zeros(int(2.0 * RATE), dtype=np.float32), speechlike(1.5, RATE)])
    peer = Peer()
    peer.pc.getSenders()[0].replaceTrack(Microphone(audio))
    try:
        session = {"type": "realtime", "audio": {"input": {"transcription": {"model": "fake-stt"},
                                                           "turn_detection": {"type": "server_vad", "silence_duration_ms": 300}}}}  # fmt: skip
        client = openai.AsyncOpenAI(base_url=agent.url + "/v1", api_key="x", max_retries=0, timeout=60)
        answer = await client.realtime.calls.create(
            sdp=await peer.offer(), session=session
        )  # multipart, as the SDK sends it
        await peer.accept(answer.text)
        events = await peer.until("output_audio_buffer.cleared")
        types = [e["type"] for e in events]
        # the reply was still playing when the user spoke again: what was queued is dropped at once
        assert types.count("input_audio_buffer.speech_started") == 2
        second = len(types) - 1 - types[::-1].index("input_audio_buffer.speech_started")
        assert types.index("output_audio_buffer.started") < second < types.index("output_audio_buffer.cleared")
        assert "output_audio_buffer.stopped" not in types
    finally:
        await peer.close()


async def test_transcription_only_call_and_bad_offers(live: LiveServer) -> None:
    peer = Peer()
    try:
        r = await post_offer(live.url + "/v1/realtime/calls?intent=transcription", await peer.offer())
        assert r.status_code == 201, r.text
        await peer.accept(r.text)
        events = await peer.until("conversation.item.input_audio_transcription.completed")
        assert events[0]["session"]["type"] == "transcription"
        assert events[-1]["transcript"].startswith("the quick brown fox")
        peer.channel.send(json.dumps({"type": "input_audio_buffer.append", "audio": "AAAA"}))
        error = (await peer.until("error"))[-1]
        assert error["error"]["code"] == "unsupported" and "audio track" in error["error"]["message"]
    finally:
        await peer.close()
    bad = await post_offer(live.url + "/v1/realtime/calls", "hello")
    assert bad.status_code == 400 and bad.json()["error"]["param"] == "sdp"


async def test_client_secrets_open_sessions_without_the_server_key(settings: Settings, llm: FakeLLM) -> None:
    settings.api_key = "sekrit"
    settings.llm_base_url, settings.llm_model = llm.url, "fake-llm"
    srv = LiveServer(settings)
    srv.start()
    try:
        url = srv.url + "/v1/realtime/client_secrets"
        assert httpx.post(url, json={}).status_code == 401  # minting needs the real key
        client = openai.AsyncOpenAI(base_url=srv.url + "/v1", api_key="sekrit", max_retries=0, timeout=60)
        secret = await client.realtime.client_secrets.create(
            expires_after={"anchor": "created_at", "seconds": 60},
            session={"type": "realtime", "instructions": "Be brief.", "audio": {"output": {"voice": "beta"}}},
        )
        assert secret.value.startswith("ek_") and 50 <= secret.expires_at - time.time() <= 61
        assert secret.session.type == "realtime" and secret.session.instructions == "Be brief."
        ek = {"Authorization": f"Bearer {secret.value}"}
        # the secret opens Realtime sessions (SDK header, browser subprotocol) with the stored configuration ...
        async with openai.AsyncOpenAI(base_url=srv.url + "/v1", api_key=secret.value).realtime.connect(
            model="gpt-realtime"
        ) as conn:
            created = await conn.recv()
            assert created.session.instructions == "Be brief." and created.session.audio.output.voice == "beta"
        async with websockets.connect(
            srv.ws_url + "/v1/realtime", subprotocols=["realtime", f"openai-insecure-api-key.{secret.value}"]
        ) as ws:
            assert json.loads(await ws.recv())["session"]["instructions"] == "Be brief."
        peer = Peer()
        try:
            r = await post_offer(srv.url + "/v1/realtime/calls", await peer.offer(), headers=ek)
            assert r.status_code == 201, r.text
            assert httpx.post(srv.url + r.headers["location"] + "/hangup", headers=ek).status_code == 200
        finally:
            await peer.close()
        # ... and nothing else
        assert httpx.get(srv.url + "/v1/models", headers=ek).status_code == 401
        assert httpx.post(url, json={}, headers=ek).status_code == 401
        assert httpx.post(srv.url + "/v1/audio/speech", json={"input": "hi"}, headers=ek).status_code == 401
        # a wrong or expired secret is refused
        async with websockets.connect(
            srv.ws_url + "/v1/realtime", additional_headers={"Authorization": "Bearer ek_nope"}
        ) as ws:
            assert json.loads(await ws.recv())["error"]["code"] == "invalid_api_key"
        short = httpx.post(
            url, json={"expires_after": {"seconds": 10}}, headers={"Authorization": "Bearer sekrit"}
        ).json()
        srv.core.client_secrets[short["value"]] = (time.time() - 1, {})
        peer = Peer()
        try:
            denied = await post_offer(
                srv.url + "/v1/realtime/calls",
                await peer.offer(),
                headers={"Authorization": f"Bearer {short['value']}"},
            )
            assert denied.status_code == 401 and short["value"] not in srv.core.client_secrets
        finally:
            await peer.close()
        # validation
        auth = {"Authorization": "Bearer sekrit"}
        assert httpx.post(url, json={"expires_after": {"seconds": 5}}, headers=auth).status_code == 400
        bad = httpx.post(url, json={"session": {"tools": [{"type": "mcp"}]}}, headers=auth)
        assert bad.status_code == 400 and "MCP" in bad.json()["error"]["message"]
        transcription = httpx.post(url, json={"session": {"type": "transcription"}}, headers=auth).json()
        assert transcription["session"]["type"] == "transcription"
    finally:
        srv.stop()
