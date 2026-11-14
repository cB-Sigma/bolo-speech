"""The inbound SIP endpoint, driven by a small SIP user agent over loopback UDP: INVITE with an SDP offer,
ACK, G.711 RTP both ways, RFC 4733 key presses, BYE from either side."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import struct
import time
from typing import Any

import httpx
import numpy as np
import pytest

from bolo.audio.codecs import alaw_encode, mulaw_decode, mulaw_encode
from bolo.config import Settings
from conftest import LiveServer
from fake_llm import FakeLLM

sipmessage = pytest.importorskip("sipmessage")

from bolo.telephony.simulator import speechlike  # noqa: E402
from bolo.telephony.sip import Offer, build_answer, parse_offer  # noqa: E402

OFFER = """v=0\r
o=- 1 1 IN IP4 127.0.0.1\r
s=call\r
c=IN IP4 127.0.0.1\r
t=0 0\r
m=audio {port} RTP/AVP {types} 101\r
a=rtpmap:0 PCMU/8000\r
a=rtpmap:8 PCMA/8000\r
a=rtpmap:101 telephone-event/8000\r
a=fmtp:101 0-16\r
a=sendrecv\r
"""


class Phone:
    """A SIP user agent that is just enough of a trunk: one call, RTP sent in real time."""

    def __init__(self, server_port: int, types: str = "0 8") -> None:
        self.server = ("127.0.0.1", server_port)
        self.sip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sip.bind(("127.0.0.1", 0))
        self.sip.setblocking(False)
        self.rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rtp.bind(("127.0.0.1", 0))
        self.rtp.setblocking(False)
        self.port = self.sip.getsockname()[1]
        self.types = types
        self.call_id = f"call-{time.time_ns()}@127.0.0.1"
        self.cseq = 1
        self.to_tag = ""
        self.remote_rtp: tuple[str, int] | None = None
        self.payload_type = 0
        self.received: list[bytes] = []  # RTP payloads from bolo
        self.markers = 0
        self.requests: list[Any] = []  # requests bolo sent (BYE)
        self.responses: list[Any] = []
        self._seq, self._ts = 0, 0
        self._tasks: list[asyncio.Task[None]] = []

    def request(self, method: str, to: str = "+15550100002", body: bytes = b"", cseq: int | None = None) -> bytes:
        number = cseq or self.cseq
        tag = f";tag={self.to_tag}" if self.to_tag and method != "INVITE" else ""
        lines = [
            f"{method} sip:{to}@127.0.0.1:{self.server[1]} SIP/2.0",
            f"Via: SIP/2.0/UDP 127.0.0.1:{self.port};branch=z9hG4bK{method}{number}{time.time_ns() % 100000};rport",
            "Max-Forwards: 70",
            f'From: "Caller" <sip:+15550100001@127.0.0.1:{self.port}>;tag=caller1',
            f"To: <sip:{to}@127.0.0.1>{tag}",
            f"Call-ID: {self.call_id}",
            f"CSeq: {number} {method}",
            f"Contact: <sip:+15550100001@127.0.0.1:{self.port}>",
            "User-Agent: test-trunk",
        ]
        if body:
            lines.append("Content-Type: application/sdp")
        lines.append(f"Content-Length: {len(body)}")
        return ("\r\n".join(lines) + "\r\n\r\n").encode() + body

    async def send(self, data: bytes) -> None:
        await asyncio.get_running_loop().sock_sendto(self.sip, data, self.server)

    async def _sip_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            data, addr = await loop.sock_recvfrom(self.sip, 65535)
            message = sipmessage.Message.parse(data)
            if isinstance(message, sipmessage.Response):
                self.responses.append(message)
            else:
                self.requests.append(message)
                reply = sipmessage.Response(200, "OK")
                reply.via, reply.from_address, reply.to_address = message.via, message.from_address, message.to_address
                reply.call_id, reply.cseq, reply.content_length = message.call_id, message.cseq, 0
                await loop.sock_sendto(self.sip, bytes(reply), addr)

    async def _rtp_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            data, _ = await loop.sock_recvfrom(self.rtp, 65535)
            if data[1] & 0x80:
                self.markers += 1
            self.received.append(data[12:])

    async def response(self, code: int, method: str = "INVITE", timeout: float = 5.0) -> Any:
        deadline = time.time() + timeout
        while time.time() < deadline:
            for r in self.responses:
                if r.code == code and r.cseq.method == method:
                    return r
            await asyncio.sleep(0.01)
        raise AssertionError(f"no {code} to {method}: got {[(r.code, r.cseq.method) for r in self.responses]}")

    async def invite(self, to: str = "+15550100002") -> Any:
        self._tasks = [asyncio.create_task(self._sip_loop()), asyncio.create_task(self._rtp_loop())]
        body = OFFER.format(port=self.rtp.getsockname()[1], types=self.types).encode()
        await self.send(self.request("INVITE", to, body))
        return body

    async def answered(self) -> Any:
        ok = await self.response(200)
        self.to_tag = ok.to_address.parameters["tag"]
        offer = parse_offer(ok.body)
        assert offer is not None
        self.remote_rtp, self.payload_type = (offer.host, offer.port), offer.payload_type
        await self.send(self.request("ACK"))
        self.cseq += 1
        return ok

    async def _packet(
        self, payload_type: int, payload: bytes, timestamp: int | None = None, marker: bool = False
    ) -> None:
        assert self.remote_rtp is not None
        self._seq += 1
        header = struct.pack(
            "!BBHII",
            0x80,
            payload_type | (0x80 if marker else 0),
            self._seq,
            self._ts if timestamp is None else timestamp,
            0x1234,
        )
        await asyncio.get_running_loop().sock_sendto(self.rtp, header + payload, self.remote_rtp)

    async def talk(self, samples: np.ndarray, silence_s: float = 0.0) -> None:
        """Send audio in 20 ms packets, in real time, then silence."""
        encode = mulaw_encode if self.payload_type == 0 else alaw_encode
        data = encode(np.concatenate([samples, np.zeros(int(silence_s * 8000), dtype=np.float32)]))
        started = time.perf_counter()
        for i in range(0, len(data) - 159, 160):
            await self._packet(self.payload_type, data[i : i + 160])
            self._ts += 160
            await asyncio.sleep(max(0.0, started + (i // 160 + 1) * 0.02 - time.perf_counter()))

    async def press(self, digit: str) -> None:
        """RFC 4733: the event repeated while the key is held, then three packets with the end bit."""
        event = "0123456789*#ABCD".index(digit)
        stamp = self._ts
        for duration in (160, 320, 480):
            await self._packet(101, struct.pack("!BBH", event, 10, duration), stamp, marker=duration == 160)
            await asyncio.sleep(0.02)
        for _ in range(3):
            await self._packet(101, struct.pack("!BBH", event, 0x80 | 10, 640), stamp)
        self._ts += 800

    def heard_s(self) -> float:
        """Seconds of audio from bolo that were not silence."""
        return sum(len(p) for p in self.received if p.strip(b"\xff\xd5")) / 8000.0

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.sip.close()
        self.rtp.close()


def sip_server(settings: Settings, llm: FakeLLM, **agent: Any) -> LiveServer:
    settings.llm_base_url, settings.llm_model = llm.url, "fake-llm"
    settings.sip, settings.sip_host, settings.sip_port = True, "127.0.0.1", 0
    settings.sip_rtp_port_min, settings.sip_rtp_port_max = 41000, 41200
    settings.telephony = {
        "agents": {"desk": {"greeting": "Hello.", "silence_timeout_s": 0, **agent}},
        "numbers": {"+15550100002": "desk"},
    }
    server = LiveServer(settings)
    server.start()
    return server


def calls(server: LiveServer) -> list[dict[str, Any]]:
    listed = httpx.get(server.url + "/api/v1/telephony/calls").json()["calls"]
    return [httpx.get(f"{server.url}/api/v1/telephony/calls/{c['id']}").json() for c in listed]


def ended(server: LiveServer, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = calls(server)
        if found and found[0]["status"] == "ended":
            return found[0]
        time.sleep(0.05)
    raise AssertionError(f"the call did not end: {calls(server)}")


def test_sdp_offer_and_answer() -> None:
    offer = parse_offer(OFFER.format(port=4000, types="8 0").encode())
    assert offer == Offer("127.0.0.1", 4000, 8, 101)  # the caller's first choice among PCMU and PCMA
    answer = build_answer("192.0.2.7", 40002, offer).decode()
    assert "c=IN IP4 192.0.2.7\r\n" in answer and "m=audio 40002 RTP/AVP 8 101\r\n" in answer
    assert "a=rtpmap:8 PCMA/8000\r\n" in answer and "a=rtpmap:101 telephone-event/8000\r\n" in answer
    assert parse_offer(build_answer("192.0.2.7", 40002, offer)) == Offer("192.0.2.7", 40002, 8, 101)
    # media-level address, no telephone-event, a video stream after the audio
    sdp = b"v=0\r\no=- 1 1 IN IP4 10.0.0.1\r\ns=-\r\nt=0 0\r\nm=audio 5004 RTP/AVP 18 0\r\nc=IN IP4 10.0.0.9\r\nm=video 6000 RTP/AVP 96\r\nc=IN IP4 10.0.0.10\r\n"
    assert parse_offer(sdp) == Offer("10.0.0.9", 5004, 0, None)
    assert parse_offer(b"v=0\r\nc=IN IP4 10.0.0.1\r\nm=audio 5004 RTP/AVP 9 18\r\n") is None  # G.722 and G.729 only
    assert parse_offer(b"v=0\r\nc=IN IP4 10.0.0.1\r\nm=audio 5004 RTP/SAVP 0\r\n") is None  # SRTP is not offered back


async def test_a_sip_call_with_audio_key_presses_and_bye(settings: Settings, llm: FakeLLM) -> None:
    server = sip_server(settings, llm)
    phone = Phone(server.core.sip.port)
    try:
        await phone.send(phone.request("OPTIONS", cseq=99))
        options = await phone_response_after_loops(phone, 200, "OPTIONS")
        assert "INVITE" in options.headers["Allow"] and options.server.startswith("bolo/")
        ok = await phone.answered()
        assert [r.code for r in phone.responses if r.cseq.method == "INVITE"][:2] == [100, 200]
        assert ok.headers["Content-Type"] == "application/sdp" and ok.contact[0].uri.port == server.core.sip.port
        assert b"m=audio" in ok.body and b"RTP/AVP 0 101" in ok.body and phone.payload_type == 0
        await phone.talk(np.zeros(0, dtype=np.float32), 0.8)  # listen to the greeting
        await phone.talk(speechlike(1.6), 1.2)
        await phone.press("4")
        await phone.press("#")
        await phone.talk(np.zeros(0, dtype=np.float32), 1.5)
        # audio came back as 20 ms packets at the pace of a phone line, the greeting and then the answers
        assert all(len(p) == 160 for p in phone.received) and phone.markers >= 2
        assert 2.0 < phone.heard_s() < 6.0 and len(phone.received) / 50 == pytest.approx(5.5, abs=1.0)
        tone = mulaw_decode(b"".join(p for p in phone.received if p.strip(b"\xff")))
        assert float(np.abs(tone).max()) > 0.1
        await phone.send(phone.request("BYE"))
        assert (await phone.response(200, "BYE")).call_id == phone.call_id
    finally:
        await phone.close()
        record = ended(server)
        server.stop()
    assert record["provider"] == "sip" and record["agent"] == "desk" and record["outcome"] == "caller_hangup"
    assert (
        record["from"] == "+15550100001"
        and record["to"] == "+15550100002"
        and record["provider_call_id"] == phone.call_id
    )
    assert record["dtmf"] == "4#" and record["timeline"][1] == {
        **record["timeline"][1],
        "encoding": "mulaw",
        "sample_rate": 8000,
    }
    texts = [(e["speaker"], e["text"]) for e in record["transcript"]]
    assert texts[0] == ("agent", "Hello.") and texts[1][0] == "caller" and texts[1][1].startswith("the quick brown")
    assert ("caller", "[The caller pressed 4 # on the phone keypad]") in texts
    assert [t["kind"] for t in record["turns"]] == ["speech", "dtmf"] and record["turns"][0]["latency"]["total_ms"] > 0


async def phone_response_after_loops(phone: Phone, code: int, method: str) -> Any:
    await phone.invite()
    return await phone.response(code, method)


async def test_the_agent_hangs_up_with_bye_and_pcma_is_negotiated(settings: Settings, llm: FakeLLM) -> None:
    server = sip_server(settings, llm, max_duration_s=1.2, max_duration_message="Time is up.")
    phone = Phone(server.core.sip.port, types="8")
    try:
        await phone.invite()
        ok = await phone.answered()
        assert b"RTP/AVP 8 101" in ok.body and b"PCMA/8000" in ok.body and phone.payload_type == 8
        await phone.talk(np.zeros(0, dtype=np.float32), 3.5)
        byes = [r for r in phone.requests if r.method == "BYE"]
        assert byes and byes[0].call_id == phone.call_id and byes[0].cseq.method == "BYE"
        # inside the dialog: our tag is the From tag, the caller's is the To tag
        assert (
            byes[0].from_address.parameters["tag"] == phone.to_tag and byes[0].to_address.parameters["tag"] == "caller1"
        )
        assert phone.heard_s() > 0.8
    finally:
        await phone.close()
        record = ended(server)
        server.stop()
    assert record["outcome"] == "max_duration" and record["timeline"][1]["encoding"] == "alaw"
    assert [e["text"] for e in record["transcript"]] == ["Hello.", "Time is up."]


async def test_sip_refusals(settings: Settings, llm: FakeLLM) -> None:
    server = sip_server(settings, llm)
    endpoint = server.core.sip
    phone = Phone(endpoint.port)
    try:
        await phone.invite("+19995550000")  # no agent answers this number and there is no default
        assert (await phone.response(404)).code == 404
        phone.call_id, phone.responses = "late-offer@127.0.0.1", []
        await phone.send(phone.request("INVITE"))  # no SDP offer
        assert (await phone.response(488)).code == 488
        phone.responses = []
        await phone.send(phone.request("REGISTER", cseq=7))
        assert (await phone.response(405, "REGISTER")).headers["Allow"].startswith("INVITE")
        # callers are admitted by address: with a list that does not include loopback, nothing is answered but 403
        import ipaddress

        endpoint._networks = [ipaddress.ip_network("203.0.113.0/24")]
        assert endpoint.allowed("203.0.113.9") and not endpoint.allowed("127.0.0.1")
        phone.call_id, phone.responses = "stranger@127.0.0.1", []
        await phone.send(phone.request("INVITE", body=OFFER.format(port=4000, types="0").encode()))
        assert (await phone.response(403)).code == 403
        assert [c["outcome"] for c in calls(server)] == []
        info = httpx.get(server.url + "/api/v1/info").json()["capabilities"]["telephony"]["sip"]
        assert info == {"enabled": True, "listening": True, "port": endpoint.port}
    finally:
        await phone.close()
        server.stop()
