"""A minimal inbound SIP endpoint: calls from a SIP trunk or PBX, straight into a voice agent.

What it does (and no more): it listens on one UDP port, accepts ``INVITE`` from allowed addresses, answers
with G.711 (PCMU or PCMA, 8 kHz) over plain RTP, reads key presses sent as RTP telephone events
(RFC 4733, formerly RFC 2833), handles ``ACK``, ``BYE``, ``CANCEL``, ``OPTIONS`` and retransmissions, and
sends ``BYE`` when the agent hangs up. The call itself is the same :class:`~bolo.telephony.session.CallSession`
the provider media streams use; audio leaves in 20 ms RTP packets paced by this module, so barge-in drops the
queue here instead of asking a provider to.

What it does not do: REGISTER, authentication by digest (callers are admitted by source address:
``sip_allowed_ips``), TCP or TLS, SRTP, re-negotiation of media, an INVITE without an SDP offer, call transfer
(REFER), outbound calls. See ``docs/telephony.md`` for the design and its limits.

SIP messages are parsed and written with `sipmessage <https://github.com/spacinov/sipmessage>`_ (BSD-2-Clause),
installed with the ``sip`` extra; dialogs, transactions, SDP and RTP are implemented here.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import re
import secrets
import socket
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sipmessage import URI, Address, CSeq, Message, Request, Response, Via

from bolo import __version__
from bolo.telephony.adapters import (
    Adapter,
    StreamAudio,
    StreamDtmf,
    StreamEvent,
    StreamMark,
    StreamStart,
    Wire,
)

if TYPE_CHECKING:
    from bolo.telephony.manager import PendingCall, TelephonyManager
    from bolo.telephony.session import CallSession

log = logging.getLogger("bolo.sip")

T1 = 0.5  # SIP retransmission base interval (RFC 3261)
FRAME = 160  # 20 ms of G.711
CODECS = {0: ("PCMU", "mulaw"), 8: ("PCMA", "alaw")}
SILENCE = {"mulaw": b"\xff" * FRAME, "alaw": b"\xd5" * FRAME}
DTMF_EVENTS = "0123456789*#ABCD"
RTP_TIMEOUT_S = 60.0
ALLOW = "INVITE, ACK, BYE, CANCEL, OPTIONS"


class SipWire(Wire):
    """Frames between the call session and the RTP sender of this process: a kind letter and a payload."""

    def __init__(self, encoding: str = "mulaw") -> None:
        super().__init__()
        self.encoding = encoding

    def _parse(self, doc: dict[str, Any]) -> list[StreamEvent]:
        return []

    def media(self, payload: bytes) -> str:
        return "A" + payload.decode("latin-1")

    def clear(self) -> str | None:
        return "C"

    def mark(self, name: str) -> str | None:
        return "M" + name


class SipAdapter(Adapter):
    name = "sip"
    label = "SIP"
    ends_by = "api"
    protection = "allowlist"
    can_transfer = False

    def __init__(self, manager: TelephonyManager, endpoint: SipEndpoint) -> None:
        super().__init__(manager)
        self.endpoint = endpoint
        self.encoding = "mulaw"

    def wire(self) -> Wire:
        return SipWire(self.encoding)

    async def hangup(self, pending: PendingCall) -> None:
        call = self.endpoint.calls.get(str(pending.record.provider_call_id))
        if call is not None:
            self.endpoint.bye(call)


@dataclass
class Offer:
    host: str
    port: int
    payload_type: int
    event_type: int | None


def parse_offer(body: bytes) -> Offer | None:
    """The first audio stream of an SDP offer, when it carries PCMU or PCMA."""
    text = body.decode(errors="replace")
    host: str | None = None
    port = 0
    types: list[int] = []
    event_type: int | None = None
    in_audio = False
    seen_media = False
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("m="):
            if in_audio:
                break  # only the first audio stream
            parts = line[2:].split()
            in_audio = len(parts) >= 4 and parts[0] == "audio" and parts[2].upper().startswith("RTP/AVP")
            seen_media = True
            if in_audio:
                port = int(parts[1]) if parts[1].split("/")[0].isdigit() else 0
                types = [int(p) for p in parts[3:] if p.isdigit()]
        elif line.startswith("c=") and (in_audio or not seen_media):
            parts = line[2:].split()
            if len(parts) >= 3:
                host = parts[2].split("/")[0]
        elif in_audio and line.startswith("a=rtpmap:"):
            found = re.match(r"a=rtpmap:(\d+)\s+telephone-event/8000", line, re.IGNORECASE)
            if found:
                event_type = int(found.group(1))
    chosen = next((t for t in types if t in CODECS), None)
    if chosen is None or not host or not port:
        return None
    return Offer(host, port, chosen, event_type if event_type in types else None)


def build_answer(address: str, port: int, offer: Offer) -> bytes:
    name = CODECS[offer.payload_type][0]
    session = int(time.time())
    types = str(offer.payload_type) + (f" {offer.event_type}" if offer.event_type is not None else "")
    lines = [
        "v=0",
        f"o=bolo {session} {session} IN IP4 {address}",
        "s=bolo",
        f"c=IN IP4 {address}",
        "t=0 0",
        f"m=audio {port} RTP/AVP {types}",
        f"a=rtpmap:{offer.payload_type} {name}/8000",
    ]
    if offer.event_type is not None:
        lines += [f"a=rtpmap:{offer.event_type} telephone-event/8000", f"a=fmtp:{offer.event_type} 0-16"]
    lines += ["a=ptime:20", "a=sendrecv"]
    return ("\r\n".join(lines) + "\r\n").encode()


class RtpStream(asyncio.DatagramProtocol):
    """One call's RTP: packets in become audio and key presses, a paced queue of frames goes out."""

    def __init__(self, call: SipCall) -> None:
        self.call = call
        self.transport: asyncio.DatagramTransport | None = None
        self.remote: tuple[str, int] = (call.offer.host, call.offer.port)
        self.queue: deque[bytes | str] = deque()
        self.sequence = secrets.randbelow(1 << 15)
        self.timestamp = secrets.randbelow(1 << 30)
        self.ssrc = secrets.randbits(32)
        self.last_received = time.monotonic()
        self.packets_in = 0
        self.packets_out = 0
        self._talking = False
        self._last_event: int | None = None

    def itself(self) -> RtpStream:
        return self

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        if len(data) < 12 or data[0] >> 6 != 2:
            return
        self.packets_in += 1
        self.last_received = time.monotonic()
        self.remote = addr  # symmetric RTP: answer where the audio comes from (NAT)
        header = 12 + 4 * (data[0] & 0x0F)
        if data[0] & 0x10 and len(data) >= header + 4:  # header extension
            header += 4 + 4 * struct.unpack("!H", data[header + 2 : header + 4])[0]
        payload = data[header:]
        if data[0] & 0x20 and payload:  # padding
            payload = payload[: -payload[-1]]
        payload_type = data[1] & 0x7F
        offer = self.call.offer
        if payload_type == offer.payload_type:
            self.call.events.put_nowait(StreamAudio(bytes(payload)))
        elif payload_type == offer.event_type and len(payload) >= 4:
            # RFC 4733: one event is sent many times with the same timestamp; report it once
            stamp = struct.unpack("!I", data[4:8])[0]
            if stamp != self._last_event and payload[0] < len(DTMF_EVENTS):
                self._last_event = stamp
                self.call.events.put_nowait(StreamDtmf(DTMF_EVENTS[payload[0]]))

    def push(self, frame: str) -> None:
        """A frame from the call session: "A" + audio, "C" (clear the queue) or "M" + mark name."""
        if frame[:1] == "A":
            data = frame[1:].encode("latin-1")
            self.queue.extend(data[i : i + FRAME] for i in range(0, len(data), FRAME))
        elif frame[:1] == "C":
            self.queue.clear()
        elif frame[:1] == "M":
            self.queue.append(frame[1:])

    async def run(self) -> None:
        """Send one packet every 20 ms: queued audio, or silence that keeps the path (and NAT bindings) open."""
        silence = SILENCE[self.call.encoding]
        started = time.perf_counter()
        tick = 0
        while not self.call.ended:
            payload: bytes | None = None
            while self.queue and payload is None:
                item = self.queue.popleft()
                if isinstance(item, str):
                    self.call.events.put_nowait(StreamMark(item))
                else:
                    payload = item.ljust(FRAME, silence[:1])
            marker = payload is not None and not self._talking
            self._talking = payload is not None
            self._send(payload or silence, marker)
            tick += 1
            await asyncio.sleep(max(0.0, started + tick * 0.02 - time.perf_counter()))

    def _send(self, payload: bytes, marker: bool) -> None:
        if self.transport is None:
            return
        self.sequence = (self.sequence + 1) & 0xFFFF
        self.timestamp = (self.timestamp + FRAME) & 0xFFFFFFFF
        first = (0x80 if marker else 0) | self.call.offer.payload_type
        header = struct.pack("!BBHII", 0x80, first, self.sequence, self.timestamp, self.ssrc)
        with contextlib.suppress(OSError):
            self.transport.sendto(header + payload, self.remote)
            self.packets_out += 1


@dataclass
class SipCall:
    call_id: str
    invite: Request
    source: tuple[str, int]
    offer: Offer
    local_tag: str
    encoding: str
    events: asyncio.Queue[StreamEvent | None] = field(default_factory=asyncio.Queue)
    rtp: RtpStream | None = None
    session: CallSession | None = None
    pending: PendingCall | None = None
    answer: bytes = b""  # the 200 OK, resent until the ACK arrives
    acked: bool = False
    ended: bool = False
    cseq: int = 0
    tasks: list[asyncio.Task[Any]] = field(default_factory=list)


class SipEndpoint(asyncio.DatagramProtocol):
    def __init__(self, manager: TelephonyManager) -> None:
        self.manager = manager
        self.core = manager.core
        s = self.core.settings
        self.host, self.port = s.sip_host, s.sip_port
        self.calls: dict[str, SipCall] = {}
        self.transport: asyncio.DatagramTransport | None = None
        self.adapter = SipAdapter(manager, self)
        self._networks = [ipaddress.ip_network(item, strict=False) for item in s.sip_allowed_ips]
        self._next_port = s.sip_rtp_port_min
        self._tasks: set[asyncio.Task[Any]] = set()

    # -- lifecycle ------------------------------------------------------------------------------------------
    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: self, local_addr=(self.host, self.port))
        assert self.transport is not None
        self.port = self.transport.get_extra_info("sockname")[1]
        self.manager.adapters["sip"] = self.adapter

    async def stop(self) -> None:
        for call in list(self.calls.values()):
            self.bye(call)
            await self._end(call, "error", "the server is shutting down")
        for task in list(self._tasks):
            task.cancel()
        if self.transport is not None:
            self.transport.close()

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def _spawn(self, coro: Any, call: SipCall | None = None) -> asyncio.Task[Any]:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if call is not None:
            call.tasks.append(task)
        return task

    def allowed(self, address: str) -> bool:
        ip = ipaddress.ip_address(address)
        if not self._networks:
            return ip.is_loopback
        return any(ip in network for network in self._networks)

    # -- SIP in ---------------------------------------------------------------------------------------------
    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        if len(data) < 12 or not data.strip():
            return  # keep-alives (CRLF)
        try:
            message = Message.parse(data)
        except (ValueError, KeyError):
            return
        try:
            if isinstance(message, Response):
                return  # answers to our BYE need nothing more
            if not self.allowed(addr[0]):
                log.warning("SIP %s from %s refused: the address is not in sip_allowed_ips", message.method, addr[0])
                if message.method != "ACK":
                    self._reply(message, addr, 403, "Forbidden")
                return
            self._on_request(message, addr)
        except (ValueError, KeyError, IndexError) as exc:
            log.warning("could not handle a SIP message from %s: %s", addr[0], exc)

    def _on_request(self, request: Request, addr: tuple[str, int]) -> None:
        method = request.method
        call = self.calls.get(request.call_id)
        if method == "OPTIONS":
            self._reply(request, addr, 200, "OK", allow=True)
        elif method == "INVITE":
            if call is not None:
                if request.cseq.sequence == call.invite.cseq.sequence:
                    self._send(call.answer, addr)  # a retransmission: the same answer again
                else:
                    self._reply(request, addr, 488, "Not Acceptable Here", tag=call.local_tag)  # no re-negotiation
                return
            self._invite(request, addr)
        elif method == "ACK":
            if call is not None:
                call.acked = True
        elif method == "BYE":
            self._reply(request, addr, 200, "OK")
            if call is not None:
                self._spawn(self._end(call, "caller_hangup"))
        elif method == "CANCEL":
            self._reply(request, addr, 200, "OK")
            if call is not None and not call.acked:
                self._spawn(self._end(call, "canceled"))
        else:
            self._reply(request, addr, 405, "Method Not Allowed", allow=True)

    def _reply(
        self, request: Request, addr: tuple[str, int], code: int, phrase: str, *, body: bytes = b"",
        tag: str | None = None, allow: bool = False, contact: bool = False,
    ) -> bytes:  # fmt: skip
        response = Response(code, phrase, body)
        response.via = request.via
        response.from_address = request.from_address
        to = request.to_address
        if code != 100 and "tag" not in to.parameters:
            to = Address(to.uri, to.name, to.parameters.replace(tag=tag or secrets.token_hex(6)))
        response.to_address = to
        response.call_id = request.call_id
        response.cseq = request.cseq
        response.server = f"bolo/{__version__}"
        if allow or contact:
            response.headers.set("Allow", ALLOW)
        if contact:
            local = self._local_address(addr)
            response.contact = [Address(URI(scheme="sip", host=local, port=self.port, user=request.uri.user))]
        if body:
            response.headers.set("Content-Type", "application/sdp")
        response.content_length = len(body)
        data = bytes(response)
        self._send(data, addr)
        return data

    def _send(self, data: bytes, addr: tuple[str, int]) -> None:
        if self.transport is not None and data:
            with contextlib.suppress(OSError):
                self.transport.sendto(data, addr)

    def _local_address(self, peer: tuple[str, int]) -> str:
        """The address the peer should send media to: ``sip_public_ip``, or the one this host reaches it from."""
        configured = self.core.settings.sip_public_ip
        if configured:
            return str(configured)
        with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(peer)
            return str(probe.getsockname()[0])
        return "127.0.0.1"

    # -- a call ---------------------------------------------------------------------------------------------
    def _invite(self, request: Request, addr: tuple[str, int]) -> None:
        self._reply(request, addr, 100, "Trying")
        offer = parse_offer(request.body) if request.body else None
        if offer is None:
            self._reply(request, addr, 488, "Not Acceptable Here")
            log.warning("SIP INVITE from %s refused: it needs an SDP offer with PCMU or PCMA audio", addr[0])
            return
        called = request.uri.user or request.to_address.uri.user or ""
        caller = request.from_address.uri.user or request.from_address.uri.host
        pending = self.manager.inbound_call("sip", request.call_id, caller, called)
        if pending is None:
            self._reply(request, addr, 404, "Not Found")
            return
        encoding = CODECS[offer.payload_type][1]
        call = SipCall(request.call_id, request, addr, offer, secrets.token_hex(6), encoding, pending=pending)
        self.calls[call.call_id] = call
        self._spawn(self._answer(call), call)

    async def _answer(self, call: SipCall) -> None:
        loop = asyncio.get_running_loop()
        s = self.core.settings
        span = max(1, s.sip_rtp_port_max - s.sip_rtp_port_min + 1)
        rtp: RtpStream | None = None
        port = 0
        for _ in range(span):
            port = self._next_port
            self._next_port = s.sip_rtp_port_min + (self._next_port + 1 - s.sip_rtp_port_min) % span
            candidate = RtpStream(call)
            try:
                await loop.create_datagram_endpoint(candidate.itself, local_addr=(self.host, port))
            except OSError:
                continue
            rtp = candidate
            break
        if rtp is None:
            self._reply(call.invite, call.source, 503, "Service Unavailable", tag=call.local_tag)
            await self._end(call, "error", "no free RTP port between sip_rtp_port_min and sip_rtp_port_max")
            return
        call.rtp = rtp
        body = build_answer(self._local_address(call.source), port, call.offer)
        call.answer = self._reply(call.invite, call.source, 200, "OK", body=body, tag=call.local_tag, contact=True)
        self.adapter.encoding = call.encoding

        async def send(frame: str) -> None:
            rtp.push(frame)

        async def close() -> None:
            await self._end(call, "agent_hangup")

        assert call.pending is not None
        call.session = self.manager.open_stream("sip", call.pending.token, {}, "", send, close, trusted=True)
        uri = call.invite.from_address.uri
        start = StreamStart(call.call_id, call.call_id, uri.user or uri.host, call.invite.uri.user)
        await call.session.handle_event(start)
        self._spawn(rtp.run(), call)
        self._spawn(self._pump(call), call)
        self._spawn(self._retransmit_answer(call), call)
        self._spawn(self._watch(call), call)

    async def _pump(self, call: SipCall) -> None:
        """Audio and key presses in the order they arrived."""
        while not call.ended and call.session is not None:
            event = await call.events.get()
            if event is None:
                return
            await call.session.handle_event(event)

    async def _retransmit_answer(self, call: SipCall) -> None:
        """A 200 OK to INVITE is repeated until its ACK arrives (RFC 3261, 13.3.1.4)."""
        delay = T1
        deadline = time.monotonic() + 64 * T1
        while not call.acked and not call.ended and time.monotonic() < deadline:
            await asyncio.sleep(delay)
            if not call.acked and not call.ended:
                self._send(call.answer, call.source)
            delay = min(delay * 2, 4.0)
        if not call.acked and not call.ended:
            self.bye(call)
            await self._end(call, "error", "the caller never acknowledged the answer (no ACK)")

    async def _watch(self, call: SipCall) -> None:
        while not call.ended:
            await asyncio.sleep(1.0)
            if call.rtp is not None and time.monotonic() - call.rtp.last_received > RTP_TIMEOUT_S:
                self.bye(call)
                await self._end(call, "error", f"no RTP received for {RTP_TIMEOUT_S:.0f} seconds")
                return

    def bye(self, call: SipCall) -> None:
        """Hang up from this side: a BYE inside the dialog the INVITE created."""
        if call.ended and call.cseq:
            return
        invite = call.invite
        contacts = invite.contact
        target = contacts[0].uri if contacts else invite.from_address.uri
        request = Request("BYE", target)
        branch = "z9hG4bK" + secrets.token_hex(8)
        local = self._local_address(call.source)
        request.via = [Via.parse(f"SIP/2.0/UDP {local}:{self.port};branch={branch};rport")]
        request.max_forwards = 70
        to = invite.to_address
        request.from_address = Address(to.uri, to.name, to.parameters.replace(tag=call.local_tag))
        request.to_address = invite.from_address
        request.call_id = call.call_id
        call.cseq += 1
        request.cseq = CSeq(call.cseq, "BYE")
        routes = invite.record_route
        if routes:
            request.route = routes
        request.user_agent = f"bolo/{__version__}"
        request.content_length = 0
        data = bytes(request)
        self._send(data, call.source)
        # no transaction layer: two repeats cover a lost datagram; a duplicate BYE is answered 481 and ignored
        loop = asyncio.get_running_loop()
        loop.call_later(T1, self._send, data, call.source)
        loop.call_later(4 * T1, self._send, data, call.source)

    async def _end(self, call: SipCall, outcome: str, error: str | None = None) -> None:
        if call.ended:
            return
        call.ended = True
        self.calls.pop(call.call_id, None)
        call.events.put_nowait(None)
        current = asyncio.current_task()
        for task in call.tasks:
            if task is not current and not task.done():
                task.cancel()
        if call.rtp is not None and call.rtp.transport is not None:
            call.rtp.transport.close()
        if call.session is not None:
            await call.session.finalize(outcome, error)
        elif call.pending is not None:
            self.manager.close_record(call.pending, outcome, error)


async def start_sip(manager: TelephonyManager) -> SipEndpoint:
    endpoint = SipEndpoint(manager)
    await endpoint.start()
    return endpoint
