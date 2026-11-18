"""A call simulator: a scripted caller behind a fake telephony provider.

It exists to make calls testable without a telephony account. A :class:`Caller` plays a scenario (speech from
text-to-speech, from a file or synthetic; silences; key presses; interruptions; background noise) as the 8 kHz
audio a provider would send, in real time, and listens the way a phone does: audio the agent sends is "played"
at its real speed, ``clear`` empties what has not been played, marks come back when playback reaches them. From
that it scores what a caller experiences: how long after the end of each utterance the agent was heard, whether
barge-in cut the agent off, and whether the expected behaviours happened.

The caller reaches the agent through a *link*:

* :class:`DirectLink`: inside the server process, straight into a call session (``POST
  /api/v1/telephony/simulate``; no network);
* :class:`WebLink`: the whole path a provider takes: the signed answer webhook, the document it returns, the
  WebSocket media stream, and the ``after`` webhook when bolo hands the call back (the test-suite and
  ``bolo telephony simulate``).

:class:`FakeProviderAPI` is the other half of a fake provider: a small HTTP server that accepts the REST calls
bolo makes (placing a call, Telnyx commands) and records them.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import http.server
import json
import os
import re
import secrets
import threading
import time
import urllib.parse
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np

from bolo.audio.codecs import encode_frame
from bolo.audio.resample import resample
from bolo.engines.base import FloatArray
from bolo.errors import InvalidRequest
from bolo.telephony import security
from bolo.telephony.config import AgentConfig

if TYPE_CHECKING:
    from bolo.core import Bolo

SCENARIOS: dict[str, dict[str, Any]] = {
    "basic": {
        "description": "The caller waits for the greeting, says one thing, hears the answer and hangs up.",
        "steps": [{"wait_for_agent": True}, {"speech_s": 2.0}, {"wait_for_agent": True}, {"hangup": True}],
        "expect": [
            {"type": "agent_spoke"},
            {"type": "turns_min", "value": 1},
            {"type": "outcome", "is": "caller_hangup"},
        ],
    },
    "barge-in": {
        "description": "The caller interrupts the agent's answer; the queued audio must be cleared.",
        "steps": [
            {"wait_for_agent": True},
            {"speech_s": 2.0},
            {"interrupt": {"after_s": 1.0, "speech_s": 1.5}},
            {"wait_for_agent": True},
            {"hangup": True},
        ],
        "expect": [{"type": "barge_in"}, {"type": "turns_min", "value": 2}],
    },
    "silence": {
        "description": "The caller says nothing: the agent must ask whether they are there, then hang up.",
        "steps": [{"wait_for_end": True}],
        "expect": [{"type": "timeline", "event": "silence_prompt"}, {"type": "outcome", "is": "no_response"}],
    },
    "dtmf": {
        "description": "The caller presses keys; the agent must receive them and answer.",
        "steps": [{"wait_for_agent": True}, {"dtmf": "42#"}, {"wait_for_agent": True}, {"hangup": True}],
        "expect": [{"type": "timeline", "event": "dtmf"}, {"type": "turns_min", "value": 1}],
    },
    "noisy": {
        "description": "The basic call with background noise at 20 dB signal-to-noise ratio.",
        "noise_db": 20,
        "steps": [{"wait_for_agent": True}, {"speech_s": 2.0}, {"wait_for_agent": True}, {"hangup": True}],
        "expect": [{"type": "turns_min", "value": 1}],
    },
}
SPEECH_RMS = 0.15
CALLER, CALLED = "+15550100001", "+15550100002"


def speechlike(seconds: float, sample_rate: int = 8000) -> FloatArray:
    """A loud modulated tone: what voice-activity detectors without a model take for speech."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    out: FloatArray = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))).astype(np.float32)
    return out


# -- the provider's side of each media-stream protocol ------------------------------------------------------
class ProviderSim:
    """Builds the frames a provider sends and reads the ones bolo sends back."""

    name = ""
    encoding = "mulaw"
    rate = 8000
    frame_ms = 20

    def __init__(self) -> None:
        self.stream_id = "MZ" + secrets.token_hex(8)
        self.call_id = "CA" + secrets.token_hex(8)
        self.seq = 0
        self.chunk = 0

    def _n(self) -> int:
        self.seq += 1
        return self.seq

    def opening(self, from_number: str, to_number: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def media(self, payload: bytes, timestamp_ms: int) -> dict[str, Any]:
        raise NotImplementedError

    def dtmf(self, digit: str) -> dict[str, Any]:
        raise NotImplementedError

    def mark(self, name: str) -> dict[str, Any]:
        raise NotImplementedError

    def stop(self) -> dict[str, Any] | None:
        raise NotImplementedError

    def parse(self, doc: dict[str, Any]) -> tuple[str, Any]:
        """("media", bytes), ("clear", None), ("mark", name) or ("other", None)."""
        raise NotImplementedError


class TwilioSim(ProviderSim):
    name = "twilio"

    def opening(self, from_number: str, to_number: str) -> list[dict[str, Any]]:
        start = {
            "accountSid": "ACsim", "streamSid": self.stream_id, "callSid": self.call_id, "tracks": ["inbound"],
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}, "customParameters": {},
        }  # fmt: skip
        return [
            {"event": "connected", "protocol": "Call", "version": "1.0.0"},
            {"event": "start", "sequenceNumber": str(self._n()), "start": start, "streamSid": self.stream_id},
        ]

    def media(self, payload: bytes, timestamp_ms: int) -> dict[str, Any]:
        self.chunk += 1
        media = {
            "track": "inbound", "chunk": str(self.chunk), "timestamp": str(timestamp_ms),
            "payload": base64.b64encode(payload).decode(),
        }  # fmt: skip
        return {"event": "media", "sequenceNumber": str(self._n()), "streamSid": self.stream_id, "media": media}

    def dtmf(self, digit: str) -> dict[str, Any]:
        return {
            "event": "dtmf", "streamSid": self.stream_id, "sequenceNumber": str(self._n()),
            "dtmf": {"track": "inbound_track", "digit": digit},
        }  # fmt: skip

    def mark(self, name: str) -> dict[str, Any]:
        return {"event": "mark", "streamSid": self.stream_id, "sequenceNumber": str(self._n()), "mark": {"name": name}}

    def stop(self) -> dict[str, Any] | None:
        return {
            "event": "stop", "sequenceNumber": str(self._n()), "streamSid": self.stream_id,
            "stop": {"accountSid": "ACsim", "callSid": self.call_id},
        }  # fmt: skip

    def parse(self, doc: dict[str, Any]) -> tuple[str, Any]:
        kind = doc.get("event")
        if kind == "media" and doc.get("streamSid") == self.stream_id:
            return "media", base64.b64decode(doc["media"]["payload"])
        if kind == "clear" and doc.get("streamSid") == self.stream_id:
            return "clear", None
        if kind == "mark":
            return "mark", doc["mark"]["name"]
        return "other", None


class PlivoSim(ProviderSim):
    name = "plivo"

    def opening(self, from_number: str, to_number: str) -> list[dict[str, Any]]:
        start = {
            "callId": self.call_id, "streamId": self.stream_id, "accountId": "MAsim", "tracks": ["inbound"],
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000},
        }  # fmt: skip
        return [{"event": "start", "sequenceNumber": self._n(), "start": start, "extra_headers": ""}]

    def media(self, payload: bytes, timestamp_ms: int) -> dict[str, Any]:
        self.chunk += 1
        media = {
            "track": "inbound", "timestamp": str(timestamp_ms), "chunk": self.chunk,
            "payload": base64.b64encode(payload).decode(),
        }  # fmt: skip
        return {
            "event": "media",
            "sequenceNumber": self._n(),
            "streamId": self.stream_id,
            "media": media,
            "extra_headers": "",
        }

    def dtmf(self, digit: str) -> dict[str, Any]:
        return {
            "event": "dtmf", "sequenceNumber": self._n(), "streamId": self.stream_id,
            "dtmf": {"track": "inbound", "digit": digit, "timestamp": "0"}, "extra_headers": "",
        }  # fmt: skip

    def mark(self, name: str) -> dict[str, Any]:
        return {"event": "playedStream", "sequenceNumber": self._n(), "streamId": self.stream_id, "name": name}

    def stop(self) -> dict[str, Any] | None:
        return None  # Plivo closes the socket; its "stopped" notice goes to the status callback

    def parse(self, doc: dict[str, Any]) -> tuple[str, Any]:
        kind = doc.get("event")
        if kind == "playAudio":
            media = doc["media"]
            if media.get("contentType") != "audio/x-mulaw" or int(media.get("sampleRate") or 0) != 8000:
                raise ValueError("playAudio must match the stream's contentType and sampleRate")
            return "media", base64.b64decode(media["payload"])
        if kind == "clearAudio" and doc.get("streamId") == self.stream_id:
            return "clear", None
        if kind == "checkpoint" and doc.get("streamId") == self.stream_id:
            return "mark", doc["name"]
        return "other", None


class TelnyxSim(ProviderSim):
    name = "telnyx"

    def opening(self, from_number: str, to_number: str) -> list[dict[str, Any]]:
        start: dict[str, Any] = {
            "user_id": "sim", "call_control_id": self.call_id, "call_session_id": "sess-" + self.call_id,
            "from": from_number, "to": to_number, "tags": [], "client_state": None,
            "media_format": {"encoding": "PCMU", "sample_rate": 8000, "channels": 1},
        }  # fmt: skip
        return [
            {"event": "connected", "version": "1.0.0"},
            {"event": "start", "sequence_number": str(self._n()), "start": start, "stream_id": self.stream_id},
        ]

    def media(self, payload: bytes, timestamp_ms: int) -> dict[str, Any]:
        self.chunk += 1
        media = {
            "track": "inbound", "chunk": str(self.chunk), "timestamp": str(timestamp_ms),
            "payload": base64.b64encode(payload).decode(),
        }  # fmt: skip
        return {"event": "media", "sequence_number": str(self._n()), "media": media, "stream_id": self.stream_id}

    def dtmf(self, digit: str) -> dict[str, Any]:
        return {
            "event": "dtmf", "stream_id": self.stream_id, "occurred_at": "2026-01-01T00:00:00.000000Z",
            "sequence_number": str(self._n()), "dtmf": {"digit": digit},
        }  # fmt: skip

    def mark(self, name: str) -> dict[str, Any]:
        return {"event": "mark", "stream_id": self.stream_id, "sequence_number": str(self._n()), "mark": {"name": name}}

    def stop(self) -> dict[str, Any] | None:
        return {
            "event": "stop", "sequence_number": str(self._n()), "stream_id": self.stream_id,
            "stop": {"user_id": "sim", "call_control_id": self.call_id},
        }  # fmt: skip

    def parse(self, doc: dict[str, Any]) -> tuple[str, Any]:
        kind = doc.get("event")
        if kind == "media":
            return "media", base64.b64decode(doc["media"]["payload"])
        if kind == "clear":
            return "clear", None
        if kind == "mark":
            return "mark", doc["mark"]["name"]
        return "other", None


class ExotelSim(ProviderSim):
    name = "exotel"
    encoding = "pcm16"
    frame_ms = 100

    def opening(self, from_number: str, to_number: str) -> list[dict[str, Any]]:
        start = {
            "stream_sid": self.stream_id, "call_sid": self.call_id, "account_sid": "sim", "from": from_number,
            "to": to_number, "custom_parameters": {},
            "media_format": {"encoding": "raw", "sample_rate": "8000", "bit_rate": "128"},
        }  # fmt: skip
        return [
            {"event": "connected"},
            {"event": "start", "sequence_number": self._n(), "stream_sid": self.stream_id, "start": start},
        ]

    def media(self, payload: bytes, timestamp_ms: int) -> dict[str, Any]:
        self.chunk += 1
        media = {"chunk": self.chunk, "timestamp": str(timestamp_ms), "payload": base64.b64encode(payload).decode()}
        return {"event": "media", "sequence_number": self._n(), "stream_sid": self.stream_id, "media": media}

    def dtmf(self, digit: str) -> dict[str, Any]:
        return {
            "event": "dtmf", "sequence_number": self._n(), "stream_sid": self.stream_id,
            "dtmf": {"duration": "200", "digit": digit},
        }  # fmt: skip

    def mark(self, name: str) -> dict[str, Any]:
        return {"event": "mark", "sequence_number": self._n(), "stream_sid": self.stream_id, "mark": {"name": name}}

    def stop(self) -> dict[str, Any] | None:
        return {
            "event": "stop", "sequence_number": self._n(), "stream_sid": self.stream_id,
            "stop": {"call_sid": self.call_id, "account_sid": "sim", "reason": "callended"},
        }  # fmt: skip

    def parse(self, doc: dict[str, Any]) -> tuple[str, Any]:
        kind = doc.get("event")
        if doc.get("stream_sid") != self.stream_id:
            return "other", None
        if kind == "media":
            payload = base64.b64decode(doc["media"]["payload"])
            if len(payload) % 320 or len(payload) < 3200:
                raise ValueError(f"Exotel media must be at least 3200 bytes in multiples of 320, got {len(payload)}")
            return "media", payload
        if kind == "clear":
            return "clear", None
        if kind == "mark":
            return "mark", doc["mark"]["name"]
        return "other", None


SIMS: dict[str, type[ProviderSim]] = {"twilio": TwilioSim, "plivo": PlivoSim, "telnyx": TelnyxSim, "exotel": ExotelSim}


# -- links ------------------------------------------------------------------------------------------------
class Link:
    """How the simulated provider reaches bolo."""

    async def connect(self, sim: ProviderSim, from_number: str, to_number: str) -> None:
        raise NotImplementedError

    async def send(self, doc: dict[str, Any]) -> None:
        raise NotImplementedError

    async def receive(self) -> str | None:
        """The next frame from bolo; None once bolo has closed the stream."""
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError

    async def after(self, sim: ProviderSim) -> str | None:
        """What bolo tells the provider to do once the stream is over: "hangup", "dial:<number>" or None."""
        return None

    async def record(self, sim: ProviderSim) -> dict[str, Any] | None:
        raise NotImplementedError


class DirectLink(Link):
    """Straight into a call session of this process."""

    def __init__(self, core: Bolo, agent: AgentConfig) -> None:
        self.core = core
        self.agent = agent
        self._out: asyncio.Queue[str | None] = asyncio.Queue()
        self._session: Any = None
        self._record: Any = None

    async def connect(self, sim: ProviderSim, from_number: str, to_number: str) -> None:
        from bolo.telephony.records import CallRecord

        manager = self.core.telephony
        record = CallRecord(sim.name, "inbound", self.agent.name, from_number, to_number, sim.call_id)
        record.metadata["simulated"] = True
        record.event("ringing")
        pending = manager._new_pending(sim.name, self.agent, record)
        self._record = record

        async def send(text: str) -> None:
            await self._out.put(text)

        async def close() -> None:
            await self._out.put(None)

        self._session = manager.open_stream(sim.name, pending.token, {}, "", send, close, trusted=True)

    async def send(self, doc: dict[str, Any]) -> None:
        await self._session.handle_message(json.dumps(doc))

    async def receive(self) -> str | None:
        return await self._out.get()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.finalize("caller_hangup")
        await self._out.put(None)

    async def after(self, sim: ProviderSim) -> str | None:
        to = self._record.transfer_to if self._record is not None else None
        return f"dial:{to}" if to else "hangup"

    async def record(self, sim: ProviderSim) -> dict[str, Any] | None:
        return None if self._record is None else dict(self._record.to_dict())


class WebLink(Link):
    """The path a real provider takes: signed webhook, answer document, WebSocket, ``after`` webhook.

    ``credentials`` are the secrets the fake provider signs with (they must match the server's telephony
    configuration): ``auth_token`` for Twilio and Plivo, ``seed`` (32 bytes, the Ed25519 private key whose public
    key the server has) for Telnyx, ``stream_username`` / ``stream_password`` for Exotel. ``api`` is the
    :class:`FakeProviderAPI` the server's Telnyx adapter is pointed at.
    """

    def __init__(
        self, base_url: str, credentials: dict[str, Any], *, api: FakeProviderAPI | None = None,
        api_key: str | None = None, public_base_url: str | None = None,
    ) -> None:  # fmt: skip
        self.base_url = base_url.rstrip("/")
        self.public = (public_base_url or base_url).rstrip("/")
        self.credentials = credentials
        self.api = api
        self.api_key = api_key
        self.token: str | None = None
        self.answer: str | None = None
        self._ws: Any = None
        #: set to skip the answer webhook (an outbound call: bolo already said where the stream is)
        self.stream_url: str | None = None

    async def webhook(
        self, sim: ProviderSim, hook: str, params: dict[str, str], token: str | None = None, method: str = "POST"
    ) -> httpx.Response:
        """Send one signed webhook the way ``sim``'s provider would."""
        path = f"/telephony/{sim.name}/{hook}" + (f"/{token}" if token else "")
        signed_url = self.public + path
        headers: dict[str, str] = {}
        kwargs: dict[str, Any] = {}
        c = self.credentials
        if sim.name == "twilio":
            headers["X-Twilio-Signature"] = security.sign_twilio(str(c.get("auth_token")), signed_url, params)
            kwargs["data"] = params
        elif sim.name == "plivo":
            nonce = secrets.token_hex(8)
            headers["X-Plivo-Signature-V3-Nonce"] = nonce
            headers["X-Plivo-Signature-V3"] = security.sign_plivo(str(c.get("auth_token")), signed_url, params, nonce)
            kwargs["data"] = params
        elif sim.name == "telnyx":
            body = json.dumps(params.get("_json") or params).encode()
            stamp = str(int(time.time()))
            headers["telnyx-timestamp"] = stamp
            headers["telnyx-signature-ed25519"] = security.sign_telnyx(c["seed"], stamp, body)
            headers["Content-Type"] = "application/json"
            kwargs["content"] = body
        else:
            headers["Authorization"] = security.basic_header(
                str(c.get("stream_username")), str(c.get("stream_password"))
            )
            kwargs["params"] = params
            method = "GET"
        async with httpx.AsyncClient(timeout=20.0) as client:
            return await client.request(method, self.base_url + path, headers=headers, **kwargs)

    async def answer_url(self, sim: ProviderSim, from_number: str, to_number: str) -> str:
        """Announce the call and return the stream URL bolo answered with."""
        if sim.name == "twilio":
            params = {
                "CallSid": sim.call_id,
                "From": from_number,
                "To": to_number,
                "Direction": "inbound",
                "CallStatus": "ringing",
            }
            resp = await self.webhook(sim, "voice", params)
            self.answer = resp.text
            found = re.search(r'<Stream url="([^"]+)"', resp.text)
        elif sim.name == "plivo":
            params = {
                "CallUUID": sim.call_id,
                "From": from_number.lstrip("+"),
                "To": to_number.lstrip("+"),
                "Direction": "inbound",
                "Event": "StartApp",
            }
            resp = await self.webhook(sim, "voice", params)
            self.answer = resp.text
            found = re.search(r"<Stream[^>]*>([^<]+)</Stream>", resp.text)
        elif sim.name == "telnyx":
            payload = {
                "call_control_id": sim.call_id, "connection_id": "sim", "call_leg_id": "leg", "call_session_id": "s",
                "client_state": None, "from": from_number, "to": to_number, "direction": "incoming", "state": "parked",
            }  # fmt: skip
            event = {"data": {"record_type": "event", "event_type": "call.initiated", "id": "ev1", "payload": payload}}
            resp = await self.webhook(sim, "voice", {"_json": event})  # type: ignore[dict-item]
            self.answer = resp.text
            command = self.api.last("answer") if self.api is not None else None
            found = re.search(r"(wss?://\S+)", str((command or {}).get("stream_url") or ""))
        else:
            base = self.base_url.replace("http", "ws", 1)
            return f"{base}/telephony/exotel/stream"
        if resp.status_code >= 300 or not found:
            raise RuntimeError(f"bolo did not answer the {sim.name} call: HTTP {resp.status_code} {resp.text[:300]}")
        url = found.group(1).replace("&amp;", "&")
        self.token = url.rsplit("/", 1)[-1]
        return url

    async def connect(self, sim: ProviderSim, from_number: str, to_number: str) -> None:
        import websockets

        url = self.stream_url or await self.answer_url(sim, from_number, to_number)
        headers: dict[str, str] = {}
        c = self.credentials
        if sim.name == "telnyx":
            stamp = str(int(time.time()))
            headers = {
                "telnyx-timestamp": stamp,
                "telnyx-signature-ed25519": security.sign_telnyx(c["seed"], stamp, url.encode()),
            }
        elif sim.name == "exotel":
            headers = {
                "Authorization": security.basic_header(str(c.get("stream_username")), str(c.get("stream_password")))
            }
        # the document carries the public address; the simulator dials the server where it really is
        local = self.base_url.replace("http", "ws", 1) + urllib.parse.urlsplit(url).path
        self._ws = await websockets.connect(local, additional_headers=headers, max_size=None)

    async def send(self, doc: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            await self._ws.send(json.dumps(doc))

    async def receive(self) -> str | None:
        try:
            message = await self._ws.recv()
        except Exception:
            return None
        return message if isinstance(message, str) else message.decode()

    async def close(self) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()

    async def after(self, sim: ProviderSim) -> str | None:
        if sim.name == "telnyx":
            command = self.api.last("transfer") if self.api is not None else None
            if command:
                return f"dial:{command.get('to')}"
            return "hangup" if self.api is not None and self.api.last("hangup") is not None else None
        if sim.name == "exotel":
            resp = await self.webhook(sim, "after", {"CallSid": sim.call_id})
            return f"dial:{resp.text}" if resp.status_code == 200 else "hangup"
        key = "CallSid" if sim.name == "twilio" else "CallUUID"
        resp = await self.webhook(sim, "after", {key: sim.call_id}, token=self.token)
        dial = re.search(r"<Dial>(?:<Number>)?([^<]+)", resp.text)
        return f"dial:{dial.group(1)}" if dial else "hangup" if "<Hangup" in resp.text else None

    async def record(self, sim: ProviderSim) -> dict[str, Any] | None:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=20.0) as client:
            calls = (
                (await client.get(self.base_url + "/api/v1/telephony/calls", headers=headers)).json().get("calls", [])
            )
            match = next((c for c in calls if c.get("provider_call_id") == sim.call_id), None)
            if match is None:
                return None
            doc: dict[str, Any] = (
                await client.get(f"{self.base_url}/api/v1/telephony/calls/{match['id']}", headers=headers)
            ).json()
            return doc


# -- the fake provider REST API -----------------------------------------------------------------------------
class FakeProviderAPI:
    """Accepts the REST requests bolo sends to Twilio, Plivo, Telnyx and Exotel, and records them."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.fail_with: int | None = None
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                ctype = self.headers.get("Content-Type", "")
                if "json" in ctype:
                    body = json.loads(raw or b"{}")
                else:
                    body = {k: v[0] for k, v in urllib.parse.parse_qs(raw.decode()).items()}
                path = self.path
                action = path.rstrip("/").rsplit("/", 1)[-1]
                entry = {
                    "path": path,
                    "action": action,
                    "body": body,
                    "authorization": self.headers.get("Authorization"),
                }
                owner.requests.append(entry)
                if owner.fail_with:
                    doc: dict[str, Any] = {"message": "the fake provider refused", "errors": [{"detail": "refused"}]}
                    status = owner.fail_with
                elif "/2010-04-01/" in path:
                    doc, status = {"sid": "CA" + secrets.token_hex(8), "status": "queued"}, 201
                elif "/v1/Account/" in path:
                    doc, status = {"message": "call fired", "request_uuid": secrets.token_hex(8), "api_id": "x"}, 201
                elif path.endswith("/v2/calls"):
                    data = {
                        "call_control_id": "v3:" + secrets.token_hex(8),
                        "call_leg_id": "l",
                        "call_session_id": "s",
                        "is_alive": False,
                    }
                    doc, status = {"data": data}, 200
                elif "/Calls/connect" in path:
                    doc, status = {"Call": {"Sid": secrets.token_hex(8), "Status": "in-progress"}}, 200
                else:
                    doc, status = {"data": {"result": "ok"}}, 200
                entry["response"] = doc
                out = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def last(self, action: str) -> dict[str, Any] | None:
        """The body of the latest request whose path ends in ``action`` (answer, hangup, transfer, calls, ...)."""
        for entry in reversed(self.requests):
            if entry["action"].lower().split(".")[0] == action.lower():
                body: dict[str, Any] = entry["body"]
                return body
        return None

    def start(self) -> FakeProviderAPI:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)


# -- the caller -------------------------------------------------------------------------------------------
class Caller:
    """Plays a scenario over a link and keeps what a caller would have heard."""

    def __init__(
        self, link: Link, sim: ProviderSim, *, noise_db: float | None = None,
        speech: Callable[[dict[str, Any]], Awaitable[FloatArray]] | None = None,
    ) -> None:  # fmt: skip
        self.link = link
        self.sim = sim
        self.noise_db = noise_db
        self._speech = speech
        self._rng = np.random.default_rng(7)
        self._queue = np.zeros(0, dtype=np.float32)  # caller audio waiting to be sent (8 kHz)
        self._t0 = 0.0
        self._sent_ms = 0
        self.closed = False  # bolo closed the stream
        # playback of the agent's audio
        self._play_end = 0.0
        self._marks: list[tuple[float, str]] = []
        self.heard_s = 0.0
        self.cleared = 0
        self.cleared_s = 0.0
        self.first_audio: list[float] = []  # times agent audio started after silence
        self.media_frames = 0
        self.utterances: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._agent_started = asyncio.Event()

    def now(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def agent_speaking(self) -> bool:
        return time.perf_counter() < self._play_end

    def _bytes_per_s(self) -> int:
        return self.sim.rate * (2 if self.sim.encoding == "pcm16" else 1)

    # -- background loops -----------------------------------------------------------------------------------
    async def _pump(self) -> None:
        """Send one frame of caller audio (or line noise) every frame interval, like a provider does."""
        n = self.sim.rate * self.sim.frame_ms // 1000
        frame = 0
        noise_rms = 0.0 if self.noise_db is None else SPEECH_RMS / (10 ** (float(self.noise_db) / 20.0))
        while not self.closed:
            chunk = self._queue[:n]
            self._queue = self._queue[n:]
            if len(chunk) < n:
                chunk = np.concatenate([chunk, np.zeros(n - len(chunk), dtype=np.float32)])
            if noise_rms:
                chunk = chunk + self._rng.normal(0.0, noise_rms, n).astype(np.float32)
            encoding = "pcm_s16le" if self.sim.encoding == "pcm16" else self.sim.encoding
            await self.link.send(self.sim.media(encode_frame(chunk, encoding), self._sent_ms))
            self._sent_ms += self.sim.frame_ms
            frame += 1
            delay = self._t0 + frame * self.sim.frame_ms / 1000.0 - time.perf_counter()
            await asyncio.sleep(max(0.0, delay))

    async def _listen(self) -> None:
        while True:
            message = await self.link.receive()
            if message is None:
                self.closed = True
                self._agent_started.set()
                return
            try:
                kind, value = self.sim.parse(json.loads(message))
            except (ValueError, KeyError, TypeError) as exc:
                self.errors.append(f"bad frame from bolo: {exc}")
                continue
            now = time.perf_counter()
            if kind == "media":
                self.media_frames += 1
                seconds = len(value) / self._bytes_per_s()
                if now >= self._play_end:
                    self._play_end = now
                    self.first_audio.append(self.now())
                    self._agent_started.set()
                self._play_end += seconds
                self.heard_s += seconds
            elif kind == "clear":
                dropped = max(0.0, self._play_end - now)
                if dropped > 0:
                    self.cleared += 1
                    self.cleared_s += dropped
                    self.heard_s -= dropped
                self._play_end = now
                marks, self._marks = self._marks, []
                for _, name in marks:
                    await self.link.send(self.sim.mark(name))
            elif kind == "mark":
                self._marks.append((self._play_end, value))

    async def _echo_marks(self) -> None:
        while not self.closed:
            await asyncio.sleep(0.02)
            now = time.perf_counter()
            due = [m for m in self._marks if m[0] <= now]
            if due:
                self._marks = [m for m in self._marks if m[0] > now]
                for _, name in due:
                    await self.link.send(self.sim.mark(name))

    # -- steps --------------------------------------------------------------------------------------------
    async def _audio(self, step: dict[str, Any]) -> FloatArray:
        if "speech_s" in step:
            return speechlike(float(step["speech_s"]))
        if "audio" in step:
            from bolo.audio.decode import decode_audio

            data = await asyncio.to_thread(Path(str(step["audio"])).expanduser().read_bytes)
            samples, rate = await asyncio.to_thread(decode_audio, data, filename=str(step["audio"]))
            return resample(samples, rate, 8000)
        if "say" in step:
            if self._speech is None:
                raise InvalidRequest("this simulation has no text-to-speech for 'say' steps")
            return await self._speech(step)
        raise InvalidRequest(f"a speaking step needs 'say', 'audio' or 'speech_s': {step}")

    async def speak(self, samples: FloatArray, label: str) -> None:
        self._queue = np.concatenate([self._queue, samples.astype(np.float32)])
        while len(self._queue) and not self.closed:  # noqa: ASYNC110
            await asyncio.sleep(0.02)
        self._agent_started.clear()
        self.utterances.append(
            {"label": label, "end_s": round(self.now(), 3), "seconds": round(len(samples) / 8000, 2)}
        )

    async def wait_for_agent(self, timeout_s: float = 20.0, quiet_s: float = 0.4) -> bool:
        """Wait until the agent has started speaking (since the caller last spoke) and has finished."""
        deadline = time.perf_counter() + timeout_s
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._agent_started.wait(), timeout_s)
        quiet_from: float | None = None
        while time.perf_counter() < deadline and not self.closed:
            if self.agent_speaking:
                quiet_from = None
            elif quiet_from is None:
                quiet_from = time.perf_counter()
            elif time.perf_counter() - quiet_from >= quiet_s:
                return True
            await asyncio.sleep(0.02)
        return self.closed

    async def run(self, scenario: dict[str, Any], from_number: str = CALLER, to_number: str = CALLED) -> None:
        await self.link.connect(self.sim, from_number, to_number)
        self._t0 = time.perf_counter()
        for doc in self.sim.opening(from_number, to_number):
            await self.link.send(doc)
        tasks = [asyncio.create_task(c) for c in (self._pump(), self._listen(), self._echo_marks())]
        hung_up = False
        try:
            deadline = time.perf_counter() + float(scenario.get("max_s", 90.0))
            for step in scenario.get("steps") or []:
                if self.closed or time.perf_counter() > deadline:
                    break
                remaining = max(0.1, deadline - time.perf_counter())
                if step.get("wait_for_agent"):
                    await self.wait_for_agent(min(remaining, float(step.get("timeout_s", 20.0))))
                elif step.get("wait_for_end"):
                    while not self.closed and time.perf_counter() < deadline:  # noqa: ASYNC110
                        await asyncio.sleep(0.05)
                elif "silence" in step:
                    await asyncio.sleep(min(remaining, float(step["silence"])))
                elif "dtmf" in step:
                    for digit in str(step["dtmf"]):
                        await self.link.send(self.sim.dtmf(digit))
                        await asyncio.sleep(0.15)
                    self._agent_started.clear()
                elif "interrupt" in step:
                    spec = dict(step["interrupt"])
                    samples = await self._audio(spec)
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self._agent_started.wait(), min(remaining, 20.0))
                    await asyncio.sleep(float(spec.get("after_s", 0.5)))
                    was_speaking = self.agent_speaking
                    await self.speak(samples, "interruption")
                    self.utterances[-1]["agent_was_speaking"] = was_speaking
                elif step.get("hangup"):
                    hung_up = True
                    break
                else:
                    await self.speak(await self._audio(step), str(step.get("say") or step.get("audio") or "speech"))
            if not self.closed and (hung_up or not scenario.get("keep_open")):
                stop = self.sim.stop()
                if stop is not None:
                    await self.link.send(stop)
        finally:
            self.closed = True
            for task in tasks[:1] + tasks[2:]:
                task.cancel()
            await self.link.close()
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError, Exception):
                await asyncio.wait_for(tasks[1], 2.0)
            tasks[1].cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    def response_times(self) -> list[float | None]:
        """For each caller utterance: milliseconds from its end until agent audio was next heard."""
        out: list[float | None] = []
        for i, utt in enumerate(self.utterances):
            until = self.utterances[i + 1]["end_s"] if i + 1 < len(self.utterances) else float("inf")
            first = next((t for t in self.first_audio if utt["end_s"] <= t < until), None)
            out.append(None if first is None else round((first - utt["end_s"]) * 1000.0, 1))
        return out


def _check(expect: dict[str, Any], caller: Caller, record: dict[str, Any] | None, after: str | None) -> dict[str, Any]:
    kind = str(expect.get("type"))
    rec = record or {}
    transcript = rec.get("transcript") or []
    timeline = rec.get("timeline") or []
    ok, detail = False, ""
    if kind == "agent_spoke":
        ok, detail = caller.heard_s > 0.2, f"{caller.heard_s:.1f} s of agent audio heard"
    elif kind == "agent_says":
        needle = str(expect.get("contains") or "").lower()
        said = " ".join(str(e.get("full_text") or e["text"]) for e in transcript if e["speaker"] == "agent")
        ok, detail = needle in said.lower(), f"agent said: {said[:120]!r}"
    elif kind == "caller_heard_as":
        needle = str(expect.get("contains") or "").lower()
        said = " ".join(str(e["text"]) for e in transcript if e["speaker"] == "caller")
        ok, detail = needle in said.lower(), f"caller transcribed as: {said[:120]!r}"
    elif kind == "barge_in":
        events = [e for e in timeline if e["type"] == "barge_in"]
        ok = bool(events) and caller.cleared > 0
        detail = f"{len(events)} barge-in(s) recorded, {caller.cleared} clear(s) reached the provider, {caller.cleared_s:.1f} s dropped"
    elif kind == "timeline":
        events = [e for e in timeline if e["type"] == expect.get("event")]
        ok, detail = len(events) >= int(expect.get("min", 1)), f"{len(events)} '{expect.get('event')}' event(s)"
    elif kind == "tool_called":
        names = [e.get("name") for e in timeline if e["type"] == "tool_call"]
        ok, detail = expect.get("name") in names, f"tools called: {names}"
    elif kind == "outcome":
        ok, detail = rec.get("outcome") == expect.get("is"), f"outcome {rec.get('outcome')!r}"
    elif kind == "turns_min":
        answered = [t for t in rec.get("turns") or [] if t.get("assistant_text")]
        ok, detail = len(answered) >= int(expect.get("value", 1)), f"{len(answered)} answered turn(s)"
    elif kind == "max_response_ms":
        times = [t for t in caller.response_times() if t is not None]
        ok = bool(times) and max(times) <= float(expect.get("value", 0))
        detail = f"slowest response {max(times) if times else None} ms"
    elif kind == "after":
        ok, detail = after == expect.get("is"), f"provider was told: {after!r}"
    else:
        detail = "unknown expectation"
    return {**expect, "ok": bool(ok), "detail": detail}


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "p50": ordered[(len(ordered) - 1) // 2], "max": ordered[-1], "mean": round(sum(ordered) / len(ordered), 1),
    }  # fmt: skip


def load_scenario(scenario: Any) -> tuple[str, dict[str, Any]]:
    """A built-in scenario by name, a JSON file, or a scenario object."""
    if isinstance(scenario, dict):
        return str(scenario.get("name") or "custom"), scenario
    name = str(scenario)
    if name in SCENARIOS:
        return name, SCENARIOS[name]
    path = Path(name).expanduser()
    if path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict):
            return path.stem, doc
    raise InvalidRequest(
        f"unknown scenario '{name}' (built in: {', '.join(SCENARIOS)}; or a JSON file)", param="scenario"
    )


def simulator_agent(core: Bolo, name: str | None) -> AgentConfig:
    config = core.telephony.config
    if name:
        agent = config.agents.get(str(name).strip().lower())
        if agent is None:
            raise InvalidRequest(f"unknown agent '{name}'", param="agent")
        return agent
    if config.default_agent and config.default_agent in config.agents:
        return config.agents[config.default_agent]
    if config.agents:
        return next(iter(config.agents.values()))
    return AgentConfig(name="simulator", greeting="Hello, how can I help you?", silence_timeout_s=4.0, source="builtin")


async def _until_ended(link: Link, sim: ProviderSim, timeout_s: float = 15.0) -> None:
    """Wait until the server has closed the call's record (it does when it sees the stream end), so the report reads
    the finished call: its last turn, the outcome and the "ended" event. A busy machine takes longer than any fixed
    pause; after ``timeout_s`` the report reads the record as it is. No record (a call this link did not place):
    nothing to wait for."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        record = await link.record(sim)
        if record is None or record.get("status") == "ended":
            return
        await asyncio.sleep(0.05)


async def run_scenario(
    core: Bolo, scenario: Any, *, agent: str | None = None, provider: str = "twilio", noise_db: Any = None,
    link: Link | None = None, caller_voice: str | None = None, caller_tts: str | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Run one scenario and return the report. Without ``link`` the call goes straight into this process."""
    name, doc = load_scenario(scenario)
    if provider not in SIMS:
        raise InvalidRequest(f"unknown provider '{provider}' (known: {', '.join(SIMS)})", param="provider")
    if not isinstance(doc.get("steps"), list):
        raise InvalidRequest("a scenario needs a list of 'steps'", param="scenario")
    chosen = simulator_agent(core, agent) if link is None else None
    sim = SIMS[provider]()
    noise = noise_db if noise_db is not None else doc.get("noise_db")

    async def speech(step: dict[str, Any]) -> FloatArray:
        samples, rate = await core.synthesize_all(
            str(step["say"]), model=step.get("model") or caller_tts, voice=step.get("voice") or caller_voice
        )
        return resample(samples, rate, 8000)

    if link is None:
        link = DirectLink(core, simulator_agent(core, agent))
    caller = Caller(link, sim, noise_db=None if noise is None else float(noise), speech=speech)
    started = time.perf_counter()
    await caller.run(doc, str(doc.get("from") or CALLER), str(doc.get("to") or CALLED))
    await _until_ended(caller.link, sim)  # the server closes the record when it sees the stream end
    after = await caller.link.after(sim)
    record = await caller.link.record(sim)
    checks = [_check(e, caller, record, after) for e in doc.get("expect") or []]
    for message in caller.errors:
        checks.append({"type": "protocol", "ok": False, "detail": message})
    times = caller.response_times()
    utterances = [{**u, "response_ms": t} for u, t in zip(caller.utterances, times)]
    return {
        "scenario": name,
        "description": doc.get("description"),
        "provider": provider,
        "agent": (record or {}).get("agent") or (chosen.name if chosen else agent),
        "call_id": (record or {}).get("id"),
        "passed": all(c["ok"] for c in checks),
        "checks": checks,
        "duration_s": round(time.perf_counter() - started, 2),
        "noise_db": noise,
        "caller": {
            "utterances": utterances,
            # end of the caller's utterance to the first agent audio at the provider, as a caller hears it
            "response_ms": _quantiles([t for t in times if t is not None]),
            "agent_audio_s": round(caller.heard_s, 2),
            "clears": caller.cleared,
            "cleared_s": round(caller.cleared_s, 2),
            "frames_from_agent": caller.media_frames,
        },
        "after": after,
        "outcome": (record or {}).get("outcome"),
        "latency": (record or {}).get("latency"),
        "record": record,
    }


def test_credentials() -> tuple[dict[str, Any], dict[str, Any]]:
    """Made-up secrets for a self-contained simulation: (what the simulator signs with, the matching
    ``telephony.providers`` table)."""
    seed = os.urandom(32)
    token, password = secrets.token_hex(16), secrets.token_hex(8)
    credentials = {"auth_token": token, "seed": seed, "stream_username": "simulator", "stream_password": password}
    providers = {
        "twilio": {"account_sid": "ACsimulator", "auth_token": token},
        "plivo": {"auth_id": "MAsimulator", "auth_token": token},
        "telnyx": {
            "api_key": "KEYsimulator", "connection_id": "sim",
            "public_key": base64.b64encode(security.ed25519_public_key(seed)).decode(),
        },
        "exotel": {"stream_username": "simulator", "stream_password": password},
    }  # fmt: skip
    return credentials, providers


def simulate_with_server(
    settings: Any, scenarios: list[Any], *, provider: str = "twilio", agent: str | None = None, noise_db: Any = None,
    direct: bool = False,
) -> list[dict[str, Any]]:  # fmt: skip
    """Run scenarios against a private server started for the purpose (``bolo telephony simulate``).

    The server uses ``settings`` (the operator's models, language model and agents) on a free loopback port, with
    made-up provider credentials and a fake provider API, so the whole path a provider takes is exercised
    (signed webhook, answer document, WebSocket) without touching a real account. ``direct`` skips the HTTP path.
    """
    import dataclasses
    import socket

    import uvicorn

    from bolo.api import create_app
    from bolo.core import Bolo

    api = FakeProviderAPI().start()
    credentials, providers = test_credentials()
    for entry in providers.values():
        entry["api_base_url"] = api.url
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    telephony = dict(settings.telephony or {})
    agents = dict(telephony.get("agents") or {})
    if not agents:
        agents["simulator"] = {"greeting": "Hello, how can I help you?", "silence_timeout_s": 4}
    chosen = (agent or telephony.get("default_agent") or next(iter(agents))).strip().lower()
    if chosen not in agents:
        raise InvalidRequest(f"unknown agent '{chosen}' (configured: {', '.join(agents)})", param="agent")
    telephony.update(
        providers=providers, agents=agents, public_base_url=base, validate_signatures=True, default_agent=chosen,
        numbers={},
    )  # fmt: skip
    private = dataclasses.replace(
        settings, host="127.0.0.1", port=port, ui=False, wyoming=False, sip=False, api_key=None
    )
    private.telephony = telephony
    core = Bolo(private)
    server = uvicorn.Server(uvicorn.Config(create_app(core=core), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 30
        while not server.started:
            if time.time() > deadline or not thread.is_alive():
                raise RuntimeError("the simulator's server did not start")
            time.sleep(0.02)
        loop = core.events._loop
        assert loop is not None
        reports = []
        for scenario in scenarios:
            link = None if direct else WebLink(base, credentials, api=api)
            coro = run_scenario(core, scenario, agent=chosen, provider=provider, noise_db=noise_db, link=link)
            reports.append(asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=600))
        return reports
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        api.stop()
