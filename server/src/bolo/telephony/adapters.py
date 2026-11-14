"""The telephony providers: how each one streams a call's audio, answers a call and is asked to place one.

Every adapter is thin. It knows three things about its provider and nothing about agents:

* the **media stream** (:class:`Wire`): the JSON frames on the WebSocket, turned into the neutral
  :class:`StreamStart` / :class:`StreamAudio` / :class:`StreamDtmf` / :class:`StreamMark` / :class:`StreamStop`
  events, and the frames that play audio, clear queued audio (barge-in) and set a mark;
* the **webhooks**: checking the signature, answering a call with the document (or command) that connects it to
  the stream, reporting how a call ended, answering-machine detection;
* the **REST API**: placing a call, hanging up and transferring where that takes a request.

Protocol details were read from each provider's documentation on 2026-11-14; ``docs/telephony.md`` lists what
was verified and what could not be.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar
from xml.sax.saxutils import escape, quoteattr

import httpx

from bolo.errors import BoloError, InvalidRequest
from bolo.providers import scrub
from bolo.telephony import security
from bolo.telephony.config import ProviderSettings

if TYPE_CHECKING:
    from bolo.telephony.manager import PendingCall, TelephonyManager

log = logging.getLogger("bolo.telephony")

XML_HEAD = '<?xml version="1.0" encoding="UTF-8"?>'


class TelephonyError(BoloError):
    """The telephony provider refused a request or could not be reached."""

    status = 502
    code = "telephony_provider_error"
    type = "server_error"


class TelephonyNotConfigured(BoloError):
    status = 503
    code = "telephony_not_configured"
    type = "server_error"


# -- neutral stream events --------------------------------------------------------------------------------
@dataclass
class StreamStart:
    stream_id: str
    call_id: str | None = None
    from_number: str | None = None
    to_number: str | None = None
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class StreamAudio:
    payload: bytes


@dataclass
class StreamDtmf:
    digit: str


@dataclass
class StreamMark:
    name: str


@dataclass
class StreamStop:
    reason: str = "stopped"


StreamEvent = StreamStart | StreamAudio | StreamDtmf | StreamMark | StreamStop


@dataclass
class Inbound:
    """One HTTP request from a provider, reduced to what signatures and handlers need."""

    method: str
    url: str  # the public URL the provider called, query string included
    headers: Mapping[str, str]
    query: dict[str, str] = field(default_factory=dict)
    form: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    @property
    def params(self) -> dict[str, str]:
        return {**self.query, **self.form}

    def json(self) -> dict[str, Any]:
        try:
            doc = json.loads(self.body or b"{}")
        except ValueError:
            return {}
        return doc if isinstance(doc, dict) else {}


@dataclass
class Reply:
    status: int = 200
    body: str = ""
    media_type: str = "text/plain"


def xml(body: str) -> Reply:
    return Reply(200, f"{XML_HEAD}<Response>{body}</Response>", "application/xml")


def _b64(data: Any) -> bytes:
    try:
        return base64.b64decode(str(data or ""), validate=False)
    except (ValueError, binascii.Error):
        return b""


class Wire:
    """One media-stream connection: parses the provider's frames and builds the ones bolo sends."""

    encoding = "mulaw"  # "mulaw", "alaw" or "pcm16"
    rate = 8000
    #: bytes per media frame bolo sends (20 ms of G.711 unless the provider wants otherwise)
    frame_bytes = 160

    def __init__(self) -> None:
        self.stream_id = ""

    def parse(self, message: str) -> list[StreamEvent]:
        try:
            doc = json.loads(message)
        except ValueError:
            return []
        return self._parse(doc) if isinstance(doc, dict) else []

    def _parse(self, doc: dict[str, Any]) -> list[StreamEvent]:
        raise NotImplementedError

    def media(self, payload: bytes) -> str:
        raise NotImplementedError

    def clear(self) -> str | None:
        return None

    def mark(self, name: str) -> str | None:
        return None

    def _format(self, encoding: Any, rate: Any) -> None:
        text = str(encoding or "").lower()
        if "alaw" in text or "pcma" in text:
            self.encoding = "alaw"
        elif "mulaw" in text or "pcmu" in text or "ulaw" in text:
            self.encoding = "mulaw"
        elif "l16" in text or "raw" in text or "slin" in text or "pcm" in text:
            self.encoding = "pcm16"
        try:
            if rate:
                self.rate = int(rate)
        except (TypeError, ValueError):
            pass
        self.frame_bytes = self.rate // 50 * (2 if self.encoding == "pcm16" else 1)


class Adapter:
    name: ClassVar[str] = ""
    label: ClassVar[str] = ""
    #: how bolo ends a call: "close" (closing the stream hands the call back to the provider, which then asks
    #: the ``after`` webhook what to do) or "api" (a REST command)
    ends_by: ClassVar[str] = "close"
    #: what protects the endpoints: the provider's request signature, or credentials it sends
    protection: ClassVar[str] = "signature"
    machine_detection: ClassVar[bool] = False
    #: whether a live call can be handed to a phone number
    can_transfer: ClassVar[bool] = True
    docs_url: ClassVar[str] = ""

    def __init__(self, manager: TelephonyManager) -> None:
        self.manager = manager

    @property
    def cfg(self) -> ProviderSettings:
        return self.manager.config.provider(self.name)

    def wire(self) -> Wire:
        raise NotImplementedError

    def verify(self, inbound: Inbound) -> bool:
        raise NotImplementedError

    def verify_stream(self, headers: Mapping[str, str], url: str) -> bool | None:
        """Check what the provider sends with the WebSocket handshake. None: it sends nothing bolo can check."""
        return None

    async def webhook(self, hook: str, inbound: Inbound, token: str | None) -> Reply:
        raise NotImplementedError

    async def place_call(self, pending: PendingCall, to: str, from_number: str) -> str | None:
        raise NotImplementedError

    async def hangup(self, pending: PendingCall) -> None:
        """End the call with a REST command (providers with ``ends_by == "api"``)."""

    async def transfer(self, pending: PendingCall, number: str) -> None:
        """Transfer with a REST command (providers with ``ends_by == "api"``)."""

    def setup(self) -> dict[str, Any]:
        """What the operator pastes into the provider's dashboard."""
        raise NotImplementedError

    # -- REST helper --------------------------------------------------------------------------------------
    async def _request(
        self, url: str, *, auth: tuple[str, str] | None = None, headers: dict[str, str] | None = None,
        data: dict[str, Any] | None = None, body: dict[str, Any] | None = None,
    ) -> httpx.Response:  # fmt: skip
        secrets = self.cfg.secrets()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0)) as client:
                resp = await client.post(
                    url, auth=auth or httpx.USE_CLIENT_DEFAULT, headers=headers, data=data, json=body
                )
        except httpx.HTTPError as exc:
            raise TelephonyError(
                scrub(f"could not reach {self.label}: {type(exc).__name__}: {exc}", *secrets)
            ) from None
        if resp.status_code >= 400:
            detail = scrub(resp.text[:300], *secrets)
            raise TelephonyError(f"{self.label} answered HTTP {resp.status_code}: {detail}")
        return resp

    def _need(self, *keys: str) -> list[str]:
        missing = self.cfg.missing(keys)
        if missing:
            raise TelephonyNotConfigured(
                f"{self.label} needs {', '.join(f'telephony.providers.{self.name}.{k}' for k in missing)} "
                "before bolo can use its API"
            )
        return [str(self.cfg.get(k)) for k in keys]


# == Twilio ===============================================================================================
class TwilioWire(Wire):
    def _parse(self, doc: dict[str, Any]) -> list[StreamEvent]:
        kind = doc.get("event")
        if kind == "start":
            start = doc.get("start") or {}
            self.stream_id = str(start.get("streamSid") or doc.get("streamSid") or "")
            fmt = start.get("mediaFormat") or {}
            self._format(fmt.get("encoding") or "audio/x-mulaw", fmt.get("sampleRate") or 8000)
            params = {str(k): str(v) for k, v in (start.get("customParameters") or {}).items()}
            return [StreamStart(self.stream_id, start.get("callSid"), params.get("from"), params.get("to"), params)]
        if kind == "media":
            media = doc.get("media") or {}
            if media.get("track", "inbound") != "inbound":
                return []
            return [StreamAudio(_b64(media.get("payload")))]
        if kind == "dtmf":
            return [StreamDtmf(str((doc.get("dtmf") or {}).get("digit") or ""))]
        if kind == "mark":
            return [StreamMark(str((doc.get("mark") or {}).get("name") or ""))]
        if kind == "stop":
            return [StreamStop("stopped")]
        return []

    def media(self, payload: bytes) -> str:
        return json.dumps(
            {"event": "media", "streamSid": self.stream_id, "media": {"payload": base64.b64encode(payload).decode()}}
        )

    def clear(self) -> str | None:
        return json.dumps({"event": "clear", "streamSid": self.stream_id})

    def mark(self, name: str) -> str | None:
        return json.dumps({"event": "mark", "streamSid": self.stream_id, "mark": {"name": name}})


TWILIO_ENDED = {"completed": "caller_hangup", "busy": "busy", "no-answer": "no_answer", "failed": "failed", "canceled": "canceled"}  # fmt: skip


class TwilioAdapter(Adapter):
    name = "twilio"
    label = "Twilio"
    machine_detection = True
    docs_url = "https://www.twilio.com/docs/voice/media-streams"

    def wire(self) -> Wire:
        return TwilioWire()

    def verify(self, inbound: Inbound) -> bool:
        token = self.cfg.get("auth_token")
        if not token:
            return False
        return security.verify_twilio(token, inbound.url, inbound.form, inbound.headers.get("x-twilio-signature"))

    def _connect(self, pending: PendingCall) -> Reply:
        stream = quoteattr(self.manager.ws_url(f"/telephony/twilio/stream/{pending.token}"))
        after = escape(self.manager.url(f"/telephony/twilio/after/{pending.token}"))
        return xml(f'<Connect><Stream url={stream}/></Connect><Redirect method="POST">{after}</Redirect>')

    async def webhook(self, hook: str, inbound: Inbound, token: str | None) -> Reply:
        p = inbound.params
        manager = self.manager
        if hook == "voice":
            if token:
                pending = manager.pending.get(token)
                if pending is None:
                    return xml("<Hangup/>")
            else:
                pending = manager.inbound_call("twilio", p.get("CallSid"), p.get("From"), p.get("To"))
                if pending is None:
                    return xml('<Reject reason="rejected"/>')
            pending.record.provider_call_id = p.get("CallSid") or pending.record.provider_call_id
            answered = p.get("AnsweredBy")
            if answered:
                kind = "machine" if answered.startswith("machine") or answered == "fax" else answered
                stage = "end" if answered.startswith("machine_end") else "start"
                if manager.answered_by(pending, kind, stage, detail=answered) == "hangup":
                    await manager.finish(pending, "voicemail")
                    return xml("<Hangup/>")
            return self._connect(pending)
        pending = manager.pending.get(token or "") or manager.pending_by_call("twilio", p.get("CallSid"))
        if hook == "after":
            if pending is not None and pending.record.transfer_to:
                return xml(f"<Dial>{escape(pending.record.transfer_to)}</Dial>")
            return xml("<Hangup/>")
        if hook == "status":
            outcome = TWILIO_ENDED.get(str(p.get("CallStatus") or ""))
            if pending is not None and outcome:
                await manager.finish(pending, outcome)
            return Reply(204)
        return Reply(404, "unknown webhook")

    async def place_call(self, pending: PendingCall, to: str, from_number: str) -> str | None:
        sid, token = self._need("account_sid", "auth_token")
        auth = (self.cfg.get("api_key_sid") or sid, self.cfg.get("api_key_secret") or token)
        data = {
            "To": to,
            "From": from_number,
            "Url": self.manager.url(f"/telephony/twilio/voice/{pending.token}"),
            "Method": "POST",
            "StatusCallback": self.manager.url(f"/telephony/twilio/status/{pending.token}"),
            "StatusCallbackMethod": "POST",
        }
        if pending.agent.voicemail != "ignore":
            data["MachineDetection"] = "DetectMessageEnd" if pending.agent.voicemail == "leave_message" else "Enable"
        base = self.cfg.get("api_base_url") or "https://api.twilio.com"
        resp = await self._request(f"{base}/2010-04-01/Accounts/{sid}/Calls.json", auth=auth, data=data)
        try:
            return str(resp.json().get("sid") or "") or None
        except ValueError:
            return None

    def setup(self) -> dict[str, Any]:
        m = self.manager
        return {
            "webhooks": [
                {
                    "name": "A call comes in",
                    "where": "Phone Numbers > Manage > Active numbers > (your number) > Voice Configuration: "
                    "'A call comes in' = Webhook, HTTP POST",
                    "url": m.url("/telephony/twilio/voice"),
                    "method": "POST",
                },
                {
                    "name": "Call status changes (optional)",
                    "where": "the same page, 'Call status changes', HTTP POST",
                    "url": m.url("/telephony/twilio/status"),
                    "method": "POST",
                },
            ],
            "stream_url": m.ws_url("/telephony/twilio/stream/<token>"),
            "notes": [
                "bolo answers with TwiML <Connect><Stream>; the stream URL carries a one-time token, so nothing "
                "else has to be configured for the WebSocket.",
            ],
        }


# == Plivo ================================================================================================
class PlivoWire(Wire):
    def _parse(self, doc: dict[str, Any]) -> list[StreamEvent]:
        kind = doc.get("event")
        if kind == "start":
            start = doc.get("start") or {}
            self.stream_id = str(start.get("streamId") or doc.get("streamId") or "")
            fmt = start.get("mediaFormat") or {}
            self._format(fmt.get("encoding") or "audio/x-mulaw", fmt.get("sampleRate") or 8000)
            params: dict[str, str] = {}
            for pair in re.split(r"[;,]", str(doc.get("extra_headers") or "")):
                key, sep, value = pair.partition("=")
                if sep and key.strip():
                    params[key.strip()] = value.strip()
            return [StreamStart(self.stream_id, start.get("callId"), None, None, params)]
        if kind == "media":
            media = doc.get("media") or {}
            if media.get("track", "inbound") != "inbound":
                return []
            return [StreamAudio(_b64(media.get("payload")))]
        if kind == "dtmf":
            return [StreamDtmf(str((doc.get("dtmf") or {}).get("digit") or ""))]
        if kind == "playedStream":
            return [StreamMark(str(doc.get("name") or ""))]
        if kind == "stop":
            return [StreamStop("stopped")]
        return []  # clearedAudio and anything newer

    def _content_type(self) -> str:
        return "audio/x-l16" if self.encoding == "pcm16" else "audio/x-mulaw"

    def media(self, payload: bytes) -> str:
        media = {
            "contentType": self._content_type(), "sampleRate": self.rate, "payload": base64.b64encode(payload).decode(),
        }  # fmt: skip
        return json.dumps({"event": "playAudio", "media": media})

    def clear(self) -> str | None:
        return json.dumps({"event": "clearAudio", "streamId": self.stream_id})

    def mark(self, name: str) -> str | None:
        return json.dumps({"event": "checkpoint", "streamId": self.stream_id, "name": name})


PLIVO_ENDED = {"completed": "caller_hangup", "busy": "busy", "no-answer": "no_answer", "timeout": "no_answer", "failed": "failed", "cancel": "canceled"}  # fmt: skip


class PlivoAdapter(Adapter):
    name = "plivo"
    label = "Plivo"
    machine_detection = True
    docs_url = "https://www.plivo.com/docs/voice/xml/audio-streaming"

    def wire(self) -> Wire:
        return PlivoWire()

    def verify(self, inbound: Inbound) -> bool:
        token = self.cfg.get("auth_token")
        if not token:
            return False
        h = inbound.headers
        return security.verify_plivo(
            token, inbound.url, inbound.form if inbound.method == "POST" else {}, h.get("x-plivo-signature-v3-nonce"),
            [h.get("x-plivo-signature-v3"), h.get("x-plivo-signature-ma-v3")], inbound.method,
        )  # fmt: skip

    def _connect(self, pending: PendingCall) -> Reply:
        stream = escape(self.manager.ws_url(f"/telephony/plivo/stream/{pending.token}"))
        after = escape(self.manager.url(f"/telephony/plivo/after/{pending.token}"))
        return xml(
            '<Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">'
            f'{stream}</Stream><Redirect method="POST">{after}</Redirect>'
        )

    async def webhook(self, hook: str, inbound: Inbound, token: str | None) -> Reply:
        p = inbound.params
        manager = self.manager
        if hook == "voice":
            if token:
                pending = manager.pending.get(token)
                if pending is None:
                    return xml("<Hangup/>")
            else:
                pending = manager.inbound_call("plivo", p.get("CallUUID"), p.get("From"), p.get("To"))
                if pending is None:
                    return xml('<Hangup reason="rejected"/>')
            pending.record.provider_call_id = p.get("CallUUID") or pending.record.provider_call_id
            return self._connect(pending)
        pending = manager.pending.get(token or "") or manager.pending_by_call("plivo", p.get("CallUUID"))
        if hook == "after":
            if pending is not None and pending.record.transfer_to:
                return xml(f"<Dial><Number>{escape(pending.record.transfer_to.lstrip('+'))}</Number></Dial>")
            return xml("<Hangup/>")
        if hook == "amd":
            if pending is not None and str(p.get("Machine") or "").lower() == "true":
                await manager.machine_detected(pending, "machine", "start")
            return Reply(200, "ok")
        if hook == "status":
            if pending is not None:
                machine = str(p.get("Machine") or "").lower() == "true"
                status = str(p.get("CallStatus") or "completed").lower()
                await manager.finish(pending, "voicemail" if machine else PLIVO_ENDED.get(status, "caller_hangup"))
            return Reply(200, "ok")
        return Reply(404, "unknown webhook")

    async def place_call(self, pending: PendingCall, to: str, from_number: str) -> str | None:
        auth_id, token = self._need("auth_id", "auth_token")
        m = self.manager
        body: dict[str, Any] = {
            "from": from_number.lstrip("+"),
            "to": to.lstrip("+"),
            "answer_url": m.url(f"/telephony/plivo/voice/{pending.token}"),
            "answer_method": "POST",
            "hangup_url": m.url(f"/telephony/plivo/status/{pending.token}"),
            "hangup_method": "POST",
        }
        if pending.agent.voicemail == "hangup":
            body["machine_detection"] = "hangup"
        elif pending.agent.voicemail == "leave_message":
            body["machine_detection"] = "true"
            body["machine_detection_url"] = m.url(f"/telephony/plivo/amd/{pending.token}")
            body["machine_detection_method"] = "POST"
        base = self.cfg.get("api_base_url") or "https://api.plivo.com"
        resp = await self._request(f"{base}/v1/Account/{auth_id}/Call/", auth=(auth_id, token), body=body)
        try:
            ref = resp.json().get("request_uuid")
        except ValueError:
            return None
        return str(ref[0] if isinstance(ref, list) and ref else ref or "") or None

    def setup(self) -> dict[str, Any]:
        m = self.manager
        return {
            "webhooks": [
                {
                    "name": "Answer URL",
                    "where": "Voice > Applications > XML > (your application): Primary Answer URL, method POST; "
                    "then assign the application to your number under Phone Numbers",
                    "url": m.url("/telephony/plivo/voice"),
                    "method": "POST",
                },
                {
                    "name": "Hangup URL (optional)",
                    "where": "the same application, Hangup URL, method POST",
                    "url": m.url("/telephony/plivo/status"),
                    "method": "POST",
                },
            ],
            "stream_url": m.ws_url("/telephony/plivo/stream/<token>"),
            "notes": [
                "bolo answers with Plivo XML <Stream bidirectional keepCallAlive> carrying 8 kHz mu-law.",
            ],
        }


# == Telnyx ===============================================================================================
class TelnyxWire(Wire):
    def _parse(self, doc: dict[str, Any]) -> list[StreamEvent]:
        kind = doc.get("event")
        if kind == "start":
            start = doc.get("start") or {}
            self.stream_id = str(doc.get("stream_id") or "")
            fmt = start.get("media_format") or {}
            self._format(fmt.get("encoding") or "PCMU", fmt.get("sample_rate") or 8000)
            return [StreamStart(self.stream_id, start.get("call_control_id"), start.get("from"), start.get("to"))]
        if kind == "media":
            media = doc.get("media") or {}
            if media.get("track", "inbound") != "inbound":
                return []
            return [StreamAudio(_b64(media.get("payload")))]
        if kind == "dtmf":
            return [StreamDtmf(str((doc.get("dtmf") or {}).get("digit") or ""))]
        if kind == "mark":
            return [StreamMark(str((doc.get("mark") or {}).get("name") or ""))]
        if kind == "stop":
            return [StreamStop("stopped")]
        if kind == "error":
            log.warning("telnyx stream error: %s", json.dumps(doc.get("payload"))[:300])
        return []

    def media(self, payload: bytes) -> str:
        return json.dumps({"event": "media", "media": {"payload": base64.b64encode(payload).decode()}})

    def clear(self) -> str | None:
        return json.dumps({"event": "clear"})

    def mark(self, name: str) -> str | None:
        return json.dumps({"event": "mark", "mark": {"name": name}})


TELNYX_ENDED = {"timeout": "no_answer", "no_answer": "no_answer", "user_busy": "busy", "busy": "busy", "call_rejected": "failed"}  # fmt: skip


class TelnyxAdapter(Adapter):
    name = "telnyx"
    label = "Telnyx"
    ends_by = "api"
    machine_detection = True
    docs_url = "https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming"

    def wire(self) -> Wire:
        return TelnyxWire()

    def verify(self, inbound: Inbound) -> bool:
        key = self.cfg.get("public_key")
        if not key:
            return False
        h = inbound.headers
        return security.verify_telnyx(key, h.get("telnyx-timestamp"), inbound.body, h.get("telnyx-signature-ed25519"))

    def verify_stream(self, headers: Mapping[str, str], url: str) -> bool | None:
        key = self.cfg.get("public_key")
        signature = headers.get("telnyx-signature-ed25519")
        if not key or not signature:
            return None
        return security.verify_telnyx(key, headers.get("telnyx-timestamp"), url.encode(), signature)

    def _api(self) -> tuple[str, dict[str, str]]:
        (key,) = self._need("api_key")
        return (self.cfg.get("api_base_url") or "https://api.telnyx.com"), {"Authorization": f"Bearer {key}"}

    def _stream_fields(self, pending: PendingCall) -> dict[str, Any]:
        return {
            "stream_url": self.manager.ws_url(f"/telephony/telnyx/stream/{pending.token}"),
            "stream_track": "inbound_track",
            "stream_bidirectional_mode": "rtp",
            "stream_bidirectional_codec": "PCMU",
            "client_state": base64.b64encode(pending.token.encode()).decode(),
        }

    async def _command(self, call_control_id: str, action: str, body: dict[str, Any]) -> None:
        base, headers = self._api()
        await self._request(f"{base}/v2/calls/{call_control_id}/actions/{action}", headers=headers, body=body)

    async def webhook(self, hook: str, inbound: Inbound, token: str | None) -> Reply:
        data = inbound.json().get("data") or {}
        event = str(data.get("event_type") or "")
        payload = data.get("payload") or {}
        call_id = str(payload.get("call_control_id") or "")
        manager = self.manager
        state = _b64(payload.get("client_state")).decode(errors="replace") if payload.get("client_state") else ""
        pending = manager.pending.get(token or state) or manager.pending_by_call("telnyx", call_id)
        if event == "call.initiated" and payload.get("direction") == "incoming" and pending is None:
            pending = manager.inbound_call("telnyx", call_id, payload.get("from"), payload.get("to"))
            if pending is None:
                await self._command(call_id, "hangup", {})
            else:
                await self._command(call_id, "answer", self._stream_fields(pending))
        elif pending is None:
            pass
        elif event == "call.answered":
            pending.record.event("answered")
        elif event in ("call.machine.detection.ended", "call.machine.premium.detection.ended"):
            result = str(payload.get("result") or "")
            kind = (
                "machine"
                if result in ("machine", "fax_detected")
                else "human"
                if result.startswith("human")
                else "unknown"
            )
            if kind == "machine":
                await manager.machine_detected(pending, "machine", "start", detail=result)
            else:
                manager.answered_by(pending, kind, "start", detail=result)
        elif event == "call.machine.greeting.ended":
            if pending.record.answered_by == "machine":
                await manager.machine_detected(pending, "machine", "end", detail=str(payload.get("result") or ""))
        elif event == "call.hangup":
            cause = str(payload.get("hangup_cause") or "")
            await manager.finish(pending, TELNYX_ENDED.get(cause, "caller_hangup"))
        return Reply(200, "{}", "application/json")

    async def place_call(self, pending: PendingCall, to: str, from_number: str) -> str | None:
        (connection,) = self._need("connection_id")
        base, headers = self._api()
        body: dict[str, Any] = {
            "connection_id": connection,
            "to": to,
            "from": from_number,
            "webhook_url": self.manager.url(f"/telephony/telnyx/voice/{pending.token}"),
            **self._stream_fields(pending),
        }
        if pending.agent.voicemail != "ignore":
            body["answering_machine_detection"] = (
                "greeting_end" if pending.agent.voicemail == "leave_message" else "detect"
            )
        resp = await self._request(f"{base}/v2/calls", headers=headers, body=body)
        try:
            return str((resp.json().get("data") or {}).get("call_control_id") or "") or None
        except ValueError:
            return None

    async def hangup(self, pending: PendingCall) -> None:
        if pending.record.provider_call_id:
            await self._command(pending.record.provider_call_id, "hangup", {})

    async def transfer(self, pending: PendingCall, number: str) -> None:
        if not pending.record.provider_call_id:
            raise TelephonyError("the Telnyx call has no call_control_id to transfer")
        await self._command(pending.record.provider_call_id, "transfer", {"to": number})

    def setup(self) -> dict[str, Any]:
        m = self.manager
        return {
            "webhooks": [
                {
                    "name": "Voice API application webhook",
                    "where": "Mission Control Portal > Voice > Programmable Voice > (your Voice API application): "
                    "'Send a webhook to the URL', API v2; then assign your number to the application",
                    "url": m.url("/telephony/telnyx/voice"),
                    "method": "POST",
                },
            ],
            "stream_url": m.ws_url("/telephony/telnyx/stream/<token>"),
            "notes": [
                "bolo answers each call with the answer command (stream_url, bidirectional RTP, PCMU), so it "
                "needs api_key as well as public_key, and connection_id (the application's ID) to place calls.",
            ],
        }


# == Exotel ===============================================================================================
class ExotelWire(Wire):
    encoding = "pcm16"
    #: Exotel plays chunks of at least 3.2 kB (100 ms at 8 kHz), in multiples of 320 bytes
    frame_bytes = 3200

    def __init__(self, rate: int = 8000) -> None:
        super().__init__()
        self._set_rate(rate)

    def _set_rate(self, rate: Any) -> None:
        try:
            self.rate = int(rate) if int(rate) in (8000, 16000, 24000) else 8000
        except (TypeError, ValueError):
            self.rate = 8000
        self.frame_bytes = 3200 * (self.rate // 8000)

    def _parse(self, doc: dict[str, Any]) -> list[StreamEvent]:
        kind = doc.get("event")
        if kind == "start":
            start = doc.get("start") or {}
            self.stream_id = str(doc.get("stream_sid") or start.get("stream_sid") or "")
            self._set_rate((start.get("media_format") or {}).get("sample_rate") or self.rate)
            params = {str(k): str(v) for k, v in (start.get("custom_parameters") or {}).items()}
            return [StreamStart(self.stream_id, start.get("call_sid"), start.get("from"), start.get("to"), params)]
        if kind == "media":
            return [StreamAudio(_b64((doc.get("media") or {}).get("payload")))]
        if kind == "dtmf":
            return [StreamDtmf(str((doc.get("dtmf") or {}).get("digit") or ""))]
        if kind == "mark":
            return [StreamMark(str((doc.get("mark") or {}).get("name") or ""))]
        if kind == "stop":
            return [StreamStop(str((doc.get("stop") or {}).get("reason") or "stopped"))]
        return []

    def media(self, payload: bytes) -> str:
        return json.dumps(
            {"event": "media", "stream_sid": self.stream_id, "media": {"payload": base64.b64encode(payload).decode()}}
        )

    def clear(self) -> str | None:
        return json.dumps({"event": "clear", "stream_sid": self.stream_id})

    def mark(self, name: str) -> str | None:
        return json.dumps({"event": "mark", "stream_sid": self.stream_id, "mark": {"name": name}})


EXOTEL_ENDED = {"completed": "caller_hangup", "busy": "busy", "no-answer": "no_answer", "failed": "failed", "canceled": "canceled"}  # fmt: skip


class ExotelAdapter(Adapter):
    name = "exotel"
    label = "Exotel"
    protection = "basic"
    docs_url = "https://developer.exotel.com/docs/agentstream/stream-voicebot-applet"

    def wire(self) -> Wire:
        return ExotelWire(int(self.cfg.get("sample_rate") or 8000))

    def verify(self, inbound: Inbound) -> bool:
        return self._basic(inbound.headers)

    def _basic(self, headers: Mapping[str, str]) -> bool:
        user, password = self.cfg.get("stream_username"), self.cfg.get("stream_password")
        if not user or not password:
            return False
        return security.verify_basic(headers.get("authorization"), user, password)

    def verify_stream(self, headers: Mapping[str, str], url: str) -> bool | None:
        if not self.cfg.can_validate:
            return None
        return self._basic(headers)

    async def webhook(self, hook: str, inbound: Inbound, token: str | None) -> Reply:
        p = inbound.params
        manager = self.manager
        call_id = p.get("CallSid")
        if hook == "voice":
            # the applet's "dynamic URL" form: answer with the WebSocket address for this call
            pending = manager.pending_by_call("exotel", call_id) or manager.inbound_call(
                "exotel", call_id, p.get("CallFrom") or p.get("From"), p.get("CallTo") or p.get("To"), p.get("agent")
            )
            if pending is None:
                return Reply(404, json.dumps({"error": "no agent answers this number"}), "application/json")
            url = manager.ws_url(f"/telephony/exotel/stream/{pending.token}")
            return Reply(200, json.dumps({"url": url}), "application/json")
        pending = manager.pending.get(token or "") or manager.pending_by_call("exotel", call_id)
        if hook == "after":
            # a Passthru applet after the Voicebot applet: 200 continues to the transfer branch, 302 to the other
            if pending is not None and pending.record.transfer_to:
                return Reply(200, pending.record.transfer_to)
            return Reply(302)
        if hook == "status":
            if pending is not None:
                status = str(p.get("Status") or p.get("CallStatus") or "completed").lower()
                await manager.finish(pending, EXOTEL_ENDED.get(status, "caller_hangup"))
            return Reply(200, "ok")
        return Reply(404, "unknown webhook")

    async def place_call(self, pending: PendingCall, to: str, from_number: str) -> str | None:
        sid, key, token, app = self._need("account_sid", "api_key", "api_token", "app_id")
        base = self.cfg.get("api_base_url") or f"https://{self.cfg.get('subdomain') or 'api.exotel.com'}"
        data = {
            "From": to,
            "CallerId": from_number,
            "Url": f"http://my.exotel.com/{sid}/exoml/start_voice/{app}",
            "CustomField": pending.record.id,
            "StatusCallback": self.manager.url(f"/telephony/exotel/status/{pending.token}"),
        }
        resp = await self._request(f"{base}/v1/Accounts/{sid}/Calls/connect.json", auth=(key, token), data=data)
        try:
            return str((resp.json().get("Call") or {}).get("Sid") or "") or None
        except ValueError:
            found = re.search(r"<Sid>([^<]+)</Sid>", resp.text)
            return found.group(1) if found else None

    def setup(self) -> dict[str, Any]:
        m = self.manager
        user = self.cfg.get("stream_username") or "<stream_username>"
        stream = m.ws_url("/telephony/exotel/stream")
        with_auth = stream.replace("://", f"://{user}:<stream_password>@", 1)
        return {
            "webhooks": [
                {
                    "name": "Voicebot applet URL",
                    "where": "App Bazaar > (your flow) > Voicebot applet > URL. The user name and password before "
                    "the @ are telephony.providers.exotel.stream_username / stream_password",
                    "url": with_auth,
                    "method": "WebSocket",
                },
                {
                    "name": "Passthru applet after the Voicebot applet (for transfers)",
                    "where": "a Passthru applet placed after the Voicebot applet: its 200 branch goes to a Connect "
                    "applet that dials your transfer number, its 302 branch to Hangup. The request must carry the "
                    "same credentials (user:password@ before the host)",
                    "url": m.url("/telephony/exotel/after").replace("://", f"://{user}:<stream_password>@", 1),
                    "method": "GET",
                },
            ],
            "stream_url": stream,
            "notes": [
                "Exotel does not sign requests; its applet sends the HTTP Basic credentials placed in the URL.",
                "Audio is 16-bit linear PCM; add ?sample-rate=16000 to the applet URL for wideband.",
            ],
        }


ADAPTERS: dict[str, type[Adapter]] = {
    "twilio": TwilioAdapter, "plivo": PlivoAdapter, "telnyx": TelnyxAdapter, "exotel": ExotelAdapter,
}  # fmt: skip


def require_number(value: Any, what: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"\+?[0-9]{5,15}|sips?:[^\s]+", text):
        raise InvalidRequest(f"{what} must be a phone number in E.164 form (for example +14155550100)", param=what)
    return text
