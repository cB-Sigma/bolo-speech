"""Phone calls: provider adapters, the call session, call records, outbound calls and the call simulator.

Every provider is played by a fake (``bolo.telephony.simulator``): signed webhooks, the WebSocket media stream
with real-time playback, and a small HTTP server standing in for the provider's REST API. Nothing leaves the
machine and no account is needed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import websockets

from bolo.config import Settings, load_settings
from bolo.telephony import security
from bolo.telephony.adapters import ADAPTERS, ExotelWire, PlivoWire, StreamAudio, StreamDtmf, StreamStart, TelnyxWire
from bolo.telephony.config import TelephonyConfig, normalize_number
from bolo.telephony.simulator import (
    CALLED,
    SIMS,
    FakeProviderAPI,
    ProviderSim,
    WebLink,
    run_scenario,
    test_credentials,
)
from conftest import LiveServer
from fake_llm import FakeLLM

test_credentials.__test__ = False  # type: ignore[attr-defined]  # a helper, not a test
PROVIDERS = ("twilio", "plivo", "telnyx", "exotel")
TRANSFER_TO = "+15550109999"
REPLY = "You said: the quick brown fox jumps. That is all I heard. Goodbye for now."


class Phone:
    """A live server with telephony configured against fake providers."""

    def __init__(self, server: LiveServer, llm: FakeLLM, api: FakeProviderAPI, credentials: dict[str, Any]) -> None:
        self.server, self.llm, self.api, self.credentials = server, llm, api, credentials

    @property
    def url(self) -> str:
        return self.server.url

    def link(self) -> WebLink:
        return WebLink(self.url, self.credentials, api=self.api)

    async def run(self, scenario: dict[str, Any], provider: str = "twilio", **kw: Any) -> dict[str, Any]:
        return await run_scenario(self.server.core, scenario, provider=provider, link=self.link(), **kw)

    def asked(self) -> dict[str, Any]:
        """The last request a conversation turn sent to the language model (summaries are not streamed)."""
        return next(r for r in reversed(self.llm.requests) if r.get("stream"))

    def calls(self) -> list[dict[str, Any]]:
        return list(httpx.get(self.url + "/api/v1/telephony/calls").json()["calls"])

    def call(self, call_id: str) -> dict[str, Any]:
        return dict(httpx.get(f"{self.url}/api/v1/telephony/calls/{call_id}").json())

    def wait_ended(self, call_id: str, timeout: float = 20.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            doc = self.call(call_id)
            if doc["status"] == "ended":
                return doc
            time.sleep(0.05)
        raise AssertionError(f"call {call_id} did not end: {self.call(call_id)['timeline']}")


AGENTS: dict[str, dict[str, Any]] = {
    "desk": {"greeting": "Hello.", "instructions": "You answer the phone.", "voice": "beta", "silence_timeout_s": 0},
    "quiet": {"silence_timeout_s": 0.8, "silence_prompts": 1, "silence_prompt": "Are you there?", "goodbye": "Bye."},
    "short": {"greeting": "Hello.", "max_duration_s": 1.5, "max_duration_message": "Time is up.", "silence_timeout_s": 0},
    "consent": {"recording_consent": "This call is transcribed for quality.", "greeting": "Hi.", "silence_timeout_s": 0},
    "human": {"transfer_number": TRANSFER_TO, "transfer_message": "One moment.", "silence_timeout_s": 0},
    "vm": {"voicemail": "leave_message", "voicemail_message": "Please call us back.", "silence_timeout_s": 0},
    "dialer": {"greeting": "Hello, this is a call from the clinic.", "silence_timeout_s": 0},
}  # fmt: skip


def make_phone(settings: Settings, llm: FakeLLM, **telephony: Any) -> tuple[Phone, FakeProviderAPI]:
    api = FakeProviderAPI().start()
    credentials, providers = test_credentials()
    for name, entry in providers.items():
        entry["api_base_url"] = api.url
        entry["from_number"] = "+15550100000"
        if name == "exotel":
            entry.update(account_sid="exo", api_key="exokey", api_token="exotoken", app_id="77")
    settings.llm_base_url, settings.llm_model = llm.url, "fake-llm"
    settings.telephony = {
        "providers": providers, "default_agent": "desk", "numbers": {"+15550100777": "quiet"},
        "agents": json.loads(json.dumps(AGENTS)), **telephony,
    }  # fmt: skip
    server = LiveServer(settings)
    server.core.telephony.config.public_base_url = server.url
    server.start()
    return Phone(server, llm, api, credentials), api


@pytest.fixture
def phone(settings: Settings, llm: FakeLLM) -> Iterator[Phone]:
    made, api = make_phone(settings, llm)
    try:
        yield made
    finally:
        made.server.stop()
        api.stop()


def events(record: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in record["timeline"] if e["type"] == kind]


TALK = {"steps": [{"wait_for_agent": True}, {"speech_s": 2.0}, {"silence": 1.6}, {"hangup": True}]}


# -- signatures ---------------------------------------------------------------------------------------------
def test_twilio_signature_matches_the_documented_example() -> None:
    # the worked example of Twilio's "Webhooks security" page: auth token 12345
    url = "https://mycompany.com/myapp.php?foo=1&bar=2"
    params = {"CallSid": "CA1234567890ABCDE", "Caller": "+12349013030", "Digits": "1234", "From": "+12349013030", "To": "+18005551212"}  # fmt: skip
    assert security.sign_twilio("12345", url, params) == "0/KCTR6DLpKmkAf8muzZqo1nDgQ="
    assert security.verify_twilio("12345", url, params, "0/KCTR6DLpKmkAf8muzZqo1nDgQ=")
    assert not security.verify_twilio("12345", url, {**params, "Digits": "1"}, "0/KCTR6DLpKmkAf8muzZqo1nDgQ=")
    assert not security.verify_twilio("12345", url, params, None)


def test_plivo_signature_v3() -> None:
    url, params = (
        "https://voice.example.org/telephony/plivo/voice",
        {"To": "14155550100", "CallUUID": "abc", "From": "1415"},
    )
    assert security._plivo_base(url, params, "POST") == url + ".CallUUIDabcFrom1415To14155550100"
    assert security._plivo_base(url + "?b=2&a=1", {}, "GET") == url + "?a=1&b=2"
    signature = security.sign_plivo("token", url, params, "nonce1")
    assert security.verify_plivo("token", url, params, "nonce1", [None, f"other, {signature}"])
    assert not security.verify_plivo("token", url, params, "nonce2", [signature])
    assert not security.verify_plivo("wrong", url, params, "nonce1", [signature])
    assert not security.verify_plivo("token", url, params, None, [signature])


def test_ed25519_rfc8032_vectors_and_telnyx() -> None:
    seed = bytes.fromhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb")
    public = bytes.fromhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c")
    signature = bytes.fromhex(
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
        "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"
    )
    assert security.ed25519_public_key(seed) == public
    assert security.ed25519_sign(seed, b"\x72") == signature
    assert security.ed25519_verify(public, b"\x72", signature)
    assert not security.ed25519_verify(public, b"\x73", signature)
    assert not security.ed25519_verify(public, b"\x72", signature[:63] + bytes([signature[63] ^ 1]))
    key = base64.b64encode(public).decode()
    stamp = str(int(time.time()))
    signed = security.sign_telnyx(seed, stamp, b'{"data":{}}')
    assert security.verify_telnyx(key, stamp, b'{"data":{}}', signed)
    assert not security.verify_telnyx(key, stamp, b'{"data":{"x":1}}', signed)
    old = str(int(time.time()) - 3600)  # replayed an hour later
    assert not security.verify_telnyx(key, old, b"{}", security.sign_telnyx(seed, old, b"{}"))
    assert not security.verify_telnyx("not a key", stamp, b"{}", signed)


def test_ed25519_agrees_with_the_cryptography_package() -> None:
    ed = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    from cryptography.hazmat.primitives import serialization

    for _ in range(4):
        seed, message = os.urandom(32), os.urandom(40)
        private = ed.Ed25519PrivateKey.from_private_bytes(seed)
        public = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        assert security.ed25519_public_key(seed) == public
        assert security.ed25519_sign(seed, message) == private.sign(message)
        assert security.ed25519_verify(public, message, private.sign(message))


# -- configuration ------------------------------------------------------------------------------------------
def test_telephony_settings_from_a_file_and_the_environment(tmp_path: Path) -> None:
    config = tmp_path / "bolo.toml"
    config.write_text(
        """
[telephony]
public_base_url = "https://voice.example.org/"
default_agent = "front-desk"

[telephony.providers.twilio]
account_sid = "AC123"

[telephony.agents.front-desk]
instructions = "Answer the phone."
greeting = "Hello!"
max_duration_s = 120
tools = [{ name = "lookup", description = "Find an order", url = "https://tools.example.org/lookup" }]

[telephony.numbers]
"+1 (415) 555-0100" = "front-desk"
""",
        encoding="utf-8",
    )
    env = {"BOLO_TELEPHONY_TWILIO_AUTH_TOKEN": "tw-secret", "BOLO_TELEPHONY_TELNYX_API_KEY": "KEY-secret"}
    settings = load_settings(config, {"data_dir": str(tmp_path / "data")}, env=env)
    parsed = TelephonyConfig(settings)
    assert parsed.public_base_url == "https://voice.example.org" and parsed.validate_signatures
    assert parsed.provider("twilio").values == {"account_sid": "AC123", "auth_token": "tw-secret"}
    assert parsed.provider("twilio").can_validate and not parsed.provider("telnyx").can_validate
    agent = parsed.agent_for("+14155550100")
    assert agent is not None and agent.name == "front-desk" and agent.max_duration_s == 120.0
    assert agent.tools[0]["url"] == "https://tools.example.org/lookup"
    # national and international spellings of the same number meet; unknown numbers get the default agent
    assert parsed.agent_for("04155550100") is agent and parsed.agent_for("+4930123456") is agent
    assert normalize_number("sip:+14155550100@pbx.example.org") == "14155550100"
    shown = json.dumps(settings.public_dict()["telephony"])
    assert "tw-secret" not in shown and "KEY-secret" not in shown and "AC123" in shown
    assert load_settings(
        None, {"data_dir": str(tmp_path)}, env={"BOLO_TELEPHONY_VALIDATE_SIGNATURES": "false"}
    ).telephony == {"validate_signatures": False}


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ({"providers": {"vonage": {}}}, "unknown provider 'vonage'"),
        ({"providers": {"twilio": {"password": "x"}}}, "unknown fields: password"),
        ({"agents": {"a": {"voicemail": "leave_message"}}}, "needs a voicemail_message"),
        ({"agents": {"a": {"pipeline": "telepathy"}}}, "pipeline must be one of"),
        ({"agents": {"Bad Name": {}}}, "invalid agent name"),
        ({"public_base_url": "voice.example.org"}, "public_base_url must start with https://"),
        ({"webhook": "x"}, "unknown settings: webhook"),
    ],
)
def test_bad_telephony_settings_fail_at_start_with_the_reason(
    tmp_path: Path, table: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        load_settings(None, {"data_dir": str(tmp_path)}, env={"BOLO_TELEPHONY": json.dumps(table)})


# -- the wire formats -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("provider", PROVIDERS)
def test_adapter_and_fake_provider_speak_the_same_protocol(provider: str, core: Any) -> None:
    """What the fake provider sends, the adapter understands, and the other way round."""
    sim: ProviderSim = SIMS[provider]()
    wire = ADAPTERS[provider](core.telephony).wire()
    start = [e for doc in sim.opening("+15550100001", CALLED) for e in wire.parse(json.dumps(doc))]
    assert len(start) == 1 and isinstance(start[0], StreamStart)
    assert start[0].stream_id == sim.stream_id and start[0].call_id == sim.call_id
    assert (wire.encoding, wire.rate) == (sim.encoding, 8000)
    if provider in ("telnyx", "exotel"):  # these say who is calling whom in the stream itself
        assert (start[0].from_number, start[0].to_number) == ("+15550100001", CALLED)
    payload = bytes(range(160)) * (20 if provider == "exotel" else 1)
    (audio,) = wire.parse(json.dumps(sim.media(payload, 20)))
    assert isinstance(audio, StreamAudio) and audio.payload == payload
    (digit,) = wire.parse(json.dumps(sim.dtmf("7")))
    assert isinstance(digit, StreamDtmf) and digit.digit == "7"
    assert sim.parse(json.loads(wire.media(payload))) == ("media", payload)
    assert sim.parse(json.loads(wire.clear() or "{}")) == ("clear", None)
    assert sim.parse(json.loads(wire.mark("u1") or "{}")) == ("mark", "u1")
    (mark,) = wire.parse(json.dumps(sim.mark("u1")))
    assert mark.name == "u1"  # type: ignore[union-attr]
    assert wire.parse("not json") == [] and wire.parse('{"event": "something-new"}') == []


def test_wire_details_per_provider() -> None:
    plivo = PlivoWire()
    (start,) = plivo.parse(
        json.dumps({"event": "start", "start": {"callId": "c", "streamId": "s", "mediaFormat": {"encoding": "audio/x-l16", "sampleRate": 16000}},
                    "extra_headers": "agent=desk;tenant=42"})
    )  # fmt: skip
    assert start.params == {"agent": "desk", "tenant": "42"}  # type: ignore[union-attr]
    assert (plivo.encoding, plivo.rate, plivo.frame_bytes) == ("pcm16", 16000, 640)
    assert json.loads(plivo.media(b"\0\0"))["media"] == {
        "contentType": "audio/x-l16",
        "sampleRate": 16000,
        "payload": "AAA=",
    }
    telnyx = TelnyxWire()
    telnyx.parse(
        json.dumps(
            {"event": "start", "stream_id": "s", "start": {"media_format": {"encoding": "PCMA", "sample_rate": 8000}}}
        )
    )
    assert telnyx.encoding == "alaw"
    outbound = {"event": "media", "media": {"track": "outbound", "payload": "AAAA"}}
    assert telnyx.parse(json.dumps(outbound)) == []  # bolo's own audio echoed back is not the caller
    exotel = ExotelWire()
    exotel.parse(json.dumps({"event": "start", "stream_sid": "s", "start": {"media_format": {"sample_rate": "16000"}}}))
    assert (exotel.rate, exotel.frame_bytes) == (16000, 6400)


# -- exposure -------------------------------------------------------------------------------------------------
def test_endpoints_are_closed_until_requests_can_be_checked(live: LiveServer) -> None:
    for provider in PROVIDERS:
        resp = httpx.post(f"{live.url}/telephony/{provider}/voice", data={"CallSid": "CA1", "To": "+1555"})
        assert resp.status_code == 503 and resp.json()["error"]["code"] == "telephony_not_configured"
        message = resp.json()["error"]["message"]
        assert "telephony.public_base_url is not set" in message and "telephony.validate_signatures = false" in message
        assert f"telephony.providers.{provider}." in message
    assert httpx.post(live.url + "/telephony/vonage/voice").status_code == 404
    overview = httpx.get(live.url + "/api/v1/telephony").json()
    assert all(not p["exposed"] and p["problems"] for p in overview["providers"])


async def test_streams_without_a_token_are_refused(phone: Phone) -> None:
    for provider in ("twilio", "plivo", "telnyx"):
        for path in ("stream", "stream/not-a-token"):
            with pytest.raises(websockets.exceptions.InvalidStatus) as refused:
                async with websockets.connect(f"{phone.server.ws_url}/telephony/{provider}/{path}"):
                    pass
            assert refused.value.response.status_code == 403
    with pytest.raises(websockets.exceptions.InvalidStatus):  # Exotel: no token, so the Basic credentials decide
        async with websockets.connect(
            f"{phone.server.ws_url}/telephony/exotel/stream",
            additional_headers={"Authorization": security.basic_header("simulator", "wrong")},
        ):
            pass
    assert phone.calls() == []


async def test_wrong_signatures_are_refused_and_nothing_is_answered(phone: Phone) -> None:
    link = phone.link()
    link.credentials = {
        **phone.credentials,
        "auth_token": "not-the-token",
        "seed": os.urandom(32),
        "stream_password": "x",
    }
    for provider in PROVIDERS:
        sim = SIMS[provider]()
        body: dict[str, Any] = (
            {"_json": {"data": {"event_type": "call.initiated"}}} if provider == "telnyx" else {"CallSid": "CA1"}
        )
        resp = await link.webhook(sim, "voice", body)
        assert resp.status_code == (401 if provider == "exotel" else 403), provider
        assert resp.json()["error"]["code"] == "invalid_signature"
    unsigned = httpx.post(phone.url + "/telephony/twilio/voice", data={"CallSid": "CA1", "To": CALLED})
    assert unsigned.status_code == 403
    assert phone.calls() == [] and phone.api.requests == []


def test_opting_out_of_validation_is_explicit(settings: Settings, llm: FakeLLM) -> None:
    settings.telephony = {"validate_signatures": False, "agents": {"desk": {}}, "default_agent": "desk"}
    server = LiveServer(settings)
    server.start()
    try:
        resp = httpx.post(server.url + "/telephony/twilio/voice", data={"CallSid": "CA9", "From": "+1", "To": "+2"})
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("application/xml")
        # without public_base_url the address is taken from the request
        assert f'<Stream url="ws://127.0.0.1:{server.port}/telephony/twilio/stream/' in resp.text
        overview = httpx.get(server.url + "/api/v1/telephony").json()
        assert overview["validate_signatures"] is False and all(p["exposed"] for p in overview["providers"])
    finally:
        server.stop()


# -- calls through every provider ---------------------------------------------------------------------------
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_call_through_each_provider(phone: Phone, provider: str) -> None:
    report = await phone.run({**TALK, "expect": [{"type": "agent_spoke"}, {"type": "turns_min", "value": 1}]}, provider)
    assert report["passed"], report["checks"]
    record = report["record"]
    assert record["provider"] == provider and record["agent"] == "desk" and record["direction"] == "inbound"
    assert record["status"] == "ended" and record["outcome"] == "caller_hangup" and record["duration_s"] > 4
    assert record["to"].lstrip("+") == CALLED.lstrip("+") and record["from"]
    said = record["turns"][0]["user_text"]
    assert said.startswith("the quick brown fox jumps")
    assert [(e["speaker"], e["text"]) for e in record["transcript"]][:2] == [("agent", "Hello."), ("caller", said)]
    assert record["transcript"][0]["kind"] == "greeting"
    (turn,) = record["turns"]
    latency = turn["latency"]
    assert turn["assistant_text"].startswith("You said: the quick")
    # end of speech -> detector -> STT -> first token -> first audio -> sent: every part is there and they add up
    parts = [latency[k] for k in ("stt_ms", "llm_first_token_ms", "tts_first_audio_ms", "send_ms")]
    assert all(p is not None and p >= 0 for p in parts) and sum(parts) == pytest.approx(latency["total_ms"], abs=1.0)
    assert 350 <= latency["endpoint_ms"] <= 800  # the 500 ms of silence the detector waits for
    assert latency["response_ms"] == pytest.approx(latency["endpoint_ms"] + latency["total_ms"], abs=0.2)
    # what the caller measured at the provider agrees with the server's own account
    heard = report["caller"]["utterances"][0]["response_ms"]
    assert heard is not None and abs(heard - latency["response_ms"]) < 350, (heard, latency)
    kinds = [e["type"] for e in record["timeline"]]
    order = [
        "ringing",
        "stream_started",
        "greeting",
        "user_speech_started",
        "user_speech_stopped",
        "agent_audio_started",
        "ended",
    ]
    assert [k for k in kinds if k in order] == order
    assert report["after"] in ("hangup", None)
    assert phone.asked()["messages"][:3] == [
        {"role": "system", "content": "You answer the phone."},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": said},
    ]
    # the turn is also an ordinary agent trace, tied to the call
    traces = httpx.get(phone.url + "/api/v1/requests", params={"kind": "agent"}).json()["requests"]
    assert traces[0]["transport"] == "telephony" and traces[0]["call_id"] == record["id"]
    assert traces[0]["telephony_provider"] == provider and traces[0]["agent"] == "desk"
    listed = phone.calls()[0]
    assert listed["id"] == record["id"] and listed["turns"] == 1 and listed["latency"]["response_ms"]["p50"] > 0
    # a summary was written with the language model once the call ended
    deadline = time.time() + 5
    while time.time() < deadline and not phone.call(record["id"])["summary"]:
        await asyncio.sleep(0.05)
    assert f"Caller: {said}" in phone.call(record["id"])["summary"]
    stats = httpx.get(phone.url + "/api/v1/stats").json()["telephony"]
    assert stats["calls_total"] == 1 and stats["by_outcome"] == [
        {"provider": provider, "outcome": "caller_hangup", "calls": 1}
    ]
    assert (
        f'bolo_telephony_calls_total{{provider="{provider}",outcome="caller_hangup"}} 1'
        in httpx.get(phone.url + "/metrics").text
    )


async def test_the_answer_documents(phone: Phone) -> None:
    link = phone.link()
    await link.answer_url(SIMS["twilio"](), "+15550100001", CALLED)
    token = link.token
    assert link.answer == (
        '<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
        f'<Stream url="{phone.server.ws_url}/telephony/twilio/stream/{token}"/></Connect>'
        f'<Redirect method="POST">{phone.url}/telephony/twilio/after/{token}</Redirect></Response>'
    )
    await link.answer_url(SIMS["plivo"](), "+15550100001", CALLED)
    assert link.answer == (
        '<?xml version="1.0" encoding="UTF-8"?><Response>'
        '<Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">'
        f"{phone.server.ws_url}/telephony/plivo/stream/{link.token}</Stream>"
        f'<Redirect method="POST">{phone.url}/telephony/plivo/after/{link.token}</Redirect></Response>'
    )
    sim = SIMS["telnyx"]()
    await link.answer_url(sim, "+15550100001", CALLED)
    request = phone.api.requests[-1]
    assert (
        request["path"] == f"/v2/calls/{sim.call_id}/actions/answer"
        and request["authorization"] == "Bearer KEYsimulator"
    )
    assert request["body"] == {
        "stream_url": f"{phone.server.ws_url}/telephony/telnyx/stream/{link.token}", "stream_track": "inbound_track",
        "stream_bidirectional_mode": "rtp", "stream_bidirectional_codec": "PCMU",
        "client_state": base64.b64encode(str(link.token).encode()).decode(),
    }  # fmt: skip
    # Exotel's applet may ask an HTTPS endpoint for the WebSocket address
    resp = await link.webhook(SIMS["exotel"](), "voice", {"CallSid": "exo1", "CallFrom": "0911", "CallTo": "0922"})
    assert re.fullmatch(rf"{phone.server.ws_url}/telephony/exotel/stream/\w+", resp.json()["url"])
    # a call nobody answers: no agent for the number and no default
    phone.server.core.telephony.config.default_agent = None
    refused = await link.webhook(SIMS["twilio"](), "voice", {"CallSid": "CA2", "From": "+1", "To": "+19990000000"})
    assert "<Reject" in refused.text
    mapped = await link.webhook(SIMS["twilio"](), "voice", {"CallSid": "CA3", "From": "+1", "To": "+15550100777"})
    assert "<Connect>" in mapped.text and phone.calls()[0]["agent"] == "quiet"


async def test_a_stream_token_works_once(phone: Phone) -> None:
    link = phone.link()
    url = await link.answer_url(SIMS["twilio"](), "+15550100001", CALLED)
    async with websockets.connect(url):
        with pytest.raises(websockets.exceptions.InvalidStatus):
            async with websockets.connect(url):
                pass


# -- what happens on a call -----------------------------------------------------------------------------------
async def test_barge_in_clears_the_providers_queue_and_truncates_the_answer(phone: Phone) -> None:
    scenario = {
        "steps": [
            {"wait_for_agent": True}, {"speech_s": 2.0}, {"interrupt": {"after_s": 1.0, "speech_s": 1.5}},
            {"silence": 1.5}, {"hangup": True},
        ],
        "expect": [{"type": "barge_in"}],
    }  # fmt: skip
    report = await phone.run(scenario)
    assert report["passed"], report["checks"]
    record = report["record"]
    (barge,) = events(record, "barge_in")
    assert barge["cause"] == "speech" and 700 <= barge["heard_ms"] <= 2200 and barge["cleared_ms"] > 1500
    assert report["caller"]["clears"] == 1 and report["caller"]["cleared_s"] > 1.5
    assert report["caller"]["utterances"][1]["agent_was_speaking"] is True
    first, second = (e for e in record["transcript"] if e["speaker"] == "agent" and "kind" not in e)
    # the agent's line is what the caller heard of it, and the language model is told the same
    assert first["interrupted"] and first["full_text"].startswith("You said: the quick brown fox jumps")
    assert (
        first["text"] and len(first["text"]) < len(first["full_text"]) and first["full_text"].startswith(first["text"])
    )
    assert "interrupted" not in second
    history = phone.asked()["messages"]
    assert {"role": "assistant", "content": first["text"]} in history
    assert record["turns"][0]["interrupted"] is True and record["turns"][1]["interrupted"] is False
    assert phone.calls()[0]["interruptions"] == 1


async def test_key_presses_reach_the_agent(phone: Phone) -> None:
    scenario = {"steps": [{"wait_for_agent": True}, {"dtmf": "42#"}, {"wait_for_agent": True}, {"hangup": True}]}
    report = await phone.run(scenario, "plivo")
    record = report["record"]
    assert record["dtmf"] == "42#" and [e["digit"] for e in events(record, "dtmf")] == ["4", "2", "#"]
    said = "[The caller pressed 4 2 # on the phone keypad]"
    assert {"role": "user", "content": said} in phone.asked()["messages"]
    assert next(e for e in record["transcript"] if e.get("kind") == "dtmf")["text"] == said
    (turn,) = record["turns"]
    assert turn["kind"] == "dtmf" and turn["assistant_text"].startswith("You said: [The caller pressed 4 2 #")
    assert turn["latency"]["endpoint_ms"] is None and turn["latency"]["total_ms"] > 0


async def test_silence_prompts_then_hang_up(phone: Phone) -> None:
    report = await phone.run({"to": "+15550100777", "steps": [{"wait_for_end": True}], "max_s": 20}, "telnyx")
    record = report["record"]
    assert record["agent"] == "quiet" and record["outcome"] == "no_response"
    assert [(e["kind"], e["text"]) for e in record["transcript"]] == [
        ("silence_prompt", "Are you there?"),
        ("goodbye", "Bye."),
    ]
    assert events(record, "no_response") == [
        {"t_ms": events(record, "no_response")[0]["t_ms"], "type": "no_response", "prompts": 1}
    ]
    assert 2.0 < record["duration_s"] < 8.0
    # Telnyx is hung up with a command; the caller heard the farewell before it
    assert phone.api.requests[-1]["path"].endswith("/actions/hangup") and report["after"] == "hangup"
    assert report["caller"]["agent_audio_s"] > 0.8


async def test_longest_call_length(phone: Phone) -> None:
    phone.server.core.telephony.config.default_agent = "short"
    report = await phone.run({"steps": [{"wait_for_end": True}], "max_s": 20})
    record = report["record"]
    assert record["outcome"] == "max_duration" and events(record, "max_duration")[0]["seconds"] == 1.5
    assert [e["text"] for e in record["transcript"]] == ["Hello.", "Time is up."]
    assert 1.5 <= record["duration_s"] < 5 and report["after"] == "hangup"


async def test_hanging_up_while_the_agent_prepares_an_answer_ends_the_call(phone: Phone) -> None:
    """The Realtime session closed with the call even when its turn was waiting on a response: that wait swallowed
    the cancellation, the session's turn loop then waited for another turn, and the call was never closed."""
    phone.llm.token_delay_s = 0.5  # an answer that takes seconds to write
    scenario = {"steps": [{"wait_for_agent": True}, {"speech_s": 1.5}, {"silence": 1.2}, {"hangup": True}]}
    report = await phone.run(scenario)
    record = report["record"]
    assert record["status"] == "ended" and record["outcome"] == "caller_hangup", record["timeline"]
    assert events(record, "ended") and record["turns"][0]["user_text"]
    assert phone.calls()[0]["status"] == "ended"


async def test_recording_consent_comes_first_and_cannot_be_interrupted(phone: Phone) -> None:
    phone.server.core.telephony.config.default_agent = "consent"
    report = await phone.run({"steps": [{"silence": 0.3}, {"speech_s": 1.2}, {"silence": 2.0}, {"hangup": True}]})
    record = report["record"]
    assert [e.get("kind") for e in record["transcript"]][:2] == ["consent", "greeting"]
    assert record["transcript"][0]["text"] == "This call is transcribed for quality."
    assert "interrupted" not in record["transcript"][0]
    # the caller spoke over it: nothing was cleared while the announcement played
    consent_ms = len("This call is transcribed for quality.") * 60
    assert all(e["t_ms"] > consent_ms - 300 for e in events(record, "barge_in"))


def tool_reply(name: str, arguments: str = "{}") -> Any:
    def reply(body: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        if (body.get("messages") or [{}])[-1].get("role") == "tool":
            return FakeLLM.reply(body)
        return "", {"id": "call_t1", "type": "function", "function": {"name": name, "arguments": arguments}}

    return reply


async def test_the_agent_hangs_up(phone: Phone) -> None:
    phone.llm.reply = tool_reply("hang_up", '{"reason": "done"}')  # type: ignore[method-assign]
    phone.server.core.telephony.config.agents["desk"].goodbye = "Goodbye."
    report = await phone.run(
        {"steps": [{"wait_for_agent": True}, {"speech_s": 1.5}, {"wait_for_end": True}], "max_s": 20}
    )
    record = report["record"]
    assert record["outcome"] == "agent_hangup" and report["after"] == "hangup"
    assert events(record, "tool_call")[0]["name"] == "hang_up"
    assert record["transcript"][-1]["kind"] == "goodbye" and record["turns"][0]["tool_calls"] == ["hang_up"]
    offered = [t["function"]["name"] for t in phone.asked()["tools"]]
    assert offered == ["hang_up"]  # no transfer tool: the agent has no transfer number


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_transfer_to_a_person(phone: Phone, provider: str) -> None:
    phone.llm.reply = tool_reply("transfer_call")  # type: ignore[method-assign]
    phone.server.core.telephony.config.default_agent = "human"
    report = await phone.run({"steps": [{"speech_s": 1.5}, {"wait_for_end": True}], "max_s": 20}, provider)
    record = report["record"]
    assert record["outcome"] == "transferred" and record["transfer_to"] == TRANSFER_TO
    assert events(record, "transfer")[0]["to"] == TRANSFER_TO
    assert record["transcript"][-1] == {**record["transcript"][-1], "kind": "transfer_message", "text": "One moment."}
    # how the provider learns of it: TwiML <Dial>, Plivo XML <Dial><Number>, a Telnyx command, Exotel's Passthru
    assert report["after"] == "dial:" + (TRANSFER_TO.lstrip("+") if provider == "plivo" else TRANSFER_TO)
    assert report["caller"]["agent_audio_s"] > 0.4  # "One moment." was heard before the call moved on
    assert [t["function"]["name"] for t in phone.asked()["tools"]] == ["hang_up", "transfer_call"]


async def test_a_tool_with_a_url_is_called_and_answered(phone: Phone) -> None:
    tool = {
        "name": "get_weather",
        "description": "Weather",
        "parameters": {"type": "object"},
        "url": phone.api.url + "/tools/weather",
    }
    phone.server.core.telephony.config.agents["desk"].tools = [tool]
    phone.llm.reply = tool_reply("get_weather", '{"city": "Pune"}')  # type: ignore[method-assign]
    report = await phone.run(
        {
            "steps": [
                {"wait_for_agent": True},
                {"speech_s": 1.5},
                {"wait_for_agent": True},
                {"silence": 0.5},
                {"hangup": True},
            ]
        }
    )
    record = report["record"]
    sent = phone.api.last("weather")
    assert sent is not None and sent["name"] == "get_weather" and sent["arguments"] == {"city": "Pune"}
    assert (
        sent["call"]["id"] == record["id"] and sent["call"]["provider"] == "twilio" and sent["call"]["agent"] == "desk"
    )
    assert events(record, "tool_result")[0]["name"] == "get_weather"
    answer = [e["text"] for e in record["transcript"] if e["speaker"] == "agent"][-1]
    assert answer.startswith("The tool said:") and "ok" in answer
    assert [t["kind"] for t in record["turns"]] == ["speech", "tool"]
    assert "url" not in json.dumps(phone.asked()["tools"])  # the model sees the function, not where it lives


# -- outbound ---------------------------------------------------------------------------------------------------
def place(phone: Phone, provider: str, agent: str = "dialer", **extra: Any) -> dict[str, Any]:
    resp = httpx.post(
        phone.url + "/api/v1/telephony/calls",
        json={"provider": provider, "to": "+15550100123", "agent": agent, **extra},
    )
    assert resp.status_code == 202, resp.text
    return dict(resp.json())


def test_placing_a_call_asks_each_provider_in_its_own_way(phone: Phone) -> None:
    secrets = [str(v) for v in phone.credentials.values() if isinstance(v, str)] + ["KEYsimulator", "exotoken"]
    seen: dict[str, dict[str, Any]] = {}
    for provider in PROVIDERS:
        call = place(phone, provider, metadata={"campaign": "recall"})
        assert call["direction"] == "outbound" and call["status"] == "ringing" and call["provider_call_id"]
        assert call["metadata"] == {"campaign": "recall"} and "TRAI" in call["compliance_notice"]
        assert not any(s in json.dumps(call) for s in secrets)
        seen[provider] = phone.api.requests[-1]
    base, ws = phone.url, phone.server.ws_url
    twilio = seen["twilio"]["body"]
    assert seen["twilio"]["path"] == "/2010-04-01/Accounts/ACsimulator/Calls.json"
    token = twilio["Url"].rsplit("/", 1)[-1]
    assert twilio == {
        "To": "+15550100123", "From": "+15550100000", "Url": f"{base}/telephony/twilio/voice/{token}", "Method": "POST",
        "StatusCallback": f"{base}/telephony/twilio/status/{token}", "StatusCallbackMethod": "POST", "MachineDetection": "Enable",
    }  # fmt: skip
    plivo = seen["plivo"]["body"]
    assert (
        seen["plivo"]["path"] == "/v1/Account/MAsimulator/Call/"
        and plivo["to"] == "15550100123"
        and plivo["from"] == "15550100000"
    )
    assert plivo["answer_url"].startswith(f"{base}/telephony/plivo/voice/") and plivo["machine_detection"] == "hangup"
    telnyx = seen["telnyx"]["body"]
    assert seen["telnyx"]["path"] == "/v2/calls" and telnyx["connection_id"] == "sim" and telnyx["to"] == "+15550100123"
    assert (
        telnyx["stream_url"].startswith(f"{ws}/telephony/telnyx/stream/")
        and telnyx["answering_machine_detection"] == "detect"
    )
    assert telnyx["stream_bidirectional_mode"] == "rtp" and telnyx["webhook_url"].startswith(
        f"{base}/telephony/telnyx/voice/"
    )
    exotel = seen["exotel"]["body"]
    assert seen["exotel"]["path"] == "/v1/Accounts/exo/Calls/connect.json"
    assert exotel["From"] == "+15550100123" and exotel["CallerId"] == "+15550100000"
    assert exotel["Url"] == "http://my.exotel.com/exo/exoml/start_voice/77"
    assert all(r["authorization"] for r in seen.values())


def test_placing_a_call_reports_what_is_missing(phone: Phone) -> None:
    def refused(**body: Any) -> dict[str, Any]:
        resp = httpx.post(phone.url + "/api/v1/telephony/calls", json=body)
        assert resp.status_code >= 400
        return dict(resp.json()["error"])

    assert "'to' is required" in refused(provider="twilio")["message"]
    assert "E.164" in refused(provider="twilio", to="call me", agent="dialer")["message"]
    assert "unknown agent 'nobody'" in refused(provider="twilio", to="+15550100123", agent="nobody")["message"]
    assert refused(provider="vonage", to="+15550100123")["code"] == "model_not_found"
    phone.api.fail_with = 401
    error = refused(provider="twilio", to="+15550100123", agent="dialer")
    assert error["code"] == "telephony_provider_error" and "Twilio answered HTTP 401" in error["message"]
    failed = phone.calls()[0]
    assert failed["outcome"] == "failed" and "HTTP 401" in failed["error"]
    phone.api.fail_with = None
    del phone.server.core.telephony.config.providers["plivo"].values["auth_id"]
    assert (
        "telephony.providers.plivo.auth_id" in refused(provider="plivo", to="+15550100123", agent="dialer")["message"]
    )


async def test_an_outbound_call_answered_by_a_person(phone: Phone) -> None:
    call = place(phone, "twilio")
    token = phone.api.requests[-1]["body"]["Url"].rsplit("/", 1)[-1]
    link, sim = phone.link(), SIMS["twilio"]()
    sim.call_id = call["provider_call_id"]
    answered = await link.webhook(
        sim, "voice", {"CallSid": sim.call_id, "AnsweredBy": "human", "CallStatus": "in-progress"}, token
    )
    found = re.search(r'<Stream url="([^"]+)"', answered.text)
    assert found
    link.stream_url, link.token = found.group(1), token
    report = await run_scenario(phone.server.core, TALK, provider="twilio", link=link)
    record = phone.call(call["id"])
    assert report["caller"]["agent_audio_s"] > 2
    assert (
        record["direction"] == "outbound" and record["answered_by"] == "human" and record["outcome"] == "caller_hangup"
    )
    assert record["transcript"][0]["text"] == "Hello, this is a call from the clinic." and len(record["turns"]) == 1
    # Twilio's final status callback changes nothing: the first outcome stands
    await link.webhook(sim, "status", {"CallSid": sim.call_id, "CallStatus": "completed"}, token)
    assert phone.call(call["id"])["outcome"] == "caller_hangup"


async def test_voicemail_is_hung_up_on_or_left_a_message(phone: Phone) -> None:
    link = phone.link()
    # Twilio, "hangup": the answer webhook already says a machine picked up
    call = place(phone, "twilio")
    token = phone.api.requests[-1]["body"]["Url"].rsplit("/", 1)[-1]
    sim = SIMS["twilio"]()
    resp = await link.webhook(sim, "voice", {"CallSid": call["provider_call_id"], "AnsweredBy": "machine_start"}, token)
    assert "<Hangup/>" in resp.text and "<Connect>" not in resp.text
    record = phone.call(call["id"])
    assert record["outcome"] == "voicemail" and record["answered_by"] == "machine"
    assert events(record, "answered_by")[0]["detail"] == "machine_start"
    # Twilio, "leave_message": wait for the beep (DetectMessageEnd), say the message, hang up
    call = place(phone, "twilio", agent="vm")
    request = phone.api.requests[-1]["body"]
    assert request["MachineDetection"] == "DetectMessageEnd"
    token = request["Url"].rsplit("/", 1)[-1]
    sim = SIMS["twilio"]()
    sim.call_id = call["provider_call_id"]
    resp = await link.webhook(sim, "voice", {"CallSid": sim.call_id, "AnsweredBy": "machine_end_beep"}, token)
    link.stream_url, link.token = re.search(r'<Stream url="([^"]+)"', resp.text).group(1), token  # type: ignore[union-attr]
    report = await run_scenario(
        phone.server.core, {"steps": [{"wait_for_end": True}], "max_s": 15}, provider="twilio", link=link
    )
    record = phone.call(call["id"])
    assert record["outcome"] == "voicemail" and report["after"] == "hangup"
    assert [(e["kind"], e["text"]) for e in record["transcript"]] == [("voicemail_message", "Please call us back.")]
    assert report["caller"]["agent_audio_s"] == pytest.approx(len("Please call us back.") * 0.06, abs=0.3)
    # Telnyx: detection arrives as a webhook after the call was set up; "hangup" ends it with a command
    call = place(phone, "telnyx")
    sim = SIMS["telnyx"]()
    payload = {"call_control_id": call["provider_call_id"], "result": "machine"}
    event = {"data": {"event_type": "call.machine.detection.ended", "payload": payload}}
    assert (await link.webhook(sim, "voice", {"_json": event})).status_code == 200  # type: ignore[dict-item]
    assert phone.api.requests[-1]["path"] == f"/v2/calls/{call['provider_call_id']}/actions/hangup"
    assert phone.call(call["id"])["outcome"] == "voicemail"
    # Plivo: asked to hang up on machines itself; its hangup callback says why the call ended
    call = place(phone, "plivo")
    token = phone.api.requests[-1]["body"]["hangup_url"].rsplit("/", 1)[-1]
    await link.webhook(
        SIMS["plivo"](), "status", {"CallUUID": "x", "CallStatus": "completed", "Machine": "true"}, token
    )
    assert phone.call(call["id"])["outcome"] == "voicemail"


async def test_calls_that_never_connect(phone: Phone) -> None:
    link = phone.link()
    call = place(phone, "twilio")
    token = phone.api.requests[-1]["body"]["Url"].rsplit("/", 1)[-1]
    await link.webhook(
        SIMS["twilio"](), "status", {"CallSid": call["provider_call_id"], "CallStatus": "no-answer"}, token
    )
    assert phone.call(call["id"])["outcome"] == "no_answer" and phone.call(call["id"])["duration_s"] is None
    call = place(phone, "telnyx")
    payload = {"call_control_id": call["provider_call_id"], "hangup_cause": "user_busy"}
    event = {"data": {"event_type": "call.hangup", "payload": payload}}
    await link.webhook(SIMS["telnyx"](), "voice", {"_json": event})  # type: ignore[dict-item]
    assert phone.call(call["id"])["outcome"] == "busy"
    call = place(phone, "plivo")
    hung = httpx.post(f"{phone.url}/api/v1/telephony/calls/{call['id']}/hangup").json()
    assert hung["outcome"] == "canceled"


# -- the operator's API ---------------------------------------------------------------------------------------
def test_overview_gives_the_urls_to_paste_and_no_secrets(phone: Phone) -> None:
    doc = httpx.get(phone.url + "/api/v1/telephony").json()
    assert doc["public_base_url"] == phone.url and doc["validate_signatures"] and doc["default_agent"] == "desk"
    by_name = {p["name"]: p for p in doc["providers"]}
    assert list(by_name) == list(PROVIDERS) and all(p["exposed"] and p["configured"] for p in by_name.values())
    assert by_name["twilio"]["webhooks"][0] == {
        "name": "A call comes in", "url": phone.url + "/telephony/twilio/voice", "method": "POST",
        "where": by_name["twilio"]["webhooks"][0]["where"],
    }  # fmt: skip
    assert by_name["plivo"]["webhooks"][0]["url"] == phone.url + "/telephony/plivo/voice"
    assert by_name["telnyx"]["webhooks"][0]["url"] == phone.url + "/telephony/telnyx/voice"
    exotel_url = by_name["exotel"]["webhooks"][0]["url"]
    assert (
        exotel_url == phone.server.ws_url.replace("://", "://simulator:<stream_password>@") + "/telephony/exotel/stream"
    )
    assert (
        by_name["twilio"]["settings"]["auth_token"] == "***"
        and by_name["twilio"]["settings"]["account_sid"] == "ACsimulator"
    )
    assert by_name["telnyx"]["ends_by"] == "api" and by_name["exotel"]["protection"] == "basic"
    assert by_name["exotel"]["machine_detection"] is False and by_name["twilio"]["can_place_calls"]
    text = json.dumps(doc) + json.dumps(httpx.get(phone.url + "/api/v1/info").json())
    for secret in [v for v in phone.credentials.values() if isinstance(v, str) and v != "simulator"]:
        assert secret not in text
    assert "DND" in doc["compliance_notice"] and len(doc["agents"]) == len(AGENTS)
    capability = httpx.get(phone.url + "/api/v1/info").json()["capabilities"]["telephony"]
    assert capability["providers"] == list(PROVIDERS) and capability["exposed"] == list(PROVIDERS)


def test_agents_and_numbers_through_the_api(phone: Phone, settings: Settings) -> None:
    api = phone.url + "/api/v1/telephony"
    body = {
        "instructions": "Take orders.",
        "greeting": "Pizza!",
        "max_duration_s": 90,
        "tools": [{"name": "order", "url": "https://x.example/o"}],
    }
    saved = httpx.put(api + "/agents/pizza", json=body).json()
    assert (
        saved["name"] == "pizza"
        and saved["editable"]
        and saved["source"] == "stored"
        and saved["max_duration_s"] == 90.0
    )
    assert httpx.get(api + "/agents/pizza").json()["tools"][0]["name"] == "order"
    assert "pizza" in [a["name"] for a in httpx.get(api + "/agents").json()["agents"]]
    assert (
        httpx.put(api + "/agents/pizza", json={"colour": "red"})
        .json()["error"]["message"]
        .endswith("unknown fields: colour")
    )
    from_config = httpx.put(api + "/agents/desk", json={"greeting": "x"})
    assert from_config.status_code == 400 and "comes from the configuration" in from_config.json()["error"]["message"]
    assert httpx.put(api + "/numbers/+1 415 555 0199", json={"agent": "pizza"}).status_code == 200
    assert httpx.put(api + "/numbers/+14155550198", json={"agent": "nobody"}).status_code == 400
    numbers = httpx.get(api + "/numbers").json()
    assert {"number": "+1 415 555 0199", "agent": "pizza", "source": "stored", "editable": True} in numbers["numbers"]
    assert numbers["default_agent"] == "desk"
    stored = json.loads((settings.data_dir / "telephony.json").read_text(encoding="utf-8"))
    assert stored["numbers"] == {"+1 415 555 0199": "pizza"} and stored["agents"]["pizza"]["greeting"] == "Pizza!"
    mode = (settings.data_dir / "telephony.json").stat().st_mode & 0o777
    assert sys.platform == "win32" or mode == 0o600  # Windows has no mode bits
    # a restart reads them back
    again = TelephonyConfig(settings)
    found = again.agent_for("14155550199")
    assert found is not None and found.name == "pizza" and found.source == "stored"
    assert httpx.delete(api + "/numbers/+14155550199").status_code == 204
    assert httpx.delete(api + "/agents/pizza").status_code == 204
    assert httpx.get(api + "/agents/pizza").status_code == 404 and httpx.delete(api + "/agents/desk").status_code == 400
    assert httpx.get(api + "/calls/call_nope").status_code == 404


def test_simulate_endpoint_runs_a_call_inside_the_server(phone: Phone) -> None:
    listed = httpx.get(phone.url + "/api/v1/telephony/scenarios").json()["scenarios"]
    assert [s["name"] for s in listed] == ["basic", "barge-in", "silence", "dtmf", "noisy"]
    body = {
        "agent": "desk",
        "provider": "exotel",
        "scenario": {**TALK, "expect": [{"type": "agent_says", "contains": "you said"}]},
    }
    report = httpx.post(phone.url + "/api/v1/telephony/simulate", json=body, timeout=60).json()
    assert report["passed"] and report["provider"] == "exotel" and report["agent"] == "desk"
    assert report["caller"]["response_ms"]["p50"] > 0 and report["record"]["metadata"] == {"simulated": True}
    assert phone.calls()[0]["id"] == report["call_id"]
    bad = httpx.post(phone.url + "/api/v1/telephony/simulate", json={"scenario": "karaoke"})
    assert bad.status_code == 400 and "unknown scenario 'karaoke'" in bad.json()["error"]["message"]


async def test_the_caller_can_speak_with_tts_or_from_a_file(phone: Phone, tmp_path: Path) -> None:
    from conftest import wav_bytes

    recording = tmp_path / "caller.wav"
    recording.write_bytes(wav_bytes(1.5, 16000))
    scenario = {
        "steps": [
            {"wait_for_agent": True},
            {"say": "I would like to book a table for two."},  # spoken by the server's text-to-speech
            {"wait_for_agent": True},
            {"audio": str(recording)},  # a 16 kHz recording, resampled to the line's 8 kHz
            {"silence": 1.5},
            {"hangup": True},
        ],
        "expect": [{"type": "turns_min", "value": 2}, {"type": "caller_heard_as", "contains": "the quick"}],
    }
    report = await phone.run(scenario, "telnyx")
    assert report["passed"], report["checks"]
    spoken, played = report["caller"]["utterances"]
    assert spoken["label"].startswith("I would like") and spoken["seconds"] == pytest.approx(37 * 0.06, abs=0.1)
    assert played["seconds"] == pytest.approx(1.5, abs=0.01) and all(u["response_ms"] for u in (spoken, played))
    with pytest.raises(Exception, match="needs 'say', 'audio' or 'speech_s'"):
        await phone.run({"steps": [{"whistle": 1}]})


@pytest.mark.parametrize("name", ["basic", "barge-in", "silence", "dtmf", "noisy"])
async def test_built_in_scenarios_pass_with_the_fake_engines(phone: Phone, name: str) -> None:
    phone.server.core.telephony.config.default_agent = "quiet" if name == "silence" else "desk"
    # the fake engines' energy detector takes loud noise for speech: 40 dB keeps the noisy scenario honest
    report = await run_scenario(phone.server.core, name, noise_db=40 if name == "noisy" else None)
    assert report["passed"], report["checks"]


def test_cli_simulate_and_status(
    settings: Settings,
    llm: FakeLLM,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from bolo.cli import _print_report, main
    from bolo.telephony.simulator import simulate_with_server

    settings.llm_base_url, settings.llm_model = llm.url, "fake-llm"
    settings.telephony = {
        "agents": {"desk": {"greeting": "Hello."}},
        "providers": {"twilio": {"auth_token": "real-secret"}},
    }
    # the whole provider path (signed webhook, answer document, WebSocket) against a private server
    (report,) = simulate_with_server(
        settings, [TALK | {"expect": [{"type": "turns_min", "value": 1}]}], provider="telnyx"
    )
    assert report["passed"] and report["agent"] == "desk" and report["provider"] == "telnyx"
    assert report["record"]["transcript"][0]["text"] == "Hello."
    _print_report(report)
    out = capsys.readouterr().out
    assert "custom: PASS  (telnyx, agent desk" in out and "caller heard the agent after: p50" in out
    assert "server breakdown (p50): endpoint" in out and "[ok] turns_min: 1 answered turn(s)" in out
    with pytest.raises(Exception, match="unknown agent 'nobody'"):
        simulate_with_server(settings, ["basic"], agent="nobody")
    monkeypatch.setenv("BOLO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(
        "BOLO_TELEPHONY", json.dumps({**settings.telephony, "public_base_url": "https://voice.example.org"})
    )
    assert main(["telephony", "status"]) == 0
    out = capsys.readouterr().out
    assert "Twilio: ready" in out and "A call comes in: POST https://voice.example.org/telephony/twilio/voice" in out
    assert "Plivo: closed (telephony.providers.plivo.auth_token is not set)" in out and "real-secret" not in out
    assert "agents: desk" in out


# -- real providers (opt in) --------------------------------------------------------------------------------
@pytest.mark.real
def test_real_outbound_call() -> None:
    """Place one real call and wait for it to end. Costs money and rings a phone: opt in with
    BOLO_TEST_REAL=1 and BOLO_TEST_TELEPHONY=<provider>, a configuration file with the provider's credentials,
    an agent and public_base_url (BOLO_CONFIG), a running ``bolo serve`` reachable at that address
    (BOLO_TEST_TELEPHONY_SERVER, default http://127.0.0.1:8000) and the number to ring (BOLO_TEST_TELEPHONY_TO)."""
    provider, to = os.environ.get("BOLO_TEST_TELEPHONY"), os.environ.get("BOLO_TEST_TELEPHONY_TO")
    if not provider or not to:
        pytest.skip("set BOLO_TEST_TELEPHONY=<provider> and BOLO_TEST_TELEPHONY_TO=<number> to place a real call")
    server = os.environ.get("BOLO_TEST_TELEPHONY_SERVER", "http://127.0.0.1:8000")
    headers = {"Authorization": f"Bearer {os.environ['BOLO_API_KEY']}"} if os.environ.get("BOLO_API_KEY") else {}
    body = {"provider": provider, "to": to, "agent": os.environ.get("BOLO_TEST_TELEPHONY_AGENT")}
    resp = httpx.post(server + "/api/v1/telephony/calls", json=body, headers=headers, timeout=30)
    assert resp.status_code == 202, resp.text
    call_id = resp.json()["id"]
    deadline = time.time() + 180
    record: dict[str, Any] = {}
    while time.time() < deadline:
        record = httpx.get(f"{server}/api/v1/telephony/calls/{call_id}", headers=headers, timeout=30).json()
        if record["status"] == "ended":
            break
        time.sleep(1)
    assert record.get("status") == "ended", record.get("timeline")
    assert record["outcome"] not in ("failed", "error"), record


@pytest.mark.real
def test_real_webhook_signature() -> None:
    """Check bolo's validation against a request captured from the real provider: BOLO_TEST_TELEPHONY_CAPTURE
    names a JSON file {"provider", "url", "method", "headers", "form", "body", "secret"} (``secret`` is the auth
    token, or the Telnyx public key)."""
    path = os.environ.get("BOLO_TEST_TELEPHONY_CAPTURE")
    if not path:
        pytest.skip("set BOLO_TEST_TELEPHONY_CAPTURE to a captured webhook to check its signature")
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    headers = {str(k).lower(): str(v) for k, v in doc["headers"].items()}
    form, secret = doc.get("form") or {}, doc["secret"]
    if doc["provider"] == "twilio":
        assert security.verify_twilio(secret, doc["url"], form, headers.get("x-twilio-signature"))
    elif doc["provider"] == "plivo":
        signatures = [headers.get("x-plivo-signature-v3"), headers.get("x-plivo-signature-ma-v3")]
        assert security.verify_plivo(
            secret, doc["url"], form, headers.get("x-plivo-signature-v3-nonce"), signatures, doc.get("method", "POST")
        )
    elif doc["provider"] == "telnyx":
        stamp = headers.get("telnyx-timestamp")
        assert security.verify_telnyx(
            secret, stamp, str(doc["body"]).encode(), headers.get("telnyx-signature-ed25519"), now=float(stamp or 0)
        )
    else:
        pytest.fail(f"no signature to check for {doc['provider']}")
