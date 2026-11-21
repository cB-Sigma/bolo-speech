"""Live decisions (bolo.decision_hook): questions after each user turn of a Realtime session or a phone call, their
events, the signed webhook, and the rules that end, transfer or tag a call. With the fake decision model."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from bolo.config import Settings
from bolo.decision_hook import matched, parse_hook
from bolo.errors import InvalidRequest
from bolo.pipelines.webhooks import verify
from conftest import FakeLLM, LiveServer, speechlike
from test_realtime import raw, recv_until
from test_telephony import TRANSFER_TO, events, make_phone

HOOK = {
    "model": "fake-decision",
    "questions": {
        "escalate": {"type": "yes_no", "question": "Does the caller ask for a person?"},
        "department": {"type": "choice", "question": "Which department?", "options": ["billing", "technical"]},
    },
    "actions": [
        {"question": "department", "answer": "billing", "do": "tag", "value": "billing"},
        {"question": "escalate", "answer": True, "min_probability": 0.8, "do": "transfer"},
    ],
}


class Receiver:
    """Records the webhooks it is sent."""

    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.received.append({"headers": dict(self.headers.items()), "body": body})
                self.send_response(204)
                self.end_headers()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/hook"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def receiver() -> Iterator[Receiver]:
    r = Receiver()
    yield r
    r.stop()


def test_hooks_are_checked_and_rules_match() -> None:
    hook = parse_hook(HOOK)
    assert hook is not None and hook.model == "fake-decision" and hook.audio and len(hook.rules) == 2
    assert hook.questions["escalate"]["type"] == "noul" and hook.to_dict()["questions"] == HOOK["questions"]
    assert parse_hook(None) is None and parse_hook({}) is None
    yes = {"escalate": {"type": "yes_no", "answer": True, "probability": 0.9},
           "department": {"type": "choice", "answer": "technical", "probabilities": {"billing": 0.2, "technical": 0.8},
                          "confidence": 0.6}}  # fmt: skip
    assert [r.do for r in matched(hook, yes)] == ["transfer"]
    unsure = {**yes, "escalate": {"type": "yes_no", "answer": True, "probability": 0.7}}
    assert matched(hook, unsure) == []  # below min_probability
    scored = parse_hook({"questions": {"s": {"type": "score", "question": "?", "levels": ["a", "b", "c"]}},
                         "actions": [{"question": "s", "min_score": 1.5, "do": "end_call"}]})  # fmt: skip
    assert scored is not None
    level = {"s": {"type": "score", "score": 1.6, "level": 2, "label": "c", "probabilities": [0.1, 0.2, 0.7]}}
    assert [r.do for r in matched(scored, level)] == ["end_call"]
    assert matched(scored, {"s": {**level["s"], "score": 1.2}}) == []
    for bad in (
        "x", {"questions": {}}, {**HOOK, "extra": 1},
        {**HOOK, "actions": [{"question": "nope", "do": "tag", "value": "x"}]},
        {**HOOK, "actions": [{"question": "escalate", "do": "launch"}]},
        {**HOOK, "actions": [{"question": "escalate", "do": "tag"}]},
        {**HOOK, "actions": [{"question": "escalate", "answer": "yes", "do": "end_call"}]},
        {**HOOK, "actions": [{"question": "department", "answer": "sales", "do": "end_call"}]},
        {**HOOK, "actions": [{"question": "escalate", "min_score": 1, "do": "end_call"}]},
    ):  # fmt: skip
        with pytest.raises(InvalidRequest):
            parse_hook(bad)


@pytest.fixture
def hooked(settings: Settings, llm: FakeLLM, receiver: Receiver) -> Iterator[LiveServer]:
    settings.llm_base_url, settings.llm_model = llm.url, "fake-llm"
    settings.decision_webhook_url, settings.webhook_secret = receiver.url, "hook-secret"
    srv = LiveServer(settings)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


async def test_a_realtime_session_gets_its_answers_after_each_turn(hooked: LiveServer, receiver: Receiver) -> None:
    ws = await raw(hooked)
    try:
        await recv_until(ws, "session.created")
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "decisions": HOOK,
            "audio": {"input": {"turn_detection": None, "transcription": {"model": "fake-stt"}}}}}))  # fmt: skip
        updated = await recv_until(ws, "session.updated")
        assert updated[-1]["session"]["decisions"]["actions"][1]["do"] == "transfer"
        pcm = (speechlike(2.0, 24000) * 32767).astype("<i2").tobytes()
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        got = await recv_until(ws, "conversation.item.decisions.completed")
        done = got[-1]
        heard = next(e for e in got if e["type"] == "conversation.item.input_audio_transcription.completed")
        assert done["item_id"] == heard["item_id"] and done["model"] == "fake-decision"
        assert done["answers"]["escalate"] == {"type": "yes_no", "answer": True, "probability": 0.9}
        assert done["answers"]["department"]["answer"] == "billing"
        assert [a["do"] for a in done["actions"]] == ["tag", "transfer"]  # reported; a session has no call to move
        # a bad hook is refused with the parameter named
        await ws.send(json.dumps({"type": "session.update", "session": {"decisions": {"questions": "{"}}}))
        error = (await recv_until(ws, "error"))[-1]
        assert error["error"]["param"] == "session.decisions.questions"
    finally:
        await ws.close()
    for _ in range(100):
        if receiver.received:
            break
        await asyncio.sleep(0.05)
    sent = receiver.received[0]
    body = sent["body"]
    assert verify("hook-secret", body, sent["headers"]["X-Bolo-Signature"])
    assert sent["headers"]["X-Bolo-Event"] == "decisions"
    doc = json.loads(body)
    assert doc["type"] == "conversation.item.decisions.completed" and doc["transcript"].startswith("the quick")
    assert doc["answers"]["escalate"]["answer"] is True and doc["session_id"]


@pytest.mark.parametrize(
    ("actions", "outcome"),
    [
        ([{"question": "escalate", "answer": True, "do": "transfer"}], "transferred"),
        ([{"question": "escalate", "answer": True, "do": "end_call"}], "decision_end"),
    ],
)
async def test_rules_move_a_phone_call(settings: Settings, llm: FakeLLM, actions: list[Any], outcome: str) -> None:
    hook = {**HOOK, "actions": [HOOK["actions"][0], *actions]}
    phone, api = make_phone(settings, llm)
    try:
        agents = phone.server.core.telephony.config.agents
        from bolo.telephony.config import AgentConfig

        agents["desk"] = AgentConfig.from_dict("desk", {
            "greeting": "Hello.", "silence_timeout_s": 0, "transfer_number": TRANSFER_TO, "decisions": hook,
        })  # fmt: skip
        report = await phone.run({"steps": [{"wait_for_agent": True}, {"speech_s": 2.0}, {"wait_for_end": True}],
                                  "max_s": 20})  # fmt: skip
        record = report["record"]
        assert record["outcome"] == outcome, record["timeline"]
        decided = events(record, "decisions")[0]
        assert decided["answers"]["escalate"]["answer"] is True and record["metadata"]["tags"] == ["billing"]
        if outcome == "transferred":
            assert record["transfer_to"] == TRANSFER_TO
    finally:
        phone.server.stop()
        api.stop()


def test_agents_check_their_hooks() -> None:
    from bolo.telephony.config import AgentConfig

    agent = AgentConfig.from_dict("a", {"decisions": HOOK, "decision_webhook_url": "https://example.com/x"})
    assert agent.decisions == HOOK and agent.decision_webhook_url == "https://example.com/x"
    with pytest.raises(InvalidRequest):
        AgentConfig.from_dict("a", {"decisions": {"questions": {}}})
    with pytest.raises(InvalidRequest):
        AgentConfig.from_dict("a", {"decision_webhook_url": "ftp://x"})
