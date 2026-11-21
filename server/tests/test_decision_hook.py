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
from bolo.core import Bolo
from bolo.decision_hook import definition, parse_hook
from bolo.decision_hook import run as run_hook
from bolo.errors import InvalidRequest
from bolo.pipelines.definition import validate
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


def test_hooks_are_checked_and_become_live_pipelines() -> None:
    hook = parse_hook(HOOK)
    assert hook is not None and hook.model == "fake-decision" and hook.audio and len(hook.rules) == 2
    assert hook.questions["escalate"]["type"] == "noul" and hook.to_dict()["questions"] == HOOK["questions"]
    assert parse_hook(None) is None and parse_hook({}) is None
    pipe = definition(hook)
    assert pipe["mode"] == "live" and validate(pipe).valid
    first, tag, transfer = pipe["steps"]
    assert first["type"] == "decisions" and first["options"]["questions"] == HOOK["questions"]
    assert first["inputs"] == {"text": "{{input.text}}", "audio": "{{input.audio}}"}
    assert tag == {"id": "rule-0", "type": "tag", "options": {"value": "billing"},
                   "when": {"value": "{{steps.decisions.answers.department.answer}}", "equals": "billing"}}  # fmt: skip
    assert transfer["type"] == "transfer" and transfer["when"]["all"][1] == {
        "value": "{{steps.decisions.probabilities.escalate}}", "gte": 0.8,
    }  # fmt: skip
    # rules that end the call run last, whatever their order in the hook
    ending_first = parse_hook({**HOOK, "actions": list(reversed(HOOK["actions"]))})
    assert ending_first is not None and [s["type"] for s in definition(ending_first)["steps"]] == [
        "decisions", "tag", "transfer",
    ]  # fmt: skip
    scored = parse_hook({"questions": {"s": {"type": "score", "question": "?", "levels": ["a", "b", "c"]}},
                         "audio": False, "actions": [{"question": "s", "min_score": 1.5, "do": "end_call"},
                                                     {"question": "s", "answer": 2, "do": "notify"}]})  # fmt: skip
    assert scored is not None
    steps = definition(scored)["steps"]
    assert "audio" not in steps[0]["inputs"] and steps[1]["when"] == {"value": "{{steps.decisions.answers.s.level}}",
                                                                      "equals": 2}  # fmt: skip
    assert steps[2] == {"id": "rule-0", "type": "hangup",
                        "when": {"value": "{{steps.decisions.answers.s.score}}", "gte": 1.5}}  # fmt: skip
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


class Recorder:
    """A live target that records the actions it is asked for."""

    def __init__(self) -> None:
        self.actions: list[dict[str, Any]] = []

    async def say(self, text: str) -> str:
        raise AssertionError("live decisions never speak")

    async def respond(self, instructions: str | None) -> str:
        raise AssertionError("live decisions never answer")

    async def action(self, kind: str, **fields: Any) -> None:
        self.actions.append({"action": kind, **fields})


async def test_a_hook_runs_through_the_live_pipeline_runner(core: Bolo) -> None:
    hook = parse_hook(HOOK)
    assert hook is not None
    target = Recorder()
    result = await run_hook(core, hook, target, text="I want a person", audio=None, session={"id": "s"}, turn=0)
    assert result.model == "fake-decision" and result.answers["escalate"] == {
        "type": "yes_no", "answer": True, "probability": 0.9,
    }  # fmt: skip
    assert [r.do for r in result.matched] == ["tag", "transfer"]
    assert [(a["action"], a.get("value")) for a in target.actions] == [("tag", "billing"), ("transfer", None)]
    assert result.run.status == "ok" and result.run.actions[-1]["action"] == "transfer"
    unsure = parse_hook({**HOOK, "actions": [{**HOOK["actions"][1], "min_probability": 0.95}]})
    assert unsure is not None
    again = await run_hook(core, unsure, Recorder(), text="x", audio=None, session={"id": "s"}, turn=1)
    assert again.matched == []  # 0.9 is below 0.95: the rule's step is skipped


async def test_the_decisions_step_in_any_pipeline(core: Bolo) -> None:
    definition_ = {"steps": [{"id": "ask", "type": "decisions", "model": "fake-decision",
                              "options": {"questions": {"urgent": {"type": "yes_no", "question": "Urgent?"},
                                                        "level": {"type": "score", "question": "How angry?",
                                                                  "levels": ["calm", "upset", "furious"]}}}}],
                   "outputs": {"answers": "{{steps.ask.answers}}", "p": "{{steps.ask.probabilities}}"}}  # fmt: skip
    run = core.pipelines.runner.prepare(definition_, text="The server is down")
    await core.pipelines.run(run)
    assert run.status == "ok", run.error
    assert run.outputs is not None and run.outputs["answers"]["urgent"]["answer"] is True
    assert run.outputs["p"] == {"urgent": 0.9, "level": pytest.approx(1 / 3, abs=1e-3)}
    assert run.outputs["answers"]["level"]["label"] == "upset"


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
        assert [a["do"] for a in done["actions"]] == ["tag", "transfer"]
        # the actions went out as a live pipeline's do, marked as the session's live decisions
        acted = [e for e in got if e["type"] == "pipeline.action"]
        assert [(e["action"], e.get("value"), e["source"]) for e in acted] == [
            ("tag", "billing", "decisions"), ("transfer", None, "decisions"),
        ]  # fmt: skip
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
