"""Live pipelines and triggers (bolo.pipelines.live, bolo.pipelines.triggers): a phone call answered by a pipeline
end to end on the fake engines and a fake provider, the conversation simulator, and runs started by the end of a
call, the end of a Realtime session, an upload and a watched folder."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from bolo.config import Settings
from bolo.core import Bolo
from bolo.pipelines import validate
from bolo.realtime import RealtimeError, RealtimeSession
from conftest import LiveServer, wav_bytes
from fake_llm import FakeLLM
from test_telephony import AGENTS, TRANSFER_TO, Phone, events, make_phone

ANGRY = {"value": "yes"}

TRIAGE = {
    "id": "triage",
    "mode": "live",
    "steps": [
        {"id": "mood", "type": "decision", "options": {"question": "Is the caller angry?"}},
        {"id": "escalate", "type": "condition", "options": {
            "if": {"value": "{{steps.mood.answer}}", "equals": "yes"},
            "then": [{"id": "to_person", "type": "transfer", "options": {"say": "Putting you through."}}],
        }},
        {"id": "answer", "type": "respond", "options": {"instructions": "Turn {{input.turn}}: be brief."}},
    ],
}  # fmt: skip


def llm_reply(angry: dict[str, str]) -> Any:
    """Decisions (JSON, not streamed) answer from ``angry`` or from the department words; turns are echoed."""

    def reply(body: dict[str, Any]) -> tuple[str, None]:
        system = next((m["content"] for m in body["messages"] if m["role"] == "system"), "")
        user = body["messages"][-1]["content"] if body["messages"] else ""
        if "Answer with one of" in system:
            text = str(user)
            if "department" in text:
                return json.dumps({"answer": "billing" if "bill" in text else "other"}), None
            mood = "yes" if "furious" in text else angry["value"]
            return json.dumps({"answer": mood, "confidence": 0.9, "reason": "tone"}), None
        last = next((m["content"] for m in reversed(body["messages"]) if m["role"] == "user"), "")
        return f"You said: {last}. Goodbye for now.", None

    return reply


@pytest.fixture
def triage_phone(settings: Settings, llm: FakeLLM) -> Iterator[Phone]:
    agents = json.loads(json.dumps(AGENTS))
    agents["triage"] = {"pipeline_id": "triage", "transfer_number": TRANSFER_TO, "silence_timeout_s": 0}
    made, api = make_phone(settings, llm, agents=agents)
    made.server.core.pipelines.create(TRIAGE)
    made.server.core.telephony.config.default_agent = "triage"
    llm.reply = llm_reply(ANGRY)  # type: ignore[method-assign]
    try:
        yield made
    finally:
        made.server.stop()
        api.stop()


async def test_an_angry_caller_is_transferred_by_the_pipeline(triage_phone: Phone) -> None:
    ANGRY["value"] = "yes"
    report = await triage_phone.run({"steps": [{"speech_s": 1.5}, {"wait_for_end": True}], "max_s": 20})
    record = report["record"]
    assert record["outcome"] == "transferred" and record["transfer_to"] == TRANSFER_TO, record["timeline"]
    assert events(record, "pipeline_action")[0]["action"] == "transfer"
    assert any(t["text"] == "Putting you through." for t in record["transcript"])
    runs = triage_phone.server.core.pipelines.store.runs("triage")
    assert runs and runs[0]["trigger"] == "turn" and runs[0]["actions"][-1]["action"] == "transfer"
    assert runs[0]["input"]["session"]["call_id"] == record["id"]


async def test_a_calm_caller_gets_the_language_model_answer(triage_phone: Phone) -> None:
    ANGRY["value"] = "no"
    report = await triage_phone.run({"steps": [{"speech_s": 1.5}, {"wait_for_agent": True}, {"hangup": True}],
                                     "max_s": 20})  # fmt: skip
    record = report["record"]
    said = [t["text"] for t in record["transcript"] if t["speaker"] == "agent"]
    assert any(s.startswith("You said: the quick") for s in said), record["transcript"]
    streamed = triage_phone.asked()
    assert "Turn 0: be brief." in streamed["messages"][0]["content"]  # the respond step's instructions
    run = triage_phone.server.core.pipelines.store.runs("triage")[0]
    assert [s["status"] for s in run["steps"]] == ["ok", "ok", "skipped", "ok"]
    assert run["actions"][0]["action"] == "respond" and run["actions"][0]["text"].startswith("You said")


def test_live_pipelines_are_checked(core: Bolo) -> None:
    report = validate({"mode": "live", "steps": [{"type": "stt"}, {"type": "reply"}]}, core)
    assert "cannot run per turn" in report.errors[0].message
    report = validate({"steps": [{"type": "hangup"}]}, core)
    assert "only runs in a live pipeline" in report.errors[0].message
    report = validate({**TRIAGE, "triggers": [{"type": "call_end"}]}, core)
    assert "takes no triggers" in report.errors[0].message


async def test_a_session_takes_only_a_saved_live_pipeline(core: Bolo) -> None:
    async def nowhere(_event: dict[str, Any]) -> None:
        return None

    core.pipelines.create({"id": "file-one", "steps": [{"type": "stt"}]})
    core.pipelines.create(TRIAGE)
    session = RealtimeSession(core, nowhere)
    with pytest.raises(RealtimeError, match="not a live pipeline"):
        session.configure({"type": "realtime", "pipeline_id": "file-one"})
    with pytest.raises(RealtimeError, match="no pipeline 'nope'"):
        session.configure({"type": "realtime", "pipeline_id": "nope"})
    session.configure({"type": "realtime", "pipeline_id": "triage"})
    assert session.session_object()["pipeline_id"] == "triage"


# -- the simulator ------------------------------------------------------------------------------------------
def test_simulator_plays_typed_turns_through_a_template(agent: LiveServer) -> None:
    agent.llm.reply = llm_reply({"value": "no"})  # type: ignore[attr-defined]
    templates = httpx.get(agent.url + "/api/v1/pipelines/templates").json()["data"]
    support = next(t for t in templates if t["id"] == "support-line")
    r = httpx.post(agent.url + "/v1/pipelines/simulate", data={
        "pipeline": json.dumps(support), "turns": json.dumps(["my bill is wrong", "I am furious, get me a human",
                                                              "hello?"]),
        "instructions": "You work at Acme.",
    }, timeout=60)  # fmt: skip
    assert r.status_code == 200, r.text
    sim = r.json()
    first, second = sim["turns"][0], sim["turns"][1]
    assert first["run"]["steps"][-1]["id"] == "answer" and first["agent"].startswith("You said: my bill is wrong")
    department = next(s for s in first["run"]["steps"] if s["id"] == "department")
    assert department["output"]["answer"] == "billing"
    asked = next(q for q in agent.llm.requests if "billing desk" in str(q["messages"][0].get("content")))  # type: ignore[attr-defined]
    assert asked["messages"][0]["content"].startswith("You work at Acme.")
    assert second["run"]["actions"] == [{"action": "transfer", "to": None}]
    assert second["agent"].startswith("I am sorry about this")  # said before the transfer
    assert sim["ended"]["action"] == "transfer" and len(sim["turns"]) == 2  # nothing after the transfer
    bad = httpx.post(agent.url + "/v1/pipelines/simulate", data={"pipeline": json.dumps({"steps": [{"type": "stt"}]}),
                                                                 "turns": "[\"x\"]"})  # fmt: skip
    assert bad.status_code == 400 and "live pipeline" in bad.json()["error"]["message"]
    r = httpx.post(agent.url + "/v1/pipelines/run", data={"pipeline": json.dumps(support), "text": "x"})
    assert r.status_code == 400 and "simulate" in r.json()["error"]["message"]


def test_simulator_cuts_a_recording_into_turns(agent: LiveServer) -> None:
    live = {"mode": "live", "steps": [{"type": "template", "options": {"template": "heard: {{input.text}}"}},
                                      {"type": "reply"}]}  # fmt: skip
    files = {"file": ("call.wav", wav_bytes(5.0, gap_at=(2.0,)), "audio/wav")}
    r = httpx.post(agent.url + "/v1/pipelines/simulate", data={"pipeline": json.dumps(live)}, files=files, timeout=60)
    assert r.status_code == 200, r.text
    turns = r.json()["turns"]
    assert len(turns) == 2 and all(t["agent"].startswith("heard: the quick") for t in turns)


# -- triggers -----------------------------------------------------------------------------------------------
def wait_run(core: Bolo, pipeline: str, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        runs = [r for r in core.pipelines.store.runs(pipeline) if r["status"] not in ("queued", "running")]
        if runs:
            return runs[0]
        time.sleep(0.1)
    raise AssertionError(f"no finished run of {pipeline}")


async def test_the_end_of_a_call_runs_its_pipelines(phone: Phone) -> None:
    core = phone.server.core
    core.pipelines.create({
        "id": "after", "triggers": [{"type": "call_end", "agents": ["desk"]}],
        "steps": [{"type": "template", "options": {"template": "{{input.call.agent}} call: {{input.text}}"}}],
    })  # fmt: skip
    core.pipelines.create({"id": "other", "triggers": [{"type": "call_end", "agents": ["quiet"]}],
                           "steps": [{"type": "template", "options": {"template": "x"}}]})  # fmt: skip
    report = await phone.run({"steps": [{"wait_for_agent": True}, {"speech_s": 1.5}, {"wait_for_agent": True},
                                        {"hangup": True}]})  # fmt: skip
    run = wait_run(core, "after")
    assert run["trigger"] == "call_end" and run["outputs"]["text"].startswith("desk call: Agent: Hello.")
    assert run["input"]["call"]["id"] == report["call_id"]
    assert not core.pipelines.store.runs("other")


def test_session_end_upload_and_folder_triggers(settings: Settings, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    settings.pipeline_watch_dirs = [str(tmp_path)]
    settings.pipeline_watch_interval_s = 0.2
    srv = LiveServer(settings)
    srv.start()
    core = srv.core
    try:
        core.pipelines.create({"id": "ended", "triggers": [{"type": "session_end"}],
                               "steps": [{"type": "template", "options": {"template": "[{{input.text}}]"}}]})  # fmt: skip

        async def end() -> list[str]:
            return core.pipelines.triggers.session_ended({"id": "s1", "transport": "websocket"}, "User: hi")

        assert len(srv.call(end())) == 1  # type: ignore[arg-type]
        run = wait_run(core, "ended")
        assert run["outputs"] == {"text": "[User: hi]"} and run["input"]["session"]["id"] == "s1"

        core.pipelines.create({"id": "after-upload", "steps": [{"type": "template", "options": {
            "template": "{{input.filename}}: {{input.text}}"}}]})  # fmt: skip
        r = httpx.post(srv.url + "/v1/audio/transcriptions", data={"model": "fake-stt", "pipeline": "after-upload"},
                       files={"file": ("memo.wav", wav_bytes(2.0), "audio/wav")}, timeout=30)  # fmt: skip
        assert r.status_code == 200 and r.headers["x-bolo-pipeline-run"].startswith("run_")
        run = wait_run(core, "after-upload")
        assert run["trigger"] == "upload" and run["outputs"]["text"].startswith("memo.wav: the quick")
        r = httpx.post(srv.url + "/v1/audio/transcriptions", data={"model": "fake-stt", "pipeline": "nope"},
                       files={"file": ("memo.wav", wav_bytes(1.0), "audio/wav")})  # fmt: skip
        assert r.status_code == 404

        outside = validate({"steps": [{"type": "stt"}], "triggers": [{"type": "folder", "path": "/etc"}]}, core)
        assert "not under a folder the server may watch" in outside.errors[0].message
        core.pipelines.create({"id": "inbox", "triggers": [{"type": "folder", "path": str(inbox), "pattern": "*.wav"}],
                               "steps": [{"type": "stt", "model": "fake-stt"}]})  # fmt: skip
        (inbox / "a.wav").write_bytes(wav_bytes(2.0))
        (inbox / "notes.txt").write_text("not audio")
        run = wait_run(core, "inbox")
        assert run["trigger"] == "folder" and run["input"]["filename"] == "a.wav" and run["status"] == "ok"
        assert (inbox / "done" / "a.wav").exists() and not (inbox / "a.wav").exists()
        assert (inbox / "notes.txt").exists()
    finally:
        srv.stop()


@pytest.fixture
def phone(settings: Settings, llm: FakeLLM) -> Iterator[Phone]:
    made, api = make_phone(settings, llm)
    try:
        yield made
    finally:
        made.server.stop()
        api.stop()
