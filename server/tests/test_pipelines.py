"""Pipelines (bolo.pipelines, docs/pipelines.md): validation, the runner on the fake engines, the HTTP API with
signed callbacks and webhooks, the templates and the CLI."""

from __future__ import annotations

import http.server
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from bolo.cli import main
from bolo.config import Settings
from bolo.core import Bolo
from bolo.pipelines import Audio, PipelineStore, validate, webhooks
from bolo.pipelines.conditions import evaluate
from bolo.pipelines.runner import as_text
from bolo.pipelines.templates import all_templates
from conftest import LiveServer, speechlike, two_speakers, wav_bytes
from fake_llm import FakeLLM


class Hook:
    """A webhook receiver: records each request (headers and raw body)."""

    def __init__(self, status: int = 200) -> None:
        self.requests: list[tuple[dict[str, str], bytes]] = []
        self.status = status
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                owner.requests.append(({k.lower(): v for k, v in self.headers.items()}, body))
                out = json.dumps({"received": True}).encode()
                self.send_response(owner.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/hook"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def hook() -> Iterator[Hook]:
    h = Hook()
    try:
        yield h
    finally:
        h.stop()


def decide_by_keyword(body: dict[str, Any]) -> tuple[str, None]:
    """The fake language model, for pipelines: decisions answer from the question, other prompts are echoed."""
    system = next((m["content"] for m in body["messages"] if m["role"] == "system"), "")
    user = body["messages"][-1]["content"]
    if "Answer with one of" in system:
        if "1 (angry)" in user:
            return json.dumps({"answer": "4", "confidence": 0.7, "reason": "polite"}), None
        if "Which team" in user:
            return json.dumps({"answer": "Urgent", "confidence": 0.9, "reason": "a fire"}), None
        return json.dumps({"answer": "yes", "confidence": 0.8, "reason": "it was said"}), None
    if "JSON object" in system:
        return json.dumps({"chapters": [{"start": 0, "title": "Intro"}]}), None
    return f"SUMMARY of: {user[-60:]}", None


@pytest.fixture
def with_llm(settings: Settings, llm: FakeLLM) -> FakeLLM:
    settings.llm_base_url = llm.url
    settings.llm_model = "fake-llm"
    llm.reply = decide_by_keyword  # type: ignore[method-assign]
    return llm


# -- validation ---------------------------------------------------------------------------------------------
async def test_templates_are_valid(core: Bolo) -> None:
    for t in all_templates():
        report = validate(t, core)
        assert report.valid, (t["id"], report.to_dict()["errors"])


async def test_validation_lists_every_problem(core: Bolo) -> None:
    report = validate(
        {
            "id": "Bad Id",
            "steps": [
                {"type": "denoise", "model": "no-such-model"},
                {"type": "stt", "model": "fake-tts"},
                {"type": "teleport"},
                {"id": "t", "type": "template", "options": {"template": "{{steps.nope.text}} {{steps.stt.colour}}"}},
                {"type": "tts", "inputs": {"text": "{{steps.denoise.audio}}"}},
                {"type": "align", "inputs": {"text": "{{steps.later.text}}"}},
                {"id": "later", "type": "template", "options": {"template": "x", "colour": "red"}},
                {"type": "split-speakers", "options": {"steps": []}},
                {"id": "t", "type": "template", "options": {"template": "{{input.audio}} said"}},
                {"type": "condition", "options": {"if": {"value": "{{steps.stt.text}}", "near": "x"}}},
            ],
        },
        core,
    )
    found: dict[str, list[str]] = {}
    for e in report.errors:
        found.setdefault(e.path, []).append(e.message)
    paths = {k: " | ".join(v) for k, v in found.items()}
    assert not report.valid
    assert "id" in paths
    assert "no-such-model" in paths["steps[0].model"]
    assert "is a tts model" in paths["steps[1].model"]
    assert "unknown step type 'teleport'" in paths["steps[2].type"]
    assert "step 'nope' does not exist" in paths["steps[3].options.template"]
    assert "steps[4].inputs.text" in paths and "takes text" in paths["steps[4].inputs.text"]
    assert "comes later" in paths["steps[5].inputs.text"]
    assert "no option 'colour'" in paths["steps[6].options.colour"]
    assert "diarization step before it" in paths["steps[7].inputs.turns"]
    assert "used twice" in paths["steps[8].id"]
    assert "exactly one operator" in paths["steps[9].options.if"]
    # the template's unknown output of a known step
    assert any("has no output 'colour'" in e.message for e in report.errors)


async def test_validation_reports_defaults_and_types(core: Bolo) -> None:
    report = validate({"steps": [{"type": "denoise", "model": "fake-denoise"}, {"type": "stt"}, {"type": "tts"}]}, core)
    assert report.valid, report.to_dict()
    stt, tts = report.steps[1], report.steps[2]
    assert stt["model"] == "fake-stt" and stt["defaults"]["audio"] == ["{{steps.denoise.audio}}", "{{input.audio}}"]
    assert tts["defaults"]["text"] == ["{{steps.stt.text}}", "{{input.text}}"]
    assert stt["outputs"]["transcript"] == "transcript"
    # audio written into a text is caught
    bad = validate({"steps": [{"type": "template", "options": {"template": "x {{input.audio}}"}}]}, core)
    assert "cannot be written into a text" in bad.errors[0].message


def test_conditions() -> None:
    def check(cond: dict[str, Any], value: Any) -> bool:
        return evaluate(cond, lambda v: value if v == "{{x}}" else v, as_text)

    assert check({"value": "{{x}}", "equals": "Yes"}, " yes ")
    assert check({"value": "{{x}}", "in": ["a", "b"]}, "B")
    assert check({"value": "{{x}}", "gte": 0.7}, 0.75) and not check({"value": "{{x}}", "lt": 3}, "5")
    assert check({"value": "{{x}}", "matches": "^call"}, "Call me")
    assert check({"value": "{{x}}", "contains": "fox"}, "the quick fox")
    assert check({"value": "{{x}}", "exists": False}, "") and check({"value": "{{x}}", "truthy": True}, "yes")
    assert check({"all": [{"value": "{{x}}", "equals": 4}, {"not": {"value": "{{x}}", "equals": 5}}]}, "4")


# -- the runner ---------------------------------------------------------------------------------------------
async def test_run_every_audio_step_and_keep_files(core: Bolo) -> None:
    definition = {
        "id": "all",
        "steps": [
            {"type": "denoise", "model": "fake-denoise"},
            {"type": "diarization"},
            {"type": "stt", "inputs": {"turns": "{{steps.diarization.turns}}"}},
            {"type": "punctuation", "inputs": {"text": "{{steps.stt.transcript}}"}},
            {"type": "itn"},
            {"type": "pii"},
            {"type": "lid", "model": "fake-lid"},
            {"type": "emotion", "model": "fake-emotion"},
            {"type": "audio-tagging", "model": "fake-tagging"},
            {"type": "vad"},
            {"type": "split-speakers", "options": {"steps": [{"id": "per", "type": "stt"}]}},
            {"type": "tts", "inputs": {"text": "{{steps.pii.text}}"}},
            {"type": "save", "inputs": {"value": "{{steps.punctuation.transcript}}"}, "options": {"name": "a.srt"}},
        ],
        "outputs": {"text": "{{steps.pii.text}}", "speakers": "{{steps.diarization.speakers}}",
                    "voice": "{{steps.tts.audio}}"},
    }  # fmt: skip
    run = core.pipelines.runner.prepare(definition, audio=Audio(two_speakers(), 16000))
    await core.pipelines.run(run)
    doc = run.to_dict()
    assert doc["status"] == "ok", doc["error"]
    by = {s["id"]: s for s in doc["steps"] if not s.get("speaker")}
    assert by["denoise"]["model"] == "fake-denoise" and by["denoise"]["output"]["audio"]["file"] == "denoise.wav"
    assert by["stt"]["model"] == "fake-stt" and all(s["ms"] is not None for s in doc["steps"])
    assert all(seg.get("speaker") in ("A", "B") for seg in by["stt"]["output"]["segments"])
    assert "[PERSON]" in doc["outputs"]["text"] and "fox" not in doc["outputs"]["text"]
    assert doc["outputs"]["speakers"] == ["A", "B"]
    assert by["lid"]["output"]["language"] == "en" and by["emotion"]["output"]["emotion"]
    assert [s["speaker"] for s in by["split_speakers"]["output"]["speakers"]] == ["A", "B"]
    assert [s["speaker"] for s in doc["steps"] if s.get("speaker")] == ["A", "B"]
    assert doc["outputs"]["voice"]["sample_rate"] == 24000
    files = {f["name"] for f in doc["files"]}
    assert {"denoise.wav", "tts.wav", "a.srt"} <= files
    srt = core.pipelines.store.file_path(run.id, "a.srt").read_text()
    assert "-->" in srt
    assert core.pipelines.store.get_run(run.id)["status"] == "ok"  # in the history


async def test_failure_keeps_partial_results_and_continue(core: Bolo) -> None:
    steps = [
        {"type": "stt"},
        {"type": "align", "on_error": "continue", "inputs": {"text": "  "}},
        {"type": "denoise", "model": "fake-denoise", "inputs": {"audio": "{{input.audio}}"}},
        {"id": "boom", "type": "voice-conversion", "options": {"voice": "nobody"}, "model": "fake-denoise"},
        {"type": "template", "options": {"template": "never"}},
    ]
    report = validate({"steps": steps}, core)
    assert "cannot be used for voice-conversion" in report.errors[0].message
    steps[3] = {"id": "boom", "type": "template", "options": {"template": "{{steps.stt.segments.99.text}}"}}
    steps[3] = {"id": "boom", "type": "tts", "inputs": {"text": "{{steps.stt.segments.99.text}}"}}
    run = core.pipelines.runner.prepare({"steps": steps}, audio=Audio(speechlike(2.0), 16000))
    await core.pipelines.run(run)
    doc = run.to_dict()
    statuses = [(s["id"], s["status"]) for s in doc["steps"]]
    assert statuses == [("stt", "ok"), ("align", "error"), ("denoise", "ok"), ("boom", "error"),
                        ("template", "pending")]  # fmt: skip
    assert doc["status"] == "error" and doc["error"]["step"] == "boom" and "empty" in doc["error"]["message"]
    assert doc["steps"][0]["output"]["text"]  # kept


async def test_branches_switch_stop_and_filter(core: Bolo) -> None:
    definition = {
        "steps": [
            {"type": "stt", "options": {"word_timestamps": True}},
            {"type": "filter", "inputs": {"items": "{{steps.stt.words}}"},
             "options": {"where": {"value": "{{item.w}}", "in": ["fox", "dog"]}}},
            {"id": "pick", "type": "switch", "inputs": {"value": "{{steps.filter.count}}"}, "options": {
                "cases": {"2": [{"id": "two", "type": "template", "options": {"template": "two animals"}}]},
                "default": [{"id": "other", "type": "template", "options": {"template": "other"}}]}},
            {"type": "condition", "options": {"if": {"value": "{{steps.two.text}}", "exists": True},
                                              "then": [{"type": "stop", "options": {"reason": "done early"}}]}},
            {"id": "after", "type": "template", "options": {"template": "not reached"}},
        ],
        "outputs": {"said": "{{steps.two.text}}", "other": "{{steps.other.text}}"},
    }  # fmt: skip
    report = validate(definition, None)
    assert report.valid, report.to_dict()
    run = core.pipelines.runner.prepare(definition, audio=Audio(speechlike(4.0), 16000))
    await core.pipelines.run(run)
    doc = run.to_dict()
    assert doc["status"] == "ok" and doc["stopped"] == "done early"
    states = {s["id"]: s["status"] for s in doc["steps"]}
    assert states["two"] == "ok" and states["other"] == "skipped" and states["after"] == "skipped"
    assert doc["outputs"] == {"said": "two animals", "other": None}


async def test_llm_decision_translation_and_webhook(core: Bolo, with_llm: FakeLLM, hook: Hook) -> None:
    definition = {
        "vars": {"url": hook.url},
        "steps": [
            {"type": "stt"},
            {"id": "ok", "type": "decision", "options": {"question": "Was the disclaimer read?"}},
            {"id": "mood", "type": "decision", "options": {"question": "From 1 (angry) to 5?",
                                                           "choices": [1, 2, 3, 4, 5]}},
            {"type": "translation", "options": {"target_language": "hi"}},
            {"type": "llm", "options": {"prompt": "Summarise: {{steps.stt.text}}"}},
            {"type": "webhook", "options": {"url": "{{vars.url}}", "headers": {"x-team": "qa"}},
             "inputs": {"body": {"read": "{{steps.ok.answer}}", "mood": "{{steps.mood.answer}}"}}},
        ],
    }  # fmt: skip
    run = core.pipelines.runner.prepare(definition, audio=Audio(speechlike(3.0), 16000))
    await core.pipelines.run(run)
    doc = run.to_dict()
    assert doc["status"] == "ok", doc["error"]
    by = {s["id"]: s["output"] for s in doc["steps"]}
    assert by["ok"] == {"answer": "yes", "confidence": 0.8, "reason": "it was said"}
    assert by["mood"]["answer"] == "4"
    assert by["llm"]["text"].startswith("SUMMARY of:")
    assert by["webhook"]["status"] == 200 and by["webhook"]["json"] == {"received": True}
    translate_request = next(r for r in with_llm.requests if "into Hindi" in r["messages"][0]["content"])
    assert translate_request["model"] == "fake-llm"
    headers, body = hook.requests[0]
    assert json.loads(body) == {"read": "yes", "mood": "4"}
    assert headers["x-team"] == "qa" and headers["x-bolo-event"] == "pipeline.step"
    assert webhooks.verify(webhooks.secret(core.settings), body, headers["x-bolo-signature"])
    assert not webhooks.verify("wrong", body, headers["x-bolo-signature"])


async def test_decision_outside_the_choices_fails(core: Bolo, with_llm: FakeLLM) -> None:
    with_llm.reply = lambda body: ('{"answer": "maybe"}', None)  # type: ignore[method-assign]
    definition = {"steps": [{"type": "decision", "options": {"question": "Is it raining?"}}]}
    run = core.pipelines.runner.prepare(definition, text="it is pouring outside")
    await core.pipelines.run(run)
    assert run.status == "error" and "not one of yes, no" in run.error["message"]  # type: ignore[index]


async def test_decision_models_answer_decision_steps(core: Bolo) -> None:
    definition = {"steps": [
        {"id": "yn", "type": "decision", "model": "fake-decision", "options": {"question": "Is there a text?"}},
        {"id": "pick", "type": "decision", "model": "fake-decision",
         "options": {"question": "Which team?", "choices": ["sales", "support"]}},
    ]}  # fmt: skip
    assert validate(definition, core).valid
    run = core.pipelines.runner.prepare(definition, text="hello there")
    await core.pipelines.run(run)
    by = {s["id"]: s for s in run.to_dict()["steps"]}
    assert by["yn"]["output"]["answer"] == "yes" and by["yn"]["output"]["confidence"] == 0.9
    assert by["pick"]["output"]["answer"] == "sales" and by["pick"]["model"] == "fake-decision"
    bad = validate({"steps": [{"type": "decision", "model": "fake-stt", "options": {"question": "x"}}]}, core)
    assert "cannot be used for decision" in bad.errors[0].message


def test_signature_round_trip(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path)
    key = webhooks.secret(s)
    assert key == webhooks.secret(s) and key.startswith("whsec_")  # made once, then read back
    body = b'{"a": 1}'
    header = webhooks.signature(key, body)
    assert webhooks.verify(key, body, header) and not webhooks.verify(key, body + b" ", header)
    old = webhooks.signature(key, body, int(time.time()) - 3600)
    assert not webhooks.verify(key, body, old)  # a replay is refused
    assert webhooks.secret(Settings(data_dir=tmp_path, webhook_secret="mine")) == "mine"


# -- HTTP API -----------------------------------------------------------------------------------------------
def test_api_crud_versions_and_validation(live: LiveServer) -> None:
    base = live.url + "/api/v1/pipelines"
    definition = {"id": "notes", "name": "Notes", "steps": [{"type": "stt"}, {"type": "punctuation"}]}
    r = httpx.post(base, json=definition)
    assert r.status_code == 201, r.text
    assert r.json()["version"] == 1
    assert httpx.post(base, json=definition).status_code == 409
    yaml_body = "name: Notes\nsteps:\n  - type: stt\n  - type: itn\n"
    r = httpx.put(base + "/notes", content=yaml_body, headers={"content-type": "application/yaml"})
    assert r.status_code == 200 and r.json()["version"] == 2 and r.json()["id"] == "notes"
    assert httpx.put(base + "/notes", content=yaml_body, headers={"content-type": "application/yaml"}).json()[
        "version"] == 2  # unchanged: no new version  # fmt: skip
    assert [v["version"] for v in httpx.get(base + "/notes/versions").json()["data"]] == [1, 2]
    assert httpx.get(base + "/notes", params={"version": 1}).json()["steps"][1]["type"] == "punctuation"
    assert [p["id"] for p in httpx.get(base).json()["data"]] == ["notes"]
    bad = httpx.post(base, json={"id": "bad", "steps": [{"type": "stt", "model": "nope"}]})
    assert bad.status_code == 400 and bad.json()["validation"]["errors"][0]["path"] == "steps[0].model"
    v = httpx.post(base + "/validate", json={"steps": [{"type": "tts"}]}).json()
    assert v["valid"] and v["steps"][0]["model"] == "fake-tts"
    types = {t["type"]: t for t in httpx.get(base + "/step-types").json()["data"]}
    assert types["stt"]["kind"] == "stt" and "transcript" in types["stt"]["outputs"]
    assert {t["id"] for t in httpx.get(base + "/templates").json()["data"]} >= {"meeting-notes", "call-qa", "dub"}
    assert httpx.delete(base + "/notes").status_code == 204
    assert httpx.get(base + "/notes").status_code == 404


def test_api_run_sync_async_with_callback(live: LiveServer, hook: Hook) -> None:
    httpx.post(live.url + "/api/v1/pipelines", json={
        "id": "clean", "steps": [{"type": "denoise", "model": "fake-denoise"}, {"type": "stt"}],
        "outputs": {"text": "{{steps.stt.text}}"}})  # fmt: skip
    files = {"file": ("a.wav", wav_bytes(2.0), "audio/wav")}
    r = httpx.post(live.url + "/v1/pipelines/clean/run", files=files, timeout=60)
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["status"] == "ok" and run["outputs"]["text"] and run["input"]["filename"] == "a.wav"
    wav = httpx.get(live.url + run["steps"][0]["output"]["audio"]["url"])
    assert wav.status_code == 200 and wav.content[:4] == b"RIFF"
    assert httpx.get(live.url + f"/api/v1/pipelines/runs/{run['id']}/files/..%2Fx.wav").status_code == 404

    r = httpx.post(live.url + "/v1/pipelines/clean/run", files=files,
                   data={"async": "true", "callback_url": hook.url}, timeout=60)  # fmt: skip
    assert r.status_code == 202 and r.json()["status"] in ("queued", "running")
    run_id, job_id = r.json()["id"], r.json()["job_id"]
    deadline = time.time() + 30
    while time.time() < deadline and not hook.requests:
        time.sleep(0.1)
    headers, body = hook.requests[0]
    assert headers["x-bolo-event"] == "pipeline.run.completed" and headers["x-bolo-run"] == run_id
    assert json.loads(body)["outputs"]["text"]
    assert webhooks.verify(webhooks.secret(live.core.settings), body, headers["x-bolo-signature"])
    assert httpx.get(live.url + f"/api/v1/jobs/{job_id}").json()["state"] == "done"
    deadline = time.time() + 10
    while time.time() < deadline and "callback" not in httpx.get(live.url + f"/api/v1/pipelines/runs/{run_id}").json():
        time.sleep(0.1)
    assert httpx.get(live.url + f"/api/v1/pipelines/runs/{run_id}").json()["callback"]["delivered"] is True
    history = httpx.get(live.url + "/api/v1/pipelines/runs", params={"pipeline": "clean"}).json()["data"]
    assert {h["id"] for h in history} >= {run["id"], run_id}


def test_api_adhoc_run_and_errors(live: LiveServer) -> None:
    pipeline = json.dumps({"steps": [{"type": "template", "options": {"template": "Hi {{input.text}}"}}]})
    r = httpx.post(live.url + "/v1/pipelines/run", data={"pipeline": pipeline, "text": "there"})
    assert r.status_code == 200 and r.json()["outputs"] == {"text": "Hi there"}
    r = httpx.post(live.url + "/v1/pipelines/run", data={"pipeline": "steps: [{type: nope}]", "text": "x"})
    assert r.status_code == 400 and "unknown step type" in r.json()["error"]["message"]
    r = httpx.post(live.url + "/v1/pipelines/run", data={"pipeline": pipeline})
    assert r.status_code == 400 and "recording" in r.json()["error"]["message"]
    r = httpx.post(live.url + "/v1/pipelines/missing/run", data={"text": "x"})
    assert r.status_code == 404
    r = httpx.post(live.url + "/v1/pipelines/run", data={"pipeline": pipeline, "text": "x",
                                                          "callback_url": "http://example.com"})  # fmt: skip
    assert r.status_code == 400 and "async" in r.json()["error"]["message"]


def test_remote_urls_refused_on_an_open_bind(settings: Settings) -> None:
    from bolo.api.pipelines import remote_urls_allowed

    settings.host = "0.0.0.0"
    core = Bolo(settings)
    assert not remote_urls_allowed(core)
    settings.api_key = "k"
    assert remote_urls_allowed(core)
    settings.pipeline_remote_urls = False
    assert not remote_urls_allowed(core)


def test_api_keys_need_scopes(settings: Settings) -> None:
    from bolo.keys import scope_for

    assert scope_for("POST", "/v1/pipelines/x/run") == "stt"
    assert scope_for("POST", "/api/v1/pipelines") == "manage"
    assert scope_for("GET", "/api/v1/pipelines/runs") == "read"


# -- CLI ----------------------------------------------------------------------------------------------------
def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "data"
    common = ["--data-dir", str(data), "--fake-engine"]
    path = tmp_path / "p.yaml"
    path.write_text("id: transcribe\nsteps:\n  - type: stt\n    model: fake-stt\n  - type: template\n    options: {template: 'Said: "
                    "{{steps.stt.text}}'}\noutputs:\n  said: '{{steps.template.text}}'\n")  # fmt: skip
    audio = tmp_path / "a.wav"
    audio.write_bytes(wav_bytes(2.0))
    assert main(["pipeline", "create", str(path), *common]) == 0
    assert "created transcribe version 1" in capsys.readouterr().out
    assert main(["pipeline", "list", *common]) == 0
    assert "transcribe" in capsys.readouterr().out
    assert main(["pipeline", "show", "transcribe", "--json", *common]) == 0
    assert json.loads(capsys.readouterr().out)["steps"][0]["type"] == "stt"
    assert main(["pipeline", "run", "transcribe", str(audio), *common]) == 0
    assert capsys.readouterr().out.startswith("said: Said: the quick")
    assert main(["pipeline", "run", str(path), str(audio), "--json", *common]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"steps": [{"type": "stt", "model": "nope"}]}))
    assert main(["pipeline", "validate", str(bad), *common]) == 1
    assert "steps[0].model" in capsys.readouterr().err
    assert main(["pipeline", "create", "--template", "call-qa", *common]) == 0
    assert main(["pipeline", "templates", *common]) == 0
    assert "meeting-notes" in capsys.readouterr().out
    assert main(["pipeline", "rm", "transcribe", *common]) == 0
    assert [p["id"] for p in PipelineStore(data).all()] == ["call-qa"]
    assert main(["pipeline", "secret", *common]) == 0
    assert capsys.readouterr().out.strip().endswith(webhooks.secret(Settings(data_dir=data)))
