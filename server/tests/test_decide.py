"""Decisions (task ``decision``): question parsing, ``POST /v1/audio/decisions`` with the fake model, the
llama-decision client against a stand-in for llama-server's ``/v1/systemone``, the catalog entry and the CLI."""

from __future__ import annotations

import base64
import io
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import numpy as np
import pytest

from bolo.audio.encode import encode_audio
from bolo.catalog import Catalog
from bolo.decide import _answers, parse_questions
from bolo.engines.base import DecisionRequest
from bolo.engines.decision_llamacpp import LlamaDecision
from bolo.engines.tts_server import TTSServerEngine
from bolo.errors import EngineUnavailable, InvalidRequest
from conftest import LiveServer

SR = 16000
QUESTIONS = {
    "polite": {"type": "yes_no", "question": "Is the speaker polite?"},
    "topic": {"type": "choice", "question": "What is it about?", "options": {"delivery": "An order", "billing": None}},
    "urgency": {"type": "score", "question": "How urgent?", "levels": ["can wait", "today", "now"]},
}


def tone(seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def wav(seconds: float, amp: float = 0.3) -> bytes:
    return encode_audio(tone(seconds, amp), SR, "wav")


def test_questions_become_systemone_questions() -> None:
    q = parse_questions(json.dumps(QUESTIONS))
    assert q["polite"] == {"type": "noul", "instructions": "Is the speaker polite?"}
    assert q["topic"] == {"type": "choice", "instructions": "What is it about?",
                          "criteria": {"delivery": "An order", "billing": None}}  # fmt: skip
    assert q["urgency"]["criteria"] == ["can wait", "today", "now"]
    # llama.cpp's own names, and options as a list
    same = parse_questions({"a": {"type": "noul", "instructions": "x?"}, "b": {"type": "choice",
                            "instructions": "y?", "options": ["p", "q"]}})  # fmt: skip
    assert same["a"]["type"] == "noul" and same["b"]["criteria"] == {"p": None, "q": None}
    bad: list[Any] = [
        "not json", [], {}, {"x y": {"type": "yes_no", "question": "?"}}, {"a": {"type": "maybe", "question": "?"}},
        {"a": {"type": "yes_no"}}, {"a": {"type": "choice", "question": "?", "options": ["only"]}},
        {"a": {"type": "score", "question": "?", "levels": ["one"]}},
        {"a": {"type": "score", "question": "?", "levels": [str(i) for i in range(11)]}},
        {f"q{i}": {"type": "yes_no", "question": "?"} for i in range(33)},
    ]  # fmt: skip
    for raw in bad:
        with pytest.raises(InvalidRequest):
            parse_questions(raw)


def test_answers_in_bolo_shape() -> None:
    q = parse_questions(QUESTIONS)
    raw = {
        "polite": {"type": "noul", "noul": 0.2839},
        "topic": {"type": "choice", "choice": "delivery", "probabilities": {"delivery": 0.99, "billing": 0.01},
                  "confidence": 0.98},
        "urgency": {"type": "score", "score": 1.6, "legend": {}, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                    "confidence": 0.5},
    }  # fmt: skip
    out = _answers(raw, q)
    assert out["polite"] == {"type": "yes_no", "answer": False, "probability": 0.2839}
    assert out["topic"]["answer"] == "delivery" and out["topic"]["probabilities"]["billing"] == 0.01
    assert out["urgency"]["level"] == 2 and out["urgency"]["label"] == "now" and out["urgency"]["probabilities"] == [
        0.1, 0.2, 0.7,
    ]  # fmt: skip


def test_decisions_route(live: LiveServer) -> None:
    url = live.url + "/v1/audio/decisions"
    files = {"file": ("a.wav", io.BytesIO(wav(2.0)), "audio/wav")}
    r = httpx.post(url, files=files, data={"model": "fake-decision", "questions": json.dumps(QUESTIONS)}, timeout=30)
    assert r.status_code == 200, r.text
    body = r.json()
    assert r.headers["x-bolo-model"] == "fake-decision" and body["object"] == "audio.decision"
    assert body["duration"] == 2.0 and body["answers"]["polite"] == {
        "type": "yes_no",
        "answer": True,
        "probability": 0.9,
    }
    assert body["answers"]["topic"]["answer"] == "delivery" and body["answers"]["urgency"]["label"] == "today"
    # JSON: text only, then text and base64 audio (a quiet recording and no text: no)
    r = httpx.post(url, json={"model": "fake-decision", "text": "I was charged twice", "questions": QUESTIONS})
    assert r.status_code == 200 and "duration" not in r.json() and r.json()["answers"]["polite"]["answer"] is True
    quiet = base64.b64encode(wav(1.0, amp=0.001)).decode()
    r = httpx.post(url, json={"model": "fake-decision", "audio": f"data:audio/wav;base64,{quiet}",
                              "questions": {"q": {"type": "yes_no", "question": "?"}}})  # fmt: skip
    assert r.status_code == 200 and r.json()["answers"]["q"]["answer"] is False
    # errors: no state, a recording longer than the model hears, a model of another task, bad questions
    for kwargs, param in (
        ({"json": {"model": "fake-decision", "questions": QUESTIONS}}, "file"),
        ({"files": {"file": ("a.wav", io.BytesIO(wav(31.0)), "audio/wav")},
          "data": {"model": "fake-decision", "questions": json.dumps(QUESTIONS)}}, "file"),
        ({"json": {"model": "fake-emotion", "text": "x", "questions": QUESTIONS}}, "model"),
        ({"json": {"model": "fake-decision", "text": "x", "questions": "{"}}, "questions"),
        ({"json": {"model": "fake-decision", "text": "x", "audio": "%%%", "questions": QUESTIONS}}, "audio"),
    ):  # fmt: skip
        r = httpx.post(url, timeout=30, **kwargs)  # type: ignore[arg-type]
        assert r.status_code == 400, (param, r.text)
    caps = httpx.get(live.url + "/api/v1/info").json()["capabilities"]
    assert (
        caps["decision"]["endpoint"] == "/v1/audio/decisions" and "d1-omni-600m-llamacpp" in caps["decision"]["models"]
    )


class FakeSystemOne:
    """llama-server's /health and /v1/systemone, as llama.cpp b12521 answers for d1-omni."""

    def __init__(self, status: int = 200) -> None:
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _json(self, code: int, doc: Any) -> None:
                data = json.dumps(doc).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._json(200, {"status": "ok"})

            def do_POST(self) -> None:
                doc = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": doc})
                if status != 200:
                    self._json(status, {"error": {"code": status, "message": "nope"}})
                    return
                answers = {qid: {"type": "noul", "noul": 0.75} for qid in doc["questions"]}
                self._json(200, {"model": "d1", "answers": answers, "usage": {"input_tokens": 228, "output_tokens": 0}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def systemone() -> Iterator[FakeSystemOne]:
    s = FakeSystemOne()
    yield s
    s.stop()


def test_llama_decision_client(systemone: FakeSystemOne) -> None:
    model = LlamaDecision("d1", {"max_audio_s": 30.0}, base_url=systemone.url)
    model.connect()
    q = parse_questions({"polite": {"type": "yes_no", "question": "Polite?"}})
    out = model.decide(DecisionRequest("Recording from a call.", tone(1.0), q))
    assert out["answers"]["polite"]["noul"] == 0.75 and out["usage"]["input_tokens"] == 228
    sent = systemone.requests[-1]
    assert sent["path"] == "/v1/systemone" and sent["body"]["state"] == "Recording from a call."
    assert sent["body"]["questions"] == q and sent["body"]["files"][0].startswith("data:audio/wav;base64,")
    model.decide(DecisionRequest(None, tone(1.0), q))
    assert systemone.requests[-1]["body"]["state"] is None
    model.decide(DecisionRequest("text only", None, q))
    assert "files" not in systemone.requests[-1]["body"]
    model.close()
    for status, error in ((400, InvalidRequest), (501, EngineUnavailable), (500, EngineUnavailable)):
        bad = FakeSystemOne(status)
        try:
            m = LlamaDecision("d1", {}, base_url=bad.url)
            with pytest.raises(error):
                m.decide(DecisionRequest("x", None, q))
            m.close()
        finally:
            bad.stop()


def test_catalog_entry() -> None:
    spec = Catalog.load(None).resolve("d1-omni")
    assert spec is not None and spec.id == "d1-omni-600m-llamacpp" and spec.task == "decision"
    assert spec.kind == "llama-decision" and spec.kind in TTSServerEngine.kinds and spec.requires == ["llama-server"]
    assert spec.license == "LFM Open License v1.0" and not spec.commercial_use and "USD 10M" in spec.description
    assert spec.source.repo == "LiquidAI/d1-omni-600M-GGUF" and len(spec.source.revision) == 40
    assert {spec.options["model_file"], spec.options["mmproj_file"]} == set(spec.source.include)
    assert spec.options["server_args"] == ["-b", "4096", "-ub", "4096"] and spec.size_bytes == 669998816
