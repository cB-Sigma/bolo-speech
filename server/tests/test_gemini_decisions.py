"""Google's Gemini Decisions API (``gemini-decision-1``, provider kind ``gemini``) against a stand-in written from
Google's public documentation: ``POST /v1beta/decisions`` with ``x-goog-api-key``, the request's content blocks and
questions, the answers mapped back, errors and the cost estimate. Nothing here talks to Google."""

from __future__ import annotations

import base64
import io
import json
import threading
import wave
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import numpy as np
import pytest

from bolo.audio.encode import encode_audio
from bolo.catalog import Catalog
from bolo.config import Settings
from bolo.providers import PRICE_UNITS
from bolo.remote.gemini import Gemini
from conftest import LiveServer

KEY = "test-gemini-key"


class FakeGemini:
    """``POST /v1beta/decisions`` as the Decisions guide describes it."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 200
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

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "headers": dict(self.headers.items()), "json": body})
                if self.headers.get("x-goog-api-key") != KEY:
                    self._json(
                        403, {"error": {"code": 403, "message": "API key not valid", "status": "PERMISSION_DENIED"}}
                    )
                    return
                if outer.status != 200:
                    self._json(outer.status, {"error": {"code": outer.status, "message": "blocked by safety filters"}})
                    return
                answers: dict[str, Any] = {}
                for qid, q in body["questions"].items():
                    if q["type"] == "noul":
                        answers[qid] = {"type": "noul", "noul": 0.93, "confidence": 0.86}
                    elif q["type"] == "choice":
                        opts = list(q["criteria"])
                        probs = {o: (0.8 if i == 0 else 0.2 / (len(opts) - 1)) for i, o in enumerate(opts)}
                        answers[qid] = {"type": "choice", "choice": opts[0], "probabilities": probs, "confidence": 0.8}
                    else:
                        levels = q["criteria"]
                        answers[qid] = {
                            "type": "score", "score": 0.4, "confidence": 0.6,
                            "legend": {str(i): t for i, t in enumerate(levels)},
                            "probabilities": {str(i): (0.6 if i == 0 else 0.4 / (len(levels) - 1))
                                              for i in range(len(levels))},
                        }  # fmt: skip
                usage = {"total_input_tokens": 1200, "total_output_tokens": 0,
                         "input_tokens_by_modality": {"text": 200, "audio": 1000}}  # fmt: skip
                self._json(200, {"answers": answers, "usage": usage})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1beta"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def gemini() -> Iterator[FakeGemini]:
    s = FakeGemini()
    yield s
    s.stop()


@pytest.fixture
def srv(settings: Settings, gemini: FakeGemini) -> Iterator[LiveServer]:
    settings.providers = {
        "gemini": {"base_url": gemini.url, "api_key": KEY,
                   "prices": {"gemini-decision-1": {"per_1m_tokens": 0.10}}},
    }  # fmt: skip
    server = LiveServer(settings)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def test_questions_as_gemini_takes_them() -> None:
    q = Gemini.questions({
        "a": {"type": "noul", "instructions": "Is it urgent?"},
        "b": {"type": "choice", "instructions": "Team?", "criteria": {"billing": None, "technical": "Bugs"}},
        "c": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "upset", "furious"]},
    })  # fmt: skip
    assert q["a"] == {"type": "noul", "instructions": "Is it urgent?"}
    assert q["b"]["criteria"] == {"billing": "billing", "technical": "Bugs"}  # an option without a description
    assert q["c"]["criteria"] == ["calm", "upset", "furious"]


def test_decisions_through_bolo(srv: LiveServer, gemini: FakeGemini) -> None:
    x = (0.3 * np.sin(2 * np.pi * 300 * np.arange(16000) / 16000)).astype(np.float32)
    questions = {
        "urgent": {"type": "yes_no", "question": "Is the caller in a hurry?"},
        "team": {"type": "choice", "question": "Which team?", "options": ["billing", "technical"]},
        "anger": {"type": "score", "question": "How angry?", "levels": ["calm", "upset", "furious"]},
    }
    r = httpx.post(
        f"{srv.url}/v1/audio/decisions",
        files={"file": ("a.wav", io.BytesIO(encode_audio(x, 16000, "wav")), "audio/wav")},
        data={"model": "gemini-decision-1", "text": "Call from a customer.", "questions": json.dumps(questions)},
        timeout=60,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answers"]["urgent"] == {"type": "yes_no", "answer": True, "probability": 0.93}
    assert body["answers"]["team"]["answer"] == "billing" and body["answers"]["anger"]["label"] == "calm"
    assert body["usage"] == {"input_tokens": 1200}
    sent = gemini.requests[-1]
    assert sent["path"] == "/v1beta/decisions" and sent["headers"]["x-goog-api-key"] == KEY
    assert sent["json"]["model"] == "gemini-decision-1" and set(sent["json"]["questions"]) == {
        "urgent",
        "team",
        "anger",
    }
    text, audio = sent["json"]["input"]
    assert text == {"type": "text", "text": "Call from a customer."}
    assert audio["type"] == "audio" and audio["mime_type"] == "audio/wav"
    with wave.open(io.BytesIO(base64.b64decode(audio["data"]))) as w:
        assert w.getframerate() == 16000 and w.getnframes() == 16000
    # text only: one text block; the provider's cost estimate from its token price
    r = httpx.post(f"{srv.url}/v1/audio/decisions", json={"model": "gemini-decision-1", "text": "Refund please",
                                                           "questions": {"q": {"type": "yes_no", "question": "?"}}})  # fmt: skip
    assert r.status_code == 200 and gemini.requests[-1]["json"]["input"] == [{"type": "text", "text": "Refund please"}]


def test_token_prices() -> None:
    from bolo.providers import ProviderConfig

    p = ProviderConfig("gemini", "gemini", "https://x", prices={"gemini-decision-1": {"per_1m_tokens": 0.10}})
    assert "per_1m_tokens" in PRICE_UNITS
    assert p.estimate("gemini-decision-1", tokens=1200) == {"amount": 0.00012, "currency": "USD", "estimated": True}


def test_errors_are_mapped(srv: LiveServer, gemini: FakeGemini) -> None:
    gemini.status = 400
    r = httpx.post(f"{srv.url}/v1/audio/decisions", json={"model": "gemini-decision-1", "text": "x",
                                                           "questions": {"q": {"type": "yes_no", "question": "?"}}})  # fmt: skip
    assert r.status_code == 400 and "safety" in r.text


def test_catalog_entry() -> None:
    m = Catalog.load(None).resolve("gemini-decision-1")
    assert m is not None and m.task == "decision" and m.engine == "remote" and m.kind == "remote-decision"
    assert m.cloud and m.validated == "untested" and m.options["not_yet_run"] is True
    assert m.options["provider"] == "gemini" and m.options["model"] == "gemini-decision-1"
    assert "BOLO_PROVIDER_GEMINI_API_KEY" in m.description and "leaves this server" in m.description
