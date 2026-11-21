"""Spoken intent (task ``intent``): reading SpeechBrain's SLU output, ``POST /v1/audio/intent`` and ``bolo intent``
with the fake model, and the catalog entry."""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from bolo.audio.encode import encode_audio
from bolo.catalog import Catalog
from bolo.cli import main
from bolo.intent import parse_semantics
from conftest import LiveServer

SR = 16000


def wav(seconds: float, amp: float = 0.3) -> bytes:
    t = np.arange(int(seconds * SR)) / SR
    return encode_audio((amp * np.sin(2 * np.pi * 300 * t)).astype(np.float32), SR, "wav")


def test_slurp_output_is_read_without_running_code() -> None:
    r = parse_semantics(
        "{'scenario': 'calendar', 'action': 'set', 'entities': [{'type': 'date', 'filler': 'tomorrow'}, "
        "{'type': 'event_name', 'filler': 'dentist '}]}"
    )
    assert (r.intent, r.scenario, r.action) == ("calendar_set", "calendar", "set")
    assert r.slots == [{"type": "date", "value": "tomorrow"}, {"type": "event_name", "value": "dentist"}]
    # what the SLURP model really writes: "|" for the commas (bolo's run of slu-slurp-hubert on a synthetic command)
    real = parse_semantics("{'scenario': 'calendar'| 'action': 'set'| 'entities': [{'type': 'event_name'| 'filler': "
                           "'dentist appointment'}| {'type': 'date'| 'filler': 'friday'}]}")  # fmt: skip
    assert real.intent == "calendar_set" and real.slots == [{"type": "event_name", "value": "dentist appointment"},
                                                              {"type": "date", "value": "friday"}]  # fmt: skip
    broken = parse_semantics(
        "{'scenario': 'alarm'| 'action': 'set'| 'entities': [{'type': 'date'| 'filler': 'tomorrow'}| {'ty"
    )
    assert broken.intent == "alarm_set" and broken.slots == [{"type": "date", "value": "tomorrow"}]
    # Timers and Such ("intent" and a "slots" object) and Fluent Speech Commands (plain fields)
    t = parse_semantics("{'intent': 'SimpleMath', 'slots': {'number1': 37.67, 'number2': 75.7, 'op': ' minus '}}")
    assert t.intent == "SimpleMath" and t.scenario is None and t.slots[2] == {"type": "op", "value": "minus"}
    f = parse_semantics("{'action': 'activate', 'object': 'lights', 'location': 'kitchen'}")
    assert f.intent == "activate" and f.slots == [{"type": "object", "value": "lights"}, {"type": "location",
                                                                                         "value": "kitchen"}]  # fmt: skip
    # a broken literal (beam search can stop inside a quote): what can be read is kept
    b = parse_semantics(
        "{'scenario': 'alarm', 'action': 'query', 'entities': [{'type': 'time', 'filler': 'seven'}, {'ty"
    )
    assert b.intent == "alarm_query" and b.slots == [{"type": "time", "value": "seven"}]
    # code is never run, and nonsense is no intent
    n = parse_semantics("__import__('os').system('true')")
    assert n.intent is None and n.slots == [] and n.text.startswith("__import__")
    assert parse_semantics("").intent is None


def test_intent_route(live: LiveServer) -> None:
    url = live.url + "/v1/audio/intent"
    r = httpx.post(url, files={"file": ("a.wav", io.BytesIO(wav(2.0)), "audio/wav")}, data={"model": "fake-intent"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert r.headers["x-bolo-model"] == "fake-intent" and body["object"] == "audio.intent" and body["duration"] == 2.0
    assert body["intent"] == "alarm_set" and body["scenario"] == "alarm" and body["action"] == "set"
    assert body["slots"] == [{"type": "date", "value": "tomorrow"}, {"type": "time", "value": "seven"}]
    quiet = httpx.post(url, files={"file": ("q.wav", io.BytesIO(wav(1.0, 0.001)), "audio/wav")},
                       data={"model": "fake-intent"})  # fmt: skip
    assert quiet.json()["intent"] == "general_quirky" and quiet.json()["slots"] == []
    # errors: too long, too short, no file, a model of another task
    for files, data in (
        ({"file": ("a.wav", io.BytesIO(wav(31.0)), "audio/wav")}, {"model": "fake-intent"}),
        ({"file": ("a.wav", io.BytesIO(wav(0.05)), "audio/wav")}, {"model": "fake-intent"}),
        (None, {"model": "fake-intent"}),
        ({"file": ("a.wav", io.BytesIO(wav(1.0)), "audio/wav")}, {"model": "fake-emotion"}),
    ):
        r = httpx.post(url, files=files, data=data, timeout=30)
        assert r.status_code == 400, r.text
    caps = httpx.get(live.url + "/api/v1/info").json()["capabilities"]
    assert caps["intent"]["endpoint"] == "/v1/audio/intent" and "slu-slurp-hubert" in caps["intent"]["models"]
    assert caps["intent"]["default_model"] == "slu-slurp-hubert"


def test_intent_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("BOLO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BOLO_AUTO_PULL", "0")
    clip = tmp_path / "say.wav"
    clip.write_bytes(wav(1.5))
    assert main(["intent", str(clip), "--fake-engine", "--model", "fake-intent"]) == 0
    out = capsys.readouterr().out
    assert "fake-intent: alarm_set" in out and "date: tomorrow" in out
    assert main(["intent", str(clip), "--fake-engine", "--model", "fake-intent", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["slots"][1] == {"type": "time", "value": "seven"}


def test_catalog_entry() -> None:
    spec = Catalog.load(None).resolve("slu-slurp-hubert")
    assert spec is not None
    assert spec.task == "intent" and spec.engine == "plugin:torch" and spec.kind == "speechbrain-slu"
    assert spec.license == "Apache-2.0" and spec.commercial_use and spec.languages == ["en"]
    paths = {u["path"] for u in spec.source.urls}
    assert {"hub/config.json", "hub/preprocessor_config.json"} <= paths
    assert "custom_interface.py" not in spec.source.include  # the repository's own code is not run
