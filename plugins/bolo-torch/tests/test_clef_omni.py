"""Cloudflare Clef-Omni (kind ``clef-omni``, task decision) with a stand-in for its joint_schema_model.systemone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from bolo.catalog import Catalog
from bolo.engines.base import DecisionRequest

from bolo_torch.engine import LOADERS, NOT_IMPLEMENTED, TorchEngine
from bolo_torch.models import clef_omni
from bolo_torch.remote_code import RemoteCodeRefused
from bolo_torch.runtime import LoadContext

QUESTIONS = {
    "outage": {"type": "noul", "instructions": "Is a service down?"},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "technical": None}},
}


def test_requests_go_to_systemone_as_cloudflare_documents_them() -> None:
    calls: list[dict[str, Any]] = []

    def systemone(model: Any, processor: Any, body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        return {"model": body["model"], "answers": {"outage": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 99, "output_tokens": 0}}  # fmt: skip

    model = clef_omni.ClefOmni(systemone, object(), object(), "clef-omni", max_audio_s=60.0)
    out = model.decide(DecisionRequest("Checkout returns errors.", np.zeros(1600, np.float32), QUESTIONS))
    assert out == {
        "answers": {"outage": {"type": "noul", "noul": 0.8}},
        "usage": {"input_tokens": 99, "output_tokens": 0},
    }
    sent = calls[-1]
    assert (
        sent["model"] == "clef-omni" and sent["state"] == "Checkout returns errors." and sent["questions"] == QUESTIONS
    )
    assert len(sent["audio"]) == 1 and sent["audio"][0].dtype == np.float32 and len(sent["audio"][0]) == 1600
    model.decide(DecisionRequest(None, None, QUESTIONS))
    assert calls[-1]["state"] is None and "audio" not in calls[-1]  # Cloudflare's code renders None as null
    assert model.max_audio_s == 60.0


def test_the_kind_is_registered_and_refuses_without_the_opt_in(tmp_path: Path) -> None:
    assert LOADERS["clef-omni"] == ("bolo_torch.models.clef_omni", "load")
    assert "clef-omni" in TorchEngine().kinds and "clef-omni" not in NOT_IMPLEMENTED
    (tmp_path / "manifest.json").write_text(json.dumps({"source": {"revision": "c" * 40}}))
    (tmp_path / clef_omni.CODE).write_text("raise SystemExit('must not be imported')\n")
    ctx = LoadContext("clef-omni", "decision", tmp_path, {"remote_code": {"revision": "c" * 40,
                                                                           "files": {clef_omni.CODE: "0" * 64}}})  # fmt: skip
    with pytest.raises(RemoteCodeRefused, match="allow_remote_code=true"):
        clef_omni.load(ctx)
    ctx.options["allow_remote_code"] = True
    with pytest.raises(RemoteCodeRefused, match="does not match the reviewed file"):
        clef_omni.load(ctx)  # the file is not the reviewed one: refused before it is imported


def test_catalog_entry() -> None:
    spec = Catalog.load().resolve("clef-omni")
    assert spec is not None and spec.task == "decision" and spec.engine == "plugin:torch" and spec.kind == "clef-omni"
    assert spec.license == "Apache-2.0" and spec.commercial_use and spec.options["not_yet_run"] is True
    assert (
        spec.source.repo == "Cloudflare/clef-omni" and spec.source.revision == spec.options["remote_code"]["revision"]
    )
    assert set(spec.options["remote_code"]["files"]) == {clef_omni.CODE}
    assert "llama.cpp" in spec.description
