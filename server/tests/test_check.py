"""``bolo check``: runtime, weights at their pinned revision and placement, without downloading weights (fakes only:
no HTTP, no plugin)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from bolo import check, torch_envs
from bolo.config import Settings
from bolo.core import Bolo
from conftest import LiveServer
from test_recommend import machine, nvidia


def head_for(spec: Any, status: int = 200, sizes: dict[str, int] | None = None) -> Any:
    sizes = sizes or {}

    def head(url: str, headers: dict[str, str]) -> httpx.Response:
        f = next(u for u in spec.source.urls if u["url"] == url)
        return httpx.Response(status, headers={"content-length": str(sizes.get(f["path"], f.get("size") or 0))})

    return head


def test_a_duplex_model_without_its_server_and_too_big_for_this_gpu(settings: Settings) -> None:
    core = Bolo(settings)
    spec = core.catalog.resolve("moshiko-pytorch-q8")
    assert spec is not None
    core.advisor.use_hardware(machine(gpus=[nvidia(6)], device="auto"))
    report = check.run_check(core, spec, tiny=True, head=head_for(spec))
    by = {s["name"]: s for s in report["steps"]}
    assert not report["ok"]
    assert by["runtime"]["status"] == "fail" and "bolo components install moshi" in by["runtime"]["detail"]
    assert by["tiny"]["status"] == "skip"  # a full-duplex server's model is never built small: skipped, not passed
    assert by["weights"]["status"] == "ok" and "4 files" in by["weights"]["detail"]
    assert by["placement"]["status"] == "fail" and "full-duplex models can't be split" in by["placement"]["detail"]
    core.advisor.use_hardware(machine(gpus=[nvidia(24)], device="auto"))
    by = {s["name"]: s for s in check.run_check(core, spec, head=head_for(spec))["steps"]}
    assert by["placement"]["status"] == "ok" and by["placement"]["detail"].startswith("fits on the GPU")


def test_weights_missing_resized_or_gated(settings: Settings) -> None:
    core = Bolo(settings)
    spec = core.catalog.resolve("moshiko-pytorch-q8")
    assert spec is not None
    (step,) = check.check_weights(core, spec, head_for(spec, status=404))
    assert step["status"] == "fail" and "HTTP 404" in step["detail"]
    (step,) = check.check_weights(core, spec, head_for(spec, sizes={"model.q8.safetensors": 1}))
    assert step["status"] == "fail" and "1 bytes there" in step["detail"]
    (step,) = check.check_weights(core, spec, head_for(spec, status=401))
    assert step["status"] == "fail" and "does not say it is gated" in step["detail"]


def test_a_plugin_model_asks_the_plugins_environment(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    core = Bolo(settings)
    spec = core.catalog.resolve("veena")
    assert spec is not None and check.supports_tiny(spec)
    monkeypatch.setattr(torch_envs, "plugin_commands", lambda plugins, entry_points=True: {})
    by = {s["name"]: s for s in check.run_check(core, spec, head=lambda u, h: httpx.Response(200))["steps"]}
    assert by["runtime"]["status"] == "fail" and "bolo-torch[orpheus]" in by["runtime"]["detail"]
    monkeypatch.setattr(torch_envs, "plugin_commands", lambda plugins, entry_points=True: {"plugin:torch": ["python3"]})
    asked: list[dict[str, Any]] = []

    def runner(cmd: Sequence[str], timeout: float) -> tuple[int, str]:
        asked.append(json.loads(cmd[-1]))
        steps = [{"name": "imports", "status": "ok", "detail": "bolo_torch.models.orpheus"}]
        return 0, "[transformers] a warning\n" + json.dumps({"steps": steps})

    report = check.run_check(core, spec, runner=runner, head=lambda u, h: httpx.Response(200))
    by = {s["name"]: s for s in report["steps"]}
    assert by["imports"]["status"] == "ok" and asked[0]["kind"] == "orpheus-snac" and asked[0]["extra"] == "orpheus"
    assert "tiny" not in asked[0]


def test_check_over_http_and_the_cli(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    from bolo.cli import main

    srv = LiveServer(settings)
    srv.start()
    try:
        r = httpx.post(srv.url + "/api/v1/models/fake-stt/check", json={}, timeout=60)
        assert r.status_code == 200
        report = r.json()
        assert report["model"] == "fake-stt" and report["ok"] and report["steps"][0]["name"] == "catalog"
        assert httpx.post(srv.url + "/api/v1/models/no-such-model/check", json={}).status_code == 404
    finally:
        srv.stop()
    assert main(["check", "fake-stt", "--fake-engine", "--data-dir", str(settings.data_dir)]) == 0
    assert "fake-stt: ok" in capsys.readouterr().out
    assert main(["check", "no-such-model", "--data-dir", str(settings.data_dir)]) == 1
