from __future__ import annotations

import time

import httpx
import pytest

from bolo.catalog import ModelSpec
from bolo.cli import main
from bolo.config import Settings
from bolo.core import Bolo
from bolo.errors import ComponentMissing
from bolo.text import phonemize
from conftest import LiveServer


class StubPhonemizer(phonemize.Phonemizer):
    name = "stub"
    source = "python-package"

    def version(self) -> str | None:
        return "1.52.0"

    def clause(self, text: str, voice: str) -> str:
        return text


@pytest.fixture
def no_espeak(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Pretend espeak-ng is absent; ``install()`` makes it appear, as the real installer would."""
    state = {"installed": False, "commands": []}
    monkeypatch.setattr(phonemize, "get_phonemizer", lambda: StubPhonemizer() if state["installed"] else None)
    monkeypatch.setattr(phonemize, "reset_phonemizer", lambda: None)

    def installer(cmd: list[str]) -> tuple[int, str]:
        state["commands"].append(cmd)
        if state.get("fail"):
            return 1, "ERROR: Could not find a version that satisfies the requirement\nno network"
        state["installed"] = True
        return 0, "Successfully installed espeakng-loader"

    state["installer"] = installer
    return state


def add_piper_like(core: Bolo) -> None:
    core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": "needs-espeak", "task": "tts", "engine": "fake", "kind": "fake-tts", "family": "Piper",
                "languages": ["en"], "license": "MIT", "builtin": True, "sample_rate": 24000, "requires": ["espeak-ng"],
            }
        )
    )  # fmt: skip


def wait_job(url: str, job_id: str) -> dict:
    for _ in range(100):
        job = httpx.get(f"{url}/api/v1/jobs/{job_id}").json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_component_listing_shape(live: LiveServer, no_espeak) -> None:  # type: ignore[no-untyped-def]
    doc = httpx.get(live.url + "/api/v1/components").json()
    (c,) = doc["components"]
    assert set(c) == {
        "id", "name", "installed", "source", "version", "license", "license_url", "required_by_count", "installable",
        "install_blocked_reason", "manual_command", "description",
    }  # fmt: skip
    assert c["id"] == "espeak-ng" and c["name"] == "eSpeak NG" and c["license"] == "GPL-3.0-or-later"
    assert c["installed"] is False and c["source"] is None and c["version"] is None
    assert c["required_by_count"] >= 80 and c["installable"] is True and c["install_blocked_reason"] is None
    assert c["manual_command"] == "pip install 'bolo[espeak]'" and c["license_url"].startswith("https://")
    assert httpx.get(live.url + "/api/v1/info").json()["components"] == doc["components"]
    piper = httpx.get(live.url + "/api/v1/models/piper-en-us-amy-low").json()
    assert piper["requires"] == ["espeak-ng"] and piper["available"] is False
    kokoro = httpx.get(live.url + "/api/v1/models/kokoro-v1.0-int8").json()
    assert kokoro["requires"] == [] and kokoro["available"] is True


def test_missing_component_fails_fast_then_install_activates_without_restart(live: LiveServer, no_espeak) -> None:  # type: ignore[no-untyped-def]
    add_piper_like(live.core)
    live.core.components.run_installer = no_espeak["installer"]
    speech = {"model": "needs-espeak", "input": "Hello there.", "response_format": "pcm"}
    r = httpx.post(live.url + "/v1/audio/speech", json=speech)
    assert r.status_code == 424
    err = r.json()["error"]
    assert (
        err["code"] == "component_missing"
        and "espeak-ng" in err["message"]
        and "bolo components install espeak-ng" in err["message"]
    )
    assert not live.core.manager.loaded  # refused before the model was even loaded
    load = httpx.post(live.url + "/api/v1/models/needs-espeak/load")
    assert load.status_code == 424 and load.json()["error"]["code"] == "component_missing"

    assert httpx.post(live.url + "/api/v1/components/espeak-ng/install", json={}).status_code == 400
    refused = httpx.post(live.url + "/api/v1/components/espeak-ng/install", json={"accept_license": False})
    assert (
        refused.status_code == 400
        and refused.json()["error"]["param"] == "accept_license"
        and not no_espeak["commands"]
    )
    assert httpx.post(live.url + "/api/v1/components/nope/install", json={"accept_license": True}).status_code == 404

    r = httpx.post(live.url + "/api/v1/components/espeak-ng/install", json={"accept_license": True})
    assert r.status_code == 202
    job = wait_job(live.url, r.json()["job_id"])
    assert job["kind"] == "install" and job["model"] == "espeak-ng" and job["state"] == "done", job
    assert no_espeak["commands"][0][-1].startswith("espeakng-loader")
    c = httpx.get(live.url + "/api/v1/components").json()["components"][0]
    assert (
        c["installed"] is True
        and c["source"] == "python-package"
        and c["version"] == "1.52.0"
        and c["installable"] is False
    )
    assert httpx.get(live.url + "/api/v1/models/needs-espeak").json()["available"] is True
    ok = httpx.post(live.url + "/v1/audio/speech", json=speech)  # same server process, no restart
    assert ok.status_code == 200 and len(ok.content) > 10000
    again = httpx.post(live.url + "/api/v1/components/espeak-ng/install", json={"accept_license": True})
    assert wait_job(live.url, again.json()["job_id"])["state"] == "done" and len(no_espeak["commands"]) == 1


def test_failed_install_is_reported(live: LiveServer, no_espeak) -> None:  # type: ignore[no-untyped-def]
    no_espeak["fail"] = True
    live.core.components.run_installer = no_espeak["installer"]
    r = httpx.post(live.url + "/api/v1/components/espeak-ng/install", json={"accept_license": True})
    job = wait_job(live.url, r.json()["job_id"])
    assert job["state"] == "error" and "exited with 1" in job["error"] and "no network" in job["error"]
    assert httpx.get(live.url + "/api/v1/components").json()["components"][0]["installed"] is False


async def test_install_policy(settings: Settings, no_espeak, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    settings.host = "0.0.0.0"
    core = Bolo(settings)
    assert "non-loopback" in core.components.status()[0]["install_blocked_reason"]
    assert core.components.status()[0]["installable"] is False
    settings.allow_component_install = True  # even an explicit yes does not override that
    assert Bolo(settings).components.status()[0]["installable"] is False
    settings.api_key = "k"
    assert Bolo(settings).components.status()[0]["installable"] is True
    settings.allow_component_install = False
    assert "disabled" in Bolo(settings).components.status()[0]["install_blocked_reason"]
    settings.host, settings.api_key, settings.allow_component_install = "127.0.0.1", None, None
    core = Bolo(settings)
    monkeypatch.setattr("bolo.components.installer_command", lambda: None)
    assert "neither pip nor uv" in core.components.status()[0]["install_blocked_reason"]
    monkeypatch.setattr("bolo.components.installer_command", lambda: ["pip"])
    monkeypatch.setattr("os.access", lambda *a: False)
    assert "read-only" in core.components.status()[0]["install_blocked_reason"]


async def test_events_and_core_error(core: Bolo, no_espeak) -> None:  # type: ignore[no-untyped-def]
    import asyncio

    add_piper_like(core)
    with pytest.raises(ComponentMissing):
        await core.synthesize_all("Hello.", model="needs-espeak")
    core.components.run_installer = no_espeak["installer"]
    sub = core.events.subscribe()
    first = asyncio.ensure_future(sub.__anext__())
    await asyncio.sleep(0)
    job = core.components.start_install("espeak-ng", True)
    events = [await first]
    while not (events[-1]["type"] == "job" and events[-1]["job"]["state"] == "done"):
        events.append(await asyncio.wait_for(sub.__anext__(), 5))
    assert {"type": "component", "id": "espeak-ng", "event": "installed"} in events
    assert [e["job"]["state"] for e in events if e["type"] == "job"] == ["running", "done"] and job.kind == "install"
    audio, _ = await core.synthesize_all("Hello.", model="needs-espeak")
    assert len(audio) > 0
    await sub.aclose()


def test_cli_components(
    no_espeak, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("BOLO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("bolo.components.ComponentManager._run", staticmethod(no_espeak["installer"]))
    assert main(["components", "list"]) == 0
    out = capsys.readouterr().out
    assert "espeak-ng" in out and "not installed" in out and "GPL-3.0-or-later" in out
    assert main(["components", "install", "espeak-ng"]) == 1  # the licence must be accepted explicitly
    assert "--accept-license" in capsys.readouterr().err and not no_espeak["commands"]
    assert main(["components", "install", "espeak-ng", "--accept-license"]) == 0
    assert "installed espeak-ng" in capsys.readouterr().out
    assert main(["components", "list"]) == 0
    assert "installed (python-package, 1.52.0)" in capsys.readouterr().out
    assert main(["doctor"]) == 0
    assert "espeak-ng: found (python-package" in capsys.readouterr().out
