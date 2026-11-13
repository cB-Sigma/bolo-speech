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


class StubUccharan(StubPhonemizer):
    name = "uccharan"
    component = "uccharan"

    def version(self) -> str | None:
        return "0.1.0"

    def supports(self, profile: str, language: str) -> bool:
        return phonemize.uccharan_language(profile, language) is not None


@pytest.fixture
def no_espeak(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Pretend espeak-ng and uccharan are absent; ``install()`` makes one appear, as the real installer would."""
    state = {"installed": False, "uccharan": False, "commands": []}
    monkeypatch.setattr(phonemize, "get_phonemizer", lambda: StubPhonemizer() if state["installed"] else None)
    monkeypatch.setattr(phonemize, "get_uccharan", lambda: StubUccharan() if state["uccharan"] else None)
    monkeypatch.setattr(phonemize, "reset_phonemizer", lambda: None)

    def installer(cmd: list[str]) -> tuple[int, str]:
        state["commands"].append(cmd)
        if state.get("fail"):
            return 1, "ERROR: Could not find a version that satisfies the requirement\nno network"
        state["uccharan" if cmd[-1].startswith("uccharan") else "installed"] = True
        return 0, "Successfully installed " + cmd[-1]

    state["installer"] = installer
    return state


def add_piper_like(core: Bolo, model_id: str = "needs-espeak", language: str = "de-DE") -> None:
    """A voice that reads phonemes, in a language only espeak-ng covers (German) unless told otherwise."""
    core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": model_id, "task": "tts", "engine": "fake", "kind": "fake-tts", "family": "Piper",
                "languages": [language[:2]], "license": "MIT", "builtin": True, "sample_rate": 24000,
                "requires": ["phonemizer"], "options": {"language": language},
            }
        )
    )  # fmt: skip


def by_id(components: list[dict]) -> dict[str, dict]:
    return {c["id"]: c for c in components}


def wait_job(url: str, job_id: str) -> dict:
    for _ in range(100):
        job = httpx.get(f"{url}/api/v1/jobs/{job_id}").json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_component_listing_shape(live: LiveServer, no_espeak) -> None:  # type: ignore[no-untyped-def]
    doc = httpx.get(live.url + "/api/v1/components").json()
    components = by_id(doc["components"])
    c = components["espeak-ng"]
    assert {
        "id", "name", "kind", "installed", "source", "version", "license", "license_url", "required_by_count",
        "installable", "install_blocked_reason", "manual_command", "description",
    } <= set(c)  # fmt: skip
    assert c["name"] == "eSpeak NG" and c["license"] == "GPL-3.0-or-later" and c["kind"] == "phonemizer"
    assert c["installed"] is False and c["source"] is None and c["version"] is None
    assert c["required_by_count"] >= 80 and c["installable"] is True and c["install_blocked_reason"] is None
    assert c["manual_command"] == "pip install 'bolo[espeak]'" and c["license_url"].startswith("https://")
    u = components["uccharan"]
    assert u["license"] == "Apache-2.0" and u["installed"] is False and u["installable"] is True
    assert u["manual_command"] == "pip install 'bolo[uccharan]'" and 20 <= u["required_by_count"] < 40
    assert set(u["languages"]) == {"en-us", "en-gb", "hi"}
    assert httpx.get(live.url + "/api/v1/info").json()["components"] == doc["components"]
    # English and Hindi Piper voices: uccharan first, espeak-ng as the alternative
    piper = httpx.get(live.url + "/api/v1/models/piper-en-us-amy-low").json()
    assert piper["requires"] == ["uccharan"] and piper["available"] is False
    assert [o["id"] for o in piper["phonemizers"]] == ["uccharan", "uccharan:improved", "espeak-ng"]
    assert all(set(o) == {"id", "available", "default", "note"} for o in piper["phonemizers"])
    assert [o["id"] for o in piper["phonemizers"] if o["default"]] == ["uccharan"]
    rohan = httpx.get(live.url + "/api/v1/models/piper-hi-in-rohan-medium").json()
    assert rohan["requires"] == ["uccharan"]
    german = httpx.get(live.url + "/api/v1/models/piper-de-de-thorsten-medium").json()
    assert german["requires"] == ["espeak-ng"]
    assert [(o["id"], o["available"], o["default"]) for o in german["phonemizers"]] == [
        ("uccharan", False, False),
        ("espeak-ng", False, True),
    ]
    assert "does not cover 'de-DE'" in german["phonemizers"][0]["note"]
    kokoro = httpx.get(live.url + "/api/v1/models/kokoro-v1.0-int8").json()
    assert kokoro["requires"] == [] and kokoro["available"] is True and kokoro["phonemizers"] == []
    voices = {v["id"]: v for v in kokoro["voices"]}
    assert [o["id"] for o in voices["hf_alpha"]["phonemizers"]] == ["uccharan", "uccharan:improved", "espeak-ng"]
    assert [o["id"] for o in voices["ef_dora"]["phonemizers"]] == ["uccharan", "espeak-ng"]
    assert "phonemizers" not in voices["zf_xiaobei"]
    for m in httpx.get(live.url + "/api/v1/catalog").json()["models"]:
        assert {"gated", "commercial_use", "requires", "available", "license"} <= set(m), m["id"]


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
    c = by_id(httpx.get(live.url + "/api/v1/components").json()["components"])["espeak-ng"]
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
    assert by_id(httpx.get(live.url + "/api/v1/components").json()["components"])["espeak-ng"]["installed"] is False


async def test_install_policy(settings: Settings, no_espeak, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    settings.host = "0.0.0.0"
    core = Bolo(settings)
    assert "non-loopback" in by_id(core.components.status())["espeak-ng"]["install_blocked_reason"]
    assert by_id(core.components.status())["espeak-ng"]["installable"] is False
    settings.allow_component_install = True  # even an explicit yes does not override that
    assert by_id(Bolo(settings).components.status())["espeak-ng"]["installable"] is False
    settings.api_key = "k"
    assert by_id(Bolo(settings).components.status())["espeak-ng"]["installable"] is True
    settings.allow_component_install = False
    assert "disabled" in by_id(Bolo(settings).components.status())["espeak-ng"]["install_blocked_reason"]
    settings.host, settings.api_key, settings.allow_component_install = "127.0.0.1", None, None
    core = Bolo(settings)
    monkeypatch.setattr("bolo.components.installer_command", lambda: None)
    assert "neither pip nor uv" in by_id(core.components.status())["espeak-ng"]["install_blocked_reason"]
    monkeypatch.setattr("bolo.components.installer_command", lambda: ["pip"])
    monkeypatch.setattr("os.access", lambda *a: False)
    assert "read-only" in by_id(core.components.status())["espeak-ng"]["install_blocked_reason"]


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
    assert "uccharan" in out and "Apache-2.0" in out
    assert main(["components", "install", "espeak-ng"]) == 1  # the licence must be accepted explicitly
    assert "--accept-license" in capsys.readouterr().err and not no_espeak["commands"]
    assert main(["components", "install", "espeak-ng", "--accept-license"]) == 0
    assert "installed espeak-ng" in capsys.readouterr().out
    assert main(["components", "list"]) == 0
    assert "installed (python-package, 1.52.0)" in capsys.readouterr().out
    assert main(["doctor"]) == 0
    assert "espeak-ng: found (python-package" in capsys.readouterr().out


def test_uccharan_component_covers_english_and_hindi_only(live: LiveServer, no_espeak) -> None:  # type: ignore[no-untyped-def]
    add_piper_like(live.core, "needs-en", "en-US")
    add_piper_like(live.core, "needs-de", "de-DE")
    live.core.components.run_installer = no_espeak["installer"]
    r = httpx.post(
        live.url + "/v1/audio/speech", json={"model": "needs-en", "input": "Hello.", "response_format": "pcm"}
    )
    assert r.status_code == 424 and "bolo components install uccharan" in r.json()["error"]["message"]
    # Apache-2.0: no licence acceptance needed
    r = httpx.post(live.url + "/api/v1/components/uccharan/install", json={})
    assert r.status_code == 202 and wait_job(live.url, r.json()["job_id"])["state"] == "done"
    assert no_espeak["commands"][0][-1].startswith("uccharan")
    en = httpx.get(live.url + "/api/v1/models/needs-en").json()
    assert en["available"] is True and en["requires"] == ["uccharan"]
    de = httpx.get(live.url + "/api/v1/models/needs-de").json()
    assert de["available"] is False and de["requires"] == ["espeak-ng"]
    ok = httpx.post(
        live.url + "/v1/audio/speech", json={"model": "needs-en", "input": "Hello.", "response_format": "pcm"}
    )
    assert ok.status_code == 200
    r = httpx.post(
        live.url + "/v1/audio/speech", json={"model": "needs-de", "input": "Hallo.", "response_format": "pcm"}
    )
    assert r.status_code == 424 and "espeak-ng" in r.json()["error"]["message"]


def test_phonemizer_preference(no_espeak) -> None:  # type: ignore[no-untyped-def]
    no_espeak["installed"] = no_espeak["uccharan"] = True
    try:
        assert phonemize.select("piper", "en-us").component == "uccharan"  # type: ignore[union-attr]
        assert phonemize.select("piper", "de").component == "espeak-ng"  # type: ignore[union-attr]
        assert phonemize.select("kitten", "en-gb").component == "espeak-ng"  # type: ignore[union-attr]
        phonemize.set_preference("espeak-ng")
        assert phonemize.select("piper", "en-us").component == "espeak-ng"  # type: ignore[union-attr]
        phonemize.set_preference("uccharan")
        assert phonemize.select("piper", "de") is None
        assert "phonemizer = 'auto'" in phonemize.missing_hint("piper", "de")
        with pytest.raises(ValueError):
            phonemize.set_preference("festival")
    finally:
        phonemize.set_preference("auto")
    assert phonemize.normalise_language("hi_IN") == "hi" and phonemize.normalise_language("en-GB") == "en-gb"
    assert phonemize.uccharan_language("piper", "en-gb-x-rp") == "en-gb"


def test_choosing_the_phonemizer_per_request_and_per_voice(live: LiveServer, no_espeak) -> None:  # type: ignore[no-untyped-def]
    add_piper_like(live.core, "needs-en", "en-US")
    add_piper_like(live.core, "needs-de", "de-DE")
    url = live.url + "/v1/audio/speech"

    def speak(model: str, phonemizer: str | None) -> httpx.Response:
        body = {"model": model, "input": "Hello.", "response_format": "pcm", "phonemizer": phonemizer}
        return httpx.post(url, json=body)

    no_espeak["uccharan"] = True
    assert speak("needs-en", None).status_code == 200
    assert speak("needs-en", "uccharan:improved").status_code == 200
    r = speak("needs-en", "espeak-ng")  # chosen but not installed
    assert r.status_code == 424 and r.json()["error"]["code"] == "component_missing"
    r = speak("needs-de", "uccharan")  # uccharan does not speak German
    assert (
        r.status_code == 400 and "en-us" in r.json()["error"]["message"] and r.json()["error"]["param"] == "phonemizer"
    )
    r = speak("needs-en", "festival")
    assert r.status_code == 400 and "espeak-ng" in r.json()["error"]["message"]
    # pin a voice's phonemiser: persisted, visible in the catalog, used when the request names none
    put = httpx.put(live.url + "/api/v1/models/needs-en/settings", json={"phonemizer": "espeak-ng"})
    assert put.status_code == 200 and put.json()["settings"] == {"phonemizer": "espeak-ng"}
    entry = httpx.get(live.url + "/api/v1/models/needs-en").json()
    assert entry["settings"] == {"phonemizer": "espeak-ng"} and entry["requires"] == ["espeak-ng"]
    assert speak("needs-en", None).status_code == 424  # the pinned espeak-ng is missing
    assert speak("needs-en", "uccharan").status_code == 200  # the request still wins
    no_espeak["installed"] = True
    assert speak("needs-en", None).status_code == 200
    assert httpx.put(live.url + "/api/v1/models/needs-de/settings", json={"phonemizer": "uccharan"}).status_code == 400
    assert httpx.put(live.url + "/api/v1/models/fake-tts/settings", json={"phonemizer": "uccharan"}).status_code == 400
    assert httpx.put(live.url + "/api/v1/models/needs-en/settings", json={"phonemizer": None}).json()["settings"] == {}
    assert httpx.get(live.url + "/api/v1/models/needs-en/settings").json() == {"model": "needs-en", "settings": {}}
    assert (live.core.settings.data_dir / "model-settings.json").is_file()


def test_cli_models_config(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("BOLO_DATA_DIR", str(tmp_path))
    assert main(["models", "config", "piper-en-us-amy-low", "--phonemizer", "espeak-ng"]) == 0
    assert '"phonemizer": "espeak-ng"' in capsys.readouterr().out
    assert main(["models", "config", "piper-de-de-thorsten-medium", "--phonemizer", "uccharan"]) == 1
    assert "does not cover" in capsys.readouterr().err
    assert main(["models", "config", "piper-en-us-amy-low", "--clear"]) == 0
    assert '"settings": {}' in capsys.readouterr().out
