"""Hugging Face remote code: the catalog pin, the operator's per-model opt-in, and what reaches the plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from bolo import remote_code
from bolo.catalog import Catalog, CatalogError, ModelSpec
from bolo.cli import _model_setting
from bolo.config import Settings
from bolo.core import Bolo
from bolo.plugin.host import PluginEngine
from conftest import LiveServer
from test_plugins import REFERENCE

REV = "c" * 40
PIN = {"revision": REV, "reviewed": "read", "files": {"modeling_x.py": "0" * 64, "config.json": "1" * 64}}


def entry(**options: Any) -> dict[str, Any]:
    return {
        "id": "rc-voice", "task": "tts", "engine": "plugin:ref", "kind": "fake-tts", "family": "Fake",
        "languages": ["en"], "license": "MIT", "builtin": True, "voices": [{"id": "alpha"}],
        "source": {"repo": "org/x", "revision": REV}, "options": {"remote_code": PIN, **options},
    }  # fmt: skip


@pytest.mark.parametrize(
    ("pin", "message"),
    [
        ({**PIN, "revision": "d" * 40}, "is not the source revision"),
        ({**PIN, "revision": "main"}, "40-character commit"),
        ({**PIN, "files": {"modeling_x.py": "abc"}}, "needs a lowercase sha256"),
        ({**PIN, "files": {"config.json": "1" * 64}}, "lists no .py file"),
        ("yes", "must be a mapping"),
    ],
)
def test_catalog_pins_are_checked(pin: Any, message: str) -> None:
    with pytest.raises(CatalogError, match=message):
        ModelSpec.from_dict({**entry(), "options": {"remote_code": pin}})


def test_packaged_remote_code_entries_are_pinned() -> None:
    entries = [s for s in Catalog.load().all() if remote_code.pin(s) is not None]
    assert {"step-audio-2-mini", "glm-4-voice-9b", "kimi-audio-7b-instruct", "minicpm-o-2.6"} <= {s.id for s in entries}
    assert "minicpm-o-4.5" in {s.id for s in entries}  # bolo's own duplex server, files pinned by URL
    for spec in entries:
        pin = remote_code.pin(spec)
        assert pin is not None and pin.get("reviewed") and spec.runnable, spec.id
        if spec.source.repo:  # a repository snapshot: the reviewed commit is the one pulled
            assert pin["revision"] == spec.source.revision, spec.id
        else:  # files pinned one by one: each reviewed file is downloaded from the reviewed commit
            urls = {u["path"]: u["url"] for u in spec.source.urls}
            for name in pin["files"]:
                assert f"/resolve/{pin['revision']}/" in urls.get(name, ""), (spec.id, name)
        if spec.engine == "plugin:torch":  # the plugin's install hint names the opt-in
            assert "allow_remote_code" in str(spec.options["plugin_install"]), spec.id
        else:
            assert "allow_remote_code" in spec.description, spec.id


def test_the_opt_in_is_the_operators(tmp_path: Path) -> None:
    spec = ModelSpec.from_dict(entry(allow_remote_code=True))  # a catalog entry cannot opt in
    assert remote_code.notice(spec, {}) is not None and "will not load until you opt in" in str(
        remote_code.notice(spec, {})
    )
    assert "You opted in" in str(remote_code.notice(spec, {"allow_remote_code": True}))
    assert remote_code.allowed({"allow_remote_code": "true"}) is False and remote_code.allowed({}) is False
    api = spec.to_api(settings={})
    assert api["remote_code"]["allowed"] is False and api["remote_code"]["files"] == ["config.json", "modeling_x.py"]
    assert ModelSpec.from_dict({**entry(), "options": {}}).to_api()["remote_code"] is None
    assert _model_setting("allow_remote_code", "TRUE") is True and _model_setting("allow_remote_code", "off") is False
    with pytest.raises(ValueError):
        _model_setting("allow_remote_code", "maybe")


async def test_plugin_gets_the_setting_never_the_catalogs(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    settings.plugins = [REFERENCE]
    settings.plugin_entry_points = False
    seen: list[dict[str, Any]] = []
    original = PluginEngine.load

    def spy(self: PluginEngine, kind: str, task: str, path: Path, spec: dict[str, Any], opts: Any) -> Any:
        seen.append(dict(spec))
        return original(self, kind, task, path, spec, opts)

    monkeypatch.setattr(PluginEngine, "load", spy)
    core = Bolo(settings)
    core.catalog.add(ModelSpec.from_dict(entry(allow_remote_code=True)))
    await core.start()
    try:
        await core.synthesize_all("Hello.", model="rc-voice")
        assert seen[-1]["allow_remote_code"] is False
        await core.manager.unload("rc-voice")
        core.store.set_model_settings("rc-voice", {"allow_remote_code": True})
        await core.synthesize_all("Hello.", model="rc-voice")
        assert seen[-1]["allow_remote_code"] is True and seen[-1]["remote_code"]["revision"] == REV
        assert "quantization" not in seen[-1]
        await core.manager.unload("rc-voice")
        core.store.set_model_settings("rc-voice", {"quantization": "int4"})  # a pin reaches the plugin's options
        await core.synthesize_all("Hello.", model="rc-voice")
        assert seen[-1]["quantization"] == "int4"
    finally:
        await core.stop()


def test_settings_api_and_pull_notice(live: LiveServer) -> None:
    live.core.catalog.add(ModelSpec.from_dict({**entry(), "builtin": False}))
    put = httpx.put(live.url + "/api/v1/models/rc-voice/settings", json={"allow_remote_code": True})
    assert put.status_code == 200 and put.json()["settings"] == {"allow_remote_code": True}
    listed = httpx.get(live.url + "/api/v1/models/rc-voice").json()
    assert listed["remote_code"]["allowed"] is True and listed["remote_code"]["revision"] == REV
    assert (
        httpx.put(live.url + "/api/v1/models/rc-voice/settings", json={"allow_remote_code": None}).json()["settings"]
        == {}
    )
    live.core.store.hub.list_files = lambda repo, revision="main": []  # type: ignore[method-assign]
    pull = httpx.post(live.url + "/api/v1/models/pull", json={"id": "rc-voice"}).json()
    assert any(w["kind"] == "remote_code" and "allow_remote_code=true" in w["message"] for w in pull["warnings"])
