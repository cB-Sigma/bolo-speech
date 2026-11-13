"""Out-of-process engine plugins, exercised with the reference plugin (the fake engine in a subprocess)."""

from __future__ import annotations

import io
import sys
import time

import httpx
import numpy as np
import pytest

from bolo.catalog import ModelSpec
from bolo.config import Settings
from bolo.core import Bolo
from bolo.errors import EngineUnavailable
from bolo.plugin import host, protocol
from bolo.plugin.host import PluginConfig, PluginProcess, parse_plugin_setting
from conftest import LiveServer

REFERENCE = f"ref={sys.executable} -m bolo.plugin.reference"


def add_plugin_models(core: Bolo, plugin: str = "ref") -> None:
    for model_id, kind, task in (("plug-tts", "fake-tts", "tts"), ("plug-stt", "fake-stt", "stt")):
        core.catalog.add(
            ModelSpec.from_dict(
                {
                    "id": model_id, "task": task, "engine": f"plugin:{plugin}", "kind": kind, "family": "Fake",
                    "languages": ["en"], "license": "Apache-2.0", "builtin": True, "sample_rate": 24000,
                    "voices": [{"id": "alpha", "language": "en-US"}],
                    "options": {"plugin_install": "pip install bolo-reference-plugin"},
                }
            )
        )  # fmt: skip


@pytest.fixture
def plugin_settings(settings: Settings) -> Settings:
    settings.plugins = [REFERENCE]
    settings.plugin_entry_points = False
    return settings


def test_protocol_roundtrip() -> None:
    buf = io.BytesIO()
    audio = np.linspace(-1, 1, 7, dtype=np.float32)
    protocol.write_message(buf, {"id": "1", "op": "x"}, protocol.audio_bytes(audio))
    protocol.write_message(buf, {"id": "2", "ok": True})
    buf.seek(0)
    m, payload = protocol.read_message(buf)  # type: ignore[misc]
    assert m == {"id": "1", "op": "x", "bytes": 28} and np.array_equal(protocol.audio_from_bytes(payload), audio)
    assert protocol.read_message(buf) == ({"id": "2", "ok": True}, b"")
    assert protocol.read_message(buf) is None
    with pytest.raises(protocol.ProtocolError):
        protocol.read_message(io.BytesIO(b"[1]\n"))
    with pytest.raises(protocol.ProtocolError):
        protocol.read_message(io.BytesIO(b'{"bytes": 10}\n123'))
    assert parse_plugin_setting("torch=bolo-torch --threads 2") == ("torch", ["bolo-torch", "--threads", "2"])
    with pytest.raises(ValueError):
        parse_plugin_setting("no-command")


async def test_plugin_models_through_the_scheduler(plugin_settings: Settings) -> None:
    core = Bolo(plugin_settings)
    add_plugin_models(core)
    await core.start()
    try:
        audio, rate = await core.synthesize_all("Hello from a plugin. Second sentence.", model="plug-tts")
        assert rate == 24000 and len(audio) > 2400
        samples = np.zeros(32000, dtype=np.float32)
        samples[4000:28000] = 0.3 * np.sin(np.arange(24000) / 5)
        result = await core.transcribe(samples, 16000, model="plug-stt", vad_filter=False)
        assert result.text.strip()
        engine = core.engines.get("plugin:ref")
        proc = engine.process  # type: ignore[union-attr]
        assert proc.running and proc.hello["protocol"] == 1 and proc.hello["kinds"] == ["fake-stt", "fake-tts"]
        assert len(proc.handles) == 2
        assert core.engines.status("plugin:ref").available
    finally:
        await core.stop()
    assert not core.engines.get("plugin:ref").process.running  # type: ignore[union-attr]


async def test_crash_restart_and_reload(plugin_settings: Settings) -> None:
    core = Bolo(plugin_settings)
    add_plugin_models(core)
    await core.start()
    try:
        await core.synthesize_all("Before.", model="plug-tts")
        proc: PluginProcess = core.engines.get("plugin:ref").process  # type: ignore[union-attr]
        first_pid = proc.status()["pid"]
        with pytest.raises(Exception, match=r"stopped|plugin"):
            await core.synthesize_all("Plugin crash test.", model="plug-tts")
        for _ in range(50):
            if not proc.running and proc.crashes:
                break
            time.sleep(0.05)
        assert proc.crashes == 1 and not proc.running
        # within the back-off window the plugin is not restarted
        with pytest.raises(Exception, match="restarting"):
            await core.synthesize_all("Too soon.", model="plug-tts")
        proc._not_before = 0.0
        audio, _ = await core.synthesize_all("After the restart.", model="plug-tts")  # reloads the model transparently
        assert len(audio) > 0 and proc.restarts == 1 and proc.status()["pid"] != first_pid
    finally:
        await core.stop()


def test_idle_shutdown_and_cancel() -> None:
    proc = PluginProcess(PluginConfig("ref", [sys.executable, "-m", "bolo.plugin.reference"], idle_s=0.2))
    model = host.PluginTTS(
        proc, {"op": "load", "kind": "fake-tts", "task": "tts", "path": "/nonexistent", "options": {}, "threads": 1}
    )
    chunks = list(model.synthesize("One. Two."))
    assert chunks and all(c.dtype == np.float32 for c in chunks)
    t0 = time.monotonic()
    list(model.synthesize("A cancelled request still ends. " * 5, cancelled=lambda: True))
    assert time.monotonic() - t0 < 5
    assert proc.request({"op": "health"})["ok"] is True
    model.close()
    for _ in range(40):
        if not proc.running:
            break
        time.sleep(0.05)
    assert not proc.running  # stopped after idle_s without models
    proc.stop()


def test_missing_plugin_is_reported(live: LiveServer) -> None:
    add_plugin_models(live.core, plugin="torchy")
    entry = httpx.get(live.url + "/api/v1/models/plug-tts").json()
    assert entry["available"] is False and entry["requires"] == ["plugin:torchy"]
    comp = {c["id"]: c for c in httpx.get(live.url + "/api/v1/components").json()["components"]}["plugin:torchy"]
    assert comp["installed"] is False and comp["kind"] == "engine-plugin" and comp["installable"] is False
    assert comp["manual_command"] == "pip install bolo-reference-plugin" and comp["required_by_count"] == 2
    r = httpx.post(live.url + "/v1/audio/speech", json={"model": "plug-tts", "input": "Hi.", "response_format": "pcm"})
    assert r.status_code == 424 and "engine plugin 'torchy'" in r.json()["error"]["message"]
    assert httpx.post(live.url + "/api/v1/components/plugin:torchy/install", json={}).status_code == 400


def test_entry_point_discovery(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    class EP:
        def __init__(self, name: str, module: str, attr: str | None) -> None:
            self.name, self.module, self.attr = name, module, attr

        def load(self):  # type: ignore[no-untyped-def]
            return lambda: ["my-plugin", "--serve"]

    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group: [EP("ref", "bolo.plugin.reference", None), EP("other", "x.y", "command")],
    )
    found = host.discover_entry_points()
    assert found == {"ref": [sys.executable, "-m", "bolo.plugin.reference"], "other": ["my-plugin", "--serve"]}
    s = Settings(data_dir=tmp_path, plugins=["other=/bin/true"])
    from bolo.engines import EngineRegistry

    reg = EngineRegistry(settings=s)
    assert {"plugin:ref", "plugin:other"} <= set(reg.ids())
    assert reg.get("plugin:other").config.command == ["/bin/true"]  # type: ignore[union-attr]
    assert reg.get("plugin:other").config.source == "settings"  # type: ignore[union-attr]


def test_unknown_command_and_errors() -> None:
    proc = PluginProcess(PluginConfig("nope", ["/nonexistent/plugin-binary"]))
    with pytest.raises(EngineUnavailable, match="could not be started"):
        proc.ensure_started()
    assert host.PluginEngine(PluginConfig("nope", ["/nonexistent/plugin-binary"])).status().available is False
    ref = PluginProcess(PluginConfig("ref", [sys.executable, "-m", "bolo.plugin.reference"]))
    ref.ensure_started()
    try:
        with pytest.raises(EngineUnavailable, match="unknown_op"):
            ref.request({"op": "dance"})
        with pytest.raises(EngineUnavailable, match="KeyError"):
            ref.request({"op": "synthesize", "handle": "h99", "text": "x"})
    finally:
        ref.stop()


def test_gated_entries_explain_what_to_do(live: LiveServer) -> None:
    """A gated catalog entry is never pulled without a token, and the error says how to get access."""
    live.core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": "gated-voice", "task": "tts", "engine": "fake", "kind": "fake-tts", "family": "Fake",
                "languages": ["hi"], "license": "MIT", "gated": True, "source_url": "https://huggingface.co/x/y",
                "source": {"repo": "x/y", "revision": "0" * 40},
            }
        )
    )  # fmt: skip
    live.core.store.hub.token = None
    entry = httpx.get(live.url + "/api/v1/models/gated-voice").json()
    assert entry["gated"] is True and entry["commercial_use"] is True
    r = httpx.post(live.url + "/v1/audio/speech", json={"model": "gated-voice", "input": "x", "response_format": "pcm"})
    err = r.json()["error"]
    assert r.status_code == 401 and err["code"] == "model_gated", err
    assert "https://huggingface.co/x/y" in err["message"] and "HF_TOKEN" in err["message"]
    pull = httpx.post(live.url + "/api/v1/models/pull", json={"id": "gated-voice"}).json()
    job = httpx.get(f"{live.url}/api/v1/jobs/{pull['job_id']}").json()
    assert job["state"] == "error" and "accept its terms" in job["error"].lower()
    parler = httpx.get(live.url + "/api/v1/models/indic-parler-tts").json()
    assert parler["gated"] is True and parler["requires"] == ["plugin:torch"] and parler["available"] is False
