"""Out-of-process engine plugins, exercised with the reference plugin (the fake engine in a subprocess)."""

from __future__ import annotations

import io
import sys
import time
from collections.abc import Iterator

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
        assert (
            proc.running
            and proc.hello["protocol"] == 1
            and proc.hello["kinds"]
            == [
                "fake-stt",
                "fake-tts",
                "fake-separation",
                "fake-stt-speakers",
                "fake-diarization",
                "fake-denoise",
                "fake-s2s",
                "fake-lid",
            ]
        )
        assert len(proc.handles) == 2
        assert core.engines.status("plugin:ref").available
    finally:
        await core.stop()
    assert not core.engines.get("plugin:ref").process.running  # type: ignore[union-attr]


async def test_plugin_diarization_through_the_scheduler(plugin_settings: Settings) -> None:
    """A diarization model in a plugin (ops ``diarize`` and ``embed``): bolo's speaker turns, speaker counts and
    voice prints as for its own diarizers."""
    from bolo.diarize import DiarizeOptions, KnownSpeaker

    core = Bolo(plugin_settings)
    core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": "plug-diar", "task": "diarization", "engine": "plugin:ref", "kind": "fake-diarization",
                "family": "Fake", "languages": [], "license": "Apache-2.0", "builtin": True, "sample_rate": 16000,
            }
        )
    )  # fmt: skip
    await core.start()
    try:
        rate = 16000
        t = np.arange(rate) / rate
        low, high, gap = 0.3 * np.sin(2 * np.pi * 200 * t), 0.3 * np.sin(2 * np.pi * 600 * t), np.zeros(rate // 2)
        audio = np.concatenate([low, gap, high, gap, low, gap, high]).astype(np.float32)
        turns = await core.diarize(audio, rate, DiarizeOptions(model="plug-diar"))
        assert [x.speaker for x in turns] == [0, 1, 0, 1] and abs(turns[1].start - 1.5) < 0.1
        one = await core.diarize(audio, rate, DiarizeOptions(model="plug-diar", max_speakers=1))
        assert {x.speaker for x in one} == {0}  # bolo asks the plugin again with num_speakers
        opts = DiarizeOptions(model="plug-diar", known=[KnownSpeaker("Asha", high.astype(np.float32))])
        await core.diarize(audio, rate, opts)
        assert opts.names[1] == "Asha"  # the plugin's embeddings name a known voice
        model = core.manager.loaded["plug-diar"].model
        assert type(model).__name__ == "PluginDiarization"
    finally:
        await core.stop()


async def test_plugin_language_identification(plugin_settings: Settings) -> None:
    """A language identifier in a plugin (op ``lid``): its labels and window come with the load, the identify
    answer is the plugin's, and routing uses it like a built-in one."""
    core = Bolo(plugin_settings)
    core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": "plug-lid", "task": "lid", "engine": "plugin:ref", "kind": "fake-lid", "family": "Fake",
                "languages": ["en", "hi", "ta", "fr"], "license": "Apache-2.0", "builtin": True,
                "options": {"max_s": 12},
            }
        )
    )  # fmt: skip
    core.settings.default_lid = "plug-lid"
    await core.start()
    try:
        rate = 16000
        t = np.arange(2 * rate) / rate
        tamil = (0.3 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
        result = await core.identify_language(tamil, rate)
        assert result.model == "plug-lid" and result.language == "ta" and abs(result.probability - 0.9) < 1e-6
        model = core.manager.loaded["plug-lid"].model
        assert type(model).__name__ == "PluginLid" and model.labels == ["en", "hi", "ta", "fr"] and model.max_s == 12
        assert result.heard_s == 2.0
    finally:
        await core.stop()


def test_respond_turns_roundtrip() -> None:
    from bolo.engines.base import ConversationTurn

    turns = [
        ConversationTurn("user", audio=np.linspace(-1, 1, 5, dtype=np.float32)),
        ConversationTurn("assistant", text="Hi.", tool_calls=[{"call_id": "c1", "name": "f", "arguments": "{}"}]),
        ConversationTurn("tool", text="42", call_id="c1"),
        ConversationTurn("user", audio=np.ones(3, dtype=np.float32)),
    ]
    items, payload = protocol.turns_message(turns)
    assert [i["n"] for i in items] == [5, 0, 0, 3] and len(payload) == 32
    back = protocol.turns_from_message(items, payload)
    assert back[1].tool_calls == turns[1].tool_calls and back[2].call_id == "c1"
    assert np.array_equal(back[0].audio, turns[0].audio) and np.array_equal(back[3].audio, turns[3].audio)  # type: ignore[arg-type]
    with pytest.raises(protocol.ProtocolError):
        protocol.turns_from_message(items, payload[:-4])


async def test_plugin_speech_to_speech(plugin_settings: Settings) -> None:
    """A turn-based speech-to-speech model in a plugin (op ``respond``) behind the speech-to-speech pipeline: bolo
    keeps the conversation and sends it whole every turn; text, audio and tool calls come back; barge-in stops it."""
    import asyncio

    from bolo.s2s import AudioDelta, ModelS2SBackend, ResponseEnd, S2SConfig, TextDelta, ToolCall

    core = Bolo(plugin_settings)
    core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": "plug-s2s", "task": "speech-to-speech", "engine": "plugin:ref", "kind": "fake-s2s",
                "family": "Fake", "languages": ["en"], "license": "Apache-2.0", "builtin": True,
                "sample_rate": 24000, "features": ["speech-to-speech"], "options": {"word_delay_s": 0.05},
            }
        )
    )  # fmt: skip
    core.settings.s2s_model = "plug-s2s"
    await core.start()
    try:
        assert core.s2s.describe()["available"]
        backend = await core.s2s.backend()
        assert isinstance(backend, ModelS2SBackend) and backend.output_rate == 24000
        tools = [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}]
        conv = await backend.open(S2SConfig(instructions="Be brief.", tools=tools))
        await conv.append_audio(np.zeros(8000, dtype=np.float32))
        await conv.append_audio(np.zeros(8000, dtype=np.float32))
        await conv.commit()
        events = [e async for e in conv.respond(modalities=["audio", "text"])]
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "I heard 1.0 seconds."
        audio = [e for e in events if isinstance(e, AudioDelta)]
        assert audio and audio[0].sample_rate == 24000 and len(audio[0].samples) == 2400
        assert isinstance(events[-1], ResponseEnd) and events[-1].status == "completed"

        await conv.add_text("What is the weather?")
        events = [e async for e in conv.respond(modalities=["audio"])]
        call = next(e for e in events if isinstance(e, ToolCall))
        assert call.name == "get_weather" and call.arguments == '{"city": "Pune"}'
        await conv.add_tool_output(call.call_id, "sunny")
        events = [e async for e in conv.respond(modalities=["text"])]
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "The tool said: sunny."
        assert not any(isinstance(e, AudioDelta) for e in events)  # a text-only response
        roles = [t.role for t in conv.turns]  # type: ignore[attr-defined]
        assert roles == ["user", "assistant", "user", "assistant", "tool", "assistant"]

        # barge-in: closing the response stops the plugin's generation (op cancel)
        await conv.add_text("one two three four five six seven eight nine ten")
        gen = conv.respond(modalities=["audio"])
        first = await gen.__anext__()
        assert isinstance(first, TextDelta)
        await gen.aclose()
        await asyncio.sleep(0.3)
        said = conv.turns[-1]  # type: ignore[attr-defined]
        assert said.role == "assistant" and said.text.startswith("You") and not said.text.endswith("ten.")
        await conv.close()
        model = core.manager.loaded["plug-s2s"].model
        assert type(model).__name__ == "PluginS2S" and model.supports_tools  # type: ignore[attr-defined]
    finally:
        await core.stop()


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


@pytest.fixture
def live_without_installed_plugins(settings: Settings) -> Iterator[LiveServer]:
    """A live server that does not discover plugins installed as packages (entry points). With bolo-torch installed
    in the same environment, its entry point would register plugin:torch and make the torch catalog entries
    available, which the assertions about a missing plugin rely on not happening."""
    settings.plugin_entry_points = False
    srv = LiveServer(settings)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def test_gated_entries_explain_what_to_do(live_without_installed_plugins: LiveServer) -> None:
    """A gated catalog entry is never pulled without a token, and the error says how to get access."""
    live = live_without_installed_plugins
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


def test_plugin_commands_keep_windows_paths() -> None:
    from bolo.plugin.host import split_command

    windows = split_command(r'C:\Python312\python.exe -m bolo_torch "C:\Program Files\x"', windows=True)
    assert windows == [r"C:\Python312\python.exe", "-m", "bolo_torch", r"C:\Program Files\x"]
    assert split_command("python -m x 'a b'", windows=False) == ["python", "-m", "x", "a b"]
