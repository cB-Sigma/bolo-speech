"""Romanised text for text-to-speech: modes, feature detection, graceful degradation, and every way in.

The transliterator is uccharan's and optional. Most tests put a small fake in its place, so they pass with any
uccharan (or none); the last ones run against the installed uccharan when it has the API and a Hindi pack.
"""

from __future__ import annotations

import json
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import websockets

from bolo.catalog import ModelSpec
from bolo.config import Settings, load_settings
from bolo.core import Bolo
from bolo.errors import ComponentMissing, InvalidRequest
from bolo.telephony.config import AgentConfig
from bolo.text import romanised as rom
from bolo.text.prepare import prepare, text_input
from conftest import LiveServer
from fake_tts_server import FakeTTSServer

WORDS = {"mujhe": "मुझे", "kal": "कल", "jana": "जाना", "hai": "है", "aap": "आप", "kaise": "कैसे", "ho": "हो"}
ROMAN = "mujhe kal office jana hai"
NATIVE = "मुझे कल office जाना है"


class FakeSpan:
    def __init__(self, kind: str, text: str, written: str) -> None:
        self.kind, self.text, self.written = kind, text, written


class FakeResult:
    def __init__(self, text: str, spans: list[FakeSpan]) -> None:
        self.text, self.spans, self.pack = text, spans, True


def fake_uccharan(*, packs: tuple[str, ...] = ("xlit-hi",), loaded: tuple[str, ...] = ("hi",)) -> Any:
    """A module shaped like a uccharan with transliteration: ``transliterate``, ``has_translit``, ``load_pack``
    and a pack table. ``loaded`` are the languages whose pack is in memory."""
    mod = types.ModuleType("uccharan_fake")
    state = {"loaded": set(loaded), "calls": []}

    def transliterate(text: str, lang: str = "hi", *, english: str = "keep", alternatives: int = 3) -> FakeResult:
        state["calls"].append((text, lang, english, alternatives))
        out: list[str] = []
        spans: list[FakeSpan] = []
        for token in text.split(" "):
            core = token.strip(".,!?")
            native = WORDS.get(core.lower())
            written = token.replace(core, native) if native else token
            spans.append(FakeSpan("indic" if native else "english", token, written))
            out.append(written)
        return FakeResult(" ".join(out), spans)

    def load_pack(name: str) -> str:
        if name not in mod.packs.PACKS:
            raise RuntimeError(f"unknown pack {name!r}")
        state["loaded"].add(name.removeprefix("xlit-"))
        return name

    mod.__version__ = "9.9.9"  # type: ignore[attr-defined]
    mod.transliterate = transliterate  # type: ignore[attr-defined]
    mod.has_translit = lambda lang: lang in state["loaded"]  # type: ignore[attr-defined]
    mod.load_pack = load_pack  # type: ignore[attr-defined]
    mod.packs = types.SimpleNamespace(  # type: ignore[attr-defined]
        PACKS=dict.fromkeys(packs, ("0" * 64, 1)), cache_dir=lambda: Path("/nonexistent/uccharan-cache")
    )
    mod.state = state  # type: ignore[attr-defined]
    return mod


def use(monkeypatch: pytest.MonkeyPatch, module: Any | None, *, function: bool = True) -> Any:
    """Make bolo see ``module`` as the installed uccharan (None: not installed)."""
    found = rom.Api(module, module.transliterate if module is not None and function else None, "9.9.9")
    monkeypatch.setattr(rom, "_api", found if module is not None else rom.Api())
    monkeypatch.setattr(rom, "_loaded", set())
    return module


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Any:
    return use(monkeypatch, fake_uccharan())


def hindi_voice(model_id: str = "hi-voice", **options: object) -> dict[str, object]:
    return {
        "id": model_id,
        "task": "tts",
        "engine": "tts-server",
        "kind": "tts-server",
        "family": "Test",
        "languages": ["hi"],
        "sample_rate": 24000,
        "license": "MIT",
        "requires": ["llama-server"],
        "source": {"local": "/nonexistent"},
        "voices": [{"id": "default", "name": "Model default", "server_voice": "", "language": "hi"}],
        # the model's own normaliser: what reaches the fake server is exactly what text preparation produced
        "options": {"runtime": "llama-server", "text_normalizer": "none", **options},
    }


@pytest.fixture
def llama() -> Iterator[FakeTTSServer]:
    server = FakeTTSServer("llama", rate=16000, voices=[]).start()
    yield server
    server.stop()


@pytest.fixture
def live(settings: Settings, llama: FakeTTSServer) -> Iterator[LiveServer]:
    settings.tts_servers = [f"hi-voice={llama.url}", f"en-voice={llama.url}"]
    srv = LiveServer(settings)
    srv.core.catalog.add(ModelSpec.from_dict(hindi_voice()))
    english = hindi_voice("en-voice")
    english["languages"] = ["en"]
    english["voices"] = [{"id": "default", "name": "Model default", "server_voice": "", "language": "en"}]
    srv.core.catalog.add(ModelSpec.from_dict(english))
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def sent(llama: FakeTTSServer) -> str:
    return " ".join(str(r["body"].get("input", "")) for r in llama.requests if r["path"] == "/tts")


def speak(srv: LiveServer, llama: FakeTTSServer, **body: Any) -> httpx.Response:
    llama.requests.clear()
    return httpx.post(
        f"{srv.url}/v1/audio/speech", json={"model": "hi-voice", "response_format": "wav", **body}, timeout=30
    )


def last_trace(srv: LiveServer) -> dict[str, Any]:
    traces = httpx.get(f"{srv.url}/api/v1/requests", params={"kind": "tts"}).json()["requests"]
    return dict(traces[0])


# -- the option itself ---------------------------------------------------------------------------------------------
def test_modes_are_parsed() -> None:
    assert [rom.parse_mode(v) for v in ("off", "AUTO", " on ", True, False, "true", "0", None, "")] == [
        "off", "auto", "on", "on", "off", "on", "off", None, None,
    ]  # fmt: skip
    with pytest.raises(InvalidRequest, match="off, auto, on"):
        rom.parse_mode("sometimes")


def test_latin_share() -> None:
    assert rom.latin_share("mujhe kal jana hai") == (15, 1.0)
    assert rom.latin_share("मुझे कल जाना है") == (0, 0.0)
    latin, share = rom.latin_share("मुझे कल office जाना है")
    assert latin == 6 and 0.3 < share < 0.6
    assert rom.latin_share("123 !?") == (0, 0.0)


def test_off_and_untouched_text(fake: Any) -> None:
    assert rom.apply(ROMAN, "hi", "off").record is None
    # auto leaves alone: voices of other languages, text already in the script, mixed text that is mostly native
    for text, lang in ((ROMAN, "en"), ("मुझे कल जाना है", "hi"), ("मुझे कल office जाना है", "hi-IN")):
        out = rom.apply(text, lang, "auto")
        assert out.text == text and out.record is None
    assert fake.state["calls"] == []


def test_auto_and_on_convert(fake: Any) -> None:
    out = rom.apply(ROMAN, "hi-IN", "auto", source="voice")
    assert out.text == NATIVE
    assert out.record == {
        "mode": "auto", "source": "voice", "language": "hi", "applied": True, "latin_share": 1.0,
        "pack": "xlit-hi", "text": NATIVE, "words": 4,
    }  # fmt: skip
    assert fake.state["calls"] == [(ROMAN, "hi", "keep", 0)]  # English words stay; no alternatives are asked for
    # `on` also converts text that is mostly native script already
    mixed = rom.apply("मुझे कल office jana है", "hi", "on")
    assert mixed.text == "मुझे कल office जाना है" and mixed.record is not None and mixed.record["applied"] is True
    # ... and says so when the voice is not one it applies to
    english = rom.apply(ROMAN, "en", "on")
    assert english.text == ROMAN and "Indian languages" in (english.notice or "")
    with pytest.raises(InvalidRequest, match="Indian languages"):
        rom.apply(ROMAN, "en", "on", strict=True)


def test_result_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever the function returns (an object, a dict, JSON text, plain text), and with or without options."""
    mod = fake_uccharan()
    for result, words in (
        ({"text": NATIVE, "spans": [{"kind": "indic"}, {"kind": "english"}]}, 1),
        (json.dumps({"text": NATIVE, "spans": [{"kind": "indic"}]}), 1),
        (NATIVE, None),
    ):
        mod.transliterate = lambda text, lang, _r=result, **kw: _r
        use(monkeypatch, mod)
        out = rom.apply(ROMAN, "hi", "on")
        assert out.text == NATIVE and out.record is not None and out.record.get("words") == words
    mod.transliterate = lambda text, lang: NATIVE  # an API without `english` / `alternatives`
    use(monkeypatch, mod)
    assert rom.apply(ROMAN, "hi", "on").text == NATIVE


def test_protected_words_and_addresses(fake: Any) -> None:
    """Word-list entries are matched as written and addresses are read out: neither is converted."""
    out = rom.apply("kal Hai Tech jana hai, www.kal.example dekho", "hi", "on", protect=["Hai Tech", "absent"])
    assert out.text == "कल Hai Tech जाना है, www.kal.example dekho"
    assert [c[0] for c in fake.state["calls"]] == ["kal ", " jana hai, ", " dekho"]
    assert rom.apply("mail kal@hai.example", "hi", "on").text == "mail kal@hai.example"


# -- degradation ---------------------------------------------------------------------------------------------------
def test_without_uccharan(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, None)
    assert rom.status() == {
        "available": False,
        "reason": "romanised text needs the uccharan package with transliteration support: pip install -U uccharan",
        "modes": ["off", "auto", "on"],
        "languages": [],
        "installed": [],
    }
    for mode in ("auto", "on"):
        out = rom.apply(ROMAN, "hi", mode)
        assert out.text == ROMAN and out.record is not None and out.record["applied"] is False
        assert "pip install -U uccharan" in (out.notice or "")
    assert rom.apply(ROMAN, "hi", "auto", strict=True).text == ROMAN  # `auto` means "when you can"
    with pytest.raises(ComponentMissing, match="uccharan package"):
        rom.apply(ROMAN, "hi", "on", strict=True)
    assert rom.install_pack("hi") is not None


def test_uccharan_without_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """uccharan 0.3.0 and older: installed, no ``transliterate``."""
    old = types.ModuleType("uccharan_old")
    old.__version__ = "0.3.0"  # type: ignore[attr-defined]
    monkeypatch.setattr(rom, "_api", rom.Api(old, None, "0.3.0"))
    assert rom.pack_languages() == [] and rom.status()["available"] is False
    out = rom.apply(ROMAN, "hi", "auto")
    assert out.text == ROMAN and "the installed uccharan 0.3.0 has no transliteration support" in (out.notice or "")
    with pytest.raises(ComponentMissing, match=r"uccharan 0\.3\.0 has no transliteration"):
        rom.apply(ROMAN, "hi", "on", strict=True)


def test_detection_reads_the_installed_module(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "uccharan", fake_uccharan())
    found = rom.detect()
    assert found.available and found.version == "9.9.9" and found.reason is None
    nested = types.ModuleType("uccharan")
    nested.translit = types.SimpleNamespace(transliterate=lambda text, lang: text)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uccharan", nested)
    assert rom.detect().available
    monkeypatch.setitem(sys.modules, "uccharan", types.ModuleType("uccharan"))
    assert not rom.detect().available and rom.detect().module is not None
    monkeypatch.setenv("BOLO_UCCHARAN", "off")
    assert rom.detect().module is None
    monkeypatch.setitem(sys.modules, "uccharan", None)  # importing it raises ImportError
    monkeypatch.delenv("BOLO_UCCHARAN")
    assert rom.detect() == rom.Api()


def test_pack_not_installed_or_not_offered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = use(monkeypatch, fake_uccharan(packs=("xlit-hi", "xlit-mr"), loaded=()))
    monkeypatch.delenv("UCCHARAN_DATA_DIR", raising=False)
    assert rom.pack_languages() == ["hi", "mr"] and rom.pack_state("hi") == "not_installed"
    out = rom.apply(ROMAN, "hi", "auto")
    assert out.text == ROMAN and out.record is not None and out.record["pack"] == "xlit-hi"
    assert "bolo components install uccharan-xlit-hi --accept-license" in (out.notice or "")
    assert "CC BY-SA 4.0" in (out.notice or "")
    with pytest.raises(ComponentMissing, match="uccharan-xlit-hi"):
        rom.apply(ROMAN, "hi", "on", strict=True)
    # a language the installed uccharan has no pack for: nothing to install, so a 400 when asked for by name
    out = rom.apply("naan nalai varuven", "ta", "on")
    assert "no romanised-text pack for 'ta' (packs: hi, mr)" in (out.notice or "")
    with pytest.raises(InvalidRequest, match="no romanised-text pack for 'ta'"):
        rom.apply("naan nalai varuven", "ta", "on", strict=True)
    assert mod.state["calls"] == []
    # the pack's file where uccharan looks for it: loaded on first use, without a download
    (tmp_path / "xlit-hi.ucd").write_bytes(b"pack")
    monkeypatch.setenv("UCCHARAN_DATA_DIR", str(tmp_path))
    assert rom.pack_file("hi") == tmp_path / "xlit-hi.ucd" and rom.pack_present("hi") and not rom.pack_present("mr")
    assert rom.apply(ROMAN, "hi", "auto").text == NATIVE
    assert rom.status()["installed"] == ["hi"]
    # installing through uccharan
    assert rom.install_pack("mr") is None and rom.pack_state("mr") == "ready"
    assert "no pack 'xlit-ta'" in (rom.install_pack("ta") or "")


def test_a_failing_transliterator_never_fails_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = fake_uccharan()

    def broken(text: str, lang: str, **kw: Any) -> Any:
        raise AttributeError("module 'uccharan._native' has no attribute 'transliterate'")

    mod.transliterate = broken
    use(monkeypatch, mod)
    for strict in (False, True):
        out = rom.apply(ROMAN, "hi", "on", strict=strict)
        assert out.text == ROMAN and "the transliterator failed" in (out.notice or "")
    mod.transliterate = lambda text, lang, **kw: None  # no text at all
    use(monkeypatch, mod)
    assert rom.apply(ROMAN, "hi", "on").text == ROMAN
    mod.has_translit = lambda lang: 1 / 0  # a pack check that raises: the pack table decides
    use(monkeypatch, mod)
    monkeypatch.delenv("UCCHARAN_DATA_DIR", raising=False)
    assert rom.pack_state("hi") == "not_installed"


# -- text preparation ----------------------------------------------------------------------------------------------
async def test_prepare_resolves_the_mode(core: Bolo, fake: Any) -> None:
    spec = ModelSpec.from_dict(hindi_voice())
    core.catalog.add(spec)
    assert text_input(core, spec).romanised == "off"  # the server's default
    assert prepare(core, spec, None, ROMAN, "hi").text == ROMAN
    assert prepare(core, spec, None, ROMAN, "hi").romanised is None
    # the request
    p = prepare(core, spec, None, ROMAN, "hi", romanised="auto")
    assert p.text == NATIVE and p.romanised is not None and p.romanised["source"] == "request"
    assert p.trace_extra()["romanised"]["applied"] is True
    assert prepare(core, spec, None, ROMAN, "hi", romanised=True).text == NATIVE
    # the server's setting, then the model's pin, then the request again
    core.settings.romanised = "auto"
    assert text_input(core, spec).to_dict()["romanised"] == "auto"
    p = prepare(core, spec, None, ROMAN, "hi")
    assert p.text == NATIVE and p.romanised is not None and p.romanised["source"] == "voice"
    core.store.set_model_settings(spec.id, {"romanised": "off"})
    assert text_input(core, spec).romanised == "off" and prepare(core, spec, None, ROMAN, "hi").text == ROMAN
    assert prepare(core, spec, None, ROMAN, "hi", romanised="on").text == NATIVE
    core.store.set_model_settings(spec.id, {"romanised": "nonsense"})  # a damaged file must not fail requests
    assert text_input(core, spec).romanised == "auto"
    # a catalog option
    option = ModelSpec.from_dict(hindi_voice("hi-voice-on", romanised="on"))
    core.catalog.add(option)
    core.settings.romanised = "off"
    assert text_input(core, option).romanised == "on"


async def test_prepare_keeps_pronunciations(core: Bolo, fake: Any) -> None:
    """Inline pronunciations and ``[word](/ipa/)`` overrides are matched as written."""
    spec = ModelSpec.from_dict(hindi_voice(text_markup="kokoro"))
    core.catalog.add(spec)
    p = prepare(core, spec, None, "kal [Kal](/kæl/) jana hai", "hi", romanised="on", normalize=False)
    if p.applied:  # with a uccharan that has word lists the override reaches the voice as markup
        assert p.text == "कल [Kal](/kæl/) जाना है"
    assert "Kal" in p.text and p.text.startswith("कल ") and p.text.endswith(" जाना है")


async def test_prepare_degrades(core: Bolo, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, None)
    spec = ModelSpec.from_dict(hindi_voice())
    core.catalog.add(spec)
    p = prepare(core, spec, None, ROMAN, "hi", romanised="auto")
    assert p.text == ROMAN and p.romanised is not None and "uccharan" in p.romanised["notice"]
    with pytest.raises(ComponentMissing):
        prepare(core, spec, None, ROMAN, "hi", romanised="on")
    # a session or an agent asks without insisting
    p = prepare(core, spec, None, ROMAN, "hi", romanised="on", romanised_strict=False)
    assert p.text == ROMAN and p.romanised is not None and p.romanised["applied"] is False
    # the server's default never refuses either
    core.settings.romanised = "on"
    assert prepare(core, spec, None, ROMAN, "hi").text == ROMAN


# -- HTTP, WebSocket, sessions ---------------------------------------------------------------------------------------
def test_speech_request(live: LiveServer, llama: FakeTTSServer, fake: Any) -> None:
    r = speak(live, llama, input=ROMAN)
    assert r.status_code == 200 and sent(llama) == ROMAN  # off by default
    assert "romanised" not in last_trace(live)
    r = speak(live, llama, input=ROMAN, romanised="auto")
    assert r.status_code == 200 and sent(llama) == NATIVE
    trace = last_trace(live)
    assert trace["romanised"]["applied"] is True and trace["romanised"]["text"] == NATIVE
    assert trace["normalized_text"] == NATIVE
    for value in (True, "on"):
        assert speak(live, llama, input=ROMAN, romanised=value).status_code == 200 and sent(llama) == NATIVE
    r = speak(live, llama, input=ROMAN, romanized="on")  # the other spelling
    assert r.status_code == 200 and sent(llama) == NATIVE
    r = speak(live, llama, input=ROMAN, romanised="sometimes")
    assert r.status_code == 400
    # an English voice: `auto` does nothing, `on` is a 400
    r = speak(live, llama, input="hello there", model="en-voice", romanised="auto")
    assert r.status_code == 200 and sent(llama) == "hello there"
    r = speak(live, llama, input="hello there", model="en-voice", romanised="on")
    assert r.status_code == 400 and r.json()["error"]["param"] == "romanised"


def test_speech_request_degrades(live: LiveServer, llama: FakeTTSServer, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, None)
    r = speak(live, llama, input=ROMAN, romanised="auto")
    assert r.status_code == 200 and sent(llama) == ROMAN
    notice = last_trace(live)["romanised"]
    assert notice["applied"] is False and "pip install -U uccharan" in notice["notice"]
    r = speak(live, llama, input=ROMAN, romanised="on")
    assert r.status_code == 424 and r.json()["error"]["code"] == "component_missing"
    assert "transliteration support" in r.json()["error"]["message"]
    # the pack is missing: the message names the component to install
    use(monkeypatch, fake_uccharan(loaded=()))
    monkeypatch.delenv("UCCHARAN_DATA_DIR", raising=False)
    r = speak(live, llama, input=ROMAN, romanised="on")
    assert r.status_code == 424 and "bolo components install uccharan-xlit-hi" in r.json()["error"]["message"]
    r = speak(live, llama, input=ROMAN, romanised="auto")
    assert r.status_code == 200 and "not installed" in last_trace(live)["romanised"]["notice"]


def test_model_pin_preview_and_info(live: LiveServer, llama: FakeTTSServer, fake: Any) -> None:
    info = httpx.get(f"{live.url}/api/v1/info").json()["capabilities"]["romanised"]
    assert info == {
        "available": True, "reason": None, "modes": ["off", "auto", "on"], "languages": ["hi"], "installed": ["hi"],
        "default": "off", "request_field": "romanised", "pin_per_model": True,
    }  # fmt: skip
    r = httpx.put(f"{live.url}/api/v1/models/hi-voice/settings", json={"romanised": "auto"})
    assert r.status_code == 200 and r.json()["settings"]["romanised"] == "auto"
    assert httpx.get(f"{live.url}/api/v1/text-input/hi-voice").json()["text_input"]["romanised"] == "auto"
    assert speak(live, llama, input=ROMAN).status_code == 200 and sent(llama) == NATIVE
    assert last_trace(live)["romanised"]["source"] == "voice"
    assert speak(live, llama, input=ROMAN, romanised="off").status_code == 200 and sent(llama) == ROMAN
    assert httpx.put(f"{live.url}/api/v1/models/hi-voice/settings", json={"romanised": "maybe"}).status_code == 400
    r = httpx.put(f"{live.url}/api/v1/models/hi-voice/settings", json={"romanised": None})
    assert "romanised" not in r.json()["settings"]
    # the preview shows the record and never refuses
    body = {"text": ROMAN, "models": ["hi-voice", "en-voice"], "romanised": "on"}
    r = httpx.post(f"{live.url}/api/v1/wordlists/-/preview", json=body)
    assert r.status_code == 200, r.text
    by = {p["model"]: p for p in r.json()["previews"]}
    assert by["hi-voice"]["text"] == NATIVE and by["hi-voice"]["romanised"]["applied"] is True
    assert by["en-voice"]["text"] == ROMAN and "Indian languages" in by["en-voice"]["romanised"]["notice"]
    plain = httpx.post(f"{live.url}/api/v1/wordlists/-/preview", json={"text": ROMAN, "models": ["hi-voice"]})
    assert plain.json()["previews"][0]["romanised"] is None


async def test_tts_websocket(live: LiveServer, llama: FakeTTSServer, fake: Any) -> None:
    async def say(query: str) -> str:
        llama.requests.clear()
        async with websockets.connect(f"{live.ws_url}/api/v1/tts/stream?model=hi-voice{query}") as ws:
            assert json.loads(await ws.recv())["type"] == "ready"
            await ws.send(json.dumps({"type": "text", "text": ROMAN + "."}))
            await ws.send(json.dumps({"type": "stop"}))
            others = [json.loads(m)["type"] async for m in ws if isinstance(m, str)]
        assert "chunk" in others and others[-1] == "done"
        return sent(llama)

    assert await say("") == ROMAN + "."
    assert await say("&romanised=auto") == NATIVE + "."


def test_realtime_session_and_agents(live: LiveServer, fake: Any) -> None:
    from bolo.realtime import RealtimeError, RealtimeSession

    async def sink(event: dict[str, Any]) -> None:
        return None

    session = RealtimeSession(live.core, sink)
    assert session.romanised is None
    session.configure({"type": "realtime", "audio": {"output": {"romanised": "auto"}}})
    assert session.romanised == "auto" and session.session_object()["audio"]["output"]["romanised"] == "auto"
    session.configure({"type": "realtime", "audio": {"output": {"romanized": True}}})
    assert session.romanised == "on"
    with pytest.raises(RealtimeError) as err:
        session.configure({"type": "realtime", "audio": {"output": {"romanised": "sometimes"}}})
    assert err.value.param == "session.audio.output.romanised"
    # telephony agents
    agent = AgentConfig.from_dict("desk", {"instructions": "x", "romanised": "auto"})
    assert agent.romanised == "auto" and agent.to_dict()["romanised"] == "auto"
    assert AgentConfig.from_dict("desk", {"romanised": True}).romanised == "on"
    assert AgentConfig.from_dict("desk", {}).romanised is None
    with pytest.raises(InvalidRequest):
        AgentConfig.from_dict("desk", {"romanised": "sometimes"})


async def test_spoken_answers_never_fail_on_a_missing_transliterator(
    core: Bolo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What sessions and agents do: ``on`` without the transliterator is a notice in the trace, the text is spoken."""
    from bolo.agents import speak_stream

    use(monkeypatch, None)

    async def deltas() -> Any:
        yield "mujhe kal jana hai."

    kinds = [kind async for kind, _value in speak_stream(core, deltas(), model="fake-tts", romanised="on")]
    assert "audio" in kinds
    trace = core.traces.recent(kind="tts")[0]
    assert trace.status == "ok" and trace.extra["romanised"]["applied"] is False and trace.extra["romanised"]["notice"]


def test_setting(tmp_path: Path) -> None:
    assert load_settings(env={}).romanised == "off"
    assert load_settings(env={"BOLO_ROMANISED": "AUTO"}).romanised == "auto"
    assert load_settings(env={"BOLO_ROMANISED": "true"}).romanised == "on"
    cfg = tmp_path / "bolo.toml"
    cfg.write_text('romanised = "on"\n')
    assert load_settings(cfg, env={}).romanised == "on"
    with pytest.raises(ValueError, match="romanised"):
        load_settings(env={"BOLO_ROMANISED": "sometimes"})


def test_pack_components(live: LiveServer, monkeypatch: pytest.MonkeyPatch) -> None:
    def components() -> dict[str, dict[str, Any]]:
        return {c["id"]: c for c in httpx.get(f"{live.url}/api/v1/components").json()["components"]}

    use(monkeypatch, None)
    assert not [c for c in components() if c.startswith("uccharan-xlit-")]
    mod = use(monkeypatch, fake_uccharan(packs=("xlit-hi", "xlit-mr"), loaded=("hi",)))
    monkeypatch.delenv("UCCHARAN_DATA_DIR", raising=False)
    listed = components()
    hi, mr = listed["uccharan-xlit-hi"], listed["uccharan-xlit-mr"]
    assert hi["installed"] is True and hi["kind"] == "language-pack" and hi["license"] == "CC BY-SA 4.0"
    assert mr["installed"] is False and mr["installable"] is True and "xlit-mr" in mr["manual_command"]
    url = f"{live.url}/api/v1/components/uccharan-xlit-mr/install"
    refused = httpx.post(url, json={})
    assert refused.status_code == 400 and "CC BY-SA 4.0" in refused.json()["error"]["message"]
    unknown = httpx.post(f"{live.url}/api/v1/components/uccharan-xlit-ta/install", json={"accept_license": True})
    assert unknown.status_code == 404
    r = httpx.post(url, json={"accept_license": True})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    for _ in range(100):
        job = httpx.get(f"{live.url}/api/v1/jobs/{job_id}").json()
        if job.get("state") in ("done", "error"):
            break
    assert job["state"] == "done", job
    assert "mr" in mod.state["loaded"] and components()["uccharan-xlit-mr"]["installed"] is True


# -- the installed uccharan ------------------------------------------------------------------------------------------
def test_with_the_installed_uccharan() -> None:
    """Against the real package, whatever it can do: with the API and a Hindi pack the sentence is converted; with
    anything less the text comes back unchanged with a notice. It never raises."""
    found = rom.refresh()
    out = rom.apply(ROMAN, "hi", "on")
    if found.available and rom.pack_state("hi") == "ready" and out.record and out.record["applied"]:
        assert "office" in out.text and "मुझे" in out.text and "है" in out.text
    else:
        assert out.text == ROMAN and out.notice
    status = rom.status()
    assert status["modes"] == ["off", "auto", "on"] and isinstance(status["languages"], list)
