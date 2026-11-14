"""Text preparation: uccharan's normalise-only mode, word lists, SSML and pronunciation overrides."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from bolo.catalog import ModelSpec
from bolo.config import Settings
from bolo.core import Bolo
from bolo.errors import InvalidRequest
from bolo.text import wordlists as wl
from bolo.text.normalize import normalize_for_tts, speak_url, strip_markup
from bolo.text.prepare import prepare, text_input
from bolo.text.ssml import parse_ssml, say_as
from conftest import LiveServer
from fake_tts_server import FakeTTSServer

uccharan = pytest.importorskip("uccharan")
if not hasattr(uccharan, "normalize"):  # pragma: no cover - an older uccharan
    pytest.skip("uccharan without normalise-only mode", allow_module_level=True)

TSV = (
    "# uccharan wordlist 1\n"
    "word\tlang\tipa\tsay\tmatch\tnote\n"
    "Kokoro\ten\tkəkˈɔːɹoʊ\t\t\tproduct\n"
    "SBI\thi\t\tएस बी आई\tcase\t\n"
    "2G\ten\t\ttwo G\texact\t\n"
)


def server_entry(model_id: str, **options: object) -> dict[str, object]:
    return {
        "id": model_id,
        "task": "tts",
        "engine": "tts-server",
        "kind": "tts-server",
        "family": "Test",
        "languages": ["en", "hi"],
        "sample_rate": 24000,
        "license": "MIT",
        "requires": ["llama-server"],
        "source": {"local": "/nonexistent"},
        "voices": [{"id": "default", "name": "Model default", "server_voice": ""}],
        "options": {"runtime": "llama-server", **options},
    }


@pytest.fixture
def llama() -> Iterator[FakeTTSServer]:
    server = FakeTTSServer("llama", rate=16000, voices=[]).start()
    yield server
    server.stop()


@pytest.fixture
def live(settings: Settings, llama: FakeTTSServer) -> Iterator[LiveServer]:
    settings.tts_servers = [f"text-voice={llama.url}", f"markup-voice={llama.url}"]
    srv = LiveServer(settings)
    srv.core.catalog.add(ModelSpec.from_dict(server_entry("text-voice")))
    srv.core.catalog.add(ModelSpec.from_dict(server_entry("markup-voice", text_markup="kokoro")))
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def sent(llama: FakeTTSServer) -> str:
    return " ".join(str(r["body"].get("input", "")) for r in llama.requests if r["path"] == "/tts")


def speak(srv: LiveServer, llama: FakeTTSServer, **body: Any) -> str:
    llama.requests.clear()
    r = httpx.post(
        f"{srv.url}/v1/audio/speech", json={"model": "text-voice", "response_format": "wav", **body}, timeout=30
    )
    assert r.status_code == 200, r.text
    return sent(llama)


# -- the two fixed bugs -------------------------------------------------------------------------------------------
def test_urls_are_read_not_deleted() -> None:
    assert normalize_for_tts("see https://www.example.com/docs, now") == "see w w w dot example dot com slash docs, now"
    assert speak_url("https://github.com/x_y", "hi") == "github डॉट com स्लैश x अंडरस्कोर y"
    assert normalize_for_tts("www.bbc.co.uk.") == "w w w dot b b c dot co dot uk."


def test_kokoro_overrides_survive_markup_stripping() -> None:
    assert (
        strip_markup("[Kokoro](/kəkˈɔːɹoʊ/) and [docs](https://x.y) **bold**") == "[Kokoro](/kəkˈɔːɹoʊ/) and docs bold"
    )


def test_phonemizer_applies_overrides() -> None:
    from bolo.text.phonemize import UccharanPhonemizer, strip_overrides

    ph = UccharanPhonemizer(uccharan)
    cps = ph.voice_codepoints("[Kokoro](/kəkˈɔːɹoʊ/) is here.", "en-us", "piper")
    assert "".join(map(chr, cps[0])).startswith("kəkˈɔːɹoʊ")
    k = ph.voice_codepoints("[Kokoro](/kəkˈɔːɹoʊ/).", "en-us", "kitten")
    assert "".join(map(chr, k[0])).startswith("kəkˈɔːɹoʊ")
    assert strip_overrides("say [Kokoro](/x/) now") == "say Kokoro now"


# -- SSML ---------------------------------------------------------------------------------------------------------
def test_ssml_subset() -> None:
    s = parse_ssml(
        '<speak>Call <say-as interpret-as="telephone">98765</say-as> or <sub alias="World Wide Web">WWW</sub>.'
        '<break time="1s"/> Ask <phoneme alphabet="ipa" ph="kəkˈɔːɹoʊ">Kokoro</phoneme>'
        ' on <say-as interpret-as="date" format="dmy">15/08/2025</say-as>,'
        ' <say-as interpret-as="characters">abc</say-as>, the <say-as interpret-as="ordinal">3</say-as>.</speak>'
    )
    assert s.text == "Call 9 8 7 6 5 or World Wide Web., Ask Kokoro on 2025-08-15, A B C, the 3rd."
    assert s.entries == [{"word": "Kokoro", "ipa": "kəkˈɔːɹoʊ", "match": "case"}]
    assert parse_ssml("plain <sub alias='x'>y</sub> text").text == "plain x text"
    assert say_as("5", "ordinal", None, "hi") == "5वां"
    with pytest.raises(InvalidRequest, match="line 1, column"):
        parse_ssml("<speak>broken <sub alias='x'>y</speak>")
    with pytest.raises(InvalidRequest, match="not supported"):
        parse_ssml("<speak><video/></speak>")
    with pytest.raises(InvalidRequest, match="DOCTYPE"):
        parse_ssml('<!DOCTYPE x [<!ENTITY a "b">]><speak>&a;</speak>')


# -- capability record and preparation ---------------------------------------------------------------------------
async def test_capability_records(core: Bolo) -> None:
    piper = core.catalog.resolve("piper-hi-in-rohan-medium")
    assert piper is not None
    assert text_input(core, piper).reads == "phonemes"
    assert text_input(core, piper).pronunciation == "phonemes"
    core.catalog.add(ModelSpec.from_dict(server_entry("tv")))
    tv = core.catalog.resolve("tv")
    assert tv is not None
    cap = text_input(core, tv)
    assert (cap.reads, cap.normalizer, cap.pronunciation) == ("text", "uccharan", "respelling")
    core.store.set_model_settings("tv", {"text_normalizer": "light", "wordlists": ["acme"]})
    cap = text_input(core, tv)
    assert cap.normalizer == "light" and cap.wordlists == ("acme",)


async def test_prepare_paths(core: Bolo, monkeypatch: pytest.MonkeyPatch) -> None:
    core.catalog.add(ModelSpec.from_dict(server_entry("tv")))
    tv = core.catalog.resolve("tv")
    piper = core.catalog.resolve("piper-hi-in-rohan-medium")
    assert tv is not None and piper is not None
    # text voices: uccharan's normalise-only mode
    p = prepare(core, tv, None, "Pay ₹500 at 10:30, see www.example.com.", "en")
    assert p.normalizer == "uccharan"
    assert p.text == "Pay five hundred rupees at ten thirty, see w w w dot example dot com."
    assert prepare(core, tv, None, "₹5 करोड़ 10:30 बजे", "hi").text == "पांच करोड़ रुपये साढ़े दस बजे"
    # the request can choose another normaliser
    assert prepare(core, tv, None, "I have 3 cats", "en", normalize="none").text == "I have 3 cats"
    assert prepare(core, tv, None, "I have 3 cats", "en", normalize="light").normalizer == "light"
    with pytest.raises(InvalidRequest):
        prepare(core, tv, None, "x", "en", normalize="bogus")
    # word lists: spoken form for text voices, overrides for phoneme voices
    store = wl.store_for(core.settings.data_dir)
    store.put("default", TSV)
    p = prepare(core, tv, None, "SBI में 2G", "hi")
    assert p.text == "एस बी आई में two G"
    assert {a["text"] for a in p.applied} == {"SBI", "2G"}
    p = prepare(core, piper, None, "Kokoro is 2G", "hi")
    assert p.text.startswith("[Kokoro](/kəkˈɔːɹoʊ/)") and "two G" in p.text
    # [] turns stored lists off; inline entries and Kokoro markup still apply, first
    p = prepare(core, tv, None, "SBI and [Kokoro](/kəkˈɔːɹoʊ/)", "en", wordlists=[])
    assert p.text.startswith("S B I and ") and "Kokoro" not in p.text  # acronyms spelled, Kokoro respelled
    p = prepare(core, tv, None, "SBI", "hi", pronunciations=[{"word": "SBI", "say": "स्टेट बैंक", "match": "case"}])
    assert p.text == "स्टेट बैंक"
    with pytest.raises(Exception, match="no word list named 'nope'"):
        prepare(core, tv, None, "x", "en", wordlists=["nope"])
    # SSML input
    p = prepare(
        core, tv, None, '<speak>Room <say-as interpret-as="digits">42</say-as></speak>', "en", input_format="ssml"
    )
    assert p.text == "Room four two"
    # without uccharan: bolo's own normaliser, URLs still read
    monkeypatch.setattr(wl, "uccharan_module", lambda: None)
    p = prepare(core, tv, None, "I have 3 cats at www.a.org", "en")
    assert p.normalizer == "light" and p.text == "I have three cats at w w w dot a dot org"


# -- HTTP API ------------------------------------------------------------------------------------------------------
def test_wordlist_api(live: LiveServer) -> None:
    base = f"{live.url}/api/v1/wordlists"
    assert httpx.get(base).json() == {"wordlists": [], "available": True}
    r = httpx.put(f"{base}/acme", content=TSV.encode())
    assert r.status_code == 200, r.text
    assert r.json()["entries"] == 3 and set(r.json()["languages"]) == {"en-us", "hi"}
    listed = httpx.get(base).json()["wordlists"]
    assert [x["name"] for x in listed] == ["acme"]
    got = httpx.get(f"{base}/acme").json()
    assert got["entries"][0]["word"] == "Kokoro"
    assert httpx.get(f"{base}/acme", headers={"Accept": "text/tab-separated-values"}).text.startswith("# uccharan")
    pls = httpx.get(f"{base}/acme", params={"format": "pls", "lang": "en-us"}).text
    assert "<grapheme>Kokoro</grapheme>" in pls
    # PLS and JSON bodies are accepted too
    assert httpx.put(f"{base}/from-pls", content=pls.encode()).json()["entries"] == 2
    js = httpx.put(f"{base}/from-json", json={"entries": [{"word": "UPI", "say": "यू पी आई", "lang": "hi"}]})
    assert js.json()["entries"] == 1
    bad = httpx.put(f"{base}/bad", content=b"word\tlang\nx\tnot-a-lang\n")
    assert bad.status_code == 400 and "line 2" in bad.json()["error"]["message"]
    assert httpx.put(f"{base}/Bad%20Name", content=TSV.encode()).status_code == 400
    assert httpx.delete(f"{base}/acme").status_code == 200
    assert httpx.get(f"{base}/acme").status_code == 404
    # pins on the model are validated and stored
    r = httpx.put(
        f"{live.url}/api/v1/models/text-voice/settings", json={"wordlists": ["from-json"], "text_normalizer": "light"}
    )
    assert r.status_code == 200 and r.json()["settings"]["wordlists"] == ["from-json"]
    info = httpx.get(f"{live.url}/api/v1/text-input/text-voice").json()["text_input"]
    assert info["normalizer"] == "light" and info["wordlists"] == ["from-json"]


def test_preview_endpoint(live: LiveServer) -> None:
    httpx.put(f"{live.url}/api/v1/wordlists/acme", content=TSV.encode())
    r = httpx.post(
        f"{live.url}/api/v1/wordlists/acme/preview",
        json={
            "text": "Kokoro costs ₹500",
            "models": ["text-voice", "markup-voice", "piper-hi-in-rohan-medium"],
            "language": "en",
        },
    )
    assert r.status_code == 200, r.text
    by = {p["model"]: p for p in r.json()["previews"]}
    assert by["text-voice"]["text_input"]["pronunciation"] == "respelling"
    assert by["text-voice"]["text"].endswith("costs five hundred rupees") and "Kokoro" not in by["text-voice"]["text"]
    assert by["markup-voice"]["text"].startswith("[Kokoro](/kəkˈɔːɹoʊ/)")
    assert by["piper-hi-in-rohan-medium"]["text_input"]["reads"] == "phonemes"
    assert by["piper-hi-in-rohan-medium"]["text"].startswith("[Kokoro](/kəkˈɔːɹoʊ/)")


def test_speech_requests_use_uccharan_for_text_voices(live: LiveServer, llama: FakeTTSServer) -> None:
    assert speak(live, llama, input="Pay ₹500 at 10:30.", language="en") == "Pay five hundred rupees at ten thirty."
    assert speak(live, llama, input="Pay ₹500.", language="en", normalize=False) == "Pay ₹500."
    httpx.put(f"{live.url}/api/v1/wordlists/acme", content=TSV.encode())
    assert speak(live, llama, input="SBI में खाता", language="hi", wordlists=["acme"]) == "एस बी आई में खाता"
    assert speak(live, llama, input="SBI", language="hi", pronunciations=[{"word": "SBI", "say": "एसबीआई"}]) == "एसबीआई"
    out = speak(
        live, llama, input='<speak><sub alias="doctor">Dr.</sub> Rao, <say-as interpret-as="digits">12</say-as></speak>',
        input_format="ssml", language="en",
    )  # fmt: skip
    assert out == "doctor Rao, one two"
    r = httpx.post(
        f"{live.url}/v1/audio/speech",
        json={"model": "text-voice", "input": "<speak><x/></speak>", "input_format": "ssml"},
    )
    assert r.status_code == 400 and "not supported" in r.json()["error"]["message"]
    traces = httpx.get(f"{live.url}/api/v1/requests", params={"kind": "tts"}).json()["requests"]
    done = [t for t in traces if t.get("normalizer")]
    assert done and done[0]["normalizer"] == "uccharan" and "normalized_text" in done[0]
    assert any(t.get("pronunciations_applied") for t in done)


def test_unknown_wordlist_is_404(live: LiveServer) -> None:
    r = httpx.post(f"{live.url}/v1/audio/speech", json={"model": "text-voice", "input": "hi", "wordlists": ["missing"]})
    assert r.status_code == 404


def test_store_limits(tmp_path: Path) -> None:
    s = wl.WordListStore(tmp_path)
    with pytest.raises(InvalidRequest):
        s.put("x", "a\t\t\tb\n" * (wl.MAX_ENTRIES + 1))
    with pytest.raises(InvalidRequest):
        wl.check_name("../etc")
    assert s.names() == []
