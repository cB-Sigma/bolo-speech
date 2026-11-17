"""uccharan 0.6: every language it phonemises goes through it, and its language packs are downloaded by bolo.

Most tests put a small uccharan-shaped module in place of the real one and serve its packs from a local HTTP
server, so they pass with any uccharan (or none) and never touch the network. The last ones use the installed
uccharan when it is 0.6 or later, and one real test (BOLO_TEST_REAL=1) downloads de.ucd from uccharan's release.
"""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import os
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from bolo.catalog import ModelSpec
from bolo.cli import main
from bolo.config import Settings, load_settings
from bolo.core import Bolo
from bolo.errors import ComponentMissing
from bolo.text import packs, phonemize, prepare
from conftest import LiveServer

#: the fake packs: name -> content (served by PackServer, listed with their SHA-256 in the fake's PACKS)
CONTENT = {
    name: f"uccharan pack {name}\n".encode() * 10
    for name in ("de", "fr", "da", "pt-br", "ur", "nb", "en-gb", "kokoro-en-us", "wikipron-hi", "xlit-hi")
}
#: what the fake reports as missing per language (as uccharan.missing_packs does)
NEEDS = {"de": ["de"], "fr": ["fr"], "fr-be": ["fr"], "fr-ch": ["fr"], "da": ["da"], "pt-br": ["pt-br"],
         "ur": ["ur"], "nb": ["nb"], "en-gb": ["en-gb"]}  # fmt: skip


class PackError(RuntimeError):
    """Named like uccharan's: bolo recognises it by name."""


def fake_uccharan(cache: Path, *, needs: dict[str, list[str]] | None = None) -> Any:
    """A module shaped like uccharan 0.6: LANGUAGES, phonemize, load_pack, packs.PACKS / installed / cache_dir and
    _native.missing_packs. A language whose pack is not loaded raises PackError, as uccharan does."""
    mod = types.ModuleType("uccharan")
    state: dict[str, Any] = {"loaded": set(), "calls": []}
    table = {name: (hashlib.sha256(blob).hexdigest(), len(blob)) for name, blob in CONTENT.items()}
    wanted = NEEDS if needs is None else needs

    def installed(name: str) -> bool:
        f = cache / f"{name}.ucd"
        return f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest() == table[name][0]

    def load_pack(name: str) -> str:
        if not installed(name):
            raise PackError(f"pack {name!r} is not available offline")
        state["loaded"].add(name)
        return name

    def missing_packs(lang: str, profile: str) -> list[str]:
        names = list(wanted.get(lang, []))
        if profile == "kokoro" and lang == "en-us":
            names.append("kokoro-en-us")
        return [n for n in names if n not in state["loaded"]]

    def phonemize_(text: str, lang: str = "en-us", profile: str = "espeak-ipa", **_: Any) -> Any:
        missing = missing_packs(lang, "kokoro" if profile == "kokoro" else "espeak-ipa")
        if missing:
            raise PackError(f"pack {missing[0]!r} is not available offline")
        state["calls"].append((lang, profile))
        phones = f"{lang}ˈ{text.strip('.')}"
        if profile == "sherpa-onnx":
            return [[ord(c) for c in phones]]
        return [phones] if profile == "piper" else phones

    mod.__version__ = "0.6.0"  # type: ignore[attr-defined]
    mod.LANGUAGES = phonemize.UCCHARAN_LANGUAGES  # type: ignore[attr-defined]
    mod.PackError = PackError  # type: ignore[attr-defined]
    mod.phonemize = phonemize_  # type: ignore[attr-defined]
    mod.load_pack = load_pack  # type: ignore[attr-defined]
    mod.packs = types.SimpleNamespace(PACKS=table, installed=installed, cache_dir=lambda: cache)  # type: ignore[attr-defined]
    mod._native = types.SimpleNamespace(missing_packs=missing_packs)  # type: ignore[attr-defined]
    mod.state = state  # type: ignore[attr-defined]
    return mod


class PackServer:
    """uccharan's release, locally: GET /v<version>/<name>.ucd. ``tamper`` serves other bytes; ``down`` 404s."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.tamper = False
        self.down = False
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.requests.append(self.path)
                name = self.path.rsplit("/", 1)[-1].removesuffix(".ucd")
                if outer.down or name not in CONTENT:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = CONTENT[name] + (b"x" if outer.tamper else b"")
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def release() -> Iterator[PackServer]:
    server = PackServer()
    yield server
    server.stop()


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release: PackServer) -> Any:
    """The fake uccharan as the installed one, its packs at the local release, nothing downloaded yet."""
    mod = fake_uccharan(tmp_path / "uccharan-cache")
    ph = phonemize.UccharanPhonemizer(mod)
    monkeypatch.setattr(packs, "module", lambda: mod)
    monkeypatch.setattr(packs, "URL", release.url + "/v{version}/{name}.ucd")
    monkeypatch.setattr(packs, "_loaded", set())
    monkeypatch.setattr(packs, "_present", set())
    monkeypatch.setattr(packs, "_user_offline", False)
    monkeypatch.setattr(packs, "_auto_download", True)
    monkeypatch.setattr(packs, "_reporter", None)
    monkeypatch.delenv("UCCHARAN_OFFLINE", raising=False)  # restored after the test (bolo sets it)
    monkeypatch.delenv("UCCHARAN_DATA_DIR", raising=False)
    monkeypatch.setattr(phonemize, "get_uccharan", lambda: ph)
    monkeypatch.setattr(phonemize, "get_phonemizer", lambda: None)
    monkeypatch.setattr(prepare, "_reads_numbers", {})
    return mod


def add_voice(core: Bolo, model_id: str, language: str) -> None:
    """A Piper-like voice that reads phonemes (the fake engine speaks; bolo phonemises and fetches packs)."""
    core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": model_id, "task": "tts", "engine": "fake", "kind": "fake-tts", "family": "Piper",
                "languages": [language[:2]], "license": "MIT", "builtin": True, "sample_rate": 24000,
                "requires": ["phonemizer"], "options": {"language": language},
            }
        )
    )  # fmt: skip


# -- routing ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("voice", "tag"),
    [
        # espeak-ng voice codes, as Piper voices and Kokoro carry them
        ("es", "es"), ("pt-br", "pt-br"), ("fr-fr", "fr"), ("fr-be", "fr-be"), ("fr-ch", "fr-ch"), ("nb", "nb"),
        ("no", "nb"), ("da", "da"), ("de", "de"), ("ur", "ur"), ("sd", "sd"), ("en-us", "en-us"),
        ("en-gb-x-rp", "en-gb"), ("hi", "hi"), ("te", "te"), ("ne", "ne"), ("ca", "ca"), ("pt", "pt"),
        # BCP-47 tags, as the catalog writes them
        ("es-ES", "es"), ("pt-BR", "pt-br"), ("pt-PT", "pt"), ("fr-BE", "fr-be"), ("da-DK", "da"),
        ("de_DE", "de"), ("ur-PK", "ur"), ("nl-BE", "nl"), ("no-NO", "nb"), ("hi-IN", "hi"), ("en", "en-us"),
        # not uccharan's: espeak-ng (when installed) phonemises these
        ("cmn", None), ("ru-RU", None), ("es-419", None), ("es-MX", None), ("fr-CA", None), ("ja", None),
        ("", None),
    ],
)  # fmt: skip
def test_voice_language_to_uccharan_tag(voice: str, tag: str | None) -> None:
    assert phonemize.uccharan_tag(voice) == tag
    assert phonemize.uccharan_language("piper", voice) == tag


def test_profiles_cover_what_the_voices_speak() -> None:
    kokoro = {
        p: phonemize.uccharan_language("kokoro", p) for p in ("en-us", "en-gb", "es", "fr-fr", "it", "pt-br", "hi")
    }
    assert kokoro == {"en-us": "en-us", "en-gb": "en-gb", "es": "es", "fr-fr": "fr", "it": "it", "pt-br": "pt-br",
                      "hi": "hi"}  # fmt: skip
    assert phonemize.uccharan_language("kokoro", "ja") is None and phonemize.uccharan_language("kokoro", "de") is None
    assert phonemize.uccharan_language("kitten", "es") is None and phonemize.uccharan_language("matcha", "en-us")
    assert len(phonemize.UCCHARAN_SUPPORT["piper"]) == 41  # 39 languages, with en-us/en-gb and fr/fr-be/fr-ch


@pytest.mark.parametrize("language", ["es", "de", "fr-be", "da", "pt-br", "ur"])
def test_languages_route_through_uccharan(fake: Any, language: str) -> None:
    ph = phonemize.select("piper", language)
    assert ph is not None and ph.component == "uccharan"
    resolved, improved = phonemize.resolve("piper", language, "uccharan")
    assert resolved is ph and improved is False
    codepoints = ph.voice_codepoints("Hola.", language, "piper")
    assert "".join(map(chr, codepoints[0])) == f"{language}ˈHola"  # uccharan's own tag, not the voice's
    assert fake.state["calls"][-1] == (language, "sherpa-onnx")


def test_espeak_voice_codes_reach_uccharan_as_its_tags(fake: Any) -> None:
    ph = phonemize.select("piper", "fr-fr")
    assert ph is not None and ph.component == "uccharan"
    assert ph.phonemize("Bonjour.", "fr-fr") == ["frˈBonjour"]
    assert "".join(map(chr, ph.voice_codepoints("Hei.", "no", "piper")[0])) == "nbˈHei"
    kokoro = phonemize.select("kokoro", "fr-fr")
    assert kokoro is ph and "".join(map(chr, ph.voice_codepoints("Salut.", "fr-fr", "kokoro")[0])) == "frˈSalut"


def test_languages_uccharan_lacks_go_to_espeak_or_fail_clearly(fake: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    for language in ("cmn", "ru", "es-419"):
        assert phonemize.select("piper", language) is None
        with pytest.raises(ComponentMissing, match="espeak-ng"):
            phonemize.require("piper", language)
    espeak = phonemize.EspeakExecutable("/usr/bin/espeak-ng")
    monkeypatch.setattr(phonemize, "get_phonemizer", lambda: espeak)
    assert phonemize.select("piper", "cmn") is espeak and phonemize.select("piper", "es") is not espeak


def test_an_older_uccharan_is_not_used_for_languages_it_lacks(fake: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    fake.LANGUAGES = ("en-us", "en-gb", "hi")
    old = phonemize.UccharanPhonemizer(fake)
    monkeypatch.setattr(phonemize, "get_uccharan", lambda: old)
    assert phonemize.select("piper", "en-us") is old and phonemize.select("piper", "de") is None
    with pytest.raises(ComponentMissing, match=r"uccharan>=0\.6\.4,<0\.7"):
        phonemize.resolve("piper", "de", "uccharan")


#: the probe numbers in English
ENGLISH = {"25": "twenty-five", "52": "fifty-two", "13": "thirteen", "31": "thirty-one"}


def test_text_normalisation_language(fake: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from bolo.text import wordlists

    readings = {
        "es": {"25": "veinticinco", "52": "cincuenta y dos", "13": "trece", "31": "treinta y uno"},
        "de": ENGLISH,  # uccharan 0.6.0 read German digits in English
        "ur": {"25": "پچیس", "52": "باون", "13": "تیرہ", "31": "اکتیس"},
        "en-gb": ENGLISH,
    }
    fake.WordList = object
    fake.normalize = lambda text, lang, **kw: types.SimpleNamespace(text=readings[lang][text])
    monkeypatch.setattr(wordlists, "uccharan_module", lambda: fake)
    tags = {t: prepare.uccharan_language(t) for t in ("pt-BR", "fr-BE", "nb-NO", "ur-PK", "es-MX", "ru", None)}
    assert tags == {"pt-BR": "pt-br", "fr-BE": "fr-be", "nb-NO": "nb", "ur-PK": "ur", "es-MX": None, "ru": None,
                    None: "en-us"}  # fmt: skip
    # uccharan's normalise-only mode is used where it reads numbers in the language itself
    assert prepare.normalizes_numbers("es") is True
    assert prepare.normalizes_numbers("de") is False  # read in English: bolo's light normaliser instead
    assert prepare.normalizes_numbers("ur") is True and prepare.normalizes_numbers("en-gb") is True


#: what an uccharan writes for the probe numbers, by language: words, and the ways a normaliser can go wrong
WRITTEN = {
    "si": {"25": "විසිපහ", "52": "පනස්දෙක", "13": "දහතුන", "31": "තිස්එක"},
    "sd": {"25": "پنجويهه", "52": "ٻاونجاهه", "13": "تيرهن", "31": "ايڪٽيهه"},
    "hi": {"25": "पच्चीस", "52": "बावन", "13": "तेरह", "31": "इकतीस"},
    "fi": {"25": "kaksikymmentäviisi", "52": "viisikymmentäkaksi", "13": "kolmetoista", "31": "kolmekymmentäyksi"},
}
GARBLED = {
    # uccharan 0.6.1: lookup keys of its word table, letters of the right script that spell the digits
    "si": {"25": "ණගඡ", "52": "ණඡග", "13": "ණඛඝ", "31": "ණඝඛ"},
    "sd": {"25": "ڙتج", "52": "ڙجت", "13": "ڙپٹ", "31": "ڙٹپ"},
    # digits left in, letters of another block, English, digits read one by one
    "hi": {"25": "25", "52": "बावन", "13": "तेरह", "31": "इकतीस"},
    "bn": {"25": "पच्चीस", "52": "বাহান্ন", "13": "তেরো", "31": "একত্রিশ"},
    "ta": {"25": "twenty-five", "52": "fifty-two", "13": "thirteen", "31": "thirty-one"},
    "te": {"25": "రెండు ఐదు", "52": "ఐదు రెండు", "13": "ఒకటి మూడు", "31": "మూడు ఒకటి"},
    "ur": {"25": "پچیس", "52": "باون", "13": "تیرہ ١٣", "31": "اکتیس"},
    "de": {"25": "fünfundzwanzig", "52": "zweiundfünfzig", "13": "drei\u0441ehn", "31": "einunddreißig"},
}


def _normaliser(monkeypatch: pytest.MonkeyPatch, written: dict[str, dict[str, str]]) -> list[tuple[str, str]]:
    """A uccharan whose normalise-only mode writes ``written[lang][text]`` (the text itself when not listed) and
    marks it ``[uccharan]``; returns its calls."""
    from bolo.text import wordlists

    calls: list[tuple[str, str]] = []

    def normalize(text: str, lang: str, **_: Any) -> Any:
        calls.append((text, lang))
        return types.SimpleNamespace(text=written.get(lang, {}).get(text, f"[uccharan] {text}"), spans=[])

    mod = types.ModuleType("uccharan")
    mod.WordList = object  # type: ignore[attr-defined]
    mod.LANGUAGES = phonemize.UCCHARAN_LANGUAGES  # type: ignore[attr-defined]
    mod.normalize = normalize  # type: ignore[attr-defined]
    monkeypatch.setattr(wordlists, "uccharan_module", lambda: mod)
    monkeypatch.setattr(prepare, "_reads_numbers", {})
    return calls


@pytest.mark.parametrize("tag", sorted(WRITTEN))
def test_number_words_are_trusted(monkeypatch: pytest.MonkeyPatch, tag: str) -> None:
    calls = _normaliser(monkeypatch, WRITTEN)
    assert prepare.normalizes_numbers(tag) is True
    assert prepare.normalizes_numbers(tag) is True
    assert len(calls) == 4  # probed once, then cached


@pytest.mark.parametrize("tag", sorted(GARBLED))
def test_garbled_number_words_are_not(monkeypatch: pytest.MonkeyPatch, tag: str) -> None:
    calls = _normaliser(monkeypatch, GARBLED)
    assert prepare.normalizes_numbers(tag) is False
    assert prepare.normalizes_numbers(tag) is False
    assert len(calls) == 4


def test_number_words_ok() -> None:
    ok = prepare.number_words_ok
    assert ok("si", "විසිපහ") and ok("sd", "پنجويهه") and ok("fr", "vingt-cinq") and ok("ro", "douăzeci și cinci")
    assert ok("ml", "ഇരുപത്തിയഞ്ച്") and ok("hi", "पच्चीस") and ok("en-us", "twenty-five")
    assert not ok("si", "25") and not ok("hi", "२५") and not ok("ur", "۲۵") and not ok("si", "")
    assert not ok("bn", "पच्चीस") and not ok("ur", "पच्चीस") and not ok("sd", "twenty five") and not ok("de", "ಐದು")


@pytest.mark.parametrize("language", ["si", "sd"])
def test_garbled_digits_fall_back_to_the_light_normaliser(monkeypatch: pytest.MonkeyPatch, language: str) -> None:
    """A text voice in a language whose uccharan writes digits as lookup keys gets bolo's light normaliser."""
    _normaliser(monkeypatch, GARBLED)
    cap = prepare.TextInput(reads="text", normalizer="uccharan")
    out = prepare._normalise(None, cap, "Room 25.", language, [], True)  # type: ignore[arg-type]
    assert out.normalizer == "light" and out.text == "Room 25."
    # with an uccharan that writes words, its normaliser is used
    _normaliser(monkeypatch, WRITTEN)
    out = prepare._normalise(None, cap, "Room 25.", language, [], True)  # type: ignore[arg-type]
    assert out.normalizer == "uccharan" and out.text == "[uccharan] Room 25."


# -- packs ----------------------------------------------------------------------------------------------------------


def test_missing_pack_is_downloaded_on_first_use(fake: Any, release: PackServer) -> None:
    ph = phonemize.select("piper", "da-DK")
    assert ph is not None
    assert packs.needed("da", "piper") == ["da"] and not packs.installed("da")
    cps = ph.voice_codepoints("Hej.", "da-DK", "piper")
    assert "".join(map(chr, cps[0])) == "daˈHej"
    assert release.requests == ["/v0.6.0/da.ucd"]  # the installed uccharan's release
    cached = fake.packs.cache_dir() / "da.ucd"
    assert cached.read_bytes() == CONTENT["da"] and "da" in fake.state["loaded"] and packs.installed("da")
    ph.voice_codepoints("Hej igen.", "da", "piper")
    assert release.requests == ["/v0.6.0/da.ucd"]  # once
    # Belgian French reads with the French table; Urdu text with Latin words also needs the British English data
    assert packs.ensure("fr-be", "piper") == ["fr"]
    assert packs.ensure("ur", "piper", "میرا Wi-Fi") == ["ur", "en-gb"]
    assert packs.ensure("de", "piper", "Hallo Welt") == ["de"]  # Latin script: no en-gb for German


async def test_first_use_is_a_pull_job_with_events_and_a_log(fake: Any, core: Bolo) -> None:
    add_voice(core, "da-voice", "da-DK")
    sub = core.events.subscribe()
    first = asyncio.ensure_future(sub.__anext__())  # the subscription starts with its first read
    await asyncio.sleep(0)
    audio, _rate = await core.synthesize_all("Godmorgen.", model="da-voice")
    assert len(audio) > 0 and "da" in fake.state["loaded"]
    jobs = [j for j in core.jobs.all() if j.model == "uccharan-da"]
    assert len(jobs) == 1 and jobs[0].kind == "pull" and jobs[0].state == "done"
    assert jobs[0].bytes_done == jobs[0].bytes_total == len(CONTENT["da"])
    events: list[dict[str, Any]] = [await asyncio.wait_for(first, 5)]
    while not any(e.get("event") == "installed" for e in events):
        events.append(await asyncio.wait_for(sub.__anext__(), 5))
    assert [e["job"]["state"] for e in events if e["type"] == "job"] == ["running", "done"]
    assert {"type": "component", "id": "uccharan-da", "event": "installed"} in events
    logs = [e["message"] for e in events if e["type"] == "log"]
    assert any("downloading uccharan language pack 'da'" in m for m in logs)
    await core.synthesize_all("Igen.", model="da-voice")
    assert len([j for j in core.jobs.all() if j.model == "uccharan-da"]) == 1  # downloaded once


def test_offline_errors_name_the_pack_and_the_command(
    fake: Any, release: PackServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    ph = phonemize.select("piper", "ur")
    assert ph is not None
    monkeypatch.setattr(packs, "_auto_download", False)
    with pytest.raises(ComponentMissing) as err:
        ph.voice_codepoints("سلام", "ur", "piper")
    message = err.value.message
    assert err.value.status == 424 and "'ur'" in message and "uccharan_auto_download = false" in message
    assert "bolo pull uccharan-ur" in message and "python -m uccharan.packs ur" in message
    assert "UCCHARAN_DATA_DIR" in message and not release.requests
    # UCCHARAN_OFFLINE=1 in bolo's environment is respected too
    monkeypatch.setattr(packs, "_auto_download", True)
    monkeypatch.setattr(packs, "_user_offline", True)
    with pytest.raises(ComponentMissing, match="UCCHARAN_OFFLINE=1"):
        packs.ensure("ur", "piper")
    # automatic, but the release cannot be reached
    monkeypatch.setattr(packs, "_user_offline", False)
    release.down = True
    with pytest.raises(ComponentMissing) as err:
        ph.voice_codepoints("سلام", "ur", "piper")
    assert "could not download uccharan's language pack 'ur'" in err.value.message
    assert "python -m uccharan.packs --dir DIR ur" in err.value.message
    # a pack that does not match uccharan's SHA-256 is never written or used
    release.down, release.tamper = False, True
    with pytest.raises(ComponentMissing, match="does not match the SHA-256"):
        packs.ensure("ur", "piper")
    assert not (fake.packs.cache_dir() / "ur.ucd").exists() and "ur" not in fake.state["loaded"]
    # normalisation goes on without the pack instead (uccharan reads the language by its rules)
    assert packs.ensure("ur", "espeak-ipa", required=False) == []


def test_a_pack_error_from_uccharan_becomes_a_424(fake: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ph = phonemize.select("piper", "nb")
    assert ph is not None
    monkeypatch.setattr(packs, "ensure", lambda *a, **k: [])  # as if bolo had not fetched it
    with pytest.raises(ComponentMissing, match="not available offline"):
        ph.voice_codepoints("Hei.", "nb", "piper")


def test_share_alike_packs_are_never_downloaded_automatically(
    fake: Any, release: PackServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake._native.missing_packs = lambda lang, profile: ["wikipron-hi", "xlit-hi"] if lang == "hi" else []
    with pytest.raises(ComponentMissing, match=r"CC BY-SA 4\.0") as err:
        packs.ensure("hi", "piper")
    assert "bolo pull uccharan-wikipron-hi --accept-license" in err.value.message
    assert packs.ensure("hi", "piper", required=False) == []
    assert not release.requests and packs.auto_download() is True
    assert packs.is_share_alike("xlit-hi") and packs.is_share_alike("wikipron-en-us") and not packs.is_share_alike("de")


def test_packs_are_components(
    live: LiveServer,
    fake: Any,
    release: PackServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listed = {c["id"]: c for c in httpx.get(live.url + "/api/v1/components").json()["components"]}
    ids = {packs.component_id(n) for n in CONTENT if not n.startswith("xlit-")}
    assert ids <= set(listed)
    de = listed["uccharan-de"]
    assert de["kind"] == "language-pack" and de["license"] == "Apache-2.0" and de["size_bytes"] == len(CONTENT["de"])
    assert de["installed"] is False and de["installable"] is True and de["auto_download"] is True
    assert de["manual_command"] == "python -m uccharan.packs de" and de["required_by_count"] >= 3
    wiki = listed["uccharan-wikipron-hi"]
    assert wiki["license"] == "CC BY-SA 4.0" and wiki["auto_download"] is False
    # explicit installs: Apache-2.0 packs as they are, share-alike ones with the licence accepted
    refused = httpx.post(live.url + "/api/v1/components/uccharan-wikipron-hi/install", json={})
    assert refused.status_code == 400 and "CC BY-SA 4.0" in refused.json()["error"]["message"]
    r = httpx.post(live.url + "/api/v1/models/pull", json={"id": "uccharan-nb"})
    assert r.status_code == 202
    for _ in range(100):
        job = httpx.get(f"{live.url}/api/v1/jobs/{r.json()['job_id']}").json()
        if job["state"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "done", job
    assert "nb" in fake.state["loaded"] and release.requests == ["/v0.6.0/nb.ucd"]
    # the command line: bolo pull uccharan-<pack>, components install, doctor
    monkeypatch.setenv("BOLO_DATA_DIR", str(live.core.settings.data_dir))
    assert main(["pull", "uccharan-de"]) == 0
    assert "installed uccharan-de (SHA-256 checked)" in capsys.readouterr().out
    assert main(["pull", "uccharan-de"]) == 0 and "already installed" in capsys.readouterr().out
    assert main(["pull", "uccharan-wikipron-hi"]) == 1 and "--accept-license" in capsys.readouterr().err
    assert main(["pull", "uccharan-wikipron-hi", "--accept-license"]) == 0
    assert "wikipron-hi" in fake.state["loaded"]
    assert main(["components", "install", "uccharan-pt-br"]) == 0
    assert (fake.packs.cache_dir() / "pt-br.ucd").is_file()
    capsys.readouterr()
    assert main(["components", "list"]) == 0
    out = capsys.readouterr().out
    assert "uccharan-da" in out and "uccharan-wikipron-hi" in out and "CC BY-SA 4.0" in out
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert f"language packs ({len(CONTENT)};" in out and "uccharan-ur" in out and "(on request only)" in out


def test_pack_counts_built_in_packs_and_removal(live: LiveServer, fake: Any) -> None:
    """Who needs each pack (in the catalog and among installed models), built-in packs as such, and removal."""
    import json

    fake.packs.PACKS["hi"] = ("0" * 64, 2)  # uccharan lists its built-in data as packs too
    core = live.core
    add_voice(core, "fake-de-one", "de")
    add_voice(core, "fake-de-two", "de")
    for model_id, language in (("fake-hi", "hi"), ("fake-bho", "bho"), ("fake-mai", "mai")):
        add_voice(core, model_id, language)
    folder = core.store.model_dir("fake-de-one")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps({"id": "fake-de-one", "complete": True, "files": []}))

    def listing() -> dict[str, Any]:
        return {c["id"]: c for c in httpx.get(live.url + "/api/v1/components").json()["components"]}

    listed = listing()
    de = listed["uccharan-de"]
    assert de["required_by_count"] >= 2 and de["required_by_installed_count"] == 1 and de["built_in"] is False
    assert de["removable"] is False  # not installed
    hi = listed["uccharan-hi"]
    # Hindi's data is part of uccharan: never needed, but Hindi, Bhojpuri and Maithili voices read with it
    assert hi["built_in"] is True and hi["required_by_count"] == 0 and hi["required_by_installed_count"] == 0
    assert hi["used_by_count"] >= 3 and hi["removable"] is False

    # a downloaded pack this server has not loaded can be removed; a loaded one only after a restart
    target = fake.packs.cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    (target / "da.ucd").write_bytes(CONTENT["da"])
    assert listing()["uccharan-da"]["removable"] is True
    assert httpx.delete(live.url + "/api/v1/components/uccharan-da").status_code == 204
    assert not (target / "da.ucd").exists() and listing()["uccharan-da"]["installed"] is False
    again = httpx.delete(live.url + "/api/v1/components/uccharan-da")
    assert again.status_code == 400 and "not installed" in again.json()["error"]["message"]
    packs.install("nb")
    assert listing()["uccharan-nb"]["removable"] is False
    loaded = httpx.delete(live.url + "/api/v1/components/uccharan-nb")
    assert loaded.status_code == 400 and "restart" in loaded.json()["error"]["message"]
    assert (target / "nb.ucd").is_file()
    builtin = httpx.delete(live.url + "/api/v1/components/uccharan-hi")
    assert builtin.status_code == 400
    other = httpx.delete(live.url + "/api/v1/components/espeak-ng")
    assert other.status_code == 400 and "only uccharan's language packs" in other.json()["error"]["message"]
    assert httpx.delete(live.url + "/api/v1/components/nothing").status_code == 404


def test_auto_download_setting() -> None:
    assert Settings().uccharan_auto_download is True
    assert load_settings(env={"BOLO_UCCHARAN_AUTO_DOWNLOAD": "false"}).uccharan_auto_download is False


def test_configure_turns_uccharan_own_downloads_off(fake: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(packs, "_user_offline", None)
    packs.configure(auto_download=True)
    assert os.environ.get("UCCHARAN_OFFLINE") == "1" and packs.auto_download() is True
    packs.configure(auto_download=False)
    assert packs.auto_download() is False


# -- the installed uccharan -----------------------------------------------------------------------------------------


def _real_uccharan() -> Any:
    try:
        import uccharan  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        return None
    version = tuple(int(x) for x in str(getattr(uccharan, "__version__", "0")).split(".")[:2] if x.isdigit())
    return uccharan if version >= (0, 6) else None


needs_uccharan_06 = pytest.mark.skipif(_real_uccharan() is None, reason="needs uccharan 0.6 or later")


@needs_uccharan_06
def test_installed_uccharan_spanish_needs_no_pack() -> None:
    real = _real_uccharan()
    ph = phonemize.UccharanPhonemizer(real)
    assert ph.supports("piper", "es") and ph.supports("kokoro", "pt-br") and not ph.supports("piper", "cmn")
    assert set(phonemize.UCCHARAN_LANGUAGES) <= set(real.LANGUAGES)
    assert set(real.packs.PACKS) >= {"de", "fr", "ur", "xlit-hi", "wikipron-hi"} and len(real.packs.PACKS) == 35
    assert packs.needed("es", "piper") == []
    text = "".join(map(chr, ph.voice_codepoints("El niño come.", "es-ES", "piper")[0]))
    assert "ɲ" in text


def _uccharan_061(minimum: tuple[int, ...] = (0, 6, 1)) -> Any:
    real = _real_uccharan()
    version = tuple(int(x) for x in str(getattr(real, "__version__", "0")).split(".")[:3] if x.isdigit())
    return real if real is not None and version >= minimum else None


@pytest.mark.skipif(_uccharan_061() is None, reason="needs uccharan 0.6.1 or later")
def test_installed_uccharan_reads_numbers_in_the_latin_languages(monkeypatch: pytest.MonkeyPatch) -> None:
    """uccharan 0.6.1 reads digits in German, Spanish, French … itself, so text voices get its normaliser."""
    monkeypatch.setattr(prepare, "_reads_numbers", {})
    for tag in ("de", "es", "fr", "it", "nl", "pt-br", "fr-ch"):
        assert prepare.normalizes_numbers(tag), tag
    real = _uccharan_061()
    assert "fünfundzwanzig Euro" in real.normalize("25 €", "de", digits="request").text
    assert "tres coma cinco" in real.normalize("3,5 kg", "es", digits="request").text


@pytest.mark.skipif(_uccharan_061((0, 6, 2)) is None, reason="needs uccharan 0.6.2 or later")
def test_installed_uccharan_writes_numbers_as_words_in_every_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """uccharan 0.6.2 writes numbers as words in all its languages (0.6.1: lookup keys in Sinhala and Sindhi)."""
    monkeypatch.setattr(prepare, "_reads_numbers", {})
    for tag in phonemize.UCCHARAN_LANGUAGES:
        assert prepare.normalizes_numbers(tag), tag
    real = _uccharan_061((0, 6, 2))
    assert real.normalize("25", "si", digits="request").text == "විසිපහ"
    assert real.normalize("25", "sd", digits="request").text == "پنجويهه"


@needs_uccharan_06
@pytest.mark.real
def test_real_pack_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """German's word table from uccharan's GitHub release: downloaded, checked, loaded, used."""
    real = _real_uccharan()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(packs, "_loaded", set())
    monkeypatch.setattr(packs, "_present", set())
    assert not packs.installed("de")
    packs.install("de")
    assert (tmp_path / "uccharan" / real.__version__ / "de.ucd").is_file() and packs.installed("de")
    assert real.packs.installed("de")  # uccharan's own SHA-256 check
    ph = phonemize.UccharanPhonemizer(real)
    text = "".join(map(chr, ph.voice_codepoints("Guten Morgen.", "de-DE", "piper")[0]))
    assert "ɡuːtən" in text or "ɡˈuːtən" in text
