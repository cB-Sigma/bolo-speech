"""Grapheme-to-phoneme conversion for voices that read phonemes.

Piper-format VITS voices, KittenTTS, Matcha-TTS (English) and Kokoro are
trained on phonemes (espeak-ng IPA; misaki's tokens for Kokoro English).
Since version 2.0 the sherpa-onnx runtime no longer contains a phonemiser and
expects phonemes from the caller. Two phonemisers can provide them:

* **uccharan** (Apache-2.0, ``pip install 'bolo-speech[uccharan]'``, part of
  ``bolo-speech[all]``): the default where it covers the voice's language and
  output profile, see :func:`uccharan_language` (since uccharan 0.6: English US
  and British, Hindi and the other Indian languages, Urdu, Sindhi, and Spanish,
  German, Dutch, French with Belgian and Swiss French, Italian, Portuguese
  (Brazil, Portugal), Polish, Turkish, Czech, Swedish, Catalan, Romanian,
  Hungarian, Finnish, Norwegian Bokmål and Danish; for Kokoro English it is a
  port of misaki, the front end Kokoro was trained with). Runs in process, no
  GPL code or data. Languages other than American English and Hindi need a
  language pack, which bolo downloads on first use (:mod:`bolo.text.packs`).
* **espeak-ng** (GPL-3.0-or-later), every other language.

The ``phonemizer`` setting chooses: ``auto`` (uccharan where it covers the
voice, otherwise espeak-ng), ``uccharan`` (never espeak-ng) or ``espeak-ng``
(the behaviour of bolo 0.1).

espeak-ng is licensed GPL-3.0-or-later. bolo (Apache-2.0) does not bundle
it, link it at build time or depend on it. If you want voices that need it,
you install it yourself, and bolo uses, in this order:

1. the ``espeak-ng`` executable on ``PATH`` (``apt install espeak-ng``,
   ``brew install espeak-ng``): a separate process, text in, phonemes out;
2. an espeak-ng shared library, if one is installed: the system
   ``libespeak-ng`` or the one shipped by the ``espeakng-loader`` wheel.
   This loads GPL code into the server process; whether that matters to you
   depends on how you distribute your deployment.

Set ``BOLO_ESPEAK=executable``, ``library`` or ``off`` to force a choice.
Voices that work from a lexicon or from characters (Kokoro English and
Mandarin, MeloTTS, MMS-TTS, Coqui VITS) never touch espeak-ng.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import shutil
import subprocess
import threading
import unicodedata
from functools import lru_cache
from typing import Any

from bolo.errors import ComponentMissing, EngineUnavailable

_CLAUSE = re.compile(r"([,;:،]+(?=\s|$)|\s[-–—]\s|\n+)")
_FINAL = re.compile(r"([.!?।॥؟۔…]+)[\"'”’)\]]*\s*$")
_PUNCT_MAP = {"।": ".", "॥": ".", "؟": "?", "،": ",", "۔": ".", "…": ".", "–": ",", "—": ",", "-": ","}
_LANG_FLAG = re.compile(r"\([a-z-]+\)")

INSTALL_HINT = (
    "this voice needs the optional component 'espeak-ng' (eSpeak NG, GPL-3.0-or-later), which is not shipped "
    "with bolo. Install it from the console (Models, Components), with `bolo components install espeak-ng`, "
    "with `pip install 'bolo-speech[espeak]'`, or with your system package manager (`apt install espeak-ng`); "
    "or choose a voice that does not need it, such as Kokoro English."
)


#: Output profile a voice family was trained on, by catalog ``kind``.
PROFILE_BY_KIND = {"vits": "piper", "kokoro": "kokoro", "kitten": "kitten", "matcha": "matcha"}

#: The language tags uccharan 0.6 phonemises (``uccharan.LANGUAGES``). An older uccharan that lacks one of them is
#: not used for it (:meth:`UccharanPhonemizer.supports`).
UCCHARAN_LANGUAGES: tuple[str, ...] = (
    "en-us", "en-gb", "hi", "bn", "mr", "te", "ne", "kn", "ta", "ml", "as", "gu", "bho", "mai", "pa", "or", "kok",
    "si", "sa", "ur", "sd", "mag", "es", "de", "nl", "fr", "it", "pt-br", "pt", "pl", "tr", "cs", "sv", "ca", "ro",
    "hu", "fi", "nb", "da", "fr-be", "fr-ch",
)  # fmt: skip
#: Tags of a voice (espeak-ng voice names as Piper and Kokoro write them, BCP-47 tags as the catalog writes them)
#: that are not uccharan's own spelling of the language.
UCCHARAN_ALIASES: dict[str, str] = {
    "en": "en-us", "en-uk": "en-gb", "en-gb-x-rp": "en-gb", "en-gb-scotland": "en-gb", "fr-fr": "fr",
    "es-es": "es", "pt-pt": "pt", "no": "nb", "nb-no": "nb", "no-no": "nb", "nl-nl": "nl",
    # Piper's Flemish voices were trained on espeak-ng's Dutch phonemes ("nl"), which is what uccharan's nl gives
    "nl-be": "nl",
}  # fmt: skip
#: Languages whose regions read differently: only the listed regions map to a uccharan language. Spanish outside
#: Spain (espeak-ng's es-419, Piper's es_MX and es_AR voices) and Canadian French are not uccharan's.
_REGIONAL: dict[str, dict[str, str]] = {
    "en": {"us": "en-us", "gb": "en-gb", "uk": "en-gb"},
    "es": {"es": "es"},
    "fr": {"fr": "fr", "be": "fr-be", "ch": "fr-ch", "lu": "fr"},
    "pt": {"br": "pt-br", "pt": "pt"},
    "nl": {"nl": "nl", "be": "nl"},
}
#: profile -> uccharan languages it has a front end for (Kokoro: the languages Kokoro speaks that uccharan covers)
_PROFILE_LANGUAGES: dict[str, tuple[str, ...]] = {
    "piper": UCCHARAN_LANGUAGES,
    "kokoro": ("en-us", "en-gb", "es", "fr", "it", "pt-br", "hi"),
    "kitten": ("en-us",),
    "matcha": ("en-us",),
}
#: What uccharan covers: profile -> {uccharan language -> uccharan language}. Everything else needs espeak-ng.
UCCHARAN_SUPPORT: dict[str, dict[str, str]] = {p: {t: t for t in langs} for p, langs in _PROFILE_LANGUAGES.items()}
UCCHARAN = "uccharan"
#: the uccharan releases this bolo is written for (the ``uccharan`` extra)
UCCHARAN_REQUIREMENT = "uccharan>=0.6.2,<0.7"
ESPEAK = "espeak-ng"
PREFERENCES = ("auto", UCCHARAN, ESPEAK)


def normalise_language(tag: str) -> str:
    """``en_US`` / ``en-US`` -> ``en-us``; ``hi-IN`` -> ``hi`` (Indian region tags carry no phonemiser difference)."""
    t = tag.strip().lower().replace("_", "-")
    base = t.split("-")[0]
    return base if base in ("hi",) else t


def uccharan_tag(language: str | None) -> str | None:
    """uccharan's tag for a voice's or request's language (``es``, ``pt-BR``, ``fr-fr``, ``nb``, ``de-DE``,
    ``ur-PK``), or None when uccharan has no such language (``cmn``, ``ru``, ``es-419``)."""
    t = (language or "").strip().lower().replace("_", "-")
    if not t:
        return None
    if t in UCCHARAN_ALIASES:
        return UCCHARAN_ALIASES[t]
    if t in UCCHARAN_LANGUAGES:
        return t
    base, _, region = t.partition("-")
    if base in _REGIONAL:
        return _REGIONAL[base].get(region) if region else uccharan_tag(base)
    return base if base in UCCHARAN_LANGUAGES else None


def uccharan_language(profile: str, language: str) -> str | None:
    """The uccharan language for a voice, or None when uccharan does not cover it."""
    tag = uccharan_tag(language)
    return tag if tag is not None and tag in UCCHARAN_SUPPORT.get(profile, {}) else None


class Phonemizer:
    name = "none"
    #: the component that provides it ("uccharan" or "espeak-ng")
    component = ESPEAK
    #: "system-executable" | "python-package" | "configured"
    source = "configured"

    def supports(self, profile: str, language: str) -> bool:
        return True

    def voice_codepoints(self, text: str, language: str, profile: str, improved: bool = False) -> list[list[int]]:
        """Codepoints per sentence for ``GenerationConfig.phoneme_codepoints``."""
        return self.codepoints(strip_overrides(text), language)

    def version(self) -> str | None:
        return None

    def clause(self, text: str, voice: str) -> str:
        """IPA phonemes for one clause of text (no punctuation handling)."""
        raise NotImplementedError

    def phonemize(self, text: str, voice: str) -> list[str]:
        """Phonemise ``text`` into sentences of IPA symbols, keeping punctuation.

        The result follows Piper's conventions: NFD-normalised symbols with
        clause punctuation re-attached, one string per sentence.
        """
        from bolo.text.segment import split_sentences

        text = strip_overrides(text)
        sentences: list[str] = []
        for sentence in split_sentences(text):
            m = _FINAL.search(sentence)
            final = _PUNCT_MAP.get(m.group(1)[:1], m.group(1)[:1]) if m else ""
            body = sentence[: m.start()] if m else sentence
            clauses: list[str] = []
            parts = _CLAUSE.split(body)
            for i in range(0, len(parts), 2):
                chunk = parts[i].strip()
                if not chunk:
                    continue
                phones = _LANG_FLAG.sub("", self.clause(chunk, voice)).strip()
                if not phones:
                    continue
                sep = parts[i + 1].strip()[:1] if i + 1 < len(parts) else ""
                clauses.append(phones + _PUNCT_MAP.get(sep, sep))
            if clauses:
                joined = " ".join(clauses)
                if final and not joined.endswith(final):
                    joined = joined.rstrip(",;:") + final
                sentences.append(unicodedata.normalize("NFD", joined))
        return sentences

    def codepoints(self, text: str, voice: str) -> list[list[int]]:
        return [[ord(c) for c in s] for s in self.phonemize(text, voice)]


class EspeakExecutable(Phonemizer):
    """Runs the ``espeak-ng`` program for each clause batch."""

    name = "espeak-ng (executable)"
    source = "system-executable"

    def __init__(self, exe: str) -> None:
        self.exe = exe

    def version(self) -> str | None:
        try:
            out = subprocess.run([self.exe, "--version"], capture_output=True, check=False, timeout=10).stdout
            m = re.search(r"(\d+\.\d+(?:\.\d+)?)", out.decode(errors="replace"))
            return m.group(1) if m else None
        except OSError:
            return None

    @lru_cache(maxsize=4096)  # noqa: B019 - one instance per process
    def clause(self, text: str, voice: str) -> str:
        proc = subprocess.run(
            [self.exe, "-q", "--ipa", "-v", voice, "--", text],
            capture_output=True,
            check=False,
            timeout=30,
        )
        if proc.returncode != 0:
            raise EngineUnavailable(
                f"espeak-ng failed for voice '{voice}': {proc.stderr.decode(errors='replace')[:200]}"
            )
        return " ".join(proc.stdout.decode("utf-8", errors="replace").split())


class EspeakLibrary(Phonemizer):
    """Calls ``espeak_TextToPhonemes`` in a shared libespeak-ng through ctypes."""

    name = "espeak-ng (shared library)"
    _IPA = 0x02
    _UTF8 = 1

    def __init__(self, library: str, data_path: str | None, source: str = "configured") -> None:
        self.source = source
        self.library = library
        self._lib = ctypes.CDLL(library)
        self._lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self._lib.espeak_Initialize.restype = ctypes.c_int
        self._lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        self._lib.espeak_SetVoiceByName.restype = ctypes.c_int
        self._lib.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int]
        self._lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        # AUDIO_OUTPUT_RETRIEVAL = 1, espeakINITIALIZE_DONT_EXIT = 0x8000
        rate = self._lib.espeak_Initialize(1, 0, data_path.encode() if data_path else None, 0x8000)
        if rate <= 0:
            raise OSError("espeak_Initialize failed (espeak-ng-data not found?)")
        self._lock = threading.Lock()
        self._voice: str | None = None
        self._cache: dict[tuple[str, str], str] = {}

    def version(self) -> str | None:
        try:
            self._lib.espeak_Info.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
            self._lib.espeak_Info.restype = ctypes.c_char_p
            raw = self._lib.espeak_Info(None)
            return raw.decode(errors="replace").split()[0] if raw else None
        except Exception:
            return None

    def clause(self, text: str, voice: str) -> str:
        key = (voice, text)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        with self._lock:  # libespeak-ng keeps global state
            if voice != self._voice:
                if self._lib.espeak_SetVoiceByName(voice.encode()) != 0:
                    raise EngineUnavailable(f"espeak-ng has no voice '{voice}'")
                self._voice = voice
            buf = ctypes.c_char_p(text.encode("utf-8"))
            pieces: list[str] = []
            while buf.value:
                out = self._lib.espeak_TextToPhonemes(ctypes.byref(buf), self._UTF8, self._IPA)
                if out:
                    pieces.append(out.decode("utf-8", errors="replace"))
            result = " ".join(" ".join(pieces).split())
        if len(self._cache) < 8192:
            self._cache[key] = result
        return result


class UccharanPhonemizer(Phonemizer):
    """uccharan (Apache-2.0), in process. A language's pack is fetched before its first use (bolo.text.packs)."""

    name = "uccharan"
    component = UCCHARAN
    source = "python-package"

    def __init__(self, module: Any) -> None:
        self.module = module
        known = getattr(module, "LANGUAGES", None)
        #: the languages the installed uccharan has (None: it does not say, so every one bolo knows is tried)
        self.languages: frozenset[str] | None = frozenset(str(x) for x in known) if known else None

    def version(self) -> str | None:
        return str(getattr(self.module, "__version__", "") or "") or None

    def has_language(self, tag: str) -> bool:
        return self.languages is None or tag in self.languages

    def supports(self, profile: str, language: str) -> bool:
        tag = uccharan_language(profile, language)
        return tag is not None and self.has_language(tag)

    def _lang(self, profile: str, language: str, text: str) -> str:
        """uccharan's tag for the voice, with the packs it needs loaded (downloaded on first use)."""
        from bolo.text import packs

        tag = uccharan_language(profile, language) or uccharan_tag(language)
        if tag is None or not self.has_language(tag):
            raise EngineUnavailable(f"uccharan has no {profile!r} front end for language '{language}'")
        packs.ensure(tag, profile, text)
        return tag

    def _call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Call uccharan; a pack it could not have (``PackError``) becomes a 424 that names the pack."""
        from bolo.text import packs

        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if type(exc).__name__ != "PackError":
                raise
            raise packs.PackMissing(f"uccharan is missing a language pack: {exc}") from exc

    def clause(self, text: str, voice: str) -> str:
        plain = strip_overrides(text)
        lang = self._lang("piper", voice, plain)
        return str(self._call(self.module.phonemize, plain, lang, "espeak-ipa", punctuation=False))

    def phonemize(self, text: str, voice: str) -> list[str]:
        plain = strip_overrides(text)
        lang = self._lang("piper", voice, plain)
        return list(self._call(self.module.phonemize, plain, lang, "piper"))

    def voice_codepoints(self, text: str, language: str, profile: str, improved: bool = False) -> list[list[int]]:
        if uccharan_language(profile, language) is None:
            raise EngineUnavailable(f"uccharan has no {profile!r} front end for language '{language}'")
        lang = self._lang(profile, language, strip_overrides(text))
        return list(self._call(self._codepoints, text, lang, profile, improved))

    def _codepoints(self, text: str, lang: str, profile: str, improved: bool) -> list[list[int]]:
        overrides = OVERRIDE.findall(text)
        if overrides:
            if not hasattr(self.module, "phonemize_wordlists"):
                text = strip_overrides(text)
            else:
                # Pronunciation overrides ([word](/ipa/), from word lists, SSML or the request): the
                # entry's IPA replaces uccharan's reading, in the voice's phone set.
                seen: dict[str, str] = {}
                for word, ipa in overrides:
                    seen.setdefault(word, ipa)
                wl = self.module.WordList.from_entries(
                    [{"word": w, "ipa": ipa, "match": "case"} for w, ipa in seen.items()]
                )
                plain = strip_overrides(text)
                if profile == "piper":
                    cps = self.module.phonemize_wordlists(
                        plain, lang, profile="sherpa-onnx", improved=improved, wordlists=[wl]
                    )
                    return [list(s) for s in cps]
                result: list[list[int]] = []
                for sentence in _sentences(plain):
                    phones = str(
                        self.module.phonemize_wordlists(
                            sentence, lang, profile=profile, improved=improved, wordlists=[wl]
                        )
                    ).strip()
                    if profile == "kitten":
                        phones = " ".join(_KITTEN_TOKEN.findall(phones))
                    if phones:
                        result.append([ord(c) for c in phones])
                return result
        if profile == "piper":
            return [list(s) for s in self.module.phonemize(text, lang, "sherpa-onnx", improved=improved)]
        out: list[list[int]] = []
        for sentence in _sentences(text):
            phones = str(self.module.phonemize(sentence, lang, profile, improved=improved)).strip()
            if profile == "kitten":
                # Kitten's own front end re-tokenises the phonemes (\w+|[^\w\s]) and joins them with spaces.
                phones = " ".join(_KITTEN_TOKEN.findall(phones))
            if phones:
                out.append([ord(c) for c in phones])
        return out


_KITTEN_TOKEN = re.compile(r"\w+|[^\w\s]")
#: Kokoro-style pronunciation override ``[word](/phonemes/)``
OVERRIDE = re.compile(r"\[([^\[\]]+)\]\(/([^/()]+)/\)")


def strip_overrides(text: str) -> str:
    """``[word](/phonemes/)`` -> ``word`` (for phonemisers that cannot take the phonemes)."""
    return OVERRIDE.sub(r"\1", text)


def _sentences(text: str) -> list[str]:
    from bolo.text.segment import split_sentences

    return [s for s in split_sentences(text) if s.strip()] or [text]


def _library_candidates() -> list[tuple[str, str | None, str]]:
    out: list[tuple[str, str | None, str]] = []
    try:
        import espeakng_loader  # type: ignore[import-not-found,import-untyped,unused-ignore]

        out.append((espeakng_loader.get_library_path(), espeakng_loader.get_data_path(), "python-package"))
    except Exception:
        pass
    found = ctypes.util.find_library("espeak-ng")
    if found:
        out.append((found, None, "system-executable"))
    return out


_lock = threading.Lock()
_instance: Phonemizer | None = None
_resolved = False


def get_phonemizer() -> Phonemizer | None:
    """The process-wide phonemiser, or None when espeak-ng is not installed."""
    global _instance, _resolved
    with _lock:
        if _resolved:
            return _instance
        # Order: an explicit setting, then a system executable, then the loader package.
        mode = os.environ.get("BOLO_ESPEAK", "auto").lower()
        explicit = os.environ.get("BOLO_ESPEAK_LIBRARY")
        if mode != "off":
            exe = shutil.which("espeak-ng") or shutil.which("espeak")
            if explicit and mode in ("auto", "library"):
                try:
                    _instance = EspeakLibrary(explicit, os.environ.get("BOLO_ESPEAK_DATA"), "configured")
                except Exception:
                    _instance = None
            if _instance is None and exe and mode in ("auto", "executable"):
                _instance = EspeakExecutable(exe)
            if _instance is None and mode in ("auto", "library"):
                for lib, data, source in _library_candidates():
                    try:
                        _instance = EspeakLibrary(lib, data, source)
                        break
                    except Exception:
                        continue
        _resolved = True
        return _instance


def require_phonemizer() -> Phonemizer:
    ph = get_phonemizer()
    if ph is None:
        raise ComponentMissing(INSTALL_HINT)
    return ph


_uccharan: Phonemizer | None = None
_uccharan_resolved = False
_preference = "auto"


def get_uccharan() -> Phonemizer | None:
    """uccharan when its Python package is installed (``BOLO_UCCHARAN=off`` hides it)."""
    global _uccharan, _uccharan_resolved
    with _lock:
        if not _uccharan_resolved:
            _uccharan_resolved = True
            if os.environ.get("BOLO_UCCHARAN", "auto").lower() != "off":
                try:
                    import uccharan  # type: ignore[import-not-found,unused-ignore]

                    _uccharan = UccharanPhonemizer(uccharan)
                except Exception:
                    _uccharan = None
        return _uccharan


def set_preference(preference: str) -> None:
    """``auto`` | ``uccharan`` | ``espeak-ng`` (the ``phonemizer`` setting)."""
    global _preference
    if preference not in PREFERENCES:
        raise ValueError(f"phonemizer must be one of {', '.join(PREFERENCES)}, not {preference!r}")
    _preference = preference


def candidates(profile: str, language: str) -> list[str]:
    """Components that may phonemise for this voice under the current preference, best first."""
    covered = uccharan_language(profile, language) is not None
    if _preference == ESPEAK:
        return [ESPEAK]
    if _preference == UCCHARAN:
        return [UCCHARAN] if covered else []
    return [UCCHARAN, ESPEAK] if covered else [ESPEAK]


def select(profile: str, language: str) -> Phonemizer | None:
    """The phonemiser to use for a voice, or None when no suitable one is installed."""
    for component in candidates(profile, language):
        ph = get_uccharan() if component == UCCHARAN else get_phonemizer()
        if ph is not None and ph.supports(profile, language):
            return ph
    return None


UCCHARAN_MODES = ("compatible", "improved")
CHOICES = ("auto", UCCHARAN, f"{UCCHARAN}:compatible", f"{UCCHARAN}:improved", ESPEAK)


def parse_choice(value: str | None) -> tuple[str, bool]:
    """A ``phonemizer`` request value -> ``(component or "auto", improved)``.

    Values: ``auto``, ``uccharan`` (= ``uccharan:compatible``, what the voices were trained on),
    ``uccharan:improved`` (uccharan's cleaner readings: homographs, numbers, Hindi schwa), ``espeak-ng``.
    """
    from bolo.errors import InvalidRequest

    v = (value or "auto").strip().lower()
    if v in ("", "auto"):
        return "auto", False
    component, _, mode = v.partition(":")
    if component in ("espeak", "espeak-ng") and not mode:
        return ESPEAK, False
    if component == UCCHARAN and mode in ("", *UCCHARAN_MODES):
        return UCCHARAN, mode == "improved"
    raise InvalidRequest(f"phonemizer must be one of {', '.join(CHOICES)}, not {value!r}", param="phonemizer")


def resolve(profile: str, language: str, choice: str | None = None) -> tuple[Phonemizer, bool]:
    """The phonemiser (and uccharan's improved flag) for one request.

    ``choice`` (a request's or a voice's pinned ``phonemizer``) overrides the server-wide preference.
    Raises ``ComponentMissing`` (424) when the chosen phonemiser is not installed, and ``InvalidRequest``
    (400) when uccharan is chosen for a language it does not cover.
    """
    from bolo.errors import InvalidRequest

    component, improved = parse_choice(choice)
    if component == "auto":
        return require(profile, language), False
    if component == UCCHARAN:
        if uccharan_language(profile, language) is None:
            covered = ", ".join(sorted(UCCHARAN_SUPPORT.get(profile, {}))) or "none"
            raise InvalidRequest(
                f"uccharan has no {profile} front end for language '{language}'. Supported for {profile}: "
                f"{covered}. Use phonemizer 'espeak-ng' or 'auto' for this voice.",
                param="phonemizer",
            )
        ph = get_uccharan()
        if ph is None or not ph.supports(profile, language):
            raise ComponentMissing(missing_hint(profile, language))
        return ph, improved
    ph = get_phonemizer()
    if ph is None:
        raise ComponentMissing(INSTALL_HINT)
    return ph, False


def missing_hint(profile: str, language: str) -> str:
    wanted = candidates(profile, language)
    if not wanted:
        return (
            f"no phonemiser for this voice: the phonemizer setting is 'uccharan', which does not cover "
            f"language '{language}' ({profile}); set phonemizer = 'auto' to use eSpeak NG for it"
        )
    if wanted[0] == UCCHARAN:
        ph = get_uccharan()
        if ph is not None and not ph.supports(profile, language):
            return (
                f"this voice needs a phonemiser: the installed uccharan {ph.version() or ''} does not have language "
                f"'{language}'; upgrade it with `pip install -U '{UCCHARAN_REQUIREMENT}'`"
                + ("" if len(wanted) == 1 else "; eSpeak NG (`bolo components install espeak-ng`) works too")
            )
        return (
            "this voice needs a phonemiser: install uccharan (Apache-2.0) with `pip install 'bolo-speech[uccharan]'` or "
            "`bolo components install uccharan`"
            + ("" if len(wanted) == 1 else "; eSpeak NG (`bolo components install espeak-ng`) works too")
        )
    return INSTALL_HINT


def require(profile: str, language: str) -> Phonemizer:
    ph = select(profile, language)
    if ph is None:
        raise ComponentMissing(missing_hint(profile, language))
    return ph


def reset_phonemizer() -> None:
    """Forget the cached choices (tests, and after installing a component)."""
    global _instance, _resolved, _uccharan, _uccharan_resolved
    with _lock:
        _instance, _resolved = None, False
        _uccharan, _uccharan_resolved = None, False
