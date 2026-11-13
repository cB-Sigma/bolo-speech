"""Grapheme-to-phoneme conversion through an *external* espeak-ng.

Piper-format VITS voices, KittenTTS, Matcha-TTS (English) and the
non-English Kokoro voices are trained on IPA phonemes produced by espeak-ng.
Since version 2.0 the sherpa-onnx runtime no longer contains espeak-ng and
expects phonemes from the caller.

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

from bolo.errors import ComponentMissing, EngineUnavailable

_CLAUSE = re.compile(r"([,;:،]+(?=\s|$)|\s[-–—]\s|\n+)")
_FINAL = re.compile(r"([.!?।॥؟۔…]+)[\"'”’)\]]*\s*$")
_PUNCT_MAP = {"।": ".", "॥": ".", "؟": "?", "،": ",", "۔": ".", "…": ".", "–": ",", "—": ",", "-": ","}
_LANG_FLAG = re.compile(r"\([a-z-]+\)")

INSTALL_HINT = (
    "this voice needs the optional component 'espeak-ng' (eSpeak NG, GPL-3.0-or-later), which is not shipped "
    "with bolo. Install it from the console (Models, Components), with `bolo components install espeak-ng`, "
    "with `pip install 'bolo[espeak]'`, or with your system package manager (`apt install espeak-ng`); "
    "or choose a voice that does not need it, such as Kokoro English."
)


class Phonemizer:
    name = "none"
    #: "system-executable" | "python-package" | "configured"
    source = "configured"

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


def reset_phonemizer() -> None:
    """Forget the cached choice (tests)."""
    global _instance, _resolved
    with _lock:
        _instance, _resolved = None, False
