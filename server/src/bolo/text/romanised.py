"""Romanised text: Indian-language text typed in Latin letters, written in its own script before a voice reads it.

``"mujhe kal office jana hai"`` for a Hindi voice becomes ``"मुझे कल office जाना है"``: the words of the language
are written in the native script, English words stay as they are (the voice's normaliser or phonemiser then
deals with them as it does with any Latin word). The conversion is uccharan's transliterator; bolo only decides
when to ask for it.

Three modes:

* ``off``: text is passed on as written.
* ``auto``: converted when the voice speaks an Indian language and the text is mostly Latin script.
* ``on``: converted whenever the voice speaks an Indian language and the text has Latin-script words.

The mode comes from the request, else the model's pinned setting, else the ``romanised`` setting.

uccharan's transliteration API and its language packs (``xlit-<language>``, CC BY-SA 4.0, never fetched
automatically) are optional. What this installation can do is detected when the module is imported
(:func:`detect`; :func:`refresh` after a component was installed). When something is missing the text is passed
on unchanged with a notice in the trace; only a request that asked for ``on`` itself is refused (424
``component_missing`` when installing something would help, 400 for a language no pack exists for).
"""

from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bolo.errors import ComponentMissing, InvalidRequest

MODES = ("off", "auto", "on")
#: ``auto`` converts when at least this share of the text's letters is Latin
AUTO_LATIN_SHARE = 0.6
#: component ids of the packs: ``uccharan-xlit-hi`` installs uccharan's pack ``xlit-hi``
COMPONENT_PREFIX = "uccharan-"
PACK_PREFIX = "xlit-"
PACK_LICENSE = "CC BY-SA 4.0"
UPGRADE_HINT = "pip install -U uccharan"

#: languages whose voices can be handed romanised text (ISO 639 codes as the catalog uses them); which of them
#: can be converted depends on the packs the installed uccharan knows
INDIC_LANGUAGES = frozenset(
    "as awa bho bn brx doi gu hi hne kn kok ks mag mai ml mni mr ne or pa raj sa sat sd si ta te ur".split()  # noqa: SIM905
)

_TRUE = ("on", "true", "1", "yes")
_FALSE = ("off", "false", "0", "no", "none")


def parse_mode(value: Any, param: str = "romanised") -> str | None:
    """``off`` / ``auto`` / ``on`` from a request or settings value (booleans mean on / off); None when unset."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if not text:
        return None
    if text == "auto":
        return "auto"
    if text in _TRUE:
        return "on"
    if text in _FALSE:
        return "off"
    raise InvalidRequest(f"{param} must be one of {', '.join(MODES)}", param=param)


def language_base(language: str | None) -> str:
    return (language or "").strip().lower().replace("_", "-").split("-")[0]


def latin_share(text: str) -> tuple[int, float]:
    """How many Latin letters the text has, and their share of all its letters."""
    latin = letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        if ch.isascii() or "LATIN" in unicodedata.name(ch, ""):
            latin += 1
    return latin, (latin / letters if letters else 0.0)


# -- what the installed uccharan can do ------------------------------------------------------------------------


@dataclass(frozen=True)
class Api:
    """The transliteration support found in this environment."""

    #: the uccharan module, or None when it is not installed (or hidden with BOLO_UCCHARAN=off)
    module: Any | None = None
    #: ``transliterate(text, lang, ...)``, or None when the installed uccharan has no such function
    function: Callable[..., Any] | None = None
    #: the module's version, for messages
    version: str | None = None

    @property
    def available(self) -> bool:
        return self.function is not None

    @property
    def reason(self) -> str | None:
        """Why romanised text cannot be converted at all, with the way out; None when the API is there."""
        if self.function is not None:
            return None
        if self.module is None:
            return f"romanised text needs the uccharan package with transliteration support: {UPGRADE_HINT}"
        version = f" {self.version}" if self.version else ""
        return f"the installed uccharan{version} has no transliteration support: {UPGRADE_HINT}"


def detect() -> Api:
    """Look for uccharan's transliteration API (``uccharan.transliterate``, or ``uccharan.translit.transliterate``)."""
    if os.environ.get("BOLO_UCCHARAN", "auto").lower() == "off":
        return Api()
    try:
        import uccharan  # type: ignore[import-not-found,unused-ignore]
    except Exception:
        return Api()
    version = getattr(uccharan, "__version__", None)
    function = getattr(uccharan, "transliterate", None)
    if not callable(function):
        function = getattr(getattr(uccharan, "translit", None), "transliterate", None)
    return Api(uccharan, function if callable(function) else None, str(version) if version else None)


_lock = threading.Lock()
_api: Api = detect()
#: packs handed to ``uccharan.load_pack`` by this process
_loaded: set[str] = set()


def api() -> Api:
    return _api


def refresh() -> Api:
    """Detect again (after uccharan or a pack was installed into the running environment)."""
    global _api
    with _lock:
        _api = detect()
        _loaded.clear()
    return _api


def _pack_table(module: Any) -> dict[str, Any] | None:
    """uccharan's table of downloadable packs (name -> (sha256, size)), or None when it has none."""
    packs = getattr(module, "packs", None)
    if packs is None:
        try:
            import importlib

            packs = importlib.import_module(f"{module.__name__}.packs")
        except Exception:
            return None
    table = getattr(packs, "PACKS", None)
    return table if isinstance(table, dict) else None


def pack_name(language: str | None) -> str:
    return PACK_PREFIX + language_base(language)


def component_id(language: str | None) -> str:
    return COMPONENT_PREFIX + pack_name(language)


def pack_languages(found: Api | None = None) -> list[str]:
    """The languages the installed uccharan lists a romanised-text pack for."""
    found = found or _api
    if found.module is None or not found.available:
        return []
    table = _pack_table(found.module) or {}
    return sorted(name[len(PACK_PREFIX) :] for name in table if name.startswith(PACK_PREFIX))


def pack_file(language: str | None, found: Api | None = None) -> Path | None:
    """The pack's file where uccharan looks for it (``$UCCHARAN_DATA_DIR``, then its cache), or None."""
    found = found or _api
    name = pack_name(language)
    places: list[Path] = []
    places += [Path(d) for d in os.environ.get("UCCHARAN_DATA_DIR", "").split(os.pathsep) if d]
    packs = getattr(found.module, "packs", None)
    cache_dir = getattr(packs, "cache_dir", None)
    if callable(cache_dir):
        try:
            places.append(Path(cache_dir()))
        except Exception:  # an unreadable home directory: the pack is simply not there
            pass
    for place in places:
        path = place / f"{name}.ucd"
        if path.is_file():
            return path
    return None


def pack_present(language: str | None, found: Api | None = None) -> bool:
    """Whether the pack is on this machine (loaded, or its file is where uccharan looks)."""
    found = found or _api
    if found.module is None or not found.available:
        return False
    if pack_name(language) in _loaded or pack_file(language, found) is not None:
        return True
    has = getattr(found.module, "has_translit", None)
    if callable(has):
        try:
            return bool(has(language_base(language)))
        except Exception:
            return False
    return False


def pack_state(language: str | None, found: Api | None = None) -> str:
    """``ready`` (convert), ``not_installed`` (the pack exists but is not on this machine) or ``unsupported``
    (the installed uccharan has no pack for the language). An uccharan that says nothing about packs is ``ready``."""
    found = found or _api
    module = found.module
    if module is None or not found.available:
        return "not_installed"
    base, name = language_base(language), pack_name(language)
    has = getattr(module, "has_translit", None)
    if callable(has):
        try:
            if has(base):
                return "ready"
        except Exception:  # a language tag it does not know
            pass
    table = _pack_table(module)
    if table is None:
        return "not_installed" if callable(has) else "ready"
    if name not in table:
        return "unsupported"
    if name in _loaded:
        return "ready"
    if pack_file(language, found) is None:
        return "not_installed"
    with _lock:
        if name in _loaded:
            return "ready"
        try:
            module.load_pack(name)  # from the data directory or the cache: checked against its SHA-256, no download
        except Exception:
            return "not_installed"
        _loaded.add(name)
    return "ready"


def install_hint(language: str | None) -> str:
    return f"bolo components install {component_id(language)} --accept-license"


def manual_command(language: str | None) -> str:
    """The one-line command that fetches the pack into uccharan's cache without bolo."""
    return f"python -m uccharan.packs {pack_name(language)}"


def install_pack(language: str | None) -> str | None:
    """Download the pack into uccharan's cache (checked against the SHA-256 in the installed uccharan) and load
    it. Returns an error message, or None on success."""
    found = _api
    if found.module is None or not found.available:
        return found.reason
    name = pack_name(language)
    if name not in (_pack_table(found.module) or {}):
        return f"the installed uccharan has no pack '{name}'"
    from bolo.text import packs

    try:
        if found.module is packs.module():
            packs.install(name)  # bolo downloads it (checked against uccharan's SHA-256), uccharan loads it
        else:
            found.module.load_pack(name)
    except Exception as exc:
        return f"pack '{name}' could not be installed: {getattr(exc, 'message', exc)}"
    with _lock:
        _loaded.add(name)
    return None


# -- conversion ------------------------------------------------------------------------------------------------


def _result_text(result: Any) -> tuple[str | None, int | None]:
    """The converted text of whatever ``transliterate`` returned, and how many words it wrote in the script."""
    spans: Any = None
    text: Any = None
    if isinstance(result, str):
        stripped = result.lstrip()
        if stripped.startswith("{"):
            try:
                data = json.loads(result)
            except ValueError:
                return result, None
            if isinstance(data, dict) and isinstance(data.get("text"), str):
                text, spans = data["text"], data.get("spans")
            else:
                return result, None
        else:
            return result, None
    elif isinstance(result, dict):
        text, spans = result.get("text"), result.get("spans")
    else:
        text, spans = getattr(result, "text", None), getattr(result, "spans", None)
    if not isinstance(text, str):
        return None, None
    words: int | None = None
    if isinstance(spans, (list, tuple)):
        words = 0
        for span in spans:
            kind = span.get("kind") if isinstance(span, dict) else getattr(span, "kind", None)
            words += str(kind) == "indic"
    return text, words


def _call(function: Callable[..., Any], text: str, language: str) -> tuple[str | None, int | None]:
    try:
        result = function(text, language, english="keep", alternatives=0)
    except TypeError:  # an API without these options
        result = function(text, language)
    return _result_text(result)


#: a protected word, or (word, matched in its own case only)
Protected = str | tuple[str, bool]
#: addresses are read out by the normaliser, never converted
_ADDRESS = r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"


def _protect_pattern(text: str, protect: Iterable[Protected]) -> re.Pattern[str] | None:
    """What must reach the voice as written: word-list entries, inline pronunciations, URLs and email addresses.
    An entry is a word (any case) or ``(word, True)`` for one that is matched in its own case only."""
    lowered = text.lower()
    exact: set[str] = set()
    loose: set[str] = set()
    for item in protect:
        word, cased = (item, False) if isinstance(item, str) else item
        if word and (word in text if cased else word.lower() in lowered):
            (exact if cased else loose).add(word)
    alternatives = [f"(?i:{_ADDRESS})"]
    for words, flag in ((exact, ""), (loose, "i")):
        if words:
            body = "|".join(map(re.escape, sorted(words, key=len, reverse=True)))
            alternatives.append(
                rf"(?<![^\W\d_])(?{flag}:{body})(?![^\W\d_])" if flag else rf"(?<![^\W\d_])(?:{body})(?![^\W\d_])"
            )
    pattern = re.compile("|".join(alternatives))
    return pattern if pattern.search(text) else None


def _convert(
    function: Callable[..., Any], text: str, language: str, protect: Iterable[Protected]
) -> tuple[str, int | None]:
    pattern = _protect_pattern(text, protect)
    if pattern is None:
        out, words = _call(function, text, language)
        if out is None:
            raise ValueError("the transliterator returned no text")
        return out, words
    parts: list[str] = []
    total: int | None = 0
    last = 0
    for match in [*pattern.finditer(text), None]:
        segment = text[last : match.start()] if match else text[last:]
        if any(ch.isalpha() for ch in segment):
            out, words = _call(function, segment, language)
            if out is None:
                raise ValueError("the transliterator returned no text")
            # the transliterator may trim the ends of what it is given: keep the spacing around protected words
            lead = segment[: len(segment) - len(segment.lstrip())]
            tail = segment[len(segment.rstrip()) :]
            parts.append(lead + out.strip() + tail)
            total = None if total is None or words is None else total + words
        else:
            parts.append(segment)
        if match:
            parts.append(match.group(0))
            last = match.end()
    return "".join(parts), total


@dataclass
class Outcome:
    """What happened to one request's text."""

    text: str
    #: the trace record (``romanised`` in a TTS trace and in the preview), or None when nothing was considered
    record: dict[str, Any] | None = None

    @property
    def notice(self) -> str | None:
        return str(self.record["notice"]) if self.record and self.record.get("notice") else None


def apply(
    text: str,
    language: str | None,
    mode: str,
    *,
    source: str = "request",
    strict: bool = False,
    protect: Iterable[Protected] | Callable[[], Iterable[Protected]] = (),
) -> Outcome:
    """Convert ``text`` for a voice of ``language`` as ``mode`` asks.

    ``strict`` marks a request that named the mode itself: with ``on`` it is refused when the conversion cannot
    happen. In every other case the text comes back unchanged with a notice. This function does not raise for
    anything the transliterator does.
    """
    if mode not in MODES or mode == "off":
        return Outcome(text)
    base = language_base(language)
    latin, share = latin_share(text)
    record: dict[str, Any] = {"mode": mode, "source": source, "language": base or None, "applied": False}
    refuse = strict and mode == "on"
    if base not in INDIC_LANGUAGES:
        if mode == "auto":
            return Outcome(text)
        notice = f"romanised text is converted for voices of Indian languages; this voice speaks '{base or '?'}'"
        if refuse:
            raise InvalidRequest(notice, param="romanised")
        return Outcome(text, {**record, "notice": notice})
    if latin == 0 or (mode == "auto" and share < AUTO_LATIN_SHARE):
        return Outcome(text)  # native script already, or mixed text that `auto` leaves alone
    record["latin_share"] = round(share, 2)
    found = _api
    if not found.available or found.function is None:
        notice = found.reason or "romanised text cannot be converted here"
        if refuse:
            raise ComponentMissing(notice)
        return Outcome(text, {**record, "notice": f"not converted: {notice}"})
    state = pack_state(base, found)
    record["pack"] = pack_name(base)
    if state == "unsupported":
        known = ", ".join(pack_languages(found)) or "none"
        notice = f"uccharan has no romanised-text pack for '{base}' (packs: {known})"
        if refuse:
            raise InvalidRequest(notice, param="romanised")
        return Outcome(text, {**record, "notice": f"not converted: {notice}"})
    if state != "ready":
        notice = (
            f"the romanised-text pack for '{base}' ({pack_name(base)}, {PACK_LICENSE}) is not installed: "
            f"{install_hint(base)}"
        )
        if refuse:
            raise ComponentMissing(notice)
        return Outcome(text, {**record, "notice": f"not converted: {notice}"})
    try:
        out, words = _convert(found.function, text, base, protect() if callable(protect) else protect)
    except Exception as exc:  # whatever the transliterator does, the text is still spoken
        return Outcome(text, {**record, "notice": f"not converted: the transliterator failed ({exc})"})
    record["applied"] = out != text
    record["text"] = out[:500]
    if words is not None:
        record["words"] = words
    return Outcome(out, record)


def status() -> dict[str, Any]:
    """What this installation can do (``capabilities.romanised`` of ``GET /api/v1/info``)."""
    found = _api
    languages = pack_languages(found)
    return {
        "available": found.available,
        "reason": found.reason,
        "modes": list(MODES),
        "languages": languages,
        "installed": [lang for lang in languages if pack_present(lang, found)],
    }
