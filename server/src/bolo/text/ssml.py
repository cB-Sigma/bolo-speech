"""A subset of SSML 1.1 for ``/v1/audio/speech`` (``input_format: "ssml"``).

Supported: ``<speak>``, ``<p>``, ``<s>``, ``<sub alias>``, ``<phoneme alphabet="ipa" ph>``,
``<say-as interpret-as format>`` (characters / spell-out / verbatim, cardinal / number,
ordinal, digits / number_digit / telephone, date with ``format`` dmy, mdy, ymd, dm, md, ym,
time, currency), ``<lang xml:lang>``, ``<break>``, and ``<emphasis>`` / ``<prosody>``
(their text is kept, the styling ignored). Anything else is a 400 error.

``<sub>`` replaces its text with the alias where it stands; ``<phoneme>`` becomes an inline
pronunciation entry (the highest priority, see ``bolo.text.prepare``); ``<say-as>`` rewrites its
text into a form the normaliser reads the requested way.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from bolo.errors import InvalidRequest

_INLINE = ("emphasis", "prosody", "lang", "voice", "mark")


@dataclass
class SsmlText:
    """Plain text and the inline pronunciation entries taken from ``<phoneme>`` elements."""

    text: str
    entries: list[dict[str, str]] = field(default_factory=list)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def say_as(text: str, interpret_as: str, fmt: str | None, language: str | None) -> str:
    """Rewrite ``text`` so that the normaliser reads it as ``interpret_as`` asks."""
    kind = interpret_as.strip().lower()
    t = text.strip()
    lang = (language or "en").split("-")[0].lower()
    if kind in ("characters", "spell-out", "verbatim", "letters", "spell"):
        return " ".join(ch.upper() if ch.isalpha() else ch for ch in t if not ch.isspace())
    if kind in ("digits", "number_digit", "telephone", "phone"):
        return " ".join(ch for ch in t if ch.isdigit() or ch == "+")
    if kind in ("cardinal", "number"):
        return re.sub(r"[^\d.,+\-]", "", t) or t
    if kind == "ordinal":
        digits = re.sub(r"\D", "", t)
        if not digits:
            return t
        if lang == "hi":
            return f"{digits}वां"
        return digits + _ordinal_suffix(int(digits)) if lang == "en" else digits
    if kind == "date" and fmt:
        parts = [p for p in re.split(r"[/.\-\s]+", t) if p]
        order = fmt.strip().lower()
        if len(parts) == len(order) and set(order) <= set("dmy"):
            got = dict(zip(order, parts))
            if len(order) == 3:
                return f"{got['y'].zfill(4)}-{got['m'].zfill(2)}-{got['d'].zfill(2)}"
        return t
    return t


def parse_ssml(source: str, language: str | None = None) -> SsmlText:
    """Turn SSML into plain text plus inline entries; malformed input is a 400 error with its position."""
    src = source.strip()
    if "<!DOCTYPE" in src.upper() or "<!ENTITY" in src.upper():
        raise InvalidRequest("SSML must not contain a DOCTYPE or entity declarations", param="input")
    if not re.match(r"^(<\?xml[^>]*>\s*)?<speak[\s>/]", src):
        src = f"<speak>{src}</speak>"
    try:
        root = ET.fromstring(src)
    except ET.ParseError as exc:
        line, col = exc.position
        raise InvalidRequest(f"malformed SSML at line {line}, column {col}: {exc}", param="input") from exc
    out: list[str] = []
    entries: list[dict[str, str]] = []

    def walk(el: ET.Element, lang: str | None) -> None:
        tag = _local(el.tag)
        attrs = {_local(k): v for k, v in el.attrib.items()}
        if tag == "lang":
            lang = attrs.get("lang") or lang
        if tag == "break":
            out.append(", ")
        elif tag == "sub":
            alias = attrs.get("alias")
            if alias is None:
                raise InvalidRequest("<sub> needs an alias attribute", param="input")
            out.append(alias)
        elif tag == "phoneme":
            ph = attrs.get("ph")
            alphabet = attrs.get("alphabet", "ipa").lower()
            word = "".join(el.itertext()).strip()
            if ph is None:
                raise InvalidRequest("<phoneme> needs a ph attribute", param="input")
            if alphabet != "ipa":
                raise InvalidRequest(f"<phoneme> alphabet {alphabet!r} is not supported (only ipa)", param="input")
            if word:
                entries.append({"word": word, "ipa": ph, "match": "case"})
                out.append(word)
        elif tag == "say-as":
            kind = attrs.get("interpret-as")
            if not kind:
                raise InvalidRequest("<say-as> needs an interpret-as attribute", param="input")
            out.append(say_as("".join(el.itertext()), kind, attrs.get("format"), lang or language))
        elif tag in ("speak", "p", "s", *_INLINE):
            if el.text:
                out.append(el.text)
            for child in el:
                walk(child, lang)
                if child.tail:
                    out.append(child.tail)
            if tag in ("p", "s"):
                out.append("\n")
            return
        elif tag == "audio":
            pass
        else:
            raise InvalidRequest(f"the SSML element <{tag}> is not supported", param="input")

    walk(root, None)
    text = re.sub(r"[ \t]+", " ", "".join(out))
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    return SsmlText(text, entries)
