"""Turn request text into what one voice should receive.

Every voice gets a :class:`TextInput` record that says how it reads text:

* ``phonemes``: bolo phonemises for it (Piper, Kokoro, Kitten, Matcha through sherpa-onnx). Word-list
  entries reach it as ``[word](/ipa/)`` overrides that the phonemiser turns into the entry's IPA in the
  voice's phone set.
* ``text``: the voice reads text itself (tts-server and Orpheus llama.cpp models, plugin voices).
  By default uccharan's normalise-only mode writes numbers, dates, money, URLs and so on out in words
  (bolo's own light normaliser when uccharan is not installed), and word-list entries become their
  spoken form, or a respelling uccharan chooses from the entry's IPA. A voice whose front end takes
  Kokoro-style ``[word](/ipa/)`` markup (catalog option ``text_markup: kokoro``) gets the markup.
* ``characters``: voices with a closed character set (Coqui-style VITS, MMS): the light normaliser,
  spoken forms of word-list entries, then the voice's own fitting.

Pronunciation sources, first match wins (see uccharan's docs/wordlists.md for matching):
SSML ``<phoneme>`` and ``[word](/ipa/)`` in the input, request ``pronunciations``, request
``wordlists`` in order, lists pinned to the model, then the ``default`` list.
"""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from bolo.errors import ComponentMissing, InvalidRequest
from bolo.text import romanised as rom
from bolo.text import wordlists as wl
from bolo.text.normalize import OVERRIDE, normalize_for_tts, strip_markup

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.core import Bolo

NORMALIZERS = ("uccharan", "light", "model", "none")
INPUT_FORMATS = ("text", "ssml", "markdown")
#: engines whose voices read text themselves
TEXT_ENGINES = ("tts-server", "remote")


@dataclass(frozen=True)
class TextInput:
    """How a voice takes text (the per-voice capability record)."""

    #: "phonemes" | "text" | "characters"
    reads: str
    #: default normaliser: "uccharan" | "light" | "model" | "none"
    normalizer: str
    #: inline pronunciation markup the voice's front end understands: "none" | "kokoro"
    markup: str = "none"
    #: Latin-script words in other scripts: "keep" | "transliterate"
    latin: str = "keep"
    #: respelling style for entries with IPA only: "native" | "latin-hyphen" | "latin-plain"
    respell: str = "native"
    #: which language reads digits: "context" | "request" | a language tag
    digits: str = "context"
    #: word lists pinned to the model, in priority order
    wordlists: tuple[str, ...] = ()
    #: romanised text (an Indian language typed in Latin letters) is written in the voice's script first:
    #: "off" | "auto" | "on" (the model's pin, else the catalog option, else the server's `romanised` setting)
    romanised: str = "off"

    @property
    def pronunciation(self) -> str:
        """What a word-list entry turns into for this voice: phonemes, markup or respelling."""
        if self.reads == "phonemes":
            return "phonemes"
        return "markup" if self.markup != "none" else "respelling"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["wordlists"] = list(self.wordlists)
        d["pronunciation"] = self.pronunciation
        return d


def text_input(core: Bolo, spec: ModelSpec, voice_id: str | None = None) -> TextInput:
    """The capability record of a voice: catalog options, then the model's pinned settings."""
    opts = spec.options
    pins = core.store.model_settings(spec.id)
    if core.components.phoneme_target(spec, voice_id) is not None:
        reads = "phonemes"
    elif spec.engine in TEXT_ENGINES or spec.engine.startswith("plugin:"):
        reads = "text"
    elif spec.engine == "sherpa-onnx":
        reads = "characters"
    else:
        reads = "text"
    default_norm = "uccharan" if reads == "text" and spec.engine != "fake" else "light"
    normalizer = str(pins.get("text_normalizer") or opts.get("text_normalizer") or default_norm)
    respell = str(opts.get("text_respell") or ("latin-plain" if opts.get("lowercases") else "native"))
    return TextInput(
        reads=reads,
        normalizer=normalizer if normalizer in NORMALIZERS else default_norm,
        markup=str(opts.get("text_markup") or "none"),
        latin=str(opts.get("text_latin") or "keep"),
        respell=respell,
        digits=str(opts.get("numbers") or opts.get("text_digits") or "context"),
        wordlists=tuple(str(x) for x in (pins.get("wordlists") or opts.get("wordlists") or ())),
        romanised=_default_romanised(core, pins.get("romanised"), opts.get("romanised")),
    )


def _default_romanised(core: Bolo, pinned: Any, option: Any) -> str:
    """The voice's default mode. A stored value that is not a mode is ignored: it must not fail a request."""
    for value in (pinned, option, getattr(core.settings, "romanised", None)):
        try:
            mode = rom.parse_mode(value)
        except InvalidRequest:
            continue
        if mode is not None:
            return mode
    return "off"


def uccharan_language(language: str | None) -> str | None:
    """The uccharan language tag for a request language, or None when uccharan does not read it."""
    mod = wl.uccharan_module()
    if mod is None:
        return None
    from bolo.text.phonemize import uccharan_tag

    tag = uccharan_tag(language or "en")
    langs = getattr(mod, "LANGUAGES", ())
    return tag if tag is not None and (not langs or tag in langs) else None


#: uccharan language -> whether its normalise-only mode reads numbers in that language (see normalizes_numbers)
_reads_numbers: dict[str, bool] = {}


def normalizes_numbers(tag: str) -> bool:
    """Whether uccharan's normalise-only mode writes numbers out in the language itself.

    English, Hindi and the other South Asian languages always are. In the Latin-script languages other than English
    (Spanish, German, French …) uccharan 0.6.0's normalise-only mode reads digits in English, while its phonemiser
    reads them in the language; 0.6.1 (the extra's minimum) reads them in the language. With an uccharan that does
    not, those languages get bolo's light normaliser for text voices; the probe below finds out which case the
    installed uccharan is in."""
    from bolo.text.packs import LATIN_SCRIPT

    if tag in ("en-us", "en-gb") or tag not in LATIN_SCRIPT:
        return True
    if tag not in _reads_numbers:
        mod = wl.uccharan_module()
        try:
            out = str(mod.normalize("7", tag, digits="request").text).strip().lower() if mod is not None else ""
        except Exception:
            out = ""
        _reads_numbers[tag] = bool(out) and out not in ("7", "seven", "सात")
    return _reads_numbers[tag]


@dataclass
class Prepared:
    text: str
    normalizer: str
    applied: list[dict[str, Any]] = field(default_factory=list)
    #: what happened to romanised text (bolo.text.romanised), or None when the option did not come into play
    romanised: dict[str, Any] | None = None

    def trace_extra(self) -> dict[str, Any]:
        out: dict[str, Any] = {"normalizer": self.normalizer, "normalized_text": self.text[:500]}
        if self.applied:
            out["pronunciations_applied"] = self.applied[:50]
        if self.romanised:
            out["romanised"] = self.romanised
        return out


def extract_overrides(text: str) -> tuple[str, list[dict[str, str]]]:
    """``[word](/ipa/)`` markup -> plain text plus inline entries."""
    entries: list[dict[str, str]] = []
    for word, ipa in OVERRIDE.findall(text):
        entries.append({"word": word.strip(), "ipa": ipa.strip(), "match": "case"})
    return OVERRIDE.sub(r"\1", text), entries


def build_lists(
    core: Bolo,
    cap: TextInput,
    inline_entries: list[dict[str, str]],
    pronunciations: list[dict[str, Any]] | None,
    wordlists: list[str] | None,
) -> list[Any]:
    """The word lists that apply, highest priority first."""
    explicit = bool(pronunciations) or bool(wordlists)
    if wl.uccharan_module() is None:
        if explicit:
            wl.require_uccharan()
        return []
    store = wl.store_for(core.settings.data_dir)
    lists: list[Any] = []
    if inline_entries:
        lists.append(wl.inline(inline_entries, param="input"))
    if pronunciations:
        lists.append(wl.inline(pronunciations))
    if wordlists is not None and not wordlists:
        return lists  # [] turns stored lists off for this request
    names: list[str] = []
    for n in [*(wordlists or []), *cap.wordlists, wl.DEFAULT]:
        if n not in names:
            names.append(n)
    if len([n for n in names if n != wl.DEFAULT]) > wl.MAX_LISTS_PER_REQUEST:
        raise InvalidRequest(f"at most {wl.MAX_LISTS_PER_REQUEST} word lists per request", param="wordlists")
    for n in names:
        if n == wl.DEFAULT or n in cap.wordlists:
            if store.exists(n):
                lists.append(store.load(n))
        else:
            lists.append(store.load(wl.check_name(n)))  # a missing requested list is a 404
    return lists


def _pre_clean(text: str) -> str:
    """What every path does first: NFC and markup a voice would read out (links, emphasis, emoji)."""
    return strip_markup(unicodedata.normalize("NFC", text)).strip()


def apply_lists(
    text: str, language: str, lists: list[Any], *, as_markup: bool, respell: str = "native", latin: str = "keep"
) -> tuple[str, list[dict[str, Any]]]:
    """Apply word lists only (no other normalisation): matched words become their spoken form, or
    ``[word](/ipa/)`` overrides with ``as_markup`` when the entry has IPA."""
    if not lists:
        return text, []
    mod = wl.require_uccharan()
    ulang = uccharan_language(language) or "en-us"
    n = mod.normalize(text, ulang, urls="keep", latin=latin, respell=respell, wordlists=lists)
    entries_cache: dict[int, list[dict[str, Any]]] = {}
    out: list[str] = []
    applied: list[dict[str, Any]] = []
    for s in n.spans:
        if s.kind != "wordlist" or s.entry is None:
            out.append(s.text)
            continue
        li, ei = s.entry
        if li not in entries_cache:
            entries_cache[li] = lists[li].entries()
        entry = entries_cache[li][ei]
        ipa = str(entry.get("ipa") or "").replace(" ", "").replace(".", "").replace("/", "")
        if as_markup and ipa:
            out.append(f"[{s.text}](/{ipa}/)")
            applied.append({"text": s.text, "ipa": ipa, "word": entry.get("word")})
        else:
            out.append(s.spoken)
            applied.append({"text": s.text, "spoken": s.spoken, "word": entry.get("word")})
    return "".join(out), applied


def _protected_words(lists: list[Any]) -> list[tuple[str, bool]]:
    """Words with a pronunciation of their own: they are matched as written, so they are not converted.
    Each comes with whether its entry matches in its own case only (``match`` = case or exact)."""
    words: list[tuple[str, bool]] = []
    for item in lists:
        try:
            words += [
                (str(e.get("word") or ""), str(e.get("match") or "") in ("case", "exact")) for e in item.entries()
            ]
        except Exception:  # a list that cannot be read protects nothing
            continue
    return words


def prepare(
    core: Bolo,
    spec: ModelSpec,
    voice_id: str | None,
    text: str,
    language: str,
    *,
    normalize: bool | str = True,
    wordlists: list[str] | None = None,
    pronunciations: list[dict[str, Any]] | None = None,
    input_format: str = "text",
    romanised: bool | str | None = None,
    romanised_strict: bool = True,
) -> Prepared:
    """The text one voice receives for a request (before the voice's own character fitting).

    ``romanised`` is the request's choice (``off`` / ``auto`` / ``on``; None: the voice's default).
    ``romanised_strict`` says the caller named it for this very text: ``on`` is then refused when it cannot be
    done; a session or an agent passes False and gets a notice instead.
    """
    if input_format not in INPUT_FORMATS:
        raise InvalidRequest(f"input_format must be one of {', '.join(INPUT_FORMATS)}", param="input_format")
    cap = text_input(core, spec, voice_id)
    inline_entries: list[dict[str, str]] = []
    if input_format == "ssml":
        from bolo.text.ssml import parse_ssml

        parsed = parse_ssml(text, language)
        text, inline_entries = parsed.text, parsed.entries
    text, marked = extract_overrides(text)
    inline_entries += marked
    lists = build_lists(core, cap, inline_entries, pronunciations, wordlists)
    requested = rom.parse_mode(romanised)
    outcome = rom.apply(
        text,
        language,
        requested or cap.romanised,
        source="request" if requested else "voice",
        strict=romanised_strict and requested is not None,
        protect=lambda: [*((str(e.get("word") or ""), True) for e in inline_entries), *_protected_words(lists)],
    )
    prepared = _normalise(core, cap, outcome.text, language, lists, normalize)
    prepared.romanised = outcome.record
    return prepared


def _normalise(
    core: Bolo, cap: TextInput, text: str, language: str, lists: list[Any], normalize: bool | str
) -> Prepared:
    """Normaliser and word lists (the text is already in the voice's script)."""

    if normalize is True:
        mode = cap.normalizer
    elif normalize is False:
        mode = "none"
    else:
        mode = str(normalize)
        if mode not in NORMALIZERS:
            raise InvalidRequest(f"normalize must be true, false or one of {', '.join(NORMALIZERS)}", param="normalize")
    ulang = uccharan_language(language)
    if mode == "uccharan" and (ulang is None or not normalizes_numbers(ulang)):
        mode = "light"  # uccharan is not installed, or does not read this language's numbers

    if cap.reads == "text" and mode == "uccharan":
        mod = wl.require_uccharan()
        try:
            n = mod.normalize(
                _pre_clean(text),
                ulang,
                latin=cap.latin,
                digits=cap.digits,
                urls="read",
                respell=cap.respell,
                wordlists=lists,
            )
        except ValueError as exc:
            raise InvalidRequest(str(exc), param="input") from exc
        applied: list[dict[str, Any]] = []
        if cap.markup == "kokoro" and any(s.kind == "wordlist" for s in n.spans):
            # Rebuild with markup for entries that have IPA.
            parts: list[str] = []
            for s in n.spans:
                if s.kind == "wordlist" and s.entry is not None:
                    entry = lists[s.entry[0]].entries()[s.entry[1]]
                    ipa = str(entry.get("ipa") or "").replace(" ", "")
                    parts.append(f"[{s.text}](/{ipa}/)" if ipa else s.spoken)
                    applied.append({"text": s.text, "ipa": ipa or None, "spoken": s.spoken})
                else:
                    parts.append(s.spoken)
            return Prepared("".join(parts), mode, applied)
        for s in n.spans:
            if s.kind == "wordlist":
                applied.append({"text": s.text, "spoken": s.spoken})
        return Prepared(n.text, mode, applied)

    if mode in ("light", "uccharan"):
        out = normalize_for_tts(text, language)
        mode = "light"
    elif mode in ("model", "none"):
        out = text
    else:  # pragma: no cover - checked above
        raise InvalidRequest("unknown normaliser", param="normalize")
    if not lists:
        return Prepared(out, mode)
    try:
        out, applied = apply_lists(
            out,
            language,
            lists,
            as_markup=cap.reads == "phonemes" or cap.markup != "none",
            respell=cap.respell,
            latin=cap.latin,
        )
    except ComponentMissing:
        raise
    except ValueError as exc:
        raise InvalidRequest(str(exc), param="input") from exc
    return Prepared(out, mode, applied)
