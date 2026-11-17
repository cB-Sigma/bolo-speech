"""The language of a text to be spoken, guessed from its script and, for Latin script, its common words.

No dependency and no model: the script of the letters settles most languages (Devanagari, Tamil, Bengali,
Gurmukhi, Hangul, Thai, ...). Where several languages share a script, a few frequent words or letters choose
among them (Marathi or Nepali in Devanagari, Assamese in Bengali script, Urdu or Persian in Arabic script,
Ukrainian in Cyrillic). Latin-script text is scored against short lists of the most frequent words of twenty
languages; a short or ambiguous text is reported as unsure (``language`` None) rather than guessed.

:func:`detect` answers with the language (ISO 639-1 where there is one), the script, a confidence and the other
languages written in that script, so a caller can pick the closest language a voice speaks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: (first code point, last code point, script)
_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0041, 0x005A, "Latin"), (0x0061, 0x007A, "Latin"), (0x00C0, 0x024F, "Latin"), (0x1E00, 0x1EFF, "Latin"),
    (0x0370, 0x03FF, "Greek"), (0x1F00, 0x1FFF, "Greek"),
    (0x0400, 0x052F, "Cyrillic"),
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"), (0x0750, 0x077F, "Arabic"), (0xFB50, 0xFDFF, "Arabic"), (0xFE70, 0xFEFF, "Arabic"),
    (0x0900, 0x097F, "Devanagari"), (0xA8E0, 0xA8FF, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"),
    (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1100, 0x11FF, "Hangul"), (0x3130, 0x318F, "Hangul"), (0xAC00, 0xD7AF, "Hangul"),
    (0x1200, 0x139F, "Ethiopic"),
    (0x1780, 0x17FF, "Khmer"),
    (0x1C50, 0x1C7F, "Ol Chiki"),
    (0xABC0, 0xABFF, "Meetei Mayek"),
    (0x3040, 0x309F, "Kana"), (0x30A0, 0x30FF, "Kana"), (0x31F0, 0x31FF, "Kana"),
    (0x3400, 0x4DBF, "Han"), (0x4E00, 0x9FFF, "Han"), (0xF900, 0xFAFF, "Han"),
)  # fmt: skip

#: the languages written in a script, most widely spoken first; the first is the answer when nothing narrows it
SCRIPT_LANGUAGES: dict[str, tuple[str, ...]] = {
    "Devanagari": ("hi", "mr", "ne", "mai", "bho", "sa", "kok", "doi", "brx", "mag", "hne"),
    "Bengali": ("bn", "as", "mni"),
    "Gurmukhi": ("pa",),
    "Gujarati": ("gu",),
    "Oriya": ("or",),
    "Tamil": ("ta",),
    "Telugu": ("te",),
    "Kannada": ("kn",),
    "Malayalam": ("ml",),
    "Sinhala": ("si",),
    "Ol Chiki": ("sat",),
    "Meetei Mayek": ("mni",),
    "Arabic": ("ar", "ur", "fa", "ps", "sd", "ks"),
    "Hebrew": ("he", "yi"),
    "Greek": ("el",),
    "Cyrillic": ("ru", "uk", "bg", "sr", "be", "mk", "kk"),
    "Armenian": ("hy",),
    "Georgian": ("ka",),
    "Thai": ("th",),
    "Lao": ("lo",),
    "Khmer": ("km",),
    "Myanmar": ("my",),
    "Tibetan": ("bo",),
    "Ethiopic": ("am", "ti"),
    "Hangul": ("ko",),
    "Kana": ("ja",),
    "Han": ("zh", "yue", "ja"),
}

#: the most frequent words of Latin-script languages (lower case); a few per language are enough to tell them apart
_LATIN_WORDS: dict[str, frozenset[str]] = {
    lang: frozenset(words.split())
    for lang, words in {
        "en": "the and is are was were to of in that it you he she we they this with for not have has be on at "
        "what there will would can do does my your from but",
        "es": "el la los las de que y en un una es por con para no se lo como más pero sus le ya o fue este "
        "está muy también hay yo",
        "fr": "le la les de des et est un une que qui en du dans pour pas sur ce il elle nous vous sont avec "
        "au aux mais ne je",
        "de": "der die das und ist nicht ein eine zu den mit von sich des auf für im dem auch es ich sie wir "
        "aber wie sind war",
        "it": "il lo la gli le di che e è un una per non sono del della con si ma come anche ho questo nel alla mi io",
        "pt": "o a os as de que e do da em um uma para com não é se na no por mais dos das como mas foi ao "
        "ele ela você",
        "nl": "de het een en van is dat niet ik je op te zijn met voor er maar die wat hij ze ook als bij naar heb",
        "sv": "och att det är en som på för av med jag inte till har den de om ett var men så vi han hon kan",
        "da": "og at det er en som på for af med jeg ikke til har den de om et var men så vi han hun kan skal",
        "no": "og at det er en som på for av med jeg ikke til har den de om et var men så vi han hun kan skal ikkje",
        "fi": "ja on ei se että hän oli ovat kuin mutta tai minä sinä me te he tämä joka mitä myös kanssa olen",
        "pl": "i w nie na się z że do to jest jak co ale o po tak jego jej czy przez dla od był są",
        "cs": "a v se na je že s z do to jak ale o by jsem jsou pro jeho jako už tak není bylo",
        "ro": "și în de la a care cu nu un o pe este din să se mai ca pentru sunt dar sau fost",
        "tr": "ve bir bu da de için ile ne çok o ben sen var değil gibi daha ama olarak en kadar mi",
        "hu": "a az és hogy nem is egy van de meg ez csak már volt mint el ki vagy még kell",
        "id": "yang dan di ini itu dengan untuk tidak dari dalam akan pada ada saya kami mereka juga atau bisa sudah",
        "vi": "và của là có không một những được trong cho này người với các đã để tôi anh em",
        "ca": "el la els les de que i a en un una és per amb no es del al com però més hi",
        "sw": "na ya wa kwa ni za la katika hii kuwa yake hiyo lakini pia sana mimi wewe sisi",
    }.items()
}

#: letters (lower case) that belong to only a few Latin-script languages, with the weight they add
_LATIN_LETTERS: dict[str, dict[str, float]] = {
    "ñ": {"es": 2}, "¿": {"es": 3}, "¡": {"es": 3},
    "ã": {"pt": 2}, "õ": {"pt": 2},
    "ß": {"de": 3},
    "œ": {"fr": 3}, "ê": {"fr": 0.5, "pt": 0.5, "vi": 0.5}, "è": {"fr": 1, "it": 0.5, "ca": 0.5},
    "ł": {"pl": 3}, "ą": {"pl": 3}, "ę": {"pl": 3}, "ś": {"pl": 2}, "ź": {"pl": 2}, "ż": {"pl": 2}, "ń": {"pl": 2},
    "ř": {"cs": 3}, "ě": {"cs": 3}, "ů": {"cs": 3},
    "ș": {"ro": 3}, "ş": {"ro": 1, "tr": 2}, "ț": {"ro": 3}, "ţ": {"ro": 2}, "ă": {"ro": 2, "vi": 1},
    "ğ": {"tr": 3}, "ı": {"tr": 3},
    "ő": {"hu": 3}, "ű": {"hu": 3},
    "å": {"sv": 1, "da": 1, "no": 1}, "ø": {"da": 1.5, "no": 1.5}, "æ": {"da": 1.5, "no": 1.5},
    "ä": {"de": 0.5, "sv": 0.5, "fi": 0.7}, "ö": {"de": 0.5, "sv": 0.5, "fi": 0.5, "tr": 0.5, "hu": 0.5},
    "đ": {"vi": 3}, "ơ": {"vi": 3}, "ư": {"vi": 3}, "ạ": {"vi": 3}, "ả": {"vi": 3}, "ấ": {"vi": 3}, "ộ": {"vi": 3},
    "ệ": {"vi": 3}, "ế": {"vi": 3}, "ờ": {"vi": 3}, "ủ": {"vi": 3},
}  # fmt: skip


def _words_of(text: str) -> frozenset[str]:
    return frozenset(text.split(" "))


#: frequent words that tell languages of one non-Latin script apart
_SCRIPT_WORDS: dict[str, dict[str, frozenset[str]]] = {
    "Devanagari": {
        "hi": _words_of("है हैं और का की के में नहीं से को यह था थी थे हम आप मैं कि भी पर क्या"),
        "mr": _words_of("आहे आहेत आणि नाही हे ते मी तुम्ही आम्ही काय होते होता त्या या पण की"),
        "ne": _words_of("छ छन् हो र को मा गर्न भएको पनि त्यो यो म तिमी हुन्छ थियो छैन"),
    },
    "Cyrillic": {
        "ru": _words_of("и в не на что я с он как это по но все она так его только мне было"),
        "uk": _words_of("і в не на що я з він як це по але все вона так його тільки мені було"),
        "bg": _words_of("и в не на че аз с той как това по но всички тя така е са ще"),
    },
}

#: letters that only some languages of a script use
_SCRIPT_LETTERS: dict[str, tuple[tuple[str, str], ...]] = {
    "Bengali": (("ৰৱ", "as"),),
    "Arabic": (("ٹڈڑںےھۓ", "ur"), ("پچژگی", "fa")),
    "Cyrillic": (("іїєґ", "uk"), ("ў", "be"), ("ђјљњћџ", "sr"), ("ѓќѕ", "mk"), ("әғқңөұүһ", "kk")),
    "Kana": (),
}

# letters, with the vowel signs and other combining marks of the Indian scripts (not word characters to ``re``)
_WORD_RE = re.compile(r"(?:[^\W\d_]|[\u0300-\u036f\u0900-\u0dff\u200c\u200d])+(?:['’][^\W\d_]+)?")
_TAG_RE = re.compile(r"<[^>]{0,200}>|\[([^\]]*)\]\(/[^)]*/\)")


@dataclass
class Detection:
    """What :func:`detect` found. ``language`` is None when it is unsure."""

    language: str | None
    script: str | None
    confidence: float
    #: the other languages written in ``script`` (closest first), for a voice that does not speak ``language``
    alternatives: tuple[str, ...] = ()
    reason: str = ""
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def sure(self) -> bool:
        return self.language is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "script": self.script,
            "confidence": round(self.confidence, 2),
            **({"reason": self.reason} if self.reason else {}),
        }


def script_of(ch: str) -> str | None:
    cp = ord(ch)
    for lo, hi, name in _RANGES:
        if lo <= cp <= hi:
            return name
    return None


def script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isalpha():
            s = script_of(ch)
            if s is not None:
                counts[s] = counts.get(s, 0) + 1
    return counts


def _dominant_script(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    total = sum(counts.values())
    # Japanese mixes kana and kanji: any real share of kana makes it Japanese
    if counts.get("Kana", 0) >= max(2, 0.1 * (counts.get("Kana", 0) + counts.get("Han", 0))):
        return "Kana"
    # Indian and other texts carry English words in Latin letters: a script of its own with a third of the letters
    # wins over the Latin ones
    others = {s: n for s, n in counts.items() if s != "Latin"}
    if others:
        best = max(others, key=lambda s: others[s])
        if others[best] >= total / 3:
            return best
    return max(counts, key=lambda s: counts[s])


def _words(text: str) -> list[str]:
    return [w.lower().replace("’", "'") for w in _WORD_RE.findall(text)]


def _latin(text: str) -> Detection:
    words = _words(text)
    if not words:
        return Detection(None, "Latin", 0.0, reason="no words")
    scores: dict[str, float] = {}
    for w in words:
        for lang, common in _LATIN_WORDS.items():
            if w in common:
                scores[lang] = scores.get(lang, 0.0) + 1.0
    lowered = text.lower()
    for ch, weights in _LATIN_LETTERS.items():
        n = lowered.count(ch)
        if n:
            for lang, weight in weights.items():
                scores[lang] = scores.get(lang, 0.0) + weight * min(n, 3)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked:
        return Detection(None, "Latin", 0.0, reason="no common words of a known language", scores=scores)
    best, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    # Danish and Norwegian (Bokmål) share most frequent words: a tie between them is still Scandinavian, not sure
    confidence = (top - second) / top if top else 0.0
    if top < 2.0 or (len(words) < 3 and top < 3.0):
        return Detection(None, "Latin", confidence, reason="too few words to tell", scores=scores)
    if second and top < 1.5 * second:
        return Detection(
            None, "Latin", confidence, reason=f"{best} or {ranked[1][0]}", alternatives=(best, ranked[1][0]),
            scores=scores,
        )  # fmt: skip
    return Detection(best, "Latin", min(1.0, 0.5 + 0.5 * confidence), scores=scores)


def _within_script(script: str, text: str) -> Detection:
    langs = SCRIPT_LANGUAGES.get(script, ())
    if not langs:
        return Detection(None, script, 0.0, reason="no language known for the script")
    choice = langs[0]
    confidence = 0.9 if len(langs) == 1 else 0.75
    reason = ""
    for letters, lang in _SCRIPT_LETTERS.get(script, ()):
        if any(ch in text for ch in letters):
            choice, reason = lang, "letters"
            break
    words = _SCRIPT_WORDS.get(script)
    if words and not reason:
        tokens = _words(text)
        scores = {lang: float(sum(1 for t in tokens if t in common)) for lang, common in words.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        if ranked and ranked[0][1] >= 1 and (len(ranked) < 2 or ranked[0][1] > ranked[1][1]):
            choice, reason = ranked[0][0], "words"
            confidence = 0.85
    if script == "Han":
        confidence = 0.8  # Chinese (Cantonese and Japanese kanji read the same letters)
    alternatives = tuple(lang for lang in langs if lang != choice)
    return Detection(choice, script, confidence, alternatives, reason)


def detect(text: str) -> Detection:
    """Guess the language of ``text`` (markup such as SSML tags is ignored)."""
    clean = _TAG_RE.sub(lambda m: m.group(1) or " ", text or "")
    counts = script_counts(clean)
    script = _dominant_script(counts)
    if script is None:
        return Detection(None, None, 0.0, reason="no letters")
    if script == "Latin":
        return _latin(clean)
    return _within_script(script, clean)


def base(tag: str | None) -> str:
    """``"hi-IN"`` -> ``"hi"``; ``"nb"`` counts as Norwegian (``"no"``)."""
    code = (tag or "").replace("_", "-").split("-")[0].lower()
    return {"nb": "no", "nn": "no", "fil": "tl", "zho": "zh", "cmn": "zh"}.get(code, code)
