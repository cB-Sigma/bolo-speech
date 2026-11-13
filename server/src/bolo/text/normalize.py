"""Light text normalisation before synthesis.

The built-in normaliser is deliberately small and predictable: Unicode NFC,
whitespace, removal of markup and emoji that a voice would read out, and
spelled-out integers for English and Hindi (other Indic languages pass
digits through to the voice's own front end). Code-mixed input (for
example Hindi with English words in Latin script) is passed through
untouched: nothing is transliterated.

For evaluation (WER) :func:`normalize_for_wer` uses ``whisper-normalizer``
when it is installed (``bolo-speech[text]``) and a simple fallback otherwise.
"""

from __future__ import annotations

import re
import unicodedata

_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\U0000fe0f\U0000200d]+", flags=re.UNICODE
)
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~|`)(?=\S)(.+?)(?<=\S)\1")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", flags=re.MULTILINE)
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"[ \t ]+")

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_INDIC_DIGIT_RANGES = [0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66]

_EN_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_EN_TENS = ["zero", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

_HI_0_99 = [
    "शून्य",
    "एक",
    "दो",
    "तीन",
    "चार",
    "पाँच",
    "छह",
    "सात",
    "आठ",
    "नौ",
    "दस",
    "ग्यारह",
    "बारह",
    "तेरह",
    "चौदह",
    "पंद्रह",
    "सोलह",
    "सत्रह",
    "अठारह",
    "उन्नीस",
    "बीस",
    "इक्कीस",
    "बाईस",
    "तेईस",
    "चौबीस",
    "पच्चीस",
    "छब्बीस",
    "सत्ताईस",
    "अट्ठाईस",
    "उनतीस",
    "तीस",
    "इकतीस",
    "बत्तीस",
    "तैंतीस",
    "चौंतीस",
    "पैंतीस",
    "छत्तीस",
    "सैंतीस",
    "अड़तीस",
    "उनतालीस",
    "चालीस",
    "इकतालीस",
    "बयालीस",
    "तैंतालीस",
    "चौवालीस",
    "पैंतालीस",
    "छियालीस",
    "सैंतालीस",
    "अड़तालीस",
    "उनचास",
    "पचास",
    "इक्यावन",
    "बावन",
    "तिरपन",
    "चौवन",
    "पचपन",
    "छप्पन",
    "सत्तावन",
    "अट्ठावन",
    "उनसठ",
    "साठ",
    "इकसठ",
    "बासठ",
    "तिरसठ",
    "चौंसठ",
    "पैंसठ",
    "छियासठ",
    "सड़सठ",
    "अड़सठ",
    "उनहत्तर",
    "सत्तर",
    "इकहत्तर",
    "बहत्तर",
    "तिहत्तर",
    "चौहत्तर",
    "पचहत्तर",
    "छिहत्तर",
    "सतहत्तर",
    "अठहत्तर",
    "उन्यासी",
    "अस्सी",
    "इक्यासी",
    "बयासी",
    "तिरासी",
    "चौरासी",
    "पचासी",
    "छियासी",
    "सत्तासी",
    "अट्ठासी",
    "नवासी",
    "नब्बे",
    "इक्यानवे",
    "बानवे",
    "तिरानवे",
    "चौरानवे",
    "पंचानवे",
    "छियानवे",
    "सत्तानवे",
    "अट्ठानवे",
    "निन्यानवे",
]
assert len(_HI_0_99) == 100


def english_number(n: int) -> str:
    if n < 0:
        return "minus " + english_number(-n)
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _EN_TENS[tens] + ("-" + _EN_ONES[ones] if ones else "")
    if n < 1000:
        h, rest = divmod(n, 100)
        return _EN_ONES[h] + " hundred" + (" and " + english_number(rest) if rest else "")
    for value, name in ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (10**3, "thousand")):
        if n >= value:
            head, rest = divmod(n, value)
            if head >= 1000:
                break
            return english_number(head) + " " + name + (" " + english_number(rest) if rest else "")
    return " ".join(_EN_ONES[int(d)] for d in str(n))


def hindi_number(n: int) -> str:
    """Spell an integer in Hindi using the Indian system (हज़ार, लाख, करोड़)."""
    if n < 0:
        return "ऋण " + hindi_number(-n)
    if n < 100:
        return _HI_0_99[n]
    parts: list[str] = []
    for value, name in ((10**7, "करोड़"), (10**5, "लाख"), (10**3, "हज़ार"), (100, "सौ")):
        if n >= value:
            head, n = divmod(n, value)
            parts.append(hindi_number(head) + " " + name)
    if n:
        parts.append(_HI_0_99[n])
    return " ".join(parts)


def _to_ascii_digits(text: str) -> str:
    table: dict[int, str] = {}
    for base in _INDIC_DIGIT_RANGES:
        for i in range(10):
            table[base + i] = str(i)
    return text.translate(table)


_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d+))?(?![\w]|[.,]\d)")


def _expand_numbers(text: str, lang: str) -> str:
    spell = hindi_number if lang == "hi" else english_number
    point = "दशमलव" if lang == "hi" else "point"
    digit_names = _HI_0_99 if lang == "hi" else _EN_ONES

    def repl(m: re.Match[str]) -> str:
        whole = m.group(1).replace(",", "")
        if len(whole) > 13:
            return m.group(0)
        if len(whole) > 1 and whole.startswith("0"):
            return " ".join(digit_names[int(d)] for d in whole)
        words = spell(int(whole))
        if m.group(2):
            words += f" {point} " + " ".join(digit_names[int(d)] for d in m.group(2))
        return words

    return _NUMBER.sub(repl, text)


def strip_markup(text: str) -> str:
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_HEADING.sub("", text)
    for _ in range(2):
        text = _MD_EMPHASIS.sub(r"\2", text)
    return _EMOJI.sub("", text)


def normalize_for_tts(text: str, language: str | None = None, *, expand_numbers: bool = True) -> str:
    """Clean text for synthesis. ``language`` is a BCP-47-ish code such as ``en`` or ``hi-IN``."""
    lang = (language or "en").split("-")[0].split("_")[0].lower()
    text = unicodedata.normalize("NFC", text)
    text = strip_markup(text)
    text = _URL.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_WS.sub(" ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if expand_numbers and lang in ("en", "hi"):
        if lang == "hi":
            text = text.translate(_DEVANAGARI_DIGITS)
        text = _expand_numbers(text, lang)
    return text


def strip_punctuation(text: str, replacement: str = " ") -> str:
    """Remove punctuation and symbols but keep combining marks (Indic vowel signs are not ``\\w``)."""
    return "".join(replacement if unicodedata.category(ch)[0] in "PS" else ch for ch in text)


def normalize_for_wer(text: str, language: str | None = None) -> str:
    """Normalise a transcript for word-error-rate comparison."""
    lang = (language or "en").split("-")[0].lower()
    try:
        if lang == "en":
            from whisper_normalizer.english import EnglishTextNormalizer

            return str(EnglishTextNormalizer()(text))
        if lang in ("hi", "bn", "ta", "te", "ml", "kn", "mr", "gu", "pa", "or", "as", "ur", "ne"):
            from whisper_normalizer.indic import IndicTextNormalizer  # type: ignore[attr-defined]

            return str(IndicTextNormalizer()(text))
        from whisper_normalizer.basic import BasicTextNormalizer

        return str(BasicTextNormalizer()(text))
    except Exception:
        pass
    text = unicodedata.normalize("NFKC", _to_ascii_digits(text)).lower()
    if lang == "en":
        text = _expand_numbers(text, "en").replace("-", " ")
    text = strip_punctuation(text.replace("'", ""))
    return " ".join(text.split())


def word_error_rate(reference: str, hypothesis: str, language: str | None = None) -> float:
    """Word error rate after normalisation (Levenshtein distance over words)."""
    ref = normalize_for_wer(reference, language).split()
    hyp = normalize_for_wer(hypothesis, language).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)
