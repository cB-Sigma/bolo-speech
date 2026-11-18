"""Hotword spelling: a transcript's near-misses of the request's hotwords, written the way the hotwords are.

A recogniser given hotwords often hears the name but spells it by ear: whispercpp-small, given "Kshitij Wadhwani
Zentriva", wrote "Kshitij Waadhwaani at Zentriva Labs". This pass works on the transcript of any engine: a word, or a
run of adjacent words up to a hotword's own word count, that comes close to a hotword is replaced by the hotword's
exact spelling. Close means close after a phonetic folding (lower case, accents dropped, aa/ee/oo and doubled
letters to one, ph to f, w and v alike, an h after a consonant dropped, ...) within a small edit distance, and never
for short words or words that start differently, so that real different words stay ("wanted" is not "Wadhwani").

Punctuation around a replaced word and an all-capitals word stay as they were; word timestamps follow (a run of words
becomes one word spanning them, or the hotword's words share the run's time).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bolo.engines.base import Segment, Word

#: the largest edit distance allowed, as a share of the longer folded form (rounded down)
MAX_DISTANCE_SHARE = 0.25
#: folded forms shorter than this must match exactly (short words are too easy to confuse)
MIN_FUZZY_LENGTH = 5
_SPLIT = re.compile(r"[,;/\n]+")
_EDGE = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.UNICODE)


def phrases(hotwords: str | None) -> list[str]:
    """The hotwords of a request: phrases separated by commas, semicolons, slashes or new lines; the words of a
    phrase count on their own too (free text such as "Kshitij Wadhwani Zentriva" is a list of names), and so do its
    runs of adjacent words ("Kshitij Wadhwani"), which a recogniser may write as one word ("Kshetjwadwani")."""
    out: list[str] = []
    for phrase in _SPLIT.split(hotwords or ""):
        phrase = " ".join(phrase.split())
        if not phrase:
            continue
        words = phrase.split()
        runs = [" ".join(words[i : i + n]) for n in range(len(words) - 1, 1, -1) for i in range(len(words) - n + 1)]
        for item in [phrase, *runs, *(words if len(words) > 1 else [])]:
            if item not in out and len(fold(item)) >= 3:
                out.append(item)
    return out


def fold(text: str) -> str:
    """A spelling-insensitive form: "Waadhwaani", "Wadhwani" and "Vadvani" all become "vadvani"."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if c.isalnum() and not unicodedata.combining(c))
    for a, b in (("ph", "f"), ("ck", "k"), ("q", "k"), ("w", "v"), ("ee", "i"), ("oo", "u"), ("y", "i")):
        text = text.replace(a, b)
    text = re.sub(r"([bcdfgjklmnpqrstvxz])h", r"\1", text)  # dh -> d, bh -> b, kh -> k, sh -> s, th -> t
    return re.sub(r"(.)\1+", r"\1", text)  # aa -> a, nn -> n


def _distance(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def matches(heard: str, hotword: str) -> bool:
    """Whether ``heard`` (one word, or several joined by spaces) is ``hotword`` spelt by ear."""
    a, b = fold(heard), fold(hotword)
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) < MIN_FUZZY_LENGTH or a[0] != b[0]:
        return False
    return _distance(a, b) <= int(MAX_DISTANCE_SHARE * max(len(a), len(b)))


def _core(token: str) -> tuple[str, str, str]:
    m = _EDGE.match(token)
    assert m is not None
    return m.group(1), m.group(2), m.group(3)


def _cased(original: list[str], hotword: str) -> str:
    """The hotword's spelling, in capitals when the words it replaces were all in capitals."""
    letters = "".join(original)
    return hotword.upper() if len(letters) > 1 and letters.isupper() else hotword


def _plan(tokens: list[str], hotwords: list[str]) -> list[tuple[int, int, str]]:
    """``(start, end, replacement)`` over ``tokens``: the longest hotword first, left to right, never overlapping."""
    cores = [_core(t)[1] for t in tokens]
    taken = [False] * len(tokens)
    inner = {h for h in hotwords if " " in h and any(h != o and f" {h} " in f" {o} " for o in hotwords)}
    found: list[tuple[int, int, str]] = []
    for hotword in sorted(hotwords, key=lambda h: (-len(h.split()), -len(h))):
        width = len(hotword.split())
        for n in range(width, 0, -1):
            for i in range(len(tokens) - n + 1):
                if any(taken[i : i + n]) or not all(cores[i : i + n]):
                    continue
                run = cores[i : i + n]
                # only the first and last words of a run may carry punctuation inside the run
                if n > 1 and any(_core(t)[2] for t in tokens[i : i + n - 1]):
                    continue
                if " ".join(run) == hotword or not matches(" ".join(run), hotword):
                    continue
                # a run inside a longer phrase ("Kshitij Wadhwani" of "Kshitij Wadhwani Zentriva") is only for words
                # heard joined ("Kshetjwadwani"); heard one for one, the word-by-word pass says which word changed
                if n == width > 1 and hotword in inner:
                    continue
                found.append((i, i + n, _cased(run, hotword)))
                taken[i : i + n] = [True] * n
    return sorted(found)


def spell_text(text: str, hotwords: list[str]) -> tuple[str, list[dict[str, Any]]]:
    """``text`` with its near-misses of ``hotwords`` replaced, and what was changed (``{"from", "to"}``)."""
    tokens = text.split(" ")
    plan = _plan(tokens, hotwords)
    if not plan:
        return text, []
    changes = []
    for start, end, word in reversed(plan):
        lead, _, _ = _core(tokens[start])
        _, _, trail = _core(tokens[end - 1])
        heard = " ".join(_core(t)[1] for t in tokens[start:end])
        tokens[start:end] = [lead + word + trail]
        changes.append({"from": heard, "to": word})
    return " ".join(tokens), list(reversed(changes))


def spell_words(words: list[Word], hotwords: list[str]) -> list[Word]:
    """Word timestamps that follow the replacements: a run of words becomes the hotword's words over the same time
    (one word when the hotword is one word)."""
    from bolo.engines.base import Word

    tokens = [w.w.strip() for w in words]
    plan = _plan(tokens, hotwords)
    if not plan:
        return list(words)
    out = list(words)
    for start, end, hotword in reversed(plan):
        run = words[start:end]
        lead = run[0].w[: len(run[0].w) - len(run[0].w.lstrip())]  # the leading space engines put in a word
        first_lead, _, _ = _core(run[0].w.strip())
        _, _, last_trail = _core(run[-1].w.strip())
        parts = hotword.split()
        t0, t1 = run[0].t0, run[-1].t1
        probs = [w.p for w in run if w.p is not None]
        p = min(probs) if probs else None
        if len(parts) == len(run):
            new = [replace(w, w=(lead if k == 0 else w.w[: len(w.w) - len(w.w.lstrip())]) + part)
                   for k, (w, part) in enumerate(zip(run, parts))]  # fmt: skip
        else:
            span = (t1 - t0) / len(parts)
            new = [Word((lead if k == 0 else " ") + part, t0 + k * span, t0 + (k + 1) * span, p)
                   for k, part in enumerate(parts)]  # fmt: skip
        new[0] = replace(new[0], w=new[0].w[: len(new[0].w) - len(new[0].w.lstrip())] + first_lead + new[0].w.lstrip())
        new[-1] = replace(new[-1], w=new[-1].w + last_trail)
        out[start:end] = new
    return out


@dataclass
class Spelling:
    """The hotword pass of one request or session."""

    hotwords: list[str]

    @classmethod
    def of(cls, hotwords: str | None, enabled: bool = True) -> Spelling | None:
        found = phrases(hotwords) if enabled else []
        return cls(found) if found else None

    def text(self, text: str, words: list[Word] | None = None) -> tuple[str, list[Word], list[dict[str, Any]]]:
        new, changes = spell_text(text, self.hotwords)
        return new, spell_words(list(words or []), self.hotwords) if changes else list(words or []), changes

    def segment(self, segment: Segment) -> tuple[Segment, list[dict[str, Any]]]:
        text, words, changes = self.text(segment.text, segment.words)
        if not changes:
            return segment, []
        for c in changes:
            c.setdefault("start", round(segment.start, 3))
        return replace(segment, text=text, words=words), changes


def spell_transcript(transcript: Any, spelling: Spelling) -> list[dict[str, Any]]:
    """Apply ``spelling`` to a transcript (``bolo.core.Transcript``): its segments, its text and its words. Returns
    what was changed, by segment (the text's changes when there are no segments)."""
    changes: list[dict[str, Any]] = []
    segments = []
    for seg in transcript.segments:
        new, found = spelling.segment(seg)
        segments.append(new)
        changes.extend(found)
    transcript.segments = segments
    text, words, found = spelling.text(transcript.text, transcript.words)
    transcript.text, transcript.words = text, words
    return changes if transcript.segments else found
