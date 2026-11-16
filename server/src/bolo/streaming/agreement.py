"""Stabilising partial transcripts.

A non-streaming model re-decodes a growing audio buffer, and each pass may
revise earlier words. Showing every revision makes text flicker.
``LocalAgreement`` commits a word only once ``n`` consecutive hypotheses
agree on it (Macháček et al., "Turning Whisper into Real-Time Transcription
System", 2023); committed words are never taken back within an utterance.
The partial that is sent is the committed prefix followed by the newest
unconfirmed tail, with ``stable_chars`` marking where the committed part
ends. Each new hypothesis is aligned with the committed words before its new tail is
taken, so a re-decode that splits or merges a word before that point does not repeat one.
"""

from __future__ import annotations

import unicodedata

from bolo.text.normalize import strip_punctuation


def _key(word: str) -> str:
    return strip_punctuation(unicodedata.normalize("NFKC", word), "").lower()


def _committed_end(committed: list[str], words: list[str]) -> tuple[int, int]:
    """Where the committed words end in a new hypothesis, and how far that hypothesis strays from them.

    A re-decode can split, merge, insert or drop words before the end of the committed part ("it has finished" can
    come back as "it has, uh, finished reading"). Counting off ``len(committed)`` words would then repeat or skip a
    word at the seam, so the committed keys are aligned with the hypothesis instead (word-level edit distance against
    each of its prefixes). The best prefix wins; among equals the one that ends on the last committed word, then the
    longer one. Returns ``(prefix length, edit distance)``.
    """
    if not committed:
        return 0, 0
    a = [_key(w) for w in committed]
    b = [_key(w) for w in words]
    prev = list(range(len(b) + 1))  # distance between a[:0] and b[:j]
    for i, ka in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, kb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ka != kb))
        prev = cur
    best = min(range(len(b) + 1), key=lambda j: (prev[j], not (j and b[j - 1] == a[-1]), -j))
    return best, prev[best]


class LocalAgreement:
    def __init__(self, n: int = 2) -> None:
        self.n = max(1, n)
        self.committed: list[str] = []
        self._history: list[list[str]] = []
        self.retractions = 0  # hypotheses that disagreed with the committed words

    def reset(self) -> None:
        self.committed = []
        self._history = []

    def update(self, hypothesis: str) -> tuple[str, int]:
        """Feed a new full hypothesis. Returns ``(text to show, stable_chars)``."""
        words = hypothesis.split()
        self._history.append(words)
        if len(self._history) > self.n:
            self._history.pop(0)
        # each hypothesis, past the point where the committed words end in it
        cut, distance = _committed_end(self.committed, words)
        if distance:
            self.retractions += 1
        rest = words[cut:]
        if len(self._history) >= self.n:
            rests = [h[_committed_end(self.committed, h)[0] :] for h in self._history[:-1]] + [rest]
            agreed = 0
            shortest = min(len(r) for r in rests)
            while agreed < shortest and len({_key(r[agreed]) for r in rests}) == 1:
                agreed += 1
            self.committed.extend(rest[:agreed])
            rest = rest[agreed:]
        stable = " ".join(self.committed)
        text = " ".join([*self.committed, *rest])
        return text, len(stable)

    def finalize(self, final_text: str) -> str:
        self.reset()
        return final_text


class PrefixStability:
    """For true streaming models: how much of the text survived from the previous partial."""

    def __init__(self) -> None:
        self._prev = ""

    def reset(self) -> None:
        self._prev = ""

    def update(self, text: str) -> int:
        n = 0
        for a, b in zip(self._prev, text):
            if a != b:
                break
            n += 1
        self._prev = text
        # only count whole words as stable
        if n < len(text):
            cut = text.rfind(" ", 0, n + 1)
            n = max(cut, 0)
        return n
