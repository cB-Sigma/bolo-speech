"""Stabilising partial transcripts.

A non-streaming model re-decodes a growing audio buffer, and each pass may
revise earlier words. Showing every revision makes text flicker.
``LocalAgreement`` commits a word only once ``n`` consecutive hypotheses
agree on it (Macháček et al., "Turning Whisper into Real-Time Transcription
System", 2023); committed words are never taken back within an utterance.
The partial that is sent is the committed prefix followed by the newest
unconfirmed tail, with ``stable_chars`` marking where the committed part
ends.
"""

from __future__ import annotations

import unicodedata

from bolo.text.normalize import strip_punctuation


def _key(word: str) -> str:
    return strip_punctuation(unicodedata.normalize("NFKC", word), "").lower()


class LocalAgreement:
    def __init__(self, n: int = 2) -> None:
        self.n = max(1, n)
        self.committed: list[str] = []
        self._history: list[list[str]] = []
        self.retractions = 0  # committed words the newest hypothesis disagreed with

    def reset(self) -> None:
        self.committed = []
        self._history = []

    def update(self, hypothesis: str) -> tuple[str, int]:
        """Feed a new full hypothesis. Returns ``(text to show, stable_chars)``."""
        words = hypothesis.split()
        self._history.append(words)
        if len(self._history) > self.n:
            self._history.pop(0)
        if len(self._history) >= self.n:
            agreed = 0
            shortest = min(len(h) for h in self._history)
            while agreed < shortest and len({_key(h[agreed]) for h in self._history}) == 1:
                agreed += 1
            if agreed > len(self.committed):
                self.committed.extend(words[len(self.committed) : agreed])
        for i, w in enumerate(self.committed):
            if i >= len(words) or _key(words[i]) != _key(w):
                self.retractions += 1
                break
        tail = words[len(self.committed) :]
        stable = " ".join(self.committed)
        text = " ".join([*self.committed, *tail])
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
