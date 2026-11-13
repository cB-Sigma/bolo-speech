"""Sentence segmentation for streaming synthesis.

Synthesis latency is dominated by the first chunk, so text is cut into
sentences and synthesised one at a time. The splitter understands Latin
punctuation, the Devanagari danda (``।``, ``॥``), Urdu/Arabic full stop
(``۔``, ``؟``) and CJK stops, keeps common abbreviations and decimals
together, and never splits inside a number. Mixed-script (code-mixed) text
passes through unchanged apart from the cuts.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

_TERMINATORS = ".!?।॥۔؟。！？…"
_CLOSERS = "\"'”’)]}»"
_SOFT = ",;:—–،、，；："

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd", "co", "no", "fig", "gen",
    "col", "lt", "sgt", "capt", "rev", "hon", "mt", "ft", "e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "approx",
    "dept", "est", "min", "max", "shri", "smt", "km", "kg", "rs",
}  # fmt: skip

_BOUNDARY = re.compile(rf"([{re.escape(_TERMINATORS)}]+[{re.escape(_CLOSERS)}]*)(\s+|$)|(\n\s*\n)")
_CJK_BOUNDARY = re.compile(r"([。！？]+)")


def _is_abbreviation(before: str) -> bool:
    m = re.search(r"([A-Za-z][A-Za-z.]*)$", before)
    if not m:
        return False
    word = m.group(1).rstrip(".").lower()
    if word in _ABBREVIATIONS:
        return True
    # Single capital letter initials: "J. K. Rowling"
    return len(word) == 1 and m.group(1)[0].isupper()


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, preserving their punctuation."""
    text = text.strip()
    if not text:
        return []
    out: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        end = m.end(1) if m.group(1) else m.start(3)
        punct = m.group(1) or ""
        before = text[start : m.start(1)] if m.group(1) else text[start:end]
        if punct.startswith(".") and len(punct.rstrip(_CLOSERS)) == 1:
            if _is_abbreviation(before):
                continue
            nxt = text[m.end() : m.end() + 1]
            if nxt and nxt.islower() and nxt.isascii():
                continue  # "approx. five" style continuation
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    # CJK text has no spaces after stops: split those further.
    final: list[str] = []
    for s in out:
        if _CJK_BOUNDARY.search(s) and not re.search(r"\s", s.strip()):
            parts = _CJK_BOUNDARY.split(s)
            for i in range(0, len(parts) - 1, 2):
                piece = (parts[i] + parts[i + 1]).strip()
                if piece:
                    final.append(piece)
            if len(parts) % 2 == 1 and parts[-1].strip():
                final.append(parts[-1].strip())
        else:
            final.append(s)
    return final


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    pieces: list[str] = []
    current = ""
    for token in re.split(rf"(?<=[{re.escape(_SOFT)}])\s+", sentence):
        if current and len(current) + 1 + len(token) > max_chars:
            pieces.append(current)
            current = token
        else:
            current = f"{current} {token}".strip()
    if current:
        pieces.append(current)
    out: list[str] = []
    for p in pieces:  # still too long: cut at spaces
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def chunk_text(text: str, *, max_chars: int = 300, first_max_chars: int = 120, min_chars: int = 12) -> list[str]:
    """Cut ``text`` into synthesis chunks.

    The first chunk is kept short (``first_max_chars``) so audio starts
    early; very short sentences are merged with the next one so prosody
    does not suffer.
    """
    chunks: list[str] = []
    for sentence in split_sentences(text):
        limit = first_max_chars if not chunks else max_chars
        for piece in _split_long(sentence, limit):
            merged_limit = first_max_chars if len(chunks) == 1 else max_chars
            if chunks and len(chunks[-1]) < min_chars and len(chunks[-1]) + len(piece) + 1 <= merged_limit:
                chunks[-1] = f"{chunks[-1]} {piece}"
            else:
                chunks.append(piece)
    return chunks


class IncrementalSegmenter:
    """Feed text as it arrives (e.g. LLM tokens); get complete sentences out."""

    def __init__(self, max_chars: int = 300, first_max_chars: int = 120) -> None:
        self._buf = ""
        self._max = max_chars
        self._first_max = first_max_chars
        self._emitted = 0

    def push(self, text: str) -> list[str]:
        self._buf += text
        sentences = split_sentences(self._buf)
        if not sentences:
            return []
        ends_clean = bool(re.search(rf"[{re.escape(_TERMINATORS)}][{re.escape(_CLOSERS)}]*\s+$", self._buf)) or (
            self._buf.endswith("\n")
        )
        ready = sentences if ends_clean else sentences[:-1]
        # Keep the unfinished tail exactly as it was typed: tokens arrive with their own spaces.
        consumed = 0
        for sentence in ready:
            found = self._buf.find(sentence, consumed)
            consumed = (found if found >= 0 else consumed) + len(sentence)
        rest = self._buf[consumed:].lstrip()
        limit = self._first_max if self._emitted == 0 else self._max
        if not ready and len(rest) > limit * 2:
            # No sentence end in sight: release everything up to the last soft break.
            parts = _split_long(rest.strip(), limit)
            ready, rest = parts[:-1], parts[-1] + (" " if rest[-1:].isspace() else "")
        self._buf = rest
        out = list(self._emit(ready))
        return out

    def flush(self) -> list[str]:
        rest, self._buf = self._buf, ""
        return list(self._emit(split_sentences(rest)))

    def _emit(self, sentences: list[str]) -> Iterator[str]:
        for s in sentences:
            limit = self._first_max if self._emitted == 0 else self._max
            for piece in _split_long(s, limit):
                self._emitted += 1
                yield piece
