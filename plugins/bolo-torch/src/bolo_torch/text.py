"""Sentence chunking for streamed synthesis (Latin and Indic punctuation)."""

from __future__ import annotations

import re

# sentence ends: . ! ? and their full-width forms, the Devanagari danda and double danda, the Urdu full stop
_SENTENCE_END = re.compile(r"[.!?。！？।॥۔][\"'”’)\]]*(?=\s)")
_CLAUSE_END = re.compile(r"(?<=[,;:،])\s+")


def split_sentences(text: str, max_chars: int = 200) -> list[str]:
    """Split text into sentences no longer than ``max_chars`` (long ones at clause marks, then at spaces)."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out: list[str] = []
    for sentence in _sentences(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            out.append(sentence)
            continue
        out.extend(_pack(_CLAUSE_END.split(sentence), max_chars))
    return out


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        out.append(text[start : match.end()])
        start = match.end()
    out.append(text[start:])
    return out


def _pack(parts: list[str], max_chars: int) -> list[str]:
    out: list[str] = []
    current = ""
    for part in parts:
        for word_chunk in _hard_wrap(part, max_chars):
            candidate = f"{current} {word_chunk}".strip()
            if current and len(candidate) > max_chars:
                out.append(current)
                current = word_chunk
            else:
                current = candidate
    if current:
        out.append(current)
    return out


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    current = ""
    for word in text.split(" "):
        if current and len(current) + 1 + len(word) > max_chars:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out
