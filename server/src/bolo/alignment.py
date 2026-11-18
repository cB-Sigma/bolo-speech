"""CTC forced alignment: where in a recording each word of a known text is spoken.

A CTC acoustic model (task ``align``, :class:`bolo.engines.base.AlignModel`) gives, for every frame of audio, the
log probability of each of its tokens and of "blank". Given the text, the most likely path through the frames that
spells it out (Viterbi over the CTC graph: blank, token, blank, token, ...) puts each token on the frames where it
is heard; a word spans its tokens. This is what wav2vec 2.0 / MMS forced alignment does; it runs here in NumPy on
the model's frame scores, so any character-level CTC model in ONNX can serve.

The text is cut into words at whitespace (and, for scripts written without spaces, at each character); each word
is turned into the model's tokens (``tokenize``); a word with no token the model knows (punctuation, a symbol) gets
the time where it stands, without length. Times are seconds from the start of the audio.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from bolo.engines.base import Word

#: scripts written without spaces between words: each character counts as a word (as Whisper does for zh/ja)
_UNSPACED = re.compile("[぀-ヿ㐀-䶿一-鿿豈-﫿฀-๿຀-໿က-႟ក-៿가-힯]")
#: largest Viterbi table (frames x 2 * tokens + 1) aligned in one piece; about 5 minutes of speech with its text
MAX_CELLS = 250_000_000


class AlignmentError(ValueError):
    """The text cannot be aligned to the audio (too long, or more tokens than frames)."""


@dataclass
class AlignedWord:
    word: str
    start: float
    end: float
    #: mean probability of the word's tokens on the frames they were put on (0..1); None for a word with no token
    score: float | None

    def to_word(self) -> Word:
        return Word(self.word, self.start, self.end)


def split_words(text: str) -> list[str]:
    """The words of ``text``: whitespace-separated, and each character of a script written without spaces."""
    out: list[str] = []
    for chunk in text.split():
        if not _UNSPACED.search(chunk):
            out.append(chunk)
            continue
        buf = ""
        for ch in chunk:
            if _UNSPACED.match(ch):
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(ch)
            elif unicodedata.category(ch).startswith("P") and out and not buf:
                out[-1] += ch  # punctuation stays with the character before it
            else:
                buf += ch
        if buf:
            out.append(buf)
    return out


def char_tokenizer(vocab: dict[str, int], romanize: Callable[[str], str] | None = None) -> Callable[[str], list[int]]:
    """A word -> token ids for a character-level vocabulary: NFKC, lower case, optionally romanized (MMS), and
    without the characters the vocabulary does not have (punctuation, most symbols); a letter with a diacritic the
    vocabulary lacks counts as its base letter."""
    lowered = {k.lower(): v for k, v in vocab.items() if len(k) == 1}

    def tokenize(word: str) -> list[int]:
        text = unicodedata.normalize("NFKC", word).lower()
        if romanize is not None:
            text = romanize(text).lower()
        ids: list[int] = []
        for c in text:
            if c.isspace():
                continue
            if c in lowered:
                ids.append(lowered[c])
                continue
            # a letter with a diacritic the vocabulary has only without it
            ids.extend(lowered[b] for b in unicodedata.normalize("NFKD", c) if b in lowered)
        return ids

    return tokenize


def viterbi(log_probs: np.ndarray, targets: list[int], blank: int) -> np.ndarray:
    """The CTC state of every frame on the most likely path that spells ``targets``: state ``2i + 1`` is token
    ``i``, even states are blanks. ``log_probs`` is (frames, vocabulary)."""
    frames = log_probs.shape[0]
    n = len(targets)
    states = 2 * n + 1
    repeats = sum(1 for a, b in itertools.pairwise(targets) if a == b)
    if n + repeats > frames:
        raise AlignmentError(
            f"the text has more sounds ({n + repeats} tokens) than the audio has frames ({frames}): "
            "is it the transcript of this recording?"
        )
    if frames * states > MAX_CELLS:
        raise AlignmentError(
            f"audio and text too long to align in one piece ({frames} frames x {n} tokens); "
            "align it in parts of a few minutes"
        )
    ext = np.full(states, blank, dtype=np.int64)
    ext[1::2] = targets
    # skipping a blank (s-2 -> s) is allowed into a token that differs from the token before it
    skip = np.zeros(states, dtype=bool)
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    neg = np.float32(-1e30)
    lp = log_probs[:, ext].astype(np.float32)
    alpha = np.full(states, neg, dtype=np.float32)
    alpha[0] = lp[0, 0]
    alpha[1] = lp[0, 1]
    back = np.zeros((frames, states), dtype=np.uint8)  # 0: stay, 1: from s-1, 2: from s-2
    for t in range(1, frames):
        stay = alpha
        one = np.concatenate(([neg], alpha[:-1]))
        two = np.where(skip, np.concatenate(([neg, neg], alpha[:-2])), neg)
        best = np.maximum(stay, np.maximum(one, two))
        move = np.where(two > np.maximum(stay, one), 2, np.where(one > stay, 1, 0)).astype(np.uint8)
        back[t] = move
        alpha = best + lp[t]
    s = states - 1 if states == 1 or alpha[states - 1] >= alpha[states - 2] else states - 2
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    return path


def align_words(
    log_probs: np.ndarray,
    frame_s: float,
    words: list[str],
    tokens: list[list[int]],
    blank: int,
    separator: int | None = None,
    offset: float = 0.0,
) -> list[AlignedWord]:
    """Each word with the time it is spoken. ``tokens[i]`` are word ``i``'s token ids (may be empty);
    ``separator`` is a token the model puts between words (a space), put between them in the target too."""
    targets: list[int] = []
    owner: list[int] = []  # the word each target token belongs to (-1: a separator)
    for i, ids in enumerate(tokens):
        if not ids:
            continue
        if separator is not None and targets:
            targets.append(separator)
            owner.append(-1)
        targets.extend(ids)
        owner.extend([i] * len(ids))
    out: list[AlignedWord] = [AlignedWord(w, offset, offset, None) for w in words]
    if not targets:
        return out
    path = viterbi(log_probs, targets, blank)
    frames_of: dict[int, list[int]] = {}
    for t, s in enumerate(path):
        if s % 2:
            frames_of.setdefault(int(s) // 2, []).append(t)
    probs = np.exp(log_probs)
    spans: dict[int, tuple[int, int, list[float]]] = {}
    for k, word in enumerate(owner):
        if word < 0 or k not in frames_of:
            continue
        frames = frames_of[k]
        p = [float(probs[t, targets[k]]) for t in frames]
        first, last, scores = spans.get(word, (frames[0], frames[-1], []))
        spans[word] = (min(first, frames[0]), max(last, frames[-1]), scores + p)
    previous_end = offset
    for i, item in enumerate(out):
        if i in spans:
            first, last, scores = spans[i]
            item.start = offset + first * frame_s
            item.end = offset + (last + 1) * frame_s
            item.score = round(float(np.mean(scores)), 4)
            previous_end = item.end
        else:  # nothing to hear (punctuation): where it stands, after the word before it
            item.start = item.end = previous_end
    return out


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32)
    x = x - x.max(axis=-1, keepdims=True)
    out: np.ndarray = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
    return out
