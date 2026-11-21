"""kale-tts-1-nano (Kale Computer, Apache-2.0) in process: one ONNX graph on ONNX Runtime, no server, no PyTorch.

Ten English voices designed with Maya-1 (eight American, two British), 24 kHz, 7.5 M parameters, distilled from a
Kokoro-82M fine-tune: the graph takes Kokoro's phoneme ids, a style row per input length from the voice's table
(``voices/<id>.bin``) and the voice's other inputs (``voices/<id>.extra.bin``, laid out as ``voices.json`` says).

The text front end is a Python port of Kale's own (``frontend/words/kaleTts.js`` and ``python/kale_tts.py`` at model
revision b6ce1887268c3667994fc3b0b2c82887c069915c, Apache-2.0): its technical terms; dizione's text rules (bolo carries
dizione in :mod:`bolo.vendor.dizione`; Kale uses its JavaScript port), with Kale's size suffixes, dotted names and
letter spelling; sentences with one-word ones joined to the next; then per word Kale's word list (CMUdict-derived,
with its checked overrides), dizione's homograph picker and Kale's sense rules, stems, letter spelling and its small
GRU guesser (``g2p.bin``) for unknown words; the IPA becomes Kokoro-native ids. The British voices use Kale's British
word list and guesser (``words/en-gb/``) and British dates; not ported from ``enGb.js``: its American-to-British
rewrite of homograph readings and of the letter-sound fallback, which keep their American form.
"""

from __future__ import annotations

import json
import re
import struct
import threading
import unicodedata
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import FloatArray, LoadOptions, TTSModel, Voice

SAMPLE_RATE = 24000
GAP_S = 0.12
_TOKEN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|[.,!?;:…—–]")
_PUNCT = {".": ".", ",": ",", "!": "!", "?": "?", ";": ";", ":": ":", "…": "…", "—": ",", "–": ","}
_VOWELS = frozenset("aeiouɑɐɒæɔəɘɚɛɜɝɞɨɪʊʌɤøœɶɯɵʉʏᵻ")
_SIBILANT = ("s", "z", "ʃ", "ʒ", "tʃ", "dʒ")
_VOICELESS = ("p", "t", "k", "f", "θ")
_SUFFIXES = (
    ("ly", "li"),
    ("ness", "nəs"),
    ("ful", "fəl"),
    ("less", "ləs"),
    ("er", "ɚ"),
    ("est", "ɪst"),
    ("ment", "mənt"),
)
_LETTER_SOUND = {
    "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ", "f": "f", "g": "ɡ", "h": "h", "i": "ɪ", "j": "dʒ", "k": "k", "l": "l",
    "m": "m", "n": "n", "o": "ɑː", "p": "p", "q": "k", "r": "ɹ", "s": "s", "t": "t", "u": "ʌ", "v": "v", "w": "w", "x": "ks",
    "y": "j", "z": "z",
}  # fmt: skip
_FOLD = {
    "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe", "ø": "o", "Ø": "O", "ß": "ss", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "ı": "i", "ʻ": "", "ʼ": "", "‘": "'", "`": "'",
}  # fmt: skip
_VOWEL = re.compile(r"eɪ|aɪ|aʊ|ɔɪ|oʊ|əʊ|ɑː|iː|uː|ɜː|ɔː|oː|[aeiouæɑɒɔəɚɛɜɪʊʌᵻɐ]")
_MAX_LEARNT = 4096
# dizione's markers around letters to be named (normalize.py, spoken.js)
_SPELL_RUN = re.compile("([A-Za-z ]+)")
_LETTER_OR_DIGIT = r"[^\W_]"
_SIZE = re.compile(r"(?<![^\W_])(?<!\.)(\d+(?:\.\d+)?)([MBKk])(?![^\W_])")
_TIMES = re.compile(r"(?<![^\W_])(?<!\.)(\d+(?:\.\d+)?)[x×](?![^\W_])")
_SECONDS = re.compile(r"(?<![^\W_])(?<!\.)(\d+(?: s|\.\d+ ?s))(?![^\W_]|['’])")
_DOTTED = re.compile(r"(?<![^\W_])(?<!\.)((?:[^\W_][\w-]*\.)+[^\W\d_][\w-]*)")
_ONE_WORD = re.compile(r"[^\W_]+(?:['’][^\W\d_]+)*")
_LIST_COMMA = re.compile(r",(\s+(?:and|or)\s)", re.IGNORECASE)
_CAMEL = re.compile(r"(?<!\w)[a-z]+(?:[A-Z][a-z0-9]*)+(?!\w)")
_LEAD_METAL_NEXT = frozenset(
    "pipe pipes piping paint paints poisoning poison free based acid oxide ore solder".split()  # noqa: SIM905
)
_LEAD_METAL_PREV = frozenset(("red", "white"))
_LEAD_MADE = frozenset(("made", "built", "cast", "lined", "pipes", "weights"))
_READ_HABIT = frozenset(("daily", "often", "nightly", "regularly"))
_PERIODS = frozenset(("day", "night", "morning", "evening", "week", "weekend"))
_PAST = frozenset(("yesterday", "ago", "last", "had", "has", "have", "was", "were", "already"))
_KN_PAIRS = {"eɪ": "A", "aɪ": "I", "aʊ": "W", "ɔɪ": "Y", "tʃ": "ʧ", "dʒ": "ʤ"}
_KN_VOWELS = frozenset("aeiouɑɐɒæəɘɚɛɜɝɞɪɨʊʉʌɔøœɤɯʏyAIWOYᵊᵻ")
_KN_END = frozenset(" .,!?;:")


# -- small helpers (kaleTts.js) ------------------------------------------------------------------------------------
def read_table(text: str, into: dict[str, str] | None = None) -> dict[str, str]:
    """``word<TAB>reading`` lines (``#`` comments) into a mapping; later files override earlier ones."""
    out = {} if into is None else into
    for raw in text.split("\n"):
        line = raw.removesuffix("\r")
        if not line or line.startswith("#"):
            continue
        tab = line.find("\t")
        if tab > 0:
            out[line[:tab]] = line[tab + 1 :]
    return out


def fold(text: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.category(ch) == "Mn":
            continue
        out.append(_FOLD.get(ch, ch))
    return unicodedata.normalize("NFC", "".join(out))


def _norm(token: str) -> str:
    return token.replace("’", "'")


def _unstress(ipa: str) -> str:
    return ipa.replace("ˈ", "").replace("ˌ", "")


def _secondary(ipa: str) -> str:
    return ipa.replace("ˈ", "ˌ")


def _bare(ipa: str) -> str:
    return re.sub("[ˈˌː]", "", ipa)


def _vowel_count(ipa: str) -> int:
    return len(_VOWEL.findall(ipa))


def _edit_distance(x: str, y: str) -> int:
    row = list(range(len(y) + 1))
    for i in range(1, len(x) + 1):
        prev, row[0] = row[0], i
        for j in range(1, len(y) + 1):
            here = row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, prev + (x[i - 1] != y[j - 1]))
            prev = here
    return row[len(y)]


def _first_sound(ipa: str) -> str | None:
    return next((ch for ch in ipa if ch not in "ˈˌ"), None)


def _stress_first_vowel(ipa: str) -> str:
    plain = _unstress(ipa)
    at = next((i for i, c in enumerate(plain) if c in _VOWELS), -1)
    return plain if at < 0 else plain[:at] + "ˈ" + plain[at:]


def _plural_suffix(ipa: str) -> str:
    if ipa.endswith(_SIBILANT):
        return "ᵻz"
    return "s" if ipa.endswith(_VOICELESS) else "z"


def _past_suffix(ipa: str) -> str:
    if ipa.endswith(("t", "d")):
        return "ᵻd"
    return "t" if ipa.endswith((*_VOICELESS, "s", "ʃ", "tʃ")) else "d"


def _constant(sound: str) -> Callable[[str], str]:
    """A suffix whose sound does not depend on the stem's."""
    return lambda _ipa: sound


def _sense_rule(low: str, words: list[str], i: int) -> str | None:
    """Kale's context rules ahead of dizione's picker: "lead" the metal or not, "read" as a habit."""

    def at(k: int) -> str:
        return words[k] if 0 <= k < len(words) else ""

    if low == "lead":
        metal = (
            at(i + 1) in _LEAD_METAL_NEXT
            or at(i - 1) in _LEAD_METAL_PREV
            or (at(i - 1) in ("of", "from") and at(i - 2) in _LEAD_MADE)
        )
        return "lˈɛd" if metal else "lˈiːd"
    if low == "read":
        habit = at(i + 1) in _READ_HABIT or (at(i + 1) in ("every", "each") and at(i + 2) in _PERIODS)
        if habit and not any(x in _PAST for x in words):
            return "ɹˈiːd"
    return None


def _letter_run(tokens: list[str], start: int) -> tuple[str, bool, int] | None:
    """Two or more one-letter capitals in a row ("G P U", "L L M's"): one spelled name."""
    letters, i = "", start
    while i < len(tokens) and re.fullmatch("[A-Z]", tokens[i]):
        letters += tokens[i]
        i += 1
    plural = i < len(tokens) and re.fullmatch("[A-Z]['’]s", tokens[i]) is not None
    if plural:
        letters += tokens[i][0]
        i += 1
    return (letters, plural, i) if len(letters) >= 2 else None


# -- the guesser for unknown words (g2p.bin: a GRU encoder-decoder with attention, float16 weights) -----------------
def _sigmoid(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return 1.0 / (1.0 + np.exp(-x))


class G2pModel:
    """Kale's ``g2p.bin`` (magic ``KG2P``, version 1): letters in, IPA out, greedy, at most ``3 n + 6`` symbols."""

    def __init__(self, data: bytes) -> None:
        if data[:4] != b"KG2P":
            raise ValueError("not a Kale g2p model")
        version, e, h, d, n_in, n_out, max_out = struct.unpack_from("<7i", data, 4)
        if version != 1:
            raise ValueError(f"Kale g2p model version {version}")
        offset = 32
        inputs = struct.unpack_from(f"<{n_in}i", data, offset)
        offset += 4 * n_in
        outputs = struct.unpack_from(f"<{n_out}i", data, offset)
        offset += 4 * n_out
        halves = np.frombuffer(data, dtype="<f2", offset=offset).astype(np.float64)
        position = 0

        def take(*shape: int) -> np.ndarray[Any, Any]:
            nonlocal position
            size = int(np.prod(shape))
            block = halves[position : position + size].reshape(shape)
            position += size
            return block

        def gru(cols: int, size: int) -> dict[str, np.ndarray[Any, Any]]:
            return {
                "wih": take(3 * size, cols),
                "whh": take(3 * size, size),
                "bih": take(3 * size),
                "bhh": take(3 * size),
            }

        self.emb_in = take(n_in, e)
        self.enc_fwd, self.enc_bwd = gru(e, h), gru(e, h)
        self.init_w, self.init_b = take(d, 2 * h), take(d)
        self.emb_out = take(n_out, e)
        self.dec = gru(e + 2 * h, d)
        self.att_w = take(2 * h, d)
        self.comb_w, self.comb_b = take(d, d + 2 * h), take(d)
        self.out_w, self.out_b = take(n_out, d), take(n_out)
        if position != len(halves):
            raise ValueError("Kale g2p model size")
        self.h, self.d, self.max_out = h, d, max_out
        self.outputs = outputs
        self.in_index = {code: i for i, code in enumerate(inputs) if i >= 1}

    @staticmethod
    def _cell(g: dict[str, np.ndarray[Any, Any]], x: np.ndarray[Any, Any], h: np.ndarray[Any, Any]) -> Any:
        size = len(h)
        gi = g["wih"] @ x + g["bih"]
        gh = g["whh"] @ h + g["bhh"]
        r = _sigmoid(gi[:size] + gh[:size])
        z = _sigmoid(gi[size : 2 * size] + gh[size : 2 * size])
        c = np.tanh(gi[2 * size :] + r * gh[2 * size :])
        return (1 - z) * c + z * h

    def phonemes(self, word: str) -> str | None:
        if not word or len(word) > 24:
            return None
        ids = [self.in_index.get(ord(ch)) for ch in word]
        if any(i is None for i in ids):
            return None
        n = len(ids)
        hf = np.zeros(self.h)
        fwd = []
        for t in range(n):
            hf = self._cell(self.enc_fwd, self.emb_in[ids[t]], hf)
            fwd.append(hf)
        hb = np.zeros(self.h)
        bwd: list[Any] = [None] * n
        for t in range(n - 1, -1, -1):
            hb = self._cell(self.enc_bwd, self.emb_in[ids[t]], hb)
            bwd[t] = hb
        enc = np.stack([np.concatenate((fwd[t], bwd[t])) for t in range(n)])
        state = np.tanh(self.init_w @ np.concatenate((hf, hb)) + self.init_b)
        ctx = np.zeros(2 * self.h)
        prev, out, ended = 1, [], False  # 1: start of word
        for _ in range(min(self.max_out, 3 * n + 6)):
            state = self._cell(self.dec, np.concatenate((self.emb_out[prev], ctx)), state)
            scores = enc @ (self.att_w @ state)
            weights = np.exp(scores - scores.max())
            ctx = (weights[:, None] * enc).sum(axis=0) / weights.sum()
            hidden = np.tanh(self.comb_w @ np.concatenate((state, ctx)) + self.comb_b)
            best = int(np.argmax(self.out_w @ hidden + self.out_b))
            if best == 2:  # end of word
                ended = True
                break
            if best < 3:
                return None
            out.append(chr(self.outputs[best]))
            prev = best
        said = "".join(out)
        return said if ended and any(c in _VOWELS for c in said) else None


# -- text -> words ---------------------------------------------------------------------------------------------------
def _terms(table: dict[str, str]) -> Callable[[str], str]:
    """Kale's technical terms, whole tokens, longest first; camelCase split; looked up again."""
    if not table:
        return lambda t: t
    forms = sorted(table, key=len, reverse=True)
    pattern = re.compile(
        "(?:" + "|".join(re.escape(f) if f.startswith(".") else r"(?<!\w)" + re.escape(f) for f in forms) + r")(?!\w)"
    )

    def look_up(text: str) -> str:
        def spoken(m: re.Match[str]) -> str:
            f = m.group(0)
            said = table.get(f, f)
            before = m.string[m.start() - 1] if m.start() > 0 else ""
            return f" {said}" if f.startswith(".") and before.isalnum() else said

        return pattern.sub(spoken, text)

    def split(text: str) -> str:
        return _CAMEL.sub(lambda m: re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", m.group(0)), text)

    return lambda t: look_up(split(look_up(t)))


def _size_suffixes(text: str) -> str:
    words = {"M": "million", "B": "billion", "K": "thousand", "k": "thousand"}
    text = _SIZE.sub(lambda m: f"{m.group(1)} {words[m.group(2)]}", text)
    text = _TIMES.sub(lambda m: f"{m.group(1)} times", text)
    return _SECONDS.sub(lambda m: re.sub(" ?s$", "", m.group(1)) + " sec", text)


def _dotted_names(text: str) -> str:
    def say(m: re.Match[str]) -> str:
        name = m.group(1)
        last = name.rsplit(".", 1)[1]
        if not last[:1].islower():
            return name
        parts = name.split(".")
        return name if all(len(p) == 1 for p in parts) else " dot ".join(parts)

    return _DOTTED.sub(say, text)


def _spoken_text(text: str, rules: Any, spell: Callable[[str], str]) -> str:
    """dizione's ``spokenText``: its rules, letter runs named as Kale names them, spaces and repeated marks tidied."""
    text = rules.rewrite(text)
    text = _SPELL_RUN.sub(lambda m: f" {spell(m.group(1).replace(' ', ''))} ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r" (?=[,.;:!?…])", "", text)
    text = re.sub(r"([,.;:!?…])(?=[,.;:!?…])", lambda m: "" if m.string[m.end()] == m.group(1) else m.group(1), text)
    return text.strip()


def _drop_list_comma(s: str) -> str:
    """The comma that closes a short list ("speed, melody, clear words, and how") goes."""

    def keep_or_drop(m: re.Match[str]) -> str:
        clause = re.split("[;:]", s[: m.start()])[-1]
        parts = clause.split(",")
        if len(parts) < 2:
            return m.group(0)
        items = [len(p.split()) for p in parts[1:]]
        return m.group(1) if all(1 <= n <= 2 for n in items) else m.group(0)

    return _LIST_COMMA.sub(keep_or_drop, s)


def sentences(text: str, normalize: Callable[[str], str]) -> list[str]:
    """Sentences, each one model call; a one-word sentence rides with the next one (the last with the one before)."""
    out: list[str] = []
    carry = ""
    for part in re.split(r"(?<=[.!?…])\s+|\n+", normalize(text)):
        s = _drop_list_comma(part.strip())
        if not s:
            continue
        joined = f"{carry} {s}" if carry else s
        if len(_ONE_WORD.findall(joined)) == 1:
            carry = joined
            continue
        out.append(joined)
        carry = ""
    if carry:
        if out:
            out[-1] += f" {carry}"
        else:
            out.append(carry)
    return out


# -- words -> IPA ----------------------------------------------------------------------------------------------------
class FrontEnd:
    """Kale's front end for one accent: its word list, overrides, terms and guesser in ``directory``."""

    def __init__(self, directory: Path, homographs: Any, british: bool = False) -> None:
        from bolo.vendor.dizione.normalize import Normalizer

        lex = read_table((directory / "lexicon.tsv").read_text(encoding="utf-8"))
        read_table((directory / "override.public.tsv").read_text(encoding="utf-8"), lex)
        self.lex = lex
        self.model = G2pModel((directory / "g2p.bin").read_bytes())
        self.homographs = homographs
        self.learnt: dict[str, str | None] = {}
        self._terms = _terms(read_table((directory / "terms.tsv").read_text(encoding="utf-8")))

        class British(Normalizer):  # type: ignore[misc]
            """Day-first numeric dates, every date "the <day> of <month>"."""

            def _date(self, month: Any, day: Any, year_text: Any = None, the: bool = False) -> Any:
                return super()._date(month, day, year_text, True)

            def _slash_date(self, match: Any) -> Any:
                self.day_first = True
                return super()._slash_date(match)

            def _short_date(self, match: Any) -> Any:
                self.day_first = True
                return super()._short_date(match)

        self._rules = (British if british else Normalizer)(lambda word: word in lex)

    def _spell_letters(self, letters: str) -> str:
        upper = letters.upper()
        named = len(upper) <= 5 and not (len(upper) >= 3 and letters.lower() in self.lex)
        return upper if named else " ".join(upper)

    def normalize(self, text: str) -> str:
        text = _size_suffixes(self._terms(text))
        text = re.sub(r"\bDr\.(?=\s*$)", "Doctor", text)
        return _dotted_names(_spoken_text(text, self._rules, self._spell_letters))

    def run(self, text: str) -> list[tuple[str, str]]:
        """``[(sentence, IPA)]``"""
        return [(s, self.phonemes(s)) for s in sentences(text, self.normalize)]

    def phonemes(self, sentence: str) -> str:
        tokens = _TOKEN.findall(fold(sentence))
        out: list[str] = []
        skip = False
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            mark = _PUNCT.get(tok)
            if mark is not None:
                if out:
                    out[-1] += mark
                i += 1
                continue
            if skip:
                skip = False
                i += 1
                continue
            run = _letter_run(tokens, i)
            if run is not None:
                letters, plural, end = run
                ipa = self.spell(letters)
                out.append(ipa + _plural_suffix(ipa) if plural else ipa)
                i = end
                continue
            nx = tokens[i + 1] if i + 1 < len(tokens) else None
            following = nx if nx is not None and nx not in _PUNCT else None
            if following is not None:
                pair = self.lex.get(_norm(tok).lower() + " " + _norm(following).lower())
                if pair is not None:
                    out.append(pair)
                    skip = True
                    i += 1
                    continue
            reading = self.homograph(tok, tokens, i)
            if reading is not None:
                out.append(reading)
                i += 1
                continue
            said = self.word(tok, following)
            if said:
                out.append(said)
            i += 1
        return " ".join(out)

    def homograph(self, token: str, tokens: list[str], index: int) -> str | None:
        w = _norm(token)
        low = w.lower()
        if len(w) > 1 and w == w.upper():
            return None
        ruled = _sense_rule(low, [_norm(t).lower() for t in tokens], index)
        if ruled is not None:
            return ruled
        if self.homographs is None or not self.homographs.covers(low):
            return None
        phones, plural = self.homographs.pronounce(low, [_norm(t) for t in tokens], index)
        sense = str(phones).replace(" ", "")
        entry = self.homographs.words[low[:-1] if plural else low]
        senses = [str(s["phones"]).replace(" ", "") for s in entry["senses"]]
        own = self.lex.get(low[:-1] if plural else low)
        ipa: str = sense
        if own is not None:
            nearest = senses[0]
            for s in senses[1:]:
                if _edit_distance(s, own) < _edit_distance(nearest, own):
                    nearest = s
            if nearest == sense:
                ipa = own
        return ipa + _plural_suffix(ipa) if plural else ipa

    def word(self, token: str, following: str | None) -> str | None:
        lex = self.lex
        w = _norm(token)
        low = w.lower()
        if len(w) > 1 and w == w.upper():
            if w in lex:
                return lex[w]
            if not (len(w) >= 3 and low in lex) and len(w) <= 5:
                return self.spell(w)
        if following is not None:
            after = self.lookup(_norm(following).lower(), None)
            variant = lex.get(low + "#v")
            if after is not None and variant is not None:
                first = _first_sound(after)
                if first is not None and first in _VOWELS:
                    return variant
        return self.lookup(low, w)

    def lookup(self, low: str, original: str | None) -> str | None:
        lex = self.lex
        if low in lex:
            return self.letters(low, lex[low])
        if original is None:
            return None
        for candidates, suffix in self.stems(low):
            for stem in candidates:
                base = lex.get(stem)
                if base:
                    return base + suffix(base)
        if len(low) == 1:
            return lex[low] if low in lex else _LETTER_SOUND.get(low, "")
        said = self.fallback(low)
        if said is not None:
            return self.joined(low, said) or said
        for i in range(3, len(low) - 2):
            a, b = lex.get(low[:i]), lex.get(low[i:])
            if a is not None and b is not None:
                return a + _secondary(b)
        return self.chunks(low)

    def joined(self, low: str, said: str) -> str | None:
        """A guessed word that is two listed words read as the guess plus one stray vowel: the two words instead."""
        if len(low) < 6:
            return None
        for i in range(3, len(low) - 2):
            a, b = self.lex.get(low[:i]), self.lex.get(low[i:])
            if a is None or b is None:
                continue
            two = a + _secondary(b)
            close = _vowel_count(said) > _vowel_count(two) and _edit_distance(_bare(said), _bare(two)) <= 1
            return two if close else None
        return None

    def fallback(self, low: str) -> str | None:
        if low in self.learnt:
            return self.learnt[low]
        said = self.model.phonemes(low)
        if len(self.learnt) >= _MAX_LEARNT:
            self.learnt.clear()
        self.learnt[low] = said
        return said

    def stems(self, w: str) -> list[tuple[list[str], Callable[[str], str]]]:
        out: list[tuple[list[str], Callable[[str], str]]] = []

        def doubled(b: str) -> list[str]:
            return [b[:-1]] if len(b) > 2 and b[-1] == b[-2] else []

        def none(_ipa: str) -> str:
            return ""

        if w.endswith("'s"):
            out.append(([w[:-2]], _plural_suffix))
        if w.endswith("s'"):
            out.append(([w[:-1]], none))
        if w.endswith("ies") and len(w) > 4:
            out.append(([w[:-3] + "y"], _plural_suffix))
        if w.endswith("es") and len(w) > 3:
            out.append(([w[:-2]], _plural_suffix))
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            out.append(([w[:-1]], _plural_suffix))
        if w.endswith("ied") and len(w) > 4:
            out.append(([w[:-3] + "y"], _past_suffix))
        if w.endswith("ed") and len(w) > 4:
            b = w[:-2]
            out.append(([b, b + "e", *doubled(b)], _past_suffix))
        if w.endswith("ing") and len(w) > 5:
            b = w[:-3]
            out.append(([b, b + "e", *doubled(b)], _constant("ɪŋ")))
        for suffix, sound in _SUFFIXES:
            if w.endswith(suffix) and len(w) > len(suffix) + 2:
                b = w[: -len(suffix)]
                out.append(([b, *([b[:-1] + "y"] if b.endswith("i") else []), *doubled(b)], _constant(sound)))
        return out

    def letters(self, low: str, ipa: str) -> str:
        """A listed reading that is only the word's letter names ("cpu") is spelled as "GPU" is."""
        if len(low) < 2 or len(low) > 5 or low == "ok" or not re.fullmatch("[a-z]+", low):
            return ipa
        names = "".join("eɪ" if c == "a" else _bare(self.lex.get(c, "")) for c in low)
        return self.spell(low) if _bare(ipa) == names else ipa

    def spell(self, word: str) -> str:
        parts = []
        for ch in word:
            c = ch.lower()
            part = "ˈeɪ" if c == "a" else "ˈaɪ" if c == "i" else self.lex.get(c, _LETTER_SOUND.get(c, ""))
            if part:
                parts.append(part)
        if not parts:
            return ""
        return " ".join([*(_secondary(p) for p in parts[:-1]), parts[-1]])

    def chunks(self, w: str) -> str:
        out, i = "", 0
        while i < len(w):
            for j in range(len(w), i + 2, -1):
                piece = self.lex.get(w[i:j])
                if piece:
                    out += _unstress(piece)
                    i = j
                    break
            else:
                out += _LETTER_SOUND.get(w[i], "")
                i += 1
        return _stress_first_vowel(out)


# -- IPA -> ids ------------------------------------------------------------------------------------------------------
def kokoro_native(chars: list[str], british: bool) -> list[str]:
    """IPA characters that have ids -> Kokoro-native ones (Kale's kn_ipa): American rules, or the British convention."""
    pairs = {**_KN_PAIRS, **({"əʊ": "Q"} if british else {"oʊ": "O"})}
    out: list[str] = []
    i = 0
    while i < len(chars):
        two = chars[i] + (chars[i + 1] if i + 1 < len(chars) else "")
        if two in pairs:
            out.append(pairs[two])
            i += 2
            continue
        c = chars[i]
        i += 1
        if british:
            out.append(c)
            continue
        if c == "ː":
            continue
        out += ["ə", "ɹ"] if c == "ɚ" else ["ɜ", "ɹ"] if c == "ɝ" else ["T"] if c == "ɾ" else [c]
    for k in range(1, len(out) - 1):
        if (
            out[k] == "ə"
            and out[k + 1] in "lnm"
            and (k + 2 == len(out) or out[k + 2] in _KN_END)
            and out[k - 1] not in _KN_VOWELS
            and out[k - 1] not in "ˈˌ"
            and not out[k - 1].isspace()
        ):
            out[k] = "ᵊ"
    return out


def ipa_ids(ipa: str, vocab: dict[str, int], british: bool, native: bool = True) -> list[int]:
    chars = [c for c in ipa if c in vocab]
    return [vocab[c] for c in (kokoro_native(chars, british) if native else chars) if c in vocab]


def read_homographs(file: Path) -> Any:
    """dizione's homograph picker over Kale's copy of its data (``frontend/dizione/homographs.data.js``, the built
    ``homographs.json`` as a JavaScript module) and dizione's hand rules."""
    from bolo.vendor.dizione.homographs import Homographs, read_rules

    text = file.read_text(encoding="utf-8")
    data = json.loads(text[text.index("{", text.index("export default")) : text.rindex("}") + 1])
    picker = Homographs.__new__(Homographs)
    picker.words = data["words"]
    picker.pooled = data["pooled"]
    picker.plural_bases = frozenset(data.get("plural_bases", ()))
    picker.rules = read_rules()
    return picker


def _gender(description: str) -> str:
    words = set(re.findall(r"[a-z]+", description.lower()))
    if "woman" in words:
        return "f"
    return "m" if words & {"man", "granddad"} else ""


class KaleOnnx(TTSModel):
    """kale-tts-1-nano on ONNX Runtime (see the module docstring)."""

    sample_rate = SAMPLE_RATE

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from bolo.engines.kokoro_vocab import int8_session

        meta = json.loads((path / "voices.json").read_text(encoding="utf-8"))
        graph = str(spec.get("graph", "int8"))
        self._session = int8_session(path, str(meta.get("graphs", {}).get(graph, graph)), opts.threads, opts.device)
        self._vocab: dict[str, int] = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))["model"][
            "vocab"
        ]
        self._native = meta.get("phonemes") == "kokoro-native"
        self._inputs = list(meta["voice_inputs"])
        self._meta = {str(v["id"]): v for v in meta["voices"]}
        front = path / "frontend"
        homographs = read_homographs(front / "dizione" / "homographs.data.js")
        words = front / "words"
        self._front = {"en-us": FrontEnd(words, homographs)}
        if (words / "en-gb" / "g2p.bin").exists():
            self._front["en-gb"] = FrontEnd(words / "en-gb", homographs, british=True)
        self.voices = tuple(
            Voice(
                str(v["id"]),
                f"{v.get('name', v['id'])} ({v.get('description', '')})" if v.get("description") else str(v["id"]),
                {"en-gb": "en-GB"}.get(str(v.get("accent", "en-us")), "en-US"),
                _gender(str(v.get("description", ""))),
                i,
            )
            for i, v in enumerate(meta["voices"])
        )
        self._names = {str(v.get("name", "")).lower(): str(v["id"]) for v in meta["voices"]}
        self._tables: dict[str, tuple[FloatArray, dict[str, FloatArray]]] = {}
        self._path = path
        self._lock = threading.Lock()

    def find_voice(self, voice: str | None) -> Voice | None:
        if voice and voice.strip().lower() in self._names:
            voice = self._names[voice.strip().lower()]
        return super().find_voice(voice)

    def _voice(self, vid: str) -> tuple[FloatArray, dict[str, FloatArray]]:
        if vid not in self._tables:
            style = np.fromfile(self._path / "voices" / f"{vid}.bin", dtype=np.float32).reshape(-1, 256)
            extra = np.fromfile(self._path / "voices" / f"{vid}.extra.bin", dtype=np.float32)
            feeds, offset = {}, 0
            for item in self._inputs:
                size = int(np.prod(item["dims"]))
                feeds[str(item["name"])] = extra[offset : offset + size].reshape(item["dims"])
                offset += size
            self._tables[vid] = (style, feeds)
        return self._tables[vid]

    def accent(self, voice: Voice) -> str:
        accent = str(self._meta[voice.id].get("accent", "en-us"))
        return accent if accent in self._front else "en-us"

    def sentence_ids(self, text: str, voice: Voice) -> list[list[int]]:
        accent = self.accent(voice)
        british = accent == "en-gb"
        out = []
        for _sentence, ipa in self._front[accent].run(text):
            ids = ipa_ids(ipa, self._vocab, british, self._native)
            if ids:
                out.append(ids[:510])
        return out

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[FloatArray]:
        chosen = self.find_voice(voice) or self.voices[0]
        style, feeds = self._voice(chosen.id)
        gap = np.zeros(int(GAP_S * self.sample_rate), dtype=np.float32)
        for i, ids in enumerate(self.sentence_ids(text, chosen)):
            if cancelled is not None and cancelled():
                return
            with self._lock:
                (audio,) = self._session.run(
                    None,
                    {
                        "input_ids": np.array([[0, *ids, 0]], dtype=np.int64),
                        "style": style[min(len(ids), len(style)) - 1][None],
                        "speed": np.array([speed], dtype=np.float32),
                        **feeds,
                    },
                )[:1]
            wave = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1, 1)
            yield wave if i == 0 else np.concatenate([gap, wave])
