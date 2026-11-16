from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from bolo.streaming.agreement import LocalAgreement, PrefixStability
from bolo.text.formats import to_srt, to_vtt
from bolo.text.normalize import english_number, hindi_number, normalize_for_tts, normalize_for_wer, word_error_rate
from bolo.text.phonemize import Phonemizer
from bolo.text.segment import IncrementalSegmenter, chunk_text, split_sentences


def test_split_latin() -> None:
    assert split_sentences("Hello world. How are you? Fine!") == ["Hello world.", "How are you?", "Fine!"]
    assert split_sentences("") == []
    assert split_sentences("no terminator") == ["no terminator"]


def test_split_keeps_abbreviations_decimals_and_initials() -> None:
    text = "Dr. Smith paid Rs. 3.50 to Mr. J. K. Rao on Jan 5. It was fine."
    assert split_sentences(text) == ["Dr. Smith paid Rs. 3.50 to Mr. J. K. Rao on Jan 5.", "It was fine."]
    assert split_sentences("Version 2.0 is out. Get it.") == ["Version 2.0 is out.", "Get it."]
    assert split_sentences('He said "stop." Then left.') == ['He said "stop."', "Then left."]


def test_split_indic_urdu_and_cjk() -> None:
    assert split_sentences("नमस्ते दुनिया। आप कैसे हैं? मैं ठीक हूँ॥") == ["नमस्ते दुनिया।", "आप कैसे हैं?", "मैं ठीक हूँ॥"]
    assert split_sentences("یہ ٹھیک ہے۔ کیا حال ہے؟") == ["یہ ٹھیک ہے۔", "کیا حال ہے؟"]
    assert split_sentences("你好。今天天气很好！") == ["你好。", "今天天气很好！"]
    # code-mixed text passes through untouched
    mixed = "मेरा order कल deliver होगा। Please confirm कर दीजिए।"
    assert split_sentences(mixed) == ["मेरा order कल deliver होगा।", "Please confirm कर दीजिए।"]
    assert " ".join(split_sentences(mixed)) == mixed


def test_chunk_text_first_chunk_is_short_and_nothing_is_lost() -> None:
    long = (
        "This is a long first sentence that goes on, and on, with many clauses, so that the first chunk "
        "needs to be cut at a soft boundary somewhere, right about here, to keep audio starting early. Then more."
    )
    chunks = chunk_text(long, first_max_chars=80)
    assert len(chunks[0]) <= 80
    assert " ".join(chunks).split() == long.split()
    assert all(len(c) <= 300 for c in chunks)
    assert chunk_text("Hi. Ok.") == ["Hi. Ok."]  # tiny sentences are merged
    assert chunk_text("x" * 700, max_chars=300, first_max_chars=300) == ["x" * 300, "x" * 300, "x" * 100]


def test_incremental_segmenter() -> None:
    s = IncrementalSegmenter()
    out = [s.push(t) for t in ["Hello wor", "ld. How are", " you? I am", " fine"]]
    assert out == [[], ["Hello world."], ["How are you?"], []]
    assert s.flush() == ["I am fine"]
    assert s.flush() == []
    s = IncrementalSegmenter()  # word-by-word tokens, as a language model sends them
    text = "You said: the quick brown fox. That is all I heard. Goodbye for now."
    words = text.split(" ")
    got = [x for i, w in enumerate(words) for x in s.push(w + ("" if i == len(words) - 1 else " "))] + s.flush()
    assert got == ["You said: the quick brown fox.", "That is all I heard.", "Goodbye for now."]
    s = IncrementalSegmenter(first_max_chars=40)
    released = s.push("word " * 40)  # no sentence end in sight: text is still released
    assert released and all(len(p) <= 40 for p in released)


def test_numbers() -> None:
    assert english_number(0) == "zero"
    assert english_number(21) == "twenty-one"
    assert english_number(105) == "one hundred and five"
    assert english_number(1234) == "one thousand two hundred and thirty-four"
    assert english_number(2_000_000) == "two million"
    assert hindi_number(0) == "शून्य"
    assert hindi_number(21) == "इक्कीस"
    assert hindi_number(99) == "निन्यानवे"
    assert hindi_number(100) == "एक सौ"
    assert hindi_number(250000) == "दो लाख पचास हज़ार"
    assert hindi_number(10_000_000) == "एक करोड़"


def test_normalize_for_tts() -> None:
    out = normalize_for_tts("**Hello** [world](http://x.y) 😀  I have 1,234 apples and 3.14 pies.")
    assert out == "Hello world I have one thousand two hundred and thirty-four apples and three point one four pies."
    assert normalize_for_tts("Call 0123 now") == "Call zero one two three now"
    assert normalize_for_tts("मेरे पास १२३ रुपये हैं", "hi") == "मेरे पास एक सौ तेईस रुपये हैं"
    assert normalize_for_tts("Das sind 12 Dinge", "de") == "Das sind 12 Dinge"  # other languages keep digits
    assert normalize_for_tts("a\r\n\r\n\r\n\r\nb") == "a\n\nb"
    assert normalize_for_tts("keep 5 as is", expand_numbers=False) == "keep 5 as is"
    mixed = "मेरा order कल deliver होगा"
    assert normalize_for_tts(mixed, "hi") == mixed


def test_wer() -> None:
    assert word_error_rate("Hello, World!", "hello world") == 0.0
    assert word_error_rate("the cat sat", "the cat") == 1 / 3
    assert word_error_rate("the cat sat", "the dog sat") == 1 / 3
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "x") == 1.0
    assert normalize_for_wer("नमस्ते, दुनिया।", "hi").split() == ["नमस्ते", "दुनिया"]


@dataclass
class Seg:
    start: float
    end: float
    text: str


def test_subtitles() -> None:
    segs = [Seg(0.0, 1.5, " Hello "), Seg(3661.25, 3662.0, "World")]
    assert to_srt(segs) == "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n2\n01:01:01,250 --> 01:01:02,000\nWorld\n"
    assert to_vtt(segs).startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello\n")


def test_local_agreement_never_retracts_committed_words() -> None:
    la = LocalAgreement(2)
    assert la.update("the quick") == ("the quick", 0)  # nothing confirmed yet
    text, stable = la.update("the quick brown")
    assert (text, stable) == ("the quick brown", len("the quick"))
    text, stable = la.update("the quick brown fox")
    assert text[:stable] == "the quick brown"
    # a later hypothesis that rewrites the start cannot take committed words back
    text, stable = la.update("a quick brown fox jumps")
    assert text.startswith("the quick brown")
    assert la.retractions == 1
    la.reset()
    assert la.update("fresh start") == ("fresh start", 0)


def test_local_agreement_ignores_case_and_punctuation() -> None:
    la = LocalAgreement(2)
    la.update("Hello, world")
    text, stable = la.update("hello world again")
    assert stable == len("hello world")
    assert text == "hello world again"


def test_prefix_stability() -> None:
    ps = PrefixStability()
    assert ps.update("hello") == 0
    assert ps.update("hello wor") == 5
    assert ps.update("hello world again") == 5  # "wor" was still changing
    assert ps.update("hello world again") == len("hello world again")


class EchoPhonemizer(Phonemizer):
    def clause(self, text: str, voice: str) -> str:
        return f"<{text.lower()}>"


def test_phonemizer_keeps_punctuation_and_sentences() -> None:
    ph = EchoPhonemizer()
    assert ph.phonemize("Hello world, this is a test. How are you? Dr. Smith is here", "en-us") == [
        "<hello world>, <this is a test>.",
        "<how are you>?",
        "<dr. smith is here>",
    ]
    assert ph.phonemize("नमस्ते। ठीक है?", "hi") == ["<नमस्ते>.", "<ठीक है>?"]
    assert ph.codepoints("Hi.", "en-us") == [[ord(c) for c in "<hi>."]]
    assert ph.phonemize("   ", "en-us") == []


def test_espeak_executable_route(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The licence-clean route: phonemes come from an `espeak-ng` program run as a separate process."""
    from bolo.text import phonemize
    from fakebin import fake_program

    # Stands in for the real program: echoes the voice and the text it was asked to phonemise.
    fake_program(
        tmp_path / "espeak-ng",
        "import sys\nargs = sys.argv[1:]\nif args[:3] != ['-q', '--ipa', '-v']:\n    sys.exit(2)\n"
        "sys.stdout.write(f' {args[3]}:{args[5]}\\n')\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("BOLO_ESPEAK", "executable")
    phonemize.reset_phonemizer()
    try:
        ph = phonemize.get_phonemizer()
        assert isinstance(ph, phonemize.EspeakExecutable) and ph.name == "espeak-ng (executable)"
        assert ph.phonemize("Hello there, friend. Bye!", "en-us") == ["en-us:Hello there, en-us:friend.", "en-us:Bye!"]
        monkeypatch.setenv("BOLO_ESPEAK", "off")
        phonemize.reset_phonemizer()
        assert phonemize.get_phonemizer() is None
        import pytest

        from bolo.errors import ComponentMissing

        with pytest.raises(ComponentMissing, match="espeak-ng"):
            phonemize.require_phonemizer()
    finally:
        phonemize.reset_phonemizer()


# -- training-time cleaning of character voices (the same front end as the in-browser voices) ---------------
_UI_FIXTURES = (
    Path(__file__).resolve().parents[2] / "ui" / "src" / "lib" / "browser-tts" / "fixtures" / "sherpa-tokens.json"
)


def test_multilingual_clean_matches_the_browser_front_end() -> None:
    from bolo.text.normalize import clean_for_voice, multilingual_clean

    # the cases of ui/src/lib/browser-tts/browser-tts.test.ts
    assert multilingual_clean('  Hello;  "World" (again) - yes: NO  ') == "hello, world again yes, no"
    assert multilingual_clean("[a] <b>") == "a b"
    assert multilingual_clean("क़") == "क़"  # precomposed nukta letter -> letter + nukta (NFC)
    assert multilingual_clean("क़") == "क़"
    assert multilingual_clean("é") == "é"
    no_danda = frozenset("abc .,")
    assert clean_for_voice("Abc। ABC!", "multilingual", no_danda) == "abc. abc."
    with_danda = frozenset("abc .,।!")
    assert clean_for_voice("Abc। ABC!", "multilingual", with_danda) == "abc। abc!"
    assert clean_for_voice("ÀB;C", "lowercase", None) == "àb;c"
    assert clean_for_voice("As Is;", None, None) == "As Is;"


@pytest.mark.skipif(not _UI_FIXTURES.is_file(), reason="needs the repository checkout (ui fixtures)")
def test_cleaned_text_gives_the_verified_token_ids() -> None:
    """The fixtures hold ids checked sample for sample against sherpa-onnx; the server's cleaning must give them."""
    import json

    from bolo.text.normalize import clean_for_voice

    cases = json.loads(_UI_FIXTURES.read_text(encoding="utf-8"))["cases"]
    assert cases
    for case in cases:
        symbols: dict[str, int] = case["symbols"]
        cleaned = clean_for_voice(case["text"], "multilingual", frozenset(symbols))
        ids = [symbols[ch] for ch in cleaned if ch in symbols]
        blank = case["blank_id"]
        with_blanks = [blank]
        for i in ids:
            with_blanks += [i, blank]
        assert with_blanks == case["ids"], case["text"]
