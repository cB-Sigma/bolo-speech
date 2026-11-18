"""tiny-voca in process: the text handling around the ONNX graph (the graph itself needs the model's files)."""

from __future__ import annotations

from bolo.engines.tiny_voca import finished, shorter


def test_a_text_without_a_final_mark_gets_a_full_stop() -> None:
    assert finished("Hello there") == "Hello there."
    assert finished("Hello there!") == "Hello there!"
    assert finished('She said "go!"') == 'She said "go!"'
    assert finished("see you :)") == "see you :)."
    assert finished("wait,") == "wait,"


def test_too_long_an_utterance_is_cut_after_commas_then_between_words() -> None:
    assert shorter("one, two; three") == ["one,", "two;", "three"]
    assert shorter("a b c d e") == ["a b c", "d e"]
    assert shorter("word") == []


def test_dizione_is_carried_and_importable() -> None:
    """The phonemiser's tokeniser (its data, the dictionary, comes with the model): phones to the model's ids."""
    from bolo.vendor.dizione.symbols import tokenize

    phones = tokenize([(False, [[("h", 0), ("ə", 0), ("l", 0), ("oʊ", 1)]]), (True, ".")])
    assert phones.ids[0] == 1 and phones.ids[-1] == 2  # <bos> ... <eos>
    assert len(phones.ids) == len(phones.stress) and 1 in phones.stress
