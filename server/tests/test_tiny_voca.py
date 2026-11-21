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


def test_both_description_layouts_give_the_voices_and_the_token_limit() -> None:
    """tiny-voca's config.json keeps the settings under ``model``; little-tts's onnx/model.json is flat."""
    from bolo.engines.tiny_voca import layout
    from bolo.vendor.dizione.symbols import SYMBOLS

    tiny = {"voices": ["F2", "M2"], "model": {"speakers": ["F2", "M2"], "max_tokens": 512}, "vocab": list(SYMBOLS)}
    little = {"voices": ["92", "6097"], "max_tokens": 400, "symbols": list(SYMBOLS)}
    assert layout(tiny) == (["F2", "M2"], 512)
    assert layout(little) == (["92", "6097"], 400)
    assert layout({"model": {"speakers": ["A"]}}) == (["A"], 512)


def test_a_graph_trained_on_other_phoneme_ids_is_refused() -> None:
    import pytest

    from bolo.engines.tiny_voca import layout

    with pytest.raises(ValueError, match="dizione"):
        layout({"voices": ["92"], "max_tokens": 512, "symbols": ["<pad>", "a", "b"]})
    with pytest.raises(ValueError, match="no voices"):
        layout({"max_tokens": 512})
