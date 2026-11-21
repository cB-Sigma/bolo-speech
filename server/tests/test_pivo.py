"""PiVo-1.4M in process: the text front end around its ONNX graphs (the graphs need the model's files)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bolo.engines.pivo import (
    SYMBOL_TO_ID,
    cardinal,
    deemphasis,
    normalize,
    ordinal,
    phase_features,
    read_lexicon,
    tokens,
)


def test_numbers_money_dates_and_initialisms_are_written_out() -> None:
    assert cardinal(1205) == "one thousand two hundred five"
    assert ordinal(21) == "twenty first"
    assert normalize("It costs $3.50.") == "It costs three dollars and fifty cents."
    assert normalize("On 2026-11-21.") == "On November twenty first, two thousand twenty six."
    assert normalize("The CPU is 42% busy.") == "The C P U is forty two percent busy."
    assert normalize("NASA, café & co.") == "nasa, cafe and co."


def test_what_the_model_cannot_say_is_left_out_instead_of_refused() -> None:
    assert normalize("1/2 of 🙂 a") == "one slash two of a"


def test_tokens_use_cmudict_and_spell_unknown_words(tmp_path: Path) -> None:
    lexicon_file = tmp_path / "cmudict.dict"
    lexicon_file.write_text("hi HH AY1\nhi(2) HH IY1\nb B IY1\nz Z IY1\n", encoding="utf-8")
    lexicon = read_lexicon(lexicon_file)
    assert lexicon["hi"] == ["HH", "AY1"]  # the first reading
    ids = tokens("hi, bz.", lexicon)
    names = [k for i in ids for k, v in SYMBOL_TO_ID.items() if v == i]
    assert names == ["<bos>", "HH", "AY1", ",", "<space>", "B", "IY1", "Z", "IY1", ".", "<eos>"]


def test_phase_features_and_deemphasis() -> None:
    phase = phase_features(np.array([2, 1]), 200)
    assert phase.shape == (3, 4)
    assert np.allclose(phase[:, 0], [-0.5, 0.5, 0.0])
    out, state = deemphasis(np.array([1.0, 0.0], dtype=np.float32), 0.0)
    assert np.allclose(out, [1.0, 0.85]) and abs(state - 0.85) < 1e-6
