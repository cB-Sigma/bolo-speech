"""kale-tts-1-nano's text front end, ported from Kale's JavaScript (the model's files are not needed here)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from bolo.engines.kale import (
    FrontEnd,
    G2pModel,
    _dotted_names,
    _drop_list_comma,
    _size_suffixes,
    fold,
    ipa_ids,
    kokoro_native,
    read_table,
    sentences,
)


def _tiny_g2p(n_in: int = 4, n_out: int = 5, e: int = 2, h: int = 2, d: int = 2) -> bytes:
    """A KG2P file of the right shape (weights zero but for the output bias, which says end of word)."""
    head = b"KG2P" + struct.pack("<7i", 1, e, h, d, n_in, n_out, 8)
    head += struct.pack(f"<{n_in}i", 0, ord("a"), ord("b"), ord("c"))
    head += struct.pack(f"<{n_out}i", 0, 0, 0, ord("ə"), ord("b"))
    sizes = [
        n_in * e, *(3 * h * e, 3 * h * h, 3 * h, 3 * h) * 2, d * 2 * h, d, n_out * e,
        3 * d * (e + 2 * h), 3 * d * d, 3 * d, 3 * d, 2 * h * d, d * (d + 2 * h), d, n_out * d,
    ]  # fmt: skip
    weights = np.zeros(sum(sizes) + n_out, dtype="<f2")
    weights[-n_out:] = [0, 0, 1, 0, 0]  # always "end of word"
    return head + weights.tobytes()


def _front_end(tmp_path: Path, lexicon: str) -> FrontEnd:
    (tmp_path / "lexicon.tsv").write_text(lexicon, encoding="utf-8")
    (tmp_path / "override.public.tsv").write_text("# none\n", encoding="utf-8")
    (tmp_path / "terms.tsv").write_text("Node.js\tNode dot J S\n", encoding="utf-8")
    (tmp_path / "g2p.bin").write_bytes(_tiny_g2p())
    return FrontEnd(tmp_path, None)


def test_tables_later_files_win_and_comments_are_skipped() -> None:
    table = read_table("# c\na\tx\nb\ty\n")
    assert read_table("a\tz\r\n", table) == {"a": "z", "b": "y"}


def test_text_steps_of_kale() -> None:
    assert fold("Søren‘s café") == "Soren's cafe"  # ’ stays: the word step reads it as '
    assert fold("it’s") == "it’s"
    assert _size_suffixes("Kokoro-82M is 1.1x faster in 4.8 s") == "Kokoro-82 million is 1.1 times faster in 4.8 sec"
    assert _dotted_names("see api.kale.computer or U.S. law") == "see api dot kale dot computer or U.S. law"
    assert _drop_list_comma("speed, melody, clear words, and how") == "speed, melody, clear words and how"
    assert _drop_list_comma("we finished the work, and then went home") == "we finished the work, and then went home"
    assert sentences("Four. Thursday. Go now. Yes.", lambda t: t) == ["Four. Thursday.", "Go now. Yes."]


def test_words_spelling_stems_and_the_guesser(tmp_path: Path) -> None:
    lexicon = "jump\tdʒˈʌmp\nc\tsˈiː\np\tpˈiː\nu\tjˈuː\ncat\tkˈæt\n"
    fe = _front_end(tmp_path, lexicon)
    assert fe.phonemes("Jumped, cats.") == "dʒˈʌmpt, kˈæts."
    assert fe.phonemes("CPU") == "sˌiː pˌiː jˈuː"
    assert fe.model.phonemes("ab") is None  # the toy guesser ends at once: nothing said, so no reading
    assert fe.normalize("Node.js") == "Node dot J S"


def test_kokoro_native_ids() -> None:
    assert "".join(kokoro_native(list("oʊ ɾɚ bʌtən."), british=False)) == "O Təɹ bʌtᵊn."
    assert "".join(kokoro_native(list("əʊ"), british=True)) == "Q"
    assert ipa_ids("ab?", {"a": 1, "b": 2}, british=False) == [1, 2]


def test_a_model_file_of_another_kind_is_refused() -> None:
    import pytest

    with pytest.raises(ValueError, match="not a Kale g2p model"):
        G2pModel(b"XXXX" + bytes(40))
