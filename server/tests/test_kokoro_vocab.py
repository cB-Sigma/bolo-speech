"""Small voices that read Kokoro's phoneme vocabulary (Paradee): the text side around the ONNX graph."""

from __future__ import annotations

from bolo.catalog import ModelSpec
from bolo.engines.kokoro_vocab import fit, ids, kokoro_voice


def test_long_phonemes_are_cut_at_spaces_to_the_graph_limit() -> None:
    assert fit("ab cd ef", 6) == ["ab cd", "ef"]
    assert fit("abcdefgh", 3) == ["abc", "def", "gh"]
    assert fit("  ") == []


def test_ids_have_a_pad_at_each_end_and_skip_unknown_symbols() -> None:
    assert ids("ab?", {"a": 5, "b": 7}) == [0, 5, 7, 0]


def test_a_kokoro_voice_name_gives_accent_and_gender() -> None:
    british = kokoro_voice("bf_isabella", 1)
    assert (british.language, british.gender, british.sid) == ("en-GB", "f", 1)
    assert british.name == "Isabella (British, female)"
    assert kokoro_voice("am_puck", 2).language == "en-US"


def test_a_voice_with_its_own_language_reads_in_it() -> None:
    """Paradee's British voices read British phonemes, its American ones American."""
    spec = ModelSpec.from_dict(
        {
            "id": "paradee-test",
            "task": "tts",
            "engine": "onnxruntime",
            "kind": "paradee_onnx",
            "family": "Paradee",
            "languages": ["en"],
            "license": "Apache-2.0",
            "requires": ["phonemizer"],
            "source": {"repo": "x/y", "revision": "0" * 40},
            "voices": [{"id": "af_heart", "language": "en-US"}, {"id": "bf_isabella", "language": "en-GB"}],
        }
    )
    assert spec.phoneme_target("bf_isabella") == ("kokoro", "en-GB")
    assert spec.phoneme_target("af_heart") == ("kokoro", "en-US")
    assert spec.phoneme_target() == ("kokoro", "en-US")
    assert spec.phoneme_target("unknown") == ("kokoro", "en-US")
