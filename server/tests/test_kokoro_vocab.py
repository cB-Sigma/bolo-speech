"""Small voices that read Kokoro's phoneme vocabulary (Paradee): the text side around the ONNX graph."""

from __future__ import annotations

from typing import Any

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


def test_kitten_reads_espeak_ties_and_folds_as_its_training_lists() -> None:
    from bolo.engines.kokoro_vocab import espeak_to_misaki, kitten_fold, kitten_text

    assert espeak_to_misaki("va^ɪs t^ʃa-o") == "vIs ʧao"
    assert kitten_fold("ɬ̪l ɫ", "ar", "v1") == "ɬl l"
    assert kitten_fold("'g ab", "de", "v1") == "'g ab"
    assert kitten_fold("'g ab", "de", "v2") == "ɡ ab"
    assert kitten_fold("bʰiːr.h ɡər.", "hi", "v2", "भीड़") == "bʰiːɽʰ ɡəɽ"
    assert kitten_text("यह है।", "hi", "v2") == "यह है." and kitten_text("यह है।", "hi", "v1") == "यह है।"


def test_kitten_tokens_and_trim() -> None:
    import numpy as np

    from bolo.engines.kokoro_vocab import kitten_tokens, kitten_trim

    assert kitten_tokens("a(b)", {"a": 5, "b": 6, ",": 3}) == [0, 5, 3, 6, 3, 10, 0]
    wave = np.ones(1000, dtype=np.float32)
    first = kitten_trim(wave, np.array([3, 4, 3]), keep_start=True)  # 100 samples per frame: the end token's 2 go
    later = kitten_trim(wave, np.array([3, 4, 3]), keep_start=False)  # and the start pause's 2
    assert len(first) == 800 and len(later) == 600
    assert first[0] == 0.0 and first[400] == 1.0


def test_espeak_ng_writes_ties_when_asked(monkeypatch: Any) -> None:
    import subprocess

    from bolo.text import phonemize

    calls: list[list[str]] = []

    def run(argv: list[str], **_kw: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "v^aɪs".encode(), b"")

    monkeypatch.setattr(phonemize.subprocess, "run", run)
    exe = phonemize.EspeakExecutable("espeak-ng")
    assert exe.phonemize("weiß.", "de", tie="^") == ["v^aɪs."]
    assert "--tie=^" in calls[-1]
    exe.clause("ja", "de")
    assert not any(a.startswith("--tie") for a in calls[-1])
