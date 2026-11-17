"""Decoding options of speech to text: the repetition guard, Whisper options per request and per model,
silence removal tuning, chunk length and the detected language's probability."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import openai
import pytest

from bolo.engines.base import Segment, STTRequest, Word
from bolo.stt_options import OptionError, check, drop_repeated_segments, parse_vad_parameters, trim_repeats
from conftest import LiveServer, wav_bytes, wav_file


@pytest.fixture
def client(live: LiveServer) -> openai.OpenAI:
    return openai.OpenAI(base_url=live.url + "/v1", api_key="not-needed", max_retries=0)


def _post(live: LiveServer, seconds: float = 2.0, **fields: Any) -> httpx.Response:
    data = {"model": "fake-stt", **{k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in fields.items()}}
    return httpx.post(
        live.url + "/v1/audio/transcriptions",
        data=data,
        files={"file": ("a.wav", wav_bytes(seconds), "audio/wav")},
        timeout=30,
    )


def _seen(live: LiveServer) -> list[dict[str, Any]]:
    return live.core.manager.loaded["fake-stt"].model.seen_options  # type: ignore[attr-defined,no-any-return]


# ----------------------------------------------------------------------
# the repetition guard
# ----------------------------------------------------------------------


def test_trim_repeats_cuts_a_loop_to_one_copy() -> None:
    seg, removed = trim_repeats(Segment(0.0, 30.0, "So we went home and " + "the " * 40 + "end."))
    assert seg.text == "So we went home and the end." and removed == 39
    seg, removed = trim_repeats(Segment(0.0, 30.0, "thank you for watching " * 12))
    assert seg.text == "thank you for watching" and removed == 44
    seg, _ = trim_repeats(Segment(0.0, 30.0, "No, no, no, no, no, no, I said."))
    assert seg.text == "No, I said."
    seg, removed = trim_repeats(Segment(0.0, 30.0, "谢谢大家" * 10))  # written without spaces: characters
    assert seg.text == "谢谢大家" and removed == 36


def test_trim_repeats_leaves_ordinary_repetition_alone() -> None:
    for text in ("very very good", "no no no no", "it is what it is", "ha ha ha ha"):
        seg = Segment(0.0, 2.0, text)
        assert trim_repeats(seg) == (seg, 0)
    seg = Segment(0.0, 2.0, "the " * 40)
    assert trim_repeats(seg, max_repeats=0) == (seg, 0)  # 0: off


def test_trim_repeats_keeps_the_words_in_step_and_moves_the_end() -> None:
    words = [Word(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(["hello", "there", *["the"] * 30])]
    seg, removed = trim_repeats(Segment(0.0, 20.0, " ".join(w.w for w in words), words))
    assert seg.text == "hello there the" and removed == 29
    assert [w.w for w in seg.words] == ["hello", "there", "the"] and seg.end == pytest.approx(1.4)


def test_runs_of_identical_segments_keep_the_first() -> None:
    loop = [Segment(float(i), i + 1.0, "Thank you.") for i in range(8)]
    out = drop_repeated_segments([Segment(-1.0, 0.0, "Hi"), *loop, Segment(9.0, 10.0, "Bye")])
    assert [s.text for s in out] == ["Hi", "Thank you.", "Bye"]
    short = loop[:3]
    assert drop_repeated_segments(short) == short


def test_a_looping_engine_output_is_trimmed(client: openai.OpenAI) -> None:
    # the fake model loops ("the" forty times after its words) when the prompt is "loop"
    result = client.audio.transcriptions.create(model="fake-stt", file=wav_file(2.0), prompt="loop")
    assert result.text == "the quick brown fox jumps the"
    verbose = client.audio.transcriptions.create(
        model="fake-stt",
        file=wav_file(2.0),
        prompt="loop",
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
    )
    assert verbose.text == "the quick brown fox jumps the"
    assert [w.word for w in verbose.words or []] == verbose.text.split()
    off = client.audio.transcriptions.create(
        model="fake-stt", file=wav_file(2.0), prompt="loop", extra_body={"max_repeats": 0}
    )
    assert off.text.split()[5:] == ["the"] * 40


# ----------------------------------------------------------------------
# options per request
# ----------------------------------------------------------------------


def test_whisper_options_reach_the_engine_through_extra_body(client: openai.OpenAI, live: LiveServer) -> None:
    client.audio.transcriptions.create(
        model="fake-stt",
        file=wav_file(2.0),
        extra_body={
            "condition_on_previous_text": True,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "compression_ratio_threshold": 2.2,
            "temperature_increment_on_fallback": 0.2,
            "hallucination_silence_threshold": 2,
            "chunk_length": 20,
        },
    )
    assert _seen(live)[-1] == {
        "condition_on_previous_text": True,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "compression_ratio_threshold": 2.2,
        "temperature_increment_on_fallback": 0.2,
        "hallucination_silence_threshold": 2.0,
        "chunk_length": 20,
    }
    client.audio.transcriptions.create(model="fake-stt", file=wav_file(2.0))
    assert _seen(live)[-1] == {}  # nothing given, nothing pinned: the engine's own defaults


def _spy_vad(live: LiveServer) -> list[Any]:
    """Record the VadParams of every detector the server asks for."""
    seen: list[Any] = []
    original = live.core.vad_detector

    async def spy(params: Any = None, *args: Any, **kw: Any) -> Any:
        seen.append(params)
        return await original(params, *args, **kw)

    live.core.vad_detector = spy  # type: ignore[method-assign]
    return seen


def test_vad_parameters_as_an_object_or_json_turn_the_filter_on(client: openai.OpenAI, live: LiveServer) -> None:
    seen = _spy_vad(live)
    client.audio.transcriptions.create(
        model="fake-stt",
        file=wav_file(2.0),
        extra_body={"vad_parameters": {"threshold": 0.6, "min_silence_duration_ms": 300, "speech_pad_ms": 50}},
    )
    [params] = seen
    assert (params.threshold, params.min_silence_s, params.speech_pad_s) == pytest.approx((0.6, 0.3, 0.05))
    assert params.max_speech_s == 30.0  # not given: bolo's own value
    assert _post(live, vad_parameters='{"min_speech_duration_ms": 120, "max_speech_duration_s": 12}').status_code == 200
    assert (seen[-1].min_speech_s, seen[-1].max_speech_s) == pytest.approx((0.12, 12.0))
    assert _post(live).status_code == 200 and len(seen) == 2  # short audio, nothing asked: no silence removal
    off = _post(live, vad_filter=False, vad_parameters='{"threshold": 0.4}')
    assert off.status_code == 200 and len(seen) == 2
    assert off.headers["x-bolo-ignored-options"] == "vad_parameters"


def test_chunk_length_cuts_long_audio_into_shorter_pieces(live: LiveServer) -> None:
    r = _post(live, 24.0, chunk_length=10, response_format="verbose_json")
    assert r.status_code == 200
    segments = r.json()["segments"]
    assert len(segments) >= 3 and all(s["end"] - s["start"] <= 10.0 + 1e-6 for s in segments)
    whole = _post(live, 24.0, response_format="verbose_json")
    assert len(whole.json()["segments"]) == 1


@pytest.mark.parametrize(
    ("field", "value", "param"),
    [
        ("repetition_penalty", "0", "repetition_penalty"),
        ("repetition_penalty", "lots", "repetition_penalty"),
        ("no_repeat_ngram_size", "2.5", "no_repeat_ngram_size"),
        ("chunk_length", "31", "chunk_length"),
        ("compression_ratio_threshold", "-1", "compression_ratio_threshold"),
        ("temperature_increment_on_fallback", "1.5", "temperature_increment_on_fallback"),
        ("hallucination_silence_threshold", "0", "hallucination_silence_threshold"),
        ("condition_on_previous_text", "maybe", "condition_on_previous_text"),
        ("max_repeats", "-1", "max_repeats"),
        ("vad_parameters", '{"threshold": 2}', "vad_parameters.threshold"),
        ("vad_parameters", '{"speech_pad": 30}', "vad_parameters"),
        ("vad_parameters", "[1, 2]", "vad_parameters"),
        ("vad_parameters", "{oops", "vad_parameters"),
    ],
)
def test_bad_values_are_refused_with_a_clear_message(live: LiveServer, field: str, value: str, param: str) -> None:
    r = _post(live, 1.0, **{field: value})
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["param"] == param and error["type"] == "invalid_request_error"
    assert field in error["message"]


def test_bracketed_vad_parameters_from_the_sdk_are_read(live: LiveServer) -> None:
    seen = _spy_vad(live)
    r = httpx.post(
        live.url + "/v1/audio/transcriptions",
        data={"model": "fake-stt", "vad_parameters[min_silence_duration_ms]": "700"},
        files={"file": ("a.wav", wav_bytes(2.0), "audio/wav")},
        timeout=30,
    )
    assert r.status_code == 200 and seen[-1].min_silence_s == pytest.approx(0.7)
    bad = httpx.post(
        live.url + "/v1/audio/transcriptions",
        data={"model": "fake-stt", "vad_parameters[threshold]": "high"},
        files={"file": ("a.wav", wav_bytes(1.0), "audio/wav")},
        timeout=30,
    )
    assert bad.status_code == 400 and bad.json()["error"]["param"] == "vad_parameters.threshold"


def test_options_a_model_cannot_honour_are_dropped_and_named(live: LiveServer) -> None:
    assert _post(live).status_code == 200  # loads the model
    model = live.core.manager.loaded["fake-stt"].model
    model.decode_options = ("repetition_penalty",)  # type: ignore[misc]
    r = _post(live, condition_on_previous_text=True, repetition_penalty=1.1, chunk_length=20)
    assert r.status_code == 200 and r.json()["text"] == "the quick brown fox jumps"
    assert r.headers["x-bolo-ignored-options"] == "condition_on_previous_text"
    assert _seen(live)[-1] == {"repetition_penalty": 1.1}
    trace = live.core.traces.get(r.headers["x-request-id"])
    assert trace is not None and trace.extra["ignored_options"] == ["condition_on_previous_text"]
    assert "x-bolo-ignored-options" not in _post(live, repetition_penalty=1.1).headers
    neg = _post(live, vad_parameters='{"neg_threshold": 0.2}')
    assert neg.headers["x-bolo-ignored-options"] == "vad_parameters.neg_threshold"


def test_language_probability_in_verbose_json(client: openai.OpenAI) -> None:
    detected = client.audio.transcriptions.create(model="fake-stt", file=wav_file(1.0), response_format="verbose_json")
    assert detected.language_probability == 0.97  # type: ignore[attr-defined]
    told = client.audio.transcriptions.create(
        model="fake-stt", file=wav_file(1.0), response_format="verbose_json", language="en"
    )
    assert getattr(told, "language_probability", None) is None


# ----------------------------------------------------------------------
# per-model defaults
# ----------------------------------------------------------------------


def test_pinned_defaults_apply_and_the_request_wins(live: LiveServer) -> None:
    url = live.url + "/api/v1/models/fake-stt/settings"
    r = httpx.put(url, json={"repetition_penalty": 1.3, "max_repeats": 0, "vad_parameters": {"threshold": 0.4}})
    assert r.status_code == 200
    assert r.json()["settings"] == {"repetition_penalty": 1.3, "max_repeats": 0, "vad_parameters": {"threshold": 0.4}}
    looped = _post(live, prompt="loop")
    assert looped.json()["text"].split()[5:] == ["the"] * 40  # the guard is off for this model now
    assert _seen(live)[-1] == {"repetition_penalty": 1.3}
    assert _post(live, repetition_penalty=1.05).status_code == 200
    assert _seen(live)[-1] == {"repetition_penalty": 1.05}
    bad = httpx.put(url, json={"chunk_length": 0})
    assert bad.status_code == 400 and bad.json()["error"]["param"] == "chunk_length"
    bad = httpx.put(url, json={"vad_parameters": {"threshold": "x"}})
    assert bad.status_code == 400 and bad.json()["error"]["param"] == "vad_parameters.threshold"
    assert httpx.put(url, json={"repetition_penalty": None, "max_repeats": None}).status_code == 200
    assert _post(live, prompt="loop").json()["text"] == "the quick brown fox jumps the"


async def test_catalog_options_are_defaults_and_bad_ones_are_skipped(core: Any) -> None:
    spec = core.catalog.resolve("fake-stt")
    spec.options.update({"no_repeat_ngram_size": 4, "chunk_length": 99})
    try:
        assert core.decode_options(spec) == {"no_repeat_ngram_size": 4}
        core.store.set_model_settings(spec.id, {"no_repeat_ngram_size": 2})
        assert core.decode_options(spec) == {"no_repeat_ngram_size": 2}
        assert core.decode_options(spec, {"no_repeat_ngram_size": 0}) == {"no_repeat_ngram_size": 0}
    finally:
        spec.options.pop("no_repeat_ngram_size")
        spec.options.pop("chunk_length")


def test_check_and_parse_helpers() -> None:
    assert check({"repetition_penalty": "1.2", "other": "x"}) == {"repetition_penalty": 1.2, "other": "x"}
    assert check({"condition_on_previous_text": "yes"}) == {"condition_on_previous_text": True}
    assert parse_vad_parameters({"threshold": "0.3", "speech_pad_ms": 30.0}) == {"threshold": 0.3, "speech_pad_ms": 30}
    with pytest.raises(OptionError) as exc:
        check({"no_repeat_ngram_size": True})
    assert exc.value.name == "no_repeat_ngram_size"


# ----------------------------------------------------------------------
# the faster-whisper engine, without a model
# ----------------------------------------------------------------------


class _RecordingWhisper:
    """Stands in for faster_whisper.WhisperModel: records the keyword arguments of every transcribe call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.feature_extractor = SimpleNamespace(chunk_length=30)

    def transcribe(self, audio: Any, **kw: Any) -> tuple[list[Any], Any]:
        self.calls.append(kw)
        seg = SimpleNamespace(start=0.0, end=1.0, text=" hi", words=[], avg_logprob=-0.1, no_speech_prob=0.01)
        return [seg], SimpleNamespace(language="en", language_probability=0.91)


def _faster_whisper_stt() -> Any:
    pytest.importorskip("faster_whisper")
    from bolo.engines.faster_whisper import FasterWhisperSTT

    stt = object.__new__(FasterWhisperSTT)
    stt.model = _RecordingWhisper()
    stt.batched = _RecordingWhisper()
    stt.max_batch, stt.beam_size, stt.fixed_language, stt.decode_language = 8, 1, None, None
    return stt


def test_faster_whisper_passes_the_options_on() -> None:
    stt = _faster_whisper_stt()
    audio = np.zeros(16000, dtype=np.float32)
    [result] = stt.transcribe([STTRequest(audio)])
    kw = stt.model.calls[-1]
    assert kw["condition_on_previous_text"] is False and kw["temperature"] == 0.0 and kw["word_timestamps"] is False
    assert "repetition_penalty" not in kw and "chunk_length" not in kw
    assert result.text == "hi" and result.language_probability == 0.91
    options = {
        "condition_on_previous_text": True,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "compression_ratio_threshold": 2.0,
        "temperature_increment_on_fallback": 0.25,
        "hallucination_silence_threshold": 1.5,
        "chunk_length": 15,
    }
    stt.transcribe([STTRequest(audio, options=options)])
    kw = stt.model.calls[-1]
    assert kw["condition_on_previous_text"] is True and kw["repetition_penalty"] == 1.2
    assert kw["no_repeat_ngram_size"] == 3 and kw["compression_ratio_threshold"] == 2.0
    assert kw["temperature"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert kw["hallucination_silence_threshold"] == 1.5 and kw["chunk_length"] == 15
    assert kw["word_timestamps"] is True  # the hallucination check needs word timings
    # chunk_length went to a copy of the model: the shared one keeps its feature extractor
    assert len(stt.model.calls) == 2 and stt.model.feature_extractor.chunk_length == 30


def test_faster_whisper_batches_only_requests_with_the_same_batchable_options() -> None:
    stt = _faster_whisper_stt()
    audio = np.zeros(16000, dtype=np.float32)
    same = {"repetition_penalty": 1.1}
    requests = [
        STTRequest(audio, "en", options=same),
        STTRequest(audio, "en", options=same),
        STTRequest(audio, "en", options={"condition_on_previous_text": True}),
    ]
    results = stt.transcribe(requests)
    assert len(results) == 3
    assert len(stt.batched.calls) == 1 and stt.batched.calls[0]["repetition_penalty"] == 1.1
    assert len(stt.model.calls) == 1 and stt.model.calls[0]["condition_on_previous_text"] is True


def test_segment_and_word_times_stay_within_the_audio(live: LiveServer, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = live.core.catalog.resolve("fake-stt")
    assert spec is not None
    monkeypatch.setitem(spec.options, "overrun_s", 16.0)  # the model says its segment ends 16 s after the audio
    r = httpx.post(
        live.url + "/v1/audio/transcriptions",
        data={"model": "fake-stt", "response_format": "verbose_json", "timestamp_granularities[]": ["word", "segment"]},
        files={"file": ("a.wav", wav_bytes(6.0), "audio/wav")},
        timeout=30,
    )
    doc = r.json()
    assert doc["segments"][-1]["end"] == 6.0 and doc["words"][-1]["end"] == 6.0
    assert all(w["start"] <= w["end"] <= 6.0 for w in doc["words"])
    # cut into pieces: each piece's times stay within that piece
    long = _post(live, 24.0, chunk_length=10, response_format="verbose_json").json()
    assert len(long["segments"]) >= 3
    assert all(s["end"] - s["start"] <= 10.0 + 1e-6 and s["end"] <= 24.0 for s in long["segments"])
