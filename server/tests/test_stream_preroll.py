"""Live transcription with a streaming model: pre-roll, VAD-gated streams and per-utterance gain."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import bolo.streaming.session as session_module
from bolo.catalog import ModelSpec
from bolo.core import Bolo
from bolo.engines import fake
from bolo.engines.base import FloatArray, StreamResult, VadDetector, VadEvent, VadParams, Word
from bolo.engines.fake import FakeStreamingSTT
from bolo.streaming.session import (
    CHUNK_LEAD_S,
    DIGITAL_SILENCE_DB,
    SR,
    VAD_ROOM_DBFS,
    InputLevel,
    PreRoll,
    SessionConfig,
    STTSession,
    aligned_preroll,
    better_start,
    lift_onset,
    room_level,
    sounding,
    speech_gain,
    speech_lead,
    speech_level,
    speech_onset,
    voiced,
    without_stray_letter,
)
from bolo.streaming.vad import EnergyVad
from conftest import speechlike


def add(core: Bolo, model_id: str, **options: object) -> None:
    core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": model_id, "task": "stt", "engine": "fake", "kind": "fake-stt-streaming", "family": "Fake",
                "languages": ["en"], "license": "Apache-2.0", "builtin": True, "options": options, "streaming": True,
            }
        )
    )  # fmt: skip


async def run_stream(
    core: Bolo, model: str, audio: np.ndarray, *, chunk_ms: int = 40, pace: float = 1.0
) -> tuple[list[dict], STTSession]:
    messages: list[dict] = []

    async def send(msg: dict) -> None:
        messages.append(msg)

    session = STTSession(core, send, SessionConfig(model=model))
    await session.start()
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    step = 16 * chunk_ms * 2
    for off in range(0, len(pcm), step):
        session.feed(pcm[off : off + step])
        await asyncio.sleep(chunk_ms / 1000 * pace)
    await session.stop()
    return messages, session


def test_pre_roll_keeps_the_newest_audio_with_its_position() -> None:
    pre = PreRoll(0.5)
    a = np.arange(6000, dtype=np.float32)
    pre.push(a, 6000)
    assert (pre.start, pre.end) == (0, 6000) and len(pre.buf) == 6000
    pre.push(np.arange(6000, 10000, dtype=np.float32), 10000)  # 0.625 s pushed: only the newest 0.5 s are kept
    assert (pre.start, pre.end) == (2000, 10000)
    assert pre.buf[0] == 2000 and pre.buf[-1] == 9999
    pre.push(np.arange(10000, 20000, dtype=np.float32), 20000)  # one push longer than the buffer
    assert (pre.start, pre.end) == (12000, 20000) and pre.buf[0] == 12000
    audio, start = pre.take(since=15000)
    assert start == 15000 and audio[0] == 15000 and len(audio) == 5000
    assert len(pre.buf) == 0 and pre.end == 20000  # emptied, still knows where it is
    audio, start = pre.take(since=0)
    assert len(audio) == 0 and start == 20000
    # a request for more than is held gets what there is
    pre.push(np.ones(100, dtype=np.float32), 20100)
    audio, start = pre.take(since=0)
    assert start == 20000 and len(audio) == 100


def test_speech_gain_lifts_quiet_speech_only() -> None:
    loud = speechlike(1.0)  # 0.3 amplitude: about -13 dBFS
    assert speech_gain(loud, -20.0) == 1.0
    quiet = loud * 10 ** (-20 / 20)  # about -33 dBFS
    gain = speech_gain(quiet, -20.0)
    assert 3.0 < gain < 6.0
    # pauses do not count: the same speech with a long silence in it gets the same gain
    paused = np.concatenate([quiet[:8000], np.zeros(24000, dtype=np.float32), quiet[8000:]])
    assert speech_gain(paused, -20.0) == pytest.approx(gain, rel=0.05)
    assert speech_gain(quiet * 1e-4, -20.0) == pytest.approx(10 ** (24 / 20))  # capped
    assert speech_gain(np.zeros(16000, dtype=np.float32), -20.0) == 1.0
    assert speech_gain(np.zeros(100, dtype=np.float32), -20.0) == 1.0


def utterances(*parts: tuple[str, float]) -> np.ndarray:
    pieces = []
    for kind, seconds in parts:
        if kind == "speech":
            pieces.append(speechlike(seconds))
        else:  # a quiet room, not digital silence
            pieces.append(np.random.default_rng(len(pieces)).standard_normal(int(seconds * SR)) * 3e-4)
    return np.concatenate(pieces).astype(np.float32)


async def test_a_gated_stream_hears_speech_and_its_pre_roll_only(core: Bolo) -> None:
    add(core, "t-gated", step_delay_s=0.005, preroll_s=0.4)
    entry = await core.manager.load("t-gated")
    model = entry.model
    assert isinstance(model, FakeStreamingSTT)
    audio = utterances(("room", 3.0), ("speech", 1.6), ("room", 2.5), ("speech", 1.2), ("room", 1.5))
    messages, session = await run_stream(core, "t-gated", audio, pace=0.1)
    vad = [m for m in messages if m["type"] == "vad"]
    finals = [m for m in messages if m["type"] == "final"]
    assert [v["event"] for v in vad] == ["speech_start", "speech_end"] * 2
    assert len(finals) == 2
    # each utterance begins where the VAD heard it, and its words are timed from the pre-roll before the speech
    # (the energy VAD places its start a little before the sound)
    for final, start, sound in zip(finals, (vad[0], vad[2]), (3.0, 7.1)):
        assert final["t0"] == pytest.approx(start["t"], abs=0.01)
        assert final["words"][0]["t0"] == pytest.approx(sound - 0.4, abs=0.03)
    assert finals[0]["t1"] == pytest.approx(vad[1]["t"], abs=0.01)
    # the recogniser heard the speech and 0.4 s before each start, not the room in between: besides that, only the
    # silence the VAD waits for before it reports an end (0.5 s) and the tail padding that flushes the model (0.5 s)
    spoken = sum(v["t"] - u["t"] for u, v in zip(vad[0::2], vad[1::2]))
    assert spoken + 2 * (0.4 + 0.5) <= model.heard / SR <= spoken + 2 * (0.4 + 0.5 + 0.5) + 0.1
    assert model.heard < 0.7 * len(audio)
    assert session.trace is not None and session.trace.extra["utterances"] == 2


async def test_queued_audio_still_ends_one_utterance_where_the_next_begins(core: Bolo) -> None:
    """Audio sent much faster than real time arrives in large pieces with an end and a start in them."""
    add(core, "t-gated-q", step_delay_s=0.05, preroll_s=0.4)
    await core.manager.load("t-gated-q")
    audio = utterances(("room", 1.0), ("speech", 1.4), ("room", 1.2), ("speech", 1.4), ("room", 1.5))
    messages, _ = await run_stream(core, "t-gated-q", audio, chunk_ms=500, pace=0.01)
    vad = [m for m in messages if m["type"] == "vad"]
    finals = [m for m in messages if m["type"] == "final"]
    assert [v["event"] for v in vad] == ["speech_start", "speech_end"] * 2
    assert len(finals) == 2
    assert finals[0]["t1"] == pytest.approx(vad[1]["t"], abs=0.01) and finals[0]["t1"] < finals[1]["t0"]
    assert finals[1]["t0"] == pytest.approx(vad[2]["t"], abs=0.01)


async def test_quiet_speech_is_raised_for_the_recogniser(core: Bolo) -> None:
    add(core, "t-gain", step_delay_s=0.005, preroll_s=0.3, speech_dbfs=-20.0)
    entry = await core.manager.load("t-gain")
    model = entry.model
    assert isinstance(model, FakeStreamingSTT)
    quiet = utterances(("room", 1.0), ("speech", 1.5), ("room", 1.5)) * 10 ** (-20 / 20)
    messages, _ = await run_stream(core, "t-gain", quiet.astype(np.float32), pace=0.1)
    assert [m["type"] for m in messages].count("final") == 1
    peak_in = float(np.max(np.abs(quiet)))
    assert model.peak > 3 * peak_in  # what the VAD heard as about -33 dBFS reached the recogniser near -20
    assert model.peak <= 1.0


async def test_without_a_pre_roll_the_recogniser_hears_everything(core: Bolo) -> None:
    add(core, "t-open", step_delay_s=0.005)
    entry = await core.manager.load("t-open")
    model = entry.model
    assert isinstance(model, FakeStreamingSTT)
    audio = utterances(("room", 2.0), ("speech", 1.4), ("room", 1.5))
    messages, _ = await run_stream(core, "t-open", audio, pace=0.1)
    assert [m["type"] for m in messages].count("final") == 1
    assert model.heard >= len(audio)


# -- a quiet first word the VAD does not hear -------------------------------------------------------------------------

QUIET_WORD_AT = 2.0  # seconds: a quiet word, a breath, then the rest of the utterance at full level
SPEECH_AT = 2.5
VAD_LATE_S = 0.8  # where the VAD reports the start: on the loud part, as Silero did on microphone takes


def quiet_first_word() -> np.ndarray:
    """A room, a quiet first word (0.3 s, 20 dB below the rest), a 0.2 s breath, then 1.5 s of speech."""
    return utterances(
        ("room", QUIET_WORD_AT),
        ("speech", 0.3),
        ("room", SPEECH_AT - QUIET_WORD_AT - 0.3),
        ("speech", 1.5),
        ("room", 1.5),
    ) * np.concatenate([np.ones(int(QUIET_WORD_AT * SR)), np.full(int(0.3 * SR), 0.1), np.ones(int(3.2 * SR) + 1)])[
        : int(5.5 * SR)
    ].astype(np.float32)


class LateVad(VadDetector):
    """Reports speech starting ``VAD_LATE_S`` after it did, as Silero does when the first word is quiet: it starts
    counting at the first word loud enough for it."""

    def __init__(self, params: VadParams) -> None:
        self.inner = EnergyVad(params)
        self.pos = 0
        self.pending: VadEvent | None = None

    @property
    def in_speech(self) -> bool:
        return self.inner.in_speech and self.pending is None

    def accept(self, samples: FloatArray) -> list[VadEvent]:
        out: list[VadEvent] = []
        for ev in self.inner.accept(samples):
            if ev.kind == "speech_start":
                self.pending = VadEvent("speech_start", ev.sample + int(VAD_LATE_S * SR))
            else:
                out.append(ev)
        self.pos += len(samples)
        if self.pending is not None and self.pos >= self.pending.sample + int(0.3 * SR):
            out.insert(0, self.pending)
            self.pending = None
        return out

    def flush(self) -> list[VadEvent]:
        return self.inner.flush()


@pytest.fixture
def late_vad(core: Bolo, monkeypatch: pytest.MonkeyPatch) -> None:
    async def detector(params: VadParams | None = None, model: str | None = None, fallback: bool = False):  # type: ignore[no-untyped-def]
        return LateVad(params or VadParams()), "late"

    monkeypatch.setattr(core, "vad_detector", detector)


def test_speech_onset_finds_a_quiet_word_before_the_reported_start() -> None:
    audio = quiet_first_word()
    reported = int((QUIET_WORD_AT + VAD_LATE_S) * SR)
    assert speech_onset(audio, reported) / SR == pytest.approx(QUIET_WORD_AT, abs=0.03)
    # a start the VAD placed right is left there: the room before it is not speech
    assert speech_onset(audio, int(QUIET_WORD_AT * SR)) / SR == pytest.approx(QUIET_WORD_AT, abs=0.03)
    # too little held to tell the room from speech, or nothing at all: the reported start
    assert speech_onset(audio[:1000], 1000) == 1000
    assert speech_onset(audio, 0) == 0


def test_lift_onset_raises_the_quiet_word_to_the_level_of_the_rest() -> None:
    audio = quiet_first_word()
    a, b = int(QUIET_WORD_AT * SR), int((QUIET_WORD_AT + 0.3) * SR)
    rest = audio[int(SPEECH_AT * SR) : int((SPEECH_AT + 1.5) * SR)]
    lifted = lift_onset(audio, a, b, rest)
    assert speech_level(lifted[a:b]) == pytest.approx(speech_level(rest), rel=0.05)
    assert np.array_equal(lifted[:a], audio[:a]) and np.array_equal(lifted[b:], audio[b:])
    assert np.array_equal(lift_onset(audio, a, b, rest[:100]), audio)  # too little heard to go by: left alone


@pytest.mark.usefixtures("late_vad")
async def test_a_gated_stream_hears_the_quiet_first_word_the_vad_missed(core: Bolo) -> None:
    """Found on a Windows laptop: "t can you tell me" from the 20M Zipformer for "<word> can you tell me". The
    pre-roll was counted from where the VAD reported the start, which was on the second word."""
    add(core, "t-late", step_delay_s=0.005, preroll_s=0.4, speech_dbfs=-20.0)
    entry = await core.manager.load("t-late")
    model = entry.model
    assert isinstance(model, FakeStreamingSTT)
    messages, _ = await run_stream(core, "t-late", quiet_first_word(), pace=0.1)
    vad = [m for m in messages if m["type"] == "vad"]
    finals = [m for m in messages if m["type"] == "final"]
    assert vad[0]["event"] == "speech_start" and vad[0]["t"] > SPEECH_AT  # on the loud part
    assert len(finals) == 1
    # the recogniser's stream began the pre-roll before the quiet word, not before where the VAD heard speech
    assert finals[0]["words"][0]["t0"] == pytest.approx(QUIET_WORD_AT - 0.4, abs=0.03)


@pytest.mark.usefixtures("late_vad")
async def test_pseudo_streaming_decodes_the_quiet_first_word_the_vad_missed(core: Bolo) -> None:
    """Moonshine gave "Can you tell me" for "<word> can you tell me": the utterance began 0.25 s before where the
    VAD reported it, on the second word."""
    messages, _ = await run_stream(core, "fake-stt", quiet_first_word(), pace=0.1)
    finals = [m for m in messages if m["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["t0"] == pytest.approx(QUIET_WORD_AT - 0.25, abs=0.03)


# -- what is not a word the VAD missed (found with the samples on a Windows laptop, bolo 0.6.2) ------------------------


def noise(seconds: float, dbfs: float, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(seconds * SR)) * 10 ** (dbfs / 20)).astype(np.float32)


def test_voiced_tells_a_vowel_from_noise() -> None:
    assert voiced(speechlike(0.2) * 0.01).all()  # a 220 Hz tone, even at -53 dBFS
    assert not voiced(noise(0.2, -30.0)).any()  # a burst of noise, a click, a fricative's hiss
    assert not voiced(np.zeros(3200, dtype=np.float32)).any()


def test_a_burst_of_noise_before_the_speech_is_not_taken_for_a_quiet_word() -> None:
    """An 80 ms burst 0.3 s before the speech: the search crossed the pause to it and raised it 24 dB, which the 20M
    Zipformer wrote as "s good morning"."""
    room = noise(1.5, -70.0)
    room[int(1.12 * SR) : int(1.20 * SR)] += noise(0.08, -30.0, seed=1)
    audio = np.concatenate([room, speechlike(1.0)])
    at = int(1.5 * SR)
    assert speech_onset(audio, at) == at
    # a quiet word in the same place is still found
    word = room.copy()
    word[int(0.9 * SR) : int(1.2 * SR)] = speechlike(0.3) * 0.03
    assert speech_onset(np.concatenate([word, speechlike(1.0)]), at) / SR == pytest.approx(0.9, abs=0.03)


def test_the_hiss_that_starts_a_word_is_not_raised() -> None:
    """A fricative's hiss ("S" of "Speech") just before where the VAD placed the start was raised to the level of
    the vowel after it: "s speech servers" (Zipformer), "16. Speech servers" (Moonshine)."""
    audio = np.concatenate([np.zeros(int(0.5 * SR), dtype=np.float32), noise(0.08, -30.0), speechlike(1.0)])
    at = int(0.58 * SR)  # where the vowel begins
    assert speech_onset(audio, at) == at
    # placed 20 ms into the vowel, the hiss is found with it, and left as it is: the vowel sets the gain
    at = int(0.60 * SR)
    assert np.array_equal(lift_onset(audio, speech_onset(audio, at), at, audio[at:]), audio)


def test_a_recording_quiet_throughout_still_finds_its_start() -> None:
    """The whole clip 40 dB down after digital silence: the room was taken from the quiet speech itself (nothing
    else was louder than digital silence), so the start the VAD placed late, on a later word, stood."""
    quiet = 10 ** (-40 / 20)
    words = np.concatenate([speechlike(0.6), np.zeros(int(0.2 * SR), dtype=np.float32), speechlike(1.5)]) * quiet
    audio = np.concatenate([np.zeros(int(1.0 * SR), dtype=np.float32), words]).astype(np.float32)
    at = int(1.9 * SR)  # the VAD heard the second word only
    assert speech_onset(audio, at) / SR == pytest.approx(1.0, abs=0.03)
    # the same over a faint room
    roomy = audio + noise(len(audio) / SR, -80.0)
    assert speech_onset(roomy, at) / SR == pytest.approx(1.0, abs=0.03)


def test_a_stray_letter_at_the_start_is_dropped() -> None:
    words = [Word("o", 0.0, 0.1), Word("good", 0.1, 0.3), Word("morning", 0.3, 0.6)]
    assert without_stray_letter("o good morning", words) == ("good morning", words[1:])
    assert without_stray_letter("'s speech servers", [])[0] == "speech servers"
    assert without_stray_letter("t", [])[0] == ""
    for text in ("a good one", "i think so", "good morning", "ul my fellow", ""):
        assert without_stray_letter(text, words) == (text, words)


async def test_a_gated_stream_drops_a_stray_letter_before_the_first_word(
    core: Bolo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 20M Zipformer wrote "o good morning" for the en-short sample, from bolo 0.6.1 on: a fresh stream heard
    the start of the first word twice."""
    original = fake._FakeStream.result

    def with_stray(self: fake._FakeStream) -> StreamResult:
        r = original(self)
        return StreamResult(f"o {r.text}".strip(), [Word("o", 0.0, 0.05), *r.words]) if r.text else r

    monkeypatch.setattr(fake._FakeStream, "result", with_stray)
    add(core, "t-stray", step_delay_s=0.005, preroll_s=0.4)
    await core.manager.load("t-stray")
    audio = utterances(("room", 1.0), ("speech", 1.6), ("room", 1.5))
    messages, _ = await run_stream(core, "t-stray", audio, pace=0.1)
    finals = [m for m in messages if m["type"] == "final"]
    partials = [m for m in messages if m["type"] == "partial"]
    assert len(finals) == 1 and finals[0]["text"] and not finals[0]["text"].startswith("o ")
    assert finals[0]["words"][0]["w"] != "o"
    assert partials and not any(p["text"] == "o" or p["text"].startswith("o ") for p in partials)


def test_the_gain_is_judged_on_sound_not_on_the_silence_after_a_click() -> None:
    """A burst of noise in a near-silent file: the VAD reported a start at its end, and what it had heard by then was
    0.3 s of near silence and 40 ms of the first word. The 90th percentile of that made a 24 dB gain, and the speech
    that followed was clipped into "ular arrive" (20M Zipformer, en-short)."""
    hush = noise(1.3, -80.0)
    heard = np.concatenate([hush[: int(0.3 * SR)], speechlike(0.04) * 0.1])
    assert speech_gain(heard, -20.0) > 10.0  # what it was: over 20 dB
    room = room_level(hush[int(0.3 * SR) :])
    assert room == pytest.approx(-80.0, abs=1.0)
    assert len(sounding(heard, room)) == 0  # too little sound to judge
    assert speech_gain(sounding(heard, room), -20.0) == 1.0
    # speech that was heard is judged on its own frames, the pauses in it left out
    quiet = utterances(("speech", 0.6), ("room", 0.5), ("speech", 0.6)) * 10 ** (-20 / 20)
    room = room_level(utterances(("room", 1.0)) * 10 ** (-20 / 20))
    expected = speech_gain(quiet[: int(0.6 * SR)], -20.0)
    assert speech_gain(sounding(quiet, room), -20.0) == pytest.approx(expected, rel=0.1)


# -- a burst of noise before the speech (found on a Windows laptop with bolo 0.6.3) -----------------------------------

BURST_AT = 1.0  # an 80 ms burst of noise in a quiet room, the speech 0.3 s after it
BURST_SPEECH_AT = 1.38


def burst_then_speech() -> np.ndarray:
    audio = noise(4.5, -70.0)
    audio[int(BURST_AT * SR) : int((BURST_AT + 0.08) * SR)] += noise(0.08, -30.0, seed=1)
    audio[int(BURST_SPEECH_AT * SR) : int((BURST_SPEECH_AT + 1.5) * SR)] += speechlike(1.5)
    return audio


class ScriptedVad(VadDetector):
    """Reports the given events, each once it has heard ``lag_s`` past where it places it (Silero reported the start
    of en-short at the end of the burst before it, and the end of the speech half a second after it)."""

    def __init__(self, events: list[tuple[str, float]], lag_s: float = 0.1) -> None:
        self.events = [VadEvent(kind, int(t * SR)) for kind, t in events]
        self.lag = int(lag_s * SR)
        self.pos = 0
        self.speech = False
        self.heard: list[np.ndarray] = []

    @property
    def in_speech(self) -> bool:
        return self.speech

    def accept(self, samples: FloatArray) -> list[VadEvent]:
        self.heard.append(np.array(samples))
        self.pos += len(samples)
        out = []
        while self.events and self.pos >= self.events[0].sample + self.lag:
            ev = self.events.pop(0)
            self.speech = ev.kind == "speech_start"
            out.append(ev)
        return out

    def flush(self) -> list[VadEvent]:
        out, self.events = self.events, []
        return out


def scripted_vad(core: Bolo, monkeypatch: pytest.MonkeyPatch, events: list[tuple[str, float]]) -> list[ScriptedVad]:
    made: list[ScriptedVad] = []

    async def detector(params: VadParams | None = None, model: str | None = None, fallback: bool = False):  # type: ignore[no-untyped-def]
        made.append(ScriptedVad(events))
        return made[-1], "scripted"

    monkeypatch.setattr(core, "vad_detector", detector)
    return made


def test_speech_lead_finds_the_vowels_after_a_burst() -> None:
    audio = burst_then_speech()
    room = room_level(audio[: int(BURST_AT * SR)])
    burst_end = int((BURST_AT + 0.08) * SR)
    # from where the VAD placed the start (the end of the burst), the speech 0.3 s later
    assert speech_lead(audio, burst_end, room) / SR == pytest.approx(BURST_SPEECH_AT, abs=0.025)
    # the burst alone, before the speech has arrived: nothing yet
    assert speech_lead(audio[: int(1.3 * SR)], burst_end, room) is None
    # a hiss right before the first vowel belongs to the word: the sound before it with no pause counts
    hissed = burst_then_speech()
    h = int((BURST_SPEECH_AT - 0.08) * SR)
    hissed[h : h + int(0.08 * SR)] += noise(0.08, -30.0, seed=2)
    assert speech_lead(hissed, burst_end, room) / SR == pytest.approx(BURST_SPEECH_AT - 0.08, abs=0.025)


def test_aligned_preroll_puts_the_speech_early_in_an_encoder_chunk() -> None:
    n = int(0.5 * SR)
    assert aligned_preroll(n, 1.0, 0.0) == n  # no chunk known: as asked
    p = aligned_preroll(n, 1.0, 0.32)
    assert n <= p < n + int(0.32 * SR)
    # the speech comes CHUNK_LEAD_S into a chunk, counting the second the stream heard when it opened
    assert ((1.0 * SR + p) % int(0.32 * SR)) == pytest.approx(CHUNK_LEAD_S * SR, abs=1)
    for m in range(0, int(1.2 * SR), 777):
        assert ((1.0 * SR + aligned_preroll(m, 1.0, 0.32)) % int(0.32 * SR)) == pytest.approx(CHUNK_LEAD_S * SR, abs=1)


async def test_a_burst_the_vad_took_for_speech_does_not_move_the_stream_start(
    core: Bolo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """en-short with an 80 ms burst 0.3 s before it: Silero reported the start at the end of the burst, the fresh
    stream began half a second before that, and the 20M Zipformer gave "in could you tell me", "good morning" lost.
    The stream now opens once the speech's vowels arrive, its pre-roll counted from them."""
    scripted_vad(core, monkeypatch, [("speech_start", BURST_AT + 0.08), ("speech_end", BURST_SPEECH_AT + 2.0)])
    add(core, "t-burst", step_delay_s=0.005, preroll_s=0.4)
    await core.manager.load("t-burst")
    messages, _ = await run_stream(core, "t-burst", burst_then_speech(), chunk_ms=100, pace=0.1)
    vad = [m for m in messages if m["type"] == "vad"]
    finals = [m for m in messages if m["type"] == "final"]
    assert vad[0]["event"] == "speech_start" and vad[0]["t"] == pytest.approx(BURST_AT + 0.08, abs=0.01)
    assert len(finals) == 1
    # the stream began the pre-roll before the speech (its first word is timed from there), not before the burst
    assert finals[0]["words"][0]["t0"] == pytest.approx(BURST_SPEECH_AT - 0.4, abs=0.03)
    assert finals[0]["t0"] == pytest.approx(BURST_AT + 0.08, abs=0.01)  # the utterance starts where the VAD said


async def test_a_burst_alone_still_reaches_the_recogniser(core: Bolo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing voiced follows the start the VAD reported: after PENDING_S the stream opens on what was heard, as
    before (a whisper has no vowels either)."""
    audio = burst_then_speech()
    audio[int(BURST_SPEECH_AT * SR) :] = noise(len(audio) / SR - BURST_SPEECH_AT, -70.0, seed=3)
    scripted_vad(core, monkeypatch, [("speech_start", BURST_AT), ("speech_end", BURST_AT + 1.5)])
    add(core, "t-burst-alone", step_delay_s=0.005, preroll_s=0.4)
    entry = await core.manager.load("t-burst-alone")
    model = entry.model
    assert isinstance(model, FakeStreamingSTT)
    await run_stream(core, "t-burst-alone", audio, chunk_ms=100, pace=0.1)
    assert model.heard > 0
    # the start stays put if the VAD ends the speech first
    scripted_vad(core, monkeypatch, [("speech_start", BURST_AT), ("speech_end", BURST_AT + 0.2)])
    heard = model.heard
    await run_stream(core, "t-burst-alone", audio, chunk_ms=100, pace=0.1)
    assert model.heard > heard


# -- quiet input and the VAD -------------------------------------------------------------------------------------------


def test_quiet_input_is_raised_for_the_vad_only_when_the_room_is_quieter_than_a_microphone() -> None:
    level = InputLevel()
    level.push(np.zeros(SR, dtype=np.float32))  # a file's digital silence
    assert level.vad_gain() == pytest.approx(10 ** ((VAD_ROOM_DBFS - DIGITAL_SILENCE_DB) / 20))  # the room to -65
    level.push(speechlike(0.5) * 10 ** (-50 / 20))  # speech about 40 dB down: its loudest frames near -60 dBFS
    gain_db = 20 * np.log10(level.vad_gain())
    assert 20.0 < gain_db <= VAD_ROOM_DBFS - DIGITAL_SILENCE_DB
    # a microphone's own noise near -60 dBFS: nothing is raised
    mic = InputLevel()
    mic.push(noise(1.0, -60.0))
    mic.push(speechlike(0.5) * 10 ** (-50 / 20))
    assert mic.vad_gain() == 1.0
    # speech at an ordinary level: nothing is raised
    loud = InputLevel()
    loud.push(np.zeros(SR, dtype=np.float32))
    loud.push(speechlike(0.5))
    assert loud.vad_gain() == 1.0


async def test_the_vad_hears_quiet_speech_raised(core: Bolo, monkeypatch: pytest.MonkeyPatch) -> None:
    """en-short 40 dB down after digital silence: Silero reported no start at all. The VAD now hears it raised; the
    recogniser still gets the audio as received and its own gain."""
    made = scripted_vad(core, monkeypatch, [])
    quiet = np.concatenate([np.zeros(SR, dtype=np.float32), speechlike(1.0) * 10 ** (-45 / 20)])
    await run_stream(core, "fake-stt", quiet, chunk_ms=100, pace=0.1)
    heard = np.concatenate(made[0].heard)
    assert len(heard) == len(quiet)
    assert float(np.max(np.abs(heard[SR:]))) > 10 * float(np.max(np.abs(quiet[SR:])))


def test_better_start_takes_the_fuller_transcript() -> None:
    """The 20M Zipformer kept "Good morning" from one pre-roll and "Speech" from the other."""
    lost = StreamResult("in could you", [Word("in", 0.5, 0.6), Word("could", 0.7, 0.9), Word("you", 0.9, 1.0)])
    kept = StreamResult(
        "good morning could you",
        [Word("good", 0.3, 0.5), Word("morning", 0.5, 0.9), Word("could", 1.0, 1.2), Word("you", 1.2, 1.3)],
    )
    # the second stream began 0.3 s after the main one: its word times are moved onto the main stream's
    best = better_start(lost, kept, 0.3)
    assert best.text == "good morning could you" and best.words[0].t0 == pytest.approx(0.6)
    assert better_start(kept, lost, 0.3) is kept
    # the same start ("speed" for "speech" in one of them), or a stray letter that only looks like one more word:
    # the pre-roll as asked
    speed = StreamResult("speed servers are judged", [])
    assert better_start(speed, StreamResult("speech servers are judged", []), 0.0).text == "speech servers are judged"
    assert better_start(StreamResult("good morning", []), StreamResult("o good morning", []), 0.0).text == (
        "o good morning"  # its stray letter is dropped with the stream's
    )
    assert better_start(StreamResult("t good morning", []), StreamResult("good morning", []), 0.0).text == (
        "good morning"
    )
    # nothing in common: more words in all
    assert better_start(StreamResult("hello there", []), StreamResult("ular", []), 0.0).text == "hello there"


async def test_a_model_with_a_known_chunk_hears_each_utterance_from_two_pre_rolls(
    core: Bolo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The main stream's pre-roll puts the speech CHUNK_LEAD_S into an encoder chunk; the second stream's is the
    pre-roll asked for, counted as before from where the VAD reported the start (here the burst). Both are decoded,
    the same audio after the pre-roll; here the main stream's result is taken."""
    shifts: list[float] = []

    def take_main(main: StreamResult, alt: StreamResult, shift_s: float) -> StreamResult:
        shifts.append(shift_s)
        return main

    monkeypatch.setattr(session_module, "better_start", take_main)
    scripted_vad(core, monkeypatch, [("speech_start", BURST_AT + 0.08), ("speech_end", BURST_SPEECH_AT + 2.0)])
    add(core, "t-chunk", step_delay_s=0.005, preroll_s=0.4)
    entry = await core.manager.load("t-chunk")
    model = entry.model
    assert isinstance(model, FakeStreamingSTT)
    model.stream_chunk_s, model.stream_primed_s = 0.32, 1.0
    messages, _ = await run_stream(core, "t-chunk", burst_then_speech(), chunk_ms=100, pace=0.1)
    finals = [m for m in messages if m["type"] == "final"]
    assert len(finals) == 1
    t0 = finals[0]["words"][0]["t0"]  # where the main stream began
    assert BURST_SPEECH_AT - t0 >= 0.4
    assert (1.0 + BURST_SPEECH_AT - t0) % 0.32 == pytest.approx(CHUNK_LEAD_S, abs=0.025)
    # the second stream began 0.4 s before the reported start
    assert shifts and all(t0 + s == pytest.approx(BURST_AT + 0.08 - 0.4, abs=0.01) for s in shifts)
    # two streams heard the utterance: about twice what one did
    one = BURST_SPEECH_AT + 2.0 - (BURST_SPEECH_AT - 0.4)
    assert model.heard / SR > 2 * one
