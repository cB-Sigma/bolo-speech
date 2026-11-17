"""A deterministic engine with programmable delays.

It exists for two reasons: the test-suite uses it to check the scheduler,
the lifecycle manager and (above all) the latency accounting against known
timings; and ``bolo serve --fake-engine`` gives front-end developers a
complete API to build against without downloading a model.

Recognition is "decoded" from the audio length: a fixed vocabulary is
emitted at ``words_per_second``. Synthesis produces a tone whose length is
proportional to the text. Neither is meant to resemble speech.
"""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from bolo import __version__
from bolo.engines.base import (
    DenoiseModel,
    DiarizationModel,
    DiarizationResult,
    Engine,
    EngineStatus,
    FloatArray,
    LoadOptions,
    Model,
    Segment,
    SeparationModel,
    SpeakerActivity,
    SpeakerRequest,
    SpeakerTurn,
    StreamResult,
    STTModel,
    STTRequest,
    STTResult,
    STTStream,
    SynthesisExtras,
    TTSModel,
    TurnModel,
    VadDetector,
    VADModel,
    VadParams,
    Voice,
    Word,
)
from bolo.engines.blend import looks_like_blend, resolve_blend

VOCAB = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "the",
    "lazy",
    "dog",
    "while",
    "seven",
    "bright",
    "stars",
    "drift",
    "across",
    "a",
    "quiet",
    "northern",
    "sky",
    "and",
    "distant",
    "bells",
    "keep",
    "time",
    "for",
    "travellers",
    "on",
    "the",
    "old",
    "road",
]


def fake_words(seconds: float, words_per_second: float = 2.5) -> list[Word]:
    n = int(seconds * words_per_second + 1e-6)
    step = 1.0 / words_per_second
    return [Word(VOCAB[i % len(VOCAB)], i * step, (i + 1) * step) for i in range(n)]


class FakeSTT(STTModel):
    """Batch model. Decode time = ``base_delay_s`` + ``delay_per_audio_s`` x longest item."""

    def __init__(self, opts: dict[str, Any]) -> None:
        self.base_delay_s = float(opts.get("base_delay_s", 0.02))
        self.delay_per_audio_s = float(opts.get("delay_per_audio_s", 0.01))
        self.load_delay_s = float(opts.get("load_delay_s", 0.0))
        self.max_batch = int(opts.get("max_batch", 8))
        self.max_chunk_s = float(opts.get("max_chunk_s", 30.0))
        self.words_per_second = float(opts.get("words_per_second", 2.5))
        self.fail_on = opts.get("fail_on")
        #: a request with this prompt gets Whisper's loop: its words, then "the" forty times ("*": every request)
        self.loop_on = opts.get("loop_on", "loop")
        self.decode_options = tuple(opts.get("decode_options", self._all_options()))
        self.seen_options: list[dict[str, Any]] = []  # STTRequest.options of every request
        #: honours STTRequest.abort (option ``abortable``); ``aborted`` counts the decodes it stopped part-way
        self.abortable = bool(opts.get("abortable", True))
        #: seconds past the end of the audio the fake places its segment's end and last word (a model without word
        #: timings that overruns, as whisper.cpp does in a loop)
        self.overrun_s = float(opts.get("overrun_s", 0.0))
        self.aborted = 0
        self.languages = ("en", "hi")
        self.supports_word_timestamps = True
        self.supports_translate = True
        self.calls: list[int] = []  # batch size of every transcribe() call
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()
        if self.load_delay_s:
            time.sleep(self.load_delay_s)

    @staticmethod
    def _all_options() -> tuple[str, ...]:
        from bolo.stt_options import ENGINE_OPTIONS

        return tuple(ENGINE_OPTIONS)

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        with self._lock:
            self.calls.append(len(requests))
            self.seen_options.extend(dict(r.options) for r in requests)
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            longest = max((len(r.samples) for r in requests), default=0) / self.sample_rate
            delay = self.base_delay_s + self.delay_per_audio_s * longest
            if self.abortable:  # stops early when every request of the batch was given up
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline and not all(r.abort.is_set() for r in requests):
                    time.sleep(0.01)
                if all(r.abort.is_set() for r in requests):
                    self.aborted += 1
            else:
                time.sleep(delay)
            out: list[STTResult] = []
            for r in requests:
                if self.fail_on is not None and r.prompt == self.fail_on:
                    raise RuntimeError("fake engine failure requested")
                seconds = len(r.samples) / self.sample_rate
                words = fake_words(seconds, self.words_per_second)
                if self.loop_on and self.loop_on in (r.prompt, "*"):
                    t = words[-1].t1 if words else 0.0
                    step = max(seconds - t, 0.1) / 40
                    words = words + [Word("the", t + i * step, t + (i + 1) * step) for i in range(40)]
                if self.overrun_s and words:
                    last = words[-1]
                    words = [*words[:-1], Word(last.w, last.t0, last.t1 + self.overrun_s)]
                text = " ".join(w.w for w in words)
                seg = Segment(0.0, seconds + self.overrun_s, text, words if r.word_timestamps else [])
                out.append(
                    STTResult(
                        text,
                        r.language or "en",
                        [seg] if text else [],
                        words if r.word_timestamps else [],
                        language_probability=None if r.language else 0.97,
                    )
                )
            return out
        finally:
            with self._lock:
                self.concurrent -= 1


class FakeSpeakerSTT(FakeSTT):
    """Speaker-conditioned test model: for each speaker, one segment per stretch where that speaker's activity is
    above one half, with words in proportion to its length. Remembers every request."""

    speaker_conditioned = True

    def __init__(self, opts: dict[str, Any]) -> None:
        super().__init__(opts)
        self.requests: list[SpeakerRequest] = []

    def transcribe_speakers(self, request: SpeakerRequest) -> list[STTResult]:
        self.requests.append(request)
        time.sleep(self.base_delay_s)
        probs, step = request.activity.probs, request.activity.frame_s
        out: list[STTResult] = []
        for k in range(request.activity.num_speakers):
            on = np.concatenate([[0], (probs[:, k] > 0.5).astype(np.int8), [0]])
            edges = np.flatnonzero(np.diff(on))
            segments: list[Segment] = []
            for a, b in zip(edges[0::2], edges[1::2]):
                words = fake_words((b - a) * step, self.words_per_second)
                if words:
                    text = f"speaker{k} " + " ".join(w.w for w in words)
                    segments.append(Segment(a * step, b * step, text))
            out.append(STTResult(" ".join(s.text for s in segments), request.language or "en", segments))
        return out


class _FakeStream(STTStream):
    """Counts voiced samples; "decodes" words in proportion to them."""

    def __init__(self, model: FakeStreamingSTT) -> None:
        self.model = model
        self.voiced_received = 0
        self.voiced_decoded = 0
        self.base = 0
        self.trailing_silence = 0
        self.finished = False

    def accept(self, samples: FloatArray) -> None:
        self.model.heard += len(samples)
        self.model.peak = max(self.model.peak, float(np.max(np.abs(samples), initial=0.0)))
        quiet = np.abs(samples) < 1e-3
        self.voiced_received += int((~quiet).sum())
        if quiet.all():
            self.trailing_silence += len(samples)
        else:
            self.trailing_silence = int(np.argmax(~quiet[::-1]))

    def finish(self) -> None:
        self.finished = True

    def result(self) -> StreamResult:
        seconds = (self.voiced_decoded - self.base) / self.model.sample_rate
        words = fake_words(seconds, self.model.words_per_second)
        if self.model.loop_word:  # stuck on one word, utterance after utterance
            words = [Word(self.model.loop_word, w.t0, w.t1) for w in words]
        return StreamResult(" ".join(w.w for w in words), words)

    def is_endpoint(self) -> bool:
        spoke = self.voiced_decoded > self.base
        return spoke and self.trailing_silence >= self.model.endpoint_silence_s * self.model.sample_rate

    def reset(self) -> None:
        self.base = self.voiced_decoded
        self.trailing_silence = 0


class FakeStreamingSTT(FakeSTT):
    """Streaming model: each ``decode_streams`` call costs ``step_delay_s`` regardless of batch size."""

    streaming = True

    def __init__(self, opts: dict[str, Any]) -> None:
        super().__init__(opts)
        self.step_delay_s = float(opts.get("step_delay_s", 0.01))
        self.endpoint_silence_s = float(opts.get("endpoint_silence_s", 0.5))
        self.stream_preroll_s = float(opts.get("preroll_s", 0.0))
        level = opts.get("speech_dbfs")
        self.stream_speech_dbfs = None if level is None else float(level)
        self.stream_calls: list[int] = []
        #: every word the streams "hear" is this one (a recogniser stuck in a loop)
        self.loop_word = str(opts.get("loop_word") or "")
        self.heard = 0  # samples the streams were given
        self.peak = 0.0  # loudest of them

    def create_stream(self, language: str | None = None) -> STTStream:
        return _FakeStream(self)

    def decode_streams(self, streams: list[STTStream]) -> None:
        with self._lock:
            self.stream_calls.append(len(streams))
        time.sleep(self.step_delay_s)
        for s in streams:
            assert isinstance(s, _FakeStream)
            s.voiced_decoded = s.voiced_received


class FakeTTS(TTSModel):
    """Synthesis time = ``first_delay_s`` + ``delay_per_char_s`` x characters, per chunk."""

    sample_rate = 24000
    voices: tuple[Voice, ...] = (
        Voice("alpha", "Alpha", "en-US", "f", 0),
        Voice("beta", "Beta", "en-GB", "m", 1),
        Voice("gamma", "Gamma", "hi-IN", "f", 2),
    )

    def __init__(self, opts: dict[str, Any]) -> None:
        self.first_delay_s = float(opts.get("first_delay_s", 0.02))
        self.delay_per_char_s = float(opts.get("delay_per_char_s", 0.0005))
        self.seconds_per_char = float(opts.get("seconds_per_char", 0.06))
        self.pieces = int(opts.get("pieces", 2))
        self.calls: list[str] = []
        #: option ``cloning``: "clones" a reference by speaking at its dominant pitch
        self.supports_cloning = bool(opts.get("cloning"))
        self.accepts_extras = self.supports_cloning
        self.request_options = ("pitch",) if self.supports_cloning else ()
        #: option ``declared_voices``: the catalog entry's voices instead of the three above
        if opts.get("declared_voices"):
            self.voices = tuple(
                Voice(str(v["id"]), str(v.get("name", "")), str(v.get("language", "")), str(v.get("gender", "")), i)
                for i, v in enumerate(opts.get("voices") or [])
            )
        #: option ``blending``: voice blends (bolo.engines.blend) sound at the weighted mean of their voices' pitches
        self.supports_blending = bool(opts.get("blending"))
        self.extras: list[SynthesisExtras | None] = []

    @staticmethod
    def _pitch_of(wav: bytes) -> float:
        import soundfile as sf

        x, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
        mono = x.mean(axis=1)[: rate * 4]
        spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        return float(max(80.0, np.argmax(spectrum) * rate / len(mono)))

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
        extras: SynthesisExtras | None = None,
    ) -> Iterator[FloatArray]:
        self.calls.append(text)
        self.extras.append(extras)
        v = self.find_voice(voice)
        freq = 220.0 * (1 + (v.sid if v else 0) * 0.25)
        if v is None and voice and self.supports_blending and looks_like_blend(voice):
            blend = resolve_blend(voice, self.voices, "fake-tts")
            freq = sum(w * 220.0 * (1 + x.sid * 0.25) for x, w in blend.parts)
        if extras is not None and extras.reference_audio is not None:
            freq = self._pitch_of(extras.reference_audio)
        if extras is not None and "pitch" in extras.options:
            freq = float(extras.options["pitch"])
        total = max(1, int(len(text) * self.seconds_per_char / max(speed, 0.1) * self.sample_rate))
        delay = self.first_delay_s + self.delay_per_char_s * len(text)
        per_piece = total // self.pieces
        for i in range(self.pieces):
            if cancelled and cancelled():
                return
            time.sleep(delay / self.pieces)
            n = per_piece if i < self.pieces - 1 else total - per_piece * (self.pieces - 1)
            t = (np.arange(n) + i * per_piece) / self.sample_rate
            yield (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class FakeVAD(VADModel):
    def __init__(self, opts: dict[str, Any]) -> None:
        self.opts = opts

    def create_detector(self, params: VadParams | None = None) -> VadDetector:
        from bolo.streaming.vad import EnergyVad

        return EnergyVad(params or VadParams())


class FakeDiarization(DiarizationModel):
    """Tells "speakers" apart by the dominant frequency of each burst of sound (tests only)."""

    def __init__(self, opts: dict[str, Any]) -> None:
        self.delay_s = float(opts.get("delay_s", 0.02))
        #: find several tones at once (overlapping "speakers") in 0.25 s frames
        self.multi_pitch = bool(opts.get("multi_pitch", False))
        #: also return per-frame "probabilities" (0.9 inside a speaker's turns, 0.05 outside), as Nemotron does
        self.activity = bool(opts.get("activity", False))
        self.calls: list[tuple[int | None, float | None]] = []

    def diarize_detailed(
        self, samples: FloatArray, num_speakers: int | None = None, threshold: float | None = None
    ) -> DiarizationResult:
        turns = self.diarize(samples, num_speakers, threshold)
        if not self.activity:
            return DiarizationResult(turns)
        n_spk = max((t.speaker for t in turns), default=-1) + 1
        probs = np.full((len(samples) // 160 + 1, n_spk), 0.05, np.float32)
        for t in turns:
            probs[round(t.start * 100) : round(t.end * 100), t.speaker] = 0.9
        return DiarizationResult(turns, SpeakerActivity(probs, 0.01))

    def _multi_pitch_turns(self, samples: FloatArray) -> list[SpeakerTurn]:
        """Every tone above a fifth of the loudest frame's peak is a speaker; its frames make its turns."""
        frame = 4000
        pitches: list[float] = []
        active: list[set[int]] = []
        floor = 0.0
        spectra = []
        for i in range(0, len(samples) - frame + 1, frame):
            spectrum = np.abs(np.fft.rfft(samples[i : i + frame] * np.hanning(frame)))
            spectra.append(spectrum)
            floor = max(floor, float(spectrum.max()))
        for spectrum in spectra:
            found: set[int] = set()
            if floor > 0 and spectrum.max() > 0.2 * floor:
                peaks = [
                    k for k in range(1, len(spectrum) - 1)
                    if spectrum[k] > 0.2 * floor and spectrum[k] >= spectrum[k - 1] and spectrum[k] >= spectrum[k + 1]
                ]  # fmt: skip
                for k in peaks:
                    pitch = k * 16000 / frame
                    speaker = next((j for j, known in enumerate(pitches) if abs(known - pitch) < 40), -1)
                    if speaker < 0:
                        pitches.append(pitch)
                        speaker = len(pitches) - 1
                    found.add(speaker)
            active.append(found)
        turns: list[SpeakerTurn] = []
        for speaker in range(len(pitches)):
            start: int | None = None
            for index, found in enumerate([*active, set()]):
                if speaker in found and start is None:
                    start = index
                elif speaker not in found and start is not None:
                    turns.append(SpeakerTurn(start * frame / 16000, index * frame / 16000, speaker))
                    start = None
        return sorted(turns, key=lambda t: t.start)

    @staticmethod
    def _pitch(samples: FloatArray) -> float:
        if len(samples) < 256:
            return 0.0
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        return float(np.argmax(spectrum) * 16000 / len(samples))

    def embed(self, samples: FloatArray) -> FloatArray:
        pitch = self._pitch(samples)
        angle = (np.pi / 2) * round(min(max((pitch - 100.0) / 400.0, 0.0), 1.0) * 4) / 4  # five well separated "voices"
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    def diarize(
        self, samples: FloatArray, num_speakers: int | None = None, threshold: float | None = None
    ) -> list[SpeakerTurn]:
        from bolo.streaming.vad import EnergyVad, detect_segments

        self.calls.append((num_speakers, threshold))
        time.sleep(self.delay_s)
        if self.multi_pitch:
            return self._multi_pitch_turns(samples)
        segments = detect_segments(EnergyVad(VadParams(min_silence_s=0.3, speech_pad_s=0.0)), samples)
        pitches: list[float] = []
        turns: list[SpeakerTurn] = []
        for a, b in segments:
            pitch = self._pitch(samples[a:b])
            speaker = next((i for i, known in enumerate(pitches) if abs(known - pitch) < 40), -1)
            if speaker < 0:
                if num_speakers and len(pitches) >= num_speakers:
                    speaker = min(range(len(pitches)), key=lambda i: abs(pitches[i] - pitch))
                else:
                    pitches.append(pitch)
                    speaker = len(pitches) - 1
            turns.append(SpeakerTurn(a / 16000, b / 16000, speaker))
        return turns


class FakeSeparation(SeparationModel):
    """Keeps the enrolled voice's dominant tone and removes everything else (tests only)."""

    def __init__(self, opts: dict[str, Any]) -> None:
        self.delay_s = float(opts.get("delay_s", 0.0))
        self.delay_per_audio_s = float(opts.get("delay_per_audio_s", 0.0))
        self.fail = bool(opts.get("fail", False))
        self.mode = str(opts.get("mode", "target"))
        self.voices: dict[str, float] = {}
        self.extracted: list[tuple[str, float]] = []  # (key, seconds) per call
        self._lock = threading.Lock()

    def enrol(self, key: str, sample: FloatArray) -> None:
        if len(sample) < int(self.min_sample_s * 16000) - 1:
            raise ValueError("voice sample too short")
        with self._lock:
            self.voices[key] = FakeDiarization._pitch(sample)

    def extract(self, key: str, mixture: FloatArray) -> FloatArray:
        if self.fail:
            raise RuntimeError("fake separation failure requested")
        time.sleep(self.delay_s + self.delay_per_audio_s * len(mixture) / 16000)
        with self._lock:
            pitch = self.voices[key]
            self.extracted.append((key, len(mixture) / 16000))
        spectrum = np.fft.rfft(mixture)
        freqs = np.fft.rfftfreq(len(mixture), 1 / 16000)
        spectrum[np.abs(freqs - pitch) > 40] = 0
        return np.fft.irfft(spectrum, len(mixture)).astype(np.float32)

    def release(self, key: str) -> None:
        with self._lock:
            self.voices.pop(key, None)

    def separate(self, mixture: FloatArray) -> list[FloatArray]:
        """Blind mode: one track per strong tone, loudest first."""
        if self.fail:
            raise RuntimeError("fake separation failure requested")
        with self._lock:
            self.extracted.append(("blind", len(mixture) / 16000))
        spectrum = np.fft.rfft(mixture)
        freqs = np.fft.rfftfreq(len(mixture), 1 / 16000)
        power = np.abs(spectrum)
        tracks: list[FloatArray] = []
        taken: list[float] = []
        for k in np.argsort(power)[::-1]:
            if power[k] < 0.2 * power.max() or len(tracks) >= 3:
                break
            if any(abs(freqs[k] - f) < 80 for f in taken):
                continue
            taken.append(float(freqs[k]))
            kept = spectrum.copy()
            kept[np.abs(freqs - freqs[k]) > 40] = 0
            tracks.append(np.fft.irfft(kept, len(mixture)).astype(np.float32))
        return tracks


class FakeTurn(TurnModel):
    """Calls a turn finished once it holds ``min_complete_s`` of sound (tests only)."""

    def __init__(self, opts: dict[str, Any]) -> None:
        self.min_complete_s = float(opts.get("min_complete_s", 3.0))
        self.calls: list[float] = []

    def probability(self, samples: FloatArray) -> float:
        frames = samples[: len(samples) // 320 * 320].reshape(-1, 320)
        sound_s = float(np.sum(np.sqrt(np.mean(frames**2, axis=1)) > 0.01)) * 0.02 if len(frames) else 0.0
        self.calls.append(sound_s)
        return 0.95 if sound_s >= self.min_complete_s else 0.05


class FakeTextTurn(TurnModel):
    """Reads the transcript: a turn is finished once it holds ``min_words`` words (tests only)."""

    hears = "text"

    def __init__(self, opts: dict[str, Any]) -> None:
        self.min_words = int(opts.get("min_words", 8))
        self.texts: list[str] = []

    def probability(self, samples: FloatArray) -> float:
        raise NotImplementedError("reads text")

    def probability_text(self, text: str) -> float:
        self.texts.append(text)
        return 0.95 if len(text.split()) >= self.min_words else 0.05


class FakeDenoiser(DenoiseModel):
    """Spectral gate: frequency bins weaker than a few times the median are removed (tests only).

    Steady tones survive and broadband noise mostly goes, which is enough to see that the audio was changed.
    Remembers the length and rate of every call."""

    def __init__(self, opts: dict[str, Any]) -> None:
        self.sample_rate = int(opts.get("sample_rate", 16000))
        self.gate = float(opts.get("gate", 4.0))
        self.fail = bool(opts.get("fail", False))
        self.calls: list[tuple[int, int]] = []  # (samples, sample rate) per call
        self._lock = threading.Lock()

    def denoise(self, samples: FloatArray) -> FloatArray:
        if self.fail:
            raise RuntimeError("fake denoiser failure requested")
        with self._lock:
            self.calls.append((len(samples), self.sample_rate))
        if len(samples) == 0:
            return samples.astype(np.float32)
        spectrum = np.fft.rfft(samples)
        magnitude = np.abs(spectrum)
        spectrum[magnitude < self.gate * float(np.median(magnitude))] = 0
        return np.fft.irfft(spectrum, len(samples)).astype(np.float32)


class FakeEngine(Engine):
    id = "fake"
    name = "Fake engine (testing)"
    license = "Apache-2.0"
    kinds = (
        "fake-stt", "fake-stt-streaming", "fake-stt-speakers", "fake-tts", "fake-vad", "fake-diarization", "fake-turn",
        "fake-turn-text", "fake-separation", "fake-denoise",
    )  # fmt: skip

    def __init__(self) -> None:
        self.loaded: list[Model] = []

    def status(self) -> EngineStatus:
        return EngineStatus(self.id, True, __version__, None)

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        merged = {**spec, **opts.options}
        model: Model
        if kind == "fake-stt":
            model = FakeSTT(merged)
        elif kind == "fake-stt-streaming":
            model = FakeStreamingSTT(merged)
        elif kind == "fake-stt-speakers":
            model = FakeSpeakerSTT(merged)
        elif kind == "fake-tts":
            model = FakeTTS(merged)
        elif kind == "fake-vad":
            model = FakeVAD(merged)
        elif kind == "fake-diarization":
            model = FakeDiarization(merged)
        elif kind == "fake-turn":
            model = FakeTurn(merged)
        elif kind == "fake-turn-text":
            model = FakeTextTurn(merged)
        elif kind == "fake-separation":
            model = FakeSeparation(merged)
        elif kind == "fake-denoise":
            model = FakeDenoiser(merged)
        else:
            raise ValueError(f"fake engine cannot load kind {kind!r}")
        self.loaded.append(model)
        return model
