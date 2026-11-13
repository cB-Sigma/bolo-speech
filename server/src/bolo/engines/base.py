"""The engine plugin interface.

An *engine* wraps one inference library (sherpa-onnx, faster-whisper, …).
Engines are imported lazily: listing them or asking for their status never
imports the underlying library until it is needed, and a missing library is
reported as ``available: false`` with the reason instead of failing the
server.

A loaded model implements one of four small interfaces:

* :class:`STTModel` – batch decoding of one or more utterances at once
* :class:`STTModel` with ``streaming = True`` – incremental decoding through
  :class:`STTStream` objects that can be advanced together in one batch
* :class:`TTSModel` – synthesis that yields audio in pieces
* :class:`VADModel` – factories for stateful speech detectors
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass
class EngineStatus:
    id: str
    available: bool
    version: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "version": self.version, "available": self.available, "reason": self.reason}


@dataclass
class LoadOptions:
    threads: int = 1
    device: str = "cpu"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Word:
    w: str
    t0: float
    t1: float

    def to_dict(self) -> dict[str, Any]:
        return {"w": self.w, "t0": round(self.t0, 3), "t1": round(self.t1, 3)}


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    speaker: str | None = None  # set when the request asked for diarization


@dataclass
class STTRequest:
    samples: FloatArray  # mono float32 at the model's sample rate
    language: str | None = None
    task: str = "transcribe"  # or "translate"
    prompt: str | None = None
    word_timestamps: bool = False
    temperature: float = 0.0
    hotwords: str | None = None


@dataclass
class STTResult:
    text: str
    language: str | None = None
    segments: list[Segment] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)


@dataclass
class StreamResult:
    text: str
    words: list[Word] = field(default_factory=list)


class STTStream(abc.ABC):
    """One incremental recognition stream of a streaming model."""

    @abc.abstractmethod
    def accept(self, samples: FloatArray) -> None: ...

    @abc.abstractmethod
    def finish(self) -> None:
        """Signal the end of input (adds the tail padding the model needs)."""

    @abc.abstractmethod
    def result(self) -> StreamResult: ...

    @abc.abstractmethod
    def is_endpoint(self) -> bool: ...

    @abc.abstractmethod
    def reset(self) -> None: ...


class Model(abc.ABC):
    task: str = ""
    sample_rate: int = 16000

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release native resources."""


class STTModel(Model):
    task = "stt"
    #: largest batch ``transcribe`` decodes in one native call
    max_batch: int = 1
    #: longest utterance the model accepts in one piece (seconds); longer audio is segmented
    max_chunk_s: float = 30.0
    #: True for incremental (frame-synchronous) models
    streaming: bool = False
    #: languages the model can be told to use; empty means fixed or automatic
    languages: tuple[str, ...] = ()
    supports_translate: bool = False
    supports_word_timestamps: bool = False

    @abc.abstractmethod
    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        """Decode a batch of utterances. Must return one result per request, in order."""

    def create_stream(self, language: str | None = None) -> STTStream:
        raise NotImplementedError("this model does not support incremental decoding")

    def decode_streams(self, streams: list[STTStream]) -> None:
        """Advance every stream as far as its buffered audio allows, batched."""
        raise NotImplementedError


@dataclass
class Voice:
    id: str
    name: str = ""
    language: str = ""
    gender: str = ""
    sid: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name or self.id, "language": self.language, "gender": self.gender}


class TTSModel(Model):
    task = "tts"
    voices: tuple[Voice, ...] = ()

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[FloatArray]:
        """Yield float32 audio at ``sample_rate`` for one chunk of text."""

    def fit_text(self, text: str, language: str | None) -> str:
        """Adapt normalised text to what this voice can read; raise ``UnsupportedCharacters`` if it cannot.

        The default is to pass the text through. Voices with a closed character set override it.
        """
        return text

    def find_voice(self, voice: str | None) -> Voice | None:
        if not self.voices:
            return None
        if not voice:
            return self.voices[0]
        v = voice.strip().lower()
        for cand in self.voices:
            if cand.id.lower() == v or cand.name.lower() == v:
                return cand
        if v.isdigit():
            for cand in self.voices:
                if cand.sid == int(v):
                    return cand
        return None


@dataclass
class VadEvent:
    kind: str  # "speech_start" | "speech_end"
    sample: int  # position in the stream, at the detector's sample rate


class VadDetector(abc.ABC):
    """Stateful speech detector over a stream of 16 kHz float32 audio."""

    sample_rate: int = 16000

    @abc.abstractmethod
    def accept(self, samples: FloatArray) -> list[VadEvent]: ...

    @abc.abstractmethod
    def flush(self) -> list[VadEvent]:
        """End of input: close an open speech segment."""

    @property
    @abc.abstractmethod
    def in_speech(self) -> bool: ...


@dataclass
class VadParams:
    threshold: float = 0.5
    min_silence_s: float = 0.5
    min_speech_s: float = 0.25
    max_speech_s: float = 20.0
    speech_pad_s: float = 0.1


class VADModel(Model):
    task = "vad"

    @abc.abstractmethod
    def create_detector(self, params: VadParams | None = None) -> VadDetector: ...


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: int  # 0-based cluster index, in order of first appearance


class DiarizationModel(Model):
    """Who spoke when: speaker segmentation + speaker embeddings + clustering."""

    task = "diarization"

    @abc.abstractmethod
    def diarize(
        self, samples: FloatArray, num_speakers: int | None = None, threshold: float | None = None
    ) -> list[SpeakerTurn]:
        """Speaker turns of a 16 kHz recording, sorted by start time."""

    @abc.abstractmethod
    def embed(self, samples: FloatArray) -> FloatArray:
        """A unit-length speaker embedding for one stretch of speech."""


class TurnModel(Model):
    """End-of-turn detection: has the speaker finished, or only paused?"""

    task = "turn"

    @abc.abstractmethod
    def probability(self, samples: FloatArray) -> float:
        """Probability (0..1) that the 16 kHz utterance in ``samples`` is a finished turn."""


class Engine(abc.ABC):
    """Base class of engine plugins."""

    id: str = ""
    name: str = ""
    license: str = ""
    url: str = ""
    #: model ``kind`` values this engine can load
    kinds: tuple[str, ...] = ()

    @abc.abstractmethod
    def status(self) -> EngineStatus:
        """Report availability without raising."""

    @abc.abstractmethod
    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        """Load a model of ``kind`` from directory ``path``.

        ``spec`` is the catalog entry's ``options`` mapping (file names,
        language, …) merged with per-request overrides.
        """

    def detect(self, path: Path) -> dict[str, Any] | None:
        """Guess ``{"kind", "task", "options"}`` for an unknown model directory, or None."""
        return None
