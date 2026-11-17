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
    #: where the model runs, as placed by the lifecycle manager (``bolo.placement``): "cpu", "cuda:0", "coreml"...
    device: str = "cpu"
    options: dict[str, Any] = field(default_factory=dict)
    #: llama.cpp models: layers to offload (``-ngl``); None: the engine's own default for ``device``
    gpu_layers: int | None = None
    #: ONNX Runtime models made of several graphs: role ("encoder", "decoder", ...) -> device; a role not listed
    #: runs on ``device``
    graph_devices: dict[str, str] = field(default_factory=dict)
    #: PyTorch plugin models: accelerate's ``max_memory`` (device -> bytes) for ``device_map="auto"``
    max_memory: dict[str, int] | None = None
    #: ONNX Runtime graphs cut in two (bolo.onnx_split): role ("model" for a single-graph model) -> bytes of
    #: weights the front half may put on ``device_for(role)``; the back half runs on the CPU
    graph_split: dict[str, int] = field(default_factory=dict)
    #: llama.cpp models on NVIDIA GPUs: the GPUs the layers go on, each one's share (several GPUs) and the main GPU
    gpus: list[int] = field(default_factory=list)
    tensor_split: list[float] | None = None
    main_gpu: int | None = None
    #: NVIDIA GPUs this process can see (``--tensor-split`` takes one share per visible GPU)
    visible_gpus: int = 1
    #: requests the model works on at once (``num_parallel``); a llama-server bolo starts gets ``--parallel``
    parallel: int = 1

    def device_for(self, role: str) -> str:
        """The device of one graph of a multi-graph model."""
        return self.graph_devices.get(role, self.device)


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
    overlap: bool = False  # diarization found another speaker talking during this segment
    separated: bool = False  # decoded from this speaker's separated track, not from the mixture


@dataclass
class STTRequest:
    samples: FloatArray  # mono float32 at the model's sample rate
    language: str | None = None
    task: str = "transcribe"  # or "translate"
    prompt: str | None = None
    word_timestamps: bool = False
    temperature: float = 0.0
    hotwords: str | None = None
    #: label speakers in the model itself (only set for models with ``native_diarization``)
    diarize: bool = False


@dataclass
class STTResult:
    text: str
    language: str | None = None
    segments: list[Segment] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    #: what the request cost at a cloud provider (:class:`bolo.providers.Usage`; remote engine only)
    usage: Any = None


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

    def restart(self) -> None:
        """Start over as if new: nothing heard before carries over (``reset`` may keep the encoder's context)."""
        self.reset()


class Model(abc.ABC):
    task: str = ""
    sample_rate: int = 16000
    #: set by the model manager once the model is loaded: called (from a worker thread) with a
    #: ``bolo.cuda_libs.ProviderFallback.to_dict()`` when the model's runtime gives up its accelerator while serving
    #: and goes on on the CPU, so that its placement says so
    on_fallback: Callable[[dict[str, Any]], None] | None = None

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release native resources."""


class STTModel(Model):
    task = "stt"
    #: diarization-conditioned (target-speaker) recognition: :meth:`transcribe_speakers` decodes a whole
    #: recording once per speaker, steered by per-frame speaker activity, instead of turn by turn
    speaker_conditioned: bool = False
    #: largest batch ``transcribe`` decodes in one native call
    max_batch: int = 1
    #: longest utterance the model accepts in one piece (seconds); longer audio is segmented
    max_chunk_s: float = 30.0
    #: True for incremental (frame-synchronous) models
    streaming: bool = False
    #: streaming models in a live session with a VAD: seconds of audio from before the VAD heard speech start that
    #: a fresh stream is given when it does; the recogniser hears no idle audio. 0: it hears everything, idle audio
    #: too, and its own endpointing resets it (a cloud recogniser's live connection wants audio all the time).
    stream_preroll_s: float = 0.0
    #: streaming models with a pre-roll: quieter speech is raised to about this level (dBFS) before the recogniser
    #: hears it, one gain per utterance (the VAD hears it as received); None leaves the level alone
    stream_speech_dbfs: float | None = None
    #: streaming models with a pre-roll: the encoder's chunk (seconds; 0 when not known), and the audio a stream has
    #: heard when it is restarted, before the first sample it is given. A small streaming transducer keeps or loses a
    #: first word by where in a chunk the word starts; with the chunk known, a live session decodes each utterance
    #: from two pre-rolls, one of them aligned to a chunk, and takes the result whose start is fuller
    stream_chunk_s: float = 0.0
    stream_primed_s: float = 0.0
    #: languages the model can be told to use; empty means fixed or automatic
    languages: tuple[str, ...] = ()
    supports_translate: bool = False
    supports_word_timestamps: bool = False
    #: the model labels speakers itself (a cloud recogniser with diarization): a request's ``diarize`` is then
    #: passed to it as ``STTRequest.diarize`` instead of cutting the audio at bolo's speaker turns
    native_diarization: bool = False

    @abc.abstractmethod
    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        """Decode a batch of utterances. Must return one result per request, in order."""

    def transcribe_speakers(self, request: SpeakerRequest) -> list[STTResult]:
        """Speaker-conditioned models: one result per column of ``request.activity`` (speaker ``k`` of the
        diarization), segment times measured from the start of the recording."""
        raise NotImplementedError("this model is not conditioned on speaker diarization")

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


@dataclass
class SynthesisExtras:
    """Per-request inputs that only some voices use (passed as ``extras=`` when ``accepts_extras`` is true)."""

    #: language of the text (BCP-47 or ISO 639-1), for multilingual models that are told the language
    language: str | None = None
    #: a recording of the voice to imitate (WAV, mono PCM16), for models that clone voices
    reference_audio: bytes | None = None
    #: what is said in ``reference_audio``, for models that want the transcript of the reference
    reference_text: str | None = None
    #: filled by cloud voices with what the request cost there (:class:`bolo.providers.Usage`)
    usage: Any = None
    #: model-specific request options (the request's ``options``), e.g. Tontaube's style and sliders
    options: dict[str, Any] = field(default_factory=dict)


class TTSModel(Model):
    task = "tts"
    voices: tuple[Voice, ...] = ()
    #: True when ``synthesize`` takes a ``phonemizer`` keyword (voices that read phonemes)
    accepts_phonemizer: bool = False
    #: True when ``synthesize`` takes an ``extras`` keyword (:class:`SynthesisExtras`)
    accepts_extras: bool = False
    #: True when the model can imitate a voice from ``SynthesisExtras.reference_audio``
    supports_cloning: bool = False
    #: names of the request options (``SynthesisExtras.options``) the model reads; ``("*",)``: any
    request_options: tuple[str, ...] = ()
    #: True when ``voice`` may be a blend of the model's voices, ``af_bella(0.7)+af_sky(0.3)`` (bolo.engines.blend)
    supports_blending: bool = False

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[FloatArray]:
        """Yield float32 audio at ``sample_rate`` for one chunk of text."""

    def warm_up(self) -> None:
        """Run once after loading so that the first request does not pay for lazy initialisation."""

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


@dataclass
class SpeakerActivity:
    """Per-frame speaker probabilities of a recording: ``probs[frame, speaker]`` in 0..1.

    Column ``k`` belongs to speaker ``k`` of the turns it came with (0-based, in order of first appearance).
    Only end-to-end neural diarizers (Nemotron-3-Diarization) produce it; it is what diarization-conditioned
    recognisers take as input.
    """

    probs: FloatArray
    frame_s: float = 0.01

    @property
    def num_speakers(self) -> int:
        return int(self.probs.shape[1]) if self.probs.ndim == 2 else 0

    def frame(self, t: float) -> int:
        return max(0, min(len(self.probs), round(t / self.frame_s)))


@dataclass
class SpeakerRequest:
    """A whole recording for a speaker-conditioned recogniser (:meth:`STTModel.transcribe_speakers`)."""

    samples: FloatArray  # mono float32 at the model's sample rate
    activity: SpeakerActivity  # who talks when: one column per speaker, frames from the start of ``samples``
    language: str | None = None
    task: str = "transcribe"
    prompt: str | None = None
    word_timestamps: bool = False
    temperature: float = 0.0


@dataclass
class DiarizationResult:
    turns: list[SpeakerTurn]
    activity: SpeakerActivity | None = None


class DiarizationModel(Model):
    """Who spoke when: speaker segmentation + speaker embeddings + clustering, or an end-to-end model."""

    task = "diarization"
    #: whether bolo's speaker clean-up (:func:`bolo.diarize.consolidate_speakers`) runs unless the request says
    consolidate_default: bool = True

    def diarize_detailed(
        self, samples: FloatArray, num_speakers: int | None = None, threshold: float | None = None
    ) -> DiarizationResult:
        """Like :meth:`diarize`, plus the per-frame speaker probabilities when the model has them."""
        return DiarizationResult(self.diarize(samples, num_speakers, threshold))

    @abc.abstractmethod
    def diarize(
        self, samples: FloatArray, num_speakers: int | None = None, threshold: float | None = None
    ) -> list[SpeakerTurn]:
        """Speaker turns of a 16 kHz recording, sorted by start time."""

    @abc.abstractmethod
    def embed(self, samples: FloatArray) -> FloatArray:
        """A unit-length speaker embedding for one stretch of speech."""


class SeparationModel(Model):
    """Target-speaker extraction: keep one voice of a mixture, given a short sample of that voice.

    Callers enrol a speaker under a key of their choosing, extract as often as they like, then release
    the key. Audio in and out is 16 kHz mono.
    """

    task = "separation"
    #: "target": extracts one enrolled voice per call (enrol/extract/release);
    #: "blind": splits a mixture into tracks without knowing the voices (separate), which bolo then
    #: assigns to the diarized speakers by voice similarity
    mode: str = "target"
    #: seconds of the speaker's voice the model needs to enrol them ("target" models)
    min_sample_s: float = 3.0

    def enrol(self, key: str, sample: FloatArray) -> None:
        """Prepare an extractor for the voice in ``sample`` (16 kHz, at least ``min_sample_s`` long)."""
        raise NotImplementedError(f"{type(self).__name__} is a {self.mode} separator")

    def extract(self, key: str, mixture: FloatArray) -> FloatArray:
        """The enrolled voice from ``mixture`` (16 kHz); as long as the input."""
        raise NotImplementedError(f"{type(self).__name__} is a {self.mode} separator")

    def release(self, key: str) -> None:
        """Forget an enrolled voice (a no-op for unknown keys)."""

    def separate(self, mixture: FloatArray) -> list[FloatArray]:
        """Blind separation: one track per source found in ``mixture`` (16 kHz), each as long as the input."""
        raise NotImplementedError(f"{type(self).__name__} is a {self.mode} separator")


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

    def graphs(self, kind: str, path: Path, spec: dict[str, Any]) -> dict[str, int]:
        """Models made of several separately placeable graphs: role -> bytes on disk (empty: one piece)."""
        return {}
