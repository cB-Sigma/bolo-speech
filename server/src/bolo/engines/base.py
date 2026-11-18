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
* :class:`SpeechToSpeechModel` – a turn-based spoken conversation (the ``speech-to-speech`` pipeline)
"""

from __future__ import annotations

import abc
import threading
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
    #: the sampling temperature this segment was decoded at (a model that falls back to higher temperatures says
    #: which); None: the model does not say (bolo then reports the request's temperature, or 0 for greedy decoding)
    temperature: float | None = None
    emotion: str | None = None  # the emotion the recogniser heard in this segment (SenseVoice), when it says
    events: list[str] = field(default_factory=list)  # sounds the recogniser heard (SenseVoice: laughter, music, ...)


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
    #: decoding options the model honours (``STTModel.decode_options``; names and meanings in bolo.stt_options)
    options: dict[str, Any] = field(default_factory=dict)
    #: set when the client that asked has gone (``ModelQueue.submit`` cancelled): an engine that can stop part-way
    #: checks it and returns what it has; its result is discarded either way
    abort: threading.Event = field(default_factory=threading.Event, compare=False, repr=False)


#: the free-form fields of a request a recogniser may or may not use (``STTModel.request_fields``); a temperature
#: counts as set when it is above 0 (greedy decoding is what every model does without one)
REQUEST_FIELDS = ("prompt", "hotwords", "temperature")


@dataclass
class STTResult:
    text: str
    language: str | None = None
    segments: list[Segment] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    #: what the request cost at a cloud provider (:class:`bolo.providers.Usage`; remote engine only)
    usage: Any = None
    #: how sure the model was of ``language`` when it detected it (0..1); None when it does not say
    language_probability: float | None = None


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
    #: decoding options (``bolo.stt_options.ENGINE_OPTIONS``) ``transcribe`` honours in ``STTRequest.options``;
    #: bolo passes only these and reports the others as ignored
    decode_options: tuple[str, ...] = ()
    #: request fields (:data:`REQUEST_FIELDS`) ``transcribe`` uses; a request that sets one the model does not use
    #: is told so (the response's ``x-bolo-ignored-options`` header), the same as an engine option it does not honour
    request_fields: tuple[str, ...] = ()

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
    #: a direction for the delivery in plain words ("calm, a little amused"), for models with
    #: ``supports_instructions``; None for the others
    instructions: str | None = None


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
    #: True when the model follows ``SynthesisExtras.instructions`` (style, emotion or voice direction in words)
    supports_instructions: bool = False
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


class SpeakerEmbeddingModel(Model):
    """A speaker embedder on its own: one vector per stretch of speech; the cosine of two is the similarity."""

    task = "speaker-embedding"

    @abc.abstractmethod
    def embed(self, samples: FloatArray) -> FloatArray:
        """A unit-length speaker embedding for one stretch of 16 kHz speech."""


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


class DenoiseStream(abc.ABC):
    """Live denoising of one audio stream, piece by piece, at the model's ``sample_rate``.

    :meth:`process` returns the enhanced audio that is ready so far; sample ``k`` of everything it returned is sample
    ``k`` of everything it was given (a model's own delay is cut off, so it may hold back a few milliseconds at first).
    :meth:`finish` returns the rest: in all, the stream returns exactly as many samples as it was given."""

    #: the model keeps its state from piece to piece (False: each piece is enhanced with some audio before it)
    stateful: bool = True

    @abc.abstractmethod
    def process(self, samples: FloatArray) -> FloatArray: ...

    def finish(self) -> FloatArray:
        return np.zeros(0, dtype=np.float32)


class OverlapDenoiseStream(DenoiseStream):
    """For denoisers that only enhance whole recordings: every piece is enhanced together with up to ``context_s``
    of the audio before it, and the end of the result is kept. Nothing is held back."""

    stateful = False

    def __init__(self, model: DenoiseModel, context_s: float = 0.5) -> None:
        self.model = model
        self.keep = max(0, int(context_s * model.sample_rate))
        self.before = np.zeros(0, dtype=np.float32)

    def process(self, samples: FloatArray) -> FloatArray:
        if not len(samples):
            return np.zeros(0, dtype=np.float32)
        window = np.concatenate([self.before, np.asarray(samples, dtype=np.float32)])
        out = np.asarray(self.model.denoise(window), dtype=np.float32)[-len(samples) :]
        self.before = window[-self.keep :] if self.keep else self.before[:0]
        return np.ascontiguousarray(out, dtype=np.float32)


class FrameDenoiseStream(DenoiseStream):
    """For causal models that work on fixed frames and keep a state: ``step`` enhances one frame of ``frame`` samples;
    its output runs ``delay`` samples behind its input, which this cuts off."""

    def __init__(self, step: Callable[[FloatArray], FloatArray], frame: int, delay: int) -> None:
        self.step = step
        self.frame = frame
        self.delay = delay
        self.pending = np.zeros(0, dtype=np.float32)
        self.taken = 0  # samples given
        self.made = 0  # samples of output produced (including the delay)
        self.given = 0  # samples handed back

    def _run(self, audio: FloatArray) -> FloatArray:
        self.pending = np.concatenate([self.pending, np.asarray(audio, dtype=np.float32)])
        whole = len(self.pending) - len(self.pending) % self.frame
        if not whole:
            return np.zeros(0, dtype=np.float32)
        out = [
            np.asarray(self.step(self.pending[i : i + self.frame]), dtype=np.float32)
            for i in range(0, whole, self.frame)
        ]
        self.pending = self.pending[whole:]
        made = np.concatenate(out)
        start = max(0, self.delay - self.made)  # the model's delay is cut off once, at the start
        self.made += len(made)
        return made[start:]

    def process(self, samples: FloatArray) -> FloatArray:
        self.taken += len(samples)
        out = self._run(samples)[: max(0, self.taken - self.given)]
        self.given += len(out)
        return np.ascontiguousarray(out, dtype=np.float32)

    def finish(self) -> FloatArray:
        need = self.taken - self.given
        if need <= 0:
            return np.zeros(0, dtype=np.float32)
        pad = self.delay + need + 2 * self.frame  # enough to push the held-back audio out
        out = self._run(np.zeros(pad, dtype=np.float32))[:need]
        if len(out) < need:
            out = np.pad(out, (0, need - len(out)))
        self.given += len(out)
        return np.ascontiguousarray(out, dtype=np.float32)


class DenoiseModel(Model):
    """Speech enhancement: noise removed from a recording, voice kept. Audio in and out at ``sample_rate``."""

    task = "denoise"
    #: can enhance a live stream (a cloud API that takes a whole recording per request cannot)
    live: bool = True

    @abc.abstractmethod
    def denoise(self, samples: FloatArray) -> FloatArray:
        """The enhanced recording (mono float32 at ``sample_rate``), as long as the input."""

    def create_stream(self) -> DenoiseStream:
        """A live stream (:class:`DenoiseStream`); models without a streaming form overlap pieces instead."""
        return OverlapDenoiseStream(self)


class AudioClassifierModel(Model):
    """A recording in (mono float32 at ``sample_rate``, 16 kHz), every class the model knows with its score out,
    most likely first."""

    #: classes are independent (audio tagging: speech and music at once); False: one class wins (scores sum to 1)
    multi_label: bool = False

    @abc.abstractmethod
    def classify(self, samples: FloatArray) -> list[tuple[str, float]]: ...


class AudioTaggingModel(AudioClassifierModel):
    """Sound event detection and audio tagging: which sounds (speech, music, a dog, a siren, ...) a recording holds,
    each with its own probability (AudioSet's 527 classes for CED and the Zipformer taggers)."""

    task = "audio-tagging"
    multi_label = True


class EmotionModel(AudioClassifierModel):
    """Speech emotion recognition: how the speaker sounds (angry, happy, neutral, sad, ...), probabilities over the
    model's classes."""

    task = "emotion"


class LidModel(Model):
    """Spoken-language identification: which language is spoken in a recording (16 kHz mono in)."""

    task = "lid"
    #: the language codes the model can answer (ISO 639-1 where one exists, else ISO 639-3)
    labels: list[str] = []
    #: longest stretch of audio looked at, in seconds; a longer recording is cut to its first ``max_s``
    max_s: float = 30.0

    @abc.abstractmethod
    def identify(self, samples: FloatArray) -> list[tuple[str, float]]:
        """Every language the model knows with its probability (summing to 1), most likely first."""


class TurnModel(Model):
    """End-of-turn detection: has the speaker finished, or only paused?"""

    task = "turn"
    #: what the model judges: "audio" (the sound of the utterance, :meth:`probability`) or "text" (its transcript so
    #: far, :meth:`probability_text`)
    hears: str = "audio"

    @abc.abstractmethod
    def probability(self, samples: FloatArray) -> float:
        """Probability (0..1) that the 16 kHz utterance in ``samples`` is a finished turn."""

    def probability_text(self, text: str) -> float:
        """Probability (0..1) that an utterance with this transcript is a finished turn (models that read text)."""
        raise NotImplementedError(f"{type(self).__name__} judges audio, not text")


@dataclass
class ConversationTurn:
    """One turn of a conversation with a speech-to-speech model.

    ``role`` is "user" (``audio``: the turn as spoken, 16 kHz mono; or ``text``), "assistant" (what the model
    answered, as text, and the tool calls it made) or "tool" (``text``: the result of the call ``call_id``)."""

    role: str
    text: str = ""
    audio: FloatArray | None = None
    call_id: str | None = None
    #: assistant turns: ``{"call_id", "name", "arguments"}`` per tool call
    tool_calls: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ConversationRequest:
    """Everything a turn-based speech-to-speech model needs to answer: the whole conversation so far, every time."""

    turns: list[ConversationTurn]
    instructions: str | None = None
    #: function tools as the Realtime API describes them (``{"type": "function", "name", "description", "parameters"}``)
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str = "auto"
    voice: str | None = None
    max_output_tokens: int | None = None
    #: False: text only (a ``["text"]`` session)
    audio_out: bool = True


@dataclass
class ConversationChunk:
    """A piece of the answer: text, audio at the model's ``sample_rate``, or a tool call."""

    text: str = ""
    audio: FloatArray | None = None
    tool_call: dict[str, str] | None = None


class SpeechToSpeechModel(Model):
    """A model that hears speech and answers in speech (and text), one turn at a time.

    The caller keeps the conversation and sends all of it with each request; the model keeps no state between
    requests, so it can serve any number of conversations. Full-duplex models (that listen while they speak, frame
    by frame) do not fit this interface."""

    task = "speech-to-speech"
    sample_rate = 24000
    voices: tuple[Voice, ...] = ()
    #: whether ``tools`` reach the model (False: they are ignored and no tool calls come back)
    supports_tools: bool = False

    @abc.abstractmethod
    def respond(
        self, request: ConversationRequest, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[ConversationChunk]:
        """The answer to the conversation in ``request``, in pieces. Stops early once ``cancelled()`` is true."""


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
