"""Nemotron-3-Diarization in ONNX Runtime: end-to-end neural diarization without PyTorch.

NVIDIA's Nemotron-3-Diarization (OpenMDW-1.1, 100M parameters) is a Sortformer-style model: a 31-layer
Transformer encoder reads 80 ms stacks of log-mel frames and predicts, for every 10 ms frame, the
probability that each of up to eight speakers is talking. Speakers are numbered in order of arrival,
so no clustering step is needed, and two speakers can be active in the same frame (overlap).

Long recordings and live audio are processed in chunks. Every chunk also sees the
**Arrival-Order Speaker Cache** (a compressed selection of earlier frames, chosen per speaker) and a
FIFO queue of the most recent frames, which is how a speaker keeps the same channel across chunks.
The chunk loop and the cache policy below follow the reference implementation in Hugging Face
transformers (``models/nemotron3_diarization``, Apache-2.0) step by step; the ONNX graph is the
encoder and the speaker head for one step (inputs: the chunk's log-mel frames, the cached frame
embeddings and an attention mask; outputs: logits for every input frame, the chunk's frame
embeddings and the learned silence embedding).

Settings (catalog ``options``): ``model`` (ONNX file, default ``onnx/model_quantized.onnx``),
``embedding`` (optional speaker-embedding ONNX for named speakers, run by sherpa-onnx), and the
post-processing of :func:`bolo.diarize.activity_to_turns`: ``threshold`` (onset), ``offset``,
``pad_onset``, ``pad_offset``, ``min_duration_on``, ``min_duration_off``, ``min_speaker_s``.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bolo.audio.features import NEMO_MELS, NemoLogMel
from bolo.engines.base import DiarizationModel, DiarizationResult, FloatArray, LoadOptions, SpeakerActivity, SpeakerTurn
from bolo.ort_providers import create_session, providers_for, session_options

FRAME_S = 0.01  # one output frame per 10 ms mel frame
SUBSAMPLING = 8  # mel frames per encoder frame
HIDDEN = 512
NUM_SPEAKERS = 8


@dataclass(frozen=True)
class ChunkConfig:
    """Chunking and speaker-cache sizes, in 80 ms encoder frames (the model card's table)."""

    chunk_length: int
    right_context: int
    fifo_length: int
    update_period: int
    cache_length: int = 264


#: whole recordings: 27.2 s chunks with 3.2 s look-ahead (the card's "very high latency (offline)" setting)
OFFLINE = ChunkConfig(chunk_length=340, right_context=40, fifo_length=40, update_period=300)
#: live audio; latency = (chunk + look-ahead) x 80 ms
STREAMING = {
    "low_latency": ChunkConfig(chunk_length=9, right_context=4, fifo_length=264, update_period=222),  # 1.04 s
    "very_low_latency": ChunkConfig(chunk_length=6, right_context=2, fifo_length=264, update_period=222),  # 0.64 s
    "ultra_low_latency": ChunkConfig(chunk_length=3, right_context=1, fifo_length=264, update_period=222),  # 0.32 s
}

#: turning probabilities into turns (bolo.diarize.activity_to_turns); chosen on AMI excerpts, see docs/models.md
DEFAULT_POSTPROCESSING = {
    "threshold": 0.5,
    "offset": 0.5,
    "pad_onset": 0.0,
    "pad_offset": 0.0,
    "min_duration_on": 0.0,
    "min_duration_off": 1.0,
}

# speaker-cache policy (transformers' Nemotron3DiarizationStreamingConfig defaults)
_SILENCE_FRAMES_PER_SPEAKER = 1
_PREDICTION_SCORE_THRESHOLD = 0.25
_LATEST_FRAMES_BOOST = 0.05
_STRONG_BOOST_RATE = 0.75
_WEAK_BOOST_RATE = 1.5
_MIN_POSITIVE_SCORES_RATE = 0.5


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))), dtype=np.float32)


@dataclass
class SpeakerCache:
    """Arrival-Order Speaker Cache and FIFO queue of one stream (frame embeddings of 512 values)."""

    config: ChunkConfig
    embeds: np.ndarray = field(default_factory=lambda: np.zeros((0, HIDDEN), np.float32))
    probs: np.ndarray = field(default_factory=lambda: np.zeros((0, NUM_SPEAKERS), np.float32))
    fifo: np.ndarray = field(default_factory=lambda: np.zeros((0, HIDDEN), np.float32))
    compressed: bool = False

    def __post_init__(self) -> None:
        budget = self.config.cache_length // NUM_SPEAKERS - _SILENCE_FRAMES_PER_SPEAKER
        self.min_positive = math.floor(budget * _MIN_POSITIVE_SCORES_RATE)
        self.strong_boosted = math.floor(budget * _STRONG_BOOST_RATE)
        self.weak_boosted = math.floor(budget * _WEAK_BOOST_RATE)

    def context(self) -> np.ndarray:
        """The frames every step attends to before its chunk: the speaker cache, then the FIFO queue."""
        return np.concatenate([self.embeds, self.fifo])

    def update(self, chunk_embeds: np.ndarray, probs: np.ndarray, silence: np.ndarray) -> None:
        """Push a processed chunk (``chunk_embeds``: its frames without look-ahead) to the FIFO queue.

        ``probs`` are the step's speaker probabilities at the encoder frame rate for every input frame
        (cache, FIFO, chunk, look-ahead), zero on padding frames.
        """
        n_cache, n_fifo = len(self.embeds), len(self.fifo)
        fifo = np.concatenate([self.fifo, chunk_embeds])
        popped = 0
        if len(fifo) > self.config.fifo_length:
            popped = min(max(self.config.update_period, len(fifo) - self.config.fifo_length), len(fifo))
        if popped:
            chunk_start = n_cache + n_fifo
            fifo_probs = np.concatenate(
                [probs[n_cache : n_cache + n_fifo], probs[chunk_start : chunk_start + len(chunk_embeds)]]
            )
            # an uncompressed cache holds plain frames whose probabilities this step re-estimated
            stored = self.probs if self.compressed else probs[:n_cache]
            cache_embeds = np.concatenate([self.embeds, fifo[:popped]])
            cache_probs = np.concatenate([stored, fifo_probs[:popped]])
            fifo = fifo[popped:]
            if len(cache_embeds) > self.config.cache_length:
                cache_embeds, cache_probs = self._compress(cache_embeds, cache_probs, silence)
                self.compressed = True
            self.embeds, self.probs = cache_embeds, cache_probs.astype(np.float32)
        self.fifo = fifo

    def _scores(self, probs: np.ndarray) -> np.ndarray:
        thr = _PREDICTION_SCORE_THRESHOLD
        log_p = np.log(np.maximum(probs, thr))
        log_q = np.log(np.maximum(1.0 - probs, thr))
        scores = log_p - log_q + log_q.sum(axis=1, keepdims=True) - math.log(0.5)
        speech = probs > 0.5
        scores = np.where(speech, scores, -np.inf)
        positive = scores > 0
        enough = positive.sum(axis=0, keepdims=True) >= self.min_positive
        return np.where(~positive & speech & enough, -np.inf, scores)

    @staticmethod
    def _boost(scores: np.ndarray, count: int, boost: float) -> np.ndarray:
        if count <= 0:
            return scores
        count = min(count, len(scores))
        top = np.argpartition(-scores, count - 1, axis=0)[:count]  # per speaker, the best frames
        out: np.ndarray = scores.copy()
        np.add.at(out, (top, np.arange(scores.shape[1])[None, :].repeat(count, 0)), boost)
        return out

    def _compress(self, embeds: np.ndarray, probs: np.ndarray, silence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Keep the ``cache_length`` most useful frames, grouped by speaker in time order, plus silence slots."""
        n_frames, n_spk = probs.shape
        length = self.config.cache_length
        scores = self._scores(probs.astype(np.float64))
        scores[length:] += _LATEST_FRAMES_BOOST  # the frames just popped from the FIFO queue
        scores = self._boost(scores, self.strong_boosted, -2.0 * math.log(0.5))
        scores = self._boost(scores, self.weak_boosted, -math.log(0.5))
        scores = np.concatenate([scores, np.full((_SILENCE_FRAMES_PER_SPEAKER, n_spk), np.inf)])
        embeds = np.concatenate([embeds, silence.reshape(1, -1).astype(np.float32)])
        probs = np.concatenate([probs, np.zeros((1, n_spk), np.float32)])
        n_scored = n_frames + _SILENCE_FRAMES_PER_SPEAKER
        flat = scores.T.reshape(-1)  # speaker-major: speaker 0's frames, then speaker 1's, ...
        top = np.argpartition(-flat, length - 1)[:length]
        sentinel = n_scored * n_spk
        top = np.sort(np.where(flat[top] == -np.inf, sentinel, top))
        frames = np.where(top == sentinel, n_frames, np.minimum(top % n_scored, n_frames))
        return embeds[frames], probs[frames]


class _OnnxStep:
    """One encoder step of the ONNX graph."""

    def __init__(self, model_file: str, threads: int, device: str = "cpu", split_budget: int | None = None) -> None:
        import onnxruntime as ort

        from bolo.onnx_split import make_session

        def options() -> Any:
            o = session_options()
            o.intra_op_num_threads = max(1, threads)
            o.inter_op_num_threads = 1
            o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return o

        if split_budget:  # too large for the free GPU memory: the front on the GPU, the back on the CPU
            self.session = make_session(model_file, providers_for(device), options, split_budget)
        else:
            self.session = create_session(model_file, options(), providers_for(device))
        names = {i.name for i in self.session.get_inputs()}
        expected = {"input_features", "cached_embeds", "attention_mask"}
        if not expected <= names:
            raise ValueError(f"{model_file} is not a Nemotron-3-Diarization step graph (inputs {sorted(names)})")

    def __call__(
        self, features: FloatArray, cached: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits, chunk_embeds, silence = self.session.run(
            None,
            {
                "input_features": np.ascontiguousarray(features[None], dtype=np.float32),
                "cached_embeds": np.ascontiguousarray(cached[None], dtype=np.float32),
                "attention_mask": mask[None].astype(np.int64),
            },
        )
        return np.asarray(logits[0]), np.asarray(chunk_embeds[0]), np.asarray(silence)


def _step(
    run: _OnnxStep,
    cache: SpeakerCache,
    features: FloatArray,
    mel_valid: np.ndarray,
    num_chunk: int,
) -> FloatArray:
    """Run one chunk (``features``: its mel frames plus look-ahead) and update ``cache``.

    Returns the speaker probabilities of the chunk's own mel frames (``num_chunk`` encoder frames).
    """
    context = cache.context()
    n_ctx = len(context)
    n_enc = -(-len(features) // SUBSAMPLING)
    enc_valid = mel_valid[::SUBSAMPLING][:n_enc]
    mask = np.concatenate([np.ones(n_ctx, bool), enc_valid])
    logits, embeds, silence = run(features, context, mask)
    probs = _sigmoid(logits)
    pooled = probs[: (n_ctx + n_enc) * SUBSAMPLING].reshape(n_ctx + n_enc, SUBSAMPLING, -1).mean(axis=1)
    pooled = pooled * mask[:, None]
    cache.update(embeds[:num_chunk], pooled, silence)
    start = n_ctx * SUBSAMPLING
    return probs[start : start + num_chunk * SUBSAMPLING]


class NemotronStream:
    """Live diarization: feed 16 kHz audio as it arrives, get speaker probabilities per 10 ms frame.

    :meth:`accept` returns the frames that became final (a chunk is scored once its look-ahead has
    arrived, so frames come out ``(chunk + look-ahead) x 80 ms`` after they were spoken);
    :meth:`finish` scores whatever is left. Each returned array has shape ``(frames, 8)``.
    """

    def __init__(self, run: _OnnxStep, config: ChunkConfig, lock: threading.Lock) -> None:
        self._run = run
        self._lock = lock
        self.config = config
        self.cache = SpeakerCache(config)
        self._mel = NemoLogMel()
        self._pending = np.zeros((0, NEMO_MELS), np.float32)  # mel frames not yet scored
        self.frames_out = 0

    @property
    def latency_s(self) -> float:
        return (self.config.chunk_length + self.config.right_context) * SUBSAMPLING * FRAME_S

    def accept(self, samples: FloatArray) -> FloatArray:
        self._pending = np.concatenate([self._pending, self._mel.push(samples)])
        out: list[FloatArray] = []
        chunk = self.config.chunk_length * SUBSAMPLING
        window = (self.config.chunk_length + self.config.right_context) * SUBSAMPLING
        while len(self._pending) >= window:
            features = self._pending[:window]
            with self._lock:
                out.append(_step(self._run, self.cache, features, np.ones(window, bool), self.config.chunk_length))
            self._pending = self._pending[chunk:]
        return self._emit(out)

    def finish(self) -> FloatArray:
        self._pending = np.concatenate([self._pending, self._mel.finish()])
        out: list[FloatArray] = []
        if len(self._pending):
            n_enc = -(-len(self._pending) // SUBSAMPLING)
            with self._lock:
                probs = _step(self._run, self.cache, self._pending, np.ones(len(self._pending), bool), n_enc)
            out.append(probs[: len(self._pending)])
            self._pending = self._pending[:0]
        return self._emit(out)

    def _emit(self, out: list[FloatArray]) -> FloatArray:
        probs = np.concatenate(out) if out else np.zeros((0, NUM_SPEAKERS), np.float32)
        self.frames_out += len(probs)
        return probs


class NemotronDiarization(DiarizationModel):
    """Nemotron-3-Diarization: per-frame speaker probabilities, turned into speaker turns."""

    #: speakers come from the model, not from clustering embeddings, so bolo's speaker clean-up is off by default
    consolidate_default = False

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        model_file = _find(path, str(spec.get("model", "onnx/model_quantized.onnx")), "onnx/*.onnx")
        self._run = _OnnxStep(
            str(model_file), opts.threads, opts.device, **({"split_budget": opts.graph_split["model"]}
                                                          if opts.graph_split.get("model") else {})
        )  # fmt: skip
        self._lock = threading.Lock()  # one ONNX session; a stream and a recording take turns
        self.post = {name: float(spec.get(name, value)) for name, value in DEFAULT_POSTPROCESSING.items()}
        self.min_speaker_s = float(spec.get("min_speaker_s", 0.0))
        self._embedding_file = None
        embedding = spec.get("embedding", "embedding/*.onnx")
        found = sorted(path.glob(str(embedding)))
        if found:
            self._embedding_file = str(found[0])
        self._extractor: Any = None
        self._threads = opts.threads

    # -- probabilities ---------------------------------------------------------------------------
    def speaker_probabilities(self, samples: FloatArray, config: ChunkConfig = OFFLINE) -> FloatArray:
        """Probabilities ``(len(samples) // 160 + 1, 8)`` of a whole 16 kHz recording, chunk by chunk.

        Frame ``t`` is centred on ``t * 10 ms``; the last frame is the reference front-end's padding frame
        and is always zero."""
        n_mel = len(samples) // 160 + 1  # as the reference front-end: the last frame is padding
        valid = np.arange(n_mel) < len(samples) // 160
        frontend = NemoLogMel()
        mel = np.zeros((0, NEMO_MELS), np.float32)
        cache = SpeakerCache(config)
        out: list[FloatArray] = []
        chunk = config.chunk_length * SUBSAMPLING
        window = (config.chunk_length + config.right_context) * SUBSAMPLING
        fed = 0
        piece = 16000 * 30
        start = 0
        while start < n_mel:
            # compute the mel frames this chunk needs (and no more: memory stays flat on long audio)
            while len(mel) < min(window, n_mel - start) and fed < len(samples):
                mel = np.concatenate([mel, frontend.push(samples[fed : fed + piece])])
                fed += piece
            if fed >= len(samples) and len(mel) < min(window, n_mel - start):
                tail = frontend.finish()
                mel = np.concatenate([mel, tail, np.zeros((1, NEMO_MELS), np.float32)])  # + the padding frame
                fed = len(samples) + 1
            features = mel[: min(window, n_mel - start)]
            n_chunk = min(config.chunk_length, -(-(n_mel - start) // SUBSAMPLING))
            with self._lock:
                probs = _step(self._run, cache, features, valid[start : start + len(features)], n_chunk)
            out.append(probs)
            mel = mel[chunk:]
            start += chunk
        probs = np.concatenate(out)[:n_mel]
        probs[~valid] = 0.0  # the padding frame (as the reference applies its attention mask)
        return probs

    def stream(self, mode: str = "low_latency") -> NemotronStream:
        """A live stream in one of the card's latency modes (``low_latency`` 1.04 s, ``very_low_latency``
        0.64 s, ``ultra_low_latency`` 0.32 s)."""
        if mode not in STREAMING:
            raise ValueError(f"unknown streaming mode {mode!r}; one of {', '.join(STREAMING)}")
        return NemotronStream(self._run, STREAMING[mode], self._lock)

    # -- DiarizationModel ------------------------------------------------------------------------
    def diarize_detailed(
        self, samples: FloatArray, num_speakers: int | None = None, threshold: float | None = None
    ) -> DiarizationResult:
        from bolo.diarize import activity_to_turns, limit_speakers

        probs = self.speaker_probabilities(samples)
        post = dict(self.post)
        if threshold is not None:
            post["threshold"] = threshold
            post["offset"] = min(post.get("offset", threshold), threshold)
        activity = SpeakerActivity(probs, FRAME_S)
        turns, activity = activity_to_turns(activity, min_speaker_s=self.min_speaker_s, **post)
        if num_speakers and len({t.speaker for t in turns}) > num_speakers:
            turns, activity = limit_speakers(turns, activity, num_speakers)
        return DiarizationResult(turns, activity)

    def diarize(
        self, samples: FloatArray, num_speakers: int | None = None, threshold: float | None = None
    ) -> list[SpeakerTurn]:
        return self.diarize_detailed(samples, num_speakers, threshold).turns

    def embed(self, samples: FloatArray) -> FloatArray:
        if self._embedding_file is None:
            raise RuntimeError("this Nemotron diarization model has no speaker-embedding model (embedding/*.onnx)")
        if self._extractor is None:
            import sherpa_onnx

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=self._embedding_file, num_threads=self._threads, provider="cpu"
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        stream = self._extractor.create_stream()
        stream.accept_waveform(16000, samples)
        stream.input_finished()
        vector = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return (vector / norm).astype(np.float32) if norm > 0 else vector


def _find(path: Path, pattern: str, fallback: str) -> Path:
    for p in (pattern, fallback):
        found = sorted(path.glob(p))
        if found:
            return found[0]
    raise FileNotFoundError(f"no ONNX model ({pattern}) in {path}")
