"""SpeechBrain's SepFormer and MetricGAN+ as ONNX graphs, without PyTorch (``onnxruntime`` engine).

* kind ``sepformer_onnx``: SepFormer (Subakan et al., ICASSP 2021), a dual-path transformer that masks a learned
  filterbank. One graph (``sepformer.onnx``: ``mixture [1, T]`` -> ``sources [1, T', S]``) holds the encoder, the
  masking network and the decoder of SpeechBrain's ``SepformerSeparation.separate_batch``. As a ``separation``
  model it is a blind separator (``S`` tracks, two or three voices); as a ``denoise`` model (the DNS4 enhancement
  checkpoint, ``S`` = 1) it is a speech enhancer. The graphs were exported with ``tools/export_speechbrain_onnx.py``.
* kind ``speechbrain_ctc``: a SpeechBrain ``EncoderASR`` recogniser (wav2vec 2.0 encoder, a small DNN and a CTC
  layer) as one graph (``asr.onnx``: ``wav [1, T]`` at 16 kHz -> ``logprobs [1, frames, tokens]``, 20 ms frames,
  with SpeechBrain's normalisation of the waveform and of the encoder output inside), greedy CTC decoding here, as
  SpeechBrain's ``ctc_greedy_decode``, and its SentencePiece units (``tokens.txt``) joined back into words.
* kind ``metricgan_onnx``: MetricGAN+ (Fu et al., Interspeech 2021), a bidirectional LSTM that masks the log
  magnitude spectrum; the graph is the LSTM (``metricgan.onnx``: ``features [1, frames, 257]`` ->
  ``mask [1, frames, 257]``), the STFT and its inverse with the noisy phase are here, in NumPy, as in SpeechBrain's
  ``SpectralMaskEnhancement`` (512 points, hop 256, periodic Hamming window, centred).

SepFormer was trained with a scale-invariant loss, so the level of what it returns is arbitrary: each track is
scaled to the least-squares fit of the input (a denoised recording to the noisy one, a separated voice to the
mixture), so that the output sits at the input's level and ``denoise`` strengths below 1 mix like with like.
SpeechBrain's MetricGAN+ normalises its output to a peak of 1; bolo keeps the level the mask gives instead.

Both are offline models (non-causal): a live stream enhances each piece with the half second before it. SepFormer's
memory grows with the square of the length (6 GB for a minute at 8 kHz), so a long recording is denoised in
windows of ``window_s`` (10 s) overlapping by ``overlap_s`` (1 s) and cross-faded.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo.audio.resample import resample
from bolo.engines.base import (
    DenoiseModel,
    FloatArray,
    LoadOptions,
    SeparationModel,
    STTModel,
    STTRequest,
    STTResult,
)

KERNEL, STRIDE = 16, 8  # SepFormer's encoder: a 1-D convolution of 16 samples, stride 8
N_FFT, HOP = 512, 256  # MetricGAN+'s STFT at 16 kHz: 32 ms windows, 16 ms hop


def _session(path: Path, spec: dict[str, Any], opts: LoadOptions, pattern: str) -> Any:
    from bolo.engines.lid_onnx import _find, _options
    from bolo.ort_providers import create_session, providers_for

    file = _find(path, str(spec.get("model") or pattern))
    return create_session(str(file), _options(max(1, opts.threads)), providers_for(opts.device))


def run_sepformer(session: Any, x: FloatArray) -> FloatArray:
    """The graph's sources for ``x`` (mono, at the model's rate): ``[len(x), S]``."""
    n = len(x)
    # as separate_batch: the encoder takes whole strides (the last few samples are dropped) and the tracks are
    # padded with zeros to the input's length
    size = KERNEL + max(0, n - KERNEL) // STRIDE * STRIDE
    audio = np.zeros((1, size), dtype=np.float32)
    audio[0, : min(n, size)] = x[:size]
    out = np.asarray(session.run(None, {"mixture": audio})[0], dtype=np.float32)[0]  # (time, sources)
    if len(out) < n:
        out = np.pad(out, ((0, n - len(out)), (0, 0)))
    return np.ascontiguousarray(out[:n])


def _fit_level(track: FloatArray, reference: FloatArray) -> FloatArray:
    """``track`` scaled to its least-squares fit of ``reference``."""
    energy = float(np.dot(track, track))
    if energy <= 1e-12:
        return track
    return (track * (float(np.dot(track, reference)) / energy)).astype(np.float32)


def _peak_safe(tracks: list[FloatArray]) -> list[FloatArray]:
    peak = max((float(np.max(np.abs(t))) for t in tracks if t.size), default=0.0)
    return [(t / peak * 0.99).astype(np.float32) for t in tracks] if peak > 0.99 else tracks


class SepFormerSeparation(SeparationModel):
    """Blind separation into the graph's number of tracks; 16 kHz in and out, resampled to the model's rate."""

    mode = "blind"
    sample_rate = 16000

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        self._session = _session(path, spec, opts, "sepformer*.onnx")
        self.model_rate = int(spec.get("model_sample_rate", 8000))
        self.num_speakers = int(spec.get("speakers", 2))
        self._lock = threading.Lock()

    def separate(self, mixture: FloatArray) -> list[FloatArray]:
        x = np.asarray(mixture, dtype=np.float32)
        if not len(x):
            return [x.copy() for _ in range(self.num_speakers)]
        inside = resample(x, self.sample_rate, self.model_rate) if self.model_rate != self.sample_rate else x
        with self._lock:
            est = run_sepformer(self._session, inside)
        tracks = []
        for k in range(est.shape[1]):
            track = est[:, k]
            if self.model_rate != self.sample_rate:
                track = resample(track, self.model_rate, self.sample_rate)
            track = np.pad(track, (0, max(0, len(x) - len(track))))[: len(x)].astype(np.float32)
            tracks.append(_fit_level(track, x))
        return _peak_safe(tracks)


class SepFormerDenoiser(DenoiseModel):
    """SepFormer trained for enhancement (one source), at its own rate, in cross-faded windows."""

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        self._session = _session(path, spec, opts, "sepformer*.onnx")
        self.sample_rate = int(spec.get("model_sample_rate", 16000))
        self.window = int(float(spec.get("window_s", 10.0)) * self.sample_rate)
        self.overlap = min(self.window // 2, int(float(spec.get("overlap_s", 1.0)) * self.sample_rate))
        self._lock = threading.Lock()

    def _one(self, x: FloatArray) -> FloatArray:
        with self._lock:
            y = run_sepformer(self._session, x)[:, 0]
        return _fit_level(y, x)

    def denoise(self, samples: FloatArray) -> FloatArray:
        x = np.ascontiguousarray(samples, dtype=np.float32)
        n = len(x)
        if n <= self.window:
            return self._one(x) if n else x
        out = np.zeros(n, dtype=np.float32)
        weight = np.zeros(n, dtype=np.float32)
        step = self.window - self.overlap
        fade = np.linspace(0.0, 1.0, self.overlap + 2, dtype=np.float32)[1:-1]
        for start in range(0, n - self.overlap, step):
            end = min(n, start + self.window)
            piece = self._one(x[start:end])
            w = np.ones(end - start, dtype=np.float32)
            if start > 0:
                w[: self.overlap] = fade[: end - start]
            if end < n:
                w[-self.overlap :] = np.minimum(w[-self.overlap :], fade[::-1])
            out[start:end] += piece * w
            weight[start:end] += w
        return (out / np.maximum(weight, 1e-6)).astype(np.float32)


def _window() -> FloatArray:
    return (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float64)  # periodic Hamming


def _stft(x: FloatArray) -> Any:
    """torch.stft(center=True, pad_mode="constant", onesided): ``[frames, 257]`` complex."""
    padded = np.pad(np.asarray(x, dtype=np.float64), N_FFT // 2)
    frames = 1 + (len(padded) - N_FFT) // HOP
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    return np.fft.rfft(padded[index] * _window(), axis=1)


def _istft(spec: Any, length: int) -> FloatArray:
    """torch.istft(center=True, length=length) of ``[frames, 257]``."""
    window = _window()
    frames = np.fft.irfft(spec, n=N_FFT, axis=1) * window
    total = N_FFT + HOP * (len(spec) - 1)
    out = np.zeros(total)
    envelope = np.zeros(total)
    for i, frame in enumerate(frames):
        out[i * HOP : i * HOP + N_FFT] += frame
        envelope[i * HOP : i * HOP + N_FFT] += window**2
    out = out / np.where(envelope > 1e-11, envelope, 1.0)
    out = out[N_FFT // 2 :]
    return np.pad(out, (0, max(0, length - len(out))))[:length].astype(np.float32)


def stft_features(x: FloatArray) -> FloatArray:
    """MetricGAN+'s input: log1p of the STFT magnitude, ``[frames, 257]``."""
    out: FloatArray = np.log1p(np.sqrt(np.abs(_stft(x)) ** 2 + 1e-14)).astype(np.float32)
    return out


def metricgan_enhance(session: Any, x: FloatArray) -> FloatArray:
    """The mask applied to the log magnitude, back to audio with the noisy phase (as long as ``x``)."""
    spec = _stft(x)
    feats = np.log1p(np.sqrt(np.abs(spec) ** 2 + 1e-14)).astype(np.float32)
    mask = np.asarray(session.run(None, {"features": feats[None]})[0], dtype=np.float32)[0]
    magnitude = np.expm1(mask * feats).astype(np.float64)
    return _istft(magnitude * np.exp(1j * np.angle(spec)), len(x))


class MetricGanPlus(DenoiseModel):
    sample_rate = 16000

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        self._session = _session(path, spec, opts, "metricgan*.onnx")
        self._lock = threading.Lock()

    def denoise(self, samples: FloatArray) -> FloatArray:
        x = np.ascontiguousarray(samples, dtype=np.float32)
        if len(x) < 2:
            return x
        with self._lock:
            return metricgan_enhance(self._session, x)


#: SentencePiece's control pieces, which its decoder leaves out; ``<unk>`` comes out as " ⁇ "
_CONTROL = {"<s>", "</s>", "<pad>"}


def read_tokens(file: Path) -> list[str]:
    """``tokens.txt``: ``piece id`` per line (sherpa-onnx's form), as a list indexed by id."""
    from bolo.engines.ctc_align import read_vocab

    vocab = read_vocab(file)
    pieces = [""] * (max(vocab.values()) + 1)
    for piece, index in vocab.items():
        pieces[index] = piece
    return pieces


def ctc_greedy(logprobs: FloatArray, blank: int = 0) -> list[int]:
    """The best path with repeats merged and blanks removed (SpeechBrain's ``ctc_greedy_decode``)."""
    best = np.argmax(logprobs, axis=-1)
    keep = np.ones(len(best), dtype=bool)
    keep[1:] = best[1:] != best[:-1]
    return [int(i) for i in best[keep] if i != blank]


def detokenize(ids: list[int], pieces: list[str]) -> str:
    """SentencePiece's ``decode_ids`` for these pieces: ``▁`` starts a word."""
    out = []
    for i in ids:
        piece = pieces[i] if 0 <= i < len(pieces) else "<unk>"
        if piece in _CONTROL:
            continue
        out.append(" \u2047 " if piece == "<unk>" else piece)
    return " ".join("".join(out).replace("\u2581", " ").split())


class SpeechBrainCtc(STTModel):
    """A SpeechBrain wav2vec 2.0 CTC recogniser for one language, greedy decoding, 16 kHz."""

    sample_rate = 16000
    max_chunk_s = 30.0

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from bolo.engines.lid_onnx import _find

        self._session = _session(path, spec, opts, "asr*.onnx")
        self.pieces = read_tokens(_find(path, str(spec.get("tokens") or "tokens.txt")))
        self.blank = int(spec.get("blank", 0))
        #: the one language the model knows (it cannot be told another)
        self.language = str(spec.get("language") or "") or None
        self._lock = threading.Lock()

    def logprobs(self, samples: FloatArray) -> FloatArray:
        x = np.ascontiguousarray(samples, dtype=np.float32)
        if len(x) < 400:  # the feature encoder needs 25 ms
            x = np.pad(x, (0, 400 - len(x)))
        with self._lock:
            out: FloatArray = np.asarray(self._session.run(None, {"wav": x[None]})[0], dtype=np.float32)[0]
        return out

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        out = []
        for r in requests:
            text = detokenize(ctc_greedy(self.logprobs(r.samples), self.blank), self.pieces) if len(r.samples) else ""
            out.append(STTResult(text, self.language or r.language))
        return out
