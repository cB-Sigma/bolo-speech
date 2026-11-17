"""Log-mel features (NumPy only), as Whisper and as NVIDIA NeMo compute them.

:func:`whisper_log_mel` is used by the end-of-turn model (:mod:`bolo.engines.onnx_turn`), which is a
Whisper-tiny encoder with a classification head and expects the input of Whisper's feature extractor:
16 kHz audio, 25 ms Hann windows every 10 ms, 80 Slaney-scale mel bands, log10, clamped to 8 below the
peak and scaled to roughly [-1, 1].

:class:`NemoLogMel` is the front-end of NeMo's speech models as used by Nemotron-3-Diarization
(:mod:`bolo.engines.nemotron`) and the MarbleNet voice-activity detector (:mod:`bolo.engines.frame_vad`):
pre-emphasis 0.97, a 25 ms symmetric Hann window inside a 512-point FFT every 10 ms, zero-padded centred
frames, 128 (Nemotron) or 80 (MarbleNet) Slaney-scale mel bands, natural log with a 2^-24 guard and no
normalisation. It works on whole recordings and on audio that arrives piece by piece,
giving the same frames either way.
"""

from __future__ import annotations

import numpy as np

from bolo.engines.base import FloatArray

N_FFT = 400
HOP = 160
N_MELS = 80
RATE = 16000


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    mel = 3.0 * freq / 200.0
    log_region = freq >= 1000.0
    mel[log_region] = 15.0 + np.log(freq[log_region] / 1000.0) * (27.0 / np.log(6.4))
    return mel


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    freq = 200.0 * mel / 3.0
    log_region = mel >= 15.0
    freq[log_region] = 1000.0 * np.exp(np.log(6.4) / 27.0 * (mel[log_region] - 15.0))
    return freq


def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT, rate: int = RATE) -> np.ndarray:
    """Triangular Slaney-style filters with area normalisation, shape ``(n_fft // 2 + 1, n_mels)``."""
    fft_freqs = np.linspace(0.0, rate / 2.0, n_fft // 2 + 1)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(np.array([0.0]))[0], _hz_to_mel(np.array([rate / 2.0]))[0], n_mels + 2))
    widths = np.diff(edges)
    slopes = edges[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / widths[:-1]
    up = slopes[:, 2:] / widths[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    filters *= 2.0 / (edges[2:] - edges[:-2])
    return filters


_WINDOW = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic Hann
_FILTERS = {N_MELS: mel_filterbank()}
_WHISPER_BLOCK = 6000  # STFT frames per block (60 s): bounds the memory used on long recordings


def whisper_log_mel(audio: FloatArray, *, normalize: bool = False, n_mels: int = N_MELS) -> FloatArray:
    """Features of shape ``(n_mels, len(audio) // 160)``. ``normalize`` first gives the waveform zero mean and unit
    variance. ``n_mels`` is 80 (Whisper up to large-v2) or 128 (large-v3 and its derivatives)."""
    x = np.asarray(audio, dtype=np.float64)
    if normalize and len(x):
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    filters = _FILTERS.get(n_mels)
    if filters is None:
        filters = _FILTERS[n_mels] = mel_filterbank(n_mels)
    padded = np.pad(x, N_FFT // 2, mode="reflect")
    count = 1 + (len(padded) - N_FFT) // HOP - 1  # Whisper drops the last frame
    log = np.empty((max(0, count), n_mels))
    for first in range(0, count, _WHISPER_BLOCK):
        n = min(_WHISPER_BLOCK, count - first)
        index = np.arange(N_FFT)[None, :] + HOP * (first + np.arange(n))[:, None]
        power = np.abs(np.fft.rfft(padded[index] * _WINDOW, axis=1)) ** 2
        log[first : first + n] = np.log10(np.maximum(power @ filters, 1e-10))
    if count > 0:
        log = np.maximum(log, float(np.max(log)) - 8.0)
    return np.ascontiguousarray(((log + 4.0) / 4.0).T, dtype=np.float32)


# ---------------------------------------------------------------------------------------------------
# NeMo front-end (Nemotron-3-Diarization)
# ---------------------------------------------------------------------------------------------------
NEMO_N_FFT = 512
NEMO_WIN = 400
NEMO_MELS = 128
NEMO_PREEMPHASIS = 0.97
NEMO_LOG_GUARD = 2.0**-24

# symmetric Hann window of 400 samples, centred in the 512-point frame (as torch.stft pads a short window)
_NEMO_WINDOW = np.zeros(NEMO_N_FFT)
_NEMO_WINDOW[(NEMO_N_FFT - NEMO_WIN) // 2 : (NEMO_N_FFT + NEMO_WIN) // 2] = np.hanning(NEMO_WIN)
_NEMO_FILTERS = {NEMO_MELS: mel_filterbank(NEMO_MELS, NEMO_N_FFT).astype(np.float32).astype(np.float64)}
_BLOCK_FRAMES = 2000  # frames per FFT block: bounds the memory used on long recordings


def _nemo_filters(n_mels: int) -> np.ndarray:
    filters = _NEMO_FILTERS.get(n_mels)
    if filters is None:
        filters = _NEMO_FILTERS[n_mels] = mel_filterbank(n_mels, NEMO_N_FFT).astype(np.float32).astype(np.float64)
    return filters


def _nemo_frames(padded: np.ndarray, first: int, count: int, n_mels: int = NEMO_MELS) -> FloatArray:
    """Log-mel frames ``first .. first + count - 1`` of a pre-emphasised signal already padded by ``n_fft // 2``."""
    filters = _nemo_filters(n_mels)
    out = np.empty((count, n_mels), dtype=np.float32)
    for a in range(0, count, _BLOCK_FRAMES):
        n = min(_BLOCK_FRAMES, count - a)
        index = np.arange(NEMO_N_FFT)[None, :] + HOP * (first + a + np.arange(n))[:, None]
        power = np.abs(np.fft.rfft(padded[index] * _NEMO_WINDOW, axis=1)) ** 2
        out[a : a + n] = np.log(power @ filters + NEMO_LOG_GUARD)
    return out


class NemoLogMel:
    """NeMo log-mel frames of a signal that may arrive in pieces.

    Frame ``t`` covers samples ``t * 160 - 256 .. t * 160 + 255`` (zero before the start), so it is ready once
    ``t * 160 + 256`` samples have arrived. :meth:`push` returns the frames that became ready; :meth:`finish`
    returns the rest, ``len(audio) // 160`` frames in all, which is also what :func:`nemo_log_mel` gives.
    Only the samples still needed are kept, so a stream of any length uses constant memory. ``n_mels`` is 128
    (Nemotron-3-Diarization) or 80 (MarbleNet VAD).
    """

    def __init__(self, n_mels: int = NEMO_MELS) -> None:
        self.n_mels = n_mels
        self._buf = np.zeros(NEMO_N_FFT // 2, dtype=np.float64)  # the zero padding of the first frames
        self._buf_start = -(NEMO_N_FFT // 2)  # sample index (of the audio) of _buf[0]
        self._last = 0.0  # last raw sample, for the pre-emphasis across pieces
        self._received = 0
        self.frames_out = 0

    def push(self, samples: FloatArray) -> FloatArray:
        x = np.asarray(samples, dtype=np.float32).astype(np.float64)
        if len(x):
            emph = np.empty_like(x)
            emph[0] = x[0] - (NEMO_PREEMPHASIS * self._last if self._received else 0.0)
            emph[1:] = x[1:] - NEMO_PREEMPHASIS * x[:-1]
            # torch computes the pre-emphasis in float32
            emph = emph.astype(np.float32).astype(np.float64)
            self._last = float(x[-1])
            self._received += len(x)
            self._buf = np.concatenate([self._buf, emph])
        ready = max(0, (self._received - NEMO_N_FFT // 2) // HOP + 1 - self.frames_out)
        return self._take(ready)

    def finish(self) -> FloatArray:
        """The remaining frames, the end padded with zeros (``len(audio) // 160`` frames in all)."""
        self._buf = np.concatenate([self._buf, np.zeros(NEMO_N_FFT // 2)])
        return self._take(max(0, self._received // HOP - self.frames_out))

    def _take(self, count: int) -> FloatArray:
        if count <= 0:
            return np.zeros((0, self.n_mels), dtype=np.float32)
        offset = self.frames_out * HOP - (NEMO_N_FFT // 2) - self._buf_start
        frames = _nemo_frames(self._buf[offset:], 0, count, self.n_mels)
        self.frames_out += count
        drop = self.frames_out * HOP - (NEMO_N_FFT // 2) - self._buf_start
        if drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start += drop
        return frames


def nemo_log_mel(audio: FloatArray, n_mels: int = NEMO_MELS) -> FloatArray:
    """NeMo log-mel features of a whole recording, shape ``(len(audio) // 160, n_mels)``, float32."""
    stream = NemoLogMel(n_mels)
    head = stream.push(audio)
    return np.concatenate([head, stream.finish()]) if len(audio) else head


# ---------------------------------------------------------------------------------------------------
# Kaldi filterbank (speaker embedders: 3D-Speaker and WeSpeaker CAM++ run by ONNX Runtime)
# ---------------------------------------------------------------------------------------------------
KALDI_WIN = 400
KALDI_N_FFT = 512
_KALDI_FILTERS: dict[tuple[int, float], np.ndarray] = {}


def _kaldi_mel_filters(n_mels: int, high_freq: float) -> np.ndarray:
    """Kaldi's triangular mel filters (mel = 1127 ln(1 + f / 700), 20 Hz to ``high_freq``), ``(n_fft // 2 + 1, n_mels)``."""
    key = (n_mels, high_freq)
    if key not in _KALDI_FILTERS:

        def mel(f: np.ndarray) -> np.ndarray:
            return 1127.0 * np.log(1.0 + f / 700.0)

        edges = np.linspace(mel(np.array(20.0)), mel(np.array(high_freq)), n_mels + 2)
        bins = mel(np.arange(KALDI_N_FFT // 2 + 1) * RATE / KALDI_N_FFT)
        filters = np.zeros((KALDI_N_FFT // 2 + 1, n_mels))
        for i in range(n_mels):
            left, centre, right = edges[i], edges[i + 1], edges[i + 2]
            filters[:, i] = np.maximum(
                0.0, np.minimum((bins - left) / (centre - left), (right - bins) / (right - centre))
            )
        _KALDI_FILTERS[key] = filters
    return _KALDI_FILTERS[key]


def kaldi_fbank(
    audio: FloatArray, *, n_mels: int = 80, high_freq: float = 0.0, scale: float = 1.0, mean_norm: bool = True
) -> FloatArray:
    """Log mel filterbank as Kaldi computes it (torchaudio.compliance.kaldi.fbank with dither 0): 25 ms Povey windows
    every 10 ms without padding at the edges, DC removed, pre-emphasis 0.97, a 512-point power spectrum, ``n_mels``
    bands from 20 Hz to ``high_freq`` (0 or less: that much below 8 kHz). ``scale`` multiplies the samples first
    (WeSpeaker uses 16-bit sample values, 3D-Speaker -1..1); ``mean_norm`` subtracts each band's mean over the
    recording. Shape ``(frames, n_mels)``, float32."""
    x = np.asarray(audio, dtype=np.float64) * scale
    count = 1 + (len(x) - KALDI_WIN) // HOP if len(x) >= KALDI_WIN else 0
    if count <= 0:
        return np.zeros((0, n_mels), dtype=np.float32)
    top = high_freq if high_freq > 0 else RATE / 2 + high_freq
    filters = _kaldi_mel_filters(n_mels, top)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(KALDI_WIN) / (KALDI_WIN - 1))) ** 0.85
    out = np.empty((count, n_mels))
    for first in range(0, count, _BLOCK_FRAMES):
        n = min(_BLOCK_FRAMES, count - first)
        frames = x[np.arange(KALDI_WIN)[None, :] + HOP * (first + np.arange(n))[:, None]]
        frames = frames - frames.mean(axis=1, keepdims=True)
        frames = np.concatenate([frames[:, :1] * (1 - 0.97), frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1)
        power = np.abs(np.fft.rfft(frames * window, KALDI_N_FFT, axis=1)) ** 2
        out[first : first + n] = np.log(np.maximum(power @ filters, np.finfo(np.float32).eps))
    if mean_norm:
        out -= out.mean(axis=0, keepdims=True)
    return out.astype(np.float32)
