"""Log-mel features as Whisper computes them (NumPy only).

Used by the end-of-turn model (:mod:`bolo.engines.onnx_turn`), which is a Whisper-tiny encoder
with a classification head and expects the input of Whisper's feature extractor: 16 kHz audio,
25 ms Hann windows every 10 ms, 80 Slaney-scale mel bands, log10, clamped to 8 below the peak
and scaled to roughly [-1, 1].
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
_FILTERS = mel_filterbank()


def whisper_log_mel(audio: FloatArray, *, normalize: bool = False) -> FloatArray:
    """Features of shape ``(80, len(audio) // 160)``. ``normalize`` first gives the waveform zero mean and unit variance."""
    x = np.asarray(audio, dtype=np.float64)
    if normalize and len(x):
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    padded = np.pad(x, N_FFT // 2, mode="reflect")
    count = 1 + (len(padded) - N_FFT) // HOP
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(count)[:, None]
    power = np.abs(np.fft.rfft(padded[index] * _WINDOW, axis=1)) ** 2
    mel = np.maximum(power @ _FILTERS, 1e-10)
    log = np.log10(mel)[:-1]  # Whisper drops the last frame
    log = np.maximum(log, log.max() - 8.0)
    return np.ascontiguousarray(((log + 4.0) / 4.0).T, dtype=np.float32)
