"""Sample-rate conversion.

Uses libsoxr through the ``soxr`` wheel. If it cannot be imported a small
numpy fallback (anti-alias FIR followed by linear interpolation) keeps the
server usable, at lower quality.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]

try:  # pragma: no cover - import guard
    import soxr as _soxr

    HAVE_SOXR = True
except Exception:  # pragma: no cover
    _soxr = None
    HAVE_SOXR = False


def _lowpass(x: FloatArray, cutoff: float) -> FloatArray:
    """Windowed-sinc low-pass; ``cutoff`` is a fraction of the sampling rate (0..0.5)."""
    taps = 63
    n = np.arange(taps) - (taps - 1) / 2
    h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
    h /= h.sum()
    return np.convolve(x, h, mode="same").astype(np.float32)


def _resample_numpy(x: FloatArray, sr_in: int, sr_out: int) -> FloatArray:
    if len(x) == 0:
        return x
    if sr_out < sr_in:
        x = _lowpass(x, 0.5 * sr_out / sr_in * 0.92)
    n_out = round(len(x) * sr_out / sr_in)
    pos = np.arange(n_out, dtype=np.float64) * (sr_in / sr_out)
    return np.interp(pos, np.arange(len(x)), x).astype(np.float32)


def resample(samples: npt.NDArray[np.floating], sr_in: int, sr_out: int, *, force_numpy: bool = False) -> FloatArray:
    x = np.ascontiguousarray(samples, dtype=np.float32)
    if sr_in == sr_out or len(x) == 0:
        return x
    if HAVE_SOXR and not force_numpy:
        out: FloatArray = _soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32)
        return out
    return _resample_numpy(x, sr_in, sr_out)


class StreamResampler:
    """Stateful resampler for chunked audio (no clicks at chunk boundaries)."""

    def __init__(self, sr_in: int, sr_out: int) -> None:
        self.sr_in = sr_in
        self.sr_out = sr_out
        self._stream = None
        self._tail = np.zeros(0, dtype=np.float32)
        if sr_in != sr_out and HAVE_SOXR:
            self._stream = _soxr.ResampleStream(sr_in, sr_out, 1, dtype="float32", quality="HQ")

    def process(self, samples: FloatArray, last: bool = False) -> FloatArray:
        if self.sr_in == self.sr_out:
            return samples
        if self._stream is not None:
            out: FloatArray = self._stream.resample_chunk(samples, last=last).astype(np.float32)
            return out
        # Fallback: interpolate with a one-sample overlap so chunks join continuously.
        x = np.concatenate([self._tail, samples])
        if len(x) < 2:
            self._tail = x
            return np.zeros(0, dtype=np.float32)
        ratio = self.sr_in / self.sr_out
        n_out = int((len(x) - 1) / ratio) if not last else round(len(x) / ratio)
        pos = np.arange(n_out, dtype=np.float64) * ratio
        out = np.interp(pos, np.arange(len(x)), x).astype(np.float32)
        consumed = int(n_out * ratio)
        self._tail = x[consumed:] if not last else np.zeros(0, dtype=np.float32)
        return out
