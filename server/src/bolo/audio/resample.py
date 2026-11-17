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


class LowDelayResampler:
    """Streaming conversion between rates that are whole multiples of each other (16 kHz to 8 or 48 kHz and back)
    with a short linear-phase FIR filter, for live audio where soxr's streaming filters hold back 40 to 110 ms.

    The output is in step with the input by sample count: output sample ``k`` is input time ``k / rate_out``. The
    filter's own delay (``taps_per_phase`` samples of the lower rate: 1 ms at 16 kHz, 2 ms at 8 kHz) is cut off at the start and
    pushed out by :meth:`finish`, which returns the rest so that in all ``round(n * rate_out / rate_in)`` samples come
    out. Kaiser-windowed sinc, cut off at 45 % of the lower rate."""

    def __init__(self, rate_in: int, rate_out: int, taps_per_phase: int = 16) -> None:
        if rate_out % rate_in == 0:
            self.up, self.down = rate_out // rate_in, 1
        elif rate_in % rate_out == 0:
            self.up, self.down = 1, rate_in // rate_out
        else:
            raise ValueError(f"{rate_in} and {rate_out} Hz are not whole multiples of each other")
        self.rate_in, self.rate_out = rate_in, rate_out
        factor = max(self.up, self.down)
        taps = 2 * taps_per_phase * factor + 1
        n = np.arange(taps) - (taps - 1) / 2
        cutoff = 0.45 / factor  # of the higher rate
        h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(taps, 8.0)
        self._h = (h / h.sum() * self.up).astype(np.float32)
        self._delay = (taps - 1) // 2  # samples of the higher rate
        self._buf = np.zeros(taps - 1, dtype=np.float32)
        self._dropped = 0
        self._pos = 0  # samples of the higher rate produced after the delay (for picking every down-th)
        self.taken = 0
        self.given = 0

    def _filter(self, z: FloatArray) -> FloatArray:
        x = np.concatenate([self._buf, z])
        y = np.convolve(x, self._h, mode="valid").astype(np.float32)
        self._buf = x[len(x) - len(self._buf) :]
        cut = min(self._delay - self._dropped, len(y))
        self._dropped += cut
        y = y[cut:]
        if self.down > 1:
            start = (-self._pos) % self.down
            self._pos += len(y)
            y = y[start :: self.down]
        return y

    def _run(self, samples: FloatArray) -> FloatArray:
        x = np.asarray(samples, dtype=np.float32)
        if self.up > 1:
            z = np.zeros(len(x) * self.up, dtype=np.float32)
            z[:: self.up] = x
        else:
            z = x
        return self._filter(z) if len(z) else z

    def process(self, samples: FloatArray) -> FloatArray:
        self.taken += len(samples)
        out = self._run(samples)
        self.given += len(out)
        return np.ascontiguousarray(out, dtype=np.float32)

    def finish(self) -> FloatArray:
        need = round(self.taken * self.rate_out / self.rate_in) - self.given
        if need <= 0:
            return np.zeros(0, dtype=np.float32)
        out = self._run(np.zeros(self._delay // self.up + self.down + 1, dtype=np.float32))[:need]
        if len(out) < need:
            out = np.pad(out, (0, need - len(out)))
        self.given += len(out)
        return np.ascontiguousarray(out, dtype=np.float32)
