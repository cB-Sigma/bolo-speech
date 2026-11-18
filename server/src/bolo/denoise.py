"""Noise suppression before speech recognition (opt-in: ``denoise``).

A speech-enhancement model (``task: denoise`` in the catalog) removes background noise from the recording before
the recogniser hears it. It runs at the model's own sample rate: the audio is resampled to it and back.

``denoise`` takes, wherever it is accepted (a request, a Realtime session, a model pin, the server setting):

* ``true`` / ``"on"``: the default denoiser (``default_denoise``) at full strength;
* ``false`` / ``"off"`` / ``0``: no denoising, whatever the defaults say;
* a number between 0 and 1: the default denoiser, its output mixed with the original audio at that weight
  (``0.5`` keeps half of the original; a gentler setting loses less speech);
* a model id or alias of a denoise model;
* (JSON) an object ``{"model": ..., "strength": ...}``.

The first of request, model pin, catalog option and server setting that says anything decides. Denoising is a
trade: it helps recognisers on loud, steady noise and can hurt on clean audio, on speech that the enhancement
model distorts, and with recognisers trained on noisy data (Whisper is). A denoiser that cannot run is skipped
with a warning; the request goes on with the original audio.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from bolo.audio.resample import resample
from bolo.engines.base import DenoiseModel, FloatArray
from bolo.errors import BoloError, InvalidRequest

if TYPE_CHECKING:
    from bolo.core import Bolo
    from bolo.obs.trace import Trace

log = logging.getLogger("bolo")

_ON = {"true", "on", "yes", "default", "auto"}
_OFF = {"false", "off", "no", "none", "0", "0.0"}


@dataclass(frozen=True)
class DenoiseOptions:
    """What ``denoise`` asked for: ``enabled`` False switches it off; ``model`` None means the default model."""

    enabled: bool = True
    model: str | None = None
    strength: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "model": self.model, "strength": self.strength}


OFF = DenoiseOptions(enabled=False, strength=0.0)


def _strength(value: Any, param: str) -> float:
    try:
        strength = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidRequest(f"{param}: strength must be a number between 0 and 1", param=param) from exc
    if not 0.0 <= strength <= 1.0:
        raise InvalidRequest(f"{param}: strength must be between 0 and 1, got {strength:g}", param=param)
    return strength


def parse_denoise(value: Any, param: str = "denoise", strength: Any = None) -> DenoiseOptions | None:
    """``denoise`` as given (see the module's docstring); None when it says nothing (absent or empty).

    ``strength`` is a separate strength field (``denoise_strength``), applied on top of what ``value`` says."""
    opts: DenoiseOptions | None
    if value is None or (isinstance(value, str) and not value.strip()):
        opts = None
    elif isinstance(value, bool):
        opts = DenoiseOptions() if value else OFF
    elif isinstance(value, (int, float)):
        s = _strength(value, param)
        opts = DenoiseOptions(strength=s) if s > 0 else OFF
    elif isinstance(value, dict):
        if value.get("enabled") is False:
            opts = OFF
        else:
            model = value.get("model")
            if model is not None and not isinstance(model, str):
                raise InvalidRequest(f"{param}.model must be a model id", param=f"{param}.model")
            raw = value.get("strength")
            s = 1.0 if raw is None else _strength(raw, f"{param}.strength")
            opts = DenoiseOptions(model=(model or "").strip() or None, strength=s) if s > 0 else OFF
    elif isinstance(value, str):
        text = value.strip()
        low = text.lower()
        if low in _OFF:
            opts = OFF
        elif low in _ON:
            opts = DenoiseOptions()
        else:
            try:
                number = float(low)
            except ValueError:
                opts = DenoiseOptions(model=text)
            else:
                s = _strength(number, param)
                opts = DenoiseOptions(strength=s) if s > 0 else OFF
    else:
        raise InvalidRequest(
            f"{param} must be true, false, a strength between 0 and 1 or a denoise model id", param=param
        )
    if strength is not None and not (isinstance(strength, str) and not strength.strip()):
        s = _strength(strength, f"{param}_strength")
        if opts is None:
            opts = DenoiseOptions(strength=s)
        if opts.enabled:
            opts = DenoiseOptions(model=opts.model, strength=s) if s > 0 else OFF
    return opts


def mix(original: FloatArray, enhanced: FloatArray, strength: float) -> FloatArray:
    """``strength`` of the enhanced audio plus the rest of the original (both at the same rate and length)."""
    if strength >= 1.0:
        return np.ascontiguousarray(enhanced, dtype=np.float32)
    out = strength * enhanced + (1.0 - strength) * original
    return np.ascontiguousarray(out, dtype=np.float32)


def _fit(audio: FloatArray, n: int) -> FloatArray:
    if len(audio) < n:
        audio = np.pad(audio, (0, n - len(audio)))
    return np.ascontiguousarray(audio[:n], dtype=np.float32)


async def enhance(core: Bolo, samples: FloatArray, rate: int, model_id: str, trace: Trace) -> tuple[FloatArray, int]:
    """``samples`` (mono float32 at ``rate``) enhanced by the denoise model ``model_id``, at the same rate and length,
    and the model's own sample rate. Raises when the model cannot run (the standalone ``/v1/audio/denoise``)."""
    if len(samples) == 0:
        return samples, rate
    async with core.manager.use(model_id) as entry:
        model = entry.model
        assert isinstance(model, DenoiseModel)
        native = int(model.sample_rate)
        big = len(samples) > rate * 30
        source = samples
        if native != rate:
            source = (
                await asyncio.to_thread(resample, samples, rate, native) if big else resample(samples, rate, native)
            )
        enhanced, _ = await entry.queue.submit("denoise", np.ascontiguousarray(source, dtype=np.float32))
        enhanced = _fit(np.asarray(enhanced, dtype=np.float32), len(source))
        if native != rate:
            back = (
                await asyncio.to_thread(resample, enhanced, native, rate) if big else resample(enhanced, native, rate)
            )
            enhanced = _fit(back, len(samples))
    return enhanced, native


async def apply(core: Bolo, samples: FloatArray, rate: int, opts: DenoiseOptions, trace: Trace) -> FloatArray:
    """``samples`` (mono float32 at ``rate``) with ``opts.model`` applied, at the same rate and length.

    The model runs at its own sample rate. When it cannot run (its engine is missing, it is not installed and
    cannot be pulled, it fails), the original audio comes back and the trace says why."""
    assert opts.model is not None
    record: dict[str, Any] = {"model": opts.model, "strength": opts.strength, "status": "applied"}
    trace.extra["denoise"] = record
    if len(samples) == 0:
        return samples
    t0 = trace.now()
    try:
        enhanced, native = await enhance(core, samples, rate, opts.model, trace)
        record["sample_rate"] = native
    except asyncio.CancelledError:
        raise
    except BoloError as exc:
        return _skipped(record, exc.message, samples)
    except Exception as exc:
        return _skipped(record, f"{type(exc).__name__}: {exc}", samples)
    finally:
        trace.add_phase("denoise", t0, trace.now())
    seconds = len(samples) / rate if rate else 0.0
    record["ms"] = round((trace.now() - t0) * 1000.0, 1)
    if seconds > 0:
        record["rtf"] = round((trace.now() - t0) / seconds, 4)
    return mix(samples, enhanced, opts.strength)


def _skipped(record: dict[str, Any], detail: str, samples: FloatArray) -> FloatArray:
    record["status"] = "skipped"
    record["detail"] = detail
    log.warning("denoising with '%s' skipped, the original audio is used: %s", record["model"], detail)
    return samples


class _Resampler(Protocol):
    def process(self, samples: FloatArray) -> FloatArray: ...

    def finish(self) -> FloatArray: ...


class _Soxr:
    """soxr's streaming resampler where the rates are not whole multiples of each other."""

    def __init__(self, rate_in: int, rate_out: int) -> None:
        from bolo.audio.resample import StreamResampler

        self._r = StreamResampler(rate_in, rate_out)

    def process(self, samples: FloatArray) -> FloatArray:
        return self._r.process(samples)

    def finish(self) -> FloatArray:
        return self._r.process(np.zeros(0, dtype=np.float32), last=True)


class _Same:
    def process(self, samples: FloatArray) -> FloatArray:
        return samples

    def finish(self) -> FloatArray:
        return np.zeros(0, dtype=np.float32)


def _resampler(rate_in: int, rate_out: int) -> _Resampler:
    from bolo.audio.resample import LowDelayResampler

    if rate_in == rate_out:
        return _Same()
    if rate_out % rate_in == 0 or rate_in % rate_out == 0:
        return LowDelayResampler(rate_in, rate_out)  # 8, 16, 48 kHz: a millisecond, not soxr's 40 to 110
    return _Soxr(rate_in, rate_out)


class LiveDenoiser:
    """``denoise`` on a live stream (the speech-to-text WebSocket): pieces of audio at ``rate`` in, enhanced pieces at
    ``rate`` out, aligned with the input by sample count (sample ``k`` out is sample ``k`` in, enhanced).

    The model's stream (:meth:`DenoiseModel.create_stream`) keeps its state across pieces when the model has a
    streaming form (GTCRN and DPDFNet through sherpa-onnx's online denoiser, DeepFilterNet 3, RNNoise); other models
    enhance each piece with half a second of audio before it. The model runs at its own rate, resampled to it and
    back in step (with a short FIR filter when the rates are whole multiples of each other: 1 to 2 ms held back, where
    soxr's streaming filters hold 40 to 110 ms). ``strength`` below 1 mixes the original back in. :attr:`ms` collects
    the time each piece took; ``held_ms`` in :meth:`summary` is the most audio held back at once (the model's own delay,
    its frame and the resampling)."""

    def __init__(self, model: DenoiseModel, rate: int, strength: float) -> None:
        self.stream = model.create_stream()
        self.native = int(model.sample_rate)
        self.rate = rate
        self.strength = strength
        self._down = _resampler(rate, self.native)
        self._up = _resampler(self.native, rate)
        self._original = np.zeros(0, dtype=np.float32)  # input not yet matched by output
        self.ms: list[float] = []
        self._held = 0  # most audio held back at once (samples at ``rate``): the denoiser's own delay

    def _mix(self, out: FloatArray) -> FloatArray:
        n = min(len(out), len(self._original))
        out, original = out[:n], self._original[:n]
        self._original = self._original[n:]
        return mix(original, out, self.strength) if self.strength < 1.0 else np.ascontiguousarray(out, np.float32)

    def process(self, samples: FloatArray) -> FloatArray:
        import time

        t0 = time.perf_counter()
        samples = np.asarray(samples, dtype=np.float32)
        self._original = np.concatenate([self._original, samples])
        native = self._down.process(samples)
        enhanced = self.stream.process(native) if len(native) else native
        out = self._mix(self._up.process(enhanced) if len(enhanced) else np.zeros(0, np.float32))
        self.ms.append((time.perf_counter() - t0) * 1000.0)
        self._held = max(self._held, len(self._original))
        return out

    def finish(self) -> FloatArray:
        """The rest, at the end of the stream: in all, as many samples as were given."""
        native = self._down.finish()
        enhanced = np.concatenate([self.stream.process(native) if len(native) else native, self.stream.finish()])
        back = np.concatenate([self._up.process(enhanced), self._up.finish()])
        missing = len(self._original) - len(back)
        if missing > 0:  # resampling rounded the length down: the last few samples are the original's
            back = np.concatenate([back, self._original[len(back) :]])
        return self._mix(back)

    def summary(self) -> dict[str, Any]:
        if not self.ms:
            return {"chunks": 0}
        ms = np.asarray(self.ms)
        return {
            "chunks": len(ms),
            "mean_ms": round(float(ms.mean()), 2),
            "p95_ms": round(float(np.percentile(ms, 95)), 2),
            "max_ms": round(float(np.max(ms)), 2),
            "held_ms": round(1000.0 * self._held / self.rate, 1),
            "stateful": bool(self.stream.stateful),
        }
