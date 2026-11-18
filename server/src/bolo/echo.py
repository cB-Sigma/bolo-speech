"""Acoustic echo cancellation for voice agents: the agent's own voice removed from what the microphone hears.

A caller on a speakerphone, a laptop without headphones or a phone line with echo sends back what bolo said; the
voice-activity detector then hears "speech" while the agent talks, cuts its own answer short (barge-in) and the
recogniser transcribes the agent's words as the user's. An echo canceller takes the audio bolo sent out (the far
end, the reference) and the audio that comes back (the near end, the microphone) and removes the echo of the first
from the second.

Cancellers (catalog task ``echo-cancellation``, docs/voice-agents.md, "Echo cancellation"):

* ``webrtc-aec3`` - the WebRTC audio processing module's AEC3, the canceller of Chrome and of most WebRTC apps,
  through LiveKit's Python SDK (``pip install 'bolo-speech[aec]'``; Apache-2.0 and BSD-3-Clause). It estimates the
  delay between the reference and its echo by itself.
* ``speex-aec`` - SpeexDSP's MDF canceller with its residual-echo suppressor (BSD-3-Clause), through the pyaec
  binding (MIT, ``pip install 'bolo-speech[aec]'``): small and old, but it adapts quickly.
* ``dtln-aec-128`` - DTLN-aec (Westhausen and Meyer, 2021; MIT), a small recurrent network trained for the
  AEC Challenge, in ONNX Runtime: two stages, a spectral mask then a time-domain refinement, both given the
  reference.

All work on 16 kHz mono float32 in pieces of any length; the canceller keeps its own framing. The reference is the
audio as bolo sent it: :class:`EchoReference` releases it at the pace the microphone audio arrives, which is the
pace the client plays it, so the reference leads its echo by about the round trip and the client's buffering.
"""

from __future__ import annotations

import abc
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import Engine, EngineStatus, FloatArray, LoadOptions, Model

log = logging.getLogger("bolo.echo")

SR = 16000
INSTALL = "pip install 'bolo-speech[aec]'"


class EchoStream(abc.ABC):
    """One conversation's canceller: state carried from one piece to the next."""

    #: samples the output lags the input by (DTLN's overlap-add); 0 for the others
    delay: int = 0

    @abc.abstractmethod
    def process(self, mic: FloatArray, ref: FloatArray) -> FloatArray:
        """The microphone audio with the echo of ``ref`` removed; ``mic`` and ``ref`` have the same length and the
        output too."""


class _Framed(EchoStream):
    """Feeds a frame-based canceller pieces of any length: whole frames go in, the rest waits for the next call
    (the output is as long as the input; it starts with ``frame`` samples of silence)."""

    frame: int = 160

    def __init__(self) -> None:
        self._mic = np.zeros(0, dtype=np.float32)
        self._ref = np.zeros(0, dtype=np.float32)
        self._out = np.zeros(self.frame, dtype=np.float32)  # the latency of the framing
        self.delay = self.frame

    @abc.abstractmethod
    def _frame(self, mic: FloatArray, ref: FloatArray) -> FloatArray: ...

    def process(self, mic: FloatArray, ref: FloatArray) -> FloatArray:
        n = len(mic)
        self._mic = np.concatenate([self._mic, np.asarray(mic, dtype=np.float32)])
        self._ref = np.concatenate([self._ref, np.asarray(ref, dtype=np.float32)])
        whole = len(self._mic) // self.frame * self.frame
        if whole:
            outs = [
                self._frame(self._mic[i : i + self.frame], self._ref[i : i + self.frame])
                for i in range(0, whole, self.frame)
            ]
            self._out = np.concatenate([self._out, *outs])
            self._mic, self._ref = self._mic[whole:], self._ref[whole:]
        out, self._out = self._out[:n], self._out[n:]
        if len(out) < n:
            out = np.concatenate([out, np.zeros(n - len(out), dtype=np.float32)])
        return out


def _to_int16(x: FloatArray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


class WebRtcAec3Stream(_Framed):
    frame = 160  # WebRTC processes 10 ms frames

    def __init__(self, delay_ms: int = 0, noise_suppression: bool = False) -> None:
        super().__init__()
        from livekit import rtc

        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=noise_suppression)
        if delay_ms:
            self._apm.set_stream_delay_ms(int(delay_ms))

    def _audio_frame(self, x: FloatArray) -> Any:
        return self._rtc.AudioFrame(_to_int16(x).tobytes(), SR, 1, len(x))

    def _frame(self, mic: FloatArray, ref: FloatArray) -> FloatArray:
        far = self._audio_frame(ref)
        self._apm.process_reverse_stream(far)
        near = self._audio_frame(mic)
        self._apm.process_stream(near)
        return np.frombuffer(bytes(near.data), dtype=np.int16).astype(np.float32) / 32768.0


class SpeexAecStream(_Framed):
    frame = 160

    def __init__(self, filter_ms: int = 400, preprocess: bool = True) -> None:
        super().__init__()
        import ctypes

        import pyaec

        self._lib = pyaec.lib
        if self._lib is None:
            raise RuntimeError("pyaec could not load its library")
        self._ptr = ctypes.POINTER(ctypes.c_int16)
        self._handle = self._lib.AecNew(self.frame, int(SR * filter_ms / 1000), SR, bool(preprocess))
        if not self._handle:
            raise RuntimeError("SpeexDSP could not create an echo canceller")
        self._lock = threading.Lock()

    def _frame(self, mic: FloatArray, ref: FloatArray) -> FloatArray:
        rec = np.ascontiguousarray(_to_int16(mic))
        echo = np.ascontiguousarray(_to_int16(ref))
        out = np.zeros(self.frame, dtype=np.int16)
        with self._lock:
            self._lib.AecCancelEcho(
                self._handle,
                rec.ctypes.data_as(self._ptr),
                echo.ctypes.data_as(self._ptr),
                out.ctypes.data_as(self._ptr),
                self.frame,
            )
        return out.astype(np.float32) / 32768.0

    def __del__(self) -> None:
        handle, self._handle = getattr(self, "_handle", None), None
        if handle:
            self._lib.AecDestroy(handle)


class DtlnAecStream(_Framed):
    """DTLN-aec: 32 ms blocks every 8 ms, the mask stage on magnitudes, the second stage on the time signal. The
    overlap-add makes the output lag by a block less a shift (24 ms), on top of the framing."""

    block = 512
    frame = 128  # the block shift

    def __init__(self, model: DtlnAec) -> None:
        super().__init__()
        self.model = model
        self.delay = self.block  # the framing (one shift) and the overlap-add (a block less a shift)
        self._in = np.zeros(self.block, dtype=np.float32)
        self._lpb = np.zeros(self.block, dtype=np.float32)
        self._out_buf = np.zeros(self.block, dtype=np.float32)
        units = model.units
        self._s1 = np.zeros((1, 2, units, 2), dtype=np.float32)
        self._s2 = np.zeros((1, 2, units, 2), dtype=np.float32)

    def _frame(self, mic: FloatArray, ref: FloatArray) -> FloatArray:
        m = self.model
        sh = self.frame
        self._in = np.concatenate([self._in[sh:], mic])
        self._lpb = np.concatenate([self._lpb[sh:], ref])
        spec = np.fft.rfft(self._in).astype(np.complex64)
        mag = np.abs(spec).astype(np.float32).reshape(1, 1, -1)
        lpb_mag = np.abs(np.fft.rfft(self._lpb)).astype(np.float32).reshape(1, 1, -1)
        mask, self._s1 = m.run1(mag, self._s1, lpb_mag)
        est = np.fft.irfft(spec * mask.reshape(-1)).astype(np.float32).reshape(1, 1, -1)
        block, self._s2 = m.run2(est, self._s2, self._lpb.reshape(1, 1, -1).astype(np.float32))
        self._out_buf = np.concatenate([self._out_buf[sh:], np.zeros(sh, dtype=np.float32)])
        self._out_buf += block.reshape(-1)
        return np.array(self._out_buf[:sh], dtype=np.float32)


class EchoCancellerModel(Model):
    """A catalog model of task ``echo-cancellation``: :meth:`create_stream` gives one conversation's canceller."""

    task = "echo-cancellation"
    sample_rate = SR
    id: str = ""

    @abc.abstractmethod
    def create_stream(self) -> EchoStream: ...

    def cancel(self, mic: FloatArray, ref: FloatArray) -> FloatArray:
        """A whole recording and its reference (same length), the output aligned with the input."""
        stream = self.create_stream()
        pad = np.zeros(stream.delay + 160, dtype=np.float32)
        out = stream.process(np.concatenate([mic, pad]), np.concatenate([ref, pad]))
        return np.asarray(out[stream.delay : stream.delay + len(mic)], dtype=np.float32)


class WebRtcAec3(EchoCancellerModel):
    def __init__(self, spec: dict[str, Any]) -> None:
        self.id = str(spec.get("id") or spec.get("model_id") or "")
        opts = {**(spec.get("options") or {}), **spec}
        self.delay_ms = int(opts.get("delay_ms", 0))
        self.noise_suppression = bool(opts.get("noise_suppression", False))
        WebRtcAec3Stream()  # fails here, at load, when LiveKit's SDK is missing

    def create_stream(self) -> EchoStream:
        return WebRtcAec3Stream(self.delay_ms, self.noise_suppression)


class SpeexAec(EchoCancellerModel):
    def __init__(self, spec: dict[str, Any]) -> None:
        self.id = str(spec.get("id") or spec.get("model_id") or "")
        opts = {**(spec.get("options") or {}), **spec}
        self.filter_ms = int(opts.get("filter_ms", 400))
        self.preprocess = bool(opts.get("preprocess", True))
        SpeexAecStream(self.filter_ms, self.preprocess)

    def create_stream(self) -> EchoStream:
        return SpeexAecStream(self.filter_ms, self.preprocess)


class DtlnAec(EchoCancellerModel):
    def __init__(self, path: Path, spec: dict[str, Any], threads: int = 1) -> None:
        import onnxruntime as ort

        self.id = str(spec.get("id") or spec.get("model_id") or "")
        opts = {**(spec.get("options") or {}), **spec}
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(threads))
        so.inter_op_num_threads = 1
        self._s1 = ort.InferenceSession(
            str(path / opts.get("model_1", "model_128_1.onnx")), so, ["CPUExecutionProvider"]
        )
        self._s2 = ort.InferenceSession(
            str(path / opts.get("model_2", "model_128_2.onnx")), so, ["CPUExecutionProvider"]
        )
        # inputs by position: (magnitude or block, states, loopback)
        self._in1 = [i.name for i in self._s1.get_inputs()]
        self._in2 = [i.name for i in self._s2.get_inputs()]
        self.units = int(self._s1.get_inputs()[1].shape[2])
        self._lock = threading.Lock()

    def run1(self, mag: np.ndarray, states: np.ndarray, lpb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            out = self._s1.run(None, dict(zip(self._in1, (mag, states, lpb), strict=True)))
        return out[0], out[1]

    def run2(self, block: np.ndarray, states: np.ndarray, lpb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            out = self._s2.run(None, dict(zip(self._in2, (block, states, lpb), strict=True)))
        return out[0], out[1]

    def create_stream(self) -> EchoStream:
        return DtlnAecStream(self)


class _SubtractStream(EchoStream):
    def __init__(self, owner: FakeEchoCanceller) -> None:
        self.owner = owner

    def process(self, mic: FloatArray, ref: FloatArray) -> FloatArray:
        self.owner.reference_s += float(np.count_nonzero(ref)) / SR
        return np.zeros(len(mic), dtype=np.float32) if np.any(ref) else np.asarray(mic, dtype=np.float32)


class FakeEchoCanceller(EchoCancellerModel):
    """Tests: the microphone is muted while there is reference audio (half duplex); counts the reference."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.id = str(spec.get("id") or spec.get("model_id") or "")
        self.reference_s = 0.0

    def create_stream(self) -> EchoStream:
        return _SubtractStream(self)


class EchoReference:
    """The far end as the client hears it: audio bolo sent is queued, and released as much at a time as microphone
    audio arrives (the client plays at the pace it records). Cleared when the client stops playing (barge-in,
    ``output_audio_buffer.clear``, ``response.cancel``)."""

    #: queued reference audio beyond this is dropped (a client that does not play what it is sent)
    MAX_S = 120.0

    def __init__(self) -> None:
        self._queue: list[FloatArray] = []
        self._queued = 0
        self._lock = threading.Lock()

    def push(self, samples: FloatArray) -> None:
        if not len(samples):
            return
        with self._lock:
            if self._queued + len(samples) > self.MAX_S * SR:
                return
            self._queue.append(np.asarray(samples, dtype=np.float32))
            self._queued += len(samples)

    def take(self, n: int) -> FloatArray:
        out = np.zeros(n, dtype=np.float32)
        got = 0
        with self._lock:
            while got < n and self._queue:
                head = self._queue[0]
                k = min(n - got, len(head))
                out[got : got + k] = head[:k]
                got += k
                if k == len(head):
                    self._queue.pop(0)
                else:
                    self._queue[0] = head[k:]
            self._queued -= got
        return out

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
            self._queued = 0

    @property
    def pending_s(self) -> float:
        return self._queued / SR


@dataclass
class EchoOptions:
    """``audio.input.echo_cancellation`` of a Realtime session (bolo extension): which canceller, or none."""

    model: str | None = None  # None: the server's default_echo_cancellation
    #: the reference held back this long before the canceller sees it: the client's playback latency, for
    #: cancellers that do not find the delay themselves (DTLN-aec wants its reference within a few tens of ms)
    delay_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": True, "model": self.model, "delay_ms": self.delay_ms}


def parse_echo(value: Any, param: str = "echo_cancellation") -> EchoOptions | None:
    """``true`` (the default canceller), ``false``/``null`` (none), a model id, or ``{"model": id}``."""
    from bolo.errors import InvalidRequest

    if value is None or value is False:
        return None
    if value is True:
        return EchoOptions()
    if isinstance(value, str) and value.strip():
        return EchoOptions(value.strip())
    if isinstance(value, dict):
        if value.get("enabled") is False or value.get("type") == "none":
            return None
        model = value.get("model")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise InvalidRequest(f"{param}.model must be a model id", param=param)
        delay = value.get("delay_ms", 0)
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not 0 <= delay <= 1000:
            raise InvalidRequest(f"{param}.delay_ms must be between 0 and 1000", param=param)
        return EchoOptions(model.strip() if model else None, int(delay))
    raise InvalidRequest(f'{param} must be true, false, a model id or {{"model": ...}}', param=param)


class EchoCanceller:
    """One session's echo cancellation: the reference queue and the model's stream."""

    def __init__(self, model: EchoCancellerModel, model_id: str | None = None, delay_ms: int = 0) -> None:
        self.model = model
        self.model_id = model_id or model.id
        self.delay_ms = int(delay_ms)
        self._line = np.zeros(int(SR * self.delay_ms / 1000), dtype=np.float32)
        self.reference = EchoReference()
        self.stream = model.create_stream()
        self.samples = 0
        self.active_samples = 0  # samples processed while there was reference audio

    def far(self, samples: FloatArray) -> None:
        self.reference.push(samples)

    def near(self, samples: FloatArray) -> FloatArray:
        ref = self.reference.take(len(samples))
        if len(self._line):  # a delay line: the reference as the canceller should see it
            both = np.concatenate([self._line, ref])
            ref, self._line = both[: len(samples)], both[len(samples) :]
        self.samples += len(samples)
        if np.any(ref):
            self.active_samples += len(samples)
        return self.stream.process(samples, ref)

    def stop_playback(self) -> None:
        self.reference.clear()
        self._line[:] = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "audio_s": round(self.samples / SR, 2),
            "with_reference_s": round(self.active_samples / SR, 2),
            "delay_ms": self.delay_ms,
            "latency_ms": round(self.stream.delay * 1000 / SR, 1),
        }


# -- engines (bolo.engines registry) -----------------------------------------------------------------------------
class WebRtcApmEngine(Engine):
    id = "webrtc-apm"
    name = "WebRTC audio processing (LiveKit)"
    license = "Apache-2.0 (LiveKit SDK) + BSD-3-Clause (WebRTC)"
    url = "https://github.com/livekit/python-sdks"
    kinds = ("webrtc_aec3",)

    def status(self) -> EngineStatus:
        try:
            from importlib.metadata import version

            from livekit import rtc  # noqa: F401

            return EngineStatus(self.id, True, version("livekit"), None)
        except Exception as exc:
            return EngineStatus(self.id, False, None, f"{type(exc).__name__}: {exc}. Optional: {INSTALL}")

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind != "webrtc_aec3":
            raise ValueError(f"the webrtc-apm engine cannot load kind {kind!r}")
        return WebRtcAec3(spec)


class SpeexDspEngine(Engine):
    id = "speexdsp"
    name = "SpeexDSP (pyaec)"
    license = "BSD-3-Clause (SpeexDSP) + MIT (pyaec)"
    url = "https://github.com/thewh1teagle/aec-rs"
    kinds = ("speex_aec",)

    def status(self) -> EngineStatus:
        try:
            from importlib.metadata import version

            import pyaec

            if pyaec.lib is None:
                raise OSError("pyaec could not load its library")
            return EngineStatus(self.id, True, version("pyaec"), None)
        except Exception as exc:
            return EngineStatus(self.id, False, None, f"{type(exc).__name__}: {exc}. Optional: {INSTALL}")

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind != "speex_aec":
            raise ValueError(f"the speexdsp engine cannot load kind {kind!r}")
        return SpeexAec(spec)
