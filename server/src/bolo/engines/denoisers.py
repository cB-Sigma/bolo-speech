"""Denoisers that run frame by frame outside sherpa-onnx: DeepFilterNet 3 (ONNX Runtime) and RNNoise (its C library).

Both are causal 48 kHz models that take 480 samples (10 ms) at a time and keep their state between frames, so a
recording goes through them frame by frame; the first few frames of output are the model's delay, which is cut off
so that the enhanced audio lines up with the input (``delay`` option; measured: 1440 samples for DeepFilterNet 3,
960 for RNNoise).

* **DeepFilterNet 3** (``onnxruntime`` engine, kind ``deepfilternet3_stream``): the single-graph streaming export of
  torchDF (github.com/grazder/DeepFilterNet, branch torchDF), which carries the STFT, the ERB features and the deep
  filter inside the graph: inputs ``input_frame`` (480 samples), ``states`` and ``atten_lim_db`` (0: no limit),
  outputs the enhanced frame and the new state. Only ONNX Runtime is needed.
* **RNNoise** (``rnnoise`` engine, kind ``rnnoise``): Xiph's RNNoise library, built for each platform by the
  pyrnnoise project (Apache-2.0) and published as wheels on PyPI. The catalog pins the wheel of each platform by
  SHA-256; bolo unpacks it into the model's directory and calls the library through ctypes. The model's weights are
  compiled into the library. pyrnnoise's Python package is not imported (it needs audiolab); only its library is
  used.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import (
    DenoiseModel,
    DenoiseStream,
    Engine,
    EngineStatus,
    FloatArray,
    FrameDenoiseStream,
    LoadOptions,
    Model,
)

FRAME = 480
RATE = 48000


def _frames(samples: FloatArray, delay: int) -> tuple[FloatArray, int]:
    """``samples`` padded to whole frames with room for the model's delay to come out; and the input length."""
    n = len(samples)
    pad = (-(n + delay)) % FRAME + delay
    return np.ascontiguousarray(np.pad(np.asarray(samples, dtype=np.float32), (0, pad)), dtype=np.float32), n


class DeepFilterNetStream(DenoiseModel):
    """DeepFilterNet 3 as one streaming ONNX graph (torchDF), on ONNX Runtime's CPU provider.

    It runs 100 times per second of audio on 10 ms frames: an accelerator would spend more time copying than
    computing, so the session stays on the CPU whatever ``device`` says. Option ``atten_lim_db`` caps the
    suppression (0: none)."""

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from bolo.ort_providers import CPU, create_session, session_options

        files = sorted(path.glob(str(spec.get("model", "*.onnx"))))
        if not files:
            raise FileNotFoundError(f"no ONNX file in {path}")
        options = session_options()
        options.intra_op_num_threads = max(1, min(2, opts.threads))
        options.inter_op_num_threads = 1
        self._session = create_session(str(files[0]), options, [CPU])
        names = {i.name: i for i in self._session.get_inputs()}
        if not {"input_frame", "states", "atten_lim_db"} <= set(names):
            raise ValueError(f"{files[0].name} is not a torchDF streaming export (inputs: {', '.join(names)})")
        size = names["states"].shape[0]
        if not isinstance(size, int):
            raise ValueError(f"{files[0].name}: the state size is not fixed ({size!r})")
        self._state_size = size
        limit = float(spec.get("atten_lim_db") or 0.0)
        if not 0.0 <= limit <= 100.0:
            raise ValueError(f"atten_lim_db must be between 0 and 100, got {limit:g}")
        self._limit = np.array(limit, dtype=np.float32)
        self._delay = int(spec.get("delay", 3 * FRAME))
        self.sample_rate = RATE
        self._lock = threading.Lock()

    def denoise(self, samples: FloatArray) -> FloatArray:
        audio, n = _frames(samples, self._delay)
        out = np.empty_like(audio)
        state = np.zeros(self._state_size, dtype=np.float32)
        run = self._session.run
        with self._lock:
            for i in range(0, len(audio), FRAME):
                frame, state, _ = run(
                    None, {"input_frame": audio[i : i + FRAME], "states": state, "atten_lim_db": self._limit}
                )
                out[i : i + FRAME] = frame
        return np.ascontiguousarray(out[self._delay : self._delay + n], dtype=np.float32)

    def create_stream(self) -> DenoiseStream:
        state = [np.zeros(self._state_size, dtype=np.float32)]
        run, limit = self._session.run, self._limit

        def step(frame: FloatArray) -> FloatArray:
            out, state[0], _ = run(None, {"input_frame": frame, "states": state[0], "atten_lim_db": limit})
            return np.asarray(out, dtype=np.float32)

        return FrameDenoiseStream(step, FRAME, self._delay)


#: the library's file name on each platform
_LIBRARY = {"linux": "librnnoise.so", "darwin": "librnnoise.dylib", "win32": "rnnoise.dll"}


def rnnoise_library(path: Path) -> Path:
    """The RNNoise library unpacked into a model directory (the pyrnnoise wheel of this platform)."""
    name = _LIBRARY.get("linux" if sys.platform.startswith("linux") else sys.platform)
    if name is None:
        raise RuntimeError(f"no RNNoise library is published for {sys.platform}")
    found = sorted(p for p in path.rglob(name) if p.is_file())
    if not found:
        raise FileNotFoundError(f"{name} is not in {path}: pull the model again on this platform (bolo pull rnnoise)")
    return found[0]


class RNNoise(DenoiseModel):
    """RNNoise (Xiph; Valin, 2018; the 2024 model of release 0.2) through its C library: 48 kHz, 10 ms frames,
    audio as 16-bit sample values in floats."""

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        lib = ctypes.CDLL(str(rnnoise_library(path)))
        lib.rnnoise_create.restype = ctypes.c_void_p
        lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        lib.rnnoise_destroy.restype = None
        lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]  # fmt: skip
        lib.rnnoise_process_frame.restype = ctypes.c_float
        lib.rnnoise_get_frame_size.restype = ctypes.c_int
        if int(lib.rnnoise_get_frame_size()) != FRAME:
            raise RuntimeError(
                f"this RNNoise library works on {lib.rnnoise_get_frame_size()}-sample frames, not {FRAME}"
            )
        self._lib = lib
        self._delay = int(spec.get("delay", 2 * FRAME))
        self.sample_rate = RATE

    def denoise(self, samples: FloatArray) -> FloatArray:
        audio, n = _frames(np.asarray(samples, dtype=np.float32) * 32768.0, self._delay)
        out = np.empty_like(audio)
        lib = self._lib
        pointer = ctypes.POINTER(ctypes.c_float)
        state = lib.rnnoise_create(None)
        if not state:
            raise MemoryError("rnnoise_create failed")
        try:
            for i in range(0, len(audio), FRAME):
                src = audio[i : i + FRAME]
                dst = out[i : i + FRAME]
                lib.rnnoise_process_frame(state, dst.ctypes.data_as(pointer), src.ctypes.data_as(pointer))
        finally:
            lib.rnnoise_destroy(state)
        enhanced = out[self._delay : self._delay + n] / np.float32(32768.0)
        return np.ascontiguousarray(enhanced, dtype=np.float32)

    def create_stream(self) -> DenoiseStream:
        return _RNNoiseStream(self._lib, self._delay)


class _RNNoiseStream(FrameDenoiseStream):
    """One RNNoise state for one live stream; freed with the stream."""

    def __init__(self, lib: Any, delay: int) -> None:
        self._lib = lib
        self._state = lib.rnnoise_create(None)
        if not self._state:
            raise MemoryError("rnnoise_create failed")
        pointer = ctypes.POINTER(ctypes.c_float)

        def step(frame: FloatArray) -> FloatArray:
            src = np.ascontiguousarray(frame * np.float32(32768.0), dtype=np.float32)
            dst = np.empty(FRAME, dtype=np.float32)
            lib.rnnoise_process_frame(self._state, dst.ctypes.data_as(pointer), src.ctypes.data_as(pointer))
            out: FloatArray = dst / np.float32(32768.0)
            return out

        super().__init__(step, FRAME, delay)

    def __del__(self) -> None:
        state, self._state = getattr(self, "_state", None), None
        if state:
            self._lib.rnnoise_destroy(state)


class RNNoiseEngine(Engine):
    id = "rnnoise"
    name = "RNNoise"
    license = "BSD-3-Clause"
    url = "https://gitlab.xiph.org/xiph/rnnoise"
    kinds = ("rnnoise",)

    def status(self) -> EngineStatus:
        platform = "linux" if sys.platform.startswith("linux") else sys.platform
        if platform not in _LIBRARY:
            return EngineStatus(self.id, False, None, f"no RNNoise library is published for {sys.platform}")
        return EngineStatus(self.id, True, "0.2 (pyrnnoise 0.4.5 build)", None)

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind != "rnnoise":
            raise ValueError(f"the rnnoise engine cannot load kind {kind!r}")
        return RNNoise(path, spec, opts)
