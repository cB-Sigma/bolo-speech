"""Audio watermarkers on ONNX Runtime (task ``watermark``).

* ``audioseal_onnx``: AudioSeal (Meta, MIT), the 16-bit generator and detector as two ONNX graphs. The generator
  takes 16 kHz audio ``[1, 1, T]`` and a 16-bit message and returns the watermark, which is added to the audio; the
  detector returns, per sample, the probability that it is marked, and the 16 payload bits. ``T`` must be a multiple
  of 320 (the SEANet hop): bolo pads with zeros and cuts the padding off again. Long audio is processed in windows
  of ``window_s`` (10 s), each on its own: the mark is local, so a cut, spliced or trimmed recording keeps it.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import FloatArray, LoadOptions, WatermarkDetection, WatermarkModel

HOP = 320


def _pad(x: FloatArray, hop: int = HOP) -> tuple[FloatArray, int]:
    n = len(x)
    padded = max(hop, -(-n // hop) * hop)
    return (np.pad(x, (0, padded - n)) if padded != n else x).astype(np.float32), n


class AudioSealOnnx(WatermarkModel):
    """AudioSeal 16-bit generator and detector (see the module docstring)."""

    sample_rate = 16000
    payload_bits = 16
    hop = HOP

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from bolo.engines.lid_onnx import _find, _options
        from bolo.ort_providers import create_session, providers_for

        threads = max(1, min(4, opts.threads))
        providers = providers_for(opts.device)
        self._generator = create_session(
            str(_find(path, str(spec.get("generator", "*generator*.onnx")))), _options(threads), providers
        )
        self._detector = create_session(
            str(_find(path, str(spec.get("detector", "*detector*.onnx")))), _options(threads), providers
        )
        self.window = int(float(spec.get("window_s", 10.0)) * self.sample_rate) // HOP * HOP
        self._lock = threading.Lock()

    def _windows(self, n: int) -> list[tuple[int, int]]:
        return [(a, min(n, a + self.window)) for a in range(0, max(n, 1), self.window)]

    def embed(self, samples: FloatArray, payload: str | None = None) -> FloatArray:
        x = np.asarray(samples, dtype=np.float32)
        bits = payload or "0" * self.payload_bits
        if len(bits) != self.payload_bits or set(bits) - {"0", "1"}:
            raise ValueError(f"the payload is {self.payload_bits} bits of 0 and 1")
        message = np.array([[int(b) for b in bits]], dtype=np.int64)
        out = x.copy()
        for a, b in self._windows(len(x)):
            piece, n = _pad(x[a:b])
            with self._lock:
                mark = self._generator.run(None, {"audio": piece[None, None], "message": message})[0]
            out[a:b] += np.asarray(mark, dtype=np.float32).reshape(-1)[:n]
        return np.asarray(out, dtype=np.float32)

    def detect(self, samples: FloatArray) -> WatermarkDetection:
        x = np.asarray(samples, dtype=np.float32)
        probs: list[FloatArray] = []
        bits: list[tuple[FloatArray, float]] = []
        for a, b in self._windows(len(x)):
            piece, n = _pad(x[a:b])
            with self._lock:
                prob, bit = self._detector.run(None, {"audio": piece[None, None]})
            p = np.asarray(prob, dtype=np.float32).reshape(-1)[:n]
            probs.append(p)
            marked = float((p > 0.5).mean()) if n else 0.0
            bits.append((np.asarray(bit, dtype=np.float32).reshape(-1), marked * n))
        p = np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)
        # per 20 ms frame (the detector answers per sample; a frame's value is its mean)
        frame = HOP
        frames = len(p) // frame
        frame_probs = p[: frames * frame].reshape(frames, frame).mean(axis=1) if frames else p[:1]
        score = float((p > 0.5).mean()) if len(p) else 0.0
        weight = sum(w for _, w in bits)
        payload = None
        if weight > 0:
            mean = np.sum([b * w for b, w in bits], axis=0) / weight
            payload = "".join("1" if v > 0.5 else "0" for v in np.asarray(mean).reshape(-1).tolist())
        return WatermarkDetection(score, frame_probs.astype(np.float32), frame / self.sample_rate, payload)
