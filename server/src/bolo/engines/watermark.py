"""Audio watermarkers on ONNX Runtime (task ``watermark``).

* ``audioseal_onnx``: AudioSeal (Meta, MIT), the 16-bit generator and detector as two ONNX graphs. The generator
  takes 16 kHz audio ``[1, 1, T]`` and a 16-bit message and returns the watermark, which is added to the audio; the
  detector returns, per sample, the probability that it is marked, and the 16 payload bits. ``T`` must be a multiple
  of 320 (the SEANet hop): bolo pads with zeros and cuts the padding off again. Long audio is processed in windows
  of ``window_s`` (10 s), each on its own: the mark is local, so a cut, spliced or trimmed recording keeps it.
* ``perth_onnx``: Perth (Resemble AI, MIT), 32 kHz, no payload (:class:`PerthOnnx`).
* ``wavmark_onnx``: WavMark (MIT), 16 kHz, a 16-bit payload in every 1.1 s (:class:`WavMarkOnnx`).

Perth and WavMark are ONNX exports with their STFT and inverse STFT inside the graphs (codebucketsolutions/perth-onnx
and codebucketsolutions/wavmark-onnx; the export script and the parity results against PyTorch are there).
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


class PerthOnnx(WatermarkModel):
    """Perth (Resemble AI, MIT), the implicit watermarker, as two ONNX graphs with their STFT inside (32 kHz audio in
    and out). It marks the spectrum below 2 kHz and carries no payload; the detector scores a whole stretch of audio.
    Detection runs on the whole recording (the score) and on ``window_s`` windows (where the mark is)."""

    sample_rate = 32000
    payload_bits = 0
    HOP = 320

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from bolo.engines.lid_onnx import _find, _options
        from bolo.ort_providers import create_session, providers_for

        threads = max(1, min(4, opts.threads))
        providers = providers_for(opts.device)
        self._embed = create_session(
            str(_find(path, str(spec.get("embedder", "*embed*.onnx")))), _options(threads), providers
        )
        self._detect = create_session(
            str(_find(path, str(spec.get("detector", "*detect*.onnx")))), _options(threads), providers
        )
        self.window_s = float(spec.get("window_s", 2.0))
        self._lock = threading.Lock()

    def embed(self, samples: FloatArray, payload: str | None = None) -> FloatArray:
        x = np.asarray(samples, dtype=np.float32)
        n = len(x)
        # the graph's STFT needs a few frames and returns whole hops: pad, then cut back to the input's length
        size = max(4 * 2048, -(-n // self.HOP) * self.HOP)
        padded = np.pad(x, (0, size - n)) if size != n else x
        with self._lock:
            marked = np.asarray(self._embed.run(None, {"audio": padded})[0], dtype=np.float32)
        return np.asarray(marked[:n] if len(marked) >= n else np.pad(marked, (0, n - len(marked))), dtype=np.float32)

    def _score(self, x: FloatArray) -> float:
        if len(x) < 4 * 2048:
            x = np.pad(x, (0, 4 * 2048 - len(x)))
        with self._lock:
            return float(np.asarray(self._detect.run(None, {"audio": x.astype(np.float32)})[0]).reshape(-1)[0])

    def detect(self, samples: FloatArray) -> WatermarkDetection:
        x = np.asarray(samples, dtype=np.float32)
        score = self._score(x)
        size = int(self.window_s * self.sample_rate)
        frames = [self._score(x[a : a + size]) for a in range(0, max(1, len(x) - size // 2), size)]
        probs = np.asarray(frames or [score], dtype=np.float32)
        return WatermarkDetection(score, probs, self.window_s, None)


class WavMarkOnnx(WatermarkModel):
    """WavMark (MIT) as two ONNX graphs, one second of 16 kHz audio at a time. As in the ``wavmark`` package: every
    1.1 s, the first second carries 32 bits, 16 of a fixed pattern and the 16-bit payload; the encoding is repeated
    until the mark is quiet enough (signal-to-noise ratio of 38 dB, at most 10 times) and a second too quiet to take
    it (under 20 dB at the first try) is left alone. Detection looks for the pattern every 50 ms; the payload is the
    majority over the windows that hold it. Detection follows a found mark from second to second instead of
    decoding every 50 ms everywhere (see :meth:`detect`)."""

    sample_rate = 16000
    payload_bits = 16
    SECOND = 16000
    CHUNK = 17600
    STEP = 800
    #: the 16 pattern bits the wavmark package uses (wm_add_util.fix_pattern[:16])
    PATTERN = (1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0)

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from bolo.engines.lid_onnx import _find, _options
        from bolo.ort_providers import create_session, providers_for

        threads = max(1, min(4, opts.threads))
        providers = providers_for(opts.device)
        self._embed = create_session(
            str(_find(path, str(spec.get("embedder", "*embed*.onnx")))), _options(threads), providers
        )
        self._detect = create_session(
            str(_find(path, str(spec.get("detector", "*detect*.onnx")))), _options(threads), providers
        )
        self.min_snr = float(spec.get("min_snr", 20.0))
        self.max_snr = float(spec.get("max_snr", 38.0))
        self._lock = threading.Lock()

    @staticmethod
    def _snr(a: FloatArray, b: FloatArray) -> float:
        noise = float(np.sum((a - b) ** 2))
        return float("inf") if noise == 0 else 10 * np.log10(float(np.sum(a**2)) / noise)

    def embed(self, samples: FloatArray, payload: str | None = None) -> FloatArray:
        x = np.asarray(samples, dtype=np.float32)
        out = x.copy()
        bits = payload or "0" * self.payload_bits
        message = np.array([[*self.PATTERN, *(int(b) for b in bits)]], dtype=np.float32)
        for a in range(0, len(x) - self.CHUNK + 1, self.CHUNK):
            cover = x[a : a + self.SECOND]
            current = cover
            for attempt in range(10):
                with self._lock:
                    marked = np.asarray(
                        self._embed.run(None, {"audio": current[None], "bits": message})[0], dtype=np.float32
                    )[0]
                snr = self._snr(cover, marked)
                if attempt == 0 and snr < self.min_snr:
                    marked = cover  # too quiet to hide anything in
                    break
                if snr < self.max_snr:
                    break
                current = marked
            out[a : a + self.SECOND] = marked
        return np.asarray(out, dtype=np.float32)

    def _decode(self, x: FloatArray, points: list[int]) -> list[np.ndarray | None]:
        """The payload bits at each start point, or None where the pattern is not there."""
        out: list[np.ndarray | None] = []
        for i in range(0, len(points), 8):
            batch = np.stack([x[p : p + self.SECOND] for p in points[i : i + 8]])
            with self._lock:
                bits = np.asarray(self._detect.run(None, {"audio": batch})[0]) >= 0.5
            for row in bits:
                out.append(row[16:].astype(np.float32) if tuple(int(b) for b in row[:16]) == self.PATTERN else None)
        return out

    def detect(self, samples: FloatArray) -> WatermarkDetection:
        """The ``wavmark`` package decodes every 50 ms, a second of audio each time (several seconds of compute per
        second on a CPU). Marks sit 1.1 s apart, so once one is found the next is looked for 1.1 s on (and one step
        either side); the dense search resumes only where that chain breaks."""
        x = np.asarray(samples, dtype=np.float32)
        if len(x) < self.SECOND:
            x = np.pad(x, (0, self.SECOND - len(x)))
        last = len(x) - self.SECOND
        found_at: list[int] = []
        messages: list[np.ndarray] = []
        p = 0
        while p <= last:
            # dense search: the next 1.1 s of start points
            points = list(range(p, min(last, p + self.CHUNK) + 1, self.STEP))
            hits = [(q, m) for q, m in zip(points, self._decode(x, points)) if m is not None]
            if not hits:
                p = points[-1] + self.STEP
                continue
            q, m = hits[0]
            while True:  # follow the chain of marks 1.1 s apart
                found_at.append(q)
                messages.append(m)
                nxt = [c for c in (q + self.CHUNK, q + self.CHUNK - self.STEP, q + self.CHUNK + self.STEP) if c <= last]
                if not nxt:
                    break
                follow = next(((c, r) for c, r in zip(nxt, self._decode(x, nxt)) if r is not None), None)
                if follow is None:
                    break
                q, m = follow
            p = q + self.CHUNK
        chunks = max(1, len(x) // self.CHUNK)  # the seconds embed() marks: one per whole 1.1 s
        score = min(1.0, len(found_at) / chunks)
        payload = "".join("1" if v >= 0.5 else "0" for v in np.mean(messages, axis=0)) if messages else None
        # per 50 ms: marked inside a second that holds the pattern
        frames = np.zeros(max(1, len(x) // self.STEP), dtype=np.float32)
        for q in found_at:
            frames[q // self.STEP : (q + self.SECOND) // self.STEP] = 1.0
        return WatermarkDetection(score, frames, self.STEP / self.sample_rate, payload)
