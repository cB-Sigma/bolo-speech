"""Voice-activity detectors that score fixed frames of audio, turned into speech segments by bolo.

sherpa-onnx runs Silero and TEN VAD and does its own segmenting. The detectors here only say, frame by frame,
how likely the frame is speech; :class:`FrameVad` turns those probabilities into ``speech_start`` /
``speech_end`` events with the same :class:`~bolo.engines.base.VadParams` (threshold, minimum silence, minimum
and maximum speech, padding), so every detector is used the same way.

* **WebRTC VAD** (engine ``webrtcvad``, kind ``webrtc_vad``): the Gaussian-mixture detector of the WebRTC
  project (BSD-3-Clause) through the ``webrtcvad-wheels`` package (MIT). No model file; a yes/no answer per
  20 ms frame at one of four aggressiveness modes (0 to 3).
* **MarbleNet** (engine ``onnxruntime``, kind ``marblenet_vad``): NVIDIA's Frame-VAD Multilingual MarbleNet
  v2.0, a 91.5 k-parameter convolutional network that gives a speech probability for every 20 ms frame from
  80-band NeMo log-mel features. Its convolutions look about 1.4 s either side of a frame, so live audio is
  scored in a sliding window of features: ``context_s`` before the new frames and ``lookahead_s`` after them
  (the probabilities come that much later than the audio), every ``step_s``.
"""

from __future__ import annotations

import abc
import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import (
    Engine,
    EngineStatus,
    FloatArray,
    LoadOptions,
    Model,
    VadDetector,
    VadEvent,
    VADModel,
    VadParams,
)

SR = 16000
#: a frame the detector is in speech on counts as silence below this much under the threshold (as Silero does)
HYSTERESIS = 0.15
#: while speech is building up (shorter than min_speech_s), dips up to this long do not reset it
ONSET_GAP_S = 0.12


class FrameScorer(abc.ABC):
    """Speech probabilities of consecutive frames of ``frame`` samples, for audio that arrives in pieces."""

    frame: int = 320

    @abc.abstractmethod
    def push(self, samples: FloatArray) -> list[float]:
        """Probabilities of the frames that became ready (in order, continuing from the last call)."""

    @abc.abstractmethod
    def finish(self) -> list[float]:
        """Probabilities of the frames still pending at the end of the input (a partial last frame is padded)."""


class FrameVad(VadDetector):
    """Speech segments from per-frame speech probabilities: a frame at or above ``threshold`` is speech; once
    in speech, a frame stays speech down to ``threshold - 0.15``. A segment starts after ``min_speech_s`` of
    speech, ends after ``min_silence_s`` of silence (or at ``max_speech_s``) and is padded by ``speech_pad_s``."""

    def __init__(self, scorer: FrameScorer, params: VadParams | None = None) -> None:
        self.params = params or VadParams()
        self.scorer = scorer
        self.frame = scorer.frame
        self._index = 0  # frames scored so far
        self._speech = False
        self._candidate: int | None = None
        self._gap = 0
        self._start = 0
        self._last_voiced = 0
        self._received = 0
        self.low = max(0.01, self.params.threshold - HYSTERESIS)

    @property
    def in_speech(self) -> bool:
        return self._speech

    def _step(self, p: float, events: list[VadEvent]) -> None:
        params, rate = self.params, self.sample_rate
        start = self._index * self.frame
        end = start + self.frame
        self._index += 1
        pad = int(params.speech_pad_s * rate)
        if not self._speech:
            if p >= params.threshold:
                if self._candidate is None:
                    self._candidate = start
                self._gap = 0
                if end - self._candidate >= params.min_speech_s * rate:
                    self._speech = True
                    self._start = max(0, self._candidate - pad)
                    self._last_voiced = end
                    events.append(VadEvent("speech_start", self._start))
            elif self._candidate is not None:
                self._gap += self.frame
                if self._gap > ONSET_GAP_S * rate:
                    self._candidate, self._gap = None, 0
            return
        if p >= self.low:
            self._last_voiced = end
        silence = end - self._last_voiced
        too_long = end - self._start >= params.max_speech_s * rate
        if silence >= params.min_silence_s * rate or too_long:
            self._speech = False
            self._candidate, self._gap = None, 0
            cut = end if too_long and silence == 0 else self._last_voiced + pad
            events.append(VadEvent("speech_end", min(cut, end)))

    def accept(self, samples: FloatArray) -> list[VadEvent]:
        events: list[VadEvent] = []
        self._received += len(samples)
        for p in self.scorer.push(np.asarray(samples, dtype=np.float32)):
            self._step(p, events)
        return events

    def flush(self) -> list[VadEvent]:
        events: list[VadEvent] = []
        for p in self.scorer.finish():
            self._step(p, events)
        if self._speech:
            self._speech = False
            events.append(VadEvent("speech_end", min(self._received, self._index * self.frame)))
        return events


class _Framer:
    """Cuts a stream into whole frames, keeping the remainder for the next piece."""

    def __init__(self, frame: int) -> None:
        self.frame = frame
        self._rest = np.zeros(0, dtype=np.float32)

    def push(self, samples: FloatArray) -> FloatArray:
        x = np.concatenate([self._rest, samples]) if len(self._rest) else samples
        n = len(x) // self.frame * self.frame
        self._rest = x[n:]
        return x[:n]

    def finish(self) -> FloatArray:
        rest, self._rest = self._rest, np.zeros(0, dtype=np.float32)
        if not len(rest):
            return rest
        return np.concatenate([rest, np.zeros(self.frame - len(rest), dtype=np.float32)])


# ---------------------------------------------------------------------------------------------------
# WebRTC VAD
# ---------------------------------------------------------------------------------------------------
class _WebRtcScorer(FrameScorer):
    def __init__(self, vad: Any, frame: int) -> None:
        self.vad = vad
        self.frame = frame
        self._framer = _Framer(frame)

    def _score(self, x: FloatArray) -> list[float]:
        pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        step = 2 * self.frame
        return [1.0 if self.vad.is_speech(pcm[i : i + step], SR) else 0.0 for i in range(0, len(pcm), step)]

    def push(self, samples: FloatArray) -> list[float]:
        return self._score(self._framer.push(samples))

    def finish(self) -> list[float]:
        return self._score(self._framer.finish())


class WebRtcVAD(VADModel):
    """The WebRTC detector. ``mode`` (catalog option or pin): 0 (least aggressive: keeps the most audio as speech)
    to 3 (most aggressive: drops the most noise); ``frame_ms``: 10, 20 or 30."""

    def __init__(self, spec: dict[str, Any], opts: LoadOptions) -> None:
        import webrtcvad

        self._webrtcvad = webrtcvad
        merged = {**spec, **opts.options}
        self.mode = int(merged.get("mode", 2))
        if self.mode not in (0, 1, 2, 3):
            raise ValueError(f"webrtcvad mode must be 0, 1, 2 or 3, not {self.mode}")
        frame_ms = int(merged.get("frame_ms", 20))
        if frame_ms not in (10, 20, 30):
            raise ValueError(f"webrtcvad frame_ms must be 10, 20 or 30, not {frame_ms}")
        self.frame = SR * frame_ms // 1000

    def create_detector(self, params: VadParams | None = None) -> VadDetector:
        return FrameVad(_WebRtcScorer(self._webrtcvad.Vad(self.mode), self.frame), params)


class WebRtcVadEngine(Engine):
    id = "webrtcvad"
    name = "WebRTC VAD"
    license = "MIT (Python binding) + BSD-3-Clause (WebRTC)"
    url = "https://github.com/daanzu/py-webrtcvad-wheels"
    kinds = ("webrtc_vad",)

    def status(self) -> EngineStatus:
        try:
            from importlib.metadata import version

            import webrtcvad  # noqa: F401

            return EngineStatus(self.id, True, version("webrtcvad-wheels"), None)
        except Exception as exc:
            return EngineStatus(
                self.id, False, None, f"{type(exc).__name__}: {exc}. Optional: pip install 'bolo-speech[webrtcvad]'"
            )

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind != "webrtc_vad":
            raise ValueError(f"the webrtcvad engine cannot load kind {kind!r}")
        return WebRtcVAD(spec, opts)


# ---------------------------------------------------------------------------------------------------
# MarbleNet (ONNX Runtime)
# ---------------------------------------------------------------------------------------------------
#: MarbleNet scores 20 ms frames (two 10 ms feature frames each)
MARBLENET_FRAME = 320
MARBLENET_MELS = 80


class _MarbleNetScorer(FrameScorer):
    """Log-mel frames are computed once, as the audio arrives (:class:`bolo.audio.features.NemoLogMel` gives the same
    frames for a stream as for a whole recording); the network runs on a window of them: ``context_s`` before the
    frames to score, ``lookahead_s`` after."""

    frame = MARBLENET_FRAME

    def __init__(self, model: MarbleNetVAD) -> None:
        from bolo.audio.features import NemoLogMel

        self.model = model
        self._mel = NemoLogMel(MARBLENET_MELS)
        self._feats = np.zeros((0, MARBLENET_MELS), dtype=np.float32)
        self._base = 0  # index (10 ms feature frames, even) of _feats[0]
        self._done = 0  # 20 ms frames scored so far

    def _score(self, ready: int, lookahead: int) -> list[float]:
        """Probabilities of frames ``_done .. ready - 1``, from a window that ends ``lookahead`` feature frames
        after them."""
        if ready <= self._done:
            return []
        total = self._base + len(self._feats)
        start = max(self._base, 2 * (self._done - self.model.context_frames))
        end = min(total, 2 * ready + lookahead)
        window = self._feats[start - self._base : end - self._base]
        if len(window) < 2 * ready - start:  # the end of the input: a last half frame
            pad = np.repeat(window[-1:], 2 * ready - start - len(window), axis=0)
            window = np.concatenate([window, pad])
        probs = self.model.probabilities(window)
        first = self._done - start // 2
        out = [float(p) for p in probs[first : first + ready - self._done]]
        self._done = ready
        keep = max(0, 2 * (self._done - self.model.context_frames))
        if keep > self._base:
            self._feats = self._feats[keep - self._base :]
            self._base = keep
        return out

    def _add(self, feats: FloatArray) -> int:
        if len(feats):
            self._feats = np.concatenate([self._feats, feats]) if len(self._feats) else feats
        return self._base + len(self._feats)

    def push(self, samples: FloatArray) -> list[float]:
        total = self._add(self._mel.push(samples))
        look = self.model.lookahead_frames
        ready = max(0, (total - look) // 2)
        if ready - self._done < self.model.min_frames:
            return []
        return self._score(ready, look)

    def finish(self) -> list[float]:
        total = self._add(self._mel.finish())
        return self._score(-(-total // 2), 0)


class MarbleNetVAD(VADModel):
    """Frame-VAD MarbleNet on ONNX Runtime (CPU; 91.5 k parameters)."""

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from bolo.ort_providers import CPU, create_session, session_options

        merged = {**spec, **opts.options}
        files = sorted(path.glob(str(merged.get("file", "*.onnx"))))
        if not files:
            raise FileNotFoundError(f"no ONNX file in {path}")
        options = session_options()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = create_session(str(files[0]), options, [CPU])
        self._input = self._session.get_inputs()[0].name
        self._lock = threading.Lock()
        #: 20 ms frames of audio before the frames being scored that the network sees
        self.context_frames = int(float(merged.get("context_s", 1.6)) * SR) // MARBLENET_FRAME
        #: 10 ms feature frames after them (live audio: the probabilities come that much later)
        self.lookahead_frames = int(float(merged.get("lookahead_s", 0.4)) * 100)
        #: live audio is scored once at least this many new frames are ready (the window is recomputed each time)
        self.min_frames = max(1, int(float(merged.get("step_s", 0.1)) * SR) // MARBLENET_FRAME)
        self.create_detector()  # fail early if the file is unusable

    def probabilities(self, mel: FloatArray) -> FloatArray:
        """Speech probability of every 20 ms frame, from 80-band NeMo log-mel frames (10 ms, shape ``(n, 80)``)."""
        if len(mel) < 2:
            return np.zeros(0, dtype=np.float32)
        if len(mel) % 2:  # the model's preprocessor pads the frame count to a multiple of 2
            mel = np.concatenate([mel, mel[-1:]])
        with self._lock:
            logits = np.asarray(self._session.run(None, {self._input: mel.T[None]})[0][0], dtype=np.float64)
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs: FloatArray = (e[:, 1] / e.sum(axis=-1)).astype(np.float32)
        return probs

    def create_detector(self, params: VadParams | None = None) -> VadDetector:
        return FrameVad(_MarbleNetScorer(self), params)
