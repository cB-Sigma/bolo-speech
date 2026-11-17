"""Speaker embedders on their own (task ``speaker-embedding``): one unit-length vector per stretch of speech.

* sherpa-onnx (kind ``speaker_embedding``, :class:`bolo.engines.sherpa.SherpaSpeakerEmbedding`): the 3D-Speaker,
  WeSpeaker and NeMo exports that carry sherpa-onnx's metadata (ERes2Net, CAM++ zh-en, ResNet, TitaNet). The same code
  embeds the voices of the pyannote diarization pipelines.
* ONNX Runtime (kind ``onnx_speaker_embedding``, :class:`OnnxSpeakerEmbedding`, here): graphs that take the
  16 kHz waveform and return the embedding, front-end included, such as the ECAPA-TDNN export of SpeechBrain's
  model.
* ONNX Runtime with bolo's Kaldi filterbank (kind ``fbank_speaker_embedding``, :class:`FbankSpeakerEmbedding`): the
  CAM++ exports, which the ONNX Runtime inside sherpa-onnx 2.0.1 (1.28) computes wrong.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import FloatArray, LoadOptions, SpeakerEmbeddingModel


def unit(vector: Any) -> FloatArray:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    out: FloatArray = (v / norm).astype(np.float32) if norm > 0 else v
    return out


class OnnxSpeakerEmbedding(SpeakerEmbeddingModel):
    """A waveform-in, embedding-out ONNX graph (input ``[1, samples]`` float32 at 16 kHz)."""

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from bolo.ort_providers import CPU, create_session, session_options

        merged = {**spec, **opts.options}
        files = sorted(path.glob(str(merged.get("file", "*.onnx"))))
        if not files:
            raise FileNotFoundError(f"no ONNX file in {path}")
        options = session_options()
        options.intra_op_num_threads = max(1, opts.threads)
        options.inter_op_num_threads = 1
        self._session = create_session(str(files[0]), options, [CPU])
        self._input = self._session.get_inputs()[0].name
        self._lock = threading.Lock()

    def embed(self, samples: FloatArray) -> FloatArray:
        x = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(1, -1))
        with self._lock:
            out = self._session.run(None, {self._input: x})[0]
        return unit(out)


#: the first ONNX Runtime that computes CAM++ right (1.28, which sherpa-onnx 2.0.1 bundles, gives wrong embeddings)
FBANK_MIN_ORT = (1, 29)


class FbankSpeakerEmbedding(SpeakerEmbeddingModel):
    """A filterbank-in, embedding-out ONNX graph (input ``[1, frames, 80]``): bolo computes Kaldi features
    (:func:`bolo.audio.features.kaldi_fbank`) and ONNX Runtime runs the network. Used for the CAM++ exports, which
    give wrong answers in the ONNX Runtime that sherpa-onnx 2.0.1 bundles (docs/models.md, "Speaker embedding").
    Options: ``file``, ``scale`` (1: samples in -1..1, 3D-Speaker; 32768: 16-bit values, WeSpeaker), ``high_freq``,
    ``mean_norm``."""

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import onnxruntime

        from bolo.ort_providers import CPU, create_session, session_options

        version = tuple(int(p) for p in str(onnxruntime.__version__).split(".")[:2] if p.isdigit())
        if version < FBANK_MIN_ORT:
            raise RuntimeError(
                f"onnxruntime {onnxruntime.__version__} computes this model wrong (1.28 was checked); it needs "
                f"onnxruntime >= {FBANK_MIN_ORT[0]}.{FBANK_MIN_ORT[1]}: pip install -U onnxruntime"
            )
        merged = {**spec, **opts.options}
        files = sorted(path.glob(str(merged.get("file", "*.onnx"))))
        if not files:
            raise FileNotFoundError(f"no ONNX file in {path}")
        self.scale = float(merged.get("scale", 1.0))
        self.high_freq = float(merged.get("high_freq", 0.0))
        self.mean_norm = bool(merged.get("mean_norm", True))
        options = session_options()
        options.intra_op_num_threads = max(1, opts.threads)
        options.inter_op_num_threads = 1
        self._session = create_session(str(files[0]), options, [CPU])
        self._input = self._session.get_inputs()[0].name
        self._lock = threading.Lock()

    def embed(self, samples: FloatArray) -> FloatArray:
        from bolo.audio.features import kaldi_fbank

        feats = kaldi_fbank(samples, high_freq=self.high_freq, scale=self.scale, mean_norm=self.mean_norm)
        if len(feats) < 10:
            raise ValueError("too little audio for a speaker embedding")
        with self._lock:
            out = self._session.run(None, {self._input: feats[None]})[0]
        return unit(out)
