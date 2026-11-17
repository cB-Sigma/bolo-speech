"""Speaker embedders on their own (task ``speaker-embedding``): one unit-length vector per stretch of speech.

* sherpa-onnx (kind ``speaker_embedding``, :class:`bolo.engines.sherpa.SherpaSpeakerEmbedding`): the 3D-Speaker,
  WeSpeaker and NeMo exports that carry sherpa-onnx's metadata (ERes2Net, CAM++, ResNet, TitaNet). The same code
  embeds the voices of the pyannote diarization pipelines.
* ONNX Runtime (kind ``onnx_speaker_embedding``, :class:`OnnxSpeakerEmbedding`, here): graphs that take the
  16 kHz waveform and return the embedding, front-end included, such as the ECAPA-TDNN export of SpeechBrain's
  model.
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
