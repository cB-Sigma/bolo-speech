"""End-of-turn detection with a small ONNX model, run by ONNX Runtime directly.

The model answers one question about the audio of an utterance so far: has the speaker finished,
or are they only pausing ("I would like to, um ...")? Realtime sessions use it for
``turn_detection.type = "semantic_vad"``: the voice-activity detector finds the pause, this model
decides whether the pause ends the turn.

The supported model is Smart Turn v3 (https://huggingface.co/pipecat-ai/smart-turn-v3,
BSD-2-Clause): a Whisper-tiny encoder with a classification head, 8 MB, which looks at the last
eight seconds of 16 kHz audio and returns the probability that the turn is complete.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import Engine, EngineStatus, FloatArray, LoadOptions, Model, TurnModel

WINDOW_S = 8


class SmartTurn(TurnModel):
    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import onnxruntime as ort

        files = sorted(path.glob(str(spec.get("model", "*.onnx"))))
        if not files:
            raise FileNotFoundError(f"no ONNX file in {path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(2, opts.threads))
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(files[0]), sess_options=options, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        self._lock = threading.Lock()

    def probability(self, samples: FloatArray) -> float:
        from bolo.audio.features import whisper_log_mel

        want = WINDOW_S * self.sample_rate
        audio = np.asarray(samples[-want:], dtype=np.float32)
        if len(audio) < want:  # the model was trained with the speech at the end of the window
            audio = np.concatenate([np.zeros(want - len(audio), dtype=np.float32), audio])
        features = whisper_log_mel(audio, normalize=True)[None]
        with self._lock:
            out = self._session.run(None, {self._input: features})
        return float(np.asarray(out[0]).reshape(-1)[0])


class OnnxRuntimeEngine(Engine):
    id = "onnxruntime"
    name = "ONNX Runtime"
    license = "MIT"
    url = "https://onnxruntime.ai"
    kinds = ("smart_turn", "nemotron_diarization", "dicow_onnx")

    def status(self) -> EngineStatus:
        try:
            import onnxruntime

            return EngineStatus(self.id, True, str(onnxruntime.__version__), None)
        except Exception as exc:
            return EngineStatus(
                self.id, False, None,
                f"{type(exc).__name__}: {exc}. Optional (end-of-turn detection, Nemotron diarization, DiCoW): "
                "pip install onnxruntime",
            )  # fmt: skip

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind == "nemotron_diarization":
            from bolo.engines.nemotron import NemotronDiarization

            return NemotronDiarization(path, spec, opts)
        if kind == "dicow_onnx":
            from bolo.engines.dicow_onnx import DiCoWOnnx

            return DiCoWOnnx.from_dir(path, spec, opts)
        if kind != "smart_turn":
            raise ValueError(f"the onnxruntime engine cannot load kind {kind!r}")
        return SmartTurn(path, spec, opts)
