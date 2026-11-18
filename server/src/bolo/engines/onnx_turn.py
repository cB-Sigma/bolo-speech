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

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import Engine, EngineStatus, FloatArray, LoadOptions, Model, TurnModel
from bolo.ort_providers import CPU, create_session, providers_for, session_options
from bolo.placement import CPU_ONLY_ON

log = logging.getLogger(__name__)

WINDOW_S = 8

#: why the model runs on the CPU with ``device = "dml"`` (the scheduler places it there too)
DML_NOTICE = f"device 'dml': the end-of-turn model runs on the CPU: {CPU_ONLY_ON[('onnxruntime', 'smart_turn')]['dml']}"
_said: set[str] = set()


def _once(key: str, message: str) -> None:
    if key not in _said:
        _said.add(key)
        log.warning("%s", message)


def _family(device: str | None) -> str:
    return (device or "cpu").strip().lower().partition(":")[0]


class SmartTurn(TurnModel):
    """Smart Turn on ONNX Runtime. With ``device = "dml"`` it runs on the CPU (:data:`DML_NOTICE`). A session on an
    accelerator that fails while running is replaced by a CPU session, once, and the request is answered from it;
    the model's ``on_fallback`` tells the model manager, which moves its placement to the CPU."""

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        files = sorted(path.glob(str(spec.get("model", "*.onnx"))))
        if not files:
            raise FileNotFoundError(f"no ONNX file in {path}")
        self._file = files[0]
        self._threads = max(1, min(2, opts.threads))
        device = opts.device
        if _family(device) in ("dml", "directml"):
            _once("dml", DML_NOTICE)
            device = "cpu"
        providers = providers_for(device)
        if opts.graph_split.get("model"):
            # cut in two by the scheduler (bolo.onnx_split): front on the GPU
            from bolo.onnx_split import make_session

            self._session = make_session(self._file, providers, self._options, opts.graph_split["model"])
        else:
            self._session = create_session(str(self._file), self._options(), providers)
        got = getattr(self._session, "get_providers", None)
        running = (
            [str(p) for p in got()] if callable(got) else [str(p[0] if isinstance(p, tuple) else p) for p in providers]
        )
        #: the accelerator the session runs on, if any (a runtime failure on it moves the model to the CPU)
        self.accelerator: str | None = next((p for p in running if p != CPU), None)
        self._input = self._session.get_inputs()[0].name
        self._lock = threading.Lock()

    def _options(self) -> Any:
        options = session_options()
        options.intra_op_num_threads = self._threads
        options.inter_op_num_threads = 1
        return options

    def _to_cpu(self, exc: Exception) -> None:
        """The accelerated session failed while running: a CPU session from now on, said once."""
        from bolo.cuda_libs import ProviderFallback

        asked = self.accelerator or "an accelerator"
        self._session = create_session(str(self._file), self._options(), [CPU])
        self._input = self._session.get_inputs()[0].name
        self.accelerator = None
        detail = " ".join(str(exc).split())[:300]
        fallback = ProviderFallback("onnxruntime", asked, CPU, [], detail)
        message = (
            f"the end-of-turn model failed on {asked.removesuffix('ExecutionProvider')} while running "
            f"({type(exc).__name__}: {detail}); it runs on the CPU from now on"
        )
        _once(f"runtime:{asked}", message)
        hook = self.on_fallback
        if hook is not None:
            try:
                hook({**fallback.to_dict(), "message": message})
            except Exception as hook_exc:  # the answer matters more than the report
                log.debug("on_fallback failed: %s", hook_exc)

    def probability(self, samples: FloatArray) -> float:
        from bolo.audio.features import whisper_log_mel

        want = WINDOW_S * self.sample_rate
        audio = np.asarray(samples[-want:], dtype=np.float32)
        if len(audio) < want:  # the model was trained with the speech at the end of the window
            audio = np.concatenate([np.zeros(want - len(audio), dtype=np.float32), audio])
        features = whisper_log_mel(audio, normalize=True)[None]
        with self._lock:
            try:
                out = self._session.run(None, {self._input: features})
            except Exception as exc:
                if self.accelerator is None:
                    raise
                self._to_cpu(exc)
                out = self._session.run(None, {self._input: features})
        return float(np.asarray(out[0]).reshape(-1)[0])


class OnnxRuntimeEngine(Engine):
    id = "onnxruntime"
    name = "ONNX Runtime"
    license = "MIT"
    url = "https://onnxruntime.ai"
    kinds = (
        "smart_turn", "text_turn", "nemotron_diarization", "dicow_onnx", "lfm3_audio_onnx", "marblenet_vad",
        "deepfilternet3_stream",
        "whisper_lid", "onnx_lid", "onnx_speaker_embedding", "fbank_speaker_embedding", "onnx_emotion",
        "ctc_align",
        "dtln_aec", "openwakeword",
    )  # fmt: skip

    def status(self) -> EngineStatus:
        try:
            import onnxruntime

            return EngineStatus(self.id, True, str(onnxruntime.__version__), None)
        except Exception as exc:
            return EngineStatus(
                self.id, False, None,
                f"{type(exc).__name__}: {exc}. Optional (end-of-turn detection, MarbleNet VAD, Nemotron diarization, DiCoW, LFM3-Audio, "
                "DeepFilterNet 3, language identification, speech emotion, forced alignment, DTLN-aec, openWakeWord): "
                "pip install onnxruntime",
            )  # fmt: skip

    def graphs(self, kind: str, path: Path, spec: dict[str, Any]) -> dict[str, int]:
        """DiCoW (encoder, decoder) and LFM3-Audio (four graphs) are placed graph by graph (bolo.placement)."""
        if kind == "dicow_onnx":
            from bolo.engines.dicow_onnx import graph_files

            return graph_files(path, spec)
        if kind == "lfm3_audio_onnx":
            from bolo.s2s_onnx import graph_files as lfm3_graph_files

            return lfm3_graph_files(path, spec)
        return {}

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind == "nemotron_diarization":
            from bolo.engines.nemotron import NemotronDiarization

            return NemotronDiarization(path, spec, opts)
        if kind == "dicow_onnx":
            from bolo.engines.dicow_onnx import DiCoWOnnx

            return DiCoWOnnx.from_dir(path, spec, opts)
        if kind == "lfm3_audio_onnx":
            from bolo.s2s_onnx import Lfm3Tts

            return Lfm3Tts.from_dir(path, spec, opts)
        if kind == "text_turn":
            from bolo.engines.text_turn import TextTurn

            return TextTurn(path, spec, opts)
        if kind == "fbank_speaker_embedding":
            from bolo.engines.speaker_embedding import FbankSpeakerEmbedding

            return FbankSpeakerEmbedding(path, spec, opts)
        if kind == "onnx_speaker_embedding":
            from bolo.engines.speaker_embedding import OnnxSpeakerEmbedding

            return OnnxSpeakerEmbedding(path, spec, opts)
        if kind == "marblenet_vad":
            from bolo.engines.frame_vad import MarbleNetVAD

            return MarbleNetVAD(path, spec, opts)
        if kind == "deepfilternet3_stream":
            from bolo.engines.denoisers import DeepFilterNetStream

            return DeepFilterNetStream(path, spec, opts)
        if kind == "whisper_lid":
            from bolo.engines.lid_onnx import WhisperLid

            return WhisperLid(path, spec, opts)
        if kind == "onnx_lid":
            from bolo.engines.lid_onnx import OnnxLid

            return OnnxLid(path, spec, opts)
        if kind == "openwakeword":
            from bolo.wake import OpenWakeWord

            return OpenWakeWord(path, spec, opts)
        if kind == "dtln_aec":
            from bolo.echo import DtlnAec

            return DtlnAec(path, spec, opts.threads)
        if kind == "onnx_emotion":
            from bolo.engines.audio_classify import OnnxEmotion

            return OnnxEmotion(path, spec, opts)
        if kind == "ctc_align":
            from bolo.engines.ctc_align import CtcAligner

            return CtcAligner(path, spec, opts)
        if kind != "smart_turn":
            raise ValueError(f"the onnxruntime engine cannot load kind {kind!r}")
        return SmartTurn(path, spec, opts)
