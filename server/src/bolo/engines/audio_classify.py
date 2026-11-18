"""Audio classifiers: sound-event tagging (task ``audio-tagging``) and speech emotion (task ``emotion``).

* ``audio_tagging`` (sherpa-onnx): CED (Consistent Ensemble Distillation, tiny to base) and the Zipformer audio
  taggers from icefall, trained on AudioSet: 527 classes (speech, music, dog, siren, ...), each with its own
  probability. The class names come from AudioSet's ``class_labels_indices.csv`` shipped with the model.
* ``onnx_emotion`` (ONNX Runtime): one graph from the 16 kHz waveform to one score per emotion, such as the
  ONNX exports of emotion2vec+ (FunASR's utterance path: normalised waveform, encoder, mean over time, linear head,
  softmax). The labels are listed in the catalog entry.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import AudioTaggingModel, EmotionModel, FloatArray, LoadOptions

RATE = 16000


def read_audioset_labels(file: Path) -> list[str]:
    """``class_labels_indices.csv`` (``index,mid,display_name``) as a list of display names by index."""
    with file.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    labels: dict[int, str] = {}
    for row in rows[1:]:
        if len(row) >= 3 and row[0].strip().isdigit():
            labels[int(row[0])] = row[2].strip()
    return [labels.get(i, str(i)) for i in range(max(labels) + 1)] if labels else []


class SherpaAudioTagging(AudioTaggingModel):
    """CED or Zipformer audio tagging through sherpa-onnx's ``AudioTagging`` (option ``arch``: ``ced`` or
    ``zipformer``). Every class comes back with its probability (independent sigmoids, not a softmax)."""

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import sherpa_onnx as so

        from bolo.engines.sherpa import find_file
        from bolo.ort_providers import sherpa_provider

        model = find_file(path, str(spec.get("model", "model.int8.onnx")), "*.onnx")
        labels = find_file(path, str(spec.get("labels", "class_labels_indices.csv")))
        arch = str(spec.get("arch", "ced"))
        common: dict[str, Any] = {"num_threads": max(1, opts.threads), "provider": sherpa_provider(opts.device)}
        if arch == "ced":
            model_config = so.AudioTaggingModelConfig(ced=model, **common)
        elif arch == "zipformer":
            model_config = so.AudioTaggingModelConfig(
                zipformer=so.OfflineZipformerAudioTaggingModelConfig(model=model), **common
            )
        else:
            raise ValueError(f"audio_tagging: arch must be 'ced' or 'zipformer', not {arch!r}")
        self.labels = read_audioset_labels(Path(labels))
        self._tagger = so.AudioTagging(so.AudioTaggingConfig(model=model_config, labels=labels, top_k=len(self.labels)))
        self.max_s = float(spec.get("max_s", 30.0))
        self._lock = threading.Lock()

    def classify(self, samples: FloatArray) -> list[tuple[str, float]]:
        x = np.ascontiguousarray(samples[: int(self.max_s * RATE)], dtype=np.float32)
        if len(x) < RATE // 10:
            x = np.pad(x, (0, RATE // 10 - len(x)))
        with self._lock:
            stream = self._tagger.create_stream()
            stream.accept_waveform(RATE, x)
            events = self._tagger.compute(stream, len(self.labels))
        out = [(str(e.name), float(e.prob)) for e in events]
        out.sort(key=lambda t: -t[1])
        return out


class OnnxEmotion(EmotionModel):
    """A single-graph emotion classifier: the 16 kHz waveform ``(1, samples)`` in, one score per class out.

    Options: ``labels`` (the classes in output order), ``model`` (the graph's file name), ``outputs``
    (``probabilities``: the graph ends in a softmax; ``logits``: bolo applies one), ``normalize`` (bolo normalises
    the waveform to zero mean and unit variance first; the emotion2vec+ exports do it inside the graph),
    ``max_s`` (the longest stretch heard, 30 s)."""

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from bolo.engines.lid_onnx import _find, _options
        from bolo.ort_providers import create_session, providers_for

        threads = max(1, min(4, opts.threads))
        self._session = create_session(
            str(_find(path, str(spec.get("model", "*.onnx")))), _options(threads), providers_for(opts.device)
        )
        self.labels = [str(x) for x in spec.get("labels") or []]
        if not self.labels:
            raise ValueError("onnx_emotion: the catalog entry lists no labels")
        self._softmax = str(spec.get("outputs", "probabilities")) == "logits"
        self._normalize = bool(spec.get("normalize", False))
        self.max_s = float(spec.get("max_s", 30.0))
        self._input = self._session.get_inputs()[0].name
        self._lock = threading.Lock()

    def classify(self, samples: FloatArray) -> list[tuple[str, float]]:
        x = np.asarray(samples[: int(self.max_s * RATE)], dtype=np.float32)
        if len(x) < RATE // 2:  # the encoder needs a few frames: pad very short clips with silence
            x = np.pad(x, (0, RATE // 2 - len(x)))
        if self._normalize:
            x = ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)
        with self._lock:
            scores = np.asarray(self._session.run(None, {self._input: x[None]})[0], dtype=np.float64).reshape(-1)
        if self._softmax:
            e = np.exp(scores - scores.max())
            scores = e / e.sum()
        n = min(len(scores), len(self.labels))
        out = [(self.labels[i], float(scores[i])) for i in range(n)]
        out.sort(key=lambda t: -t[1])
        return out
