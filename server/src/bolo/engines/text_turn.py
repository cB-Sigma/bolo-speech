"""End-of-turn detection from the words: has the speaker finished the sentence, judged from its transcript so far.

Acoustic models (Smart Turn, :mod:`bolo.engines.onnx_turn`) listen to the end of an utterance; the models here read
it. A Realtime session that uses one decodes the utterance with its transcription model when the voice-activity
detector reports a pause, and asks the model about that text (:meth:`bolo.realtime.RealtimeSession._turn_complete`).

The supported family is VideoSDK's Namo Turn Detector v1 (Apache-2.0): a sequence classifier (DistilBERT for the
single-language models, mmBERT for the multilingual one) quantised to int8 ONNX, with a Hugging Face ``tokenizer.json``.
Its two logits are "not finished" and "finished"; the probability of the second is returned. It reads one utterance,
without the conversation before it.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import FloatArray, LoadOptions, TurnModel


class TextTurn(TurnModel):
    """A text classifier on ONNX Runtime (CPU) with a ``tokenizers`` tokenizer."""

    hears = "text"

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from tokenizers import Tokenizer

        from bolo.ort_providers import CPU, create_session, session_options

        merged = {**spec, **opts.options}
        files = sorted(path.glob(str(merged.get("model", "*.onnx"))))
        if not files:
            raise FileNotFoundError(f"no ONNX file in {path}")
        self.tokenizer = Tokenizer.from_file(str(path / str(merged.get("tokenizer", "tokenizer.json"))))
        self.max_tokens = int(merged.get("max_tokens", 512))
        # a long utterance loses its start, not its end, which is what the model judges
        self.tokenizer.enable_truncation(self.max_tokens, direction="left")
        self.tokenizer.no_padding()
        #: index of the "finished" class among the logits
        self.finished_label = int(merged.get("finished_label", 1))
        options = session_options()
        options.intra_op_num_threads = max(1, min(2, opts.threads))
        options.inter_op_num_threads = 1
        self._session = create_session(str(files[0]), options, [CPU])
        self._inputs = [i.name for i in self._session.get_inputs()]
        self._lock = threading.Lock()

    def probability(self, samples: FloatArray) -> float:
        raise NotImplementedError("this end-of-turn model reads text: use probability_text")

    def probability_text(self, text: str) -> float:
        encoding = self.tokenizer.encode(" ".join(text.split()))
        ids = np.asarray([encoding.ids], dtype=np.int64)
        feeds = {
            "input_ids": ids,
            "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
            "token_type_ids": np.zeros_like(ids),
        }
        with self._lock:
            logits = self._session.run(None, {k: v for k, v in feeds.items() if k in self._inputs})[0]
        row = np.asarray(logits, dtype=np.float64).reshape(-1)
        e = np.exp(row - row.max())
        return float(e[self.finished_label] / e.sum())
