"""Forced alignment with a character-level CTC model in ONNX Runtime (task ``align``, kind ``ctc_align``).

The graph takes 16 kHz audio (``x``, shape ``(1, samples)``; zero mean and unit variance first when ``normalize``)
and gives CTC logits per 20 ms frame. Catalog options:

* ``model``: the ONNX file; ``tokens``: ``tokens.txt`` (``symbol id`` per line, sherpa-onnx's form) or a
  ``vocab.json`` (``{symbol: id}``, Hugging Face's);
* ``blank``: the blank's id (default 0); ``separator``: a token the model writes between words (a space), if any;
* ``romanize: uroman``: the text is romanized with uroman first (MMS's forced aligner knows only Latin letters);
  without the ``uroman`` package, only text in Latin letters can be aligned.

Long audio is scored in windows of ``window_s`` seconds with two seconds of context on each side, so the frames
add up to the whole recording; the alignment itself is one Viterbi pass (bolo.alignment).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo.alignment import AlignedWord, align_words, char_tokenizer, log_softmax, split_words
from bolo.engines.base import AlignModel, FloatArray, LoadOptions
from bolo.ort_providers import create_session, providers_for, session_options

RATE = 16000
#: wav2vec 2.0's frame: 320 samples (20 ms)
HOP = 320
CONTEXT_S = 2.0


def read_vocab(file: Path) -> dict[str, int]:
    if file.suffix == ".json":
        data = json.loads(file.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in data.items()}
    vocab: dict[str, int] = {}
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        symbol, _, index = line.rpartition(" ")
        if not symbol and line.startswith(" "):
            symbol = " "  # the space token: "  4"
        vocab[symbol] = int(index)
    return vocab


def _uroman() -> Any:
    try:
        import uroman  # type: ignore[import-not-found,import-untyped,unused-ignore]
    except ImportError:
        return None
    romanizer = uroman.Uroman()
    return lambda text: str(romanizer.romanize_string(text))


class CtcAligner(AlignModel):
    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        options = session_options()
        options.intra_op_num_threads = max(1, opts.threads)
        options.inter_op_num_threads = 1
        model = sorted(path.glob(str(spec.get("model", "*.onnx"))))
        if not model:
            raise FileNotFoundError(f"no {spec.get('model', '*.onnx')} in {path}")
        self._session = create_session(str(model[0]), options, providers_for(opts.device))
        self._input = self._session.get_inputs()[0].name
        vocab = read_vocab(path / str(spec.get("tokens", "tokens.txt")))
        self.blank = int(spec.get("blank", 0))
        sep = spec.get("separator")
        self.separator = vocab.get(str(sep)) if sep is not None else None
        romanize = None
        if spec.get("romanize") == "uroman":
            romanize = _uroman()
            #: False when the text must already be in Latin letters (the uroman package is not installed)
            self.romanizes = romanize is not None
        else:
            self.romanizes = False
        self.tokenize = char_tokenizer(vocab, romanize)
        self.normalize = bool(spec.get("normalize", True))
        self.window_s = float(spec.get("window_s", 30.0))
        self.languages = tuple(spec.get("languages") or ())
        self._lock = threading.Lock()

    def _logits(self, x: FloatArray) -> np.ndarray:
        if self.normalize:
            x = ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)
        with self._lock:
            out = self._session.run(None, {self._input: x[None].astype(np.float32)})[0]
        logits: np.ndarray = np.asarray(out)[0]
        return logits

    def emissions(self, samples: FloatArray) -> np.ndarray:
        """Log probabilities, (frames, vocabulary), one frame per 20 ms of ``samples``."""
        x = np.asarray(samples, dtype=np.float32)
        frames = max(1, len(x) // HOP)
        if len(x) < HOP * 2:
            x = np.pad(x, (0, HOP * 2 - len(x)))
        window = int(self.window_s * RATE) // HOP * HOP
        context = int(CONTEXT_S * RATE) // HOP * HOP
        if len(x) <= window + 2 * context:
            out = log_softmax(self._logits(x))
        else:
            pieces = []
            for start in range(0, len(x), window):
                lo = max(0, start - context)
                hi = min(len(x), start + window + context)
                logits = self._logits(x[lo:hi])
                skip = (start - lo) // HOP
                keep = min(window, len(x) - start) // HOP
                pieces.append(log_softmax(logits[skip : skip + keep]))
            out = np.concatenate(pieces)
        if len(out) < frames:  # the last partial frame
            out = np.concatenate([out, np.repeat(out[-1:], frames - len(out), axis=0)])
        return out[:frames]

    def align(self, samples: FloatArray, text: str, language: str | None = None) -> list[AlignedWord]:
        words = split_words(text)
        if not words:
            return []
        tokens = [self.tokenize(w) for w in words]
        log_probs = self.emissions(samples)
        return align_words(log_probs, HOP / RATE, words, tokens, self.blank, self.separator)
