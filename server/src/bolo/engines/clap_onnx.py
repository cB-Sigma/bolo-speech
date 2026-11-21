"""CLAP (contrastive language-audio pretraining) on ONNX Runtime: kind ``clap_onnx``, task ``audio-classify``.

Two graphs from bolo's exports (codebucketsolutions/*-onnx on Hugging Face, made by ``tools/export_clap_onnx.py``):

* the audio tower, ``audio`` float32 ``[batch, clip_samples]`` at ``sample_rate`` -> ``embedding`` ``[batch, D]``,
  with the log-mel front end inside the graph;
* the text tower, ``input_ids`` and ``attention_mask`` int64 ``[batch, tokens]`` -> ``embedding`` ``[batch, D]``,
  with the tokenizer in ``tokenizer.json`` (Hugging Face ``tokenizers``).

``clap.json`` holds the rest: sample rate, clip length, how a short clip is filled (``repeatpad``: LAION's
repeat-then-zero-pad; ``repeat``: MS-CLAP's repeat-and-cut), the logit scale, the token limit, the pad id, whether
``<|endoftext|>`` ends each text (MS-CLAP's GPT-2), and the prompt a label is put into.

Text embeddings are cached per model (labels are asked for again and again), up to ``TEXT_CACHE`` texts.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import AudioTextModel, FloatArray, LoadOptions

#: text embeddings kept per model
TEXT_CACHE = 4096
#: clips per audio-tower run
AUDIO_BATCH = 4


def fill(x: FloatArray, n: int, padding: str) -> FloatArray:
    """``x`` made exactly ``n`` samples long the way the model was trained: ``repeatpad`` repeats the clip whole as
    often as it fits, then pads with zeros (LAION CLAP); ``repeat`` repeats it and cuts (MS-CLAP)."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) >= n:
        return x[:n]
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    if padding == "repeat":
        return np.tile(x, -(-n // len(x)))[:n]
    x = np.tile(x, n // len(x))
    return np.pad(x, (0, n - len(x)))


class ClapOnnx(AudioTextModel):
    """Options (catalog ``options``): ``audio`` and ``text`` (the graph files, default ``audio.onnx`` and
    ``text.onnx``), ``config`` (``clap.json``), ``tokenizer`` (``tokenizer.json``)."""

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from tokenizers import Tokenizer

        from bolo.engines.lid_onnx import _options
        from bolo.ort_providers import create_session, providers_for

        def file(key: str, default: str) -> Path:
            found = path / str(spec.get(key, default))
            if not found.is_file():
                raise FileNotFoundError(f"clap_onnx: {found.name} is missing from {path}")
            return found

        cfg = json.loads(file("config", "clap.json").read_text(encoding="utf-8"))
        self.sample_rate = int(cfg["sample_rate"])
        self.clip_samples = int(cfg["clip_samples"])
        self.clip_s = self.clip_samples / self.sample_rate
        self.padding = str(cfg.get("padding", "repeatpad"))
        self.logit_scale = float(cfg.get("logit_scale", 1.0))
        self.prompt = str(spec.get("prompt") or cfg.get("prompt") or self.prompt)
        self.max_tokens = int(cfg.get("max_tokens", 77))
        self.pad_id = int(cfg.get("pad_id", 0))
        self.append_eos = bool(cfg.get("append_eos", False))
        self.dim = int(cfg.get("embedding_dim", 512))
        threads = max(1, opts.threads)
        providers = providers_for(opts.device)
        self._audio = create_session(str(file("audio", "audio.onnx")), _options(threads), providers)
        self._text = create_session(str(file("text", "text.onnx")), _options(threads), providers)
        self._tokenizer = Tokenizer.from_file(str(file("tokenizer", "tokenizer.json")))
        self._cache: OrderedDict[str, FloatArray] = OrderedDict()
        self._lock = threading.Lock()

    def embed_audio(self, clips: list[FloatArray]) -> FloatArray:
        rows = [fill(c, self.clip_samples, self.padding) for c in clips]
        out = []
        for i in range(0, len(rows), AUDIO_BATCH):
            batch = np.stack(rows[i : i + AUDIO_BATCH]).astype(np.float32)
            with self._lock:
                out.append(np.asarray(self._audio.run(None, {"audio": batch})[0], dtype=np.float32))
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def _tokens(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        suffix = " <|endoftext|>" if self.append_eos else ""
        rows = [self._tokenizer.encode(t + suffix).ids[: self.max_tokens] for t in texts]
        width = max([1, *(len(r) for r in rows)])
        ids = np.full((len(rows), width), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = r
            mask[i, : len(r)] = 1
        return ids, mask

    def embed_text(self, texts: list[str]) -> FloatArray:
        found: dict[str, FloatArray] = {}
        with self._lock:
            for t in texts:
                if t in self._cache:
                    self._cache.move_to_end(t)
                    found[t] = self._cache[t]
        missing = [t for t in dict.fromkeys(texts) if t not in found]
        if missing:
            ids, mask = self._tokens(missing)
            with self._lock:
                vectors = np.asarray(self._text.run(None, {"input_ids": ids, "attention_mask": mask})[0], np.float32)
                for t, v in zip(missing, vectors):
                    found[t] = v
                    self._cache[t] = v
                    while len(self._cache) > TEXT_CACHE:
                        self._cache.popitem(last=False)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([found[t] for t in texts])

    @property
    def cached_texts(self) -> int:
        return len(self._cache)
