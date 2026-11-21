"""NVIDIA Canary-Qwen 2.5B in ONNX Runtime: a speech-augmented language model without NeMo or PyTorch.

Canary-Qwen (CC-BY-4.0) is a NeMo SALM: a FastConformer encoder (from Canary-1B-Flash) whose output is projected
into the embedding space of Qwen3-1.7B, which then writes the transcript. English only, with punctuation and
capitals. The ONNX export (OpenVoiceOS/canary-qwen-2.5b-onnx, made with its ``export_canary_qwen.py``) has three
graphs:

* ``encoder``: NeMo log-mel features ``(1, 128, T)`` and their length -> audio embeddings ``(1, L, 2048)``
* ``embed_tokens``: token ids -> embeddings
* ``decoder``: embeddings, an additive attention bias ``(1, 1, S, P + S)``, positions and the key/value cache of
  28 layers -> logits and the grown cache

The prompt is the Qwen chat template around the audio ("Transcribe the following: <audio>"), its token ids baked
into ``config.json`` (``prompt_prefix_ids``, ``prompt_suffix_ids``); the transcript is decoded greedily until an
end-of-turn token, and the tokens are turned back into text with the byte-level BPE vocabulary (``vocab.json``).

Settings (catalog ``options``): ``max_new_tokens`` (default 8 per second of audio plus 32, at most 440),
``encoder`` / ``embed_tokens`` / ``decoder`` (file names; the ``*_int8.onnx`` graphs by default).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo.audio.features import nemo_log_mel
from bolo.engines.base import LoadOptions, Segment, STTModel, STTRequest, STTResult
from bolo.ort_providers import create_session, providers_for, session_options

HOP = 160


def _byte_decoder() -> dict[str, int]:
    """GPT-2's map from the printable characters of byte-level BPE back to bytes."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def normalize_per_feature(mel: np.ndarray) -> np.ndarray:
    """NeMo's ``normalize: per_feature``: each mel band to zero mean and unit (unbiased) deviation, plus 1e-5."""
    if mel.shape[0] < 2:
        return mel - mel.mean(axis=0, keepdims=True)
    mean = mel.mean(axis=0, keepdims=True)
    std = mel.std(axis=0, ddof=1, keepdims=True) + 1e-5
    return ((mel - mean) / std).astype(np.float32)


class CanaryQwenOnnx(STTModel):
    #: trained on audio of up to 40 s; bolo cuts longer audio at pauses
    max_chunk_s = 30.0
    languages = ("en",)

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import onnxruntime as ort

        self.config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        self._tokens = {int(i): str(t) for t, i in vocab.items()}
        self._bytes = _byte_decoder()
        self._eos = {int(x) for x in self.config.get("eos_token_ids") or (151643, 151645)}
        self._prefix = np.asarray([self.config["prompt_prefix_ids"]], dtype=np.int64)
        self._suffix = np.asarray([self.config["prompt_suffix_ids"]], dtype=np.int64)
        self._layers = int(self.config.get("num_layers", 28))
        self._kv_heads = int(self.config.get("num_key_value_heads", 8))
        self._head_dim = int(self.config.get("head_dim", 128))
        self._max_tokens = int(spec.get("max_new_tokens", 0)) or None
        providers = providers_for(opts.device)

        def options() -> Any:
            o = session_options()
            o.intra_op_num_threads = max(1, opts.threads)
            o.inter_op_num_threads = 1
            o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return o

        def graph(role: str, default: str) -> Any:
            name = str(spec.get(role, default))
            return create_session(str(path / name), options(), providers, role=role)

        self.encoder = graph("encoder", "encoder_int8.onnx")
        self.embed = graph("embed_tokens", "embed_tokens_int8.onnx")
        self.decoder = graph("decoder", "decoder_int8.onnx")
        self._dtype = np.float16 if "float16" in self.decoder.get_inputs()[0].type else np.float32
        self._lock = threading.Lock()

    def _embed(self, ids: np.ndarray) -> np.ndarray:
        return np.asarray(self.embed.run(None, {"input_ids": ids})[0], dtype=self._dtype)

    def _text(self, ids: list[int]) -> str:
        chars = "".join(self._tokens.get(i, "") for i in ids)
        data = bytes(self._bytes[c] for c in chars if c in self._bytes)
        return data.decode("utf-8", errors="replace").strip()

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        with self._lock:
            return [self._one(r) for r in requests]

    def _one(self, req: STTRequest) -> STTResult:
        samples = np.asarray(req.samples, dtype=np.float32)
        duration = len(samples) / self.sample_rate
        if len(samples) < HOP * 4:
            return STTResult("", "en", [])
        mel = normalize_per_feature(nemo_log_mel(samples))  # (T, 128)
        feats = mel.T[None].astype(np.float32)
        audio, audio_len = self.encoder.run(
            None, {"input_features": feats, "input_features_lens": np.asarray([feats.shape[2]], dtype=np.int64)}
        )
        audio = np.asarray(audio[:, : int(np.asarray(audio_len).reshape(-1)[0])], dtype=self._dtype)
        embeds = np.concatenate([self._embed(self._prefix), audio, self._embed(self._suffix)], axis=1)
        past = [np.zeros((1, self._kv_heads, 0, self._head_dim), dtype=self._dtype) for _ in range(2 * self._layers)]
        budget = self._max_tokens or min(440, int(8 * duration) + 32)
        out: list[int] = []
        pos = 0
        for _ in range(budget):
            if req.abort.is_set():
                break
            s = embeds.shape[1]
            bias = np.zeros((1, 1, s, pos + s), dtype=self._dtype)
            if s > 1:  # causal mask over the prompt
                bias[0, 0, :, pos:] = np.triu(np.full((s, s), np.finfo(self._dtype).min, dtype=self._dtype), k=1)
            feed: dict[str, Any] = {
                "inputs_embeds": embeds,
                "attn_bias": bias,
                "position_ids": np.arange(pos, pos + s, dtype=np.int64)[None],
            }
            for i in range(self._layers):
                feed[f"past_key_values.{i}.key"] = past[2 * i]
                feed[f"past_key_values.{i}.value"] = past[2 * i + 1]
            logits, *past = self.decoder.run(None, feed)
            pos += s
            token = int(np.argmax(logits[0, -1]))
            if token in self._eos:
                break
            out.append(token)
            embeds = self._embed(np.asarray([[token]], dtype=np.int64))
        text = self._text(out)
        return STTResult(text, "en", [Segment(0.0, duration, text)] if text else [])
