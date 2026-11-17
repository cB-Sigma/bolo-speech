"""Smart Turn v2 in NumPy: end-of-turn detection from audio with Pipecat's wav2vec2-based model (kind ``smart_turn_v2``).

Smart Turn v2 (https://huggingface.co/pipecat-ai/smart-turn-v2, BSD-2-Clause) is wav2vec2-base with an attention
pooling layer and a small classifier, published as PyTorch weights only (``model.safetensors``). bolo reads the
safetensors file and computes the network itself, so neither PyTorch nor a converted file is needed. The reference
(``model.py`` and ``inference.py`` of github.com/pipecat-ai/smart-turn at the v2 release) pads every clip to 16 s with
an attention mask; the computation here gives the same answer while working on the real audio only:

* the clip is normalised to zero mean and unit variance over its real samples, as the feature extractor does;
* the first convolution has no bias, so over the zero padding it gives exact zeros, and its group norm (one group per
  channel) takes its statistics over all 799 frames of a 16 s window: the padding's share is added analytically;
* past that, the padding is the same constant in every frame of each layer, so it is carried as one vector per
  layer instead of thousands of frames;
* the transformer only runs over the frames that matter: the frames the encoder's mask keeps are its keys, and the
  frames the pooling's mask keeps (one or two more, as the reference samples the mask) are its queries; frames past
  the encoder's mask enter the positional convolution as zeros, as in the reference.

Clips longer than 16 s keep their last 16 s (the end is what the model judges).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import FloatArray, LoadOptions, TurnModel

SR = 16000
WINDOW = 16 * SR  # the reference pads (and cuts) every clip to 16 s
CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)
HEADS = 12
EPS = 1e-5


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """The float32 tensors of a safetensors file (8-byte header length, a JSON header, the raw data)."""
    data = path.read_bytes()
    size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + size])
    out: dict[str, np.ndarray] = {}
    base = 8 + size
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "F32":
            raise ValueError(f"{path.name}: tensor {name} is {info['dtype']}, expected F32")
        start, end = info["data_offsets"]
        out[name] = np.frombuffer(data, dtype="<f4", count=(end - start) // 4, offset=base + start).reshape(
            info["shape"]
        )
    return out


def erf(x: np.ndarray) -> np.ndarray:
    """Error function (Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7)."""
    sign = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-a * a))


def gelu(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))).astype(np.float32)


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    out: np.ndarray = ((x - mean) / np.sqrt(var + EPS) * w + b).astype(np.float32)
    return out


def conv_frames(n: int, kernel: int, stride: int) -> int:
    return (n - kernel) // stride + 1 if n >= kernel else 0


def conv1d(x: np.ndarray, w: np.ndarray, stride: int) -> np.ndarray:
    """Valid 1-D convolution without bias: ``x`` (time, in), ``w`` (out, in, kernel) -> (time', out)."""
    out_ch, in_ch, k = w.shape
    count = conv_frames(len(x), k, stride)
    if count <= 0:
        return np.zeros((0, out_ch), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(x, k, axis=0)[::stride][:count]  # (time', in, k)
    out: np.ndarray = (windows.reshape(count, in_ch * k) @ w.reshape(out_ch, in_ch * k).T).astype(np.float32)
    return out


class SmartTurnV2(TurnModel):
    """Smart Turn v2 computed in NumPy from its safetensors weights (CPU)."""

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        merged = {**spec, **opts.options}
        files = sorted(path.glob(str(merged.get("file", "model.safetensors"))))
        if not files:
            raise FileNotFoundError(f"no model.safetensors in {path}")
        t = read_safetensors(files[0])
        p = "wav2vec2."
        self.convs = [t[f"{p}feature_extractor.conv_layers.{i}.conv.weight"] for i in range(7)]
        self.gn = (t[f"{p}feature_extractor.conv_layers.0.layer_norm.weight"],
                   t[f"{p}feature_extractor.conv_layers.0.layer_norm.bias"])  # fmt: skip
        self.fp_ln = (t[f"{p}feature_projection.layer_norm.weight"], t[f"{p}feature_projection.layer_norm.bias"])
        self.fp = (t[f"{p}feature_projection.projection.weight"].T.copy(), t[f"{p}feature_projection.projection.bias"])
        g = t[f"{p}encoder.pos_conv_embed.conv.parametrizations.weight.original0"]
        v = t[f"{p}encoder.pos_conv_embed.conv.parametrizations.weight.original1"]
        # weight norm over every dimension but the kernel's (dim=2)
        self.pos_w = (g * v / np.sqrt((v.astype(np.float64) ** 2).sum(axis=(0, 1), keepdims=True))).astype(np.float32)
        self.pos_b = t[f"{p}encoder.pos_conv_embed.conv.bias"]
        self.enc_ln = (t[f"{p}encoder.layer_norm.weight"], t[f"{p}encoder.layer_norm.bias"])
        self.layers = []
        i = 0
        while f"{p}encoder.layers.{i}.attention.q_proj.weight" in t:
            q = f"{p}encoder.layers.{i}."
            self.layers.append({
                "qkv_w": np.concatenate([t[q + f"attention.{n}_proj.weight"] for n in "qkv"]).T.copy(),
                "qkv_b": np.concatenate([t[q + f"attention.{n}_proj.bias"] for n in "qkv"]),
                "o_w": t[q + "attention.out_proj.weight"].T.copy(), "o_b": t[q + "attention.out_proj.bias"],
                "ln": (t[q + "layer_norm.weight"], t[q + "layer_norm.bias"]),
                "ff1_w": t[q + "feed_forward.intermediate_dense.weight"].T.copy(),
                "ff1_b": t[q + "feed_forward.intermediate_dense.bias"],
                "ff2_w": t[q + "feed_forward.output_dense.weight"].T.copy(),
                "ff2_b": t[q + "feed_forward.output_dense.bias"],
                "ln2": (t[q + "final_layer_norm.weight"], t[q + "final_layer_norm.bias"]),
            })  # fmt: skip
            i += 1
        self.pool = [(t[f"pool_attention.{k}.weight"].T.copy(), t[f"pool_attention.{k}.bias"]) for k in (0, 2)]
        self.cls = {k: (t[f"classifier.{k}.weight"], t[f"classifier.{k}.bias"]) for k in (0, 1, 4, 6)}
        self.frames_total = WINDOW
        for k, s in zip(CONV_KERNEL, CONV_STRIDE, strict=True):
            self.frames_total = conv_frames(self.frames_total, k, s)
        self._lock = threading.Lock()

    # -- the network ------------------------------------------------------------------------------------
    def _features(self, x: np.ndarray) -> np.ndarray:
        """Conv feature extractor over a 16 s window whose samples past ``len(x)`` are zero: the frames that see real
        audio, then the padding's (constant) frame repeated up to the window's frame count, (frames, 512)."""
        total = WINDOW
        w0 = self.convs[0]
        # every frame whose window reaches real audio (the zeros after it are part of the window)
        h = conv1d(np.concatenate([x, np.zeros(CONV_KERNEL[0], np.float32)])[:, None], w0, CONV_STRIDE[0])
        total = conv_frames(total, CONV_KERNEL[0], CONV_STRIDE[0])
        # group norm (one group per channel) over all frames of the window: the rest are exact zeros
        mean = h.astype(np.float64).sum(axis=0) / total
        var = (h.astype(np.float64) ** 2).sum(axis=0) / total - mean**2
        scale = self.gn[0] / np.sqrt(var + EPS)
        h = gelu(((h - mean) * scale + self.gn[1]).astype(np.float32))
        tail = gelu(((0.0 - mean) * scale + self.gn[1]).astype(np.float32)[None])[0]  # the padding's frames
        for w, k, s in zip(self.convs[1:], CONV_KERNEL[1:], CONV_STRIDE[1:], strict=True):
            total = conv_frames(total, k, s)
            need = min(total, -(-len(h) // s) + 1)  # every frame whose window reaches the computed frames
            padded = np.concatenate([h, np.repeat(tail[None], need * s + k, axis=0)])
            h = gelu(conv1d(padded, w, s)[:need])
            tail = gelu((w.sum(axis=2) @ tail)[None])[0]
        return h

    def _attention(self, h: np.ndarray, layer: dict[str, Any], keys: int) -> np.ndarray:
        n, dim = h.shape
        hd = dim // HEADS
        qkv = h @ layer["qkv_w"] + layer["qkv_b"]
        # per-head products through einsum: small batched BLAS calls cost more in thread start-up than in arithmetic
        q = np.ascontiguousarray(qkv[:, :dim].reshape(n, HEADS, hd).transpose(1, 0, 2) * np.float32(hd**-0.5))
        k = np.ascontiguousarray(qkv[:keys, dim : 2 * dim].reshape(keys, HEADS, hd).transpose(1, 2, 0))
        v = np.ascontiguousarray(qkv[:keys, 2 * dim :].reshape(keys, HEADS, hd).transpose(1, 0, 2))
        scores = np.einsum("hqd,hdk->hqk", q, k)
        scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        scores /= scores.sum(axis=-1, keepdims=True)
        out = np.einsum("hqk,hkd->qhd", scores.astype(np.float32), v).reshape(n, dim)
        result: np.ndarray = (out @ layer["o_w"] + layer["o_b"]).astype(np.float32)
        return result

    def probability(self, samples: FloatArray) -> float:
        x = np.asarray(samples, dtype=np.float32)[-WINDOW:]
        n = len(x)
        if n < 400:
            return 0.0
        x = ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)
        with self._lock:
            return self._probability(x, n)

    def _probability(self, x: np.ndarray, n: int) -> float:
        feats = self._features(x)
        # frames the encoder keeps (feature-level attention mask from the real length)
        keep = n
        for k, s in zip(CONV_KERNEL, CONV_STRIDE, strict=True):
            keep = conv_frames(keep, k, s)
        # frames the pooling keeps: the reference samples the sample mask at int(i * WINDOW / frames)
        ratio = WINDOW / self.frames_total
        pooled = int(np.sum((np.arange(self.frames_total) * ratio).astype(np.int64) < n))
        frames = max(keep, pooled)
        feats = feats[:frames]
        h = layer_norm(feats, *self.fp_ln) @ self.fp[0] + self.fp[1]
        h[keep:] = 0.0  # masked frames are zeroed before the positional convolution
        # positional convolution: kernel 128, padding 64, 16 groups, last output frame dropped, GELU
        groups = 16
        cg = h.shape[1] // groups
        padded = np.concatenate([np.zeros((64, h.shape[1]), np.float32), h, np.zeros((64, h.shape[1]), np.float32)])
        pos = np.empty_like(h)
        for gi in range(groups):
            sl = slice(gi * cg, (gi + 1) * cg)
            pos[:, sl] = conv1d(padded[:, sl], self.pos_w[sl], 1)[:frames]
        h = h + gelu(pos + self.pos_b)
        h = layer_norm(h, *self.enc_ln)
        for layer in self.layers:
            h = layer_norm(h + self._attention(h, layer, keep), *layer["ln"])
            ff = gelu(h @ layer["ff1_w"] + layer["ff1_b"]) @ layer["ff2_w"] + layer["ff2_b"]
            h = layer_norm(h + ff, *layer["ln2"])
        # attention pooling over the frames the pooling mask keeps, then the classifier
        hp = h[:pooled]
        a = np.tanh(hp @ self.pool[0][0] + self.pool[0][1]) @ self.pool[1][0] + self.pool[1][1]
        a = np.exp(a - a.max())
        pooled_vec = (hp * (a / a.sum())).sum(axis=0)
        c = self.cls
        z = pooled_vec @ c[0][0].T + c[0][1]
        z = gelu(layer_norm(z, c[1][0], c[1][1]))
        z = gelu(z @ c[4][0].T + c[4][1])
        logit = float((z @ c[6][0].T + c[6][1])[0])
        return float(1.0 / (1.0 + np.exp(-logit)))
