"""Spoken-language identification with ONNX Runtime (task ``lid``).

Two kinds of model:

* ``whisper_lid``: Whisper's own language detection, run on its own. The encoder hears the first 30 seconds and
  the decoder, given only the start-of-transcript token, scores the next token; the scores of the language tokens,
  put through a softmax, are the probability of each of Whisper's 99 languages (what Whisper's ``detect_language``
  does). It reads the int8 encoder and decoder that sherpa-onnx's Whisper exports are made of, whose metadata lists
  the language tokens and codes.
* ``onnx_lid``: one graph from audio (or from filterbank features) to one score per language: the VoxLingua107
  ECAPA-TDNN exports, AmberNet, and the MMS-LID wav2vec 2.0 classifiers. Catalog options say what the graph takes
  (``features``: ``raw`` or ``fbank60``; ``normalize``: zero mean and unit variance first, as wav2vec 2.0 wants)
  and where the labels are (``labels``: a ``lang_map.json``, a ``labels.txt``, or a ``config.json`` with ``labels``
  or ``id2label``).

Both answer every language they know with a probability, most likely first; codes are made comparable with the
catalog's (:mod:`bolo.text.iso639`).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import FloatArray, LidModel, LoadOptions
from bolo.ort_providers import create_session, providers_for, session_options
from bolo.text.iso639 import normalize

RATE = 16000
WHISPER_WINDOW = 30 * RATE


def _softmax(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    x = np.exp(x - x.max())
    out: np.ndarray = x / x.sum()
    return out


def _ranked(labels: list[str], probs: np.ndarray) -> list[tuple[str, float]]:
    """Labels with their probability, most likely first; labels that normalise to one code are added up."""
    total: dict[str, float] = {}
    for label, p in zip(labels, probs, strict=False):
        total[label] = total.get(label, 0.0) + float(p)
    return sorted(total.items(), key=lambda kv: -kv[1])


def _find(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no file matching {pattern!r} in {path}")
    return files[0]


def _options(threads: int) -> Any:
    options = session_options()
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    return options


def read_labels(file: Path) -> list[str]:
    """Class index -> language code, from a ``lang_map.json`` (``{"0": {"iso": "ab"}}``), a ``config.json``
    (``labels`` list or ``id2label`` map) or a text file of one code per line (``ab`` or ``ab: Abkhazian``)."""
    text = file.read_text(encoding="utf-8")
    if file.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("labels"), list):
            return [normalize(str(x)) for x in data["labels"]]
        table = data.get("id2label", data) if isinstance(data, dict) else {}
        out: list[str] = []
        for key in sorted(table, key=int):
            value = table[key]
            out.append(normalize(str(value.get("iso") if isinstance(value, dict) else value)))
        return out
    return [normalize(line.split(":")[0]) for line in text.splitlines() if line.strip()]


class WhisperLid(LidModel):
    """Whisper's language detection on sherpa-onnx's int8 Whisper graphs (encoder and decoder)."""

    max_s = 30.0

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        threads = max(1, min(4, opts.threads))
        providers = providers_for(opts.device)
        self._encoder = create_session(
            str(_find(path, str(spec.get("encoder", "*encoder*.onnx")))), _options(threads), providers
        )
        self._decoder = create_session(
            str(_find(path, str(spec.get("decoder", "*decoder*.onnx")))), _options(threads), providers
        )
        meta = self._encoder.get_modelmeta().custom_metadata_map
        if meta.get("is_multilingual", "1") == "0" or not meta.get("all_language_tokens"):
            raise ValueError(f"{path}: an English-only Whisper model cannot identify languages")
        self._tokens = [int(t) for t in meta["all_language_tokens"].split(",")]
        self.labels = [normalize(c) for c in meta["all_language_codes"].split(",")]
        self._sot = int(meta["sot"])
        self._layers = int(meta["n_text_layer"])
        self._state = int(meta["n_text_state"])
        self._ctx = int(meta["n_text_ctx"])
        self._mels = int(meta.get("n_mels", 80))
        self._lock = threading.Lock()

    def identify(self, samples: FloatArray) -> list[tuple[str, float]]:
        from bolo.audio.features import whisper_log_mel

        audio = np.asarray(samples[:WHISPER_WINDOW], dtype=np.float32)
        audio = np.pad(audio, (0, WHISPER_WINDOW - len(audio)))  # Whisper hears 30 s, padded with silence
        mel = whisper_log_mel(audio, n_mels=self._mels)[None]
        cache = np.zeros((self._layers, 1, self._ctx, self._state), dtype=np.float32)
        with self._lock:
            cross_k, cross_v = self._encoder.run(None, {"mel": mel})
            logits = self._decoder.run(
                None,
                {
                    "tokens": np.array([[self._sot]], dtype=np.int64),
                    "in_n_layer_self_k_cache": cache,
                    "in_n_layer_self_v_cache": cache,
                    "n_layer_cross_k": cross_k,
                    "n_layer_cross_v": cross_v,
                    "offset": np.array([0], dtype=np.int64),
                },
            )[0]
        scores = np.asarray(logits)[0, -1, self._tokens]
        return _ranked(self.labels, _softmax(scores))


class OnnxLid(LidModel):
    """A single-graph language classifier: audio (or 60-band filterbanks) in, one score per language out."""

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        threads = max(1, min(4, opts.threads))
        self._session = create_session(
            str(_find(path, str(spec.get("model", "*.onnx")))), _options(threads), providers_for(opts.device)
        )
        self.labels = read_labels(_find(path, str(spec.get("labels", "*.json"))))
        self.max_s = float(spec.get("max_s", 30.0))
        self._features = str(spec.get("features", "raw"))
        if self._features not in ("raw", "fbank60"):
            raise ValueError(f"onnx_lid: features must be 'raw' or 'fbank60', not {self._features!r}")
        self._normalize = bool(spec.get("normalize", False))
        inputs = self._session.get_inputs()
        self._input = inputs[0].name
        self._length = inputs[1].name if len(inputs) > 1 else None  # AmberNet: audio_len
        self._lock = threading.Lock()

    def identify(self, samples: FloatArray) -> list[tuple[str, float]]:
        x = np.asarray(samples[: int(self.max_s * RATE)], dtype=np.float32)
        if len(x) < RATE // 10:
            x = np.pad(x, (0, RATE // 10 - len(x)))
        if self._normalize:
            x = ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)
        feed: dict[str, Any] = {self._input: fbank60(x) if self._features == "fbank60" else x[None]}
        if self._length is not None:
            feed[self._length] = np.array([len(x)], dtype=np.int64)
        with self._lock:
            scores = np.asarray(self._session.run(None, feed)[0]).reshape(-1)
        n = min(len(scores), len(self.labels))
        return _ranked(self.labels[:n], _softmax(scores[:n]))


# -- SpeechBrain's Fbank(n_mels=60) in NumPy ----------------------------------------------------------------
_FB_N_FFT = 400
_FB_HOP = 160
_FB_WINDOW = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(_FB_N_FFT) / _FB_N_FFT)  # periodic Hamming


def _fbank_matrix(n_mels: int = 60, n_fft: int = _FB_N_FFT, rate: int = RATE) -> np.ndarray:
    """SpeechBrain's triangular filters on the HTK mel scale, 0 to rate/2, shape ``(n_fft // 2 + 1, n_mels)``."""

    def to_mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    points = to_hz(np.linspace(to_mel(np.array(0.0)), to_mel(np.array(rate / 2.0)), n_mels + 2))
    band = np.diff(points)[:-1]
    central = points[1:-1]
    freqs = np.linspace(0.0, rate / 2.0, n_fft // 2 + 1)
    slope = (freqs[:, None] - central[None, :]) / band[None, :]
    matrix: np.ndarray = np.maximum(0.0, np.minimum(slope + 1.0, 1.0 - slope))
    return matrix


_FB_MATRIX = _fbank_matrix()


def fbank60(audio: FloatArray) -> FloatArray:
    """SpeechBrain ``Fbank(n_mels=60)`` features ``(1, frames, 60)``: centred 25 ms Hamming frames every 10 ms
    (zero padded), power spectrum, 60 mel bands, 10 log10 floored 80 dB under the loudest value."""
    x = np.pad(np.asarray(audio, dtype=np.float64), _FB_N_FFT // 2)
    count = 1 + (len(x) - _FB_N_FFT) // _FB_HOP
    index = np.arange(_FB_N_FFT)[None, :] + _FB_HOP * np.arange(count)[:, None]
    power = np.abs(np.fft.rfft(x[index] * _FB_WINDOW, axis=1)) ** 2
    db = 10.0 * np.log10(np.maximum(power @ _FB_MATRIX, 1e-10))
    db = np.maximum(db, db.max() - 80.0)
    return np.ascontiguousarray(db[None], dtype=np.float32)
