"""Rev's Reverb ASR (English, Rev Model Non-Production License) from its TorchScript export, without Rev's package.

Reverb ASR is a WeNet joint CTC/attention Conformer trained on 200,000 hours of English. Rev's own inference package
(``rev-reverb``) pins torch 2.2.2 and its WeNet code no longer imports with current torch, so the plugin runs the
TorchScript file the repository also ships (``reverb_asr_v1.jit.zip``), through the interface WeNet exports for its
C++ runtime: ``forward_encoder_chunk`` (the whole utterance as one chunk; the global CMVN is part of the encoder)
and ``ctc_activation``, then greedy CTC decoding over the BPE units of ``tk.units.txt``.

Features are Kaldi filterbanks (torchaudio's ``compliance.kaldi.fbank``) of the waveform at 16-bit scale, with the
sizes of ``config.yaml``'s ``dataset_conf.fbank_conf`` (read with ``yaml.safe_load``; 80 bands, 25 ms windows, 10 ms
shift otherwise). Reverb's "verbatimicity" (1.0: every word as spoken, fillers and repetitions included; 0.0: a
cleaned-up transcript) is the option ``verbatimicity``. No timestamps from this decoder; no translation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from bolo.engines.base import Segment, STTModel, STTRequest, STTResult

from bolo_torch.runtime import InvalidRequest, LoadContext, PluginError, model_dir, require

SR = 16000
JIT_FILE = "reverb_asr_v1.jit.zip"
UNITS_FILE = "tk.units.txt"
SPECIAL = frozenset({"<blank>", "<unk>", "<sos/eos>", "<sos>", "<eos>", "<pad>"})

Features = Callable[[np.ndarray], Any]


def read_units(path: Path) -> dict[int, str]:
    """``tk.units.txt``: one ``piece id`` per line."""
    units: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("-").isdigit():
            units[int(parts[-1])] = " ".join(parts[:-1])
    if not units:
        raise PluginError(f"{path} holds no units")
    return units


def ctc_greedy(log_probs: np.ndarray, blank: int = 0) -> list[int]:
    """Best path of a CTC output ``(T, V)``: repeats merged, blanks dropped."""
    best = np.asarray(log_probs).argmax(axis=-1).tolist()
    out: list[int] = []
    prev = -1
    for t in best:
        if t not in (prev, blank):
            out.append(int(t))
        prev = t
    return out


def pieces_to_text(ids: list[int], units: dict[int, str]) -> str:
    text = "".join(units.get(i, "") for i in ids if units.get(i, "") not in SPECIAL)
    return " ".join(text.replace("▁", " ").split())


def read_fbank_conf(root: Path) -> dict[str, float]:
    conf: dict[str, float] = {"num_mel_bins": 80, "frame_length": 25, "frame_shift": 10}
    path = root / "config.yaml"
    if path.is_file():
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        fbank = ((data.get("dataset_conf") or {}).get("fbank_conf") or {}) if isinstance(data, dict) else {}
        for key in conf:
            if isinstance(fbank.get(key), (int, float)):
                conf[key] = fbank[key]
    return conf


def kaldi_features(conf: dict[str, float]) -> Features:
    """Kaldi filterbanks as Rev computes them (16-bit scale waveform, no dither)."""
    torch = require("torch", "reverb")
    kaldi = require("torchaudio.compliance.kaldi", "reverb")

    def features(audio: np.ndarray) -> Any:
        wave = torch.from_numpy(np.asarray(audio, dtype=np.float32) * 32768.0).unsqueeze(0)
        feats = kaldi.fbank(
            wave, num_mel_bins=int(conf["num_mel_bins"]), frame_length=float(conf["frame_length"]),
            frame_shift=float(conf["frame_shift"]), dither=0.0, energy_floor=0.0, sample_frequency=SR,
        )  # fmt: skip
        return feats.unsqueeze(0)

    return features


class ReverbAsr(STTModel):
    """Reverb ASR as a bolo speech-to-text model."""

    sample_rate = SR
    supports_translate = False
    supports_word_timestamps = False
    languages = ("en",)

    def __init__(
        self, module: Any, units: dict[int, str], features: Features, *, verbatimicity: float = 1.0,
        max_chunk_s: float = 20.0, device: str = "cpu",
    ) -> None:  # fmt: skip
        if not 0.0 <= verbatimicity <= 1.0:
            raise InvalidRequest("verbatimicity is between 0.0 (clean) and 1.0 (verbatim)")
        self.module = module
        self.units = units
        self.features = features
        self.verbatimicity = verbatimicity
        self.max_chunk_s = max_chunk_s
        self.device = device
        self._lock = threading.Lock()

    def _decode(self, audio: np.ndarray) -> str:
        import torch

        feats = self.features(audio).to(self.device)
        cat = torch.tensor([self.verbatimicity, 1.0 - self.verbatimicity], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            enc, _att, _cnn = self.module.forward_encoder_chunk(
                feats,
                0,
                -1,
                torch.zeros(0, 0, 0, 0, device=self.device),
                torch.zeros(0, 0, 0, 0, device=self.device),
                cat,
            )
            log_probs = self.module.ctc_activation(enc)
        return pieces_to_text(ctc_greedy(np.asarray(log_probs[0].cpu().numpy())), self.units)

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        out: list[STTResult] = []
        for r in requests:
            if r.task == "translate":
                raise InvalidRequest("Reverb ASR transcribes English only; it does not translate")
            code = (r.language or "en").lower().split("-", 1)[0]
            if code != "en":
                raise InvalidRequest(f"Reverb ASR transcribes English only, not {r.language!r}")
            audio = np.ascontiguousarray(r.samples, dtype=np.float32)
            duration = len(audio) / SR
            if len(audio) < int(0.1 * SR):
                out.append(STTResult("", "en"))
                continue
            with self._lock:
                text = self._decode(audio)
            out.append(STTResult(text, "en", [Segment(0.0, duration, text)] if text else []))
        return out


def load(ctx: LoadContext) -> ReverbAsr:
    root = model_dir(ctx.path, ctx.options)
    jit = root / str(ctx.option("jit_file", JIT_FILE))
    if not jit.is_file():
        raise PluginError(f"{jit} is missing (pull the model again)")
    torch = require("torch", "reverb")
    module = torch.jit.load(str(jit), map_location=ctx.device)
    module.eval()
    return ReverbAsr(
        module,
        read_units(root / UNITS_FILE),
        kaldi_features(read_fbank_conf(root)),
        verbatimicity=float(ctx.option("verbatimicity", 1.0)),
        max_chunk_s=float(ctx.option("max_chunk_s", 20.0)),
        device=ctx.device,
    )
