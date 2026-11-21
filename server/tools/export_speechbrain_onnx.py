#!/usr/bin/env python3
"""Export SpeechBrain's SepFormer and MetricGAN+ checkpoints to the ONNX graphs bolo's core runs, and check them.

    python tools/export_speechbrain_onnx.py sepformer SNAPSHOT_DIR OUT_DIR [--repo R --revision REV]
    python tools/export_speechbrain_onnx.py metricgan SNAPSHOT_DIR OUT_DIR [--repo R --revision REV]
    python tools/export_speechbrain_onnx.py asr SNAPSHOT_DIR OUT_DIR [--repo R --revision REV] [--clips a.wav ...]

``SNAPSHOT_DIR`` holds a SpeechBrain model repository's ``hyperparams.yaml`` and checkpoints (for SepFormer
``encoder.ckpt``, ``masknet.ckpt``, ``decoder.ckpt``; for MetricGAN+ ``enhance_model.ckpt``; for a wav2vec 2.0 CTC
recogniser ``wav2vec2.ckpt``, ``asr.ckpt``, ``tokenizer.ckpt`` and the encoder's ``config.json`` and
``preprocessor_config.json``). Needs PyTorch (CPU is enough), ``speechbrain`` 1.1, ``onnx`` and ``onnxruntime``, and
bolo itself for the parity check (it runs the graph through bolo's own code, ``bolo.engines.speechbrain_onnx``); asr
also needs bolo-torch (``config_only_encoders``). None of this is part of bolo's own dependencies.

* **sepformer** writes ``sepformer.onnx``: ``mixture`` float32 ``[1, T]`` at the model's rate ->
  ``sources`` float32 ``[1, 8 * ((T - 16) // 8) + 16, S]``, the encoder, the masking network and the decoder of
  ``SepformerSeparation.separate_batch`` (``S`` = 1 for the enhancement models). The weights are unchanged.
* **metricgan** writes ``metricgan.onnx``: ``features`` float32 ``[1, frames, 257]`` (log1p of the STFT magnitude:
  512 points, hop 256, periodic Hamming window, centred, zero padded) -> ``mask`` float32 ``[1, frames, 257]``,
  the bidirectional LSTM generator. The STFT, the masking (in the log domain, as SpeechBrain does) and the inverse
  STFT with the noisy phase run in NumPy in bolo.

* **asr** writes ``asr.onnx``: ``wav`` float32 ``[1, T]`` at 16 kHz -> ``logprobs`` float32 ``[1, frames, tokens]``
  (20 ms frames), SpeechBrain ``EncoderASR``'s ``encoder`` (wav2vec 2.0, DNN, CTC layer, log-softmax), and
  ``tokens.txt`` (``piece id`` per line, the SentencePiece units). SpeechBrain's wav2vec 2.0 wrapper normalises the
  waveform over time and the encoder output over time and features (``layer_norm`` with the input's own shape); the
  graph does the same with means and variances, so that its length stays free. The wav2vec 2.0 network is built from
  its configuration and filled from ``wav2vec2.ckpt`` (nothing is downloaded from facebook/).

Every family writes ``parity.json``: the graph run through bolo's code against SpeechBrain's own inference class
(``separate_batch``; ``enhance_batch`` without its final peak normalisation, which bolo leaves out) on mixtures of
bolo's sample clips (``server/src/bolo/samples``), and ``speechbrain_onnx.json`` (format, source, sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "src" / "bolo" / "samples"
FORMAT = 1
#: --reuse: check a graph exported before instead of exporting it again
REUSE = False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clip(name: str, rate: int) -> np.ndarray:
    import soundfile as sf

    from bolo.audio.resample import resample

    x, sr = sf.read(SAMPLES / name, dtype="float32")
    return np.asarray(resample(x, sr, rate) if sr != rate else x, dtype=np.float32)


def si_snr(est: np.ndarray, ref: np.ndarray) -> float:
    est, ref = est - est.mean(), ref - ref.mean()
    s = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12) * ref
    e = est - s
    return float(10 * np.log10((np.dot(s, s) + 1e-12) / (np.dot(e, e) + 1e-12)))


def mixtures(rate: int, sources: int) -> dict[str, np.ndarray]:
    """Equal-power mixtures of bolo's three voices (all start together), and each voice alone."""
    voices = {"en-long": clip("en-long.wav", rate), "en-short": clip("en-short.wav", rate)}
    voices["hi-short"] = clip("hi-short.wav", rate)
    out = {}
    for names in itertools.combinations(voices, max(1, sources)):
        n = min(len(voices[v]) for v in names)
        parts = [voices[v][:n] / (np.sqrt(np.mean(voices[v][:n] ** 2)) + 1e-9) * 0.05 for v in names]
        out["+".join(names)] = np.sum(parts, axis=0).astype(np.float32)
    if sources <= 1:  # enhancement: the clips with white noise at 5 dB SNR
        rng = np.random.default_rng(7)
        for name, x in list(out.items()):
            noise = rng.standard_normal(len(x)).astype(np.float32)
            noise *= np.sqrt(np.mean(x**2) / (np.mean(noise**2) * 10 ** (5 / 10)))
            out[f"{name}+white@5dB"] = (x + noise).astype(np.float32)
    return out


def export_sepformer(src: Path, out: Path) -> dict[str, Any]:
    import onnxruntime as ort
    import torch
    from speechbrain.inference.separation import SepformerSeparation

    from bolo.engines.speechbrain_onnx import run_sepformer

    model = SepformerSeparation.from_hparams(source=str(src), savedir=str(src), run_opts={"device": "cpu"})
    model.eval()
    rate = int(model.hparams.sample_rate)
    speakers = int(model.hparams.num_spks)

    class Graph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder, self.masknet, self.decoder = model.mods.encoder, model.mods.masknet, model.mods.decoder

        def forward(self, mixture: torch.Tensor) -> torch.Tensor:
            w = self.encoder(mixture)
            masks = self.masknet(w)
            return torch.stack([self.decoder(w * masks[i]) for i in range(speakers)], dim=-1)

    graph = Graph().eval()
    path = out / "sepformer.onnx"
    if not (REUSE and path.exists()):
        torch.onnx.export(
            graph, (torch.randn(1, 3 * rate) * 0.05,), str(path), input_names=["mixture"], output_names=["sources"],
            dynamic_axes={"mixture": {1: "time"}, "sources": {1: "time_out"}}, opset_version=17, dynamo=False,
        )  # fmt: skip
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    checks: dict[str, Any] = {}
    # lengths that hit every branch of the chunking (the masking network pads to chunks of 250 frames)
    for n in (rate // 2 + 3, 3008, 3 * rate, 7 * rate + 5):
        x = (np.random.default_rng(n).standard_normal(n) * 0.05).astype(np.float32)
        with torch.inference_mode():
            ref = model.separate_batch(torch.from_numpy(x)[None])[0].numpy()
        got = run_sepformer(session, x)
        checks[f"noise, {n} samples"] = {
            "max_abs_diff": float(np.max(np.abs(ref - got))),
            "min_si_snr_db_onnx_vs_torch": round(min(si_snr(got[:, k], ref[:, k]) for k in range(speakers)), 1),
        }
    for name, x in mixtures(rate, speakers).items():
        with torch.inference_mode():
            ref = model.separate_batch(torch.from_numpy(x)[None])[0].numpy()
        got = run_sepformer(session, x)
        checks[name] = {
            "seconds": round(len(x) / rate, 2),
            "max_abs_diff": float(np.max(np.abs(ref - got))),
            "peak": float(np.max(np.abs(ref))),
            "min_si_snr_db_onnx_vs_torch": round(min(si_snr(got[:, k], ref[:, k]) for k in range(speakers)), 1),
        }
    return {"graph": path.name, "sample_rate": rate, "sources": speakers, "checks": checks}


def export_metricgan(src: Path, out: Path) -> dict[str, Any]:
    import onnxruntime as ort
    import torch
    from speechbrain.inference.enhancement import SpectralMaskEnhancement
    from speechbrain.processing.signal_processing import resynthesize

    from bolo.engines.speechbrain_onnx import metricgan_enhance, stft_features

    model = SpectralMaskEnhancement.from_hparams(source=str(src), savedir=str(src), run_opts={"device": "cpu"})
    model.eval()
    net = model.mods.enhance_model

    class Graph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = net

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.net(features, lengths=None)

    path = out / "metricgan.onnx"
    if not (REUSE and path.exists()):
        torch.onnx.export(
            Graph().eval(), (torch.rand(1, 120, 257),), str(path), input_names=["features"], output_names=["mask"],
            dynamic_axes={"features": {1: "frames"}, "mask": {1: "frames"}}, opset_version=17, dynamo=False,
        )  # fmt: skip
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    checks: dict[str, Any] = {}
    for name, x in mixtures(16000, 1).items():
        wav = torch.from_numpy(x)[None]
        with torch.inference_mode():
            feats = model.compute_features(wav)
            mask = net(feats, lengths=None)
            ref = resynthesize(
                torch.expm1(mask * feats), wav, model.hparams.compute_stft, model.hparams.compute_istft,
                normalize_wavs=False,
            )[0].numpy()  # fmt: skip
        ours = stft_features(x)
        got = metricgan_enhance(session, x)
        checks[name] = {
            "seconds": round(len(x) / 16000, 2),
            "features_max_abs_diff": float(np.max(np.abs(ours - feats[0].numpy()))),
            "max_abs_diff": float(np.max(np.abs(ref - got))),
            "peak": float(np.max(np.abs(ref))),
            "si_snr_db_onnx_vs_torch": round(si_snr(got, ref), 1),
        }
    return {"graph": path.name, "sample_rate": 16000, "sources": 1, "checks": checks}


def export_asr(src: Path, out: Path, clips: list[Path]) -> dict[str, Any]:
    import onnxruntime as ort
    import soundfile as sf
    import torch
    from bolo_torch.models.slu import config_only_encoders
    from speechbrain.inference.ASR import EncoderASR

    from bolo.engines.speechbrain_onnx import ctc_greedy, detokenize

    with config_only_encoders():
        model = EncoderASR.from_hparams(
            source=str(src), savedir=str(src), overrides={"wav2vec2_hub": str(src)}, run_opts={"device": "cpu"}
        )
    model.eval()
    enc = model.mods.encoder
    w2v = enc.wav2vec2

    class Graph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w2v, self.enc, self.ctc_lin, self.log_softmax = w2v.model, enc.enc, enc.ctc_lin, enc.log_softmax

        def forward(self, wav: torch.Tensor) -> torch.Tensor:
            if w2v.normalize_wav:  # F.layer_norm(wav, wav.shape[1:])
                wav = (wav - wav.mean(dim=1, keepdim=True)) / torch.sqrt(
                    wav.var(dim=1, keepdim=True, unbiased=False) + 1e-5
                )
            h = self.w2v(wav).last_hidden_state
            if w2v.output_norm:  # F.layer_norm(h, h.shape[1:])
                mean = h.mean(dim=(1, 2), keepdim=True)
                h = (h - mean) / torch.sqrt(((h - mean) ** 2).mean(dim=(1, 2), keepdim=True) + 1e-5)
            return self.log_softmax(self.ctc_lin(self.enc(h)))

    path = out / "asr.onnx"
    graph = Graph().eval()
    if not (REUSE and path.exists()):
        with torch.no_grad():
            torch.onnx.export(
                graph, (torch.randn(1, 3 * 16000) * 0.05,), str(path), input_names=["wav"], output_names=["logprobs"],
                dynamic_axes={"wav": {1: "time"}, "logprobs": {1: "frames"}}, opset_version=17, dynamo=False,
            )  # fmt: skip
    tok = model.tokenizer
    pieces = [tok.id_to_piece(i) for i in range(tok.vocab_size())]
    (out / "tokens.txt").write_text("".join(f"{p} {i}\n" for i, p in enumerate(pieces)), encoding="utf-8")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    checks: dict[str, Any] = {}
    audio = {c.name: c for c in clips}
    for name, file in audio.items():
        x, sr = sf.read(file, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != 16000:
            from bolo.audio.resample import resample

            x = resample(x, sr, 16000)
        wav = torch.from_numpy(np.ascontiguousarray(x))[None]
        with torch.no_grad():
            ref_text = model.transcribe_batch(wav, torch.tensor([1.0]))[0][0]
            ref = enc(wav, torch.tensor([1.0]))[0].numpy()
        got = np.asarray(session.run(None, {"wav": wav.numpy()})[0])[0]
        text = detokenize(ctc_greedy(got), pieces)
        checks[name] = {
            "seconds": round(len(x) / 16000, 2),
            "logprobs_max_abs_diff": float(np.max(np.abs(ref - got))),
            "same_best_path": bool(np.array_equal(ref.argmax(-1), got.argmax(-1))),
            "speechbrain": str(ref_text), "onnx": text, "same_text": str(ref_text) == text,
        }  # fmt: skip
    return {"graph": path.name, "sample_rate": 16000, "sources": 0, "checks": checks}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("family", choices=["sepformer", "metricgan", "asr"])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--repo", default="", help="the source repository, for speechbrain_onnx.json")
    ap.add_argument("--revision", default="", help="its revision")
    ap.add_argument("--clips", nargs="*", type=Path, default=[], help="audio files for the parity check (asr)")
    ap.add_argument("--reuse", action="store_true", help="check the graph already in OUT_DIR, do not export")
    args = ap.parse_args(argv)
    global REUSE
    REUSE = args.reuse
    args.out.mkdir(parents=True, exist_ok=True)
    if args.family == "asr":
        result = export_asr(args.src, args.out, args.clips)
    else:
        result = (export_sepformer if args.family == "sepformer" else export_metricgan)(args.src, args.out)
    import onnxruntime as ort
    import speechbrain
    import torch

    result["versions"] = {
        "torch": torch.__version__,
        "speechbrain": speechbrain.__version__,
        "onnxruntime": ort.__version__,
    }
    (args.out / "parity.json").write_text(json.dumps(result, indent=1) + "\n")
    meta = {
        "format": FORMAT, "family": args.family, "source": {"repo": args.repo, "revision": args.revision},
        "sample_rate": result["sample_rate"], "sources": result["sources"],
        "sources_sha256": {p.name: sha256(p) for p in sorted(args.src.glob("*.ckpt"))},
        "graph": {"file": result["graph"], "sha256": sha256(args.out / result["graph"])},
    }  # fmt: skip
    (args.out / "speechbrain_onnx.json").write_text(json.dumps(meta, indent=1) + "\n")
    json.dump(result["checks"], sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
