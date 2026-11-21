"""Export a CLAP model (contrastive language-audio pretraining) to two ONNX graphs, an audio tower and a text
tower, and check them against the PyTorch original.

    python export_clap_onnx.py laion  MODEL_DIR OUT clip.wav [clip.wav ...]   # a Hugging Face transformers CLAP
    python export_clap_onnx.py msclap WEIGHTS.pth OUT clip.wav [clip.wav ...]  # Microsoft CLAP 2023 (msclap)

Both graphs end in the projection and an L2 normalisation, so the dot product of an audio embedding and a text
embedding is their cosine similarity; ``clap.json`` carries the logit scale (the model's learned temperature) that
turns similarities into the scores a softmax over labels expects.

* ``audio.onnx``: ``audio`` float32 ``[batch, clip_samples]``, mono at ``sample_rate`` (LAION: 48 kHz, 10 s; MS-CLAP:
  44.1 kHz, 7 s), already repeated and padded to length (``padding`` in clap.json) -> ``embedding`` ``[batch, D]``.
  The log-mel front end is inside the graph (a strided convolution with the windowed DFT, the mel filter bank and
  the log), so nothing else is needed.
* ``text.onnx``: ``input_ids`` and ``attention_mask`` int64 ``[batch, tokens]`` (``tokenizer.json``) ->
  ``embedding`` ``[batch, D]``.
* ``audio.int8.onnx``: the audio tower with the weights of its matrix products in int8 (onnxruntime dynamic
  quantization; the front end stays in float).

The parity numbers go to ``OUT/parity.json``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr
import torch
from torch import nn

LABELS = [
    "a man speaking", "a woman speaking", "people talking", "music", "a piano", "a guitar", "drums",
    "white noise", "static", "a pure tone", "a beep", "silence", "a dog barking", "a siren", "rain",
]  # fmt: skip
PROMPT = "This is a sound of {}."


def synthetic(rate: int, seconds: float = 6.0, seed: int = 0) -> dict[str, np.ndarray]:
    """Clips that are not speech: a chord progression with a beat (music), white noise, and a 1 kHz tone."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    music = np.zeros_like(t)
    chords = [[261.63, 329.63, 392.0], [220.0, 261.63, 329.63], [174.61, 220.0, 261.63], [196.0, 246.94, 293.66]]
    seg = len(t) // len(chords)
    for i, chord in enumerate(chords):
        tt = t[i * seg : (i + 1) * seg]
        env = np.exp(-3 * (tt - tt[0]) % 0.5)
        for f in chord:
            for h in (1, 2, 3):
                music[i * seg : (i + 1) * seg] += env * np.sin(2 * np.pi * f * h * tt) / (h * 3)
    beat = (np.sin(2 * np.pi * 2 * t) > 0.95).astype(float) * rng.normal(0, 1, len(t))
    music = music / np.max(np.abs(music)) * 0.5 + 0.15 * beat
    noise = rng.normal(0, 0.2, len(t))
    tone = 0.4 * np.sin(2 * np.pi * 1000 * t)
    return {k: v.astype(np.float32) for k, v in {"music": music, "noise": noise, "tone": tone}.items()}


def repeatpad(x: np.ndarray, n: int) -> np.ndarray:
    """transformers' ClapFeatureExtractor ``padding="repeatpad"``: the clip repeated whole, then zeros to length."""
    if len(x) >= n:
        return x[:n]
    x = np.tile(x, n // len(x))
    return np.pad(x, (0, n - len(x)))


def dft_kernels(n_fft: int, window: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    k = np.arange(n_fft // 2 + 1)[:, None]
    n = np.arange(n_fft)[None, :]
    ang = 2 * np.pi * k * n / n_fft
    real = (np.cos(ang) * window[None, :]).astype(np.float32)
    imag = (-np.sin(ang) * window[None, :]).astype(np.float32)
    return torch.from_numpy(real[:, None, :]), torch.from_numpy(imag[:, None, :])


class LogMel(nn.Module):
    """transformers' ClapFeatureExtractor in torch: power spectrogram by strided convolution (centred, reflect
    padding), the mel filter bank, then 10 log10(max(x, 1e-10))."""

    def __init__(self, n_fft: int, hop: int, window: np.ndarray, mel: np.ndarray, log: str, top_db: float | None):
        super().__init__()
        real, imag = dft_kernels(n_fft, window)
        self.register_buffer("real", real)
        self.register_buffer("imag", imag)
        self.register_buffer("mel", torch.from_numpy(mel.astype(np.float32)))  # [bins, mels]
        self.n_fft, self.hop, self.log, self.top_db = n_fft, hop, log, top_db

    def forward(self, audio: torch.Tensor) -> torch.Tensor:  # [B, T] -> [B, frames, mels]
        x = nn.functional.pad(audio[:, None, :], (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        re = nn.functional.conv1d(x, self.real, stride=self.hop)
        im = nn.functional.conv1d(x, self.imag, stride=self.hop)
        power = (re * re + im * im).transpose(1, 2)  # [B, frames, bins]
        mel = torch.matmul(power, self.mel)
        db = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        if self.top_db is not None:
            db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - self.top_db)
        return db


# -- LAION CLAP (transformers) ---------------------------------------------------------------------------------
class LaionAudio(nn.Module):
    def __init__(self, model: Any, fe: Any) -> None:
        super().__init__()
        from transformers.audio_utils import window_function

        window = window_function(fe.fft_window_size, "hann").astype(np.float64)
        self.front = LogMel(fe.fft_window_size, fe.hop_length, window, fe.mel_filters_slaney, "db", None)
        self.audio_model = model.audio_model
        self.audio_projection = model.audio_projection

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self.front(audio)[:, None, :, :]
        out = self.audio_model(input_features=mel, return_dict=True)
        emb = self.audio_projection(out.pooler_output)
        return nn.functional.normalize(emb, dim=-1)


class LaionText(nn.Module):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.text_model = model.text_model
        self.text_projection = model.text_projection

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return nn.functional.normalize(self.text_projection(out.pooler_output), dim=-1)


def laion(model_dir: Path) -> dict[str, Any]:
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(str(model_dir)).eval()
    proc = ClapProcessor.from_pretrained(str(model_dir))
    fe = proc.feature_extractor
    rate, n = fe.sampling_rate, fe.nb_max_samples

    def ref_audio(x: np.ndarray) -> np.ndarray:
        inputs = fe(x, sampling_rate=rate, return_tensors="pt")
        with torch.no_grad():
            return model.get_audio_features(**inputs).numpy()

    def ref_text(texts: list[str]) -> np.ndarray:
        tok = proc.tokenizer(texts, padding=True, return_tensors="pt")
        with torch.no_grad():
            return model.get_text_features(**tok).numpy()

    return {
        "audio": LaionAudio(model, fe).eval(),
        "text": LaionText(model).eval(),
        "rate": rate,
        "samples": n,
        "logit_scale": float(model.logit_scale_a.exp()),
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "tokenizer_file": model_dir / "tokenizer.json",
        "config": {"family": "laion-clap", "padding": "repeatpad", "max_tokens": 77, "append_eos": False},
    }


# -- MS-CLAP 2023 (microsoft/CLAP, msclap) ----------------------------------------------------------------------
class MsAudio(nn.Module):
    """msclap's audio encoder whole: torchlibrosa's log-mel front end (convolutions), HTSAT and the projection."""

    def __init__(self, clap: Any) -> None:
        super().__init__()
        self.encoder = clap.audio_encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.encoder(audio)[0], dim=-1)


class MsText(nn.Module):
    def __init__(self, clap: Any) -> None:
        super().__init__()
        self.caption = clap.caption_encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.caption({"input_ids": input_ids, "attention_mask": attention_mask})
        return nn.functional.normalize(emb, dim=-1)


def msclap(weights: Path) -> dict[str, Any]:
    from msclap import CLAP

    wrapper = CLAP(model_fp=str(weights), version="2023", use_cuda=False)
    clap = wrapper.clap.eval()
    args = wrapper.args
    rate, n = int(args.sampling_rate), int(args.duration * args.sampling_rate)

    def ref_audio(x: np.ndarray) -> np.ndarray:
        audio = torch.from_numpy(repeatpad_msclap(x, n))[None]
        with torch.no_grad():
            emb = clap.audio_encoder(audio)[0]
        return nn.functional.normalize(emb, dim=-1).numpy()

    def ref_text(texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            tok = wrapper.preprocess_text(texts)
            return nn.functional.normalize(clap.caption_encoder(tok), dim=-1).numpy()

    return {
        "audio": MsAudio(clap).eval(),
        "text": MsText(clap).eval(),
        "rate": rate,
        "samples": n,
        "logit_scale": float(clap.logit_scale.exp()),
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "wrapper": wrapper,
        "config": {"family": "ms-clap", "padding": "repeat", "max_tokens": 77, "append_eos": True},
    }


def repeatpad_msclap(x: np.ndarray, n: int) -> np.ndarray:
    """msclap's ``load_audio_into_tensor`` for a clip shorter than the duration: repeated whole, then cut."""
    if len(x) >= n:
        return x[:n].astype(np.float32)
    reps = math.ceil(n / len(x))
    return np.tile(x, reps)[:n].astype(np.float32)


# -- export and parity -----------------------------------------------------------------------------------------
def export(m: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, m["samples"])
    with torch.no_grad():
        torch.onnx.export(
            m["audio"], (x,), str(out / "audio.onnx"), input_names=["audio"], output_names=["embedding"],
            dynamic_axes={"audio": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17, do_constant_folding=True,
        )  # fmt: skip
        ids = torch.ones(2, 8, dtype=torch.long)
        mask = torch.ones(2, 8, dtype=torch.long)
        torch.onnx.export(
            m["text"], (ids, mask), str(out / "text.onnx"), input_names=["input_ids", "attention_mask"],
            output_names=["embedding"], opset_version=17, do_constant_folding=True,
            dynamic_axes={"input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"},
                          "embedding": {0: "batch"}},
        )  # fmt: skip
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # the audio tower with int8 weights: matrix products only, and not the front end (the DFT convolutions and the
    # mel filter bank stay in float). The text tower is not quantized: RoBERTa and GPT-2 lose too much in int8
    # (cosine to the original about 0.96), and label embeddings are computed once and cached anyway
    graph = onnx_load(out / "audio.onnx")
    front = [
        n.name
        for n in graph.graph.node
        if any(k in n.name for k in ("/front/", "spectrogram_extractor", "logmel_extractor"))
    ]
    quantize_dynamic(
        str(out / "audio.onnx"), str(out / "audio.int8.onnx"), weight_type=QuantType.QInt8, per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=front,
    )  # fmt: skip


def onnx_load(path: Path) -> Any:
    import onnx

    return onnx.load(str(path), load_external_data=False)


def tokenize(m: dict[str, Any], out: Path, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    cfg = m["config"]
    rows = []
    for t in texts:
        enc = tok.encode(t + (" <|endoftext|>" if cfg["append_eos"] else ""))
        rows.append(enc.ids[: cfg["max_tokens"]])
    width = max(len(r) for r in rows)
    pad = int(cfg.get("pad_id", 1))
    ids = np.full((len(rows), width), pad, dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.int64)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = r
        mask[i, : len(r)] = 1
    return ids, mask


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.min(np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))))


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def parity(m: dict[str, Any], out: Path, clips: dict[str, np.ndarray]) -> dict[str, Any]:
    import onnxruntime as ort

    texts = [PROMPT.format(label) for label in LABELS]
    pad = repeatpad if m["config"]["padding"] == "repeatpad" else repeatpad_msclap
    ref_t = m["ref_text"](texts)
    report: dict[str, Any] = {"labels": LABELS, "prompt": PROMPT, "clips": {}, "graphs": {}}
    ref_a = {name: m["ref_audio"](x) for name, x in clips.items()}
    for suffix in ("", ".int8"):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        audio = ort.InferenceSession(str(out / f"audio{suffix}.onnx"), so, providers=["CPUExecutionProvider"])
        text = ort.InferenceSession(str(out / "text.onnx"), so, providers=["CPUExecutionProvider"])
        ids, mask = tokenize(m, out, texts)
        onnx_t = text.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        g: dict[str, Any] = {
            "text_max_abs": float(np.abs(onnx_t - ref_t).max()),
            "text_min_cosine": cos(onnx_t, ref_t),
            "audio_max_abs": 0.0,
            "audio_min_cosine": 1.0,
            "prob_max_abs": 0.0,
            "top1_agree": 0,
            "clips": 0,
        }
        for name, x in clips.items():
            a = audio.run(None, {"audio": pad(x, m["samples"])[None].astype(np.float32)})[0]
            ra = ref_a[name]
            g["audio_max_abs"] = max(g["audio_max_abs"], float(np.abs(a - ra).max()))
            g["audio_min_cosine"] = min(g["audio_min_cosine"], cos(a, ra))
            p_ref = softmax(m["logit_scale"] * ra @ ref_t.T)[0]
            p_onnx = softmax(m["logit_scale"] * a @ onnx_t.T)[0]
            g["prob_max_abs"] = max(g["prob_max_abs"], float(np.abs(p_ref - p_onnx).max()))
            g["top1_agree"] += int(np.argmax(p_ref) == np.argmax(p_onnx))
            g["clips"] += 1
            if suffix == "":
                top = np.argsort(-p_ref)[:3]
                report["clips"][name] = {
                    "pytorch_top3": [[LABELS[i], round(float(p_ref[i]), 4)] for i in top],
                    "onnx_top3": [[LABELS[i], round(float(p_onnx[i]), 4)] for i in np.argsort(-p_onnx)[:3]],
                }
        report["graphs"]["fp32" if suffix == "" else "int8_audio"] = g
    return report


def main() -> int:
    family, src, out_s, *wavs = sys.argv[1:]
    out = Path(out_s)
    m = laion(Path(src)) if family == "laion" else msclap(Path(src))
    out.mkdir(parents=True, exist_ok=True)
    if family == "laion":
        (out / "tokenizer.json").write_bytes(Path(m["tokenizer_file"]).read_bytes())
    else:
        m["wrapper"].tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
        m["config"]["pad_id"] = int(m["wrapper"].tokenizer.pad_token_id)
    if "pad_id" not in m["config"]:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(out / "tokenizer.json"))
        m["config"]["pad_id"] = int(tok.token_to_id("<pad>") or 1)
    export(m, out)
    clips: dict[str, np.ndarray] = {}
    for w in wavs:
        x, sr = sf.read(w, dtype="float32", always_2d=True)
        clips[Path(w).stem] = soxr.resample(x.mean(1), sr, m["rate"]).astype(np.float32)[: m["samples"]]
    clips.update(synthetic(m["rate"]))
    config = {
        **m["config"],
        "sample_rate": m["rate"],
        "clip_samples": m["samples"],
        "logit_scale": m["logit_scale"],
        "prompt": PROMPT,
        "embedding_dim": int(m["ref_text"](["x"]).shape[-1]),
    }
    (out / "clap.json").write_text(json.dumps(config, indent=2) + "\n")
    report = parity(m, out, clips)
    report["torch"] = torch.__version__
    (out / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["graphs"], indent=2))
    for name, c in report["clips"].items():
        print(name, c["pytorch_top3"], "| onnx", c["onnx_top3"][0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
