#!/usr/bin/env python3
"""Export the SNAC 24 kHz decoder (hubertsiuzdak/snac_24khz, MIT) to the ONNX graph bolo's Orpheus route runs.

Orpheus-style text-to-speech models (Orpheus, Veena, Svara, snorTTS) write SNAC codes; bolo runs the language
model in llama.cpp and decodes the codes with this graph in onnxruntime (bolo/engines/orpheus_llamacpp.py).

    python tools/export_snac_onnx.py OUT_DIR [--snac-dir DIR] [--verify-tokens FILE.json ...] [--cross-check DIR]

Needs PyTorch (CPU is enough), the ``snac`` package (1.2.x), ``onnx``, ``onnxruntime`` and
``onnxconverter-common`` (fp16); none of this is part of bolo's own dependencies. ``--snac-dir`` is a snapshot of
``hubertsiuzdak/snac_24khz`` (``config.json``, ``pytorch_model.bin``); without it the pinned revision is fetched
from Hugging Face. Writes to ``OUT_DIR``:

    snac_24khz_decoder.onnx       fp32  codes_0 [B, T], codes_1 [B, 2T], codes_2 [B, 4T] (int64, 0..4095)
                                        -> audio [B, 1, 2048 T] (float32, 24 kHz, in [-1, 1])
    snac_24khz_decoder_fp16.onnx  fp16  the same graph with fp16 weights and arithmetic, float32 inputs/outputs
    snac_onnx.json                format, source revision, sha256 and size of every file, and the verification
    README.md

What the graph computes is ``SNAC.decode``: codebook lookup, output projection and repetition per level
(strides 4, 2, 1), then the decoder. Weight normalisation is folded into plain weights before export. The
decoder's three noise blocks add ``randn * conv(x)`` like the PyTorch module does (``RandomNormalLike`` in ONNX),
so two decodes of the same codes differ slightly, as they do in PyTorch.

Verification (written to ``snac_onnx.json`` and printed): the graph is exported a second time with the noise
blocks' random term set to zero and compared with PyTorch under the same change, which makes both sides
deterministic, on random codes and on every code sequence given with ``--verify-tokens`` (JSON files holding the
generated token ids of a real run, ``{"tokens": [...]}`` or a plain list; Orpheus token ids are mapped to codes
the way the engine maps them). It reports max absolute difference and SNR (dB) per sequence. With noise on, it
reports the SNR of each side against the noise-free PyTorch output, which shows that both add noise of the same
size. ``--cross-check`` compares with another export (a folder holding ``decoder_model.onnx`` and
``decoder_model_fp16.onnx``, e.g. a snapshot of ``onnx-community/snac_24khz-ONNX``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_REPO = "hubertsiuzdak/snac_24khz"
SOURCE_REVISION = "d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"
FORMAT = 1
OPSET = 17
FP32_NAME = "snac_24khz_decoder.onnx"
FP16_NAME = "snac_24khz_decoder_fp16.onnx"
INPUTS = ["codes_0", "codes_1", "codes_2"]
AUDIO_OFFSET = 128266
CODEBOOK = 4096
FRAME = 7
SAMPLES_PER_FRAME = 2048

README = """# SNAC 24 kHz decoder (ONNX)

The decoder of [SNAC 24 kHz](https://huggingface.co/{repo}) (revision `{revision}`, MIT licence, by Hubert
Siuzdak), exported to ONNX for torch-free decoding of the audio codes that Orpheus-style text-to-speech models
(Orpheus, Veena, Svara, snorTTS) generate. Exported with `tools/export_snac_onnx.py` from the bolo speech server.

| file | precision | size |
| --- | --- | --- |
{rows}

Inputs: `codes_0` `[B, T]`, `codes_1` `[B, 2T]`, `codes_2` `[B, 4T]` (int64, 0..4095). Output: `audio`
`[B, 1, 2048·T]`, float32 at 24 kHz. The noise blocks of the decoder are kept (random, as in PyTorch).

Checked against PyTorch SNAC with the noise term disabled on both sides: see `snac_onnx.json`. Use fp32 or fp16;
int8 quantisation of this decoder is known to fail (audible noise), so none is provided.

Licence: MIT, as the original SNAC weights (Copyright (c) 2024 Hubert Siuzdak).
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_snac(snac_dir: Path | None) -> Any:
    import torch
    from snac import SNAC

    if snac_dir is None:
        from huggingface_hub import snapshot_download

        snac_dir = Path(snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION))
    model = SNAC.from_config(str(snac_dir / "config.json"))
    state = torch.load(str(snac_dir / "pytorch_model.bin"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    from torch.nn.utils import parametrize

    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    return model


def set_noise(model: Any, enabled: bool) -> None:
    """Keep the decoder's noise blocks (``enabled``) or zero their random term (for deterministic checks)."""
    from snac.layers import NoiseBlock

    for module in model.modules():
        if isinstance(module, NoiseBlock):
            module.noise_enabled = enabled
            if not hasattr(NoiseBlock, "_bolo_patched"):
                original = NoiseBlock.forward

                def forward(self: Any, x: Any, _original: Any = original) -> Any:
                    if getattr(self, "noise_enabled", True):
                        return _original(self, x)
                    return x

                NoiseBlock.forward = forward
                NoiseBlock._bolo_patched = True


def make_wrapper(model: Any) -> Any:
    import torch

    class Decoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.snac = model

        def forward(self, codes_0: Any, codes_1: Any, codes_2: Any) -> Any:
            # SNAC.decode, with repeat_interleave written as expand + flatten: the tracer bakes the example's
            # length into repeat_interleave, which breaks other lengths
            z_q = 0.0
            for q, codes in zip(self.snac.quantizer.quantizers, (codes_0, codes_1, codes_2)):
                z = q.out_proj(q.decode_code(codes))
                if q.stride > 1:
                    z = z.unsqueeze(-1).expand(-1, -1, -1, q.stride).flatten(2)
                z_q = z_q + z
            return self.snac.decoder(z_q)

    return Decoder().eval()


def export(model: Any, path: Path) -> None:
    import torch

    wrapper = make_wrapper(model)
    t = 8
    example = (
        torch.randint(0, CODEBOOK, (1, t)),
        torch.randint(0, CODEBOOK, (1, 2 * t)),
        torch.randint(0, CODEBOOK, (1, 4 * t)),
    )
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example,
            str(path),
            input_names=INPUTS,
            output_names=["audio"],
            dynamic_axes={
                "codes_0": {0: "batch", 1: "t"},
                "codes_1": {0: "batch", 1: "t2"},
                "codes_2": {0: "batch", 1: "t4"},
                "audio": {0: "batch", 2: "samples"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
    import onnx

    graph = onnx.load(str(path))
    graph.doc_string = f"SNAC 24 kHz decoder from {SOURCE_REPO}@{SOURCE_REVISION} (MIT)"
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))


def to_fp16(src: Path, dst: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    # RandomNormalLike declares a float32 output type; keep it in float32 (the converter adds the casts)
    blocked = [*float16.DEFAULT_OP_BLOCK_LIST, "RandomNormalLike"]
    graph = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True, op_block_list=blocked)
    onnx.save(graph, str(dst))


def codes_from_tokens(tokens: list[int]) -> list[list[int]]:
    """Orpheus token ids -> the three SNAC levels (the engine's mapping, complete valid frames only)."""
    audio = [t for t in tokens if AUDIO_OFFSET <= t < AUDIO_OFFSET + FRAME * CODEBOOK]
    codes = [t - AUDIO_OFFSET - (i % FRAME) * CODEBOOK for i, t in enumerate(audio)]
    frames = [codes[i : i + FRAME] for i in range(0, len(codes) - len(codes) % FRAME, FRAME)]
    frames = [f for f in frames if all(0 <= c < CODEBOOK for c in f)]
    levels: list[list[int]] = [[], [], []]
    for f in frames:
        levels[0].append(f[0])
        levels[1].extend((f[1], f[4]))
        levels[2].extend((f[2], f[3], f[5], f[6]))
    return levels


def torch_decode(model: Any, levels: list[list[int]]) -> np.ndarray:
    import torch

    with torch.no_grad():
        out = model.decode([torch.tensor([lv], dtype=torch.long) for lv in levels])
    return np.asarray(out.reshape(-1).numpy(), dtype=np.float32)


def ort_decode(session: Any, levels: list[list[int]]) -> np.ndarray:
    names = [i.name for i in session.get_inputs()]
    feeds = {n: np.asarray([lv], dtype=np.int64) for n, lv in zip(names, levels)}
    return np.asarray(session.run(None, feeds)[0], dtype=np.float32).reshape(-1)


def compare(ref: np.ndarray, out: np.ndarray) -> dict[str, float]:
    n = min(len(ref), len(out))
    ref, out = ref[:n].astype(np.float64), out[:n].astype(np.float64)
    err = ref - out
    snr = 10 * np.log10(np.sum(ref**2) / max(np.sum(err**2), 1e-30))
    return {"max_abs": round(float(np.max(np.abs(err))), 8), "snr_db": round(float(snr), 2), "samples": n}


def sessions(paths: dict[str, Path]) -> dict[str, Any]:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    return {k: ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"]) for k, p in paths.items()}


def load_sequences(files: list[Path], seed: int) -> dict[str, list[list[int]]]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[list[int]]] = {}
    for t in (1, 4, 12):
        levels = [rng.integers(0, CODEBOOK, n * t).tolist() for n in (1, 2, 4)]
        out[f"random-{t}-frames"] = levels
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        tokens = doc["tokens"] if isinstance(doc, dict) else doc
        levels = codes_from_tokens([int(x) for x in tokens])
        if levels[0]:
            out[f.stem] = levels
    return out


def verify(model: Any, out_dir: Path, token_files: list[Path], cross: Path | None) -> dict[str, Any]:
    import torch

    seqs = load_sequences(token_files, seed=0)
    report: dict[str, Any] = {"deterministic": {}, "noise": {}, "cross_check": {}}
    with tempfile.TemporaryDirectory() as tmp:
        set_noise(model, False)
        quiet32 = Path(tmp) / "quiet.onnx"
        quiet16 = Path(tmp) / "quiet16.onnx"
        export(model, quiet32)
        to_fp16(quiet32, quiet16)
        quiet = sessions({"fp32": quiet32, "fp16": quiet16})
        shipped = sessions({"fp32": out_dir / FP32_NAME, "fp16": out_dir / FP16_NAME})
        other: dict[str, Any] = {}
        if cross is not None:
            other = sessions(
                {k: cross / name for k, name in (("fp32", "decoder_model.onnx"), ("fp16", "decoder_model_fp16.onnx"))
                 if (cross / name).is_file()}
            )  # fmt: skip
        for name, levels in seqs.items():
            set_noise(model, False)
            ref = torch_decode(model, levels)
            report["deterministic"][name] = {k: compare(ref, ort_decode(s, levels)) for k, s in quiet.items()}
            set_noise(model, True)
            torch.manual_seed(0)
            noisy = torch_decode(model, levels)
            report["noise"][name] = {
                "torch": compare(ref, noisy)["snr_db"],
                **{k: compare(ref, ort_decode(s, levels))["snr_db"] for k, s in shipped.items()},
            }
            if other:
                ours = ort_decode(shipped["fp32"], levels)
                report["cross_check"][name] = {
                    k: {"vs_torch_quiet_snr_db": compare(ref, ort_decode(s, levels))["snr_db"],
                        "vs_ours_snr_db": compare(ours, ort_decode(s, levels))["snr_db"]}
                    for k, s in other.items()
                }  # fmt: skip
    set_noise(model, True)
    return report


def versions() -> dict[str, str]:
    """Package versions of the export (the graph bytes, and so the SHA-256, depend on them)."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in ("torch", "snac", "onnx", "onnxconverter-common", "onnxruntime"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "?"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--snac-dir", type=Path, default=None)
    parser.add_argument("--verify-tokens", type=Path, nargs="*", default=[])
    parser.add_argument("--cross-check", type=Path, default=None)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    model = load_snac(args.snac_dir)
    set_noise(model, True)
    export(model, out / FP32_NAME)
    to_fp16(out / FP32_NAME, out / FP16_NAME)
    files = [out / FP32_NAME, out / FP16_NAME]
    report = None if args.no_verify else verify(model, out, list(args.verify_tokens), args.cross_check)
    doc = {
        "format": FORMAT,
        "source": {"repo": SOURCE_REPO, "revision": SOURCE_REVISION, "license": "MIT"},
        "opset": OPSET,
        "exported_with": versions(),
        "inputs": INPUTS,
        "output": "audio",
        "sample_rate": 24000,
        "samples_per_frame": SAMPLES_PER_FRAME,
        "files": {p.name: {"size": p.stat().st_size, "sha256": sha256(p)} for p in files},
        "verification": report,
    }
    (out / "snac_onnx.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    rows = "\n".join(
        f"| `{p.name}` | {'fp16' if 'fp16' in p.name else 'fp32'} | {p.stat().st_size / 1e6:.1f} MB |" for p in files
    )
    (out / "README.md").write_text(
        README.format(repo=SOURCE_REPO, revision=SOURCE_REVISION, rows=rows), encoding="utf-8"
    )
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
