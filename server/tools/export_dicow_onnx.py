#!/usr/bin/env python3
"""Export BUT-FIT DiCoW v3.3 (diarization-conditioned Whisper) to the ONNX graphs bolo's core runs.

DiCoW is Whisper large-v3-turbo with FDDT layers in the encoder: before the positional embeddings and before
every encoder layer, each 20 ms frame is transformed by four learned diagonal affine maps weighted by the
frame's STNO mask (silence / target speaker / non-target / overlap). The decoder is stock Whisper. The model
also has a CTC head on the encoder (``lm_head``, an extra self-attention layer and two subsampling
convolutions), used for training and for joint CTC/attention decoding when ``generation_config.ctc_weight``
is above zero; the published generation config sets it to 0, and bolo's PyTorch plugin decodes greedily
without it, so the CTC head is not exported. Decoding (bolo/engines/dicow_onnx.py) is transformers'
Whisper long-form generation as the model's own ``generation.py`` modifies it.

    python tools/export_dicow_onnx.py MODEL_DIR OUT_DIR [--int8-only] [--drop-fp32]

``MODEL_DIR`` is a snapshot of ``BUT-FIT/DiCoW_v3_3`` (revision c34b64d9a9c5148c65fd355bb188d60343a6b44f:
``model.safetensors``, its Python files, config, generation config and tokenizer files). Needs PyTorch (CPU is
enough), ``transformers==4.55.*`` (the model's code needs it), ``onnx`` and ``onnxruntime``; none of this is
part of bolo's own dependencies. Writes to ``OUT_DIR``:

    encoder.onnx (+ .data)       fp32   input_features [batch, 128, 3000], stno_mask [batch, 4, 1500]
                                        -> cross_k, cross_v [4 layers, batch, 20 heads, 1500, 64]
    decoder.onnx (+ .data)       fp32   input_ids [batch, tokens], self_k, self_v [4, batch, 20, past, 64],
                                        cross_k, cross_v -> logits [batch, 51866] (last position only),
                                        new_self_k, new_self_v [4, batch, 20, past + tokens, 64]
    encoder.int8.onnx (+ .data)  int8   dynamic quantisation (MatMul weights per channel, int8), except the
                                        feed-forward output matrices (fc2), kept in fp32 (see KEEP_FP32)
    decoder.int8.onnx (+ .data)  int8   the same, the token embedding as an int8 Gather
    config.json, generation_config.json, preprocessor_config.json, vocab.json, added_tokens.json,
    tokenizer_config.json, dicow_onnx.json (format, source revision, sha256 of every file), README.md

The encoder graph ends with the decoder's cross-attention key and value projections, so the decoder never
sees the 1280-dimensional encoder output. The decoder is one graph for the first step (an empty past) and
every following step. ``bolo pull ./OUT_DIR`` registers the directory (kind ``dicow_onnx``).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

SOURCE_REPO = "BUT-FIT/DiCoW_v3_3"
SOURCE_REVISION = "c34b64d9a9c5148c65fd355bb188d60343a6b44f"
FORMAT = 1
COPY = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "vocab.json",
    "added_tokens.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
)

README = """---
license: cc-by-4.0
base_model: {repo}
tags: [onnx, whisper, speaker-diarization, target-speaker-asr, dicow]
---

# DiCoW v3.3, ONNX

ONNX export of [{repo}](https://huggingface.co/{repo}) (revision `{revision}`) for
[bolo](https://github.com/cb-Sigma/bolo), which runs it with ONNX Runtime and NumPy, without PyTorch.
DiCoW (Diarization-Conditioned Whisper) is Whisper large-v3-turbo conditioned on per-frame speaker activity
(silence / target / non-target / overlap masks), so a recording is transcribed once per diarized speaker,
overlapping speech included. The model is by the Speech@FIT group of Brno University of Technology and is
licensed CC-BY-4.0; this conversion keeps that licence. Changes: the weights are exported unchanged to two
graphs (encoder with FDDT conditioning and the decoder's cross-attention projections; decoder with a key/value
cache); the `*.int8.onnx` graphs are dynamically quantised (int8 weights, per output channel; the encoder's
feed-forward output matrices stay fp32). The CTC head is not included (the published generation settings do not
use it). On 24 AMI excerpts the int8 graphs scored the same cpWER as the PyTorch model (26.0 %); details in bolo's
docs/models.md, "DiCoW without PyTorch".

Exported with `server/tools/export_dicow_onnx.py` from the bolo repository; checksums in `dicow_onnx.json`.

If you use the model, please cite the papers its model card lists:

- A. Polok et al., "DiCoW: Diarization-conditioned Whisper for target speaker automatic speech recognition",
  Computer Speech & Language 95, 2026, doi:10.1016/j.csl.2025.101841.
- A. Polok et al., "Target Speaker ASR with Whisper", ICASSP 2025, doi:10.1109/ICASSP49660.2025.10887683.
- A. Polok, D. Klement, S. Cornell, M. Wiesner, J. Černocký, S. Khudanpur, L. Burget, "SE-DiCoW: Self-Enrolled
  Diarization-Conditioned Whisper", arXiv:2601.19194, 2026.
"""


def import_model_code(root: Path) -> Any:
    """The repository's ``modeling_dicow`` imported as a package (what ``trust_remote_code`` would run)."""
    name = (
        "dicow_export_" + hashlib.sha256(b"".join(f.read_bytes() for f in sorted(root.glob("*.py")))).hexdigest()[:12]
    )
    if name not in sys.modules:
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        package = importlib.util.module_from_spec(spec)
        package.__path__ = [str(root)]
        sys.modules[name] = package
    return importlib.import_module(f"{name}.modeling_dicow")


def build_wrappers(model: Any) -> tuple[Any, Any]:
    import torch
    from torch import nn

    dec = model.model.decoder
    heads = dec.layers[0].self_attn.num_heads
    head_dim = dec.layers[0].self_attn.head_dim

    def split(x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, heads, head_dim).transpose(1, 2)

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = model.model.encoder
            self.layers = dec.layers

        def forward(self, input_features: torch.Tensor, stno_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.encoder(input_features, stno_mask=stno_mask, return_dict=False)[0]
            keys = [split(layer.encoder_attn.k_proj(hidden)) for layer in self.layers]
            values = [split(layer.encoder_attn.v_proj(hidden)) for layer in self.layers]
            return torch.stack(keys), torch.stack(values)

    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dec = dec
            self.register_buffer("zero_bias", torch.zeros(dec.embed_tokens.weight.shape[0]))

        def forward(
            self,
            input_ids: torch.Tensor,
            self_k: torch.Tensor,
            self_v: torch.Tensor,
            cross_k: torch.Tensor,
            cross_v: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            past = self_k.shape[3]
            steps = input_ids.shape[1]
            positions = torch.arange(steps) + past
            x = self.dec.embed_tokens(input_ids) + self.dec.embed_positions.weight[positions]
            keys = torch.arange(steps + past)
            causal = (keys[None, :] > positions[:, None]).to(x.dtype) * -1e9
            new_k, new_v = [], []
            for i, layer in enumerate(self.dec.layers):
                attn = layer.self_attn
                h = layer.self_attn_layer_norm(x)
                q = split(attn.q_proj(h)) * attn.scaling
                k = torch.cat([self_k[i], split(attn.k_proj(h))], dim=2)
                v = torch.cat([self_v[i], split(attn.v_proj(h))], dim=2)
                new_k.append(k)
                new_v.append(v)
                w = torch.softmax(q @ k.transpose(-1, -2) + causal, dim=-1)
                x = x + attn.out_proj((w @ v).transpose(1, 2).flatten(2))
                cross = layer.encoder_attn
                h = layer.encoder_attn_layer_norm(x)
                q = split(cross.q_proj(h)) * cross.scaling
                w = torch.softmax(q @ cross_k[i].transpose(-1, -2), dim=-1)
                x = x + cross.out_proj((w @ cross_v[i]).transpose(1, 2).flatten(2))
                h = layer.final_layer_norm(x)
                x = x + layer.fc2(nn.functional.gelu(layer.fc1(h)))
            last = self.dec.layer_norm(x[:, -1])
            # proj_out is tied to the token embedding; with a (zero) bias the exporter writes Gemm(transB=1), which
            # shares the embedding's initializer instead of folding a transposed copy into the graph
            logits = nn.functional.linear(last, self.dec.embed_tokens.weight, self.zero_bias)
            return logits, torch.stack(new_k), torch.stack(new_v)

    return Encoder().eval(), Decoder().eval()


def check_against_model(model: Any, encoder: Any, decoder: Any) -> dict[str, float]:
    """The wrappers against the model's own forward pass (random features, a random mask, 6 tokens)."""
    import torch

    torch.manual_seed(0)
    feats = torch.randn(2, 128, 3000) * 0.3
    stno = torch.softmax(torch.randn(2, 4, 1500) * 3, dim=1)
    ids = torch.tensor([[50258, 50259, 50360, 50365, 400, 1000], [50258, 50259, 50360, 50365, 291, 50]])
    with torch.inference_mode():
        ref = model(input_features=feats, stno_mask=stno, decoder_input_ids=ids, use_cache=False).logits
        ck, cv = encoder(feats, stno)
        layers, b, h, _, d = ck.shape
        empty = torch.zeros(layers, b, h, 0, d)
        full, _, _ = decoder(ids, empty, empty, ck, cv)
        _, sk, sv = decoder(ids[:, :4], empty, empty, ck, cv)
        step, _, _ = decoder(ids[:, 4:], sk, sv, ck, cv)
    return {
        "logits_full": float((full - ref[:, -1]).abs().max()),
        "logits_cached": float((step - ref[:, -1]).abs().max()),
        "logits_scale": float(ref[:, -1].abs().max()),
    }


def export(model_dir: Path, out: Path) -> None:
    import torch
    from transformers import WhisperTokenizer  # noqa: F401  (fails early without transformers)

    code = import_model_code(model_dir)
    model = code.DiCoWForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    if float(getattr(model.generation_config, "ctc_weight", 0) or 0) > 0:
        raise SystemExit("this export leaves out the CTC head, but the generation config decodes with it")
    if getattr(model.config, "use_enrollments", False):
        raise SystemExit("SE-DiCoW (use_enrollments) needs an enrolment input; only DiCoW v3.x is supported here")
    encoder, decoder = build_wrappers(model)
    print("wrappers vs model:", check_against_model(model, encoder, decoder), flush=True)
    for name in ("tmp-encoder", "tmp-decoder"):
        (out / name).mkdir(parents=True, exist_ok=True)
    feats = torch.zeros(1, 128, 3000)
    stno = torch.zeros(1, 4, 1500)
    stno[:, 1] = 1
    with torch.inference_mode():
        torch.onnx.export(
            encoder,
            (feats, stno),
            str(out / "tmp-encoder" / "encoder.onnx"),
            input_names=["input_features", "stno_mask"],
            output_names=["cross_k", "cross_v"],
            dynamic_axes={"input_features": {0: "batch"}, "stno_mask": {0: "batch"}, "cross_k": {1: "batch"},
                          "cross_v": {1: "batch"}},
            opset_version=17,
            dynamo=False,
        )  # fmt: skip
        print("encoder exported", flush=True)
        ck, cv = encoder(torch.zeros(2, 128, 3000), torch.zeros(2, 4, 1500))
        ids = torch.tensor([[50258, 50259, 50360], [50258, 50259, 50360]])
        sk = torch.zeros(4, 2, 20, 2, 64)
        torch.onnx.export(
            decoder,
            (ids, sk, sk.clone(), ck, cv),
            str(out / "tmp-decoder" / "decoder.onnx"),
            input_names=["input_ids", "self_k", "self_v", "cross_k", "cross_v"],
            output_names=["logits", "new_self_k", "new_self_v"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "tokens"},
                "self_k": {1: "batch", 3: "past"},
                "self_v": {1: "batch", 3: "past"},
                "cross_k": {1: "batch"},
                "cross_v": {1: "batch"},
                "logits": {0: "batch"},
                "new_self_k": {1: "batch", 3: "total"},
                "new_self_v": {1: "batch", 3: "total"},
            },
            opset_version=17,
            dynamo=False,
        )
        print("decoder exported", flush=True)
    for name in ("encoder", "decoder"):
        consolidate(out / f"tmp-{name}" / f"{name}.onnx", out / f"{name}.onnx")


def consolidate(src: Path, dst: Path) -> None:
    """Save a graph with all its weights in one ``<name>.data`` file next to it (``src`` may equal ``dst``).

    The weights are read into memory first and ``src``'s directory is emptied before writing (a temporary
    directory of the exporter is removed), so a large graph never exists twice on disk.
    """
    import onnx

    model = onnx.load(str(src))
    if src.parent != dst.parent:
        shutil.rmtree(src.parent)
    else:
        for stale in (src, src.with_name(src.name + ".data"), dst.with_name(dst.name + ".data")):
            stale.unlink(missing_ok=True)
    path = dst
    onnx.save_model(
        model, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location=path.name + ".data",
        size_threshold=1024,
    )  # fmt: skip


def untie_logits(src: Path, dst: Path) -> None:
    """The decoder's ``Gemm(last, embedding, transB=1)`` as ``MatMul(last, embedding^T)`` for the quantiser."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(src))
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    for node in list(graph.node):
        if node.op_type != "Gemm":
            continue
        trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
        weight = numpy_helper.to_array(inits[node.input[1]])
        w = np.ascontiguousarray(weight.T if trans_b else weight)
        name = node.input[1] + "_t"
        graph.initializer.append(numpy_helper.from_array(w, name))
        matmul = helper.make_node("MatMul", [node.input[0], name], list(node.output), name=node.name + "_matmul")
        index = list(graph.node).index(node)
        graph.node.remove(node)
        graph.node.insert(index, matmul)
    onnx.save_model(model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=dst.name + ".data", size_threshold=1024)  # fmt: skip


#: encoder MatMuls left in fp32: the second feed-forward matrix of every layer, whose input (after GELU) has outlier
#: channels that per-tensor activation quantisation handles badly. Quantising them too cost 2-3 cpWER points on
#: bolo's synthetic mixes (docs/models.md, "DiCoW without PyTorch").
KEEP_FP32 = r"/fc2/MatMul"


def quantize(out: Path, keep_fp32: str = KEEP_FP32) -> None:
    import re

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tmp = out / "decoder.untied.onnx"
    untie_logits(out / "decoder.onnx", tmp)
    jobs = [("encoder.onnx", "encoder.int8.onnx", ["MatMul"]), (tmp.name, "decoder.int8.onnx", ["MatMul", "Gather"])]
    for src, dst, ops in jobs:
        exclude: list[str] = []
        if src == "encoder.onnx" and keep_fp32:
            graph = onnx.load(str(out / src), load_external_data=False).graph
            exclude = [n.name for n in graph.node if n.op_type == "MatMul" and re.search(keep_fp32, n.name)]
        quantize_dynamic(
            str(out / src),
            str(out / dst),
            op_types_to_quantize=ops,
            per_channel=True,
            weight_type=QuantType.QInt8,
            nodes_to_exclude=exclude,
            use_external_data_format=True,
            extra_options={"EnableSubgraph": False},
        )
        consolidate(out / dst, out / dst)
        print("quantised", dst, f"({len(exclude)} MatMuls kept in fp32)", flush=True)
    tmp.unlink()
    (out / (tmp.name + ".data")).unlink(missing_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def finish(model_dir: Path, out: Path) -> None:
    for name in COPY:
        if (model_dir / name).exists():
            shutil.copy2(model_dir / name, out / name)
    (out / "README.md").write_text(README.format(repo=SOURCE_REPO, revision=SOURCE_REVISION), encoding="utf-8")
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "dicow_onnx.json")
    manifest = {
        "format": FORMAT,
        "model": "dicow",
        "source": {"repo": SOURCE_REPO, "revision": SOURCE_REVISION},
        "license": "CC-BY-4.0",
        "files": {p.name: {"size": p.stat().st_size, "sha256": sha256(p)} for p in files},
    }
    (out / "dicow_onnx.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--int8-only", action="store_true", help="quantise an existing fp32 export in OUT_DIR")
    parser.add_argument("--fp32-only", action="store_true", help="export, do not quantise")
    parser.add_argument("--drop-fp32", action="store_true", help="delete the fp32 graphs once quantised")
    parser.add_argument(
        "--keep-fp32", default=KEEP_FP32, help="regular expression: encoder MatMul nodes to leave in fp32 ('' for none)"
    )
    args = parser.parse_args()
    if not args.int8_only:
        export(args.model_dir, args.out_dir)
    if not args.fp32_only:
        quantize(args.out_dir, args.keep_fp32)
        if args.drop_fp32:
            for name in ("encoder.onnx", "encoder.onnx.data", "decoder.onnx", "decoder.onnx.data"):
                (args.out_dir / name).unlink(missing_ok=True)
    finish(args.model_dir, args.out_dir)
    print(json.dumps(json.loads((args.out_dir / "dicow_onnx.json").read_text(encoding="utf-8"))["files"], indent=1))


if __name__ == "__main__":
    main()
