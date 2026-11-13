#!/usr/bin/env python3
"""Export NVIDIA Nemotron-3-Diarization to the ONNX step graph bolo runs, from the official weights.

bolo's catalog downloads the int8 ONNX conversion published by the onnx-community organisation on
Hugging Face. This script makes the same graph ourselves from ``nvidia/Nemotron-3-Diarization``
(``model.safetensors``, OpenMDW-1.1), so the conversion can be checked or replaced:

    python tools/export_nemotron_onnx.py MODEL_DIR OUT_DIR

``MODEL_DIR`` holds the official ``config.json``, ``processor_config.json`` and ``model.safetensors``.
Needs PyTorch (CPU is enough), ``transformers>=5.23``, ``onnx`` and ``onnxruntime``; it is not part of
bolo's own dependencies. Writes ``OUT_DIR/onnx/model.onnx`` (fp32) and ``OUT_DIR/onnx/model_quantized.onnx``
(8-bit block-wise weight-only quantisation, ``MatMulNBits``, of the encoder layers; the input projection and
the speaker head stay fp32), plus the config files: a directory ``bolo pull ./OUT_DIR`` can register.

Measured on 24 AMI excerpts and two 4.5-minute recordings against transformers' PyTorch model (fp32), with
bolo's front-end and chunk loop (2026-11-13): this script's fp32 graph differs by 0.00002 in mean absolute
probability (99.998 % of frame-and-speaker decisions at 0.5 identical), its int8 graph by 0.0014 (99.9 %);
the onnx-community int8 conversion bolo's catalog uses by 0.0009 (99.95 %).

The graph is one encoder step of the chunked model (see ``bolo/engines/nemotron.py``):

    inputs   input_features [batch, mel_frames, 128]   log-mel frames of the chunk and its look-ahead
             cached_embeds  [batch, cached, 512]       speaker-cache + FIFO frame embeddings
             attention_mask [batch, cached + ceil(mel_frames / 8)]
    outputs  logits         [batch, (cached + ceil(mel_frames / 8)) * 8, 8]
             chunk_embeds   [batch, ceil(mel_frames / 8), 512]
             silence_embeds [512]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> None:
    import onnx
    import torch
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
    from transformers import AutoModelForAudioFrameClassification

    model_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    (out_dir / "onnx").mkdir(parents=True, exist_ok=True)
    model = AutoModelForAudioFrameClassification.from_pretrained(model_dir, attn_implementation="eager").eval()

    class Step(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.m = model

        def forward(
            self, input_features: torch.Tensor, cached_embeds: torch.Tensor, attention_mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            chunk = self.m.model.audio_tower.embedder(input_features)
            embeds = torch.cat([cached_embeds, chunk], dim=1)
            positions = torch.arange(embeds.shape[1], device=embeds.device)[None, :]
            hidden = self.m.model(
                inputs_embeds=embeds, attention_mask=attention_mask.bool(), position_ids=positions
            ).last_hidden_state
            return self.m.classifier(hidden), chunk, self.m.silence_embeds * 1.0

    features = torch.randn(1, 304, 128) - 10.0
    cached = torch.randn(1, 50, 512)
    mask = torch.ones(1, 50 + 38, dtype=torch.int64)
    fp32 = out_dir / "onnx" / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            Step(),
            (features, cached, mask),
            str(fp32),
            input_names=["input_features", "cached_embeds", "attention_mask"],
            output_names=["logits", "chunk_embeds", "silence_embeds"],
            dynamic_axes={
                "input_features": {0: "batch_size", 1: "num_mel_frames"},
                "cached_embeds": {0: "batch_size", 1: "num_cached_frames"},
                "attention_mask": {0: "batch_size", 1: "num_frames"},
                "logits": {0: "batch_size", 1: "num_logit_frames"},
                "chunk_embeds": {0: "batch_size", 1: "num_chunk_frames"},
            },
            opset_version=18,
            dynamo=False,
        )
    keep_fp32 = [
        "/embedder/projection/MatMul",
        "/model/proj/MatMul",
        "/classifier/dense/MatMul",
        "/classifier/out_proj/MatMul",
    ]
    quantizer = MatMulNBitsQuantizer(
        onnx.load(str(fp32)),
        algo_config=DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=8),
        nodes_to_exclude=keep_fp32,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(out_dir / "onnx" / "model_quantized.onnx"), use_external_data_format=True)
    for name in ("config.json", "processor_config.json"):
        if (model_dir / name).exists():
            shutil.copy(model_dir / name, out_dir / name)
    print(f"wrote {out_dir / 'onnx'}")


if __name__ == "__main__":
    main()
