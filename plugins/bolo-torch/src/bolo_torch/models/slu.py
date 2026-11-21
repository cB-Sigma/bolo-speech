"""Spoken intent (bolo task ``intent``): SpeechBrain's direct spoken-language-understanding models.

``speechbrain/SLU-direct-SLURP-hubert-enc`` (Apache-2.0) is a HuBERT base encoder fine-tuned on SLURP (an English
corpus of commands to a home assistant, CC BY 4.0) and an attentional LSTM decoder that writes the meaning of the
utterance as a Python dict literal, token by token (beam search, 58 SentencePiece units):
``{'scenario': 'alarm', 'action': 'set', 'entities': [{'type': 'time', 'filler': 'seven am'}]}``. bolo reads that
text (``bolo.intent.parse_semantics``, without evaluating it).

The model repository ships its inference interface as Python code (``custom_interface.py``); the plugin does not run
it. It builds the model from ``hyperparams.yaml`` with SpeechBrain's own classes and decodes as that file does
(``hubert`` encoder, ``beam_searcher``, ``tokenizer.decode_ids``). SpeechBrain's HuBERT wrapper would download
facebook/hubert-base-ls960's weights (360 MB) only for the repository's ``hubert.ckpt`` to replace them: the plugin
hands it the configuration alone (``hub/config.json`` and ``hub/preprocessor_config.json``, which bolo downloads
with the model) and builds the network from it.

Options: ``max_s`` (the longest utterance taken, 30 s), ``beam_size`` (80, as trained; smaller is faster).
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
from bolo.engines.base import FloatArray, IntentModel, IntentResult

from bolo_torch.runtime import InvalidRequest, LoadContext, model_dir, require

SR = 16000


class SpeechBrainSlu(IntentModel):
    sample_rate = SR

    def __init__(self, model: Any, device: str, max_s: float = 30.0) -> None:
        self.model = model
        self.device = device
        self.max_audio_s = max_s
        self._lock = threading.Lock()

    def understand(self, samples: FloatArray) -> IntentResult:
        import torch
        from bolo.intent import parse_semantics

        x = np.ascontiguousarray(samples, dtype=np.float32)
        with self._lock, torch.inference_mode():
            wavs = torch.from_numpy(x).unsqueeze(0).to(self.device)
            encoded = self.model.mods.hubert(wavs)
            tokens, _, _, _ = self.model.mods.beam_searcher(encoded, torch.tensor([1.0], device=self.device))
            ids = [int(t) for t in tokens[0]]
            text = str(self.model.hparams.tokenizer.decode_ids(ids))
        return parse_semantics(text)


@contextlib.contextmanager
def config_only_encoders() -> Iterator[None]:
    """While SpeechBrain builds a model, its Hugging Face encoder wrappers build their network from the
    configuration instead of downloading pretrained weights (the model's own checkpoint fills it afterwards)."""
    from speechbrain.integrations.huggingface.huggingface import HFTransformersInterface

    original = HFTransformersInterface._from_pretrained

    def from_config(self: Any, source: Any, save_path: Any, cache_dir: Any, device: Any = None, **kwargs: Any) -> None:
        self.model = self.auto_class.from_config(self.config)
        if device is not None:
            self.model.to(device)

    HFTransformersInterface._from_pretrained = from_config  # type: ignore[method-assign]
    try:
        yield
    finally:
        HFTransformersInterface._from_pretrained = original  # type: ignore[method-assign]


def load(ctx: LoadContext) -> SpeechBrainSlu:
    require("transformers", "slu")
    require("sentencepiece", "slu")
    interfaces = require("speechbrain.inference.interfaces", "slu")
    root = model_dir(ctx.path, ctx.options)
    hub = root / str(ctx.option("encoder_config_dir", "hub"))
    if not (hub / "config.json").is_file():
        raise InvalidRequest(f"no encoder configuration (config.json) in {hub}")
    overrides: dict[str, Any] = {"wav2vec2_hub": str(hub), "pretrained_path": str(root)}
    with config_only_encoders():
        model = interfaces.Pretrained.from_hparams(
            source=str(root), savedir=str(root), hparams_file="hyperparams.yaml", overrides=overrides,
            run_opts={"device": ctx.device},
        )  # fmt: skip
    model.eval()
    if ctx.option("beam_size"):
        model.mods.beam_searcher.beam_size = int(ctx.option("beam_size"))
    return SpeechBrainSlu(model, ctx.device, float(ctx.option("max_s", 30.0)))
