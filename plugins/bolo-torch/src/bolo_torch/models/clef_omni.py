"""Cloudflare Clef-Omni: a decision model (typed answers, no generated text) on Qwen3-Omni 30B-A3B, through transformers.

Clef-Omni is the thinker of Qwen3-Omni-30B-A3B-Instruct, post-trained, with a small "joint schema head" that reads the
backbone's last hidden states and scores every option of every question at once. It answers llama.cpp's and
Cloudflare's ``/v1/systemone`` requests: a ``state``, questions of type ``noul``, ``choice`` and ``score``, and
media. Cloudflare's code for it is ``joint_schema_model.py`` in the model repository (the head, the record encoding
and ``systemone``); the backbone is transformers' own ``Qwen3OmniMoeForConditionalGeneration``.

That file is remote code: the plugin imports it only after :mod:`bolo_torch.remote_code`'s check (the operator's
``allow_remote_code`` for this model, the file pinned by sha256 at the reviewed commit). The recording goes to it as
16 kHz samples, so its media decoding (PyAV) and URL fetching are never used. llama.cpp does not run Clef-Omni yet
(its decision server knows the dense ``clef`` only; the ``clef_omni`` architecture is an unmerged branch).
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo_torch import remote_code
from bolo_torch.runtime import LoadContext, MissingDependency, model_dir, require

try:
    from bolo.engines.base import DecisionModel, DecisionRequest
except ImportError as exc:  # a bolo without decision models
    raise MissingDependency(
        "Clef-Omni needs a bolo release with decision models (the release after 1.0.0): pip install -U bolo-speech"
    ) from exc

log = logging.getLogger("bolo_torch")

CODE = "joint_schema_model.py"


class ClefOmni(DecisionModel):
    """Clef-Omni as a bolo decision model."""

    sample_rate = 16000
    placement_report: dict[str, Any] | None = None

    def __init__(self, systemone: Any, model: Any, processor: Any, name: str, *, max_audio_s: float = 60.0) -> None:
        self._systemone = systemone
        self.model = model
        self.processor = processor
        self.name = name
        self.max_audio_s = max_audio_s
        self._lock = threading.Lock()

    def decide(self, request: DecisionRequest) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.name, "state": request.text, "questions": request.questions}
        if request.samples is not None:
            body["audio"] = [np.asarray(request.samples, dtype=np.float32)]
        with self._lock:
            out = self._systemone(self.model, self.processor, body)
        return {"answers": dict(out.get("answers") or {}), "usage": dict(out.get("usage") or {})}


def _module(path: Path) -> Any:
    """``joint_schema_model.py`` under a name of its own, without putting the model folder on ``sys.path``."""
    spec = importlib.util.spec_from_file_location("bolo_torch_clef_omni_joint_schema_model", path / CODE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path / CODE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(ctx: LoadContext) -> ClefOmni:
    path = model_dir(ctx.path, ctx.options)
    # the check (operator opt-in, pinned revision and sha256 of every .py file) runs before anything is imported
    remote_code.trust_remote_code(ctx, path)
    transformers = require("transformers", "clef-omni")
    if not hasattr(transformers, "Qwen3OmniMoeForConditionalGeneration"):
        raise MissingDependency(
            f"transformers {getattr(transformers, '__version__', '?')} has no Qwen3OmniMoeForConditionalGeneration; "
            'Clef-Omni needs transformers 5.10.2 or later: pip install "bolo-torch[clef-omni]"'
        )
    code = _module(path)
    model, processor = code.load_release_model(path, device=ctx.device, dtype=ctx.dtype)
    return ClefOmni(
        code.systemone, model, processor, ctx.model_id or "clef-omni",
        max_audio_s=float(ctx.option("max_audio_s", 60.0)),
    )  # fmt: skip
