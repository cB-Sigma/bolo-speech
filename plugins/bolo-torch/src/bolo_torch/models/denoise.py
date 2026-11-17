"""Speech enhancement (bolo task ``denoise``) in PyTorch: Facebook Research's Demucs denoiser.

``denoiser`` (github.com/facebookresearch/denoiser) is a causal waveform-to-waveform U-Net with an LSTM (Defossez
et al., Interspeech 2020) at 16 kHz. Its code and its pretrained weights (``dns48``, ``dns64``, ``master64``,
``valentini_nc``) are under CC-BY-NC 4.0: **not for commercial use**. The weights are plain state dicts, downloaded
by bolo (pinned by SHA-256); the model class comes from the ``denoiser`` package (extra ``demucs-denoiser``; its
other dependencies are only needed for training, ``pip install --no-deps denoiser julius`` is enough here).

Options: ``model`` (the weights' file name), ``hidden`` (48 or 64), ``causal`` (true), ``stride`` (4),
``resample`` (4), ``depth`` (5), ``dry`` (0: the share of the original mixed back in by the model's own recipe; bolo's
``denoise_strength`` does the same on bolo's side).
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from bolo.engines.base import DenoiseModel, FloatArray

from bolo_torch.runtime import InvalidRequest, LoadContext, model_dir, require

SR = 16000


class DemucsDenoiser(DenoiseModel):
    sample_rate = SR

    def __init__(self, model: Any, device: str, dry: float = 0.0) -> None:
        self.model = model
        self.device = device
        self.dry = dry
        self._lock = threading.Lock()

    def denoise(self, samples: FloatArray) -> FloatArray:
        import torch

        x = np.ascontiguousarray(samples, dtype=np.float32)
        if not len(x):
            return x
        with self._lock, torch.inference_mode():
            noisy = torch.from_numpy(x).to(self.device).view(1, 1, -1)
            out = self.model(noisy)
        y = out.reshape(-1).to(torch.float32).cpu().numpy()
        y = np.pad(y, (0, len(x) - len(y))) if len(y) < len(x) else y[: len(x)]
        if self.dry:
            y = (1 - self.dry) * y + self.dry * x
        return np.ascontiguousarray(y, dtype=np.float32)


def load_demucs(ctx: LoadContext) -> DemucsDenoiser:
    demucs = require("denoiser.demucs", "demucs-denoiser")
    import torch

    root = model_dir(ctx.path, ctx.options)
    files = sorted(root.glob(str(ctx.option("model", "*.th"))))
    if not files:
        raise InvalidRequest(f"no Demucs denoiser weights (*.th) in {root}")
    kwargs: dict[str, Any] = {
        "hidden": int(ctx.option("hidden", 48)),
        "causal": bool(ctx.option("causal", True)),
        "stride": int(ctx.option("stride", 4)),
        "resample": int(ctx.option("resample", 4)),
        "depth": int(ctx.option("depth", 5)),
    }
    model = demucs.Demucs(**kwargs, sample_rate=SR)
    state = torch.load(str(files[0]), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(ctx.device)
    return DemucsDenoiser(model, ctx.device, float(ctx.option("dry", 0.0)))
