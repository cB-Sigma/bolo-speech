"""Blind speech separation: SpeechBrain SepFormer (2 speakers) and pyannote PixIT (diarization + separation).

Both are bolo ``separation`` models in ``"blind"`` mode: a 16 kHz mixture in, one 16 kHz track per source out,
which bolo assigns to its diarized speakers by voice similarity (``separate_overlaps``).

SepFormer WHAMR! checkpoints are Apache-2.0 and not gated, but they were trained on WHAMR!, which is built from
WSJ0 (LDC licence): a provenance risk for commercial use. The 8 kHz checkpoint gets 16 kHz audio resampled to
8 kHz and returns its tracks resampled to 16 kHz. PixIT (``pyannote/speech-separation-ami-1.0``) is MIT and
gated; it is trained on AMI meetings.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from bolo.engines.base import FloatArray, SeparationModel

from bolo_torch.runtime import LoadContext, ModelGated, PluginError, gated_message, hf_token, model_dir, require

SR = 16000


def resample(samples: FloatArray, rate: int, target: int) -> FloatArray:
    if rate == target:
        return np.asarray(samples, dtype=np.float32)
    import soxr

    return np.asarray(soxr.resample(np.asarray(samples, dtype=np.float32), rate, target), dtype=np.float32)


def _fit(track: FloatArray, n: int) -> FloatArray:
    track = np.asarray(track, dtype=np.float32).reshape(-1)
    return np.pad(track, (0, n - len(track))) if len(track) < n else track[:n]


def _scale(tracks: list[FloatArray]) -> list[FloatArray]:
    peak = max((float(np.max(np.abs(t))) for t in tracks if t.size), default=0.0)
    return [(t / peak * 0.99).astype(np.float32) for t in tracks] if peak > 0.99 else tracks


class SepFormer(SeparationModel):
    mode = "blind"
    sample_rate = SR

    def __init__(self, separate_batch: Callable[[Any], Any], model_rate: int, num_speakers: int = 2) -> None:
        self.separate_batch = separate_batch
        self.model_rate = model_rate
        self.num_speakers = num_speakers
        self._lock = threading.Lock()

    def separate(self, mixture: FloatArray) -> list[FloatArray]:
        import torch

        mix = resample(mixture, SR, self.model_rate)
        with self._lock, torch.inference_mode():
            out = self.separate_batch(torch.from_numpy(np.ascontiguousarray(mix)).unsqueeze(0))
        est = out[0].to(torch.float32).cpu().numpy()  # (time, speakers)
        tracks = [_fit(resample(est[:, k], self.model_rate, SR), len(mixture)) for k in range(est.shape[1])]
        return _scale(tracks)


def load_sepformer(ctx: LoadContext) -> SepFormer:
    sb = require("speechbrain.inference.separation", "sepformer")
    root = model_dir(ctx.path, ctx.options)
    model = sb.SepformerSeparation.from_hparams(source=str(root), savedir=str(root), run_opts={"device": ctx.device})
    model.eval()
    rate = int(getattr(model.hparams, "sample_rate", 0) or ctx.option("model_sample_rate", 8000))
    speakers = int(getattr(model.hparams, "num_spks", 2) or 2)
    return SepFormer(model.separate_batch, rate, speakers)


class PixIT(SeparationModel):
    """pyannote speech-separation-ami-1.0: one track per speaker it finds (16 kHz)."""

    mode = "blind"
    sample_rate = SR

    def __init__(self, pipeline: Callable[..., Any], speakers: dict[str, int] | None = None) -> None:
        self.pipeline = pipeline
        #: ``num_speakers`` / ``min_speakers`` / ``max_speakers`` for every call (catalog options)
        self.speakers = dict(speakers or {})
        self._lock = threading.Lock()
        #: who spoke when in the last call: (start s, end s, track index)
        self.last_turns: list[tuple[float, float, int]] = []

    def separate(self, mixture: FloatArray) -> list[FloatArray]:
        import torch

        audio = np.ascontiguousarray(mixture, dtype=np.float32)
        with self._lock, torch.inference_mode():
            out = self.pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SR}, **self.speakers)
        if isinstance(out, tuple) and len(out) == 2:
            diarization, sources = out
        else:
            diarization = getattr(out, "speaker_diarization", None)
            sources = getattr(out, "sources", None)
        if diarization is None or sources is None:
            raise PluginError(f"unexpected pyannote output {type(out).__name__}", "engine_unavailable")
        labels = [str(x) for x in diarization.labels()]
        data = np.asarray(sources.data, dtype=np.float32)  # (time, speakers)
        count = len(labels) if labels else data.shape[1]
        index = {label: i for i, label in enumerate(labels)}
        self.last_turns = sorted(
            (float(seg.start), float(seg.end), index[str(label)])
            for seg, _, label in diarization.itertracks(yield_label=True)
            if str(label) in index
        )
        return _scale([_fit(data[:, k], len(mixture)) for k in range(min(count, data.shape[1]))])


def _speechbrain_fetch_shim() -> None:
    """pyannote.audio 4.0 calls SpeechBrain's ``from_hparams(..., token=, huggingface_cache_dir=, revision=)``;
    SpeechBrain 1.1 takes those inside ``fetch_config`` and rejects them as keywords
    (``Pretrained.__init__() got an unexpected keyword argument 'token'``), and it wants ``run_opts["device"]``
    as a string, not a ``torch.device``. Move and convert them.
    PixIT's speaker embedding (``speechbrain/spkrec-ecapa-voxceleb``, pinned by revision) is not gated."""
    try:
        from speechbrain.inference.interfaces import Pretrained
        from speechbrain.utils.fetching import FetchConfig
    except ImportError:
        return
    original = Pretrained.from_hparams.__func__
    if getattr(original, "_bolo_torch_shim", False):
        return
    names = ("token", "revision", "huggingface_cache_dir")
    if any(n in inspect.signature(original).parameters for n in names):
        return  # a SpeechBrain that takes them as keywords

    def from_hparams(cls: Any, *args: Any, **kwargs: Any) -> Any:
        moved = {n: kwargs.pop(n) for n in names if n in kwargs}
        run_opts = kwargs.get("run_opts")
        if isinstance(run_opts, dict) and "device" in run_opts:
            kwargs["run_opts"] = {**run_opts, "device": str(run_opts["device"])}  # 1.1 wants a string
        if moved and "fetch_config" not in kwargs:
            kwargs["fetch_config"] = FetchConfig(
                token=moved.get("token") or False,
                revision=moved.get("revision"),
                huggingface_cache_dir=moved.get("huggingface_cache_dir"),
            )
        return original(cls, *args, **kwargs)

    from_hparams._bolo_torch_shim = True  # type: ignore[attr-defined]
    Pretrained.from_hparams = classmethod(from_hparams)


def load_pixit(ctx: LoadContext) -> PixIT:
    import torch

    audio = require("pyannote.audio", "pyannote")
    repo = str(ctx.option("repo", "pyannote/speech-separation-ami-1.0"))
    root = model_dir(ctx.path, {**ctx.options, "repo": repo})
    config = root / "config.yaml"
    source = str(config) if config.is_file() else repo
    _speechbrain_fetch_shim()
    params = inspect.signature(audio.Pipeline.from_pretrained).parameters
    auth = {"token": hf_token()} if "token" in params else {"use_auth_token": hf_token()}
    try:
        pipeline = audio.Pipeline.from_pretrained(source, **auth)
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        if any(s in text for s in ("401", "403", "GatedRepo", "gated", "restricted")):
            raise ModelGated(gated_message("pyannote/separation-ami-1.0")) from exc
        raise
    if pipeline is None:  # pyannote 3.x returns None (and prints a notice) when a download is refused
        raise ModelGated(gated_message("pyannote/separation-ami-1.0"))
    leakage = ctx.option("leakage_removal")
    if leakage is not None:  # the pipeline's default (true) zeroes a track wherever its speaker is judged silent
        params = pipeline.parameters(instantiated=True)
        params.setdefault("separation", {})["leakage_removal"] = bool(leakage)
        pipeline.instantiate(params)
    pipeline.to(torch.device(ctx.device))
    speakers = {k: int(ctx.option(k)) for k in ("num_speakers", "min_speakers", "max_speakers") if ctx.option(k)}
    return PixIT(pipeline, speakers)
