"""pyannote's own speaker diarization pipelines in PyTorch: ``speaker-diarization-3.1`` and ``community-1``.

bolo's built-in diarization runs without PyTorch (the ONNX export of pyannote segmentation-3.0 plus speaker
embeddings, clustered by sherpa-onnx). These are the original pipelines, for people who want pyannote's exact
results: ``pyannote.audio`` ``Pipeline.from_pretrained`` on the files bolo downloaded, run as a bolo
``diarization`` model (``POST /api/v1/diarize``, ``diarize=true`` and ``diarized_json`` transcriptions).

* ``pyannote/speaker-diarization-3.1`` (MIT, gated) holds only the pipeline configuration; the pipeline downloads
  ``pyannote/segmentation-3.0`` (MIT, gated) and ``pyannote/wespeaker-voxceleb-resnet34-LM`` (CC-BY-4.0) itself.
* ``pyannote/speaker-diarization-community-1`` (CC-BY-4.0, gated) holds everything: segmentation, the WeSpeaker
  embedding and the PLDA of its VBx clustering. It needs pyannote.audio 4.

The pipeline is built from a checked copy of its configuration, never from the file itself
(:mod:`bolo_torch.pyannote_config`): only pyannote's own classes, only pinned sub-models.

pyannote.audio 4 sends usage metrics to pyannote's servers unless ``PYANNOTE_METRICS_ENABLED`` says otherwise; the
plugin turns them off unless that variable is already set.
"""

from __future__ import annotations

import inspect
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from bolo.engines.base import DiarizationModel, DiarizationResult, FloatArray, SpeakerTurn

from bolo_torch.pyannote_config import COMMIT, UnsafeConfig, model_ref, safe_config
from bolo_torch.runtime import (
    InvalidRequest,
    LoadContext,
    ModelGated,
    PluginError,
    gated_message,
    hf_token,
    model_dir,
    require,
)

SR = 16000
SPEAKER_OPTIONS = ("num_speakers", "min_speakers", "max_speakers")


def import_pyannote() -> Any:
    """``pyannote.audio``, with its usage metrics off unless ``PYANNOTE_METRICS_ENABLED`` is set already."""
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
    return require("pyannote.audio", "pyannote")


def is_gated_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(s in text for s in ("401", "403", "GatedRepo", "gated", "restricted"))


def pinned_root(ctx: LoadContext, repo: str) -> Path:
    """The model folder: the one bolo downloaded (pinned revision, hashes verified), else the plugin downloads
    ``repo`` itself, but only at a commit given in the option ``revision``."""
    if ctx.path.is_dir() and any(ctx.path.iterdir()):
        return ctx.path
    revision = str(ctx.option("revision", ""))
    if not COMMIT.match(revision):
        raise UnsafeConfig(
            f"{ctx.path} is empty and the option 'revision' is not a commit hash: the plugin downloads pyannote "
            "pipelines only at a pinned commit (pull the model through bolo, or give 'revision')"
        )
    return model_dir(ctx.path, {**ctx.options, "repo": repo, "revision": revision})


def load_pipeline(ctx: LoadContext, repo: str, also_gated: list[str] | None = None) -> Any:
    """Build the pipeline from its checked configuration (:mod:`bolo_torch.pyannote_config`), with the token in
    the keyword this pyannote.audio takes. A refused download (the pipeline's or a gated sub-model's) becomes
    :class:`ModelGated` naming ``repo`` and ``also_gated``."""
    root = pinned_root(ctx, repo)
    revisions = {str(k): str(v) for k, v in dict(ctx.option("revisions", {})).items()}
    hashes = {str(k): str(v) for k, v in dict(ctx.option("sha256", {})).items()}
    # checked before pyannote is imported: nothing of the configuration reaches pyannote unless it passes
    config = safe_config(root, revisions=revisions, hashes=hashes)
    audio = import_pyannote()
    params = config["pipeline"]["params"]
    if (
        config["pipeline"]["name"].endswith("SpeakerDiarization")
        and "plda" not in params
        and "plda" in inspect.signature(_pipelines(audio).SpeakerDiarization.__init__).parameters
    ):
        # pyannote.audio 4 loads a PLDA for every SpeakerDiarization, even with agglomerative clustering that does
        # not use it: community-1's, from its latest revision. Give it the pinned copy instead.
        default = str(ctx.option("default_plda", "pyannote/speaker-diarization-community-1"))
        params["plda"] = model_ref("plda", default, root, revisions)
    accepts = inspect.signature(audio.Pipeline.from_pretrained).parameters
    auth = {"token": hf_token()} if "token" in accepts else {"use_auth_token": hf_token()}
    try:
        pipeline = audio.Pipeline.from_pretrained(config, **auth)
    except Exception as exc:
        if is_gated_error(exc):
            raise ModelGated(gated_message(repo, _also(also_gated))) from exc
        raise
    if pipeline is None:  # pyannote 3.x returns None (and prints a notice) when a download is refused
        raise ModelGated(gated_message(repo, _also(also_gated)))
    return pipeline


def _pipelines(audio: Any) -> Any:
    """``pyannote.audio.pipelines`` (a submodule that ``import pyannote.audio`` need not load)."""
    found = getattr(audio, "pipelines", None)
    return found if found is not None else require("pyannote.audio.pipelines", "pyannote")


def _also(repos: list[str] | None) -> str:
    if not repos:
        return ""
    pages = ", ".join(f"https://huggingface.co/{r}" for r in repos)
    return f" The pipeline also downloads gated models: accept the conditions of {pages} as well."


def annotation_turns(annotation: Any) -> list[SpeakerTurn]:
    """A pyannote ``Annotation`` as bolo turns, speakers numbered in order of first appearance."""
    tracks = sorted(
        ((float(seg.start), float(seg.end), str(label)) for seg, _, label in annotation.itertracks(yield_label=True)),
        key=lambda t: (t[0], t[1]),
    )
    index: dict[str, int] = {}
    turns: list[SpeakerTurn] = []
    for start, end, label in tracks:
        if end > start:
            turns.append(SpeakerTurn(round(start, 3), round(end, 3), index.setdefault(label, len(index))))
    return turns


class PyannoteDiarization(DiarizationModel):
    """A pyannote.audio speaker diarization pipeline as a bolo diarization model (16 kHz)."""

    sample_rate = SR
    #: the pipeline's own speaker count stands: bolo does not merge or drop speakers unless the request asks
    consolidate_default = False

    def __init__(
        self,
        pipeline: Callable[..., Any],
        embedding: Callable[[Any], Any] | None,
        speakers: dict[str, int] | None = None,
        exclusive: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.embedding = embedding
        #: ``num_speakers`` / ``min_speakers`` / ``max_speakers`` from the catalog options (a request's
        #: ``num_speakers`` wins)
        self.speakers = dict(speakers or {})
        #: community-1's exclusive diarization: one speaker at a time, no overlapping turns
        self.exclusive = exclusive
        self._lock = threading.Lock()

    def diarize_detailed(
        self, samples: FloatArray, num_speakers: int | None = None, threshold: float | None = None
    ) -> DiarizationResult:
        """``threshold`` is not used: the pipeline's own clustering parameters apply."""
        import torch

        audio = np.ascontiguousarray(samples, dtype=np.float32)
        if len(audio) == 0:
            return DiarizationResult([])
        kwargs = {"num_speakers": int(num_speakers)} if num_speakers else dict(self.speakers)
        with self._lock, torch.inference_mode():
            out = self.pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SR}, **kwargs)
        if isinstance(out, tuple):  # pyannote 3.x with return_embeddings
            out = out[0]
        annotation = out
        if hasattr(out, "speaker_diarization"):  # pyannote 4: DiarizeOutput
            exclusive = getattr(out, "exclusive_speaker_diarization", None)
            annotation = exclusive if self.exclusive and exclusive is not None else out.speaker_diarization
        if not hasattr(annotation, "itertracks"):
            raise PluginError(f"unexpected pyannote output {type(out).__name__}", "engine_unavailable")
        return DiarizationResult(annotation_turns(annotation))

    def diarize(
        self, samples: FloatArray, num_speakers: int | None = None, threshold: float | None = None
    ) -> list[SpeakerTurn]:
        return self.diarize_detailed(samples, num_speakers, threshold).turns

    def embed(self, samples: FloatArray) -> FloatArray:
        """The pipeline's own speaker embedding of one stretch of speech, unit length."""
        import torch

        if self.embedding is None:
            raise InvalidRequest("this pyannote pipeline has no speaker embedding model")
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        minimum = int(getattr(self.embedding, "min_num_samples", 0) or 0)
        if len(audio) < max(minimum, 1):
            audio = np.pad(audio, (0, max(minimum, 1) - len(audio)))
        with self._lock, torch.inference_mode():
            vector = self.embedding(torch.from_numpy(audio)[None, None, :])
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()
        vec = np.nan_to_num(np.asarray(vector, dtype=np.float32).reshape(-1))
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32) if norm > 0 else vec


def load(ctx: LoadContext) -> PyannoteDiarization:
    import torch

    repo = str(ctx.option("repo", "pyannote/speaker-diarization-3.1"))
    # gated sub-models the pipeline needs (3.1: pyannote/segmentation-3.0), named when access is refused
    also = [str(r) for r in ctx.option("also_gated", [])]
    pipeline = load_pipeline(ctx, repo, also)
    pipeline.to(torch.device(ctx.device))
    speakers = {k: int(ctx.option(k)) for k in SPEAKER_OPTIONS if ctx.option(k)}
    return PyannoteDiarization(
        pipeline, getattr(pipeline, "_embedding", None), speakers, exclusive=bool(ctx.option("exclusive", False))
    )
