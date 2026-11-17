"""DiariZen (BUT Speech@FIT): speaker diarization with a pruned WavLM Large and Conformer layers, then VBx.

What runs is the authors' own pipeline, ``diarizen.pipelines.inference.DiariZenPipeline`` from
github.com/BUTSpeechFIT/DiariZen (MIT) at a pinned commit, on the fork of pyannote.audio 3.1.1 that lives in the
same repository (``pyannote-audio/``, MIT). Neither is on PyPI, so both are installed from that commit::

    pip install "bolo-torch[diarizen]" \\
        "pyannote.audio @ git+https://github.com/BUTSpeechFIT/DiariZen@<commit>#subdirectory=pyannote-audio" \\
        "diarizen @ git+https://github.com/BUTSpeechFIT/DiariZen@<commit>"

(:data:`INSTALL` has the exact command). The fork replaces pyannote.audio and pins pyannote.core 5.0.0 and its
siblings, so DiariZen needs a Python environment of its own (not the one with the ``pyannote`` extra). The loader
refuses any other commit of either package (pip records it in ``direct_url.json``).

The pipeline, as the authors run it (``DiariZenPipeline.__call__``): 16 s windows every 1.6 s through the
segmentation network (up to four speakers per window, powerset output), a median filter, speaker counting,
WeSpeaker ResNet34 embeddings (``pyannote/wespeaker-voxceleb-resnet34-LM``, CC-BY-4.0, fetched at a pinned revision)
of each local speaker, VBx clustering with the model's PLDA (``plda/``), and reconstruction into turns. Audio goes
in as an in-memory 16 kHz float WAV, which the pipeline reads with ``torchaudio.load``: the result is the authors'.

VBx decides the number of speakers itself: a requested ``num_speakers`` (or the catalog's ``min_speakers`` /
``max_speakers``) is not used, as in the reference, which ignores them too.

Weights: the segmentation model is CC-BY-NC-4.0 (not for commercial use), the PLDA files CC-BY-4.0.

PyTorch 2.6 and later load checkpoints with ``weights_only=True``; the two checkpoints hold pyannote's task
description and torch's version object, which are allowed by name (:data:`SAFE_GLOBALS`) for the load and nothing
else.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from bolo_torch.models.diarization import SR, PyannoteDiarization
from bolo_torch.runtime import InvalidRequest, LoadContext, MissingDependency, hub_snapshot, model_dir, require

log = logging.getLogger("bolo_torch.diarizen")

REPO = "BUTSpeechFIT/DiariZen"
COMMIT = "844f5555b0a98acd0931511fc641a8c5b8ba92c7"
EMBEDDING_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"
EMBEDDING_REVISION = "837717ddb9ff5507820346191109dc79c958d614"
INSTALL = (
    f'pip install "bolo-torch[diarizen]" "pyannote.audio @ git+https://github.com/{REPO}@{COMMIT}'
    f'#subdirectory=pyannote-audio" "diarizen @ git+https://github.com/{REPO}@{COMMIT}"'
)
#: the distributions that must come from the pinned commit
PINNED = ("diarizen", "pyannote.audio")
#: files of the model repository the pipeline reads
MODEL_FILES = ("config.toml", "pytorch_model.bin", "plda/plda.npz", "plda/xvec_transform.npz")
#: (module, name) of the classes the two checkpoints pickle besides tensors
SAFE_GLOBALS = (
    ("torch.torch_version", "TorchVersion"),
    ("pyannote.audio.core.task", "Specifications"),
    ("pyannote.audio.core.task", "Problem"),
    ("pyannote.audio.core.task", "Resolution"),
)


def installed_commit(dist: str, read: Any = None) -> str | None:
    """The git commit a distribution was installed from (pip's ``direct_url.json``), or None."""
    read = read or (lambda name: metadata.distribution(name).read_text("direct_url.json"))
    try:
        raw = read(dist)
    except metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    try:
        info = json.loads(raw)
    except ValueError:
        return None
    vcs = info.get("vcs_info") or {}
    commit = vcs.get("commit_id")
    return str(commit) if commit else None


def check_install(read: Any = None) -> None:
    """Refuse unless diarizen and the pyannote.audio fork were both installed from :data:`COMMIT`."""
    wrong = []
    for dist in PINNED:
        commit = installed_commit(dist, read)
        if commit != COMMIT:
            wrong.append(f"{dist} ({'not from a git commit' if commit is None else 'commit ' + commit[:12]})")
    if wrong:
        raise MissingDependency(
            f"DiariZen runs on its authors' code at commit {COMMIT[:12]}; this environment has "
            f"{', '.join(wrong)}. Install both in an environment of their own (the fork replaces pyannote.audio): "
            f"{INSTALL}"
        )


def wav_bytes(samples: Any) -> io.BytesIO:
    """16 kHz mono float samples as an in-memory 32-bit float WAV (what the pipeline's ``torchaudio.load`` reads)."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.ascontiguousarray(samples, dtype=np.float32), SR, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf


class DiariZenRunner:
    """The authors' pipeline behind the call shape :class:`PyannoteDiarization` uses (``{"waveform", "sample_rate"}``
    and speaker-count keywords, which VBx does not take)."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self._ignored = False

    def __call__(self, file: dict[str, Any], **speakers: Any) -> Any:
        if speakers and not self._ignored:
            log.info("DiariZen's VBx clustering finds the number of speakers itself; %s not used", speakers)
            self._ignored = True
        waveform = file["waveform"]
        samples = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)
        if int(file.get("sample_rate", SR)) != SR:
            raise InvalidRequest(f"DiariZen takes {SR} Hz audio")
        return self.pipeline(wav_bytes(np.asarray(samples, dtype=np.float32).reshape(-1)))

    def to(self, device: Any) -> DiariZenRunner:
        self.pipeline.to(device)
        return self


def safe_globals() -> list[Any]:
    import importlib

    return [getattr(importlib.import_module(module), name) for module, name in SAFE_GLOBALS]


def load(ctx: LoadContext) -> PyannoteDiarization:
    torch = require("torch", "diarizen")
    check_install()
    inference = require("diarizen.pipelines.inference", "diarizen")
    root = model_dir(ctx.path, ctx.options)
    missing = [f for f in MODEL_FILES if not (root / f).is_file()]
    if missing:
        raise InvalidRequest(f"{root} is not a DiariZen model folder (missing {', '.join(missing)})")
    embedding = hub_snapshot(
        str(ctx.option("embedding_repo", EMBEDDING_REPO)),
        str(ctx.option("embedding_revision", EMBEDDING_REVISION)),
        ["pytorch_model.bin", "config.yaml"],
    )
    allow = getattr(torch.serialization, "safe_globals", None)  # torch 2.5 and later
    with allow(safe_globals()) if allow is not None else contextlib.nullcontext():
        # a dictionary, not the path: pyannote picks its ONNX WeSpeaker loader for any path containing "wespeaker"
        pipeline = inference.DiariZenPipeline(
            diarizen_hub=Path(root).resolve(),
            embedding_model={"checkpoint": str(Path(embedding) / "pytorch_model.bin")},
        )
    runner = DiariZenRunner(pipeline).to(torch.device(ctx.device))
    return PyannoteDiarization(runner, getattr(pipeline, "_embedding", None))
