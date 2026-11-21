"""Models that are not in the curated catalog: ``bolo pull hf:<repo>`` and ``bolo pull ./directory``.

The repository listing is inspected and matched against the file layouts
the engines understand. When the layout is ambiguous (a lone ``model.onnx``
with ``tokens.txt`` could be several architectures) the caller must name
the loader with ``--kind``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bolo.catalog import ModelSpec
from bolo.errors import InvalidRequest
from bolo.store import RemoteFile, Source

if TYPE_CHECKING:
    from bolo.core import Bolo

# kind -> (engine, task, streaming)
KINDS: dict[str, tuple[str, str, bool]] = {
    "ct2-whisper": ("faster-whisper", "stt", False),
    "whisper": ("sherpa-onnx", "stt", False),
    "moonshine": ("sherpa-onnx", "stt", False),
    "moonshine_v2": ("sherpa-onnx", "stt", False),
    "transducer": ("sherpa-onnx", "stt", False),
    "nemo_transducer": ("sherpa-onnx", "stt", False),
    "nemo_ctc": ("sherpa-onnx", "stt", False),
    "zipformer_ctc": ("sherpa-onnx", "stt", False),
    "sense_voice": ("sherpa-onnx", "stt", False),
    "paraformer": ("sherpa-onnx", "stt", False),
    "dolphin_ctc": ("sherpa-onnx", "stt", False),
    "omnilingual_ctc": ("sherpa-onnx", "stt", False),
    "qwen3_asr": ("sherpa-onnx", "stt", False),
    "cohere_transcribe": ("sherpa-onnx", "stt", False),
    "online_transducer": ("sherpa-onnx", "stt", True),
    "online_nemo_ctc": ("sherpa-onnx", "stt", True),
    "online_zipformer2_ctc": ("sherpa-onnx", "stt", True),
    "online_paraformer": ("sherpa-onnx", "stt", True),
    "vits": ("sherpa-onnx", "tts", False),
    "kokoro": ("sherpa-onnx", "tts", False),
    "kitten": ("sherpa-onnx", "tts", False),
    "matcha": ("sherpa-onnx", "tts", False),
    "moonshine-onnx": ("moonshine", "stt", False),
    "whispercpp": ("whisper.cpp", "stt", False),
    "kokoro-onnx": ("kokoro-onnx", "tts", False),
    "silero_vad": ("sherpa-onnx", "vad", False),
    "ten_vad": ("sherpa-onnx", "vad", False),
    "nemotron_diarization": ("onnxruntime", "diarization", False),
    "dicow_onnx": ("onnxruntime", "stt", False),
}

#: what a model of these kinds is, whatever directory or repository it comes from: features, default options and
#: languages (the catalog entry of DiCoW's ONNX export, for a directory registered before the files are hosted)
KIND_DEFAULTS: dict[str, dict[str, Any]] = {
    "dicow_onnx": {
        "features": ["timestamps", "speaker-conditioned"],
        "options": {"diarization_model": "nemotron-3-diarization", "conditioning": "turns"},
        "languages": ["en"],
    },
}


def detect_kind(repo: str, paths: list[str]) -> str | None:
    """Guess the loader kind from a repository's file names."""
    names = [p.rsplit("/", 1)[-1].lower() for p in paths]
    name = repo.lower()

    def has(pattern: str) -> bool:
        return any(re.search(pattern, n) for n in names)

    if "model.bin" in names and "config.json" in names and (has(r"^tokenizer\.json$") or has(r"^vocabulary\.")):
        return "ct2-whisper"
    if "dicow_onnx.json" in names or ("dicow" in name and has(r"^encoder(\.int8)?\.onnx$") and has(r"^decoder")):
        return "dicow_onnx"
    if "nemotron" in name and "diar" in name and has(r"\.onnx$"):
        return "nemotron_diarization"
    if has(r"silero_vad.*\.onnx$"):
        return "silero_vad"
    if has(r"ten-vad.*\.onnx$"):
        return "ten_vad"
    if has(r"-encoder(\.int8)?\.onnx$") and has(r"-decoder(\.int8)?\.onnx$") and has(r"-tokens\.txt$"):
        return "whisper"
    if has(r"^preprocess\.onnx$") and has(r"^uncached_decode"):
        return "moonshine"
    if has(r"^encoder_model\.(ort|onnx)$") and has(r"^decoder_model_merged"):
        return "moonshine_v2"
    if has(r"^conv_frontend.*\.onnx$") and has(r"^encoder.*\.onnx$") and has(r"^decoder.*\.onnx$"):
        return "qwen3_asr"  # Qwen3-ASR and models built on it (sherpa-onnx's from_qwen3_asr layout)
    if "tokens.txt" not in names:
        return None
    if "cohere" in name and has(r"^encoder.*\.onnx$") and has(r"^decoder.*\.onnx$") and not has(r"^joiner"):
        return "cohere_transcribe"
    if has(r"^encoder.*\.onnx$") and has(r"^decoder.*\.onnx$") and has(r"^joiner.*\.onnx$"):
        if "streaming" in name:
            return "online_transducer"
        return "nemo_transducer" if ("nemo" in name or "parakeet" in name) else "transducer"
    if has(r"^voices\.bin$"):
        return "kitten" if "kitten" in name else "kokoro"
    if has(r"\.onnx\.json$") or "vits" in name or "piper" in name or "mms" in name:
        return "vits"
    if has(r"^model.*\.onnx$"):
        streaming = "streaming" in name
        if "sense-voice" in name or "sensevoice" in name:
            return "sense_voice"
        if "paraformer" in name:
            return "online_paraformer" if streaming else "paraformer"
        if "dolphin" in name:
            return "dolphin_ctc"
        if "omnilingual" in name:
            return "omnilingual_ctc"
        if "nemo" in name or "parakeet" in name or "conformer" in name:
            return "online_nemo_ctc" if streaming else "nemo_ctc"
        if "zipformer" in name:
            return "online_zipformer2_ctc" if streaming else "zipformer_ctc"
    return None


def choose_files(kind: str, files: list[RemoteFile], precision: str = "int8") -> list[str]:
    """Exclude patterns: skip docs and test data, and fetch one precision when both exist."""
    exclude: list[str] = ["test_wavs/*", "*.md", "*.sh", "*.py", "*.wav", "*.mp4", "*.mov", "*.srt", ".gitattributes"]
    if kind == "ct2-whisper":
        return exclude
    if kind == "nemotron_diarization":  # one int8 graph (onnx/model_quantized.onnx + its data file), no other precision
        return [*exclude, "*.nemo", "*.gguf", "*.safetensors", "onnx/model.onnx*", "onnx/model_fp16*", "onnx/model_q4*"]
    if kind == "dicow_onnx":  # the int8 graphs, or the fp32 ones with precision="float"
        if precision == "float":
            return [*exclude, "*.int8.onnx", "*.int8.onnx.data"]
        return [*exclude, "encoder.onnx", "encoder.onnx.data", "decoder.onnx", "decoder.onnx.data"]
    paths = {f.path for f in files}
    for path in sorted(paths):
        if not path.endswith(".onnx") or path.endswith(".int8.onnx"):
            continue
        twin = path[: -len(".onnx")] + ".int8.onnx"
        if twin in paths:
            exclude.append(twin if precision == "float" else path)
    exclude.extend(p for p in sorted(paths) if ".fp16." in p)
    return exclude


def register_hf_model(core: Bolo, repo: str, kind: str | None = None, task: str | None = None) -> ModelSpec:
    """Create (or return) a custom catalog entry for a Hugging Face repository."""
    repo = repo.strip().strip("/")
    revision = "main"
    if "@" in repo:
        repo, revision = repo.split("@", 1)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise InvalidRequest("expected hf:<owner>/<repo>[@revision]", param="id")
    model_id = "hf-" + re.sub(r"[^a-z0-9._-]+", "-", repo.lower().replace("/", "--"))
    existing = core.catalog.resolve(model_id) or core.catalog.resolve(repo)
    if existing is not None:
        return existing
    files = core.store.hub.list_files(repo, revision)
    detected = kind or detect_kind(repo, [f.path for f in files])
    if detected is None:
        raise InvalidRequest(
            f"could not tell what kind of model '{repo}' is from its files. Pass kind explicitly "
            f"(one of: {', '.join(sorted(KINDS))}).",
            param="kind",
        )
    if detected not in KINDS:
        raise InvalidRequest(f"unknown kind '{detected}'. Known kinds: {', '.join(sorted(KINDS))}", param="kind")
    engine, default_task, streaming = KINDS[detected]
    exclude = choose_files(detected, files)
    source = Source(repo=repo, revision=revision, exclude=exclude)
    from bolo.store import match_patterns

    size = sum(f.size or 0 for f in files if match_patterns(f.path, [], exclude))
    info: dict[str, Any] = {}
    try:
        info = core.store.hub.repo_info(repo)
    except Exception:
        info = {}
    card = info.get("cardData") or {}
    license_ = str(card.get("license") or "see model card")
    langs = card.get("language") or []
    if isinstance(langs, str):
        langs = [langs]
    defaults = KIND_DEFAULTS.get(detected, {})
    spec = ModelSpec(
        id=model_id,
        aliases=[repo],
        task=task or default_task,
        engine=engine,
        kind=detected,
        family="Custom",
        languages=[str(x) for x in langs][:50] or list(defaults.get("languages", [])),
        features=list(defaults.get("features", [])),
        options=dict(defaults.get("options", {})),
        tier="balanced",
        streaming=streaming,
        size_bytes=size,
        license=license_,
        gated=bool(info.get("gated")),
        description=f"User-added model from huggingface.co/{repo} (not validated by the bolo maintainers).",
        source_url=f"https://huggingface.co/{repo}",
        validated="untested",
        source=source,
        custom=True,
    )
    spec.validate()
    core.catalog.add_custom(spec)
    return spec


def register_local_model(
    core: Bolo,
    path: str,
    kind: str | None = None,
    task: str | None = None,
    *,
    name: str | None = None,
    languages: list[str] | None = None,
    options: dict[str, Any] | None = None,
) -> ModelSpec:
    """Create (or return) a custom catalog entry for a model directory on this machine.

    The directory is used in place (the store keeps a link to it), so nothing is copied and the
    files must stay where they are. The layout is the one the engines expect, for example
    ``model.onnx`` + ``tokens.txt`` for a VITS voice.
    """
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise InvalidRequest(f"'{path}' is not a directory", param="id")
    files = sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file())
    if not files:
        raise InvalidRequest(f"'{path}' is empty", param="id")
    model_id = name or "local-" + re.sub(r"[^a-z0-9._-]+", "-", directory.name.lower()).strip("-")
    existing = core.catalog.resolve(model_id)
    if existing is not None:
        if existing.source.local and Path(existing.source.local) == directory:
            return existing
        raise InvalidRequest(f"the name '{model_id}' is already taken; choose another with --name", param="name")
    detected = kind or detect_kind(directory.name, files)
    if detected is None:
        raise InvalidRequest(
            f"could not tell what kind of model '{directory.name}' is from its files. Pass --kind "
            f"(one of: {', '.join(sorted(KINDS))}).",
            param="kind",
        )
    if detected not in KINDS:
        raise InvalidRequest(f"unknown kind '{detected}'. Known kinds: {', '.join(sorted(KINDS))}", param="kind")
    engine, default_task, streaming = KINDS[detected]
    defaults = KIND_DEFAULTS.get(detected, {})
    opts = {**defaults.get("options", {}), **(options or {})}
    if languages and "language" not in opts and (task or default_task) == "tts":
        opts["language"] = languages[0]
    spec = ModelSpec(
        id=model_id,
        task=task or default_task,
        engine=engine,
        kind=detected,
        family="Custom",
        languages=list(languages or defaults.get("languages", [])),
        features=list(defaults.get("features", [])),
        tier="balanced",
        streaming=streaming,
        size_bytes=sum((directory / f).stat().st_size for f in files if f != "manifest.json"),
        license="see the model's own terms (local directory)",
        description=f"User-added model from the local directory {directory} (not validated by the bolo maintainers).",
        source_url="",
        validated="untested",
        source=Source(local=str(directory)),
        options=opts,
        custom=True,
    )
    spec.validate()
    core.catalog.add_custom(spec)
    return spec
