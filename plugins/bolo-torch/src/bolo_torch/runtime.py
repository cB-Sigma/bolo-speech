"""Device, dtype and thread selection, Hugging Face downloads and the plugin's error codes."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

HF_TOKEN_VARS = ("HF_TOKEN", "BOLO_HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
DEVICES = ("auto", "cpu", "cuda", "mps")
DTYPES = ("auto", "float32", "float16", "bfloat16")


class PluginError(RuntimeError):
    """An error with a code that the plugin sends to bolo (``error.code`` in the reply)."""

    code = "plugin_error"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


class ModelGated(PluginError):
    """Hugging Face refused a download: the repository is gated (bolo's ``model_gated``)."""

    code = "model_gated"


class InvalidRequest(PluginError):
    code = "invalid_request"


class MissingDependency(PluginError):
    """A library of an optional extra is not installed in the plugin's environment."""

    code = "component_missing"


def gated_message(repo: str, extra: str = "") -> str:
    token = "a token is set" if hf_token() else "no token is set"
    return (
        f"Hugging Face denied access to '{repo}' ({token}). The repository is gated: open "
        f"https://huggingface.co/{repo}, accept its conditions with your account, create a read token at "
        f"https://huggingface.co/settings/tokens and set HF_TOKEN (or BOLO_HF_TOKEN) for bolo and the plugin.{extra}"
    )


def require(module: str, extra: str) -> Any:
    """Import an optional library, or explain which extra brings it."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependency(
            f"'{module}' is not installed in the plugin's environment ({exc}). "
            f'Install it with: pip install "bolo-torch[{extra}]"'
        ) from exc


def hf_token() -> str | None:
    for var in HF_TOKEN_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return None


def cache_dir() -> str | None:
    """Where the plugin's own downloads go: ``BOLO_TORCH_CACHE``, else the Hugging Face default (``HF_HOME``)."""
    return os.environ.get("BOLO_TORCH_CACHE") or None


def hub_snapshot(
    repo: str, revision: str | None = None, allow_patterns: list[str] | None = None, *, extra_hint: str = ""
) -> Path:
    """Download (or find in the cache) a Hugging Face repository; gated repositories raise :class:`ModelGated`."""
    hub = require("huggingface_hub", "parler")
    errors = require("huggingface_hub.utils", "parler")
    try:
        path = hub.snapshot_download(
            repo, revision=revision, allow_patterns=allow_patterns, token=hf_token(), cache_dir=cache_dir()
        )
    except errors.GatedRepoError as exc:
        raise ModelGated(gated_message(repo, extra_hint)) from exc
    except errors.RepositoryNotFoundError as exc:
        raise PluginError(
            f"Hugging Face repository '{repo}' was not found, or is private and the token has no access",
            "download_failed",
        ) from exc
    except errors.HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise ModelGated(gated_message(repo, extra_hint)) from exc
        raise PluginError(f"download of '{repo}' failed: {exc}", "download_failed") from exc
    return Path(path)


def model_dir(path: Path, options: dict[str, Any]) -> Path:
    """The model's files: the directory bolo downloaded, else ``options.repo`` fetched by the plugin itself."""
    if path.is_dir() and any(path.iterdir()):
        return path
    repo = options.get("repo")
    if repo:
        return hub_snapshot(str(repo), options.get("revision"))
    raise InvalidRequest(f"model directory {path} is missing or empty and no 'repo' option names a download")


def _torch() -> Any:
    return require("torch", "parler")


def pick_device(requested: str | None, options: dict[str, Any] | None = None) -> str:
    """``BOLO_TORCH_DEVICE`` > the model's ``device`` option > bolo's ``device`` setting; ``auto`` detects."""
    options = options or {}
    choice = (os.environ.get("BOLO_TORCH_DEVICE") or options.get("device") or requested or "auto").strip().lower()
    base = choice.split(":", 1)[0]
    if base not in DEVICES:
        raise InvalidRequest(f"unknown device {choice!r} (use one of {', '.join(DEVICES)}, or cuda:N)")
    torch = _torch()
    if base == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if base == "cuda" and not torch.cuda.is_available():
        raise InvalidRequest(
            f"device {choice!r} was requested but this torch build sees no CUDA device "
            f"(torch {torch.__version__}; a CPU-only build cannot use a GPU)"
        )
    if base == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise InvalidRequest("device 'mps' was requested but Metal is not available")
    return choice


def pick_dtype(device: str, options: dict[str, Any] | None = None) -> torch.dtype:
    """``BOLO_TORCH_DTYPE`` > the model's ``dtype`` option > auto (fp32 on CPU, bf16/fp16 on GPUs)."""
    options = options or {}
    choice = (os.environ.get("BOLO_TORCH_DTYPE") or options.get("dtype") or "auto").strip().lower()
    if choice not in DTYPES:
        raise InvalidRequest(f"unknown dtype {choice!r} (use one of {', '.join(DTYPES)})")
    import torch

    if choice == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device == "mps":
            return torch.float16
        return torch.float32
    dtype: torch.dtype = getattr(torch, choice)
    return dtype


_threads_lock = threading.Lock()


def set_threads(requested: int) -> int:
    """Intra-op threads of this process (``BOLO_TORCH_THREADS`` wins). A process-wide torch setting."""
    env = os.environ.get("BOLO_TORCH_THREADS")
    n = int(env) if env else int(requested or 0)
    if n > 0:
        with _threads_lock:
            _torch().set_num_threads(n)
    return int(_torch().get_num_threads())


@dataclass
class LoadContext:
    """What a model loader receives."""

    kind: str
    task: str
    path: Path
    options: dict[str, Any]
    device: str = "cpu"
    dtype: Any = None
    threads: int = 1
    model_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    #: bolo's scheduler found less free accelerator memory than the model needs: accelerate's ``max_memory``
    #: (``{"0": bytes, "cpu": bytes}``) for ``device_map="auto"``; None: the whole model on ``device``
    max_memory: dict[str, int] | None = None

    def option(self, name: str, default: Any = None) -> Any:
        value = self.options.get(name)
        return default if value is None else value


def split_kwargs(ctx: LoadContext) -> dict[str, Any]:
    """``from_pretrained`` arguments that split a model across the GPU and the CPU (accelerate's
    ``device_map="auto"`` limited by ``max_memory``), when bolo asked for a split and accelerate is installed;
    empty otherwise (load whole, then ``.to(device)``)."""
    if not ctx.max_memory or ctx.device == "cpu":
        return {}
    import importlib.util

    if importlib.util.find_spec("accelerate") is None:
        return {}
    limits: dict[Any, Any] = {}
    for key, value in ctx.max_memory.items():
        limits[int(key) if str(key).isdigit() else str(key)] = int(value)
    return {"device_map": "auto", "max_memory": limits}


def device_map_report(model: Any) -> dict[str, Any] | None:
    """How a loaded model's weights are spread: ``{"gpu_bytes", "cpu_bytes", "device_map"}`` (sent to bolo as
    ``info.placement``), or None for something without parameters."""
    params = getattr(model, "parameters", None)
    if params is None:
        return None
    gpu = cpu = 0
    for p in params():
        size = int(p.numel()) * int(p.element_size())
        if getattr(p.device, "type", "cpu") in ("cuda", "mps", "xpu"):
            gpu += size
        else:
            cpu += size
    out: dict[str, Any] = {"gpu_bytes": gpu, "cpu_bytes": cpu}
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        out["device_map"] = {str(k): str(v) for k, v in hf_map.items()}
    return out
