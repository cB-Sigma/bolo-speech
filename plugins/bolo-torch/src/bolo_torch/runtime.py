"""Device, dtype and thread selection, Hugging Face downloads and the plugin's error codes."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

log = logging.getLogger("bolo_torch")

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


#: bolo's own ``device`` names for an accelerator, and the torch device each one is: Apple silicon (bolo resolves
#: ``auto`` to ``coreml`` there) is MPS; an AMD GPU is ``cuda`` in a ROCm build of torch (HIP serves torch.cuda)
BOLO_ALIASES = {"coreml": "mps", "metal": "mps", "mlx": "mps", "rocm": "cuda", "hip": "cuda", "gpu": "cuda"}


def _mps_available(torch: Any) -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


_fallback = threading.local()


def take_device_fallback() -> str | None:
    """Why the last :func:`pick_device` in this thread ran on the CPU instead of the GPU it was asked for (once)."""
    note: str | None = getattr(_fallback, "note", None)
    _fallback.note = None
    return note


def _fall_back(note: str) -> str:
    log.warning("%s", note)
    _fallback.note = note
    return "cpu"


#: how to get a CUDA build of torch (PyPI's torch is CPU only on Windows; docs/configuration.md)
CUDA_TORCH = "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128"


def _no_cuda(torch: Any) -> str:
    version = str(torch.__version__)
    if getattr(torch.version, "cuda", None) is None and not getattr(torch.version, "hip", None):
        why = f"torch {version} is a CPU-only build"
    else:
        why = f"torch {version} sees no CUDA device (no NVIDIA GPU or driver visible)"
    return f"{why}, so the model runs on the CPU. For the GPU install a CUDA build of torch: {CUDA_TORCH}"


def pick_device(requested: str | None, options: dict[str, Any] | None = None) -> str:
    """``BOLO_TORCH_DEVICE`` > the model's ``device`` option > bolo's ``device`` setting; ``auto`` detects.

    bolo's names for accelerators are accepted too: ``coreml`` / ``metal`` is MPS and ``rocm`` / ``hip`` is
    ``cuda`` on a ROCm build of torch. A torch build that cannot reach the GPU bolo gave (its ``auto`` resolved to
    the machine's GPU: a CPU-only torch next to an AMD GPU, a virtual Mac without Metal) runs on the CPU with a
    warning instead of failing, and so do ``cuda`` and ``rocm`` asked for by name on a torch without them
    (docs/rocm.md), and a model option ``auto`` when bolo placed the model on its GPU; :func:`take_device_fallback`
    says why. ``mps`` or ``coreml`` asked for by name without Metal fails."""
    _fallback.note = None
    options = options or {}
    chosen = os.environ.get("BOLO_TORCH_DEVICE") or options.get("device")
    from_bolo = not chosen
    choice = (chosen or requested or "auto").strip().lower()
    base, _, index = choice.partition(":")
    alias = BOLO_ALIASES.get(base)
    if alias is not None:
        torch = _torch()
        hip = getattr(torch.version, "hip", None)
        if alias == "mps" and not _mps_available(torch):
            if not from_bolo:
                raise InvalidRequest(f"device {choice!r} was requested but Metal (MPS) is not available")
            return _fall_back(f"device {choice!r}: this torch sees no Metal (MPS) device, so the model runs on the CPU")
        if alias == "cuda" and base in ("rocm", "hip") and not (hip and torch.cuda.is_available()):
            what = "is not a ROCm build" if not hip else "sees no AMD GPU"
            return _fall_back(
                f"device {choice!r}: torch {torch.__version__} {what}, so the model runs on the CPU (for the GPU, "
                "install torch from https://download.pytorch.org/whl/rocm6.4 or newer: docs/rocm.md)"
            )
        choice = alias + (f":{index}" if index and alias == "cuda" else "")
        base = alias
    if base not in DEVICES:
        raise InvalidRequest(f"unknown device {choice!r} (use one of {', '.join(DEVICES)}, rocm, coreml, or cuda:N)")
    torch = _torch()
    if base == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available(torch):
            return "mps"
        placed = (requested or "").strip().lower()
        if not from_bolo and placed.partition(":")[0] not in ("", "auto", "cpu"):
            # the model's option says auto, and bolo placed it on its GPU: it runs on the CPU, so say so (bolo moves
            # its placement to the CPU; `bolo ps` said "100% GPU" otherwise)
            return _fall_back(f"device {placed!r}: {_no_cuda(torch)}")
        return "cpu"
    if base == "cuda" and not torch.cuda.is_available():
        # bolo's device (cuda:0 next to an NVIDIA GPU), BOLO_TORCH_DEVICE or the model's option: the CPU with the
        # reason, as the built-in engines do, rather than refusing the model
        return _fall_back(f"device {choice!r}: {_no_cuda(torch)}")
    if base == "mps" and not _mps_available(torch):
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


def split_kwargs(
    ctx: LoadContext, model_cls: Any = None, path: Path | str | None = None, **build: Any
) -> dict[str, Any]:
    """``from_pretrained`` arguments that split a model across the GPU and the CPU (accelerate's
    ``device_map="auto"`` limited by ``max_memory``), when bolo asked for a split and accelerate is installed;
    empty otherwise (load whole, then ``.to(device)``).

    With ``model_cls`` (what ``from_pretrained`` is called on; an ``Auto`` class too) and the model's ``path``, a
    model class that declares no ``_no_split_modules`` (Hugging Face remote code often does not, and transformers
    then refuses ``device_map="auto"``) is planned here instead: :func:`planned_split`. ``build`` holds what the
    model needs to be built: ``trust_remote_code`` and arguments of the model class itself."""
    if not ctx.max_memory or ctx.device == "cpu":
        return {}
    import importlib.util

    if importlib.util.find_spec("accelerate") is None:
        return {}
    limits: dict[Any, Any] = {}
    for key, value in ctx.max_memory.items():
        limits[int(key) if str(key).isdigit() else str(key)] = int(value)
    split = {"device_map": "auto", "max_memory": limits}
    if model_cls is None or path is None:
        return split
    return planned_split(ctx, split, model_cls, Path(path), **build)


def repeated_blocks(model: Any) -> list[str]:
    """The class names of a model's repeated layers, the largest stack first: the class of the elements of each
    ``nn.ModuleList`` of two or more modules of one class that have modules of their own (a decoder's layers, an
    audio encoder's blocks). Each such layer adds its input to its output, so it must stay on one device."""
    from torch import nn

    stacks: list[tuple[int, str]] = []
    for module in model.modules():
        if not isinstance(module, nn.ModuleList) or len(module) < 2:
            continue
        classes = {type(m) for m in module}
        first = module[0]
        if len(classes) != 1 or next(first.children(), None) is None:  # mixed, or leaves (Linear) that never split
            continue
        stacks.append((sum(int(p.numel()) for p in module.parameters()), type(first).__name__))
    names: list[str] = []
    for _, name in sorted(stacks, key=lambda s: -s[0]):
        if name not in names:
            names.append(name)
    return names


def _empty_model(model_cls: Any, path: Path, trust_remote_code: bool, build: dict[str, Any]) -> Any:
    """The model built without weights (accelerate's ``init_empty_weights``: every parameter on the meta device)."""
    import accelerate
    import transformers

    config = transformers.AutoConfig.from_pretrained(str(path), trust_remote_code=trust_remote_code)
    with accelerate.init_empty_weights():
        if hasattr(model_cls, "_model_mapping"):  # AutoModel, AutoModelForCausalLM, ...
            return model_cls.from_config(config, trust_remote_code=trust_remote_code, **build)
        return model_cls._from_config(config, **build)


#: the output layer's names: bitsandbytes leaves it unquantized
OUTPUT_LAYERS = ("lm_head", "output_layer", "embed_out")


def unquantized(model: Any, dtype: Any) -> dict[str, Any]:
    """The parameters a bitsandbytes load keeps in ``dtype``: all but those of the linear layers (embeddings,
    norms), and the output layer's."""
    from torch import nn

    out = getattr(model, "get_output_embeddings", lambda: None)()
    special: dict[str, Any] = {}
    for name, module in model.named_modules():
        last = name.rsplit(".", 1)[-1]
        if isinstance(module, nn.Linear) and module is not out and last not in OUTPUT_LAYERS:
            continue
        for pname, _ in module.named_parameters(recurse=False):
            special[f"{name}.{pname}" if name else pname] = dtype
    return special


def quantized_device_map(
    ctx: LoadContext, split: dict[str, Any], model_cls: Any, path: Path | str, *, bits: int = 4,
    trust_remote_code: bool = False, **build: Any,
) -> dict[str, Any]:  # fmt: skip
    """An explicit GPU/CPU device map for a bitsandbytes 4-bit (or 8-bit) load that bolo asked to split
    (``split``: :func:`split_kwargs`' ``{"device_map": "auto", "max_memory"}``).

    transformers converts a model's layers to 4-bit before it works out ``device_map="auto"``, so it cannot leave the
    layers bound for the CPU unconverted: they become 4-bit layers on the CPU, whose weights accelerate's offload keeps
    on the meta device and cannot load ("Cannot copy out of meta tensor", GLM-4-Voice int4 on a 6 GB GPU). With an
    explicit map the CPU's share stays in its own dtype (``llm_int8_enable_fp32_cpu_offload``), as transformers'
    documentation asks. The map is sized for the quantized weights (each GPU budget less transformers' 10 % margin)
    and keeps the model's layers whole."""
    import accelerate
    import torch

    model = _empty_model(model_cls, Path(path), trust_remote_code, build)
    no_split = list(getattr(model, "_no_split_modules", None) or []) or repeated_blocks(model)
    dtype: Any = torch.int8
    if bits == 4:
        with contextlib.suppress(ImportError, AttributeError):
            from accelerate.utils import CustomDtype

            dtype = CustomDtype.INT4
    limits = {k: int(v * 0.9) for k, v in dict(split["max_memory"]).items()}
    device_map = dict(
        accelerate.infer_auto_device_map(
            model, max_memory=limits, no_split_module_classes=no_split, dtype=dtype,
            special_dtypes=unquantized(model, ctx.dtype or torch.bfloat16),
        )
    )  # fmt: skip
    if "disk" in device_map.values():
        raise PluginError(
            f"{ctx.model_id or type(model).__name__} does not fit in the GPU and CPU memory bolo allows, even in "
            f"{bits}-bit",
            code="out_of_memory",
        )
    log.info(
        "%s: %d-bit with a GPU/CPU split over %s (an explicit map: the CPU's share stays unquantized)",
        ctx.model_id, bits, ", ".join(sorted({str(v) for v in device_map.values()})),
    )  # fmt: skip
    return device_map


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f}"


def planned_split(
    ctx: LoadContext,
    split: dict[str, Any],
    model_cls: Any,
    path: Path,
    *,
    trust_remote_code: bool = False,
    **build: Any,
) -> dict[str, Any]:
    """``split`` itself when the model names the layers to keep whole (``_no_split_modules``); otherwise an
    explicit device map: the model is built without weights, its repeated layers found (:func:`repeated_blocks`)
    and accelerate's ``infer_auto_device_map`` given them as ``no_split_module_classes`` and bolo's ``max_memory``.
    A model with no repeated layers cannot be split: :class:`PluginError` (``out_of_memory``) says how much memory
    it needs on one device, before anything is loaded."""
    if getattr(model_cls, "_no_split_modules", None) and not hasattr(model_cls, "_model_mapping"):
        return split
    try:
        model = _empty_model(model_cls, path, trust_remote_code, build)
    except Exception as exc:  # leave it to from_pretrained, as before
        log.warning("%s: could not build the model without weights to plan its split: %s", ctx.model_id, exc)
        return split
    # transformers 4 refuses device_map="auto" without it; transformers 5 makes it an empty set and splits
    # anywhere, also inside a layer, whose output then meets its input on another device
    if getattr(model, "_no_split_modules", None):
        return split
    import accelerate
    from accelerate.utils import compute_module_sizes

    name = type(model).__name__
    dtype = ctx.dtype
    needed = int(compute_module_sizes(model, dtype=dtype)[""])
    limits: dict[Any, int] = split["max_memory"]
    gpu = sum(v for k, v in limits.items() if k not in ("cpu", "disk"))
    blocks = repeated_blocks(model)
    if not blocks:
        raise PluginError(
            f"{ctx.model_id or name} cannot be split between the GPU and the CPU ({name} names no layers to keep "
            f"whole and has no repeated layers): it needs about {_gb(needed)} GB on one device, and bolo found "
            f"{_gb(gpu)} GB free on the GPU. Free GPU memory, or load it on the CPU (device = cpu)",
            code="out_of_memory",
        )
    device_map = accelerate.infer_auto_device_map(model, max_memory=limits, no_split_module_classes=blocks, dtype=dtype)
    if "disk" in device_map.values():
        raise PluginError(
            f"{ctx.model_id or name} needs about {_gb(needed)} GB and does not fit in the {_gb(gpu)} GB free on the "
            f"GPU and the {_gb(limits.get('cpu', 0))} GB bolo allows on the CPU",
            code="out_of_memory",
        )
    log.info(
        "%s: %s declares no _no_split_modules; keeping each %s on one device (%s)",
        ctx.model_id, name, ", ".join(blocks), ", ".join(sorted({str(v) for v in device_map.values()})),
    )  # fmt: skip
    return {"device_map": device_map}


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
