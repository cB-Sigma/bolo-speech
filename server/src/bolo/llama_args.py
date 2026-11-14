"""Command-line arguments for the llama-server processes bolo starts.

Three groups, each left out when the operator already wrote the same flag (in the catalog entry's
``server_args``, ``tts_server_args`` or ``s2s_server_args``), because an operator's explicit argument wins:

* speed and memory (:func:`tuning_args`): flash attention (``-fa``), the KV cache type
  (``--cache-type-k`` / ``--cache-type-v``), the context size (``-c``) and the batch size (``-b``). Each comes from,
  first that is set: the model's pinned setting (``PUT /api/v1/models/{id}/settings``), the catalog entry's
  ``options``, the server-wide ``llama_*`` setting. Unset everywhere: llama-server's own default.
* GPUs (:func:`gpu_args`): ``-ngl`` from the placement, and on a machine with several GPUs ``--split-mode none
  --main-gpu N`` (one GPU) or ``--split-mode layer --tensor-split ... --main-gpu N`` (layers spread over several).
* parallel requests (:func:`parallel_args`): ``--parallel N`` (llama-server's slots).

The values are checked (:func:`check_tuning`) when the settings are read and when a model setting is pinned.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

#: tuning keys, as model options and pinned model settings; the server-wide settings are ``llama_<key>``
TUNING_KEYS = ("flash_attention", "kv_cache_type", "context_size", "batch_size")
FLASH_VALUES = ("on", "off", "auto")
#: cache types llama-server accepts (``--cache-type-k``); q8_0 and q4_0 are the ones Ollama offers
KV_CACHE_TYPES = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")

_FLASH_FLAGS = ("-fa", "--flash-attn")
_KV_FLAGS = ("-ctk", "--cache-type-k", "-ctv", "--cache-type-v")
_CTX_FLAGS = ("-c", "--ctx-size")
_BATCH_FLAGS = ("-b", "--batch-size")
_GPU_FLAGS = ("-sm", "--split-mode", "-mg", "--main-gpu", "-ts", "--tensor-split", "-dev", "--device")
_PARALLEL_FLAGS = ("-np", "--parallel")


def has_flag(args: Iterable[str], names: Iterable[str]) -> bool:
    """``args`` contains one of ``names``, as ``-x value`` or ``-x=value``."""
    wanted = tuple(names)
    return any(a in wanted or a.split("=", 1)[0] in wanted for a in (str(x) for x in args))


def normalise_flash(value: Any) -> str:
    """``true``/``on``/``1`` -> ``on``, ``false``/``off``/``0`` -> ``off``, ``auto``; ``""`` for unset."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on", "enabled"):
        return "on"
    if text in ("0", "false", "no", "off", "disabled"):
        return "off"
    return text


def check_tuning(values: Mapping[str, Any], prefix: str = "") -> None:
    """Raise ``ValueError`` (naming ``<prefix><key>``) for a value llama-server would reject."""
    flash = normalise_flash(values.get("flash_attention"))
    if flash and flash not in FLASH_VALUES:
        raise ValueError(f"{prefix}flash_attention: expected on, off or auto, got {values.get('flash_attention')!r}")
    kv = str(values.get("kv_cache_type") or "").strip().lower()
    if kv and kv not in KV_CACHE_TYPES:
        raise ValueError(f"{prefix}kv_cache_type: expected one of {', '.join(KV_CACHE_TYPES)}, got {kv!r}")
    for key in ("context_size", "batch_size"):
        raw = values.get(key)
        if raw in (None, ""):
            continue
        try:
            number = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{prefix}{key}: expected a whole number, got {raw!r}") from exc
        if number < 0:
            raise ValueError(f"{prefix}{key}: expected 0 (llama-server's default) or more, got {number}")


def resolve_tuning(settings: Any, options: Mapping[str, Any], pins: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The four tuning values for one model: pin, else catalog option, else ``llama_<key>`` setting.

    Values that are unset everywhere are left out (llama-server's default applies)."""
    out: dict[str, Any] = {}
    for key in TUNING_KEYS:
        for source in (pins or {}, options, {key: getattr(settings, f"llama_{key}", None)}):
            raw = source.get(key)
            if raw in (None, "") or (key in ("context_size", "batch_size") and str(raw) == "0"):
                continue
            if key == "flash_attention":
                out[key] = normalise_flash(raw)
            elif key == "kv_cache_type":
                out[key] = str(raw).strip().lower()
            else:
                out[key] = int(raw)
            break
    return out


def tuning_args(tuning: Mapping[str, Any], extra: Iterable[str]) -> list[str]:
    """llama-server arguments for :func:`resolve_tuning`'s values, minus what ``extra`` already sets.

    A quantized V cache needs flash attention in llama.cpp, so ``-fa on`` is added with one unless flash attention
    was set (to anything) somewhere."""
    extra = [str(a) for a in extra]
    out: list[str] = []
    flash = str(tuning.get("flash_attention") or "")
    kv = str(tuning.get("kv_cache_type") or "")
    if not flash and kv and kv not in ("f32", "f16", "bf16"):
        flash = "on"
    if flash and not has_flag(extra, _FLASH_FLAGS):
        out += ["-fa", flash]
    if kv and not has_flag(extra, _KV_FLAGS):
        out += ["--cache-type-k", kv, "--cache-type-v", kv]
    if tuning.get("context_size") and not has_flag(extra, _CTX_FLAGS):
        out += ["-c", str(int(tuning["context_size"]))]
    if tuning.get("batch_size") and not has_flag(extra, _BATCH_FLAGS):
        out += ["-b", str(int(tuning["batch_size"]))]
    return out


def gpu_args(
    device: str,
    gpu_layers: int | None,
    extra: Iterable[str],
    *,
    gpus: list[int] | None = None,
    tensor_split: list[float] | None = None,
    main_gpu: int | None = None,
    visible: int = 1,
) -> list[str]:
    """``-ngl`` and the multi-GPU flags for a llama-server bolo starts.

    ``-ngl``: none when the operator set it; the scheduler's count when it computed one; else all layers (``99``)
    for any device other than the CPU. GPU selection: nothing on a machine with one GPU; with several, the GPUs
    the placement chose: ``--split-mode none --main-gpu N`` for one, ``--split-mode layer --tensor-split a,b,..
    --main-gpu N`` (one share per visible GPU, 0 for those not used) for several. Left out when the operator set
    any of ``-sm``, ``-mg``, ``-ts`` or ``--device``."""
    from bolo.placement import operator_layers

    extra = [str(a) for a in extra]
    out: list[str] = []
    if operator_layers(extra) is None:
        if gpu_layers is not None:
            out += ["-ngl", str(max(0, gpu_layers))]
        elif device != "cpu":
            out += ["-ngl", "99"]
    if device == "cpu" or not gpus or has_flag(extra, _GPU_FLAGS) or (gpu_layers is not None and gpu_layers <= 0):
        return out
    if len(gpus) > 1 and tensor_split:
        n = max(visible, max(gpus) + 1)
        shares = [0.0] * n
        for idx, share in zip(gpus, tensor_split):
            shares[idx] = share
        text = ",".join(f"{s:g}" for s in shares)
        out += [
            "--split-mode",
            "layer",
            "--tensor-split",
            text,
            "--main-gpu",
            str(main_gpu if main_gpu is not None else gpus[0]),
        ]
    elif visible > 1 or gpus[0] != 0:
        out += ["--split-mode", "none", "--main-gpu", str(gpus[0])]
    return out


def parallel_args(n: int, extra: Iterable[str], always: bool = False) -> list[str]:
    """``--parallel N`` (llama-server's slots) when more than one request should run at once (``always``: also 1)."""
    if (n <= 1 and not always) or has_flag(extra, _PARALLEL_FLAGS):
        return []
    return ["--parallel", str(n)]


def has_gpu_flags(args: Iterable[str]) -> bool:
    """The operator chose the GPUs of a llama-server (``-sm``, ``-mg``, ``-ts`` or ``--device``)."""
    return has_flag(args, _GPU_FLAGS)
