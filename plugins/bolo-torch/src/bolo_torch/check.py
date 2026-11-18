"""``bolo-torch check``: what ``bolo check <model>`` asks this plugin's environment, without the model's weights.

* ``imports``: the loader of the model's kind and the libraries of its extra import here;
* ``code``: an inference-code bundle the loader needs (``bolo-torch code``) is installed and matches its pins, or a
  load will install it;
* ``tiny`` (``bolo check --tiny``): the model's own architecture, from its ``config.json`` (and, for remote code,
  its reviewed ``.py`` files, checked against the catalog's pins), built with tiny dimensions and random weights,
  saved, then loaded the way the loader loads it: bolo's GPU/CPU split planning (:func:`runtime.split_kwargs`, with
  a ``max_memory`` that forces a split), the dtype the GPU would use, and one forward pass. What cannot be shrunk
  or run that way is reported as skipped, never as passed.

Input: a JSON object (the command's argument); output: ``{"steps": [{"name", "status", "detail"}]}`` on stdout,
``status`` one of ``ok``, ``fail``, ``skip``.
"""

from __future__ import annotations

import contextlib
import importlib
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

#: kinds whose model is a transformers model built from its config.json: the class ``from_pretrained`` is called on
#: (None: the class the config's ``architectures`` names, from transformers)
TINY_KINDS: dict[str, str | None] = {
    "orpheus-snac": "AutoModelForCausalLM",
    "step-audio-2": "AutoModelForCausalLM",
    "glm-4-voice": "AutoModel",
    "minicpm-o": "AutoModel",
    "qwen-omni": None,
}
#: the libraries each extra brings (bolo_torch.cli.FAMILIES, by extra name)
HIDDEN = ("hidden_size", "d_model", "n_embd", "dim", "embed_dim")
LAYERS = ("num_hidden_layers", "num_layers", "n_layer", "encoder_layers", "decoder_layers", "num_encoder_layers",
          "num_decoder_layers", "depth")  # fmt: skip
FFN = ("intermediate_size", "ffn_dim", "encoder_ffn_dim", "decoder_ffn_dim", "n_inner")
TINY_HEADS = 2
TINY_LAYERS = 2


def _step(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def check_imports(kind: str, extra: str | None) -> dict[str, str]:
    from bolo_torch.cli import FAMILIES
    from bolo_torch.engine import LOADERS, NOT_IMPLEMENTED

    if kind in NOT_IMPLEMENTED:
        return _step("imports", "fail", f"the kind {kind!r} is not implemented: {NOT_IMPLEMENTED[kind]}")
    if kind not in LOADERS:
        return _step("imports", "fail", f"this bolo-torch has no loader for the kind {kind!r}")
    module, _ = LOADERS[kind]
    missing = []
    for name in [module, *FAMILIES.get(extra or "", [])]:
        try:
            importlib.import_module(name)
        except Exception as exc:  # an import error of any kind is what this step is for
            missing.append(f"{name} ({type(exc).__name__}: {exc})")
    if missing:
        hint = f'; pip install "bolo-torch[{extra}]"' if extra else ""
        return _step("imports", "fail", "cannot import " + "; ".join(missing) + hint)
    return _step("imports", "ok", f"{module}" + (f" and the {extra} extra's libraries" if extra else ""))


def check_code(extra: str | None) -> dict[str, str] | None:
    from bolo_torch import upstream

    if not extra or extra not in upstream.bundles():
        return None
    bundle = upstream.get(extra)
    if not upstream.bundle_dir(bundle).is_dir():
        return _step("code", "ok", f"the {extra} code bundle is not installed yet; a load installs it (pinned)")
    problems = upstream.verify(bundle)
    if problems:
        return _step("code", "fail", f"the {extra} code bundle does not match its pins: " + "; ".join(problems))
    return _step("code", "ok", f"the {extra} code bundle matches its pins")


# -- tiny ------------------------------------------------------------------------------------------------------------
def shrink(config: Any) -> list[str]:
    """Make ``config`` (and every sub-configuration) tiny in place: 2 attention heads of the same head size (rotary
    sections stay valid), 2 layers, a feed-forward 4 times the width. Returns what was changed."""
    changed: list[str] = []
    seen: set[int] = set()

    def visit(c: Any, where: str) -> None:
        if id(c) in seen:
            return
        seen.add(id(c))
        attrs = vars(c)
        heads = attrs.get("num_attention_heads") or attrs.get("n_head") or attrs.get("encoder_attention_heads")
        hidden_key = next((k for k in HIDDEN if isinstance(attrs.get(k), int)), None)
        if hidden_key and isinstance(heads, int) and heads > 0 and attrs[hidden_key] % heads == 0:
            head_dim = int(attrs.get("head_dim") or attrs[hidden_key] // heads)
            width = head_dim * TINY_HEADS
            if width < attrs[hidden_key]:
                changed.append(f"{where}{hidden_key} {attrs[hidden_key]}->{width}")
                setattr(c, hidden_key, width)
                for key in ("num_attention_heads", "n_head", "encoder_attention_heads", "decoder_attention_heads"):
                    if isinstance(attrs.get(key), int):
                        setattr(c, key, TINY_HEADS)
                for key in ("num_key_value_heads", "multi_query_group_num", "num_kv_heads"):
                    if isinstance(attrs.get(key), int):
                        setattr(c, key, min(int(attrs[key]), TINY_HEADS) if attrs[key] > 1 else 1)
                if "head_dim" in attrs and attrs.get("head_dim") is None:
                    c.head_dim = head_dim
                for key in FFN:
                    if isinstance(attrs.get(key), int) and attrs[key] > 4 * width:
                        setattr(c, key, 4 * width)
        for key in LAYERS:
            value = attrs.get(key)
            if isinstance(value, int) and value > TINY_LAYERS:
                changed.append(f"{where}{key} {value}->{TINY_LAYERS}")
                setattr(c, key, TINY_LAYERS)
        if isinstance(attrs.get("layer_types"), list):
            c.layer_types = list(attrs["layer_types"])[:TINY_LAYERS]
        for key, value in list(attrs.items()):
            if hasattr(value, "to_dict") and hasattr(value, "__dict__") and not isinstance(value, type):
                visit(value, f"{where}{key}.")

    visit(config, "")
    return changed


def _model_class(transformers: Any, config: Any, wanted: str | None, kind: str = "", options: Any = None) -> Any:
    if kind == "qwen-omni":  # the class the loader uses for the entry's family (bolo_torch.models.omni)
        from bolo_torch.models.omni import FAMILIES

        family = FAMILIES[str((options or {}).get("family") or "qwen2.5-omni")]
        return getattr(transformers, family.model_class)
    if wanted is None:
        name = (getattr(config, "architectures", None) or [None])[0]
        cls = getattr(transformers, str(name), None) if name else None
        if cls is None:
            raise LookupError(f"transformers {transformers.__version__} has no class {name!r} (config.architectures)")
        return cls
    return getattr(transformers, wanted)


def check_tiny(request: dict[str, Any]) -> list[dict[str, str]]:
    kind = str(request["kind"])
    if kind not in TINY_KINDS:
        return [_step("tiny", "skip", f"the kind {kind!r} is not a transformers model bolo can shrink; not checked")]
    try:
        import torch
        import transformers
    except ImportError as exc:
        return [_step("tiny", "skip", f"transformers is not installed here ({exc})")]
    source = Path(str(request["config_dir"]))
    options = dict(request.get("options") or {})
    trust = False
    if options.get("remote_code"):
        if not request.get("allow_remote_code"):
            return [_step("tiny", "skip", "the model runs remote code: opt in (allow_remote_code) to build it tiny")]
        from bolo_torch import remote_code

        try:  # the same review a load does: the .py files and auto_map JSON against the catalog's sha256 pins
            remote_code.check(source, {**options, "allow_remote_code": True}, str(request.get("model") or kind))
        except Exception as exc:
            return [_step("tiny", "fail", f"remote code: {exc}")]
        trust = True
    steps: list[dict[str, str]] = []
    work = Path(tempfile.mkdtemp(prefix="bolo-tiny-"))
    try:
        try:
            config = transformers.AutoConfig.from_pretrained(str(source), trust_remote_code=trust)
            changed = shrink(config)
            cls = _model_class(transformers, config, TINY_KINDS[kind], kind, options)
            if hasattr(cls, "_model_mapping"):
                model = cls.from_config(config, trust_remote_code=trust)
            else:
                model = cls._from_config(config)
            with torch.no_grad():  # random weights of a sane size: remote code built with skip_init has none
                for p in model.parameters():
                    if p.is_floating_point():
                        p.normal_(0.0, 0.02)
            params = sum(int(p.numel()) for p in model.parameters())
            model.save_pretrained(str(work), safe_serialization=True)
            for other in source.iterdir():  # remote code and small companion files the saved copy needs too
                if other.is_file() and not (work / other.name).exists():
                    shutil.copy2(other, work / other.name)
            del model
        except Exception as exc:
            steps.append(_step("tiny", "skip", f"could not build it tiny ({type(exc).__name__}: {exc}); not checked"))
            return steps
        steps.append(_step("tiny-build", "ok", f"{cls.__name__} with {params:,} parameters ({', '.join(changed[:6])})"))
        steps.extend(_tiny_load(request, kind, cls, work, trust, torch, transformers))
        return steps
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _tiny_load(
    request: dict[str, Any], kind: str, cls: Any, path: Path, trust: bool, torch: Any, transformers: Any
) -> list[dict[str, str]]:
    from bolo_torch import runtime

    steps: list[dict[str, str]] = []
    options = dict(request.get("options") or {})
    size = sum(f.stat().st_size for f in path.glob("*.safetensors"))
    # the GPU holds a third of it: bolo's scheduler would ask for a split
    ctx = runtime.LoadContext(
        kind, str(request.get("task") or ""), path, options, device="cuda", dtype=torch.bfloat16,
        model_id=str(request.get("model") or kind), max_memory={"0": max(1, size // 3), "cpu": 64 * 1024**3},
    )  # fmt: skip
    import importlib.util

    if importlib.util.find_spec("accelerate") is None:
        steps.append(_step("tiny-split", "skip", "accelerate is not installed: no split to plan"))
        split: dict[str, Any] = {}
    else:
        try:
            split = runtime.split_kwargs(ctx, cls, path, trust_remote_code=trust)
        except runtime.PluginError as exc:
            steps.append(_step("tiny-split", "fail", str(exc)))
            return steps
        except Exception as exc:
            steps.append(_step("tiny-split", "fail", f"{type(exc).__name__}: {exc}"))
            return steps
        device_map = split.get("device_map")
        if isinstance(device_map, dict):
            where = sorted({str(v) for v in device_map.values()})
            steps.append(_step("tiny-split", "ok", f"an explicit device map over {', '.join(where)}"))
        elif device_map == "auto":
            steps.append(_step("tiny-split", "ok", "device_map='auto' (the model names its layers to keep whole)"))
    quantization = str(options.get("quantization") or "").lower()
    if quantization in ("int4", "nf4", "int8"):
        steps.append(_step("tiny-quantized", "skip", f"{quantization} needs bitsandbytes on CUDA; not checked here"))
    kwargs: dict[str, Any] = {"trust_remote_code": trust, "torch_dtype": torch.bfloat16}
    on_gpu = torch.cuda.is_available()
    if isinstance(split.get("device_map"), dict):
        # without a GPU the plan's GPU share is loaded on the CPU: the load still goes through the explicit map
        kwargs["device_map"] = {
            k: (v if on_gpu or v in ("cpu", "disk") else "cpu") for k, v in split["device_map"].items()
        }
    elif split.get("device_map") == "auto" and on_gpu:
        kwargs.update(split)
    try:
        model = cls.from_pretrained(str(path), **kwargs)
        model.eval()
    except Exception as exc:
        steps.append(_step("tiny-load", "fail", f"{type(exc).__name__}: {exc}"))
        return steps
    steps.append(
        _step(
            "tiny-load",
            "ok",
            "from_pretrained in bfloat16" + (" with the device map" if "device_map" in kwargs else ""),
        )
    )
    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 1000)
    ids = torch.randint(0, max(2, min(vocab, 1000)), (1, 8))
    device = next(model.parameters()).device
    # the model itself, else the language model inside it (Qwen-Omni's thinker, MiniCPM-o's llm)
    targets = [("the model", model)] + [
        (name, getattr(model, name)) for name in ("thinker", "llm", "language_model") if hasattr(model, name)
    ]
    try:
        used = targets[0][0]
        for i, (name, target) in enumerate(targets):
            try:
                with torch.no_grad():
                    out = target(input_ids=ids.to(device))
                used = name
                break
            except TypeError:
                if i == len(targets) - 1:
                    raise
        logits = getattr(out, "logits", None)
        shape = tuple(logits.shape) if logits is not None else None
        steps.append(_step("tiny-request", "ok", f"a forward pass of 8 tokens through {used} (logits {shape})"))
    except (TypeError, ValueError, AttributeError, NotImplementedError, KeyError) as exc:
        steps.append(
            _step("tiny-request", "skip", f"its forward needs more than text tokens ({type(exc).__name__}: {exc})")
        )
    except Exception as exc:
        steps.append(_step("tiny-request", "fail", f"{type(exc).__name__}: {exc}"))
    with contextlib.suppress(Exception):
        del model
    return steps


def run(request: dict[str, Any]) -> dict[str, Any]:
    kind, extra = str(request["kind"]), request.get("extra")
    steps = [check_imports(kind, extra)]
    code = check_code(extra)
    if code is not None:
        steps.append(code)
    if request.get("tiny"):
        if steps[0]["status"] != "ok":
            steps.append(_step("tiny", "skip", "the libraries are missing (above)"))
        elif request.get("config_dir"):
            try:
                steps.extend(check_tiny(request))
            except Exception as exc:  # report, never crash: bolo shows the step
                steps.append(_step("tiny", "fail", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"))
        else:
            steps.append(_step("tiny", "skip", "no config.json to build it from"))
    return {"steps": steps}
