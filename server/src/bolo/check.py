"""``bolo check <model>``: whether a model would run, found out without downloading its weights.

Today's surprises (a 4-bit model that failed on meta tensors, a component whose environment could not import its
own code, a model that could not be split, a q8 checkpoint with the wrong dtype) all came after many GB of downloads.
This finds what it can first.

Level 1 (always):

* ``catalog``: the entry can be pulled and has a runtime;
* ``runtime``: its engine is available, its components are installed (a full-duplex server's environment imports its
  own code; ``install=True`` installs a missing one), and for a bolo-torch model the plugin's environment imports
  the loader and the libraries of its extra, with its inference-code bundle matching its pins;
* ``weights``: every file of the model, asked for (``HEAD``) at its pinned revision or URL: there (200) or gated
  (401/403, for a gated entry), with its size as the catalog says;
* ``placement``: what the advisor (``bolo recommend``) makes of it on this machine: on the GPU, split across the GPU
  and the CPU, on the CPU, or refused (a full-duplex model that does not fit the GPU).

Level 2 (``tiny=True``): a transformers-based bolo-torch model is built from its ``config.json`` (and, after the
remote-code opt-in, its reviewed code) with tiny dimensions and random weights, and goes through the plugin's own
split planning, a bfloat16 load and one forward pass (``bolo_torch.check``). Models that cannot be shrunk that way
(the full-duplex servers' models, non-transformers models) say so and are skipped, never passed.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from bolo import torch_envs
from bolo.errors import BoloError

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.core import Bolo

OK, FAIL, SKIP, WARN = "ok", "fail", "skip", "warn"
#: kinds of bolo-torch models ``--tiny`` builds (bolo_torch.check.TINY_KINDS)
TINY_KINDS = ("orpheus-snac", "step-audio-2", "glm-4-voice", "minicpm-o", "qwen-omni")
#: the files the tiny build fetches: those at the root of the model's files up to this size (configuration, code and
#: small companions such as Qwen-Omni's speaker table), never the weights
TINY_FILE_MAX = 5_000_000


def step(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def supports_tiny(spec: ModelSpec) -> bool:
    return spec.engine == "plugin:torch" and spec.kind in TINY_KINDS


def _extra(spec: ModelSpec) -> str | None:
    m = re.search(r"bolo-torch\[([^\]]+)\]", str(spec.options.get("plugin_install") or ""))
    return m.group(1).split(",")[0] if m else None


# -- weights --------------------------------------------------------------------------------------------------------
def check_weights(
    core: Bolo, spec: ModelSpec, head: Callable[[str, dict[str, str]], httpx.Response] | None = None
) -> list[dict[str, str]]:
    """``HEAD`` each file at its pinned location: there or gated, and the size the listing or catalog gives."""
    if spec.builtin or spec.cloud or spec.source is None:
        return [step("weights", SKIP, "no files to download")]
    if spec.source.local:
        return [step("weights", SKIP, f"a folder on this machine ({spec.source.local})")]
    try:
        files = core.store.plan(spec.source)
    except BoloError as exc:
        status = OK if spec.gated and exc.code == "model_gated" else FAIL
        return [step("weights", status, f"the file listing: {exc.message}")]
    except httpx.HTTPError as exc:
        return [step("weights", FAIL, f"the file listing could not be read: {exc}")]
    hub = core.store.hub
    do_head = head or (lambda url, headers: httpx.head(url, headers=headers, follow_redirects=True, timeout=30.0))
    problems: list[str] = []
    gated = 0
    total = 0
    for f in files:
        headers = hub.headers() if httpx.URL(f.url).host == httpx.URL(hub.endpoint).host else {}
        try:
            r = do_head(f.url, headers)
        except httpx.HTTPError as exc:
            problems.append(f"{f.path}: {type(exc).__name__}: {exc}")
            continue
        if r.status_code in (401, 403):
            gated += 1
            if not spec.gated:
                problems.append(f"{f.path}: HTTP {r.status_code}, and the catalog does not say it is gated")
            continue
        if r.status_code >= 400:
            problems.append(f"{f.path}: HTTP {r.status_code}")
            continue
        size = r.headers.get("x-linked-size") or r.headers.get("content-length")
        if f.size is not None and size is not None and size.isdigit() and int(size) != f.size:
            problems.append(f"{f.path}: {int(size):,} bytes there, {f.size:,} expected")
        total += f.size or (int(size) if size and size.isdigit() else 0)
    if problems:
        return [step("weights", FAIL, f"{len(problems)} of {len(files)} files: " + "; ".join(problems[:5]))]
    detail = f"{len(files)} files there at their pinned revision ({total / 1e9:.2f} GB)"
    if gated:
        detail += f"; {gated} gated (accept the terms and set BOLO_HF_TOKEN to download)"
    out = [step("weights", OK, detail)]
    if spec.size_bytes and total and abs(total - spec.size_bytes) > 0.02 * spec.size_bytes and not gated:
        out.append(step("size", WARN, f"the files add up to {total:,} bytes, the catalog says {spec.size_bytes:,}"))
    return out


def fetch_tiny_files(core: Bolo, spec: ModelSpec, folder: Path) -> list[str]:
    """The model's small root files (configuration, code, small companions; never weights) into ``folder``."""
    assert spec.source is not None
    fetched = []
    for f in core.store.plan(spec.source):
        if "/" in f.path or f.size is None or f.size > TINY_FILE_MAX:
            continue
        r = httpx.get(f.url, headers=core.store.hub.headers(), follow_redirects=True, timeout=60.0)
        r.raise_for_status()
        (folder / f.path).write_bytes(r.content)
        fetched.append(f.path)
    return fetched


# -- the runtime ------------------------------------------------------------------------------------------------------
def check_plugin(
    core: Bolo, spec: ModelSpec, tiny: bool, allow_remote_code: bool, runner: torch_envs.Runner
) -> list[dict[str, str]]:
    s = core.settings
    commands = torch_envs.plugin_commands(s.plugins, s.plugin_entry_points)
    argv = commands.get(spec.engine)
    if argv is None:
        hint = str(spec.options.get("plugin_install") or "pip install bolo-torch")
        return [step("runtime", FAIL, f"the {spec.engine} plugin is not installed: {hint}")]
    python = torch_envs.python_of(argv)
    if python is None:
        return [step("runtime", SKIP, f"the {spec.engine} plugin's command is not a Python one; not checked")]
    request: dict[str, Any] = {
        "kind": spec.kind, "extra": _extra(spec), "model": spec.id, "task": spec.task,
        # the revision the files are fetched at: remote code runs only at the reviewed commit
        "options": {
            **{k: v for k, v in spec.options.items() if _jsonable(v)},
            **({"revision": spec.source.revision} if spec.source is not None and spec.source.repo else {}),
        },
        "allow_remote_code": allow_remote_code or bool(core.store.model_settings(spec.id).get("allow_remote_code")),
    }  # fmt: skip
    steps = [step("runtime", OK, f"{spec.engine} in {python}")]
    work = None
    try:
        if tiny and supports_tiny(spec):
            work = Path(tempfile.mkdtemp(prefix="bolo-check-"))
            try:
                fetched = fetch_tiny_files(core, spec, work)
                request.update(tiny=True, config_dir=str(work))
                if "config.json" not in fetched:
                    request.pop("config_dir")
            except (BoloError, httpx.HTTPError) as exc:
                steps.append(step("tiny", FAIL, f"its configuration could not be fetched: {exc}"))
        elif tiny:
            steps.append(step("tiny", SKIP, f"the kind {spec.kind!r} is not a transformers model bolo can shrink"))
        # (``python -m bolo_torch`` serves the plugin protocol: the CLI is called by name)
        entry = "import sys; from bolo_torch.cli import main; sys.exit(main(['check', sys.argv[1]]))"
        code, out = runner([python, "-c", entry, json.dumps(request)], 1800.0)
        report = _report(out)
        if code != 0 or not isinstance(report, dict):
            tail = " | ".join(out.strip().splitlines()[-3:])
            steps.append(step("plugin", FAIL, f"`bolo-torch check` gave no report (exit {code}): {tail}"))
        else:
            steps.extend(step(str(x["name"]), str(x["status"]), str(x["detail"])) for x in report.get("steps", []))
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
    return steps


def _report(output: str) -> dict[str, Any] | None:
    """The JSON report among a command's output (libraries log to the same stream)."""
    for line in reversed(output.strip().splitlines()):
        if line.startswith('{"steps"'):
            try:
                report = json.loads(line)
            except ValueError:
                return None
            return report if isinstance(report, dict) else None
    return None


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def check_duplex(core: Bolo, spec: ModelSpec, install: bool, runner: torch_envs.Runner) -> list[dict[str, str]]:
    from bolo.duplex import runtime as duplex_runtime

    server, build = duplex_runtime.server_of(spec)
    problem = duplex_runtime.missing_reason(core.settings, spec)
    if problem and install and duplex_runtime.unsupported_reason(server, build) is None:
        error = core.components.install_blocking(component=server, flavour=build)
        problem = error and f"installing it failed: {error}"
    if problem:
        return [step("runtime", FAIL, problem)]
    python = duplex_runtime.installed_path(core.settings, server, build)
    info = duplex_runtime.SERVERS[server].builds[build]
    steps = [step("runtime", OK, f"the {build} build of {server} is installed")]
    if info.crate is not None or python is None:
        return steps
    root = duplex_runtime.install_dir(core.settings, server, build)
    prelude = ""
    if info.requirements and info.git is not None:
        prelude = f"import sys; sys.path.insert(0, {str(root / 'src' / info.git[2])!r}); "
    names = ", ".join(info.imports)
    code, out = runner([str(python), "-c", f"{prelude}import {names}"], 600.0)
    if code != 0:
        tail = " | ".join(out.strip().splitlines()[-3:])
        steps.append(step("imports", FAIL, f"its environment cannot import {names}: {tail}"))
    else:
        steps.append(step("imports", OK, f"its environment imports {names}"))
    state = duplex_runtime.compile_state(core.settings, server, str(core.settings.device or "").lower() != "cpu")
    if state is not None and not state["speed_up"]:
        steps.append(step("speed-up", WARN, f"torch.compile will be off: {state['reason']}"))
    return steps


def check_runtime(
    core: Bolo, spec: ModelSpec, tiny: bool, install: bool, allow_remote_code: bool, runner: torch_envs.Runner
) -> list[dict[str, str]]:
    if spec.cloud:
        try:
            core._require_provider(spec)
        except BoloError as exc:
            return [step("runtime", FAIL, exc.message)]
        return [step("runtime", OK, "a cloud provider answers it")]
    if spec.engine.startswith("plugin:"):
        return check_plugin(core, spec, tiny, allow_remote_code, runner)
    steps: list[dict[str, str]] = []
    if spec.engine == "duplex":
        steps = check_duplex(core, spec, install, runner)
    else:
        status = core.engines.status(spec.engine)
        steps.append(step("runtime", OK if status.available else FAIL,
                          f"engine {spec.engine} {status.version or ''}".strip() if status.available else
                          f"the {spec.engine} engine is not available: {status.reason}"))  # fmt: skip
    missing = None if spec.engine == "duplex" else core.components.missing_reason(spec)  # said above for duplex
    if missing:
        steps.append(step("components", FAIL, missing))
    if tiny:
        why = (
            "a full-duplex server's model runs in that server's own code, which bolo cannot build small"
            if spec.engine == "duplex"
            else "only transformers-based bolo-torch models are built tiny"
        )
        steps.append(step("tiny", SKIP, why))
    return steps


# -- placement ------------------------------------------------------------------------------------------------------
def check_placement(core: Bolo, spec: ModelSpec) -> dict[str, str]:
    from bolo.hardware import gigabytes
    from bolo.placement import split_mode

    fit = core.advisor.fit_of(spec)
    need, have = fit.memory_needed_bytes, fit.memory_available_bytes
    memory = f" (needs about {gigabytes(need)} of {gigabytes(have)})" if need and have else ""
    if fit.gpu_too_small:
        return step("placement", FAIL, f"refused here: {fit.reason}")
    if fit.level == "unavailable":
        return step("placement", FAIL, fit.reason)
    if fit.device == "gpu" and need and have and need > have:
        mode = split_mode(spec.engine, spec.kind, spec.options)
        verdict = (
            f"split across the GPU and the CPU ({mode})" if mode != "whole" else "on the CPU (does not fit the GPU)"
        )
    elif fit.device == "gpu":
        verdict = "fits on the GPU"
    elif fit.device == "cloud":
        verdict = "at the provider"
    else:
        verdict = "on the CPU"
    status = WARN if fit.level in ("slow", "unlikely") else OK
    label = fit.to_dict()["label"]
    return step("placement", status, f"{verdict}{memory}: {label}; {fit.reason}".rstrip("; "))


# -- all of it -------------------------------------------------------------------------------------------------------
def run_check(
    core: Bolo,
    spec: ModelSpec,
    *,
    tiny: bool = False,
    install: bool = False,
    allow_remote_code: bool = False,
    runner: torch_envs.Runner = torch_envs.run,
    head: Callable[[str, dict[str, str]], httpx.Response] | None = None,
) -> dict[str, Any]:
    """The report of ``bolo check``: ``{"model", "ok", "steps": [{"name", "status", "detail"}]}``; ``ok`` is false
    when any step failed."""
    steps: list[dict[str, str]] = []
    if spec.status == "pending":
        steps.append(step("catalog", FAIL, "its files are not published yet"))
    elif not spec.runnable:
        steps.append(step("catalog", FAIL, f"no runtime bolo has can run it: {spec.not_runnable_reason}"))
    else:
        steps.append(step("catalog", OK, f"{spec.task}, engine {spec.engine}, kind {spec.kind}"))
    parts: list[Callable[[], list[dict[str, str]]]] = [
        lambda: check_runtime(core, spec, tiny, install, allow_remote_code, runner),
        lambda: check_weights(core, spec, head),
        lambda: [check_placement(core, spec)],
    ]
    for part in parts:
        try:
            steps.extend(part())
        except Exception as exc:  # a check never crashes: the failure is the finding
            steps.append(step("check", FAIL, f"{type(exc).__name__}: {exc}"))
    return {"model": spec.id, "ok": all(s["status"] != FAIL for s in steps), "tiny": tiny, "steps": steps}
