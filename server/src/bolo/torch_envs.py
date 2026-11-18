"""The PyTorch builds of the bolo-torch plugin's environments, and switching them to a GPU build in one step.

PyPI's torch is CPU only on Windows and macOS, and a package extra cannot name another index, so a plain
``pip install bolo-torch`` there runs every PyTorch model on the CPU. ``bolo torch use cuda`` (also
``bolo components install torch-cuda``, ``POST /api/v1/components/torch-cuda/install`` and the console's
Components page) does what a person would do by hand, in every environment a bolo-torch plugin runs in (bolo's own,
when the plugin is installed there, and each one named in the ``plugins`` setting):

1. find the GPU and what its driver supports (``nvidia-smi``: "CUDA Version: 12.8") and pick the PyTorch index
   folder for it (``cu128``: the newest CUDA build the driver can run);
2. stop the plugin (its process holds torch's libraries open, which Windows will not let pip replace);
3. reinstall torch, and torchaudio and torchvision where they are installed, from that index: the installed
   version when the index has it, else the newest it has for that environment's Python and platform;
4. check it in that environment (``torch.cuda.is_available()`` and the device's name); the plugin starts again
   on the next load.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PYTORCH_INDEX = "https://download.pytorch.org/whl/"
TARGETS = ("cuda", "rocm", "xpu", "cpu")
#: PyTorch's CUDA index folders and the CUDA version each needs from the driver, newest first
CUDA_INDEXES: tuple[tuple[tuple[int, int], str], ...] = (
    ((13, 0), "cu130"), ((12, 9), "cu129"), ((12, 8), "cu128"), ((12, 6), "cu126"), ((12, 4), "cu124"),
    ((12, 1), "cu121"), ((11, 8), "cu118"),
)  # fmt: skip
#: PyTorch's ROCm index folders (Linux), newest first
ROCM_INDEXES: tuple[tuple[tuple[int, int], str], ...] = (
    ((7, 1), "rocm7.1"), ((7, 0), "rocm7.0"), ((6, 4), "rocm6.4"), ((6, 3), "rocm6.3"),
)  # fmt: skip
#: what a GPU build of torch with its libraries takes to download, about (for the console's confirmation)
DOWNLOAD_BYTES = 2_500_000_000
#: torch versions bolo-torch supports (its requirement ``torch>=2.4``)
MIN_TORCH = (2, 4)
PACKAGES = ("torch", "torchaudio", "torchvision")

#: run in an environment's Python: what it has installed, read from the package metadata (torch is not imported)
PROBE = """
import importlib.metadata as m, json, sys, sysconfig
def v(n):
    try:
        return m.version(n)
    except m.PackageNotFoundError:
        return None
print(json.dumps({"python": sys.executable, "tag": "cp%d%d" % sys.version_info[:2], "platform": sysconfig.get_platform(),
                  "bolo_torch": v("bolo-torch"), "torch": v("torch"), "torchaudio": v("torchaudio"),
                  "torchvision": v("torchvision")}))
"""
#: run in an environment's Python after a reinstall: does torch reach the GPU
VERIFY = """
import json, torch
out = {"version": torch.__version__, "cuda": bool(torch.cuda.is_available()), "hip": getattr(torch.version, "hip", None),
       "xpu": bool(getattr(torch, "xpu", None) and torch.xpu.is_available()), "device": None}
if out["cuda"]:
    out["device"] = torch.cuda.get_device_name(0)
elif out["xpu"]:
    out["device"] = torch.xpu.get_device_name(0)
print(json.dumps(out))
"""

Runner = Callable[[Sequence[str], float], tuple[int, str]]


def run(cmd: Sequence[str], timeout: float = 60.0) -> tuple[int, str]:
    """Run a command; ``(exit code, stdout and stderr)``. A missing program is exit code 127."""
    try:
        proc = subprocess.run(list(cmd), capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return proc.returncode, (proc.stdout + proc.stderr).decode(errors="replace")


# -- the builds ---------------------------------------------------------------------------------------------------
def build_of(version: str | None, platform: str = sys.platform) -> str | None:
    """Which build a torch version is: ``cu128``, ``rocm6.4``, ``xpu``, ``cpu``; ``pypi-cuda`` for PyPI's Linux
    wheel (a CUDA build without a local version), ``cpu`` for PyPI's Windows and macOS wheels. None: no torch."""
    if not version:
        return None
    local = version.partition("+")[2]
    if local:
        return local
    return "pypi-cuda" if platform.startswith("linux") else "cpu"


def build_label(build: str | None, gpu: str | None = None) -> str:
    """For people: "CUDA 12.8 build", "CPU build, GPU not used"."""
    if build is None:
        return "torch not installed"
    if build == "cpu":
        return "CPU build, GPU not used" if gpu else "CPU build"
    if build == "pypi-cuda":
        return "CUDA build (PyPI)"
    m = re.fullmatch(r"cu(\d+)(\d)", build)
    if m:
        return f"CUDA {m[1]}.{m[2]} build"
    if build.startswith("rocm"):
        return f"ROCm {build[4:]} build"
    if build == "xpu":
        return "Intel XPU build"
    return f"{build} build"


def family_of(build: str | None) -> str | None:
    """``cuda``, ``rocm``, ``xpu`` or ``cpu`` (the targets of ``bolo torch use``)."""
    if build is None:
        return None
    if build.startswith("cu") or build == "pypi-cuda":
        return "cuda"
    if build.startswith("rocm"):
        return "rocm"
    return "xpu" if build == "xpu" else "cpu"


# -- the GPU --------------------------------------------------------------------------------------------------------
@dataclass
class Gpu:
    #: "nvidia", "amd" or None (none found)
    vendor: str | None = None
    name: str | None = None
    #: the newest CUDA (NVIDIA) or the installed ROCm (AMD) version, (major, minor)
    version: tuple[int, int] | None = None

    @property
    def target(self) -> str | None:
        return {"nvidia": "cuda", "amd": "rocm"}.get(self.vendor or "")


def _version(text: str) -> tuple[int, int] | None:
    m = re.search(r"(\d+)\.(\d+)", text)
    return (int(m[1]), int(m[2])) if m else None


def detect_gpu(runner: Runner = run, which: Callable[[str], str | None] = shutil.which) -> Gpu:
    """An NVIDIA GPU through ``nvidia-smi`` (its header names the newest CUDA the driver runs), else an AMD GPU
    with ROCm (Linux: ``rocminfo``, the version from ``/opt/rocm/.info/version``)."""
    if which("nvidia-smi"):
        code, out = runner(["nvidia-smi"], 20.0)
        if code == 0:
            m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
            name_code, names = runner(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 20.0)
            name = names.strip().splitlines()[0].strip() if name_code == 0 and names.strip() else None
            return Gpu("nvidia", name, _version(m[1]) if m else None)
    if sys.platform.startswith("linux") and (which("rocminfo") or which("rocm-smi")):
        info = Path("/opt/rocm/.info/version")
        version = _version(info.read_text(encoding="utf-8")) if info.is_file() else None
        return Gpu("amd", None, version)
    return Gpu()


def index_for(target: str, gpu: Gpu | None = None) -> list[str]:
    """The PyTorch index folders that can serve ``target`` on this GPU, the best first: for CUDA those whose CUDA
    the driver runs (all of them when it is not known), newest first."""
    if target == "cuda":
        version = gpu.version if gpu and gpu.vendor == "nvidia" else None
        return [name for need, name in CUDA_INDEXES if version is None or need <= version]
    if target == "rocm":
        version = gpu.version if gpu and gpu.vendor == "amd" else None
        return [name for need, name in ROCM_INDEXES if version is None or need <= version]
    if target in ("xpu", "cpu"):
        return [target]
    raise ValueError(f"unknown target {target!r}: one of {', '.join(TARGETS)}")


# -- environments -------------------------------------------------------------------------------------------------
@dataclass
class TorchEnv:
    """An environment a bolo-torch plugin runs in."""

    #: the plugin's engine id ("plugin:torch"); several when plugins share an environment
    plugins: list[str]
    python: str
    tag: str = ""  # "cp312"
    platform: str = ""  # sysconfig.get_platform(): "win-amd64", "linux-x86_64"
    bolo_torch: str | None = None
    versions: dict[str, str | None] = field(default_factory=dict)

    @property
    def torch(self) -> str | None:
        return self.versions.get("torch")

    @property
    def build(self) -> str | None:
        return build_of(self.torch, "linux" if self.platform.startswith("linux") else self.platform)

    def to_dict(self, gpu: Gpu | None = None) -> dict[str, Any]:
        return {
            "plugins": self.plugins, "python": self.python, "bolo_torch": self.bolo_torch, "torch": self.torch,
            "torchaudio": self.versions.get("torchaudio"), "torchvision": self.versions.get("torchvision"),
            "build": self.build, "family": family_of(self.build),
            "label": build_label(self.build, gpu.name or gpu.vendor if gpu else None),
        }  # fmt: skip


def python_of(argv: Sequence[str]) -> str | None:
    """The Python interpreter a plugin command runs in: the command itself (``.../python -m bolo_torch``), or the
    Python next to a console script (``venv/bin/bolo-torch``, ``venv\\Scripts\\bolo-torch.exe``)."""
    if not argv:
        return None
    exe = argv[0]
    if re.fullmatch(r"(python|pythonw|py)(\d+(\.\d+)?)?(\.exe)?", Path(exe).name.lower()):
        return shutil.which(exe) or exe
    found = shutil.which(exe)
    if not found:
        return None
    folder = Path(found).resolve().parent
    for name in ("python.exe", "python", "python3"):
        if (folder / name).is_file():
            return str(folder / name)
    return None


def plugin_commands(settings_plugins: Sequence[str], entry_points: bool = True) -> dict[str, list[str]]:
    """Every plugin bolo would start: ``engine id -> command`` (the ``plugins`` setting, then installed packages)."""
    from bolo.plugin.host import discover_entry_points, parse_plugin_setting

    found: dict[str, list[str]] = {}
    if entry_points:
        found.update({f"plugin:{n}": argv for n, argv in discover_entry_points().items()})
    for item in settings_plugins:
        try:
            name, argv = parse_plugin_setting(item)
        except ValueError:
            continue
        found[f"plugin:{name}"] = argv
    return found


def discover(commands: dict[str, list[str]], runner: Runner = run) -> list[TorchEnv]:
    """The environments of ``commands`` that have bolo-torch, one per interpreter."""
    envs: dict[str, TorchEnv] = {}
    for plugin, argv in commands.items():
        python = python_of(argv)
        if python is None:
            continue
        key = os.path.normcase(os.path.realpath(python))
        if key in envs:
            envs[key].plugins.append(plugin)
            continue
        code, out = runner([python, "-c", PROBE], 60.0)
        try:
            info = json.loads(out.strip().splitlines()[-1]) if code == 0 and out.strip() else None
        except ValueError:
            info = None
        if not isinstance(info, dict) or not info.get("bolo_torch"):
            continue
        envs[key] = TorchEnv(
            [plugin], python, str(info.get("tag") or ""), str(info.get("platform") or ""), str(info["bolo_torch"]),
            {p: info.get(p) for p in PACKAGES},
        )  # fmt: skip
    return list(envs.values())


# -- the index ----------------------------------------------------------------------------------------------------
def _wheel_platform_ok(wheel_platform: str, platform: str) -> bool:
    if platform.startswith("win"):
        arch = "amd64" if "amd64" in platform else "arm64"
        return wheel_platform == f"win_{arch}"
    if platform.startswith("linux"):
        arch = platform.rsplit("-", 1)[-1]
        return wheel_platform.endswith(arch) and "linux" in wheel_platform
    if platform.startswith("macosx"):
        return wheel_platform.startswith("macosx")
    return False


def versions_on_index(page: str, package: str, index: str, tag: str, platform: str) -> list[str]:
    """The versions of ``package`` a PyTorch index page lists for ``index`` (``cu128``), a Python tag (``cp312``)
    and a platform (``win-amd64``), oldest first."""
    pattern = re.compile(
        rf"{re.escape(package)}-(\d+\.\d+\.\d+(?:\.post\d+)?)(?:\+|%2B){re.escape(index)}-{tag}-{tag}-([A-Za-z0-9_.]+?)\.whl"
    )
    found = {m[1] for m in pattern.finditer(page) if _wheel_platform_ok(m[2], platform)}
    return sorted(found, key=_key)


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", version.partition("+")[0]))


def fetch_page(index: str, package: str) -> str:
    import httpx

    r = httpx.get(f"{PYTORCH_INDEX}{index}/{package}/", timeout=60.0, follow_redirects=True)
    r.raise_for_status()
    return r.text


@dataclass
class Plan:
    """What ``bolo torch use`` does in one environment."""

    env: TorchEnv
    target: str
    index: str | None = None
    #: package -> version (without the local part) to install from the index
    pins: dict[str, str] = field(default_factory=dict)
    #: nothing to do (the environment has that build already), or why it cannot be done
    done: bool = False
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def command(self) -> list[str]:
        if self.index is None or not self.pins:
            return []
        reqs = [f"{p}=={v}+{self.index}" for p, v in self.pins.items()]
        return [
            self.env.python, "-m", "pip", "install", *reqs, "--index-url", f"{PYTORCH_INDEX}{self.index}",
            "--extra-index-url", "https://pypi.org/simple",
        ]  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugins": self.env.plugins, "python": self.env.python, "target": self.target, "index": self.index,
            "pins": self.pins, "done": self.done, "error": self.error, "notes": self.notes,
            "command": " ".join(self.command()),
        }  # fmt: skip


def _pick(available: list[str], series: tuple[int, int], patch: int = 0) -> str | None:
    """The release of ``series`` (major, minor) with torch's patch number (torchaudio 2.9.1 and torchvision 0.24.1
    go with torch 2.9.1), else the newest of the series, else the newest before it (torchaudio stopped at 2.11,
    which goes with any later torch)."""
    same = [v for v in available if _key(v)[:2] == series]
    exact = [v for v in same if len(_key(v)) > 2 and _key(v)[2] == patch]
    if exact:
        return exact[-1]
    if same:
        return same[-1]
    older = [v for v in available if _key(v)[:2] <= series]
    return older[-1] if older else None


def plan(
    env: TorchEnv,
    target: str,
    gpu: Gpu | None = None,
    page: Callable[[str, str], str] = fetch_page,
    latest: bool = False,
) -> Plan:
    """The reinstall that gives ``env`` the ``target`` build. The index folder: the best for this GPU (the newest
    CUDA its driver runs) that has the installed torch's version for this environment's Python and platform, else
    the best that has any torch bolo-torch supports, with its newest (and with ``latest``, always that). Then the
    torchaudio and torchvision of that torch's release, for those installed."""
    out = Plan(env, target)
    if family_of(env.build) == target:
        out.done = True
        out.notes.append(f"already a {build_label(env.build)} (torch {env.torch})")
        return out
    installed = env.torch.partition("+")[0] if env.torch and not latest else None
    offered: dict[str, list[str]] = {}
    for index in index_for(target, gpu):
        try:
            listed = versions_on_index(page(index, "torch"), "torch", index, env.tag, env.platform)
        except Exception as exc:  # an index that cannot be read: say so, try the next
            out.notes.append(f"{index}: {exc}")
            continue
        offered[index] = [v for v in listed if _key(v)[:2] >= MIN_TORCH]
    chosen = next((i for i, vs in offered.items() if installed and installed in vs), None)
    if chosen is None:
        chosen = next((i for i, vs in offered.items() if vs), None)
    if chosen is None:
        out.error = (
            f"no {target} build of torch for Python {env.tag} on {env.platform} on PyTorch's index "
            f"({', '.join(offered) or 'none could be read'})"
        )
        return out
    torch = installed if installed in offered[chosen] else offered[chosen][-1]
    out.index, out.pins = chosen, {"torch": torch}
    if installed and torch != installed:
        out.notes.append(f"torch {installed} has no {target} build for this environment; installing {torch}")
    major, minor, patch = (*_key(torch), 0)[:3]
    series = (major, minor)
    for package, want in (("torchaudio", series), ("torchvision", (0, minor + 15))):
        if not env.versions.get(package):
            continue
        try:
            available = versions_on_index(page(chosen, package), package, chosen, env.tag, env.platform)
        except Exception as exc:
            out.notes.append(f"{chosen}/{package}: {exc}")
            available = []
        version = _pick(available, want, patch)
        if version is None:
            out.notes.append(f"{chosen} has no {package} for torch {torch}; it is left as it is")
            continue
        out.pins[package] = version
    return out


def verify(env: TorchEnv, target: str, runner: Runner = run) -> tuple[bool, str]:
    """Import torch in ``env`` and check that it reaches the GPU: ``(ok, what it says)``."""
    code, out = runner([env.python, "-c", VERIFY], 300.0)
    try:
        info = json.loads(out.strip().splitlines()[-1]) if code == 0 and out.strip() else None
    except ValueError:
        info = None
    if not isinstance(info, dict):
        return False, f"torch could not be imported: {' | '.join(out.strip().splitlines()[-3:])}"
    if target in ("cuda", "rocm"):
        ok = bool(info.get("cuda")) and (target == "cuda") != bool(info.get("hip"))
    elif target == "xpu":
        ok = bool(info.get("xpu"))
    else:
        ok = True
    device = f" on {info['device']}" if info.get("device") else ""
    what = f"torch {info.get('version')}{device}"
    return ok, what if ok else f"{what} does not reach the GPU ({target})"


def doctor_lines(envs: list[TorchEnv], gpu: Gpu, program: str = "bolo") -> list[str]:
    """One line per bolo-torch environment on a CPU build while the machine has a GPU it could use."""
    target = gpu.target
    if target is None:
        return []
    out = []
    for env in envs:
        if family_of(env.build) == "cpu":
            out.append(
                f"{', '.join(env.plugins)} ({env.python}) has torch {env.torch}, a CPU build: its models do not use "
                f"the {gpu.name or gpu.vendor} GPU. Fix: {program} torch use {target}"
            )
    return out


def apply(
    plans: list[Plan],
    runner: Runner = run,
    stop: Callable[[list[str]], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Carry out ``plans``: per environment, stop its plugins (``stop``), run pip, and check the result in that
    environment. Returns one result per plan (``ok``, ``detail``)."""
    results: list[dict[str, Any]] = []
    todo = [p for p in plans if not p.done and p.error is None]
    for p in plans:
        result = {**p.to_dict(), "ok": p.done, "detail": "; ".join(p.notes)}
        if p.error is not None:
            result["ok"], result["detail"] = False, p.error
        elif not p.done:
            step = todo.index(p)
            if progress:
                progress(step / max(1, len(todo)), f"{', '.join(p.env.plugins)}: installing from {p.index}")
            if stop is not None:
                stop(p.env.plugins)  # its process holds torch's libraries open (Windows will not replace them)
            code, output = runner(p.command(), 3600.0)
            if code != 0:
                tail = " | ".join(output.strip().splitlines()[-4:])
                result["ok"], result["detail"] = False, f"pip exited with {code}: {tail}"
            else:
                ok, what = verify(p.env, p.target, runner)
                result["ok"], result["detail"] = ok, what
        log.info("torch in %s: %s", p.env.python, result["detail"])
        results.append(result)
    if progress:
        progress(1.0, "done")
    return results
