"""The servers of full-duplex models that bolo installs and starts: Kyutai's ``moshi`` runtimes and NVIDIA's
PersonaPlex server, all speaking Kyutai's WebSocket protocol (:mod:`bolo.duplex.kyutai`).

Components (``bolo components install <id> [--flavour <build>]``), each in its own folder under
``<data_dir>/components/<id>/<version>/<build>``:

* ``moshi`` (Kyutai; code MIT, the Rust server Apache-2.0 / MIT), three builds:

  - ``pytorch``: the ``moshi`` package 0.2.13 (wheel pinned by SHA-256) with PyTorch 2.9.1 (the CUDA 12.8 build from
    download.pytorch.org) and bitsandbytes 0.49.2 (the q8 checkpoints), in a virtual environment of its own;
    the server is ``python -m moshi.server``. NVIDIA GPU only: Kyutai's server synchronises CUDA while it warms
    up. Linux and Windows; the default build there.
  - ``mlx``: the ``moshi_mlx`` package 0.3.0 (wheel pinned by SHA-256) for Apple silicon, ``python -m
    moshi_mlx.local_web``; the default build on macOS arm64.
  - ``candle``: the Rust server ``moshi-backend`` 0.6.4 (crate pinned by SHA-256), built with ``cargo install``
    (CUDA with ``device = cuda``, Metal on macOS, else the CPU, which is far from real time). Needs a Rust
    toolchain, and for CUDA the CUDA toolkit; ``moshi-backend --config <json> standalone``.

* ``personaplex`` (NVIDIA; code MIT, a fork of Kyutai's moshi package): one build, ``pytorch``: NVIDIA/personaplex at
  a pinned commit (``git clone``, checked) with PyTorch 2.4.1 (CUDA 12.4), ``python -m moshi.server``.

The Python builds need ``uv`` (which fetches the Python they want) or a Python 3.10 to 3.12 running bolo, and
download several GB (PyTorch with CUDA). Model files are bolo's own downloads (catalog, pinned by SHA-256): the
servers are given their paths and run with ``HF_HUB_OFFLINE=1``, so they fetch nothing themselves.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bolo.tts_runtimes import components_dir, platform_key

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.config import Settings
    from bolo.engines.tts_server import ServerProcess
    from bolo.jobs import Job

MOSHI = "moshi"
PERSONAPLEX = "personaplex"
PYTORCH, MLX, CANDLE = "pytorch", "mlx", "candle"


@dataclass(frozen=True)
class Pinned:
    """A file downloaded for an install, checked against its SHA-256."""

    url: str
    sha256: str
    size: int | None = None

    @property
    def name(self) -> str:
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Build:
    """One build of a server: what is installed, and on which platforms it can run."""

    id: str
    version: str
    platforms: tuple[str, ...]
    #: Python builds: the package (a pinned wheel, or a pinned git commit and a subfolder) and pinned requirements
    wheel: Pinned | None = None
    git: tuple[str, str, str] | None = None  # (url, commit, subfolder)
    torch: str | None = None  # "torch==2.9.1"
    torch_index: str | None = None  # PyTorch's own index for the CUDA build
    packages: tuple[str, ...] = ()
    python: str = "3.12"
    #: a server that is not a package: its checkout is kept (``<build folder>/src``) and run from there, after
    #: ``pip install -r`` of this requirements file of the checkout
    requirements: str | None = None
    #: modules the install check imports (from the checkout for ``requirements`` builds)
    imports: tuple[str, ...] = ("moshi", "sphn")
    #: the Rust build: the crate archive and its binary
    crate: Pinned | None = None
    binary: str | None = None
    note: str = ""


@dataclass(frozen=True)
class Server:
    id: str
    name: str
    license: str
    license_url: str
    homepage: str
    builds: dict[str, Build]
    description: str
    #: the build each platform installs by default
    defaults: dict[str, str] = field(default_factory=dict)


_LINUX_WIN = ("linux-x86_64", "windows-x86_64")
SERVERS: dict[str, Server] = {
    MOSHI: Server(
        id=MOSHI,
        name="Kyutai Moshi servers (moshi, moshi_mlx, moshi-backend)",
        license="MIT (Python packages); MIT or Apache-2.0 (Rust server)",
        license_url="https://github.com/kyutai-labs/moshi/blob/main/LICENSE-MIT",
        homepage="https://github.com/kyutai-labs/moshi",
        builds={
            PYTORCH: Build(
                PYTORCH,
                "0.2.13",
                _LINUX_WIN,
                wheel=Pinned(
                    "https://files.pythonhosted.org/packages/21/a9/95a09abaca35ce07e4bcf5b3ea85e41727fd04f8378da5c408e08b8ea76f/moshi-0.2.13-py3-none-any.whl",
                    "beeb3ebc1d769a800e22b26bf4ec0d25f9525d54ae77979b37c86a2f782f4e64",
                    110991,
                ),
                torch="torch==2.9.1",
                torch_index="https://download.pytorch.org/whl/cu128",
                packages=("bitsandbytes==0.49.2",),
                note="NVIDIA GPU (CUDA 12.8 build of PyTorch); bf16 checkpoints want 24 GB of GPU memory, q8 about 12",
            ),
            MLX: Build(
                MLX,
                "0.3.0",
                ("darwin-arm64",),
                wheel=Pinned(
                    "https://files.pythonhosted.org/packages/41/75/3c00e7392a7ea80887440da615c300679636f9e6f89da10e1082f4970a95/moshi_mlx-0.3.0-py3-none-any.whl",
                    "2ec210340df8bf34025df6c272ea3ed618f5c22e176c3513f5dba4e529420621",
                    55777,
                ),
                note="Apple silicon (MLX); q4 checkpoints need about 6 GB of memory",
            ),
            CANDLE: Build(
                CANDLE,
                "0.6.4",
                (*_LINUX_WIN, "linux-aarch64", "darwin-arm64"),
                crate=Pinned(
                    "https://static.crates.io/crates/moshi-backend/moshi-backend-0.6.4.crate",
                    "768a1ab829815bbed8a7f7ac1f28213f8f5481018457950923058bb0bf665f33",
                    45966,
                ),
                binary="moshi-backend",
                note="built from source with cargo (and the CUDA toolkit for device = cuda); Metal on macOS",
            ),
        },
        defaults={"darwin-arm64": MLX, "linux-x86_64": PYTORCH, "windows-x86_64": PYTORCH},
        description=(
            "Kyutai's servers for Moshi and Hibiki (full-duplex speech-to-speech and simultaneous speech translation): "
            "the PyTorch server (NVIDIA GPU; Linux and Windows), the MLX server (Apple silicon) and the Rust/candle "
            "server (CUDA, Metal or CPU; built from source). bolo starts the one a model needs on a loopback port "
            "and bridges it to Realtime and WebRTC sessions."
        ),
    ),
    PERSONAPLEX: Server(
        id=PERSONAPLEX,
        name="NVIDIA PersonaPlex server (moshi-personaplex)",
        license="MIT (code; the PersonaPlex weights are under the NVIDIA Open Model License)",
        license_url="https://github.com/NVIDIA/personaplex/blob/main/LICENSE-MIT",
        homepage="https://github.com/NVIDIA/personaplex",
        builds={
            PYTORCH: Build(
                PYTORCH,
                "16419a2e",
                _LINUX_WIN,
                git=("https://github.com/NVIDIA/personaplex.git", "16419a2ef7b98611f58c9df07477d00e5ad61d5f", "moshi"),
                torch="torch==2.4.1",
                torch_index="https://download.pytorch.org/whl/cu124",
                note="NVIDIA GPU with about 24 GB (bf16 7 B model); --cpu-offload with accelerate for less",
            ),
        },
        defaults={"linux-x86_64": PYTORCH, "windows-x86_64": PYTORCH},
        description=(
            "NVIDIA's server for PersonaPlex, a full-duplex model built on Moshi whose role (a text prompt) and voice "
            "(a voice prompt) are chosen per conversation. bolo installs NVIDIA/personaplex at a pinned commit with "
            "PyTorch and starts it on a loopback port."
        ),
    ),
}
HIBIKI_ZERO = "hibiki-zero"
SERVERS[HIBIKI_ZERO] = Server(
    id=HIBIKI_ZERO,
    name="Kyutai Hibiki-Zero server (hibiki-zero)",
    license="MIT (code; the Hibiki-Zero weights are CC-BY-NC-SA-4.0)",
    license_url="https://github.com/kyutai-labs/hibiki-zero/blob/main/LICENSE",
    homepage="https://github.com/kyutai-labs/hibiki-zero",
    builds={
        PYTORCH: Build(
            PYTORCH,
            "0.0.4",
            _LINUX_WIN,
            git=("https://github.com/kyutai-labs/hibiki-zero.git", "871e89d078202c7d9d18d0924bd76cf161cd6606", "."),
            torch="torch==2.9.1",
            torch_index="https://download.pytorch.org/whl/cu128",
            packages=("moshi==0.2.13",),
            python="3.12",
            note="NVIDIA GPU (the server refuses to start without CUDA); about 8 GB for the 3 B model",
        ),
    },
    defaults={"linux-x86_64": PYTORCH, "windows-x86_64": PYTORCH},
    description=(
        "Kyutai's server for Hibiki-Zero, simultaneous speech translation from French, Spanish, Portuguese and German "
        "into English (speech and text). bolo installs kyutai-labs/hibiki-zero at a pinned commit with PyTorch and "
        "starts it on a loopback port."
    ),
)
FLM_AUDIO = "flm-audio"
SERVERS[FLM_AUDIO] = Server(
    id=FLM_AUDIO,
    name="CofeAI FLM-Audio server (flmaudio.server)",
    license="Apache-2.0 (code; the FLM-Audio weights state no licence and are for research use only)",
    license_url="https://github.com/cofe-ai/flm-audio/blob/main/LICENSE",
    homepage="https://github.com/cofe-ai/flm-audio",
    builds={
        PYTORCH: Build(
            PYTORCH,
            "41d768c9",
            _LINUX_WIN,
            git=("https://github.com/cofe-ai/flm-audio.git", "41d768c9f61d954907c1c518709f2d9239de46a8", "."),
            torch="torch==2.6.0",
            torch_index="https://download.pytorch.org/whl/cu124",
            requirements="requirements-server.txt",
            imports=("flmaudio.models", "sphn"),
            note="NVIDIA GPU with about 24 GB (a 7 B model in bfloat16); its server synchronises CUDA while warming up",
        ),
    },
    defaults={"linux-x86_64": PYTORCH, "windows-x86_64": PYTORCH},
    description=(
        "CofeAI's server for FLM-Audio, a full-duplex English and Chinese conversation model on Mimi that speaks "
        "Kyutai's protocol. bolo clones cofe-ai/flm-audio at a pinned commit, installs its requirements with PyTorch "
        "and runs it from the checkout on a loopback port."
    ),
)
FREEZE_OMNI = "freeze-omni"
SERVERS[FREEZE_OMNI] = Server(
    id=FREEZE_OMNI,
    name="Tencent Freeze-Omni server (bin/server.py, Socket.IO)",
    license="Freeze-Omni License: academic, research and education use only (code and weights)",
    license_url="https://github.com/VITA-MLLM/Freeze-Omni/blob/main/License.txt",
    homepage="https://github.com/VITA-MLLM/Freeze-Omni",
    builds={
        PYTORCH: Build(
            PYTORCH,
            "163a2488",
            _LINUX_WIN,
            git=("https://github.com/VITA-MLLM/Freeze-Omni.git", "163a24880e533b2a07038fb8dcfe02dbbb8457e6", "."),
            torch="torch==2.2.0",
            torch_index="https://download.pytorch.org/whl/cu121",
            python="3.10",
            requirements="requirements.txt",
            imports=("flask_socketio", "torchaudio"),
            note="NVIDIA GPU (its code calls .cuda()); Qwen2-7B-Instruct in bfloat16 wants about 24 GB",
        ),
    },
    defaults={"linux-x86_64": PYTORCH, "windows-x86_64": PYTORCH},
    description=(
        "Tencent's server for Freeze-Omni, a speech-to-speech model on a frozen Qwen2-7B-Instruct that listens while "
        "it speaks and is interrupted by the caller (its own voice-activity detection and state prediction). It speaks "
        "Socket.IO, not Kyutai's protocol (bolo.duplex.freeze_omni). Non-commercial licence."
    ),
)
MINICPM_O = "minicpm-o"
SERVERS[MINICPM_O] = Server(
    id=MINICPM_O,
    name="MiniCPM-o 4.5 duplex server (bolo's, on the model's own code)",
    license="Apache-2.0 (bolo's server and minicpmo-utils; the model's remote code and weights are Apache-2.0)",
    license_url="https://huggingface.co/openbmb/MiniCPM-o-4_5",
    homepage="https://github.com/OpenBMB/MiniCPM-o",
    builds={
        PYTORCH: Build(
            PYTORCH,
            "1.0.6",
            _LINUX_WIN,
            torch="torch==2.8.0 torchaudio==2.8.0",
            torch_index="https://download.pytorch.org/whl/cu128",
            python="3.10",
            packages=(
                "transformers==4.51.0",
                "accelerate==1.10.1",
                "minicpmo-utils[all]==1.0.6",
                "sphn==0.2.1",
                "websockets==15.0.1",
                "soundfile==0.12.1",
            ),
            imports=("transformers", "stepaudio2", "sphn", "websockets"),
            note="NVIDIA GPU with about 24 GB (bfloat16, 9 B); the model card's tested set: Python 3.10, transformers 4.51.0",
        ),
    },
    defaults={"linux-x86_64": PYTORCH, "windows-x86_64": PYTORCH},
    description=(
        "An environment for MiniCPM-o 4.5's full-duplex mode: PyTorch, transformers 4.51.0 and OpenBMB's minicpmo-utils "
        "(token2wav), in which bolo runs a small server of its own (bolo/duplex/servers/minicpmo_duplex.py) that "
        "drives the model's as_duplex() API and speaks Kyutai's protocol. The model's code is remote code: it runs "
        "only after you opt in for the model (allow_remote_code)."
    ),
)
COMPONENTS = tuple(SERVERS)


def build_of(server: str, flavour: str | None, key: str | None = None) -> Build | None:
    info = SERVERS[server]
    key = key or platform_key()
    return info.builds.get(flavour or info.defaults.get(key, "") or "")


def install_dir(settings: Settings, server: str, flavour: str) -> Path:
    build = SERVERS[server].builds[flavour]
    return components_dir(settings) / server / build.version / flavour


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _cargo_binary(root: Path, name: str) -> Path:
    return root / "bin" / (name + (".exe" if os.name == "nt" else ""))


def installed_path(settings: Settings, server: str, flavour: str) -> Path | None:
    """The interpreter (Python builds) or binary (Rust build) of an installed build, else None."""
    build = SERVERS[server].builds.get(flavour)
    if build is None:
        return None
    root = install_dir(settings, server, flavour)
    if not (root / ".bolo-complete").is_file():
        return None
    path = _cargo_binary(root, build.binary) if build.binary else _venv_python(root)
    return path if path.is_file() else None


def installed_flavours(settings: Settings, server: str) -> list[str]:
    return [f for f in SERVERS[server].builds if installed_path(settings, server, f) is not None]


def unsupported_reason(server: str, flavour: str | None = None, key: str | None = None) -> str | None:
    key = key or platform_key()
    info = SERVERS[server]
    build = build_of(server, flavour, key)
    if build is None:
        builds = ", ".join(info.builds)
        if flavour:
            return f"{info.name} has no build '{flavour}' (builds: {builds})"
        return f"{info.name} has no build for {key} (builds: {builds}, each for its own platforms)"
    if key not in build.platforms:
        return f"the {build.id} build of {info.name} runs on {', '.join(build.platforms)}, not on {key}"
    return None


def blocked_reason(server: str, flavour: str | None) -> str | None:
    """Why this machine cannot install the build (missing tools), or None."""
    problem = unsupported_reason(server, flavour)
    if problem:
        return problem
    build = build_of(server, flavour)
    assert build is not None
    if build.crate is not None and shutil.which("cargo") is None:
        return "cargo (the Rust toolchain, https://rustup.rs) is not on PATH; it builds moshi-backend from source"
    if build.git is not None and shutil.which("git") is None:
        return "git is not installed (it fetches the pinned source)"
    if build.wheel is not None or build.git is not None:
        ok = (3, 10) <= sys.version_info[:2] <= (3, 12)
        if shutil.which("uv") is None and not ok:
            return (
                f"the server needs Python 3.10 to 3.12 for its own environment: install uv (it fetches one) or run "
                f"bolo on such a Python (this is {sys.version_info[0]}.{sys.version_info[1]})"
            )
    return None


Run = Callable[[list[str]], tuple[int, str]]


def _check(run: Run, cmd: list[str], what: str) -> None:
    code, output = run(cmd)
    if code != 0:
        tail = " | ".join(output.strip().splitlines()[-4:])
        raise RuntimeError(f"{what} failed ({code}): {tail}")


def install(
    settings: Settings, server: str, flavour: str | None, fetch: Callable[[Pinned, Path], Path], run: Run,
    job: Job | None = None,
) -> str | None:  # fmt: skip
    """Install a build. ``fetch(pinned, folder)`` downloads a file there (checking its SHA-256) and returns its path;
    ``run`` runs a command and returns (exit code, output). Returns an error message, or None."""
    problem = blocked_reason(server, flavour)
    if problem:
        return problem
    build = build_of(server, flavour)
    assert build is not None
    target = install_dir(settings, server, build.id)
    work = target.with_name(target.name + ".download")
    shutil.rmtree(target, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        if build.crate is not None:
            _install_crate(settings, build, target, work, fetch, run)
        else:
            _install_python(build, target, work, fetch, run, job)
        (target / ".bolo-complete").write_text(
            json.dumps({"server": server, "build": build.id, "version": build.version}), encoding="utf-8"
        )
    except Exception as exc:
        shutil.rmtree(target, ignore_errors=True)
        return str(exc) if isinstance(exc, RuntimeError) else f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return None


def _install_python(
    build: Build, target: Path, work: Path, fetch: Callable[[Pinned, Path], Path], run: Run, job: Job | None
) -> None:
    uv = shutil.which("uv")
    if uv:
        _check(run, [uv, "venv", "--quiet", "--python", build.python, str(target)], "creating the environment")
        python = _venv_python(target)
        pip = [uv, "pip", "install", "--python", str(python)]
    else:
        _check(run, [sys.executable, "-m", "venv", str(target)], "creating the environment")
        python = _venv_python(target)
        pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
    if job is not None:
        job.progress = 0.2
    if build.torch:
        cmd = [*pip, *build.torch.split()]
        if build.torch_index and not platform_key().startswith("darwin"):
            cmd += ["--index-url", build.torch_index]
        _check(run, cmd, f"installing {build.torch}")
    if job is not None:
        job.progress = 0.6
    source: Path | None = None
    if build.wheel is not None:
        install = [str(fetch(build.wheel, work))]
    elif build.git is None:
        install = []  # only the pinned packages
    else:
        url, commit, sub = build.git
        checkout = target / "src" if build.requirements else work / "src"
        _check(run, ["git", "clone", "--quiet", url, str(checkout)], f"git clone {url}")
        _check(run, ["git", "-C", str(checkout), "checkout", "--quiet", commit], f"git checkout {commit}")
        code, head = run(["git", "-C", str(checkout), "rev-parse", "HEAD"])
        if code != 0 or head.strip() != commit:
            raise RuntimeError(f"{url} is not at the pinned commit {commit} (got {head.strip()[:40]})")
        if build.requirements:
            source = checkout / sub
            install = ["-r", str(source / build.requirements)]
        else:
            install = [str(checkout / sub)]
    _check(run, [*pip, *install, *build.packages], "installing the server package")
    names = ", ".join(build.imports)
    prelude = f"import sys; sys.path.insert(0, {str(source)!r}); " if source is not None else ""
    _check(run, [str(python), "-c", f"{prelude}import {names}"], f"checking the install (import {names})")


def _install_crate(
    settings: Settings, build: Build, target: Path, work: Path, fetch: Callable[[Pinned, Path], Path], run: Run
) -> None:
    assert build.crate is not None
    archive = fetch(build.crate, work)
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.startswith("/") or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                raise RuntimeError(f"unsafe member in the crate archive: {m.name}")
        tf.extractall(work / "src")  # members checked above
    source = next((work / "src").iterdir())
    features: list[str] = []
    kind = str(settings.device or "").split(":", 1)[0].lower()
    if platform_key() == "darwin-arm64":
        features = ["--features", "metal"]
    elif kind in ("cuda", "auto") and shutil.which("nvcc"):
        features = ["--features", "cuda"]
    elif kind == "cuda":
        raise RuntimeError("device = cuda, but nvcc (the CUDA toolkit) is not on PATH: candle's CUDA build needs it")
    _check(
        run, ["cargo", "install", "--locked", "--path", str(source), "--root", str(target), *features],
        "cargo install moshi-backend",
    )  # fmt: skip


def status(settings: Settings, server: str) -> dict[str, Any]:
    """What the components page shows for one server."""
    info = SERVERS[server]
    builds = installed_flavours(settings, server)
    wanted = info.defaults.get(platform_key())
    path = next((installed_path(settings, server, f) for f in [wanted, *builds] if f in builds), None)
    return {
        "path": str(path) if path else None, "flavours": builds, "wanted_flavour": wanted,
        "builds": {
            b.id: {"version": b.version, "platforms": list(b.platforms), "note": b.note,
                   "available": unsupported_reason(server, b.id) is None}
            for b in info.builds.values()
        },
    }  # fmt: skip


# -- starting a server for a model --------------------------------------------------------------------------------
#: Moshi 7B's language-model configuration, for checkpoints published without a config.json (the moshiko- and
#: moshika-pytorch-bf16 repositories); the values of moshi.models.loaders._lm_kwargs in moshi 0.2.13 (Copyright (c)
#: Kyutai, MIT), as Kyutai's own q8 repositories write them in their config.json
MOSHI_7B_CONFIG: dict[str, Any] = {
    "dim": 4096, "text_card": 32000, "existing_text_padding_id": 3, "n_q": 16, "dep_q": 8, "card": 2048,
    "num_heads": 32, "num_layers": 32, "hidden_scale": 4.125, "causal": True, "layer_scale": None, "context": 3000,
    "max_period": 10000, "gating": "silu", "norm": "rms_norm_f32", "positional_embedding": "rope",
    "depformer_dim": 1024, "depformer_dim_feedforward": 4224, "depformer_num_heads": 16, "depformer_num_layers": 6,
    "depformer_causal": True, "depformer_layer_scale": None, "depformer_multi_linear": True, "depformer_context": 8,
    "depformer_max_period": 10000, "depformer_gating": "silu", "depformer_pos_emb": "none",
    "depformer_weights_per_step": True, "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}  # fmt: skip


#: Kyutai's and NVIDIA's Python servers are started through this small script (written next to the run
#: configuration and run by the server's own Python), which changes three things without editing their code:
#: aiohttp binds to the address bolo gives (PersonaPlex's server binds every interface), Hugging Face downloads of
#: the files bolo already has are answered from disk (PersonaPlex fetches config.json at every start), and
#: torch.cuda.synchronize does nothing when the device is the CPU (Kyutai's warm-up calls it unconditionally).
LAUNCHER = """\
# Started by bolo (bolo.duplex.runtime): run a full-duplex model server with a fixed bind address and local files.
import json, runpy, sys

spec = json.loads(sys.argv[1])
sys.argv = [spec["module"], *spec["args"]]
if spec.get("cwd"):
    import os

    os.chdir(spec["cwd"])
    sys.path.insert(0, spec["cwd"])
try:
    import aiohttp.web

    _run_app = aiohttp.web.run_app

    def run_app(app, *args, **kwargs):
        kwargs["host"] = spec["host"]
        kwargs["port"] = spec["port"]
        return _run_app(app, *args, **kwargs)

    aiohttp.web.run_app = run_app
except ImportError:
    pass
try:
    import huggingface_hub

    _download = huggingface_hub.hf_hub_download

    def hf_hub_download(repo_id, filename, *args, **kwargs):
        local = spec["files"].get(filename)
        return local if local else _download(repo_id, filename, *args, **kwargs)

    huggingface_hub.hf_hub_download = hf_hub_download
except ImportError:
    pass
if spec.get("cpu"):
    import torch

    torch.cuda.synchronize = lambda *a, **k: None
if spec["module"].endswith(".py"):
    runpy.run_path(spec["module"], run_name="__main__")
else:
    runpy.run_module(spec["module"], run_name="__main__", alter_sys=True)
"""


@dataclass
class Launch:
    """How to start the server of one model."""

    process: ServerProcess
    #: conversations it can hold at once
    max_sessions: int
    #: the server keeps the model's state between connections (MLX): bolo stops it after each conversation
    restart_after_session: bool
    runtime: str


def server_of(spec: ModelSpec) -> tuple[str, str]:
    """``(server, build)`` a catalog entry runs on (options ``server`` and ``build``)."""
    server = str(spec.options.get("server") or MOSHI)
    if server not in SERVERS:
        raise ValueError(f"{spec.id}: unknown full-duplex server {server!r}")
    return server, str(spec.options.get("build") or PYTORCH)


def missing_reason(settings: Settings, spec: ModelSpec) -> str | None:
    """Why the server of ``spec`` cannot start here (not installed, wrong platform), or None."""
    server, build = server_of(spec)
    problem = unsupported_reason(server, build)
    if problem:
        return problem
    if installed_path(settings, server, build) is None:
        flag = "" if SERVERS[server].defaults.get(platform_key()) == build else f" --flavour {build}"
        return f"the {build} build of {SERVERS[server].name} is not installed (bolo components install {server}{flag})"
    return None


def _torch_device(settings: Settings, spec: ModelSpec, server: str) -> str:
    device = str(settings.device or "auto").strip().lower()
    if device in ("auto", "cuda", ""):
        return "cuda"
    if device.startswith("cuda:"):
        return device
    if device == "cpu" and server == MOSHI:
        return "cpu"  # far slower than real time for the 7 B models; Hibiki 1 B is the one that may keep up
    raise ValueError(
        f"model '{spec.id}' runs on a PyTorch server that needs an NVIDIA GPU (device = cuda or auto; this server has "
        f"device = {device!r})"
    )


def _voices(archive: Path, run_dir: Path) -> Path:
    """PersonaPlex's voice prompts (``voices.tgz`` of the model, a ``voices/`` folder of ``.pt`` and ``.wav`` files),
    unpacked once next to the run configuration: plain files only, no paths that leave the folder."""
    target = run_dir / "voices"
    if (target / ".bolo-complete").is_file():
        return target
    if not archive.is_file():
        raise FileNotFoundError(f"the voice prompts {archive} are missing; pull the model again")
    shutil.rmtree(target, ignore_errors=True)
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf.getmembers():
            parts = Path(m.name).parts
            if m.isdir() or not m.isfile() or not parts or parts[0] != "voices" or ".." in parts or len(parts) != 2:
                continue
            source = tf.extractfile(m)
            if source is None:
                continue
            target.mkdir(parents=True, exist_ok=True)
            with source, (target / parts[1]).open("wb") as out:
                shutil.copyfileobj(source, out)
    if not target.is_dir():
        raise FileNotFoundError(f"{archive} holds no voices/ folder")
    (target / ".bolo-complete").write_text("", encoding="utf-8")
    return target


def _placeholder_static(run_dir: Path) -> Path:
    """An almost empty web folder: the servers otherwise download (or need) Kyutai's web client."""
    static = run_dir / "static"
    static.mkdir(exist_ok=True)
    (static / "index.html").write_text("<!doctype html><title>bolo</title>served by bolo\n", encoding="utf-8")
    return static


def launch(settings: Settings, spec: ModelSpec, path: Path, port: int, extra_args: list[str]) -> Launch:
    """The server process (not started) that serves ``spec`` from its files in ``path``."""
    from bolo.engines.tts_server import ServerProcess

    server, build = server_of(spec)
    exe = installed_path(settings, server, build)
    if exe is None:
        raise FileNotFoundError(missing_reason(settings, spec) or "the server is not installed")
    opts = spec.options
    run_dir = Path(settings.data_dir) / "run" / "duplex" / spec.id
    run_dir.mkdir(parents=True, exist_ok=True)
    files = {k: str(path / str(opts[k])) for k in ("model_file", "mimi_file", "tokenizer_file") if opts.get(k)}
    needed = (
        ("model_file",)
        if server in (FLM_AUDIO, FREEZE_OMNI, MINICPM_O)
        else ("model_file", "mimi_file", "tokenizer_file")
    )
    for key in needed:
        if key not in files or not Path(files[key]).is_file():
            raise FileNotFoundError(f"model '{spec.id}': its {key} ({files.get(key)}) is missing; pull the model again")
    files.setdefault("mimi_file", "")
    files.setdefault("tokenizer_file", "")
    repo = str(opts.get("repo") or "")
    env = {"HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_TELEMETRY": "1"}
    extra = [str(a) for a in opts.get("server_args") or []] + [str(a) for a in extra_args]
    weights = [
        "--moshi-weight",
        files["model_file"],
        "--mimi-weight",
        files["mimi_file"],
        "--tokenizer",
        files["tokenizer_file"],
    ]
    tls, sessions, restart = False, 1, False
    gpu = str(settings.device or "").lower() != "cpu"
    if build == CANDLE:
        config = {
            "instance_name": "bolo", "hf_repo": repo, "lm_model_file": files["model_file"],
            "text_tokenizer_file": files["tokenizer_file"], "mimi_model_file": files["mimi_file"],
            "mimi_num_codebooks": int(opts.get("mimi_codebooks") or 8), "log_dir": str(run_dir / "logs"),
            "static_dir": str(_placeholder_static(run_dir)), "addr": "127.0.0.1", "port": port, "cert_dir": str(run_dir),
        }  # fmt: skip
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "candle.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        argv = [
            str(exe),
            "--config",
            str(run_dir / "candle.json"),
            "standalone",
            *(["--cpu"] if not gpu else []),
            *extra,
        ]
        tls = True  # the Rust server serves https only (with a certificate it makes in cert_dir)
        sessions = max(1, int(getattr(settings, "duplex_max_sessions", 1)))
    else:
        local: dict[str, str] = {}
        if build == MLX:
            module = "moshi_mlx.local_web"
            args = ["--static", "none", "--no-browser", "--hf-repo", repo or "kyutai/moshiko-mlx-q4", *weights,
                    "--steps", str(int(opts.get("max_steps") or 7500))]  # fmt: skip
            if opts.get("config_file"):
                args += ["--lm-config", str(path / str(opts["config_file"]))]
            if opts.get("quantized"):
                args += ["-q", str(int(opts["quantized"]))]
            restart, gpu = True, True  # the MLX server keeps the model's state from one connection to the next
        elif server == FLM_AUDIO:
            module = "flmaudio.server"
            device = _torch_device(settings, spec, server)
            # its loaders take folders: the model's own (config.json, the weights, the tokenizer, Mimi's file)
            args = ["--device", device, "--model-path", str(path), "--mimi-path", str(path)]
        elif server == MINICPM_O:
            module = str(Path(__file__).with_name("servers") / "minicpmo_duplex.py")
            device = _torch_device(settings, spec, server)
            args = ["--device", device, "--model-path", str(path)]
            for name, rel in (opts.get("voice_files") or {}).items():
                args += ["--voice", f"{name}={rel}"]
        elif server == FREEZE_OMNI:
            module = "bin/server.py"
            sessions = max(1, int(getattr(settings, "duplex_max_sessions", 1)))
            args = ["--model_path", str(path / "checkpoints"), "--llm_path", str(path / "Qwen2-7B-Instruct"),
                    "--max_users", str(sessions), "--llm_exec_nums", str(sessions), "--timeout", "3600",
                    "--top_p", "0.8", "--top_k", "20", "--temperature", "0.8"]  # fmt: skip
            tls = True  # its server makes a self-signed certificate and serves https only
        elif server == HIBIKI_ZERO:
            module = "hibiki_zero.run"
            device = _torch_device(settings, spec, server)
            args = ["serve", "--static", str(_placeholder_static(run_dir)), "--hf-repo", repo, "--device", device,
                    "--model-weight", files["model_file"], "--mimi-weight", files["mimi_file"],
                    "--tokenizer", files["tokenizer_file"], "--config-path", str(path / str(opts["config_file"]))]  # fmt: skip
        else:
            module = "moshi.server"
            device = _torch_device(settings, spec, server)
            args = [
                "--static",
                "none",
                "--device",
                device,
                "--hf-repo",
                repo or "kyutai/moshiko-pytorch-bf16",
                *weights,
            ]
            if server == PERSONAPLEX:
                args += [
                    "--voice-prompt-dir",
                    str(_voices(path / str(opts.get("voices_archive") or "voices.tgz"), run_dir)),
                ]
                # its server asks the Hub for config.json at every start, only to count the download
                stub = run_dir / "hub-config.json"
                stub.write_text("{}\n", encoding="utf-8")
                local["config.json"] = str(stub)
            else:
                config_path = path / str(opts["config_file"]) if opts.get("config_file") else run_dir / "config.json"
                if not opts.get("config_file"):
                    names = {"moshi_name": Path(files["model_file"]).name, "mimi_name": Path(files["mimi_file"]).name,
                             "tokenizer_name": Path(files["tokenizer_file"]).name}  # fmt: skip
                    config_path.write_text(json.dumps({**names, **MOSHI_7B_CONFIG}, indent=2), encoding="utf-8")
                args += ["--config-path", str(config_path)]
                if opts.get("cfg_coef"):
                    args += ["--cfg-coef", str(opts["cfg_coef"])]
            gpu = device != "cpu"
            if opts.get("config_file"):
                local["config.json"] = str(path / str(opts["config_file"]))
        local.update({Path(v).name: v for v in files.values() if v})
        script = run_dir / "bolo_launch.py"
        script.write_text(LAUNCHER, encoding="utf-8")
        flag = "--ip" if server == FREEZE_OMNI else "--host"
        host = [flag, "127.0.0.1", "--port", str(port)]
        # (hibiki_zero.run takes its options after the command name, serve)
        full = [*args[:1], *host, *args[1:], *extra] if module == "hibiki_zero.run" else [*host, *args, *extra]
        launch_spec: dict[str, Any] = {
            "module": module, "args": full, "host": "127.0.0.1", "port": port, "files": local, "cpu": not gpu,
        }  # fmt: skip
        build_info = SERVERS[server].builds[build]
        if build_info.requirements:  # run from the kept checkout
            launch_spec["cwd"] = str(
                install_dir(settings, server, build) / "src" / (build_info.git or ("", "", "."))[2]
            )
        argv = [str(exe), "-I", "-u", str(script), json.dumps(launch_spec)]
    process = ServerProcess(
        f"duplex:{spec.id}", argv, port, start_timeout_s=float(getattr(settings, "duplex_start_timeout_s", 900.0)),
        env=env, health_path="/", gpu=gpu and build != MLX, flavour=build, tls=tls, ready_any_status=True,
    )  # fmt: skip
    return Launch(process, sessions, restart, f"{server}:{build}")


def default_fetch(store: Any) -> Callable[[Pinned, Path], Path]:
    """Downloads through bolo's store (resumable, checked against the SHA-256), copied into ``folder`` under the
    file's own name (pip reads a wheel's name)."""
    from bolo.store import RemoteFile

    def fetch(pinned: Pinned, folder: Path) -> Path:
        blob, _ = store.fetch_blob(RemoteFile(pinned.name, pinned.url, pinned.size, pinned.sha256))
        out = folder / pinned.name
        shutil.copyfile(blob, out)
        return out

    return fetch


def run_command(cmd: list[str]) -> tuple[int, str]:
    """Run an install step (PyTorch with CUDA is a large download: up to an hour)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=3600, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    return proc.returncode, (proc.stdout + proc.stderr).decode(errors="replace")


def remove(settings: Settings, server: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(components_dir(settings) / server)
