"""External speech-synthesis servers that the ``tts-server`` engine forwards to.

Three runtimes are known:

* **llama-server** from llama.cpp (MIT, https://github.com/ggml-org/llama.cpp): ``POST /tts`` with
  ``input``, ``lang``, ``voice``, ``speaker_ref_b64``, ``response_format`` (``wav`` | ``pcm`` | ``pcm16``) and
  ``stream``; ``GET /props`` lists a model's predefined ``voices``; ``GET /health`` is 503 while the model loads.
  One model per process, started as ``llama-server -m <model.gguf> -mm <mmproj.gguf>``.
* **audiocpp_server** from audio.cpp (Apache-2.0, https://github.com/0xShug0/audio.cpp): ``POST
  /v1/audio/speech`` with ``model`` (the configured id), ``input``, ``language``, ``voice``, ``voice_ref``
  (``{"type": "base64", "data": …}``), ``reference_text``, ``speed`` and ``options``; WAV by default, raw PCM16
  with ``stream_format: "audio"`` for models configured with ``mode: "streaming"``. Models are declared in a
  JSON config (``--config``).
* **tontaube** from craitech/tontaube (Apache-2.0, https://github.com/craitech/tontaube): ``POST /predict`` with
  ``text``, ``language``, ``tag`` (audiobook | conversational | agentic), ``voice_audio_b64`` or ``voice_path``,
  ``controls`` (the style sliders) and ``format``, answering ``{"audio_b64", "mime_type", "sample_rate"}``;
  ``GET /controls`` lists the sliders and ``GET /readyz`` is 503 while the model loads. Not a binary but a
  Python application: bolo clones the pinned tag and runs ``uv run --python 3.12 --frozen tontaube serve``
  (``uv`` must be on PATH), pointing ``MODEL_PATH`` at the weights bolo downloaded. Its first start installs its
  own dependencies (PyTorch and more, several GB) into the checkout.

bolo either *connects* to such a server (setting ``tts_servers``: ``"<model id or runtime>=<url>"``) or
*manages* one: it starts the binary on a loopback port when the model is loaded and stops it when the model is
unloaded. The binary is found through a setting (``llama_server_bin`` / ``audiocpp_server_bin``), in bolo's
components folder (``bolo components install llama-server``: a pinned upstream release archive, checked
against its SHA-256) or on ``PATH``.

This module has no heavy imports: the catalog and the components page use it to say whether a model can run.
"""

from __future__ import annotations

import contextlib
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.config import Settings

LLAMA = "llama-server"
AUDIOCPP = "audio.cpp"
TONTAUBE = "tontaube"
RUNTIMES = (LLAMA, AUDIOCPP, TONTAUBE)
ENGINE_ID = "tts-server"


@dataclass(frozen=True)
class Release:
    """A prebuilt upstream archive for one platform. The SHA-256 was computed from the download itself."""

    url: str
    sha256: str
    size: int | None = None
    #: more archives unpacked into the same folder (the CUDA runtime DLLs of the Windows CUDA builds)
    extras: tuple[Release, ...] = ()


@dataclass(frozen=True)
class GitSource:
    """A pinned source checkout (an application run from source rather than a release binary)."""

    url: str
    tag: str
    commit: str
    platforms: tuple[str, ...]


@dataclass(frozen=True)
class Runtime:
    id: str
    name: str
    binary: str
    license: str
    license_url: str
    homepage: str
    version: str
    releases: dict[str, Release]
    setting: str  # the setting that names an operator-provided binary
    description: str
    git: GitSource | None = None


RUNTIME_INFO: dict[str, Runtime] = {
    LLAMA: Runtime(
        id=LLAMA,
        name="llama.cpp server (llama-server)",
        binary="llama-server",
        license="MIT",
        license_url="https://github.com/ggml-org/llama.cpp/blob/master/LICENSE",
        homepage="https://github.com/ggml-org/llama.cpp",
        version="b12375",
        releases={
            "linux-x86_64": Release(
                "https://github.com/ggml-org/llama.cpp/releases/download/b12375/llama-b12375-bin-ubuntu-x64.tar.gz",
                "89fdf5a4066170fa408a5ebe1510b50bd8a5bab750e49ae531bb1d4cdbae65e3",
                23332809,
            ),
            # the Vulkan build (AMD, Intel and NVIDIA GPUs); it needs the Vulkan loader (libvulkan1) and a driver
            "linux-x86_64-vulkan": Release(
                "https://github.com/ggml-org/llama.cpp/releases/download/b12375/llama-b12375-bin-ubuntu-vulkan-x64.tar.gz",
                "e5144c8499399c115c7f5319421942252e9e84bf7d78b46f968cd5dbc769df72",
                36187166,
            ),
            "linux-aarch64": Release(
                "https://github.com/ggml-org/llama.cpp/releases/download/b12375/llama-b12375-bin-ubuntu-arm64.tar.gz",
                "122ae6ea1a24d713ca9a3e9094fcc1c766e6547373a7356f97668f24310daf13",
                18963094,
            ),
            "darwin-arm64": Release(
                "https://github.com/ggml-org/llama.cpp/releases/download/b12375/llama-b12375-bin-macos-arm64.tar.gz",
                "bdd4a9681383933b1323e5351edbaa2501bede6209c3bd4359c5262339299e47",
                16913580,
            ),
            "darwin-x86_64": Release(
                "https://github.com/ggml-org/llama.cpp/releases/download/b12375/llama-b12375-bin-macos-x64.tar.gz",
                "e61d5511936adc349db7af0dd5dec2134a3fde108fb3759782a133bfa0106365",
            ),
            # the Windows CPU build loads the best of its CPU back ends (SSE4.2 to AVX-512) at start-up
            "windows-x86_64": Release(
                "https://github.com/ggml-org/llama.cpp/releases/download/b12375/llama-b12375-bin-win-cpu-x64.zip",
                "3aa7046f1b310c404418625d4e7e7b057c2d0e0eef5557925561381373d8f026",
                25116305,
            ),
            "windows-x86_64-cuda": Release(
                "https://github.com/ggml-org/llama.cpp/releases/download/b12375/llama-b12375-bin-win-cuda-12.4-x64.zip",
                "8b7bf8f3b5269531cbf00c52e4ee5d8b0c77cb4ea7af694ed3da3a34e47a7099",
                319401248,
                extras=(
                    Release(
                        "https://github.com/ggml-org/llama.cpp/releases/download/b12375/cudart-llama-bin-win-cuda-12.4-x64.zip",
                        "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
                        391443627,
                    ),
                ),
            ),
            "windows-x86_64-vulkan": Release(
                "https://github.com/ggml-org/llama.cpp/releases/download/b12375/llama-b12375-bin-win-vulkan-x64.zip",
                "8990615f25d4fa12a7f1d8fc9da995538ce739fec01838f003fa9ff099d643f0",
                38113510,
            ),
            "windows-arm64": Release(
                "https://github.com/ggml-org/llama.cpp/releases/download/b12375/llama-b12375-bin-win-cpu-arm64.zip",
                "dcbbd1217956893f91714e58bba95fe87dda272a3e72aabe46513a1ce9113cac",
                13459532,
            ),
        },
        setting="llama_server_bin",
        description=(
            "llama.cpp's HTTP server, which speaks text-to-speech for Chatterbox, Qwen3-TTS, Pocket TTS, Soprano, "
            "LFM3-Audio and Fish Drama 3 GGUF models. bolo starts it on a loopback port for the model in use, or "
            "connects to one you run (setting tts_servers). The prebuilt release is the CPU build (Metal on Apple "
            "Silicon); device = vulkan installs the Vulkan build on Linux x86-64 and Windows, device = cuda the CUDA 12 "
            "build on Windows. llama.cpp publishes no CUDA build for Linux: there, use the Vulkan build or point "
            "llama_server_bin at a CUDA build of your own."
        ),
    ),
    AUDIOCPP: Runtime(
        id=AUDIOCPP,
        name="audio.cpp server (audiocpp_server)",
        binary="audiocpp_server",
        license="Apache-2.0",
        license_url="https://github.com/0xShug0/audio.cpp/blob/main/LICENSE",
        homepage="https://github.com/0xShug0/audio.cpp",
        version="v0.9.4",
        releases={
            # the portable build (every x86-64 CPU level); built on ubuntu-latest, so it needs glibc 2.43 or newer
            "linux-x86_64": Release(
                "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.4/audio-v0.9.4-bin-ubuntu-x64-cpu-portable.tar.gz",
                "02756c556d78ccca8040cab6fbc6c7bf57d4f6ed70fe534e8a9e846bab464687",
                70154506,
            ),
            "darwin-arm64": Release(
                "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.4/audio-v0.9.4-bin-macos-arm64-metal.tar.gz",
                "5722c4f33e56114345965c0dee472a9239218f7499ace6393bba6a951a41b86a",
            ),
            # the portable Windows builds bring the Visual C++ runtime DLLs they need
            "windows-x86_64": Release(
                "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.4/audio-v0.9.4-bin-windows-x64-cpu-portable.zip",
                "9c11a03b7ea21979792fde05f2f355ea1fc0da26ccd8f675dd61c628250ae76f",
                32190461,
            ),
            "windows-x86_64-cuda": Release(
                "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.4/audio-v0.9.4-bin-windows-x64-cuda12.4.zip",
                "e85d829e121a424a960d9ad293536b7ada74ea845b1c12fcf2991ec09a224d1e",
                467800419,
                extras=(
                    Release(
                        "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.4/audio-v0.9.4-cudart-windows-x64-cuda12.4.zip",
                        "efb58667b95bcb32e89ddf87811664e6ffdced3fe7015ddcc077599ed5f44bb2",
                        607273675,
                    ),
                ),
            ),
            "windows-x86_64-vulkan": Release(
                "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.4/audio-v0.9.4-bin-windows-x64-vulkan.zip",
                "804f70cf47c7b8450ef37e76fc04ecee71728345061f6e749a412932dd7db7a7",
                84083745,
            ),
        },
        setting="audiocpp_server_bin",
        description=(
            "audio.cpp's HTTP server: one ggml runtime for 60+ text-to-speech families (VoxCPM2, Magpie, "
            "Chatterbox, CosyVoice3, Maya1, Qwen3-TTS, …) behind an OpenAI-style /v1/audio/speech. bolo writes "
            "its config, starts it on a loopback port for the model in use, or connects to one you run (setting "
            "tts_servers). The prebuilt Linux release is the CPU build; audio.cpp is fastest on CUDA, so for a GPU "
            "point audiocpp_server_bin at a CUDA build. On Windows, device = cuda or vulkan installs the CUDA 12 or "
            "Vulkan build."
        ),
    ),
    TONTAUBE: Runtime(
        id=TONTAUBE,
        name="Tontaube inference server (tontaube serve)",
        binary="tontaube",
        license="Apache-2.0 (inference code; the model weights have their own licence)",
        license_url="https://github.com/craitech/tontaube/blob/v1.1.0/LICENSE",
        homepage="https://github.com/craitech/tontaube",
        version="v1.1.0",
        releases={},
        setting="tontaube_dir",
        description=(
            "The Tontaube inference server, the only runtime for TontaubeV1.1 (expressive voice cloning, English "
            "and German first). bolo clones the pinned release and starts `uv run --python 3.12 --frozen tontaube "
            "serve` for the model in use; its first start installs PyTorch and the server's other dependencies "
            "(several GB). Linux only: an NVIDIA GPU (Ampere or newer for bf16), or an x86-64 CPU with AVX2, FMA "
            "and F16C and a C++ compiler. Or run it yourself and set tts_servers."
        ),
        git=GitSource(
            "https://github.com/craitech/tontaube.git",
            "v1.1.0",
            "ef254ca3d7dcbf536e83b04fd38798aea6b5e759",
            ("linux-x86_64", "linux-aarch64"),
        ),
    ),
}


def platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "arm64" if system == "darwin" else "aarch64"}.get(machine, machine)
    return f"{system}-{machine}"


def unsupported_reason(runtime: str, key: str | None = None) -> str | None:
    """Why ``runtime`` cannot run on this platform at all (``key``: :func:`platform_key`), or None. Tontaube runs on
    Linux only: on Windows it can run under WSL2."""
    info = RUNTIME_INFO[runtime]
    key = key or platform_key()
    if info.git is None or key in info.git.platforms:
        return None
    short = info.name.split(" ", 1)[0]
    if key.startswith("windows"):
        return (
            f"{short} is Linux only (use WSL2): this is {key}; run bolo inside WSL2, or run `{info.binary} serve` "
            f'there and set tts_servers = ["{runtime}=http://host:port"]'
        )
    return f"{short} is Linux only: this is {key}; run `{info.binary} serve` on Linux and set tts_servers"


def components_dir(settings: Settings) -> Path:
    return Path(settings.data_dir) / "components"


def install_dir(settings: Settings, runtime: str, flavour: str | None = None) -> Path:
    """Where bolo installs ``runtime``: ``<data_dir>/components/<runtime>/<version>``, and each build of a release
    runtime in its own folder below that (``.../cpu``, ``.../cuda``, ``.../vulkan``), so that installing the CUDA
    build does not find the CPU build "already installed" and the build matching ``device`` is the one started."""
    info = RUNTIME_INFO[runtime]
    base = components_dir(settings) / runtime / info.version
    return base / flavour if flavour else base


def is_checkout(path: Path) -> bool:
    return (path / "pyproject.toml").is_file()


#: the builds of a release runtime: "cpu" is the platform's default build (on Apple Silicon it has Metal)
CPU_FLAVOUR = "cpu"
FLAVOURS = (CPU_FLAVOUR, "cuda", "vulkan")


def _flavour(device: str | None) -> str | None:
    """The build a GPU device asks for (``cuda:1`` → ``cuda``), when a platform has one; None: the default build."""
    kind = (device or "").split(":", 1)[0].strip().lower()
    return kind if kind in ("cuda", "vulkan") else None


def _release_key(flavour: str) -> str:
    key = platform_key()
    return key if flavour == CPU_FLAVOUR else f"{key}-{flavour}"


def pinned_flavours(runtime: str) -> list[str]:
    """The builds pinned for this platform."""
    info = RUNTIME_INFO[runtime]
    return [f for f in FLAVOURS if _release_key(f) in info.releases]


#: how to get a GPU build where none is published (Linux CUDA)
GPU_BUILD_DOCS = (
    "https://github.com/cB-Sigma/bolo-speech/blob/main/docs/configuration.md#a-gpu-build-of-llama-server-on-linux"
)


def missing_build_note(runtime: str, device: str | None = None) -> str | None:
    """What to say when ``device`` asks for a GPU build that is not pinned for this platform, so that the CPU build
    is what gets installed (and runs): the builds that are pinned here, and the setting for a build of your own."""
    wanted = _flavour(device)
    if not wanted or wanted in pinned_flavours(runtime):
        return None
    info = RUNTIME_INFO[runtime]
    if info.git is not None or not info.releases:
        return None
    gpu = [f for f in pinned_flavours(runtime) if f != CPU_FLAVOUR]
    other = f"install the {gpu[0]} build (--flavour {gpu[0]}), or " if gpu else ""
    return (
        f"device {device!r}: {info.name} {info.version} has no prebuilt {wanted} build for {platform_key()}, so the "
        f"CPU build is installed and runs on the CPU; for the GPU, {other}build {info.binary} with "
        f"{'CUDA' if wanted == 'cuda' else 'Vulkan'} and set {info.setting} to it: see "
        f"{GPU_BUILD_DOCS}"
    )


def flavour_for(runtime: str, device: str | None = None) -> str:
    """The build ``device`` asks for on this platform: ``cuda`` or ``vulkan`` where that build is pinned (CUDA on
    Windows, Vulkan on Windows and Linux x86-64), else ``cpu``, the platform's default build
    (:func:`missing_build_note` says so)."""
    wanted = _flavour(device)
    return wanted if wanted and wanted in pinned_flavours(runtime) else CPU_FLAVOUR


def release_for(runtime: str, device: str | None = None, flavour: str | None = None) -> Release | None:
    """The pinned release archive for this platform: the build ``flavour`` names, else the one for ``device`` (the
    CUDA or Vulkan build where one is pinned, such as on Windows), else the platform's default build."""
    return RUNTIME_INFO[runtime].releases.get(_release_key(flavour or flavour_for(runtime, device)))


def detect_flavour(folder: Path, skip: tuple[str, ...] = ()) -> str:
    """The build unpacked in ``folder`` (not counting its subfolders named in ``skip``), from its libraries
    (``ggml-cuda.dll``, ``cudart64_12.dll``, ``ggml-vulkan.dll``, ...): for installs made before each build had its
    own folder."""
    names = " ".join(
        p.name.lower()
        for p in folder.rglob("*")
        if p.is_file() and not (skip and p.relative_to(folder).parts[0] in skip)
    )
    if "cuda" in names or "cublas" in names:
        return "cuda"
    if "vulkan" in names:
        return "vulkan"
    return CPU_FLAVOUR


def _binary_in(folder: Path, binary: str, skip: tuple[str, ...] = ()) -> Path | None:
    for name in executable_names(binary):
        for candidate in sorted(folder.rglob(name)):
            rel = candidate.relative_to(folder).parts
            if skip and len(rel) > 1 and rel[0] in skip:
                continue
            if is_executable(candidate):
                return candidate
    return None


def installed_flavours(settings: Settings, runtime: str) -> dict[str, Path]:
    """The builds of a release runtime in bolo's components folder: flavour → binary. An install from before the
    per-build folders (the files right in the version folder) counts as the build its libraries show."""
    info = RUNTIME_INFO[runtime]
    base = install_dir(settings, runtime)
    out: dict[str, Path] = {}
    if info.git is not None or not base.is_dir():
        return out
    for flavour in FLAVOURS:
        if (base / flavour).is_dir():
            found = _binary_in(base / flavour, info.binary)
            if found is not None:
                out[flavour] = found
    legacy = _binary_in(base, info.binary, skip=FLAVOURS)
    if legacy is not None:
        out.setdefault(detect_flavour(base, skip=FLAVOURS), legacy)
    return out


def _preference(runtime: str, device: str | None) -> list[str]:
    """The installed builds to start, best first: the one ``device`` asks for (with ``auto``, the GPU builds first:
    whichever one was installed); then the others, so that a model still runs on the build that is there."""
    if (device or "").strip().lower() == "auto":
        return ["cuda", "vulkan", CPU_FLAVOUR]
    wanted = flavour_for(runtime, device)
    rest = ["vulkan", CPU_FLAVOUR] if wanted == "cuda" else [CPU_FLAVOUR, "vulkan", "cuda"]
    return [wanted, *(f for f in rest if f != wanted)]


def migrate_legacy(settings: Settings, runtime: str) -> None:
    """Move an install from before the per-build folders into the folder of the build it is."""
    info = RUNTIME_INFO[runtime]
    base = install_dir(settings, runtime)
    if info.git is not None or not base.is_dir():
        return
    loose = [p for p in base.iterdir() if p.name not in FLAVOURS and not p.name.endswith(".partial")]
    if not loose or _binary_in(base, info.binary, skip=FLAVOURS) is None:
        return
    target = base / detect_flavour(base, skip=FLAVOURS)
    if target.exists():
        return
    target.mkdir(parents=True)
    for item in loose:
        item.rename(target / item.name)


def _windows() -> bool:
    return os.name == "nt"


def _pathext() -> list[str]:
    return [e.lower() for e in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";") if e]


def executable_names(binary: str) -> list[str]:
    """The file names ``binary`` may have here: itself, and on Windows with each extension in ``PATHEXT``
    (``llama-server.exe``), as ``shutil.which`` looks for it."""
    if not _windows():
        return [binary]
    exts = _pathext()
    names = [binary + ext for ext in exts]
    if Path(binary).suffix.lower() in exts:
        names.insert(0, binary)
    return names


def is_executable(path: Path) -> bool:
    """A file this system can run: the executable bit on POSIX, an extension in ``PATHEXT`` on Windows."""
    if not path.is_file():
        return False
    if _windows():
        return path.suffix.lower() in _pathext()
    return os.access(path, os.X_OK)


def _configured_binary(value: str) -> Path | None:
    path = Path(value).expanduser()
    for name in executable_names(path.name):
        candidate = path.with_name(name)
        if is_executable(candidate):
            return candidate
    found = shutil.which(value)
    return Path(found) if found else None


def find_binary(
    settings: Settings, runtime: str, flavour: str | None = None, device: str | None = None
) -> tuple[Path | None, str | None]:
    """``(path, source)`` of the server binary for ``runtime``: the setting, bolo's components folder, then PATH.
    For a runtime run from source (Tontaube) the path is the checkout folder. On Windows a name without an
    extension also finds ``<name>.exe`` (and the other ``PATHEXT`` extensions), as on ``PATH``. Of the builds in
    the components folder, the one ``flavour`` names (only that one), else the one ``device`` asks for, else
    another one that is installed (see :func:`_preference`). ``device`` replaces the setting's in that choice (a
    model moved to the CPU after its GPU was lost starts the CPU build when there is one)."""
    info = RUNTIME_INFO[runtime]
    if info.git is not None:
        configured = getattr(settings, info.setting, None)
        if configured:
            path = Path(str(configured)).expanduser()
            return (path, "settings") if is_checkout(path) else (None, None)
        local = install_dir(settings, runtime) / info.binary
        return (local, "bolo") if is_checkout(local) else (None, None)
    configured = getattr(settings, info.setting, None)
    if configured:
        found_path = _configured_binary(str(configured))
        return (found_path, "settings") if found_path else (None, None)
    builds = installed_flavours(settings, runtime)
    if flavour is not None:
        return (builds[flavour], "bolo") if flavour in builds else (None, None)
    for wanted in _preference(runtime, device or getattr(settings, "device", None)):
        if wanted in builds:
            return builds[wanted], "bolo"
    found = shutil.which(info.binary)
    return (Path(found), "path") if found else (None, None)


def flavour_of(settings: Settings | None, runtime: str, binary: Path) -> str:
    """The build ``binary`` is: the folder bolo installed it in, else what the libraries next to it show
    (``ggml-vulkan.dll``, ``ggml-cuda.dll``, ...)."""
    if settings is not None:
        with contextlib.suppress(Exception):
            for flavour, path in installed_flavours(settings, runtime).items():
                if path == binary:
                    return flavour
    with contextlib.suppress(OSError):
        names = " ".join(p.name.lower() for p in binary.parent.iterdir() if p.is_file())
        if "cuda" in names or "cublas" in names:
            return "cuda"
        if "vulkan" in names:
            return "vulkan"
    return CPU_FLAVOUR


def child_env(flavour: str | None) -> dict[str, str]:
    """Variables a server binary of this build needs on this machine, besides bolo's own environment: the Vulkan
    build on a Windows machine with AMD graphics runs without AMD's switchable-graphics layer
    (:func:`bolo.winhw.vulkan_child_env`)."""
    from bolo import winhw

    return winhw.vulkan_child_env(flavour)


def run_env(path: Path, settings: Settings | None = None, runtime: str = LLAMA) -> dict[str, str] | None:
    """The environment to run ``path`` in for a quick question (``--help``, ``--version``, ``--list-devices``):
    None (bolo's own) unless its build needs something more (:func:`child_env`)."""
    extra = child_env(flavour_of(settings, runtime, path))
    return {**os.environ, **extra} if extra else None


def parse_server_setting(item: str) -> tuple[str, str]:
    """``"target=url"`` → ``(target, url)``; ``target`` is a model id or a runtime name."""
    target, sep, url = item.partition("=")
    if not sep or not target.strip() or not url.strip():
        raise ValueError(f"a TTS server is configured as 'model-or-runtime=url', got {item!r}")
    return target.strip(), url.strip().rstrip("/")


def runtime_of(spec: ModelSpec) -> str:
    return str(spec.options.get("runtime") or LLAMA)


def url_for(settings: Settings, names: list[str]) -> str | None:
    """The first ``tts_servers`` URL configured for one of ``names`` (model id, aliases, runtime), or None."""
    by_target: dict[str, str] = {}
    for item in settings.tts_servers:
        with contextlib.suppress(ValueError):
            target, url = parse_server_setting(item)
            by_target[target.lower()] = url
    return next((by_target[n.lower()] for n in names if n and n.lower() in by_target), None)


def remote_url(settings: Settings | None, spec: ModelSpec) -> str | None:
    """The URL of an operator-run server for this model (by model id, alias, then runtime), or None."""
    if settings is None or spec.engine != ENGINE_ID:
        return None
    return url_for(settings, [spec.id, *spec.aliases, runtime_of(spec)])


def check_runs(path: Path) -> str | None:
    """None when the binary starts on this machine (``--help``), else why not (e.g. a too old C library)."""
    try:
        proc = subprocess.run([str(path), "--help"], capture_output=True, timeout=30, check=False, env=run_env(path))
    except subprocess.TimeoutExpired:
        return f"`{path.name} --help` did not answer within 30 s" + _hang_hint(path)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{type(exc).__name__}: {exc}"
    text = (proc.stdout + proc.stderr).decode(errors="replace")
    if "GLIBC_" in text and "not found" in text:
        needed = sorted(set(re.findall(r"GLIBC_[0-9.]+", text)))
        return (
            f"the prebuilt binary needs a newer C library than this system has ({', '.join(needed)}); "
            "build it from source or use a newer distribution"
        )
    if proc.returncode not in (0, 1) and not text.strip():
        return f"it exited with {proc.returncode}"
    return None


def _hang_hint(path: Path) -> str:
    """What to try when a Vulkan build hangs on start."""
    if flavour_of(None, LLAMA, path) != "vulkan":
        return ""
    return (
        " (a Vulkan build that hangs while it lists the GPUs: on a laptop with AMD and NVIDIA graphics try the CPU "
        'or CUDA build, or set DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1=1; docs/configuration.md, "Windows")'
    )


def binary_version(path: Path) -> str | None:
    """What ``<binary> --version`` prints first (llama-server prints ``version: …``)."""
    try:
        proc = subprocess.run([str(path), "--version"], capture_output=True, timeout=15, check=False, env=run_env(path))
    except (OSError, subprocess.SubprocessError):
        return None
    text = (proc.stdout + proc.stderr).decode(errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("version"):
            return line.split(":", 1)[-1].strip() or line
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return first[:80] or None


# -- installing a release archive -------------------------------------------------------------------------------


def _is_root(name: str) -> bool:
    return name.replace("\\", "/").strip("/") in ("", ".")


def _member_path(name: str) -> PurePosixPath | None:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(p in ("..", "") for p in path.parts) or ":" in path.parts[0]:
        return None
    return path


def extract_release(archive: Path, dest: Path) -> list[str]:
    """Unpack a release archive (tar.gz or zip) into ``dest``, keeping executable bits.

    Unlike model archives, release tarballs hold shared-library symlinks (``libllama.so -> libllama.so.1``);
    a symlink is accepted only when its target is a plain name in the same folder. Anything absolute, climbing
    out of ``dest`` or of another type (devices, hard links) rejects the whole archive.
    """
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if archive.name.endswith(".zip") or zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if _is_root(info.filename):
                    continue
                rel = _member_path(info.filename)
                if rel is None:
                    raise ValueError(f"unsafe path in archive: {info.filename!r}")
                target = dest.joinpath(*rel.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)
                mode = (info.external_attr >> 16) & 0o777
                if mode:
                    target.chmod(mode)
                written.append(str(rel))
        return written
    with tarfile.open(archive, "r:*") as tf:
        members = [m for m in tf.getmembers() if not (m.isdir() and _is_root(m.name))]
        for m in members:
            rel = _member_path(m.name)
            if rel is None:
                raise ValueError(f"unsafe path in archive: {m.name!r}")
            if m.issym():
                link = PurePosixPath(m.linkname)
                if link.is_absolute() or len(link.parts) != 1 or link.parts[0] in ("..", "."):
                    raise ValueError(f"unsafe link in archive: {m.name} -> {m.linkname}")
            elif not (m.isfile() or m.isdir()):
                raise ValueError(f"unsupported member in archive: {m.name!r}")
        for m in members:
            rel = _member_path(m.name)
            assert rel is not None
            target = dest.joinpath(*rel.parts)
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if m.issym():
                with contextlib.suppress(FileNotFoundError):
                    target.unlink()
                target.symlink_to(m.linkname)
                written.append(str(rel))
                continue
            member = tf.extractfile(m)
            assert member is not None
            with member, target.open("wb") as out:
                shutil.copyfileobj(member, out)
            target.chmod((m.mode & 0o777) | stat.S_IRUSR | stat.S_IWUSR)
            written.append(str(rel))
    return written


def install_status(settings: Settings, runtime: str) -> dict[str, Any]:
    """What the components page shows for one runtime (without running anything slow)."""
    path, source = find_binary(settings, runtime)
    release = release_for(runtime, settings.device)
    builds = installed_flavours(settings, runtime)
    flavour = next((f for f, p in builds.items() if p == path), None) if source == "bolo" else None
    urls: list[str] = []
    for item in settings.tts_servers:
        with contextlib.suppress(ValueError):
            target, url = parse_server_setting(item)
            if target.lower() == runtime:
                urls.append(url)
    return {
        "path": str(path) if path else None, "source": source, "release": release, "urls": urls,
        "flavour": flavour, "flavours": sorted(builds), "wanted_flavour": flavour_for(runtime, settings.device),
    }  # fmt: skip
