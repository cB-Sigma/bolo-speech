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
        },
        setting="llama_server_bin",
        description=(
            "llama.cpp's HTTP server, which speaks text-to-speech for Chatterbox, Qwen3-TTS, Pocket TTS, Soprano, "
            "LFM3-Audio and Fish Drama 3 GGUF models. bolo starts it on a loopback port for the model in use, or "
            "connects to one you run (setting tts_servers). The prebuilt release is the CPU build; for a GPU point "
            "llama_server_bin at a CUDA/Vulkan/Metal build."
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
        },
        setting="audiocpp_server_bin",
        description=(
            "audio.cpp's HTTP server: one ggml runtime for 60+ text-to-speech families (VoxCPM2, Magpie, "
            "Chatterbox, CosyVoice3, Maya1, Qwen3-TTS, …) behind an OpenAI-style /v1/audio/speech. bolo writes "
            "its config, starts it on a loopback port for the model in use, or connects to one you run (setting "
            "tts_servers). The prebuilt Linux release is the CPU build; audio.cpp is fastest on CUDA, so for a GPU "
            "point audiocpp_server_bin at a CUDA build."
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


def components_dir(settings: Settings) -> Path:
    return Path(settings.data_dir) / "components"


def install_dir(settings: Settings, runtime: str) -> Path:
    info = RUNTIME_INFO[runtime]
    return components_dir(settings) / runtime / info.version


def is_checkout(path: Path) -> bool:
    return (path / "pyproject.toml").is_file()


def find_binary(settings: Settings, runtime: str) -> tuple[Path | None, str | None]:
    """``(path, source)`` of the server binary for ``runtime``: the setting, bolo's components folder, then PATH.
    For a runtime run from source (Tontaube) the path is the checkout folder."""
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
        path = Path(str(configured)).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path, "settings"
        found = shutil.which(str(configured))
        return (Path(found), "settings") if found else (None, None)
    local = install_dir(settings, runtime)
    if local.is_dir():
        for candidate in sorted(local.rglob(info.binary)):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate, "bolo"
    found = shutil.which(info.binary)
    return (Path(found), "path") if found else (None, None)


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
        proc = subprocess.run([str(path), "--help"], capture_output=True, timeout=30, check=False)
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


def binary_version(path: Path) -> str | None:
    """What ``<binary> --version`` prints first (llama-server prints ``version: …``)."""
    try:
        proc = subprocess.run([str(path), "--version"], capture_output=True, timeout=15, check=False)
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
    info = RUNTIME_INFO[runtime]
    path, source = find_binary(settings, runtime)
    release = info.releases.get(platform_key())
    urls: list[str] = []
    for item in settings.tts_servers:
        with contextlib.suppress(ValueError):
            target, url = parse_server_setting(item)
            if target.lower() == runtime:
                urls.append(url)
    return {"path": str(path) if path else None, "source": source, "release": release, "urls": urls}
