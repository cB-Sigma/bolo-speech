"""The CUDA libraries ONNX Runtime's CUDA provider and CTranslate2 (faster-whisper) load, and what to do without them.

Two runtimes in one process want two CUDA majors: ``onnxruntime-gpu`` 1.32 on PyPI is built for CUDA 13 and cuDNN 9
(``cublasLt64_13.dll``), CTranslate2 4.x for CUDA 12 (``cublas64_12.dll``). NVIDIA's pip wheels bring both
(``onnxruntime-gpu[cuda,cudnn]``, ``nvidia-cublas-cu12``, ``nvidia-cudnn-cu12``), but nothing tells Windows where
they are: they sit in ``site-packages\\nvidia\\<package>\\bin`` (``nvidia\\cu13\\bin\\x86_64`` for CUDA 13), which is
neither on ``PATH`` nor next to the DLL that needs them. ONNX Runtime then falls back to the CPU with a line on
standard error and nothing else; CTranslate2 fails the first request. So before either one starts on a GPU, bolo
adds those folders (``os.add_dll_directory`` and the front of ``PATH``), calls ``onnxruntime.preload_dlls()`` where
it exists (1.21 and newer, Windows and Linux), and afterwards checks what it got:

* :func:`bolo.ort_providers.create_session` compares a session's providers with the ones asked for; a missing one is
  a :class:`ProviderFallback`, logged with the library that was missing and recorded for the model's load (the
  lifecycle manager turns it into the model's notice and moves its placement to the CPU).
* the faster-whisper engine warms the model up on the GPU and, when a CUDA library is missing, loads it again on the
  CPU and records the same.
* ``bolo doctor`` (:func:`doctor_lines`) loads both on CUDA and names what is missing, with the install command.

This module has no heavy imports; ``onnxruntime`` and ``ctranslate2`` are imported only when asked about.
"""

from __future__ import annotations

import contextlib
import ctypes
import io
import logging
import os
import re
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: what makes ONNX Runtime's CUDA provider work (one ONNX Runtime package at a time: they share the module)
ORT_CUDA_RECIPE = 'pip uninstall -y onnxruntime, then pip install "onnxruntime-gpu[cuda,cudnn]"'
#: what CTranslate2 4.x (faster-whisper) needs on the GPU: CUDA 12's cuBLAS and cuDNN 9
CT2_CUDA_RECIPE = 'pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"'
#: both, in bolo's environment (docs/configuration.md, "Windows")
CUDA_RECIPE = (
    'pip uninstall -y onnxruntime, then pip install "onnxruntime-gpu[cuda,cudnn]" nvidia-cublas-cu12 '
    '"nvidia-cudnn-cu12>=9,<10" in bolo\'s environment (uv: uv pip install ...)'
)
DOCS = "docs/configuration.md#cuda-on-windows"

_lock = threading.RLock()
_dirs_added: set[str] = set()
_dll_handles: list[Any] = []
_ort_prepared = False
_ct2_prepared = False


def _windows() -> bool:
    return os.name == "nt"


# -- where NVIDIA's pip wheels put their libraries ------------------------------------------------------------
def nvidia_roots() -> list[Path]:
    """The ``nvidia`` folders (namespace package of NVIDIA's wheels) in this environment's ``sys.path``."""
    roots: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        root = Path(entry) / "nvidia"
        with contextlib.suppress(OSError):
            if root.is_dir() and root not in roots:
                roots.append(root)
    return roots


def nvidia_lib_dirs(roots: list[Path] | None = None, windows: bool | None = None) -> list[Path]:
    """The folders of NVIDIA's wheels that hold libraries: ``nvidia/<package>/bin`` and ``bin/x86_64`` (the CUDA 13
    wheels' layout, ``nvidia/cu13/bin/x86_64``) holding DLLs on Windows, ``nvidia/<package>/lib`` elsewhere."""
    win = _windows() if windows is None else windows
    out: list[Path] = []
    for root in nvidia_roots() if roots is None else roots:
        try:
            packages = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            continue
        for package in packages:
            for folder in [package / "bin", package / "bin" / "x86_64"] if win else [package / "lib"]:
                with contextlib.suppress(OSError):
                    if folder.is_dir() and any(folder.glob("*.dll" if win else "*.so*")):
                        out.append(folder)
    return out


def add_dll_directories(dirs: list[Path] | None = None) -> list[Path]:
    """Windows: make the DLLs of NVIDIA's wheels loadable. Each folder goes to ``os.add_dll_directory`` (DLLs loaded
    with the default search directories, as ONNX Runtime's providers are) and to the front of ``PATH`` (plain
    ``LoadLibrary``, as CTranslate2 loads cuBLAS). Elsewhere it does nothing. Returns the folders added now."""
    if not _windows():
        return []
    added: list[Path] = []
    with _lock:
        for folder in nvidia_lib_dirs() if dirs is None else dirs:
            key = str(folder)
            if key in _dirs_added:
                continue
            _dirs_added.add(key)
            add = getattr(os, "add_dll_directory", None)
            if add is not None:
                with contextlib.suppress(OSError):
                    _dll_handles.append(add(key))
            path = os.environ.get("PATH", "")
            if key.lower() not in (p.lower() for p in path.split(os.pathsep)):
                os.environ["PATH"] = key + (os.pathsep + path if path else "")
            added.append(folder)
    if added:
        log.info("CUDA libraries of NVIDIA's pip wheels: %s", ", ".join(str(d) for d in added))
    return added


def prepare_ort_cuda() -> None:
    """Once per process, before the first CUDA session: the NVIDIA wheels' folders (Windows) and
    ``onnxruntime.preload_dlls()`` (onnxruntime-gpu 1.21 and newer), which loads CUDA and cuDNN from those wheels on
    Windows and Linux. What it prints about libraries it could not load goes to the debug log."""
    global _ort_prepared
    with _lock:
        if _ort_prepared:
            return
        _ort_prepared = True
    add_dll_directories()
    try:
        import onnxruntime as ort
    except Exception:
        return
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    said = io.StringIO()
    try:
        with contextlib.redirect_stdout(said):
            preload()
    except Exception as exc:
        log.debug("onnxruntime.preload_dlls() failed: %s: %s", type(exc).__name__, exc)
    if said.getvalue().strip():
        log.debug("onnxruntime.preload_dlls(): %s", " | ".join(said.getvalue().strip().splitlines()))


def _load(name: str) -> bool:
    try:
        if _windows():
            ctypes.WinDLL(name, winmode=0)  # type: ignore[attr-defined,unused-ignore]  # PATH too, as the runtimes search
        else:
            ctypes.CDLL(name, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    except OSError:
        return False
    return True


def prepare_ct2_cuda() -> None:
    """Once per process, before CTranslate2 runs on CUDA: on Windows the NVIDIA wheels' folders (``nvidia-cublas-cu12``,
    ``nvidia-cudnn-cu12``); on Linux, the CUDA 12 cuBLAS and cuDNN of those wheels when the loader cannot find them
    by name (they are not on ``LD_LIBRARY_PATH``)."""
    global _ct2_prepared
    with _lock:
        if _ct2_prepared:
            return
        _ct2_prepared = True
    if _windows():
        add_dll_directories()
        return
    if sys.platform != "linux":
        return
    wanted = [n for n in ("libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9") if not _load(n)]
    if not wanted:
        return
    files = [f for d in nvidia_lib_dirs(windows=False) for f in sorted(d.glob("lib*.so.*"))]
    pending = [f for f in files if f.name in wanted or (f.name.startswith("libcudnn") and f.name.endswith(".so.9"))]
    while pending:  # cuDNN's parts depend on each other: load in rounds until nothing more loads
        loaded = [f for f in pending if _load(str(f))]
        if not loaded:
            break
        pending = [f for f in pending if f not in loaded]


# -- what a failed load says ----------------------------------------------------------------------------------
_DEPENDS = re.compile(r'depends on "([^"]+)" which is missing')
_ERROR_LOADING = re.compile(r'Error loading "([^"]+)"')
_SHARED_OBJECT = re.compile(r"([\w.+-]+\.so[\w.]*): cannot open shared object file")
_CT2_LIBRARY = re.compile(r"Library (\S+) is not found or cannot be loaded")


def missing_libraries(text: str) -> list[str]:
    """The libraries a loader message names as missing (ONNX Runtime's log, a CTranslate2 error)."""
    names: list[str] = []
    for pattern in (_DEPENDS, _SHARED_OBJECT, _CT2_LIBRARY):
        names += [m.group(1) for m in pattern.finditer(text)]
    if not names:  # the provider library itself would not load (no "depends on" in the message)
        names = [m.group(1) for m in _ERROR_LOADING.finditer(text)]
    out: list[str] = []
    for name in names:
        base = re.split(r"[\\/]", name)[-1]
        if base not in out:
            out.append(base)
    return out


def ort_cuda_major() -> int | None:
    """The CUDA major the installed onnxruntime-gpu was built for (12, 13), or None (no CUDA build)."""
    try:
        from onnxruntime.capi import build_and_package_info as info
    except Exception:
        return None
    version = str(getattr(info, "cuda_version", "") or "")
    major = version.split(".", 1)[0]
    return int(major) if major.isdigit() else None


def ort_cuda_libraries(major: int | None, windows: bool | None = None) -> list[str]:
    """The CUDA and cuDNN libraries ONNX Runtime's CUDA provider links against, for CUDA ``major``."""
    if major is None:
        return []
    win = _windows() if windows is None else windows
    cufft = major - 1  # cuFFT's major trails CUDA's (CUDA 12: cufft64_11, CUDA 13: cufft64_12)
    if win:
        return [f"cudart64_{major}.dll", f"cublasLt64_{major}.dll", f"cublas64_{major}.dll", f"cufft64_{cufft}.dll",
                "cudnn64_9.dll"]  # fmt: skip
    return [f"libcudart.so.{major}", f"libcublasLt.so.{major}", f"libcublas.so.{major}", f"libcufft.so.{cufft}",
            "libcudnn.so.9"]  # fmt: skip


def ct2_cuda_libraries(windows: bool | None = None) -> list[str]:
    """What CTranslate2 4.x loads on CUDA: CUDA 12's cuBLAS (at the first matrix product) and cuDNN 9."""
    if _windows() if windows is None else windows:
        return ["cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"]
    return ["libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9"]


def unloadable(names: list[str], load: Callable[[str], bool] | None = None) -> list[str]:
    """Of ``names``, the libraries this process cannot load."""
    loader = load or _load
    return [n for n in names if not loader(n)]


_capture_lock = threading.Lock()


@contextlib.contextmanager
def native_stderr() -> Iterator[list[str]]:
    """Catch what native code writes to the process's standard error (file descriptor 2) inside the block, such as
    ONNX Runtime's "Error loading ... which is missing" line; it is passed on to the real standard error afterwards.
    The list it yields holds the text once the block has ended (empty where file descriptors cannot be swapped)."""
    out: list[str] = []
    with _capture_lock:
        try:
            saved = os.dup(2)
        except OSError:
            yield out
            return
        try:
            with tempfile.TemporaryFile() as tmp:
                with contextlib.suppress(Exception):
                    sys.stderr.flush()
                os.dup2(tmp.fileno(), 2)
                try:
                    yield out
                finally:
                    with contextlib.suppress(Exception):
                        sys.stderr.flush()
                    os.dup2(saved, 2)
                    tmp.seek(0)
                    data = tmp.read()
                    with contextlib.suppress(OSError):
                        view = memoryview(data)
                        while view:
                            view = view[os.write(2, view) :]
                    out.append(data.decode(errors="replace"))
        finally:
            os.close(saved)


# -- a model that asked for a GPU and runs on the CPU ---------------------------------------------------------
_SHORT = {
    "CUDAExecutionProvider": "CUDA",
    "cuda": "CUDA",
    "DmlExecutionProvider": "DirectML",
    "CoreMLExecutionProvider": "Core ML",
    "ROCMExecutionProvider": "ROCm",
    "MIGraphXExecutionProvider": "MIGraphX",
    "OpenVINOExecutionProvider": "OpenVINO",
}


@dataclass
class ProviderFallback:
    """An accelerator a runtime was asked for and did not start: the work runs on the CPU instead."""

    #: "onnxruntime" or "ctranslate2"
    runtime: str
    #: what was asked for: an ONNX Runtime provider ("CUDAExecutionProvider") or a CTranslate2 device ("cuda")
    requested: str
    #: what it runs on
    running: str = "cpu"
    #: the libraries the loader named, or that could not be loaded
    missing: list[str] = field(default_factory=list)
    #: the runtime's own words
    detail: str = ""
    #: how to fix it
    recipe: str = ""
    #: the graph of a multi-graph model ("encoder", ...), when it is one
    role: str | None = None

    def message(self) -> str:
        name = {"onnxruntime": "ONNX Runtime", "ctranslate2": "CTranslate2"}.get(self.runtime, self.runtime)
        asked = _SHORT.get(self.requested, self.requested)
        why = (
            f" ({', '.join(self.missing)} {'is' if len(self.missing) == 1 else 'are'} missing)" if self.missing else ""
        )
        if not why and self.detail:
            why = f" ({self.detail.strip().splitlines()[-1][:200]})"
        part = f" for its {self.role}" if self.role else ""
        fix = f"; fix: {self.recipe}" if self.recipe else ""
        return f"{name} could not start {asked}{part}{why} and runs on the CPU{fix}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "requested": self.requested,
            "running": self.running,
            "missing": list(self.missing),
            "role": self.role,
            "recipe": self.recipe or None,
            "message": self.message(),
        }


@dataclass
class Watch:
    """What the sessions opened inside :func:`watch` got: fallbacks, and how many started their accelerator."""

    fallbacks: list[ProviderFallback] = field(default_factory=list)
    accelerated: int = 0

    @property
    def whole(self) -> bool:
        """Every session that asked for an accelerator runs on the CPU."""
        return bool(self.fallbacks) and self.accelerated == 0

    def notice(self) -> str | None:
        seen: list[str] = []
        for fb in self.fallbacks:
            text = fb.message()
            if text not in seen:
                seen.append(text)
        return "; ".join(seen) or None


_local = threading.local()


@contextlib.contextmanager
def watch() -> Iterator[Watch]:
    """Collect the fallbacks recorded in this thread inside the block (a model's load)."""
    w = Watch()
    stack: list[Watch] = getattr(_local, "stack", [])
    _local.stack = [*stack, w]
    try:
        yield w
    finally:
        _local.stack = stack


def record(fallback: ProviderFallback | None = None, accelerated: bool = False) -> None:
    """Note a session's outcome for the :func:`watch` blocks of this thread."""
    for w in getattr(_local, "stack", []):
        if fallback is not None:
            w.fallbacks.append(fallback)
        if accelerated:
            w.accelerated += 1


# -- bolo doctor ------------------------------------------------------------------------------------------------
def check_ort_cuda() -> tuple[bool, str]:
    """Open a tiny in-memory graph on ONNX Runtime's CUDA provider and run it once. ``(ok, what happened)``."""
    try:
        import onnxruntime as ort
    except Exception as exc:
        return False, f"onnxruntime cannot be imported ({type(exc).__name__}: {exc})"
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False, f"this ONNX Runtime has no CUDA provider (it is the CPU build): {ORT_CUDA_RECIPE}"
    import numpy as np

    from bolo.hardware import benchmark_model

    prepare_ort_cuda()
    options = ort.SessionOptions()
    with native_stderr() as said:
        try:
            session = ort.InferenceSession(benchmark_model(), sess_options=options, providers=["CUDAExecutionProvider"])
        except Exception as exc:
            session, error = None, f"{type(exc).__name__}: {exc}"
        else:
            error = ""
    text = "".join(said) + error
    if session is not None and "CUDAExecutionProvider" in session.get_providers():
        try:
            session.run(None, {
                "x": np.zeros((1, 64, 96, 96), np.float32), "w": np.zeros((64, 64, 3, 3), np.float32),
                "a": np.zeros((512, 512), np.float32),
            })  # fmt: skip
        except Exception as exc:
            return False, f"the CUDA provider started but a test graph failed: {type(exc).__name__}: {exc}"
        return True, f"CUDA provider loads and runs (CUDA {ort_cuda_major() or '?'} build)"
    missing = missing_libraries(text) or unloadable(ort_cuda_libraries(ort_cuda_major()))
    why = f"{', '.join(missing)} missing" if missing else (text.strip().splitlines() or ["no reason given"])[-1][:200]
    return False, f"the CUDA provider does not start ({why}): {CUDA_RECIPE if _windows() else ORT_CUDA_RECIPE}"


def check_ct2_cuda() -> tuple[bool, str]:
    """CTranslate2's CUDA devices and the CUDA 12 libraries it loads. ``(ok, what happened)``."""
    try:
        import ctranslate2
    except Exception as exc:
        return False, f"ctranslate2 cannot be imported ({type(exc).__name__}: {exc})"
    try:
        count = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:
        return False, f"ctranslate2 cannot count CUDA devices ({type(exc).__name__}: {exc})"
    if count == 0:
        return False, "ctranslate2 sees no CUDA device (no NVIDIA driver, or a build without CUDA)"
    prepare_ct2_cuda()
    missing = unloadable(ct2_cuda_libraries())
    if missing:
        return (
            False,
            f"ctranslate2 sees {count} CUDA device(s) but {', '.join(missing)} cannot be loaded: {CT2_CUDA_RECIPE}",
        )
    return (
        True,
        f"ctranslate2 {getattr(ctranslate2, '__version__', '')} sees {count} CUDA device(s); cuBLAS 12 and cuDNN 9 load",
    )


def doctor_wants_cuda(device: str | None, nvidia_gpu: bool) -> bool:
    """``bolo doctor`` tests CUDA when ``device`` is ``cuda`` (``gpu``), or ``auto`` on a machine with an NVIDIA GPU."""
    name = (device or "cpu").strip().lower().partition(":")[0]
    return name in ("cuda", "gpu") or (name == "auto" and nvidia_gpu)


def doctor_lines(
    ort_check: Callable[[], tuple[bool, str]] | None = None,
    ct2_check: Callable[[], tuple[bool, str]] | None = None,
    have_ort: bool | None = None,
    have_ct2: bool | None = None,
) -> list[str]:
    """``bolo doctor``'s CUDA section: one line per runtime that is installed."""
    import importlib.util

    lines = ["cuda:"]
    if have_ort is None:
        have_ort = importlib.util.find_spec("onnxruntime") is not None
    if have_ct2 is None:
        have_ct2 = importlib.util.find_spec("ctranslate2") is not None
    for name, present, check in (
        ("onnxruntime", have_ort, ort_check or check_ort_cuda),
        ("faster-whisper", have_ct2, ct2_check or check_ct2_cuda),
    ):
        if not present:
            continue
        ok, text = check()
        lines.append(f"  {'ok ' if ok else '-- '}{name:<15} {text}")
    if len(lines) == 1:
        lines.append("  neither onnxruntime nor ctranslate2 is installed")
    elif any(line.startswith("  -- ") for line in lines):
        lines.append(f"  the Windows recipe: {DOCS}")
    return lines
