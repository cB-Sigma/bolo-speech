"""ONNX Runtime execution providers for bolo's ``device`` setting.

The engines that run ONNX graphs themselves (Nemotron diarization, DiCoW, the SNAC decoder of the
Orpheus-family voices, the end-of-turn model and in-process speech-to-speech) ask here which
providers to use. A device this ONNX Runtime build cannot serve falls back to the CPU with one
warning; the CPU is always last in the list, so operators a provider lacks still run.

They open their sessions through :func:`create_session`, which readies CUDA's libraries first (bolo.cuda_libs) and
then checks what the session got: ONNX Runtime falls back to the CPU on its own when a provider does not start (a
CUDA DLL missing, say), with nothing but a line on standard error, so bolo logs it with the missing library and
records it for the model's load (its notice, and its placement moved to the CPU). Every session it opens asks for
exact int8 products on x86 CPUs without VNNI (:func:`exact_int8`).
"""

from __future__ import annotations

import contextlib
import functools
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CPU = "CPUExecutionProvider"

#: ``device`` values (before an optional ``:index``) and the provider each one asks for
PROVIDER_BY_DEVICE = {
    "cuda": "CUDAExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "mps": "CoreMLExecutionProvider",
    "metal": "CoreMLExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
}

#: tried in this order for ``device = "auto"``
AUTO_ORDER = ("CUDAExecutionProvider", "ROCMExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")

_WITH_DEVICE_ID = {
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
}
_warned: set[str] = set()


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return [CPU]
    return list(ort.get_available_providers())


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(message)


def default_dml_index() -> int | None:
    """DirectML's ``device_id`` for ``dml`` with no number: Windows' high-performance adapter (bolo.winhw)."""
    from bolo import winhw

    return winhw.default_dml_index()


def _dml(index: int | None) -> Any:
    return ("DmlExecutionProvider", {"device_id": index}) if index is not None else "DmlExecutionProvider"


def providers_for(device: str | None, available: list[str] | None = None) -> list[Any]:
    """The ``providers`` argument for ``onnxruntime.InferenceSession`` on ``device``.

    ``cpu`` (the default) gives the CPU only; ``cuda``, ``cuda:1``, ``rocm``, ``dml``, ``coreml``
    (also ``mps`` and ``metal``) or ``openvino`` give that provider first and the CPU after it;
    ``auto`` takes the first accelerator this build has. ``dml`` with no number runs on Windows'
    high-performance adapter (the discrete GPU of a hybrid laptop), not on DXGI's first adapter, which
    is DirectML's own default and on such a laptop the integrated GPU; ``dml:N`` pins adapter N.
    """
    kind = (device or "cpu").strip().lower()
    name, _, index = kind.partition(":")
    have = available if available is not None else available_providers()
    if name in ("", "cpu"):
        return [CPU]
    if name == "auto":
        chosen = next((p for p in AUTO_ORDER if p in have), None)
        if chosen == "DmlExecutionProvider":
            return [_dml(default_dml_index()), CPU]
        return [chosen, CPU] if chosen else [CPU]
    wanted = PROVIDER_BY_DEVICE.get(name)
    if wanted is None:
        _warn_once(kind, f"device {device!r} is not known to ONNX Runtime engines; they run on the CPU")
        return [CPU]
    if wanted not in have:
        hint = cuda_hint(have) if wanted == "CUDAExecutionProvider" else ""
        _warn_once(
            kind,
            f"device {device!r} asks for {wanted}, which this ONNX Runtime build does not have "
            f"(it has {', '.join(have)}); the ONNX Runtime engines run on the CPU" + (f": {hint}" if hint else ""),
        )
        from bolo import cuda_libs

        recipe = {"CUDAExecutionProvider": cuda_hint(have), "DmlExecutionProvider": DIRECTML_RECIPE}
        cuda_libs.record(
            cuda_libs.ProviderFallback(
                "onnxruntime", wanted, CPU, [], f"this ONNX Runtime build has no {wanted}", recipe.get(wanted, "")
            )
        )
        return [CPU]
    if index.isdigit() and wanted in _WITH_DEVICE_ID:
        return [(wanted, {"device_id": int(index)}), CPU]
    if wanted == "DmlExecutionProvider":
        return [_dml(default_dml_index()), CPU]
    return [wanted, CPU]


#: the ``device`` that reaches each provider of :data:`AUTO_ORDER`
DEVICE_BY_PROVIDER = {
    "CUDAExecutionProvider": "cuda",
    "ROCMExecutionProvider": "rocm",
    "DmlExecutionProvider": "dml",
    "CoreMLExecutionProvider": "coreml",
}


def auto_device(found: str | None, available: list[str] | None = None) -> str:
    """What ``device = "auto"`` means for the engines that open ONNX Runtime sessions themselves.

    ``found`` is the accelerator bolo.accel found (``cuda:0``, ``coreml`` or None). The answer is the first provider
    of :data:`AUTO_ORDER` this ONNX Runtime build has, as a ``device``: ``cuda`` only when an NVIDIA GPU was found
    (onnxruntime-gpu on a machine without one accelerates nothing), ``dml`` for onnxruntime-directml (Windows'
    high-performance adapter, as :func:`providers_for` runs ``dml``), ``rocm``, ``coreml``; ``cpu`` when the build
    has none. So onnxruntime-directml on a laptop with an NVIDIA GPU runs on DirectML, not on the CPU for want of a
    CUDA provider it does not have; engines with their own CUDA (faster-whisper, sherpa-onnx) still get ``cuda``."""
    have = available if available is not None else available_providers()
    for provider in AUTO_ORDER:
        if provider not in have:
            continue
        if provider == "CUDAExecutionProvider" and not (found or "").startswith("cuda"):
            continue
        return DEVICE_BY_PROVIDER[provider]
    return "cpu"


def cuda_hint(have: list[str]) -> str:
    """How to get ONNX Runtime's CUDA provider, given the providers this build has: onnxruntime-directml (it has
    DirectML) is told that ``dml`` (or ``auto``) uses its GPU, and how to swap it for onnxruntime-gpu."""
    if "DmlExecutionProvider" in have:
        return (
            'this is onnxruntime-directml: device = "dml" (or "auto") runs them on the GPU through DirectML; for CUDA '
            'instead: pip uninstall -y onnxruntime-directml, then pip install "onnxruntime-gpu[cuda,cudnn]"'
        )
    packages = installed_ort_packages()
    if packages:
        return no_cuda_provider(packages)
    from bolo import cuda_libs

    return cuda_libs.ORT_CUDA_RECIPE


def _name(provider: Any) -> str:
    return str(provider[0] if isinstance(provider, (tuple, list)) else provider)


#: ONNX Runtime session option that makes int8 matrix products exact on x86 CPUs without VNNI (:func:`session_options`)
X64_QUANT_PRECISION = "session.x64quantprecision"


def exact_int8(options: Any) -> Any:
    """``options`` with :data:`X64_QUANT_PRECISION` set, so int8 graphs give the same output on every x86 CPU.

    Quantized graphs multiply uint8 activations by int8 weights (MatMulInteger, ConvInteger and their fused forms,
    and the QLinear operators ONNX Runtime makes of QuantizeLinear / DequantizeLinear pairs). On x86 CPUs with AVX2
    but without VNNI (AVX-512 VNNI or AVX-VNNI: most AMD Ryzen before Zen 4, Intel before Ice Lake / Alder Lake),
    ONNX Runtime's uint8 x int8 kernel adds pairs of products in 16 bits, which saturate: a pair can reach
    2 x 255 x 127 = 64770 > 32767. That emptied DiCoW's transcripts and moved the end-of-turn model's probabilities by
    up to 0.57 on such CPUs. With this option ONNX Runtime converts the int8 weights to uint8 at load time there
    (uint8 x uint8 has no such overflow); on CPUs with VNNI, on other architectures and for float graphs it changes
    nothing. A stand-in without ``add_session_config_entry`` (tests) is returned as it is."""
    add = getattr(options, "add_session_config_entry", None)
    if add is not None:
        add(X64_QUANT_PRECISION, "1")
    return options


def session_options() -> Any:
    """New ``onnxruntime.SessionOptions`` for bolo's own sessions, with exact int8 products (:func:`exact_int8`)."""
    import onnxruntime as ort

    return exact_int8(ort.SessionOptions())


def create_session(
    model: Any, sess_options: Any = None, providers: list[Any] | None = None, *, role: str | None = None
) -> Any:
    """``onnxruntime.InferenceSession(model, sess_options, providers)``, readied and checked.

    The options always ask for exact int8 products (:func:`exact_int8`; new options from :func:`session_options`
    when none are given). Before a session that asks for CUDA, the CUDA libraries of NVIDIA's pip wheels are made loadable (once:
    :func:`bolo.cuda_libs.prepare_ort_cuda`). After it, an accelerator that was asked for and is not among the
    session's providers is a :class:`~bolo.cuda_libs.ProviderFallback`: logged once with the library ONNX Runtime
    said was missing, and recorded for the load in progress (:func:`bolo.cuda_libs.watch`)."""
    import onnxruntime as ort

    from bolo import cuda_libs

    sess_options = session_options() if sess_options is None else exact_int8(sess_options)
    chosen = list(providers or [CPU])
    asked = [n for n in (_name(p) for p in chosen) if n != CPU]
    if not asked:
        return ort.InferenceSession(model, sess_options=sess_options, providers=chosen)
    if "CUDAExecutionProvider" in asked:
        cuda_libs.prepare_ort_cuda()
    with cuda_libs.native_stderr() as said:
        session = ort.InferenceSession(model, sess_options=sess_options, providers=chosen)
    got = getattr(session, "get_providers", None)
    if got is None:  # a stand-in
        return session
    running = [str(p) for p in got()]
    lost = [p for p in asked if p not in running]
    if not lost:
        cuda_libs.record(accelerated=True)
        return session
    text = "".join(said)
    for wanted in lost:
        missing = cuda_libs.missing_libraries(text)
        recipe = ""
        if wanted == "CUDAExecutionProvider":
            if not missing:
                missing = cuda_libs.unloadable(cuda_libs.ort_cuda_libraries(cuda_libs.ort_cuda_major()))
            recipe = cuda_libs.CUDA_RECIPE
        elif wanted == "DmlExecutionProvider":
            recipe = DIRECTML_RECIPE
        lines = [ln for ln in text.splitlines() if wanted.removesuffix("ExecutionProvider") in ln or "rror" in ln]
        fallback = cuda_libs.ProviderFallback(
            "onnxruntime", wanted, running[0] if running else CPU, missing, "\n".join(lines[-3:]), recipe, role
        )
        _warn_once(f"fallback:{wanted}:{','.join(missing)}", fallback.message())
        cuda_libs.record(fallback)
    return session


#: how to get a sherpa-onnx that runs on CUDA
SHERPA_CUDA_RECIPE = (
    "install sherpa-onnx's CUDA build in place of the one from PyPI (https://k2-fsa.github.io/sherpa/onnx/cuda.html)"
)


def sherpa_build(package_dir: Path | None, versions: list[str] | None = None) -> bool | None:
    """Whether a sherpa-onnx installation can use CUDA: True for a GPU build, False for a CPU build, None when it
    cannot be told.

    sherpa-onnx carries its own ONNX Runtime library in its package's ``lib`` folder (``onnxruntime.dll``,
    ``libonnxruntime.so``); its CUDA builds have ONNX Runtime's CUDA provider next to it
    (``onnxruntime_providers_cuda.dll``, ``libonnxruntime_providers_cuda.so``) and a version such as ``1.12.0+cuda``.
    The wheels on PyPI are CPU builds: asked for ``cuda``, they say "Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON"
    and run on the CPU."""
    if any("+cuda" in (v or "").lower() for v in versions or []):
        return True
    if package_dir is None:
        return None
    names: list[str] = []
    for folder in (package_dir / "lib", package_dir, package_dir.parent / f"{package_dir.name}.libs"):
        with contextlib.suppress(OSError):
            if folder.is_dir():
                names += [f.name.lower() for f in folder.iterdir()]
    if any("onnxruntime_providers_cuda" in n for n in names):
        return True
    if any(n.startswith(("onnxruntime", "libonnxruntime")) for n in names):
        return False
    return None


@functools.lru_cache(maxsize=1)
def sherpa_gpu_build() -> bool | None:
    """:func:`sherpa_build` for the installed sherpa-onnx (None when it is not installed or cannot be told). Read
    from the files, without importing it."""
    import importlib.util
    from importlib import metadata

    try:
        spec = importlib.util.find_spec("sherpa_onnx")
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    origin = spec.origin or next(iter(spec.submodule_search_locations or []), None)
    package_dir = (Path(origin).parent if spec.origin else Path(origin)) if origin else None
    versions: list[str] = []
    for dist in ("sherpa-onnx", "sherpa-onnx-core"):
        with contextlib.suppress(metadata.PackageNotFoundError):
            versions.append(metadata.version(dist))
    return sherpa_build(package_dir, versions)


def sherpa_cpu_notice(model_id: str | None = None) -> str:
    """Why a sherpa-onnx model asked to run on CUDA runs on the CPU (a CPU build of sherpa-onnx)."""
    from importlib import metadata

    version = ""
    with contextlib.suppress(metadata.PackageNotFoundError):
        version = " " + metadata.version("sherpa-onnx")
    who = f"'{model_id}' runs" if model_id else "its models run"
    return (
        f"the installed sherpa-onnx{version} is a CPU build (no CUDA provider in its package): {who} on the CPU; "
        f"for CUDA, {SHERPA_CUDA_RECIPE}"
    )


def sherpa_provider(device: str | None) -> str:
    """sherpa-onnx's ``provider`` string for ``device`` (bolo uses ``cpu``, ``cuda`` and ``coreml``).

    sherpa-onnx knows a ``directml`` provider only when it was built with DirectML, and its wheels on PyPI are not
    (asked for it, they fall back to the CPU saying "DirectML is for Windows only"), so ``device = "dml"`` runs
    sherpa-onnx engines on the CPU, with one notice. ``cuda`` with a CPU build of sherpa-onnx (:func:`sherpa_gpu_build`,
    the wheels on PyPI) is the CPU too, with one notice."""
    name = (device or "cpu").strip().lower().partition(":")[0]
    if name in ("cuda", "gpu"):
        if sherpa_gpu_build() is False:
            _warn_once("sherpa-cpu-build", sherpa_cpu_notice())
            return "cpu"
        return "cuda"
    if name in ("coreml", "mps", "metal"):
        return "coreml"
    if name in ("dml", "directml"):
        _warn_once(
            "sherpa-dml",
            "device 'dml': sherpa-onnx's wheels have no DirectML provider, so its engines (most speech recognition "
            "and voices) run on the CPU; DirectML serves the engines that use ONNX Runtime directly",
        )
    return "cpu"


#: the PyPI distributions that each install the same ``onnxruntime`` module
ORT_PACKAGES = (
    "onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino", "onnxruntime-qnn",
    "onnxruntime-rocm", "onnxruntime-silicon",
)  # fmt: skip

DIRECTML_RECIPE = (
    "pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml, then pip install onnxruntime-directml "
    "(after bolo and its extras: installing faster-whisper again brings plain onnxruntime back)"
)


def installed_ort_packages(versions: dict[str, str | None] | None = None) -> dict[str, str]:
    """The ONNX Runtime distributions installed here, with their versions."""
    if versions is None:
        from importlib import metadata

        versions = {}
        for name in ORT_PACKAGES:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
    return {name: v for name, v in versions.items() if v}


#: what each ONNX Runtime distribution is, for messages that name it
_BUILD = {
    "onnxruntime": "the CPU build", "onnxruntime-directml": "the DirectML build",
    "onnxruntime-openvino": "the OpenVINO build", "onnxruntime-qnn": "the Qualcomm QNN build",
    "onnxruntime-rocm": "the ROCm build", "onnxruntime-silicon": "the Apple silicon build",
}  # fmt: skip


def no_cuda_provider(packages: dict[str, str]) -> str:
    """Why this ONNX Runtime has no CUDA provider, naming the installed distribution, and how to get one."""
    gpu = 'pip install "onnxruntime-gpu[cuda,cudnn]"'
    if len(packages) > 1 or "onnxruntime-gpu" in packages:
        listed = ", ".join(f"{name} {version}" for name, version in sorted(packages.items()))
        names = " ".join(sorted(packages))
        return (
            f"the installed ONNX Runtime ({listed}) has no CUDA provider (these packages overwrite each other's "
            f"files): pip uninstall -y {names}, then {gpu}"
        )
    if not packages:
        return f"onnxruntime is not installed: {gpu}"
    name, version = next(iter(packages.items()))
    what = f"{name} {version} ({_BUILD.get(name, 'not a CUDA build')})"
    if name == "onnxruntime-directml":
        return (
            f'this ONNX Runtime is {what}, which has no CUDA provider; device = "dml" (or "auto") runs the ONNX '
            f"Runtime engines on the GPU through DirectML, or for CUDA: pip uninstall -y {name}, then {gpu}"
        )
    return f"this ONNX Runtime is {what}, which has no CUDA provider: pip uninstall -y {name}, then {gpu}"


def cuda_dropped(
    device: str | None, nvidia_gpu: bool, packages: dict[str, str], providers: list[str] | None
) -> str | None:
    """The one-line warning for an NVIDIA machine whose ONNX Runtime lost its CUDA provider: ``device`` is ``cuda``
    (``gpu``), or ``auto`` with an NVIDIA GPU and no other accelerator in this ONNX Runtime (with
    onnxruntime-directml, ``auto`` runs on DirectML, which is not a loss). That happens when ``uv tool upgrade`` or
    ``uv tool install --force`` reinstalls plain onnxruntime over onnxruntime-gpu. None when there is nothing to
    say (no ONNX Runtime at all, or one with CUDA)."""
    name = (device or "cpu").strip().lower().partition(":")[0]
    if not packages or providers is None or "CUDAExecutionProvider" in providers:
        return None
    if name == "auto":
        if not nvidia_gpu or auto_device("cuda:0", providers) != "cpu":
            return None
    elif name not in ("cuda", "gpu"):
        return None
    return (
        f"device {device!r} on a machine with an NVIDIA GPU, but {no_cuda_provider(packages)}; until then the ONNX "
        "Runtime engines run on the CPU (`uv tool upgrade` and `uv tool install --force` put plain onnxruntime back: "
        'docs/configuration.md, "CUDA on Windows")'
    )


def ort_problems(device: str | None, packages: dict[str, str], providers: list[str] | None) -> list[str]:
    """What ``bolo doctor`` warns about: several ONNX Runtime packages sharing the one module, and a device whose
    provider this ONNX Runtime lacks (``dml`` without onnxruntime-directml, ``cuda`` without onnxruntime-gpu)."""
    out: list[str] = []
    if len(packages) > 1:
        listed = ", ".join(f"{name} {version}" for name, version in sorted(packages.items()))
        out.append(
            f"several ONNX Runtime packages are installed ({listed}); they all install the same `onnxruntime` "
            "module and overwrite each other's files: keep one (uninstall all of them, then install the one you want)"
        )
    name = (device or "cpu").strip().lower().partition(":")[0]
    wanted = PROVIDER_BY_DEVICE.get(name)
    if wanted and providers is not None and wanted not in providers and packages:
        if wanted == "DmlExecutionProvider":
            out.append(f"device {device!r} needs DirectML in ONNX Runtime, which this one lacks: {DIRECTML_RECIPE}")
        elif wanted == "CUDAExecutionProvider":
            out.append(f"device {device!r}: {no_cuda_provider(packages)} (faster-whisper brings its own CUDA support)")
        else:
            out.append(f"device {device!r}: this ONNX Runtime has no {wanted}")
    return out
