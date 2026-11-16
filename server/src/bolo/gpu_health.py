"""Is the GPU still there? A health check for the accelerator bolo's models run on.

A laptop's NVIDIA GPU can drop off the bus under load ("Unable to determine the device handle for GPU0: ...: GPU
is lost. Reboot the system to recover this GPU" from nvidia-smi, then "No devices were found"). The runtimes do
not always fail when that happens: a CUDA llama-server went on answering, slower and slower, burning CPU, while
placement still said ``cuda:0``. So while models are on a GPU, bolo asks the driver every
``gpu_health_interval_s`` seconds (and right after a request whose real-time factor is far worse than that model's
own recent median), and treats a CUDA (or Vulkan) error a managed server prints on its console the same way:

* NVML (``pynvml``): ``nvmlDeviceGetHandleByIndex`` / ``nvmlDeviceGetMemoryInfo`` failing with
  ``NVML_ERROR_GPU_IS_LOST``, or no device at all where bolo has GPU models;
* else nvidia-smi: its output saying "GPU is lost", "Unable to determine the device handle" or
  "No devices were found" (where bolo has GPU models).

Once lost, the GPU stays lost for this server's life (the driver needs a reboot). bolo.lifecycle then marks the
models placed on a GPU as degraded and, with ``gpu_lost_fallback = "cpu"`` (the default), loads them again on
the CPU (managed llama-server / audio.cpp on their CPU build); with ``"fail"`` their requests fail instead.

Checks never raise; tests pass a fake NVML module and a fake command runner.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import re
import shutil
import statistics
import subprocess
import threading
import time
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("bolo.gpu_health")

#: NVML's return code for a GPU that fell off the bus (nvml.h)
NVML_ERROR_GPU_IS_LOST = 15
#: what nvidia-smi prints for a lost GPU (lower case)
SMI_LOST = ("gpu is lost", "unable to determine the device handle")
SMI_NONE = "no devices were found"
#: a server's console line that means its GPU stopped working. "out of memory" is not one of them: a model too
#: big for the GPU is a placement problem, and the server's crash handling covers it. ggml prints
#: "ggml_cuda_init: found 1 CUDA devices" on every start: only its failures count.
GPU_ERROR_RE = re.compile(
    r"CUDA error|cudaError\w+|ggml_cuda\w*: .*\bfail(?:ed|ure)?\b|GPU is lost|ErrorDeviceLost|VK_ERROR_DEVICE_LOST",
    re.IGNORECASE,
)
#: the advice shown wherever a lost GPU is reported
REBOOT = "reboot to recover"

#: a request this many times slower (real-time factor) than the model's recent median asks for a check now
SLOW_FACTOR = 2.0
#: real-time factors kept per model, and how many are needed before one counts as "far worse"
RTF_WINDOW = 20
RTF_MIN_SAMPLES = 2
#: at most one extra check (after a slow request) this often
KICK_MIN_INTERVAL_S = 5.0

Runner = Callable[[list[str], float], "tuple[int | None, str] | None"]


def gpu_error_line(line: str) -> bool:
    """A server's console line that reports its GPU failing (CUDA error, Vulkan device lost)."""
    return bool(GPU_ERROR_RE.search(line)) and "out of memory" not in line.lower()


@dataclass
class GpuHealth:
    #: "ok", "lost", "none" (no NVIDIA GPU found, where none was expected), "unknown" (nothing answered) or
    #: "unchecked" (no GPU model has been loaded, so nothing was asked)
    state: str = "unchecked"
    #: what was seen ("NVML: GPU is lost", "nvidia-smi: No devices were found", "llama-server: CUDA error ...")
    reason: str | None = None
    #: "nvml", "nvidia-smi", or the server that reported it ("llama-server:svara-tts-v1-llamacpp")
    source: str | None = None
    #: NVIDIA GPUs that answered (state "ok")
    gpus: int | None = None
    checked_at: float | None = None
    #: when the loss was first seen (epoch seconds)
    lost_at: float | None = None
    #: what bolo does about it: "cpu" (models moved to the CPU) or "fail" (their requests fail)
    fallback: str | None = None
    #: models moved off the lost GPU, and models still on it (degraded)
    moved: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)

    @property
    def lost(self) -> bool:
        return self.state == "lost"

    def notice(self) -> str | None:
        """The one-line notice for the log, ``bolo ps``, the console and doctor."""
        if not self.lost:
            return None
        then = (
            "bolo is using the CPU"
            if self.fallback != "fail"
            else "models placed on it fail (gpu_lost_fallback = fail)"
        )
        return f"GPU LOST ({self.reason or 'the driver lost the GPU'}): {REBOOT}; {then}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "lost": self.lost,
            "reason": self.reason,
            "source": self.source,
            "gpus": self.gpus,
            "checked_at": self.checked_at,
            "lost_at": self.lost_at,
            "fallback": self.fallback if self.lost else None,
            "moved_to_cpu": list(self.moved),
            "degraded": list(self.degraded),
            "notice": self.notice(),
        }


def _run(argv: list[str], timeout: float) -> tuple[int | None, str] | None:
    """(exit code, standard output and error) of a command; None when it cannot be started. A command that does
    not answer in time is killed and reads (None, "")."""
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except (OSError, ValueError):
        return None
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        with contextlib.suppress(Exception):
            proc.communicate(timeout=5.0)
        return None, ""
    return proc.returncode, (out or b"").decode(errors="replace")


def _lost_error(exc: BaseException) -> bool:
    return getattr(exc, "value", None) == NVML_ERROR_GPU_IS_LOST or "gpu is lost" in str(exc).lower()


def _check_nvml(nv: Any, expect_gpu: bool) -> GpuHealth | None:
    """NVML's answer, or None when NVML cannot tell (not initialised, an unrelated error)."""
    try:
        nv.nvmlInit()
    except Exception as exc:
        if _lost_error(exc):
            return GpuHealth("lost", f"NVML: {exc}", "nvml")
        log.debug("NVML is not usable for the health check: %s", exc)
        return None
    try:
        count = int(nv.nvmlDeviceGetCount())
        if count == 0:
            if expect_gpu:
                return GpuHealth("lost", "NVML finds no NVIDIA GPU (the driver lost it)", "nvml", gpus=0)
            return GpuHealth("none", "NVML finds no NVIDIA GPU", "nvml", gpus=0)
        for i in range(count):
            handle = nv.nvmlDeviceGetHandleByIndex(i)
            nv.nvmlDeviceGetMemoryInfo(handle)
        return GpuHealth("ok", None, "nvml", gpus=count)
    except Exception as exc:
        if _lost_error(exc):
            return GpuHealth("lost", f"NVML: {exc}", "nvml")
        log.debug("NVML health query failed: %s", exc)
        return None
    finally:
        with contextlib.suppress(Exception):
            nv.nvmlShutdown()


def _check_smi(exe: str, runner: Runner, expect_gpu: bool) -> GpuHealth:
    answer = runner([exe, "--query-gpu=index,name,memory.used", "--format=csv,noheader"], 10.0)
    if answer is None:
        return GpuHealth("unknown", "nvidia-smi could not be started", "nvidia-smi")
    code, text = answer
    if code is None:
        return GpuHealth("unknown", "nvidia-smi did not answer within 10 s", "nvidia-smi")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lower = text.lower()
    if any(k in lower for k in SMI_LOST):
        line = next((ln for ln in lines if any(k in ln.lower() for k in SMI_LOST)), lines[0] if lines else "")
        return GpuHealth("lost", f"nvidia-smi: {line}", "nvidia-smi")
    if SMI_NONE in lower:
        if expect_gpu:
            return GpuHealth("lost", "nvidia-smi: No devices were found", "nvidia-smi", gpus=0)
        return GpuHealth("none", "nvidia-smi: No devices were found", "nvidia-smi", gpus=0)
    if code == 0 and lines:
        return GpuHealth("ok", None, "nvidia-smi", gpus=len(lines))
    return GpuHealth("unknown", f"nvidia-smi exited with {code}" + (f": {lines[-1]}" if lines else ""), "nvidia-smi")


_UNSET: Any = object()


def check(
    *,
    expect_gpu: bool = True,
    nvml: Any = _UNSET,
    runner: Runner | None = None,
    which: Callable[[str], str | None] | None = None,
) -> GpuHealth:
    """Ask the NVIDIA driver whether its GPUs are there: NVML first, else nvidia-smi. ``expect_gpu``: bolo has
    models on a GPU, so finding none means the driver lost it. Never raises."""
    try:
        if nvml is _UNSET:
            try:
                import pynvml  # type: ignore[import-not-found,import-untyped,unused-ignore]

                nvml = pynvml
            except Exception:
                nvml = None
        found = _check_nvml(nvml, expect_gpu) if nvml is not None else None
        if found is None or found.state != "ok":
            # NVML could not tell, or says lost: nvidia-smi gives the driver's own words (and a second opinion)
            exe = (which or shutil.which)("nvidia-smi")
            if exe:
                smi = _check_smi(exe, runner or _run, expect_gpu)
                if found is None or (smi.lost and not found.lost):
                    found = smi
        if found is None:
            found = GpuHealth("unknown", "neither NVML nor nvidia-smi is available", None)
    except Exception as exc:  # pragma: no cover - a health check must never break the server
        found = GpuHealth("unknown", f"{type(exc).__name__}: {exc}", None)
    found.checked_at = time.time()
    return found


# -- the monitor ------------------------------------------------------------------------------------------------
_monitors: weakref.WeakSet[GpuMonitor] = weakref.WeakSet()


def report_gpu_error(source: str, line: str) -> None:
    """A managed server printed a GPU error (called from its output thread): every running monitor hears it."""
    for monitor in list(_monitors):
        monitor.report(source, line)


class GpuMonitor:
    """Checks the GPU while models are on it: every ``interval_s`` seconds, after a request far slower than that
    model's recent median, and when a managed server reports a GPU error. On the first loss it calls ``on_lost``
    once; the loss then stays (the GPU needs a reboot)."""

    def __init__(
        self,
        *,
        interval_s: float,
        has_gpu_models: Callable[[], bool],
        on_lost: Callable[[GpuHealth], Awaitable[None]],
        checker: Callable[[], GpuHealth] | None = None,
        fallback: str = "cpu",
    ) -> None:
        self.interval_s = interval_s
        self._has_gpu_models = has_gpu_models
        self._on_lost = on_lost
        self._checker = checker or check
        self.health = GpuHealth(fallback=fallback)
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._kick: asyncio.Event | None = None
        self._pending: GpuHealth | None = None
        self._lock = threading.Lock()
        self._rtf: dict[str, collections.deque[float]] = {}
        self._last_kick = 0.0
        self.checks = 0
        _monitors.add(self)

    @property
    def lost(self) -> bool:
        return self.health.lost

    @property
    def pending(self) -> bool:
        """A server reported a GPU error that has not been acted on yet."""
        return self._pending is not None

    def ensure_running(self) -> None:
        """Start the check loop (from the event loop) when it is not running; a no-op with the interval at 0."""
        if self.interval_s <= 0 and self._pending is None:
            return
        if self._task is not None and not self._task.done():
            return
        self._loop = asyncio.get_running_loop()
        self._kick = asyncio.Event()
        if self._pending is not None:
            self._kick.set()
        self._task = asyncio.create_task(self._run(), name="bolo-gpu-health")

    async def close(self) -> None:
        _monitors.discard(self)
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None

    def _wake(self) -> None:
        loop, kick = self._loop, self._kick
        if loop is None or kick is None or loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(kick.set)

    def report(self, source: str, line: str) -> None:
        """A managed server's GPU error (any thread): treated as a lost GPU."""
        if self.lost:
            return
        with self._lock:
            if self._pending is None:
                self._pending = GpuHealth("lost", f"{source.split(':', 1)[0]}: {line.strip()[:200]}", source)
        log.error("%s reported a GPU error: %s", source, line.strip()[:300])
        self._wake()

    def note_rtf(self, model_id: str, rtf: float | None) -> None:
        """A finished request on a GPU model (any thread): far slower than the model's recent median asks for a
        check now."""
        if rtf is None or rtf <= 0 or self.lost:
            return
        with self._lock:
            seen = self._rtf.setdefault(model_id, collections.deque(maxlen=RTF_WINDOW))
            slow = len(seen) >= RTF_MIN_SAMPLES and rtf > SLOW_FACTOR * statistics.median(seen)
            seen.append(rtf)
            now = time.monotonic()
            if not slow or now - self._last_kick < KICK_MIN_INTERVAL_S:
                return
            self._last_kick = now
        log.info("%s: real-time factor %.2f, far above its recent median: checking the GPU", model_id, rtf)
        self._wake()

    def forget(self, model_id: str) -> None:
        with self._lock:
            self._rtf.pop(model_id, None)

    async def check_now(self) -> GpuHealth:
        """One check (in a worker thread), acting on a loss. Returns the health afterwards."""
        with self._lock:
            pending, self._pending = self._pending, None
        if self.lost:
            return self.health
        found = pending
        if found is None:
            found = await asyncio.get_running_loop().run_in_executor(None, self._checker)
        self.checks += 1
        if found.lost:
            found.lost_at = time.time()
            found.checked_at = found.checked_at or found.lost_at
            found.fallback = self.health.fallback
            self.health = found
            log.error("%s", found.notice())
            try:
                await self._on_lost(found)
            except Exception:
                log.exception("moving models off the lost GPU failed")
        else:
            found.fallback = self.health.fallback
            self.health = found
        return self.health

    async def _run(self) -> None:
        assert self._kick is not None
        while not self.lost and (self._has_gpu_models() or self._pending is not None):
            if self._pending is None:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._kick.wait(), timeout=self.interval_s if self.interval_s > 0 else None)
            self._kick.clear()
            if not self._has_gpu_models() and self._pending is None:
                break
            with contextlib.suppress(Exception):
                await self.check_now()
