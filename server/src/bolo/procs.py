"""Child processes bolo starts: llama-server and audio.cpp servers, and the plugin host's workers.

They are tied to bolo's lifetime and stopped gracefully before they are killed:

* **Windows:** each child goes into its own Job Object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. bolo holds
  the job's handle, so when bolo ends, however it ends (Task Manager, a service stop, closing the console
  window), Windows closes the handle and ends the child and everything the child started. The job also carries
  the plugin host's memory limit (``JOB_OBJECT_LIMIT_PROCESS_MEMORY``: committed memory per process, where POSIX
  limits the address space), and a positive ``nice`` starts the child with ``BELOW_NORMAL_PRIORITY_CLASS``.
  Children get their own console process group, so the graceful stop is a ``CTRL_BREAK_EVENT`` (console programs
  such as llama-server exit on it), then ``TerminateJobObject``.
* **Linux:** the child asks the kernel for ``SIGTERM`` when bolo dies (``prctl(PR_SET_PDEATHSIG)``), which covers
  ``kill -9``. The kernel sends it when the *thread* that started the child ends, so children are started from
  one thread that lives as long as bolo.
* **macOS** (and Linux as well): children stay in bolo's process group, so Ctrl+C in a terminal and a service
  manager stopping the group reach them, and the children still running are stopped when bolo exits normally
  (``atexit``) and on ``SIGHUP``. A ``kill -9`` of bolo leaves them running on macOS.
* **Memory limit:** ``RLIMIT_AS`` on Linux; macOS neither enforces ``RLIMIT_AS`` nor accepts one below the
  address space every process reserves there, so on macOS the limit is not applied (bolo logs that once) and the
  child starts without it.
* **Everywhere:** :func:`stop` asks first (``SIGTERM`` / ``CTRL_BREAK_EVENT``) and kills after a grace period.

Only the standard library is used (``ctypes`` on Windows and Linux).
"""

from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, replace
from typing import Any

log = logging.getLogger("bolo.procs")

WINDOWS = sys.platform == "win32"
LINUX = sys.platform.startswith("linux")
MACOS = sys.platform == "darwin"

# Windows constants (winbase.h, winnt.h)
CREATE_NEW_PROCESS_GROUP = 0x00000200
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
PR_SET_PDEATHSIG = 1


@dataclass(frozen=True)
class Limits:
    """Limits for a child: a lower priority (``nice`` > 0) and a memory limit in MB (0: none)."""

    nice: int = 0
    memory_limit_mb: int = 0


# -- Windows Job Objects ------------------------------------------------------------------------------------------
class _Job:
    """A Windows Job Object that ends its processes when its last handle closes."""

    def __init__(self, memory_limit_mb: int = 0) -> None:
        import ctypes

        k32 = _kernel32()
        handle = k32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined,unused-ignore]
        self.handle: int | None = handle
        info = _extended_limit_info()()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_limit_mb > 0:
            info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY
            info.ProcessMemoryLimit = memory_limit_mb * 1024 * 1024
        ok = k32.SetInformationJobObject(
            handle, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS, ctypes.byref(info), ctypes.sizeof(info)
        )
        if not ok:
            error = ctypes.get_last_error()  # type: ignore[attr-defined,unused-ignore]
            self.close()
            raise ctypes.WinError(error)  # type: ignore[attr-defined,unused-ignore]

    def assign(self, proc: subprocess.Popen[Any]) -> None:
        import ctypes

        if not _kernel32().AssignProcessToJobObject(self.handle, int(proc._handle)):  # type: ignore[attr-defined]
            raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined,unused-ignore]

    def contains(self, proc: subprocess.Popen[Any]) -> bool:
        import ctypes

        result = ctypes.c_int(0)
        _kernel32().IsProcessInJob(int(proc._handle), self.handle, ctypes.byref(result))  # type: ignore[attr-defined]
        return bool(result.value)

    def terminate(self, exit_code: int = 1) -> None:
        if self.handle:
            _kernel32().TerminateJobObject(self.handle, exit_code)

    def close(self) -> None:
        handle, self.handle = self.handle, None
        if handle:
            _kernel32().CloseHandle(handle)


_K32: Any = None
_LIMIT_INFO: Any = None


def _kernel32() -> Any:
    global _K32
    if _K32 is None:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(ctypes.c_int)]
        k32.IsProcessInJob.restype = wintypes.BOOL
        k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.TerminateJobObject.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        _K32 = k32
    return _K32


def _extended_limit_info() -> Any:
    global _LIMIT_INFO
    if _LIMIT_INFO is None:
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]  # fmt: skip

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        _LIMIT_INFO = ExtendedLimitInformation
    return _LIMIT_INFO


# -- Linux: parent-death signal, from one long-lived thread ----------------------------------------------------
_PRCTL: Any = None


def _prctl() -> Any:
    global _PRCTL
    if _PRCTL is None:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        _PRCTL = libc.prctl
    return _PRCTL


def _preexec(limits: Limits, parent: int, prctl: Any) -> None:  # pragma: no cover - runs in the child
    if sys.platform == "win32":  # never called there (no fork)
        return
    if prctl is not None:
        prctl(PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)
        if os.getppid() != parent:  # bolo died between fork and prctl
            os.kill(os.getpid(), signal.SIGTERM)
    if limits.nice > 0:
        os.nice(limits.nice)
    if limits.memory_limit_mb > 0:
        import resource

        limit = limits.memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


_warned_macos_memory = False


def _no_memory_limit_on_macos() -> None:
    global _warned_macos_memory
    if not _warned_macos_memory:
        _warned_macos_memory = True
        log.warning(
            "the memory limit for child processes (plugin_memory_limit_mb) is not applied on macOS, which does not "
            "enforce an address-space limit; children start without one"
        )


class _Spawner:
    """Starts children from one daemon thread that lives as long as bolo (see PR_SET_PDEATHSIG above)."""

    def __init__(self) -> None:
        self._jobs: queue.SimpleQueue[tuple[concurrent.futures.Future[Any], dict[str, Any]]] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def popen(self, kwargs: dict[str, Any]) -> subprocess.Popen[Any]:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="bolo-spawner", daemon=True)
                self._thread.start()
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._jobs.put((future, kwargs))
        result: subprocess.Popen[Any] = future.result()
        return result

    def _run(self) -> None:
        while True:
            future, kwargs = self._jobs.get()
            try:
                future.set_result(subprocess.Popen(**kwargs))
            except BaseException as exc:
                future.set_exception(exc)


_spawner = _Spawner()

# -- registry ------------------------------------------------------------------------------------------------------
_children: dict[int, tuple[subprocess.Popen[Any], _Job | None]] = {}
_registry_lock = threading.Lock()
_hooks_installed = False


def install_hooks() -> None:
    """Stop the children still running when bolo exits (``atexit``) and, on POSIX, on ``SIGHUP`` (a closed
    terminal), unless something else handles that signal. The signal handler needs the main thread: ``bolo serve``
    calls this at start-up; :func:`spawn` calls it too (then only ``atexit`` when not on the main thread)."""
    global _hooks_installed
    on_main = threading.current_thread() is threading.main_thread()
    if _hooks_installed and not on_main:
        return
    if not _hooks_installed:
        atexit.register(stop_all)
    _hooks_installed = True
    if sys.platform != "win32" and on_main:
        with contextlib.suppress(ValueError, OSError, AttributeError):
            current = signal.getsignal(signal.SIGHUP)
            if current is signal.SIG_DFL or current is None:  # not ours already, nor anyone else's

                def on_hangup(signum: int, frame: object) -> None:  # pragma: no cover - needs a hang-up
                    stop_all(timeout=3.0)
                    signal.signal(signal.SIGHUP, signal.SIG_DFL)
                    os.kill(os.getpid(), signal.SIGHUP)

                signal.signal(signal.SIGHUP, on_hangup)


def spawn(argv: list[str], *, limits: Limits | None = None, **kwargs: Any) -> subprocess.Popen[Any]:
    """Start a child tied to bolo's lifetime (see the module docstring); ``kwargs`` go to :class:`subprocess.Popen`."""
    limits = limits or Limits()
    kwargs = {"args": argv, **kwargs}
    job: _Job | None = None
    if WINDOWS:
        flags = int(kwargs.pop("creationflags", 0)) | CREATE_NEW_PROCESS_GROUP
        if limits.nice > 0:
            flags |= BELOW_NORMAL_PRIORITY_CLASS
        kwargs["creationflags"] = flags
        try:
            job = _Job(limits.memory_limit_mb)
        except OSError as exc:  # pragma: no cover - Job Objects exist on every supported Windows
            log.warning("could not create a Job Object for %s: %s", argv[0], exc)
        proc: subprocess.Popen[Any] = subprocess.Popen(**kwargs)
        if job is not None:
            try:
                job.assign(proc)
            except OSError as exc:  # pragma: no cover - e.g. a job that forbids nested jobs
                log.warning("could not put %s (pid %d) in a Job Object: %s", argv[0], proc.pid, exc)
                job.close()
                job = None
    else:
        if MACOS and limits.memory_limit_mb > 0:
            _no_memory_limit_on_macos()
            limits = replace(limits, memory_limit_mb=0)
        prctl = None
        if LINUX:
            with contextlib.suppress(OSError, AttributeError):
                prctl = _prctl()
        if prctl is not None or limits.nice > 0 or limits.memory_limit_mb > 0:
            parent = os.getpid()
            kwargs["preexec_fn"] = lambda: _preexec(limits, parent, prctl)
        proc = _spawner.popen(kwargs) if prctl is not None else subprocess.Popen(**kwargs)
    with _registry_lock:
        install_hooks()
        _children[proc.pid] = (proc, job)
    return proc


def job_of(proc: subprocess.Popen[Any]) -> _Job | None:
    """The Job Object a child is in (Windows), else None."""
    with _registry_lock:
        entry = _children.get(proc.pid)
    return entry[1] if entry and entry[0] is proc else None


def _forget(proc: subprocess.Popen[Any]) -> None:
    with _registry_lock:
        entry = _children.get(proc.pid)
        if entry and entry[0] is proc:
            del _children[proc.pid]
    if entry and entry[0] is proc and entry[1] is not None:
        entry[1].close()


def ask_to_stop(proc: subprocess.Popen[Any]) -> bool:
    """The graceful step: SIGTERM on POSIX, CTRL_BREAK_EVENT to the child's console process group on Windows (only
    for a child :func:`spawn` started, which has its own group). False when the child could not be asked."""
    if proc.poll() is not None:
        return True
    try:
        if WINDOWS:
            with _registry_lock:
                own_group = proc.pid in _children
            if not own_group:
                return False
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined,unused-ignore]
        else:
            proc.terminate()
    except OSError as exc:  # no console to share (a service), or already gone
        log.debug("could not ask pid %d to stop: %s", proc.pid, exc)
        return False
    return True


def kill(proc: subprocess.Popen[Any]) -> None:
    """The hard stop: the whole Job Object on Windows (the child and what it started), else SIGKILL."""
    job = job_of(proc)
    if job is not None:
        job.terminate()
    with contextlib.suppress(OSError):
        proc.kill()


def stop(proc: subprocess.Popen[Any], timeout: float = 10.0, *, graceful: bool = True) -> int | None:
    """Stop a child: ask (unless ``graceful`` is False), wait up to ``timeout`` seconds, then kill. Returns its exit
    code. The child's Job Object (Windows) is closed, which also ends whatever the child left behind."""
    try:
        if proc.poll() is None:
            if graceful and ask_to_stop(proc):
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    log.info("pid %d did not stop within %.0f s; killing it", proc.pid, timeout)
            if proc.poll() is None:
                kill(proc)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=10)
        return proc.returncode
    finally:
        _forget(proc)


def stop_all(timeout: float = 5.0) -> None:
    """Stop every child still running (at exit)."""
    with _registry_lock:
        procs = [proc for proc, _ in _children.values()]
    for proc in procs:
        with contextlib.suppress(Exception):
            stop(proc, timeout)
