"""Child processes tied to bolo's lifetime (bolo.procs): graceful stop, Job Objects, the parent-death signal."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import psutil
import pytest

from bolo import procs

SRC = str(Path(procs.__file__).resolve().parents[1])


def python(code: str) -> list[str]:
    return [sys.executable, "-c", textwrap.dedent(code)]


def gone(pid: int, within: float = 15.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.1)
    return False


def read_line(proc: subprocess.Popen[bytes]) -> str:
    assert proc.stdout is not None
    return proc.stdout.readline().decode().strip()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals; Windows is asked with CTRL_BREAK_EVENT")
def test_stop_asks_before_it_kills(tmp_path: Path) -> None:
    marker = tmp_path / "stopped"
    proc = procs.spawn(
        python(f"""
            import signal, sys, time
            def bye(*_):
                open({str(marker)!r}, "w").close()
                sys.exit(0)
            signal.signal(signal.SIGTERM, bye)
            print("ready", flush=True)
            time.sleep(60)
        """),
        stdout=subprocess.PIPE,
    )
    assert read_line(proc) == "ready"
    assert procs.stop(proc, timeout=10) == 0 and marker.exists()
    # a child that ignores the request is killed after the grace period
    stubborn = procs.spawn(
        python("""
            import signal, time
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            print("ready", flush=True)
            time.sleep(60)
        """),
        stdout=subprocess.PIPE,
    )
    assert read_line(stubborn) == "ready"
    started = time.monotonic()
    assert procs.stop(stubborn, timeout=0.5) is not None and time.monotonic() - started < 10
    assert stubborn.pid not in procs._children


def test_stop_ends_the_child_on_every_platform() -> None:
    proc = procs.spawn(python("import time; print('ready', flush=True); time.sleep(60)"), stdout=subprocess.PIPE)
    assert read_line(proc) == "ready"
    procs.stop(proc, timeout=2)
    assert proc.poll() is not None and proc.pid not in procs._children


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects")
def test_a_job_object_ends_the_whole_tree() -> None:
    proc = procs.spawn(
        python("""
            import subprocess, sys, time
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            print(child.pid, flush=True)
            time.sleep(60)
        """),
        stdout=subprocess.PIPE,
    )
    grandchild = int(read_line(proc))
    job = procs.job_of(proc)
    assert job is not None and job.contains(proc)
    procs.stop(proc, timeout=0.5)  # no console to share on CI: straight to TerminateJobObject
    assert gone(proc.pid) and gone(grandchild)


@pytest.mark.skipif(sys.platform not in ("win32", "linux"), reason="macOS has no parent-death signal or Job Object")
def test_children_die_with_a_killed_parent() -> None:
    """bolo killed hard (kill -9, Task Manager): its children go too (PR_SET_PDEATHSIG; the Job Object's handle)."""
    parent = subprocess.Popen(
        python(f"""
            import subprocess, sys, threading, time
            sys.path.insert(0, {SRC!r})
            from bolo import procs
            # started from a short-lived thread, as bolo's loaders do: the child must outlive that thread
            out = []
            t = threading.Thread(target=lambda: out.append(procs.spawn([sys.executable, "-c", "import time; time.sleep(60)"])))
            t.start(); t.join()
            print(out[0].pid, flush=True)
            time.sleep(60)
        """),
        stdout=subprocess.PIPE,
    )
    try:
        child = int(read_line(parent))
        time.sleep(0.5)
        assert psutil.pid_exists(child) and not gone(child, within=0.5)  # the thread's end did not take it
        parent.kill()
        parent.wait(10)
        assert gone(child), "the child outlived its killed parent"
    finally:
        parent.kill()


def test_limits_lower_priority_and_memory() -> None:
    proc = procs.spawn(
        python("""
            import time
            try:
                block = bytearray(1500 * 1024 * 1024)
                print("allocated", flush=True)
            except MemoryError:
                print("limited", flush=True)
            time.sleep(30)
        """),
        limits=procs.Limits(nice=5, memory_limit_mb=400),
        stdout=subprocess.PIPE,
    )
    try:
        assert read_line(proc) == "limited"
        priority = psutil.Process(proc.pid).nice()
        if sys.platform == "win32":
            assert priority == psutil.BELOW_NORMAL_PRIORITY_CLASS  # type: ignore[attr-defined,unused-ignore]
        else:
            assert priority >= os.nice(0) + 5
    finally:
        procs.stop(proc, timeout=1)
