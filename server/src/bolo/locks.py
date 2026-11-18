"""Locks between processes: a lock file held with the operating system's own locking (``fcntl.flock`` on POSIX,
``msvcrt.locking`` on Windows), which the system lets go of when the holder exits or crashes, so a lock is never left
stale. The server and the CLI (``bolo keys``, ``bolo pull``) take them around files both of them write.

The lock is not re-entrant, and it is per open file: two threads of one process that each open the lock exclude
each other as two processes do.
"""

from __future__ import annotations

import contextlib
import errno
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

#: how often a waiting process tries the lock again (seconds)
POLL_S = 0.05


class LockTimeout(TimeoutError):
    """The lock was still held by another process when the time to wait for it ran out."""


def _try_lock(fd: int) -> bool:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EDEADLK):
                return False
            raise
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


class FileLock:
    """An exclusive lock on ``path`` (created if missing, never deleted: deleting a lock file another process is
    waiting on would let two processes in at once).

    ``timeout``: how long :meth:`acquire` waits (seconds) before raising :class:`LockTimeout`; None waits for as
    long as it takes."""

    def __init__(self, path: Path | str, timeout: float | None = None) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self, cancelled: Callable[[], bool] | None = None, on_wait: Callable[[], None] | None = None) -> bool:
        """Take the lock, waiting while another process (or another open of it) holds it. Returns whether it had to
        wait. ``cancelled`` is asked while waiting: when it says so, :class:`InterruptedError` is raised.
        ``on_wait`` is called once, when the lock is found taken."""
        if self._fd is not None:
            raise RuntimeError(f"{self.path} is already held by this lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        waited = False
        try:
            while not _try_lock(fd):
                if not waited and on_wait is not None:
                    on_wait()
                waited = True
                if cancelled is not None and cancelled():
                    raise InterruptedError(f"gave up waiting for {self.path}")
                if deadline is not None and time.monotonic() >= deadline:
                    raise LockTimeout(f"{self.path} is still locked by another process after {self.timeout:g} s")
                time.sleep(POLL_S)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return waited

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            with contextlib.suppress(OSError):
                _unlock(fd)
        finally:
            os.close(fd)

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.release()


# -- renames that wait out Windows sharing violations ----------------------------------------------------------
#: Windows errors that mean another program has the file open for a moment (ERROR_ACCESS_DENIED,
#: ERROR_SHARING_VIOLATION): an antivirus scanner looking at a file that was just written, an indexer, a reader
SHARING_ERRORS = (5, 32)
#: the pauses between tries of :func:`replace_file` (about 3 s in all)
REPLACE_BACKOFF_S = (0.05, 0.1, 0.2, 0.4, 0.8, 1.5)


def sharing_violation(exc: BaseException) -> bool:
    """A Windows "in use by another process" error (WinError 32, or 5 for a file being deleted or scanned)."""
    return isinstance(exc, OSError) and getattr(exc, "winerror", None) in SHARING_ERRORS


def retry_sharing(
    action: Callable[[], object],
    backoff: tuple[float, ...] = REPLACE_BACKOFF_S,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """Run ``action`` (a rename or a deletion), again with a growing pause while Windows says a file it touches is
    in use by another process (WinError 32 or 5); the last error is raised as it is. Other errors are not waited
    for."""
    wait = sleep or time.sleep
    for pause in (*backoff, None):
        try:
            action()
            return
        except OSError as exc:
            if pause is None or not sharing_violation(exc):
                raise
            wait(pause)


def replace_file(
    src: Path | str,
    dst: Path | str,
    backoff: tuple[float, ...] = REPLACE_BACKOFF_S,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """``os.replace(src, dst)`` (``Path.replace``: atomic, and over an existing ``dst`` on Windows too), tried again
    while Windows says one of the two files is in use (:func:`retry_sharing`)."""
    retry_sharing(lambda: Path(src).replace(dst), backoff, sleep)
