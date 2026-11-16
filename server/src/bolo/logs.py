"""Where the server's log goes, file logging with rotation, and reading the log back (``bolo logs``).

The server always logs to its standard error (the terminal, journald under systemd, ``docker logs`` in a
container). With ``log_file`` it also writes a file, rotated at ``log_max_mb`` with ``log_backups`` old files kept:

* ``log_file = "auto"``: the platform's log directory (:func:`default_log_path`);
* ``log_file = "/some/path.log"``: that file;
* ``log_file = "off"``: no file;
* unset (the default): no file, except when ``bolo serve`` runs as a service (systemd sets ``INVOCATION_ID``,
  launchd sets ``XPC_SERVICE_NAME`` to the job's label), where it means ``auto``.

Windows cannot rename a file another program holds open without ``FILE_SHARE_DELETE``. ``bolo logs -f`` opens the
log with that share mode, so it does not stop a rotation; when another program (an editor) blocks the rename, the
handler copies the file to the first backup and empties it instead (copy-truncate), and tries again a minute later
if even that fails, rather than printing "--- Logging error ---" for every record.
"""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
import os
import shutil
import sys
import time
from collections import deque
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

#: the file the running server writes, once :func:`configure` has set it up
ACTIVE_LOG_FILE: Path | None = None
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_OFF = {"off", "none", "false", "no", "0"}


def default_log_path(environ: Mapping[str, str] | None = None, system: str | None = None) -> Path:
    """The platform's place for bolo's log file.

    * Linux and other Unixes: ``$XDG_STATE_HOME/bolo/logs/server.log`` (``~/.local/state/bolo/logs/server.log``)
    * macOS: ``~/Library/Logs/bolo/server.log``
    * Windows: ``%LOCALAPPDATA%\\bolo\\logs\\server.log``
    """
    env = os.environ if environ is None else environ
    name = (system or sys.platform).lower()
    home = Path(env.get("HOME") or env.get("USERPROFILE") or str(Path.home()))
    if name.startswith("win"):
        base = Path(env["LOCALAPPDATA"]) if env.get("LOCALAPPDATA") else home / "AppData" / "Local"
        return base / "bolo" / "logs" / "server.log"
    if name == "darwin":
        return home / "Library" / "Logs" / "bolo" / "server.log"
    state = Path(env["XDG_STATE_HOME"]) if env.get("XDG_STATE_HOME") else home / ".local" / "state"
    return state / "bolo" / "logs" / "server.log"


def running_as_service(environ: Mapping[str, str] | None = None) -> str | None:
    """``"systemd"`` or ``"launchd"`` when this process was started by one of them, else None."""
    env = os.environ if environ is None else environ
    if env.get("INVOCATION_ID") and env.get("JOURNAL_STREAM") is not None:
        return "systemd"
    label = env.get("XPC_SERVICE_NAME", "")
    if label and label != "0" and not label.startswith("application."):  # Terminal and apps have "0" / application.*
        return "launchd"
    return None


def resolve_log_file(
    value: str | None, environ: Mapping[str, str] | None = None, serving: bool = True, system: str | None = None
) -> Path | None:
    """The file the server writes for a ``log_file`` value, or None for none. ``serving``: asked for ``bolo serve``
    (an unset value then follows :func:`running_as_service`)."""
    text = str(value or "").strip()
    if text.lower() in _OFF:
        return None
    if not text:
        if not (serving and running_as_service(environ)):
            return None
        text = "auto"
    if text.lower() in ("auto", "default", "on", "true", "yes", "1"):
        return default_log_path(environ, system)
    return Path(text).expanduser()


class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """:class:`logging.handlers.RotatingFileHandler` that survives a rotation another process blocks (Windows)."""

    retry_s = 60.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._blocked_until = 0.0

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        if time.monotonic() < self._blocked_until:
            return False
        return bool(super().shouldRollover(record))

    def doRollover(self) -> None:
        try:
            super().doRollover()
            return
        except OSError as exc:  # PermissionError on Windows: the file (or a backup) is open in another program
            reason = exc
        if self.stream is not None:
            self.stream.close()
            self.stream = None  # type: ignore[assignment]
        try:
            if self.backupCount > 0:
                shutil.copyfile(self.baseFilename, self.rotation_filename(self.baseFilename + ".1"))
            with open(self.baseFilename, "r+b") as fh:  # noqa: PTH123 - the handler works with str paths
                fh.truncate(0)
        except OSError as exc:
            self._blocked_until = time.monotonic() + self.retry_s
            sys.stderr.write(
                f"bolo: could not rotate the log file {self.baseFilename} ({reason}; {exc}); trying again in "
                f"{self.retry_s:.0f} s\n"
            )
        self.stream = self._open()


def configure(settings: Any, environ: Mapping[str, str] | None = None) -> Path | None:
    """Set up the root logger for ``bolo serve``: standard error, and the log file when there is one. Returns the
    file. A file that cannot be opened is reported and skipped: logging must never stop the server starting."""
    global ACTIVE_LOG_FILE
    level = getattr(logging, str(settings.log_level).upper(), logging.INFO)
    logging.basicConfig(level=level, format=FORMAT, datefmt="%H:%M:%S")
    logging.getLogger().setLevel(level)
    path = resolve_log_file(settings.log_file, environ)
    ACTIVE_LOG_FILE = None
    if path is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=max(1, int(settings.log_max_mb)) * 1024 * 1024, backupCount=max(0, int(settings.log_backups)),
            encoding="utf-8",
        )  # fmt: skip
    except OSError as exc:
        logging.getLogger("bolo").warning(
            "cannot write the log file %s (%s); logging to standard error only", path, exc
        )
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s " + FORMAT.split(" ", 1)[1], "%Y-%m-%d %H:%M:%S"))
    handler.setLevel(level)
    handler.set_name("bolo-file")
    root = logging.getLogger()
    for old in [h for h in root.handlers if h.get_name() == "bolo-file"]:
        root.removeHandler(old)
        old.close()
    root.addHandler(handler)
    ACTIVE_LOG_FILE = path
    return path


def attach_file_handler(*loggers: str) -> None:
    """Also send these loggers (uvicorn's, which do not reach the root logger) to the log file."""
    root = logging.getLogger()
    handler = next((h for h in root.handlers if h.get_name() == "bolo-file"), None)
    if handler is None:
        return
    for name in loggers:
        logger = logging.getLogger(name)
        if handler not in logger.handlers and logger.propagate is False:
            logger.addHandler(handler)


def tail(path: Path, lines: int) -> list[str]:
    """The last ``lines`` lines of a file (all of them for ``lines <= 0``)."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        if lines <= 0:
            return [line.rstrip("\n") for line in fh]
        return [line.rstrip("\n") for line in deque(fh, maxlen=lines)]


def open_shared(path: Path, binary: bool = False) -> Any:
    """Open a file for reading so that the writer may still rename or delete it: on Windows with
    ``FILE_SHARE_DELETE`` (Python's ``open`` leaves it out, so a log rotation would fail while the file is open);
    elsewhere a plain ``open``. Text (UTF-8) unless ``binary``."""
    if sys.platform != "win32":
        return path.open("rb") if binary else path.open("r", encoding="utf-8", errors="replace")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    generic_read, share_all, open_existing, normal = 0x80000000, 0x7, 3, 0x80
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
        wintypes.HANDLE,
    ]  # fmt: skip
    k32.CreateFileW.restype = wintypes.HANDLE
    handle = k32.CreateFileW(str(path), generic_read, share_all, None, open_existing, normal, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        error = ctypes.get_last_error()
        raise OSError(0, ctypes.FormatError(error), str(path), error)
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    if binary:
        return open(fd, "rb")  # noqa: PTH123 - a file descriptor
    return open(fd, encoding="utf-8", errors="replace")  # noqa: PTH123 - a file descriptor


#: at most this much of the followed file's first line identifies it (log lines start with a timestamp)
_HEAD = 4096


def _text(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace").rstrip("\r\n")


def _first_line(handle: Any) -> bytes:
    handle.seek(0)
    line: bytes = handle.readline(_HEAD)
    return line


def follow(
    path: Path, poll_s: float = 0.5, stop: Any = None, backup: Path | None = None, from_start: bool = False
) -> Iterator[str]:
    """New lines of a log file as they are written, across rotations (``tail -F``). ``stop()`` ends it (tests).

    Both kinds of rotation are followed without losing lines:

    * rename (the normal one): the file at ``path`` is a new file. What was written to the old one since the last
      read is read from the still-open handle first, then the new file from its start.
    * copy-truncate (:class:`RotatingFileHandler` on Windows when another program blocks the rename): the file is
      copied to ``backup`` (``path.1``) and emptied in place, then written again, possibly past the position
      reached before it is next read. It is told by the file being shorter than that position, or by its first
      line having changed. The lines written before the copy that were not read yet are read from ``backup`` (when
      its first line is that of the file followed), then the emptied file from its start.

    ``from_start``: read the file from its start rather than from its end."""
    backup = backup or path.with_name(path.name + ".1")
    fh: Any = None
    inode = 0
    pos = 0  # bytes of the current file read (complete lines only)
    head = b""  # its first line (or as much of it as there was)
    at_start = from_start  # after a rotation the new file is read from its start, however late it is opened

    def lines_from(handle: Any, offset: int) -> list[str]:
        handle.seek(offset)
        return [_text(raw) for raw in handle.read().splitlines(keepends=True)]

    def emptied() -> bool:
        """The file was emptied (and maybe written again) since ``head`` was read."""
        if not head:
            return False
        return bool(_first_line(fh)[: len(head)] != head)

    try:
        while stop is None or not stop():
            if fh is None:
                try:
                    fh = open_shared(path, binary=True)
                    inode = os.fstat(fh.fileno()).st_ino
                    head = _first_line(fh)
                    pos = 0 if at_start else fh.seek(0, os.SEEK_END)
                except OSError:
                    fh = None
                    time.sleep(poll_s)
                    continue
            renamed = truncated = False
            try:
                fh.seek(pos)
                data = fh.read()
                end = data.rfind(b"\n") + 1  # complete lines only: the writer may be in the middle of one
                if end and not emptied():
                    pos += end
                    if not head.endswith(b"\n"):
                        head = _first_line(fh)
                    yield from (_text(raw) for raw in data[:end].splitlines(keepends=True))
                    continue
                st = path.stat()
                renamed = bool(st.st_ino and inode and st.st_ino != inode)
                truncated = not renamed and (st.st_size < pos or emptied())
            except OSError:
                renamed = True
            if not (renamed or truncated):
                time.sleep(poll_s)
                continue
            # rotated: first what was written before the rotation and not read yet, then the new file from its start
            missed: list[str] = []
            if renamed:  # the open handle is the old file (now the backup)
                with contextlib.suppress(OSError):
                    missed = lines_from(fh, pos)
            else:  # emptied in place: the rest is in the copy, when it is a copy of this file
                with contextlib.suppress(OSError), open_shared(backup, binary=True) as old:
                    if head and _first_line(old)[: len(head)] == head:
                        missed = lines_from(old, pos)
            fh.close()
            fh, at_start = None, True
            yield from missed
            try:
                fh = open_shared(path, binary=True)
                inode = os.fstat(fh.fileno()).st_ino
                head, pos = _first_line(fh), 0
            except OSError:
                fh = None
    finally:
        if fh is not None:
            fh.close()
