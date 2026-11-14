"""Where the server's log goes, file logging with rotation, and reading the log back (``bolo logs``).

The server always logs to its standard error (the terminal, journald under systemd, ``docker logs`` in a
container). With ``log_file`` it also writes a file, rotated at ``log_max_mb`` with ``log_backups`` old files kept:

* ``log_file = "auto"``: the platform's log directory (:func:`default_log_path`);
* ``log_file = "/some/path.log"``: that file;
* ``log_file = "off"``: no file;
* unset (the default): no file, except when ``bolo serve`` runs as a service (systemd sets ``INVOCATION_ID``,
  launchd sets ``XPC_SERVICE_NAME`` to the job's label), where it means ``auto``.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
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
        handler = logging.handlers.RotatingFileHandler(
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


def follow(path: Path, poll_s: float = 0.5, stop: Any = None) -> Iterator[str]:
    """New lines of a log file as they are written, across rotations (``tail -F``). ``stop()`` ends it (tests)."""
    fh = None
    inode = None
    try:
        while stop is None or not stop():
            if fh is None:
                try:
                    fh = path.open("r", encoding="utf-8", errors="replace")
                    inode = os.fstat(fh.fileno()).st_ino
                    fh.seek(0, os.SEEK_END)
                except OSError:
                    time.sleep(poll_s)
                    continue
            line = fh.readline()
            if line:
                yield line.rstrip("\n")
                continue
            try:
                st = path.stat()
                rotated = st.st_ino != inode or st.st_size < fh.tell()
            except OSError:
                rotated = True
            if rotated:  # the file was rotated or truncated: read the new one from its start
                fh.close()
                try:
                    fh = path.open("r", encoding="utf-8", errors="replace")
                    inode = os.fstat(fh.fileno()).st_ino
                except OSError:
                    fh = None
                continue
            time.sleep(poll_s)
    finally:
        if fh is not None:
            fh.close()
