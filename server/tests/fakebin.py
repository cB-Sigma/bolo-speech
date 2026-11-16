"""Stand-in programs for tests: a script run by this Python, which bolo finds and starts like a real binary.

On POSIX that is an executable file with a ``#!`` line. Windows has no ``#!``: there the program is a ``.cmd``
file that runs the script with this Python (the name bolo looks for gets the ``.cmd`` extension, which is in
``PATHEXT``), so ``shutil.which``, ``subprocess`` and bolo's own lookup find and start it as they would an
``.exe``.
"""

from __future__ import annotations

import sys
from pathlib import Path

WINDOWS = sys.platform == "win32"


def fake_program(path: Path, source: str) -> Path:
    """Write a program at ``path`` (``path.cmd`` on Windows) that runs the Python ``source``; returns its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if WINDOWS:
        script = path.with_name(path.name + "-script.py")
        script.write_text(source, encoding="utf-8")
        launcher = path.with_name(path.name + ".cmd")
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        return launcher
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(0o755)
    return path


def printing_program(path: Path, stdout: str = "", stderr: str = "", code: int = 0) -> Path:
    """A program that prints ``stdout`` and ``stderr`` and exits with ``code``."""
    return fake_program(
        path,
        f"import sys\nsys.stdout.write({stdout!r})\nsys.stderr.write({stderr!r})\nsys.exit({code})\n",
    )
