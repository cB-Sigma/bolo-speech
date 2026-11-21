"""Files a pipeline run writes (audio outputs and ``save`` steps), kept in the run's folder."""

from __future__ import annotations

import re

from bolo.errors import InvalidRequest

EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".txt", ".json", ".srt", ".vtt")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
MEDIA_TYPES = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".opus": "audio/ogg",
    ".txt": "text/plain; charset=utf-8", ".json": "application/json", ".srt": "text/plain; charset=utf-8",
    ".vtt": "text/vtt; charset=utf-8",
}  # fmt: skip


def check_name(name: str) -> str:
    """A file name a run may write: no folders, one of :data:`EXTENSIONS`."""
    if not _NAME_RE.match(name) or ".." in name:
        raise InvalidRequest(f"'{name}' is not a file name (letters, digits, '.', '_' and '-'; no folders)")
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in EXTENSIONS:
        raise InvalidRequest(f"'{name}': the file name must end in one of {', '.join(EXTENSIONS)}")
    return name


def extension(name: str) -> str:
    return "." + name.rsplit(".", 1)[-1].lower()
