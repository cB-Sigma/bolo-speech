"""Subtitle renderers for transcription output."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class _Seg(Protocol):
    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...
    @property
    def text(self) -> str: ...


def _speaker(seg: object) -> str | None:
    return getattr(seg, "speaker", None)


def _stamp(seconds: float, sep: str) -> str:
    ms_total = max(0, round(seconds * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: Iterable[_Seg]) -> str:
    blocks = [
        f"{i}\n{_stamp(s.start, ',')} --> {_stamp(s.end, ',')}\n"
        + (f"[{_speaker(s)}] " if _speaker(s) else "")
        + f"{s.text.strip()}\n"
        for i, s in enumerate(segments, 1)
    ]
    return "\n".join(blocks)


def to_vtt(segments: Iterable[_Seg]) -> str:
    blocks = [
        f"{_stamp(s.start, '.')} --> {_stamp(s.end, '.')}\n"
        + (f"<v {_speaker(s)}>" if _speaker(s) else "")
        + f"{s.text.strip()}\n"
        for s in segments
    ]
    return "WEBVTT\n\n" + "\n".join(blocks)
