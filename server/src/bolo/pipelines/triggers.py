"""When pipelines run on their own (``triggers``; docs/pipelines.md, "Triggers")."""

from __future__ import annotations

from typing import Any

from bolo.pipelines.definition import Problem

TRIGGERS: tuple[str, ...] = ()


def check_triggers(triggers: Any, mode: str) -> list[Problem]:
    if not isinstance(triggers, list):
        return [Problem("triggers", "must be a list")]
    if triggers:
        return [Problem("triggers", "this server runs pipelines on request only; triggers are not available")]
    return []
