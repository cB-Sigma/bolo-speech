"""Conditions of pipeline steps (``when``, ``condition``, ``filter``): see docs/pipelines.md, "Conditions"."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

OPERATORS = (
    "equals", "not_equals", "in", "not_in", "contains", "matches", "gt", "gte", "lt", "lte", "exists", "truthy",
)  # fmt: skip


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return v


def _same(a: Any, b: Any) -> bool:
    a, b = _norm(a), _norm(b)
    if isinstance(a, str) and isinstance(b, float):
        try:
            return float(a) == b
        except ValueError:
            return False
    if isinstance(b, str) and isinstance(a, float):
        return _same(b, a)
    if isinstance(a, bool) and isinstance(b, str):
        return str(a).lower() == b
    if isinstance(b, bool) and isinstance(a, str):
        return str(b).lower() == a
    return bool(a == b)


def _number(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("", "false", "no", "0", "none", "null")
    return bool(v)


def evaluate(cond: dict[str, Any], resolve: Callable[[Any], Any], text: Callable[[Any], str]) -> bool:
    """``resolve`` turns a value or reference into its value; ``text`` writes a value as text."""
    if "all" in cond:
        return all(evaluate(c, resolve, text) for c in cond["all"])
    if "any" in cond:
        return any(evaluate(c, resolve, text) for c in cond["any"])
    if "not" in cond:
        return not evaluate(cond["not"], resolve, text)
    value = resolve(cond.get("value"))
    op = next(k for k in cond if k != "value")
    arg = resolve(cond[op])
    if op == "exists":
        present = value is not None and value not in ("", [])
        return present == bool(arg)
    if op == "truthy":
        return truthy(value) == bool(arg)
    if op == "equals":
        return _same(value, arg)
    if op == "not_equals":
        return not _same(value, arg)
    if op in ("in", "not_in"):
        found = any(_same(value, a) for a in (arg or []))
        return found if op == "in" else not found
    if op == "contains":
        if isinstance(value, list):
            return any(_same(v, arg) for v in value)
        return text(arg).strip().lower() in text(value).lower()
    if op == "matches":
        return re.search(str(arg), text(value), re.IGNORECASE) is not None
    a, b = _number(value), _number(arg)
    if a is None or b is None:
        return False
    return {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b}[op]
