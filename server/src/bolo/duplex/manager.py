"""Which full-duplex backend serves ``s2s_model``, and whether it can (for the console and ``/v1/capabilities``)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from bolo.duplex.base import DuplexBackend
from bolo.errors import ComponentMissing, EngineUnavailable

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.core import Bolo

#: catalog ``kind`` of the models that speak Kyutai's WebSocket protocol (bolo.duplex.kyutai)
KYUTAI_KIND = "kyutai"
KINDS = (KYUTAI_KIND,)


def _codec_missing() -> str | None:
    try:
        import av  # noqa: F401
    except ImportError:
        from bolo.duplex.ogg_opus import INSTALL_HINT

        return f"PyAV is not installed ({INSTALL_HINT})"
    return None


def readiness(core: Bolo, spec: ModelSpec) -> tuple[bool, bool, str | None]:
    """``(configured, available, reason)`` of a full-duplex ``s2s_model``."""
    from bolo.duplex import runtime

    s = core.settings
    if spec.kind not in KINDS:
        return True, False, f"model '{spec.id}' is of kind {spec.kind!r}, which no full-duplex backend serves"
    problem = _codec_missing()
    if problem:
        return True, False, problem
    if s.duplex_server_url:
        return True, True, None
    problem = runtime.missing_reason(s, spec)
    if problem:
        return True, False, problem
    if not (core.store.is_installed(spec.id) or (s.auto_pull and not spec.gated)):
        return True, False, f"model '{spec.id}' is not installed (bolo pull {spec.id})"
    return True, True, None


async def build_backend(core: Bolo, spec: ModelSpec) -> DuplexBackend:
    from bolo.duplex import runtime
    from bolo.duplex.kyutai import KyutaiBackend
    from bolo.engines.tts_server import free_port

    s = core.settings
    if spec.kind not in KINDS:
        raise EngineUnavailable(f"model '{spec.id}' is of kind {spec.kind!r}, which no full-duplex backend serves")
    problem = _codec_missing()
    if problem:
        raise ComponentMissing(f"the full-duplex model '{spec.id}' sends Opus audio: {problem}")
    query = dict(spec.options.get("query") or {})
    common: dict[str, Any] = {
        "model": spec.id, "query": query, "voice_param": spec.options.get("voice_param"),
        "voice_suffix": str(spec.options.get("voice_suffix") or ""), "default_voice": spec.options.get("default_voice"),
        "instructions_param": spec.options.get("instructions_param"),
    }  # fmt: skip
    if s.duplex_server_url:
        return KyutaiBackend(s.duplex_server_url, max_sessions=s.duplex_max_sessions, runtime="external", **common)
    problem = runtime.missing_reason(s, spec)
    if problem:
        raise ComponentMissing(f"the full-duplex model '{spec.id}' cannot start: {problem}")
    if not core.store.is_installed(spec.id):
        await core._auto_pull(spec)
    try:
        launch = await asyncio.to_thread(
            runtime.launch, s, spec, core.store.model_dir(spec.id), free_port(), list(s.duplex_server_args)
        )
    except (ValueError, FileNotFoundError) as exc:
        raise EngineUnavailable(str(exc)) from exc
    return KyutaiBackend(
        process=launch.process, max_sessions=launch.max_sessions, restart_after_session=launch.restart_after_session,
        runtime=launch.runtime, **common,
    )  # fmt: skip
