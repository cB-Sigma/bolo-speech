"""Which full-duplex backend serves ``s2s_model``, and whether it can (for the console and ``/v1/capabilities``)."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bolo.duplex.base import DuplexBackend
from bolo.errors import ComponentMissing, EngineUnavailable

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.core import Bolo

#: catalog ``kind`` of the models that speak Kyutai's WebSocket protocol (bolo.duplex.kyutai)
KYUTAI_KIND = "kyutai"
#: Tencent's Freeze-Omni server (Socket.IO; bolo.duplex.freeze_omni)
FREEZE_OMNI_KIND = "freeze-omni"
KINDS = (KYUTAI_KIND, FREEZE_OMNI_KIND)


def _codec_missing() -> str | None:
    try:
        import av  # noqa: F401
    except ImportError:
        from bolo.duplex.ogg_opus import INSTALL_HINT

        return f"PyAV is not installed ({INSTALL_HINT})"
    return None


def remote_code_problem(core: Bolo, spec: ModelSpec, path: Path | None = None) -> str | None:
    """Why the model's remote code may not run (bolo.remote_code): the operator has not opted in for it, or, with
    ``path``, its files are not the pinned ones (a changed file, an unlisted ``.py`` file, an ``auto_map`` into another
    repository). None: no remote code, or the opt-in and the files are in order."""
    from bolo import remote_code

    pin = remote_code.pin(spec)
    if pin is None:
        return None
    if not remote_code.allowed(core.store.model_settings(spec.id)):
        return remote_code.notice(spec, core.store.model_settings(spec.id))
    if path is None:
        return None
    files = {str(k): str(v) for k, v in (pin.get("files") or {}).items()}
    files.update({str(k): str(v) for k, v in (spec.options.get("sha256") or {}).items()})
    for name, digest in files.items():
        target = path / name
        if not target.is_file():
            return f"model '{spec.id}': the pinned file {name} is missing; pull the model again"
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            return f"model '{spec.id}': {name} is not the reviewed file (sha256 differs); its remote code will not run"
    extra = sorted(str(p.relative_to(path)) for p in path.rglob("*.py") if str(p.relative_to(path)) not in files)
    if extra:
        return f"model '{spec.id}': unreviewed code in its folder ({', '.join(extra)}); its remote code will not run"
    for name in files:
        if name.endswith(".json"):
            with contextlib.suppress(ValueError, OSError):
                auto_map = json.loads((path / name).read_text(encoding="utf-8")).get("auto_map") or {}
                if "--" in json.dumps(auto_map):
                    return f"model '{spec.id}': {name} maps classes to another repository; its remote code will not run"
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
    problem = runtime.missing_reason(s, spec) or remote_code_problem(core, spec)
    if problem:
        return True, False, problem
    if not (core.store.is_installed(spec.id) or (s.auto_pull and not spec.gated)):
        return True, False, f"model '{spec.id}' is not installed (bolo pull {spec.id})"
    return True, True, None


async def build_backend(core: Bolo, spec: ModelSpec) -> DuplexBackend:
    from bolo.duplex import runtime
    from bolo.duplex.freeze_omni import FreezeOmniBackend
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
    if spec.kind == FREEZE_OMNI_KIND:
        if s.duplex_server_url:
            return FreezeOmniBackend(
                s.duplex_server_url, model=spec.id, max_sessions=s.duplex_max_sessions, runtime="external"
            )
    elif s.duplex_server_url:
        return KyutaiBackend(s.duplex_server_url, max_sessions=s.duplex_max_sessions, runtime="external", **common)
    problem = runtime.missing_reason(s, spec)
    if problem:
        raise ComponentMissing(f"the full-duplex model '{spec.id}' cannot start: {problem}")
    if not core.store.is_installed(spec.id):
        await core._auto_pull(spec)
    problem = remote_code_problem(core, spec, core.store.model_dir(spec.id))
    if problem:
        raise EngineUnavailable(problem)
    try:
        launch = await asyncio.to_thread(
            runtime.launch, s, spec, core.store.model_dir(spec.id), free_port(), list(s.duplex_server_args)
        )
    except (ValueError, FileNotFoundError) as exc:
        raise EngineUnavailable(str(exc)) from exc
    if spec.kind == FREEZE_OMNI_KIND:
        return FreezeOmniBackend(
            process=launch.process, model=spec.id, max_sessions=launch.max_sessions, runtime=launch.runtime
        )
    return KyutaiBackend(
        process=launch.process, max_sessions=launch.max_sessions, restart_after_session=launch.restart_after_session,
        runtime=launch.runtime, **common,
    )  # fmt: skip
