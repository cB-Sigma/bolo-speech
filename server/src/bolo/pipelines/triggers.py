"""When pipelines run on their own (``triggers``; docs/pipelines.md, "Triggers"):

* ``call_end``: at the end of each phone call (of some agents), on the call's transcript;
* ``session_end``: at the end of each Realtime session, on its transcript;
* ``folder``: on each new recording in a watched folder (under a ``pipeline_watch_dirs`` folder).

A recording uploaded for transcription runs a pipeline too when the request names one (``pipeline=<id>``), and a
``live`` pipeline runs per user turn of the session or call that uses it (bolo.pipelines.live).
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bolo.pipelines.definition import Problem

if TYPE_CHECKING:
    from bolo.core import Bolo

log = logging.getLogger("bolo.pipelines")

TRIGGERS = ("call_end", "session_end", "folder")
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".webm", ".aac")


def check_triggers(triggers: Any, mode: str, watch_dirs: list[str] | None = None) -> list[Problem]:
    if not isinstance(triggers, list):
        return [Problem("triggers", "must be a list")]
    out: list[Problem] = []
    for i, t in enumerate(triggers):
        p = f"triggers[{i}]"
        if not isinstance(t, dict) or t.get("type") not in TRIGGERS:
            out.append(Problem(p, f"a trigger is an object with a type: {', '.join(TRIGGERS)}"))
            continue
        if mode == "live":
            out.append(Problem(p, "a live pipeline runs per turn of the calls and sessions that use it; it takes no "
                                  "triggers"))  # fmt: skip
            continue
        allowed = {"call_end": ("agents",), "session_end": (), "folder": ("path", "pattern", "move_to")}[t["type"]]
        for key in t:
            if key != "type" and key not in allowed:
                out.append(Problem(f"{p}.{key}", f"a {t['type']} trigger has no '{key}'"))
        if (
            t["type"] == "call_end"
            and t.get("agents") is not None
            and not (isinstance(t["agents"], list) and all(isinstance(a, str) for a in t["agents"]))
        ):
            out.append(Problem(f"{p}.agents", "must be a list of agent names"))
        if t["type"] == "folder":
            path = t.get("path")
            if not isinstance(path, str) or not path.strip():
                out.append(Problem(f"{p}.path", "a folder trigger needs a path"))
            elif watch_dirs is not None and not allowed_folder(path, watch_dirs):
                roots = ", ".join(watch_dirs) or "none"
                out.append(Problem(f"{p}.path", f"'{path}' is not under a folder the server may watch "
                                                f"(pipeline_watch_dirs: {roots})"))  # fmt: skip
            move_to = t.get("move_to", "done")
            if not isinstance(move_to, str) or not move_to or "/" in move_to or "\\" in move_to or ".." in move_to:
                out.append(Problem(f"{p}.move_to", "must be the name of a folder inside the watched folder"))
    return out


def allowed_folder(path: str, roots: list[str]) -> bool:
    try:
        target = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    for root in roots:
        try:
            base = Path(root).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if target == base or base in target.parents:
            return True
    return False


class Triggers:
    """Starts triggered runs (as jobs) for the pipelines whose ``triggers`` ask for them."""

    def __init__(self, core: Bolo) -> None:
        self.core = core
        self._watcher: asyncio.Task[Any] | None = None
        #: folder -> file name -> (size, mtime) seen at the last look: a file is taken once it stops changing
        self._seen: dict[str, dict[str, tuple[int, float]]] = {}

    def _with(self, kind: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        out = []
        for p in self.core.pipelines.store.all():
            if p.get("mode", "file") != "file":
                continue
            for t in p.get("triggers") or []:
                if isinstance(t, dict) and t.get("type") == kind:
                    out.append((p, t))
        return out

    def _start(self, definition: dict[str, Any], trigger: str, **kw: Any) -> str | None:
        service = self.core.pipelines
        try:
            run = service.runner.prepare(definition, trigger=trigger, **kw)
        except Exception as exc:
            log.warning("pipeline %s (%s trigger) did not start: %s", definition.get("id"), trigger,
                        getattr(exc, "message", exc))  # fmt: skip
            return None
        service.start(run)
        return run.id

    # -- calls and sessions -----------------------------------------------------------------------------------
    def call_ended(self, record: Any) -> list[str]:
        started = []
        text = record.transcript_text()
        call = {
            "id": record.id, "agent": record.agent, "from": record.from_number, "to": record.to_number,
            "direction": record.direction, "outcome": record.outcome, "duration_s": record.duration_s,
            "provider": record.provider,
        }  # fmt: skip
        for definition, t in self._with("call_end"):
            agents = t.get("agents")
            if agents and record.agent not in agents:
                continue
            run_id = self._start(definition, "call_end", text=text or None, extra_inputs={"call": call})
            if run_id:
                started.append(run_id)
                record.event("pipeline_started", pipeline=definition.get("id"), run=run_id)
        return started

    def session_ended(self, session_info: dict[str, Any], transcript: str) -> list[str]:
        if not transcript.strip():
            return []
        started = []
        for definition, _ in self._with("session_end"):
            run_id = self._start(definition, "session_end", text=transcript, extra_inputs={"session": session_info})
            if run_id:
                started.append(run_id)
        return started

    # -- watched folders ------------------------------------------------------------------------------------------
    def start(self) -> None:
        if self._watcher is None and self.core.settings.pipeline_watch_dirs:
            self._watcher = asyncio.create_task(self._watch(), name="bolo-pipeline-folders")

    async def close(self) -> None:
        task, self._watcher = self._watcher, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _watch(self) -> None:
        while True:
            try:
                await self.scan()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("watching pipeline folders")
            await asyncio.sleep(max(0.2, self.core.settings.pipeline_watch_interval_s))

    async def scan(self) -> list[str]:
        """Look at every watched folder once; start a run for each recording that has stopped changing."""
        started: list[str] = []
        roots = self.core.settings.pipeline_watch_dirs
        for definition, t in self._with("folder"):
            path = str(t.get("path") or "")
            if not allowed_folder(path, roots):
                continue
            folder = Path(path).expanduser()
            pattern = str(t.get("pattern") or "*")
            done = folder / str(t.get("move_to") or "done")
            seen = self._seen.setdefault(f"{definition['id']}:{folder}", {})
            try:
                entries = [p for p in folder.iterdir() if p.is_file()]
            except OSError:
                continue
            for f in entries:
                if f.suffix.lower() not in AUDIO_SUFFIXES or not fnmatch.fnmatch(f.name, pattern):
                    continue
                try:
                    st = f.stat()
                except OSError:
                    continue
                stamp = (st.st_size, st.st_mtime)
                if seen.get(f.name) != stamp:  # new or still being written: look again next time
                    seen[f.name] = stamp
                    continue
                seen.pop(f.name, None)
                run_id = await self._run_file(definition, f, done)
                if run_id:
                    started.append(run_id)
        return started

    async def _run_file(self, definition: dict[str, Any], f: Path, done: Path) -> str | None:
        from bolo.audio import decode_audio
        from bolo.pipelines.runner import Audio

        done.mkdir(parents=True, exist_ok=True)
        target = done / f.name
        try:
            shutil.move(str(f), target)  # taken: never run twice, even after a restart
            data = target.read_bytes()
            samples, rate = await asyncio.to_thread(decode_audio, data, filename=target.name)
        except Exception as exc:
            log.warning("pipeline %s: %s could not be read: %s", definition.get("id"), f, exc)
            return None
        return self._start(definition, "folder", audio=Audio(samples, rate), filename=f.name)
