"""The plugin's protocol server: bolo's SDK server, with error codes that bolo understands.

bolo's SDK reports an exception by its class name; this server sends the ``code`` of
:class:`bolo_torch.runtime.PluginError` instead (``model_gated``, ``invalid_request``, ``component_missing``,
``download_failed``), so bolo answers a gated download with 401 ``model_gated`` rather than 503.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import IO, Any

from bolo.engines.base import Engine
from bolo.plugin.sdk import _Server


def error_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    if isinstance(exc, MemoryError):
        return "out_of_memory"
    return type(exc).__name__


class TorchServer(_Server):
    def handle(self, rid: str, op: Any, message: dict[str, Any], payload: bytes) -> None:
        try:
            if op == "load":
                self.load(rid, message)
            elif op == "unload":
                model = self.models.pop(str(message.get("handle")), None)
                if model is not None:
                    model.close()
                self.reply(rid)
            elif op == "transcribe":
                self.transcribe(rid, message, payload)
            elif op == "speakers" and hasattr(self, "speakers"):  # bolo 0.2.0 and later
                self.speakers(rid, message, payload)
            elif op == "synthesize":
                self.synthesize(rid, message)
            elif op in ("separate", "enrol", "extract", "release"):
                self.separation(rid, op, message, payload)
            elif op in ("diarize", "embed") and hasattr(self, "diarization"):  # bolo 0.6.7 and later
                self.diarization(rid, op, message, payload)
            else:
                self.fail(rid, "unknown_op", f"unknown op {op!r}")
        except Exception as exc:
            logging.getLogger("bolo_torch").warning("%s failed: %s: %s", op, type(exc).__name__, exc)
            self.fail(rid, error_code(exc), str(exc)[:2000])
        finally:
            self.cancelled.discard(rid)


def serve(engine: Engine, stdin: IO[bytes] | None = None, stdout: IO[bytes] | None = None) -> int:
    """Like :func:`bolo.plugin.sdk.serve`, with :class:`TorchServer`."""
    out = stdout or sys.stdout.buffer
    if stdout is None:
        sys.stdout = sys.stderr  # stray prints of model libraries must not corrupt the protocol stream
    code = TorchServer(engine, stdin or sys.stdin.buffer, out).run()
    if stdin is None:
        # bolo closed the pipe or asked for shutdown: leave now. Waiting for the worker pool would keep a
        # half-finished synthesis or separation running (and holding memory) after bolo has gone.
        sys.stderr.flush()
        os._exit(code)
    return code
