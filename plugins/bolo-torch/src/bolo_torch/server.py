"""The plugin's protocol server: bolo's SDK server, with error codes that bolo understands.

bolo's SDK reports an exception by its class name; this server sends the ``code`` of
:class:`bolo_torch.runtime.PluginError` instead (``model_gated``, ``invalid_request``, ``component_missing``,
``download_failed``), so bolo answers a gated download with 401 ``model_gated`` rather than 503.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from typing import IO, Any

from bolo.engines.base import Engine
from bolo.plugin.sdk import _Server, protocol_streams

from bolo_torch.runtime import release_memory


def error_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    if isinstance(exc, MemoryError):
        return "out_of_memory"
    return type(exc).__name__


class TorchServer(_Server):
    def handle(self, rid: str, op: Any, message: dict[str, Any], payload: bytes) -> None:
        failed_load: tuple[str, str] | None = None
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
            elif op == "respond" and hasattr(self, "respond"):  # bolo 0.6.7 and later
                self.respond(rid, message, payload)
            elif op == "denoise" and hasattr(self, "denoise"):  # bolo 0.6.7 and later
                self.denoise(rid, message, payload)
            elif op == "lid" and hasattr(self, "lid"):  # bolo 0.6.7 and later
                self.lid(rid, message, payload)
            elif op == "intent" and hasattr(self, "intent"):  # bolo with the intent task
                self.intent(rid, message, payload)
            else:
                self.fail(rid, "unknown_op", f"unknown op {op!r}")
        except Exception as exc:
            logging.getLogger("bolo_torch").warning("%s failed: %s: %s", op, type(exc).__name__, str(exc))
            if op == "load":
                failed_load = (error_code(exc), str(exc)[:2000])
                # whoever still holds the exception (a log handler, a debugger) must not keep the loader's locals
                traceback.clear_frames(exc.__traceback__)
            else:
                self.fail(rid, error_code(exc), str(exc)[:2000])
        finally:
            self.cancelled.discard(rid)
        if failed_load is not None:
            # out of the except block: the exception, its traceback and the loader's frames (which hold the weights
            # loaded so far) are gone, so the collector can free them and torch can give their memory back
            self.load_failed(rid, *failed_load)

    def load_failed(self, rid: str, code: str, text: str) -> None:
        """Answer a failed load after freeing what it left on the GPU; ``error.gpu`` says what is still held, so
        bolo can restart the process when no other model needs it (the CUDA context stays until the process ends)."""
        error: dict[str, Any] = {"code": code, "message": text}
        try:
            gpu = release_memory()
        except Exception as exc:  # never let the cleanup hide the load's own error
            logging.getLogger("bolo_torch").warning("could not free memory after the failed load: %s", exc)
            gpu = {"initialized": True}
        error["gpu"] = gpu
        if gpu.get("initialized"):
            logging.getLogger("bolo_torch").info(
                "after the failed load the GPU holds %.0f MB allocated, %.0f MB reserved by this process",
                gpu.get("allocated_bytes", 0) / 1e6, gpu.get("reserved_bytes", 0) / 1e6,
            )  # fmt: skip
        self.send({"id": rid, "ok": False, "error": error})


def serve(engine: Engine, stdin: IO[bytes] | None = None, stdout: IO[bytes] | None = None) -> int:
    """Like :func:`bolo.plugin.sdk.serve`, with :class:`TorchServer`."""
    standalone = stdin is None and stdout is None
    if standalone:
        # the protocol gets private copies of stdin and stdout; model libraries get an empty stdin (a first-use
        # prompt ends at once instead of reading the protocol pipe forever) and stdout goes to stderr
        stdin, stdout = protocol_streams()
    code = TorchServer(engine, stdin or sys.stdin.buffer, stdout or sys.stdout.buffer).run()
    if standalone:
        # bolo closed the pipe or asked for shutdown: leave now. Waiting for the worker pool would keep a
        # half-finished synthesis or separation running (and holding memory) after bolo has gone.
        sys.stderr.flush()
        os._exit(code)
    return code
