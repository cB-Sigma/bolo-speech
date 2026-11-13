"""The wire format between bolo and an out-of-process engine plugin (protocol version 1).

A plugin is a separate process that bolo starts and talks to over the plugin's standard input and
output. Each message is one line of UTF-8 JSON (an object) terminated by ``\\n``. When the object
has a ``"bytes": n`` member, exactly ``n`` raw bytes follow the newline: audio as little-endian
float32 samples. Standard error is free for the plugin's own logging (bolo forwards it to its log).

Every request carries an ``id`` (a string chosen by bolo); every reply to it carries the same
``id``. A request is answered by exactly one final message (``"ok": true`` with results, or
``"ok": false`` with ``"error": {"code", "message"}``); ``synthesize`` sends any number of
``{"id", "chunk": seq, "bytes": n}`` messages before its final one. Replies to different requests
may interleave: a plugin may work on several requests at once, but need not.

Requests (``op``):

========== ============================================================ ==============================
op         request members                                              final reply members
========== ============================================================ ==============================
hello      ``protocol`` (int, the highest version bolo speaks)          ``protocol``, ``name``,
                                                                        ``version``, ``kinds``
health     -                                                            ``rss_mb`` (optional)
load       ``model``, ``kind``, ``task``, ``path``, ``options``,        ``handle``, ``sample_rate``,
           ``threads``, ``device``                                      ``info`` (see below)
unload     ``handle``                                                   -
transcribe ``handle``, ``requests``: list of ``{n, language, task,      ``results``: list of
           prompt, word_timestamps, temperature, hotwords}``; the       ``{text, language, segments,
           audio of all requests follows, concatenated (``n``           words}``
           samples each)
synthesize ``handle``, ``text``, ``voice``, ``speed``                   ``sample_rate`` (chunks before)
cancel     ``target`` (the id of a running request)                     -
shutdown   -                                                            - (then the process exits)
========== ============================================================ ==============================

``info`` of a loaded model may hold ``voices`` (list of ``{id, name, language, gender}``),
``languages``, ``max_batch`` (default 1: no batching), ``supports_translate`` and
``supports_word_timestamps``. Segments are ``{start, end, text, words}``, words ``{w, t0, t1}``.

Versioning: bolo sends the highest version it speaks in ``hello``; the plugin answers with the
version it will use (at most bolo's). A plugin that cannot speak any version bolo knows answers
with ``ok: false``. New optional members may be added within a version; anything incompatible
gets a new version number.
"""

from __future__ import annotations

import json
from typing import IO, Any

import numpy as np

from bolo.engines.base import FloatArray

PROTOCOL_VERSION = 1
MAX_LINE = 16 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


def write_message(stream: IO[bytes], message: dict[str, Any], payload: bytes | None = None) -> None:
    """Write one message (and its payload). The caller serialises concurrent writers."""
    if payload is not None:
        message = {**message, "bytes": len(payload)}
    stream.write(json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n")
    if payload:
        stream.write(payload)
    stream.flush()


def read_message(stream: IO[bytes]) -> tuple[dict[str, Any], bytes] | None:
    """The next message and its payload, or None at end of stream."""
    line = stream.readline(MAX_LINE)
    if not line:
        return None
    if not line.endswith(b"\n"):
        raise ProtocolError("message line too long or truncated")
    try:
        message = json.loads(line)
    except ValueError as exc:
        raise ProtocolError(f"not a JSON line: {line[:120]!r}") from exc
    if not isinstance(message, dict):
        raise ProtocolError("a message must be a JSON object")
    size = int(message.get("bytes") or 0)
    payload = b""
    while len(payload) < size:
        part = stream.read(size - len(payload))
        if not part:
            raise ProtocolError("stream ended inside a payload")
        payload += part
    return message, payload


def audio_bytes(samples: FloatArray) -> bytes:
    return np.ascontiguousarray(samples, dtype="<f4").tobytes()


def audio_from_bytes(payload: bytes) -> FloatArray:
    return np.frombuffer(payload, dtype="<f4").astype(np.float32)
