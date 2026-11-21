"""The wire format between bolo and an out-of-process engine plugin (protocol version 1).

A plugin is a separate process that bolo starts and talks to over the plugin's standard input and
output. Each message is one line of UTF-8 JSON (an object) terminated by ``\\n``. When the object
has a ``"bytes": n`` member, exactly ``n`` raw bytes follow the newline: audio as little-endian
float32 samples. Standard error is free for the plugin's own logging (bolo forwards it to its log).

Every request carries an ``id`` (a string chosen by bolo); every reply to it carries the same
``id``. A request is answered by exactly one final message (``"ok": true`` with results, or
``"ok": false`` with ``"error": {"code", "message"}``); ``synthesize`` sends any number of
``{"id", "chunk": seq, "bytes": n}`` messages before its final one; ``respond`` sends chunks that may also carry
``text`` (a piece of the answer's text) or ``tool_call`` (``{call_id, name, arguments}``), with or without audio. Replies to different requests
may interleave: a plugin may work on several requests at once, but need not.

Requests (``op``):

========== ============================================================ ==============================
op         request members                                              final reply members
========== ============================================================ ==============================
hello      ``protocol`` (int, the highest version bolo speaks)          ``protocol``, ``name``,
                                                                        ``version``, ``kinds``
health     -                                                            ``rss_mb`` (optional)
load       ``model``, ``kind``, ``task``, ``path``, ``options``,        ``handle``, ``sample_rate``,
           ``threads``, ``device``, ``placement`` (optional)            ``info`` (see below)
unload     ``handle``                                                   -
transcribe ``handle``, ``requests``: list of ``{n, language, task,      ``results``: list of
           prompt, word_timestamps, temperature, hotwords}``; the       ``{text, language, segments,
           audio of all requests follows, concatenated (``n``           words}``
           samples each)
speakers   ``handle``, ``n``, ``language``, ``task``, ``prompt``,          ``results``: one per speaker
           ``word_timestamps``, ``temperature``, ``frame_s``,             (as for ``transcribe``), times
           ``frames``, ``speakers``; ``n`` samples of audio follow,       from the start of the audio
           then the speaker activity: ``frames`` x ``speakers``
           float32, row by row (frame by frame)
synthesize ``handle``, ``text``, ``voice``, ``speed``                   ``sample_rate`` (chunks before)
separate   ``handle``; the mixture follows (16 kHz)                     ``tracks`` (count); the tracks
                                                                        follow, concatenated, equal length
enrol      ``handle``, ``key``; the voice sample follows (16 kHz)       -
extract    ``handle``, ``key``; the mixture follows (16 kHz)            the enrolled voice's track follows
release    ``handle``, ``key``                                          -
diarize    ``handle``, ``num_speakers``, ``threshold`` (each may be     ``turns``: list of
           null); the recording follows (16 kHz)                        ``[start, end, speaker]``
embed      ``handle``; a stretch of one voice follows (16 kHz)          the unit-length embedding
                                                                        follows (float32)
denoise    ``handle``; the recording follows (at the model's            the enhanced recording follows,
           ``sample_rate``)                                             as long as the input
lid        ``handle``; the recording follows (16 kHz)                     ``languages``: list of
                                                                        ``[code, probability]``, most
                                                                        likely first
intent     ``handle``; one utterance follows (16 kHz)                   ``intent``: ``{intent, scenario,
                                                                        action, slots, text}``, slots
                                                                        a list of ``{type, value}``
respond    ``handle``, ``turns``: list of ``{role, text, n, call_id,     ``sample_rate``, ``chunks``,
           tool_calls}``, ``instructions``, ``tools``,                    ``cancelled`` (chunks before)
           ``tool_choice``, ``voice``, ``max_output_tokens``,
           ``audio_out``; the audio of the turns follows,
           concatenated (``n`` samples each, 16 kHz)
cancel     ``target`` (the id of a running request)                     -
shutdown   -                                                            - (then the process exits)
========== ============================================================ ==============================

``info`` of a loaded model may hold ``voices`` (list of ``{id, name, language, gender}``),
``languages``, ``max_batch`` (default 1: no batching), ``supports_translate`` and
``supports_word_timestamps``, ``request_fields`` (after bolo 0.6.7: which of ``prompt``, ``hotwords`` and
``temperature`` the model uses; bolo reports the others as ignored; not given: all three) and ``speaker_conditioned`` (true: the model answers ``speakers``, diarization-
conditioned recognition of a whole recording, bolo sends it only to such models); a separation model (task ``separation``) gives ``mode``
(``"blind"``: answers ``separate``; ``"target"``: answers ``enrol``/``extract``/``release``) and
``min_sample_s`` (seconds of voice a target model needs); a diarization model (task ``diarization``, bolo 0.6.7)
gives ``consolidate_default`` (false: bolo does not merge or drop its speakers); a speech-to-speech model (task
``speech-to-speech``, bolo 0.6.7) gives ``voices`` and ``supports_tools``; a language identifier (task ``lid``,
bolo 0.6.7) gives ``labels`` (the language codes it can answer) and ``max_s`` (the seconds of audio it looks at); a
spoken-intent model (task ``intent``) may give ``max_s`` (the longest utterance it takes, default 30). The model keeps no conversation: each
``respond`` carries the whole conversation (user turns as audio or text, the model's earlier answers as text, tool
results), and the answer streams back as chunks at the model's ``sample_rate``. ``placement`` in a ``load``
request (bolo 0.5) is
``{"device_map": "auto", "max_memory": {"0": bytes, "cpu": bytes}}`` when bolo's scheduler found less free
accelerator memory than the model needs: a plugin that can split the model (accelerate's ``device_map``) may use it,
and answers with ``info.placement`` = ``{"gpu_bytes", "cpu_bytes", "device_map"}``; a plugin that cannot ignores
it. ``info.device`` (optional, bolo 0.6.11) is the device the plugin put the model on (``mps``, ``cuda:0``, ``cpu``):
when that is an accelerator and bolo placed the model on the CPU, bolo moves its placement there. Segments are
``{start, end, text, words}``, words ``{w, t0, t1}``.

Versioning: bolo sends the highest version it speaks in ``hello``; the plugin answers with the
version it will use (at most bolo's). A plugin that cannot speak any version bolo knows answers
with ``ok: false``. New optional members may be added within a version, and new ops that bolo
only sends to models announcing them in ``info`` (``speakers``, added in bolo 0.3; ``diarize`` and ``embed`` are sent only to models loaded with task
``diarization``, bolo 0.6.7; ``denoise`` only to models loaded with task ``denoise``, bolo 0.6.7; ``respond`` only
to models loaded with task ``speech-to-speech``, bolo 0.6.7; ``lid`` only to models loaded with task ``lid``, bolo
0.6.7; ``intent`` only to models loaded with task ``intent``); anything
incompatible gets a new version number.
"""

from __future__ import annotations

import json
from typing import IO, Any

import numpy as np

from bolo.engines.base import ConversationTurn, FloatArray

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


def speaker_payload(samples: FloatArray, probs: FloatArray) -> bytes:
    """The payload of a ``speakers`` request: the audio, then the activity matrix (frames x speakers)."""
    return audio_bytes(samples) + np.ascontiguousarray(probs, dtype="<f4").tobytes()


def split_speaker_payload(payload: bytes, n: int, frames: int, speakers: int) -> tuple[FloatArray, FloatArray]:
    """Audio and activity ``(frames, speakers)`` of a ``speakers`` request."""
    data = audio_from_bytes(payload)
    if len(data) != n + frames * speakers:
        raise ProtocolError(f"speakers payload holds {len(data)} values, expected {n} + {frames} x {speakers}")
    return data[:n], data[n:].reshape(frames, speakers)


def turns_message(turns: list[ConversationTurn]) -> tuple[list[dict[str, Any]], bytes]:
    """The ``turns`` of a ``respond`` request and its payload (the audio of the turns, 16 kHz, concatenated)."""
    items: list[dict[str, Any]] = []
    audio: list[bytes] = []
    for t in turns:
        n = 0 if t.audio is None else len(t.audio)
        item: dict[str, Any] = {"role": t.role, "text": t.text, "n": n}
        if t.call_id:
            item["call_id"] = t.call_id
        if t.tool_calls:
            item["tool_calls"] = [dict(c) for c in t.tool_calls]
        items.append(item)
        if t.audio is not None and n:
            audio.append(audio_bytes(t.audio))
    return items, b"".join(audio)


def turns_from_message(items: list[dict[str, Any]], payload: bytes) -> list[ConversationTurn]:
    """The conversation of a ``respond`` request (the inverse of :func:`turns_message`)."""
    data = audio_from_bytes(payload)
    need = sum(int(i.get("n") or 0) for i in items)
    if need != len(data):
        raise ProtocolError(f"respond payload holds {len(data)} samples, the turns {need}")
    turns: list[ConversationTurn] = []
    offset = 0
    for i in items:
        n = int(i.get("n") or 0)
        turns.append(
            ConversationTurn(
                role=str(i.get("role") or "user"), text=str(i.get("text") or ""),
                audio=data[offset : offset + n] if n else None, call_id=i.get("call_id"),
                tool_calls=[{k: str(v) for k, v in c.items()} for c in i.get("tool_calls") or []],
            )
        )  # fmt: skip
        offset += n
    return turns
