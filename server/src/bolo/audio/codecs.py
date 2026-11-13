"""Sample-format conversion and the G.711 telephony codecs.

G.711 μ-law (PCMU, North America/Japan) and A-law (PCMA, elsewhere) are
implemented from ITU-T G.711 with lookup tables, so encoding and decoding
are single vectorised numpy indexing operations. There is no dependency on
the ``audioop`` module, which was removed from the standard library in
Python 3.13.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
Int16Array = npt.NDArray[np.int16]

TELEPHONY_RATE = 8000
RTP_FRAME_MS = 20


def pcm16_to_float(data: bytes | bytearray | memoryview | Int16Array) -> FloatArray:
    arr = data if isinstance(data, np.ndarray) else np.frombuffer(_even(data), dtype="<i2")
    return (arr.astype(np.float32) / 32768.0).astype(np.float32)


def float_to_pcm16(samples: npt.NDArray[Any]) -> Int16Array:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return np.round(clipped * 32767.0).astype("<i2")


def _even(data: bytes | bytearray | memoryview, width: int = 2) -> bytes | bytearray | memoryview:
    extra = len(data) % width
    return data[: len(data) - extra] if extra else data


def pcm_f32_to_float(data: bytes | bytearray | memoryview) -> FloatArray:
    return np.frombuffer(_even(data, 4), dtype="<f4").astype(np.float32)


# -- G.711 --------------------------------------------------------------

_MULAW_BIAS = 0x84
_MULAW_CLIP = 32635


def _build_mulaw_decode() -> Int16Array:
    u = (~np.arange(256, dtype=np.int32)) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = (((mantissa << 3) + _MULAW_BIAS) << exponent) - _MULAW_BIAS
    return np.where(sign != 0, -sample, sample).astype(np.int16)


def _build_mulaw_encode() -> npt.NDArray[np.uint8]:
    # Follows the ITU/Sun reference: work on the 14-bit magnitude.
    pcm = np.arange(-32768, 32768, dtype=np.int32) >> 2
    mask = np.where(pcm < 0, 0x7F, 0xFF)
    mag = np.minimum(np.abs(pcm), _MULAW_CLIP >> 2) + (_MULAW_BIAS >> 2)
    seg_end = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)
    seg = np.searchsorted(seg_end, mag, side="left").astype(np.int32)
    uval = np.where(seg >= 8, 0x7F, (seg << 4) | ((mag >> (seg + 1)) & 0x0F))
    return ((uval ^ mask) & 0xFF).astype(np.uint8)


def _build_alaw_decode() -> Int16Array:
    a = np.arange(256, dtype=np.int32) ^ 0x55
    t = (a & 0x0F) << 4
    seg = (a & 0x70) >> 4
    t = np.where(seg == 0, t + 8, np.where(seg == 1, t + 0x108, (t + 0x108) << np.maximum(seg - 1, 0)))
    return np.where((a & 0x80) != 0, t, -t).astype(np.int16)


def _build_alaw_encode() -> npt.NDArray[np.uint8]:
    pcm = np.arange(-32768, 32768, dtype=np.int32) >> 3
    mask = np.where(pcm >= 0, 0xD5, 0x55)
    mag = np.where(pcm >= 0, pcm, -pcm - 1)
    seg_end = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF], dtype=np.int32)
    seg = np.searchsorted(seg_end, mag, side="left").astype(np.int32)
    low = np.where(seg < 2, (mag >> 1) & 0x0F, (mag >> np.maximum(seg, 1)) & 0x0F)
    aval = np.where(seg >= 8, 0x7F, (seg << 4) | low)
    return ((aval ^ mask) & 0xFF).astype(np.uint8)


_MULAW_DECODE = _build_mulaw_decode()
_MULAW_ENCODE = _build_mulaw_encode()
_ALAW_DECODE = _build_alaw_decode()
_ALAW_ENCODE = _build_alaw_encode()


def mulaw_decode_pcm16(data: bytes | bytearray | memoryview) -> Int16Array:
    return _MULAW_DECODE[np.frombuffer(data, dtype=np.uint8)]


def alaw_decode_pcm16(data: bytes | bytearray | memoryview) -> Int16Array:
    return _ALAW_DECODE[np.frombuffer(data, dtype=np.uint8)]


def mulaw_decode(data: bytes | bytearray | memoryview) -> FloatArray:
    """G.711 μ-law bytes to float32 samples in [-1, 1]."""
    return pcm16_to_float(mulaw_decode_pcm16(data))


def alaw_decode(data: bytes | bytearray | memoryview) -> FloatArray:
    """G.711 A-law bytes to float32 samples in [-1, 1]."""
    return pcm16_to_float(alaw_decode_pcm16(data))


def _as_pcm16(samples: npt.NDArray[Any]) -> Int16Array:
    if samples.dtype == np.int16:
        return samples
    return float_to_pcm16(samples)


def mulaw_encode(samples: npt.NDArray[Any]) -> bytes:
    """Float32 (or int16) samples to G.711 μ-law bytes."""
    pcm = _as_pcm16(samples)
    return _MULAW_ENCODE[pcm.astype(np.int32) + 32768].tobytes()


def alaw_encode(samples: npt.NDArray[Any]) -> bytes:
    """Float32 (or int16) samples to G.711 A-law bytes."""
    pcm = _as_pcm16(samples)
    return _ALAW_ENCODE[pcm.astype(np.int32) + 32768].tobytes()


# -- framing ------------------------------------------------------------


def frame_bytes(sample_rate: int = TELEPHONY_RATE, frame_ms: int = RTP_FRAME_MS, bytes_per_sample: int = 1) -> int:
    """Payload size of one frame: 160 bytes for 20 ms of 8 kHz G.711."""
    return sample_rate * frame_ms // 1000 * bytes_per_sample


def iter_frames(payload: bytes, size: int, *, pad: bytes | None = None) -> Iterator[bytes]:
    """Split ``payload`` into fixed-size frames (RTP-style packetisation).

    The last short frame is padded with ``pad`` (one byte, repeated) when
    given, otherwise it is yielded as is.
    """
    if size <= 0:
        raise ValueError("frame size must be positive")
    for off in range(0, len(payload), size):
        chunk = payload[off : off + size]
        if len(chunk) < size and pad is not None:
            chunk = chunk + pad * (size - len(chunk))
        yield chunk


class Framer:
    """Re-chunk an arbitrary byte stream into fixed-size frames."""

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("frame size must be positive")
        self.size = size
        self._buf = bytearray()

    def push(self, data: bytes) -> list[bytes]:
        self._buf.extend(data)
        out: list[bytes] = []
        while len(self._buf) >= self.size:
            out.append(bytes(self._buf[: self.size]))
            del self._buf[: self.size]
        return out

    def flush(self, pad: bytes | None = None) -> bytes | None:
        if not self._buf:
            return None
        chunk = bytes(self._buf)
        self._buf.clear()
        if pad is not None:
            chunk += pad * (self.size - len(chunk))
        return chunk


MULAW_SILENCE = b"\xff"
ALAW_SILENCE = b"\xd5"

ENCODINGS = ("pcm_s16le", "pcm_f32le", "mulaw", "alaw")


def decode_frame(data: bytes | bytearray | memoryview, encoding: str) -> FloatArray:
    """Decode one raw audio frame of a streaming ``encoding`` to float32 mono."""
    if encoding == "pcm_s16le":
        return pcm16_to_float(data)
    if encoding == "pcm_f32le":
        return pcm_f32_to_float(data)
    if encoding in ("mulaw", "ulaw", "pcmu"):
        return mulaw_decode(data)
    if encoding in ("alaw", "pcma"):
        return alaw_decode(data)
    raise ValueError(f"unsupported encoding {encoding!r}; expected one of {', '.join(ENCODINGS)}")


def encode_frame(samples: npt.NDArray[np.floating], encoding: str) -> bytes:
    if encoding == "pcm_s16le":
        return float_to_pcm16(samples).tobytes()
    if encoding == "pcm_f32le":
        return np.asarray(samples, dtype="<f4").tobytes()
    if encoding in ("mulaw", "ulaw", "pcmu"):
        return mulaw_encode(samples)
    if encoding in ("alaw", "pcma"):
        return alaw_encode(samples)
    raise ValueError(f"unsupported encoding {encoding!r}; expected one of {', '.join(ENCODINGS)}")
