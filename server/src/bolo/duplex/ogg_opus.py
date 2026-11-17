"""Streaming Ogg Opus: the audio framing of Kyutai's WebSocket protocol (Moshi, Hibiki and the models built on them).

Each audio message of that protocol carries the bytes of an Ogg stream of Opus packets (mono, 24 kHz input rate),
written page by page as audio is produced: first the ``OpusHead`` and ``OpusTags`` pages, then normally one page
per Opus packet (RFC 3533 for Ogg pages, RFC 7845 for Opus in Ogg). The Ogg layer is written and read here in
plain Python; the Opus packets are encoded and decoded with libopus through PyAV (``av``, which the ``duplex`` and
``webrtc`` extras bring).

:class:`OggOpusWriter` turns float PCM into such bytes and :class:`OggOpusReader` turns bytes (in any split, a page
may arrive in pieces) back into float PCM. Kyutai's servers read and write the same framing with their ``sphn``
library (Python) or the ``ogg`` and ``opus`` crates (Rust); the tests check that bolo's framing and sphn's read
each other.
"""

from __future__ import annotations

import struct
from fractions import Fraction
from typing import Any

import numpy as np

from bolo.audio.resample import StreamResampler
from bolo.engines.base import FloatArray
from bolo.errors import ComponentMissing

INSTALL_HINT = 'pip install "bolo-speech[duplex]" (PyAV, which brings libopus)'
#: Opus always decodes at 48 kHz; granule positions in Ogg Opus count 48 kHz samples
OPUS_RATE = 48000
#: the usual encoder delay at 48 kHz (RFC 7845, 4.2)
PRE_SKIP = 312
_CAPTURE = b"OggS"
_HEADER = struct.Struct("<4sBBqIIIB")  # capture, version, flags, granule, serial, sequence, crc, segment count
_CONTINUED, _BOS, _EOS = 0x01, 0x02, 0x04


def _crc_table() -> list[int]:
    table = []
    for i in range(256):
        r = i << 24
        for _ in range(8):
            r = ((r << 1) ^ 0x04C11DB7) if r & 0x80000000 else (r << 1)
        table.append(r & 0xFFFFFFFF)
    return table


_TABLE = _crc_table()


def ogg_crc(data: bytes) -> int:
    """The CRC-32 of an Ogg page (polynomial 0x04C11DB7, no reflection, initial value 0)."""
    crc = 0
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _TABLE[((crc >> 24) & 0xFF) ^ byte]
    return crc


def ogg_page(packet: bytes, *, serial: int, sequence: int, granule: int, flags: int = 0) -> bytes:
    """One Ogg page holding exactly one packet (up to 65025 bytes)."""
    if len(packet) > 255 * 255:
        raise ValueError("a packet of more than 65025 bytes does not fit in one page")
    lacing = [255] * (len(packet) // 255) + [len(packet) % 255]
    header = _HEADER.pack(_CAPTURE, 0, flags, granule, serial, sequence, 0, len(lacing)) + bytes(lacing)
    page = bytearray(header + packet)
    struct.pack_into("<I", page, 22, ogg_crc(bytes(page)))
    return bytes(page)


def opus_head(input_rate: int, channels: int = 1) -> bytes:
    return b"OpusHead" + struct.pack("<BBHIhB", 1, channels, PRE_SKIP, input_rate, 0, 0)


def opus_tags(vendor: str = "bolo") -> bytes:
    raw = vendor.encode()
    return b"OpusTags" + struct.pack("<I", len(raw)) + raw + struct.pack("<I", 0)


def _av() -> Any:
    try:
        import av
    except ImportError as exc:
        raise ComponentMissing(
            f"full-duplex models send their audio as Opus, which needs PyAV: {INSTALL_HINT}"
        ) from exc
    return av


class OggOpusWriter:
    """Float PCM (mono, ``sample_rate``) in, Ogg Opus bytes out; the first call's bytes start with the headers."""

    def __init__(
        self, sample_rate: int = 24000, *, frame_ms: int = 20, bitrate: int = 32000, serial: int = 0x0B010
    ) -> None:
        av = _av()
        self.sample_rate = sample_rate
        self.frame = sample_rate * frame_ms // 1000
        self.serial = serial
        enc = av.CodecContext.create("libopus", "w")
        enc.sample_rate = sample_rate
        enc.layout = "mono"
        enc.format = "s16"
        enc.time_base = Fraction(1, sample_rate)
        enc.bit_rate = bitrate
        enc.options = {"frame_duration": str(frame_ms), "application": "voip"}
        enc.open()
        self._enc = enc
        self._av = av
        self._pending = np.zeros(0, dtype=np.float32)
        self._pts = 0
        self._sequence = 0
        self._granule = PRE_SKIP
        self._started = False

    def _page(self, packet: bytes, granule: int, flags: int = 0) -> bytes:
        page = ogg_page(packet, serial=self.serial, sequence=self._sequence, granule=granule, flags=flags)
        self._sequence += 1
        return page

    def header(self) -> bytes:
        """The OpusHead and OpusTags pages (written once, before the first audio)."""
        if self._started:
            return b""
        self._started = True
        return self._page(opus_head(self.sample_rate), 0, _BOS) + self._page(opus_tags(), 0)

    def append(self, samples: FloatArray) -> bytes:
        """Encode whole frames of ``samples`` (the rest waits for the next call). Returns the pages written."""
        out = bytearray(self.header())
        pcm = np.concatenate([self._pending, np.asarray(samples, dtype=np.float32)])
        whole = len(pcm) - len(pcm) % self.frame
        self._pending = pcm[whole:].copy()
        step = OPUS_RATE // self.sample_rate
        for i in range(0, whole, self.frame):
            chunk = (np.clip(pcm[i : i + self.frame], -1.0, 1.0) * 32767.0).astype("<i2").reshape(1, -1)
            frame = self._av.AudioFrame.from_ndarray(chunk, format="s16", layout="mono")
            frame.sample_rate = self.sample_rate
            frame.pts = self._pts
            self._pts += self.frame
            for packet in self._enc.encode(frame):
                self._granule += self.frame * step
                out += self._page(bytes(packet), self._granule)
        return bytes(out)


class OggOpusReader:
    """Ogg Opus bytes in (any split), float PCM out at ``sample_rate`` (mono)."""

    def __init__(self, sample_rate: int = 24000) -> None:
        av = _av()
        self.sample_rate = sample_rate
        dec = av.CodecContext.create("libopus", "r")
        dec.layout = "mono"
        self._dec = dec
        self._av = av
        self._buf = bytearray()
        self._partial = bytearray()  # a packet continued on the next page
        self._resampler: StreamResampler | None = None
        self._decoded_rate: int | None = None
        #: pages whose CRC did not match (they are skipped)
        self.bad_pages = 0

    def _packets(self) -> list[bytes]:
        packets: list[bytes] = []
        buf = self._buf
        while True:
            start = buf.find(_CAPTURE)
            if start < 0:
                del buf[: max(0, len(buf) - 3)]
                return packets
            if start:
                del buf[:start]
            if len(buf) < _HEADER.size:
                return packets
            _cap, _version, flags, _granule, _serial, _seq, crc, count = _HEADER.unpack_from(buf)
            head = _HEADER.size + count
            if len(buf) < head:
                return packets
            lacing = buf[_HEADER.size : head]
            size = head + sum(lacing)
            if len(buf) < size:
                return packets
            page = bytearray(buf[:size])
            del buf[:size]
            struct.pack_into("<I", page, 22, 0)
            if ogg_crc(bytes(page)) != crc:
                self.bad_pages += 1
                self._partial.clear()
                continue
            if not flags & _CONTINUED:
                self._partial.clear()
            pos = head
            for lace in lacing:
                self._partial += page[pos : pos + lace]
                pos += lace
                if lace < 255:  # the packet ends in this segment
                    packets.append(bytes(self._partial))
                    self._partial.clear()

    def append(self, data: bytes) -> FloatArray:
        """Decode what ``data`` completes; returns the new samples (possibly none)."""
        self._buf += data
        pieces: list[FloatArray] = []
        for packet in self._packets():
            if packet.startswith((b"OpusHead", b"OpusTags")) or not packet:
                continue
            try:
                frames = self._dec.decode(self._av.Packet(packet))
            except Exception:  # a damaged packet: Opus resynchronises on the next one
                continue
            for frame in frames:
                pcm = frame.to_ndarray()
                x = pcm.astype(np.float32) / 32768.0 if pcm.dtype == np.int16 else pcm.astype(np.float32)
                x = x.reshape(-1)  # mono: one plane, or one interleaved channel
                rate = int(frame.sample_rate)
                if self._decoded_rate != rate:
                    self._decoded_rate = rate
                    self._resampler = StreamResampler(rate, self.sample_rate) if rate != self.sample_rate else None
                pieces.append(self._resampler.process(x) if self._resampler is not None else x)
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32)
