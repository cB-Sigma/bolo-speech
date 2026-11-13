"""Encode synthesised audio for delivery, in one shot or as a stream.

==========  ==========================================================
format      route
==========  ==========================================================
``pcm``     raw 16-bit little-endian mono, built in
``wav``     built in (streaming header with unknown length)
``flac``    libsndfile via ``soundfile``
``mp3``     libsndfile >= 1.1 via ``soundfile`` (bundled in its wheels)
``opus``    Ogg/Opus, libsndfile >= 1.1 via ``soundfile``
``aac``     ADTS AAC through PyAV (``bolo-speech[av]``) or ``ffmpeg``
``mulaw``   raw 8 kHz G.711 μ-law, built in (telephony extension)
``alaw``    raw 8 kHz G.711 A-law, built in (telephony extension)
==========  ==========================================================
"""

from __future__ import annotations

import io
import shutil
import struct
import subprocess
from typing import Any

import numpy as np
import numpy.typing as npt

from bolo.audio.codecs import TELEPHONY_RATE, alaw_encode, float_to_pcm16, mulaw_encode
from bolo.audio.resample import StreamResampler, resample
from bolo.errors import UnsupportedFormat

FloatArray = npt.NDArray[np.float32]

FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm", "mulaw", "alaw")

_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg; codecs=opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mulaw": "audio/basic",
    "alaw": "audio/x-alaw",
}

_OPUS_RATES = (8000, 12000, 16000, 24000, 48000)
_MP3_RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)
_AAC_RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)


def media_type_for(fmt: str, sample_rate: int | None = None) -> str:
    fmt = normalize_format(fmt)
    mt = _MEDIA_TYPES[fmt]
    if fmt == "pcm" and sample_rate:
        mt += f"; rate={sample_rate}"
    return mt


def normalize_format(fmt: str) -> str:
    f = fmt.lower().strip()
    aliases = {"ulaw": "mulaw", "pcmu": "mulaw", "pcma": "alaw", "pcm_s16le": "pcm", "ogg": "opus", "wave": "wav"}
    f = aliases.get(f, f)
    if f not in FORMATS:
        raise UnsupportedFormat(f"unsupported response_format {fmt!r}; expected one of {', '.join(FORMATS)}")
    return f


def _nearest_rate(rate: int, allowed: tuple[int, ...]) -> int:
    if rate in allowed:
        return rate
    higher = [r for r in allowed if r >= rate]
    return min(higher) if higher else max(allowed)


def output_rate(fmt: str, model_rate: int, requested: int | None = None) -> int:
    """The sample rate that will actually be delivered for ``fmt``."""
    fmt = normalize_format(fmt)
    if fmt in ("mulaw", "alaw"):
        return requested or TELEPHONY_RATE
    rate = requested or model_rate
    if fmt == "opus":
        return _nearest_rate(rate, _OPUS_RATES)
    if fmt == "mp3":
        return _nearest_rate(rate, _MP3_RATES)
    if fmt == "aac":
        return _nearest_rate(rate, _AAC_RATES)
    return rate


def wav_header(sample_rate: int, n_samples: int | None = None) -> bytes:
    """A 44-byte PCM16 mono WAV header. ``None`` marks a stream of unknown length."""
    data_bytes = 0xFFFFFFFF - 36 if n_samples is None else n_samples * 2
    riff = 0xFFFFFFFF if n_samples is None else 36 + data_bytes
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff, b"WAVE", b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16, b"data", data_bytes,
    )  # fmt: skip


def have_av() -> bool:
    try:
        import av  # noqa: F401

        return True
    except Exception:
        return False


def aac_available() -> bool:
    return have_av() or shutil.which("ffmpeg") is not None


def available_formats() -> dict[str, bool]:
    return {f: (aac_available() if f == "aac" else True) for f in FORMATS}


class _SndfileSink:
    """Incremental libsndfile writer into memory; drains bytes as they appear."""

    def __init__(self, sample_rate: int, fmt: str, subtype: str) -> None:
        import soundfile as sf

        self._buf = io.BytesIO()
        kwargs: dict[str, Any] = {}
        if fmt == "MP3":
            # Constant bitrate keeps the stream decodable without a (rewritten) length header.
            kwargs = {"bitrate_mode": "CONSTANT", "compression_level": 0.5}
        try:
            self._file: Any = sf.SoundFile(
                self._buf, mode="w", samplerate=sample_rate, channels=1, format=fmt, subtype=subtype, **kwargs
            )
        except TypeError:  # soundfile < 0.13 has no bitrate_mode
            self._file = sf.SoundFile(
                self._buf, mode="w", samplerate=sample_rate, channels=1, format=fmt, subtype=subtype
            )
        except Exception as exc:
            raise UnsupportedFormat(
                f"this libsndfile build cannot write {fmt}/{subtype} ({exc}); upgrade the soundfile wheel"
            ) from exc
        self._sent = 0
        # libsndfile starts an MP3 with a Xing/LAME info frame that it rewrites on close.
        # A stream cannot be rewritten, so the info frame is dropped from streamed output.
        self._strip_info = fmt == "MP3"

    def _drain(self) -> bytes:
        view = self._buf.getbuffer()
        if self._strip_info:
            if len(view) < 4:
                return b""
            flen = mp3_frame_length(bytes(view[:4]))
            if flen is None:
                self._strip_info = False
            else:
                if len(view) < flen:
                    return b""
                head = bytes(view[:flen])
                if b"Xing" in head or b"Info" in head:
                    self._sent = flen
                self._strip_info = False
        data = view[self._sent :].tobytes()
        self._sent += len(data)
        return data

    def whole(self) -> bytes:
        """Close and return the complete file, with headers finalised."""
        self._file.close()
        return self._buf.getvalue()

    def feed(self, samples: FloatArray) -> bytes:
        self._file.write(samples)
        return self._drain()

    def finish(self) -> bytes:
        self._file.close()
        return self._drain()


_MP3_BITRATES_V1 = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
_MP3_BITRATES_V2 = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
_MP3_RATES_BY_VERSION = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


def mp3_frame_length(header: bytes) -> int | None:
    """Length in bytes of the MPEG Layer III frame starting with ``header``."""
    if len(header) < 4 or header[0] != 0xFF or (header[1] & 0xE0) != 0xE0:
        return None
    version = (header[1] >> 3) & 0x03
    layer = (header[1] >> 1) & 0x03
    br_idx = header[2] >> 4
    sr_idx = (header[2] >> 2) & 0x03
    padding = (header[2] >> 1) & 0x01
    if version == 1 or layer != 1 or br_idx in (0, 15) or sr_idx == 3:
        return None
    bitrate = (_MP3_BITRATES_V1 if version == 3 else _MP3_BITRATES_V2)[br_idx] * 1000
    rate = _MP3_RATES_BY_VERSION[version][sr_idx]
    return (144 if version == 3 else 72) * bitrate // rate + padding


class _AvAacSink:
    def __init__(self, sample_rate: int) -> None:
        import av

        self._av = av
        self._buf = io.BytesIO()
        self._container = av.open(self._buf, mode="w", format="adts")
        self._stream = self._container.add_stream("aac", rate=sample_rate)
        self._stream.layout = "mono"
        self._rate = sample_rate
        self._sent = 0
        self._pts = 0

    def _drain(self) -> bytes:
        data = self._buf.getbuffer()[self._sent :].tobytes()
        self._sent += len(data)
        return data

    def feed(self, samples: FloatArray) -> bytes:
        frame = self._av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="fltp", layout="mono")
        frame.sample_rate = self._rate
        frame.pts = self._pts
        self._pts += len(samples)
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        return self._drain()

    def finish(self) -> bytes:
        for packet in self._stream.encode(None):
            self._container.mux(packet)
        self._container.close()
        data = self._buf.getvalue()[self._sent :]
        self._sent += len(data)
        return data


class StreamEncoder:
    """Encode float32 mono chunks to ``fmt`` incrementally.

    ``feed`` returns whatever encoded bytes are ready (possibly empty for
    codecs with look-ahead); ``finish`` returns the tail.
    """

    def __init__(self, fmt: str, model_rate: int, requested_rate: int | None = None) -> None:
        self.format = normalize_format(fmt)
        self.model_rate = model_rate
        self.sample_rate = output_rate(self.format, model_rate, requested_rate)
        self.media_type = media_type_for(self.format, self.sample_rate)
        self._resampler = StreamResampler(model_rate, self.sample_rate)
        self._header_sent = False
        self._sink: _SndfileSink | _AvAacSink | None = None
        self._ffmpeg_pcm: list[bytes] | None = None
        self.samples_out = 0
        if self.format == "flac":
            self._sink = _SndfileSink(self.sample_rate, "FLAC", "PCM_16")
        elif self.format == "mp3":
            self._sink = _SndfileSink(self.sample_rate, "MP3", "MPEG_LAYER_III")
        elif self.format == "opus":
            self._sink = _SndfileSink(self.sample_rate, "OGG", "OPUS")
        elif self.format == "aac":
            if have_av():
                self._sink = _AvAacSink(self.sample_rate)
            elif shutil.which("ffmpeg"):
                self._ffmpeg_pcm = []
            else:
                raise UnsupportedFormat(
                    "response_format 'aac' needs FFmpeg: install \"bolo-speech[av]\" or put ffmpeg on PATH"
                )

    def feed(self, samples: npt.NDArray[np.floating], last: bool = False) -> bytes:
        x = self._resampler.process(np.ascontiguousarray(samples, dtype=np.float32), last=last)
        if len(x) == 0:
            return b""
        self.samples_out += len(x)
        if self.format == "pcm":
            return float_to_pcm16(x).tobytes()
        if self.format == "wav":
            body = float_to_pcm16(x).tobytes()
            if not self._header_sent:
                self._header_sent = True
                return wav_header(self.sample_rate) + body
            return body
        if self.format == "mulaw":
            return mulaw_encode(x)
        if self.format == "alaw":
            return alaw_encode(x)
        if self._ffmpeg_pcm is not None:
            self._ffmpeg_pcm.append(float_to_pcm16(x).tobytes())
            return b""
        assert self._sink is not None
        return self._sink.feed(x)

    def finish(self) -> bytes:
        tail = b""
        if self._resampler.sr_in != self._resampler.sr_out:
            tail = self.feed(np.zeros(0, dtype=np.float32), last=True)
        if self.format == "wav" and not self._header_sent:
            self._header_sent = True
            return wav_header(self.sample_rate, 0)
        if self._ffmpeg_pcm is not None:
            return tail + _ffmpeg_encode_aac(b"".join(self._ffmpeg_pcm), self.sample_rate)
        if self._sink is not None:
            return tail + self._sink.finish()
        return tail


def _ffmpeg_encode_aac(pcm: bytes, rate: int) -> bytes:
    exe = shutil.which("ffmpeg")
    assert exe is not None
    proc = subprocess.run(
        [
            exe,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-f",
            "adts",
            "pipe:1",
        ],
        input=pcm,
        capture_output=True,
        check=False,
        timeout=600,
    )
    if proc.returncode != 0:
        raise UnsupportedFormat(f"ffmpeg failed to encode AAC: {proc.stderr.decode(errors='replace')[:200]}")
    return proc.stdout


def encode_audio(
    samples: npt.NDArray[np.floating],
    sample_rate: int,
    fmt: str = "wav",
    requested_rate: int | None = None,
) -> bytes:
    """One-shot encode. WAV output carries a correct length header."""
    fmt = normalize_format(fmt)
    if fmt == "wav":
        rate = output_rate(fmt, sample_rate, requested_rate)
        x = resample(samples, sample_rate, rate)
        return wav_header(rate, len(x)) + float_to_pcm16(x).tobytes()
    enc = StreamEncoder(fmt, sample_rate, requested_rate)
    if isinstance(enc._sink, _SndfileSink):
        x = resample(samples, sample_rate, enc.sample_rate)
        if len(x):
            enc._sink._file.write(x)
        return enc._sink.whole()
    return enc.feed(samples) + enc.finish()
