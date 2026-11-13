"""Decode uploaded audio to mono float32.

Route, in order:

1. libsndfile through the ``soundfile`` wheel: WAV, FLAC, OGG/Vorbis,
   OGG/Opus, MP3, AIFF, AU and friends. No system packages needed.
2. PyAV (``pip install 'bolo-speech[av]'``; the wheel bundles FFmpeg's
   libraries): M4A/AAC, WebM, MP4, AMR, WMA and anything else FFmpeg reads.
3. An ``ffmpeg`` executable on ``PATH`` as a last resort.

Raw headerless audio (PCM, μ-law, A-law) is handled by :func:`decode_raw`.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess

import numpy as np
import numpy.typing as npt

from bolo.audio.codecs import decode_frame
from bolo.errors import AudioDecodeError

FloatArray = npt.NDArray[np.float32]

_NEEDS_AV = ("m4a", "mp4", "aac", "webm", "mka", "mkv", "amr", "wma", "3gp", "mov")

_RAW_TYPES = {
    "audio/pcm": "pcm_s16le",
    "audio/l16": "pcm_s16le",
    "audio/x-raw": "pcm_s16le",
    "audio/basic": "mulaw",
    "audio/pcmu": "mulaw",
    "audio/x-mulaw": "mulaw",
    "audio/mulaw": "mulaw",
    "audio/pcma": "alaw",
    "audio/x-alaw": "alaw",
    "audio/alaw": "alaw",
}
_RAW_EXT = {
    "pcm": "pcm_s16le",
    "raw": "pcm_s16le",
    "ulaw": "mulaw",
    "mulaw": "mulaw",
    "ul": "mulaw",
    "alaw": "alaw",
    "al": "alaw",
}


def have_av() -> bool:
    try:
        import av  # noqa: F401

        return True
    except Exception:
        return False


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def raw_encoding_for(content_type: str | None, filename: str | None) -> tuple[str, int] | None:
    """Return ``(encoding, sample_rate)`` if the upload is headerless raw audio."""
    if content_type:
        base = content_type.split(";")[0].strip().lower()
        enc = _RAW_TYPES.get(base)
        if enc:
            m = re.search(r"rate=(\d+)", content_type)
            default = 8000 if enc in ("mulaw", "alaw") else 16000
            return enc, int(m.group(1)) if m else default
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        enc = _RAW_EXT.get(ext)
        if enc:
            return enc, 8000 if enc in ("mulaw", "alaw") else 16000
    return None


def decode_raw(data: bytes, encoding: str, sample_rate: int) -> tuple[FloatArray, int]:
    try:
        return decode_frame(data, encoding), sample_rate
    except ValueError as exc:
        raise AudioDecodeError(str(exc)) from exc


def _to_mono(x: npt.NDArray[np.floating]) -> FloatArray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


def _decode_soundfile(data: bytes) -> tuple[FloatArray, int]:
    import soundfile as sf

    with sf.SoundFile(io.BytesIO(data)) as f:
        samples = f.read(dtype="float32", always_2d=False)
        return _to_mono(samples), int(f.samplerate)


def _decode_av(data: bytes) -> tuple[FloatArray, int]:
    import av

    chunks: list[FloatArray] = []
    with av.open(io.BytesIO(data), mode="r") as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise AudioDecodeError("the file contains no audio stream")
        rate = int(stream.codec_context.sample_rate or 16000)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1).astype(np.float32))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1).astype(np.float32))
    if not chunks:
        return np.zeros(0, dtype=np.float32), rate
    return np.concatenate(chunks), rate


def _decode_ffmpeg(data: bytes) -> tuple[FloatArray, int]:
    exe = shutil.which("ffmpeg")
    assert exe is not None
    proc = subprocess.run(
        [exe, "-nostdin", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        input=data,
        capture_output=True,
        check=False,
        timeout=600,
    )
    if proc.returncode != 0:
        raise AudioDecodeError(f"ffmpeg could not decode the audio: {proc.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32), 16000


def decode_audio(
    data: bytes,
    *,
    filename: str | None = None,
    content_type: str | None = None,
) -> tuple[FloatArray, int]:
    """Decode an uploaded audio file. Returns ``(mono float32 samples, sample_rate)``."""
    if not data:
        raise AudioDecodeError("the uploaded audio file is empty")
    raw = raw_encoding_for(content_type, filename)
    looks_contained = data[:4] in (b"RIFF", b"fLaC", b"OggS", b"FORM", b".snd") or data[:3] == b"ID3"
    if raw is not None and not looks_contained:
        return decode_raw(data, raw[0], raw[1])

    errors: list[str] = []
    try:
        return _decode_soundfile(data)
    except AudioDecodeError:
        raise
    except Exception as exc:
        errors.append(f"libsndfile: {exc}")
    if have_av():
        try:
            return _decode_av(data)
        except AudioDecodeError:
            raise
        except Exception as exc:
            errors.append(f"PyAV: {exc}")
    elif have_ffmpeg():
        return _decode_ffmpeg(data)

    ext = filename.rsplit(".", 1)[-1].lower() if filename and "." in filename else ""
    hint = ""
    if not have_av() and (ext in _NEEDS_AV or not ext):
        hint = " This container needs FFmpeg: install the bundled decoder with \"pip install 'bolo-speech[av]'\" or put ffmpeg on PATH."
    raise AudioDecodeError(f"could not decode the audio file ({'; '.join(errors)}).{hint}")
