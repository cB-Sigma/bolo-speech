from __future__ import annotations

import io

import numpy as np
import pytest

from bolo.audio import codecs, decode_audio, encode_audio, resample
from bolo.audio.codecs import (
    Framer,
    alaw_decode,
    alaw_encode,
    decode_frame,
    encode_frame,
    float_to_pcm16,
    frame_bytes,
    iter_frames,
    mulaw_decode,
    mulaw_encode,
    pcm16_to_float,
)
from bolo.audio.decode import decode_raw, raw_encoding_for
from bolo.audio.encode import StreamEncoder, media_type_for, mp3_frame_length, normalize_format, output_rate, wav_header
from bolo.audio.resample import StreamResampler
from bolo.errors import AudioDecodeError, UnsupportedFormat


def tone(freq: float = 440.0, seconds: float = 1.0, rate: int = 16000, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def snr_db(reference: np.ndarray, signal: np.ndarray) -> float:
    n = min(len(reference), len(signal))
    noise = reference[:n] - signal[:n]
    return float(10 * np.log10(np.sum(reference[:n] ** 2) / (np.sum(noise**2) + 1e-12)))


def dominant_freq(x: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.argmax(spectrum) * rate / len(x))


# -- G.711 ----------------------------------------------------------------
def test_g711_reference_points() -> None:
    # Values from ITU-T G.711: digital silence and full scale.
    assert mulaw_encode(np.array([0], dtype=np.int16)) == b"\xff"
    assert alaw_encode(np.array([0], dtype=np.int16)) == b"\xd5"
    assert mulaw_encode(np.array([32767], dtype=np.int16)) == b"\x80"
    assert mulaw_encode(np.array([-32768], dtype=np.int16)) == b"\x00"
    assert alaw_encode(np.array([32767], dtype=np.int16)) == b"\xaa"
    assert alaw_encode(np.array([-32768], dtype=np.int16)) == b"\x2a"
    assert codecs.mulaw_decode_pcm16(b"\xff")[0] == 0
    assert codecs.mulaw_decode_pcm16(b"\x80")[0] == 32124
    assert codecs.alaw_decode_pcm16(b"\xd5")[0] == 8
    assert codecs.alaw_decode_pcm16(b"\xaa")[0] == 32256


def test_g711_matches_stdlib_audioop_when_available() -> None:
    audioop = pytest.importorskip("audioop", reason="audioop was removed in Python 3.13")
    pcm = np.arange(-32768, 32768, dtype=np.int16)
    assert mulaw_encode(pcm) == audioop.lin2ulaw(pcm.tobytes(), 2)
    assert alaw_encode(pcm) == audioop.lin2alaw(pcm.tobytes(), 2)
    every = bytes(range(256))
    assert codecs.mulaw_decode_pcm16(every).tobytes() == audioop.ulaw2lin(every, 2)
    assert codecs.alaw_decode_pcm16(every).tobytes() == audioop.alaw2lin(every, 2)


@pytest.mark.parametrize(("enc", "dec"), [(mulaw_encode, mulaw_decode), (alaw_encode, alaw_decode)])
def test_g711_round_trip_quality(enc, dec) -> None:
    x = tone(440, 1.0, 8000, 0.5)
    y = dec(enc(x))
    assert len(y) == len(x)
    assert snr_db(x, y) > 30  # G.711 gives roughly 38 dB on a half-scale tone
    # idempotent on its own code points
    assert enc(dec(enc(x))) == enc(x)


def test_g711_encode_is_monotonic_and_symmetric() -> None:
    pcm = np.arange(0, 32768, 16, dtype=np.int16)
    decoded = codecs.mulaw_decode_pcm16(mulaw_encode(pcm)).astype(np.int32)
    assert np.all(np.diff(decoded) >= 0)
    neg = codecs.mulaw_decode_pcm16(mulaw_encode((-pcm[1:]).astype(np.int16))).astype(np.int32)
    assert np.array_equal(neg, -decoded[1:])


def test_pcm_conversions() -> None:
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    pcm = float_to_pcm16(x)
    assert pcm.tolist() == [-32767, -16384, 0, 16384, 32767, 32767]
    assert np.allclose(pcm16_to_float(pcm.tobytes()), np.clip(x, -1, 1), atol=1e-4)
    assert len(pcm16_to_float(b"\x00\x01\x02")) == 1  # odd trailing byte ignored
    for encoding in ("pcm_s16le", "pcm_f32le", "mulaw", "alaw"):
        y = decode_frame(encode_frame(tone(300, 0.1, 8000), encoding), encoding)
        assert len(y) == 800
    with pytest.raises(ValueError, match="unsupported encoding"):
        decode_frame(b"", "g729")


def test_rtp_framing() -> None:
    assert frame_bytes() == 160  # 20 ms of 8 kHz G.711
    assert frame_bytes(16000, 20, 2) == 640
    payload = mulaw_encode(tone(440, 0.05, 8000))  # 400 bytes
    frames = list(iter_frames(payload, 160, pad=codecs.MULAW_SILENCE))
    assert [len(f) for f in frames] == [160, 160, 160]
    assert frames[-1].endswith(b"\xff" * 80)
    assert [len(f) for f in iter_frames(payload, 160)] == [160, 160, 80]
    framer = Framer(160)
    out = framer.push(payload[:100]) + framer.push(payload[100:])
    assert b"".join(out) == payload[:320]
    assert framer.flush(pad=b"\xff") == payload[320:] + b"\xff" * 80
    assert framer.flush() is None


# -- resampling -----------------------------------------------------------
@pytest.mark.parametrize("force_numpy", [False, True])
@pytest.mark.parametrize(
    ("sr_in", "sr_out"), [(8000, 16000), (16000, 8000), (22050, 16000), (48000, 16000), (24000, 8000)]
)
def test_resample_preserves_tone(sr_in: int, sr_out: int, force_numpy: bool) -> None:
    x = tone(440, 1.0, sr_in)
    y = resample(x, sr_in, sr_out, force_numpy=force_numpy)
    assert abs(len(y) - sr_out) <= 2
    assert abs(dominant_freq(y, sr_out) - 440) < 3
    assert 0.3 < float(np.abs(y[sr_out // 4 : -sr_out // 4]).max()) < 0.6


def test_resample_removes_aliasing() -> None:
    x = tone(3000, 1.0, 16000) + tone(7000, 1.0, 16000)  # 7 kHz cannot exist at 8 kHz
    y = resample(x, 16000, 8000)
    spectrum = np.abs(np.fft.rfft(y))
    assert abs(np.argmax(spectrum) - 3000) < 5
    alias_bin = 1000  # 8000 - 7000
    assert spectrum[alias_bin] < 0.05 * spectrum.max()


def test_stream_resampler_matches_one_shot_length_and_is_continuous() -> None:
    x = tone(440, 2.0, 8000)
    r = StreamResampler(8000, 16000)
    parts = [r.process(x[i : i + 160]) for i in range(0, len(x), 160)]
    parts.append(r.process(np.zeros(0, dtype=np.float32), last=True))
    y = np.concatenate(parts)
    assert abs(len(y) - 32000) <= 16
    assert abs(dominant_freq(y[:16000], 16000) - 440) < 3
    assert float(np.max(np.abs(np.diff(y)))) < 0.12  # no clicks at chunk boundaries
    same = StreamResampler(16000, 16000)
    assert same.process(x) is x


# -- decode / encode ------------------------------------------------------
@pytest.mark.parametrize("fmt", ["wav", "flac", "mp3", "opus"])
def test_encode_decode_round_trip(fmt: str) -> None:
    x = tone(440, 1.0, 24000)
    data = encode_audio(x, 24000, fmt)
    y, rate = decode_audio(data, filename=f"a.{fmt}")
    assert rate == 24000
    assert abs(len(y) - 24000) <= 2400
    assert abs(dominant_freq(y[:24000], rate) - 440) < 5


def test_wav_header_and_pcm() -> None:
    x = tone(440, 0.5, 16000)
    wav = encode_audio(x, 16000, "wav")
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert int.from_bytes(wav[40:44], "little") == len(x) * 2
    assert int.from_bytes(wav[24:28], "little") == 16000
    assert encode_audio(x, 16000, "pcm") == float_to_pcm16(x).tobytes()
    streaming = wav_header(16000)
    assert int.from_bytes(streaming[4:8], "little") == 0xFFFFFFFF


def test_telephony_output_is_8k_g711() -> None:
    x = tone(440, 1.0, 24000)
    mu = encode_audio(x, 24000, "mulaw")
    assert len(mu) == 8000
    assert abs(dominant_freq(mulaw_decode(mu), 8000) - 440) < 3
    al = encode_audio(x, 24000, "alaw")
    assert len(al) == 8000
    assert media_type_for("mulaw") == "audio/basic"


def test_stream_encoders_emit_incrementally_and_decode() -> None:
    x = tone(440, 2.0, 22050)
    for fmt in ("pcm", "wav", "mp3", "flac", "mulaw"):
        enc = StreamEncoder(fmt, 22050)
        parts = [enc.feed(x[i : i + 4410]) for i in range(0, len(x), 4410)]
        assert sum(1 for p in parts if p) >= 5, fmt  # audio leaves before the stream ends
        data = b"".join(parts) + enc.finish()
        if fmt == "mp3":
            assert mp3_frame_length(data[:4]) is not None
            assert b"Xing" not in data[:200] and b"Info" not in data[:200]
        if fmt in ("wav", "flac", "mp3"):
            y, rate = decode_audio(data, filename=f"a.{fmt}")
            assert abs(len(y) / rate - 2.0) < 0.15, fmt
    opus = StreamEncoder("opus", 22050)
    assert opus.sample_rate == 24000  # Opus has no 22.05 kHz mode: resampled
    data = opus.feed(x) + opus.finish()
    y, rate = decode_audio(data, filename="a.opus")
    assert rate == 24000 and abs(len(y) / rate - 2.0) < 0.1


def test_aac_needs_ffmpeg_or_is_valid() -> None:
    from bolo.audio.encode import aac_available

    x = tone(440, 1.0, 24000)
    if not aac_available():
        with pytest.raises(UnsupportedFormat, match="FFmpeg"):
            encode_audio(x, 24000, "aac")
        return
    data = encode_audio(x, 24000, "aac")
    assert data[0] == 0xFF and (data[1] & 0xF0) == 0xF0  # ADTS sync word
    y, rate = decode_audio(data, filename="a.aac")
    assert abs(len(y) / rate - 1.0) < 0.15


def test_format_helpers() -> None:
    assert normalize_format("ULAW") == "mulaw"
    assert normalize_format("pcm_s16le") == "pcm"
    with pytest.raises(UnsupportedFormat):
        normalize_format("wma")
    assert output_rate("opus", 22050) == 24000
    assert output_rate("mulaw", 24000) == 8000
    assert output_rate("wav", 24000, 16000) == 16000
    assert media_type_for("pcm", 24000) == "audio/pcm; rate=24000"


def test_decode_errors_and_raw() -> None:
    with pytest.raises(AudioDecodeError, match="empty"):
        decode_audio(b"")
    with pytest.raises(AudioDecodeError):
        decode_audio(b"this is not audio at all" * 10, filename="x.bin")
    assert raw_encoding_for("audio/basic", None) == ("mulaw", 8000)
    assert raw_encoding_for("audio/pcm;rate=8000", None) == ("pcm_s16le", 8000)
    assert raw_encoding_for(None, "call.ulaw") == ("mulaw", 8000)
    assert raw_encoding_for("audio/wav", "a.wav") is None
    x = tone(440, 0.5, 8000)
    y, rate = decode_audio(mulaw_encode(x), content_type="audio/basic")
    assert rate == 8000 and len(y) == len(x)
    y, rate = decode_raw(float_to_pcm16(x).tobytes(), "pcm_s16le", 8000)
    assert snr_db(x, y) > 80


def test_stereo_is_downmixed() -> None:
    import soundfile as sf

    left = tone(440, 0.5, 16000)
    stereo = np.stack([left, np.zeros_like(left)], axis=1)
    buf = io.BytesIO()
    sf.write(buf, stereo, 16000, format="WAV")
    y, rate = decode_audio(buf.getvalue())
    assert y.ndim == 1 and rate == 16000
    assert np.allclose(y, left / 2, atol=1e-3)
