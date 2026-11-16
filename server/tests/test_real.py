"""Real models on CPU. Opt-in: ``BOLO_TEST_REAL=1 pytest -m real``.

These download models (about 600 MB for everything not named ``large``) into
``BOLO_DATA_DIR`` and run them. They assert intelligibility with word error
rate thresholds that leave room for model noise, and print the latency they
measured so a run doubles as a sanity check of the numbers in the README.

Voices that read phonemes are skipped when no phonemiser for their language is installed (uccharan covers English
and Hindi; espeak-ng the rest).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import websockets

from bolo.audio import decode_audio, mulaw_decode, resample
from bolo.config import Settings
from bolo.core import Bolo
from bolo.obs.trace import Trace
from bolo.text.normalize import word_error_rate
from bolo.text.phonemize import get_phonemizer, get_uccharan, select
from conftest import LiveServer

pytestmark = [pytest.mark.real, pytest.mark.timeout(900)]

EN_TEXT = "Good morning. Could you tell me when my order will arrive? Thank you very much for your help."
HI_TEXT = "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।"
LIBRISPEECH = ("csukuangfj/sherpa-onnx-whisper-tiny", "test_wavs/0.wav")
LIBRISPEECH_REF = (
    "after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels"
)

needs_espeak = pytest.mark.skipif(get_phonemizer() is None, reason="needs an espeak-ng installation (not bundled: GPL)")
needs_phonemes_en_hi = pytest.mark.skipif(
    select("piper", "en-us") is None, reason="needs a phonemiser: uccharan (bolo-speech[uccharan]) or espeak-ng"
)
needs_uccharan = pytest.mark.skipif(get_uccharan() is None, reason="needs uccharan (bolo-speech[uccharan])")


def real_settings() -> Settings:
    data_dir = Path(os.environ.get("BOLO_DATA_DIR", Path.home() / ".cache" / "bolo"))
    return Settings(data_dir=data_dir, auto_pull=True, stt_ttl_s=-1, tts_ttl_s=-1)


@pytest.fixture
async def core() -> AsyncIterator[Bolo]:
    c = Bolo(real_settings())
    await c.start()
    try:
        yield c
    finally:
        await c.stop()


@pytest.fixture
def live() -> LiveServer:  # type: ignore[misc]
    srv = LiveServer(real_settings())
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def librispeech(data_dir: Path) -> tuple[np.ndarray, int]:
    """A 6.6 s LibriSpeech utterance (CC BY 4.0) fetched from a model repository's test files."""
    repo, path = LIBRISPEECH
    dest = data_dir / "validation" / repo.replace("/", "--") / Path(path).name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(
            httpx.get(f"https://huggingface.co/{repo}/resolve/main/{path}", follow_redirects=True, timeout=60).content
        )
    return decode_audio(dest.read_bytes(), filename=dest.name)


@pytest.mark.parametrize(
    ("model", "max_wer"),
    [
        ("faster-whisper-tiny", 0.15),
        ("whisper-tiny", 0.15),
        ("moonshine-tiny-en", 0.12),
        ("zipformer-en-20m-streaming", 0.35),
    ],
)
async def test_real_speech_transcription(core: Bolo, model: str, max_wer: float) -> None:
    samples, rate = librispeech(core.settings.data_dir)
    await core.transcribe(samples, rate, model=model, language="en")  # warm-up
    trace = Trace("stt", "bench", model)
    result = await core.transcribe(samples, rate, model=model, language="en", trace=trace)
    wer = word_error_rate(LIBRISPEECH_REF, result.text, "en")
    print(
        f"\n{model}: WER {wer:.3f}, RTF {trace.rtf:.3f}, {trace.total_ms:.0f} ms for {trace.audio_seconds:.1f}s | {result.text}"
    )
    assert wer <= max_wer
    assert trace.rtf is not None and trace.rtf < 1.0  # faster than real time on a few CPU cores


async def test_word_timestamps_real(core: Bolo) -> None:
    samples, rate = librispeech(core.settings.data_dir)
    result = await core.transcribe(samples, rate, model="faster-whisper-tiny", language="en", word_timestamps=True)
    assert len(result.words) >= 15
    starts = [w.t0 for w in result.words]
    assert starts == sorted(starts) and result.words[-1].t1 <= len(samples) / rate + 0.5
    assert result.words[0].w.lower().strip(".,") == "after"


async def test_streaming_model_emits_incremental_partials(live: LiveServer) -> None:
    samples, rate = librispeech(live.core.settings.data_dir)
    pcm = (resample(samples, rate, 16000) * 32767).astype("<i2").tobytes() + b"\x00" * 48000
    partials, finals = [], []
    async with websockets.connect(
        live.ws_url + "/api/v1/stt/stream?model=zipformer-en-20m-streaming", open_timeout=120
    ) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), 300))
        assert ready["mode"] == "native"

        async def send() -> None:
            for off in range(0, len(pcm), 3200):  # 100 ms frames at real-time pace
                await ws.send(pcm[off : off + 3200])
                await asyncio.sleep(0.1)
            await ws.send(json.dumps({"type": "stop"}))

        task = asyncio.create_task(send())
        async for message in ws:
            msg = json.loads(message)
            if msg["type"] == "partial":
                partials.append(msg)
            elif msg["type"] == "final":
                finals.append(msg)
        await task
    text = " ".join(f["text"] for f in finals)
    wer = word_error_rate(LIBRISPEECH_REF, text, "en")
    lat = sorted(p["latency_ms"] for p in partials)
    print(
        f"\nzipformer stream: {len(partials)} partials, median latency {lat[len(lat) // 2]:.0f} ms, final {finals[-1]['latency_ms']:.0f} ms, WER {wer:.3f}"
    )
    assert len(partials) >= 6  # text arrives while the audio is still being sent
    assert partials[0]["t1"] < 3.0
    assert wer <= 0.35
    assert lat[len(lat) // 2] < 1000
    assert finals[0]["words"]


async def test_pseudo_streaming_partials_are_stable(live: LiveServer) -> None:
    samples, rate = librispeech(live.core.settings.data_dir)
    pcm = (resample(samples, rate, 16000) * 32767).astype("<i2").tobytes() + b"\x00" * 48000
    partials, finals, vad = [], [], []
    async with websockets.connect(live.ws_url + "/api/v1/stt/stream?model=moonshine-tiny-en", open_timeout=120) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), 300))
        assert ready["mode"] == "pseudo" and ready["vad"] == "silero-vad"

        async def send() -> None:
            for off in range(0, len(pcm), 3200):
                await ws.send(pcm[off : off + 3200])
                await asyncio.sleep(0.1)
            await ws.send(json.dumps({"type": "stop"}))

        task = asyncio.create_task(send())
        async for message in ws:
            msg = json.loads(message)
            {"partial": partials, "final": finals, "vad": vad}.get(msg["type"], []).append(msg)
        await task
    text = " ".join(f["text"] for f in finals)
    wer = word_error_rate(LIBRISPEECH_REF, text, "en")
    lat = sorted(p["latency_ms"] for p in partials)
    print(
        f"\nmoonshine pseudo-stream: {len(partials)} partials, median latency {lat[len(lat) // 2]:.0f} ms, final {finals[-1]['latency_ms']:.0f} ms, WER {wer:.3f}"
    )
    assert [v["event"] for v in vad][:1] == ["speech_start"] and "speech_end" in [v["event"] for v in vad]
    assert len(partials) >= 4 and wer <= 0.15
    stable = [p["text"][: p["stable_chars"]] for p in partials]
    assert all(b.startswith(a) for a, b in zip(stable, stable[1:]))  # the stable prefix never retracts
    assert stable[-1]


def browser_microphone(clips: list[np.ndarray], *, pause_s: float, speech_dbfs: float) -> list[bytes]:
    """What the console sends for a microphone: 48 kHz capture in the worklet's 20 ms blocks, resampled to 16 kHz in
    the page (an anti-alias FIR, then linear interpolation, as ui/src/lib/audio/resample.ts) and sent as 16-bit PCM.

    The 48 kHz ``clips`` are spoken at ``speech_dbfs`` (browsers' automatic gain control leaves speech around
    -30 dBFS, a quiet voice lower) in a quiet room, with ``pause_s`` of it before, between and after them."""
    room = np.zeros(int(pause_s * 48000))
    parts = [room]
    for clip in clips:
        x = clip.astype(np.float64)
        parts += [x * 10 ** (speech_dbfs / 20) / np.sqrt(np.mean(x[np.abs(x) > 0.01 * np.abs(x).max()] ** 2)), room]
    x = np.concatenate(parts)
    x += np.random.default_rng(7).standard_normal(len(x)) * 10 ** (-62 / 20)
    taps = 63
    cutoff = 0.5 * 16000 * 0.92 / 48000
    k = np.arange(taps) - (taps - 1) / 2
    fir = np.where(k == 0, 2 * cutoff, np.sin(2 * np.pi * cutoff * k) / (np.pi * np.where(k == 0, 1, k)))
    n = np.arange(taps) / (taps - 1)
    fir *= 0.42 - 0.5 * np.cos(2 * np.pi * n) + 0.08 * np.cos(4 * np.pi * n)
    fir /= fir.sum()
    filtered = np.convolve(np.concatenate([np.zeros(taps - 1), x]), fir[::-1], mode="valid")
    # output sample p interpolates input samples 3p - 1 and 3p at fraction 0: it is input sample 3p - 1
    pcm = np.clip(np.concatenate([[0.0], filtered])[0 : len(filtered) : 3], -1, 1)
    pcm = np.where(pcm < 0, np.round(pcm * 32768), np.round(pcm * 32767)).astype("<i2").tobytes()
    return [pcm[off : off + 640] for off in range(0, len(pcm), 640)]  # 320 samples: one 20 ms block


ASK_NOT = Path(__file__).parents[2] / "ui" / "src" / "mock" / "assets" / "en-ask-not.wav"  # J. F. Kennedy, 1961
ASK_NOT_REF = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"


@pytest.mark.parametrize("model", ["zipformer-en-20m-streaming", "moonshine-tiny-en"])
async def test_simulated_browser_microphone(live: LiveServer, model: str) -> None:
    """The console's microphone path, end to end: two utterances at a microphone's level, each after a pause, sent
    in 20 ms frames at real-time pace.

    Found on a Windows laptop: the 20M Zipformer gave no text at all for microphone speech, and dropped the start
    of what it did hear (its own endpointing reset the stream when it had decoded nothing for 2.4 s, mid-word)."""
    clips = []
    for samples, rate in (
        decode_audio(ASK_NOT.read_bytes(), filename=ASK_NOT.name),
        librispeech(live.core.settings.data_dir),
    ):
        clips.append(resample(samples, rate, 48000))
    frames = browser_microphone(clips, pause_s=1.5, speech_dbfs=-30.0)
    url = live.ws_url + f"/api/v1/stt/stream?model={model}&sample_rate=16000&encoding=pcm_s16le&interim=true&vad=true"
    finals, vad = [], []
    async with websockets.connect(url, open_timeout=120) as ws:
        json.loads(await asyncio.wait_for(ws.recv(), 300))

        async def send() -> None:
            for frame in frames:
                await ws.send(frame)
                await asyncio.sleep(0.02)
            await ws.send(json.dumps({"type": "stop"}))

        task = asyncio.create_task(send())
        async for message in ws:
            msg = json.loads(message)
            {"final": finals, "vad": vad}.get(msg["type"], []).append(msg)
        await task
    texts = [f["text"].lower().replace(",", "").replace(".", "") for f in finals]
    wer = word_error_rate(f"{ASK_NOT_REF} {LIBRISPEECH_REF}", " ".join(texts), "en")
    print(f"\n{model} from a simulated microphone: WER {wer:.3f} | " + " / ".join(texts))
    assert vad and vad[0]["event"] == "speech_start" and 1.3 < vad[0]["t"] < 2.0
    assert len(finals) >= 2 and wer <= 0.2
    # each utterance starts with its first word
    assert texts[0].split()[0] == "and" and texts[-1].split()[0] == "after"
    words = " ".join(texts).split()
    assert all(a != b for a, b in zip(words, words[1:]))  # no word repeated at a seam


async def test_kokoro_round_trip(core: Bolo) -> None:
    trace = Trace("tts", "bench", "kokoro-v1.0-int8")
    parts, rate = [], 24000
    try:
        async for chunk in core.synthesize(EN_TEXT, model="kokoro-v1.0-int8", voice="af_heart", trace=trace):
            trace.mark_ttfa()
            parts.append(chunk.samples)
            rate = chunk.sample_rate
    finally:
        core.complete_tts(trace)
    audio = np.concatenate(parts)
    heard = await core.transcribe(audio, rate, model="moonshine-tiny-en")
    wer = word_error_rate(EN_TEXT, heard.text, "en")
    print(f"\nkokoro: TTFA {trace.ttfa_ms:.0f} ms, RTF {trace.rtf:.3f}, round-trip WER {wer:.3f} | {heard.text}")
    assert rate == 24000 and 3.0 < len(audio) / rate < 12.0
    assert wer <= 0.2


@needs_phonemes_en_hi
async def test_piper_voice_round_trip(core: Bolo) -> None:
    audio, rate = await core.synthesize_all(EN_TEXT, model="piper-en-us-amy-low")
    heard = await core.transcribe(audio, rate, model="moonshine-tiny-en")
    wer = word_error_rate(EN_TEXT, heard.text, "en")
    print(f"\npiper amy: round-trip WER {wer:.3f} | {heard.text}")
    assert rate == 16000 and wer <= 0.2


@needs_phonemes_en_hi
async def test_hindi_end_to_end(core: Bolo) -> None:
    """Hindi text -> Hindi voice (Piper format) -> Hindi speech recognition (Whisper fine-tune)."""
    audio, rate = await core.synthesize_all(HI_TEXT, model="piper-hi-in-rohan-medium")
    heard = await core.transcribe(audio, rate, model="faster-whisper-tiny-hindi")
    wer = word_error_rate(HI_TEXT, heard.text, "hi")
    print(f"\nhindi: {len(audio) / rate:.1f}s, round-trip WER {wer:.3f} | {heard.text}")
    assert any("ऀ" <= ch <= "ॿ" for ch in heard.text)  # Devanagari came back
    assert wer <= 0.35


async def test_builtin_hindi_sample_is_recognised(core: Bolo) -> None:
    from bolo import samples

    audio, rate, meta = samples.load_audio("hi-short")
    heard = await core.transcribe(audio, rate, model="faster-whisper-tiny-hindi")
    assert word_error_rate(meta["text"], heard.text, "hi") <= 0.35


async def test_telephony_mulaw_8k_path(live: LiveServer) -> None:
    """Speech out as 8 kHz mu-law, then the same mu-law streamed back in 20 ms frames and transcribed."""
    r = httpx.post(
        live.url + "/v1/audio/speech",
        json={"model": "kokoro-v1.0-int8", "voice": "af_heart", "input": EN_TEXT, "response_format": "mulaw"},
        timeout=600,
    )
    assert r.status_code == 200 and r.headers["x-sample-rate"] == "8000"
    ulaw = r.content
    assert 3.0 < len(ulaw) / 8000 < 12.0 and np.abs(mulaw_decode(ulaw)).max() > 0.1
    # batch upload of raw mu-law
    up = httpx.post(
        live.url + "/v1/audio/transcriptions",
        files={"file": ("call.ulaw", ulaw, "audio/basic")},
        data={"model": "moonshine-tiny-en"},
        timeout=600,
    )
    assert word_error_rate(EN_TEXT, up.json()["text"], "en") <= 0.3
    # realtime: RTP-sized frames
    finals = []
    url = live.ws_url + "/api/v1/stt/stream?model=moonshine-tiny-en&encoding=mulaw&sample_rate=8000"
    async with websockets.connect(url, open_timeout=120) as ws:
        await asyncio.wait_for(ws.recv(), 300)
        payload = ulaw + b"\xff" * 12000

        async def send() -> None:
            for off in range(0, len(payload), 160):
                await ws.send(payload[off : off + 160])
                await asyncio.sleep(0.02)
            await ws.send(json.dumps({"type": "stop"}))

        task = asyncio.create_task(send())
        async for message in ws:
            msg = json.loads(message)
            if msg["type"] == "final":
                finals.append(msg)
        await task
    text = " ".join(f["text"] for f in finals)
    wer = word_error_rate(EN_TEXT, text, "en")
    print(f"\ntelephony mu-law stream: WER {wer:.3f}, final latency {finals[-1]['latency_ms']:.0f} ms | {text}")
    assert wer <= 0.3


async def test_long_audio_is_segmented_with_silero(core: Bolo) -> None:
    samples, rate = librispeech(core.settings.data_dir)
    x = resample(samples, rate, 16000)
    gap = np.zeros(16000 * 2, dtype=np.float32)
    long = np.concatenate([x, gap, x, gap, x, gap, x, gap, x, gap, x])  # 50 s: longer than one 30 s window
    seen = []

    async def on_segment(seg) -> None:  # type: ignore[no-untyped-def]
        seen.append(seg)

    trace = Trace("stt", "bench", "moonshine-tiny-en")
    result = await core.transcribe(long, 16000, model="moonshine-tiny-en", trace=trace, on_segment=on_segment)
    assert 2 <= len(result.segments) <= 8 and len(seen) == len(result.segments)
    assert "vad" in [p.name for p in trace.phases]
    assert result.text.lower().count("nightfall") >= 5
    assert trace.first_partial_ms is not None and trace.first_partial_ms <= (trace.final_ms or 0)


def test_openai_sdk_against_real_models(live: LiveServer) -> None:
    import io

    import openai

    client = openai.OpenAI(base_url=live.url + "/v1", api_key="x", max_retries=0, timeout=600)
    speech = client.audio.speech.create(
        model="kokoro-v1.0-int8", voice="af_heart", input=EN_TEXT, response_format="wav"
    )
    f = io.BytesIO(speech.content)
    f.name = "speech.wav"
    result = client.audio.transcriptions.create(model="moonshine-tiny-en", file=f, response_format="verbose_json")
    assert word_error_rate(EN_TEXT, result.text, "en") <= 0.2
    assert result.duration and 3.0 < result.duration < 12.0


async def test_hindi_whisper_large(core: Bolo) -> None:
    """The multilingual route for Hindi: a general Whisper model told the language (bigger download)."""
    from bolo import samples

    audio, rate, meta = samples.load_audio("hi-short")
    heard = await core.transcribe(audio, rate, model="faster-whisper-small", language="hi")
    wer = word_error_rate(meta["text"], heard.text, "hi")
    print(f"\nfaster-whisper-small on Hindi: WER {wer:.3f} | {heard.text}")
    assert wer <= 0.5


@pytest.mark.parametrize(
    "model", ["kokoro-v1.0-int8", "piper-en-us-amy-low", "kitten-nano-0.8-int8", "piper-hi-in-rohan-medium"]
)
async def test_every_voice_returns_audio_or_a_clear_error(live: LiveServer, model: str) -> None:
    """A text-to-speech model must never answer with silence and no error (WebSocket and HTTP paths)."""
    text = HI_TEXT if "-hi-" in model else "Hello there. This is a short test."
    audio, messages = bytearray(), []
    async with websockets.connect(live.ws_url + f"/api/v1/tts/stream?model={model}", open_timeout=120) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), 600))
        if ready["type"] != "error":
            await ws.send(json.dumps({"type": "text", "text": text, "flush": True}))
            await ws.send(json.dumps({"type": "stop"}))
            async for message in ws:
                if isinstance(message, bytes):
                    audio.extend(message)
                else:
                    messages.append(json.loads(message))
        else:
            messages.append(ready)
    errors = [m for m in messages if m["type"] == "error"]
    r = httpx.post(
        live.url + "/v1/audio/speech", json={"model": model, "input": text, "response_format": "pcm"}, timeout=600
    )
    if select("piper", "en-us") is None and model != "kokoro-v1.0-int8":
        # no phonemiser on this machine: the only acceptable outcome is an explicit error that says so
        assert errors and "uccharan" in errors[0]["message"]
        assert r.status_code >= 400 and "uccharan" in r.json()["error"]["message"]
        return
    assert not errors, errors
    seconds = len(audio) / 2 / ready["sample_rate"]
    assert seconds > 1.0, f"{model} returned {seconds:.2f}s of audio over the WebSocket"
    assert np.abs(np.frombuffer(bytes(audio), dtype="<i2")).max() > 1000
    assert r.status_code == 200 and len(r.content) / 2 / int(r.headers["x-sample-rate"]) > 1.0
    chunks = [m for m in messages if m["type"] == "chunk"]
    assert chunks and chunks[0]["ttfa_ms"] is not None


async def test_band_limited_phone_line_audio_real(live: LiveServer) -> None:
    """Speech filtered to the 300-3400 Hz telephone band, 8 kHz mu-law, 20 ms frames: VAD events and a final."""
    from bolo import samples
    from bolo.audio import mulaw_encode

    audio, rate, meta = samples.load_audio("en-long")
    x = resample(audio, rate, 8000)
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / 8000)
    spectrum[(freqs < 300) | (freqs > 3400)] = 0
    narrow = np.fft.irfft(spectrum, len(x)).astype(np.float32)
    payload = mulaw_encode(np.concatenate([narrow, np.zeros(12000, dtype=np.float32)]))
    for model in ("moonshine-tiny-en", "zipformer-en-20m-streaming"):
        vad, finals = [], []
        url = live.ws_url + f"/api/v1/stt/stream?model={model}&encoding=mulaw&sample_rate=8000"
        async with websockets.connect(url, open_timeout=120) as ws:
            ready = json.loads(await asyncio.wait_for(ws.recv(), 600))
            assert ready["vad"] == "silero-vad"

            async def send(ws=ws) -> None:  # type: ignore[no-untyped-def]
                for off in range(0, len(payload), 160):
                    await ws.send(payload[off : off + 160])
                    await asyncio.sleep(0.02)
                await ws.send(json.dumps({"type": "stop"}))

            task = asyncio.create_task(send())
            async for message in ws:
                msg = json.loads(message)
                {"vad": vad, "final": finals}.get(msg["type"], []).append(msg)
            await task
        text = " ".join(f["text"] for f in finals)
        wer = word_error_rate(meta["text"], text, "en")
        print(f"\n{model} on band-limited mu-law: {len(finals)} finals, WER {wer:.3f} | {text[:80]}")
        assert "speech_start" in [v["event"] for v in vad] and "speech_end" in [v["event"] for v in vad]
        assert finals and wer <= (0.25 if model.startswith("moonshine") else 0.5)


async def test_two_voices_are_told_apart(core: Bolo) -> None:
    """Diarization on a dialogue made of two Kokoro voices, scored against the known turn boundaries."""
    from bolo.diarize import DiarizeOptions, diarization_error_rate
    from bolo.engines.base import SpeakerTurn

    lines = [
        ("af_heart", "Good morning. I would like to ask about my order from last week."),
        ("am_adam", "Of course. Could you give me the order number, please?"),
        ("af_heart", "Yes, it is four seven one nine. It was supposed to arrive on Monday."),
        ("am_adam", "Thank you. I can see that it left the warehouse yesterday evening."),
    ]
    rate = 16000
    parts: list[np.ndarray] = []
    reference: list[SpeakerTurn] = []
    position = 0
    for voice, text in lines:
        audio, native = await core.synthesize_all(text, model="kokoro-v1.0-int8", voice=voice)
        speech = resample(audio, native, rate)
        reference.append(SpeakerTurn(position / rate, (position + len(speech)) / rate, 0 if voice == "af_heart" else 1))
        gap = np.zeros(int(0.6 * rate), dtype=np.float32)
        parts += [speech, gap]
        position += len(speech) + len(gap)
    dialogue = np.concatenate(parts)

    turns = await core.diarize(dialogue, rate, DiarizeOptions())
    score = diarization_error_rate(reference, turns)
    print(f"\ndiarization: {len({t.speaker for t in turns})} speakers, DER {score['der']:.3f} ({score})")
    assert len({t.speaker for t in turns}) == 2
    assert score["confusion"] <= 0.05 and score["der"] <= 0.2

    result = await core.transcribe(dialogue, rate, model="moonshine-tiny-en", diarize=DiarizeOptions(num_speakers=2))
    speakers = [s.speaker for s in result.segments]
    print("speakers per segment:", speakers, "|", result.text)
    assert speakers[0] == "A" and set(speakers) == {"A", "B"}
    assert word_error_rate(" ".join(t for _, t in lines), result.text, "en") <= 0.25


def _dominant_speakers(reference: list[Any], turns: list[Any]) -> list[int | None]:
    """For every reference turn, the speaker the diarizer gave most of its time to (None: nothing heard)."""
    dominant: list[int | None] = []
    for ref in reference:
        heard: dict[int, float] = {}
        for t in turns:
            overlap = min(ref.end, t.end) - max(ref.start, t.start)
            if overlap > 0:
                heard[t.speaker] = heard.get(t.speaker, 0.0) + overlap
        dominant.append(max(heard, key=lambda k: heard[k]) if heard else None)
    return dominant


@pytest.mark.parametrize("rate", [16000, 8000])
async def test_nemotron_tells_two_similar_voices_apart(core: Bolo, rate: int) -> None:
    """Two female Kokoro voices that clustering merges into one speaker, in a conversation where the second
    talks over the first for 2.5 s, wideband and as G.711 mu-law audio.

    This is the kind of recording the measurements in docs/models.md are about (A, B, A's long sentence with B
    starting before it ends, A). Kokoro's output differs a little from one synthesis to the next, so the test
    needs a recording whose result does not depend on that: over 12 syntheses this one gave two speakers, every
    line with its own speaker and a DER of at most 2 % at both rates. A clean four-line dialogue does not
    (the next test)."""
    from bolo.audio import mulaw_encode
    from bolo.diarize import DiarizeOptions, diarization_error_rate
    from bolo.engines.base import SpeakerTurn

    async def say(voice: str, text: str) -> np.ndarray:
        audio, native = await core.synthesize_all(text, model="kokoro-v1.0-int8", voice=voice)
        return resample(audio, native, 16000)

    a1 = await say("af_sky", "Good morning. I would like to ask about my order from last week.")
    b1 = await say("af_bella", "Of course. Could you give me the order number, please?")
    a2 = await say(
        "af_sky",
        "Good morning. I would like to ask about my order from last week. It was supposed to arrive on Monday, "
        "but nothing has come yet, and I have not had any message about it.",
    )
    b2 = await say(
        "af_bella",
        "Of course, I am sorry about the delay. Could you give me the order number, please? I will look it up "
        "straight away and see where the parcel is at the moment.",
    )
    a3 = await say("af_sky", "Yes, it is four seven one nine. It was supposed to arrive on Monday.")
    gap = int(0.6 * 16000)
    start_b2 = len(a1) + gap + len(b1) + gap + len(a2) - int(2.5 * 16000)  # b2 starts 2.5 s before a2 ends
    start_a3 = start_b2 + len(b2) + gap
    placed = [
        (0, a1, 0),
        (len(a1) + gap, b1, 1),
        (len(a1) + gap + len(b1) + gap, a2, 0),
        (start_b2, b2, 1),
        (start_a3, a3, 0),
    ]
    dialogue = np.zeros(start_a3 + len(a3) + gap, dtype=np.float32)
    reference: list[SpeakerTurn] = []
    for start, speech, speaker in placed:
        dialogue[start : start + len(speech)] += speech
        reference.append(SpeakerTurn(start / 16000, (start + len(speech)) / 16000, speaker))
    if rate == 8000:
        dialogue = mulaw_decode(mulaw_encode(resample(dialogue, 16000, 8000)))

    opts = DiarizeOptions(model="nemotron-3-diarization")
    turns = await core.diarize(dialogue, rate, opts)
    score = diarization_error_rate(reference, turns)
    print(f"\nnemotron at {rate} Hz: {len({t.speaker for t in turns})} speakers, DER {score['der']:.3f} ({score})")
    # What the test shows: the two voices come out as two speakers, and every line is given to its own speaker.
    assert len({t.speaker for t in turns}) == 2
    dominant = _dominant_speakers(reference, turns)
    assert None not in dominant, dominant
    by_voice = {voice: {d for d, r in zip(dominant, reference, strict=True) if r.speaker == voice} for voice in (0, 1)}
    assert all(len(v) == 1 for v in by_voice.values()) and by_voice[0] != by_voice[1], (dominant, by_voice)
    # Measured 1.4-1.9 % DER over 12 syntheses at each rate; the bounds leave room for another CPU.
    assert score["confusion"] <= 0.05 and score["der"] <= 0.15
    assert opts.activity is not None and opts.activity.num_speakers == 2
    assert abs(len(opts.activity.probs) - len(dialogue) / rate * 100) <= 2


async def test_nemotron_on_a_clean_dialogue_as_telephone_audio(core: Bolo) -> None:
    """The limit of the test above: four short lines without overlap by the same two voices, as 8 kHz mu-law.

    Here Nemotron's answer depends on small differences between one synthesis of the same text and the next
    (docs/models.md, "Clean two-voice dialogues"): of 40 dialogues put together from five syntheses of each line
    it found two speakers in 38 and gave every line its own speaker in 31, and it missed up to 44 % of the speech
    (median 23 %). On GitHub's CPUs the same dialogue came out as one speaker once, and with 25 % and 41 %
    missed speech. So this test asserts only what held every time: speech is found, there are never more than
    two speakers, and little is reported outside the lines. It does not assert that the voices are told apart."""
    from bolo.audio import mulaw_encode
    from bolo.diarize import DiarizeOptions, diarization_error_rate
    from bolo.engines.base import SpeakerTurn

    lines = [
        ("af_sky", "Good morning. I would like to ask about my order from last week."),
        ("af_bella", "Of course. Could you give me the order number, please?"),
        ("af_sky", "Yes, it is four seven one nine. It was supposed to arrive on Monday."),
        ("af_bella", "Thank you. I can see that it left the warehouse yesterday evening."),
    ]
    parts: list[np.ndarray] = []
    reference: list[SpeakerTurn] = []
    position = 0
    for voice, text in lines:
        audio, native = await core.synthesize_all(text, model="kokoro-v1.0-int8", voice=voice)
        speech = resample(audio, native, 16000)
        reference.append(SpeakerTurn(position / 16000, (position + len(speech)) / 16000, int(voice == "af_bella")))
        parts += [speech, np.zeros(int(0.6 * 16000), dtype=np.float32)]
        position += len(speech) + int(0.6 * 16000)
    dialogue = mulaw_decode(mulaw_encode(resample(np.concatenate(parts), 16000, 8000)))

    opts = DiarizeOptions(model="nemotron-3-diarization")
    turns = await core.diarize(dialogue, 8000, opts)
    score = diarization_error_rate(reference, turns)
    print(f"\nnemotron, clean dialogue at 8000 Hz: {len({t.speaker for t in turns})} speakers ({score})")
    assert 1 <= len({t.speaker for t in turns}) <= 2
    # False alarm: when the two voices are heard as one, the three 0.6 s pauses between the lines are closed
    # (pauses under 1 s inside one speaker's turn are), which alone is 13 % of the speech; the most seen was that.
    assert score["false_alarm"] <= 0.2 and score["missed"] <= 0.6
    assert opts.activity is not None and abs(len(opts.activity.probs) - len(dialogue) / 8000 * 100) <= 2


async def test_overlap_separation_recovers_the_second_voice(core: Bolo) -> None:
    """Two Kokoro voices, the second talking over the first for about 2.5 s: with separate_overlaps the second
    speaker's overlapped words are transcribed from their own track, and the whole transcript gets better."""
    from bolo.diarize import DiarizeOptions
    from bolo.separate import SeparateOptions

    rate = 16000
    a1 = "The library opens at nine and closes late on Thursdays for the reading group."
    b1 = "Please send the updated schedule to everyone before the meeting starts tomorrow."
    a2 = (
        "When the committee finally met in the spring, they agreed to repair the roof first, then repaint the hall, "
        "and leave the garden until the autumn when there would be more volunteers around."
    )
    b2 = (
        "The engineers spent most of the week testing the new pumps, and although two of them failed early, the "
        "rest kept running smoothly through the night without any trouble at all."
    )

    async def say(voice: str, text: str) -> np.ndarray:
        audio, native = await core.synthesize_all(text, model="kokoro-v1.0-int8", voice=voice)
        return resample(audio, native, rate)

    parts = [await say("af_heart", a1), await say("am_michael", b1), await say("af_heart", a2)]
    b_over = await say("am_michael", b2)
    gap = np.zeros(int(0.6 * rate), dtype=np.float32)
    head = np.concatenate([parts[0], gap, parts[1], gap, parts[2]])
    start_b2 = len(head) - int(2.5 * rate)
    mix = np.zeros(start_b2 + len(b_over) + rate // 2, dtype=np.float32)
    mix[: len(head)] += head
    mix[start_b2 : start_b2 + len(b_over)] += b_over

    reference = " ".join([a1, b1, a2, b2])
    plain = await core.transcribe(mix, rate, model="whisper-base", language="en", diarize=DiarizeOptions())
    trace = Trace("stt", "internal", "")
    opts = SeparateOptions()
    separated = await core.transcribe(
        mix, rate, model="whisper-base", language="en", diarize=DiarizeOptions(), separate=opts, trace=trace
    )
    report = separated.separation
    assert report is not None, separated
    print("\nseparation:", report["status"], report["windows"], f"{report['time_ms']:.0f} ms")
    print("plain:", plain.text)
    print("separated:", separated.text)
    assert report["status"] == "applied" and report["windows"][0]["separated"] == ["A", "B"]
    assert any(s.separated and s.speaker == "B" for s in separated.segments)
    b_text = " ".join(s.text for s in separated.segments if s.speaker == "B")
    b_plain = " ".join(s.text for s in plain.segments if s.speaker == "B")
    assert word_error_rate(b1 + " " + b2, b_text, "en") < word_error_rate(b1 + " " + b2, b_plain, "en")
    assert word_error_rate(reference, separated.text, "en") <= 0.2
    assert "separate" in [p.name for p in trace.phases]
    # telephone audio is refused unless asked for
    from bolo.errors import InvalidRequest

    with pytest.raises(InvalidRequest, match="16 kHz"):
        await core.transcribe(
            resample(mix, rate, 8000), 8000, model="whisper-base", diarize=DiarizeOptions(), separate=SeparateOptions()
        )


async def test_end_of_turn_model_tells_finished_from_unfinished(core: Bolo) -> None:
    """Smart Turn v3 (8 MB): the whole English sample is a finished turn, its first half is not."""
    from bolo import samples

    audio, rate, _meta = samples.load_audio("en-short")
    x = resample(audio, rate, 16000)
    complete = await core.end_of_turn(x)
    partial = await core.end_of_turn(x[: int(len(x) * 0.45)])
    print(f"\nend of turn: whole {complete:.3f}, first 45 % {partial:.3f}")
    assert complete is not None and partial is not None
    assert complete > 0.5 > partial


async def test_upstream_piper_voice_is_converted_and_speaks(core: Bolo) -> None:
    """`piper:<voice>`: the Piper project's own files, converted after the download (needs espeak-ng to speak)."""
    from bolo.piper import register_piper_voice

    spec = register_piper_voice(core, "en_US-danny-low")  # not in the catalog
    job = await core.pull_and_wait(spec.id)
    assert job.state == "done", job.error
    directory = core.store.model_dir(spec.id)
    assert (directory / "tokens.txt").is_file() and (directory / "model.onnx").is_file()
    if select("piper", "en-us") is None:
        pytest.skip("speaking needs a phonemiser (uccharan or espeak-ng)")
    audio, rate = await core.synthesize_all("Good morning. Could you tell me when my order will arrive?", model=spec.id)
    heard = await core.transcribe(audio, rate, model="moonshine-tiny-en", language="en")
    print(f"\n{spec.id}: {heard.text}")
    assert word_error_rate("Good morning. Could you tell me when my order will arrive?", heard.text, "en") <= 0.2


@needs_uccharan
@pytest.mark.parametrize("choice", ["uccharan", "uccharan:improved", "espeak-ng"])
async def test_phonemizer_choice_per_request(core: Bolo, choice: str) -> None:
    """The same Piper voice with each phonemiser, chosen per request: all intelligible."""
    if choice == "espeak-ng" and get_phonemizer() is None:
        pytest.skip("espeak-ng is not installed")
    audio, rate = await core.synthesize_all(EN_TEXT, model="piper-en-us-amy-low", phonemizer=choice)
    heard = await core.transcribe(audio, rate, model="moonshine-tiny-en")
    wer = word_error_rate(EN_TEXT, heard.text, "en")
    print(f"\namy with {choice}: round-trip WER {wer:.3f} | {heard.text}")
    assert wer <= 0.2


@needs_uccharan
async def test_kokoro_english_and_hindi_through_uccharan(core: Bolo) -> None:
    for voice, text, stt, lang in (
        ("af_heart", EN_TEXT, "moonshine-tiny-en", "en"),
        ("hf_alpha", HI_TEXT, "faster-whisper-tiny-hindi", "hi"),
    ):
        audio, rate = await core.synthesize_all(text, model="kokoro-v1.0-int8", voice=voice, phonemizer="uccharan")
        heard = await core.transcribe(audio, rate, model=stt, language=lang)
        wer = word_error_rate(text, heard.text, lang)
        print(f"\nkokoro {voice} via uccharan: WER {wer:.3f} | {heard.text}")
        assert wer <= (0.2 if lang == "en" else 0.4)


@needs_uccharan
async def test_kitten_through_uccharan(core: Bolo) -> None:
    audio, rate = await core.synthesize_all(EN_TEXT, model="kitten-nano-0.8-int8", phonemizer="uccharan")
    heard = await core.transcribe(audio, rate, model="moonshine-tiny-en")
    assert word_error_rate(EN_TEXT, heard.text, "en") <= 0.2


@pytest.mark.skipif(
    not os.environ.get("BOLO_DICOW_ONNX_DIR"),
    reason="set BOLO_DICOW_ONNX_DIR to an export of tools/export_dicow_onnx.py (a 1.5 GB model)",
)
async def test_dicow_onnx_transcribes_both_voices_of_an_overlap(core: Bolo) -> None:
    """Torch-free DiCoW on two Kokoro voices that talk at once for 2 s: each speaker's own words, both intelligible."""
    from bolo.custom import register_local_model
    from bolo.diarize import DiarizeOptions

    spec = register_local_model(core, os.environ["BOLO_DICOW_ONNX_DIR"], name="dicow-onnx-test")
    assert spec.kind == "dicow_onnx" and "speaker-conditioned" in spec.features
    core.store.pull(spec.id, spec.source)
    texts = {
        "af_heart": "Please send the updated schedule to everyone before the meeting starts tomorrow morning.",
        "am_michael": "We moved the old printer into the storage room next to the kitchen last week.",
    }
    voices = {}
    for voice, text in texts.items():
        audio, native = await core.synthesize_all(text, model="kokoro-v1.0-int8", voice=voice)
        voices[voice] = resample(audio, native, 16000)
    a, b = voices["af_heart"], voices["am_michael"]
    start_b = len(a) - 2 * 16000  # B starts 2 s before A ends
    mix = np.zeros(start_b + len(b) + 8000, np.float32)
    mix[: len(a)] += a
    mix[start_b : start_b + len(b)] += b
    result = await core.transcribe(mix, 16000, model=spec.id, language="en", diarize=DiarizeOptions())
    by_speaker: dict[str, str] = {}
    for seg in result.segments:
        by_speaker[seg.speaker or "?"] = (by_speaker.get(seg.speaker or "?", "") + " " + seg.text).strip()
    print("\ndicow-onnx:", by_speaker)
    assert len(by_speaker) == 2
    first, second = (by_speaker[k] for k in sorted(by_speaker))
    assert word_error_rate(texts["af_heart"], first, "en") <= 0.25
    assert word_error_rate(texts["am_michael"], second, "en") <= 0.25


def _llama_server_settings() -> Settings:
    s = real_settings()
    s.llama_server_bin = os.environ.get("BOLO_LLAMA_SERVER_BIN") or None
    s.threads = int(os.environ.get("BOLO_THREADS") or 0)  # a shared machine: fewer threads than cores is faster
    return s


async def test_pocket_tts_on_a_managed_llama_server() -> None:
    """tts-server engine, managed mode: bolo starts llama-server (a release build) for Pocket TTS (219 MB)."""
    from bolo.tts_runtimes import find_binary

    settings = _llama_server_settings()
    if find_binary(settings, "llama-server")[0] is None:
        pytest.skip("needs llama-server: bolo components install llama-server, or BOLO_LLAMA_SERVER_BIN")
    core = Bolo(settings)
    await core.start()
    try:
        await core.manager.load("pocket-tts-en-llamacpp")  # pull + start the server; not part of the timing
        results = []
        for voice in ("heart", "michael"):
            trace = Trace("tts", "bench", "pocket-tts-en-llamacpp")
            parts, rate = [], 24000
            try:
                async for chunk in core.synthesize(EN_TEXT, model="pocket-tts-en-llamacpp", voice=voice, trace=trace):
                    trace.mark_ttfa()
                    parts.append(chunk.samples)
                    rate = chunk.sample_rate
            finally:
                core.complete_tts(trace)
            audio = np.concatenate(parts)
            heard = await core.transcribe(audio, rate, model="moonshine-tiny-en")
            wer = word_error_rate(EN_TEXT, heard.text, "en")
            results.append(wer)
            print(
                f"\npocket-tts-en-llamacpp ({voice}): {len(audio) / rate:.2f} s audio, TTFA {trace.ttfa_ms:.0f} ms, "
                f"RTF {trace.rtf:.3f}, round-trip WER {wer:.3f} | {heard.text}"
            )
            assert rate == 24000 and 3.0 < len(audio) / rate < 15.0
        assert min(results) <= 0.3
    finally:
        await core.stop()


async def test_saved_voice_clone_and_speaker_similarity() -> None:
    """A saved voice made from one of bolo's reference clips (Kokoro's michael, not a real person), cloned by Pocket
    TTS on a managed llama-server and scored with the speaker-embedding model: the clone must be closer to its own
    reference than to a different voice, and be understood. Uses pocket-tts-en-llamacpp, pyannote-3.0-eres2net and
    moonshine-tiny-en (pulled when missing)."""
    from bolo import samples, similarity
    from bolo.tts_runtimes import find_binary
    from bolo.voices import clean_reference

    settings = _llama_server_settings()
    if find_binary(settings, "llama-server")[0] is None:
        pytest.skip("needs llama-server: bolo components install llama-server, or BOLO_LLAMA_SERVER_BIN")
    core = Bolo(settings)
    await core.start()
    try:
        audio, rate, _meta = samples.load_audio("en-long")
        if core.voices.find("real-test-michael") is None:
            core.voices.create(
                name="real-test-michael", cleaned=clean_reference([(audio, rate)]), consent="permission",
                created_by={"name": "test_real"}, language="en",
            )  # fmt: skip
        result = await similarity.score_voice(core, "real-test-michael", "pocket-tts-en-llamacpp", EN_TEXT, True)
        other, other_rate, _ = samples.load_audio("en-short")  # Kokoro's heart: another voice
        versus = await similarity.score(core, other, other_rate, result["samples"], result["sample_rate"])
        heard = await core.transcribe(result["samples"], result["sample_rate"], model="moonshine-tiny-en")
        wer = word_error_rate(EN_TEXT, heard.text, "en")
        print(
            f"\nclone similarity {result['similarity']:.3f} vs another voice {versus['similarity']:.3f}, WER {wer:.3f}"
        )
        assert result["similarity"] > versus["similarity"] and wer <= 0.3
        core.voices.delete("real-test-michael")
    finally:
        await core.stop()


async def test_orpheus_family_on_llama_server_with_snac_onnx() -> None:
    """orpheus-llamacpp kind: Svara or Veena on a managed llama-server, SNAC decoded in onnxruntime, Hindi round trip.

    The catalog entries are ``pending`` until bolo's SNAC decoder export is published, so this test needs local
    files: ``BOLO_ORPHEUS_GGUF`` (the GGUF of the entry named by ``BOLO_ORPHEUS_MODEL``, default
    ``svara-tts-v1-llamacpp``; e.g. ``svara-tts-v1.Q4_K_M.gguf`` from mradermacher/svara-tts-v1-GGUF or
    ``Veena-q4_k_m.gguf`` from Mungert/Veena-GGUF for ``veena-llamacpp``) and ``BOLO_SNAC_DECODER``
    (``tools/export_snac_onnx.py`` output, or any SNAC 24 kHz decoder ONNX file). About 15 times slower than real
    time on a 4-core CPU. Q2_K files are not good enough for this check (Veena Q2_K garbles Hindi).
    """
    pytest.importorskip("onnxruntime")
    from bolo.catalog import ModelSpec, packaged_entries
    from bolo.tts_runtimes import find_binary

    gguf, decoder = os.environ.get("BOLO_ORPHEUS_GGUF"), os.environ.get("BOLO_SNAC_DECODER")
    if not gguf or not decoder:
        pytest.skip("needs BOLO_ORPHEUS_GGUF (a Svara or Veena GGUF) and BOLO_SNAC_DECODER (a SNAC decoder ONNX file)")
    settings = _llama_server_settings()
    if find_binary(settings, "llama-server")[0] is None:
        pytest.skip("needs llama-server: bolo components install llama-server, or BOLO_LLAMA_SERVER_BIN")
    base = os.environ.get("BOLO_ORPHEUS_MODEL") or "svara-tts-v1-llamacpp"
    entry = dict(next(e for e in packaged_entries() if e["id"] == base))
    entry.pop("status", None)
    entry["id"] = f"{base}-realtest"
    entry["source"] = {"local": str(Path(gguf).resolve().parent)}
    entry["options"] = {**entry["options"], "model_file": Path(gguf).name, "snac_path": str(Path(decoder).resolve())}
    voice = next(str(v["id"]) for v in entry["voices"] if str(v.get("language", "")).startswith("hi"))
    core = Bolo(settings)
    await core.start()
    try:
        spec = ModelSpec.from_dict(entry)
        core.catalog.add(spec)
        if not core.store.is_installed(spec.id):
            core.store.pull(spec.id, spec.source)
        await core.manager.load(spec.id)  # starts llama-server; not part of the timing
        trace = Trace("tts", "bench", spec.id)
        parts, rate = [], 24000
        try:
            async for chunk in core.synthesize(HI_TEXT, model=spec.id, voice=voice, trace=trace):
                trace.mark_ttfa()
                parts.append(chunk.samples)
                rate = chunk.sample_rate
        finally:
            core.complete_tts(trace)
        audio = np.concatenate(parts)
        heard = await core.transcribe(audio, rate, model="indicconformer-hi-community", language="hi")
        wer = word_error_rate(HI_TEXT, heard.text, "hi")
        print(
            f"\n{base} ({Path(gguf).name}, {voice}): {len(audio) / rate:.2f} s audio, TTFA {trace.ttfa_ms:.0f} ms, "
            f"RTF {trace.rtf:.2f}, round-trip WER {wer:.3f} | {heard.text}"
        )
        assert rate == 24000 and 2.0 < len(audio) / rate < 15.0
        assert wer <= 0.35
    finally:
        await core.stop()


# -- cloud providers (docs/providers.md) ------------------------------------------------------------------------------
# Opt-in twice over: BOLO_TEST_REAL=1 as for everything in this file, and the provider's key in the environment
# (the vendor's usual variable, or BOLO_PROVIDER_<NAME>_API_KEY). These calls are billed by the provider: a few
# seconds of audio and a few sentences of text per test. Where a credential proxy adds the real key to requests
# for the provider's host, any placeholder value in the variable will do.
CLOUD_KEYS = {
    "openai": ("OPENAI_API_KEY", "BOLO_PROVIDER_OPENAI_API_KEY"),
    "sarvam": ("SARVAM_API_KEY", "BOLO_PROVIDER_SARVAM_API_KEY"),
    "elevenlabs": ("ELEVENLABS_API_KEY", "BOLO_PROVIDER_ELEVENLABS_API_KEY"),
    "deepgram": ("DEEPGRAM_API_KEY", "BOLO_PROVIDER_DEEPGRAM_API_KEY"),
}


def cloud_key(provider: str) -> str | None:
    return next((os.environ[name] for name in CLOUD_KEYS[provider] if os.environ.get(name)), None)


def needs_cloud(provider: str) -> pytest.MarkDecorator:
    names = " or ".join(CLOUD_KEYS[provider])
    return pytest.mark.skipif(cloud_key(provider) is None, reason=f"needs a real {provider} key in {names}")


@pytest.fixture
async def cloud_core(tmp_path: Path) -> AsyncIterator[Bolo]:
    providers = {name: {"api_key": key} for name in CLOUD_KEYS if (key := cloud_key(name))}
    c = Bolo(Settings(data_dir=tmp_path / "data", auto_pull=False, providers=providers))
    await c.start()
    try:
        yield c
    finally:
        await c.stop()


async def cloud_round_trip(core: Bolo, tts: str, stt: str, text: str, language: str, voice: str | None = None) -> float:
    """Speak ``text`` with one cloud model, transcribe it with another; returns the word error rate."""
    trace = Trace("tts", "test", tts)
    audio, rate = await core.synthesize_all(text, model=tts, voice=voice, language=language, trace=trace)
    core.complete_tts(trace)
    assert len(audio) > rate, "less than a second of audio came back"
    heard = await core.transcribe(audio, rate, model=stt, language=language)
    wer = word_error_rate(text, heard.text)
    print(f"\n{tts} -> {stt}: WER {wer:.2f}; provider {trace.extra.get('provider')}; heard: {heard.text!r}")
    return wer


@needs_cloud("openai")
async def test_cloud_openai(cloud_core: Bolo) -> None:
    check = await asyncio.to_thread(cloud_core.providers.test, "openai")
    assert check["ok"] and check["checked"] == "credentials", check
    assert (
        await cloud_round_trip(cloud_core, "openai-gpt-4o-mini-tts", "openai-gpt-4o-mini-transcribe", EN_TEXT, "en")
        < 0.2
    )
    assert await cloud_round_trip(cloud_core, "openai-tts-1", "openai-whisper-1", EN_TEXT, "en", "nova") < 0.2
    audio, rate = await cloud_core.synthesize_all("Word timestamps, please.", model="openai-tts-1")
    timed = await cloud_core.transcribe(audio, rate, model="openai-whisper-1", language="en", word_timestamps=True)
    assert len(timed.words) >= 3 and timed.words[-1].t1 <= len(audio) / rate + 0.5


@needs_cloud("sarvam")
async def test_cloud_sarvam(cloud_core: Bolo) -> None:
    check = await asyncio.to_thread(cloud_core.providers.test, "sarvam")
    assert check["ok"], check
    assert await cloud_round_trip(cloud_core, "sarvam-bulbul-v3", "sarvam-saaras-v4", HI_TEXT, "hi") < 0.4
    assert await cloud_round_trip(cloud_core, "sarvam-bulbul-v4-flash", "sarvam-saaras-v3", HI_TEXT, "hi") < 0.4
    audio, rate = await cloud_core.synthesize_all(HI_TEXT, model="sarvam-bulbul-v2", language="hi")
    english = await cloud_core.transcribe(audio, rate, model="sarvam-saaras-v4", task="translate")
    assert any(word in english.text.lower() for word in ("weather", "hello", "how are you"))


@needs_cloud("elevenlabs")
async def test_cloud_elevenlabs(cloud_core: Bolo) -> None:
    check = await asyncio.to_thread(cloud_core.providers.test, "elevenlabs")
    assert check["ok"] and check["checked"] == "credentials", check
    async with cloud_core.manager.use("elevenlabs-flash-v2.5") as entry:
        voices = entry.model.voices  # type: ignore[attr-defined]
    assert voices, "the account lists no voices"
    wer = await cloud_round_trip(
        cloud_core, "elevenlabs-flash-v2.5", "elevenlabs-scribe-v3", EN_TEXT, "en", voices[0].id
    )
    assert wer < 0.2


@needs_cloud("deepgram")
async def test_cloud_deepgram(cloud_core: Bolo) -> None:
    check = await asyncio.to_thread(cloud_core.providers.test, "deepgram")
    assert check["ok"] and check["checked"] == "credentials", check
    samples, rate = librispeech(real_settings().data_dir)
    heard = await cloud_core.transcribe(samples, rate, model="deepgram-nova-3", language="en", word_timestamps=True)
    assert word_error_rate(LIBRISPEECH_REF, heard.text) < 0.15 and len(heard.words) > 10
    # the live socket, at the pace of a microphone
    async with cloud_core.manager.use("deepgram-nova-3") as entry:
        stream = entry.model.create_stream("en")  # type: ignore[attr-defined]
        try:
            audio = resample(samples, rate, 16000)
            for off in range(0, len(audio), 1600):
                stream.accept(audio[off : off + 1600])
                await asyncio.sleep(0.1)
            stream.finish()
            await asyncio.to_thread(stream.settle)
            live_text = stream.result().text
        finally:
            stream.close()
    print(f"\ndeepgram live: {live_text!r}")
    assert word_error_rate(LIBRISPEECH_REF, live_text) < 0.25


# -- LFM3-Audio in ONNX Runtime (in-process speech-to-speech) ---------------------------------------------------------
# A 2.7 GB download and about 3.4 GB of memory: also opt-in by name, BOLO_TEST_S2S_ONNX=1 (needs bolo-speech[s2s]).
needs_s2s_onnx = pytest.mark.skipif(
    os.environ.get("BOLO_TEST_S2S_ONNX") != "1",
    reason="set BOLO_TEST_S2S_ONNX=1 (2.7 GB model, needs bolo-speech[s2s])",
)


@needs_s2s_onnx
@needs_phonemes_en_hi
async def test_lfm3_audio_onnx_speech_to_speech() -> None:
    """A spoken question, the spoken answer, a follow-up that needs the first turn, and a barge-in."""
    import time

    from bolo.s2s import AudioDelta, ResponseEnd, S2SConfig, TextDelta

    settings = real_settings()
    settings.s2s_model = "lfm3-audio-2b-onnx"
    core = Bolo(settings)
    await core.start()
    try:
        backend = await core.s2s.backend()
        assert backend.describe()["backend"] == "onnx" and core.s2s.describe()["available"]
        conversation = await backend.open(S2SConfig(instructions="Answer in one short sentence."))

        async def turn(question: str, stop_after_s: float | None = None) -> tuple[str, np.ndarray, str]:
            asked, rate = await core.synthesize_all(question, model="piper-en-us-amy-low")
            asked = resample(asked, rate, 16000)
            for off in range(0, len(asked), 1600):  # at the pace of a microphone: the audio is prefilled
                await conversation.append_audio(asked[off : off + 1600])
                await asyncio.sleep(0.1)
            await conversation.commit()
            started, first, text, parts, status = time.perf_counter(), None, "", [], "cancelled"
            stream = conversation.respond(modalities=["text", "audio"], max_output_tokens=300)
            async for event in stream:
                if isinstance(event, TextDelta):
                    text += event.text
                elif isinstance(event, AudioDelta):
                    first = first or time.perf_counter() - started
                    parts.append(event.samples)
                    if stop_after_s and sum(map(len, parts)) / event.sample_rate >= stop_after_s:
                        break
                elif isinstance(event, ResponseEnd):
                    status = event.status
            await stream.aclose()
            audio = np.concatenate(parts)
            took = time.perf_counter() - started
            print(
                f"\nlfm3-audio-2b-onnx: {question!r} -> {text!r} [{status}] first audio {first:.2f} s, "
                f"{len(audio) / 24000:.1f} s of speech in {took:.1f} s (RTF {took / (len(audio) / 24000):.2f})"
            )
            return text, audio, status

        text, audio, status = await turn("What is the capital of France?")
        assert status == "completed" and "paris" in text.lower()
        heard = await core.transcribe(audio, 24000, model="moonshine-tiny-en")
        assert word_error_rate(text, heard.text, "en") <= 0.3, heard.text
        text, audio, status = await turn("And what about Italy?")
        assert "rome" in text.lower()
        _text, audio, status = await turn("Tell me a long story about a fox.", stop_after_s=1.0)
        assert status == "cancelled" and len(audio) / 24000 < 4.0
        text, _audio, status = await turn("How many days are there in a week?")
        assert status == "completed" and "seven" in text.lower()
        await conversation.close()
        # the same graphs as a voice
        spoken, rate = await core.synthesize_all("The weather is lovely today.", model="lfm3-audio-2b-onnx")
        heard = await core.transcribe(spoken, rate, model="moonshine-tiny-en")
        print(f"\nlfm3-audio-2b-onnx as a voice: {heard.text!r}")
        assert rate == 24000 and word_error_rate("The weather is lovely today.", heard.text, "en") <= 0.34
    finally:
        await core.stop()


async def test_dicow_onnx_large_transcribes_each_speaker(core: Bolo) -> None:
    """Diarized transcription with the core DiCoW ONNX engine (int8, 1.5 GB) and its Nemotron diarizer on the
    demo's two-voice dialogue, cut to 20.8 s. Every speaker gets text of their own, close to what they said.
    On x86 CPUs without VNNI both speakers came back empty before the engine asked ONNX Runtime for exact int8
    products (``session.x64quantprecision``)."""
    pytest.importorskip("onnxruntime")
    import itertools

    import bolo
    from bolo.diarize import DiarizeOptions

    example = Path(bolo.__file__).parent / "_ui" / "diarization"
    audio, rate = decode_audio((example / "two-speakers-en.mp3").read_bytes(), filename="two-speakers-en.mp3")
    audio = audio[: int(20.8 * rate)]
    turns = json.loads((example / "two-speakers-en.json").read_text(encoding="utf-8"))["turns"]
    said: dict[str, list[str]] = {}
    for turn in turns:
        if turn["end"] <= 20.8:
            said.setdefault(turn["speaker"], []).append(turn["text"])
    trace = Trace("stt", "test", "dicow-v3.3-onnx")
    result = await core.transcribe(
        audio, rate, model="dicow-v3.3-onnx", language="en", diarize=DiarizeOptions(), trace=trace
    )
    heard: dict[str | None, list[str]] = {}
    for seg in result.segments:
        heard.setdefault(seg.speaker, []).append(seg.text)
    print("\n" + "\n".join(f"{k}: {' '.join(v)}" for k, v in heard.items()), trace.extra)
    assert trace.extra.get("conditioning") == "turns"  # the conditioned path, not cut-at-turns decoding
    assert len(heard) == 2 and all(" ".join(v).strip() for v in heard.values())
    refs = [" ".join(v) for v in said.values()]
    hyps = [" ".join(v) for v in heard.values()]
    # each speaker's words against the reference speaker they match best (cpWER's assignment)
    best = min(itertools.permutations(hyps), key=lambda p: sum(word_error_rate(r, h, "en") for r, h in zip(refs, p)))
    rates = [word_error_rate(r, h, "en") for r, h in zip(refs, best)]
    print("per-speaker WER", [round(x, 3) for x in rates])
    # Nemotron hears the first sentence for both voices, so one stream may carry it twice: a loose bound
    assert all(x <= 0.5 for x in rates) and sum(rates) / len(rates) <= 0.35
