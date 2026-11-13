"""Real models on CPU. Opt-in: ``BOLO_TEST_REAL=1 pytest -m real``.

These download models (about 600 MB for everything not named ``large``) into
``BOLO_DATA_DIR`` and run them. They assert intelligibility with word error
rate thresholds that leave room for model noise, and print the latency they
measured so a run doubles as a sanity check of the numbers in the README.

Voices that need espeak-ng phonemes are skipped when no espeak-ng is installed.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import numpy as np
import pytest
import websockets

from bolo.audio import decode_audio, mulaw_decode, resample
from bolo.config import Settings
from bolo.core import Bolo
from bolo.obs.trace import Trace
from bolo.text.normalize import word_error_rate
from bolo.text.phonemize import get_phonemizer
from conftest import LiveServer

pytestmark = [pytest.mark.real, pytest.mark.timeout(900)]

EN_TEXT = "Good morning. Could you tell me when my order will arrive? Thank you very much for your help."
HI_TEXT = "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।"
LIBRISPEECH = ("csukuangfj/sherpa-onnx-whisper-tiny", "test_wavs/0.wav")
LIBRISPEECH_REF = (
    "after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels"
)

needs_espeak = pytest.mark.skipif(get_phonemizer() is None, reason="needs an espeak-ng installation (not bundled: GPL)")


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


@needs_espeak
async def test_piper_voice_round_trip(core: Bolo) -> None:
    audio, rate = await core.synthesize_all(EN_TEXT, model="piper-en-us-amy-low")
    heard = await core.transcribe(audio, rate, model="moonshine-tiny-en")
    wer = word_error_rate(EN_TEXT, heard.text, "en")
    print(f"\npiper amy: round-trip WER {wer:.3f} | {heard.text}")
    assert rate == 16000 and wer <= 0.2


@needs_espeak
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
    if get_phonemizer() is None and model != "kokoro-v1.0-int8":
        # no espeak-ng on this machine: the only acceptable outcome is an explicit error that says so
        assert errors and "espeak-ng" in errors[0]["message"]
        assert r.status_code >= 400 and "espeak-ng" in r.json()["error"]["message"]
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
    if get_phonemizer() is None:
        pytest.skip("speaking needs espeak-ng")
    audio, rate = await core.synthesize_all("Good morning. Could you tell me when my order will arrive?", model=spec.id)
    heard = await core.transcribe(audio, rate, model="moonshine-tiny-en", language="en")
    print(f"\n{spec.id}: {heard.text}")
    assert word_error_rate("Good morning. Could you tell me when my order will arrive?", heard.text, "en") <= 0.2
