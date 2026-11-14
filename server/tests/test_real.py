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
    select("piper", "en-us") is None, reason="needs a phonemiser: uccharan (bolo[uccharan]) or espeak-ng"
)
needs_uccharan = pytest.mark.skipif(get_uccharan() is None, reason="needs uccharan (bolo[uccharan])")


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
    (median 23 %; false alarm at most 9 %). On GitHub's CPUs the same dialogue came out as one speaker once and
    with 25 % and 41 % missed speech. So this test asserts only what held every time: speech is found, there are never more than
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
    assert score["false_alarm"] <= 0.15 and score["missed"] <= 0.6
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
    reason="set BOLO_DICOW_ONNX_DIR to an export of tools/export_dicow_onnx.py (dicow-v3.3-onnx is not hosted yet)",
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
