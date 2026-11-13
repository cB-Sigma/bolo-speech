<p align="center">
  <a href="https://cb-sigma.github.io/bolo/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/banner-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="docs/assets/brand/banner-light.png">
      <img src="docs/assets/brand/banner-dark.png" alt="bolo: Ollama for speech. Pull a model, serve it over the APIs you already use, and watch every millisecond." width="100%">
    </picture>
  </a>
</p>

<p align="center">
  <a href="https://github.com/cb-Sigma/bolo/actions/workflows/ci.yml"><img src="https://github.com/cb-Sigma/bolo/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-1f62b3" alt="License: Apache-2.0"></a>
  <a href="https://github.com/cb-Sigma/bolo/actions/workflows/pages.yml"><img src="https://github.com/cb-Sigma/bolo/actions/workflows/pages.yml/badge.svg" alt="Live demo build status"></a>
  <a href="https://cb-sigma.github.io/bolo/"><img src="https://img.shields.io/badge/live%20demo-runs%20in%20your%20browser-1f62b3" alt="Live demo: runs in your browser"></a>
  <img src="https://img.shields.io/badge/API-OpenAI%20compatible-1f62b3" alt="OpenAI-compatible API">
  <img src="https://img.shields.io/badge/torch-free-1f62b3" alt="No PyTorch needed">
</p>

**Ollama for speech: a small, PyTorch-free server that pulls speech models by name and serves speech-to-text, text-to-speech and voice-activity detection over the APIs your clients already use.**

*bolo* (बोलो) is Hindi for "speak".

<p align="center">
  <a href="https://cb-sigma.github.io/bolo/"><img src="docs/assets/brand/button-live-demo.png" alt="Try the live demo" width="233"></a>
</p>

<p align="center">
  <sub>The demo runs speech recognition entirely in your browser with WebAssembly: no install, no upload, works offline after the first visit.</sub>
</p>

<p align="center">
  <img src="docs/assets/console-playground.gif" alt="Animated capture of the speech-to-text playground: words appear as audio streams in, then the latency timeline shows how long the final transcript took after speech ended" width="900">
</p>

<p align="center">
  <sub>The speech-to-text playground, live against a real server on a 4-core CPU. Lighter text is still changing; solid text is final.</sub>
</p>


- **Drop-in.** OpenAI-compatible `/v1/audio/transcriptions`, `/v1/audio/speech` and `/v1/models`; the model-management routes of [speaches](https://github.com/speaches-ai/speaches); the Wyoming protocol for Home Assistant.
- **Light.** No PyTorch anywhere. Inference runs on ONNX Runtime (through [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)) and CTranslate2 (through [faster-whisper](https://github.com/SYSTRAN/faster-whisper)). The base install has no engine at all; you add the ones you want.
- **Realtime on CPU.** A low-latency tier, true streaming models, and stabilised partial results for models that were never designed to stream. Measured numbers below, from a 4-vCPU machine.
- **A real server.** Per-model queues with micro-batching, backpressure, cancellation, lazy loading, idle unloading and a memory budget, with every request traced phase by phase.
- **Indian languages and phone audio.** Hindi, Bengali, Urdu, Nepali and Gujarati voices; Hindi and multilingual recognition; 8 kHz G.711 μ-law/A-law in and out.
- **Voice agents.** An OpenAI Realtime API endpoint (`/v1/realtime`) and audio in `/v1/chat/completions`: speech in, your own language model in the middle (Ollama, llama.cpp, vLLM, any OpenAI-compatible endpoint), streamed speech out, with barge-in and per-turn latency traces.
- **Who said what.** Speaker diarization without PyTorch, as `diarized_json`, in subtitles, and as speaker labels on live transcripts.
- **Honest.** 180 models in the catalog, each with its licence; the 71 that were actually run on CPU are marked, with what was measured. Everything else says *untested*.

> **Status: 0.1, alpha.** Developed and measured on Linux x86-64, CPU only. GPUs, Apple Silicon, arm64 and Windows are untested; see [What is not tested](#what-is-not-tested).

## The console

<table>
  <tr>
    <td width="50%" valign="top">
      <a href="docs/assets/console-playground.png"><img src="docs/assets/console-playground.png" alt="Speech-to-text playground with transcript, level strip and latency timeline"></a>
      <br><b>Playground.</b> Talk, upload a file or play a sample; see partial and final text, and where the wait after you stop speaking went.
    </td>
    <td width="50%" valign="top">
      <a href="docs/assets/console-latency.png"><img src="docs/assets/console-latency.png" alt="Latency dashboard with percentile charts and a request waterfall"></a>
      <br><b>Latency.</b> Live percentiles against your target, the distribution, and a waterfall of every request with its phases.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="docs/assets/console-compare.png"><img src="docs/assets/console-compare.png" alt="Compare mode showing the server and the in-browser model side by side"></a>
      <br><b>Browser vs server.</b> The same audio through the server and through a model running in the page, with both timelines.
    </td>
    <td width="50%" valign="top">
      <a href="docs/assets/console-tts.png"><img src="docs/assets/console-tts.png" alt="Text-to-speech playground with time to first audio and a chunk timeline"></a>
      <br><b>Text to speech.</b> Streamed playback, time to first audio, a per-chunk timeline, underrun detection, and a round trip back through recognition.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="docs/assets/console-models.png"><img src="docs/assets/console-models.png" alt="Model catalog with tiers, licences and load state"></a>
      <br><b>Models.</b> Search and filter the catalog, pull with progress, load and unload, and see what was actually validated on CPU.
    </td>
    <td width="50%" valign="top">
      <a href="docs/assets/console-benchmark.png"><img src="docs/assets/console-benchmark.png" alt="Benchmark showing latency and throughput against concurrency"></a>
      <br><b>Benchmark.</b> Sweep concurrency, chart latency and throughput, and find the ceiling for your latency target.
    </td>
  </tr>
</table>

<p align="center">
  <img src="docs/assets/console-timeline.png" alt="Close-up of the latency timeline: speech span, first partial, and 658 ms from end of speech to the final transcript, split into endpointing, model and network" width="420">
  &nbsp;&nbsp;&nbsp;
  <img src="docs/assets/console-phone.png" alt="The playground on a phone-width screen" width="220">
</p>

<p align="center">
  <sub>Left: the latency timeline for one utterance. Right: the same screen at phone width. A light theme is included (<a href="docs/assets/console-light.png">screenshot</a>), as is the browser-only <a href="docs/assets/console-demo.png">live demo</a>. Indian languages are listed under their own names (<a href="docs/assets/console-hindi.png">Hindi on the server</a>).</sub>
</p>

<p align="center">
  <sub>All screenshots are of the real console, taken on a 4-core CPU with no GPU. See <a href="docs/assets/README.md">docs/assets/README.md</a> for what each one was taken against.</sub>
</p>

## Quick start

```bash
pipx install 'bolo-speech[all] @ git+https://github.com/cb-Sigma/bolo#subdirectory=server'
bolo serve
```

(`pip install` into a virtual environment works the same way. Once the package is on PyPI this becomes `pipx install 'bolo-speech[all]'`, or `pipx run --spec 'bolo-speech[all]' bolo serve`.)

Open <http://127.0.0.1:8000> for the console, or use the API. The first request for a model downloads it:

```bash
# speech to text (pulls whisper-base, about 160 MB, the first time)
curl http://127.0.0.1:8000/v1/audio/transcriptions -F file=@recording.wav -F model=whisper-1

# text to speech (pulls kokoro-v1.0-int8, about 170 MB, the first time)
curl http://127.0.0.1:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "tts-1", "voice": "af_heart", "input": "Hello from bolo."}' -o hello.mp3
```

Or choose models yourself, the Ollama way:

```bash
bolo list --all -q english        # browse the catalog
bolo pull moonshine-tiny-en       # 124 MB, the fastest English recogniser here
bolo transcribe recording.wav -m moonshine-tiny-en
bolo speak "Hello there." -o hello.wav -m kokoro-v1.0-int8 --voice af_heart
bolo ps                           # what a running server has in memory
bolo doctor                       # engines, codecs and phonemiser found on this machine
```

With the official OpenAI SDK, only the base URL changes:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
text = client.audio.transcriptions.create(model="whisper-1", file=open("recording.wav", "rb")).text
client.audio.speech.create(model="tts-1", voice="alloy", input=text).write_to_file("echo.mp3")
```

Installation extras: `sherpa` (most models), `faster-whisper`, `av` (M4A/WebM/AAC through bundled FFmpeg), `wyoming`, `all` (those four), and `espeak` (see below; never part of `all`).

**eSpeak NG is optional and not shipped with bolo.** Voices in Piper format, KittenTTS and the non-English Kokoro voices need the eSpeak NG phonemiser, which is GPL-3.0-or-later. The default voice (Kokoro English) does not. If you want those voices, install it when you choose to: one click in the console (Models → Components), `bolo components install espeak-ng --accept-license`, `pip install 'bolo-speech[espeak]'`, or your system's package (`apt install espeak-ng`). It takes effect without a restart. Until then those voices answer with a clear `component_missing` error. Details: [Licences](#licences).

## What it does

**Protocols**

| | |
| --- | --- |
| OpenAI HTTP | transcriptions (json, text, srt, vtt, verbose_json with word timestamps, SSE streaming), translations, speech (mp3, opus, aac, flac, wav, pcm, chunked as it is synthesised), models |
| speaches routes | `/v1/registry`, `POST`/`DELETE /v1/models/{id}`, `/v1/audio/voices`, `/api/ps`, `/v1/audio/speech/timestamps` and its environment variables |
| OpenAI Realtime API | `WS /v1/realtime`: transcription sessions, and conversation sessions (speech → your LLM → speech) with server VAD, barge-in, text turns and function calling. GA event names, tested with the official SDK's realtime client |
| Chat completions with audio | `POST /v1/chat/completions`: `input_audio` parts are transcribed, the request goes to your LLM, the reply is spoken when `modalities` includes `audio` (streaming too). Everything else is proxied unchanged |
| Speaker diarization | `diarize=true` or `response_format=diarized_json` on transcriptions, `POST /api/v1/diarize`, `diarize=true` on the realtime WebSocket |
| Realtime WebSockets | `WS /api/v1/stt/stream` (partials, finals with word times, VAD events, optional speaker labels) and `WS /api/v1/tts/stream` (text in pieces, audio out) |
| Wyoming | speech-to-text and text-to-speech on one TCP port, including streaming synthesis |
| Native JSON API | catalog, pulls with progress, load/unload, optional components, live statistics, request traces, Server-Sent Events, benchmark jobs |
| Operations | Prometheus `/metrics`, `/healthz`, `/readyz`, optional bearer key, CORS |

**Server behaviour**

- Models load on first use, unload after five idle minutes, and are evicted least-recently-used first under a memory budget. A newly installed model is first loaded in a child process, so a broken file cannot crash the server.
- Each model has a bounded queue. Requests that arrive while the engine is busy are decoded together in one batched call; when the queue is full the answer is `429` with `Retry-After`, not a growing delay. Work for clients that hung up is dropped.
- Long recordings are split at pauses (Silero VAD) and the pieces decoded as a batch; with `stream=true` you get text as each piece finishes.
- Downloads resume after interruptions and are verified (SHA-256 for LFS files) into a content-addressed cache.

**The console** (served at `/`): a playground for both directions with a latency timeline, the model catalog, a live latency dashboard with a per-request waterfall, and a benchmark page that sweeps concurrency.

## Supported models

<!-- BEGIN GENERATED: models-summary -->
| Task | Family | Models | Languages | Validated on CPU | Best measured RTF |
| --- | --- | --- | --- | --- | --- |
| DIARIZATION | pyannote + speaker embeddings | 3 | - | 1 | 0.106 (`pyannote-3.0-eres2net`) |
| STT | Whisper | 31 | af, am, ar, as, az, ba, be, bg +92 | 14 | 0.081 (`faster-whisper-tiny.en`) |
| STT | Moonshine | 11 | ar, en, es, ja, ko, uk, vi, zh | 4 | 0.033 (`moonshine-tiny-en`) |
| STT | Parakeet | 7 | bg, cs, da, de, el, en, es, et +17 | 4 | 0.030 (`parakeet-tdt-ctc-110m`) |
| STT | NeMo FastConformer | 8 | be, de, en, es, fr, hr, it, pl +2 | 3 | 0.334 (`nemo-fastconformer-ctc-en-streaming-480ms`) |
| STT | Canary | 1 | de, en, es, fr | 1 | 0.358 (`canary-180m-flash`) |
| STT | Zipformer | 16 | ar, bn, de, en, es, fr, id, ja +5 | 7 | 0.028 (`zipformer-en-gigaspeech`) |
| STT | SenseVoice | 1 | en, ja, ko, yue, zh | 1 | 0.039 (`sense-voice-small`) |
| STT | Paraformer | 2 | en, zh | 1 | 0.091 (`paraformer-zh`) |
| STT | Dolphin | 2 | ar, bn, fa, gu, hi, id, ja, jv +26 | 1 | 0.021 (`dolphin-base`) |
| STT | Omnilingual ASR | 1 | ar, as, bn, de, en, es, fr, gu +23 | 1 | 0.112 (`omnilingual-ctc-300m`) |
| STT | IndicConformer | 1 | as, bn, brx, gu, hi, kn, ks, mr | 1 | 0.131 (`indicconformer-multi-community`) |
| TTS | Piper | 74 | ar, ca, cs, cy, da, de, el, en +30 | 17 | 0.040 (`piper-ne-np-google-medium`) |
| TTS | Kokoro | 2 | en, es, fr, hi, it, ja, pt, zh | 2 | 0.759 (`kokoro-v1.0`) |
| TTS | KittenTTS | 4 | en | 4 | 0.107 (`kitten-nano-0.8`) |
| TTS | Matcha-TTS | 2 | en, zh | 1 | 0.205 (`matcha-en-ljspeech`) |
| TTS | MeloTTS | 1 | en | 1 | 0.278 (`melo-tts-en`) |
| TTS | MMS-TTS | 7 | de, en, es, fr, ru, th, uk | 1 | 0.196 (`mms-tts-eng`) |
| TTS | VITS | 4 | bn, gu, ne | 4 | 0.046 (`mimic3-gu-cmu-indic`) |
| VAD | Silero VAD | 2 | - | 2 | 0.006 (`silero-vad-v5`) |

**180 models in the catalog, 71 validated on CPU, 109 untested.** Full list: [docs/models.md](docs/models.md).
<!-- END GENERATED: models-summary -->

"Validated on CPU" means downloaded and run on the machine described under [Performance](#performance) by `server/tools/validate_models.py`; the table in [docs/models.md](docs/models.md) has every model with its measured real-time factor and licence. Anything else that sherpa-onnx or faster-whisper can load can be added with `bolo pull hf:<owner>/<repo>`.

Not supported: models that need PyTorch at run time (XTTS, F5-TTS, Chatterbox, Orpheus and other LLM-based voices), and voice cloning. bolo runs no language models itself: voice agents use the one you connect.

## The realtime tier

The default realtime model is `moonshine-tiny-en` (27M parameters, MIT); `zipformer-en-20m-streaming` is the true-streaming alternative. One live stream, audio sent at real-time pace through the same code path as the WebSocket, on the 4-vCPU machine described under [Performance](#performance):

| Model | How it streams | First partial (p50 / p95) | Final after end of speech is detected (p50 / p95) | Notes |
| --- | --- | --- | --- | --- |
| `zipformer-en-20m-streaming` | native, incremental | 19 / 21 ms | 22 / 30 ms | lower-case, no punctuation; least accurate of the four, and it can drop the first words of audio that starts abruptly |
| `moonshine-tiny-en` | re-decode + LocalAgreement | 62 / 69 ms | 420 / 465 ms on a 12 s utterance; 161 / 196 ms on a 3 s one | punctuated; the default |
| `whisper-tiny` (sherpa-onnx) | re-decode + LocalAgreement | 95 / 111 ms | 918 / 978 ms | 99 languages |
| `faster-whisper-tiny` | re-decode + LocalAgreement | 570 / 640 ms | 860 / 885 ms | 99 languages, word timestamps |

These are `latency_ms` values: server time from receiving the last audio that contributed to the text until the message is sent. They do **not** include the pause the endpointer waits for before it calls an utterance finished, which is 500 ms by default and usually the larger part of what a user perceives. Measured from the true end of speech to the final text with `moonshine-tiny-en` on a 3-second utterance: **355 ms** with `endpointing=fast` (250 ms), **615 ms** with the default (500 ms), **1,212 ms** with `patient` (900 ms).

Every other speech-to-text model also works on the realtime WebSocket; it is just slower. Word error rates on a LibriSpeech test clip and the full table are in [docs/models.md](docs/models.md).

How it works, and how to tune it: [docs/realtime.md](docs/realtime.md).

## Indian languages and telephony

**Recognition.** Hindi fine-tunes of Whisper (`faster-whisper-tiny-hindi`, 79 MB, and `-small-hindi`), multilingual Whisper with `language=hi` (or bn, ta, te, mr, gu, kn, ml, pa, ur, …), Meta's Omnilingual CTC model (1,600+ languages), Dolphin (40 Eastern languages), a streaming Bengali Zipformer, and a community export of AI4Bharat's IndicConformer. The official IndicConformer repositories are gated and not loadable yet; bolo never downloads gated models without your token.

**Synthesis.** Hindi (three Piper-format voices and four Kokoro voices), Nepali, Urdu, Bengali and Gujarati voices. Sentences are split at the danda, Hindi numbers are spelled out in the Indian system (`250000` → दो लाख पचास हज़ार), and code-mixed text passes through untouched. No Tamil, Telugu, Marathi, Kannada, Malayalam or Punjabi voice yet.

**Checked end to end:** Hindi text → `piper-hi-in-rohan-medium` → `faster-whisper-tiny-hindi` returns the sentence with 0% word error rate in the test-suite; the community IndicConformer export and `faster-whisper-small-hindi` also transcribed synthesised Hindi without errors. That is one sentence of clean synthetic speech, not an accuracy benchmark. Voices for Nepali, Urdu, Bengali and Gujarati produced audio that nobody who speaks those languages has reviewed.

**Telephony.** 8 kHz G.711 is native in both directions: `encoding=mulaw|alaw` on the realtime WebSocket, raw `audio/basic` uploads, and `"response_format": "mulaw"` for speech, with helpers for 20 ms RTP-sized frames. The codecs match the reference implementation bit for bit. In the test-suite, text → Kokoro → 8 kHz μ-law → 160-byte frames at real-time pace → `moonshine-tiny-en` came back with 0% word error rate, and the same speech filtered to the 300 to 3,400 Hz telephone band with 17% (the first words were clipped).

Details, including what was *not* tested (no real phone network): [docs/indic-telephony.md](docs/indic-telephony.md).

## API compatibility

**OpenAI.** The request and response shapes follow the OpenAI API reference as typed in the official `openai` Python SDK (3.50.0), and are exercised with that SDK in the test-suite:

- *Audio*: plain and verbose transcription, word and segment timestamps, SSE transcription events, `diarized_json` (the `gpt-4o-transcribe-diarize` shape), translation, speech in every format, streamed speech, `stream_format: "sse"`, the error envelope. `whisper-1`, `gpt-4o-transcribe`, `tts-1` and friends map to your default models, and OpenAI voice names map onto the selected model's voices.
- *Realtime* (`/v1/realtime`, GA event names such as `response.output_audio.delta`): `session.update`, `input_audio_buffer.*` with server VAD or manual commit, `conversation.item.*`, input transcription events, `response.create` / `response.cancel` with audio, transcript and text deltas, function-call events, truncation, errors; 24 kHz PCM16 and G.711 μ-law/A-law. Tested with the SDK's realtime client and a raw WebSocket.
- *Chat completions with audio*: `input_audio` content parts and `modalities: ["text", "audio"]`, non-streaming and streaming.

bolo is a cascade (speech-to-text → your language model → text-to-speech), not a speech-to-speech model. Not implemented: WebRTC and SIP transports, ephemeral client secrets, MCP tools, image input, semantic VAD (accepted, handled as server VAD), known-speaker references for diarization. See [docs/voice-agents.md](docs/voice-agents.md).

**speaches.** Point existing speaches clients at bolo and keep their model ids (`Systran/faster-whisper-small`, `speaches-ai/Kokoro-82M-v1.0-ONNX`, …) and environment variables (including `CHAT_COMPLETION_BASE_URL`). What matches, what differs and what is missing (WebRTC realtime, speaker embeddings) is spelled out in [docs/migration-from-speaches.md](docs/migration-from-speaches.md).

**Wyoming / Home Assistant.** `bolo serve --wyoming` exposes installed models on port 10300. Tested with the `wyoming` Python client; **not** tested against a Home Assistant instance. Setup notes: [examples/home_assistant.md](examples/home_assistant.md).

Full reference: [docs/api.md](docs/api.md) and [docs/openapi.json](docs/openapi.json). Client examples: [examples/](examples/).

## Architecture

```
   OpenAI HTTP · speaches routes · native /api/v1 · WebSockets · Wyoming TCP · CLI
                                      │
        ┌─────────────────────────────┴──────────────────────────────┐
        │ core: transcribe · synthesize · realtime sessions · pulls   │
        ├──────────────┬───────────────┬───────────────┬──────────────┤
        │ audio I/O    │ text          │ observability │ model store  │
        │ G.711, soxr, │ sentences,    │ traces, p50/  │ resumable,   │
        │ codecs       │ normalisation │ p95, events   │ verified     │
        ├──────────────┴───────────────┴───────────────┴──────────────┤
        │ lifecycle: lazy load · probe · TTL · memory budget · LRU     │
        ├──────────────────────────────────────────────────────────────┤
        │ scheduler: one bounded queue per model · micro-batching ·    │
        │            worker threads · backpressure · cancellation      │
        ├───────────────────────┬──────────────────┬───────────────────┤
        │ sherpa-onnx           │ faster-whisper   │ optional plugins  │
        │ (ONNX Runtime)        │ (CTranslate2)    │                   │
        └───────────────────────┴──────────────────┴───────────────────┘
```

Engines are plugins behind four small interfaces (batch recognition, streaming recognition, synthesis, VAD), imported only when a model needs them. More in [docs/architecture.md](docs/architecture.md).

## Performance

All numbers were measured with bolo's own load generator (`bolo bench`, in process) on **4 vCPUs of an Intel Xeon at 2.1 GHz, 15 GB RAM, no GPU, Linux x86-64, Python 3.12, sherpa-onnx 2.0.1, faster-whisper 1.2.1, 4 inference threads**. The machine is a shared cloud VM that was running other workloads at the same time (load average 2 to 4 before bolo started), so expect better on an idle machine of the same size. Raw results: [docs/measurements/](docs/measurements/).

**Single request** (best of three after warm-up; RTF = processing time / audio duration):

| Model | RTF | Meaning |
| --- | --- | --- |
| `moonshine-tiny-en` | 0.033 | 30× faster than real time |
| `parakeet-tdt-ctc-110m` | 0.030 | 33× |
| `sense-voice-small` | 0.039 | 26× |
| `parakeet-tdt-0.6b-v2` | 0.081 | 12× |
| `faster-whisper-tiny` / `whisper-base` / `whisper-small` | 0.09 / 0.11 / 0.36 | 11× / 9× / 2.7× |
| `whisper-large-v3-turbo` (int8) | 0.30 | 3.3× |
| `piper-en-us-amy-low` (TTS) | 0.033 to 0.049, first audio after 140 to 190 ms | 20 to 30× |
| `kitten-nano-0.8-int8` (TTS) | 0.27, first audio after about 1.0 s | 3.7× |
| `kokoro-v1.0-int8` (TTS) | 0.88, first audio after 2.4 to 3.4 s | barely faster than real time |

Kokoro is the default voice because it sounds best and needs no extra install, but on four shared cores it is only just real-time; the float32 export (`kokoro-v1.0`) measured slightly faster (RTF 0.76). For low latency on CPU use a Piper-format voice. Two streaming models were slower than real time here and are not usable live on this hardware: `parakeet-unified-en-streaming-560ms` (RTF 4.7) and, nearly, `nemo-fastconformer-ctc-en-streaming-80ms` (0.76).

**Concurrent realtime streams** (each client streams at 1×; p50 / p95 in ms):

| Streams | `zipformer-en-20m-streaming` first partial | final | same, batching off: first partial | final | `moonshine-tiny-en` first partial | final | same, batching off: first partial | final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 21 / 22 | 32 / 35 | 22 / 24 | 34 / 64 | 66 / 96 | 161 / 196 | 56 / 65 | 150 / 161 |
| 2 | 28 / 31 | 40 / 44 | 25 / 52 | 48 / 109 | 104 / 182 | 314 / 358 | 83 / 134 | 224 / 338 |
| 4 | 37 / 55 | 54 / 114 | 34 / 144 | 115 / 176 | 262 / 329 | 652 / 792 | 81 / 267 | 684 / 795 |
| 8 | 75 / 86 | 172 / 274 | 448 / 525 | 874 / 2,837 | 514 / 621 | 1,338 / 1,977 | 456 / 924 | 1,719 / 2,386 |

(The Zipformer rows used the 12-second sample, the Moonshine rows the 3-second one.)

**Back-to-back requests** (throughput in seconds of audio per second, 12-second clip):

| Clients | `zipformer-en-20m-streaming` batched / unbatched | `moonshine-tiny-en` batched / unbatched | `faster-whisper-tiny` batched / unbatched |
| --- | --- | --- | --- |
| 1 | 9.8 / 10.3 | 26.6 / 30.2 | 14.1 / 14.0 |
| 2 | 15.3 / 12.1 | 28.1 / 28.5 | 14.0 / 13.6 |
| 4 | 21.5 / 12.2 | 28.5 / 27.4 | 12.2 / 14.0 |
| 8 | 25.7 / 11.3 | 28.3 / 27.5 | 13.4 / 13.8 |

**What batching buys on CPU, honestly:**

- For the **streaming transducer** it is a clear win: with 8 live streams, batched decoding kept the median first partial at 75 ms against 448 ms unbatched, and whole-file throughput more than doubled (25.7× against 11.3× real time). A single stream uses little of the CPU per step, so stepping eight streams in one call shares the cores well.
- For **Moonshine and Whisper it buys almost nothing in throughput** (28× against 27×; 13× against 14×). A single decode already saturates four cores, so a batch of eight takes about eight times as long. It still helps fairness under load: at 8 streams Moonshine's p95 final was 1,977 ms batched against 2,386 ms unbatched, at the price of slower partials when only 2 to 4 streams are active.
- Batching is therefore on by default but never waits for a batch to fill (`batch_wait_ms = 0`): an idle server answers a lone request immediately.
- Four cores carry about 4 live Moonshine streams or 8 live Zipformer streams before latency climbs past half a second. Beyond that the queue bounds latency by refusing work (`429`), which is what you want from a server.

**Speaker diarization** (`pyannote-3.0-eres2net`: pyannote segmentation-3.0 + ERes2Net embeddings, ONNX): on a 21-second clip of two synthetic voices (Kokoro `af_heart` and `am_michael`) taking six turns, the diarization error rate was **5.2%**, all of it speech missed at turn boundaries, with no speaker confusion (0% with a 250 ms collar), at RTF 0.11. Labelling live on the WebSocket (one embedding per utterance) got all six turns right and added about 90 ms per final. On four short public recordings the *number* of speakers was right for the three two-speaker clips and one too many (5 for 4) on the four-speaker one; with the default threshold of other embedding models it over-counts more. Pass `num_speakers` when you know it. That is a smoke test, not a benchmark: no overlapping speech, no noise.

**A voice-agent turn** (Realtime API; `moonshine-tiny-en` → SmolLM2-360M-Instruct on llama-cpp-python, 2 threads, on the same 4 cores → `piper-en-us-amy-low`), measured from the moment the end of speech is detected, median of three turns: transcript after 339 ms, first LLM token after 494 ms, **first audio after 2.4 s**. With Kokoro as the voice the first audio came after 3.8 s. Most of the wait is the tiny model writing its first sentence while it shares the CPU with recognition and synthesis; a language model on its own hardware changes these numbers completely, so read them as "the pipeline works", not as what to expect. The answers of a 360M model are not worth quoting.

**Text to speech under load** (`piper-en-us-amy-low`, a 200-character paragraph): first audio after 138 ms with one client, 402 ms with two, 826 ms with four; throughput stays near 27× real time. Synthesis is not batched; requests interleave sentence by sentence.

## How it compares

bolo is not the only way to serve speech, and often not the best one. Fairly:

| | What it is | Choose it over bolo when | Choose bolo when |
| --- | --- | --- | --- |
| [speaches](https://github.com/speaches-ai/speaches) | OpenAI-compatible STT/TTS server (faster-whisper, Kokoro, Piper), MIT | you need its WebRTC realtime transport or speaker-embedding endpoint, or you prefer pyannote's PyTorch diarization pipeline | you want its API (including the Realtime API and chat-completions audio) with micro-batching, backpressure, more model families, telephony audio and latency tooling, without PyTorch. Its last commit was in April 2026 and the question "[Actively supported?](https://github.com/speaches-ai/speaches/issues/659)" has been open since July 2026 |
| [LocalAI](https://github.com/mudler/LocalAI) | a large, very active platform for LLMs, images, audio and more, with backends pulled as separate images, MIT | you want one gateway that also *runs* the language models, images and embeddings, GPU acceleration, or its much bigger ecosystem | you want a single small Python process for the speech side, with realtime streaming and per-request latency traces, in front of whatever LLM you already run |
| [vLLM-Omni](https://github.com/vllm-project/vllm-omni) | vLLM extended to omni-modal and speech models | you have GPUs and want true speech-to-speech models, expressive LLM-based TTS or full-duplex omni models at scale | you are on CPUs or small machines and want a cascade of classic ASR/TTS models around any text LLM, with low latency |
| [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) | realtime speech-to-text with state-of-the-art streaming policies (SimulStreaming/AlignAtt), streaming diarization and translation, Apache-2.0 | realtime transcription quality is the whole job: its streaming policies for Whisper are more advanced than bolo's LocalAgreement, and its live diarization handles overlapping speakers, which bolo's per-utterance labelling does not | you also need text-to-speech, batch transcription, an OpenAI-compatible API (including Realtime) and model management in the same server, without PyTorch |
| [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) | a polished OpenAI-compatible server for Kokoro TTS, with GPU images | you only need Kokoro, want GPU speed, word-level captions or voice mixing | you want Kokoro *and* other voices and speech-to-text behind one API |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) servers | the WebSocket servers that ship with the engine bolo builds on | you want the thinnest possible layer, or C++/mobile/embedded targets | you want HTTP APIs clients already speak, a catalog, scheduling across models and observability |

bolo stands on these projects' shoulders: most of what it runs comes from the sherpa-onnx and faster-whisper teams and from the people who trained and exported the models.

## What is not tested

Stated plainly, so you do not find out the hard way:

- **GPU.** No GPU was available. `device = "cuda"` is passed to faster-whisper and sherpa-onnx, and there is a `Dockerfile.gpu`, but none of it has been run.
- **Apple Silicon and macOS, Linux arm64, Windows.** Not run. The dependencies publish wheels for macOS and arm64, so it should work there; Windows is unknown.
- **Python 3.10 and 3.11.** The package targets 3.10 to 3.13; the test-suite was run on 3.12 and 3.13 only.
- **Real telephony.** The G.711 paths are tested end to end with synthesised speech, not on a phone network.
- **Home Assistant, Pipecat, LiveKit, browsers on the Realtime endpoint.** The protocols they use are implemented and tested with protocol clients; the integrations themselves were not run. The example files say so.
- **The container images, the systemd unit and the Kubernetes manifest** were written but not built or deployed here (there was no container runtime).
- **Most of the catalog.** 109 of 180 models are marked untested: their files were checked on the Hub and a sibling of the same architecture was run, but that model was not.
- **The `espeak-ng` executable route** for phonemes (the measurements used the shared-library route from the `espeakng-loader` package).
- **Voice agents with a capable language model.** The conversation path was run end to end with a 360M-parameter model on CPU only (see [Performance](#performance)); the answers of such a model are poor and the turn latency is not representative of a real deployment. Function calling was exercised with a scripted fake model, not a real one. Ollama, vLLM and hosted APIs were not tried.
- **Diarization accuracy.** Measured on one clip made of two synthetic voices taking turns, plus speaker counts on four short public recordings. Overlapping speech, more than a handful of speakers and noisy rooms were not evaluated.
- **Accuracy at scale.** Word error rates here come from a handful of clips. They show that the pipelines work, not how good the models are; for that, see the model publishers' evaluations.
- **Long soak, memory growth over days, hostile input.** Not done.
- Engines `moonshine` (useful-moonshine-onnx), `whisper.cpp` and `kokoro-onnx` have plugins that were written against their documentation and not run.

## Licences

bolo itself is [Apache-2.0](LICENSE). It contains no third-party source code; [NOTICE](NOTICE) lists what it works with.

- **Engines**: sherpa-onnx (Apache-2.0), ONNX Runtime (MIT), faster-whisper and CTranslate2 (MIT). The optional `kokoro-onnx` plugin is MIT but pulls in the GPL `phonemizer` package, so it is not part of `[all]`.
- **eSpeak NG (GPL-3.0-or-later) is a separate, optional component.** It turns text into phonemes for Piper-format voices, KittenTTS, Matcha-TTS (English) and the non-English Kokoro voices. bolo's wheel does not contain it, the `all` extra does not install it, and the container image does not include it unless you build with `--build-arg WITH_ESPEAK=1`. It arrives only when you ask: the console's install button (which asks you to accept the licence), `bolo components install espeak-ng --accept-license`, `pip install 'bolo-speech[espeak]'` (the `espeakng-loader` package: a prebuilt library plus data), or a system package. bolo's own code stays Apache-2.0 either way: Apache-2.0 code may be combined with GPLv3 code, and bolo only calls the component's public interface. But **if you redistribute a bundle that contains eSpeak NG** (an image built with `WITH_ESPEAK=1`, an appliance, an installer) **you must comply with the GPL for that component**. Without it you still have Kokoro English and Mandarin, MeloTTS, MMS-TTS (non-commercial) and the Coqui Bengali voices. sherpa-onnx removed espeak-ng from its own binaries in version 2.0, and the GPL `piper-tts` runtime is not used at all. More in [docs/models.md](docs/models.md#licences-and-the-espeak-ng-question).
- **Models** are downloaded from their publishers under their own licences, which the catalog records per model and the console shows. Most are MIT, Apache-2.0 or CC-BY-4.0. Some are **non-commercial** (MMS-TTS, non-English Moonshine, several Piper voices, flagged ⚠) or have custom terms (SenseVoice, Paraformer). Nothing in the default catalog is gated, and bolo never downloads a gated model without your own token.

## Roadmap

- Realtime API over WebRTC, ephemeral client secrets, MCP tools.
- Overlap-aware online diarization and known-speaker enrolment.
- The official (gated) AI4Bharat IndicConformer models with a user token; Telugu and Malayalam Piper voices; more Indian-language voices.
- Loading raw Piper voices without a conversion step.
- GPU validation, Apple Silicon validation, and published container images.
- Models trained for 8 kHz telephone speech.
- An attention-based streaming policy for Whisper as an alternative to LocalAgreement.
- Authentication beyond a single shared key.

## Contributing

Issues and pull requests are welcome, especially measurements on hardware we do not have and new catalog entries: [CONTRIBUTING.md](CONTRIBUTING.md) explains the setup and how to add and validate a model. Please read the [Code of Conduct](CODE_OF_CONDUCT.md); report security problems as described in [SECURITY.md](SECURITY.md).
