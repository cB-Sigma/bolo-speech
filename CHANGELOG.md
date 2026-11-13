# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
aims to follow [Semantic Versioning](https://semver.org/) once it reaches 1.0.

## [0.1.0] - unreleased

First public version.

### Added

- Speech-to-text, text-to-speech and voice-activity detection behind one
  server, without PyTorch: engines are plugins for sherpa-onnx (ONNX Runtime)
  and faster-whisper (CTranslate2), with optional plugins for
  useful-moonshine-onnx, whisper.cpp and kokoro-onnx.
- A data-driven model catalog with licences, pinned revisions and a
  `validated` mark for the models that were run on CPU; `bolo pull hf:<repo>`
  for models outside it.
- Resumable, checksum-verified downloads into a content-addressed cache.
- OpenAI-compatible `/v1/audio/transcriptions` (json, text, srt, vtt,
  verbose_json, SSE streaming), `/v1/audio/translations`, `/v1/audio/speech`
  (mp3, opus, aac, flac, wav, pcm, plus 8 kHz μ-law and A-law) and `/v1/models`.
- speaches-compatible model management, `/api/ps` and VAD timestamp routes,
  and its environment variable names.
- Native API under `/api/v1`: catalog, pulls with progress, load/unload,
  statistics, per-request traces with phase timings, Server-Sent Events,
  benchmark jobs, built-in samples.
- Realtime speech-to-text over WebSocket for every model: incremental
  decoding for streaming models, VAD-segmented re-decoding with
  LocalAgreement stabilisation for the others; PCM, μ-law and A-law input.
- Streaming text-to-speech over WebSocket with incremental text input.
- Wyoming protocol server (speech-to-text and text-to-speech, including
  streaming synthesis) for Home Assistant.
- Per-model scheduler with natural micro-batching, bounded queues with
  `429` backpressure, cancellation on disconnect; lifecycle manager with lazy
  loading, a crash-isolating first load, idle unloading, memory budget and
  LRU eviction.
- Telephony helpers: G.711 μ-law/A-law codecs, RTP-sized framing, 8 kHz paths.
- Indic-aware sentence segmentation and Hindi number normalisation.
- Prometheus metrics, health and readiness endpoints.
- `bolo` command: `serve`, `pull`, `rm`, `list`, `ps`, `show`, `transcribe`,
  `speak`, `bench`, `doctor`.
- Speaker diarization without PyTorch (pyannote segmentation-3.0 and a
  speaker-embedding model through sherpa-onnx): `diarize=true` and
  `diarized_json` on `/v1/audio/transcriptions`, speaker labels in
  `verbose_json`, SRT and VTT, `POST /api/v1/diarize`, and online speaker
  labels on realtime finals.
- OpenAI Realtime API over WebSocket at `/v1/realtime`: transcription
  sessions, and conversation sessions (speech in, language model, speech out)
  with server VAD, barge-in, function-call events and per-turn latency
  traces. The language model is any OpenAI-compatible chat-completions
  endpoint (`llm_base_url`, `llm_model`, `llm_api_key`).
- `/v1/chat/completions` with audio input (`input_audio`) and audio output
  (`modalities`, `audio`), streaming included; other requests are proxied to
  the configured language model.
- Optional components: eSpeak NG is not shipped, and can be installed on
  request from the console, `bolo components install espeak-ng` or
  `POST /api/v1/components/espeak-ng/install` after accepting its licence
  (GPL-3.0-or-later), taking effect without a restart. Catalog entries show
  `requires` and `available`; voices that need a missing component answer
  `component_missing`.
- `bolo components` and a language-model check in `bolo doctor`.
- Web console served from the package.

### Known limitations

- Only Linux x86-64 on CPU has been tested. GPU, Apple Silicon, arm64 and
  Windows are untested.
- The Realtime API is WebSocket only (no WebRTC or SIP), has no ephemeral
  client secrets, MCP tools, image input or semantic turn detection, and
  user transcripts arrive whole rather than as incremental deltas.
- Diarization: no known-speaker references; the realtime labels assume one
  speaker per utterance and do not handle overlapping speech.
- Voice-agent function calling was exercised with a scripted stand-in model
  only; Ollama, vLLM and hosted models were not run.
- Piper, KittenTTS, Matcha-TTS (English) and non-English Kokoro voices need
  the optional eSpeak NG component.
