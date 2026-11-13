# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
aims to follow [Semantic Versioning](https://semver.org/) once it reaches 1.0.

## [0.2.0] - unreleased

### Added

- **Speech synthesis in the browser.** The live demo's text-to-speech screen
  now speaks: nine voices run in the page with ONNX Runtime Web
  (WebAssembly, up to 4 threads), downloaded once and kept by the browser.
  IISc SYSPIN voices for Hindi, Maithili (female, male), Bhojpuri (female,
  male), Marathi, Kannada and Indian English read characters directly (no
  phonemiser); the US English Piper voice "amy" gets its phonemes from a
  vendored copy of uccharan (Apache-2.0), so no eSpeak NG is involved. The
  front end matches sherpa-onnx (cleaning, token ids, pause shortening; checked
  sample for sample), spells numbers out in Hindi, Marathi, Kannada and
  English, and lists the characters a voice cannot say. Sentence-by-sentence
  streaming playback, time to first audio, real-time factor, the chunk
  timeline, WAV download and a round trip through the in-browser recogniser
  (Hindi, English). Voices are shrunk for the browser (8-bit convolution
  weights outside the text encoder, 16-bit elsewhere: 114 MB to 36.5 MB, no
  measurable change in round-trip error) and published same-origin with the
  Pages site by the "Live demo" workflow (`ui/scripts/voices/build.py`); they
  are not in git. The console served by `bolo serve` gets the same screen as
  an "In browser" mode next to "Server", fetching the voices from the public
  demo. On iPhone and iPad (WebKit) only one in-browser model runtime is kept
  at a time and the synthesiser's runtime memory is capped at 1 GiB, because a
  second 4 GiB WebAssembly memory does not fit there ("RangeError: Out of
  memory"); any out-of-memory load is retried once on one thread with
  everything else released, then explained ("close other tabs").
- **Indian languages.** Speech-to-text entries for all 22 scheduled languages
  of India (`indicconformer-<lang>-community`: AI4Bharat IndicConformer, MIT,
  ungated community exports pinned to a commit), 16 of them run on real
  recordings with the measured error rates in the catalog; among them
  Maithili. Also `indicconformer-ta` / `-te` / `-ml` (smaller exports),
  Hinglish in Latin script (`faster-whisper-base-hinglish`), Whisper
  fine-tunes for Tamil, Kannada, Telugu, Bengali, Punjabi and Hindi
  (large-v2), NVIDIA Nemotron 3.5 streaming, and the Nepali Piper voice
  `chitwan`. Bhojpuri is documented honestly: no dedicated model, the Hindi
  model as a fallback with its measured error.
- **Any Piper voice by name**: `bolo pull piper:<voice>` fetches a voice from
  the Piper project and converts it for sherpa-onnx after the download,
  without any machine-learning library (`source.convert: piper` in catalog
  entries). Ten Indian voices defined that way: Telugu (3), Tamil (2),
  Malayalam (2), Marathi, Bengali, Urdu.
- **Local model directories**: `bolo pull ./dir --kind … --name … --language …`.
- **Archive sources**: catalog entries can point at a `.tar.gz` / `.zip`
  (for example a GitHub release asset) that is verified by SHA-256 and
  unpacked safely. The Hugging Face token is no longer sent to other hosts.
- **Voices converted by this project**: `tools/voices/` and a "Voices"
  workflow convert the IISc SYSPIN VITS voices (Maithili, Bhojpuri, Hindi,
  Bengali, Marathi, Telugu, Kannada, Gujarati, Magahi, Chhattisgarhi, Indian
  English) and publish them as release assets (release `voices-v1`, 22
  voices).
- **Closed character sets**: character-input voices no longer drop what they
  cannot read. Digits are spelled out where a number normaliser exists;
  otherwise the request fails with `unsupported_characters`, naming them.
- **Realtime API**: incremental user-transcript deltas while the user is
  speaking; a WebRTC transport (`POST /v1/realtime/calls` and `/hangup`,
  extra `webrtc`, with `output_audio_buffer.*` events); ephemeral client
  secrets (`POST /v1/realtime/client_secrets`); optional semantic turn
  detection (`turn_detection.type = "semantic_vad"`) with the Smart Turn v3
  end-of-turn model, a new `turn` model task and an `onnxruntime` engine.
- **Diarization**: speaker labels on words (diarization and word timestamps
  together); `known_speaker_names[]` / `known_speaker_references[]` to name
  speakers from reference recordings (also on `/api/v1/diarize`); a
  `speakers` message with revised labels at the end of a realtime stream;
  consolidation of spurious speakers after clustering
  (`diarization_min_speaker_s`, `diarization_merge_similarity`).
- `capabilities` in `/api/v1/info` describe all of the above; catalog entries
  have a `status` field.
- Ogg/Opus is streamed in 100 ms pages when PyAV is installed.
- Catalog option `token_timestamps` for Whisper exports with attention
  outputs (sherpa-onnx); none of the current entries has them.
- The release workflow can be started by hand, publishes to PyPI with an API
  token and creates the tag and the GitHub release itself.
- **uccharan as the phonemiser.** Piper-format, KittenTTS, Matcha-TTS and
  Kokoro voices get their phonemes from uccharan (Apache-2.0; new extra
  `uccharan`, part of `all` and of the Docker image) wherever it covers the
  voice's language: English (US, British) and Hindi. 26 catalog voices plus
  Kokoro's English and Hindi voices therefore work without eSpeak NG; the 67
  phoneme-based voices in other languages still need it. For Kokoro English
  uccharan is a port of misaki, the front end Kokoro was trained with: in our
  run it halved the error of the lexicon-only path. Measured voice by voice
  against eSpeak NG (docs/models.md, "Phonemisers").
- **Choosing the phonemiser**: setting `phonemizer` (`auto`, `uccharan`,
  `espeak-ng`), a `phonemizer` field per request (speech endpoint, TTS
  WebSocket, Realtime sessions, chat completions), a per-voice pin
  (`bolo models config <id> --phonemizer …`, `PUT /api/v1/models/{id}/settings`)
  and uccharan's `improved` mode (`uccharan:improved`). Catalog entries list the
  choices for each voice.
- **Low-latency default voice**: `default_realtime_tts` (`piper-en-us-amy-low`)
  for voice agents, Wyoming, `tts-1` and `model: "realtime"`; Kokoro stays the
  quality default.
- **Engine plugins**: out-of-process engines discovered from installed
  packages (entry points `bolo.engines`) or a configured command, with a
  versioned JSON-lines protocol, restart with back-off after a crash, idle
  shutdown and resource limits; their models use bolo's queues, traces and
  metrics. A reference plugin is used in the tests. Catalog entries for models
  that need PyTorch (Indic Parler-TTS, IndicF5, Veena, Svara, SraVaani, the
  official IndicConformer) point at the future `torch` plugin
  (docs/plugins.md).
- **Meta MMS-TTS voices for Indian languages** (CC-BY-NC-4.0, non-commercial):
  conversion in `tools/voices/export_mms.py` and the Voices workflow; 18
  catalog entries, published as release assets and on Hugging Face; the first
  voices for Punjabi, Odia, Assamese and Dogri among them.
- **Gated models** in the catalog, labelled `gated`; the error now says where
  to accept the terms before anything else is tried.
- **Hugging Face hosting** of converted models (one public repository per
  model under `codebucketsolutions/`, with a model card) when the workflow has
  an `HF_TOKEN`; catalog entries can name a fallback source.
- **Diarization**: overlapping speech is reported (`overlaps` from
  `/api/v1/diarize`, `overlap: true` on transcript segments). Transcribing both
  overlapping turns whole was tried and made the second speaker's text worse,
  so overlapping audio is still transcribed once.
- **Wyoming**: transcript streaming (`transcript-start`, `-chunk`, `-stop`).
- Kokoro: a shorter first chunk (`first_chunk_chars`) and a warm-up on load
  for earlier first audio.
- A CI job builds the container image and smoke-tests it on every push.
- **Separating overlapping speech (opt-in).** `separate_overlaps=true` on a
  diarized transcription or on `/api/v1/diarize` gives each speaker in an
  overlap a track of their own before transcription: VoxLock target-speaker
  extraction (`voxlock-int8`, Apache-2.0, run by sherpa-onnx 2.0.1, no
  PyTorch), enrolled from each speaker's own speech outside overlaps.
  Segments transcribed from a track are marked `separated: true`; the response
  carries a `separation` report (what was separated, what was skipped and why)
  and an `x-bolo-separation` header; a `separate` phase in traces, new
  metrics. Audio below 16 kHz is refused unless
  `separation_allow_narrowband=true`. Off by default; docs/models.md,
  "Overlapping speech", says for which audio it passed our evaluation. New
  catalog task `separation`; engine plugins can serve separation models
  (blind or target-speaker) through new protocol operations.
- **The PyTorch engine plugin, bolo-torch 0.1.0** (`plugins/bolo-torch`, its
  own package and changelog). Runs Indic Parler-TTS, Parler-TTS Tiny, IndicF5,
  Veena and Svara (text to speech) and SpeechBrain SepFormer (WHAMR! 8 kHz and
  16 kHz) and pyannote PixIT (separation) out of process, with extras per
  model family. New catalog entries `parler-tts-tiny-v1`,
  `sepformer-whamr16k`, `sepformer-whamr` and `pixit-ami`; Indic Parler-TTS,
  IndicF5, Veena and Svara list their voices. All run on CPU here except Veena
  and Svara (3B models); every one is slower than real time on CPU
  (plugins/bolo-torch/README.md has the numbers).
- Plugin errors coded `model_gated` or `invalid_request` now reach the client
  as `401 model_gated` / `400 invalid_request` instead of `503`.

### Changed

- Character-input voices (IISc SYSPIN, MMS-TTS) now get the text cleaning their
  training text had (SYSPIN: Coqui's `multilingual_cleaners`, the same front end
  as the in-browser voices; MMS: lower case), so capitals, `;`, `:`, hyphens,
  brackets and precomposed nukta letters map to the symbols the voices learnt.
- The 22 SYSPIN voices are published (release `voices-v1`) and pullable.
- The container image includes Moonshine (the default realtime recogniser was
  missing from it) and uccharan; `PYTHON_IMAGE` selects the base image.

- A loaded model's memory is never estimated below the size of its files
  (the estimate could read far too low, or zero).
- Examples for Pipecat and LiveKit Agents are now runnable scripts that were
  run against bolo (Pipecat 1.16.0, LiveKit Agents 1.8.11).

### Validated in this release (see the docs for numbers and conditions)

- Voice-agent turns through Ollama and through llama.cpp's `llama-server`
  with a real (small) model; a turn over WebRTC with real models.
- Pipecat and LiveKit Agents, each through the Realtime endpoint, a Realtime
  transcription session and the HTTP endpoints.
- The eSpeak NG executable route with a real `espeak-ng` binary.

### Known limitations

- WebRTC was tested with a WebRTC library, not with browsers, NAT or TURN.
  SIP is not implemented.
- The semantic end-of-turn model judges sound, not grammar; in our small
  test it helped with a pause in mid-phrase and was indecisive on Hindi.
- Of the published MMS-TTS voices, eight were run by the maintainers (from
  local conversions with the same script); the other ten are untested.
- Speaker counting was evaluated on 18 recordings; no diarization error
  rate on real recordings was measured. Known-speaker naming was tested with
  synthetic voices only.
- Six scheduled languages (Bodo, Dogri, Kashmiri, Konkani, Manipuri,
  Santali) have recognisers that were not run, and eleven have no voice.
- Word timestamps are still unavailable for Whisper through sherpa-onnx and
  for Moonshine.
- uccharan is not on PyPI yet at the time of writing; until it is, install its
  wheel yourself (the `uccharan` extra cannot be resolved), and CI builds the
  image without it.
- The PyTorch engine plugin is not written yet: the models that need it are
  listed and unusable.
- Online (realtime) speaker labels still assume one speaker per utterance;
  overlap is handled for whole recordings only.
- Overlap separation was evaluated on synthetic two-voice mixes made with
  bolo's own voices and on 24 AMI meeting excerpts; not on real telephone
  calls, and not with more than one recogniser per language.

## [0.1.0]

First public version (source only; never tagged or published to PyPI).

### Added

- Console: the microphone is requested at the tap, in parallel with loading the
  in-browser model; speech said in the meantime is buffered and transcribed
  once the model is ready. Loading shows its stage and bytes, a stuck load is
  reported with a single-threaded retry, and a refused microphone is explained
  with steps for the device (including in-app browsers). `?debug=1` adds a
  diagnostics panel.
- Console: Indian languages. The language select lists them under their own
  names, the model picker narrows to models that support the chosen language,
  and the in-browser mode gains a Hindi model and multilingual Whisper Tiny and
  Base, with measured error rates shown next to them. Fonts for Indian scripts
  ship with the console and load on demand.
- New mark (a person speaking) and a "bolo · बोलो" wordmark.

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
  user transcripts arrive whole rather than as incremental deltas
  (addressed in 0.2.0, except SIP, MCP tools and image input).
- Diarization: no known-speaker references (added in 0.2.0); the realtime
  labels assume one speaker per utterance and do not handle overlapping
  speech.
- Voice-agent function calling was exercised with a scripted stand-in model
  only; Ollama, vLLM and hosted models were not run (Ollama was in 0.2.0).
- Piper, KittenTTS, Matcha-TTS (English) and non-English Kokoro voices need
  the optional eSpeak NG component.
