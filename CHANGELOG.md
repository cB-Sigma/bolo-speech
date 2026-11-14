# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
aims to follow [Semantic Versioning](https://semver.org/) once it reaches 1.0.

## [0.2.0] - 2026-11-14

### Added

- **uccharan's normalise-only mode for voices that read text** (`tts-server` models,
  Orpheus-family voices on llama.cpp, plugin voices), on by default: numbers, money with
  lakh/crore, dates, times, phone numbers, units, abbreviations, URLs and email addresses written
  out in English, Hindi and the other Indian languages uccharan reads, Latin-script English kept
  inside Indic text. Per-model catalog option and pin `text_normalizer` (`uccharan`, `light`,
  `model`, `none`); `normalize` on `/v1/audio/speech` also takes these names. bolo's own
  normaliser stays the fallback when uccharan is not installed.
- **Pronunciation word lists** (`<data_dir>/wordlists/<name>.tsv`, uccharan's TSV format, PLS
  and JSON import, PLS export): `GET/PUT/DELETE /api/v1/wordlists/{name}`, `GET /api/v1/wordlists`,
  `POST /api/v1/wordlists/{name}/preview` (what each chosen voice would receive). Applied by
  priority: SSML and inline `[word](/ipa/)`, request `pronunciations`, request `wordlists`, lists
  pinned to the model, the `default` list. Phoneme voices get the entry's IPA in their phone set,
  text voices its spoken form or a respelling, voices with `text_markup: kokoro` the markup.
  `GET /api/v1/text-input/{model}` shows each voice's record. Traces record the normaliser, the
  normalised text and the entries applied.
- **SSML subset** on `/v1/audio/speech` (`input_format: "ssml"`): `<sub>`, `<phoneme>`,
  `<say-as>` (characters, cardinal, ordinal, digits/telephone, date with format), `<break>`,
  `<p>`, `<s>`, `<lang>`, `<emphasis>`, `<prosody>`. `WS /api/v1/tts/stream` takes `normalize` and
  `wordlists` query parameters.
- Console: a Word lists screen (editor, import, and a preview of what each voice receives; shown
  when the server has word lists), a Conversation screen for talking to a voice agent and picking
  its pipeline, a Components screen for optional software, per-model settings, voice cloning on
  the text-to-speech screen and speaker results on the speech-to-text screen.

- **Orpheus-family voices without PyTorch** (kind `orpheus-llamacpp` of the
  `tts-server` engine, `bolo/engines/orpheus_llamacpp.py`): llama.cpp's
  `llama-server` runs the language model from a GGUF; bolo builds the prompt
  token ids (formats `svara`, `veena`, `orpheus`), streams the generated ids
  from `/completion` (token-array prompt, `return_tokens`), groups them into
  7-token SNAC frames and decodes them with the SNAC 24 kHz decoder in
  onnxruntime, streaming PCM with the same windows as bolo-torch (4 frames per
  chunk, 3 frames of left context, one frame held back). Sampling per the model
  cards, stop at end-of-speech (the stream is closed, which frees the slot), a
  token budget per character. Managed or operator-run llama-server, as for
  every `tts-server` model; new extra `bolo-speech[orpheus]` (onnxruntime). New tool
  `server/tools/export_snac_onnx.py`: bolo's own fp32 and fp16 export of the
  SNAC decoder, checked against PyTorch SNAC (fp32: 112-127 dB SNR, max abs
  difference at most 2.9e-6 on random and real Svara code sequences, noise
  term off on both sides; bolo's streaming windows match bolo-torch's).
  Catalog (`catalog/tts-orpheus.yaml`, all `pending` until the decoder is
  published at `codebucketsolutions/snac-24khz-onnx`): `svara-tts-v1-llamacpp`
  (Q4_K_M) and `svara-tts-v1-q8-llamacpp`, `veena-llamacpp` (Q4_K_M) and
  `veena-q8-llamacpp`, `orpheus-3b-0.1-ft-llamacpp` (English),
  `snortts-indic-v0-llamacpp` (nine Indian languages). Svara and Veena Q4_K_M
  were run on the 4-vCPU test machine (RTF 11-16, CPU is for offline use;
  Hindi, Bengali, Tamil and English transcribed back; Q2_K is too lossy). The
  Svara entries use the Orpheus prompt layout, with which Bengali no longer
  speaks the speaker label; see docs/models.md.
- **Realtime pipelines** (opt-in; the cascade stays the default): setting
  `realtime_pipeline` and the bolo extension `session.pipeline`. `audio-llm`
  sends the user's turn as `input_audio` to an audio-understanding
  chat-completions model (settings `audio_llm_base_url`, `audio_llm_model`,
  `audio_llm_api_key`, falling back to `llm_*`; e.g. llama-server with Gemma 4
  E2B/E4B or Ultravox) and bolo's TTS speaks the answer. `speech-to-speech`
  hands turns to a speech-to-speech model through a backend interface
  (`bolo/s2s.py`); the first backend is llama-server's WebSocket
  `/v1/realtime` (LFM3-Audio 2B, English, tool calls; LFM Open License, so
  never a default), either one the operator runs (`s2s_server_url`,
  `s2s_server_api_key`) or one bolo starts for `s2s_model` (default
  `lfm3-audio-2b-llamacpp`; `s2s_parallel`, `s2s_server_args`) with the
  tts-server engine's process handling. bolo keeps auth, codecs and resampling,
  turn detection, barge-in (`response.cancel` upstream), items, tool-call
  items, user transcripts (`realtime_transcribe_user`) and traces.
  `capabilities.realtime.pipelines` in `GET /api/v1/info`. Tested against
  scripted fake upstream servers only. `websockets` is now a direct dependency.
- Catalog: `f5-tts-v1-base` (SWivid F5-TTS v1, CC-BY-NC-4.0, bolo-torch
  `f5-original` extra), `indic-speak` (Bodhan AI, gated, Indic Open Model
  License, bolo-torch `indic-speak` extra), `indextts-2.5-audiocpp` (bilibili
  licence, audio.cpp); `mira-tts-audiocpp` (CC-BY-NC-SA) is `pending` (no
  audio.cpp package published); `spark-tts-0.5b`, `maskgct` and
  `diarizen-wavlm-large-s80-md-v2` (non-commercial) are listed with a note on
  what is missing, as bolo-torch kinds that are not implemented (no runtime
  bolo can use yet). All `validated: untested`.
- **External TTS servers** (engine `tts-server`): text-to-speech forwarded to
  llama.cpp's `llama-server` (`POST /tts`, streamed PCM16) or audio.cpp's
  `audiocpp_server` (`POST /v1/audio/speech`). bolo either connects to a server
  the operator runs (new setting `tts_servers`, `"<model id or runtime>=<url>"`,
  and `tts_server_api_key`; nothing is downloaded) or starts one per loaded
  model on a loopback port, waits for `/health`, stops it when the model is
  unloaded and restarts it after a crash with the plugin host's back-off
  (settings `llama_server_bin`, `audiocpp_server_bin`, `tts_server_args`,
  `tts_server_start_timeout_s`, `tts_server_timeout_s`). Audio is resampled to
  the catalog rate; `/v1/audio/speech`, SSE, the TTS WebSocket and voice
  listings work as for any voice. A missing binary is `424 component_missing`,
  an unreachable or crashed server `503 engine_unavailable`.
- 51 catalog entries on those servers (`catalog/tts-server.yaml`): Chatterbox
  Multilingual V3 (Hindi), Chatterbox Turbo, Qwen3-TTS, Pocket TTS (six
  languages), LFM3-Audio on llama-server; VoxCPM2, Magpie, Chatterbox,
  CosyVoice3, Maya1 and 34 more audio.cpp families. Every entry states
  licence, `commercial_use` and `gated` (non-commercial models are included and
  marked); `options.hardware` (GPU advice and approximate memory) is returned as
  `hardware` by the catalog API. Soprano and Fish Drama 3 on llama-server are
  `pending`. `pocket-tts-en-llamacpp` was run on CPU: RTF 0.72, first audio
  0.77 s, round-trip WER 0 (2 threads, shared 4-vCPU Xeon).
- **Voice cloning** on `/v1/audio/speech`: `reference_audio` (base64 or data
  URL) and `reference_text`, for models with the `voice-cloning` feature;
  others answer 400. Engines opt in with `TTSModel.accepts_extras` /
  `supports_cloning` and receive `SynthesisExtras` (language, reference).
- Components `llama-server` (MIT) and `audio.cpp` (Apache-2.0), `kind:
  "tts-server"`: `bolo components install llama-server` /
  `POST /api/v1/components/llama-server/install` download the pinned upstream
  release (llama.cpp b12375, audio.cpp v0.9.4), check its SHA-256, unpack it
  into `<data_dir>/components/` and check that the binary starts (the audio.cpp
  Linux release needs glibc 2.43).

- **Diarization-conditioned recognition** (speaker-conditioned STT models).
  With such a model a diarized transcription is no longer cut at speaker
  turns: bolo diarizes, then the model transcribes the whole recording once
  per speaker, told frame by frame who is talking, so overlapping speech is
  transcribed for each speaker. The first such models are BUT-FIT's DiCoW
  v3.3 (`dicow-v3.3`, CC-BY-4.0) and SE-DiCoW (`se-dicow`, Apache-2.0) in
  the bolo-torch plugin; they use `nemotron-3-diarization`
  unless the request names another diarizer, and are conditioned on the
  diarizer's speaker turns (a clustering diarizer works too). On 24 AMI
  far-field excerpts cpWER fell from 69.4 % (bolo's default,
  pyannote + whisper-base) to 26.0 %, on 44 synthetic two-voice mixes
  from 55.4 to 6.6 % (SE-DiCoW: 25.3 and 7.9 %), at RTF 0.45 instead of
  0.25 on a 4-vCPU CPU at 2 threads (bfloat16 with AMX; docs/models.md, "Diarization-conditioned
  recognition"). `separate_overlaps` cannot be combined with it.
- Plugin protocol: the `speakers` request (audio plus per-frame speaker
  activity, one result per speaker), sent only to models whose `load` reply
  announces `info.speaker_conditioned`; the protocol stays at version 1.
  Engines implement `STTModel.transcribe_speakers(SpeakerRequest)`.
- `GET /api/v1/info` lists speaker-conditioned models under
  `capabilities.diarization.speaker_conditioned_models`.
- **DiCoW without PyTorch** (`dicow-v3.3-onnx`, catalog entry pending until
  the export is hosted). A core engine runs DiCoW v3.3 in ONNX Runtime with
  NumPy (`bolo/engines/dicow_onnx.py`): Whisper's 128-band log-mel, the STNO
  masks and transformers' long-form greedy decoding as DiCoW's own code
  changes it, reproduced step by step. `server/tools/export_dicow_onnx.py`
  makes the graphs (encoder with the FDDT conditioning, decoder with a
  key/value cache; fp32 and int8). `bolo pull ./dir` registers an export as a
  speaker-conditioned model with `nemotron-3-diarization` as its diarizer.
  On 24 AMI excerpts the int8 graphs scored the same cpWER as the plugin
  (26.0 %), on 40 English synthetic mixes 6.0 against 5.5 %; the fp32 graphs
  gave the plugin's float32 transcripts word for word on two reference
  excerpts. RTF 0.7-1.0 on a shared 4-vCPU machine, no PyTorch needed
  (docs/models.md, "DiCoW without PyTorch"; docs/measurements/dicow-onnx.json).
- `whisper_log_mel` takes `n_mels` (128 for Whisper large-v3 derivatives) and
  computes long recordings block by block.
- `capabilities.diarization.speaker_conditioned_models` leaves out pending
  catalog entries; `bolo pull ./dir` gives a directory of a known kind its
  catalog features and options (`custom.KIND_DEFAULTS`).
- **Nemotron-3-Diarization as a second diarizer** (`nemotron-3-diarization`,
  opt-in). NVIDIA's end-to-end diarization model (OpenMDW-1.1, up to 8
  speakers, overlap-aware) runs in ONNX Runtime without PyTorch: NeMo's
  log-mel front-end in NumPy and the chunked speaker-cache loop for long
  recordings and live audio, both checked against the transformers reference
  (probabilities within 0.001). Choose it with `diarization_model` /
  `model` per request or `default_diarization`. On 24 AMI far-field excerpts
  DER fell from 38.6 to 24.1 % (speaker confusion 14.2 to 2.0 %, speaker
  count right in 18 instead of 7) and cpWER with whisper-base from 69 to
  61 %; on same-gender and telephone mixes it removed the speaker confusion
  of clustering; it runs five times faster (RTF 0.04). It stays opt-in: on
  clean turn-taking dialogues of synthetic voices it merged some clearly
  different voices that clustering separates (docs/models.md, "Speaker
  diarization"). The per-frame speaker probabilities are kept for later
  diarization-conditioned recognition. `server/tools/export_nemotron_onnx.py`
  re-exports the model from NVIDIA's weights.
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

### Fixed

- Kokoro-style `[word](/phonemes/)` overrides were removed with other Markdown links before they
  reached the voice; they now set the word's pronunciation (for every voice).
- URLs were deleted from text before synthesis; they are now read out ("w w w dot example dot
  com slash docs").

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
