# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
aims to follow [Semantic Versioning](https://semver.org/) once it reaches 1.0.

## [Unreleased]

### Added

- **Remote code as an explicit, pinned opt-in per model.** A catalog entry whose model exists only as Python in its
  Hugging Face repository (transformers' `trust_remote_code`) pins it in `options.remote_code`: the reviewed commit
  (the entry's source revision), what the review found, and the sha256 of every `.py` file and of every JSON file
  with an `auto_map`; catalog validation refuses an incomplete pin. The code runs only after the operator opts in for
  that model: the per-model setting `allow_remote_code` (`bolo models config <id> --set allow_remote_code=true`,
  `PUT /api/v1/models/{id}/settings`), off by default; bolo sends plugins that setting in `load` options and
  overwrites whatever a catalog entry says. `bolo pull`, a pull job's `warnings`, `bolo show`, the catalog API
  (`remote_code`: `revision`, `files`, `reviewed`, `allowed`, `notice`) and every load (a log warning) say what runs.
  docs/models.md, "Remote code". Catalog entries may carry `options.license_note` (shown by `bolo pull` and the API).
- **Step-Audio 2 mini runs** (`step-audio-2-mini`, bolo-torch `step-audio-2` extra): text streamed, tool calls, speech
  from the authors' token2wav decoder in 25-token pieces, voices `female` and `male`. Its model code is remote code
  (opt-in, pinned); token2wav comes from github.com/stepfun-ai/Step-Audio2 (Apache-2.0) at a pinned commit. No longer
  listed as not runnable. Tested with fakes only; nothing ran for real.
- **GLM-4-Voice 9B runs** (`glm-4-voice-9b`, bolo-torch `glm-4-voice` extra): text and speech interleaved and
  streamed (22.05 kHz, first audio after 25 speech tokens), `quantization: int4` for about 9 GB of GPU memory. The
  download now includes glm-4-voice-tokenizer and glm-4-voice-decoder, pinned by revision and sha256. ChatGLM's code
  is remote code (opt-in); the tokenizer and decoder code come from THUDM/GLM-4-Voice and Matcha-TTS at pinned commits.
  Licence labelled everywhere: commercial use only after registering with Zhipu (`license_note`). Fakes only.
- **Kimi-Audio 7B Instruct runs** (`kimi-audio-7b-instruct`, bolo-torch `kimi-audio` extra and flash-attn, NVIDIA
  only): the authors' `kimia_infer` at a pinned commit, its generation loop run step by step so text streams and
  speech comes in 30-token chunks. The download now includes the pinned glm-4-voice-tokenizer, which kimia_infer
  otherwise fetches unpinned from the Hub. Remote code opt-in. Fakes only.

- **More denoisers, chosen and pulled like any other model** (`denoise=<model id>` on transcriptions, Realtime
  sessions, pins and the `denoise` setting; `bolo list`, `bolo pull`, recommendations). Every entry states its licence;
  non-commercial ones are in, marked. Runnable: **DPDFNet** (Ceva, Apache-2.0; `dpdfnet-baseline`, `dpdfnet2`,
  `dpdfnet4`, `dpdfnet8` at 16 kHz, `dpdfnet2-48k`, `dpdfnet8-48k` full band, `dpdfnet2-8k`, `dpdfnet8-8k` for
  telephone audio; 9 to 15 MB each, sherpa-onnx, option `attenuation_limit_db`), **DeepFilterNet 3**
  (`deepfilternet3`, MIT or Apache-2.0, 16 MB, torchDF's single streaming ONNX graph on onnxruntime, 48 kHz),
  **RNNoise** (`rnnoise`, BSD-3-Clause, 48 kHz: engine `rnnoise`, the library from the pyrnnoise wheel of the
  platform, pinned by sha256 and called through ctypes, no new dependency), the **Demucs denoiser** (`demucs-dns48`,
  `demucs-dns64`, `demucs-master64`, `demucs-valentini-nc`; CC-BY-NC-4.0, not for commercial use; PyTorch, on the
  bolo-torch plugin, extra `demucs-denoiser`) and **ElevenLabs Voice Isolator** (`elevenlabs-voice-isolator`, cloud,
  `POST /v1/audio-isolation`; not run against a real account). Every entry runs; the ones looked at and left out
  (ClearVoice FRCRN, MossFormer2 and MossFormerGAN, Resemble Enhance, DeepFilterNet 2's three-graph export) are in
  docs/models-unsupported.md with the reason. `gtcrn-simple` stays the default. Measured with the harness of the first
  denoiser (docs/measurements/denoise.json, docs/models.md, "Denoising").
- **`denoise` on the live speech-to-text WebSocket** (`/api/v1/stt/stream`): query or `config` message, the same
  values as for transcriptions (`true`, `false`, a strength, a model id, `denoise_strength`), and the model's pin, its
  catalog option and the `denoise` setting when the stream says nothing. The denoiser keeps its state from chunk to
  chunk where the model has a streaming form (GTCRN and DPDFNet through sherpa-onnx's online denoiser, DeepFilterNet 3,
  RNNoise; the output is in step with the input, the model's own delay cut off) and enhances each chunk with half a
  second of audio before it otherwise. `ready.denoise` says what runs; `metrics.denoise` reports the time per chunk
  and the audio held back (docs/api.md). A cloud denoiser is refused for live streams.
- Catalog downloads can be per platform (`platform` on a `source.urls` entry: `linux-x86_64`, `linux-aarch64`,
  `darwin`, `win32-x86_64`), and a `.whl` is unpacked like a `.zip`.
- Plugin protocol: op `denoise` for models loaded with task `denoise` (docs/plugins.md); the SDK serves a
  `DenoiseModel`, the reference plugin has `fake-denoise`.
- **End-of-turn models to choose from, and models that read the transcript.** `smart-turn-v3.1`, `smart-turn-v3.0`
  and `smart-turn-v3-fp32` (the float32 v3.2 build), BSD-2-Clause; `namo-turn-en` and `namo-turn-multilingual`
  (VideoSDK's Namo Turn Detector v1, Apache-2.0), which read text: at each pause a `semantic_vad` session decodes the
  utterance so far with its transcription model (not for cloud recognisers) and asks the model about that text
  (`tokenizers` joins the `turn` extra); and `silence-only`, no model, which makes `semantic_vad` behave like
  `server_vad`. A session chooses its model with the bolo extension `turn_detection.turn_model` (default
  `default_turn`); traces name the model, and a text model's checks record the text and the decode time;
  `capabilities.realtime.semantic_vad.input` says `audio` or `text`. Catalogued with their terms but not run
  (`runnable: false`, with the reason): `smart-turn-v2` (PyTorch weights only), `livekit-turn-detector` and
  `livekit-turn-detector-en` (LiveKit Model License: only with the LiveKit Agents framework) and
  `ten-turn-detection` (a 7.6 B model; Apache-2.0 plus Agora's conditions in its GitHub licence). Downloads are pinned
  by SHA-256. Measured on a scripted set of 20 English and 6 Hindi finished and unfinished utterances
  (docs/measurements/turn-detectors.json): on English, accuracy 0.63 for the default `smart-turn-v3`, 0.80 for its
  float32 build, 0.98 for `namo-turn-en` on clean text (0.85 without final punctuation), chance for
  `namo-turn-multilingual`. The default stays `smart-turn-v3` (docs/models.md, "End-of-turn detection").
- **More voice-activity detectors, chosen by name.** `silero-vad-v6` (MIT), `ten-vad` and `ten-vad-int8` (TEN VAD:
  Apache-2.0 with Agora's conditions, labelled `commercial_use: false`: only for your own applications, not in a way
  that competes with Agora), `marblenet-vad` (NVIDIA Frame-VAD Multilingual MarbleNet v2.0, NVIDIA Open Model
  License, a third-party ONNX export run by ONNX Runtime with bolo's NeMo log-mel front-end), `webrtc-vad` and
  `webrtc-vad-aggressive` (WebRTC's detector, BSD-3-Clause, through the new `webrtcvad` extra; nothing to download)
  and `energy-vad` (bolo's own energy detector, built in). Downloads are pinned by SHA-256. Any of them works as
  `default_vad`, as `model` of `POST /v1/audio/speech/timestamps` and, new, per Realtime session as
  `turn_detection.vad_model` (bolo extension). Detectors that score frames (WebRTC, MarbleNet) share one segmenter
  with Silero's parameters (threshold with hysteresis, minimum silence and speech, maximum speech, padding).
  Measured on a 74 s stream of the repository's samples, clean and with seven kinds of added noise
  (docs/measurements/vad-detectors.json, `server/tools/measure_detectors.py`): mean frame accuracy over the noisy
  conditions 0.93 for Silero v5 and v6, 0.92 for MarbleNet (0.4 s later), 0.89 for the default Silero v4, 0.85 for
  WebRTC, 0.78 for TEN VAD and 0.74 for the energy detector. The default stays `silero-vad`, the best of them on
  clean audio (docs/models.md, "Voice-activity detection").
- **Console: language packs in one row.** The "Optional components" list on Connect and Models showed each of
  uccharan's ~35 language packs as a card of its own (the Connect column was about 4,800 px tall). The packs are now
  one row, "Language packs · 2 of 35 installed · 1 needed by your models", closed by default (the choice is kept in
  the browser). View opens a compact list, one line per pack: search, Needed / Installed / All, eight per page,
  packs needed by installed models first, Install or Remove on each and "Install needed (N)" for all of them. The
  Components page shows the same list, open.
- `GET /api/v1/components`: language packs carry `required_by_installed_count`, `built_in`, `used_by_count` and
  `removable`; `DELETE /api/v1/components/uccharan-<pack>` removes a downloaded pack this server has not loaded.
- **Turn-based speech-to-speech models behind the `speech-to-speech` pipeline**, loaded by bolo's model manager:
  catalog task `speech-to-speech`, `s2s_model = "<id>"`. bolo keeps the conversation (the user's turns as audio,
  the last eight spoken turns; the model's answers as text; tool calls and results) and sends all of it with every
  response, so the model keeps no state and one copy serves every session; text, audio and tool calls stream back,
  a barge-in stops the model before its next piece. The audio of a turn goes to the model once the turn ends.
  Plugin protocol: op `respond` for models loaded with task `speech-to-speech` (docs/plugins.md), `info.voices` and
  `info.supports_tools`; the SDK's `SpeechToSpeechModel` (`respond(ConversationRequest, cancelled)`), the reference
  plugin's `fake-s2s` and the fake engine's `fake-s2s` model. Tested with those fakes only.
- **More speech-to-speech and audio-in models in the catalog, each with its licence** (non-commercial ones marked,
  none left out; docs/models.md, "Speech-to-speech and audio-in models"). New catalog tasks `speech-to-speech` and
  `audio-llm` (`/api/v1/catalog?task=`). Speech-to-speech on bolo-torch (`omni` extra): `qwen2.5-omni-3b-torch`
  (Qwen Research License, non-commercial), `qwen2.5-omni-7b-torch` and `qwen3-omni-30b-a3b-instruct-torch`
  (Apache-2.0). Listed with the reason they cannot run yet: Kyutai Moshi (`moshiko-` / `moshika-pytorch-bf16`,
  CC-BY-4.0; full-duplex, which bolo's turn-based interface cannot carry), `glm-4-voice-9b` (glm-4-voice License,
  commercial use after registration), `minicpm-o-2.6` (Apache-2.0; remote code only), `step-audio-2-mini`
  (Apache-2.0) and `kimi-audio-7b-instruct` (MIT; no packaged inference code). Audio-in models on llama-server:
  `lfm2.5-audio-1.5b-llamacpp` (LFM Open License, conditional), `ultravox-0.5-llama-3.2-1b-llamacpp` (MIT + Llama 3.2
  Community License), `qwen2.5-omni-3b-llamacpp` (Qwen Research License), `qwen2.5-omni-7b-llamacpp`,
  `qwen3-omni-30b-a3b-instruct-llamacpp`, `gemma-4-e2b-it-llamacpp` and `gemma-4-e4b-it-llamacpp` (Apache-2.0),
  every file pinned by revision and SHA-256. Only `lfm2.5-audio-1.5b-llamacpp` was run (it heard the question but
  transcribed it instead of answering); the others are untested.
- **The `audio-llm` pipeline's model on a llama-server bolo starts**: without `audio_llm_base_url`, an
  `audio_llm_model` naming a catalog entry of task `audio-llm` is pulled and served by a managed llama-server
  (loopback port, random API key, `--parallel audio_llm_parallel`, `audio_llm_server_args`, GPU/CPU placement,
  restarted after a crash, stopped with bolo). `GET /api/v1/info` reports it under the `audio-llm` pipeline.
- **Spoken-language identification**, a new model task `lid`, and **routing by language**. `POST /v1/audio/language`
  (multipart `file`, optional `model`, `top_k`) answers the most likely language with its probability, the `top_k`
  most likely ones, and which recogniser a routed transcription would use. A transcription with `model=auto` and no
  `language` is heard by the identifier first (`default_lid`, `voxlingua107-ecapa`) and sent to the recogniser
  `language_routes` names for the language, else the best installed local recogniser that speaks it (the one the
  recommendations name for the language, then a recogniser made for it), else `default_stt`; the routed recogniser
  is told the language. Below `lid_min_probability` (0.5) or when the identifier cannot run, `default_stt` answers.
  Routing never downloads a recogniser it was not told to use and never picks a cloud model on its own; a request
  that names a model or a language is never changed; translations are not routed. The decision is in the new
  `x-bolo-route` header (`auto -> indicconformer-ta-community; reason=recommended; language=ta; probability=1;
  lid=voxlingua107-ecapa`, plus `recommended=<id>` when a better recogniser for the language is not installed), in
  `x-bolo-language` (`ta; source=detected; probability=1; applied=true`) and in the trace's `route`.
  `GET /api/v1/info` describes it under `capabilities.lid`. Settings `default_lid`, `language_routing` (on),
  `language_routes` and `lid_min_probability`; extra `bolo-speech[lid]` (ONNX Runtime, which `all` already has
  through faster-whisper).
- Language identifiers in the catalog, each with its licence: `voxlingua107-ecapa` (SpeechBrain VoxLingua107
  ECAPA-TDNN, Apache-2.0, 87 MB, 107 languages), `voxlingua107-ecapa-int8` (the same quantised, 22 MB, filterbanks
  computed in NumPy), `ambernet-langid` (NVIDIA AmberNet, NGC Terms of Use, marked not for commercial use),
  `whisper-tiny-lid` and `whisper-base-lid` (MIT: Whisper's own language detection with probabilities, on
  sherpa-onnx's int8 Whisper graphs), all run in ONNX Runtime and measured; `mms-lid-126` and `mms-lid-4017`
  (CC-BY-NC-4.0, int8 ONNX, 978 MB, runnable but not measured); and, not runnable yet,
  `voxlect-indic-lid-whisper-small` (OpenRAIL), `voxlect-indic-lid-mms-lid-256` (CC-BY-NC-4.0) and `lid-indic-vaani`
  (Apache-2.0). On the repository's English and Hindi clips and 39 clips synthesised with bolo's voices in 13 Indian
  languages, the VoxLingua107 model was right on 11/11 and 33/39 (it does not know Odia and heard Urdu as Hindi) at
  RTF 0.11 on one thread; Whisper base 11/11 and 24/39 at 0.61 (docs/models.md, "Language identification";
  docs/measurements/lid.json).
- **Console: voice blends and the language decision.** The text-to-speech page has a "Blend voices" panel for models
  with `voice-blending` (Kokoro): pick up to 8 of the model's voices with weights, see the blend as it is sent
  (`af_bella(0.7)+af_sky(0.3)`) and keep it in the voice library under a name. Saved blends are listed in the library
  apart from the recordings (no consent, no audio) and can be used, or chosen in the voice picker, or deleted. A
  "Detect language" switch sends `detect_language`, and the voice picker offers "From the text" (no voice named, so
  the server picks one that speaks the detected language). After synthesis over chunked HTTP a "What the server
  decided" panel shows `x-bolo-model`, `x-bolo-voice`, `x-bolo-blend` and `x-bolo-language` in words, with a warning
  when a named voice speaks another language than the text.
- **Console: compare two voices.** The Speakers page has a "Compare two voices" panel: upload or record two clips,
  and the server's speaker embeddings (`POST /v1/audio/speech/embedding`, the default embedder or a chosen
  diarization model) give their cosine, read as same speaker / similar / different with the default model's
  thresholds (and as a bare cosine with another model).
- Console: the diarizer list names each model's licence, its runtime (`plugin torch` for pyannote's PyTorch
  pipelines), whether it is gated and whether it is non-commercial; the model pickers mark gated and non-commercial
  models; the Speakers page lists pyannote's PyTorch pipelines among the diarizers a server runs.
- **Console: decoding options and denoising.** The "Who spoke when" transcription panel has an "Advanced decoding"
  section (folded away; empty fields are the defaults): `condition_on_previous_text`, `repetition_penalty`,
  `no_repeat_ngram_size`, `compression_ratio_threshold`, `log_prob_threshold`, `no_speech_threshold`,
  `temperature_increment_on_fallback`, `hallucination_silence_threshold`, `chunk_length`, `max_repeats`,
  `vad_filter` and `vad_parameters`, and a "Denoise first" control (server default, off, or on with a strength and
  a denoise model). The result shows the language with `language_probability`, `x-bolo-denoise` and
  `x-bolo-ignored-options`. The Conversation page sends the same denoise choice as `audio.input.denoise` and shows
  what the session reports. The Models screen lists the `denoise` task as "Denoising".
- `GET /api/v1/info` reports OpenTelemetry export under `capabilities.opentelemetry` (switched on, running, endpoint
  without credentials or query string, protocol, service name, spans exported), read-only; the console's Connect
  screen shows it.
- **pyannote's own PyTorch diarization pipelines** through the bolo-torch plugin, for pyannote's exact results:
  `pyannote-speaker-diarization-3.1` (MIT pipeline and segmentation, CC-BY-4.0 WeSpeaker embedding and PLDA files)
  and `pyannote-speaker-diarization-community-1` (CC-BY-4.0), both gated (accept the conditions on Hugging Face and
  set `BOLO_HF_TOKEN`). They are ordinary `diarization` models: `POST /api/v1/diarize`, `diarize=true` and
  `diarized_json` transcriptions, `num_speakers` / `min_speakers` / `max_speakers`, known speakers. bolo downloads
  the pipeline configuration and every sub-model (segmentation, embedding, PLDA, 33 MB) at pinned revisions with
  sha256. On the demo's two-voice dialogue both gave DER 6.05 % (the ONNX default 5.87 %, Nemotron 5.87 %) at RTF
  about 0.9 on 4 CPU threads (docs/models.md, "pyannote's PyTorch pipelines").
- Plugin protocol: ops `diarize` and `embed` for models loaded with task `diarization` (docs/plugins.md); the SDK
  and the reference plugin serve them. Engine plugins can now serve diarization models.
- bolo-torch builds pyannote pipelines (the new diarizers and PixIT) only from a checked copy of `config.yaml`:
  read with `yaml.safe_load`, sha256 checked against the catalog, only pyannote's own pipeline classes and
  clustering names, plain parameter values and pinned sub-models; anything else is refused before pyannote is
  imported. PixIT's separation model is now part of its pinned, hashed catalog download. pyannote.audio 4's usage
  metrics are off unless `PYANNOTE_METRICS_ENABLED` is set.
- **`POST /v1/audio/speech/embedding`**, speaches' speaker-embedding route: multipart `file` (and `model`) in, the
  speaker embedding out in the shape of OpenAI's embeddings response. The embedder is the one that scores cloned
  voices (`voice_similarity_model`, else the default diarization model's ERes2Net, 192 numbers); speaches' own model
  id falls back to it. With the real model, two halves of one speaker's clip scored 0.81 and another voice 0.09.
- **OpenTelemetry export** (extra `otel`, off by default): each request's trace becomes a span (`bolo.stt`,
  `bolo.tts`, `bolo.agent` ...) with a child span per phase and the model, device, transport, sizes and latencies as
  attributes, sent over OTLP. Switched on by the setting `otel` or by `OTEL_EXPORTER_OTLP_ENDPOINT`; the other
  standard `OTEL_*` variables apply (`OTEL_SERVICE_NAME`, headers, protocol, sampler). Switched on without the extra,
  bolo logs one warning and carries on. Prometheus `/metrics` and the request traces are unchanged.
- **Kokoro voice blends**: `voice` takes a mix of a Kokoro model's voices, `af_bella+af_sky` (equal parts) or
  `af_bella(0.7)+af_sky(0.3)` (weights, normalised to sum to 1); the style vectors are added up with those weights
  and the text is read in the language of the heaviest voice. Unknown voices are a `400` naming the near matches
  (an unknown single voice now names them too). The response carries `x-bolo-blend`. `GET /v1/audio/voices` marks
  the voices that can be blended (`blendable`), and models that blend voices have the feature `voice-blending`.
  With sherpa-onnx, blends are spoken by a second instance of the model that holds the blends in use: the first
  request with a new blend loads it (a few seconds on CPU) and it takes about as much memory again as the model.
- **Saved blends** in the voice library: `POST /api/v1/voices/blends` keeps a blend under a name, used as `voice`
  like any saved voice (without `model`, the blend's model answers). A saved blend is marked `"kind": "blend"`,
  listed under `blends` in `GET /api/v1/voices` (apart from the recordings), needs no recording and no consent,
  and goes into exports.
- **Soniox as a cloud provider** (`kind = "soniox"`, `[providers.soniox]` or `BOLO_PROVIDER_SONIOX_API_KEY`):
  speech-to-text only. `soniox-stt-async-v6` transcribes recordings through Soniox's async API (upload, create,
  poll, read the transcript, then delete the transcription and the file at Soniox) with word timestamps, language
  identification, hotwords and prompt as context, and translation to English. `soniox-stt-rt-v6` uses Soniox's
  WebSocket on the STT WebSocket, with interim text and Soniox's endpoint detection; its recorded audio goes to
  `stt-async-v6`. Errors map as for the other providers (401 `provider_auth`, 429 with retry, timeouts 504, 402
  as `provider_error`), and fallback works both ways. Paid cloud API; audio leaves this server. Not yet run
  against a real Soniox account: tested against a stand-in written from Soniox's documentation.
- **Speaker labels from the recogniser itself**: with `diarize=true` or `diarized_json`, a cloud recogniser that
  labels speakers (Soniox) is asked to, instead of cutting the recording at bolo's speaker turns; no local
  diarization model is needed. Naming `diarization_model`, giving known speakers or asking for overlap
  separation still uses bolo's diarizer.
- **WebRTC from the browser.** The console's Conversation page has a transport switch, WebSocket / WebRTC: with WebRTC
  the microphone is an `RTCPeerConnection` audio track, the Realtime events travel on the `oai-events` data channel,
  the agent's voice plays from the remote track, and the page shows the connection, ICE and data-channel states and
  the audio received. `examples/webrtc_browser.html` does the same in one self-contained page. Tested in headless
  Chromium against a real `bolo serve` (`npm run e2e:webrtc`, `docs/measurements/webrtc-browser.json`); Chrome and
  Edge themselves are next.
- **Voxtral speech-to-text on llama.cpp** (Mistral, Apache-2.0), without PyTorch: `voxtral-mini-3b-llamacpp`
  (ggml-org's Q4_K_M weights and Q8_0 audio encoder, 3.2 GB; run on CPU) and `voxtral-small-24b-llamacpp` (Q4_K_M,
  15.7 GB; catalogued, not run, wants a GPU with about 20 GB). New kind `voxtral-llamacpp` of the `tts-server` engine:
  bolo starts llama-server with the model and its audio encoder (`bolo components install llama-server`; the pinned
  build b12375 already has Voxtral's audio projector) or uses one you run (`tts_servers`). Transcription uses
  Voxtral's own transcription mode with the request's language (`lang:<code>[TRANSCRIBE]`); translation into English
  is an instruction in the chat format; long audio is cut at pauses into pieces of at most 15 s, and generation stops
  at a blank line, after which the model sometimes went on as if chatting. The model gives no timestamps:
  `verbose_json` segments are those pieces, and word timestamps are unavailable. On the shared 4-vCPU test machine:
  RTF 2.0 on an 11.8 s sample (WER 2.8 %), 3 s and Hindi samples exact, load 42 s, 5.7 GB resident. docs/models.md,
  "Voxtral speech-to-text on llama.cpp".
- A catalog entry can set its own `queue_timeout_s` (0: no limit): the pieces of a long recording are queued
  together, and with a slow model the later ones were refused after the usual 30 s wait (`Overloaded`).
- **Language detection for speech**: without `language`, `/v1/audio/speech` and `bolo speak` guess the language of
  the text for a multilingual model (Kokoro, the Indian-language voices, multilingual `tts-server` models) and,
  unless a voice of the model is named, choose a voice that speaks it. The script decides most languages
  (Devanagari, Tamil, Bengali, ...), frequent words tell Latin-script languages and those sharing a script apart;
  no model and no new dependency. An unsure guess keeps the default. An explicit voice or language is never
  changed. The response header `x-bolo-language` reports the decision (`hi-IN; source=detected; detected=hi;
  script=Devanagari; voice=hf_alpha`); `detect_language: false` turns it off.
- **Whisper decoding options and a repetition guard** on `POST /v1/audio/transcriptions` (and translations), as
  form fields or, from the OpenAI SDKs, `extra_body`: `condition_on_previous_text` (still off by default),
  `repetition_penalty`, `no_repeat_ngram_size`, `compression_ratio_threshold`, `log_prob_threshold`,
  `no_speech_threshold`, `temperature_increment_on_fallback`, `hallucination_silence_threshold`, `chunk_length`
  and `vad_parameters` (`threshold`, `min_speech_duration_ms`, `max_speech_duration_s`, `min_silence_duration_ms`,
  `speech_pad_ms`). faster-whisper honours all of them; whisper.cpp the ones it has a parameter for
  (`condition_on_previous_text`, `log_prob_threshold`, `no_speech_threshold`,
  `temperature_increment_on_fallback`); silence removal and `chunk_length` are applied by bolo for every engine.
  Options a model cannot honour are dropped and named in the `x-bolo-ignored-options` header and the request log.
  Values out of range get a `400` naming the field.
- `max_repeats` (default 4): a phrase repeated back to back more often than this in a segment ("the the the ...")
  is cut to one copy, and so is a run of identical segments; `0` turns it off.
- The same options as per-model defaults: catalog `options`, or pinned with `PUT /api/v1/models/{id}/settings`
  and `bolo models config <id> --set key=value`. The request wins over the pin, the pin over the catalog.
- `verbose_json` has `language_probability` when the model detected the language (faster-whisper).
- Denoising before speech recognition: `denoise` on `POST /v1/audio/transcriptions` and `/v1/audio/translations`
  (`true`, `false`, a strength between 0 and 1 that mixes the denoised audio with the original, or a model id;
  `denoise_strength`), on Realtime sessions (OpenAI's `audio.input.noise_reduction`, and bolo's
  `audio.input.denoise`), as a pin per model (`bolo models config <id> --set denoise=0.5`,
  `PUT /api/v1/models/{id}/settings`) and as a server setting (`denoise`, off by default; `default_denoise`). The
  model runs at its own sample rate; the response header `x-bolo-denoise` says what ran, and a denoiser that cannot
  run is skipped with a warning. New catalog task `denoise` with `gtcrn-simple` (GTCRN, MIT, 0.5 MB, sherpa-onnx, RTF
  about 0.06 on one CPU thread). It is opt-in because it can hurt: on noisy mixes of the repository's samples it
  lowered `moonshine-v2-tiny-en`'s word error rate from 31.7% to 23.4% and raised `whisper-base`'s from 15.4% to
  26.8% (11.7% at strength 0.5); docs/models.md, "Denoising".
- `bolo serve --console-port N` (`console_port`, `BOLO_CONSOLE_PORT`) serves the web console on a port of its own
  while the API keeps `--port`, so the API can be exposed alone; `--console-host` and `--console-api-url` set where
  the console listens and which API address it calls. The API accepts the console's cross-origin requests by
  itself. `--console-port 0` or `--no-console` serves no console. The default is unchanged: both on one port.
- The repetition guard on live transcripts (`WS /api/v1/stt/stream` and the Realtime API): each final is checked
  together with the finals already sent, so a loop that runs on from one utterance into the next is cut. Only the new
  final changes; a final that only repeats earlier ones is not sent (an empty `completed` transcript in the Realtime
  API); partials and Realtime deltas are not trimmed. The model's pinned `max_repeats` applies, and Whisper options
  pinned on a model now reach its live decodes as well.

### Fixed

- `test_gated_entries_explain_what_to_do` failed whenever bolo-torch was installed next to the server: the test's
  live server discovered the plugin through its entry point, so `plugin:torch` was available and the assertion that
  `indic-parler-tts` is unavailable failed. It now runs a server that does not discover installed plugins.

- whisper.cpp: a request's `language`, `prompt` or decoding options stayed set for every later request
  (pywhispercpp keeps parameters between calls). Every request now sets all of them, from the model's defaults where
  it gives none, and calls are serialised. A real-model test (`tests/test_real_whispercpp.py`, opt-in) checks each
  option reaches whisper.cpp and does not carry over.
- The `en-us` and `hi` language packs read "needed by 0 models". That is right (American English and Hindi are part
  of uccharan, so their packs are never needed, and Bhojpuri, Maithili and Magahi use Hindi's built-in data), but it
  read like a broken count: the console now says "built in, not needed · used by N models".
- A model could be unloaded as idle while it was still decoding: when a client disconnected, its request let go of
  the model at once, but the engine went on decoding in its worker thread, and after the idle time (`keep_alive`,
  `stt_ttl_s`, ...) the model was unloaded under it (seen with whisper.cpp, whose decode ran on for minutes). A model
  now counts as in use while its queue has work running or waiting, and its idle time starts when the last decode
  ends; this applies to idle unloading, `keep_alive: 0`, eviction to make room and moving off a lost GPU. A
  disconnected client's decode is stopped where the engine can stop part-way (whisper.cpp through its abort
  callback, faster-whisper at the next 30 s window); elsewhere it runs to the end and its result is discarded.
- Segment and word times are kept within the audio that was decoded: a model without word timings could place a
  segment's end past the end of the recording (whisper.cpp ended one at 40 s on 24 s of audio).

## [0.6.6] - 2026-11-17

### Fixed

- `bolo ps` prints each model's placement notice (for example that DirectML runs Nemotron's Gelu nodes on the CPU,
  or that a provider fell back) without `-v`; the API already returned them.
- The real-model tests read `BOLO_DEVICE` (they always ran on the CPU), so `BOLO_DEVICE=dml:1 pytest ...` runs where it
  says.

## [0.6.5] - 2026-11-17

### Fixed

- Every ONNX Runtime session printed "Config with key [session.x64quantprecision] already exists" to the console:
  the exact-int8 option was set twice (by `session_options()` and again by `create_session`). It is set once.
- **Live transcription: a junk "ular" before "good morning" after a click** (found on a Windows laptop with 0.6.4:
  en-short sent over `/api/v1/stt/stream` with an 80 ms Hann-windowed burst of noise at 0.3 peak, 0.3 s before the
  speech, gave "ular good morning could you tell me when my order will arrive" with `zipformer-en-20m-streaming`).
  Silero reported the start on the speech, not on the click, but the pre-roll lengthened to put the speech 0.05 s
  into an encoder chunk began 0.65 s before the speech and so took the click in, and so did the second stream's
  (0.5 s before the reported start); the chunk-aligned decode made "ular" of it and, having more words before the two
  first agreed, was taken. A burst in the pre-roll (sound above the room with no voiced frames in it, ending 0.1 s or
  more before the speech) is now brought down to the room before either stream hears it, to digital silence after
  digital silence (brought down only to -90 dBFS, it still gave "ul"); a quiet first word and the hiss that starts a
  word are left alone. The laptop's recipe, seeds 1-3, sends of 20, 100 and 250 ms: "good morning" first in 3 of 9
  takes before ("ular good morning" 3, "in could you" 3), 6 of 9 now; the three left are the 250 ms sends, which give
  "in could you" without the click as well. `tools/measure_stream_onset.py`, 252 takes per model, before -> now:
  Zipformer first word right 150 -> 150, stray tokens 0 -> 0, WER 0.100 -> 0.101 (three takes changed: en-long after
  a burst without a room right now, over a room at -60 dBFS "servers are" now; one quiet first word "ul could you" ->
  "tell me"); Moonshine 192 -> 192, 2 -> 2, 0.062 -> 0.062, every transcript the same.
- `bolo ps` and the placement of `nemotron-3-diarization` on DirectML said "100% GPU" with no notice although part
  of the graph runs on the CPU. From ONNX Runtime's kernel lists (OperatorKernels.md and DirectML's operator
  registrations, 1.18 to 1.24.4, the DirectML package's latest), DirectML has kernels for every operator in the int8
  graph but one: its 127 MatMulNBits nodes (8 bits, block 32, no g_idx or bias) pass DirectML's support check, as do
  RotaryEmbedding and SkipLayerNormalization, but Gelu is registered only in the com.microsoft domain, so the graph's
  31 ONNX Gelu nodes (opset 20) run on the CPU between the GPU's parts. The weights are all on the GPU, so the
  placement stays "100% GPU" (it counts bytes) and its notice now says that the Gelu nodes run on the CPU. The same
  diarization to every printed digit on CPU and DirectML does not by itself show that the graph ran on the CPU:
  speaker activity is thresholded frame by frame, and small float differences need not move anything printed.

## [0.6.4] - 2026-11-17

### Added

- Real-model test: `nemotron-3-diarization` on the demo's two-voice dialogue (`two-speakers-en`, the first 20.8 s
  and the whole 27.2 s) wants two speakers, every line with its own speaker and DER at most 12 %, and prints the
  per-speaker peak probability and the ONNX Runtime build. Checked after a report that this clip gave one speaker on
  a Windows laptop: on Linux it gives two speakers with every line right (DER 5.2 % and 5.9 %, no confusion), the
  same with onnxruntime 1.24.4 (the DirectML package's version), 1.31.1 and 1.32.0, with 1 to 16 threads, in the
  three live latency modes, and on an emulated AVX2 CPU without VNNI like the laptop's (probabilities within
  1.2e-6 of native, no decision changed), so no fault was found in the engine's chunking, thresholds or
  post-processing. In DiCoW's
  test, the comment blamed Nemotron's turns for speaker B's stream repeating A's first sentence: the turns are
  right; DiCoW's stream for B does that when the recording starts with 0.17 s that nobody speaks in.

### Changed

- Documentation: the README, status, models, recommendations, architecture, configuration and performance pages no
  longer say that GPUs, Windows and DirectML are untested. They give what was measured on one Windows 11 laptop (CPU,
  CUDA, DirectML, Vulkan, WSL2, Chrome and Edge), collected in docs/status.md, "A Windows 11 laptop", and keep Apple
  Silicon, ROCm and discrete Radeon cards, native Linux GPUs, several NVIDIA GPUs, arm64 and phones as untested.

### Fixed

- DiCoW: when a recording opened with a short silence (the demo's two-voice dialogue starts at 0.17 s), the
  second speaker's text began with the first speaker's sentence ("good morning did you get a chance…"). A first
  speaker turn that starts within 0.5 s now starts at 0 s for the speaker masks.
- **Live transcription: "Good morning" after a burst of noise, and speech 40 dB down** (found on a Windows laptop
  with 0.6.3, the samples sent over `/api/v1/stt/stream` in 100 ms pieces: `zipformer-en-20m-streaming` gave "in
  could you tell me..." for en-short with an 80 ms burst 0.3 s before it, and "ular" at -40 dB, where Moonshine lost
  the start). The burst was not raised, did not reset the stream and did not set the gain (0 dB in every take): Silero
  reported the start on the burst, the fresh stream began half a second before that, and "Good morning" came 0.4 s
  later into it than without the burst. The 20M Zipformer keeps or loses a first word by where in its 0.32 s encoder
  chunks the word starts: "Good morning" 0.05 s into a chunk was kept, 0.2 s in it was "in", which a file with half a
  second of digital silence before the speech also gave; "Speech servers" the other way round. A stream now opens
  once the speech's vowels have arrived (at most 0.5 s after the start the VAD reported), and a model that gives its
  encoder chunk (`stream_chunk_s`, 0.32 for the 20M Zipformer) decodes each utterance twice, from a pre-roll that puts
  the speech 0.05 s into a chunk and from the pre-roll as before, and takes the one with more words before the two
  first agree. At -40 dB the VAD was the cause: Silero reported no start, or one seconds late, for speech near -55
  dBFS; the recognisers' gain cap was not reached first. The VAD (only the VAD) now hears quiet input raised so its
  loudest recent frame is near -30 dBFS, never so far that the room goes above -65 dBFS, so by 25 dB at most after
  digital silence and not at all over a microphone's own noise near -60 dBFS. Over a room at -60 dBFS a clip 40 dB down
  has its speech about 6 dB above the room, and neither model gets anything from it: that is the models' limit.
  `tools/measure_stream_onset.py`, 252 takes per model, 547c42d -> now, first word right, takes with a stray token,
  word error rate:
  - Zipformer: samples as they are 29/36 -> 31/36 (en-short after a burst 0/3 -> 3/3, en-long after a burst 3/3 ->
    1/3); 30 dB down 7/12 -> 10/12; 40 dB down 1/12 -> 5/12 (en-short "" -> "good morning could you..."); quiet first
    word 42/96 -> 52/96; first 300 ms quieter 38/96 -> 52/96; stray tokens 0 -> 0; WER 0.136 -> 0.100.
  - Moonshine: 40 dB down 2/12 -> 6/12; the rest as it was (samples 36/36, quiet first word 68/96, first 300 ms
    quieter 73/96); stray tokens 2 -> 2; WER 0.072 -> 0.062.

## [0.6.3] - 2026-11-17

### Fixed

- Live demo and console, "Your device": a WebGPU adapter request or storage estimate the browser never answered (seen
  on the speech-to-text page while a model was starting) left "GPU in use" on "checking…" for good. The probe now
  gives up after 5 s (3 s for storage) and says the browser did not answer.
- The live demo is rebuilt when bolo's version changes, so its footer shows the released version (it said 0.6.1
  after 0.6.2 was out).
- README: `--refresh` for `uv tool upgrade` right after a release (uv offered the previous version from its cache).
- **Live transcription: no stray letter or token before the first word, and quiet input no worse than before** (found
  on a Windows laptop, the samples sent over `/api/v1/stt/stream` in 100 ms pieces: `zipformer-en-20m-streaming` gave
  "o good morning...", "s speech servers..." and, with a burst of noise before the speech, "s good morning"; Moonshine
  "16. Speech servers..."; at -40 dB, "ular" and "When my order will arise."). The search 0.6.2 added for a quiet
  first word the VAD missed (8ccf736) took any sound above the room for that word and raised it up to 24 dB: a burst
  of noise 0.3 s before the speech, or the hiss that starts "Speech", became a letter or a number. Its room was the
  quietest tenth of the frames that were not digital silence, so after a file's silent lead-in, or in a recording
  quiet throughout, the room was the speech itself. Now a stretch of sound counts only from the first one with 40 ms
  of voiced (periodic) frames in it, digital silence counts as a very quiet room, the margin over the room is 6 dB
  rather than 9, and the gain is judged on the voiced frames of the part raised. Two older causes, from the streaming
  pre-roll and gain (a6104c8, also 0.6.2): the utterance's gain was judged on everything heard after the VAD's start,
  which after a click can be mostly silence (a 24 dB gain that clipped en-short into "ular arrive"); it is now judged
  on the frames that sound. And the 20M Zipformer writes the first sound of a fresh stream twice when it starts
  abruptly (digital silence, then speech): a lone letter other than "a" and "I" at the start of an utterance on a
  VAD-gated stream is dropped. Measured with `server/tools/measure_stream_onset.py` (252 takes per model: the laptop's
  cases as files, as they are and over a -70 and -60 dBFS room, and 8ccf736's simulated microphone takes with four
  sets of room noise), first word right / takes with a stray token / word error rate, a6104c8 -> 8ccf736 (0.6.2) ->
  now. Zipformer: the samples as they are, trimmed and with the burst 20/36, 9, 0.096 -> 20/36, 3, 0.106 -> 29/36, 0,
  0.061; 30 dB down 5/12 -> 6/12 -> 7/12, WER 0.184 -> 0.164 -> 0.163; 40 dB down 1/12 -> 2/12 -> 1/12, WER 0.694 ->
  0.659 -> 0.684; a quiet first word, then a pause 13/96 -> 33/96 -> 42/96; only the first 300 ms quieter 18/96 ->
  32/96 -> 38/96; stray tokens over all takes 24 -> 18 -> 0. Moonshine: the samples 36/36 throughout; 30 dB down 9/12
  throughout; 40 dB down 2/12 throughout (WER 0.688 -> 0.660 -> 0.658); a quiet first word 34/96 -> 63/96 -> 68/96;
  the first 300 ms quieter 42/96 -> 70/96 -> 73/96; stray tokens 2 -> 5 -> 2 (left: "At the early nightfall" for
  "After early nightfall" twice). At 40 dB down Silero hears little or nothing of the speech in any version.

## [0.6.2] - 2026-11-16

### Added

- Docs: "At log-on, restarted after a crash" (docs/configuration.md, Windows). Task Scheduler's "restart on failure"
  settings apply only when the task fails to start, not when bolo exits later; the section gives a PowerShell loop
  started at log-on by a scheduled task, and NSSM/WinSW with `sc.exe failure` recovery settings for a service
  (`sc.exe` alone cannot run bolo as a service).
- Docs: Windows PowerShell 5.1 needs `[Console]::OutputEncoding` and `$OutputEncoding` set to UTF-8 to pipe text out
  of and into `bolo`; cmd and PowerShell 7 need nothing.
- README: validated on Windows 11 natively (RTX 3060 Laptop GPU, CPU-only and CUDA) and under WSL2; on Windows,
  install natively and add the CUDA wheels for an NVIDIA GPU, WSL2 only for Linux-only runtimes such as Tontaube.
- llama-server: the Vulkan build for Linux x86-64 (llama.cpp b12375's `ubuntu-vulkan-x64`) is pinned, so
  `BOLO_DEVICE=vulkan bolo components install llama-server` (or `--flavour vulkan`) installs it on Linux too, as on
  Windows. It needs the Vulkan loader and a Vulkan driver.
- Docs: "A GPU build of llama-server on Linux": the Vulkan build, and building llama-server with CUDA (also under
  WSL2) for `llama_server_bin`.
- `bolo doctor` with `device = "cuda"` (or `auto` and an NVIDIA GPU) loads both CUDA runtimes for real: it opens a
  tiny graph on ONNX Runtime's CUDA provider and runs it, and asks CTranslate2 (faster-whisper) for its CUDA devices
  and loads CUDA 12's cuBLAS and cuDNN, then names each missing library with the command that installs it.
- `cuda` extra: CUDA 12's cuBLAS and cuDNN 9 from NVIDIA's wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`;
  Windows and Linux), for faster-whisper on an NVIDIA GPU. ONNX Runtime's CUDA build stays a manual swap
  (`onnxruntime-gpu` must replace `onnxruntime`, which an extra cannot do). docs/configuration.md has the Windows
  recipe ("CUDA on Windows").
- `bolo components install llama-server --flavour cpu|cuda|vulkan` (also audio.cpp, and `{"flavour": ...}` in
  `POST /api/v1/components/{id}/install`) picks the build; `bolo components list` names the build that runs.
- **A lost GPU is noticed** (`bolo.gpu_health`). While models are on a GPU, bolo asks the NVIDIA driver every
  `gpu_health_interval_s` seconds (30) whether the GPU is still there, and right after a request whose real-time
  factor is more than twice that model's recent median: NVML's `NVML_ERROR_GPU_IS_LOST` (or no GPU), else
  nvidia-smi's "GPU is lost" / "Unable to determine the device handle" / "No devices were found". A CUDA or Vulkan
  device error printed by a llama-server or audio.cpp bolo started for a GPU (`CUDA error`, `cudaError...`,
  `ggml_cuda_...: ... failed`, `ErrorDeviceLost`; not `out of memory`) counts the same. Then the models on the GPU
  are marked degraded and, with the new `gpu_lost_fallback = "cpu"` (the default), loaded again on the CPU (managed
  servers on their CPU build, or with `--device none` and `CUDA_VISIBLE_DEVICES=-1`), and later loads go there
  too; `"fail"` fails their requests instead. The notice ("GPU LOST (...): reboot to recover; bolo is using the
  CPU") is in the log, `bolo ps` (`GPU LOST` in its first line, as the processor of a model still on the lost GPU,
  and the models moved), `GET /api/v1/ps` and `/api/v1/stats` (`system.gpu_health`, `degraded` per model),
  `GET /healthz` (still 200, `"status": "degraded"`), `GET /api/v1/info` (`health`), the console's Loaded models
  panel and `bolo doctor` (`gpu health:`). Found on an RTX 3060 Laptop GPU (Windows 11, driver 621.24): during a
  CUDA llama-server run, requests slowed from a real-time factor of 11.8 to 29.4 (the CPU build: 4.69) until
  nvidia-smi said "GPU is lost. Reboot the system to recover this GPU", while bolo went on serving and `bolo ps`
  said `cuda:0`. docs/configuration.md, "A lost GPU", has the Windows laptop notes.

### Changed

- Live demo: **Auto runs the small speech recognisers on the CPU** (WebAssembly, multi-threaded when the page is
  cross-origin isolated) instead of WebGPU: Moonshine Tiny and Base, Whisper Tiny and Base and the other
  recognisers up to 100 M parameters. Measured in Chrome and Edge on a Windows laptop (Ryzen 7 5800H, RTX 3060
  Laptop GPU; 2.7 s of English; real-time factor on WebAssembly vs WebGPU): Moonshine Tiny 0.097 vs 0.94, Moonshine
  Base 0.17 vs 1.55 (loading 1.4 s vs 11.4 s), Whisper Base (English) 0.99 vs 1.92 (1.9 s vs 12.3 s), with the same
  transcripts; with the RTX 3060 forced, Moonshine Tiny on WebGPU was still 0.87 to 0.90. The picker says why
  ("a small recogniser: faster on the CPU than on WebGPU"), and that the CPU has one thread when the page is not
  cross-origin isolated. WebGPU stays selectable by hand, larger and 16-bit models still start there, and once a
  model has run on two placements Auto follows the faster one measured on that device.
- Live demo: the Models screen has the downloaded-models manager (size, last use, delete, storage limit), as the
  screens that use the models do; `downloads-panel` was not on `#/models` at all.

### Fixed

- `bolo doctor`'s CUDA section names the ONNX Runtime package that is installed when it has no CUDA provider
  (`onnxruntime-directml 1.24.4 (the DirectML build)`, with `pip uninstall -y onnxruntime-directml`), instead of
  calling every such build "the CPU build" and telling it to uninstall `onnxruntime`.
- A lost CUDA provider is said out loud: with `device = "cuda"`, or `auto` on a machine with an NVIDIA GPU, an ONNX
  Runtime without a CUDA provider (`uv tool upgrade` and `uv tool install --force` put plain `onnxruntime` back over
  `onnxruntime-gpu`) gets one warning line in the server's log at start and a `WARNING:` line in `bolo doctor`, each
  naming the package and the fix. docs/configuration.md, "CUDA on Windows", has the upgrade path
  (`uv tool install ... --with "onnxruntime-gpu[cuda,cudnn]" --excludes <file naming onnxruntime>`, which
  `uv tool upgrade` keeps).
- `bolo logs -f` no longer skips lines when the log is copied and emptied (the rotation's fallback on Windows when
  another program blocks the rename): the emptied file is told by its size or by its first line having changed,
  even when it has already grown past the position read; the lines written before the copy are read from
  `server.log.1`, then the new file from its start. On Windows 328 of 4200 lines had been skipped.
- `GET /api/v1/hardware?refresh=1` no longer runs the slow probes every time (about 3 s per call on a Windows
  laptop): the answers of nvidia-smi, the server binaries' `--version` and `--list-devices`, the torch plugin's
  `doctor` and CTranslate2's CUDA device count are reused for 60 s, and forgotten when a component is installed;
  `refresh=force` runs them again at once.
- `bolo components install tontaube` said "unknown component" (the command's own list left it out, on every
  platform). On Windows and macOS `components list` now shows Tontaube as unavailable with the reason, and installing
  it says it is Linux only (on Windows: "use WSL2"); the API's `components` entries have `available` and
  `unavailable_reason`.
- audio.cpp models: a voice that is neither the catalog's nor one the server lists is refused with 400 and the
  voices there are, instead of being passed on (audio.cpp takes it for a file path and answered 500).
- `GET /v1/audio/voices?model=<id>` lists that model's voices only (it listed every model's); an unknown model is 404.
- Removing a model (`bolo rm`, `DELETE /api/v1/models/{id}`) waits out a Windows sharing violation (WinError 32 or
  5: an antivirus scanner holding a file that was just written) with a few growing pauses (about 3 s in all). When
  the folder is still in use after that, the error (409 `files_in_use`) names the folder and the file instead of a
  traceback; the manifest goes first, so a folder only partly deleted is not taken for an installed model.
- The cause of that WinError 32 on the laptop (3 of 3 times right after a load failed with ComponentMissing, never
  after one that loaded): the request had started the hardware advisor's background refresh, which reads every
  installed model's `manifest.json`, and the removal that followed at once met the file open in that thread.
  Manifest reads and a model's removal now take turns (a lock in the store), and on Windows a manifest is read
  with `FILE_SHARE_DELETE`, so a read never makes a deletion fail.
- tools/validate_models.py: the CPU's name comes from bolo.hardware (on Windows it read `/proc/cpuinfo`), and a GPU
  that is there is named; without the `--en-voice` model it takes the first installed English voice, and with none
  the models that need test speech fail with a reason instead of a `KeyError: 'en'`; diarization and end-of-turn
  models get checks of their own instead of the VAD check; a model whose removal fails (WinError 32 on Windows) is
  recorded with `cleanup_error` and the run goes on.
- Saved voices: the "sounds noisy" warning no longer fires on clean synthetic speech whose DC offset comes and goes
  with the voice (bolo's own reference clips among them): the offset is removed with a 0.1 s moving average instead
  of the clip's mean, which had turned digital silence into a background 27 dB under the speech. Noise 10 dB under
  the speech still earns the warning.
- Live demo: **Chatterbox Turbo would not load on the CPU**, neither chosen by hand nor as the retry after a WebGPU
  failure: "Could not find an implementation for GatherBlockQuantized(1) node with name '/text_emb/Gather_Quant'"
  (Chrome and Edge 157 on Windows). transformers.js points every device at ONNX Runtime Web's asyncify build (the
  one WebGPU needs), whose CPU kernels lack `GatherBlockQuantized`; every 4- and 8-bit Turbo graph uses it (q4,
  q4f16 and quantized alike; fp16 and fp32 do not). The plain WebAssembly build has it: with onnxruntime-web
  1.31.0-dev.20260914 all four q4 graphs open on the plain build and none on the asyncify one. The cloning worker
  now picks the build per device (plain on the CPU), the CPU always gets the CPU's files (q4, never the WebGPU-only
  q4f16 graph), and a model that fails to load on WebGPU is loaded again in a new worker on the CPU build (a worker
  cannot change builds). In headless Chromium, Turbo then loads on the CPU in 21 s and speaks. The clone page states
  both downloads before anything is fetched (725 MB on the CPU, 565 MB on WebGPU, different files), and the note
  for the CPU retry says its files are a separate 725 MB download. Unit tests for the dtype and build per device.
- Live demo, Your device: "GPU in use" is always listed (`data-testid="gpu-in-use"`): "checking…" until the WebGPU
  probe answers, then the adapter, or "none" without one; it used to appear only once an adapter was found. The
  panel (`data-testid="device-panel"`) is also on the Clone page now, besides Speech to text and Text to speech.
- **Live transcription from a microphone still lost the first word when it was quiet** (Edge and Chrome on Windows:
  `zipformer-en-20m-streaming` gave "t can you tell me...", Moonshine "Can you tell me..." for a sentence with a word
  before "Can"). Silero reports a start once it has heard a quarter second of confident speech; a quiet first word
  (a browser's gain control and noise suppression still settling, or a soft voice), or one followed by a breath, is
  not that, and the start landed on the next word, 0.4-0.8 s late in simulated takes. The streaming pre-roll (0.5 s)
  and the pseudo-streaming pad (0.25 s) were both counted from that start, so the recogniser got the end of the word
  or none of it. Nothing else dropped audio: the console holds the microphone's 20 ms blocks until the stream is
  ready and sends them all, the server queues them, its resampler and the gain keep every sample. Now the speech
  before a reported start is looked for in the audio held back (frames 9 dB above the room, through pauses shorter
  than 0.4 s, at most 1.2 s back); the pre-roll and pad are counted from where it began, and that part is raised to
  the level of the speech the VAD heard (one gain, at most 24 dB). Simulated microphone takes (real and synthetic
  speech at -26 to -36 dBFS, the first word 14-24 dB quieter and then a pause, 0.4-1.5 s of room before, 100 and
  250 ms sends, some after a burst of held audio, the console's resampler), first word right in N of 24, two sets of
  room noise: Zipformer 1 and 2 before, 14 and 9 after; Moonshine 3 and 4 before, 18 and 19 after. Takes whose first
  300 ms only are quieter (no pause after it): Zipformer 12 before, 13 after; Moonshine 20 before, 21 after.
- **The end-of-turn model (`smart-turn-v3`, `semantic_vad`) gave wrong probabilities on x86 CPUs without VNNI**
  (AVX2 only: AMD Ryzen before Zen 4, Intel before Ice Lake / Alder Lake), the same ONNX Runtime kernel that emptied
  DiCoW's transcripts: its int8 graph (QuantizeLinear / DequantizeLinear pairs, which ONNX Runtime fuses into uint8 x
  int8 kernels) saturates there. Under emulation of such a CPU, on 34 windows of the bundled samples, the
  probability moved by up to 0.57 (mean 0.12) and 4 of the 34 end-of-turn decisions at 0.5 flipped; with the fix it
  matches the native result exactly. Every ONNX Runtime session bolo opens itself (Nemotron diarization, DiCoW, the
  SNAC decoder, smart-turn, in-process LFM3-Audio and the halves of split graphs) now sets
  `session.x64quantprecision` through `bolo.ort_providers` (`session_options`, and `create_session` for options made
  elsewhere). The other graphs were checked and were not affected (SNAC, the LFM3-Audio encoder, codec and depth
  decoder are float; Nemotron and the LFM3-Audio decoder use MatMulNBits, which the option does not touch: under
  emulation Nemotron's probabilities matched to 7e-7 and LFM3-Audio's greedy text was the same); on a CPU with VNNI
  every output is identical with the option. sherpa-onnx brings its own ONNX Runtime and takes no
  session option from bolo; docs/measurements/int8-without-vnni.json records what its int8 models do on such CPUs.
  New unit test: every engine's session carries the option.
- A request running when its model was unloaded (bolo stop, keep_alive, or a model moved off a lost GPU) could wait
  forever: closing the model's queue failed the queued requests but not the running one. The queue now lets running
  requests finish for up to 10 s, then fails them with a clear error; a model moved off a lost GPU waits for the
  requests using it (up to 30 s) before it is unloaded.
- `bolo doctor` names the GPUs llama-server's `--list-devices` found (`GPU build: Vulkan0: ...`), and a CUDA or
  Vulkan build that found none says so instead of reading "CPU build".
- `bolo components install llama-server` / `audio.cpp` with a GPU `device` that has no prebuilt build on this
  platform (CUDA on Linux, where neither project publishes one) installed the CPU build and reported a plain install.
  It now says that the CPU build is what runs, and names the way to the GPU (the Vulkan build, or a build of your own
  in `llama_server_bin` / `audiocpp_server_bin`): on the command line, in the log and as a warning on the job.
- **The end-of-turn model (`smart-turn-v3`) failed on DirectML**, on every adapter (ONNX Runtime: Reshape node
  `node_view_1`, `MLOperatorAuthorImpl.cpp(2597)`, 8007023E), and the error ended realtime sessions with
  `semantic_vad`. With `device = "dml"` it is now placed on the CPU (8 MB, 13-39 ms there), with a notice saying why;
  its placement and `bolo ps` say `cpu`. An accelerated end-of-turn session that fails while running is replaced by
  a CPU session once, the request is answered from it, and the model's placement moves to the CPU; any other error
  of the end-of-turn model makes a session fall back to ending turns on silence instead of failing. Found on a
  Windows 11 laptop (Radeon iGPU and RTX 3060 Laptop GPU, onnxruntime-directml 1.24.4).
- **`device = "auto"` with onnxruntime-directml ran the ONNX Runtime engines on the CPU** on a machine with an
  NVIDIA GPU: `auto` meant `cuda:0`, which onnxruntime-directml has no provider for, and the warning advised
  installing onnxruntime-gpu. For the ONNX Runtime engines `auto` now means the first provider the installed
  `onnxruntime` has (CUDA with an NVIDIA GPU, ROCm, DirectML on the high-performance adapter, Core ML; else the CPU);
  faster-whisper keeps CUDA. Asked for `cuda`, onnxruntime-directml now says that `dml` (or `auto`) uses the GPU, and
  how to swap it for onnxruntime-gpu; `bolo doctor` says the same.
- **sherpa-onnx models were reported on the GPU while they ran on the CPU** (`whisper-base` with `device = "cuda"`:
  "100% GPU", `cuda:0`, while sherpa-onnx said "Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON ... Fallback to
  cpu!"). bolo now tells a CUDA build of sherpa-onnx from a CPU build (ONNX Runtime's CUDA provider library in its
  package's `lib` folder, or a `+cuda` version) and places a CPU build's models on the CPU, with one notice naming
  why; a build it cannot tell apart that falls back at load time is recorded like ONNX Runtime's provider fallbacks
  and moved to the CPU. `bolo doctor`, `bolo recommend` and `GET /api/v1/hardware` (`runtimes.sherpa-onnx.gpu`) say
  whether the installed sherpa-onnx is a CPU or a GPU build.
- **Live transcription from a microphone with a streaming model (`zipformer-en-20m-streaming`) gave no text, or lost
  its first words** ("Speech servers" came out as "ers"). Seen in the console in Edge and Chrome on Windows. The
  recogniser heard the room before anyone spoke, and its own endpointing reset the stream whenever it had decoded
  nothing for 2.4 s, often mid-word; quiet speech (browsers' automatic gain control leaves it around -30 dBFS) was
  decoded late enough for that to happen on every take. With a VAD, a streaming model now hears speech only: the idle
  audio is held back, and when the VAD hears speech start a fresh stream gets the 0.5 s before it (the pre-roll) and
  then the live audio until the VAD hears it end, which ends the utterance (the model's own endpointing no longer
  does). Quiet speech is raised to about -20 dBFS for the recogniser, one gain per utterance (the VAD still hears it
  as it arrived). The 20M Zipformer decodes with beam search (`modified_beam_search`, four paths), which keeps it from
  dropping whole phrases where greedy search committed to blanks. A cloud recogniser's live connection, and every
  session without a VAD, still hear everything. Simulated microphone takes (quiet speech after a pause, 20 ms frames,
  the console's resampler) of six recordings at four levels and pauses: word error rate 0.37 before, 0.10 after;
  the first word right in 3 of 24 takes before, 13 after (synthetic voices that start abruptly still lose it
  often); two takes gave no text and one only "ul america" before, none after. A restarted stream starts on a second
  of digital silence before the pre-roll. Model options `stream_preroll_s` (0.5; 0 turns this off) and
  `stream_speech_dbfs` (-20; empty leaves the level alone) for sherpa-onnx streaming models.
- Pseudo-streaming partials (Moonshine, Whisper, …) no longer repeat a word ("finished finished") or drop one when a
  re-decode splits, merges or inserts a word before the end of the confirmed part: each new hypothesis is aligned with
  the confirmed words (word-level edit distance) before its new tail is taken, instead of counting words off. Two
  utterances close together no longer share audio either, so a word on the seam is not in both finals.
Found on a Windows 11 laptop; the first fix applies everywhere.

- `bolo rm` and `bolo cp` with a data directory (`--data-dir`, `BOLO_DATA_DIR`, the config file) other than the one
  of the server at `--url` (`BOLO_URL`, default `http://127.0.0.1:8000`) asked that server to do it: the server
  deleted (or copied) the model in its own store, the command printed "removed X", and the files in the CLI's data
  directory stayed. They now go through the server only when its `data_dir` (`GET /api/v1/info`) is the CLI's
  own, and otherwise change the CLI's data directory and say which (`note: the server at ... uses the data
  directory A; this command changes B`). `bolo pull`, `show` and `voices` already act on the CLI's data directory.
- A configuration file with a Windows path in double quotes (`data_dir = "C:\Users\..."`) stopped `bolo serve`
  with only `error: Invalid hex value (at line 1, column 17)`. A TOML error now names the file and the line, and an
  escape problem says to write the path in single quotes (`'C:\Users\...'`, a literal string) or with forward
  slashes. A path that parsed into a control character (`"C:\temp"` holds a tab) is refused with the same advice
  instead of being used, and a file saved with a byte-order mark (Notepad) is read.
- `bolo doctor`'s "logs:" line spoke of journalctl and `docker logs` on every platform. On Windows it now gives the
  log file (`%LOCALAPPDATA%\bolo\logs\server.log` with `log_file = "auto"`), `bolo logs -f` and the Task
  Scheduler, NSSM and WinSW recipes in deploy/README.md; on macOS, launchd and `brew services`.
- `bolo show` of a Piper voice said "phonemes come from espeak-ng" even where espeak-ng is absent and uccharan
  phonemises it. The catalog's Piper descriptions no longer name one phonemiser, and `bolo show` has a "phonemes"
  line with the one this installation uses for the voice's language (installed components and a pinned
  `phonemizer` considered), and the others that are available.
- A micro-benchmark stored before bolo read the CPU's name from the Windows registry was silently ignored (it had
  recorded Windows's processor identifier as the name), and `bolo doctor` fell back to "estimated". A benchmark
  now also records the processor identifier and matches on it, so such a benchmark counts again; one that is not
  used (another CPU, other cores, another version) is named in the speed line with `run bolo recommend
  --benchmark again`, instead of being dropped without a word.
- Install commands in the README, the docs and `bolo components` (`manual_command`) use double quotes
  (`uv tool install "bolo-speech[all]"`), which work in bash, zsh, PowerShell and cmd; with single quotes cmd
  failed with "Failed to parse: `'bolo-speech[all]'`". The Windows docs also say that the firewall prompt names
  `python.exe` (uv's interpreter), not `bolo.exe`, that Windows PowerShell 5.1 needs
  `[Console]::OutputEncoding = [Text.Encoding]::UTF8` to pipe UTF-8 output, and that 5.1 drops the double quotes
  inside a JSON argument to `curl.exe`.
- **Windows: every model on an NVIDIA GPU read as "100% CPU"** in the log, `bolo ps` and `/api/v1/ps` (seen on an
  RTX 3060 Laptop GPU, Windows 11, driver 621.24). Under the WDDM driver model the driver keeps no per-process GPU
  memory: NVML reports `NVML_VALUE_NOT_AVAILABLE` (or 0) and nvidia-smi "N/A", which bolo counted as 0, so each load
  measured 0 bytes on the GPU. Such figures are now unknown, never 0 (`process_bytes` null, the GPU's new
  `driver_model` is "WDDM"). A load is then measured by what DXGI says bolo's own process uses on that GPU (the
  adapter with the GPU's LUID; runtimes in bolo's process), else by the growth of the GPU's used memory, with such
  loads made one at a time; when neither grows the memory is unknown. A model placed on the GPU always shows as on
  it: `processor` follows the placement, and when the memory is unknown `vram_measured` is false, the new
  `vram_source` is "plan", `bolo ps` prints `100% GPU ?` with an "unknown VRAM" note and the console's Loaded
  models panel says "VRAM unknown". The same holds for Apple Silicon's unified memory (`vram_source` "unified").
- `mypy` failed with `nvidia-ml-py` installed (`pynvml` has no type information).
- **A model placed on an NVIDIA GPU on Windows ran on the CPU without a word.** Found on an RTX 3060 Laptop GPU
  (Windows 11, driver 621.24): `onnxruntime-gpu` 1.32.0 is built for CUDA 13 and cuDNN 9, and its CUDA provider did
  not start (`... depends on "cublasLt64_13.dll" which is missing (Error 126)`), even with
  `onnxruntime-gpu[cuda,cudnn]` installed, because nothing put NVIDIA's wheel folders (`site-packages\nvidia\...\bin`)
  on the DLL search path; ONNX Runtime then ran the session on the CPU while placement and `bolo ps` said `cuda:0`.
  Before its first CUDA session bolo now adds those folders (`os.add_dll_directory` and `PATH`) and calls
  `onnxruntime.preload_dlls()` (Windows and Linux). After each session it compares the providers it got with the
  ones it asked for (CUDA, DirectML, Core ML, ...) in all five engines that open ONNX Runtime sessions themselves;
  a provider that did not start is logged with the library ONNX Runtime said was missing, becomes the model's notice
  (log, "loaded" event, `bolo ps`), and moves the model's placement to the CPU (`device`, `processor`, and a new
  `provider_fallback` field in the placement).
- **faster-whisper on CUDA failed every request on Windows** (`Library cublas64_12.dll is not found or cannot be
  loaded`: CTranslate2 4.9 needs CUDA 12). bolo now adds the folders of `nvidia-cublas-cu12` and
  `nvidia-cudnn-cu12` before loading (on Linux it loads those libraries from NVIDIA's wheels when they are not on
  the library path), warms the model up on the GPU, and when a CUDA library is still missing loads it on the CPU
  with the same notice instead of failing requests.
- `bolo components install llama-server` put the CPU, CUDA and Vulkan builds in one folder, so after the CPU build
  `BOLO_DEVICE=cuda bolo components install llama-server` said "already installed" and kept it. Each build now has
  its own folder (`components/llama-server/<version>/cpu|cuda|vulkan`; an install from before moves into the
  folder of the build its libraries show), installing checks the build `device` asks for, and bolo starts the build
  matching `device` (with `auto`, a GPU build when one is installed). The same for audio.cpp.
- `bolo recommend` no longer suggests sherpa-onnx models for the GPU on Windows (its CUDA wheel bundles its own
  `onnxruntime.dll`, which fails with Error 1114 once `onnxruntime` is loaded, and the catalog's int8 sherpa models
  gain nothing on CUDA); faster-whisper on CUDA, measured 6-19x faster than on the CPU on that laptop, is.
- **`dicow-v3.3-onnx` returned empty text for every speaker on x86 CPUs without VNNI** (AVX2 only: AMD Ryzen
  before Zen 4, such as the Ryzen 7 5800H of the Windows test laptop, and Intel before Ice Lake / Alder Lake), on
  Windows and Linux alike. The int8 graphs multiply uint8 activations by int8 weights, and ONNX Runtime's kernel
  for that on such CPUs adds pairs of products in 16 bits, which overflow (up to 2 x 255 x 127); the decoder then
  produced no text tokens at all, up to its 448-token limit (checked under emulation of such a CPU). The machine the export was measured on has AVX-512 VNNI, where this does not
  happen. Both DiCoW sessions now set `session.x64quantprecision`, so ONNX Runtime turns the weights into uint8 at
  load time on those CPUs (no overflow); elsewhere nothing changes, the output on the measurement machine is
  identical. New unit tests (the int8 product is exact, the option reaches both sessions) and a real-model test
  (diarized transcription of the demo's two-voice dialogue, text for each speaker); docs/measurements/dicow-onnx.json
  (`cpu_without_vnni`).
- **The Vulkan build of llama-server hung on a laptop with AMD and NVIDIA graphics** (Windows 11, AMD Radeon
  integrated graphics and an RTX 3060 Laptop GPU): `llama-server --list-devices` never answered and Windows logged
  `RADAR_PRE_LEAK_64` for `llama-server.exe`, because of AMD's switchable-graphics Vulkan layer. When bolo starts
  a Vulkan build of llama-server or audio.cpp on Windows and DXGI lists an AMD adapter, it now sets
  `DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1=1` in the server's environment (unless the variable is set in bolo's).
  `bolo doctor`'s `--list-devices` question uses the same environment, gives up after 20 s with a warning that
  says so (it used to print "GPU support unknown" without a word), and a managed Vulkan server that never becomes
  ready names the variable in its error.
- **Live demo: Chatterbox Turbo on the clone page failed silently on WebGPU** (Chrome and Edge on Windows: "ready
  on WebGPU", then Generate produced nothing for more than six minutes, while the console showed `failed to call
  OrtRun() ... webgpu/buffer_manager.cc:643 BufferManager::Download` from the decoder). Worker and ONNX Runtime
  errors now reach the page: a run failure that transformers.js only logs fails the request at once, an error the
  worker cannot catch fails whatever was waiting on it, and a stall watchdog (no sign of life for 60 s on WebGPU,
  180 s on the CPU; the language model reports each step) ends a generation that never comes back. After a failure
  on WebGPU the model is loaded again on the CPU (WebAssembly) and the clip made once more, with a visible note
  ("Generation failed on WebGPU: ... Made on the CPU instead"); the worker is replaced, because transformers.js
  keeps a failed run at the head of its queue and every later run in that worker fails too. Auto then keeps that
  model on the CPU in this browser. If the CPU fails as well, the page shows both errors. Stop lets go of a worker
  that does not answer within 5 s. The in-page recogniser also fails a pending transcription when its worker
  crashes after loading, instead of waiting forever.
- **Live demo: "Your device" and the GPU request on Windows.** 0.6.1 asked WebGPU for
  `powerPreference: "high-performance"` and assumed that picked the discrete GPU; Chrome and Edge on Windows ignore
  the option (the console says so; https://crbug.com/369219127) and Windows decides which GPU the browser gets. The
  request stays, but "Your device" now says which adapter WebGPU actually got ("GPU in use": vendor, architecture
  and description from `adapter.info`), that the request is only a hint those browsers ignore, and, when it is an
  integrated GPU on Windows: "Windows picks the GPU for the browser: Settings > System > Display > Graphics >
  (browser) > High performance". An AMD `gcn-5` adapter with no description (what Chrome reports for a Ryzen
  processor's Radeon graphics by default) counts as most likely integrated.
- Live demo: the footer said "bolo 0.1.0 demo" (the version of the server the catalog snapshot was exported from).
  The build now takes bolo's version from the server package (`server/src/bolo/__init__.py`, the version hatch
  builds from), so it reads "bolo 0.6.1 demo" and follows each release.
- Live demo, memory across reloads (private memory of the browser profile grew by about 10 MB per reload in
  Chrome on Windows). In the page: the model workers are now terminated on `pagehide` (not when the page goes into
  the back/forward cache), so their WebAssembly memory and WebGPU devices are handed back before the next page
  loads; the recogniser's worker no longer leaves the cached ONNX Runtime binary's response body (14 to 27 MB)
  unread when it only checks that the file is cached; the clone page revokes its last clip's object URL when it is
  left, not only when the next clip replaces it. Nothing else in the page outlives a reload (listeners, timers and
  IndexedDB connections belong to the page); growth that remains is the browser's own (code and shader caches,
  the service worker process) and was not chased.

## [0.6.1] - 2026-11-16

### Fixed

- A preset made with `bolo cp` (or `POST /api/v1/models/copy`) could come back with empty settings on Windows, and
  any settings change could read back stale there: the per-model settings (model-settings.json) were cached by the
  file's mtime, which Windows takes from a coarse clock, so a write in the same tick as the read before it kept the
  old copy. Our own writes now refresh the cache, and the check also compares the file's size and id.
- **Hybrid laptops (integrated graphics and a discrete GPU) used the integrated GPU.** Seen on a Windows laptop with
  a Ryzen 7 5800H (Radeon graphics, 0.5 GB) and an RTX 3060 Laptop GPU (6 GB):
  - The live demo asked WebGPU for an adapter without options, and Chrome and Edge then give the power-saving one:
    the "Your device" panel, the verdicts and the models themselves all ran on the Radeon. Every adapter request and
    every WebGPU session (ONNX Runtime Web in the voice and voice-cloning workers, transformers.js in speech
    recognition, the language model and Chatterbox) now asks for `powerPreference: "high-performance"` through one
    helper. "Your device" names the adapter (vendor, architecture, description), says when the browser still gave
    the integrated GPU, with the fix (Windows: Settings > System > Display > Graphics, the browser, High
    performance; or the NVIDIA Control Panel), and lets the visitor choose power saving instead (kept per browser).
  - The server: `device = "dml"` ran on DXGI's adapter 0, which on such a laptop is the integrated GPU. It now runs
    on the adapter Windows calls high-performance (`IDXGIFactory6::EnumAdapterByGpuPreference`, else the one with
    the most dedicated memory), and placement and `bolo ps` read that adapter's memory; `dml:N` still pins one. The
    NVIDIA card is listed once, as CUDA with its DirectML number. Integrated graphics (Intel UHD and Iris, AMD APU
    Radeon, 512 MB or less) never count as the best GPU, so `bolo doctor` and `bolo recommend` point at
    `device = "cuda"` there, and warn when `dml:N` pins the integrated GPU.

## [0.6.0] - 2026-11-16

Voice cloning and Windows. Saved voices with a consent statement, speaker similarity, OpenAI's custom-voice routes
and `bolo voices`; cloning measured on CPU for 14 models (llama.cpp and audio.cpp), and the Tontaube server as a new
runtime. Windows is supported: CI runs the server and plugin tests on `windows-latest`, the llama.cpp and audio.cpp
release builds install there, and servers bolo starts end with it. GPU paths on Windows (CUDA, DirectML, Vulkan)
are untested on real hardware. The live demo gains a Clone a voice page (Pocket TTS, MOSS-TTS-Nano, Chatterbox).
uccharan 0.6.4 is the new minimum.

### Added

- **Windows.** CI runs the server's test suite, the bolo-torch plugin's fast tests, `bolo doctor` and `bolo serve`
  on `windows-latest` (CPU, Python 3.12), and the package declares `Operating System :: Microsoft :: Windows`. GPU
  paths on Windows are untested. What differs there: docs/configuration.md, "Windows"; running bolo at log-on or as
  a service (Task Scheduler, NSSM, WinSW, the firewall): deploy/README.md.
- `bolo components install llama-server` and `audio.cpp` on Windows x64: the pinned releases (llama.cpp b12375,
  audio.cpp v0.9.4) now include the Windows builds, the CPU build by default, the CUDA 12.4 build with its CUDA
  runtime archive for `device = "cuda"` and the Vulkan build for `device = "vulkan"` (llama.cpp also Windows on ARM),
  each checked against its SHA-256.
- Servers bolo starts are tied to its lifetime (bolo.procs): on Windows each llama-server, audio.cpp and plugin
  worker runs in a Job Object that ends it (and whatever it started) when bolo ends, however it ends; on Linux the
  child asks for `SIGTERM` when bolo dies (`PR_SET_PDEATHSIG`, also after `kill -9`); everywhere the children still
  running are stopped at exit and on `SIGHUP`. Stopping one asks first (`SIGTERM`; `CTRL_BREAK_EVENT` on Windows)
  and kills it 10 s later. On Windows `plugin_nice` lowers the worker's priority class and `plugin_memory_limit_mb`
  limits its committed memory through the Job Object.
- Hardware detection on Windows: the CPU's name and instruction sets (SSE4.2, AVX, AVX2, AVX-512F; NEON, dot
  product and SVE on ARM64) and the graphics adapters DXGI lists (name, dedicated memory, `dml:N`); with
  `device = "dml"`, free GPU memory for placement and `bolo ps` from DXGI's per-process budget.
- `bolo doctor` shows the installed ONNX Runtime packages and providers, and warns when several ONNX Runtime packages
  share the one `onnxruntime` module or when `device` asks for a provider (DirectML, CUDA) the installed one lacks.
- PowerShell and cmd variants of the install and curl commands in the README.
- **Saved voices (a voice library) for the models that clone voices** ([docs/voices.md](docs/voices.md)).
  `POST /api/v1/voices` (multipart: recordings, name, transcript, language, description, default model) stores a
  cleaned reference (mono 24 kHz, silence trimmed, pauses shortened, loudness normalised, 2 to 60 s) and warns when
  it is short, noisy, clipped or sounds like more than one person; list, get, change, delete, the reference audio,
  export and import (.zip). `voice="priya"` or OpenAI's `voice={"id": "priya"}` in `/v1/audio/speech` (and every
  other place a voice is named) speaks from it with any model that clones voices; a model that cannot answers 400
  naming the installed ones that can; a voice's default model answers when the request names none. OpenAI's
  custom-voice routes as in openai-python 3.x: `POST /v1/audio/voices`, `DELETE /v1/audio/voices/{id}` and
  `/v1/audio/voice_consents`. `bolo voices list|add|show|rm|export|import|score`.
- **Consent and labelling.** Creating a voice requires a consent statement (`self`: "this is my voice", or
  `permission`: "I have the speaker's permission"; refused otherwise with `400 consent_required`), stored with the
  time and who made it. Cloned speech carries the headers `x-bolo-cloned: true` and `x-bolo-voice`, and the trace
  records the voice and its consent. A short "Voice cloning responsibly" section in docs/voices.md; no bolo-side
  watermark yet (AudioSeal needs PyTorch; documented as a follow-up).
- **Speaker similarity** (`bolo.similarity`): the cosine similarity of the speaker embeddings (the diarization
  model's ERes2Net) of a reference and a clone, with thresholds calibrated on LibriSpeech dev-clean (same person:
  0.45 to 0.93, median 0.80; different people: median 0.11, 95th percentile 0.36; "same speaker" from 0.50,
  docs/measurements/speaker-similarity-calibration.json). Shown when a voice is created with a score model, by
  `GET /api/v1/voices/{id}/score`, `bolo voices score`, and as `speaker_similarity` in `bolo bench` / `POST
  /api/v1/bench` for models that clone voices. Setting `voice_similarity_model`.
- **Cloning measured on CPU** (docs/voices.md, "Measured: cloning on CPU"; `server/tools/measure_cloning.py`): on
  llama-server, Pocket TTS in its six languages, Chatterbox Turbo and Qwen3-TTS 1.7B Base; on audio.cpp v0.9.4 (built
  from source), Pocket TTS, MOSS-TTS Nano, XTTS v2, Plapre 3, OmniVoice (ran, did not clone) and Qwen3-TTS 0.6B
  Base. Those 14 entries are now `validated: cpu` with their speaker similarity.
- **Tontaube** (`tontaube-v1.1-int4`, alias `tontaube`, and `tontaube-v1.1`): a new tts-server runtime `tontaube`
  for the Tontaube inference server (craitech/tontaube v1.1.0), operator-run (`tts_servers`) or started by bolo
  (`bolo components install tontaube` clones the pinned tag; `uv run --python 3.12 --frozen tontaube serve` with
  the downloaded weights). Saved voices and `reference_audio` go to `voice_audio_b64`, its bundled voices to
  `voice_path`, and the new request field `options` carries its style (audiobook, conversational, agentic) and its
  five sliders. Licence: Tontaube Community Model License 1.0 (a speech-generation service always needs a
  commercial licence), labelled in the catalog. Not run here. Settings `tontaube_dir`, `tontaube_start_timeout_s`.
- `/v1/audio/speech` takes `options`, model-specific request options (audio.cpp: passed through to the server);
  a model answers 400 for an option it does not read.
- Console: a "Saved voices" panel on Text to speech (play the reference, use a voice in the playground, score it
  with the current model, save a new one from an upload or a recording with a consent checkbox, delete, export),
  saved voices in the voice picker of cloning models, and "Save as a voice" in the clone panel.
- **Brand**: the tagline under "Ollama for speech." is now "Speech-to-text, text-to-speech and voice agents on your
  own machine." (banners, social preview, README, the console's share descriptions).
- **Live demo: Clone a voice** (<https://cb-sigma.github.io/bolo/>, everything in the visitor's browser, nothing
  uploaded; tested in headless Chromium on a CPU). Record (after a consent tick) or upload 5 to 30 s of speech, or
  pick one of three synthetic example voices; the recogniser in the page fills in the transcript; the voice is spoken
  back with streaming playback, time to first audio, real-time factor, a WAV download and a speaker-similarity score
  (CAM++). Voices are kept in the browser (IndexedDB). Models, from the Hugging Face Hub at pinned revisions:
  - **Pocket TTS 100M** (Kyutai, CC BY 4.0; 8-bit ONNX) in English, German, Spanish, Italian, Portuguese and French,
    driven by our own loop with a SentencePiece tokenizer. The 24-layer French model runs on the CPU, slower than
    real time.
  - **MOSS-TTS-Nano 100M** (OpenMOSS, Apache-2.0, 20 languages) from its official ONNX exports, with our own port of
    its reference runtime (tokenizer, text normaliser, chunking, voice-clone prompt, sampler, streaming codec); a
    parity test against the reference gives equal codes in every run. No transcript needed.
  - **Chatterbox Turbo** (English) and **Chatterbox Multilingual** (23 languages, Hindi included), MIT, through
    transformers.js; WebGPU advised. No watermark is applied in the browser.
  - **Numbers in words**: before generating, the page writes numbers, times, dates and money out in words with
    uccharan for the chosen language (English, German, Spanish, French, Italian, Portuguese, Dutch, Polish, Czech,
    Swedish, Danish, Norwegian, Finnish, Hungarian, Turkish, Hindi, Catalan, Romanian) and shows what the model was
    given. Measured on a German sentence: word error rate 49 % to 8 % (Pocket TTS), 22 % to 10 % (MOSS-TTS-Nano).
  - The demo's uccharan (WebAssembly) moves to 0.6.3, which reads ordinals and dates in the European languages.

### Fixed

- Windows: `bolo` writes UTF-8 to pipes and files (Hindi or any non-Latin text printed by `bolo transcribe` to a
  pipe or a redirect failed with `UnicodeEncodeError` under the legacy code page), and every JSON and text file bolo
  reads or writes names UTF-8 (the LFM3-Audio configuration, the ONNX graph cut's `split.json`, manifests, saved
  voices' metadata, consents); ruff's `PLW1514` now enforces it.
- Windows: the components folder finds `llama-server.exe` and `audiocpp_server.exe`, and `llama_server_bin` /
  `audiocpp_server_bin` without an extension find the `.exe` (`PATHEXT`), as `PATH` does.
- Windows: registering a local model folder (`bolo pull <folder>`, `file:`) no longer needs Developer Mode or
  administrator rights: without the right to make a directory symlink bolo makes a directory junction, else copies
  the folder (with a notice). Removing such a model removes the link, never the folder.
- Windows: log rotation works while `bolo logs -f` follows the file (it opens the file with `FILE_SHARE_DELETE`);
  when another program blocks the rename, the handler copies and empties the file instead and retries a minute
  later rather than printing "--- Logging error ---" for every record.
- audio.cpp servers bolo starts get a backend audio.cpp knows: `rocm` becomes `hip`, `coreml`/`mps` become `metal`,
  and `dml` or `openvino` run it on the CPU (audio.cpp refused to start with "unsupported server backend").
- `device = "dml"`: sherpa-onnx engines say once that they run on the CPU (its wheels have no DirectML provider).
- Live transcription with a streaming model no longer merges two utterances into one final when audio queued up
  while the model was busy (a slow machine, a client sending faster than real time): the queued audio is cut where
  the VAD heard the next utterance start, and a final ends where the VAD heard speech end rather than where the
  audio that carried that news ended. Found on Windows, where it happened in the tests.
- The least recently used model is unloaded first on Windows too: model use is timed with `perf_counter` (Windows'
  `monotonic` clock ticks every 15.6 ms, so two uses close together looked simultaneous).
- Windows: a plugin command with a Windows path (`plugins = ["torch=C:\\...\\python.exe -m bolo_torch"]`) keeps
  its backslashes (they were read as POSIX escapes, so the plugin was "not found").
- `chatterbox-multilingual-v3-llamacpp` sets a 4096-token context: llama-server's default reserved about 13 GB and was
  killed on a 15 GB machine.
- `xtts-v2-audiocpp` names its GPT GGUF (audio.cpp refuses a folder with two) and no longer sends `reference_text`,
  which audio.cpp refuses for XTTS v2 (new entry option `reference_text: false`).
- The Chatterbox entries no longer say their output carries Resemble's Perth watermark: llama.cpp and audio.cpp do
  not run the watermarker, and Resemble's detector found none in Chatterbox Turbo's output.
- `zonos3-mini-audiocpp` says that audio.cpp v0.9.4 has no zonos3 family.
- A Realtime API call over WebRTC gets 10 s for its peer connection to close and is then let go, so a peer that
  never finishes closing no longer keeps the call's close waiting for ever (seen on Windows).
- Live demo: choosing an example voice no longer replaces text the visitor typed.

### Changed

- **Text voices trust uccharan's normaliser only where it writes numbers as words.** bolo now asks the installed
  uccharan to write out a few numbers in each language (once per language, cached) and uses its normalise-only mode
  for that language's text voices only when the results are words of the language's script: no digits, no
  characters of another script, not the English reading, and not the same letters for 25 and 52 and for 13 and 31
  (digits spelled one by one). uccharan 0.6.1 wrote Sinhala and Sindhi numbers as internal lookup keys ("25" was
  ණගඡ, ڙتج), which the old check let through; with such a release those voices now get bolo's light normaliser
  (digits left for the voice). The check used to cover only the Latin-script languages.
- The `uccharan` extra (and `all`) asks for `uccharan>=0.6.4,<0.7` (was `>=0.6.1`): 0.6.2 writes Sinhala and
  Sindhi numbers as words; 0.6.3 and 0.6.4 read dates, ordinals and clock times in full ("10h30" is *dix heures
  trente*, "10:30" *zehn Uhr dreißig*) and put currency words in the plural (*vingt-cinq euros*).

## [0.5.0] - 2026-11-15

Ollama-style model handling: each model is placed on the GPU or the CPU (or split between them) when it loads,
unloaded after a keep-alive, listed by `bolo ps`, `/api/v1/ps` and the console's Loaded models panel, spread over
several GPUs, and managed with `show`, `rm`, `cp` and `logs`. Every ONNX Runtime engine follows the `device`
setting. uccharan 0.6.1 gives phonemes and number reading in its 39 languages, with language packs downloaded on
first use. The live demo gains 28 recognisers, 100 voices, a Speakers page and a voice conversation in the page.
Everything on the GPU side is tested with fakes only: no GPU was used.

### Added

- **More of Ollama's model handling.** GPU-side behaviour is tested with a fake NVML of several made-up GPUs and
  with llama-server command lines that are built but not run: untested on hardware.
  - **Several GPUs** (docs/configuration.md, "Several GPUs"). New setting `gpus` (`[0, 2]`, `BOLO_GPUS`,
    `bolo serve --gpus`): the GPUs bolo may use, numbered as CUDA numbers them after `CUDA_VISIBLE_DEVICES` (which
    bolo now applies to its NVML and nvidia-smi readings; `/api/v1/ps` gives `index` and `physical_index`). With
    `device = "cuda"` (or `auto`) each model goes on the allowed GPU with the most memory available, so models
    are spread over the GPUs as they load. A llama.cpp model that does not fit whole on one GPU is spread over
    several by layers (`--tensor-split`, `--main-gpu`); new `gpu_spread` (like `OLLAMA_SCHED_SPREAD`) spreads every
    llama.cpp model over all allowed GPUs. One-GPU placements on a machine with several get `--split-mode none
    --main-gpu N`. faster-whisper models get the chosen GPU's index; sherpa-onnx models use GPU 0 only. `-sm`,
    `-mg`, `-ts` or `--device` in your server arguments win.
  - **GPU per model** in `bolo ps` (a GPU column; GPUs left out by `gpus` are marked) and `/api/v1/ps` (`gpus`,
    `placement.gpu_bytes_by_gpu`, `tensor_split`, `main_gpu`; per GPU `allowed` and `models_vram_bytes`).
  - **Faster llama.cpp voices**: new settings `llama_flash_attention` (`-fa on|off|auto`), `llama_kv_cache_type`
    (`--cache-type-k/-v`: `f16`, `q8_0`, `q4_0`, ..., like `OLLAMA_KV_CACHE_TYPE`), `llama_context_size` (`-c`)
    and `llama_batch_size` (`-b`) for the llama-servers bolo starts (tts-server and Orpheus-family voices, the
    speech-to-speech server). Per model as catalog options (`flash_attention`, `kv_cache_type`, `context_size`,
    `batch_size`) or pins (`PUT /api/v1/models/{id}/settings`); an operator's own arguments win. A quantized KV
    cache turns flash attention on unless it is set. The quality and speed trade-offs are documented, with
    [estimate] marks: nothing was measured on a GPU.
  - **Parallel requests per model**: `num_parallel` (like `OLLAMA_NUM_PARALLEL`; the existing `workers` under
    Ollama's name, used when `num_parallel` is 0) and `max_queue` (existing; like `OLLAMA_MAX_QUEUE` but per
    model) can now be set per model (catalog `options.num_parallel` / `options.max_queue`, or pins). A llama-server
    bolo starts for a voice gets `--parallel N`. `bolo ps -v` shows ACTIVE, QUEUED and PARALLEL per model;
    `/api/v1/ps` has `running`, `queued`, `num_parallel` and `max_queue`.
  - **`bolo show <model>`** now prints licence, gated and commercial-use flags, files and sizes, the settings that
    apply (and pins), languages, voices, engine and speed (the catalog's measurements, the expected real-time
    factor here); `--license` prints the licence and the licence file shipped with the model, `--json` the
    document. New **`GET /api/v1/models/{id}/show`** (`?license=true`).
  - **`bolo cp <source> <destination>`** and **`POST /api/v1/models/copy`**: a preset, a new name that uses the
    source's files and starts with its settings, then has its own (`preset_of` in `custom.yaml`, the catalog and
    `bolo list`). Deleting a preset leaves the files.
  - **`bolo rm`** refuses a model the running server (`--url`, `$BOLO_URL`) has loaded unless `--force`, which
    unloads it first, and removes through the server when one answers.
  - **`bolo list`** shows downloaded models as `ollama list` does: ID, TASK, ENGINE, SIZE, MODIFIED, LICENCE
    (`--all` keeps the catalog view). Catalog entries carry `modified_at` and `preset_of`.
  - **`bolo models config <id> --set key=value / --unset key`** for every pinnable setting; `--clear` now removes
    every pin.
  - **Logs** (docs/configuration.md, "Logs"): new settings `log_file` (`auto`: `~/.local/state/bolo/logs/server.log`
    on Linux, `~/Library/Logs/bolo/server.log` on macOS, `%LOCALAPPDATA%\bolo\logs\server.log` on Windows; a
    path; `off`), `log_max_mb` (10) and `log_backups` (3): the log is also written to a rotating file. Off by
    default, on when `bolo serve` runs under systemd or launchd; the systemd unit writes
    `/var/lib/bolo/logs/server.log`. New **`bolo logs [-f] [--lines N]`**; `bolo doctor` says where the logs are;
    `/api/v1/info` has `log_file`.
  - **"Coming from Ollama"** in docs/configuration.md: each `OLLAMA_*` variable and its bolo setting. Every new
    setting has a `BOLO_*` variable and a TOML key; a test checks that every setting is documented.

- **Ollama-style model scheduling: GPU/CPU placement, keep-alive, `bolo ps`.** Untested on a GPU: everything
  GPU-side is tested with fake NVML readings and made-up memory figures.
  - **Placement per model** (docs/configuration.md, "GPU and CPU placement"). With an accelerator, each model goes
    on the GPU when its estimated memory fits in the free accelerator memory minus `gpu_reserve_mb` (default 512),
    after unloading idle GPU-resident models when that makes it fit; otherwise on the CPU, with a notice. Nothing is
    refused. llama.cpp models (tts-server, Orpheus-family voices, the speech-to-speech server) get `-ngl N` computed
    from the free memory and the GGUF's layer count and size, unless `-ngl` is set in their arguments. ONNX Runtime
    models made of several graphs (DiCoW ONNX, LFM3-Audio) are placed graph by graph. A single ONNX graph that
    does not fit (Nemotron diarization, the end-of-turn model, or the largest graph of a multi-graph model) is cut
    in two at a layer boundary: the front on the GPU, the back on the CPU (new setting `onnx_split`, default
    `auto`; needs `onnx`, new extra `bolo-speech[onnx-split]`). The halves are cached next to the model; tested
    with both halves on the CPU against the whole graph. bolo-torch plugin models get
    accelerate's `device_map="auto"` with a `max_memory` and report their split back (plugin protocol: optional
    `placement` in `load`, `info.placement` in the reply). sherpa-onnx and faster-whisper models move whole. A model
    can pin its device (`PUT /api/v1/models/{id}/settings {"device": "cpu"}`, or `options.device` in the catalog).
  - **Accelerator memory**: NVML (new optional extra `bolo-speech[nvidia]`, `nvidia-ml-py`) or `nvidia-smi`, with
    the memory held by bolo and the servers and plugins it started; Apple Silicon reported as unified memory; the
    growth of RAM and VRAM during each load is attributed to the model. New `gpu_memory_budget_mb`: LRU eviction
    among GPU-resident models, then the CPU.
  - **Keep-alive** (docs/api.md, "Keep-alive"): a `keep_alive` field on transcriptions, translations and speech, and
    `?keep_alive=` on the STT, TTS and Realtime WebSockets (applied when the session ends); seconds or durations
    (`"5m"`), `0` unloads after the request, `-1` keeps the model. New setting `keep_alive` (default `5m`);
    `stt_ttl_s`, `tts_ttl_s` and `diarization_ttl_s` now default to unset and override it when set (`vad_ttl_s`
    stays `-1`). Models can pin their own `keep_alive`.
  - **`GET /api/v1/ps`**, Ollama's `/api/ps` shape (`size`, `size_vram`, `processor`, `expires_at`, ...) with the
    machine's CPU, RAM and GPUs. bolo's `/api/ps` stays speaches' route. `/api/v1/stats` carries the same fields.
  - **`bolo ps`** shows MODEL, TASK, SIZE, PROCESSOR ("100% GPU", "35%/65% CPU/GPU"), DEVICE and UNTIL ("4 minutes
    from now", "Forever", "Stopping...") under a line with CPU, RSS, RAM and each GPU's used/total memory; `-v` adds
    the request and latency columns. New **`bolo load <model> --keep-alive 10m`** (also `bolo run`) and
    **`bolo stop <model>`**. `POST /api/v1/models/{id}/load` takes `keep_alive` and says where the model went.
  - **Console**: a "Loaded models" panel on the Models page: each loaded model with its size, CPU/GPU split, VRAM
    share, a countdown to unloading, and Unload / Keep loaded buttons, plus CPU, RAM and GPU totals. GPUs are
    numbered as `gpus` numbers them, ones bolo may not use are marked, and a model spread over several lists them.

- **Phonemes and text normalisation through uccharan 0.6 in its 39 languages** (models.md, "Phonemisers").
  Piper, Mimic 3 and Kokoro voices in Spanish, German, Dutch, French (also Belgian and Swiss), Italian,
  Portuguese (Brazil, Portugal), Polish, Turkish, Czech, Swedish, Catalan, Romanian, Hungarian, Finnish,
  Norwegian Bokmål, Danish, Urdu, Sindhi and the South Asian languages (Bengali, Marathi, Telugu, Nepali, Tamil,
  Malayalam, Gujarati and the rest) now get their phonemes from uccharan (Apache-2.0) instead of espeak-ng: 69
  catalog voices work without the GPL component (26 before), plus Kokoro's Spanish, French, Italian and
  Portuguese voices. A voice's espeak-ng voice name or language tag is mapped to uccharan's (`fr-fr` to `fr`,
  `no` to `nb`, `pt-BR` to `pt-br`, `en-gb-x-rp` to `en-gb`); languages uccharan does not have (Mandarin,
  Russian, Latin American Spanish `es-419` and others) still go to espeak-ng when it is installed, or fail with
  `424` as before. Text voices get uccharan's normalise-only mode in the languages where it reads numbers in
  the language itself: English, Hindi, the South Asian languages, Urdu, Sindhi and, with uccharan 0.6.1, the
  Latin-script languages (Spanish, German, French …). bolo asks the installed uccharan; with 0.6.0, which reads
  digits in English in those languages, they keep bolo's light normaliser.
- **uccharan's language packs are downloaded on first use** and are components. The word tables of languages
  other than American English and Hindi, British English and Kokoro's English dictionaries are packs on
  uccharan's release: bolo fetches one the first time a voice needs it, as a `pull` job with events and a log
  line, checked against the SHA-256 the installed uccharan lists (and again by uccharan when it loads it). New
  setting `uccharan_auto_download` (default `true`; `UCCHARAN_OFFLINE=1` also turns it off): when off, or when the
  release cannot be reached, the request fails with `424 component_missing` naming the pack and the command
  that fetches it. All 35 packs are listed as components `uccharan-<pack>` with size and licence (`bolo
  components list`, `bolo doctor`, `GET /api/v1/components`) and can be fetched with `bolo pull uccharan-de`,
  `bolo components install uccharan-de` or `POST /api/v1/models/pull` (`{"id": "uccharan-de"}`). The CC BY-SA
  4.0 packs (`wikipron-*`, `xlit-*`) are never downloaded automatically, only when installed with the licence
  accepted (`--accept-license`, `"accept_license": true`).
- `GET /api/v1/recommendations` without `task` answers an overview of every task (each with its default, its
  recommendations and its first `limit` results, 3 by default) instead of `400`.
- **Homebrew**: `brew install cb-sigma/tap/bolo` (macOS 14 or later, Linux), then `brew services start bolo`. The
  formula lives in [cB-Sigma/homebrew-tap](https://github.com/cB-Sigma/homebrew-tap), installs `all` and follows
  new releases by itself.
- **Live demo** (<https://cb-sigma.github.io/bolo/>, everything in the visitor's browser; tested in headless
  Chromium on a CPU, with WebGPU and phones faked):
  - **Speech to text: 28 models** (Moonshine, Whisper Tiny to Large v3 Turbo, Distil-Whisper, Hindi and Hinglish
    Whisper fine-tunes, IndicWav2Vec Hindi, Moonshine for Arabic, Chinese, Japanese, Korean, Ukrainian, Vietnamese
    and French), grouped by language with size, licence and needs. Moonshine is the default recogniser.
  - **Text to speech: 100 voices**: the 22 SYSPIN voices (8-bit, and the original 32-bit export), 42 Piper voices
    in 31 languages and Kokoro-82M in 8, 16 or 32 bits, with phonemes from uccharan 0.6.1 in the page, and a text
    playground for uccharan in all its languages (romanised text into native script, spoken form, phonemes).
  - **Device warnings**: a probe of memory, WebGPU and its limits, threads and storage gives each model a verdict
    in the pickers and before a download; a model not expected to work asks first and can free space.
  - **GPU/CPU placement per model** (Auto, GPU, CPU), WebGPU for the voices, and a voice cut in two at build time
    (front on WebGPU, back on the CPU); a "Loaded models" panel with an in-browser keep-alive.
  - **Speakers**: who spoke when, in the page (pyannote segmentation with WeSpeaker or CAM++ embeddings), with
    example conversations, their reference and the DER.
  - **Voice conversation**: microphone, recogniser, a small language model (in the page or your own endpoint) and
    a voice, with latencies and barge-in.

### Changed

- The `uccharan` extra (and `all`) asks for `uccharan>=0.6.1,<0.7` (was `>=0.4`).
- The `model` event of `/api/v1/events` for a load carries `processor` (and `notice` when the model is not all on
  the accelerator); the unload answer carries `unloaded` (false when the model was not loaded).

### Fixed

- **The `device` setting now reaches every ONNX Runtime engine.** Nemotron diarization, DiCoW ONNX,
  the SNAC decoder of the Orpheus-family voices, the end-of-turn model and in-process
  speech-to-speech always ran on the CPU, whatever `device` said. They now use the matching
  execution provider (CUDA, CoreML, ROCm, DirectML, OpenVINO) when the installed `onnxruntime`
  has it, and fall back to the CPU with one warning when it does not. sherpa-onnx models also
  accept `coreml` (`mps` and `metal` mean the same). New value `auto`: the first accelerator found.
  Still untested on a GPU.
- **Live demo**: the page no longer crashes once several models are downloaded (WebKit read every cached model
  into memory at start-up; iPhones closed the page): downloads are listed in local storage, with a panel to delete
  them, clear all and set a storage limit. The browser voices no longer garble the first word, and the first piece
  of audio is not lost on iOS. Maithili, Bhojpuri and Magahi in the text playground no longer ask uccharan for
  data files that do not exist.

## [0.4.0] - 2026-11-14

### Added

- **Hardware detection and model recommendations**
  ([docs/recommendations.md](docs/recommendations.md)). bolo looks at the machine it runs on and
  rates every catalog model for it. **It recommends and warns; it never restricts**: no model is
  hidden, refused or blocked because of hardware.
  - `bolo/hardware.py` detects, best effort and without new dependencies: the CPU (model,
    physical / logical / usable cores, AVX2, AVX-512, AVX-512 VNNI / BF16, AMX, NEON, dot product,
    SVE), memory, free disk in the data directory, NVIDIA GPUs (`nvidia-smi`, else `pynvml`),
    Apple Silicon's unified memory, Intel and AMD graphics from sysfs, what the runtimes can use
    (ONNX Runtime providers, CTranslate2 CUDA devices, the torch plugin's device, `llama-server`
    and `audio.cpp` builds), containers, virtual machines and cgroup CPU and memory limits. It
    never raises; what cannot be read is reported as unknown.
  - Every catalog entry gains a **`fit`** for this machine: a level (`comfortable`, `usable`,
    `slow`, `unlikely`, plus `cloud` and `unavailable`), the expected real-time factor with a
    range, whether it is measured-and-scaled, estimated from a sibling model or unknown, the
    memory or VRAM it needs against what there is, and a one-sentence reason. Untested entries say
    so instead of carrying an invented number.
  - **A micro-benchmark** (`bolo recommend --benchmark`, `POST /api/v1/hardware/benchmark`; about
    three seconds, nothing downloaded) times a tiny fixed workload (a hand-built ONNX graph in
    onnxruntime, else numpy) and stores the result in `<data_dir>/hardware-benchmark.json`; the
    catalog's measured real-time factors are then scaled by it. Without it they are scaled by a
    documented rule from the core count and instruction sets, and every surface says the estimate
    is rougher.
  - **`bolo recommend`** `[--task … --language … --commercial --local-only --realtime | --batch
    --max-size-mb N --use live|files|voiceover --all --why --json --benchmark]`: this machine,
    then a default pick per task, a ranked table for one task, or a whole setup for a use.
    `bolo doctor` gains "this machine" and a short recommended setup.
  - API: `GET /api/v1/hardware`, `POST /api/v1/hardware/benchmark`,
    `GET /api/v1/recommendations?task=&language=&commercial=&local_only=&realtime=&max_size_mb=`
    (every model of the task stays in the list, the unlikely ones last, each with its reason; at
    most three are marked recommended, one of them the default pick) and
    `GET /api/v1/recommendations/setup?use=&language=`. `GET /api/v1/info` gains
    `capabilities.recommendations`.
  - **Warnings, not restrictions**: pulling or loading a model that is unlikely to fit or to run
    in real time adds a `warnings` list to the answers of `POST /api/v1/models/pull` and
    `/load`, to the job and to request traces, a log line, and a `warning:` line from
    `bolo pull`. Status codes and behaviour do not change. New setting `hardware_warnings`
    (default `true`) turns the notices off.
  - **Console**: on the Models screen a "This machine" card (with "Run benchmark"), a fit line on
    every card and an "On this machine" section in the details, a "Recommended for this machine"
    badge, "Sort: best for this machine", an optional "Hide unlikely to fit" toggle (off by
    default, not remembered, and it says how many it hides), a notice when a pull or load comes
    back with warnings, and **"Pick for me"**: choose a language and a use (live conversation,
    transcribing files, voice-over), see the proposed models with their reasons, pull them in one
    click. The browser-only demo shows none of it: there the machine is the visitor's browser.

- **Phone calls** ([docs/telephony.md](docs/telephony.md)). Telephony providers connect a call to a
  bolo voice agent: **Twilio Media Streams**, **Plivo audio streams**, **Telnyx media streaming**
  and the **Exotel Voicebot applet**, each a thin adapter (`bolo/telephony/adapters.py`) in front
  of one provider-neutral call session that bridges into the existing Realtime session, so turn
  detection, the three pipelines, tools and traces are the ones `/v1/realtime` has. Routes
  `/telephony/{provider}/voice` (the answer webhook: TwiML `<Connect><Stream>`, Plivo XML
  `<Stream bidirectional keepCallAlive>`, the Telnyx `answer` command, Exotel's `{"url": …}`),
  `/telephony/{provider}/stream/{token}` (WebSocket), `/after` (hang up or `<Dial>` once bolo
  closed the stream) and `/status`. **No real call has been made with it**: it was written against
  the providers' documentation as read on 2026-11-14 and is tested against fake providers; the
  document lists what was verified and what was assumed for each.
- **On a call**: a recording-consent announcement (never interrupted), a greeting, **barge-in**
  that clears the audio queued at the provider (`clear` / `clearAudio`) and truncates the
  assistant's message to what was played, key presses (DTMF) handed to the agent as a user
  message and to tools, built-in `hang_up` and `transfer_call` tools (transfer by TwiML / Plivo
  XML `<Dial>`, the Telnyx `transfer` command, or Exotel's Passthru applet), tools with a `url`
  called by HTTP POST, "are you still there?" prompts and a hang-up after silence, a longest
  call length.
- **Agents and numbers**: named agents (instructions, greeting, voice, language, STT / TTS / LLM
  models, pipeline, tools, limits, transfer number, voicemail policy) and a called-number → agent
  map in the `telephony` setting (`[telephony.agents.<name>]`, `[telephony.numbers]`) and through
  `GET/PUT/DELETE /api/v1/telephony/agents/{name}` and `/numbers/{number}`
  (`<data_dir>/telephony.json`, mode 0600).
- **Outbound calls**: `POST /api/v1/telephony/calls` asks the provider to place a call and attach
  an agent (Twilio `Calls.json`, Plivo `Call/`, Telnyx `POST /v2/calls`, Exotel connect-to-flow),
  with the provider's **answering-machine detection** where it has one (Twilio `MachineDetection`,
  Plivo `machine_detection`, Telnyx `answering_machine_detection`; none for Exotel) and a per-agent
  policy: hang up, leave a message, or ignore. Compliance with outbound-calling rules (TRAI and
  DND in India, the TCPA in the United States, …) is the operator's responsibility; bolo checks
  none of it, and the API's answer says so.
- **Security**: every provider request is checked (Twilio `X-Twilio-Signature`, Plivo
  `X-Plivo-Signature-V3`, Telnyx Ed25519 with a five-minute tolerance, Exotel's Basic
  credentials); media streams need a one-time token handed out by the verified answer webhook.
  A provider's routes answer `503 telephony_not_configured`, naming the missing settings, until
  requests can be checked or `telephony.validate_signatures = false` is set on purpose. New
  setting `telephony.public_base_url`. Provider credentials come from the configuration or the
  environment (`BOLO_TELEPHONY_<PROVIDER>_<FIELD>`), are never logged or returned, and are removed
  from provider error messages. Ed25519 verification is implemented from RFC 8032 (checked against
  its test vectors and the `cryptography` package), so Telnyx needs no extra dependency.
- **Call records**: `GET /api/v1/telephony/calls` and `/calls/{id}`: a timeline, turns with a
  latency breakdown (`endpoint_ms`, `stt_ms`, `llm_first_token_ms`, `tts_first_audio_ms`,
  `send_ms`, `total_ms`, `response_ms`), the transcript with speakers (an interrupted line holds
  what was heard, with the full text beside it), the outcome, and a summary written by the
  language model when one is configured. In memory, like request traces. Each turn is also an
  `agent` trace with transport `telephony` and the `call_id`. `GET /api/v1/stats` gains
  `telephony`, `/metrics` gains `bolo_telephony_active_calls` and `bolo_telephony_calls_total`,
  the event stream gains `call` events, `GET /api/v1/info` gains `capabilities.telephony`.
- **Call simulator**: `bolo telephony simulate` (and `POST /api/v1/telephony/simulate`, and a
  button in the console). A scripted caller (speech from text-to-speech or files, silences, key
  presses, interruptions, noise) rings an agent through a fake provider that behaves like a phone
  line (8 kHz frames in real time, real-time playback, `clear`, marks) over the whole provider
  path (signed webhook, answer document, WebSocket), then reports what the caller heard, the
  response time as the caller experiences it, and whether the expected behaviours happened.
  Built-in scenarios `basic`, `barge-in`, `silence`, `dtmf`, `noisy`. `bolo telephony status`
  prints what is configured and the URLs to paste into each provider's dashboard.
- **Inbound SIP** (`sip = true`, `pip install 'bolo-speech[sip]'`; experimental): a minimal
  endpoint for trunks and PBXs: `INVITE` with an SDP offer, PCMU / PCMA RTP, key presses as RTP
  telephone events (RFC 4733), `ACK`, `BYE` from either side, `CANCEL`, `OPTIONS`,
  retransmissions; callers admitted by source address (`sip_allowed_ips`, loopback only by
  default). No REGISTER, digest authentication, TLS, SRTP, re-INVITE or transfer. SIP messages are
  parsed with sipmessage (BSD-2-Clause). Tested against the suite's own SIP user agent, not
  against a trunk or a softphone.
- **Console**: a **Phone** page: each provider's state, what is missing and the webhook URLs to
  paste (with where to paste them), agents (add, edit, remove), the number → agent map, recent
  calls with per-turn latency bars, transcript and timeline, and "Simulate a call".
- Realtime sessions: `RealtimeSession.responding`, `.user_speaking`, `.input_ms`, `.interrupt()`
  and per-session `tts_model` / `llm_model` / `trace_extra`, used by the telephony bridge.

- **Romanised text for text-to-speech** (docs/api.md, "Romanised text"). An Indian language typed
  in Latin letters ("mujhe kal office jana hai") is written in the voice's script ("मुझे कल office
  जाना है") by uccharan's transliterator before the normaliser and the phonemiser see it; English
  words stay English. Modes `off` (default), `auto` (the voice speaks an Indian language and at
  least 60 % of the text's letters are Latin) and `on`, set per request (`romanised` in
  `/v1/audio/speech`, `?romanised=` on the TTS WebSocket, `session.audio.output.romanised`,
  `audio.romanised` in chat completions, `romanised` on a telephony agent,
  `bolo speak --romanised`), per model (`PUT /api/v1/models/{id}/settings`, catalog option
  `romanised`) and server-wide (new setting `romanised`). Word-list entries, inline
  pronunciations, URLs and email addresses are not converted.
  - The transliterator and its language packs are optional and detected when bolo starts
    (`bolo/text/romanised.py`). With uccharan 0.3.0 or older, without uccharan, or without the
    language's pack, text is passed on unchanged and the request's trace carries a `romanised`
    record with a notice that says what to install; only a request that itself says
    `"romanised": "on"` is refused (`424 component_missing`, or `400` for a language or voice it
    cannot apply to). Sessions, chat completions and phone calls never fail on it.
  - uccharan's packs (`xlit-<language>`, CC BY-SA 4.0) are components: `uccharan-xlit-hi`, … in
    `GET /api/v1/components` (kind `language-pack`), installed on request only with
    `bolo components install uccharan-xlit-hi --accept-license` or the API with
    `{"accept_license": true}`; uccharan downloads and checks them.
  - `GET /api/v1/info` gains `capabilities.romanised`; `GET /api/v1/text-input/{model}` and the
    word-list preview gain `romanised`.
  - Console: a "Romanised text" choice on the Text to speech screen for voices of Indian
    languages, a pin in a model's settings, and a line in a request's "What the voice received"
    and in the word-list preview that says what was converted, or why not.
  - Tested against a stand-in transliterator, against uccharan 0.3.0 (which has none), without
    uccharan, and against uccharan 0.4.0 with its Hindi pack (the component install and one
    request on a Piper Hindi voice); no listening test of converted text was made.

### Changed

- `bolo list` has a `FIT` column (how the model suits this machine, as on the catalog's `fit`).
- The `uccharan` extra and the `uccharan` component ask for uccharan 0.4 or later, so that word
  lists and the romanised-text option work from a plain install.
- The README shown on PyPI has absolute links and images (`tools/pypi_readme.py` writes
  `server/README.md` from the repository's README).

### Fixed

- `bolo recommend` and `GET /api/v1/recommendations` without a language no longer suggest
  single-language voices of arbitrary languages for text to speech ("also recommended:
  mimic3-ne-google"): English and multilingual voices come first when no language is asked for.

## [0.3.0] - 2026-11-14

### Added

- **Speech-to-speech in bolo's own process: `lfm3-audio-2b-onnx`** ([docs/voice-agents.md](docs/voice-agents.md)).
  `s2s_model = "lfm3-audio-2b-onnx"` runs Liquid AI's LFM3-Audio 2B from its ONNX graphs with ONNX
  Runtime and NumPy (`bolo/s2s_onnx.py`; `pip install 'bolo-speech[s2s]'`: onnxruntime and tokenizers), with
  no llama-server: log-mel front end, the audio encoder (the turn is evaluated while the user
  speaks), the language model with its caches, the depth decoder (three MaskGIT passes per 80 ms
  speech frame), the codec decoder and inverse STFT (24 kHz audio in chunks as it is generated),
  interleaved text and speech, tool calls, barge-in between steps. The catalog entry pins every
  file by SHA-256 (2.7 GB), states the LFM Open License (commercial use only below USD 10M yearly
  revenue) and is also a text-to-speech voice (engine `onnxruntime`, kind `lfm3_audio_onnx`; two
  voices, 18 languages). Without the packages a session answers 424, with missing files 503, and
  `GET /api/v1/info` reports the pipeline as configured but not available. On 4 shared CPU cores it
  generates speech at 0.4 to 0.5 times real time (first audio after about 1 s): it needs a faster
  CPU or a GPU to hold a conversation. Opt-in real test: `BOLO_TEST_REAL=1 BOLO_TEST_S2S_ONNX=1`.
- **Cloud providers: the `remote` engine** ([docs/providers.md](docs/providers.md)). bolo can hand
  speech-to-text and text-to-speech to a cloud API and treat the answer like a local model's. One
  engine, four adapters: `openai-compatible` (OpenAI and every server that copies its audio API,
  including another bolo: transcriptions with `json` / `verbose_json` / streamed answers,
  translations, streamed speech), `sarvam` (Saaras v3/v4 recognition and translation, Bulbul
  v2/v3/v4-flash voices, the streaming recognition socket), `elevenlabs` (streamed speech, Scribe)
  and `deepgram` (Nova pre-recorded and live). **These are paid cloud APIs and audio leaves the
  server**; nothing is sent unless a cloud model is named. Request fields (language, voice, speed,
  prompt, hotwords, timestamps, translation) are mapped per provider; options a provider lacks
  answer a clear 400 (or are dropped with `ignore_unsupported`).
- **`providers` setting** (`[providers.<name>]` tables: `kind`, `base_url`, `api_key`, `options`,
  `prices`; `BOLO_PROVIDERS`, `BOLO_PROVIDER_<NAME>_API_KEY` / `_BASE_URL` / `_KIND`), and
  `GET/PUT/DELETE /api/v1/providers[/{name}]`, `POST /api/v1/providers/{name}/test` for providers
  added at run time (`<data_dir>/providers.json`, mode 0600; `allow_provider_edit`). Keys are
  write-only: never logged, never returned, removed from provider error messages. CLI:
  `bolo providers list`, `bolo providers test <name>`, and a providers line in `bolo doctor`.
- **Cloud catalog entries** (20: `openai-*`, `sarvam-*`, `elevenlabs-*`, `deepgram-*`) with
  `engine: remote`, `cloud: true`, the notice "paid cloud API; audio leaves this server", the
  provider's terms in the licence field, languages and voices. They count as installed when their
  provider is configured. Any other model works as `<provider>:<model>`.
- **bolo's features on cloud models**: diarization and long-recording segmentation (bolo cuts,
  the provider transcribes each piece), word lists and text normalisation before the text leaves
  (per-model `text_normalizer`), every output format including 8 kHz G.711, the STT WebSocket
  (Deepgram's and Sarvam's live sockets), Realtime cascades, traces (`provider`: requests, time,
  bytes; `cost`: an estimate from the operator's price table), `stats.providers`, and
  `bolo_provider_*` metrics.
- **Fallback**: a per-model `fallback` list (setting `fallbacks`, a pin through
  `PUT /api/v1/models/{id}/settings`, or `options.fallback` in the catalog), local to cloud or
  cloud to local, used on `engine_unavailable`, `overloaded` and provider failures (5xx, timeout,
  rate limit, refused key). The trace records it (`fallback`), responses carry `x-bolo-model` and
  `x-bolo-fallback`, and `bolo_fallbacks_total` counts it.
- **Cloud speech-to-speech**: `s2s_model = "<provider>:<model>"` (for example
  `openai:gpt-realtime-2.1`) backs the Realtime `speech-to-speech` pipeline with OpenAI's
  Realtime API over WebSocket. bolo keeps turn detection, barge-in, codecs and transcripts.
  Gemini Live is not done.
- **For clients (found by the console)**: `capabilities.realtime.pipelines.*` have `configured`
  and `available`; catalog entries have `runnable` / `not_runnable_reason` (`spark-tts-0.5b`,
  `maskgct`, `diarizen-wavlm-large-s80-md-v2`); voice-agent traces have `pipeline`; `info` has
  `auto_pull` and `providers`; `info.capabilities` has `wordlists` and `providers`.
- **Console**: a Providers page (add, edit, test; the key field is write-only), a "cloud" mark and
  filter on the Models page, and a Compare page that sends the same audio or text to two or more
  models and shows output, latency and estimated cost side by side.

### Changed

- `GET /api/v1/catalog` now lists entries whose files are not published yet (`status:
  "pending"`, `installed: false`) instead of leaving them out. `GET /v1/models` is unchanged
  (installed models only).
- `GET /api/v1/stats` has two more keys, `providers` and `fallbacks` (empty lists without cloud
  requests).
- **Published model files**: `dicow-v3.3-onnx` (bolo's ONNX export of DiCoW v3.3,
  `codebucketsolutions/dicow-v3.3-onnx`) and the SNAC 24 kHz decoder of the Orpheus-family
  llama.cpp voices (`codebucketsolutions/snac-24khz-onnx`) are on Hugging Face, rebuilt by the
  "Publish a converted model" workflow and compared with the recorded SHA-256 sums before upload.
  The catalog pins their revisions and the entries are no longer `pending`: `dicow-v3.3-onnx`
  (validated on CPU), `svara-tts-v1-llamacpp`, `svara-tts-v1-q8-llamacpp`, `veena-llamacpp`,
  `veena-q8-llamacpp`, `orpheus-3b-0.1-ft-llamacpp` and `snortts-indic-v0-llamacpp` (untested from
  the published files).
- Install instructions lead with `uv` (`uv tool install 'bolo-speech[all]'`,
  `uvx --from 'bolo-speech[all]' bolo serve`).

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
