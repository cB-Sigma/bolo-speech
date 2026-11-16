# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
aims to follow [Semantic Versioning](https://semver.org/) once it reaches 1.0.

## [Unreleased]

### Added

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

### Fixed

- `chatterbox-multilingual-v3-llamacpp` sets a 4096-token context: llama-server's default reserved about 13 GB and was
  killed on a 15 GB machine.
- `xtts-v2-audiocpp` names its GPT GGUF (audio.cpp refuses a folder with two) and no longer sends `reference_text`,
  which audio.cpp refuses for XTTS v2 (new entry option `reference_text: false`).
- The Chatterbox entries no longer say their output carries Resemble's Perth watermark: llama.cpp and audio.cpp do
  not run the watermarker, and Resemble's detector found none in Chatterbox Turbo's output.
- `zonos3-mini-audiocpp` says that audio.cpp v0.9.4 has no zonos3 family.
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

### Changed

- **Text voices trust uccharan's normaliser only where it writes numbers as words.** bolo now asks the installed
  uccharan to write out a few numbers in each language (once per language, cached) and uses its normalise-only mode
  for that language's text voices only when the results are words of the language's script: no digits, no
  characters of another script, not the English reading, and not the same letters for 25 and 52 and for 13 and 31
  (digits spelled one by one). uccharan 0.6.1 wrote Sinhala and Sindhi numbers as internal lookup keys ("25" was
  ණගඡ, ڙتج), which the old check let through; with such a release those voices now get bolo's light normaliser
  (digits left for the voice). The check used to cover only the Latin-script languages.
- The `uccharan` extra (and `all`) asks for `uccharan>=0.6.2,<0.7` (was `>=0.6.1`): 0.6.2 writes Sinhala and
  Sindhi numbers as words.

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
