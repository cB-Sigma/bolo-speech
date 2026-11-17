# Configuration

Settings come from, lowest to highest priority:

1. built-in defaults;
2. a TOML file: `--config FILE`, else `$BOLO_CONFIG`, else `./bolo.toml`, else
   `~/.config/bolo/config.toml`;
3. environment variables `BOLO_<NAME>`: every setting below has one, its name in capitals (`gpu_spread` is
   `BOLO_GPU_SPREAD`, `llama_kv_cache_type` is `BOLO_LLAMA_KV_CACHE_TYPE`); a few `speaches` names are honoured
   too, see below, and Ollama users will find their `OLLAMA_*` variables mapped in "Coming from Ollama";
4. command-line flags of `bolo serve`.

Lists can be written as TOML arrays, JSON arrays or comma-separated strings.
Booleans accept `1/0`, `true/false`, `yes/no`, `on/off`.

## Settings

| Setting | Default | Flag | Meaning |
| --- | --- | --- | --- |
| `host` | `127.0.0.1` | `--host` | bind address |
| `port` | `8000` | `--port` | HTTP port |
| `api_key` | none | `--api-key` | if set, required as `Authorization: Bearer <key>` (or `X-API-Key`, or `?api_key=` for WebSockets and SSE) |
| `cors_origins` | `[]` | `--cors-origin` (repeatable) | allowed browser origins; `*` allows any |
| `ui` | `true` | `--no-ui` | serve the web console at `/` |
| `cross_origin_isolation` | `true` | | send COOP/COEP headers with the console so its in-browser WebAssembly recogniser can use threads. API responses are not affected. Turn off if you embed the console in another site's frame |
| `max_upload_mb` | `200` | | largest accepted audio upload |
| `data_dir` | `~/.cache/bolo` (`$XDG_CACHE_HOME/bolo`) | `--data-dir` | models and the blob cache |
| `auto_pull` | `true` | `--no-auto-pull` | download a catalog model the first time it is requested (never for gated models) |
| `hf_token` | none (falls back to `$HF_TOKEN`) | `--hf-token` | Hugging Face token for gated or private repositories |
| `hf_endpoint` | `https://huggingface.co` | | Hub endpoint (for a mirror) |
| `default_stt` | `whisper-base` | `--default-stt` | model behind `whisper-1` and requests that name no model |
| `default_realtime_stt` | `moonshine-tiny-en` | `--default-realtime-stt` | model for the STT WebSocket when none is named (`model=realtime`) |
| `default_tts` | `kokoro-v1.0-int8` | `--default-tts` | the quality voice: requests that name no model, `tts-1-hd`, `gpt-4o-mini-tts` |
| `default_realtime_tts` | `piper-en-us-amy-low` | | the low-latency voice: `tts-1`, `model=realtime`, voice agents (Realtime API conversations, `/v1/chat/completions` audio without `speech_model`) and Wyoming synthesis without a voice. A voice that only `default_tts` has (e.g. `af_heart`) still selects `default_tts`. See models.md, "Default voices" |
| `phonemizer` | `auto` | | who turns text into phonemes for Piper, Kitten, Matcha and Kokoro voices: `auto` (uccharan where it covers the voice's language: its 39 languages, English, Hindi and the South Asian languages, Urdu, Sindhi and nineteen European ones; espeak-ng otherwise), `uccharan` (never espeak-ng), `espeak-ng` (as in bolo 0.1). Requests (`phonemizer` field) and per-model pins (`bolo models config <id> --phonemizer …`, `PUT /api/v1/models/{id}/settings`, stored in `<data_dir>/model-settings.json`) override it. See models.md, "Phonemisers" |
| `uccharan_auto_download` | `true` | | download a uccharan language pack (Apache-2.0: a language's word table, British English, Kokoro's English dictionaries) the first time a voice needs it, as a `pull` job, checked against the SHA-256 the installed uccharan lists, into uccharan's cache. `false` (or `UCCHARAN_OFFLINE=1` in bolo's environment): a voice whose pack is missing answers `424 component_missing` with the pack's name and the command that fetches it (`bolo pull uccharan-<pack>`, `python -m uccharan.packs <pack>`; offline machines: `UCCHARAN_DATA_DIR`). The CC BY-SA 4.0 packs (`wikipron-*`, `xlit-*`) are never downloaded automatically. See models.md, "Phonemisers" |
| `romanised` | `off` | | romanised text for text-to-speech: an Indian language typed in Latin letters is written in the voice's script before it is read. `off`; `auto` (when the voice speaks an Indian language and the text is mostly Latin script); `on`. A request (`romanised`) and a model pin override it. Needs a uccharan release with a transliterator and the language's pack (component `uccharan-xlit-<language>`, CC BY-SA 4.0); when one is missing the text is passed on unchanged with a notice in the request's trace. See api.md, "Romanised text" |
| `default_vad` | `silero-vad` | | detector for long-file segmentation and realtime endpointing |
| `default_diarization` | `pyannote-3.0-eres2net` | | speaker-diarization model used when a request names none; `nemotron-3-diarization` is the end-to-end alternative (models.md, "Speaker diarization") |
| `voice_similarity_model` | unset | | diarization model whose speaker embeddings score cloned voices (speaker similarity, `docs/voices.md`). Unset: `default_diarization`. The verdicts ("same speaker" at 0.50, "similar" at 0.30) were calibrated with `pyannote-3.0-eres2net` only |
| `diarization_min_speaker_s` | `3` | | after clustering, a "speaker" with less speech than this in the whole recording (and never more than a tenth of the speech) is folded into the most similar speaker; `0` switches it off. Not applied to `nemotron-3-diarization`, whose speakers come from the model rather than from clustering. See api.md, "How many speakers" |
| `diarization_merge_similarity` | `0.6` | | speakers whose voice prints have at least this cosine similarity are merged; `0` switches it off |
| `default_separation` | `voxlock-int8` | | model used by `separate_overlaps` (opt-in per request; pulled on first use). See api.md, "Separating overlapping speech" |
| `separation_context_s` | `2` | | seconds of audio on each side of an overlap that are separated and transcribed again |
| `separation_min_overlap_s` | `0.5` | | overlaps shorter than this are left alone |
| `separation_sample_max_s` | `10` | | at most this much of a speaker's speech without overlap is used as their voice sample (VoxLock needs 3 s and uses the loudest 3 s of it) |
| `separation_timeout_s` | `300` | | stop separating after this long per request; what was not separated is transcribed from the mixture and the response says `reason: "timeout"` |
| `default_turn` | `smart-turn-v3` | | end-of-turn model for Realtime sessions that ask for `semantic_vad` (pulled on first use; without it `semantic_vad` is plain silence detection) |
| `plugins` | `[]` | | out-of-process engine plugins, `name=command args` each (docs/plugins.md) |
| `plugin_entry_points` | `true` | | also use plugins installed as packages (entry-point group `bolo.engines`) |
| `plugin_idle_s` | `300` | | stop a plugin process when none of its models has been loaded for this long |
| `plugin_memory_limit_mb` | `0` | | address-space limit for plugin processes (POSIX); `0`: none |
| `plugin_nice` | `0` | | niceness added to plugin processes |
| `webrtc_ice_servers` | `[]` | | STUN/TURN URLs for the WebRTC transport of the Realtime API (`stun:host:port`, `turn:…`). Empty: no external server is contacted and calls connect on the same machine or network only |
| `llm_base_url` | none | `--llm-base-url` | OpenAI-compatible chat-completions endpoint for voice agents (`/v1/realtime` conversations, `/v1/chat/completions`), e.g. `http://127.0.0.1:11434/v1` |
| `llm_model` | none | `--llm-model` | model name sent to that endpoint; unset: the client's model name is passed on |
| `llm_api_key` | none | `--llm-api-key` | bearer token for that endpoint |
| `llm_timeout_s` | `120` | | |
| `llm_instructions` | a short "helpful voice assistant" prompt | | default system prompt of Realtime conversations |
| `realtime_pipeline` | `cascade` | | what answers a Realtime conversation turn: `cascade` (bolo's STT, then the `llm_*` language model, then bolo's TTS), `audio-llm` (the user's audio goes to an audio-understanding model, `audio_llm_*`; bolo's TTS speaks the answer) or `speech-to-speech` (a speech-to-speech model hears the turn and answers with speech and tool calls, `s2s_*`). A session can pick another one with the bolo extension `session.pipeline`. See voice-agents.md, "Pipelines" |
| `audio_llm_base_url`, `audio_llm_model`, `audio_llm_api_key` | unset (the `llm_*` values) | | `audio-llm`: a chat-completions endpoint whose model accepts `input_audio`, e.g. a llama-server running Gemma 4 E2B/E4B or Ultravox |
| `realtime_transcribe_user` | `true` | | `audio-llm` and `speech-to-speech`: also transcribe the user's turns with bolo's STT (transcript events and conversation history), in parallel with the response |
| `s2s_model` | `lfm3-audio-2b-llamacpp` | | `speech-to-speech`: the catalog model of a llama-server that bolo starts (needs `bolo components install llama-server` or `llama_server_bin`; the model is pulled on first use). LFM Open License: commercial use only below USD 10M yearly revenue. Or `lfm3-audio-2b-onnx`: the same model from its ONNX graphs, in bolo's own process with ONNX Runtime (`pip install "bolo-speech[s2s]"`; no llama-server, a 2.7 GB download, about 3.4 GB of memory; without the packages a session answers 424, with missing files 503). Or `<provider>:<model>` (a provider of kind `openai-compatible`, e.g. `openai:gpt-realtime-2.1`): OpenAI's Realtime API over WebSocket answers instead; a paid cloud API, the conversation's audio leaves this server (providers.md, "Cloud speech-to-speech") |
| `s2s_server_url` | unset | | `speech-to-speech`: a llama-server you run (`http://host:port`); bolo opens its WebSocket `/v1/realtime`, one upstream session per Realtime session. Nothing is downloaded or started then |
| `s2s_server_api_key` | unset | | sent as `Authorization: Bearer …` to `s2s_server_url` |
| `s2s_parallel` | `2` | | sessions the managed server holds at once (`llama-server --parallel`; one slot per session) |
| `s2s_server_args` | `[]` | | extra command-line arguments for the managed server (bolo adds `-ngl N`, the layers that fit in free accelerator memory, when `device` is not `cpu` and the arguments have no `-ngl`) |
| `allow_component_install` | unset | | whether optional components (eSpeak NG) may be installed through the API. Unset: allowed on a loopback bind or when an API key is set. Never allowed on a non-loopback bind without a key |
| `preload` | `[]` | `--preload` (repeatable) | models to load at startup; `/readyz` waits for them |
| `keep_alive` | `5m` | | how long a model stays loaded once idle, as in Ollama: seconds (`300`) or a duration (`"30s"`, `"5m"`, `"1h30m"`); `0` unloads a model as soon as the request using it ends; a negative value (`-1`) keeps models loaded until they are unloaded on purpose or evicted. A request's own `keep_alive` (api.md, "Keep-alive") and the per-task values below override it. Environment: `BOLO_KEEP_ALIVE`. Ollama's `OLLAMA_KEEP_ALIVE` is deliberately not read: bolo often runs next to an Ollama whose setting is meant for Ollama |
| `stt_ttl_s`, `tts_ttl_s` | unset (`keep_alive`) | | per-task override of `keep_alive`, in seconds; `-1` never, `0` at once |
| `vad_ttl_s` | `-1` | | same, for the VAD (kept loaded by default) |
| `diarization_ttl_s` | unset (`keep_alive`) | | same, for diarization pipelines and separation models |
| `memory_budget_mb` | `0` (off) | `--memory-budget-mb` | evict idle models (LRU) to stay under this estimate. A model's memory is estimated as the growth of the process during its load, and never less than the size of its files |
| `max_loaded_models` | `0` (off) | | cap on models in memory |
| `gpu_reserve_mb` | `512` | | with an accelerator: accelerator memory left free when placing a model (the display, other programs, allocation spikes). A model goes on the GPU only when its estimated memory fits in the free memory minus this; see "GPU and CPU placement" below |
| `onnx_split` | `auto` | | `auto`: an ONNX Runtime model that does not fit in free accelerator memory whole (Nemotron diarization, the end-of-turn model, or the largest graph of DiCoW ONNX and LFM3-Audio) is cut in two at a layer boundary; the front half runs on the accelerator, the back half on the CPU. Needs the `onnx` package (`bolo-speech[onnx-split]`); without it, or with `off`, such a model runs on the CPU. The halves are written once to `.bolo-split/` next to the model. See "GPU and CPU placement" |
| `gpus` | `[]` (all) | `--gpus` | the NVIDIA GPUs bolo may use, as CUDA numbers them in bolo's process (after `CUDA_VISIBLE_DEVICES`), e.g. `[0, 2]`. A model with `device = "cuda"` goes on the one of them with the most memory available; a pin to a GPU outside the list puts the model on the CPU with a notice. See "Several GPUs" |
| `gpu_spread` | `false` | `--gpu-spread` | spread every llama.cpp model over all allowed GPUs (`--tensor-split`), like `OLLAMA_SCHED_SPREAD`, instead of keeping it on one GPU when it fits there. Other runtimes cannot split a model over GPUs and take one GPU each either way |
| `llama_flash_attention` | unset | | flash attention (`-fa on\|off\|auto`) for the llama-servers bolo starts (tts-server and Orpheus-family voices, the speech-to-speech server); unset: llama-server's default (`auto`). `true`/`false` mean `on`/`off`. See "llama.cpp speed and memory" |
| `llama_kv_cache_type` | unset (`f16`) | | the K and V cache type (`--cache-type-k`/`--cache-type-v`) of those servers, like `OLLAMA_KV_CACHE_TYPE`: `f16`, `q8_0`, `q4_0` (also `f32`, `bf16`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`). A quantized type turns flash attention on unless it is set |
| `llama_context_size` | `0` (llama-server's default) | | context size (`-c`) of those servers, like `OLLAMA_CONTEXT_LENGTH` |
| `llama_batch_size` | `0` (llama-server's default) | | logical batch size (`-b`) of those servers |
| `gpu_memory_budget_mb` | `0` (off) | | accelerator memory bolo's models may take in all. Loading a model that would go over it first unloads idle GPU-resident models, least recently used first; when every one is in use, the new model is placed on the CPU (with a notice) instead. Nothing is refused |
| `probe_models` | `true` | | load a newly installed model once in a child process first |
| `threads` | `0` = min(4, CPUs) / `num_parallel` | `--threads` | inference threads per engine call. If the machine also runs other work (a language model, other services), set this below the number of cores: oversubscribed inference threads slow each other down badly |
| `workers` | `1` | `--workers` | concurrent engine calls per model (the older name of `num_parallel`, used when that is `0`) |
| `num_parallel` | `0` (= `workers`) | `--num-parallel` | requests one model works on at once, like `OLLAMA_NUM_PARALLEL`. A model can have its own (`options.num_parallel` in the catalog, or a pin); a llama-server bolo starts gets `--parallel N` with it. See "Parallel requests and queues" |
| `max_batch` | `8` | `--max-batch` | largest micro-batch |
| `batch_wait_ms` | `0` | `--batch-wait-ms` | linger for a fuller batch (adds latency; see architecture.md) |
| `max_queue` | `64` | `--max-queue` | requests waiting per model before `429`, like `OLLAMA_MAX_QUEUE` (which counts the whole server). A model can have its own (`options.max_queue`, or a pin) |
| `queue_timeout_s` | `30` | | fail jobs that waited longer than this |
| `partial_interval_ms` | `350` | | how often a pseudo-streaming session re-decodes for a partial |
| `endpoint_silence_ms` | `500` | | trailing silence that ends an utterance. Per stream: `?endpointing=fast\|balanced\|patient` or `?endpoint_ms=` (see realtime.md for the trade-off) |
| `max_utterance_s` | `20` | | force a final after this much continuous speech |
| `agreement` | `2` | | hypotheses that must agree before a word is shown as stable |
| `wyoming` | `false` | `--wyoming` | serve the Wyoming protocol |
| `wyoming_host`, `wyoming_port` | `127.0.0.1`, `10300` | `--wyoming-host`, `--wyoming-port` | Wyoming bind address |
| `device` | `cpu` | | where models run (per model, as placed by the scheduler: see "GPU and CPU placement" below; a model can pin its own with `PUT /api/v1/models/{id}/settings` `{"device": "cpu"}` or a catalog `options.device`): `cpu`, `cuda` (or `cuda:1`), `coreml` (also `mps`, `metal`), `rocm`, `dml`, `openvino`, or `auto` (the first accelerator found; for the ONNX Runtime engines, the first one the installed `onnxruntime` has a provider for: see "GPU and CPU placement"). sherpa-onnx models use `cuda` (only with a CUDA build of sherpa-onnx: the wheels on PyPI are CPU builds, and their models are placed on the CPU with a notice) or `coreml`; faster-whisper uses `cuda`; the ONNX Runtime engines (Nemotron diarization, DiCoW ONNX, the Orpheus-family decoder, the end-of-turn model, in-process speech-to-speech) use the matching execution provider when the installed `onnxruntime` has it (for `cuda`, install `onnxruntime-gpu`; on Windows also its CUDA libraries, see "CUDA on Windows"), and otherwise run on the CPU with one warning; llama-server gets `-ngl N`, the layers that fit in the free accelerator memory (all of them when everything fits; none when nothing does), unless you set `-ngl` yourself in `tts_server_args`, `s2s_server_args` or the entry's `server_args`; audio.cpp gets the device as its backend (`cuda`, `hip` for `rocm`, `vulkan`, `metal` for `coreml`; the CPU for `dml` and `openvino`, which it does not have). With `dml` (Windows, DirectML) only the ONNX Runtime engines use the GPU; sherpa-onnx and faster-whisper run on the CPU (see "Windows" below). **Measured on one laptop's GPU only** (CUDA, DirectML, Vulkan: [status.md](status.md#a-windows-11-laptop)) |
| `gpu_health_interval_s` | `30` | | while models are on a GPU, ask the driver this often whether the GPU is still there (NVML, else nvidia-smi), and also right after a request whose real-time factor is more than twice that model's recent median. `0`: no such checks; a CUDA or Vulkan error printed by a llama-server or audio.cpp bolo started still counts. See "A lost GPU" |
| `gpu_lost_fallback` | `cpu` | | what happens to the models on a GPU that was lost: `cpu` loads them again on the CPU (the CPU build of llama-server and audio.cpp, every GPU hidden) and places later loads there; `fail` leaves them in place and fails their requests (and later GPU loads) with `engine_unavailable`, for a deployment that would rather fail than run many times slower. See "A lost GPU" |
| `log_level` | `info` | `--log-level` | `debug`, `info`, `warning`, `error` |
| `log_file` | unset (off, except under systemd or launchd) | `--log-file` | also write the server's log to a file: `auto` (the platform's place, see "Logs"), a path, or `off`. Unset: no file, unless `bolo serve` runs as a service |
| `log_max_mb` | `10` | | rotate the log file at this size |
| `log_backups` | `3` | | rotated files kept (`server.log.1`, `server.log.2`, ...) |
| `trace_buffer` | `1000` | | request traces kept in memory |
| `otel` | unset (off unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set) | | export each request's trace as OpenTelemetry spans over OTLP; needs the `otel` extra. `true` / `false` force it on or off; unset, it is on when `OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is set. See "OpenTelemetry" below |
| `fake_engine` | `false` | `--fake-engine` | add deterministic test models (development) |
| `tts_servers` | `[]` | | external TTS servers the operator runs, as `"<model id or runtime>=<url>"`, e.g. `"llama-server=http://127.0.0.1:8080"` (every llama-server model goes there) or `"voxcpm2-audiocpp=http://gpu-box:8080"` (one model). A model id or alias wins over a runtime name. Such a model needs no download and no binary; for audio.cpp the server's config must declare the model under the id in the entry's `options.server_model` (the catalog id). See models.md, "External TTS servers" |
| `tts_server_api_key` | unset | | sent as `Authorization: Bearer …` to the servers in `tts_servers` (llama-server `--api-key`) |
| `llama_server_bin`, `audiocpp_server_bin` | unset | | the binary bolo starts for a model whose runtime has no URL in `tts_servers`. Unset: the one `bolo components install llama-server` / `audio.cpp` put in `<data_dir>/components/`, then `llama-server` / `audiocpp_server` on `PATH` (on Windows `llama-server.exe` / `audiocpp_server.exe`: a name without an extension finds the `.exe`, as `PATH` does). Point it at a CUDA/Vulkan/Metal build to use a GPU; `bolo components install` itself installs the Vulkan build with `device = "vulkan"` on Linux x86-64 and Windows, and the CUDA 12 build with `device = "cuda"` on Windows. llama.cpp and audio.cpp publish no CUDA build for Linux: there `device = "cuda"` installs the CPU build and says so (`bolo components install` prints a note, the job carries it as a warning); use `--flavour vulkan` (llama-server; it needs the Vulkan loader, `libvulkan1`, and a Vulkan driver: Mesa for AMD and Intel, NVIDIA's own driver) or point this setting at a CUDA build of your own (or the one `--flavour cpu\|cuda\|vulkan` names), each build in its own folder (`components/llama-server/<version>/cpu`, `.../cuda`, `.../vulkan`); the one matching `device` is started |
| `tts_server_args` | `[]` | | extra command-line arguments for every server bolo starts (e.g. `["-ngl", "20"]` for llama-server, which bolo then leaves alone; without one bolo adds `-ngl N` itself when `device` is not `cpu`, N being the layers that fit in free accelerator memory) |
| `tts_server_start_timeout_s` | `300` | | how long a started server may take to answer `GET /health` (loading its model) |
| `tontaube_dir` | unset | | a checkout of [craitech/tontaube](https://github.com/craitech/tontaube) (release v1.1.0) that bolo runs with `uv run --python 3.12 --frozen tontaube serve` for the Tontaube models. Unset: the one `bolo components install tontaube` cloned into `<data_dir>/components/`. `uv` must be on `PATH` |
| `tontaube_start_timeout_s` | `3600` | | how long a Tontaube server bolo started may take to answer `GET /readyz`: its first start installs its dependencies (PyTorch and more, several GB) and prepares the model |
| `tts_server_timeout_s` | `300` | | longest wait for the next bytes of a synthesis answer |
| `providers` | `{}` | | cloud providers for `engine: remote` models: one `[providers.<name>]` table each with `kind` (`openai-compatible`, `sarvam`, `elevenlabs`, `deepgram`; implied for those four names, `openai` meaning `openai-compatible`), `base_url` (default: the vendor's), `api_key`, `options` and `prices`. In the environment: `BOLO_PROVIDERS` (the table as JSON) or `BOLO_PROVIDER_<NAME>_API_KEY` / `_BASE_URL` / `_KIND` (these add to the file's table). Keys are never logged or returned by any API. Paid cloud APIs; audio leaves this server. See [providers.md](providers.md) |
| `fallbacks` | `[]` | | `"<model>=<fallback>[,<fallback>…]"` per model: what answers instead when the model's engine is unavailable, its queue is full, or its provider fails (5xx, timeout, rate limit). Local and cloud models can be mixed (`"whisper-base=openai-whisper-1"`, `"sarvam-saaras-v4=whisper-base"`). A pin through `PUT /api/v1/models/{id}/settings` wins; a catalog entry's `options.fallback` is the default. See providers.md, "Fallback" |
| `allow_provider_edit` | unset | | add, change and remove providers through the API and the console (stored in `<data_dir>/providers.json`, mode 0600). Unset: allowed on a loopback bind or with `api_key`; never on a non-loopback bind without one |
| `provider_timeout_s` | `60` | | longest wait for a cloud provider's answer, or for its next bytes while streaming (a provider or model option `timeout_s` overrides it) |
| `telephony` | `{}` | | phone calls: a `[telephony]` table with `public_base_url` (the `https://` address providers reach this server at), `validate_signatures` (default `true`; the provider endpoints stay closed until requests can be checked, unless this is `false`), `default_agent`, `call_buffer` (call records kept, default 200), and the tables `[telephony.providers.<twilio\|plivo\|telnyx\|exotel>]` (credentials), `[telephony.agents.<name>]` (voice agents) and `[telephony.numbers]` (`"<called number>" = "<agent>"`). In the environment: `BOLO_TELEPHONY` (the table as JSON), `BOLO_TELEPHONY_PUBLIC_BASE_URL`, `BOLO_TELEPHONY_DEFAULT_AGENT`, `BOLO_TELEPHONY_VALIDATE_SIGNATURES`, `BOLO_TELEPHONY_<PROVIDER>_<FIELD>` (these add to the file's table). Credentials are never logged or returned by any API. Every field: [telephony.md](telephony.md) |
| `hardware_warnings` | `true` | | attach a notice when a model that is pulled or loaded is unlikely to fit this machine or to run in real time: a `warnings` list in the answers of `POST /api/v1/models/pull` and `/load`, on the job and on request traces, a log line, a `warning:` line from `bolo pull`, a notice in the console. Notices only: nothing is ever refused, hidden or slowed down, with the setting on or off. `false` removes the notices; the `fit` on catalog entries and `bolo recommend` stay. See [recommendations.md](recommendations.md) |
| `sip` | `false` | | listen for inbound SIP calls from a trunk or PBX (UDP, G.711 RTP; needs `bolo-speech[sip]`). See telephony.md, "SIP" |
| `sip_host`, `sip_port` | `0.0.0.0`, `5060` | | the SIP listening address (UDP) |
| `sip_public_ip` | unset | | the media address written into SDP answers; unset: the local address the INVITE reached. Set it behind NAT |
| `sip_allowed_ips` | `[]` | | source addresses and networks (`"203.0.113.10"`, `"10.0.0.0/8"`) allowed to send SIP requests; empty: loopback only. There is no other authentication |
| `sip_rtp_port_min`, `sip_rtp_port_max` | `40000`, `40100` | | UDP port range for RTP, one port per call |

Four more environment variables are read outside the settings object. `BOLO_UCCHARAN=off` hides an installed
uccharan (to test the espeak-ng route). For
the optional eSpeak NG component: `BOLO_ESPEAK_LIBRARY` (with
`BOLO_ESPEAK_DATA`) points at a specific libespeak-ng and takes precedence;
otherwise an `espeak-ng` executable on `PATH` is used, then the
`espeakng-loader` package. `BOLO_ESPEAK` (`auto`, `executable`, `library`,
`off`) restricts that search. See
[models.md](models.md#licences-and-the-espeak-ng-question).

## Example file

```toml
# bolo.toml
host = "0.0.0.0"
port = 8000
api_key = "change-me"
cors_origins = ["https://console.example.com"]

[models]
default_stt = "faster-whisper-small"
default_realtime_stt = "zipformer-en-20m-streaming"
default_tts = "kokoro-v1.0-int8"
preload = ["silero-vad", "zipformer-en-20m-streaming"]

[lifecycle]
keep_alive = "10m"                    # every model, unless a task or a request says otherwise
stt_ttl_s = 900
memory_budget_mb = 3000

# with an accelerator (device = "cuda"): models that do not fit in free VRAM go on the CPU
device = "cuda"
gpus = [0, 1]                         # the GPUs bolo may use (after CUDA_VISIBLE_DEVICES)
[gpu]
reserve_mb = 1024                     # gpu_reserve_mb
memory_budget_mb = 8000               # gpu_memory_budget_mb
spread = false                        # gpu_spread

[scheduler]
threads = 4
num_parallel = 2                      # requests each model works on at once (OLLAMA_NUM_PARALLEL)
max_batch = 8
max_queue = 32

[llama]                               # llama-servers bolo starts: llama_kv_cache_type, llama_flash_attention
kv_cache_type = "q8_0"
flash_attention = "on"

[log]
file = "auto"                         # log_file: also write the platform's log file (see "Logs")

[tts]
# a llama-server you run yourself serves every llama-server model; audio.cpp models are started by bolo
servers = ["llama-server=http://127.0.0.1:8080"]

# cloud providers (paid cloud APIs; audio leaves this server): docs/providers.md
[providers.sarvam]
api_key = "…"                         # or BOLO_PROVIDER_SARVAM_API_KEY in the environment

[providers.sarvam.prices]             # optional, for cost estimates: what you pay
"saaras:v4" = { per_hour = 30, currency = "INR" }

# phone calls: docs/telephony.md
[telephony]
public_base_url = "https://voice.example.org"
default_agent = "front-desk"

[telephony.providers.twilio]
auth_token = "…"                      # or BOLO_TELEPHONY_TWILIO_AUTH_TOKEN in the environment

[telephony.agents.front-desk]
instructions = "You answer the phone for Example Clinic."
greeting = "Example Clinic, how can I help?"

[telephony.numbers]
"+14155550100" = "front-desk"
```

`fallbacks = ["whisper-base=openai-whisper-1"]` and the other top-level
settings go above the first `[section]` (TOML reads everything after a
section header as part of that section).

Section names are mostly for your own orientation: `[models]`, `[scheduler]`
and so on are flattened, so every key must be one of the settings above.
The one rule: if `<section>_<key>` is a setting, that is what is set, so
`[llm]` with `base_url`, `model`, `api_key` means `llm_base_url`, `llm_model`,
`llm_api_key`. `[providers.<name>]` tables and the `[telephony]` table are
the exceptions: they stay nested and make up the `providers` and `telephony`
settings.
An unknown key is an error.

## speaches environment variables

To make switching painless, these `speaches` names are read as fallbacks
(a `BOLO_*` variable for the same setting wins):

| speaches | bolo setting |
| --- | --- |
| `API_KEY` | `api_key` |
| `PRELOAD_MODELS` | `preload` |
| `STT_MODEL_TTL`, `TTS_MODEL_TTL`, `VAD_MODEL_TTL` | `stt_ttl_s`, `tts_ttl_s`, `vad_ttl_s` |
| `ALLOW_ORIGINS` | `cors_origins` |
| `UVICORN_HOST`, `UVICORN_PORT` | `host`, `port` |
| `LOG_LEVEL` | `log_level` |
| `ENABLE_UI` | `ui` |
| `WHISPER__CPU_THREADS`, `WHISPER__NUM_WORKERS` | `threads`, `workers` |
| `HF_TOKEN` | `hf_token` |
| `CHAT_COMPLETION_BASE_URL`, `CHAT_COMPLETION_API_KEY` | `llm_base_url`, `llm_api_key` |

Note that speaches binds `0.0.0.0` by default and bolo binds `127.0.0.1`: set
`UVICORN_HOST=0.0.0.0` (or `BOLO_HOST`) in a container.

## GPU and CPU placement

With an accelerator (`device` = `cuda`, `cuda:1`, `coreml`, `auto`, ...), bolo decides per model, at load time, where
it goes, the way Ollama does:

1. Estimate the model's memory: its files on disk × 1.2, or the catalog's `options.hardware.vram_gb` when larger.
2. Read the free accelerator memory (NVIDIA: NVML through the optional `nvidia-ml-py` package, else `nvidia-smi`;
   Apple Silicon: the machine's available memory, since it is shared), subtract `gpu_reserve_mb`, and cap it by
   what is left of `gpu_memory_budget_mb`.
3. If the model fits, it goes on the accelerator. If it would fit after unloading idle GPU-resident models, those
   are unloaded first (least recently used first). Otherwise it is split where its runtime allows, or placed on
   the CPU, with a notice in the log, in the load answer and in `bolo ps`. Nothing is refused.
4. After the load, bolo checks that the runtime really started the accelerator. ONNX Runtime falls back to the CPU
   on its own when a provider does not start (a CUDA DLL not found, say) and says so only on standard error; bolo
   compares each session's providers with the ones it asked for, and CTranslate2 (faster-whisper) is warmed up on
   the GPU and loaded again on the CPU when a CUDA library is missing. Either way the model runs on the CPU: the log,
   the model's notice, the "loaded" event and `bolo ps` say so with the missing library and the install command,
   and its placement (`device`, `processor`, `provider_fallback` in `GET /api/v1/ps`) says CPU. `bolo doctor`
   checks the same before anything is loaded (see "CUDA on Windows").

How a model can be split depends on its runtime:

| Runtime | Placement |
| --- | --- |
| llama.cpp (`tts-server` models on llama-server, Orpheus-family voices, the speech-to-speech server) | by layers: `-ngl N` from the free memory, the GGUF's layer count (read from its header) and its size, the projector (`-mm`) first, plus 10% of the model for the KV cache and buffers. An `-ngl` you set yourself wins |
| ONNX Runtime models made of several graphs (DiCoW ONNX: encoder and decoder; LFM3-Audio: four graphs) | by graph: the largest graphs go on the accelerator while they fit, the rest run on the CPU; with `onnx_split = "auto"` the largest graph left over is cut in two with the memory that is left |
| single-graph ONNX Runtime models (Nemotron diarization, the end-of-turn model) | with `onnx_split = "auto"`: cut in two at a layer boundary, the front (as many weights as fit) on the accelerator, the back on the CPU; otherwise whole on one device |
| bolo-torch plugin models (PyTorch) | the plugin gets accelerate's `device_map="auto"` with a `max_memory` from the free memory and reports back how it split the weights (docs/plugins.md); models whose loader supports it (Orpheus SNAC so far) are split by layers, others load whole |
| sherpa-onnx | whole model on one device: sherpa-onnx builds its sessions itself, with one provider per model |
| faster-whisper (CTranslate2) | whole model on one device (CTranslate2 cannot split a model); CUDA only, so `coreml` runs it on the CPU |
| the SNAC decoder of Orpheus-family voices | whole, on the device of the voice's layers |
| kokoro-onnx, Moonshine, whisper.cpp, cloud and operator-run servers | the CPU (or wherever the operator's server runs) |

`device = "auto"` means the accelerator bolo can read: the first NVIDIA GPU (NVML or nvidia-smi) or Apple Silicon's
GPU, else the CPU; for ROCm, DirectML or OpenVINO name the device. The ONNX Runtime engines (Nemotron diarization,
DiCoW ONNX, the end-of-turn model, in-process speech-to-speech) take `auto` as the first provider the installed
`onnxruntime` has, in the order CUDA (with an NVIDIA GPU found), ROCm, DirectML, Core ML: with `onnxruntime-directml`
that is `dml` on Windows' high-performance adapter (the discrete GPU of a laptop), even with an NVIDIA GPU, while
faster-whisper and a CUDA build of sherpa-onnx still get `cuda`; with plain `onnxruntime` (no accelerator) they are
placed on the CPU. A model placed on an accelerator its runtime does not start (a CUDA library missing, sherpa-onnx
built without CUDA saying "Fallback to cpu!") is moved to the CPU after its load, with a notice: its placement and
`bolo ps` say `cpu` and "100% CPU". When the free memory cannot be read (those, or a
GPU without driver tools), a model goes on the device as before, and `bolo ps` shows the plan rather than a measurement. After each load bolo measures what the
load took: the growth of its own resident memory and that of the processes it started (llama-server, audio.cpp,
plugins), and the growth of their accelerator memory (NVML's per-process figures, else the whole device's). On Apple
Silicon the accelerator's share cannot be told apart from ordinary memory; it is reported as planned.

**Windows and WDDM.** On Windows a GeForce card (and most others) runs under the WDDM driver model, where the NVIDIA
driver keeps no per-process memory figures: NVML reports them as not available (or 0) and `nvidia-smi` as "N/A".
bolo takes that as unknown, never as 0. It measures a load instead by what DXGI says bolo's own process uses on that
GPU (`QueryVideoMemoryInfo`, for runtimes in bolo's process: ONNX Runtime, sherpa-onnx, CTranslate2), else by the
growth of the GPU's used memory before and after the load (llama-server and other server processes), with such
loads made one at a time so two are not counted together; a program outside bolo that takes GPU memory during a
load is counted in that load. When neither grows, the memory is unknown. Either way a model placed on the GPU
shows as on it: `processor` follows the placement, `vram_measured` is false, `vram_source` is `"plan"`, `size_vram`
is the planned figure, and `bolo ps` prints `100% GPU ?` with a note ("unknown VRAM"); the console's Loaded models
panel shows "VRAM unknown". Only a measurement that found nothing on the GPU (a runtime that fell back to the CPU)
reads "100% CPU". `/api/v1/ps` gives each GPU's `driver_model` ("WDDM", "TCC") and, under WDDM, `process_bytes` is
null and `own_bytes` is bolo's own process's use. A card in TCC mode (`nvidia-smi -dm 1`, not GeForce) reports
per-process figures as on Linux.

**Cutting a graph in two** (`onnx_split`, `bolo/onnx_split.py`). ONNX graphs list their nodes in order, so any
point in the list divides a graph into a front and a back. bolo picks a layer boundary (a point where few tensors
cross, typically the hidden state and what every layer shares) that puts the most weights in the front within the
free memory. Nodes that only turn weights into weights (the DequantizeLinear of a quantized weight, a Constant) go
with the half that reads them. A point is refused when an If, Loop or Scan would read a value from the other half
through its subgraph. Weights both halves read are copied into both. The two halves run as two ONNX Runtime
sessions: the crossing tensors come back to the host between them (a few MB per call for a speech model). The
halves are written to `<model dir>/.bolo-split/<model sha>-<cut>/` once and reused while the model file is the same.
If a split cannot be made or opened, the model runs whole on the CPU with a warning. Tested on CPU only: the halves
answer as the whole graph does (bit for bit with ONNX Runtime's graph optimisations off; within float rounding
with them on, since ONNX Runtime fuses nodes differently on each side of the cut).

`memory_budget_mb` counts the RAM side of each model, `gpu_memory_budget_mb` the accelerator side.

None of this runs on a real GPU in bolo's tests: it is tested with fake NVML readings and made-up memory figures.
On real hardware, whole models were placed on one laptop's GPU ([status.md](status.md#a-windows-11-laptop)).

### Several GPUs

On a machine with several NVIDIA GPUs:

* **Which GPUs.** `gpus = [0, 2]` (`BOLO_GPUS=0,2`, `bolo serve --gpus 0,2`) limits bolo to those; empty means all
  of them. The numbers are CUDA's as bolo's process sees them: with `CUDA_VISIBLE_DEVICES=2,3` set, GPU 2 is `0`
  and GPU 3 is `1`, exactly as for `cuda:N` (bolo reads `CUDA_VISIBLE_DEVICES`, by index or `GPU-<uuid>`, and maps
  NVML's numbers to CUDA's; `/api/v1/ps` lists both, `index` and `physical_index`). The servers and plugins bolo
  starts inherit the variable, so the numbers mean the same to them.
* **Which GPU for a model.** `device = "cuda"` (or `auto`) names no GPU: each model goes on the allowed GPU with
  the most memory available when it is loaded (free memory minus `gpu_reserve_mb`, within what is left of
  `gpu_memory_budget_mb`, which counts all GPUs together). `cuda:1`, set or pinned for a model, means that GPU.
* **Spreading a model.** A llama.cpp model that does not fit whole on the best GPU is spread over the allowed GPUs
  by layers, in proportion to the memory each has, when that puts more layers on GPUs than one GPU would (Ollama's
  rule); the GPU with the most memory is the main one (it also holds the projector and the KV cache share). bolo
  starts llama-server with `--split-mode layer --tensor-split a,b,... --main-gpu N`. A model that fits on one GPU
  gets `--split-mode none --main-gpu N`, because llama-server would otherwise spread it over every visible GPU.
  `gpu_spread = true` (Ollama's `OLLAMA_SCHED_SPREAD`) spreads every llama.cpp model over all allowed GPUs, which
  can help a model that is slow on one GPU and costs PCIe traffic between them. Arguments you give llama-server
  yourself (`-sm`, `-mg`, `-ts`, `--device`) win, and bolo then leaves the GPUs to them.
* **Other runtimes take one GPU each.** ONNX Runtime and faster-whisper models get the chosen GPU's number
  (`cuda:N`); bolo-torch plugin models get it in accelerate's `max_memory`. sherpa-onnx cannot choose a GPU and
  always uses GPU 0, so its models are placed on GPU 0 or, when `gpus` leaves GPU 0 out, on the CPU with a notice.
  audio.cpp servers get the `cuda` backend and use their default GPU.
* **What you see.** `bolo ps` has a GPU column (`0`, `0,1` for a spread model, `-` on the CPU) and marks GPUs the
  `gpus` setting leaves out; `/api/v1/ps` gives `gpus` per model, `placement.gpu_bytes_by_gpu`,
  `placement.tensor_split` and `main_gpu` for a spread model, and per GPU `allowed` and `models_vram_bytes`. The
  console's Loaded models panel shows a memory bar per GPU.

None of this has run on a machine with two NVIDIA GPUs (the one laptop it was checked on has one NVIDIA GPU and integrated AMD graphics): it is tested with a fake NVML of several made-up GPUs and with
llama-server command lines that are built but not run.

### llama.cpp speed and memory

Four settings reach every llama-server bolo starts (tts-server and Orpheus-family voices, and the
speech-to-speech server). Each can be set for all of them (`llama_flash_attention`, `llama_kv_cache_type`,
`llama_context_size`, `llama_batch_size`), per catalog entry (`options.flash_attention`, `kv_cache_type`,
`context_size`, `batch_size`) or per model at run time (`PUT /api/v1/models/{id}/settings`, or
`bolo models config <id> --set kv_cache_type=q8_0`). The pin wins over the catalog, the catalog over the setting,
and an argument you wrote yourself (`tts_server_args`, `s2s_server_args`, a catalog `server_args`: `-fa`,
`--cache-type-k`, `-c`, `-b` in any spelling) wins over all of them. A change applies when the model is next
loaded (`bolo stop <model>` unloads it).

| Setting | llama-server | What it trades |
| --- | --- | --- |
| flash attention: `on`, `off`, `auto` | `-fa` | the same attention computed in fewer passes over memory: less memory for the KV cache's work buffers and faster prompt processing on CUDA and Metal. Output is the same up to float rounding. For a voice the prompts are short, so the gain is small: about 0 to 15% faster generation on a recent NVIDIA GPU [estimate]; on a CPU it can be slower [estimate]. llama-server's default `auto` turns it on where the backend supports it |
| KV cache type: `f16`, `q8_0`, `q4_0` | `--cache-type-k`, `--cache-type-v` | the memory of the cache that holds every token so far. A 3B Llama-architecture voice (Orpheus 3B: 28 layers, 8 KV heads of 128) needs 112 KB per token in `f16`, so about 900 MB for an 8192-token context; `q8_0` about 500 MB, `q4_0` about 260 MB (arithmetic, not measured). `q8_0` is close to lossless for text models; `q4_0` loses more. For voices, whose tokens are audio codes, nobody has measured it here: expect `q8_0` to sound the same and `q4_0` to risk mispronunciations and artefacts on long sentences [estimate]. Speed is about the same [estimate]. A quantized V cache needs flash attention, so bolo adds `-fa on` with one unless flash attention is set |
| context size | `-c` | memory (the KV cache grows with it) against the longest prompt plus generated audio a request can have. An Orpheus-family voice writes up to `max_new_tokens` (2048) audio tokens per piece of text, so keep each slot at 3072 or more [estimate]. With `--parallel N` some llama-server builds divide the context between the slots |
| batch size | `-b` | prompt tokens processed per step: a larger batch processes long prompts faster and needs a larger compute buffer. Voice prompts are short, so it rarely matters [estimate] |

None of these has been measured on a GPU with bolo yet; the values marked [estimate] are expectations from
llama.cpp's own documentation and the arithmetic above.

### Parallel requests and queues

Every loaded model has a queue (`bolo/scheduler.py`): requests wait there, `num_parallel` of them run at once,
compatible requests waiting together run as one micro-batch where the engine can (`max_batch`, `batch_wait_ms`),
and when `max_queue` requests are already waiting the next one is refused at once with `429` and a `Retry-After`
(the backpressure), rather than waiting without limit; a request that waited longer than `queue_timeout_s` fails.

| Ollama | bolo | Notes |
| --- | --- | --- |
| `OLLAMA_NUM_PARALLEL` | `num_parallel` (the same knob as the older `workers`, used when `num_parallel` is `0`) | per model; a catalog entry's `options.num_parallel` (or `options.workers`) or a pin (`--set num_parallel=2`) wins. A llama-server bolo starts for a voice gets `--parallel N` with it (above 1), so that the server holds N requests at once; the speech-to-speech server keeps `s2s_parallel` |
| `OLLAMA_MAX_QUEUE` | `max_queue` | per model in bolo (Ollama counts the whole server); `options.max_queue` or a pin wins |
| (micro-batching) | `max_batch`, `batch_wait_ms` | nothing like it in Ollama |

Changes apply when the model is next loaded. `bolo ps -v` shows per model ACTIVE (requests and sessions using
it), QUEUED (requests waiting) and PARALLEL; `/api/v1/ps` has `running`, `queued`, `num_parallel` and `max_queue`
per model. More in parallel helps only while the hardware has room: on a CPU, each request's threads (`threads`)
share the cores, so `num_parallel` × `threads` should stay near the number of cores.

## Logs

`bolo serve` always logs to its standard error. With `log_file` it also writes a file, rotated at `log_max_mb`
(10 MB) with `log_backups` (3) older files kept. `log_file = "auto"` puts it in the platform's place:

| Where bolo runs | Where the log is |
| --- | --- |
| Linux, `bolo serve` in a terminal | the terminal; with `log_file = "auto"`: `$XDG_STATE_HOME/bolo/logs/server.log` (`~/.local/state/bolo/logs/server.log`) |
| macOS | the terminal; with `log_file = "auto"`: `~/Library/Logs/bolo/server.log` (Console.app shows it) |
| Windows | the console; with `log_file = "auto"`: `%LOCALAPPDATA%\bolo\logs\server.log`. Run at log-on or as a service (Task Scheduler, NSSM, WinSW: [deploy/README.md](../deploy/README.md#windows-at-log-on-at-start-up-or-as-a-service); restarted after a crash: [below](#at-log-on-restarted-after-a-crash)), set `BOLO_LOG_FILE=auto`: bolo cannot tell it is a service there |
| the systemd unit (`deploy/systemd/bolo.service`) | `journalctl -u bolo`, and `/var/lib/bolo/logs/server.log` (the unit sets `BOLO_LOG_FILE`; without it, a service writes the `auto` place of its user) |
| launchd (macOS) | the job's `StandardErrorPath`, and `~/Library/Logs/bolo/server.log` |
| Docker (`deploy/Dockerfile`) | `docker logs <container>`; no file unless you set `BOLO_LOG_FILE=/data/logs/server.log` (in the `/data` volume) |

Unset, `log_file` is off, except when `bolo serve` runs under systemd (it sets `INVOCATION_ID`) or launchd (it sets
`XPC_SERVICE_NAME` to the job's label), where it means `auto`; `log_file = "off"` switches that off too.

`bolo logs` prints the last 100 lines of the log file (`--lines N`, `0` for all; `-f` keeps following it, across
rotations). It asks the running server (`--url`, default `$BOLO_URL` or `http://127.0.0.1:8000`) which file it
writes (`GET /api/v1/info` has `log_file`), else uses the settings, else the platform's place when a file is there.
`bolo doctor` says where the logs are.

## OpenTelemetry

`pip install "bolo-speech[otel]"` (opentelemetry-sdk and the OTLP/HTTP exporter, Apache-2.0) and point bolo at a
collector:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4318 OTEL_SERVICE_NAME=bolo-eu1 bolo serve
```

Each finished request becomes one span, `bolo.stt`, `bolo.tts`, `bolo.agent` (a voice-agent turn), `bolo.diarization`,
`bolo.embedding` or `bolo.vad`, with a child span per phase of its trace (`queue`, `resample`, `decode`,
`synthesize_first`, `llm_first_token` ...), placed where the phase ran. The request span's attributes:
`bolo.request_id` (the id `/api/v1/requests/{id}` knows), `bolo.kind`, `bolo.transport`, `bolo.model`, `bolo.device`
(where the model was loaded: `cpu`, `cuda:0`, `remote` ...; absent when it was not), `bolo.status`, the sizes
`bolo.audio_seconds` and `bolo.chars`, `bolo.language`, the latencies bolo measured (`bolo.total_ms`, `bolo.ttfa_ms`,
`bolo.final_ms`, `bolo.first_partial_ms`, `bolo.rtf`, `bolo.batch_size`), numeric extras such as `bolo.first_token_ms`,
and the model ids an agent turn used (`bolo.stt_model`, `bolo.voice`, `bolo.pipeline`). A failed request's span has
status `ERROR` and `error.type`. Transcripts and the text that was spoken are not exported.

What the standard variables do: `OTEL_SERVICE_NAME` names the service (default `bolo`), `OTEL_RESOURCE_ATTRIBUTES`
adds resource attributes, `OTEL_EXPORTER_OTLP_HEADERS` sends headers (an API key for a hosted backend),
`OTEL_TRACES_SAMPLER` samples, `OTEL_BSP_*` tunes the batching, `OTEL_SDK_DISABLED=true` switches export off.
`OTEL_EXPORTER_OTLP_PROTOCOL` is `http/protobuf` (port 4318) by default; `grpc` (port 4317, what speaches' examples
use) also needs `opentelemetry-exporter-otlp-proto-grpc`. The setting `otel` (`BOLO_OTEL`) forces export on or off
whatever the variables say.

The spans are built from the trace bolo keeps anyway, after the request has finished, and sent in batches off the
request path; an unreachable collector costs a log line, not latency. `/metrics`, `/api/v1/requests` and the event
stream are unchanged. When export is switched on but the extra is missing, bolo logs one warning naming the package
and carries on without it. Only traces are exported: metrics stay on Prometheus `/metrics`, and the log is not sent.

## Windows

What differs on Windows (CI runs the server's tests on `windows-latest`; the CPU, CUDA, DirectML and Vulkan notes below were found on a Windows 11 laptop with an RTX 3060 Laptop GPU and Radeon integrated graphics, whose numbers are in [status.md](status.md#a-windows-11-laptop)):

| | |
| --- | --- |
| install | PowerShell or cmd: `uv tool install "bolo-speech[all]"` (double quotes: cmd.exe does not treat single quotes as quotes) |
| models (`data_dir`) | `%USERPROFILE%\.cache\bolo` (`C:\Users\<you>\.cache\bolo`), the same place as on Linux and macOS, not `%LOCALAPPDATA%`: bolo keeps it so that existing installs never lose their models. To keep models elsewhere set `BOLO_DATA_DIR` (for example `%LOCALAPPDATA%\bolo`), after moving the folder there if you want to keep what you downloaded. `XDG_CACHE_HOME` is honoured too |
| settings file | `.\bolo.toml`, else `%USERPROFILE%\.config\bolo\config.toml` (or `--config`, `BOLO_CONFIG`). Write Windows paths in single quotes (`data_dir = 'C:\Users\you\bolo'`, a TOML literal string) or with forward slashes (`"C:/Users/you/bolo"`): in double quotes a backslash starts an escape, so `"C:\Users\..."` is an error and `"C:\temp"` holds a tab |
| log file | `log_file = "auto"`: `%LOCALAPPDATA%\bolo\logs\server.log`. `bolo logs -f` opens it so that the rotation can still rename it; when another program blocks the rename, bolo copies the file to `server.log.1` and empties it instead, and `bolo logs -f` follows that too (the lines written before the copy are read from `server.log.1`, then the emptied file from its start) |
| uccharan's language packs | `%USERPROFILE%\.cache\uccharan\<version>\` (uccharan's own cache) |
| environment variables | PowerShell: `$env:BOLO_DEVICE = "dml"` (this session), `[Environment]::SetEnvironmentVariable("BOLO_DEVICE", "dml", "User")` (from the next start on); cmd: `set BOLO_DEVICE=dml` |
| text output | `bolo` reads and writes UTF-8 on pipes and files, so Hindi and other scripts survive `bolo transcribe x.wav > out.txt` and `bolo speak - -o out.wav < text.txt`. Windows PowerShell 5.1 uses other encodings at both ends of a pipe: it decodes what a program writes with `[Console]::OutputEncoding` (the console's code page) and encodes what it pipes into a program with `$OutputEncoding` (ASCII). Before piping text out of `bolo` (or `curl.exe`) into another PowerShell command, or into `bolo` (`"नमस्ते" \| bolo speak - -o out.wav`), set both, once per session or in your profile: `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` and `$OutputEncoding = [System.Text.Encoding]::UTF8`. cmd and PowerShell 7 need nothing |
| `curl.exe` and JSON | Windows PowerShell 5.1 drops the double quotes inside an argument it passes to a program, so `curl.exe -d '{"model": "tts-1"}'` sends `{model: tts-1}` and the server answers 400. In 5.1 escape them (`-d '{\"model\": \"tts-1\"}'`), put the JSON in a file (`-d "@body.json"`), or use `Invoke-RestMethod` as in the README. PowerShell 7.3 and later pass them as written; cmd needs `-d "{\"model\": \"tts-1\"}"` |
| firewall | with `--host 0.0.0.0` the Windows Defender Firewall prompt names `python.exe` ("Python"), not `bolo.exe`: uv's `bolo.exe` is a launcher that starts uv's Python interpreter, which is what listens. Prefer a rule for the port ([deploy/README.md](../deploy/README.md#windows-at-log-on-at-start-up-or-as-a-service)) to one for that interpreter, whose path changes when uv installs another Python |
| local model folders | `bolo pull <folder>` / `file:` link the folder into the store: a directory symlink where allowed (Developer Mode or administrator), else a directory junction (no rights needed, same volume or not, local disks only), else a copy, with a notice |
| `llama-server`, `audio.cpp` | `bolo components install llama-server` / `audio.cpp` install the pinned Windows release: the CPU build (with every CPU back end, from SSE4.2 to AVX-512, chosen at start-up), the CUDA 12.4 build with its CUDA runtime DLLs for `device = "cuda"`, the Vulkan build for `device = "vulkan"`. Each build goes in its own folder, so `BOLO_DEVICE=cuda bolo components install llama-server` after the CPU build installs the CUDA build next to it (`--flavour cpu\|cuda\|vulkan` picks one explicitly); bolo starts the build matching `device` (with `auto`, a GPU build when one is installed), and `bolo components list` shows which one runs. Tontaube is Linux only: `components list` shows it as unavailable with the reason, and `components install tontaube` says "Linux only (use WSL2)"; run bolo in WSL2, or run `tontaube serve` there and set `tts_servers` |
| Vulkan build on AMD + NVIDIA laptops | on a laptop with AMD graphics (Radeon integrated graphics next to an NVIDIA GPU) the Vulkan build of llama-server hung while it listed the GPUs (`llama-server --list-devices` never answered; Windows logged `RADAR_PRE_LEAK_64` for `llama-server.exe`): AMD's switchable-graphics Vulkan layer, which the Vulkan loader loads into every Vulkan program, is the cause. When bolo starts a Vulkan build of llama-server or audio.cpp on Windows and DXGI lists an AMD adapter, it sets `DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1=1` in that server's environment (logged at start), unless you set that variable yourself (any value: set `0` to keep the layer). `bolo doctor`'s `--list-devices` question gets the same environment and gives up after 20 s with a warning that names the variable, and a managed server that never answers `GET /health` says the same when it is a Vulkan build. Running `llama-server.exe` yourself: `set DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1=1` (cmd) or `$env:DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1 = "1"` (PowerShell) first |
| servers bolo starts | llama-server, audio.cpp and plugin workers each run in a Job Object that Windows closes when bolo ends, however it ends (Task Manager, closing the console, a service stop), which ends them and whatever they started. bolo stops one by sending it `CTRL_BREAK_EVENT` and ends the Job Object 10 s later (at once when bolo has no console, as a service). `plugin_nice` > 0 means `BELOW_NORMAL_PRIORITY_CLASS`; `plugin_memory_limit_mb` limits each worker's committed memory (POSIX limits its address space) |

**DirectML** (`device = "dml"`, or `dml:N` to pin adapter N in DXGI's order: `bolo doctor` and
`GET /api/v1/hardware` list them as `... (dml:N)`, and an NVIDIA card bolo reaches through CUDA shows its DirectML
number as "DirectML adapter dml:N"). `dml` with no number runs on the adapter Windows calls high-performance
(`IDXGIFactory6::EnumAdapterByGpuPreference`; without it, the adapter with the most dedicated memory), not on
DXGI's first adapter, which is DirectML's own default: on a laptop with integrated graphics and a discrete GPU,
Windows lists the integrated GPU (the one driving the display) first, so adapter 0 is the slow one. Placement and
`bolo ps` read the memory of the same adapter. Integrated graphics (Intel UHD and Iris, an AMD APU's Radeon, or
512 MB of dedicated memory or less) never count as the machine's best GPU: on a laptop with an NVIDIA card,
`bolo doctor` and `bolo recommend` suggest `device = "cuda"`, and point out a `dml:N` that pins the integrated GPU.
Only the engines that open ONNX Runtime sessions themselves use
it: Nemotron diarization, DiCoW ONNX, the SNAC decoder of the Orpheus-family voices, the end-of-turn model and
in-process speech-to-speech (`lfm3-audio-2b-onnx`), except the end-of-turn model (`smart-turn-v3`): DirectML fails
to run it on every adapter (its Reshape node `node_view_1`, error 8007023E), so with `dml` it is placed on the CPU,
with a notice, where it takes 13-39 ms (8 MB). `device = "auto"` with `onnxruntime-directml` installed means `dml`
for these engines. sherpa-onnx's wheels have no DirectML provider and CTranslate2
(faster-whisper) has none at all, so those models stay on the CPU (bolo says so once); llama-server and audio.cpp
have no DirectML back end (use `cuda` or `vulkan` for them). DirectML needs the `onnxruntime-directml` package in
place of `onnxruntime`. All the ONNX Runtime packages install the same `onnxruntime` module and overwrite each other,
and faster-whisper (in `all`) asks for plain `onnxruntime`, so install DirectML last, and again after any later
`pip install` or `uv tool upgrade` that brings `onnxruntime` back:

```powershell
uv tool install "bolo-speech[all,turn]"
$py = "$(uv tool dir)\bolo-speech\Scripts\python.exe"     # the Python of uv's bolo-speech tool
uv pip uninstall --python $py onnxruntime onnxruntime-gpu onnxruntime-directml
uv pip install --python $py onnxruntime-directml
bolo doctor        # "onnxruntime: onnxruntime-directml 1.24.4; providers Dml, CPU"
```

In a plain virtual environment: `pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml`, then
`pip install onnxruntime-directml`. (Run on a Windows 11 laptop with onnxruntime-directml 1.24.4: see [status](status.md).) `bolo doctor` warns when more than one of these packages is installed, and when
`device` asks for a provider the installed ONNX Runtime lacks. There is no `directml` extra: pip and uv cannot
replace a dependency's `onnxruntime` with `onnxruntime-directml`, so an extra would install both. On 2026-11-16
`onnxruntime-directml` was at 1.24.4 while `onnxruntime` was at 1.32.0; the in-process speech-to-speech model needs
1.20 or newer, which 1.24.4 is. With DirectML, free GPU memory comes from DXGI (the budget Windows gives bolo's
process, minus what it uses), so placement and `bolo ps` have a figure for dedicated GPUs; for integrated graphics
the free memory is unknown and `bolo ps` shows the plan.

### CUDA on Windows

An NVIDIA GPU on Windows needs CUDA libraries for two runtimes, of two CUDA majors. Found on an RTX 3060 Laptop GPU
(Windows 11, driver 621.24):

- **ONNX Runtime** (Nemotron diarization, DiCoW ONNX, the Orpheus-family decoder, the end-of-turn model, in-process
  speech-to-speech): `onnxruntime-gpu` 1.32.0 from PyPI is built for CUDA 13 and cuDNN 9. Without them its CUDA
  provider does not start (`... depends on "cublasLt64_13.dll" which is missing (Error 126)`), and ONNX Runtime
  quietly runs the model on the CPU. NVIDIA's wheels (`onnxruntime-gpu[cuda,cudnn]`) bring the DLLs, into
  `site-packages\nvidia\cu13\bin\x86_64`, `nvidia\cudnn\bin` and others that Windows does not search; bolo adds
  those folders and calls `onnxruntime.preload_dlls()` before its first CUDA session.
- **faster-whisper** (CTranslate2 4.9): CUDA 12's cuBLAS and cuDNN 9 (`Library cublas64_12.dll is not found or
  cannot be loaded`). `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` bring them (`nvidia\cublas\bin`,
  `nvidia\cudnn\bin`); bolo adds those folders too, and when CUDA still cannot be used it loads the model on the CPU
  with a notice instead of failing requests. On that laptop faster-whisper ran 6-19x faster on the GPU than on the
  CPU: on Windows it is the GPU route for speech-to-text.

In bolo's environment (about 2.5 GB for the two CUDA majors):

```powershell
uv tool install "bolo-speech[all,cuda]"                      # `cuda`: nvidia-cublas-cu12 and nvidia-cudnn-cu12
$py = "$(uv tool dir)\bolo-speech\Scripts\python.exe"        # the Python of uv's bolo-speech tool
uv pip uninstall --python $py onnxruntime onnxruntime-gpu onnxruntime-directml
uv pip install --python $py "onnxruntime-gpu[cuda,cudnn]" nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
$env:BOLO_DEVICE = "cuda"
bolo doctor        # "cuda:" with "ok onnxruntime ... CUDA provider loads and runs" and "ok faster-whisper ..."
```

In a plain virtual environment: `pip uninstall -y onnxruntime`, then
`pip install "onnxruntime-gpu[cuda,cudnn]" nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"`. `uv tool install
"bolo-speech[all]" --with "onnxruntime-gpu[cuda,cudnn]" --with nvidia-cublas-cu12 --with "nvidia-cudnn-cu12>=9,<10"`
installs everything in one go, but also keeps plain `onnxruntime` (faster-whisper asks for it): `onnxruntime` and
`onnxruntime-gpu` install the same `onnxruntime` module and overwrite each other's files, so uninstall
`onnxruntime` and install `onnxruntime-gpu` again afterwards, as above. For the same reason the `cuda` extra holds
only faster-whisper's CUDA 12 libraries: an extra can add `onnxruntime-gpu` but not take `onnxruntime` away. Swap
them again after any `uv tool upgrade` or `pip install` that brings `onnxruntime` back. `nvidia-cudnn-cu12` and the
`nvidia-cudnn-cu13` that `onnxruntime-gpu[cudnn]` brings put the same cuDNN 9 files in `nvidia\cudnn\bin`; the one
installed last stays, and `bolo doctor` tests both runtimes with it.

**Upgrading drops CUDA.** `uv tool upgrade bolo-speech` and `uv tool install --force "bolo-speech[...]"` resolve
bolo's dependencies again, and faster-whisper asks for plain `onnxruntime`: it is installed over `onnxruntime-gpu`,
and the ONNX Runtime engines quietly run on the CPU. bolo says so: with `device = "cuda"`, or `auto` on a machine
with an NVIDIA GPU (`nvidia-smi` on `PATH`), when the installed ONNX Runtime has no CUDA provider, the server logs
one warning at start and `bolo doctor` prints a `WARNING:` line, each naming the installed package
(`onnxruntime 1.32.0 (the CPU build)`) and the fix. Either run the swap above again after every upgrade, or install
the tool so that plain `onnxruntime` is never part of it: uv's `--excludes` takes a requirements file whose
packages are left out of the resolution, and uv keeps it (with the `--with` packages) in the tool's receipt, so a
later `uv tool upgrade bolo-speech` keeps `onnxruntime-gpu` (checked with uv 0.11):

```powershell
Set-Content -Path "$env:TEMP\no-onnxruntime.txt" -Value "onnxruntime"
uv tool install --force "bolo-speech[all,cuda]" --with "onnxruntime-gpu[cuda,cudnn]" --excludes "$env:TEMP\no-onnxruntime.txt"
uv tool upgrade bolo-speech     # later: keeps onnxruntime-gpu and leaves onnxruntime out
```

`uv tool upgrade` has no `--with` of its own; it reuses what the install recorded. A later `uv tool install --force`
replaces that record, so give it `--with` and `--excludes` again (without them it brings plain `onnxruntime` back,
and the warning says so).

`bolo doctor` with `device = "cuda"` (or `auto` on a machine with an NVIDIA GPU) opens a tiny graph on ONNX
Runtime's CUDA provider and runs it, asks CTranslate2 for its CUDA devices and loads CUDA 12's cuBLAS and cuDNN, and
names each library that is missing with the command that installs it; an ONNX Runtime without a CUDA provider is
named by its package (`onnxruntime-directml 1.24.4 (the DirectML build)`), with the command that swaps it. On Linux the same wheels work: bolo calls
`preload_dlls()` for ONNX Runtime and loads CTranslate2's cuBLAS and cuDNN from NVIDIA's wheels when they are not
on the library path.

sherpa-onnx's CUDA wheel cannot be used inside bolo on Windows: it brings its own `onnxruntime.dll`, which fails to
load (Error 1114) once `onnxruntime` has been imported, as bolo does; and the catalog's int8 sherpa-onnx models gain
nothing on CUDA. With `device = "cuda"` on Windows they run on the CPU, and `bolo recommend` suggests faster-whisper
models for the GPU.

### A GPU build of llama-server on Linux

`bolo components install llama-server` installs llama.cpp's prebuilt releases. For Linux x86-64 llama.cpp
publishes a CPU build and a Vulkan build, but no CUDA build (it ships CUDA on Linux only as Docker images); neither
does audio.cpp. So on Linux:

**Vulkan (AMD, Intel and NVIDIA GPUs; nothing to compile).** Install the Vulkan loader and a driver, then the build:

```sh
sudo apt install libvulkan1 vulkan-tools mesa-vulkan-drivers   # NVIDIA cards: the NVIDIA driver brings its own
vulkaninfo --summary                                           # your GPU should be listed
BOLO_DEVICE=vulkan bolo components install llama-server        # or: bolo components install llama-server --flavour vulkan
```

and run bolo with `device = "vulkan"`. Under WSL2 Vulkan goes through a translation layer and is slow: use CUDA there.

**CUDA (NVIDIA; usually the fastest).** Build llama-server once with the CUDA toolkit (12.x; under WSL2 install the
toolkit only, the Windows driver serves the GPU) and point bolo at it:

```sh
sudo apt install build-essential cmake git nvidia-cuda-toolkit # Ubuntu's CUDA toolkit; or NVIDIA's own packages
nvcc --version                                                 # the CUDA toolkit is installed
git clone --depth 1 --branch b12375 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j 4 --target llama-server
./build/bin/llama-server --list-devices                        # should list CUDA0
```

Keep `-j` to a number: each CUDA compile job takes about 0.7 GB, and an unbounded `-j` ran a WSL2 machine out of
memory (it hung); `-j 4` peaked at 2.7 GB and took 13 minutes on an 8-core laptop. The shallow clone has no history, so
`llama-server --version` says `build 1` rather than 12375; that is only the label.

Then set `llama_server_bin = "/path/to/llama.cpp/build/bin/llama-server"` in the settings file (or
`BOLO_LLAMA_SERVER_BIN=...`) and `device = "cuda"`; `bolo components list` shows it as your own binary and
`bolo doctor` asks it for its devices. b12375 is the version bolo pins and tests; a newer tag usually works. Measured
under WSL2 on an RTX 3060 Laptop GPU: Pocket TTS ran at the same speed as the prebuilt Windows CUDA build (real-time
factor 0.034 against 0.032). The build takes several minutes; `-DCMAKE_CUDA_ARCHITECTURES=86` (your card's compute capability: 86 for RTX 30, 89 for
RTX 40) makes it shorter. audio.cpp builds the same way (`-DGGML_CUDA=ON`; see its README) for
`audiocpp_server_bin`.

### A lost GPU

Found on an RTX 3060 Laptop GPU next to AMD Radeon integrated graphics (Windows 11, driver 621.24): during a long
CUDA llama-server run, requests became slower and slower (real-time factor 11.8, then 18.9, then 29.4; the CPU
build managed 4.69), llama-server used a CPU core flat out, and then `nvidia-smi` said `Unable to determine the
device handle for GPU0: 0000:01:00.0: GPU is lost. Reboot the system to recover this GPU`, later `No devices were
found`. llama-server kept answering, and until this check `bolo ps` kept saying `cuda:0`.

While models are on a GPU, bolo checks every `gpu_health_interval_s` seconds (30) and right after a request far
slower than that model's recent median:

- NVML (`nvidia-ml-py`): `nvmlDeviceGetHandleByIndex` or `nvmlDeviceGetMemoryInfo` failing with
  `NVML_ERROR_GPU_IS_LOST`, or no GPU at all; else
- `nvidia-smi`: "GPU is lost", "Unable to determine the device handle" or "No devices were found";
- and a llama-server or audio.cpp bolo started for a GPU printing a CUDA or Vulkan device error (`CUDA error: ...`,
  `cudaErrorDevicesUnavailable`, `ggml_cuda_...: ... failed`, `ErrorDeviceLost`; not `out of memory`).

When the GPU is lost, bolo logs `GPU LOST (<what was seen>): reboot to recover; bolo is using the CPU`, marks the
models on a GPU as degraded, and with `gpu_lost_fallback = "cpu"` (the default) loads each one again on the CPU once
its requests in flight have finished: llama-server and audio.cpp restart on their CPU build (`bolo components
install llama-server --flavour cpu` installs it; without it the GPU build starts with `--device none` and
`CUDA_VISIBLE_DEVICES=-1`), ONNX Runtime and CTranslate2 models load with the CPU provider. Models loaded later go
on the CPU too. With `gpu_lost_fallback = "fail"` the models stay where they are and their requests fail. Where it
shows:

- `bolo ps`: `GPU LOST` in the first line, the notice under the table, a model still on the lost GPU has `GPU LOST`
  as its processor, the ones moved are listed;
- `GET /api/v1/ps` and `/api/v1/stats`: `system.gpu_health` (`state` "lost", `reason`, `source`, `lost_at`,
  `moved_to_cpu`, `degraded`, `notice`), and `degraded` on each model;
- `GET /healthz`: still 200 (restarting bolo does not bring the GPU back), with `"status": "degraded"`, `gpu` and
  `notice`; `GET /api/v1/info`: `health`;
- the console's Loaded models panel: a red notice;
- `bolo doctor`: a `gpu health:` line wherever NVIDIA's tools are installed.

The loss lasts until bolo restarts: once the GPU is back (after a reboot), start bolo again.

Laptops with NVIDIA Optimus (or AMD switchable graphics) power the discrete GPU up and down, and under load a lid,
display or power-source change can make the driver lose it; only a reboot brings it back. Not verified, but worth
trying when it happens again: run on mains power, and in the NVIDIA Control Panel set "Power management mode" to
"Prefer maximum performance" for the program (`python.exe` of bolo's environment and `llama-server.exe`), or
globally.

### At log-on, restarted after a crash

Task Scheduler starts bolo at log-on but does not start it again when it crashes. Its "If the task fails, restart
every ..." setting (`-RestartCount` and `-RestartInterval` of `New-ScheduledTaskSettingsSet`) applies only when the
task fails to start: the program cannot be found, the account cannot log on, and the like. Once `bolo.exe` has
started, the task counts as having run; when bolo later exits, whatever its exit code (an unhandled error, "End
task" in Task Manager, a lost GPU taking the process down), Task Scheduler records the code as the "Last Run Result"
and does nothing more. Keep those settings for the start-up case, but something else has to watch the process.

**A PowerShell loop started by the task** (no extra software; runs as you, with your models and settings). Save
this as `%USERPROFILE%\bolo\run-bolo.ps1`:

```powershell
# Starts `bolo serve` and starts it again whenever it exits, waiting longer after quick repeated exits.
$env:BOLO_LOG_FILE = "auto"                                   # %LOCALAPPDATA%\bolo\logs\server.log
$bolo  = "$env:USERPROFILE\.local\bin\bolo.exe"               # (Get-Command bolo).Source
$log   = "$env:LOCALAPPDATA\bolo\logs\restarts.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
$delay = 5
while ($true) {
    $started = Get-Date
    & $bolo serve
    $code = $LASTEXITCODE
    "$(Get-Date -Format s) bolo exited with code $code; starting it again in $delay s" | Add-Content -Path $log
    Start-Sleep -Seconds $delay
    # a run that lasted 10 minutes was not a crash loop: start over at 5 s; otherwise back off up to 5 minutes
    $delay = if (((Get-Date) - $started).TotalMinutes -ge 10) { 5 } else { [Math]::Min($delay * 2, 300) }
}
```

Register it to run at log-on (in a normal PowerShell, as yourself):

```powershell
$script   = "$env:USERPROFILE\bolo\run-bolo.ps1"
$action   = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
$trigger  = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
  -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName "bolo" -Action $action -Trigger $trigger -Settings $settings `
  -Description "bolo speech server, restarted when it exits"
Start-ScheduledTask -TaskName "bolo"          # now, without logging off and on
```

`-ExecutionTimeLimit ([TimeSpan]::Zero)` keeps Task Scheduler from ending the task after its default three days;
the restart settings cover only a failed start of `powershell.exe`, as above. `-WindowStyle Hidden` hides the
PowerShell window, which may still flash at log-on. The task ends when you log off. The loop also starts bolo again
after it exits cleanly, so to stop bolo, stop the task first and then bolo:

```powershell
Stop-ScheduledTask -TaskName "bolo"
Get-Process bolo -ErrorAction SilentlyContinue | Stop-Process     # if it outlived the loop
Unregister-ScheduledTask -TaskName "bolo" -Confirm:$false        # to remove it for good
```

To try it, end `bolo.exe` in Task Manager: `restarts.log` gets a line and `bolo serve` is back after 5 s.

**As a service: NSSM or WinSW.** Both run bolo as a Windows service and start it again when it exits
(`nssm set bolo AppExit Default Restart`, WinSW's `<onfailure action="restart" .../>`); the commands are in
[deploy/README.md](../deploy/README.md#windows-at-log-on-at-start-up-or-as-a-service). A service runs before anyone
logs on and needs no window. On top of that, `sc.exe failure` tells Windows' service manager what to do when the
wrapper itself (`nssm.exe` or the WinSW executable) dies: start it again after 10 s, twice, then after a minute,
and count afresh after a day without failures:

```powershell
sc.exe failure bolo reset= 86400 actions= restart/10000/restart/10000/restart/60000
sc.exe qfailure bolo        # shows what is set
```

`sc.exe` alone cannot run bolo as a service: `sc.exe create bolo binPath= "...\bolo.exe serve"` registers it, but
`bolo.exe` is an ordinary program, not a service, so it never reports to the service manager, which gives up on it
after 30 s (error 1053, "The service did not respond to the start or control request in a timely fashion"). Use
NSSM or WinSW as the service, and `sc.exe` for its recovery settings.

**What was run.** What Task Scheduler's restart setting does, as described above, follows Microsoft's documentation
of the setting and was seen on a Windows 11 laptop: a crashed `bolo serve` was not started again. The PowerShell
loop, the task that starts it, the NSSM and WinSW services and the `sc.exe failure` settings have not been run by
the maintainers yet.

## Coming from Ollama

bolo borrows Ollama's model handling. What each `OLLAMA_*` variable is called here (as a setting; its environment
variable is `BOLO_` and the name in capitals):

| Ollama | bolo | Notes |
| --- | --- | --- |
| `OLLAMA_HOST` | `host`, `port` | two settings; bolo listens on `127.0.0.1:8000` |
| `OLLAMA_MODELS` | `data_dir` | models are under `<data_dir>/models` |
| `OLLAMA_KEEP_ALIVE` | `keep_alive` | the same values (`5m`, `0`, `-1`); per-task `stt_ttl_s`, `tts_ttl_s`, ... override it |
| `OLLAMA_MAX_LOADED_MODELS` | `max_loaded_models` | plus `memory_budget_mb` and `gpu_memory_budget_mb` |
| `OLLAMA_NUM_PARALLEL` | `num_parallel` | per model, see "Parallel requests and queues" |
| `OLLAMA_MAX_QUEUE` | `max_queue` | per model |
| `OLLAMA_SCHED_SPREAD` | `gpu_spread` | llama.cpp models only |
| `OLLAMA_GPU_OVERHEAD` | `gpu_reserve_mb` | in MB, per GPU |
| `CUDA_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES`, or `gpus` | both are honoured; `gpus` numbers the GPUs that are visible |
| `OLLAMA_FLASH_ATTENTION` | `llama_flash_attention` | llama-server voices and the speech-to-speech server |
| `OLLAMA_KV_CACHE_TYPE` | `llama_kv_cache_type` | the same values (`f16`, `q8_0`, `q4_0`) |
| `OLLAMA_CONTEXT_LENGTH` | `llama_context_size` | |
| `OLLAMA_ORIGINS` | `cors_origins` | |
| `OLLAMA_DEBUG` | `log_level = "debug"` | |
| `OLLAMA_LOAD_TIMEOUT` | `tts_server_start_timeout_s` | for the servers bolo starts |
| `OLLAMA_NOPRUNE`, `OLLAMA_NOHISTORY`, `OLLAMA_LLM_LIBRARY` | none | |

And the commands: `ollama list` / `ps` / `show` / `rm` / `cp` / `run` / `stop` are `bolo list` / `ps` / `show` /
`rm` / `cp` / `load` (or `run`) / `stop`; `ollama pull` is `bolo pull`; the server's log is `bolo logs`. `bolo cp`
makes a preset: a new name for the same files with settings of its own (README, "Models"; api.md,
`POST /api/v1/models/copy`).

## Per-model options

A catalog entry (or a custom entry in `<data_dir>/custom.yaml`) may carry
`options` that override the server-wide values for that model: `threads`,
`num_parallel` (or its older name `workers`), `max_queue`, `max_batch`, `max_chunk_s`,
the llama.cpp settings `flash_attention`, `kv_cache_type`, `context_size` and
`batch_size`, and engine-specific keys such as `beam_size` or `compute_type` for
faster-whisper. `num_parallel`, `max_queue` and the four llama.cpp settings can
also be pinned with `PUT /api/v1/models/{id}/settings` (or
`bolo models config <id> --set key=value`), which wins over the catalog.

`fallback` (a list of model ids of the same task) names what answers instead
when the model is unavailable, overloaded or, for a cloud model, its provider
fails; the setting `fallbacks` and a pin through
`PUT /api/v1/models/{id}/settings` override it. Entries with `engine: remote`
(cloud models) take `provider`, `model` and the adapter options listed in
[providers.md](providers.md#model-options).

Text handling (see api.md, "Text preparation, SSML and pronunciation word lists"):
`text_normalizer` (`uccharan`, the default for voices that read text; `light`,
bolo's own; `model`, numbers left to the model; `none`), `text_markup` (`kokoro`
when the voice's front end takes `[word](/ipa/)` markup; default `none`),
`text_latin` (`keep` Latin-script words in Indic text, the default, or
`transliterate` them into the voice's script), `text_respell` (`native`,
`latin-hyphen` or `latin-plain` respellings of word-list entries that only have
IPA; `latin-plain` when the entry also sets `lowercases: true`), `numbers` (the
language that reads digits, e.g. `hi`; default: the request language, English
next to English words) and `wordlists` (lists applied to every request for the
model) and `romanised` (`off`, `auto` or `on` for this model's voices; default: the
server's `romanised` setting). `text_normalizer`, `wordlists` and `romanised` can
also be pinned at run time with `PUT /api/v1/models/{id}/settings`, which wins
over the catalog.

Placement and keep-alive: `device` (where this model runs, overriding the `device`
setting; it still goes on the CPU when it does not fit) and `keep_alive` (how long
it stays loaded once idle, overriding `keep_alive` and the per-task TTLs). Both can
be pinned with `PUT /api/v1/models/{id}/settings` as well, which wins. For
llama.cpp models, `block_count` gives the layer count when the GGUF header cannot
be read.

Pronunciation word lists are files in `<data_dir>/wordlists/<name>.tsv`
(uccharan's TSV format; manage them with `/api/v1/wordlists`); the list named
`default` applies to every request that does not choose its own lists. They need
the `uccharan` package, which is also what enables the normalise-only mode; without
it bolo uses its light normaliser.

Orpheus-family voices on llama.cpp (`kind: orpheus-llamacpp`; see models.md,
"Orpheus-family voices on llama.cpp") read: `prompt_format` (`svara`, `veena`
or `orpheus`), `temperature`, `top_p`, `repetition_penalty` (defaults: the
model card's values for the format), `seed`, `eos_tokens`, `max_chars` (longest
piece of text per generation, 200), `max_new_tokens` (2048), `frames_per_char`
(the token budget per character of text, 2.0 frames), `stream_frames` (frames
decoded per streamed chunk, 4), `context_frames` (left context per decode, 3),
`speaker_field` (`name` or `id`: what a voice is called in the prompt when it has no `speaker`), `decoder_threads` (onnxruntime threads of the SNAC decoder, at most 2 by
default), `request` (extra `/completion` fields), `snac` (the decoder file:
`file`, `url`, `sha256`, `size`) and `snac_path` (an absolute path to another
SNAC decoder ONNX file, which wins). The llama-server options of the
`tts-server` engine apply too (`model_file`, `server_args`, the settings
`tts_servers`, `llama_server_bin`, `tts_server_args`). Note that
`tts_servers = ["llama-server=…"]` sends these models to that server as well:
name an Orpheus model by its id (`"svara-tts-v1-llamacpp=http://…"`) when your
llama-server runs another model.
