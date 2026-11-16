# Architecture

bolo is one Python process. Everything that makes it a server rather than a
wrapper around an inference library sits between the protocol front ends and
the engines:

```
          OpenAI HTTP    speaches routes    native /api/v1    WebSockets    Wyoming TCP    CLI
               \              |                  |                |             |          /
                +-------------+------------------+----------------+-------------+---------+
                |                              core (bolo.core)                           |
                |   transcribe()   synthesize()   realtime sessions   pulls   benchmark   |
                +----+--------------+----------------+-------------------+----------------+
                     |              |                |                   |
              audio I/O         text            observability        model store
        decode, resample,   normalise,        traces with phases,   resumable downloads,
        encode, G.711,      segment into      rolling quantiles,    checksums, content-
        RTP framing         sentences         SSE events, metrics   addressed blobs
                     |              |
                +----+--------------+-----------------------------------------------------+
                |                lifecycle manager (bolo.lifecycle)                       |
                |  lazy load · first load in a probe process · keep-alive unload · RAM    |
                |  and VRAM budgets · LRU eviction · GPU/CPU placement · preload          |
                +----+--------------------------------------------------------------------+
                     |  one queue per loaded model
                +----+--------------------------------------------------------------------+
                |                    scheduler (bolo.scheduler)                           |
                |  bounded queue · natural micro-batching · worker threads · cancellation |
                +----+--------------------------------------------------------------------+
                     |
          +----------+-----------+----------------+----------------+
          |  sherpa-onnx         |  faster-whisper |  optional: moonshine, whisper.cpp,
          |  (ONNX Runtime)      |  (CTranslate2)  |  kokoro-onnx, fake (tests)
          +----------------------+-----------------+----------------------------------
```

The source tree mirrors the picture:

| Path (under `server/src/bolo/`) | Role |
| --- | --- |
| `api/` | FastAPI routers: `openai.py`, `speaches.py`, `native.py`, `ws.py`, `static.py` (console) |
| `wyoming_server.py` | Wyoming protocol handler |
| `cli.py` | the `bolo` command |
| `core.py` | the `Bolo` object every front end calls |
| `streaming/` | realtime sessions, VAD helpers, partial-result stabilisation |
| `scheduler.py` | per-model queue with micro-batching |
| `lifecycle.py`, `_probe.py` | model loading, unloading, eviction, crash isolation |
| `placement.py`, `accel.py`, `keepalive.py`, `onnx_split.py` | GPU/CPU placement (llama.cpp layers, ONNX Runtime graphs, one graph cut in two), accelerator memory (NVML, nvidia-smi, unified memory), keep-alive parsing |
| `hardware.py`, `recommend.py` | what this machine is (CPU, memory, GPUs, runtimes, container limits), the micro-benchmark, a fit per catalog model and the rankings; advice and notices only, never a restriction ([recommendations.md](recommendations.md)) |
| `engines/` | the plugin interface (`base.py`) and one module per engine |
| `catalog/` | YAML model catalog and its validation |
| `store.py`, `custom.py` | downloads, the blob cache, `hf:` pulls |
| `audio/`, `text/` | codecs, resampling, encoders; segmentation, normalisation, word lists, romanised text, phonemes |
| `obs/` | traces, quantile sketches, event bus, Prometheus text |
| `bench.py`, `samples/` | load generator and built-in samples |

## Engines are plugins

An engine implements `Engine.status()` and `Engine.load(kind, task, path, spec, opts)`
and returns a model object with one of four small interfaces
(`bolo/engines/base.py`):

* `STTModel.transcribe(requests) -> results` decodes a *batch* of utterances
  in one call. `max_batch` says how many the engine can take.
* A streaming `STTModel` also has `create_stream()` and
  `decode_streams(streams)`, which advances many independent streams in one
  batched call.
* `TTSModel.synthesize(text, voice, speed, cancelled)` yields audio for one
  sentence and polls `cancelled()` so abandoned requests stop early.
* `VADModel.create_detector(params)` returns a stateful detector that turns
  audio into `speech_start` / `speech_end` events with sample positions.

Engines are imported lazily. `GET /api/v1/info` (and `bolo doctor`) asks each
one for its status; a missing library shows up as `available: false` with the
import error and an install hint, and never stops the server from starting.
The base install has no engine at all, which is why the wheel is small.

## The scheduler

Every loaded model owns one `ModelQueue`: a bounded FIFO, a few worker
threads, and a handler per job kind (`batch`, `stream`, `tts`).

**Natural micro-batching.** A worker takes the oldest job and then every
compatible job that is already waiting, up to the engine's `max_batch`. When
the server is idle a job runs at once, alone: batching adds no latency. When
it is busy, whatever queued up during the previous engine call goes out
together. `batch_wait_ms` can add a deliberate linger for hardware where a
batch costs little more than one item; it defaults to 0 because on CPU it
only adds latency (see the measurements in the README).

**What gets batched.**

* whole-utterance requests from different clients (`/v1/audio/transcriptions`,
  Wyoming, the final and partial re-decodes of pseudo-streaming sessions);
* the 30-second chunks of one long file (they are all submitted at once);
* the decode steps of concurrent native streaming sessions: sherpa-onnx
  advances all ready streams in one `decode_streams` call;
* text-to-speech is not batched (the engines synthesise one sentence at a
  time); sentences from different requests interleave fairly instead.

**Backpressure.** The queue is bounded (`max_queue`, default 64). A request
that finds it full gets `429` with `Retry-After` immediately instead of
making everyone's latency grow. Jobs that waited longer than
`queue_timeout_s` are failed rather than run late.

**Cancellation.** When an HTTP client disconnects or a WebSocket closes, the
task waiting on the job is cancelled; a job that is still queued is removed
before it reaches the engine. A decode that is already running in native code
cannot be interrupted, but synthesis checks the cancel flag between pieces
and later sentences of an abandoned request are never synthesised.

**Isolation.** If a batched engine call raises, the jobs are retried one by
one, so a single bad input fails alone.

**Threads.** `num_parallel` (Ollama's name; `workers`, default 1, when it is
0) is the number of concurrent engine calls per model, and can be set per
model like `max_queue` (configuration.md, "Parallel requests and queues");
`threads` is the intra-op thread count each call may use (default:
CPU count, at most 4, divided by `num_parallel`). One worker with all cores and
batching is the right default for speech-to-text on a small machine. More
workers with fewer threads each can help text-to-speech throughput.

## The lifecycle manager

* **Lazy.** A model is loaded by the first request that names it. Concurrent
  first requests share one load.
* **Probe.** The first load after installation happens in a child process
  (`python -m bolo._probe`). Native runtimes sometimes abort the process when
  a model file is not what they expect; the probe turns that into an error
  message for one model instead of an outage.
* **Keep-alive.** Idle models are unloaded after their keep-alive, as in
  Ollama: the `keep_alive` of the last request or load that gave one, else the
  model's pin, the per-task TTL (`stt_ttl_s`, ...; the VAD stays,
  `vad_ttl_s = -1`) or the `keep_alive` setting (`5m`). `0` unloads a model
  when the request using it ends (a session's: when the session ends); `-1`
  keeps it. A model in use is never unloaded. A request's value travels in a
  context variable (`bolo.keepalive.REQUEST_KEEP_ALIVE`) set by the route, so
  the core's call paths did not change.
* **Budget.** With `memory_budget_mb` or `max_loaded_models` set, loading a
  model first evicts idle ones, least recently used first. If everything
  loaded is busy the request gets `429` rather than the machine swapping.
  Memory per model is estimated from the growth of the process RSS during the
  load, which is approximate.
* **Auto-pull.** If a requested model is in the catalog but not on disk it is
  downloaded first (unless `auto_pull` is off, or the model is gated).

## Placement: GPU, CPU or both

Before a model is loaded the lifecycle manager places it (`bolo/placement.py`,
`ModelManager.place`), the way Ollama's scheduler does:

```
 requested device (pin > catalog options.device > `device`; auto -> the GPU found)
        |
 free accelerator memory (bolo/accel.py: NVML, nvidia-smi, unified memory on Apple)
   - gpu_reserve_mb, capped by what is left of gpu_memory_budget_mb
        |
 fits whole? ---- yes ----------------------------------------------> on the GPU
        | no
 would fit after unloading idle GPU-resident models? -- yes -> unload them (LRU), place again
        | no
 the runtime can split?  llama.cpp: -ngl N layers | ONNX Runtime: graph by graph |
                         PyTorch plugin: device_map + max_memory
        | no
 on the CPU, with a notice (nothing is refused)
```

The decision reaches the engine in `LoadOptions`: `device` (the device for
the whole model, or for the graphs not listed), `gpu_layers` (llama-server's
`-ngl`), `graph_devices` (role -> device, read with `opts.device_for(role)`)
and `max_memory` (sent to plugins). Engines that cannot use the device family
(faster-whisper on Metal, kokoro-onnx, Moonshine) are placed on the CPU up
front rather than failing later.

After the load the manager measures what it took: the growth of the resident
memory of this process and every process it started (llama-server, audio.cpp,
plugins), and the growth of their accelerator memory (NVML's per-process
figures, else the device's used memory). A plugin's own report of its split
wins. On Apple Silicon the accelerator share is the plan, since unified memory
cannot be told apart. `memory_budget_mb` counts each model's RAM side,
`gpu_memory_budget_mb` its accelerator side.

On Windows under the WDDM driver model the NVIDIA driver keeps no per-process
figures (NVML: `NVML_VALUE_NOT_AVAILABLE` or 0; nvidia-smi: "N/A"; NVML's
`nvmlDeviceGetDriverModel` says WDDM). `accel.py` reads that as unknown
(`process_bytes` None) and adds this process's own use from DXGI
(`own_bytes`: `QueryVideoMemoryInfo` on the adapter whose LUID the CUDA driver
gives for the GPU's PCI address, else the one NVIDIA adapter of that name).
`lifecycle.attribute_vram` then takes, in order: per-process growth; this
process's DXGI growth (runtimes in bolo's process); the device's used-memory
growth, with such loads serialised by a lock; else the plan, not measured
(`vram_source = "plan"`). `LoadedModel.processor` follows the placement for a
model placed on the accelerator unless a measurement found nothing there.

`GET /api/v1/ps`, `bolo ps` and the console's "Loaded models" panel show the
result: size, `size_vram`, `PROCESSOR` ("100% GPU", "100% CPU",
"35%/65% CPU/GPU"), device and time until unload.

**Several GPUs.** With NVIDIA GPUs, `accel.py` numbers them as CUDA does in
this process (it applies `CUDA_VISIBLE_DEVICES`), the `gpus` setting narrows
them, and `ModelManager._place_gpus` runs the decision above once per allowed
GPU and keeps the one with the most memory available. For a llama.cpp model
that does not fit whole there (or always, with `gpu_spread`),
`placement.plan_spread` shares its layers over the GPUs in proportion to their
free memory; the result reaches llama-server as `--tensor-split` and
`--main-gpu` (`bolo/llama_args.py`), and one-GPU placements get
`--split-mode none --main-gpu N` so llama-server does not spread them itself.
The eviction step counts what unloading idle models would give back per GPU.

A single ONNX graph that does not fit (or the largest graph of a multi-graph
model) is cut in two at a layer boundary when `onnx_split = "auto"`
(`bolo/onnx_split.py`): every prefix of a graph's node list is a valid front,
so bolo scans them once, keeping the weights on each side and the tensors that
cross up to date, refuses points where a control-flow subgraph would read
across, moves weight-only nodes (dequantisation) with their reader, and takes
the layer boundary that puts the most weights within the free memory. The
halves run as two sessions behind `SplitSession`, which answers `run`,
`get_inputs` and `get_outputs` like one, and are cached next to the model by
its hash and the cut. CTranslate2 cannot split a model; sherpa-onnx builds its
own sessions, so its models move whole.

**Untested on hardware.** CI has no GPU. The placement arithmetic, the NVML
and nvidia-smi readings, the GPU budget and eviction, `-ngl` and the per-graph
providers are tested with fake NVML modules and made-up memory figures only;
graph cuts are tested with both halves on the CPU (outputs compared with the
whole graph on small built graphs, and on Smart Turn and Nemotron where cached).
Several GPUs (`gpus`, `gpu_spread`, `CUDA_VISIBLE_DEVICES`, `--tensor-split`)
are tested with a fake NVML of several made-up GPUs, and the llama-server
arguments for GPUs, flash attention, the KV cache type and parallel slots by
the command lines bolo builds, without running llama-server. The WDDM path
(per-process N/A, DXGI usage, device-level growth) is tested with a fake NVML
and fake DXGI adapters; the CUDA-driver LUID lookup (`winhw.cuda_luids`) has
not run in CI.

## The model store

`<data_dir>/blobs/sha256-<digest>` holds each unique file once;
`<data_dir>/models/<id>/` holds hard links with the layout the engine expects
plus a `manifest.json`. Downloads resume with HTTP `Range` after an
interruption. LFS files are verified against the SHA-256 the Hub publishes,
small files against their git object id. Catalog entries pin the repository
revision to a commit, so a pull is reproducible.

## Realtime sessions

`bolo/streaming/session.py` backs the STT WebSocket and the realtime benchmark:

```
socket reader ──► feed() ──► [per-session queue] ──► session task
   (never blocks)                                      │ decode G.711/PCM, resample (stateful)
                                                       │ VAD ──► vad events
                                                       │ native model: accept audio ► "stream" job ► partial / endpoint ► final
                                                       │ other models: every 350 ms re-decode the utterance as a "batch" job
                                                       │               ► LocalAgreement ► partial;  on speech end ► final
                                                       ▼
                                                 send JSON messages
```

The reader and the decoder are decoupled so a slow decode never stalls the
socket; if audio arrives faster than it can be transcribed for more than
30 seconds of backlog, the stream is closed with an `overloaded` error.
[realtime.md](realtime.md) explains the two decoding strategies and the
latency definitions.

## Diarization, voice agents and components

* **Diarization** (`bolo/diarize.py`, `SherpaDiarization`) is one more model
  kind behind the same lifecycle manager and scheduler. A diarized
  transcription runs the pipeline first, merges adjacent turns of one speaker
  and then transcribes each turn as an ordinary batch job. On realtime
  streams, each finalised utterance is embedded and matched against the
  speakers seen so far (`OnlineSpeakerLabeler`).
* **Voice agents** (`bolo/realtime.py`, `bolo/agents.py`, `bolo/llm.py`) are a
  cascade over the pieces above: a realtime STT session, an HTTP client for
  any OpenAI-compatible chat-completions endpoint, and `speak_stream`, which
  cuts the model's token stream into sentences and synthesises each as soon
  as it is complete. Speech detected while a response is playing cancels the
  language-model stream and the queued synthesis (barge-in). Every turn is
  one trace of kind `agent`.
* **Phone calls** (`bolo/telephony/`) put a phone line in front of a voice
  agent. Each provider (Twilio, Plivo, Telnyx, Exotel) is a thin adapter:
  its media-stream frames become neutral events, its webhooks are verified
  and answered, its REST API places calls. One `CallSession` per call owns
  a `RealtimeSession` and adds what only a call has: tracking what the
  provider has played, clearing its queue on barge-in, greeting and prompts,
  key presses, hang-up and transfer, and the call record. The SIP endpoint
  (`sip.py`) feeds the same session from RTP. `simulator.py` plays the
  provider and the caller for tests. See [telephony.md](telephony.md).
* **Components** (`bolo/components.py`) tracks optional pieces: the two
  phonemisers (uccharan, eSpeak NG), reports which one each voice would use
  and whether it is installed, and can install them into the running
  environment on request. It also lists the engine plugins catalog models
  need.
* **Phonemisers** (`bolo/text/phonemize.py`) turn text into phonemes for
  voices that read phonemes: uccharan in process where it covers the
  language (a data table), eSpeak NG otherwise; a request, a per-model pin
  (`model-settings.json` in the data directory) or the server setting chooses.
* **Engine plugins** (`bolo/plugin/`) run engines in separate processes over a
  JSON-lines protocol on stdio; their models look like any other model to
  the scheduler (docs/plugins.md).

## Observability

Each request owns a `Trace` (an id, the transport, and a list of named phases
with start and duration from one monotonic clock). Finished traces go into a
ring buffer served by `/api/v1/requests`, are published on `/api/v1/events`,
and feed per-model rolling quantiles (the most recent 2,048 observations, at
most 15 minutes old) that `/api/v1/stats` and `/metrics` report. The timing
tests run the whole pipeline against a fake engine with known delays and
check every reported number against them.
