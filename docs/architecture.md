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
                |  lazy load · first load in a probe process · TTL unload · memory budget |
                |  · LRU eviction · preload                                               |
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
| `engines/` | the plugin interface (`base.py`) and one module per engine |
| `catalog/` | YAML model catalog and its validation |
| `store.py`, `custom.py` | downloads, the blob cache, `hf:` pulls |
| `audio/`, `text/` | codecs, resampling, encoders; segmentation, normalisation, phonemes |
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

**Threads.** `workers` (default 1) is the number of concurrent engine calls
per model; `threads` is the intra-op thread count each call may use (default:
CPU count, at most 4, divided by `workers`). One worker with all cores and
batching is the right default for speech-to-text on a small machine. More
workers with fewer threads each can help text-to-speech throughput.

## The lifecycle manager

* **Lazy.** A model is loaded by the first request that names it. Concurrent
  first requests share one load.
* **Probe.** The first load after installation happens in a child process
  (`python -m bolo._probe`). Native runtimes sometimes abort the process when
  a model file is not what they expect; the probe turns that into an error
  message for one model instead of an outage.
* **TTL.** Idle models are unloaded after `stt_ttl_s` / `tts_ttl_s` (300 s);
  the VAD stays (`vad_ttl_s = -1`). A model in use is never unloaded.
* **Budget.** With `memory_budget_mb` or `max_loaded_models` set, loading a
  model first evicts idle ones, least recently used first. If everything
  loaded is busy the request gets `429` rather than the machine swapping.
  Memory per model is estimated from the growth of the process RSS during the
  load, which is approximate.
* **Auto-pull.** If a requested model is in the catalog but not on disk it is
  downloaded first (unless `auto_pull` is off, or the model is gated).

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
* **Components** (`bolo/components.py`) tracks optional third-party pieces
  that are not shipped (eSpeak NG), reports whether they are installed, and
  can install them into the running environment on request.

## Observability

Each request owns a `Trace` (an id, the transport, and a list of named phases
with start and duration from one monotonic clock). Finished traces go into a
ring buffer served by `/api/v1/requests`, are published on `/api/v1/events`,
and feed per-model rolling quantiles (the most recent 2,048 observations, at
most 15 minutes old) that `/api/v1/stats` and `/metrics` report. The timing
tests run the whole pipeline against a fake engine with known delays and
check every reported number against them.
