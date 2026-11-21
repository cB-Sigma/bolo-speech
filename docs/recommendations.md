# Hardware detection and model recommendations

bolo's catalog has several hundred models. Which of them suit the machine
bolo runs on depends on its CPU, its memory, whether a GPU is there and
whether anything bolo starts can use it. bolo looks at the machine, rates
every catalog model for it, and ranks the models of a task.

One rule holds everywhere in this document:

> **bolo recommends; it never restricts.** No model is hidden, refused or
> blocked because of hardware. A model that is unlikely to fit is still
> listed (last, with the reason), can still be pulled and can still be
> loaded. You can always pick anything.

```bash
bolo recommend                              # this machine, and a default pick per task
bolo recommend --task stt --language hi     # every Hindi recogniser, best fit first
bolo recommend --use live --language en     # a whole setup: STT + TTS (+ VAD, end of turn)
bolo recommend --benchmark                  # measure this machine first (a few seconds)
bolo doctor                                 # includes "this machine" and a short recommended setup
```

The console shows the same on its Models screen; the API is
`GET /api/v1/hardware` and `GET /api/v1/recommendations`
([api.md](api.md#hardware-and-recommendations)).

## What is detected

`bolo.hardware` reads, without any extra dependency:

| What | From |
| --- | --- |
| CPU model, vendor, architecture | `/proc/cpuinfo` (Linux), `sysctl` (macOS), `platform` elsewhere |
| physical and logical cores | `/proc/cpuinfo`, `sysctl`, `psutil` |
| usable cores | the smallest of the logical cores, the process's CPU affinity and the cgroup CPU quota |
| instruction sets that matter | x86: AVX, AVX2, AVX-512, AVX-512 VNNI / BF16 / FP16, AVX-VNNI, AMX; ARM: NEON, dot product, SVE, i8mm, BF16 |
| memory: total and available | `/proc/meminfo`, `sysctl`, `psutil` |
| free disk | the file system of the data directory |
| NVIDIA GPUs: name, VRAM total and free, driver | `nvidia-smi`, else the `pynvml` module if it is installed |
| Apple Silicon | the platform: the GPU shares the machine's memory ("unified") |
| Intel and AMD graphics | `/sys/class/drm` (AMD with its VRAM counters). Listed, but no runtime bolo starts uses them out of the box, and the recommender does not count on them |
| ONNX Runtime | the execution providers the installed build really has |
| sherpa-onnx | version. Whether its bundled ONNX Runtime is the CUDA build cannot be told and is reported as unknown |
| CTranslate2 (faster-whisper) | `get_cuda_device_count()` |
| the torch plugin | installed or not; its device (`cuda`, `mps`, `cpu`) from `bolo-torch doctor`, which is slow and only run when asked |
| `llama-server`, `audio.cpp` | the binary bolo would start; a release bolo installed itself is a CPU build on Linux and a Metal build on Apple Silicon; for `llama-server`, `--list-devices` when asked |
| container, virtual machine | `/.dockerenv`, `/run/.containerenv`, `/proc/1/cgroup`, Kubernetes' environment, the DMI vendor, the CPU's `hypervisor` flag |
| cgroup limits | `cpu.max` and `memory.max` (cgroup v2), `cpu.cfs_quota_us` and `memory.limit_in_bytes` (v1) |

Detection is best effort and never fails. A value that cannot be read is
`null` in the API and "unknown" on the command line and in the console: it is
never replaced by a guess. The server looks once at start-up, off the request
path. The slower probes (the torch plugin's `doctor`, the server binaries'
`--version`) run for `bolo recommend`, `bolo doctor` and
`GET /api/v1/hardware?refresh=true`.

Two things detection points out without acting on them:

* **A CPU quota below bolo's thread count.** bolo derives its inference
  threads from the machine's core count, not from the container's quota, so a
  container limited to one core on a 64-core host runs four threads on one
  core. The hardware report says so and names the `threads` value that avoids
  it. bolo does not change the setting for you.
* **A GPU that the `device` setting does not use.** With `device = "cpu"`
  (the default) the built-in engines run on the CPU whatever GPU is present;
  the torch plugin picks its own device. Fits are computed for the machine
  *as bolo is configured*, and the note tells you what to set.

## How fast is this machine: measured or estimated

The catalog's real-time factors were measured by the maintainers on one
**reference machine**: 4 vCPUs of an Intel Xeon at 2.1 GHz (AVX-512, AMX), no
GPU, 4 inference threads ([models.md](models.md#what-validated-means)). To
say anything about *your* machine, bolo needs a **speed factor**: how fast
your machine runs inference against that one (2.0 = twice as fast). A
catalog RTF is divided by it.

There are two ways to get the factor, and every surface says which one was
used.

### The micro-benchmark (measured)

```bash
bolo recommend --benchmark
```

runs a tiny fixed workload for about three seconds and stores the result in
`<data_dir>/hardware-benchmark.json` with a timestamp. `POST
/api/v1/hardware/benchmark` and the "Run benchmark" button in the console do
the same on a running server. It downloads nothing.

* With **onnxruntime** installed: a hand-built ONNX graph (a 3x3 convolution
  over 64 channels on a 96x96 map with a ReLU, and a 512x512 matrix product),
  run with the number of threads bolo uses on this machine. The best time of
  the run is compared with the reference machine's 4-thread time, so the
  result covers the speed of a core, its instruction sets and how many cores
  there are.
* Without it: a single-threaded numpy workload (element-wise float32
  arithmetic) gives the speed of one core; the thread count is then applied
  by the rule below. The result says `workload: numpy`.

The stored result is used from then on, by the command line and by the
server (a server that was already running picks up a benchmark taken from the
command line when it restarts or on `GET /api/v1/hardware?refresh=true`). It
is ignored when the CPU model, the core count, the usable cores
or bolo's thread count differ from when it was taken (the data directory was
moved, the container's limit changed): run it again.

What the benchmark is not: it is two operators on synthetic data, not a
speech model. A recogniser's decoder loop, a vocoder or a quantised model
can scale differently. A benchmark-scaled figure is given a range of
**x0.75 to x1.33** around the expected value. The reference timings
(`REFERENCE_MS` in `bolo/hardware.py`: 2.9 ms for the onnxruntime workload on
4 threads with onnxruntime 1.29, 6.1 ms for the numpy workload) are the best
of about twenty runs on the reference machine, which was shared with other
work when they were taken, as it was for the catalog's own measurements. On
a busy machine the benchmark reads low, which is what the models will see
too.

### The heuristic (estimated)

Without a stored benchmark the factor is

```
factor = core_speed(instruction sets) x thread_scale(inference threads)
```

| CPU | `core_speed` |
| --- | --- |
| x86-64 with AVX-512 and VNNI or AMX (the reference) | 1.0 |
| x86-64 with AVX-512 | 0.9 |
| x86-64 with AVX2 | 0.7 |
| x86-64 with AVX only | 0.45 |
| x86-64 without AVX | 0.3 |
| x86-64, instruction sets unknown | 0.7 |
| Apple Silicon | 1.4 |
| 64-bit ARM with dot product or SVE (Graviton, Ampere, recent Cortex) | 0.7 |
| 64-bit ARM with NEON only (Raspberry Pi 4 class) | 0.3 |
| anything else | 0.5 |

`thread_scale(n) = min(1, (n / 4) ^ 0.8)`: sub-linear below the reference's
four threads, flat above (bolo uses at most four threads per worker unless
`threads` is set). `n` is bolo's thread count capped by the usable cores, so
a CPU quota of 1.5 counts as 1.5.

**These multipliers are documented guesses, not measurements.** They ignore
the CPU's generation, clock speed, cache and memory bandwidth. A
heuristic-scaled figure is given a range of **x0.5 to x2**, the command line
and the console say "rough estimate", and `bolo recommend` reminds you that
the benchmark has not run.

GPU speed is not estimated: only a few models were measured on a GPU, on
one laptop ([status.md](status.md#a-windows-11-laptop)), which is not
enough to scale from. Only faster-whisper's measured CUDA speed-up is quoted
in its fit.

## How a fit is computed

Each catalog entry gets a **fit** for this machine:

| Level | Meaning |
| --- | --- |
| `comfortable` | expected to run faster than real time with room to spare (RTF up to 0.4; up to 0.25 for live use) and to fit in memory with room |
| `usable` | expected to keep up (RTF up to 1.0; up to 0.6 for live use), or to fit in memory only tightly, or to fit on the GPU with unknown speed |
| `slow` | expected to be slower than real time (RTF up to 5): fine for batch work, not for live use |
| `unlikely` | expected not to fit in memory or on the GPU, to need a GPU that is not there, or to be far slower than real time (RTF above 5) |
| `cloud` | runs at a provider: this machine does not matter |
| `unavailable` | its files are not published yet, or no runtime can run it: not a hardware question |

A fit carries the expected real-time factor with its range, where the number
comes from, the memory it needs against what there is, and a one-sentence
reason. In order:

1. **Cloud, pending and not-runnable entries** get `cloud` or `unavailable`
   and no number.
2. **Speed on the CPU.**
   * *Measured* (`rtf_basis: measured`): the entry has a real-time factor
     from the maintainers' run. It is divided by the speed factor.
   * *Estimated* (`rtf_basis: estimated`): the entry was never run, but a
     measured model with the same task, engine and loader kind exists (for
     the torch plugin and the TTS servers: also the same family). The
     nearest one in size, within a factor of four, lends its figure, scaled
     by the difference in size. The range is widened by another x0.5 to x2
     and the reason names the model it was taken from.
   * *Unknown* (`rtf_basis: unknown`): neither. No number is invented. The
     level then comes from what the catalog does say: `gpu: required`
     without a usable GPU is `unlikely`; `gpu: recommended` without one is
     `slow`; a VAD, end-of-turn or very small model is `comfortable`;
     otherwise the tier decides (`accuracy`: `slow`, else `usable`). The
     reason says it is a guess.
3. **Memory.** The need is the catalog's figure when it has one
   (`options.hardware.ram_gb`), else an estimate: the model's files x 1.2
   plus 150 MB (the same factor bolo's memory budget uses), and never less
   than the memory growth the maintainers saw or the catalog's `vram_gb`.
   It is compared with the usable memory: the machine's, or the container's
   limit when that is lower. More than 90 % of it: `unlikely`. More than
   60 %: at most `usable`, and the reason says "tight". If less than the need
   is free right now, a note says so; the level does not change, because
   free memory changes by the minute.
4. **A GPU**, when one is present *and bolo would use it for this model*:
   the torch plugin unless it is set to the CPU or its PyTorch has no CUDA;
   faster-whisper with `device = "cuda"` and a CTranslate2 that has CUDA;
   `llama-server` and `audio.cpp` models with a `device` other than `cpu`,
   unless the installed binary is known to be a CPU build; sherpa-onnx and
   onnxruntime models with `device = "cuda"`. The need is the catalog's
   `vram_gb` (itself an estimate: weights x 1.3 + 0.5 GB), else the same
   formula. On unified memory 70 % of the machine's memory counts as
   available. If it fits, a model that wants a GPU or is slow on the CPU
   becomes `usable` with device `gpu` and **no speed figure**; if it does not
   fit and the catalog says `gpu: required`, it is `unlikely`.
5. **The engine.** A missing engine (faster-whisper not installed, say) does
   not change the level: that is one `pip install` away, not hardware. It is
   noted, and it ranks the model after equally suitable ones whose engine is
   there.

### Three example machines

What the recommender picks by default changes with the machine. This table is
written by `server/tools/gen_docs.py` from the catalog, with the heuristic
speed factor (no benchmark) and every engine taken as installed:

<!-- BEGIN GENERATED: example-machines -->
**The reference machine: 4 x86-64 cores with AVX-512 and AMX, 16 GB, no GPU** (speed factor 1.00; of 554 entries: 279 comfortable, 97 usable, 99 slow, 56 unlikely to fit, 23 cloud)

| For | Default pick | Fit | Expected RTF (range) | Basis |
| --- | --- | --- | --- | --- |
| Speech to text, English, live | `kroko-en-streaming` | comfortable | 0.085 (0.043 to 0.17) | measured, scaled |
| Speech to text, English, files | `parakeet-tdt-0.6b-v2` | comfortable | 0.081 (0.04 to 0.16) | measured, scaled |
| Speech to text, Hindi, files | `indicconformer-hi-community` | comfortable | 0.066 (0.033 to 0.13) | measured, scaled |
| Text to speech, English, live | `piper-en-us-amy-low` | comfortable | 0.049 (0.024 to 0.098) | measured, scaled |
| Text to speech, Hindi, voice-over | `kokoro-v1.0` | usable | 0.76 (0.38 to 1.52) | measured, scaled |
| Speaker diarization | `nemotron-3-diarization` | comfortable | 0.039 (0.019 to 0.078) | measured, scaled |

**A small board: 4 ARM cores with NEON only, 4 GB (Raspberry Pi 4 class)** (speed factor 0.30; of 554 entries: 130 comfortable, 153 usable, 130 slow, 118 unlikely to fit, 23 cloud)

| For | Default pick | Fit | Expected RTF (range) | Basis |
| --- | --- | --- | --- | --- |
| Speech to text, English, live | `moonshine-tiny-en` | comfortable | 0.11 (0.055 to 0.22) | measured, scaled |
| Speech to text, English, files | `parakeet-tdt-0.6b-v2` | comfortable | 0.27 (0.13 to 0.54) | measured, scaled |
| Speech to text, Hindi, files | `indicconformer-hi-community` | comfortable | 0.22 (0.11 to 0.44) | measured, scaled |
| Text to speech, English, live | `piper-en-us-amy-low` | comfortable | 0.16 (0.082 to 0.33) | measured, scaled |
| Text to speech, Hindi, voice-over | `piper-hi-in-rohan-medium` | comfortable | 0.31 (0.15 to 0.61) | measured, scaled |
| Speaker diarization | `nemotron-3-diarization` | comfortable | 0.13 (0.065 to 0.26) | measured, scaled |

**A container: 1.5 cores of an AVX2 server, 2 GB memory limit** (speed factor 0.32; of 554 entries: 149 comfortable, 139 usable, 97 slow, 146 unlikely to fit, 23 cloud)

| For | Default pick | Fit | Expected RTF (range) | Basis |
| --- | --- | --- | --- | --- |
| Speech to text, English, live | `moonshine-tiny-en` | comfortable | 0.1 (0.052 to 0.21) | measured, scaled |
| Speech to text, English, files | `parakeet-tdt-0.6b-v2` | comfortable | 0.25 (0.13 to 0.5) | measured, scaled |
| Speech to text, Hindi, files | `indicconformer-hi-community` | comfortable | 0.21 (0.1 to 0.41) | measured, scaled |
| Text to speech, English, live | `piper-en-us-amy-low` | comfortable | 0.15 (0.077 to 0.31) | measured, scaled |
| Text to speech, Hindi, voice-over | `piper-hi-in-rohan-medium` | comfortable | 0.29 (0.14 to 0.58) | measured, scaled |
| Speaker diarization | `nemotron-3-diarization` | comfortable | 0.12 (0.061 to 0.24) | measured, scaled |

**A workstation: 16 AVX2 cores, 64 GB, an NVIDIA GPU with 24 GB, `device = "cuda"`** (speed factor 0.70; of 554 entries: 269 comfortable, 245 usable, 3 slow, 14 unlikely to fit, 23 cloud)

| For | Default pick | Fit | Expected RTF (range) | Basis |
| --- | --- | --- | --- | --- |
| Speech to text, English, live | `kroko-en-streaming` | comfortable | 0.12 (0.061 to 0.24) | measured, scaled |
| Speech to text, English, files | `parakeet-tdt-0.6b-v2` | comfortable | 0.12 (0.058 to 0.23) | measured, scaled |
| Speech to text, Hindi, files | `indicconformer-hi-community` | comfortable | 0.095 (0.047 to 0.19) | measured, scaled |
| Text to speech, English, live | `piper-en-us-amy-low` | comfortable | 0.07 (0.035 to 0.14) | measured, scaled |
| Text to speech, Hindi, voice-over | `piper-hi-in-rohan-medium` | comfortable | 0.13 (0.066 to 0.26) | measured, scaled |
| Speaker diarization | `nemotron-3-diarization` | comfortable | 0.056 (0.028 to 0.11) | measured, scaled |
<!-- END GENERATED: example-machines -->

## How uncertain this is

Be clear about what the numbers are worth:

* Of the catalog's entries, roughly a third were run by the maintainers.
  Their real-time factors come from one shared cloud VM and are "indicative
  and somewhat pessimistic". The rest are estimated from a sibling or not
  estimated at all.
* The speed factor is a synthetic benchmark at best and a table lookup at
  worst.
* Memory needs are estimates from file sizes; the catalog's `vram_gb`
  figures are a formula too.
* GPU speed was measured for a few models on one laptop's RTX 3060
  ([status.md](status.md#a-windows-11-laptop)); for every other model and
  GPU it is a guess.
* Quality figures (word error rate, round-trip error rate, diarization error
  rate) come from each model's *own* test audio. They show that a model
  works; they do not rank two models against each other. The recommender
  does not rank by them, with one exception: a voice whose round trip
  through a recogniser failed badly (more than 25 %) is not preferred.

So a fit is a starting point. Once a model is loaded, the console and
`GET /api/v1/stats` show what it really does on this machine ("Measured on
this server"), which beats any of the above.

## Ranking

`bolo recommend --task T` (and `GET /api/v1/recommendations?task=T`) orders
the models of a task. The filters you pass are applied first:

| Filter | Flag | Query parameter |
| --- | --- | --- |
| language (a model that covers it, or a language-independent one) | `--language hi` | `language=hi` |
| licences that allow commercial use | `--commercial` | `commercial=true` |
| no cloud models | `--local-only` | `local_only=true` |
| largest download | `--max-size-mb 500` | `max_size_mb=500` |
| live use or batch work | `--realtime` / `--batch` | `realtime=true` / `false` |

Filters are yours and remove models; the answer counts what each one removed
(`filtered_out`). **Hardware removes nothing.** Every remaining model is in
the list, in this order:

1. by fit: `comfortable`, `usable`, `cloud`, `slow`, `unlikely`,
   `unavailable` (for batch work `comfortable` and `usable` count the same:
   anything that keeps up will do);
2. models whose engine is installed before the others;
3. for text-to-speech without a language: English and multilingual voices
   (more than three languages) before voices of one other language, which
   are no general suggestion;
4. measured before estimated before guessed, so an estimate never outranks a
   measurement at the same level;
5. voices that failed their round-trip check last;
6. for speech-to-text with a language: a recogniser made for that language
   (up to three languages) before one that covers dozens; without a language
   the other way round, the broad model being the safer general suggestion;
7. licences that allow commercial use before those that do not;
8. by tier: for live use `realtime` first and streaming models ahead; else
   the most capable first (`accuracy`, `balanced`, `realtime`);
9. the model the server's settings name as the default, then installed
   models, then the faster one (live) or the larger one (otherwise).

The first up to three `comfortable` or `usable` models of *different
families* are marked **recommended for this machine**; the first of them is
the **default pick**. For text-to-speech without a language, voices of one
other language are marked only when fewer than three general voices keep up.
Cloud models are never marked: they are not about this
machine. If nothing keeps up with real time, the default pick is the best of
what runs slowly and the answer says so; if nothing runs at all, there is no
default pick and the list still gives the reason for each model.

`speech-to-speech` is accepted as a task (`s2s` for short): it ranks the
entries that can be a Realtime speech-to-speech backend.

### A whole setup

`bolo recommend --use live|files|voiceover --language L` (and
`GET /api/v1/recommendations/setup`, and "Pick for me" in the console)
proposes one model per task:

| Use | Tasks |
| --- | --- |
| `live` (a voice agent, live captions) | speech-to-text and text-to-speech ranked for live use; optionally VAD and end-of-turn detection |
| `files` (recordings, meetings) | speech-to-text ranked for batch work; optionally diarization |
| `voiceover` (narration) | text-to-speech ranked for batch work |

Local models only, unless you ask for cloud ones (`--cloud`,
`local_only=false`). It lists what would have to be pulled and how large
the download is. Nothing is pulled until you say so.

## Checking a model before downloading it

`bolo check <model> [--tiny] [--json]`, `POST /api/v1/models/{id}/check` (`{"tiny": true}`) and the "Check" button on
the console's Models page answer "would this run?" without downloading a model's weights. Exit code 1 (`"ok": false`)
when any step failed. The steps (`{"name", "status": "ok" | "fail" | "warn" | "skip", "detail"}`):

| Step | What is checked |
| --- | --- |
| `catalog` | the entry can be pulled and has a runtime |
| `runtime` (`imports`, `code`, `components`) | its engine is available and its components installed; for a full-duplex server, its environment imports its own code (`--install` installs a missing one: several GB), and whether torch.compile would be on; for a bolo-torch model, the plugin's environment imports the loader and the libraries of its extra, and its inference-code bundle matches its pins |
| `weights` | every file, asked for (HEAD) at its pinned revision or URL: there, or gated (for a gated entry), and as large as the listing or catalog says |
| `placement` | what the advisor makes of it here: fits on the GPU, split across the GPU and the CPU, on the CPU, or refused (a full-duplex model that does not fit the GPU) |
| `tiny-build`, `tiny-split`, `tiny-load`, `tiny-request` (`--tiny`) | a transformers-based bolo-torch model (Orpheus voices, Qwen-Omni, Step-Audio 2, GLM-4-Voice, MiniCPM-o) built from its `config.json` with two attention heads of the same head size and two layers, random weights; saved; split planning as when bolo's scheduler asks for a GPU/CPU split (bolo-torch's `split_kwargs`); loaded in bfloat16; one forward pass. Remote code is used only with the model's opt-in or `--allow-remote-code`, and only after the same pin check a load does |

What `--tiny` cannot check is reported as `skip`, never as `ok`: the full-duplex servers' models (they run in their
servers' own code), models that are not transformers models, 4-bit and 8-bit loading (bitsandbytes needs CUDA), and a
forward pass that needs more than text tokens. Only configuration, code and small companion files (up to 5 MB) are
fetched. A CI job runs `bolo check --all --tiny` over every entry it can build (not required yet).

## Warnings, and how to turn them off

When a model whose fit is `slow` or `unlikely` is pulled or loaded, bolo
attaches a notice and carries on:

* **API**: `POST /api/v1/models/pull` and `POST /api/v1/models/{id}/load`
  answer as before, with an extra `warnings` list (`code: hardware_fit`,
  `level`, `message`); the pull's job has the same list. A pull larger than
  the free disk space adds `code: disk_space`. The status codes do not
  change: the pull has started, the model is loaded.
* **Traces**: a request served by such a model has `warnings` on its trace
  (`GET /api/v1/requests`).
* **Log**: one line at warning level the first time the model is loaded.
* **Command line**: `bolo pull` prints `warning: ...` to standard error and
  downloads the model.
* **Console**: a notice when you pull or load the model, and a note in its
  details. The buttons stay enabled.

To silence the notices:

```toml
# bolo.toml
hardware_warnings = false
```

or `BOLO_HARDWARE_WARNINGS=false`. That removes the `warnings` lists, the
log line and the command-line message. The `fit` on catalog entries, the
rankings and `bolo recommend` stay: they are information you ask for, not
notices pushed at you. With the setting on or off, nothing else about pulling
or loading changes.

## In the console

On the Models screen:

* a **"This machine" card** with what was detected, the speed factor and
  whether it was measured, and a "Run benchmark" button;
* a **fit line** on every card ("Comfortable here · 9× real time (measured,
  roughly scaled)") with the reason as its tooltip, and an "On this machine"
  section in a model's details;
* a **"Recommended for this machine"** badge on the few models the server
  marks, judged for the language filter when one is set;
* **Sort: best for this machine**, which reorders;
* **Hide unlikely to fit**, an optional toggle that is off by default, is
  not remembered between visits, and says how many models it is hiding
  while it is on. Without it, hardware only reorders and highlights;
* **Pick for me**, offered first thing on a server with no model installed
  and always available on the hardware card: choose a language and a use,
  see the proposal with its reasons, pull it in one click.

The browser-only live demo has no server, so "this machine" would be the
visitor's browser; it shows none of this.

## Limits

* Windows: the CPU's name (registry), cores and memory; instruction sets
  through `IsProcessorFeaturePresent` (SSE4.2, AVX, AVX2, AVX-512F; NEON,
  dot product and SVE on ARM64), so AVX-512 extensions such as VNNI and AMX
  are not seen; no container detection. Graphics adapters through DXGI
  (name, dedicated memory, `dml:N`), counted on when the DirectML build of
  ONNX Runtime is installed. An NVIDIA card read through NVML or nvidia-smi
  is listed once (as CUDA, with its `dml:N`); integrated graphics are never
  the best GPU, so a hybrid laptop is judged by its discrete card.
* Intel Macs: no GPU detection (it would need `system_profiler`, which is
  slow).
* Intel and AMD GPUs are listed but never counted on (except through
  DirectML on Windows); Vulkan, ROCm and OpenVINO builds of the runtimes are
  not detected.
* Several NVIDIA GPUs: the one with the most memory is used for the fit;
  models are not spread over GPUs.
* Free memory and free VRAM are read when the hardware is detected, not
  per request; `GET /api/v1/hardware?refresh=true` looks again.
* The fits are computed in a background thread when the server starts,
  cached, and refreshed in the background when they are a minute old, when a
  model is installed or removed, and when the benchmark runs. A request never
  waits for them: in the first moments after start-up a trace or a pull
  started by a request can lack the notice it would otherwise carry.
