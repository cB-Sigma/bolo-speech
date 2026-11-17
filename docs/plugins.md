# Engine plugins (out of process)

bolo never imports PyTorch, and its built-in engines (sherpa-onnx, faster-whisper, Moonshine,
whisper.cpp, ONNX Runtime) run in the server process. Some good models need other libraries -
PyTorch above all (Indic Parler-TTS, IndicF5, Veena, Svara, SraVaani). An **engine plugin** runs
such models in a separate process that bolo starts, talks to over a pipe, restarts when it crashes
and stops when it is idle. The server stays small and torch-free; the plugin brings whatever it
needs.

Status: the server side, the protocol, the plugin-side helper and a reference plugin are part of
bolo 0.2.0 and are covered by tests (`server/tests/test_plugins.py`, which runs the reference
plugin as a real subprocess). The PyTorch plugin is the separate package **bolo-torch**
([plugins/bolo-torch](../plugins/bolo-torch/README.md)); until it is installed the catalog lists the
models that need it as "needs plugin `torch` (not installed)" with its install command.

## Using a plugin

A plugin is found in two ways:

* **installed package**: a Python package that declares an entry point in the group
  `bolo.engines`. The entry point names a module, which bolo starts as
  `python -m <module>` with its own interpreter, or a function (`module:function`) that returns the
  command line to start:

  ```toml
  # pyproject.toml of the plugin package
  [project.entry-points."bolo.engines"]
  torch = "bolo_torch:plugin"   # a function returning [sys.executable, "-m", "bolo_torch"]
  ```

  The entry point is loaded in bolo's own process, so the module it names must not import heavy
  libraries (bolo-torch's `__init__` imports nothing but `sys`).

* **configured command** (wins over an entry point of the same name), for a plugin in another
  virtual environment, a container, or any executable:

  ```toml
  # bolo.toml
  plugins = ["torch=/opt/bolo-torch/venv/bin/python -m bolo_torch.plugin"]
  ```

  or `BOLO_PLUGINS="torch=/opt/bolo-torch/venv/bin/python -m bolo_torch.plugin"`.

The plugin then appears as engine `plugin:torch` in `bolo doctor`, `/api/v1/info` and
`/api/v1/components`, and catalog entries with `engine: plugin:torch` become usable. Until it is
installed such entries have `requires: ["plugin:torch"]`, `available: false`, and using one is
answered with HTTP 424 `component_missing` and the install hint from the entry's
`options.plugin_install`.

| Setting | Default | Meaning |
| --- | --- | --- |
| `plugins` | `[]` | `name=command args` per plugin |
| `plugin_entry_points` | `true` | discover installed plugin packages |
| `plugin_idle_s` | `300` | stop a plugin process when none of its models has been loaded for this long |
| `plugin_memory_limit_mb` | `0` (none) | address-space limit of a plugin process (POSIX `RLIMIT_AS`) |
| `plugin_nice` | `0` | added scheduling niceness of plugin processes |

## What bolo does for a plugin

* **Lifecycle.** The process starts on the first model load, not with the server. A crash fails the
  requests in flight with 503 `engine_unavailable`; the next request restarts the process after a
  back-off (1 s, doubling up to 60 s while crashes continue) and loads its models again, so callers
  see one failed request, not a broken model. Unloading the last model of a plugin starts the idle
  timer; the process is asked to shut down (then killed after 5 s) when it expires and on server
  shutdown.
* **Scheduling and observability.** Plugin models are ordinary `STTModel` / `TTSModel` objects to
  the rest of bolo: they get the same per-model queues, backpressure (`429`), cancellation on client
  disconnect, idle unloading, memory budget, traces with phases (`queue`, `decode`,
  `synthesize_first`, …), Prometheus metrics and the console's views. Batching is off unless the
  plugin reports `max_batch > 1` when it loads a model. The first-load probe in a child process is
  skipped: the plugin is a separate process already.
* **Downloads.** Catalog entries for plugin models are pulled by bolo like any other (Hugging Face
  repositories pinned to a commit, gated repositories with `HF_TOKEN`); the plugin receives the
  local directory.

## Writing a plugin

Implement bolo's ordinary engine interface (`bolo.engines.base.Engine`, returning `STTModel` or
`TTSModel` objects) in the plugin package, and serve it:

```python
# bolo_torch/plugin.py
from bolo.plugin.sdk import serve
from bolo_torch.engine import TorchEngine

if __name__ == "__main__":
    raise SystemExit(serve(TorchEngine()))
```

`serve` answers the protocol on stdin/stdout, runs requests on a small thread pool (default 4,
`BOLO_PLUGIN_WORKERS`), redirects stray `print` output to stderr (which bolo copies into its log),
and passes cancellation to `TTSModel.synthesize` through its `cancelled` callback.
`python -m bolo.plugin.reference` is a complete plugin (bolo's fake engine served out of process)
and the shortest example.

## Protocol (version 1)

JSON lines over the plugin's stdin/stdout. A message is one UTF-8 JSON object on one line; when it
has `"bytes": n`, exactly `n` raw bytes follow (audio: little-endian float32). Requests carry an
`id`; every request gets exactly one final reply with the same `id` (`"ok": true` plus results, or
`"ok": false, "error": {"code", "message"}`). `synthesize` and `respond` send `{"id", "chunk": n, "bytes": …}`
messages before their final reply. Replies to different requests may interleave.

| `op` | request | final reply |
| --- | --- | --- |
| `hello` | `protocol` (highest version bolo speaks) | `protocol` (version used), `name`, `version`, `kinds` |
| `health` | - | `rss_mb` |
| `load` | `model`, `kind`, `task`, `path`, `options`, `threads`, `device`, `placement` (optional) | `handle`, `sample_rate`, `info` (`voices`, `languages`, `max_batch`, `supports_translate`, `supports_word_timestamps`, `speaker_conditioned`, `placement`) |
| `unload` | `handle` | - |
| `transcribe` | `handle`, `requests: [{n, language, task, prompt, word_timestamps, temperature, hotwords}]` + the audio of all requests, concatenated | `results: [{text, language, segments: [{start, end, text, words}], words: [{w, t0, t1}]}]` |
| `speakers` | `handle`, `n`, `language`, `task`, `prompt`, `word_timestamps`, `temperature`, `frame_s`, `frames`, `speakers` + `n` audio samples, then the speaker activity (`frames` x `speakers` float32, frame by frame) | `results`: one per speaker, as for `transcribe`, times from the start of the audio |
| `synthesize` | `handle`, `text`, `voice`, `speed` | chunks, then `sample_rate`, `chunks`, `cancelled` |
| `separate` | `handle` + the mixture (16 kHz) | `tracks` (count) + the tracks, concatenated, equal length |
| `enrol` | `handle`, `key` + a voice sample (16 kHz) | - |
| `extract` | `handle`, `key` + the mixture (16 kHz) | the enrolled voice's track |
| `release` | `handle`, `key` | - |
| `diarize` | `handle`, `num_speakers`, `threshold` (either may be null) + the recording (16 kHz) | `turns`: `[[start, end, speaker], ...]` |
| `embed` | `handle` + one voice (16 kHz) | the unit-length speaker embedding (float32) |
| `denoise` | `handle` + the recording (at the model's `sample_rate`) | the enhanced recording, as long as the input |
| `respond` | `handle`, `turns: [{role, text, n, call_id, tool_calls}]`, `instructions`, `tools`, `tool_choice`, `voice`, `max_output_tokens`, `audio_out` + the audio of the turns (16 kHz), concatenated (`n` samples each) | chunks (`text`, `tool_call`, audio at the model's rate), then `sample_rate`, `chunks`, `cancelled` |
| `cancel` | `target` (id of a running request) | - |
| `shutdown` | - | - (the process exits) |

A denoise model (`task: denoise`, bolo 0.6.7 and later) answers `denoise`: audio in and out at the `sample_rate` of
its `load` reply, which bolo resamples to and from. bolo sends it the whole recording of a transcription; on the live
speech-to-text WebSocket it sends each chunk with the half second before it (a plugin model keeps no state between
requests). In a plugin built on bolo's SDK, the model is a `DenoiseModel`.

A separation model's `load` reply carries `info.mode` (`"blind"`: answers `separate`; `"target"`:
answers `enrol` / `extract` / `release`) and `info.min_sample_s`.

A diarization model (`task: diarization`, bolo 0.6.7 and later) answers `diarize` with its speaker turns (speakers
numbered from 0; bolo renumbers them in order of appearance) and `embed` with a voice print, which bolo uses for
known speakers, its speaker clean-up and assigning separated tracks. `info.consolidate_default: false` turns bolo's
speaker clean-up off for the model, keeping its own speaker count. `min_speakers` / `max_speakers` are bolo's: it
asks again with `num_speakers` when the count falls outside them. In a plugin built on bolo's SDK, the model is a
`DiarizationModel`.

A speech-to-speech model (`task: speech-to-speech`, bolo 0.6.7 and later) holds spoken conversations one turn at a
time, behind the Realtime API's `speech-to-speech` pipeline (voice-agents.md). Its `load` reply may carry
`info.voices` and `info.supports_tools`. The model keeps nothing between requests: bolo keeps the conversation and
sends all of it with every `respond`: user turns as audio (`n` samples of the payload, 16 kHz; the last eight spoken
turns) or `text`, the model's earlier answers as `text` with their `tool_calls` (`{call_id, name, arguments}`), and
tool results (`role: tool`, `call_id`, `text`). The answer streams back as chunks: `text` (the next piece of what
the model says), audio (float32 at `sample_rate`, with or without text) or `tool_call`. A barge-in sends `cancel`;
the plugin stops before its next chunk. `audio_out: false` is a text-only response. In a plugin built on bolo's SDK,
the model is a `SpeechToSpeechModel` that implements `respond(ConversationRequest, cancelled) ->
Iterator[ConversationChunk]`. Full-duplex models (that listen and speak at once, frame by frame) do not fit this op.

A speech-recognition model whose `load` reply says `info.speaker_conditioned: true` is a
diarization-conditioned (target-speaker) recogniser. For a diarized transcription bolo then does not
cut the recording at speaker turns: it diarizes, and sends the whole recording once with `speakers`,
together with per-frame speaker activity (column `k` is speaker `k` of the diarization, values 0..1,
`frame_s` seconds per row). The plugin answers with one result per speaker. bolo sends `speakers`
only to models that announce it, so older plugins are unaffected; the protocol version stays 1. In a
plugin built on bolo's SDK, an `STTModel` with `speaker_conditioned = True` implements
`transcribe_speakers(SpeakerRequest) -> list[STTResult]`.

**GPU/CPU placement.** `device` in `load` is where bolo's scheduler placed the model (`cpu`, `cuda:0`, ...;
configuration.md, "GPU and CPU placement"). When the free accelerator memory is smaller than the model's estimate,
bolo also sends `placement: {"device_map": "auto", "max_memory": {"0": <bytes>, "cpu": <bytes>}}`: a plugin that
can split a model across the GPU and the CPU (accelerate's `device_map="auto"` with that `max_memory`) does so, and
answers with `info.placement: {"gpu_bytes", "cpu_bytes", "device_map"}`, which `bolo ps` shows as
`"x%/y% CPU/GPU"`. A plugin that cannot split ignores `placement` and loads the model whole on `device`. In a
plugin built on bolo's SDK, `LoadOptions.max_memory` carries it, and a model attribute `placement_report` (a dict)
is sent back. Both members are optional (bolo 0.5); the protocol version stays 1. bolo-torch splits the
transformers language model of Orpheus SNAC voices this way; its other loaders load whole.

Errors: `error.code` `model_gated` becomes bolo's `401 model_gated` and `invalid_request` its
`400 invalid_request` (so a gated download inside the plugin reaches the client as such); any other
code is a `503 engine_unavailable` naming the code. bolo's SDK reports an exception by its class
name; bolo-torch sends the code of its own errors instead.

Compatibility: new optional members, and new ops that bolo sends only to models announcing them in
`info` (`speakers`) or loaded for a new task (`diarize`, `embed`, `respond`), can appear within version 1; an incompatible change gets
version 2, and bolo will keep speaking version 1 to plugins that answer `hello` with 1. The exact
definitions are in `server/src/bolo/plugin/protocol.py`.

## The PyTorch plugin: bolo-torch

[`plugins/bolo-torch`](../plugins/bolo-torch/README.md) serves these catalog entries
(`server/src/bolo/catalog/plugin-torch.yaml`, `engine: plugin:torch`, pinned revisions):

| Model | Task | Extra | Licence | Gated | Run here (CPU, 2 threads, shared machine) |
| --- | --- | --- | --- | --- | --- |
| `indic-parler-tts` (AI4Bharat) | TTS, voice described in words, 69 named voices | `parler` | Apache-2.0 | yes | RTF 21-23, first audio 21-64 s |
| `parler-tts-tiny-v1` | TTS, English | `parler` | Apache-2.0 | no | RTF 4.5-4.8, first audio 4.6-11.5 s |
| `indic-f5` (AI4Bharat IndicF5) | TTS, voice cloning, 11 Indian languages | `f5` | MIT | yes | RTF 120-154 |
| `veena` (Maya Research) | TTS, 3B Llama + SNAC, Hindi and English | `orpheus` | Apache-2.0 | no | not run (only the SNAC decoder) |
| `svara-tts-v1` (Kenpath) | TTS, 3B Llama + SNAC, 19 languages | `orpheus` | Apache-2.0 | no | not run (only the SNAC decoder) |
| `sepformer-whamr16k` | separation (blind, 2 speakers, 16 kHz) | `sepformer` | Apache-2.0, WSJ0-derived data | no | RTF 3.7 |
| `sepformer-whamr` | separation (blind, 2 speakers, 8 kHz) | `sepformer` | Apache-2.0, WSJ0-derived data | no | RTF 1.4 |
| `pixit-ami` (pyannote PixIT) | separation with diarization | `pyannote` | MIT | yes (two repositories) | RTF 5.7; little gain on a synthetic mix |
| `pyannote-speaker-diarization-3.1` | diarization (pyannote's PyTorch pipeline) | `pyannote` | MIT + CC-BY-4.0 (embedding, PLDA) | yes (three repositories) | RTF 0.9; DER 6.0 % on the demo dialogue (ONNX default 5.9 %) |
| `pyannote-speaker-diarization-community-1` | diarization (VBx clustering) | `pyannote` | CC-BY-4.0 | yes | RTF 0.9; DER 6.0 % on the demo dialogue |
| `dicow-v3.3` (BUT-FIT DiCoW) | STT, diarization-conditioned (speaker request) | `dicow` | CC-BY-4.0 | no | RTF 0.45 with Nemotron diarization (bf16, AMX); AMI cpWER 69 → 26 % |
| `se-dicow` (BUT-FIT SE-DiCoW) | STT, diarization-conditioned, self-enrolment | `dicow` | Apache-2.0 | no | RTF 0.59 with Nemotron diarization (bf16); AMI cpWER 69 → 25 % |
| `demucs-dns48`, `demucs-dns64`, `demucs-master64`, `demucs-valentini-nc` (Facebook Research Demucs denoiser) | denoise (16 kHz) | `demucs-denoiser` | **CC-BY-NC-4.0, not for commercial use** | no | docs/models.md, "Denoising" |
| `qwen2.5-omni-3b-torch` | speech-to-speech (text and speech out) | `omni` | Qwen Research License (non-commercial) | no | not run (12 GB; fakes only) |
| `qwen2.5-omni-7b-torch` | speech-to-speech | `omni` | Apache-2.0 | no | not run (22 GB; fakes only) |
| `qwen3-omni-30b-a3b-instruct-torch` | speech-to-speech, speech out in 10 languages | `omni` | Apache-2.0 | no | not run (70 GB; fakes only) |
| `sravaani-1.0`, `indicconformer-600m-official` | STT | - | MIT | yes | not implemented in bolo-torch yet |
| `moshiko-pytorch-bf16`, `moshika-pytorch-bf16`, `glm-4-voice-9b`, `minicpm-o-2.6`, `step-audio-2-mini`, `kimi-audio-7b-instruct` | speech-to-speech | - | CC-BY-4.0, glm-4-voice License, Apache-2.0, MIT | no | not runnable yet (full-duplex, or no packaged inference code; models.md) |

Install (CPU): `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`, then
`pip install "bolo-torch[parler,sepformer]"` with the extras you need. `parler` needs a
different `transformers` version from `pyannote` and `dicow`; run PixIT and the pyannote diarizers from a second environment with
`BOLO_PLUGINS="torch=/path/to/env/bin/python -m bolo_torch"`. GPUs: set `BOLO_TORCH_DEVICE=auto`
(bolo's own `device` setting defaults to `cpu`); no GPU was used in testing. Details, measurements
and licences: [plugins/bolo-torch/README.md](../plugins/bolo-torch/README.md).
