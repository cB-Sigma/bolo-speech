# Engine plugins (out of process)

bolo never imports PyTorch, and its built-in engines (sherpa-onnx, faster-whisper, Moonshine,
whisper.cpp, ONNX Runtime) run in the server process. Some good models need other libraries -
PyTorch above all (Indic Parler-TTS, IndicF5, Veena, Svara, SraVaani). An **engine plugin** runs
such models in a separate process that bolo starts, talks to over a pipe, restarts when it crashes
and stops when it is idle. The server stays small and torch-free; the plugin brings whatever it
needs.

Status: the server side, the protocol, the plugin-side helper and a reference plugin are part of
bolo 0.2.0 and are covered by tests (`server/tests/test_plugins.py`, which runs the reference
plugin as a real subprocess). The PyTorch plugin itself (`plugins/bolo-torch`, package name not
decided yet) is a separate package and is not written yet; the catalog already lists the models
that will need it, as "needs plugin `torch` (not installed)".

## Using a plugin

A plugin is found in two ways:

* **installed package**: a Python package that declares an entry point in the group
  `bolo.engines`. The entry point names a module, which bolo starts as
  `python -m <module>` with its own interpreter, or a function (`module:function`) that returns the
  command line to start:

  ```toml
  # pyproject.toml of the plugin package
  [project.entry-points."bolo.engines"]
  torch = "bolo_torch.plugin"
  ```

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
`"ok": false, "error": {"code", "message"}`). `synthesize` sends `{"id", "chunk": n, "bytes": …}`
messages before its final reply. Replies to different requests may interleave.

| `op` | request | final reply |
| --- | --- | --- |
| `hello` | `protocol` (highest version bolo speaks) | `protocol` (version used), `name`, `version`, `kinds` |
| `health` | - | `rss_mb` |
| `load` | `model`, `kind`, `task`, `path`, `options`, `threads`, `device` | `handle`, `sample_rate`, `info` (`voices`, `languages`, `max_batch`, `supports_translate`, `supports_word_timestamps`) |
| `unload` | `handle` | - |
| `transcribe` | `handle`, `requests: [{n, language, task, prompt, word_timestamps, temperature, hotwords}]` + the audio of all requests, concatenated | `results: [{text, language, segments: [{start, end, text, words}], words: [{w, t0, t1}]}]` |
| `synthesize` | `handle`, `text`, `voice`, `speed` | chunks, then `sample_rate`, `chunks`, `cancelled` |
| `cancel` | `target` (id of a running request) | - |
| `shutdown` | - | - (the process exits) |

Compatibility: new optional members can appear within version 1; an incompatible change gets
version 2, and bolo will keep speaking version 1 to plugins that answer `hello` with 1. The exact
definitions are in `server/src/bolo/plugin/protocol.py`.

## Models waiting for the PyTorch plugin

These are in the catalog (`server/src/bolo/catalog/plugin-torch.yaml`) with `engine: plugin:torch`,
pinned revisions and their licences. None was run by the maintainers.

| Model | Task | Languages | Licence | Gated |
| --- | --- | --- | --- | --- |
| `indic-parler-tts` (AI4Bharat) | TTS, voice described in words | 17 incl. 11 Indian | Apache-2.0 | yes |
| `indic-f5` (AI4Bharat IndicF5) | TTS, voice cloning | 11 Indian | MIT | yes |
| `veena` (Maya Research) | TTS, 3B Llama + SNAC | Hindi, English | Apache-2.0 | no |
| `svara-tts-v1` (Kenpath) | TTS, Llama + SNAC | 19 incl. Bhojpuri, Magahi, Maithili, Chhattisgarhi, Bodo, Dogri | Apache-2.0 | no |
| `sravaani-1.0` (ARTPARK-IISc) | STT (TorchScript) | 13 Indian + English | MIT | yes |
| `indicconformer-600m-official` (AI4Bharat) | STT, 22 scheduled languages | 22 | MIT | yes (the ungated community exports run without a plugin) |

The 3B-class models (Veena, Svara) need a GPU in practice; GPU use was not tested by the
maintainers.
