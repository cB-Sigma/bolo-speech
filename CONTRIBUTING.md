# Contributing to bolo

Thanks for considering it. Bug reports, measurements on hardware we do not
have, new catalog entries and code are all welcome.

## Development setup

```bash
git clone https://github.com/cb-Sigma/bolo && cd bolo/server
python -m venv .venv && . .venv/bin/activate          # Python 3.10 to 3.13
pip install -e '.[all,dev]'

ruff check src tests tools && ruff format --check src tests tools
mypy
pytest -q                                              # about 90 seconds, needs no model and no network
```

The suite uses a deterministic fake engine, so it runs anywhere. The tests
that download and run real models are opt-in:

```bash
BOLO_TEST_REAL=1 pytest -m real -k "not large"         # about 600 MB of models
BOLO_TEST_NETWORK=1 pytest -m network                  # checks every catalog source against the Hub
```

To work on the API with nothing downloaded:

```bash
bolo serve --fake-engine --cors-origin '*'
```

The web console lives in `ui/` (`npm ci && npm run dev`); `npm run build`
writes the bundle into `server/src/bolo/_ui/`, which is committed so that
`pip install` needs no Node.

## Ground rules

* **No PyTorch** in the core or in any default extra. Inference goes through
  ONNX Runtime, CTranslate2 or another native library with wheels.
* **Licences are part of the code.** bolo is Apache-2.0. Do not add a
  dependency under GPL/AGPL, and do not copy code from projects whose licence
  is incompatible. Every catalog entry states the licence of its model.
* **Do not claim what was not run.** `validated: cpu` and every number in the
  docs come from a real run. Mark things you could not test as untested.
* Tests come with the change. Timing-sensitive code is tested against the
  fake engine with known delays.
* Keep the public API (`docs/api.md`) backwards compatible; additive changes
  are fine.

## Adding a model to the catalog

The catalog is data: `server/src/bolo/catalog/*.yaml`. An entry looks like this:

```yaml
- id: zipformer-xx-streaming            # lowercase, digits, '.', '_' and '-'
  aliases: [owner/original-repo-name]   # optional extra names clients may send
  task: stt                             # stt | tts | vad
  engine: sherpa-onnx                   # which plugin loads it
  kind: online_transducer               # the loader inside that engine (see below)
  family: Zipformer
  languages: [xx]                       # ISO 639-1/3 codes
  tier: realtime                        # realtime | balanced | accuracy
  streaming: true                       # only for natively incremental models
  size_bytes: 123456789                 # what will be downloaded
  params: 66000000                      # optional
  license: Apache-2.0                   # the MODEL's licence, as its publisher states it
  commercial_use: false                 # only when the licence forbids commercial use
  features: [timestamps]
  description: One or two honest sentences.
  source_url: https://…                 # where the model comes from (upstream, not the mirror)
  validated: untested                   # leave this; see below
  source:
    repo: owner/repo                    # Hugging Face model repository
    revision: <40-character commit>     # pinned, so pulls are reproducible
    include: ["encoder*.int8.onnx", "decoder*.onnx", "joiner*.int8.onnx", "tokens.txt"]
    exclude: ["test_wavs/*", "*.md"]
  options: {}                           # per-model overrides (threads, max_batch, …)
```

Loader kinds:

| Engine | `kind` values |
| --- | --- |
| `sherpa-onnx` (speech-to-text) | `whisper`, `moonshine`, `moonshine_v2`, `transducer`, `nemo_transducer`, `nemo_ctc`, `zipformer_ctc`, `sense_voice`, `paraformer`, `dolphin_ctc`, `omnilingual_ctc`, `telespeech_ctc`, `wenet_ctc`, `canary`; streaming: `online_transducer`, `online_nemo_ctc`, `online_zipformer2_ctc`, `online_paraformer`, `online_t_one_ctc` |
| `sherpa-onnx` (text-to-speech) | `vits` (Piper format, MMS, Coqui, MeloTTS), `kokoro`, `kitten`, `matcha` |
| `sherpa-onnx` (VAD) | `silero_vad`, `ten_vad` |
| `faster-whisper` | `ct2-whisper` |

Steps:

1. Try it first without touching the catalog: `bolo pull hf:owner/repo` (add
   `--kind …` if the layout is ambiguous), then `bolo transcribe` / `bolo speak`.
2. Add the entry. Find the commit with
   `curl -s https://huggingface.co/api/models/owner/repo | jq -r .sha`.
   Use `include` patterns that select one precision (usually int8).
3. Run `pytest tests/test_store_catalog.py` (shape) and
   `BOLO_TEST_NETWORK=1 pytest -m network` (the source resolves and the size matches).
4. Validate it on your machine and record the result:

   ```bash
   python tools/validate_models.py --out report.json your-model-id
   python tools/apply_validation.py report.json
   python tools/gen_docs.py                    # refreshes the tables in README.md and docs/
   ```

   `validated: cpu` may only be set by that tool from a real run. If you
   validated on other hardware (a GPU, Apple Silicon, arm64), say so in the
   pull request: we would like to record those too.

Voices that need phonemes from espeak-ng are welcome in the catalog; give them
`requires: [espeak-ng]` so that they show as unavailable until that optional
component is installed. Do not add anything GPL-licensed to the dependencies
or to the `all` extra. Gated models may be added with `gated: true`; they are
never downloaded without the user's own token.

## Adding an engine

Implement `bolo.engines.base.Engine` in `server/src/bolo/engines/<name>.py`,
register it in `BUILTIN_ENGINES` (`engines/__init__.py`), import the
underlying library inside functions only (never at module import), add an
optional extra to `pyproject.toml`, and add the module to the mypy override
list if it has no type hints. `engines/faster_whisper.py` is a compact
example; `engines/fake.py` shows every interface.

## Pull requests

Small, focused pull requests are easiest to review. Describe what you ran.
If the change affects latency or throughput, include `bolo bench` output from
before and after, and the hardware.

By contributing you agree that your contribution is licensed under the
Apache License 2.0.

## Releasing (maintainers)

1. Set `__version__` in `server/src/bolo/__init__.py` and move the
   "Unreleased" section of `CHANGELOG.md` under the new version.
2. Merge to `main` with CI green.
3. Run the **Release** workflow from the Actions tab ("Run workflow" on
   `main`). It builds the console and the distributions, publishes them to
   PyPI with the `PYPI_API_TOKEN` repository secret, creates the tag
   `vX.Y.Z` and the GitHub release with generated notes, and pushes the
   container image. Pushing a tag `vX.Y.Z` by hand starts the same workflow.
