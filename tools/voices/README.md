# Voices converted by the bolo project

bolo runs without PyTorch, but some good, permissively licensed voices exist only as PyTorch
checkpoints. The files here convert them once and publish the result as assets of a GitHub
release of this repository, so that the conversion is public and repeatable.

| File | What it does |
| --- | --- |
| `syspin.json` | the voices: source repository, pinned revision, language, a test sentence |
| `export_syspin.py` | checkpoint → `model.onnx` + `tokens.txt` + `voice.json` + licence files → `<id>.tar.gz`, `SHA256SUMS`, `voices.json` |
| `update_catalog.py` | writes the published sizes and checksums into the catalog and removes `status: pending` |
| `requirements.txt` | what the export needs besides CPU PyTorch |

The voices are the IISc SYSPIN VITS voices (`SYSPIN/vits_<Language>_<Gender>` on Hugging Face):
weights under the MIT licence, trained on the SYSPIN corpus (CC-BY-4.0), by the SYSPIN project,
SPIRE Lab, Indian Institute of Science. Every archive carries `LICENSE` and `ATTRIBUTION`.

## Publishing

1. Actions → **Voices** → "Run workflow" (inputs: release tag, default `voices-v1`; optional
   comma-separated voice ids). It exports each voice, runs a smoke synthesis with sherpa-onnx
   and creates or updates the release.
2. `python tools/voices/update_catalog.py` (reads `voices.json` from the release), then
   `python server/tools/gen_docs.py`, and commit the catalog.

## Locally

```bash
python -m venv venv && . venv/bin/activate
pip install "torch==2.8.*" "torchaudio==2.8.*" --index-url https://download.pytorch.org/whl/cpu
pip install -r tools/voices/requirements.txt
python tools/voices/export_syspin.py --out dist --only syspin-mai-female
```

About 2 minutes and 4 GB of scratch disk per voice on a laptop CPU (the checkpoint is 950 MB).
A voice that is already converted can be packed and checked without PyTorch:
`--exported syspin-mai-female=/path/to/dir`. To use a converted directory with bolo directly:
`bolo pull ./dir --kind vits --name syspin-mai-female --language mai`.

Status: the packing, smoke-test, index and catalog-update steps were run by the maintainers on
voices exported earlier with the same recipe; the export step as written in this script and the
workflow itself had not been run when this was committed.
