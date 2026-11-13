# Models converted by the bolo project

bolo runs without PyTorch, but some good voices and recognisers exist only as PyTorch
checkpoints. The files here convert them once and publish the result as assets of a GitHub
release of this repository and, when the `HF_TOKEN` secret is set, as one public Hugging Face
repository per model under `codebucketsolutions/`, so that the conversion is public and repeatable.

| File | What it does |
| --- | --- |
| `syspin.json` + `export_syspin.py` | IISc SYSPIN VITS voices (MIT): checkpoint → `model.onnx` + `tokens.txt` + `voice.json` + licence files → `<id>.tar.gz`, `SHA256SUMS`, `voices.json` |
| `mms.json` + `export_mms.py` | Meta MMS-TTS voices for Indian languages (**CC-BY-NC-4.0, non-commercial**), with the original VITS inference code pinned; each archive carries the CC-BY-NC-4.0 text and the attribution |
| `whisper.json` + `convert_whisper.py` | Whisper fine-tunes → CTranslate2 int8 for faster-whisper (empty today: see below) |
| `publish_hf.py` | uploads each model with a model card (licence, attribution, provenance, pinned source, measured numbers from `measured.json`) and records the Hub commit in `voices.json` |
| `update_catalog.py` | points the catalog entries at the Hub repository (pinned to the uploaded commit) with the release archive as `source.fallback`, or at the archive alone, and removes `status: pending` |
| `requirements-*.txt` | what each converter needs besides CPU PyTorch |

The voices are the IISc SYSPIN VITS voices (`SYSPIN/vits_<Language>_<Gender>` on Hugging Face):
weights under the MIT licence, trained on the SYSPIN corpus (CC-BY-4.0), by the SYSPIN project,
SPIRE Lab, Indian Institute of Science. Every archive carries `LICENSE` and `ATTRIBUTION`.

## Publishing

1. Actions → **Voices** → "Run workflow" (inputs: release tag, default `voices-v1`; optional
   comma-separated model ids from any of the three lists). It converts each model, runs a smoke
   test and creates or updates the release; with the `HF_TOKEN` secret it then uploads every model
   to `codebucketsolutions/bolo-<family>-<language>[-<gender>]` (created if missing, public) and
   re-uploads `voices.json` with the Hub commits. Without the secret that step says so and does
   nothing.
2. `python tools/voices/update_catalog.py` (reads `voices.json` from the release), then
   `python server/tools/gen_docs.py`, and commit the catalog.

## Locally

```bash
python -m venv venv && . venv/bin/activate
pip install "torch==2.8.*" "torchaudio==2.8.*" --index-url https://download.pytorch.org/whl/cpu
pip install -r tools/voices/requirements-syspin.txt
python tools/voices/export_syspin.py --out dist --only syspin-mai-female
```

About 2 minutes and 4 GB of scratch disk per voice on a laptop CPU (the checkpoint is 950 MB).
A voice that is already converted can be packed and checked without PyTorch:
`--exported syspin-mai-female=/path/to/dir`. To use a converted directory with bolo directly:
`bolo pull ./dir --kind vits --name syspin-mai-female --language mai`.

Status (2026-11-13): the workflow published the 22 SYSPIN voices as release `voices-v1`, and the
catalog entries were switched on with `update_catalog.py`. `export_mms.py` was run by the
maintainers for Punjabi, Odia, Assamese, Dogri, Maithili, Magahi, Chhattisgarhi and Urdu (each
loaded by bolo and transcribed back, `docs/measurements/mms-tts-round-trip.json`); the MMS voices
are not published yet. `publish_hf.py` and the Hub source were tested with a stand-in for the Hub
API (`server/tests/test_voice_tools.py`), not against the Hub.

`convert_whisper.py` converted `Aditya109/whisper-small-bhojpuri-merged` (CC-BY-4.0) here. On 120
RESPIN Bhojpuri test utterances it did worse than the Hindi IndicConformer model that bolo already
uses for Bhojpuri (see docs/indic-telephony.md), so it is not listed in `whisper.json` and not
published. No permissively licensed Maithili or Magahi Whisper fine-tune was found.
