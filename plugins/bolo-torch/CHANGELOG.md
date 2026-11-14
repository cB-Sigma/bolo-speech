# Changelog of bolo-torch

The PyTorch engine plugin of bolo. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] - 2026-11-14

### Changed

- Installs with bolo 0.3 as well as 0.2 (`bolo-speech>=0.2,<0.4`). No change to the plugin's code.

## [0.1.0] - 2026-11-14

First release, for bolo 0.2.

### Added

- The original F5-TTS checkpoints (`f5-tts` kind with `layout: f5-tts`, extra `f5-original`): catalog entry
  `f5-tts-v1-base` (CC-BY-NC-4.0). F5 voices can name bolo's built-in reference clips (`reference:`).
- Indic-Speak (`indic-speak` kind, extra `indic-speak`): Bodhan AI's gated Indic TTS through its repository's own
  `inference.py`; catalog entry `indic-speak`. Kinds `diarizen`, `spark-tts` and `maskgct` are listed as not
  implemented (catalogued with a note on what is missing). Unit-tested with fakes only.
- DiCoW (`dicow` kind, extra `dicow`): BUT-FIT's diarization-conditioned Whisper as a speaker-conditioned STT
  model (bolo's `speakers` plugin request). Catalog entries `dicow-v3.3` (CC-BY-4.0) and `se-dicow` (Apache-2.0,
  self-enrolment), both run here on CPU. STNO masks and self-enrolment windows are built as BUT's inference
  pipeline builds them; on two AMI excerpts the transcripts were identical to that pipeline's for every speaker. bfloat16 on CPUs
  with AMX or AVX512-BF16. Needs transformers 4.55.
- Engine plugin `torch` (entry point `bolo.engines: torch = bolo_torch:plugin`), served with bolo's plugin SDK;
  errors keep their meaning across the process boundary (`model_gated`, `invalid_request`,
  `component_missing`, `download_failed`).
- Parler-TTS (`parler-tts` kind): Indic Parler-TTS and Parler-TTS Tiny v1 in bolo's catalog, named voices or
  free-text descriptions, sentence-by-sentence streaming, optional in-sentence streaming (`stream_seconds`).
- IndicF5 (`f5-tts` kind): loaded from the model repository's bundled `f5_tts` and weights, reference voices from
  the catalog or a folder (`BOLO_TORCH_VOICES_DIR`).
- Veena and Svara (`orpheus-snac` kind): Llama-style language model plus SNAC decoder, prompts per model card,
  audio streamed while tokens are generated. Not run end to end (3B models, no GPU here).
- Speech separation (`sepformer`, `pixit` kinds) as bolo `separation` models in blind mode: SpeechBrain SepFormer
  WHAMR! 8 kHz and 16 kHz, pyannote PixIT `speech-separation-ami-1.0`; `bolo-torch separate` runs them without
  bolo.
- Device (`auto`, `cpu`, `cuda`, `mps`), dtype and thread selection; downloads through the Hugging Face hub with
  `HF_TOKEN`; `bolo-torch doctor`.
- Extras per model family: `parler`, `f5`, `orpheus`, `sepformer`, `pyannote`.

