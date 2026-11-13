# Changelog of bolo-torch

The PyTorch engine plugin of bolo. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - unreleased

First release, for bolo 0.2.

### Added

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
