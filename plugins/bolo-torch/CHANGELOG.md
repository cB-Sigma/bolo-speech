# Changelog of bolo-torch

The PyTorch engine plugin of bolo. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **MaskGCT** (kind `maskgct`, extra `maskgct`, code bundle `maskgct`): Amphion's `MaskGCT_Inference_Pipeline`
  methods unchanged, as `maskgct_inference.py` runs them, with the networks' weights from the model folder,
  w2v-bert-2.0 at a pinned revision and the feature statistics from the bundle (`weights_only`); github.com/open-
  mmlab/Amphion at a pinned commit, 46 files sha256-pinned. Voice cloning from reference clips with their
  transcripts; Chinese and English mixed, Japanese, Korean, French and German; `speed` scales the length rule. The
  GPL-licensed parts of the phoneme front end (phonemizer with espeak-ng, pykakasi, Unidecode) are left out of the
  extra and installed by the operator; phonemizer is pointed at espeakng-loader's espeak-ng; LangSegment 0.2.0's
  broken package import is worked around. transformers is held below 4.48 (Amphion's Llama layers). The front end ran
  here in all six languages and the pipeline with tiny random networks; the weights were not run. CC-BY-NC-4.0.
- **Spark-TTS 0.5B** (kind `spark-tts`, extra `spark-tts`, code bundle `spark-tts`): the authors' recipe
  (`cli/SparkTTS.py`): the Qwen2.5-0.5B language model through transformers with their sampling, and the BiCodec
  tokenizer and decoder from github.com/SparkAudio/Spark-TTS at a pinned commit (20 files, sha256-pinned). Created
  voices (gender and pitch level; the request's speed as one of five speed levels) and cloned voices (bolo's
  reference clips or `voices_dir`, with their transcripts). Tested with fakes; the bundle's BiCodec code ran here
  with random weights through the plugin's token paths. Weights CC-BY-NC-SA-4.0.
- **DiariZen** (kind `diarizen`, extra `diarizen`): BUT-FIT's DiariZen v2 diarization through the authors' own
  `DiariZenPipeline` and their pyannote.audio 3.1.1 fork, both installed from github.com/BUTSpeechFIT/DiariZen at a
  pinned commit (the loader refuses any other commit, read from pip's `direct_url.json`), in an environment of their
  own. Audio reaches the pipeline as an in-memory float WAV; the WeSpeaker embedding is fetched at a pinned revision
  and passed as a checkpoint dictionary; the two checkpoints load under `weights_only` with pyannote's task classes
  and torch's version object allowed by name. Served through the `diarize` and `embed` ops. Run here on CPU through
  bolo (DER 5.0 % on the demo dialogue, RTF 12 on a shared 4-vCPU machine). Weights CC-BY-NC-4.0.
- **Remote code, opted in per model and pinned** (`bolo_torch.remote_code`): `trust_remote_code=True` is passed only
  when the load options say `allow_remote_code: true` (bolo's per-model setting) and the catalog's
  `options.remote_code` pins the folder's revision and the sha256 of every `.py` file and of every JSON file with an
  `auto_map`; an unlisted `.py` file, a changed file, another revision or an `auto_map` into another repository is
  refused (`remote_code_refused`). Each load logs a warning naming the files. `bolo-torch pin-remote-code <folder>
  --revision <commit>` prints a pin. Configuration files that name classes (hyperpyyaml) are checked against the
  entry's `sha256` option before they are read.
- **Inference code from GitHub, pinned** (`bolo_torch.upstream`): bundles of files from the authors' repositories at
  a fixed commit, each file pinned by sha256 and each repository's licence recorded
  (`src/bolo_torch/upstream/*.json`, written by `tools/pin_upstream.py`). `bolo-torch code list | install <name> |
  verify`; a load installs a missing bundle (`BOLO_TORCH_CODE_AUTO_INSTALL=0` turns that off) into
  `BOLO_TORCH_CODE_DIR` and checks every file before importing.
- **Step-Audio 2 mini** (`kind: step-audio-2`, `step-audio-2` extra): the authors' prompt format, the model's remote
  code (opt-in), token2wav from stepfun-ai/Step-Audio2 at a pinned commit; text streamed as the model writes it, tool calls
  from `<tool_call>` blocks, speech in pieces of 25 speech tokens (first audio after about 1 s of speech), voices
  `female` and `male`, cancel inside generation. Without CUDA it answers in text only, with a warning. Tested with
  fakes only.
- **GLM-4-Voice 9B** (`kind: glm-4-voice`, `glm-4-voice` extra): the authors' web-demo recipe (speech tokens from
  `WhisperVQEncoder`, the interleaved prompt, `AudioDecoder.token2wav` in blocks of 25 to 200 tokens), ChatGLM's
  remote code (opt-in), the GLM-4-Voice and Matcha-TTS code at pinned commits, the decoder's `config.yaml` checked by
  sha256, `quantization: int4` (bitsandbytes NF4). Each load logs the licence: commercial use only after registering
  with Zhipu. Tested with fakes only.
- **Kimi-Audio 7B Instruct** (`kind: kimi-audio`, `kimi-audio` extra, flash-attn installed separately): `KimiAudio` from
  `kimia_infer` (MoonshotAI/Kimi-Audio at a pinned commit, with GLM-4-Voice's speech tokenizer from its submodule),
  remote code opt-in, the speech tokenizer from the pinned folder instead of the Hub, `_generate_loop` run one step at
  a time: text streamed, speech through `detokenize_streaming` in 30-token chunks, cancel between steps. CUDA only, with
  a clear error elsewhere and a hint for flash-attn. Tested with fakes only.
- The `moshi` kind's message says where Moshi runs now: in Kyutai's own server, which bolo starts for a full-duplex
  `s2s_model` (not in this plugin).
- README: running the four speech-to-speech models on a Windows laptop with an NVIDIA GPU, step by step.
- **MiniCPM-o 2.6** (`kind: minicpm-o`, `minicpm-o` extra): the model card's recipe on the repository's remote code
  (opt-in, pinned): `init_tts`, the audio-assistant system prompt with the card's reference voices, the streaming API
  with the conversation prefilled (`mode: stream`, default) or `chat()` (`mode: chat`, and text-only answers). Refuses
  to load when Vocos or the ChatTTS tokenizer is missing, instead of letting its code download them. Tested with fakes
  only.
- **Spoken-language identifiers** (bolo task `lid`), served through the protocol's new `lid` op (needs a bolo that
  sends it): `voxlect-lid` (Voxlect's classifiers for India, rebuilt from transformers' Whisper or wav2vec 2.0
  encoder and the checkpoint's weights with the LoRA products merged, without the Voxlect code; `voxlect` extra),
  `speechbrain-lid` (SpeechBrain ECAPA-TDNN classifiers built from SpeechBrain's classes, without its hub loader;
  `speechbrain` extra) and `hf-audio-classification` (any transformers audio classifier whose labels are languages,
  Auto classes, no remote code). Run here: `voxlect-indic-lid-whisper-small`, `lid-indic-vaani`, `indic-slid-mhubert`
  and `indic-slid-mhubert-task2`.

- **Demucs denoiser** (`kind: demucs-denoiser`, `demucs-denoiser` extra): `demucs-dns48`, `demucs-dns64`,
  `demucs-master64` and `demucs-valentini-nc` from facebookresearch/denoiser as bolo `denoise` models, served through
  the protocol's new `denoise` op (needs a bolo that sends it). Code and weights are CC-BY-NC-4.0: not for commercial
  use. `pip install --no-deps denoiser julius` is enough (the package's other dependencies are for training).
- **Qwen2.5-Omni and Qwen3-Omni as speech-to-speech models** (`kind: qwen-omni`, `omni` extra): the catalog's
  `qwen2.5-omni-3b-torch` (Qwen Research License, non-commercial), `qwen2.5-omni-7b-torch` and
  `qwen3-omni-30b-a3b-instruct-torch` (Apache-2.0), served through the protocol's new `respond` op (needs a bolo that
  sends it): the whole conversation in (turns as 16 kHz audio or text), the answer's text and 24 kHz speech out in
  one piece, the model's voices, Qwen2.5-Omni's required system prompt before the session's instructions;
  transformers' own model classes, no remote code. Tested with fake model objects only; no real model was run.
- Model kind `moshi` is known and answers that it cannot be loaded yet (full-duplex). `glm-4-voice`, `minicpm-o`,
  `step-audio-2` and `kimi-audio`, listed the same way at first, now load (above).
- **pyannote speaker diarization** (`kind: pyannote-diarization`, `pyannote` extra): `pyannote-speaker-diarization-3.1`
  and `pyannote-speaker-diarization-community-1` as bolo `diarization` models, served through the protocol's new
  `diarize` and `embed` ops (needs a bolo that sends them). Turns in bolo's format, `num_speakers` from the request,
  `num_speakers` / `min_speakers` / `max_speakers` / `exclusive` from the catalog, voice prints from the pipeline's
  own embedding model, device cpu/cuda/mps as for every model.
- **Configuration guard** for pyannote pipelines (diarization and PixIT): the plugin never hands pyannote a
  `config.yaml`. It checks the files' sha256 against the catalog, parses the configuration with `yaml.safe_load`,
  allows only `pyannote.audio.pipelines.SpeakerDiarization` / `SpeechSeparation`, the clustering names
  `AgglomerativeClustering` / `VBxClustering`, plain parameter values and sub-models that bolo downloaded or that
  are pinned to a commit, and passes pyannote the checked dictionary. Anything else (another class, `os.system`,
  `preprocessors`, `_target_`, an unpinned repository) is refused before pyannote is imported.

### Changed

- PixIT loads its separation model from bolo's pinned download (or the pinned revision) instead of the Hub's
  latest, and pyannote.audio 4's usage metrics are off unless `PYANNOTE_METRICS_ENABLED` is set.

## [0.1.4] - 2026-11-16

Windows: the fast tests run on Windows in CI. Installs with bolo 0.6.

### Added

- CI runs the fast tests on Windows (`windows-latest`, CPU torch); the package declares
  `Operating System :: Microsoft :: Windows`. With bolo 0.6 the worker runs in a Windows Job Object:
  `plugin_nice` and `plugin_memory_limit_mb` apply there too, and the worker ends with bolo. GPU paths on Windows
  are untested.

### Changed

- Installs with bolo 0.6 as well as 0.2 to 0.5 (`bolo-speech>=0.2,<0.7`).

## [0.1.3] - 2026-11-15

Placement protocol: bolo 0.5 can ask for a GPU/CPU split with `device_map` placement. Installs with bolo 0.5.

### Added

- **GPU/CPU splits.** When bolo 0.5's scheduler finds less free GPU memory than a model needs, it sends a
  `placement` with the load (`device_map="auto"` and a `max_memory`); the Orpheus SNAC loader then lets accelerate
  put as many layers on the GPU as fit and the rest on the CPU, and reports the split back
  (`info.placement`: GPU and CPU bytes, the device map), which `bolo ps` shows as "x%/y% CPU/GPU". Other loaders
  load whole as before. With an older bolo nothing changes. Tested with fakes only; no GPU was used.

### Changed

- Installs with bolo 0.5 as well as 0.2 to 0.4 (`bolo-speech>=0.2,<0.6`).

## [0.1.2] - 2026-11-14

### Changed

- Installs with bolo 0.4 as well as 0.2 and 0.3 (`bolo-speech>=0.2,<0.5`). No change to the plugin's code.

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

