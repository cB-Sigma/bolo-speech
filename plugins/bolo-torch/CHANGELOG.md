# Changelog of bolo-torch

The PyTorch engine plugin of bolo. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.11] - 2026-11-22

### Added

- **SpeechBrain direct spoken-language understanding** (kind `speechbrain-slu`, extra `slu`, op `intent`):
  `slu-slurp-hubert` built from its `hyperparams.yaml` with SpeechBrain's classes, HuBERT from its configuration
  only (`config_only_encoders`: SpeechBrain's wrapper would download facebook/hubert-base-ls960's weights just for
  the checkpoint to replace them), decoded as the repository's interface does, `beam_size` option. Run on CPU:
  8 of 8 synthetic commands understood, RTF 10.6 at beam 8.

## [0.1.10] - 2026-11-21

### Added

- **Cloudflare Clef-Omni** (kind `clef-omni`, extra `clef-omni`, task `decision`): typed answers (yes/no, a choice, a
  score) about a text and a recording from Qwen3-Omni 30B-A3B's thinker and Cloudflare's joint head, through the
  model repository's `joint_schema_model.py` (remote code, opt in per model). Served on the plugin protocol's new
  `decide` op. Wiring tested with a stand-in for its `systemone`; the model itself (70.8 GB) not run yet.

- **A generic transformers speech-recognition loader** (kind `hf-asr-pipeline`, extra `hf-asr`): a model's
  `automatic-speech-recognition` pipeline, the repository's own one through remote code (opt in per model), with the
  language passed under a name and mapping the catalog gives. First user: **IFM Jais-ASR** (`jais-asr`, extra
  `jais-asr`, transformers 5.16.1). Wiring tested with a fake pipeline; the model (11.4 GB) not run yet.
- **Rev's Reverb ASR** (kind `reverb-asr`, extra `reverb`): the repository's TorchScript export through WeNet's export
  interface (encoder, CTC), greedy CTC over its BPE units, Kaldi filterbanks from torchaudio and Rev's
  verbatimicity. Rev's own package pins torch 2.2.2 and is not used. Run on CPU: an English clip word for word at 0.05
  of real time.
- **Rev's Reverb diarization v2** (`reverb-diarization-v2-torch`) on the existing pyannote loader. Runs on CPU, very
  slowly (16 times real time); torchaudio downloads the WavLM Base+ weights on its first load.
- **Salesforce SOV-Transcribe-1** (kind `sov-transcribe`, extra `sov-transcribe`): speech to text in English, French,
  German, Spanish, Italian, Portuguese and Dutch through transformers 5.19 to 5.22 and the model's own code (remote
  code, opt in per model), with the prompts of its card (language hint, hotwords as words the audio may contain).
  Wiring tested with a fake network; the model itself (53 GB) not run yet.

## [0.1.9] - 2026-11-19

### Changed

- Needs bolo 1.0.0 or later (below 2).

### Fixed

- **Parler-TTS spoke noise on Apple silicon's GPU.** On MPS (torch 2.15, an M4 Pro) its speech came out as noise in
  float16 and in float32 while the CPU's was right; Parler models now run on the CPU when MPS is chosen, with the
  reason as the load's `device_fallback` (bolo shows it in `bolo ps`).
- **bolo showed models on the CPU that the plugin ran on MPS.** With `BOLO_TORCH_DEVICE=mps`, demucs-dns48 loaded
  "on mps, float16" while `bolo ps` said "100% CPU on cpu": the load reply now carries the device (`info.device`) and
  bolo moves the placement there.
- **GLM-4-Voice could not import its libraries in a new environment.** lightning 2.2 imports `pkg_resources`, which
  setuptools 81 and later no longer have; the `glm-4-voice` extra now keeps setuptools below 81.
- **`bolo check --tiny` could not load Step-Audio 2 mini.** Its model code rebuilds the text model's configuration
  with `num_attention_groups` at its default and asserts it equals `num_key_value_heads`, which the tiny build had
  reduced; the tiny configuration now names both.

Checked on an M4 Pro with MPS (warm): pyannote speaker-diarization-community-1 0.83 s against 6.7 s on the CPU for
22 s of audio, the same two speakers; SepFormer 3.4 s against 5.9 s, the same two clean tracks; demucs-dns48 works.

## [0.1.8] - 2026-11-18

### Added

- **`bolo-torch check`**, what `bolo check <model>` asks the plugin's environment: the loader and the extra's
  libraries import, the code bundle matches its pins, and with `tiny` the model's architecture built small from its
  configuration, split-planned, loaded in bfloat16 and run once.

### Fixed

- **GLM-4-Voice int4 split across the GPU and the CPU failed with meta tensors.** bolo-torch passed
  `device_map="auto"`, and transformers converts the layers to 4-bit before it resolves that, so the layers bound
  for the CPU became 4-bit there and accelerate's offload could not load them ("Cannot copy out of meta tensor"). The
  plugin now works out an explicit device map first (transformers' documented way to keep the CPU's share
  unquantized), sized for 4-bit layers with the embeddings and output layer in 16-bit, and the decoder blocks whole:
  for GLM-4-Voice 9B on a 6 GB GPU, 29 of its 40 blocks on the GPU. Reproduced and checked on CPU with GLM-4-Voice's
  own code (transformers 4.44.1); the 4-bit load itself needs CUDA and was not run here.
- **Extras that bring librosa ask for 0.10.2 or later** (`f5`, `step-audio-2`, `glm-4-voice`, `kimi-audio`,
  `minicpm-o`, `maskgct`): older librosa imports `pkg_resources`, which setuptools 81 no longer has.
- **A failed load gives its GPU memory back.** Before answering a failed `load`, the plugin clears the loader's
  frames, runs the garbage collector, then `torch.cuda.empty_cache()` and `torch.cuda.ipc_collect()` (MPS:
  `torch.mps.empty_cache()`), and adds `error.gpu` to the reply: `{"initialized": false}` when CUDA was never started,
  else the bytes still allocated and reserved. bolo uses it to restart the plugin when nothing else is loaded in it
  (bolo bug 17).

## [0.1.7] - 2026-11-18

### Security

- **README: "Security notes".** The `glm-4-voice`, `minicpm-o` and `step-audio-2` extras pin transformers 4.44.1,
  4.44.2 and 4.49.0, which their models' remote code needs and which have published advisories. The pins stay; the
  README says so and recommends a separate plugin environment for each.

### Added

- **`bolo-torch install` keeps a GPU build of torch even when pip replaces it**: it reinstalls the build that was
  there from its PyTorch index and says so. bolo itself installs the GPU build in one step (`bolo torch use cuda`,
  bolo-speech's CHANGELOG).

### Fixed

- **A CPU-only torch next to an NVIDIA GPU (PyPI's torch on Windows).** With `device = "cuda"` the
  model was refused ("this torch build sees no CUDA device"), so a Demucs denoiser was skipped and live and file
  denoising returned the audio untouched (`x-bolo-denoise: skipped`); with the model's option `device = "auto"` it
  ran on the CPU while bolo, `bolo ps` and `/api/v1/ps` said "100% GPU cuda:0". In both cases the model now loads on
  the CPU with a notice saying why and how to get a CUDA build of torch, and the plugin reports it, so the placement
  says CPU (as `rocm` already did). `bolo doctor` warns when bolo-torch has a CPU-only torch on a machine with an
  NVIDIA GPU, and the plugin's README and docs/configuration.md give the CUDA index
  (`https://download.pytorch.org/whl/cu128`). The `pyannote` extra's floor stays at pyannote.audio 3.3.2 (pip takes
  4.x, which needs torch 2.8, unless `bolo-torch install pyannote` keeps an older torch); the README says so.
- **Installing an extra replaced a CUDA build of torch with a CPU one.** On Windows with CUDA torch installed,
  `pip install "bolo-torch[pyannote]"` let pip take the newest torchaudio, which needs a newer torch, so pip
  installed PyPI's torch 2.15, a CPU build, and pyannote ran on the CPU. A new command, `bolo-torch install
  <extras>` (`--dry-run` prints it), installs extras with the installed torch pinned, torchaudio and torchvision
  held to its release, and the PyTorch index of its build (`+cu128`, `+rocm6.4`, `+xpu`, `+cpu`) added.
- **Step-Audio 2 mini could not load on a GPU under 24 GB.** When bolo split it between the GPU and the CPU, transformers
  refused: "StepAudio2ForCausalLM does not support `device_map='auto'`" (its remote code declares no
  `_no_split_modules`). When a model class names no layers to keep whole, the plugin now builds the model without
  weights (accelerate's `init_empty_weights`), takes its repeated layers (the elements of its largest
  `nn.ModuleList`) as the layers to keep on one device, and passes accelerate's device map for bolo's `max_memory`
  to `from_pretrained`. A model with no repeated layers is refused before loading, saying how many GB it needs on one
  device. Applies to every loader that splits (Step-Audio 2, Qwen-Omni, GLM-4-Voice, MiniCPM-o, Orpheus SNAC);
  under transformers 5, which splits such a class anywhere, also inside a layer, the layers are now kept whole too.

## [0.1.6] - 2026-11-18

### Fixed

- **A model library that asks a question on first use no longer stalls the plugin** (DiariZen's first request
  hung until bolo's 10-minute timeout): it gets an empty stdin and its output goes to the log (bolo-speech 0.6.8's
  plugin SDK).
- **GLM-4-Voice in 4-bit (`quantization = "int4"`) loads again.** The `glm-4-voice` extra pins transformers 4.44.1
  but left accelerate free, so accelerate 1.x was installed, and its dispatch refuses to move a bitsandbytes
  model ("`.to` is not supported for 4-bit or 8-bit bitsandbytes models"). The extras that pin transformers
  4.44 (`glm-4-voice`, `minicpm-o`) now also pin `accelerate<1.0`.
- **GLM-4-Voice in 4-bit follows bolo's GPU/CPU split.** It was always loaded whole on the GPU, so on a 6 GB card it
  took 7 GB and spilled into Windows' shared memory. When bolo asks for a split, what does not fit stays on the CPU
  (bitsandbytes' CPU offload); otherwise it loads whole on the device as before.

## [0.1.5] - 2026-11-18

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

