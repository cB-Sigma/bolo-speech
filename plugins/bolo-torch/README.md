# bolo-torch

The PyTorch engine plugin for [bolo](https://github.com/cB-Sigma/bolo-speech), the torch-free speech server.

bolo never imports PyTorch. When a catalog entry says `engine: plugin:torch`, bolo starts this package in a
separate process (`python -m bolo_torch`) and talks to it over the plugin protocol
([docs/plugins.md](https://github.com/cB-Sigma/bolo-speech/blob/main/docs/plugins.md)): it loads models there, streams synthesis back sentence by sentence,
restarts the process if it crashes and stops it when it has been idle. Everything else (queues, traces, metrics,
the console, the OpenAI-compatible API) works as for bolo's own engines.

## What it runs

| Catalog id | Task | Model | Extra | Licence | Gated | Tested here |
| --- | --- | --- | --- | --- | --- | --- |
| `indic-parler-tts` | TTS | AI4Bharat Indic Parler-TTS, 69 named voices, 21 languages | `parler` | Apache-2.0 | yes | CPU, see below |
| `parler-tts-tiny-v1` | TTS | Parler-TTS Tiny v1, English | `parler` | Apache-2.0 | no | CPU, see below |
| `indic-f5` | TTS | AI4Bharat IndicF5, voice cloning from a reference clip, 11 languages | `f5` | MIT | yes | CPU, see below |
| `f5-tts-v1-base` | TTS | SWivid F5-TTS v1 Base, English and Chinese voice cloning (bolo's reference clips or `voices_dir`) | `f5-original` | CC-BY-NC-4.0 (non-commercial) | no | **no** |
| `indic-speak` | TTS | Bodhan AI Indic-Speak, 3B Llama + SNAC + Vocos, 22 Indian languages and English | `indic-speak` | Indic Open Model License v1.0 (conditional) | yes | **no** |
| `veena` | TTS | Maya Research Veena, 3B Llama + SNAC, Hindi and English | `orpheus` | Apache-2.0 | no | **no** (only the SNAC decoder) |
| `svara-tts-v1` | TTS | Kenpath Svara v1, 3B Llama + SNAC, 19 languages, 38 voices | `orpheus` | Apache-2.0 | no | **no** (only the SNAC decoder) |
| `spark-tts-0.5b` | TTS | SparkAudio Spark-TTS 0.5B, English and Chinese, created voices (female, male) and voice cloning | `spark-tts` + the authors' BiCodec code (bundle `spark-tts`) | CC-BY-NC-SA-4.0 (non-commercial) | no | **no** (fakes; BiCodec code with random weights) |
| `maskgct` | TTS | Amphion MaskGCT, voice cloning in English, Chinese, Japanese, Korean, French and German | `maskgct` + GPL front-end packages + the authors' code (bundle `maskgct`) | CC-BY-NC-4.0 (non-commercial) | no | **no** (fakes; the authors' pipeline with tiny random networks) |
| `sepformer-whamr16k` | separation | SpeechBrain SepFormer, WHAMR!, 16 kHz, 2 speakers | `sepformer` | Apache-2.0 (WSJ0 data, see below) | no | CPU, see below |
| `sepformer-whamr` | separation | SpeechBrain SepFormer, WHAMR!, 8 kHz, 2 speakers | `sepformer` | Apache-2.0 (WSJ0 data, see below) | no | CPU, see below |
| `pixit-ami` | separation | pyannote PixIT `speech-separation-ami-1.0`, diarization + separation | `pyannote` | MIT | yes (two repositories) | CPU, see below |
| `pyannote-speaker-diarization-3.1` | diarization | pyannote `speaker-diarization-3.1`, the original PyTorch pipeline | `pyannote` | MIT (pipeline, segmentation) + CC-BY-4.0 (WeSpeaker embedding, PLDA files) | yes (three repositories) | CPU, see below |
| `pyannote-speaker-diarization-community-1` | diarization | pyannote `speaker-diarization-community-1` (VBx clustering) | `pyannote` | CC-BY-4.0 | yes | CPU, see below |
| `diarizen-wavlm-large-s80-md-v2` | diarization | BUT-FIT DiariZen v2: pruned WavLM Large + Conformer, VBx clustering (the authors' pipeline) | `diarizen` + the authors' code (see below) | CC-BY-NC-4.0 (non-commercial; PLDA CC-BY-4.0) | no | CPU, see below |
| `dicow-v3.3` | STT, diarization-conditioned | BUT-FIT DiCoW v3.3: Whisper large-v3-turbo told who talks when | `dicow` | CC-BY-4.0 | no | CPU, see below |
| `se-dicow` | STT, diarization-conditioned | BUT-FIT SE-DiCoW: DiCoW with self-enrolment | `dicow` | Apache-2.0 | no | CPU, see below |
| `sov-transcribe-1` | STT | Salesforce SOV-Transcribe-1: Whisper large-v3 encoder + Gemma 4 26B A4B, English, French, German, Spanish, Italian, Portuguese, Dutch | `sov-transcribe` + remote code opt-in | Apache-2.0 | no | **no** (fakes only; 53 GB) |
| `jais-asr` | STT | IFM Jais-ASR: Arabic (MSA and 19 dialects) with English, French, Spanish, Italian; transformers pipeline from the model repository | `jais-asr` + remote code opt-in | Apache-2.0 | no | **no** (fakes only; 11.4 GB) |
| `reverb-asr` | STT | Rev Reverb ASR: English WeNet Conformer from its TorchScript export, greedy CTC, verbatimicity | `reverb` | Rev Model Non-Production License (non-commercial) | yes | CPU (RTF 0.05) |
| `reverb-diarization-v2-torch` | diarization | Rev Reverb diarization v2: Rev's pyannote 3.1 pipeline with a WavLM-based segmentation model | `pyannote` + torchaudio | Rev Model Non-Production License (non-commercial; embedding and PLDA CC-BY-4.0) | yes | CPU, very slow (RTF 16); see the entry's note |
| `qwen2.5-omni-3b-torch` | speech-to-speech | Qwen2.5-Omni 3B, text and speech out (voices Chelsie, Ethan) | `omni` | Qwen Research License (non-commercial) | no | **no** (fakes only) |
| `qwen2.5-omni-7b-torch` | speech-to-speech | Qwen2.5-Omni 7B | `omni` | Apache-2.0 | no | **no** (fakes only) |
| `qwen3-omni-30b-a3b-instruct-torch` | speech-to-speech | Qwen3-Omni 30B-A3B Instruct, speech out in 10 languages (voices Ethan, Chelsie, Aiden) | `omni` | Apache-2.0 | no | **no** (fakes only) |
| `glm-4-voice-9b` | speech-to-speech | Zhipu GLM-4-Voice 9B, Chinese and English, text and speech interleaved, streamed | `glm-4-voice` + remote code opt-in | glm-4-voice License: **commercial use only after registering with Zhipu** | no | **no** (fakes only) |
| `kimi-audio-7b-instruct` | speech-to-speech | Moonshot Kimi-Audio 7B Instruct, English and Chinese, text and speech streamed | `kimi-audio` + flash-attn + remote code opt-in | MIT | no | **no** (fakes only) |
| `minicpm-o-2.6` | speech-to-speech | OpenBMB MiniCPM-o 2.6, English and Chinese, text and speech streamed (voices female, male, default_female) | `minicpm-o` + remote code opt-in (the whole model) | Apache-2.0 | no | **no** (fakes only) |
| `step-audio-2-mini` | speech-to-speech | StepFun Step-Audio 2 mini, text, speech (voices female, male) and tool calls, streamed | `step-audio-2` + remote code opt-in | Apache-2.0 | no | **no** (fakes only) |

The separation models are bolo `separation` models in "blind" mode: bolo uses them for `separate_overlaps=true`
on diarized transcription and on `POST /api/v1/diarize`, and assigns their tracks to its own diarized speakers.
They can also be run without bolo: `bolo-torch separate mix.wav -m sepformer-whamr16k -o out/`.

The pyannote diarizers are bolo `diarization` models: `POST /api/v1/diarize` with `model=pyannote-speaker-diarization-3.1`
(or `-community-1`), or `diarization_model=...` with `diarize=true` / `response_format=diarized_json` on
transcriptions. bolo's own diarization runs pyannote's segmentation model exported to ONNX with its own embeddings and
clustering; these run pyannote.audio's pipelines unchanged, for pyannote's exact results. See "pyannote speaker
diarization" below.

Spark-TTS 0.5B (`spark-tts-0.5b`, kind `spark-tts`) follows the authors' `cli/SparkTTS.py`: its language model
(Qwen2.5-0.5B, `LLM/`) through transformers' `AutoModelForCausalLM`, sampling as theirs (temperature 0.8, top-k 50,
top-p 0.95), and the BiCodec tokenizer and decoder from github.com/SparkAudio/Spark-TTS (Apache-2.0) at a pinned
commit (bundle `spark-tts`, see "Inference code from GitHub"). Two kinds of voice: *created* voices (`female`,
`male`: a gender and a pitch level, with the request's `speed` as one of the model's five speed levels), for which
the model writes the voice tokens itself, and *cloned* voices (`heart`, `michael`: bolo's reference clips, or
`voices_dir`), whose voice tokens come from the reference recording with its transcript. 16 kHz, one sentence at a
time. Weights CC-BY-NC-SA-4.0: not for commercial use. Not run here (3.9 GB).

MaskGCT (`maskgct`, kind `maskgct`) runs the authors' `MaskGCT_Inference_Pipeline` (github.com/open-mmlab/Amphion,
MIT, bundle `maskgct`) as `maskgct_inference.py` does: w2v-bert-2.0 and the semantic codec turn the reference
recording into semantic tokens, the text-to-semantic model writes the new text's (25 steps, guidance 2.5; the length
follows the reference's speaking rate, divided by `speed`), the two semantic-to-acoustic models and the acoustic
codec make 24 kHz audio. Every voice is a reference clip with its transcript (`heart`, `michael`, or `voices_dir`).
Languages: Chinese and English (mixed in one sentence), Japanese and Korean (by script), French and German (by
bolo's text-language guess, else the voice's `language`). What the pipeline's constructor would download (w2v-bert-2.0's latest
revision) comes from a pinned revision instead; the statistics file loads with `weights_only`. The phoneme front end
is Amphion's: `phonemizer` with espeak-ng, `pykakasi` and `Unidecode` are GPL-licensed, so they are not in the
`maskgct` extra; install them yourself if their licences suit you (`pip install phonemizer espeakng-loader pykakasi
Unidecode`; the loader points phonemizer at espeakng-loader's espeak-ng). LangSegment's only release on PyPI (0.2.0)
cannot be imported as published (its `__init__` names functions it lacks); the loader loads its working module
directly. pyopenjtalk downloads its dictionary (open_jtalk 1.11, 23 MB, from GitHub) on the first Japanese text. The
code's top-level packages are `models` and `utils`, and its front end reads files relative to the working directory
while it is imported (done under a lock, with the bundle as the working directory): give MaskGCT a plugin
environment of its own. transformers must be older than 4.48 (Amphion's Llama layers call the attention the old
way). Weights CC-BY-NC-4.0: not for commercial use. Not run here (6.6 GB plus 2.3 GB).

DiariZen (`diarizen-wavlm-large-s80-md-v2`, kind `diarizen`) is a bolo `diarization` model too, run by the authors'
own `DiariZenPipeline` (github.com/BUTSpeechFIT/DiariZen, MIT) on their fork of pyannote.audio 3.1.1 (in the same
repository). Neither is on PyPI, so both are installed from the pinned commit, in an environment of their own
because the fork replaces pyannote.audio (it cannot share one with the `pyannote` extra):

```bash
python -m venv /opt/bolo-torch-diarizen
/opt/bolo-torch-diarizen/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
/opt/bolo-torch-diarizen/bin/pip install ./server "./plugins/bolo-torch[diarizen]" \
  "pyannote.audio @ git+https://github.com/BUTSpeechFIT/DiariZen@844f5555b0a98acd0931511fc641a8c5b8ba92c7#subdirectory=pyannote-audio" \
  "diarizen @ git+https://github.com/BUTSpeechFIT/DiariZen@844f5555b0a98acd0931511fc641a8c5b8ba92c7"
BOLO_PLUGINS="torch=/opt/bolo-torch-diarizen/bin/python -m bolo_torch" bolo serve
```

(pip needs git for these two; on Windows install Git for Windows first.) The loader refuses any other commit of
either package. It fetches the WeSpeaker embedding (`pyannote/wespeaker-voxceleb-resnet34-LM`, CC-BY-4.0, not gated)
at a pinned revision; the PLDA comes with the model. VBx finds the number of speakers: `num_speakers` and the
`min_speakers` / `max_speakers` options are not used, as in the authors' pipeline. Its weights are CC-BY-NC-4.0:
not for commercial use. Slow on a CPU (more than ten times slower than real time on the test machine); a GPU is
recommended. See "DiariZen" below.

The DiCoW models are bolo `stt` models that are *speaker-conditioned*: in a diarized transcription
(`diarize=true` or `response_format=diarized_json`) bolo diarizes the recording (with
`nemotron-3-diarization` unless the request names another diarizer), then sends the whole recording to the
plugin with the speakers' frame activity, and the model transcribes it once per speaker. Overlapping speech is
transcribed for each speaker instead of being cut at turn boundaries. Without diarization they work as an
ordinary (slow) Whisper. See "DiCoW" below and docs/models.md, "Diarization-conditioned recognition".
DiCoW v3.3 also runs without PyTorch, in bolo's own process, as the catalog entry `dicow-v3.3-onnx`
(docs/models.md, "DiCoW without PyTorch").

The Qwen-Omni models are bolo `speech-to-speech` models, behind the Realtime API's `speech-to-speech` pipeline
(`s2s_model = "qwen2.5-omni-3b-torch"`), over the protocol's `respond` op (bolo 0.6.7 or later): bolo sends the whole
conversation each turn (the user's turns as 16 kHz audio, the model's earlier answers as text) and the plugin runs
the processor's chat template and `generate` with the Talker on (transformers' own `Qwen2_5OmniForConditionalGeneration`
and `Qwen3OmniMoeForConditionalGeneration`, no remote code). Qwen2.5-Omni gets the system prompt it needs for
speech ("You are Qwen, a virtual human ...") with the session's instructions after it. The answer, text and 24 kHz
speech, comes back in one piece when generation ends; tools are not passed. None of them was run here: the
downloads are 12 to 70 GB.

Step-Audio 2 mini (`step-audio-2-mini`, kind `step-audio-2`) streams: its text as the model writes it, tool calls as
they close, and speech from the authors' token2wav decoder in pieces of 25 speech tokens (the first after about 1 s
of speech). It needs two things beyond the extra: the model's own code from its Hugging Face repository,
which runs only after you opt in for this model (`bolo models config step-audio-2-mini --set allow_remote_code=true`;
see "Remote code" below), and token2wav from github.com/stepfun-ai/Step-Audio2 (Apache-2.0) at a pinned commit, which
the plugin installs (see "Inference code from GitHub" below). token2wav calls `.cuda()`: without an NVIDIA GPU the
model answers in text only, with a warning. The model's `forward` keeps no key-value cache, so every token
reprocesses the conversation. Not run here.

GLM-4-Voice 9B (`glm-4-voice-9b`, kind `glm-4-voice`) follows the authors' web demo: the user's speech as tokens of
the Whisper-based `glm-4-voice-tokenizer`, the demo's prompt and system instruction, text and speech tokens
interleaved in one stream, and speech through `glm-4-voice-decoder` (`AudioDecoder.token2wav`, flow matching and
HiFi-GAN, 22.05 kHz) in blocks of 25, 50, 100, 150 and 200 tokens, each conditioned on the earlier ones. The first
audio comes after 25 speech tokens (about 2 s of speech). The model's own code (ChatGLM) is remote code (opt in per
model); the tokenizer and decoder code comes from github.com/THUDM/GLM-4-Voice and Matcha-TTS at pinned commits
(bundle `glm-4-voice`). `quantization: int4` (or `bolo models config glm-4-voice-9b --set quantization=int4`) loads the language model with
bitsandbytes NF4, as the authors' `model_server.py --dtype int4`. **Licence:** the weights are under the glm-4-voice License: free for academic
research, commercial use only after registering with Zhipu (https://open.bigmodel.cn/mla/form, then free), "Built
with glm-4" shown in the product, Chinese law. bolo marks it non-commercial, and `bolo pull` and every load print
this. Not run here.

Kimi-Audio 7B Instruct (`kimi-audio-7b-instruct`, kind `kimi-audio`) is built by the authors' `kimia_infer`
(`KimiAudio`: the language model, the prompt manager with GLM-4-Voice speech tokens and Whisper large-v3 features,
the streaming detokenizer and BigVGAN vocoder), installed from github.com/MoonshotAI/Kimi-Audio at a pinned commit.
The plugin runs `KimiAudio._generate_loop` one step at a time, so text streams as the model writes it and speech goes out
in chunks of 30 speech tokens through `detokenize_streaming` (24 kHz; first audio after about 2.4 s of speech). Its
prompt manager loads the speech tokenizer by name from the Hub (any revision); the plugin gives it the copy bolo
pinned and downloaded with the model instead. NVIDIA GPU only (its code uses `torch.cuda` and imports flash-attn,
which is not in the extra: `pip install flash-attn --no-build-isolation` on Linux; on Windows a community wheel or
WSL2). Not run here.

MiniCPM-o 2.6 (`minicpm-o-2.6`, kind `minicpm-o`) is Python in its Hugging Face repository and nothing else, so it
loads only after the per-model opt-in (see "Remote code"). It follows the model card: the "audio assistant" system
prompt with a voice's reference recording (`female`, `male`: stable; `default_female`: more natural, less stable,
per the card), and by default the card's streaming API: a new session per answer, the conversation prefilled (user
speech in 1 s pieces), `streaming_generate` giving text and 24 kHz speech piece by piece. Earlier answers are
prefilled as text with the chat markup that the streaming API does not add for the assistant role, an adaptation of
the card's single-session example. `mode: chat` uses `chat()` instead (the whole answer when generation ends); a
text-only answer always does. bolo downloads the voices, Vocos and the ChatTTS tokenizer with the model, so its
`init_tts` fetches nothing at load. Not run here.

The catalog also lists `sravaani-1.0` and `indicconformer-600m-official` for this plugin; bolo-torch does not
implement them yet. Moshi, Hibiki and PersonaPlex (full-duplex) are not plugin models: bolo runs them in their own
servers (bolo's docs/voice-agents.md, "Full-duplex models").

### How each family runs

* **Parler-TTS** (`parler-tts` 0.2.3): the voice is a description. A voice id from the catalog (`divya`, `rohit`,
  `thoma`, ...) becomes "*Name*'s voice is clear and natural, at a moderate pace, in a very close recording that
  almost has no background noise. Very clear audio."; a catalog voice can carry its own `description`. Indic
  Parler-TTS reads the description with its text encoder's tokenizer (`google/flan-t5-large`, downloaded once,
  tokenizer files only). `speed` above 1.2 or below 0.8 adds "The speaker talks fast/slowly." Audio streams one
  sentence at a time (bolo also splits text into sentences before it reaches the plugin). The option
  `stream_seconds` turns on Parler's own streamer inside a sentence; on CPU it did not start audio sooner and
  almost doubled the real-time factor (measured below), so it is off by default.
* **F5-TTS v1 Base** (`layout: f5-tts`): `f5_tts.infer.utils_infer.load_model` builds the v1 DiT from
  `F5TTS_v1_Base/model_1250000.safetensors` and its `vocab.txt`, the Vocos vocoder comes from the pinned
  `charactr/vocos-mel-24khz` files, and generation is the package's `infer_batch_process` (as for IndicF5).
  Catalog voices with `reference: <bolo sample id>` use bolo's built-in synthetic clips and their transcripts.
* **Indic-Speak**: the gated repository ships its own `inference.py`; the loader imports it from the model
  directory and calls `TTS(<dir>)(text, speaker=…, style=…)` as the model card documents (24 kHz). Voice ids are
  the speaker names (`amit`, `kavya`, …), optionally `<speaker>:<STYLE>`; unknown names are refused because the
  model would silently average them.
* **IndicF5**: the loader builds the DiT from the `f5_tts` copy that ships in the model repository and loads
  `model.safetensors` from the local directory. The repository's `model.py` is not used: it looks its vocabulary
  up as a Hub repository id (which fails for a local directory), wraps the models in `torch.compile` and reads
  audio with `torchaudio.load`, which needs `torchcodec` in torchaudio 2.9+. Only Vocos's `config.yaml`
  (`charactr/vocos-mel-24khz`, MIT) is downloaded; the vocoder weights are in the IndicF5 file. A voice is a
  reference clip plus its transcript: `pan_f_happy` (the Punjabi prompt in the repository, transcript from the
  IndicF5 README) and any `<name>.wav` + `<name>.txt` pair in `BOLO_TORCH_VOICES_DIR` (or the `voices_dir`
  option). Option `nfe_step` (default 32) sets the number of flow steps.
* **Veena and Svara** (Orpheus-style): the language model writes SNAC codec tokens, 7 per 85 ms frame. Prompts
  follow the Veena model card (`<spk_kavya> text`) and the Svara inference code (`Hindi (Male): text`, with the
  `<|audio|>` token). Audio is streamed while the model generates: every 4 frames (option `stream_frames`) the new
  frames are decoded with 3 frames of left context and one frame of look-ahead held back. The SNAC decoder
  (`hubertsiuzdak/snac_24khz`, MIT, pinned) was run here and gives 2048 samples per frame as the code expects;
  the 3B language models were not (7.6 GB and 13.2 GB, and they need a GPU in practice).
* **SepFormer**: SpeechBrain's `SepformerSeparation`. bolo sends 16 kHz audio; the 8 kHz model gets it resampled
  to 8 kHz and returns its tracks resampled to 16 kHz.
* **PixIT**: `pyannote.audio` `Pipeline.from_pretrained` on the pipeline configuration bolo downloaded. The
  pipeline then downloads `pyannote/separation-ami-1.0` (gated), `microsoft/wavlm-large` and
  `speechbrain/spkrec-ecapa-voxceleb` itself (2.6 GB together). One track per speaker the pipeline finds.
  pyannote.audio 4.0.7 calls SpeechBrain with arguments SpeechBrain 1.1.2 no longer takes as keywords (`token`,
  `revision`, `huggingface_cache_dir`) and a `torch.device` where it wants a string; the loader moves them into
  SpeechBrain's `FetchConfig` before loading, otherwise the pipeline does not load. The pinned revision of the
  ECAPA embedding was not honoured in that path (the current `main` was downloaded). Option `leakage_removal`
  overrides the pipeline's setting (default true), which zeroes a track wherever its speaker is judged silent.

* **pyannote speaker diarization** (`kind: pyannote-diarization`): `Pipeline.from_pretrained` on a checked copy
  of the pipeline configuration (see "Configuration guard" below), run on the 16 kHz recording bolo sends. bolo
  downloads everything at pinned revisions with sha256: for 3.1 the configuration, `pyannote/segmentation-3.0`
  (5.9 MB), `pyannote/wespeaker-voxceleb-resnet34-LM` (26.6 MB) and community-1's two PLDA files (0.3 MB:
  pyannote.audio 4 loads a PLDA for every speaker-diarization pipeline, agglomerative clustering included, and
  would otherwise fetch community-1's latest revision); for community-1 its configuration, segmentation,
  embedding and PLDA (32.8 MB). Nothing is downloaded at load time. Turns come back in bolo's format (speakers
  numbered in order of appearance); `num_speakers` of a request goes to the pipeline, the catalog options
  `num_speakers` / `min_speakers` / `max_speakers` are passed on every call, and a request's `min_speakers` /
  `max_speakers` make bolo ask again with that number when the pipeline's count is outside them. `threshold` is
  ignored (pyannote's tuned clustering parameters apply) and bolo's speaker clean-up is off for these models.
  Voice prints for known speakers and overlap separation come from the pipeline's own embedding model
  (`pipeline._embedding`, unit length). Option `exclusive: true` returns community-1's exclusive diarization (no
  overlapping turns). pyannote.audio 4 sends usage metrics to pyannote's servers by default; the plugin sets
  `PYANNOTE_METRICS_ENABLED=false` unless you set it yourself (PixIT too).
* **Configuration guard** (pyannote diarization and PixIT). pyannote builds a pipeline from what `config.yaml`
  names: it imports the class in `pipeline.name`, any `preprocessors`, and downloads the sub-models it lists from
  their latest revision, and it unpickles model checkpoints. A changed or malicious repository could make it run any
  installed callable. The plugin never gives pyannote the file: it checks the sha256 of the configuration and the
  sub-model files against the catalog, parses the configuration with `yaml.safe_load`, accepts only
  `pyannote.audio.pipelines.SpeakerDiarization` / `SpeechSeparation`, the clustering names
  `AgglomerativeClustering` / `VBxClustering`, plain parameter values (no `name`, `_target_` or dotted paths, no
  `preprocessors`) and sub-models that are folders bolo downloaded or Hub repositories pinned to a commit, and
  hands pyannote the checked dictionary with every sub-model resolved. Anything else is refused (`400
  invalid_request`, saying what) before pyannote is imported. Without a model folder from bolo the plugin downloads
  a pipeline only at a commit given in the option `revision`. PixIT's separation model
  (`pyannote/separation-ami-1.0`, 1.3 GB) is now part of the catalog download, pinned and hashed; the WavLM it
  builds on is still fetched by the model itself (not pinned by bolo).
* **DiCoW** (BUT-FIT, `BUT-FIT/DiCoW_v3_3`; Whisper large-v3-turbo with frame-level diarization-dependent
  transformations, FDDT). For each diarized speaker the plugin builds the four-class STNO mask (silence / only the
  target / only others / target in overlap) every 20 ms from the activity bolo sends, with the formula of BUT's
  inference pipeline (`BUTSpeechFIT/DiCoW`, `pipeline.py`, Apache-2.0), and runs the model's own generation code
  (Whisper long-form decoding in 30 s windows, greedy, all speakers of a recording in one batch, up to the option
  `batch`, default 4). The repository's Python files are imported as a package (what `trust_remote_code` runs):
  transformers 4.55 cannot load them from a local directory because it copies only the modules
  `modeling_dicow.py` imports directly. Segment times come from the generator's own segments. A speaker with less
  than 0.2 s of speech (option `skip_silent_s`) is not decoded. On a CPU with AMX or AVX512-BF16 the weights are
  loaded in bfloat16 unless `dtype` is set; otherwise float32. Output is lower-case without punctuation and with
  contractions spelled out ("we are", "do not"), the normalised style of its training references. SE-DiCoW uses
  the same loader and also passes each speaker's self-enrolment window (the 30 s where that speaker talks alone the
  most, chosen as BUT's pipeline does).

## Install

bolo-torch is on PyPI and installs bolo (the `bolo-speech` distribution) with it:

```bash
# 1. PyTorch. The CPU-only build is much smaller than the default (CUDA) build on Linux:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# 2. the plugin, with the model families you want:
pip install "bolo-torch[parler,sepformer]"
```

To add model families to an environment that already has a CUDA (or ROCm, XPU) build of torch, use
`bolo-torch install`, not `pip install "bolo-torch[...]"`:

```bash
bolo-torch install pyannote             # --dry-run prints the pip command
```

A plain pip install lets pip take the newest torchaudio, which needs a newer torch than the one installed, and pip
then replaces your torch with PyPI's, which on Windows is the CPU build: the models run on the CPU. `bolo-torch
install` pins the installed torch, holds torchaudio and torchvision to its release (a constraints file), and adds
the PyTorch index the build came from (`+cu128` -> download.pytorch.org/whl/cu128).

**pyannote and older torch.** The `pyannote` extra asks for pyannote.audio 3.3.2 or later (torch 2.0 or later), but
left free pip takes pyannote.audio 4, which needs torch 2.8 or later, and with an older CUDA torch (2.6.0+cu124, say)
it replaces that torch with PyPI's newest, CPU only on Windows. The floor cannot go lower than it is: pyannote
`speaker-diarization-community-1` needs pyannote.audio 4. So either upgrade torch first from the CUDA index (2.8 or
later), or run `bolo-torch install pyannote`, which keeps your torch: with torch 2.6 pip then resolves pyannote.audio
3.4, which runs `pyannote-speaker-diarization-3.1` and `pixit-ami` but not community-1.

From a checkout of the bolo repository:

```bash
# 1. PyTorch. The CPU-only build is much smaller than the default (CUDA) build on Linux:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# 2. bolo and the plugin, with the model families you want:
pip install ./server "./plugins/bolo-torch[parler,sepformer]"
```


Extras: `parler` (Parler-TTS), `f5` (IndicF5), `f5-original` (the original F5-TTS checkpoints, on the `f5-tts`
package, which pins its own libraries: give it its own environment), `indic-speak` (transformers 5 and SNAC),
`orpheus` (Veena, Svara), `sepformer`, `pyannote` (PixIT, pyannote speaker diarization),
`dicow` (DiCoW, SE-DiCoW: transformers 4.55 and pandas), `sov-transcribe` (SOV-Transcribe-1: transformers 5.19 to 5.22, accelerate), `hf-asr` (speech recognition through a transformers pipeline: transformers, accelerate), `jais-asr` (Jais-ASR: transformers 5.16.1, accelerate, soundfile, soxr), `reverb` (Reverb ASR: torchaudio, PyYAML), `diarizen` (DiariZen: toml, accelerate, scipy, torchaudio
below 2.9; with the authors' two packages from GitHub, see above), `spark-tts` (Spark-TTS: transformers, einops, einx,
omegaconf, soxr, torchaudio; the BiCodec code is the bundle `spark-tts`), `maskgct` (MaskGCT: transformers before 4.48,
librosa, onnxruntime, jieba, cn2an, pypinyin, LangSegment, pyopenjtalk-plus and the rest of Amphion's front end
except its GPL-licensed packages, see above), `omni` (Qwen2.5-Omni and Qwen3-Omni: transformers 4.57
or later, accelerate, torchvision), `step-audio-2` (Step-Audio 2 mini: transformers 4.49.0, which the authors use, and
token2wav's libraries: s3tokenizer, onnxruntime, hyperpyyaml, diffusers, torchaudio, librosa), `glm-4-voice` (GLM-4-Voice: transformers 4.44.1, diffusers 0.27.2 with huggingface-hub below 0.26, conformer,
hyperpyyaml, hydra-core, lightning, tiktoken, bitsandbytes, as in the authors' requirements), `kimi-audio`
(Kimi-Audio: transformers 4.48 to 4.51, tiktoken, torchdyn, timm, loguru; flash-attn separately), `minicpm-o`
(MiniCPM-o 2.6: transformers 4.44.2, vector-quantize-pytorch 1.18.5, vocos 0.1.0, as on the model card). DiCoW uses the plugin protocol's `speakers` request (bolo 0.2.0 or later), the
Qwen-Omni models its `respond` request (bolo 0.6.7 or later).
Installed in bolo's environment, the plugin is found through its entry point (`bolo.engines: torch`) and
`bolo doctor` lists `plugin:torch`. Check the plugin's own environment with `bolo-torch doctor`.

**Some extras do not mix.** `parler-tts` 0.2.3 pins `transformers==4.46.1`, while `pyannote.audio[separation]`
needs `transformers>=4.48.3` and DiCoW's model code needs transformers 4.55. Put PixIT and the pyannote diarizers in a second environment and point bolo at it (a configured command wins
over the entry point):

```bash
python -m venv /opt/bolo-torch-pyannote
/opt/bolo-torch-pyannote/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
/opt/bolo-torch-pyannote/bin/pip install ./server "./plugins/bolo-torch[pyannote,sepformer]"
BOLO_PLUGINS="torch=/opt/bolo-torch-pyannote/bin/python -m bolo_torch" bolo serve
```

That is how PixIT was run here. One plugin name serves all `plugin:torch` entries, so the environment the
plugin runs in must hold every family you want to use at the same time.

The `parler` extra also pins `sentencepiece<0.2.1`: Parler's audio library pins `protobuf<5`, and sentencepiece
0.2.3 cannot be imported with protobuf 4 (`cannot import name 'runtime_version'`), which broke loading Indic
Parler-TTS here until it was pinned.

## Remote code

A model whose code exists only in its Hugging Face repository (transformers' `trust_remote_code=True`) loads only
when all of this holds: the catalog entry pins the reviewed commit and the sha256 of every `.py` file and of every
JSON file with an `auto_map` (`options.remote_code`); the downloaded folder is exactly that revision, every pinned
file matches, no unpinned `.py` file is present and no `auto_map` points at another repository; and the operator set
`allow_remote_code = true` for this model (`bolo models config <id> --set allow_remote_code=true`). bolo sends the
plugin that setting and overrides whatever a catalog entry says. The check runs right before the import, and each
load logs a warning that names the files. Otherwise the load fails with `remote_code_refused` and says what to do.
The code runs inside this plugin's process with its rights. `bolo-torch pin-remote-code <folder> --revision <commit>`
prints the pin for a new entry; read the files it lists first. bolo's docs/models.md, "Remote code", has the details.

## Security notes

Three extras pin an older transformers because their models' remote code needs it: `glm-4-voice` (4.44.1),
`minicpm-o` (4.44.2) and `step-audio-2` (4.49.0). These releases have published security advisories (16 to 18
each), fixed in later 4.x releases and in 5.x. Other extras also keep transformers below 5 (`kimi-audio`,
`dicow`, `maskgct`, and `parler` through parler-tts). They are opt-in: nothing installs them unless you name the
extra, and the models' remote code runs only after `allow_remote_code = true`. Install each in its own plugin
environment (a separate venv that bolo starts through `BOLO_PLUGINS`, as under "Some extras do not mix"), not in
the one bolo or your other models use.

## Inference code from GitHub

Code that is not a package (Step-Audio 2's token2wav, ...) comes from the authors' GitHub repository at a pinned
commit, each file pinned by sha256 (`src/bolo_torch/upstream/<name>.json`, written by `tools/pin_upstream.py` from a
checkout at that commit):

```bash
bolo-torch code list                     # bundles, their repositories, commits and licences, installed or not
bolo-torch code install step-audio-2     # download now (a load installs a missing bundle too)
bolo-torch code verify                   # every installed file against its sha256
```

Files go to `BOLO_TORCH_CODE_DIR` (default `~/.cache/bolo-torch/code`, `%LOCALAPPDATA%\bolo-torch\code` on Windows),
one folder per pin. `BOLO_TORCH_CODE_AUTO_INSTALL=0` makes a load fail instead of downloading. A changed file is
refused at load.

| Bundle | Repository @ commit | Licence |
| --- | --- | --- |
| `step-audio-2` | stepfun-ai/Step-Audio2 @ `76e272b56c39` (`utils.py`, `token2wav.py`, `cosyvoice2/`, part of `flashcosyvoice/`, the two prompt voices) | Apache-2.0 |
| `glm-4-voice` | THUDM/GLM-4-Voice @ `eb00ce9142e8` (`speech_tokenizer/`, `flow_inference.py`, `cosyvoice/`) and shivammehta25/Matcha-TTS @ `dd9105b34bf2` (`matcha/`, its submodule) | Apache-2.0; MIT |
| `spark-tts` | SparkAudio/Spark-TTS @ `2f1ea9082400` (`sparktts/`: BiCodec, its modules and token maps) | Apache-2.0 |
| `maskgct` | open-mmlab/Amphion @ `26f688311018` (MaskGCT's models and pipeline, the two codecs, `utils/`, the g2p front end with its lexicons and Chinese polyphone ONNX model, 46 files) | MIT |
| `kimi-audio` | MoonshotAI/Kimi-Audio @ `349251e1d8f4` (`kimia_infer/`) and THUDM/GLM-4-Voice @ `eb00ce9142e8` (`speech_tokenizer/` in `kimia_infer/models/tokenizer/glm4`, its submodule) | MIT (stated in README and pyproject; no LICENSE file; Qwen-derived code Apache-2.0); Apache-2.0 |

## Speech-to-speech models on a Windows laptop with an NVIDIA GPU

None of the four models below was run here (no GPU, about 5 GB of disk). These are the steps to run each for real,
from a checkout of the bolo repository (the remote-code opt-in and these loaders are not in a released bolo-torch
yet).

**What fits.** Approximate GPU memory with a short conversation; laptop GPUs have 8 GB (RTX 4060/4070 Laptop), 12 GB
(RTX 4080/5070 Ti Laptop), 16 GB (RTX 4090/5080 Laptop) or 24 GB (RTX 5090 Laptop).

| Model | Needs | 8 GB | 12 GB | 16 GB | 24 GB |
| --- | --- | --- | --- | --- | --- |
| `glm-4-voice-9b` | about 22 GB in bfloat16; about 9 GB with `quantization=int4` | no | yes, int4 | yes, int4 | yes, bfloat16 |
| `step-audio-2-mini` | about 20 GB (17 GB of weights and token2wav) | no | no | only split (part on the CPU, slow) | yes |
| `minicpm-o-2.6` | about 18 GB | no | no | only split (part on the CPU, slow) | yes |
| `kimi-audio-7b-instruct` | 24 GB or more, no split, 48 GB of RAM to load | no | no | no | at the limit |

A split happens when bolo's scheduler finds too little free GPU memory (`bolo ps` shows it); it was tested with fakes
only. Close other GPU programs first and check free memory with `nvidia-smi`.

**1. bolo, from a checkout** (PowerShell; Python 3.12 from python.org, git):

```powershell
git clone https://github.com/cB-Sigma/bolo-speech C:\bolo-src
py -3.12 -m venv C:\bolo-env
C:\bolo-env\Scripts\pip install "C:\bolo-src\server[all,cuda]"
```

**2. The plugin, in an environment of its own per model** (their transformers versions differ). For GLM-4-Voice:

```powershell
py -3.12 -m venv C:\bolo-s2s\glm
C:\bolo-s2s\glm\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
C:\bolo-s2s\glm\Scripts\pip install C:\bolo-src\server "C:\bolo-src\plugins\bolo-torch[glm-4-voice]"
C:\bolo-s2s\glm\Scripts\bolo-torch doctor                    # cuda_available: true, "glm-4-voice": true
C:\bolo-s2s\glm\Scripts\bolo-torch code install glm-4-voice  # GLM-4-Voice + Matcha-TTS code, sha256-checked
```

For the others the extra and the bundle change: `step-audio-2` (`bolo-torch code install step-audio-2`),
`minicpm-o` (no bundle: the model is all remote code). Kimi-Audio also needs flash-attn, which has no official
Windows build: run steps 1 to 5 under WSL2 (Ubuntu, `python3.12 -m venv`, the same commands with Linux paths) and add
`pip install flash-attn --no-build-isolation` (it compiles against the CUDA toolkit 12.x; `pip install ninja` first)
or install a prebuilt flash-attn wheel that matches your Python, torch and CUDA versions; then
`bolo-torch code install kimi-audio`.

**3. Point bolo at that environment**, on the GPU:

```powershell
$env:BOLO_PLUGINS = "torch=C:\bolo-s2s\glm\Scripts\python.exe -m bolo_torch"
$env:BOLO_DEVICE = "cuda"
$env:BOLO_TORCH_DEVICE = "cuda"
C:\bolo-env\Scripts\bolo doctor                # lists plugin:torch as available
```

**4. Pull, read the remote code, opt in.** No Hugging Face token is needed for these four.

```powershell
C:\bolo-env\Scripts\bolo pull glm-4-voice-9b   # 21 GB; warns that the model runs remote code; prints the licence note
C:\bolo-env\Scripts\bolo show glm-4-voice-9b   # "Remote code": the files, the commit, what the review found
C:\bolo-env\Scripts\bolo models config glm-4-voice-9b --set allow_remote_code=true
C:\bolo-env\Scripts\bolo models config glm-4-voice-9b --set quantization=int4   # on 12 or 16 GB
```

Downloads: `step-audio-2-mini` 17.9 GB, `minicpm-o-2.6` 17.4 GB, `kimi-audio-7b-instruct` 44.1 GB. GLM-4-Voice's
licence allows commercial use only after registering with Zhipu (https://open.bigmodel.cn/mla/form).

**5. Serve and talk to it** through the Realtime API's speech-to-speech pipeline:

```powershell
$env:BOLO_REALTIME_PIPELINE = "speech-to-speech"
$env:BOLO_S2S_MODEL = "glm-4-voice-9b"
C:\bolo-env\Scripts\bolo serve
# in a second window
C:\bolo-env\Scripts\pip install openai soundfile numpy
C:\bolo-env\Scripts\python C:\bolo-src\examples\realtime_agent.py question.wav --out reply.wav
```

`question.wav` is a short spoken question in English (or Chinese); without a file the script asks bolo's
text-to-speech for one, which needs a voice installed. The first request loads the model (minutes for 20 GB).

**What to check.**

* bolo's log: `remote code: <model> imports Python files from its own model repository (...)` once per load (and
  for GLM-4-Voice the licence note); no `remote_code_refused` (if it appears, the message says what to do: the opt-in
  is missing, or the folder is not the pinned revision, or a file changed).
* `reply.wav` holds a spoken answer that fits the question, and the script prints the answer's text and the
  latency figures of the turn. Expect the first audio after about 1 s of the model's speech for Step-Audio 2,
  2 s for GLM-4-Voice, 2.4 s for Kimi-Audio, plus the time those tokens take on your GPU; MiniCPM-o streams in
  pieces; Step-Audio 2 slows down as the conversation grows (its model keeps no key-value cache).
* `C:\bolo-env\Scripts\bolo ps`: the model on `cuda`, its GPU memory; `nvidia-smi` agrees.
* Talk over the answer (barge-in, with a microphone client such as `examples/webrtc_browser.html`): the answer
  stops within a token or a speech chunk.
* Voices: `--voice male` with Step-Audio 2 and MiniCPM-o (`female`, `male`; MiniCPM-o also `default_female`).
* Step-Audio 2 tool calls: give the session a function tool; check the `<tool_call>` format the model writes (the
  plugin expects `function`, the name and JSON arguments on separate lines, or a JSON object).
* Errors to expect and their fixes: `component_missing` (install the extra in the plugin's environment, or
  flash-attn), `device 'cuda' was requested but this torch build sees no CUDA device` (install the CUDA torch wheel
  in the plugin's environment), out of memory (`quantization=int4` for GLM-4-Voice, or a smaller model).

## Spark-TTS, MaskGCT and DiariZen on a Windows laptop with an NVIDIA GPU

DiariZen ran here on a CPU only; Spark-TTS (3.9 GB) and MaskGCT (8.9 GB) were not run. Steps to run all three on the
GPU, from a checkout. Spark-TTS needs about 4 GB of GPU memory, MaskGCT about 12 GB (float32, as its authors run it;
an 8 GB laptop GPU will not hold it), DiariZen about 2 GB.

**1. bolo**, as above (`git clone https://github.com/cB-Sigma/bolo-speech C:\bolo-src`, `C:\bolo-env`).

**2. One plugin environment per model** (DiariZen's pyannote.audio fork must not meet other pyannote versions):

```powershell
py -3.12 -m venv C:\bolo-tts\spark
C:\bolo-tts\spark\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
C:\bolo-tts\spark\Scripts\pip install C:\bolo-src\server "C:\bolo-src\plugins\bolo-torch[spark-tts]"
C:\bolo-tts\spark\Scripts\bolo-torch code install spark-tts      # sparktts at the pinned commit, sha256-checked

py -3.12 -m venv C:\bolo-tts\maskgct
C:\bolo-tts\maskgct\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
C:\bolo-tts\maskgct\Scripts\pip install C:\bolo-src\server "C:\bolo-src\plugins\bolo-torch[maskgct]"
# the GPL-licensed parts of its phoneme front end (read their licences first):
C:\bolo-tts\maskgct\Scripts\pip install phonemizer espeakng-loader pykakasi Unidecode
C:\bolo-tts\maskgct\Scripts\bolo-torch code install maskgct        # Amphion's files at the pinned commit (80 MB)

py -3.12 -m venv C:\bolo-tts\diarizen
C:\bolo-tts\diarizen\Scripts\pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
C:\bolo-tts\diarizen\Scripts\pip install C:\bolo-src\server "C:\bolo-src\plugins\bolo-torch[diarizen]" `
  "pyannote.audio @ git+https://github.com/BUTSpeechFIT/DiariZen@844f5555b0a98acd0931511fc641a8c5b8ba92c7#subdirectory=pyannote-audio" `
  "diarizen @ git+https://github.com/BUTSpeechFIT/DiariZen@844f5555b0a98acd0931511fc641a8c5b8ba92c7"
C:\bolo-tts\diarizen\Scripts\bolo-torch doctor                  # cuda_available: true
```

(The DiariZen line needs git on PATH: Git for Windows.)

**3. Serve with one of them**, on the GPU:

```powershell
$env:BOLO_PLUGINS = "torch=C:\bolo-tts\spark\Scripts\python.exe -m bolo_torch"   # or ...\maskgct\... / ...\diarizen\...
$env:BOLO_DEVICE = "cuda"; $env:BOLO_TORCH_DEVICE = "cuda"
C:\bolo-env\Scripts\bolo pull spark-tts-0.5b        # 3.9 GB; or: bolo pull maskgct (6.6 GB; the plugin adds
                                                      # w2v-bert-2.0, 2.3 GB, on the first load), bolo pull diarizen
C:\bolo-env\Scripts\bolo serve
```

**4. Try them** (second window):

```powershell
# Spark-TTS: a created voice, a faster one, and a cloned one
curl.exe -s http://127.0.0.1:8000/v1/audio/speech -H "Content-Type: application/json" `
  -d '{"model":"spark-tts-0.5b","voice":"female","input":"The quick brown fox jumps over the lazy dog."}' -o female.wav
curl.exe -s http://127.0.0.1:8000/v1/audio/speech -H "Content-Type: application/json" `
  -d '{"model":"spark-tts-0.5b","voice":"male","speed":1.4,"input":"This one should be noticeably faster."}' -o male.wav
curl.exe -s http://127.0.0.1:8000/v1/audio/speech -H "Content-Type: application/json" `
  -d '{"model":"spark-tts-0.5b","voice":"heart","input":"Cloned from bolo''s English reference clip."}' -o heart.wav
# MaskGCT: English, then Chinese and English mixed, then French, cloning bolo's reference clip
curl.exe -s http://127.0.0.1:8000/v1/audio/speech -H "Content-Type: application/json" `
  -d '{"model":"maskgct","voice":"heart","input":"The quick brown fox jumps over the lazy dog."}' -o mask-en.wav
curl.exe -s http://127.0.0.1:8000/v1/audio/speech -H "Content-Type: application/json" `
  -d '{"model":"maskgct","voice":"michael","input":"我们明天 meet 一下吧。"}' -o mask-zh.wav
curl.exe -s http://127.0.0.1:8000/v1/audio/speech -H "Content-Type: application/json" `
  -d '{"model":"maskgct","voice":"heart","input":"Bonjour, comment allez-vous ?"}' -o mask-fr.wav
# DiariZen: the demo dialogue (two voices taking turns)
curl.exe -s http://127.0.0.1:8000/api/v1/diarize -F model=diarizen -F file=@C:\bolo-src\server\src\bolo\_ui\diarization\two-speakers-en.mp3
```

**What to check.** Spark-TTS: each file speaks the text (`bolo transcribe -m whisper-base female.wav`), `female`
and `male` differ, `heart` sounds like `C:\bolo-src\server\src\bolo\samples\en-short.wav`, and a Chinese sentence
works; the log shows `spark-tts inference code: SparkAudio/Spark-TTS@2f1ea9082400`. Note the first-audio time and
RTF that bolo logs, and `bolo ps` (on `cuda`, about 4 GB). MaskGCT: each file speaks its text in the reference's
voice (the timbre of `en-short.wav` / `en-long.wav`), the Chinese-English sentence switches language mid-sentence,
`speed` 1.5 makes it shorter; the log shows `maskgct inference code: open-mmlab/Amphion@26f688311018` and no
`component_missing` for the front end (if phonemizer cannot find espeak-ng, set `PHONEMIZER_ESPEAK_LIBRARY` to
`libespeak-ng.dll`). DiariZen: speakers `A` and `B` taking seven turns, as on the
CPU here (DER 5.0 % against the clip's script); the time should be well under the 27 s of audio (on the CPU it took
337 s). Errors to expect: `component_missing` naming the commit (install the two DiariZen packages exactly as above),
`device 'cuda' was requested but this torch build sees no CUDA device` (the CUDA torch wheel is missing in that
environment).

## GPUs

* bolo passes its own `device` setting (default `cpu`) to the plugin. To use a GPU set `BOLO_TORCH_DEVICE=auto`
  (CUDA if available, then Apple MPS, else CPU) or `cuda`, `cuda:1`, `mps`. A model's catalog option `device`
  also works. Asking for `cuda` with a CPU-only torch answers `400` with that explanation.
* `dtype` is automatic: float32 on CPU, bfloat16 on CUDA when supported (else float16), float16 on MPS. Override
  with `BOLO_TORCH_DTYPE` or the option `dtype` (`float32`, `float16`, `bfloat16`).
* **The GPU build of torch in one step**: `bolo torch use cuda` (or the console's Components page, "Install GPU
  version", or `bolo components install torch-cuda`). It finds the GPU and the newest CUDA its driver runs
  (`nvidia-smi`), picks PyTorch's index for it (`cu128`, ...), and in every environment a bolo-torch plugin runs in
  (bolo's own and each in the `plugins` setting) reinstalls torch, and torchaudio and torchvision where present, at
  the installed version when that index has it, checks `torch.cuda.is_available()` and the GPU's name there, and
  restarts the plugin. `--dry-run` prints the pip commands; `bolo torch` shows each environment's build;
  `bolo torch use rocm` (Linux), `xpu` and `cpu` work the same way. About 2.5 GB to download.
* By hand, on Windows or Linux: install torch from PyTorch's CUDA index before the plugin:
  `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128` (or the `cu12x` folder for your
  driver, https://pytorch.org/get-started/locally/). On Linux PyPI's default torch is a CUDA build too; **on Windows
  (and macOS) PyPI's torch is CPU only**, so a plain `pip install bolo-torch` there gets a CPU-only torch. Add model
  families with `bolo-torch install <extras>`, which keeps the torch you installed (see Install). `bolo doctor` warns
  when a bolo-torch environment has a CPU-only torch on a machine with an NVIDIA (or ROCm) GPU, naming the
  environment and the command. `bolo-torch install` also checks after pip: if the GPU build was replaced anyway, it
  reinstalls it from its index and says so.
* A GPU this torch cannot reach (a CPU-only build with `device = "cuda"`, `rocm`, or bolo placing the model on its
  GPU while the model's option is `auto`) is not an error: the model loads on the CPU with a warning saying why and
  how to get a CUDA build, and bolo's placement (`bolo ps`, `/api/v1/ps`) says CPU with that notice.
* With bolo 0.5 or later, when its scheduler finds less free GPU memory than a model needs, the load carries a `placement`
  (`device_map="auto"` and a `max_memory`): the Orpheus SNAC, Qwen-Omni, Step-Audio 2, GLM-4-Voice and MiniCPM-o
  loaders let accelerate put as many layers on the GPU as fit and the rest on the CPU, and report the split back for
  `bolo ps`. Other loaders load whole on `device`. A model class that does not name the layers to keep whole
  (`_no_split_modules`; remote code often does not, and transformers 4 then refuses `device_map="auto"`) is built
  without weights first, its repeated layers (the elements of its largest `nn.ModuleList`) are kept whole, and an
  explicit device map is passed instead; a model with no repeated layers is refused before loading with the memory
  it needs on one device.
* **No GPU was used in testing.** The GPU, MPS and half-precision paths are untested; the placement protocol is
  tested on CPU torch and with fakes.
* **Windows**: CI runs the fast tests on `windows-latest` with CPU torch. With bolo 0.6 the worker runs in a Job
  Object, so it ends with bolo and `plugin_nice` / `plugin_memory_limit_mb` apply. CUDA on Windows is untested.

## Settings

| Environment variable | Meaning |
| --- | --- |
| `BOLO_TORCH_DEVICE` | `auto`, `cpu`, `cuda`, `cuda:N`, `mps`; wins over bolo's `device` |
| `BOLO_TORCH_DTYPE` | `auto`, `float32`, `float16`, `bfloat16` |
| `BOLO_TORCH_THREADS` | torch intra-op threads (default: bolo's `threads` setting) |
| `BOLO_TORCH_CACHE` | where the plugin's own downloads go (default: the Hugging Face cache, `HF_HOME`) |
| `HF_TOKEN` / `BOLO_HF_TOKEN` | Hugging Face token for gated repositories |
| `BOLO_TORCH_VOICES_DIR` | extra IndicF5 voices: `<name>.wav` + `<name>.txt` |

Catalog `options` the plugin reads: `device`, `dtype`, `seed`, `max_chars` (sentence length limit), and per family
`description_tokenizer` / `description_template` / `stream_seconds` / `max_new_tokens` (Parler), `nfe_step` /
`voices_dir` (IndicF5), `prompt_format` / `temperature` / `top_p` / `repetition_penalty` / `max_new_tokens` /
`stream_frames` / `context_frames` / `codec_repo` (Veena, Svara). Without a model directory from bolo, the option
`repo` (and `revision`) makes the plugin download the model itself.

## Gated models

Indic Parler-TTS, IndicF5, PixIT and the pyannote diarizers are gated on Hugging Face. To use them:

1. Sign in to Hugging Face and accept the conditions on each model page:
   [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts),
   [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5), for PixIT **both**
   [pyannote/speech-separation-ami-1.0](https://huggingface.co/pyannote/speech-separation-ami-1.0) and
   [pyannote/separation-ami-1.0](https://huggingface.co/pyannote/separation-ami-1.0); for
   `pyannote-speaker-diarization-3.1` **all three** of
   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1),
   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   (its PLDA files, which pyannote.audio 4 loads for 3.1 as well); for `pyannote-speaker-diarization-community-1`
   the last one. pyannote's forms ask for your company or university, website or use case, and pyannote may email
   you about its products.
2. Create a read token at https://huggingface.co/settings/tokens and set `BOLO_HF_TOKEN` (or `HF_TOKEN`) in the
   environment bolo runs in; bolo uses it for its download and the plugin inherits it. For example:
   `BOLO_HF_TOKEN=hf_... BOLO_PLUGINS="torch=/opt/bolo-torch-pyannote/bin/python -m bolo_torch" bolo serve`, then
   `bolo pull pyannote-speaker-diarization-3.1` and
   `curl -F file=@meeting.wav -F model=pyannote-speaker-diarization-3.1 localhost:8000/api/v1/diarize`.

Without access, bolo's own download answers `401 model_gated`; a download the plugin makes (PixIT's second
repository, a tokenizer) answers the same, with the repository to accept, instead of a generic 503.

## Measured here

One 4-vCPU Xeon (2.1 GHz) virtual machine, no GPU, torch 2.11.0+cpu, Python 3.12, 2 threads, 2026-11-13. The
machine was shared with other jobs (load average 2-8), so these are pessimistic and only roughly comparable.
TTS figures are through bolo's HTTP API (`/v1/audio/speech`, `response_format=pcm`, streamed); load time is
`POST /api/v1/models/{id}/load` and includes starting the plugin process when it was not running. RTF =
processing time / audio duration.

| Model | Load | Text | First audio | Total | Audio | RTF |
| --- | --- | --- | --- | --- | --- | --- |
| `parler-tts-tiny-v1` (voice jon) | 7.4 s | 42 chars, 2 sentences | 4.6 s | 13.6 s | 2.9 s | 4.8 |
| | | 101 chars, 2 sentences | 11.5 s | 24.9 s | 5.5 s | 4.5 |
| same, in process, `stream_seconds=1.0` | 6.2 s | 42 chars | 4.9 s | 24.2 s | 3.0 s | 8.0 |
| `indic-parler-tts` (voice divya, Hindi) | 16.1 s | 20 chars | 64.3 s | 64.3 s | 3.0 s | 21.1 |
| | | 47 chars, 2 sentences | 39.4 s | 95.8 s | 4.3 s | 22.5 |
| `indic-parler-tts` (voice thoma, English) | | 42 chars, 2 sentences | 20.8 s | 75.9 s | 3.6 s | 21.1 |
| `indic-f5` (voice pan_f_happy, Hindi) | 11.9 s | 20 chars | 235 s | 235 s | 1.5 s | 154 |
| | | 47 chars, 2 sentences | 211 s | 446 s | 3.7 s | 120 |

All outputs were transcribed back with faster-whisper-tiny (English) or faster-whisper-tiny-hindi and read as the
input text, with small errors ("ताज़ा जाएंगी" for "बाज़ार जाएँगे" once). On CPU these models are far slower than
real time; IndicF5 and Indic Parler-TTS are GPU models in practice.

Separation of a synthetic two-voice mix (bolo's samples `en-long.wav` + `hi-short.wav` overlapping from 6.0 s at
0 dB, 11.8 s; scored by SI-SDR over the overlap, permutation-invariant):

| Model | Load | Separate 11.8 s | RTF | Female voice | Male voice |
| --- | --- | --- | --- | --- | --- |
| `sepformer-whamr16k` | 2.6 s | 43.6 s | 3.7 | 0.5 → 5.5 dB | −0.5 → −0.3 dB |
| `sepformer-whamr` (8 kHz) | 2.3 s | 17.0 s | 1.4 | 0.5 → 5.8 dB | −0.5 → −1.8 dB |
| `pixit-ami`, pipeline defaults | 10.8 s | 67.2 s | 5.7 | 3 speakers found; the whole mixture on one track, the others silent | |
| `pixit-ami`, `leakage_removal` false, 2 speakers | | | | 0.5 → 1.9 dB | −0.5 → −1.0 dB |

SepFormer pulled the female voice out but did not improve the male one on this mix; PixIT, trained on real AMI
meetings, did little on these two synthetic TTS voices. Neither result says much about meetings or calls: the
test is one synthetic clip.

Through bolo (`POST /api/v1/diarize` with `separate_overlaps=true`), separating the one 7.7 s overlap window, the
whole request (diarization included) took 36.0 s with `sepformer-whamr16k`, 21.0 s with `sepformer-whamr` and
33.8 s with `pixit-ami` (load 10.7 s). In each run bolo kept only speaker A's track (speaker B, who talks only
during the overlap, had 0.5 s of clean speech, less than the 1 s bolo needs for a voice print). With
`sepformer-whamr16k` that track scored −10.7 dB against the male reference, worse than the −0.9 dB of the
plugin's own track for the same window (`pixit-ami`: −9.9 dB). That was bolo's track assignment as of the
afternoon of 2026-11-13; it was reworked afterwards (it now also weighs each track's share of the speakers' solo
speech) and checked with perfect tracks, so this single observation may no longer apply. The word-error
evaluation of both separators through the current bolo is in docs/models.md, "Overlapping speech".

### pyannote speaker diarization

2026-11-17, same machine type, torch 2.15.0+cpu, pyannote.audio 4.1.0. DER through bolo (`Bolo.diarize`, the
path of `POST /api/v1/diarize`) after bolo downloaded the catalog entries with a token; load time and RTF from the
loader run directly at 4 threads while the machine was quiet (load average below 2). The demo's two-voice dialogue (two SYSPIN voices taking turns, 27.2 s; reference turns from its script,
frame DER without collar; the clip and reference of bolo's `test_nemotron_separates_the_demo_dialogue`):

| Diarizer | DER | Missed | False alarm | Confusion | Speakers | Load | RTF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pyannote-speaker-diarization-3.1` | 6.05 % | 3.8 % | 2.3 % | 0 | 2 | 7.5 s | 0.91 |
| `pyannote-speaker-diarization-community-1` | 6.05 % | 3.8 % | 2.3 % | 0 | 2 | 8.2 s | 0.88 |
| bolo's `pyannote-3.0-eres2net` (ONNX, default) | 5.87 % | 3.6 % | 2.3 % | 0 | 2 | | 0.23 |
| bolo's `nemotron-3-diarization` (ONNX) | 5.87 % | 5.1 % | 0.8 % | 0 | 2 | | 0.06 |

Both PyTorch pipelines found the same seven turns, identical through bolo and from pyannote called directly, and
identical from run to run. On the first 20.8 s: 3.1 4.93 %, ONNX default 4.93 %, Nemotron 5.22 %. With a 250 ms
collar all are under 1 %. `num_speakers=3` gave three speakers and `max_speakers=1` one. The machine was shared
(load average up to 17 on 4 cores); through bolo under that load the RTFs were 1.0-1.5 (3.1, 4 threads) and 4.1
(community-1, 2 threads), and one run at 4 threads stalled for minutes inside the segmentation LSTM under that load (lower `BOLO_TORCH_THREADS` on a busy
machine). One clean synthetic dialogue: it shows the pipelines run and agree, not which is better on meetings
(pyannote's model cards: AMI SDM DER 22.7 % for 3.1, 19.9 % for community-1).

### DiariZen

2026-11-17, same machine type, torch 2.8.0+cpu, Python 3.13, diarizen and the pyannote.audio fork from
BUTSpeechFIT/DiariZen @ `844f5555b0a9`, 4 threads, the machine shared (load average about 15). The demo's two-voice
dialogue as above, through bolo (`POST /api/v1/diarize` with `model=diarizen`, plugin process started by bolo):

| Diarizer | DER | Missed | False alarm | Confusion | Speakers | Load | RTF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `diarizen-wavlm-large-s80-md-v2` | 5.03 % | 3.3 % | 1.8 % | 0 | 2 | 14.7 s | 12.4 |

The same seven turns through bolo as from the loader called directly; with a 250 ms collar DER 0. Its 16 s windows
every 1.6 s through WavLM Large make it slow on a CPU (337 s for 27 s of audio here). The authors' example
(`example/EN2002a_30s.wav`, an AMI excerpt with four speakers) ran with their pipeline the same way and gave four
speakers. On the model card, AMI SDM DER 13.9 % (pyannote 3.1: 22.4 %).

### DiCoW

2026-11-14, same machine type (it has AMX, so DiCoW ran in bfloat16), bolo and the plugin at 2 threads, the
machine idle for the speeds. Full numbers in docs/models.md, "Diarization-conditioned recognition", and
docs/measurements/dicow-conditioned.json.

* **Against the reference.** On two AMI excerpts with Nemotron's speaker turns, the plugin's STNO masks were
  identical to those of BUT's `pipeline.py` and every speaker's transcript was word for word the reference
  pipeline's (6 of 6 speaker streams, DiCoW v3.3; the same for SE-DiCoW, including its self-enrolment windows).
* **Accuracy through bolo** (cpWER, diarized transcription):

| Audio | Files | pyannote + whisper-base (bolo default) | Nemotron + whisper-base | Nemotron + `dicow-v3.3` | Nemotron + `se-dicow` |
| --- | --- | --- | --- | --- | --- |
| AMI far-field excerpts (45 s) | 24 | 69.4 % | 61.2 % | **26.0 %** | 25.3 % |
| Synthetic two-voice mixes (16 and 8 kHz) | 44 | 55.4 % | 25.4 % | **6.6 %** | 7.9 % |

* **Speed** (diarization included, 45 s excerpts with 3-4 speakers): RTF 0.45 at 2 threads, 0.26 at 4 threads;
  1.41 at 2 threads in float32. Load about 4.5 s; the plugin process holds about 3 GB in bfloat16. SE-DiCoW: RTF
  0.59 at 2 threads.
* **Known failure:** SE-DiCoW once repeated one word for the last 6 s of a speaker's stream (a 16 kHz Hindi mix);
  decoding is greedy without Whisper's temperature fallback, as in BUT's pipeline.

## Not tested

* **Veena and Svara** end to end: the language models were not downloaded (7.6 GB / 13.2 GB, no GPU here). The
  prompt layouts, token-to-SNAC mapping and streaming are unit-tested with fakes; the SNAC decoder ran for real.
* **PixIT** on real meeting audio, and with pyannote.audio 3.x (it ran with 4.0.7 and the shim above). PixIT
  through the configuration guard (and with its separation model from bolo's pinned download) was tested with
  fakes only; it was not loaded for real after that change.
* **pyannote diarization** with pyannote.audio 3.x (the pipeline outputs of 3.x are handled, untested), on GPU or
  MPS, on real meetings or calls, and `min_speakers` above the count found through bolo (that run stalled under
  machine load, see above). Before
  `pyannote/separation-ami-1.0` was accepted for the test account, loading it came back through bolo as
  `401 model_gated` naming that repository.
* GPU (CUDA, MPS), float16, bfloat16 other than DiCoW on CPU, Python 3.10, 3.11 and 3.13 for the plugin (it ran
  on 3.12), macOS.
* DiCoW on whole meetings (only 45 s excerpts and ~30 s mixes), on real telephone calls, on Hindi beyond one
  synthetic voice pair, and on single-speaker audio against plain Whisper large-v3-turbo; on CPUs without AMX /
  AVX512-BF16 only float32 speed was measured (forced on this machine). DiCoW gives no word timestamps.
* Telephone (8 kHz) recordings with the 8 kHz SepFormer; real meeting audio with any separator.
* `sravaani-1.0` and `indicconformer-600m-official`: not implemented.
* `maskgct`: the weights (6.6 GB, plus 2.3 GB of w2v-bert-2.0) were not downloaded. Tested with fakes of the authors'
  pipeline, builders and transformers' classes. Here, from the bundle (downloaded and matched to its pins): the
  phoneme front end ran for real in all six languages (phonemizer with espeakng-loader's espeak-ng, transformers
  4.47.1), and the whole pipeline (`text2semantic`, `extract_acoustic_code`, `semantic2acoustic`, the decoder) ran
  through the plugin's backend with tiny random networks and the real codec configurations, giving 24 kHz audio of
  the length the rule asks for. transformers 4.48 and later fail in Amphion's Llama layers (`position_embeddings`),
  hence the pin. Speech quality and cloning were not heard.
* `spark-tts-0.5b`: the weights (3.9 GB) were not downloaded. Tested with fakes of transformers' classes and of
  `BiCodecTokenizer`; the bundle was downloaded and matched its pins here, and the authors' BiCodec code from it ran
  with random weights through the plugin's `tokenize` / `detokenize` (32 voice tokens, 50 semantic tokens per
  second, 320 samples per token), so the shapes are the real ones. Speech quality, the created voices' pitch levels
  and Chinese were not heard.
* DiariZen on a GPU, on real meetings, with torch 2.9 or later (torchaudio 2.9 reads audio through torchcodec, which
  the extra keeps out), and on Windows (the install needs git).
* `glm-4-voice-9b`: tested with fakes of ChatGLM, its tokenizer, `WhisperVQEncoder`, the feature extractor and
  `AudioDecoder` only; the bundle was downloaded and matched its pins, not imported. The speech-token extraction is
  the authors' `extract_speech_token` with the device made a parameter (theirs moves everything to CUDA).
  `quantization: int4` was not tried.
* `minicpm-o-2.6`: tested with a fake of its remote-code model only. Prefilling earlier answers into the streaming
  API, `init_vision: false` and `mode: chat` with long conversations were not checked against the real model.
* `kimi-audio-7b-instruct`: tested with a fake `KimiAudio` (language model, prompt manager, sampler, detokenizer)
  only; the step-by-step loop mirrors `KimiAudio._generate_loop` at the pinned commit. The transformers range of the
  extra (4.48 to 4.51) is a guess from when the code was published; flash-attn on Windows was not tried.
* `step-audio-2-mini`: tested with fakes of its model, tokenizer, `utils.py` and `Token2wav` only; the bundle of
  inference code was downloaded and checked against its pins here, but not imported (its libraries and a GPU were
  not available). The format of its tool calls (`<tool_call>function\nname\narguments</tool_call>`) follows the
  authors' examples and could not be checked against the model.
* `f5-tts-v1-base` and `indic-speak`: unit-tested with fake backends only; the weights (1.3 GB / 7.1 GB, Indic-Speak
  gated) were not downloaded, and Indic-Speak's `inference.py` (gated) was not read.

## Licences

The plugin's code is Apache-2.0. Model weights and their training data carry their own terms; read each model
card before commercial use.

| Model | Weights | Notes |
| --- | --- | --- |
| Indic Parler-TTS | Apache-2.0 | Gated (accept conditions). |
| Parler-TTS Tiny v1 | Apache-2.0 | Text encoder from `google/flan-t5` (Apache-2.0). |
| IndicF5 | MIT | Gated. Trained on Rasa, IndicTTS, LIMMITS and IndicVoices-R; those datasets' terms were not reviewed here. Uses Vocos (MIT). |
| Veena | Apache-2.0 | Training data "proprietary" per its card. |
| Svara v1 | Apache-2.0 | Base `canopylabs/3b-hi-ft-research_release` (Llama 3.2 lineage); trained on SYSPIN, RASA, IndicTTS, SPICOR. SNAC codec MIT. |
| SepFormer WHAMR! (8 kHz, 16 kHz) | Apache-2.0 | Trained on WHAMR!, built from WSJ0 (LDC licence, research use) and WHAM! noise (CC-BY-NC-4.0): a provenance risk for commercial use. |
| DiCoW v3.3 | CC-BY-4.0 | Attribution to BUT Speech@FIT (cite the DiCoW and SE-DiCoW papers, model card). Fine-tuned from Whisper large-v3-turbo (MIT) on AMI and NOTSOFAR-1 (CC BY 4.0), LibriMix and synthetic LibriSpeech mixtures (LibriSpeech CC BY 4.0; LibriMix's noisy variants use WHAM! noise, CC-BY-NC-4.0, and which variants were used is not stated: unverified). The model repository's Python code is under the repository's licence; BUT's inference code (`BUTSpeechFIT/DiCoW`) is Apache-2.0. |
| SE-DiCoW | Apache-2.0 | Same base model and training-data notes as DiCoW v3.3. |
| pyannote speaker-diarization-3.1 | MIT | Gated (with `pyannote/segmentation-3.0`, MIT, gated). Downloads `pyannote/wespeaker-voxceleb-resnet34-LM` (CC-BY-4.0, WeSpeaker trained on VoxCeleb) and the PLDA files of community-1 (CC-BY-4.0, gated). |
| pyannote speaker-diarization-community-1 | CC-BY-4.0 | Gated (asks for contact details). Attribution required. |
| GLM-4-Voice 9B | glm-4-voice License | **Commercial use only after registering with Zhipu** (https://open.bigmodel.cn/mla/form, then free); academic research free; show "Built with glm-4" and ship the licence with the product; no military or illegal use; Chinese law, Haidian court. The same licence covers glm-4-voice-tokenizer and glm-4-voice-decoder. Code: Apache-2.0 (GLM-4-Voice), MIT (Matcha-TTS). |
| MiniCPM-o 2.6 | Apache-2.0 | Weights and code (the repository's remote code) per the model card; an optional registration questionnaire. |
| Kimi-Audio 7B Instruct | MIT | Per the model card; the model is built on Qwen2.5-7B (Apache-2.0). Its code repository has no LICENSE file: README and pyproject say MIT, Qwen-derived code Apache-2.0. Uses glm-4-voice-tokenizer, whose weights are under the glm-4-voice License (commercial use after registering with Zhipu): check that before commercial use. |
| Step-Audio 2 mini | Apache-2.0 | Model and code (the model repository's remote code, the Step-Audio2 repository's token2wav) under Apache-2.0. |
| Spark-TTS 0.5B | CC-BY-NC-SA-4.0 | **Not for commercial use**; derivatives share alike. Code (`sparktts`): Apache-2.0. The language model is fine-tuned from Qwen2.5-0.5B (Apache-2.0); the tokenizer uses `wav2vec2-large-xlsr-53` (Apache-2.0), shipped in the model repository. |
| MaskGCT | CC-BY-NC-4.0 | **Not for commercial use.** Code (Amphion): MIT. Uses `facebook/w2v-bert-2.0` (MIT). Its phoneme front end needs GPL-licensed packages that are not in the extra: phonemizer (GPL-3.0) with espeak-ng (GPL-3.0-or-later), pykakasi (GPL-3.0-or-later), Unidecode (GPL-2.0-or-later); LangSegment is BSD, pyopenjtalk-plus MIT, jieba, cn2an and pypinyin MIT. |
| DiariZen v2 (`BUT-FIT/diarizen-wavlm-large-s80-md-v2`) | CC-BY-NC-4.0 | **Not for commercial use** (the authors apply NC because some training sets forbid commercial use). The PLDA files are CC-BY-4.0. Code (DiariZen and its pyannote.audio fork): MIT. Downloads `pyannote/wespeaker-voxceleb-resnet34-LM` (CC-BY-4.0). |
| PixIT `speech-separation-ami-1.0` | MIT | Gated (both repositories). Downloads `speechbrain/spkrec-ecapa-voxceleb` (Apache-2.0) and `microsoft/wavlm-large` (no licence stated on its model card; check before commercial use). |

Apart from GLM-4-Voice (commercial use after registration), Spark-TTS, MaskGCT and DiariZen (non-commercial), no
model here is marked non-commercial by its publisher; the SepFormer training-data caveat above is the one to
weigh for commercial use. Nothing is downloaded unless you ask for that model.

## Development

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ./server -e "./plugins/bolo-torch[dev]"
cd plugins/bolo-torch
ruff check src tests && ruff format --check src tests && mypy && pytest -q
```

The tests use fake model backends and run the real plugin server as a subprocess through bolo's plugin host;
they need torch but no model libraries and no downloads.
