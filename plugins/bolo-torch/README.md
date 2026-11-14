# bolo-torch

The PyTorch engine plugin for [bolo](https://github.com/cb-Sigma/bolo), the torch-free speech server.

bolo never imports PyTorch. When a catalog entry says `engine: plugin:torch`, bolo starts this package in a
separate process (`python -m bolo_torch`) and talks to it over the plugin protocol
([docs/plugins.md](../../docs/plugins.md)): it loads models there, streams synthesis back sentence by sentence,
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
| `sepformer-whamr16k` | separation | SpeechBrain SepFormer, WHAMR!, 16 kHz, 2 speakers | `sepformer` | Apache-2.0 (WSJ0 data, see below) | no | CPU, see below |
| `sepformer-whamr` | separation | SpeechBrain SepFormer, WHAMR!, 8 kHz, 2 speakers | `sepformer` | Apache-2.0 (WSJ0 data, see below) | no | CPU, see below |
| `pixit-ami` | separation | pyannote PixIT `speech-separation-ami-1.0`, diarization + separation | `pyannote` | MIT | yes (two repositories) | CPU, see below |
| `dicow-v3.3` | STT, diarization-conditioned | BUT-FIT DiCoW v3.3: Whisper large-v3-turbo told who talks when | `dicow` | CC-BY-4.0 | no | CPU, see below |
| `se-dicow` | STT, diarization-conditioned | BUT-FIT SE-DiCoW: DiCoW with self-enrolment | `dicow` | Apache-2.0 | no | CPU, see below |

The separation models are bolo `separation` models in "blind" mode: bolo uses them for `separate_overlaps=true`
on diarized transcription and on `POST /api/v1/diarize`, and assigns their tracks to its own diarized speakers.
They can also be run without bolo: `bolo-torch separate mix.wav -m sepformer-whamr16k -o out/`.

The DiCoW models are bolo `stt` models that are *speaker-conditioned*: in a diarized transcription
(`diarize=true` or `response_format=diarized_json`) bolo diarizes the recording (with
`nemotron-3-diarization` unless the request names another diarizer), then sends the whole recording to the
plugin with the speakers' frame activity, and the model transcribes it once per speaker. Overlapping speech is
transcribed for each speaker instead of being cut at turn boundaries. Without diarization they work as an
ordinary (slow) Whisper. See "DiCoW" below and docs/models.md, "Diarization-conditioned recognition".

The catalog also lists `sravaani-1.0` and `indicconformer-600m-official` for this plugin. bolo-torch 0.1.0 does
not implement them; loading one answers `400 invalid_request` saying so.

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

From a checkout of the bolo repository:

```bash
# 1. PyTorch. The CPU-only build is much smaller than the default (CUDA) build on Linux:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# 2. bolo and the plugin, with the model families you want:
pip install ./server "./plugins/bolo-torch[parler,sepformer]"
```


Extras: `parler` (Parler-TTS), `f5` (IndicF5), `f5-original` (the original F5-TTS checkpoints, on the `f5-tts`
package, which pins its own libraries: give it its own environment), `indic-speak` (transformers 5 and SNAC),
`orpheus` (Veena, Svara), `sepformer`, `pyannote` (PixIT),
`dicow` (DiCoW, SE-DiCoW: transformers 4.55 and pandas). DiCoW uses the plugin
protocol's `speakers` request (bolo 0.2.0 or later).
Installed in bolo's environment, the plugin is found through its entry point (`bolo.engines: torch`) and
`bolo doctor` lists `plugin:torch`. Check the plugin's own environment with `bolo-torch doctor`.

**Some extras do not mix.** `parler-tts` 0.2.3 pins `transformers==4.46.1`, while `pyannote.audio[separation]`
needs `transformers>=4.48.3` and DiCoW's model code needs transformers 4.55. Put PixIT in a second environment and point bolo at it (a configured command wins
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

## GPUs

* bolo passes its own `device` setting (default `cpu`) to the plugin. To use a GPU set `BOLO_TORCH_DEVICE=auto`
  (CUDA if available, then Apple MPS, else CPU) or `cuda`, `cuda:1`, `mps`. A model's catalog option `device`
  also works. Asking for `cuda` with a CPU-only torch answers `400` with that explanation.
* `dtype` is automatic: float32 on CPU, bfloat16 on CUDA when supported (else float16), float16 on MPS. Override
  with `BOLO_TORCH_DTYPE` or the option `dtype` (`float32`, `float16`, `bfloat16`).
* For CUDA install the default torch wheel (`pip install torch`) or the one for your CUDA version from
  https://pytorch.org/get-started/locally/, then the plugin.
* **No GPU was used in testing.** The GPU, MPS and half-precision paths are untested.

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

Indic Parler-TTS, IndicF5 and PixIT are gated on Hugging Face. To use them:

1. Sign in to Hugging Face and accept the conditions on each model page:
   [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts),
   [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5), and for PixIT **both**
   [pyannote/speech-separation-ami-1.0](https://huggingface.co/pyannote/speech-separation-ami-1.0) and
   [pyannote/separation-ami-1.0](https://huggingface.co/pyannote/separation-ami-1.0).
2. Create a read token at https://huggingface.co/settings/tokens and set `HF_TOKEN` (or `BOLO_HF_TOKEN`) in the
   environment bolo runs in; the plugin inherits it.

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
* **PixIT** on real meeting audio, and with pyannote.audio 3.x (it ran with 4.0.7 and the shim above). Before
  `pyannote/separation-ami-1.0` was accepted for the test account, loading it came back through bolo as
  `401 model_gated` naming that repository.
* GPU (CUDA, MPS), float16, bfloat16 other than DiCoW on CPU, Python 3.10, 3.11 and 3.13 for the plugin (it ran
  on 3.12), macOS.
* DiCoW on whole meetings (only 45 s excerpts and ~30 s mixes), on real telephone calls, on Hindi beyond one
  synthetic voice pair, and on single-speaker audio against plain Whisper large-v3-turbo; on CPUs without AMX /
  AVX512-BF16 only float32 speed was measured (forced on this machine). DiCoW gives no word timestamps.
* Telephone (8 kHz) recordings with the 8 kHz SepFormer; real meeting audio with any separator.
* `sravaani-1.0` and `indicconformer-600m-official`: not implemented. `spark-tts-0.5b`, `maskgct` and
  `diarizen-wavlm-large-s80-md-v2` are catalogued but not implemented (no packaged inference code; the plugin
  protocol has no diarization request).
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
| PixIT `speech-separation-ami-1.0` | MIT | Gated (both repositories). Downloads `speechbrain/spkrec-ecapa-voxceleb` (Apache-2.0) and `microsoft/wavlm-large` (no licence stated on its model card; check before commercial use). |

No model here is marked non-commercial by its publisher; the SepFormer training-data caveat above is the one to
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
