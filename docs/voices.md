# Saved voices and voice cloning

Some text-to-speech models can speak in a voice they have never been trained
on, from a short recording of it: their catalog entries list
`features: [voice-cloning]` (`bolo list --all`, or the Models screen). bolo
keeps such recordings as **saved voices**, so that a voice is recorded and
checked once and then named in any request, like the custom voices of OpenAI
or ElevenLabs:

```bash
bolo voices add Priya priya.wav --consent self --language en
curl localhost:8000/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model": "pocket-tts-en-llamacpp", "voice": "priya", "input": "Hello from bolo."}' -o hello.mp3
```

* [Creating a voice](#creating-a-voice)
* [Using a voice](#using-a-voice)
* [Voice cloning responsibly](#voice-cloning-responsibly)
* [How alike is the clone? Speaker similarity](#how-alike-is-the-clone-speaker-similarity)
* [Measured: cloning on CPU](#measured-cloning-on-cpu)
* [Tontaube](#tontaube)

## Creating a voice

From the console (Text to speech, "Saved voices", **New voice**, or **Save as
a voice** on a reference in the clone panel), the command line, or the API:

```bash
bolo voices add "Priya S." a.wav b.wav --consent permission --transcript "…" --language en \
  --default-model pocket-tts-en-llamacpp           # id: priya-s
curl localhost:8000/api/v1/voices -F name=Priya -F consent=self -F file=@priya.wav \
  -F score_model=pocket-tts-en-llamacpp            # also speaks a test sentence and scores it
```

Give 5 to 30 seconds of one person speaking clearly in a quiet room; several
files of the same person are joined. bolo stores a cleaned copy under
`<data_dir>/voices/<id>/` (`reference.wav` and `voice.json`):

* converted to mono 24 kHz, leading and trailing silence trimmed (0.15 s
  kept), pauses longer than 0.4 s shortened to 0.4 s;
* speech brought to about -20 dBFS with peaks under -1 dBFS;
* at most 60 s (cut at a pause); less than 2 s of speech is refused.

It then checks the recording and adds **warnings** (the voice is still saved):
shorter than 5 s; noisy (the loud frames are less than 16 dB above the quiet
ones; clean LibriSpeech read speech measured 18 to 55 dB); clipped; and, when
the speaker-embedding model is installed, **more than one person** (3-second
windows compared with each other; one person's windows scored 0.475 or more
on LibriSpeech, two people's a median of 0.23, and the warning starts below
0.40). Without a transcript the default speech-to-text model transcribes the
recording when it is installed (`transcript_source: auto`); some models
(Qwen3-TTS Base, CosyVoice3, F5-TTS) clone better with the words of the
reference.

The rest of the library: `bolo voices list | show <id> | rm <id> | score <id>
--model <m> | export [<id>…] -o voices.zip | import voices.zip [--replace]`,
and the routes in [api.md](api.md#saved-voices-apiv1voices). An export is a
.zip of each voice's `voice.json` and `reference.wav`; importing one needs a
consent record in each voice (or `--consent` for all of them).

## Using a voice

Name it as the `voice` of a model that clones voices, wherever a voice is
named: `/v1/audio/speech`, the TTS WebSocket, Realtime sessions, `bolo speak
--voice`, `bolo bench --voice`.

* `"voice": "priya"`: the model's own voice of that name if it has one, else
  the saved voice.
* `"voice": {"id": "priya"}`: OpenAI's shape for a custom voice; the saved
  voice wins over a model voice of the same name. The OpenAI SDKs send it with
  `voice={"id": "priya"}`, and `client.audio.voices.create(...)` /
  `client.audio.voice_consents.create(...)` work against bolo too
  ([api.md](api.md#openais-custom-voices)).
* Without `model`, the voice's `default_model` answers (set it with
  `--default-model` or `PATCH /api/v1/voices/{id}`).
* With a model that cannot clone, the answer is `400`: `'priya' is a saved
  voice, and model 'kokoro-v1.0-int8' cannot clone voices. Installed models
  that clone voices: …`.

The voice's transcript goes with the reference (`reference_text`), and its
language is the request's default language. A fallback model (setting
`fallbacks`) stands in for a cloned voice only if it clones voices too.
OpenAI's own voice names (`alloy`, `nova`, …) cannot be used as saved-voice
ids: stock clients send them for ordinary voices.

Speech in a cloned voice carries the response headers `x-bolo-cloned: true`
and `x-bolo-voice: <id>`, and the request trace says which saved voice was
used and on what consent (`extra.voice`).

## Voice cloning responsibly

A cloned voice can be mistaken for the person. bolo asks for one thing and
records it; the rest is up to whoever runs the server.

* **Consent is required.** Creating a voice needs a consent statement:
  `self` ("this is my voice") or `permission` ("I have the speaker's
  permission"); OpenAI's route also takes the id of a recorded consent
  statement (`/v1/audio/voice_consents`). A request without one is refused
  (`400 consent_required`); the console has a checkbox for it. The statement
  is stored with the voice, with the time and who made it: whether an API key
  authorised the request (bolo has one key, so the `created_by` name the client
  sends is what tells people apart), the client's address, or the user name
  for `bolo voices add`. bolo cannot check that the statement is true.
* **Outputs are labelled** for whoever receives them through bolo: the
  `x-bolo-cloned` and `x-bolo-voice` headers and the request trace. Plain audio
  files carry no label once they leave the server.
* **Watermarks.** No model in the catalog watermarks its output as bolo
  runs it. Chatterbox's reference implementation (Resemble AI's Python
  package) adds the Perth watermark after generation, but llama.cpp and
  audio.cpp do not run that step: Resemble's own detector (resemble-perth
  1.0.1) found no watermark in Chatterbox Turbo's output through bolo (score
  0.0; 1.0 once Perth had embedded one), and audio.cpp's source has no
  watermarker. bolo adds no watermark of its own yet:
  [AudioSeal](https://github.com/facebookresearch/audioseal) (Meta) is MIT
  licensed, weights included since April 2024, but it needs PyTorch, which the
  core server does not use, and there is no ONNX or numpy port whose detection
  rate has been checked; a weak homemade "watermark" would only give a false
  sense of safety. Adding AudioSeal (through the torch plugin, or an ONNX
  export with a measured detection rate) is a planned follow-up.
* Speaker similarity (below) measures likeness. It is not a way to verify who
  someone is.
* Check the model's licence as well: several cloning models are
  non-commercial, and Tontaube's licence restricts speech-generation services
  ([below](#tontaube)).

## How alike is the clone? Speaker similarity

bolo scores a clone with the speaker-embedding model its diarization uses
(the default diarization model, ERes2Net from 3D-Speaker; setting
`voice_similarity_model`): the cosine similarity between the embeddings of
the reference and of the synthesised speech, from -1 to 1.

| Score | Verdict | What real recordings did (LibriSpeech dev-clean, 16 speakers) |
| --- | --- | --- |
| 0.50 and above | same speaker | 99.4 % of pairs of utterances by the same person; 0.1 % of pairs by different people |
| 0.30 to 0.50 | similar | where most of the remaining overlap is: 10 % of different-person pairs reach 0.30 |
| below 0.30 | different | 90 % of different-person pairs |

Same-person pairs scored 0.45 to 0.93 (median 0.80), different-person pairs
-0.30 to 0.53 (median 0.11, 95th percentile 0.36): 160 and 3000 pairs, in
[measurements/speaker-similarity-calibration.json](measurements/speaker-similarity-calibration.json),
measured with `server/tools/calibrate_similarity.py`. These are real recordings
of native English speakers reading; synthesised speech, other languages and
noisy rooms were not part of the set, so read the verdicts as a rule of thumb.
Another embedding model has another scale (`calibrated: false` in the
answer).

Where the score appears:

* when a voice is created with `score_model` (or a `default_model`): a test
  sentence in the voice's language is spoken and scored;
* `GET /api/v1/voices/{id}/score?model=…` and `bolo voices score <id> --model
  …`; scores are kept in the voice's `scores`;
* `bolo bench --kind tts --model <cloning model> --voice <saved voice>` and
  `POST /api/v1/bench`: `speaker_similarity` after the sweep (also for a
  catalog voice that has a reference clip);
* the console's Saved voices panel, per model, with a button to score.

## Measured: cloning on CPU

Every model that clones voices and could run on the test machine (4 vCPUs of
a 2.1 GHz Xeon, no GPU, 15 GB RAM, **shared with other jobs**, so speeds are
pessimistic), through bolo's own code path (a saved voice, `core.synthesize`,
a server started by bolo, 4 threads: llama.cpp b12375's `llama-server`, and
audio.cpp v0.9.4's `audiocpp_server` built from source here because the
prebuilt release needs glibc 2.43), measured with
`server/tools/measure_cloning.py` on 2026-11-15/16. The voices: two LibriSpeech
dev-clean speakers (CC BY 4.0; ids 2277 and 84), about 15 s each with
transcripts. Each model spoke one test sentence in its language. Figures are
"speaker 2277 / speaker 84"; raw data in
[measurements/voice-cloning-cpu.json](measurements/voice-cloning-cpu.json).

| Model | Runtime | Language | Cloned | Similarity to the reference | Highest to 14 other people | Its stock voice vs the reference | RTF | First audio | Round-trip WER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pocket-tts-en-llamacpp` | llama-server | en | partly | 0.48 / 0.66 | 0.38 / 0.45 | 0.18 / 0.11 | 0.43 / 0.47 | 1.4 / 1.6 s | 0 / 0 % |
| `pocket-tts-de-llamacpp` | llama-server | de | weakly | 0.27 / 0.46 | 0.29 / 0.28 | 0.02 / 0.18 | 0.45 / 0.49 | 1.4 / 1.6 s | 18 / 0 % |
| `pocket-tts-es-llamacpp` | llama-server | es | weakly | 0.22 / 0.44 | 0.27 / 0.33 | 0.07 / 0.16 | 0.47 / 0.46 | 1.4 / 1.6 s | 79 / 37 % |
| `pocket-tts-fr-llamacpp` | llama-server | fr | weakly | 0.32 / 0.44 | 0.34 / 0.21 | -0.02 / 0.18 | 0.66 / 0.69 | 1.9 / 2.1 s | 50 / 50 % |
| `pocket-tts-it-llamacpp` | llama-server | it | partly | 0.39 / 0.37 | 0.38 / 0.30 | 0.12 / 0.32 | 0.40 / 0.44 | 1.3 / 1.6 s | 50 / 28 % |
| `pocket-tts-pt-llamacpp` | llama-server | pt | weakly | 0.24 / 0.39 | 0.31 / 0.28 | 0.00 / 0.32 | 0.68 / 0.63 | 2.1 / 2.3 s | 28 / 56 % |
| `chatterbox-turbo-llamacpp` | llama-server | en | yes | 0.73 / 0.70 | 0.37 / 0.51 | 0.23 / 0.06 | 3.63 / 3.06 | 20.3 / 17.8 s | 5 / 0 % |
| `qwen3-tts-1.7b-base-llamacpp` | llama-server | en | yes | 0.55 / 0.69 | 0.33 / 0.40 | 0.20 / 0.24 | 2.32 / 2.43 | 12.0 / 11.8 s | 0 / 0 % |
| `pocket-tts-en-audiocpp` | audio.cpp | en | yes | 0.55 / 0.58 | 0.37 / 0.45 | 0.13 / 0.18 | 9.71 / 18.91 | 45.1 / 92.3 s | 0 / 0 % |
| `moss-tts-nano-100m-audiocpp` | audio.cpp | en | yes | 0.59 / 0.71 | 0.32 / 0.62 | 0.09 / 0.05 | 1.05 / 1.47 | 6.0 / 9.4 s | 5 / 5 % |
| `xtts-v2-audiocpp` | audio.cpp | en | yes | 0.52 / 0.71 | 0.44 / 0.49 | 0.17 / 0.30 | 1.36 / 1.67 | 6.3 / 11.4 s | 0 / 0 % |
| `plapre-3-audiocpp` | audio.cpp | da | partly | 0.36 / 0.65 | 0.29 / 0.45 | 0.04 / 0.01 | 3.49 / 3.71 | 29.6 / 27.6 s | n/a (English text) |
| `omnivoice-audiocpp` | audio.cpp | en | no | 0.09 / 0.15 | 0.25 / 0.21 | -0.04 / 0.33 | 45.09 / 37.85 | 229.1 / 210.4 s | 100 / 10 % |
| `qwen3-tts-0.6b-base-audiocpp` | audio.cpp | en | yes | 0.71 / 0.73 | 0.32 / 0.46 | 0.16 / 0.23 | 3.24 / 4.34 | 17.6 / 20.1 s | 0 / 0 % |

* **Cloned**: yes when both clones score 0.50 or more (what the same real
  person scores); partly when one does, or when both are closer to their own
  reference than to any of the 14 other speakers; no when neither is; weakly
  otherwise (some likeness, above the model's stock voice, but not clearly
  the person). Chatterbox Turbo and
  Qwen3-TTS Base (both sizes), MOSS-TTS Nano and XTTS v2 copy the voice best;
  OmniVoice on audio.cpp ran but did not clone; Pocket TTS clones in English and less so
  in its other languages, where the English references are spoken in German,
  Spanish, French, Italian or Portuguese.
* **Round-trip WER**: `moonshine-tiny-en` for English, `whisper-base` for the
  others; whisper-base is a weak recogniser for those languages, so most of
  their WER is the recogniser's (for example French "Le renard brun rapide" came
  back as "Le Bernard Brown rapide").
* **First audio**: llama-server returns a sentence when it is done, so first
  audio is about the time of the first sentence. Chatterbox Turbo (RTF about 3)
  and Qwen3-TTS 1.7B (RTF about 2.4) are GPU models; on this CPU they are for
  files, not conversations.
* **Not run to the end**: `chatterbox-multilingual-v3-llamacpp`. With
  llama-server's default context it reserved about 13 GB and was killed for
  lack of memory; the entry now sets a 4096-token context (2.3 GB). It then
  cloned ("Hello there." came back exactly, similarity 0.53 to one of bolo's
  reference clips) but took 306 s for 1.2 s of audio, past bolo's 300 s read
  timeout.
* **audio.cpp fixes found by running it**: `xtts-v2-audiocpp` needed its GPT
  GGUF named (`model_file`; the server refuses a folder with two GGUFs) and
  must not be sent `reference_text` (audio.cpp answers 500 for it); both are now
  in the entry. A source build has no compiled-in model specs, so these runs
  passed `--model-spec-override <audio.cpp>/model_specs` (setting
  `tts_server_args`); the release builds compile them in.
  `zonos3-mini-audiocpp` cannot run on v0.9.4: that release has no zonos3
  family. Plapre 3 is Danish and was given the English sentence, so its WER is
  not a measure. audio.cpp on a CPU is slow (Pocket TTS ran at RTF 10 to 19
  there against 0.4 to 0.5 on llama.cpp; OmniVoice at RTF 40); its authors
  target CUDA.
* **Not run**: the other 25 audio.cpp cloning entries (larger downloads and
  slower on this CPU than the time allowed), the bolo-torch models F5-TTS,
  Spark-TTS and MaskGCT (PyTorch plugin, several GB, non-commercial licences)
  and Tontaube (below). They stay `validated: untested`.

## Tontaube

[TontaubeV1.1](https://huggingface.co/TontaubeAI/TontaubeV1.1-int4)
(Cremer & Cremer Technologies) is an expressive voice-cloning model, English
and German first, also Spanish, French, Italian, Dutch and Portuguese. It runs
only in its own inference server, [craitech/tontaube](https://github.com/craitech/tontaube)
(Apache-2.0), which bolo knows as the tts-server runtime `tontaube`.

| Entry | Weights | Runs on |
| --- | --- | --- |
| `tontaube-v1.1-int4` (alias `tontaube`) | 4-bit (GPTQ), 1.8 GB | NVIDIA GPU (about 4 GiB, peaks at about 6.5 GiB by the model card), or an x86-64 Linux CPU with AVX2, FMA and F16C (about real time on a 12-core desktop, by the card) |
| `tontaube-v1.1` | full precision, 6.4 GB | the server's vLLM backend: an NVIDIA GPU with 24 GiB or more |

**Licence: Tontaube Community Model License 1.0**, not an open-source
licence. In short (the licence text decides): research, noncommercial use and
testing are free; commercial use is free while your group's revenue is below
EUR 200,000, with the attribution "Powered by TontaubeV1" (a credits line or a
docs page is enough), and needs a commercial licence above it; **offering the
model to others as a speech-generation service always needs a commercial
licence**, whatever your revenue (a small free demo is fine; using it inside
your own product is not a service); fine-tunes and other derivatives you
distribute carry the same licence. Running bolo so that other people can
generate speech with Tontaube is such a service, which is why the entries say
`commercial_use: false`.

Two ways to run it:

* **Your own server**: run `tontaube serve` yourself (any machine with a GPU)
  and set `tts_servers = ["tontaube=http://gpu-box:8080"]` (and
  `tts_server_api_key` if it has `REQUIRE_API_KEY=1`; bolo sends it as
  `x-api-key`).
* **Managed by bolo**: `bolo components install tontaube` clones the pinned
  release (v1.1.0, commit checked), `bolo pull tontaube` downloads the weights
  (bolo passes them as `MODEL_PATH`), and the first request starts `uv run
  --python 3.12 --frozen tontaube serve` on a loopback port with a random API
  key, `LITE_DEVICE=cpu` and `LITE_CPU_THREADS` on the CPU. `uv` must be on
  `PATH`. **The first start installs the server's own dependencies (PyTorch,
  vLLM and more: several GB) and downloads DualCodec, W2V-BERT 2.0 (2.3 GB)
  and VibeVoice-1.5B (5.4 GB, for its output adapter)**; it can take a long
  time, so the start timeout is `tontaube_start_timeout_s` (an hour).

Requests: `voice` is one of the server's 46 bundled reference voices (MIT
licensed; `marcus` first, each with a language and a default style), a saved
voice, or `reference_audio`. `options` takes `style` (`audiobook`,
`conversational`, `agentic`), the five sliders `rate` (0-4), `pause` (0-4),
`pitch_range` (0-5), `pitch_level` (0-4) and `legato` (0-2; 2 is normal
articulation), and `temperature` and `seed`:

```json
{"model": "tontaube", "voice": "priya", "input": "Take it slowly.", "language": "de",
 "options": {"style": "audiobook", "rate": 1, "pause": 3}}
```

`speed` other than 1.0 is refused (use `rate`). The server answers each
sentence whole (`POST /predict`), so the first audio comes when the first
sentence is done.

**Not run here.** The test machine (4 vCPUs with AVX2, FMA and F16C, so the
CPU path would apply) had 8 to 14 GB of disk free and 15 GB of memory, both
shared with other jobs. The first start needs the 1.8 GB weights, the
server's locked dependencies (PyTorch 2.9.1 with its CUDA libraries, vLLM
0.16, transformers and more: several GB), W2V-BERT 2.0 (2.3 GB) and, unless
`ENABLE_VIBEVOICE=0`, VibeVoice-1.5B (5.4 GB), and loads W2V-BERT and the four
codebook models into memory; other jobs on the machine were already being
killed for lack of memory. Both entries stay `validated: untested`; the
engine is tested against a fake Tontaube server, managed start included.
