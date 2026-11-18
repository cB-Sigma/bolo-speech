# Roadmap

Things bolo does not do yet and that are planned for later. Effort: S (days), M (one or two weeks), L (more).
Suggestions and pull requests are welcome; open an issue first for anything marked M or L.

## Speech-to-text

| Feature | Today | Effort | Likely approach |
|---|---|---|---|
| Profanity filter | missing | S | Word-list masking, reusing the existing word-list machinery |
| Pronunciation assessment (goodness of pronunciation) | missing | M | A phoneme CTC model (wav2vec2-xlsr-53-espeak-cv-ft, Apache-2.0) plus GOP scoring on top of forced alignment |

## Speakers and audio analysis

| Feature | Today | Effort | Likely approach |
|---|---|---|---|
| Speaker age and gender | missing | S | audeering wav2vec2 age-gender (CC-BY-NC-SA-4.0), ECAPA gender classifiers |
| Synthetic speech (deepfake) detection | missing | M | AASIST / AASIST-L (MIT), SSL-AASIST |
| Standalone source-separation endpoint | only through `separate_overlaps` | S | The SepFormer, PixIT and VoxLock models already in the catalog |

## Audio enhancement

| Feature | Today | Effort | Likely approach |
|---|---|---|---|
| Dereverberation | missing | S-M | WPE (nara_wpe, MIT, no model), MossFormer2 (Apache-2.0), VoiceFixer (MIT) |
| Bandwidth extension / super-resolution | missing | M | MossFormer2_SR_48K (Apache-2.0), AudioSR (MIT code), NU-Wave 2 |
| Loudness normalisation of synthesised speech | missing | S | EBU R128 with a -16 LUFS default (pyloudnorm, MIT) |

## Text-to-speech and voice

| Feature | Today | Effort | Likely approach |
|---|---|---|---|
| Voice design from a text description | Parler only | S | Qwen3-TTS VoiceDesign, Maya1, Parler-TTS (all in the catalog) |
| Singing voice synthesis | missing | M | DiffSinger / OpenUtau voicebanks (licence per voicebank), YuE |
| Accent conversion | missing | L | Voice-conversion models with an accent-target reference |
| Dubbing (translate and re-voice with timing) | missing | M-L | Recognition, translation, a cloning voice and duration fitting, on top of forced alignment |
| Watermarking with the original PyTorch models | ONNX ports only (AudioSeal; Perth and WavMark next) | S | bolo-torch catalog entries for `audioseal`, `resemble-perth` and `wavmark`, behind the same `watermark` option and detect endpoint |
| Swappable phonemisers | each voice uses the phonemiser it was trained with (uccharan, dizione, espeak-ng, ...) | M | Phonemisers as a catalog model kind; a voice names the one it was trained with, and uccharan (or any other) can be swapped in with a symbol mapping, mainly for retrained Indian-language voices |

## Voice agents and telephony

| Feature | Today | Effort | Likely approach |
|---|---|---|---|
| Backchannels ("mm-hm") in turn-based agents | missing | M | Short filler clips driven by the turn model; full-duplex models do this natively |
| Memory across sessions | in-session history only | S | An optional per-agent summary store handed to the language model |
| Call audio recording | transcript and latency only | S | A two-channel WAV per call |

## Operations

| Feature | Today | Effort | Likely approach |
|---|---|---|---|
| Async transcription jobs with a webhook callback | missing | S | The existing job runner plus signed webhooks |
| Air-gapped install bundle | partial (`auto_pull = false`, local wheels) | S | `bolo pull --export` / `--import` of the data directory |
| Helm chart | missing | S | A chart around `deploy/kubernetes/bolo.yaml` |
| Horizontal autoscaling with shared state | missing (one replica) | L | Shared session state and autoscaling on the existing Prometheus metrics |

## Clients

| Feature | Today | Effort | Likely approach |
|---|---|---|---|
| Client SDKs for bolo's own routes | OpenAI SDKs work; examples only | S | Thin Python and TypeScript wrappers |
| Native mobile / edge SDK | browser demo only | L | sherpa-onnx's mobile bindings |
| Fine-tuning and adaptation | missing | L | Outside an inference server; adapters trained with external tools could be loaded |
