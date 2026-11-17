# Models bolo does not run

Models that were looked at for the catalog and left out because no runtime
bolo has can run them yet. Every catalog entry is runnable; these are not in
the catalog. Each has the concrete reason and what would change it.

## Denoising

| model | licence | size | why it is not in the catalog |
| --- | --- | --- | --- |
| ClearVoice FRCRN_SE_16K ([alibabasglab/FRCRN_SE_16K](https://huggingface.co/alibabasglab/FRCRN_SE_16K)) | Apache-2.0 | 161 MB | PyTorch only; the network comes only with the `clearvoice` package (ClearerVoice-Studio), which pins numpy below 2, librosa 0.10.2, opencv, scenedetect and torchvision and needs `audioop-lts` on Python 3.13. A trial loader through that package ran about 50 times slower than real time on one shared 4-vCPU machine, and its output on a noisy sample correlated less with the clean speech than the noisy input did, so it was not added. No ONNX export is published. |
| ClearVoice MossFormer2_SE_48K ([alibabasglab/MossFormer2_SE_48K](https://huggingface.co/alibabasglab/MossFormer2_SE_48K)) | Apache-2.0 | 222 MB | the same package and dependencies as FRCRN; not tried after FRCRN's speed. |
| ClearVoice MossFormerGAN_SE_16K ([alibabasglab/MossFormerGAN_SE_16K](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K)) | Apache-2.0 | 39 MB (+ 98 MB discriminator, not needed) | the same package and dependencies as FRCRN; not tried after FRCRN's speed. |
| Resemble Enhance ([ResembleAI/resemble-enhance](https://huggingface.co/ResembleAI/resemble-enhance)) | MIT | 713 MB | the `resemble-enhance` package pins torch 2.1.1, deepspeed 0.12.4 and gradio 4.8; the enhancer is a DeepSpeed checkpoint with no ONNX export; over the 500 MB this project tries on its shared test machine. |
| DeepFilterNet 2 (`models/DeepFilterNet2_onnx.tar.gz` of [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) 0.5.6) | MIT or Apache-2.0 | 9 MB | the official ONNX export is three graphs (encoder, ERB decoder, deep-filter decoder) that need DeepFilterNet's own STFT, ERB features, normalisation and deep filtering around them (libDF); bolo has none of that. DeepFilterNet 3 is in the catalog (`deepfilternet3`) as torchDF's single streaming graph, which carries all of it. |

## Text-to-speech

| model | licence | size | why it is not in the catalog |
| --- | --- | --- | --- |
| MiraTTS on audio.cpp ([YatharthS/MiraTTS](https://huggingface.co/YatharthS/MiraTTS) at `ff750bd7`) | CC-BY-NC-SA-4.0 | 1.6 GB | audio.cpp runs it (`mira_tts` family, experimental there), but only from a GGUF made with audio.cpp's own converter (`tools/community_models/convert_mira_tts.py`, then `audiocpp_gguf --family mira_tts`), and nobody publishes that file: audio.cpp's model repository (`audio-cpp/audio.cpp-gguf`, checked 2026-11-17) has no MiraTTS, and the GGUFs on the Hub (`Mungert/MiraTTS-GGUF`, `mradermacher/MiraTTS-GGUF`, `eugenehp/miratts`) hold only the Qwen2 language model, without the speaker encoder, DAC decoder and upsampler audio.cpp needs. Anyone with the converted folder can still use it: `bolo pull ./dir` with `runtime: audio.cpp`, `family: mira_tts`. A published audio.cpp package would bring it back. |

## Full-duplex speech-to-speech

The full-duplex models that run are in the catalog (models.md, "Full-duplex models"). These were looked at and are
not:

| model | licence | size | why it is not in the catalog |
| --- | --- | --- | --- |
| Hibiki Rust builds ([kyutai/hibiki-1b-rs-q6k](https://huggingface.co/kyutai/hibiki-1b-rs-q6k), [hibiki-1b-rs-q8](https://huggingface.co/kyutai/hibiki-1b-rs-q8), [hibiki-1b-rs-bf16](https://huggingface.co/kyutai/hibiki-1b-rs-bf16), [hibiki-2b-rs-bf16](https://huggingface.co/kyutai/hibiki-2b-rs-bf16)) | CC-BY-4.0 | 1.9 to 6.2 GB | the only program that loads them is Kyutai's `hibiki-rs` (kyutai-labs/hibiki), a command line that translates a file (`gen in.mp3 out.wav`), not a server; Kyutai's Rust server (`moshi-backend`) loads Moshi's configuration only. The same models are in the catalog in their PyTorch and MLX builds. A streaming server for these files would change this. |
| Third-party conversions of Moshi and PersonaPlex (GGUF such as `Codes4Fun/personaplex-7b-v1-q4_k-GGUF`, MLX 4- and 8-bit such as `aufklarer/PersonaPlex-7B-MLX-4bit`, bitsandbytes 4-bit, ONNX exports) | as their source (CC-BY-4.0, NVIDIA Open Model License) | 4 to 9 GB | not published by Kyutai or NVIDIA, and none comes with a server that speaks Kyutai's protocol and, for PersonaPlex, takes its voice and text prompts; Kyutai's MLX server loads Moshi checkpoints only. Not checked one by one. |

## End-of-turn detection

| model | licence | why bolo does not run it |
| --- | --- | --- |
| Smart Turn v2 ([pipecat-ai/smart-turn-v2](https://huggingface.co/pipecat-ai/smart-turn-v2) at `3267e96b`, `model.safetensors`, 379 MB) | BSD-2-Clause | Published only as PyTorch weights (a wav2vec2-base model with an end-of-turn head); there is no ONNX or other export that bolo's runtimes load, and bolo does not carry PyTorch in the core. Smart Turn v3 (`smart-turn-v3`, 9 MB, in the catalog) replaced it. |
| LiveKit turn detector, multilingual ([livekit/turn-detector](https://huggingface.co/livekit/turn-detector) tag `v0.4.1-intl`, `onnx/model_q8.onnx`, 396 MB) and English ([main](https://huggingface.co/livekit/turn-detector) at `fba34c38`, `model_quantized.onnx`, 165 MB) | LiveKit Model License | Its licence allows use **only together with the LiveKit Agents framework**, "not on a standalone basis or with any other frameworks", and forbids using its outputs to improve other models. bolo is another framework, so running it inside bolo would break the licence. (Technically it would run: int8 ONNX, Qwen2.5-0.5B, reading the conversation as text.) |
| TEN Turn Detection ([TEN-framework/TEN_Turn_Detection](https://huggingface.co/TEN-framework/TEN_Turn_Detection) at `41845390`, four safetensors files, 15.2 GB) | Hugging Face card: Apache-2.0; the [GitHub licence](https://github.com/TEN-framework/ten-turn-detection/blob/b808d8a2eccc0e771965f31b74a04f136f55113f/LICENSE) adds Agora's conditions (not on end-user devices; only for your own applications; nothing that competes with Agora) | A 7.6-billion-parameter language model (Qwen2.5-7B) in bfloat16 PyTorch weights: no ONNX or GGUF export from its authors, no in-process runtime in bolo for a causal language model used as a classifier, and 15 GB to load for a decision taken at every pause. |

The text-reading end-of-turn models bolo does run are the Namo Turn Detectors (`namo-turn-en`,
`namo-turn-multilingual`, Apache-2.0), wired to the transcript of the utterance at each pause.
| MiniCPM-o 2.6 in full-duplex ([openbmb/MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6)) | Apache-2.0 | 17.4 GB | it has no full-duplex mode: its "live streaming" prefills the user's audio and then generates (the application decides when the turn ends; OpenBMB's demo server uses a voice-activity detector), so it never listens while it speaks. Its turn-based use is a separate catalog entry; its successor MiniCPM-o 4.5 is in the catalog as a full-duplex model (`minicpm-o-4.5`). |
| SALMONN-omni (Tsinghua and ByteDance) | - | - | no weights or inference code published (a paper only); checked on Hugging Face and GitHub on 2026-11-17. |
| OmniFlatten (Alibaba) | - | - | no weights or code from the authors (a paper and a project page; an unofficial reimplementation has no weights). |
| SyncLLM (Meta) and LSLM (listening-while-speaking) | - | - | no weights published by the authors; LSLM's public code is an unofficial reimplementation without weights. |
| Covo-Audio-Chat-FD (Tencent) | - | - | the full-duplex variant described in the Covo-Audio report is not published; only the turn-based `tencent/Covo-Audio-Chat` is. |
| RoboEgo / FLM-Ego (CofeAI) | - | - | not published; its audio-only sibling FLM-Audio is in the catalog (`flm-audio`). |
| NTPP ([aigc-x/NTPP](https://huggingface.co/aigc-x/NTPP), gated, and [NTPP-0.5B](https://huggingface.co/aigc-x/NTPP-0.5B)) | Apache-2.0 | 1 to 15 GB | the published code (Chaos96/NTPP) turns a recorded file into a reply file (`inference.py --input_audio`); there is no streaming loop that could hold a live conversation. A streaming server for it would change this. |
| DuplexOmni ([MuyeHuang/DuplexOmni](https://huggingface.co/MuyeHuang/DuplexOmni)) | Apache-2.0 | 70.5 GB | runs on its authors' modified vLLM runtime, recommended on eight NVIDIA H20 GPUs; not wired here: a bridge to that runtime has not been written, and nothing short of a multi-GPU server runs it. |
| BayLing-Duplex ([BayLing-Models/BayLing-Duplex](https://huggingface.co/BayLing-Models/BayLing-Duplex)) | glm-4-voice License (commercial use after registering with Zhipu) | 19 GB + GLM-4-Voice's tokenizer and decoder | weights and the authors' scripts exist (ChatGLM remote code, GLM-4-Voice's speech tokenizer and decoder); not wired here yet: its scripts have no server, so it needs a server of bolo's own like MiniCPM-o 4.5's, written against code that could not be run here. |
| VITA-1.5 ([VITA-MLLM/VITA-1.5](https://huggingface.co/VITA-MLLM/VITA-1.5)) | no licence on its model card (the VITA code is Apache-2.0) | 19.6 GB | its real-time demo interrupts a turn-based model with a voice-activity detector, like Freeze-Omni from the same team, and needs a vLLM build of its own; not wired here yet. |
| LFM3-Audio 2B full-duplex adapter ([MAdel121/LFM3-Audio-2B-Full-Duplex](https://huggingface.co/MAdel121/LFM3-Audio-2B-Full-Duplex)) | LFM Open License v1.0 | 0.1 GB (+ the base model) | a third-party LoRA adapter with its own Python package in the model repository, not from the base model's authors; not checked further. |
| OSUM-EChat, LLaMA-Omni2 | Apache-2.0 | 1 to 25 GB | not full-duplex (turn-based speech conversation); out of scope for this table. |

## Language identification

Every language identifier that states a licence is in the catalog and runs (docs/models.md, "Language
identification"). These were found and left out because their repositories state no licence at all: with no
terms, nobody has been given the right to copy or use the weights, so bolo can neither download them for you nor
label them. They are fine-tunes for the 22 scheduled languages (no English) with transformers' audio-classification
classes, which bolo-torch's `hf-audio-classification` kind would load as they are; a licence added upstream is all it
takes to catalogue them.

| model | licence | size | why it is not in the catalog |
| --- | --- | --- | --- |
| [radioduran/indic-language-identification](https://huggingface.co/radioduran/indic-language-identification) (HuBERT) | none stated | 379 MB | no licence on the model card or in the repository |
| [PremKumar12/indic-language-identification](https://huggingface.co/PremKumar12/indic-language-identification) (HuBERT) | none stated | 378 MB | no licence on the model card or in the repository |
| [ansc00053/indic-language-identification](https://huggingface.co/ansc00053/indic-language-identification) (wav2vec 2.0) | none stated | 1.3 GB | no licence on the model card or in the repository |
| [vensti14/indic-SLID-improved](https://huggingface.co/vensti14/indic-SLID-improved) (wav2vec 2.0) | none stated | 1.3 GB | no licence on the model card or in the repository |
