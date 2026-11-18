# Apple Silicon (M1 and later)

On a Mac with Apple Silicon, `device = "auto"` (or `coreml`; `mps` and `metal` mean the same) puts models on the GPU
and the Neural Engine through the runtime each engine has. The GPU shares the machine's memory: bolo plans with the
memory macOS calls available, and `bolo ps` shows it as unified memory.

```
pip install "bolo-speech[all]"          # sherpa-onnx, ONNX Runtime (Core ML provider), faster-whisper, ...
pip install "bolo-speech[whispercpp]"   # whisper.cpp with Metal
BOLO_DEVICE=auto bolo serve
```

## What runs where

| engine | on Apple Silicon | from |
|---|---|---|
| sherpa-onnx (most speech recognition and voices) | Core ML (ONNX Runtime's Core ML provider inside sherpa-onnx) | `bolo-speech[sherpa]` |
| ONNX Runtime engines (Nemotron diarization, DiCoW ONNX, end-of-turn models, in-process speech-to-speech) | `CoreMLExecutionProvider` | `onnxruntime` (the PyPI wheel for macOS arm64 has it) |
| llama-server (voices, audio LLMs, speech-to-speech) | Metal | `bolo components install llama-server` (the macOS arm64 release is the Metal build) |
| audio.cpp | Metal | a build of your own (`audiocpp_server_bin`) |
| whisper.cpp | Metal | `bolo-speech[whispercpp]` (pywhispercpp's macOS arm64 wheels are built with Metal) |
| bolo-torch plugin (PyTorch models) | MPS | `pip install bolo-torch[...]`; torch's macOS wheels have MPS |
| MLX models (Whisper, Parakeet, Kokoro and more) | Metal, through MLX | `bolo-speech[mlx]` (see "MLX" below) |
| Moshi (full duplex) | Metal, through `moshi_mlx` | `bolo components install moshi` (the MLX build is the default on macOS arm64) |
| faster-whisper (CTranslate2) | the CPU: CTranslate2 has no Metal backend | |

A model whose runtime cannot reach the GPU is placed on the CPU with a notice (`bolo ps`, `GET /api/v1/ps`).
Core ML compiles a model the first time it loads it; the first load of a model is slower than the next ones.

## What bolo doctor says

`bolo doctor` lists the GPU as `Apple M... GPU` with unified memory and suggests `device = "coreml"` (or `auto`). It
reports ONNX Runtime's providers (`CoreMLExecutionProvider`), sherpa-onnx, the llama-server build (Metal),
pywhispercpp's GPU back ends (`metal`), the torch plugin's `MPS: yes/no`, and MLX (`Metal`, mlx-audio's version).
`bolo recommend` counts the GPU for each engine that can use it.

## MLX

`pip install "bolo-speech[mlx]"` adds the MLX engine (`engine: mlx`, in bolo's own process, not a plugin): Apple's
array framework running models on the GPU through Metal. The extra installs `mlx` (MIT), `mlx-whisper` (MIT),
`parakeet-mlx` (Apache-2.0), `mlx-audio` (MIT) and uccharan, only on macOS arm64
(`sys_platform == "darwin" and platform_machine == "arm64"`); elsewhere it installs nothing.

| catalog id | what | licence |
|---|---|---|
| `whisper-tiny-mlx`, `whisper-base-mlx` | Whisper on mlx-whisper: 99 languages, translation, word timestamps | MIT |
| `parakeet-tdt-ctc-110m-mlx` | NVIDIA Parakeet TDT-CTC 110M (English) on parakeet-mlx, word timestamps | CC-BY-4.0 |
| `kokoro-v1.0-mlx` | Kokoro-82M on mlx-audio: 41 voices in English, Spanish, French, Hindi, Italian, Portuguese; voice blends | Apache-2.0 |
| `moshiko-mlx-q4`, `-q8`, `-bf16` | Moshi, full duplex, through `moshi_mlx` (`bolo components install moshi`) | CC-BY-4.0 |

```
bolo pull whisper-tiny-mlx kokoro-v1.0-mlx
curl -F model=whisper-tiny-mlx -F file=@sample.wav http://127.0.0.1:8000/v1/audio/transcriptions
```

Kokoro on MLX reads phonemes that bolo makes itself (uccharan's Kokoro front end, the same as for the sherpa-onnx
Kokoro voices), so mlx-audio's own front end (misaki, which pulls in spaCy and espeak-ng) is not needed; the Japanese
and Chinese voices are left out because bolo has no front end for them.

Placement: a model placed on the GPU (`device = "auto"` or `coreml` on a Mac) runs on MLX's GPU stream, one placed on
the CPU on its CPU stream. When MLX has no Metal device (MLX's Linux CPU wheels, a virtual Mac without one) the model
runs on the CPU and its placement says so (`bolo ps`, notice "MLX could not start Metal"). MLX keeps one default device
per process, so bolo runs one MLX call at a time.

MLX also has Linux CPU wheels (`pip install "mlx[cpu]" mlx-whisper parakeet-mlx mlx-audio`); the models run there on
the CPU, slowly. That is how they were first run.

## The torch plugin on MPS

bolo passes its own device to the plugin: `coreml` (what `auto` becomes on a Mac) means MPS there. When torch sees no
MPS device (a virtual Mac, a CPU-only torch), the plugin runs on the CPU with a warning instead of failing. Asked for by
name (`BOLO_TORCH_DEVICE=mps`, or a model's `device` option), a missing MPS device is an error. Some PyTorch operations
have no MPS kernel yet; `PYTORCH_ENABLE_MPS_FALLBACK=1` runs those on the CPU.

## What CI proves

The macOS job (`macos-15`, GitHub's Apple Silicon runner) installs the wheel in a clean environment and, through the
HTTP API: transcribes a sample with Moonshine on the CPU and through Core ML; installs llama-server's Metal build and
asks it for its devices; with `device = "auto"`, transcribes the sample with whisper.cpp (`whispercpp-tiny`) and the MLX
models (`whisper-tiny-mlx`, `parakeet-tdt-ctc-110m-mlx`), speaks a sentence with `kokoro-v1.0-mlx` and transcribes it
back; prints where each model ran (`GET /api/v1/ps`: device, runtime, notice); installs the torch plugin and separates a
two-speaker mix with SepFormer (`bolo-torch separate --device auto`), printing whether MPS was used; and runs
`bolo doctor`. GitHub's macOS runners are virtual machines: what they do not expose (the Neural Engine; Metal or MPS
where the virtual GPU lacks them) is reported by the job, and the model then runs on the CPU with a notice.
