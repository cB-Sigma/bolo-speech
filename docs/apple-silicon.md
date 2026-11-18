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

## The torch plugin on MPS

bolo passes its own device to the plugin: `coreml` (what `auto` becomes on a Mac) means MPS there. When torch sees no
MPS device (a virtual Mac, a CPU-only torch), the plugin runs on the CPU with a warning instead of failing. Asked for by
name (`BOLO_TORCH_DEVICE=mps`, or a model's `device` option), a missing MPS device is an error. Some PyTorch operations
have no MPS kernel yet; `PYTORCH_ENABLE_MPS_FALLBACK=1` runs those on the CPU.

## What CI proves

The macOS job (`macos-15`, GitHub's Apple Silicon runner) installs the wheel in a clean environment and, through the
HTTP API, transcribes a sample with Moonshine on the CPU and through Core ML, installs llama-server's Metal build and
asks it for its devices, and runs `bolo doctor`. GitHub's macOS runners are virtual machines: whatever they do not
expose (the Neural Engine, MPS where the virtual GPU lacks it) is reported by the job rather than assumed.
