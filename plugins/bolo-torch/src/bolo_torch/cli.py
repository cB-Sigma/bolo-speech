"""``bolo-torch``: serve the plugin (what bolo runs), check the environment, or separate a recording directly.

bolo-torch serve                       # the plugin protocol on stdin/stdout (bolo starts this)
bolo-torch doctor                      # torch build, devices, which model families are installed
bolo-torch separate mix.wav -o out/    # speech separation without bolo (SepFormer / PixIT)
bolo-torch code list                   # inference code from GitHub repositories, pinned by commit and sha256
bolo-torch code install step-audio-2   # download one bundle now (a load also installs it when missing)
bolo-torch pin-remote-code DIR --revision COMMIT   # the remote_code pin of a downloaded model folder
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from bolo_torch import __version__

#: separation models for ``bolo-torch separate`` (the same files as bolo's catalog entries)
SEPARATION_MODELS: dict[str, dict[str, Any]] = {
    "sepformer-whamr": {
        "kind": "sepformer", "repo": "speechbrain/sepformer-whamr",
        "revision": "9950fd8055b38679d2af5bef2366b5dba600ec66",
        "allow": ["hyperparams.yaml", "masknet.ckpt", "encoder.ckpt", "decoder.ckpt"],
        "about": "SpeechBrain SepFormer, WHAMR! (noisy, reverberant), 8 kHz, 2 speakers. Apache-2.0; WSJ0-derived data",
    },
    "sepformer-whamr16k": {
        "kind": "sepformer", "repo": "speechbrain/sepformer-whamr16k",
        "revision": "21a5b500c6f52fddc387c5d9e5fb13ffd6f039c5",
        "allow": ["hyperparams.yaml", "masknet.ckpt", "encoder.ckpt", "decoder.ckpt"],
        "about": "SpeechBrain SepFormer, WHAMR! at 16 kHz, 2 speakers. Apache-2.0; WSJ0-derived data",
    },
    "pixit-ami": {
        "kind": "pixit", "repo": "pyannote/speech-separation-ami-1.0",
        "revision": "9486b106945ae0cc0784041a08bfcdba5edadfb9", "allow": None,
        "about": "pyannote PixIT, joint diarization and separation, AMI meetings, 16 kHz. MIT, gated",
    },
}  # fmt: skip

FAMILIES = {
    "parler": ["parler_tts", "transformers"],
    "f5": ["transformers", "vocos", "x_transformers", "torchdiffeq", "pydub", "jieba", "pypinyin"],
    "orpheus": ["transformers", "snac"],
    "sepformer": ["speechbrain", "torchaudio"],
    "pyannote": ["pyannote.audio"],
    "omni": ["transformers", "accelerate"],
    "step-audio-2": ["transformers", "s3tokenizer", "hyperpyyaml", "onnxruntime", "librosa", "torchaudio"],
    "glm-4-voice": ["transformers", "tiktoken", "hyperpyyaml", "diffusers", "conformer", "lightning", "hydra"],
}


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def doctor() -> dict[str, Any]:
    import torch

    from bolo_torch.runtime import hf_token, pick_device

    mps = getattr(torch.backends, "mps", None)
    return {
        "bolo_torch": __version__,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "mps_available": bool(mps is not None and mps.is_available()),
        "auto_device": pick_device("auto"),
        "threads": torch.get_num_threads(),
        "hf_token": bool(hf_token()),
        "families": {name: all(_installed(m) for m in mods) for name, mods in FAMILIES.items()},
    }


def separate(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    from bolo.engines.base import LoadOptions, SeparationModel

    from bolo_torch.engine import TorchEngine
    from bolo_torch.models.separation import SR, resample
    from bolo_torch.runtime import hub_snapshot

    entry = SEPARATION_MODELS[args.model]
    t0 = time.perf_counter()
    root = hub_snapshot(entry["repo"], entry["revision"], entry["allow"])
    t_download = time.perf_counter() - t0
    t0 = time.perf_counter()
    model = TorchEngine().load(
        entry["kind"],
        "separation",
        root,
        {"model_id": args.model},
        LoadOptions(threads=args.threads, device=args.device),
    )
    t_load = time.perf_counter() - t0
    assert isinstance(model, SeparationModel)
    samples, rate = sf.read(args.input, dtype="float32", always_2d=True)
    mono = resample(samples.mean(axis=1).astype(np.float32), int(rate), SR)
    t0 = time.perf_counter()
    tracks = model.separate(mono)
    t_run = time.perf_counter() - t0
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for i, track in enumerate(tracks):
        path = out / f"SPEAKER_{i:02d}.wav"
        sf.write(str(path), track, SR)
        files.append(str(path))
    duration = len(mono) / SR
    turns = getattr(model, "last_turns", [])
    return {
        "model": args.model, "device": args.device, "threads": args.threads, "audio_s": round(duration, 2),
        "download_s": round(t_download, 2), "load_s": round(t_load, 2), "separate_s": round(t_run, 2),
        "rtf": round(t_run / duration, 3) if duration else None, "tracks": files,
        "turns": [[round(s, 2), round(e, 2), k] for s, e, k in turns],
    }  # fmt: skip


def code_command(args: argparse.Namespace) -> int:
    from bolo_torch import upstream
    from bolo_torch.runtime import PluginError

    if args.action == "list":
        print(json.dumps(upstream.describe(), indent=2))
        return 0
    names = [args.name] if args.name else sorted(upstream.bundles())
    status = 0
    for name in names:
        try:
            bundle = upstream.get(name)
            if args.action == "install":
                print(f"{name}: {bundle.licences()}", file=sys.stderr)
                path = upstream.install(bundle, force=args.force)
                print(f"{name}: installed in {path}")
            else:
                problems = upstream.verify(bundle)
                print(f"{name}: " + ("matches its pins" if not problems else "; ".join(problems)))
                status = status or (1 if problems else 0)
        except PluginError as exc:
            print(f"error ({exc.code}): {exc}", file=sys.stderr)
            status = 2
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bolo-torch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"bolo-torch {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="serve the plugin protocol on stdin/stdout")
    sub.add_parser("doctor", help="report torch, devices and installed model families")
    sep = sub.add_parser("separate", help="split a recording into one track per speaker")
    sep.add_argument("input")
    sep.add_argument("-o", "--output", default="separated")
    sep.add_argument("-m", "--model", default="sepformer-whamr16k", choices=sorted(SEPARATION_MODELS))
    sep.add_argument("--device", default=os.environ.get("BOLO_TORCH_DEVICE", "auto"))
    sep.add_argument("--threads", type=int, default=0)
    code = sub.add_parser("code", help="inference code from GitHub repositories (pinned by commit and sha256)")
    code.add_argument("action", choices=["list", "install", "verify"])
    code.add_argument("name", nargs="?")
    code.add_argument("--force", action="store_true", help="install again even when present")
    pin = sub.add_parser("pin-remote-code", help="print the remote_code pin of a downloaded model folder")
    pin.add_argument("folder")
    pin.add_argument("--revision", required=True, help="the commit the folder was downloaded at")
    args = parser.parse_args(argv)
    if args.command in (None, "serve"):
        logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(name)s: %(message)s")
        from bolo_torch.engine import TorchEngine
        from bolo_torch.server import serve

        return serve(TorchEngine())
    if args.command == "doctor":
        print(json.dumps(doctor(), indent=2))
        return 0
    if args.command == "code":
        return code_command(args)
    if args.command == "pin-remote-code":
        from bolo_torch.remote_code import pins

        print(json.dumps({"remote_code": pins(Path(args.folder), args.revision)}, indent=2))
        print(
            "Read every listed file before you put this pin in a catalog entry: the code runs in the plugin's process.",
            file=sys.stderr,
        )
        return 0
    if args.command == "separate":
        from bolo_torch.runtime import PluginError

        try:
            print(json.dumps(separate(args), indent=2, ensure_ascii=False))
        except PluginError as exc:
            print(f"error ({exc.code}): {exc}", file=sys.stderr)
            return 2
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
