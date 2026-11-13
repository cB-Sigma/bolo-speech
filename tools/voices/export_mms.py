#!/usr/bin/env python3
"""Convert Meta MMS-TTS voices (``facebook/mms-tts``) to the layout sherpa-onnx (and so bolo) loads, and pack them.

Meta's Massively Multilingual Speech project publishes one single-speaker VITS voice per language
as a PyTorch checkpoint (``models/<iso>/G_100000.pth`` in ``facebook/mms-tts``), licensed
**CC-BY-NC-4.0**: non-commercial use only, with attribution. The licence allows redistributing the
weights in another format for non-commercial purposes; each archive therefore carries the
licence, the attribution and a note that the weights were only converted.

The conversion follows the recipe sherpa-onnx used for its own ``vits-mms-*`` exports: the
original VITS inference code (from the ``mms-meta/MMS`` space, pinned), ``torch.onnx.export``
of ``infer()``, and the metadata sherpa-onnx reads (``comment: mms``, ``add_blank``,
``frontend: characters``). Voices whose training text was romanised (uroman) are refused: they
would need a romaniser at synthesis time.

    python -m venv venv && . venv/bin/activate
    pip install "torch==2.8.*" --index-url https://download.pytorch.org/whl/cpu
    pip install -r tools/voices/requirements-mms.txt
    python tools/voices/export_mms.py --out dist                  # every voice in mms.json
    python tools/voices/export_mms.py --out dist --only mms-tts-pan,mms-tts-ory

About 150 MB of download and 400 MB of scratch disk per voice, under a minute on a laptop CPU.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import shutil
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from export_syspin import download, sha256, smoke_test  # noqa: E402

VITS_FILES = ("models.py", "commons.py", "modules.py", "attentions.py", "transforms.py", "LICENSE")
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt"


def attribution(voice: dict[str, Any], spec: dict[str, Any]) -> str:
    return (
        f"{voice['id']}: {voice['language_name']} voice from Meta's Massively Multilingual Speech (MMS) project\n\n"
        f"Model: https://huggingface.co/{spec['repo']}/tree/{spec['revision']}/models/{voice['code']}\n"
        'Authors: Vineel Pratap, Andros Tjandra, Bowen Shi, et al. (Meta AI), "Scaling Speech Technology to\n'
        '1,000+ Languages", 2023, https://arxiv.org/abs/2305.13516\n\n'
        "Licence: Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC-4.0),\n"
        "https://creativecommons.org/licenses/by-nc/4.0/ (full text in LICENSE). NON-COMMERCIAL USE ONLY.\n\n"
        "Changes: format conversion only (PyTorch checkpoint to ONNX with the original VITS inference code,\n"
        f"{spec['code_repo']} at revision {spec['code_revision']}), made with tools/voices/export_mms.py of the\n"
        "bolo project (https://github.com/cb-Sigma/bolo). The weights were not retrained or otherwise changed.\n"
        "Inference code: VITS, MIT licence, Copyright (c) 2021 Jaehyeon Kim.\n"
    )


def fetch_code(spec: dict[str, Any], work: Path) -> Path:
    code = work / "vits"
    code.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/spaces/{spec['code_repo']}/resolve/{spec['code_revision']}/vits"
    for name in VITS_FILES:
        download(f"{base}/{name}", code / name)
    return code


def export(voice: dict[str, Any], spec: dict[str, Any], work: Path, code: Path) -> dict[str, Any]:
    """Checkpoint -> model.onnx + tokens.txt in ``work``. Needs torch and onnx."""
    import onnx
    import torch

    base = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/models/{voice['code']}"
    for name in ("config.json", "vocab.txt", "G_100000.pth"):
        download(f"{base}/{name}", work / name)
    hps = json.loads((work / "config.json").read_text(encoding="utf-8"))
    if hps["data"]["training_files"].split(".")[-1] == "uroman":
        raise SystemExit(f"{voice['id']}: trained on romanised text (uroman); not supported")
    # Inference never aligns; the training-only Cython extension is replaced by an empty module.
    sys.modules.setdefault("monotonic_align", types.ModuleType("monotonic_align"))
    sys.path.insert(0, str(code))
    import models  # type: ignore[import-not-found]

    symbols = [line.rstrip("\n") for line in (work / "vocab.txt").open(encoding="utf-8")]
    net = models.SynthesizerTrn(
        len(symbols),
        hps["data"]["filter_length"] // 2 + 1,
        hps["train"]["segment_size"] // hps["data"]["hop_length"],
        **hps["model"],
    )
    checkpoint = torch.load(work / "G_100000.pth", map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    missing, _ = net.load_state_dict(state, strict=False)
    if [k for k in missing if not k.startswith("enc_q")]:
        raise SystemExit(f"{voice['id']}: checkpoint does not match the model: {missing[:5]}")
    net.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, m: Any) -> None:
            super().__init__()
            self.m = m

        def forward(self, x: Any, x_length: Any, noise_scale: Any, length_scale: Any, noise_scale_w: Any) -> Any:
            return self.m.infer(
                x, x_length, noise_scale=noise_scale, length_scale=length_scale, noise_scale_w=noise_scale_w
            )[0]

    x = torch.randint(low=1, high=10, size=(1, 50), dtype=torch.int64)
    args = (x, torch.tensor([50]), torch.tensor([0.667]), torch.tensor([1.0]), torch.tensor([0.8]))
    model_path = work / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            Wrapper(net), args, str(model_path), opset_version=13, dynamo=False,
            input_names=["x", "x_length", "noise_scale", "length_scale", "noise_scale_w"], output_names=["y"],
            dynamic_axes={"x": {0: "N", 1: "L"}, "x_length": {0: "N"}, "y": {0: "N", 2: "L"}},
        )  # fmt: skip
    meta = {
        "model_type": "vits", "comment": "mms", "url": f"https://huggingface.co/{spec['repo']}",
        "add_blank": int(hps["data"]["add_blank"]), "language": voice["language_name"], "frontend": "characters",
        "n_speakers": int(hps["data"]["n_speakers"]), "sample_rate": int(hps["data"]["sampling_rate"]),
    }  # fmt: skip
    model = onnx.load(str(model_path))
    for key, value in meta.items():
        prop = model.metadata_props.add()
        prop.key, prop.value = key, str(value)
    onnx.save(model, str(model_path))
    (work / "G_100000.pth").unlink()
    return {"symbols": symbols, "sample_rate": meta["sample_rate"]}


def write_tokens(symbols: list[str], path: Path) -> list[str]:
    """tokens.txt as sherpa-onnx's MMS exports have it (upper-case twins: the MMS front end lower-cases)."""
    uppers = collections.Counter(s.upper() for s in symbols)
    lines: list[str] = []
    for index, token in enumerate(symbols):
        lines.append(f"{token} {index}")
        if token.lower() != token.upper() and len(token.upper()) == 1 and uppers[token.upper()] == 1:
            lines.append(f"{token.upper()} {index}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [s for s in symbols if len(s) == 1 and s not in ("_",)]


def pack(voice: dict[str, Any], spec: dict[str, Any], work: Path, out: Path, symbols: list[str], licence: str,
         smoke: bool) -> dict[str, Any]:  # fmt: skip
    characters = write_tokens(symbols, work / "tokens.txt")
    result = smoke_test(work, voice["sample_text"], characters) if smoke else None
    info = {
        "id": voice["id"], "language": voice["language"], "language_name": voice["language_name"], "speakers": 1,
        "sample_rate": result["sample_rate"] if result else 16000,
        "architecture": "VITS (MMS), character input, no phonemiser",
        "cleaner": "lowercase",  # the MMS front end lower-cases; bolo applies the same
        "source": {"repo": spec["repo"], "revision": spec["revision"], "path": f"models/{voice['code']}",
                   "url": f"https://huggingface.co/{spec['repo']}"},
        "license": {"weights": "CC-BY-NC-4.0", "commercial_use": False, "credit": "Meta AI (MMS project)"},
        "characters": "".join(sorted(c for c in characters if c.strip())),
        "smoke_test": result,
    }  # fmt: skip
    (work / "voice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (work / "LICENSE").write_text(licence, encoding="utf-8")
    (work / "ATTRIBUTION").write_text(attribution(voice, spec), encoding="utf-8")
    names = ("model.onnx", "tokens.txt", "voice.json", "LICENSE", "ATTRIBUTION")
    archive = out / f"{voice['id']}.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for name in names:
            member = tar.gettarinfo(str(work / name), arcname=f"{voice['id']}/{name}")
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            member.mtime = 0
            member.mode = 0o644
            with (work / name).open("rb") as handle:
                tar.addfile(member, handle)
    return {
        **{k: info[k] for k in ("id", "language", "language_name", "sample_rate", "source", "license")},
        "file": archive.name, "size": archive.stat().st_size, "sha256": sha256(archive),
        "unpacked_size": sum((work / n).stat().st_size for n in names), "characters": len(characters),
        "smoke_test": result,
    }  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default="", help="comma-separated voice ids (default: all in mms.json)")
    parser.add_argument("--keep", action="store_true", help="keep the unpacked directories under <out>/work")
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--voices", default=str(HERE / "mms.json"))
    args = parser.parse_args()
    spec = json.loads(Path(args.voices).read_text(encoding="utf-8"))
    wanted = [v.strip() for v in args.only.split(",") if v.strip()]
    voices = [v for v in spec["voices"] if not wanted or v["id"] in wanted]
    if set(wanted) - {v["id"] for v in voices}:
        print(f"unknown voice ids: {sorted(set(wanted) - {v['id'] for v in voices})}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    code = fetch_code(spec, out / "work")
    licence_file = out / "work" / "CC-BY-NC-4.0.txt"
    download(LICENSE_URL, licence_file)
    licence = licence_file.read_text(encoding="utf-8")
    index_path = out / "voices.json"
    index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    index.update(release=spec["release"])
    entries: dict[str, Any] = {e["id"]: e for e in index.get("voices", [])}
    for voice in voices:
        print(f"== {voice['id']} ({spec['repo']}/models/{voice['code']} @ {spec['revision'][:12]})", flush=True)
        work = out / "work" / voice["id"]
        work.mkdir(parents=True, exist_ok=True)
        exported = export(voice, spec, work, code)
        entries[voice["id"]] = pack(voice, spec, work, out, exported["symbols"], licence, not args.no_smoke)
        e = entries[voice["id"]]
        print(f"   {e['file']}  {e['size']} bytes  sha256 {e['sha256']}  smoke {e['smoke_test']}", flush=True)
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        index["voices"] = [entries[k] for k in sorted(entries)]
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        sums = io.StringIO()
        for key in sorted(entries):
            sums.write(f"{entries[key]['sha256']}  {entries[key]['file']}\n")
        (out / "SHA256SUMS").write_text(sums.getvalue(), encoding="utf-8")
    if not args.keep:
        shutil.rmtree(out / "work", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
