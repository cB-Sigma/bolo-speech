#!/usr/bin/env python3
"""Convert Whisper fine-tunes (Hugging Face transformers checkpoints) to CTranslate2 int8 for faster-whisper.

bolo runs Whisper through faster-whisper (CTranslate2) without PyTorch. Fine-tunes published only as
transformers checkpoints are converted once, here, with CTranslate2's own converter (which needs
transformers and PyTorch at conversion time only), and packed as ``<id>.tar.gz`` with ``LICENSE``,
``ATTRIBUTION`` and ``model.json`` (provenance). ``SHA256SUMS`` and ``voices.json`` (the index that
``update_catalog.py`` and ``publish_hf.py`` read) are written next to the archives.

    pip install "torch==2.8.*" --index-url https://download.pytorch.org/whl/cpu
    pip install -r tools/voices/requirements-whisper.txt
    python tools/voices/convert_whisper.py --out dist [--only faster-whisper-small-bhojpuri]
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from export_syspin import download, sha256  # noqa: E402

LICENSE_URLS = {"CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/legalcode.txt"}


def attribution(model: dict[str, Any]) -> str:
    return (
        f"{model['id']}: {model['language_name']} speech recognition (Whisper fine-tune)\n\n"
        f"Model: https://huggingface.co/{model['repo']} (revision {model['revision']}), by {model['author']}.\n"
        f"Base model: {model['base']}. Training data: {model['data']}.\n"
        f"Licence: {model['license']} ({model['license_url']}); full text in LICENSE.\n\n"
        "Changes: format conversion only (transformers checkpoint to CTranslate2, int8 weights), made with\n"
        "tools/voices/convert_whisper.py of the bolo project (https://github.com/cB-Sigma/bolo-speech).\n"
        "The weights were not retrained.\n"
    )


def fix_tokenizer_config(path: Path) -> None:
    """Checkpoints saved by transformers 5 list ``extra_special_tokens``; transformers 4 (what CTranslate2's
    converter supports) expects a mapping. The tokens are Whisper's standard ones, already in tokenizer.json."""
    if not path.is_file():
        return
    config = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(config.get("extra_special_tokens"), list):
        del config["extra_special_tokens"]
        path.write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")


def convert(model: dict[str, Any], work: Path) -> Path:
    src = work / "src"
    src.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}"
    for name in model["files"]:
        download(f"{base}/{name}", src / name)
    fix_tokenizer_config(src / "tokenizer_config.json")
    out = work / model["id"]
    shutil.rmtree(out, ignore_errors=True)
    converter = shutil.which("ct2-transformers-converter") or "ct2-transformers-converter"
    subprocess.run(
        [converter, "--model", str(src), "--output_dir", str(out), "--quantization", "int8",
         "--copy_files", "tokenizer.json"],
        check=True,
    )  # fmt: skip
    for name in model.get("copy", []):
        if (src / name).is_file():
            shutil.copy2(src / name, out / name)
    shutil.rmtree(src)
    return out


def smoke_test(directory: Path, language: str) -> dict[str, Any]:
    import numpy as np
    from faster_whisper import WhisperModel

    model = WhisperModel(str(directory), device="cpu", compute_type="int8", cpu_threads=2)
    segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language=language, beam_size=1)
    list(segments)
    return {"loaded": True}


def pack(
    model: dict[str, Any], directory: Path, out: Path, licence: str, smoke: dict[str, Any] | None
) -> dict[str, Any]:
    info = {
        "id": model["id"], "language": model["language"], "language_name": model["language_name"],
        "architecture": "Whisper (CTranslate2, int8)", "decode_language": model["decode_language"],
        "source": {"repo": model["repo"], "revision": model["revision"], "url": f"https://huggingface.co/{model['repo']}"},
        "license": {"weights": model["license"], "data": model["data"], "credit": model["author"]},
    }  # fmt: skip
    (directory / "model.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (directory / "LICENSE").write_text(licence, encoding="utf-8")
    (directory / "ATTRIBUTION").write_text(attribution(model), encoding="utf-8")
    archive = out / f"{model['id']}.tar.gz"
    names = sorted(p.name for p in directory.iterdir() if p.is_file())
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for name in names:
            member = tar.gettarinfo(str(directory / name), arcname=f"{model['id']}/{name}")
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            member.mtime = 0
            member.mode = 0o644
            with (directory / name).open("rb") as handle:
                tar.addfile(member, handle)
    return {
        **{k: info[k] for k in ("id", "language", "language_name", "source", "license")},
        "task": "stt", "file": archive.name, "size": archive.stat().st_size, "sha256": sha256(archive),
        "unpacked_size": sum((directory / n).stat().st_size for n in names), "smoke_test": smoke,
    }  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--keep", action="store_true", help="keep the converted directory under <out>/work")
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--models", default=str(HERE / "whisper.json"))
    args = parser.parse_args()
    spec = json.loads(Path(args.models).read_text(encoding="utf-8"))
    wanted = [v.strip() for v in args.only.split(",") if v.strip()]
    models = [m for m in spec["voices"] if not wanted or m["id"] in wanted]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "voices.json"
    index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    index["release"] = spec["release"]
    entries = {e["id"]: e for e in index.get("voices", [])}
    for model in models:
        print(f"== {model['id']} ({model['repo']} @ {model['revision'][:12]})", flush=True)
        work = out / "work"
        directory = convert(model, work)
        licence_file = work / f"{model['license']}.txt"
        download(LICENSE_URLS[model["license"]], licence_file)
        smoke = None if args.no_smoke else smoke_test(directory, model["decode_language"])
        entries[model["id"]] = pack(model, directory, out, licence_file.read_text(encoding="utf-8"), smoke)
        print(f"   {entries[model['id']]['file']}  {entries[model['id']]['size']} bytes", flush=True)
        if not args.keep:
            shutil.rmtree(directory)
        index["voices"] = [entries[k] for k in sorted(entries)]
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        sums = io.StringIO()
        for key in sorted(entries):
            sums.write(f"{entries[key]['sha256']}  {entries[key]['file']}\n")
        (out / "SHA256SUMS").write_text(sums.getvalue(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
