#!/usr/bin/env python3
"""Convert the IISc SYSPIN VITS voices to the layout sherpa-onnx (and so bolo) loads, and pack them.

The SYSPIN project (SPIRE Lab, Indian Institute of Science) publishes single-speaker Coqui VITS
voices for Indian languages on Hugging Face (``SYSPIN/vits_<Language>_<Gender>``; weights MIT,
trained on the SYSPIN corpus, CC-BY-4.0). They are PyTorch checkpoints. bolo runs without
PyTorch, so each voice is converted once, here, into:

    model.onnx    the VITS graph, with the metadata sherpa-onnx reads
    tokens.txt    the character table ("<symbol> <id>" per line)
    voice.json    provenance: source repository and revision, licence, language, sample rate, characters
    LICENSE, ATTRIBUTION

and packed as ``<id>.tar.gz``. ``SHA256SUMS`` and ``voices.json`` (the index that
``update_catalog.py`` reads) are written next to the archives.

Run it yourself (needs about 4 GB of disk per voice while it works; CPU is enough):

    python -m venv venv && . venv/bin/activate
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -r tools/voices/requirements-syspin.txt
    python tools/voices/export_syspin.py --out dist                    # every voice in syspin.json
    python tools/voices/export_syspin.py --out dist --only syspin-mai-female,syspin-bho-female

Already have exported directories (model.onnx + tokens.txt [+ config.json])? Pack and check them
without PyTorch:

    python tools/voices/export_syspin.py --out dist --exported syspin-mai-female=/path/to/dir

The voices read characters from a closed set: anything else in the text (digits, Latin letters in
a Devanagari voice) has no symbol. ``voice.json`` lists the set so that callers can check.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import shutil
import sys
import tarfile
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPECIAL = ("<PAD>", "<EOS>", "<BOS>", "<BLNK>")

LICENSE_TEXT = """MIT License

Copyright (c) SYSPIN project, SPIRE Lab, Indian Institute of Science (IISc), Bengaluru

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def attribution(voice: dict[str, Any]) -> str:
    return (
        f"{voice['id']}: {voice['language_name']} {voice['gender']} voice\n\n"
        f"Model: https://huggingface.co/{voice['repo']} (revision {voice['revision']}),\n"
        "by the SYSPIN project, SPIRE Lab, Indian Institute of Science (IISc), Bengaluru.\n"
        "The model weights are published under the MIT licence (see LICENSE).\n\n"
        "Training data: the SYSPIN text-to-speech corpus (IISc), licensed CC-BY-4.0\n"
        "(https://creativecommons.org/licenses/by/4.0/).\n\n"
        "This archive is a format conversion (Coqui VITS checkpoint to ONNX) made with\n"
        "tools/voices/export_syspin.py of the bolo project (https://github.com/cb-Sigma/bolo).\n"
        "The weights were not retrained or otherwise changed.\n"
    )


def download(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    print(f"  downloading {url}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "bolo-voices"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    tmp.replace(dest)


def write_tokens(char_to_id: dict[str, int], path: Path) -> list[str]:
    """Write tokens.txt; returns the characters of the voice.

    Control characters (one SYSPIN voice has a TAB in its table) cannot be written as a
    "<symbol> <id>" line and are never wanted in text, so they are dropped. An upper-case twin is
    added for cased letters whose upper-case form is not a symbol of its own, because the original
    front end lower-cases its input and sherpa-onnx does not.
    """
    uppers = collections.Counter(symbol.upper() for symbol in char_to_id)
    lines: list[str] = []
    characters: list[str] = []
    for symbol, index in char_to_id.items():
        if symbol not in SPECIAL and (len(symbol) != 1 or (unicodedata.category(symbol) == "Cc")):
            print(f"  dropping symbol {symbol!r} (id {index}) from tokens.txt")
            continue
        lines.append(f"{symbol} {index}")
        if symbol in SPECIAL:
            continue
        characters.append(symbol)
        upper = symbol.upper()
        if upper != symbol.lower() and len(upper) == 1 and uppers[upper] == 1 and upper not in char_to_id:
            lines.append(f"{upper} {index}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return characters


def export(voice: dict[str, Any], work: Path) -> dict[str, Any]:
    """Checkpoint -> model.onnx + tokens.txt in ``work``. Needs torch, coqui-tts and onnx."""
    import onnx
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import Vits

    base = f"https://huggingface.co/{voice['repo']}/resolve/{voice['revision']}"
    download(f"{base}/config.json", work / "config.json")
    download(f"{base}/best_model.pth", work / "best_model.pth")
    config = VitsConfig()
    config.load_json(str(work / "config.json"))
    vits = Vits.init_from_config(config)
    if vits.config.use_phonemes:
        raise SystemExit(f"{voice['id']}: this checkpoint uses phonemes; only character voices are handled here")
    vits.load_checkpoint(config, str(work / "best_model.pth"))
    model_path = work / "model.onnx"
    vits.export_onnx(output_path=str(model_path), verbose=False)
    chars = vits.tokenizer.characters
    meta = {
        "model_type": "vits", "comment": "coqui", "language": voice["language_name"], "frontend": "characters",
        "add_blank": int(vits.config.add_blank), "blank_id": chars.blank_id,
        "n_speakers": int(getattr(vits, "num_speakers", 0) or 0), "use_eos_bos": int(vits.tokenizer.use_eos_bos),
        "bos_id": chars.bos_id, "eos_id": chars.eos_id, "pad_id": chars.pad_id,
        "sample_rate": int(vits.ap.sample_rate),
    }  # fmt: skip
    model = onnx.load(str(model_path))
    del model.metadata_props[:]
    for key, value in meta.items():
        prop = model.metadata_props.add()
        prop.key, prop.value = key, str(value)
    onnx.save(model, str(model_path))
    (work / "char_to_id.json").write_text(json.dumps(chars._char_to_id, ensure_ascii=False), encoding="utf-8")
    (work / "best_model.pth").unlink()  # 950 MB that is not needed any more
    return {"sample_rate": meta["sample_rate"]}


def read_token_table(path: Path) -> dict[str, int]:
    table: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        symbol, _, index = line.rpartition(" ")
        if index.strip().isdigit():
            table.setdefault(symbol if symbol else " ", int(index))
    return table


def smoke_test(directory: Path, text: str, characters: list[str]) -> dict[str, Any]:
    """Load the converted voice with sherpa-onnx and say one sentence."""
    import numpy as np
    import sherpa_onnx

    allowed = set(characters)
    spoken = "".join(ch for ch in text if ch in allowed or ch.lower() in allowed)
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(directory / "model.onnx"), tokens=str(directory / "tokens.txt")
            ),
            num_threads=2,
        )
    )
    if not config.validate():
        raise SystemExit(f"{directory.name}: sherpa-onnx rejected the converted voice")
    audio = sherpa_onnx.OfflineTts(config).generate(spoken, sid=0, speed=1.0)
    samples = np.asarray(audio.samples, dtype=np.float32)
    seconds = len(samples) / audio.sample_rate
    peak = float(np.abs(samples).max()) if len(samples) else 0.0
    if seconds < 0.5 or peak < 0.02:
        raise SystemExit(
            f"{directory.name}: the smoke test produced no audible speech ({seconds:.2f} s, peak {peak:.3f})"
        )
    return {"text": spoken, "seconds": round(seconds, 2), "sample_rate": int(audio.sample_rate), "peak": round(peak, 3)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(voice: dict[str, Any], directory: Path, out: Path, licence: dict[str, str], smoke: bool) -> dict[str, Any]:
    """tokens.txt, voice.json, LICENSE, ATTRIBUTION, smoke test and ``<id>.tar.gz``. Returns the index entry."""
    staged = out / "work" / (voice["id"] + "-pack")
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir(parents=True)
    shutil.copy2(directory / "model.onnx", staged / "model.onnx")
    table_file = directory / "char_to_id.json"
    table = (
        json.loads(table_file.read_text(encoding="utf-8"))
        if table_file.is_file()
        else read_token_table(directory / "tokens.txt")
    )
    characters = write_tokens(table, staged / "tokens.txt")
    result = smoke_test(staged, voice["sample_text"], characters) if smoke else None
    info = {
        "id": voice["id"],
        "language": voice["language"],
        "language_name": voice["language_name"],
        "gender": voice["gender"],
        "speakers": 1,
        "sample_rate": result["sample_rate"] if result else 22050,
        "architecture": "VITS (Coqui), character input, no phonemiser",
        "cleaner": "multilingual",  # Coqui multilingual_cleaners ran on the training text; bolo applies the same
        "source": {
            "repo": voice["repo"],
            "revision": voice["revision"],
            "url": f"https://huggingface.co/{voice['repo']}",
        },
        "license": licence,
        "characters": "".join(sorted(characters)),
        "has_digits": sorted(ch for ch in characters if ch.isdigit()),
        "smoke_test": result,
    }
    (staged / "voice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (staged / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")
    (staged / "ATTRIBUTION").write_text(attribution(voice), encoding="utf-8")
    archive = out / f"{voice['id']}.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for name in ("model.onnx", "tokens.txt", "voice.json", "LICENSE", "ATTRIBUTION"):
            member = tar.gettarinfo(str(staged / name), arcname=f"{voice['id']}/{name}")
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            member.mtime = 0  # the same input gives the same archive
            member.mode = 0o644
            with (staged / name).open("rb") as handle:
                tar.addfile(member, handle)
    entry = {
        **{k: info[k] for k in ("id", "language", "language_name", "gender", "sample_rate", "source", "license")},
        "file": archive.name,
        "size": archive.stat().st_size,
        "sha256": sha256(archive),
        "unpacked_size": sum(
            (staged / n).stat().st_size for n in ("model.onnx", "tokens.txt", "voice.json", "LICENSE", "ATTRIBUTION")
        ),
        "characters": len(characters),
        "smoke_test": result,
    }
    shutil.rmtree(staged)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory for the archives, SHA256SUMS and voices.json")
    parser.add_argument("--only", default="", help="comma-separated voice ids (default: all in syspin.json)")
    parser.add_argument("--exported", action="append", default=[], metavar="ID=DIR",
                        help="use an already converted directory for this voice instead of exporting (repeatable)")  # fmt: skip
    parser.add_argument("--no-smoke", action="store_true", help="skip the sherpa-onnx smoke synthesis")
    parser.add_argument("--voices", default=str(HERE / "syspin.json"))
    args = parser.parse_args()
    spec = json.loads(Path(args.voices).read_text(encoding="utf-8"))
    exported = dict(item.split("=", 1) for item in args.exported)
    wanted = [v.strip() for v in args.only.split(",") if v.strip()] or (list(exported) if exported else [])
    voices = [v for v in spec["voices"] if not wanted or v["id"] in wanted]
    unknown = set(wanted) - {v["id"] for v in voices}
    if unknown:
        print(f"unknown voice ids: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "voices.json"
    index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    index.update(release=spec["release"], license=spec["license"])
    entries: dict[str, Any] = {e["id"]: e for e in index.get("voices", [])}
    for voice in voices:
        print(f"== {voice['id']} ({voice['repo']} @ {voice['revision'][:12]})", flush=True)
        if voice["id"] in exported:
            directory = Path(exported[voice["id"]])
        else:
            directory = out / "work" / voice["id"]
            directory.mkdir(parents=True, exist_ok=True)
            export(voice, directory)
        entries[voice["id"]] = pack(voice, directory, out, spec["license"], not args.no_smoke)
        print(
            f"   {entries[voice['id']]['file']}  {entries[voice['id']]['size']} bytes  sha256 {entries[voice['id']]['sha256']}"
        )
        if voice["id"] not in exported:
            shutil.rmtree(directory, ignore_errors=True)
        index["voices"] = [entries[k] for k in sorted(entries)]
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        sums = io.StringIO()
        for key in sorted(entries):
            sums.write(f"{entries[key]['sha256']}  {entries[key]['file']}\n")
        (out / "SHA256SUMS").write_text(sums.getvalue(), encoding="utf-8")
    shutil.rmtree(out / "work", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
