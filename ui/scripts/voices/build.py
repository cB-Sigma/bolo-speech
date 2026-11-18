#!/usr/bin/env python3
"""Prepare the voices that the live demo speaks with, in the browser.

The demo runs speech synthesis in the page with ONNX Runtime Web. GitHub release assets are sent
without CORS headers, so the browser cannot fetch voices from there; instead the Pages workflow
runs this script and puts the result next to the demo (same origin, under ``voices/``).

For every voice in voices.json it:

1. gets the converted voice: SYSPIN voices from the archives that tools/voices/export_syspin.py
   published as release assets (``voices-v1``), Piper voices from Hugging Face at a pinned
   revision (checked against a SHA-256);
2. shrinks the ONNX graph (shrink.py: 8-bit convolution weights outside the text encoder, 16-bit
   elsewhere; the browser still computes in fp32) and checks the result against the original:
   same length, log-mel distance on the loudest frames below a threshold;
3. for voices marked ``"halves": true``, cuts the shrunk graph in two at a layer boundary (the server's
   bolo/onnx_split.py) for the demo's split placement, front half on WebGPU and back half on the CPU,
   and checks that the halves together give the whole graph's output;
4. writes ``<id>/model.onnx``, ``<id>/voice.json`` (what the browser front end needs: symbol
   table, blank handling, sample rate, inference scales), the licence and attribution files, and
   an ``index.json`` with sizes and SHA-256 of everything.

Needs Python 3.10+, ``onnx``, ``onnxruntime`` and ``numpy`` (no PyTorch):

    python -m pip install onnx onnxruntime numpy
    python ui/scripts/voices/build.py --out ui/dist-demo/voices
    python ui/scripts/voices/build.py --out /tmp/voices --local syspin-mai-female=/path/to/export_dir

``--local ID=DIR`` takes a directory with model.onnx + tokens.txt (an unpacked archive, or the
output of the export) instead of downloading. ``--skip-missing`` leaves out voices whose source is
not published yet (the index then lists only the others) instead of failing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shrink import shrink  # noqa: E402

# The vendored copy of uccharan (text to phonemes for Piper voices) whose data file is published with them.
UCCHARAN = HERE.parents[1] / "src" / "vendor" / "uccharan"
# uccharan's transliteration packs (romanised text to native script) for the text playground: CC BY-SA 4.0,
# from the uccharan release that matches the vendored copy, checked against these SHA-256 sums.
XLIT_RELEASE = "https://github.com/cB-Sigma/uccharan/releases/download/v0.6.3/"
XLIT = {
    "xlit-hi.ucd": "59e5dde0354e4d4c55530baedc3034d5ed7eff0a382056a1d269d012b0c9ca52",
    "xlit-mr.ucd": "623f1a488324d18819d84177f21f3a900bd47025904fb80d5f7dbda2efb4bec8",
    "xlit-bn.ucd": "0273a468778c752d8a070852abfb39cd0cd0a61b2148a58c20626a06e99fded1",
    "xlit-te.ucd": "c6d63469c0184646d0ff4f12198313f409a536f2fbf269a127648ad488e59c83",
    "xlit-gu.ucd": "4f5c4e575c3d465643b8a3ef2ade0b4b1206736d156914f99213558c9cfd0e39",
    "xlit-LICENSE-CC-BY-SA-4.0.txt": "28a9529c7d0bb4dc51f4bf5c116a3d16ef247a052f7591466768ddf563fd1cf5",
    "xlit-NOTICE.txt": "414e5f6105b9d8cefec1be2866f4f16f16f6ae2f7188d3abac6fd77b0ea8bc4b",
}
# Parts of each architecture whose weights stay 16-bit (see shrink.py for why).
KEEP = {"coqui": ("text_encoder", "duration_predictor"), "piper": ("enc_p", "dp", "dec")}
# Mean |Δ log10 mel| over the loudest frames, shrunk vs original, with the noise switched off.
MAX_DISTANCE = 0.08
# Output frame of the VITS decoder, in samples, and how many frames the shrunk voice may differ by.
HOP = 256
MAX_HOPS = 1

PIPER_ATTRIBUTION = """{id}: Piper voice "{name}" ({quality}), {language}

Model: {url}
from the rhasspy/piper-voices repository (MIT licence), revision {revision}.
Training data and its licence: see MODEL_CARD in this directory.

Weights shrunk for the browser by ui/scripts/voices/build.py of the bolo project
(https://github.com/cB-Sigma/bolo-speech): 8-bit convolution weights outside the text
encoder, 16-bit elsewhere. Not retrained.
"""

SYSPIN_NOTE = """
Weights shrunk for the browser by ui/scripts/voices/build.py of the bolo project:
8-bit convolution weights outside the text encoder, 16-bit elsewhere. Not retrained.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path) -> None:
    print(f"  fetching {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "bolo-voices"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
            tmp.replace(dest)
            return
        except urllib.error.HTTPError as err:
            if err.code == 404:
                raise
            if attempt == 2:
                raise
        except OSError:
            if attempt == 2:
                raise
        time.sleep(5 * (attempt + 1))


def read_tokens(path: Path) -> dict[str, int]:
    """tokens.txt as written by export_syspin.py: "<symbol> <id>" per line (the symbol may be a space)."""
    table: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        symbol, _, index = line.rpartition(" ")
        if index.isdigit():
            table.setdefault(symbol if symbol else " ", int(index))
    return table


def get_syspin(voice: dict[str, Any], release: str, local: dict[str, str], work: Path) -> Path:
    if voice["id"] in local:
        return Path(local[voice["id"]])
    archive = work / f"{voice['id']}.tar.gz"
    fetch(f"{release}{voice['id']}.tar.gz", archive)
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter="data")
    archive.unlink()
    return work / voice["id"]


def get_piper(voice: dict[str, Any], local: dict[str, str], work: Path) -> Path:
    directory = work / voice["id"]
    directory.mkdir(parents=True, exist_ok=True)
    for name, spec in voice["files"].items():
        dest = directory / name
        if voice["id"] in local:
            shutil.copy2(Path(local[voice["id"]]) / name, dest)
        else:
            fetch(voice["base"] + spec["from"], dest)
        if sha256(dest) != spec["sha256"]:
            raise SystemExit(f"{voice['id']}: {name} does not match the pinned SHA-256")
    return directory


# Where the SYSPIN checkpoints come from (repository and pinned revision): tools/voices/syspin.json.
SYSPIN_SPEC = HERE.parents[2] / "tools" / "voices" / "syspin.json"
SHERPA_DEFAULTS = {"noise_scale": 0.667, "length_scale": 1.0, "noise_scale_w": 0.8}


def syspin_inference(voice_id: str) -> dict[str, Any]:
    """The checkpoint's own inference settings (Coqui config.json, model_args), not sherpa-onnx's defaults.

    Coqui synthesises SYSPIN voices with inference_noise_scale 0.667 and inference_noise_scale_dp 1.0
    (the stochastic duration predictor's noise); sherpa-onnx uses 0.8 for the latter, which gives a flatter,
    more regular rhythm. Falls back to sherpa-onnx's values (and says so) when the config cannot be read.
    """
    try:
        spec = {v["id"]: v for v in json.loads(SYSPIN_SPEC.read_text(encoding="utf-8"))["voices"]}[voice_id]
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/config.json"
        request = urllib.request.Request(url, headers={"User-Agent": "bolo-voices"})
        with urllib.request.urlopen(request, timeout=60) as response:
            cfg = json.loads(response.read().decode("utf-8"))
        args = cfg.get("model_args", {})
        return {
            "noise_scale": float(args.get("inference_noise_scale", 0.667)),
            "length_scale": float(args.get("length_scale", 1.0)),
            "noise_scale_w": float(args.get("inference_noise_scale_dp", 1.0)),
            "source": f"{spec['repo']}@{spec['revision'][:12]} config.json",
        }
    except (OSError, KeyError, ValueError) as err:
        print(f"   {voice_id}: checkpoint settings not readable ({err}); using sherpa-onnx defaults")
        return {**SHERPA_DEFAULTS, "source": "sherpa-onnx defaults"}


def coqui_config(directory: Path, session_meta: dict[str, str]) -> dict[str, Any]:
    tokens = read_tokens(directory / "tokens.txt")
    meta = session_meta
    if meta.get("frontend") != "characters" or meta.get("comment") != "coqui":
        raise SystemExit(f"{directory}: not a character-input Coqui VITS export")
    info = json.loads((directory / "voice.json").read_text(encoding="utf-8")) if (directory / "voice.json").is_file() else {}
    special = {"<PAD>", "<EOS>", "<BOS>", "<BLNK>"}
    return {
        "kind": "coqui",
        "sample_rate": int(meta["sample_rate"]),
        "add_blank": meta["add_blank"] == "1",
        "blank_id": int(meta["blank_id"]),
        "use_eos_bos": meta["use_eos_bos"] == "1",
        "bos_id": int(meta["bos_id"]),
        "eos_id": int(meta["eos_id"]),
        # Coqui's multilingual_cleaners ran on the training text: NFC, lower case, ; : → , and - → space,
        # brackets and double quotes removed, whitespace collapsed. The browser applies the same.
        "cleaner": "multilingual",
        "inference": dict(SHERPA_DEFAULTS),  # replaced by the checkpoint's own settings in build()
        "symbols": {k: v for k, v in tokens.items() if k not in special},
        "source": info.get("source"),
        "license": info.get("license"),
    }


def piper_config(directory: Path) -> dict[str, Any]:
    cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if cfg.get("phoneme_type", "espeak") != "espeak" or cfg.get("num_speakers", 1) != 1:
        raise SystemExit(f"{directory}: only single-speaker espeak-style Piper voices are handled")
    inference = cfg.get("inference", {})
    return {
        "kind": "piper",
        "sample_rate": int(cfg["audio"]["sample_rate"]),
        "phoneme_language": cfg["espeak"]["voice"],
        "phoneme_id_map": cfg["phoneme_id_map"],
        "inference": {
            "noise_scale": inference.get("noise_scale", 0.667),
            "length_scale": inference.get("length_scale", 1.0),
            "noise_scale_w": inference.get("noise_w", 0.8),
        },
    }


def coqui_ids(cfg: dict[str, Any], text: str) -> list[int]:
    import re
    import unicodedata

    text = unicodedata.normalize("NFC", text).lower().replace(";", ",").replace("-", " ").replace(":", ",")
    text = re.sub(r"\s+", " ", re.sub(r'[<>()\[\]"]+', "", text)).strip()
    ids = [cfg["symbols"][c] for c in text if c in cfg["symbols"]]
    if cfg["add_blank"]:
        spread = [cfg["blank_id"]] * (2 * len(ids) + 1)
        spread[1::2] = ids
        ids = spread
    return [cfg["bos_id"], *ids, cfg["eos_id"]] if cfg["use_eos_bos"] else ids


def piper_ids(cfg: dict[str, Any], phonemes: str) -> list[int]:
    m = cfg["phoneme_id_map"]
    ids = list(m["^"]) + list(m["_"])
    for ch in phonemes:
        if ch in m:
            ids += list(m[ch]) + list(m["_"])
    return ids + list(m["$"])


def logmel(x: np.ndarray, sr: int) -> np.ndarray:
    n, hop, bands = 1024, 256, 80
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (n // 2, n // 2)), n)[::hop] * np.hanning(n)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.linspace(0, sr / 2, power.shape[1])

    def mel(f: np.ndarray | float) -> np.ndarray | float:
        return 2595 * np.log10(1 + np.asarray(f) / 700)

    points = 700 * (10 ** (np.linspace(0, mel(sr / 2), bands + 2) / 2595) - 1)
    bank = np.zeros((bands, len(freqs)))
    for i in range(bands):
        lo, mid, hi = points[i], points[i + 1], points[i + 2]
        bank[i] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return np.log10(power @ bank.T + 1e-6)


def check(original: Path, shrunk: Path, ids: list[int], sr: int) -> dict[str, Any]:
    import onnxruntime as ort

    feed = {
        "input": np.array([ids], dtype=np.int64),
        "input_lengths": np.array([len(ids)], dtype=np.int64),
        "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32),  # no noise: the two must agree
    }
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    a = ort.InferenceSession(str(original), options).run(None, feed)[0].reshape(-1)
    b = ort.InferenceSession(str(shrunk), options).run(None, feed)[0].reshape(-1)
    # VITS rounds each symbol's predicted duration up to whole frames of 256 samples. A weight that moves by
    # a rounding error can tip one symbol over a frame boundary: one frame (11.6 ms at 22.05 kHz) longer or
    # shorter, well under what a listener notices in a segment's length (a few tens of ms). That much is
    # allowed; anything more is a changed voice. The frames are then compared along the best alignment.
    hops = abs(len(a) - len(b)) // HOP
    if len(a) % HOP or len(b) % HOP or hops > MAX_HOPS:
        raise SystemExit(f"{shrunk}: the shrunk voice changed the timing ({len(a)} vs {len(b)} samples)")
    ma, mb = logmel(a, sr), logmel(b, sr)
    loud = ma.max(axis=1) > ma.max() - 4
    distance = aligned_distance(ma, mb, loud)
    if distance > MAX_DISTANCE:
        raise SystemExit(f"{shrunk}: log-mel distance {distance:.3f} is above {MAX_DISTANCE}")
    return {"seconds": round(len(a) / sr, 2), "logmel_distance": round(distance, 4), "frames_off": hops}


def cut_in_two(model: Path, dest: Path, ids: list[int]) -> dict[str, Any]:
    """Cut the voice's graph in two at a layer boundary with the server's bolo/onnx_split.py, for the demo's
    split placement (front half on WebGPU, back half on the CPU), and check that the halves run together give
    the whole graph's output. Writes ``front.onnx`` and ``back.onnx`` next to ``model.onnx``."""
    import importlib.util

    import onnx
    import onnxruntime as ort

    path = HERE.parents[2] / "server" / "src" / "bolo" / "onnx_split.py"
    spec = importlib.util.spec_from_file_location("onnx_split", path)
    assert spec and spec.loader
    split = importlib.util.module_from_spec(spec)
    sys.modules["onnx_split"] = split  # its dataclasses look themselves up there
    spec.loader.exec_module(split)
    whole = onnx.load(str(model))
    points = split.cut_points(whole)
    weights = max(p.front_bytes + p.back_bytes for p in points)
    cut = split.choose_cut(points, front_budget=weights // 2)
    if cut is None:
        raise SystemExit(f"{model}: no layer boundary to cut at")
    front, back = split.build_halves(whole, cut)
    onnx.save(front, str(dest / "front.onnx"))
    onnx.save(back, str(dest / "back.onnx"))
    feed = {
        "input": np.array([ids], dtype=np.int64),
        "input_lengths": np.array([len(ids)], dtype=np.int64),
        "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    a = ort.InferenceSession(str(model), options).run(None, feed)[0].reshape(-1)
    fs = ort.InferenceSession(str(dest / "front.onnx"), options)
    bs = ort.InferenceSession(str(dest / "back.onnx"), options)
    fin = {i.name for i in fs.get_inputs()}
    mid = dict(zip([o.name for o in fs.get_outputs()], fs.run(None, {k: v for k, v in feed.items() if k in fin})))
    back_feed = {i.name: mid[i.name] if i.name in mid else feed[i.name] for i in bs.get_inputs()}
    out_name = whole.graph.output[0].name
    b = (mid[out_name] if out_name in mid else bs.run([out_name], back_feed)[0]).reshape(-1)
    if len(a) != len(b) or float(np.abs(a - b).max()) > 1e-3:
        raise SystemExit(f"{model}: the two halves do not give the whole graph's output")
    print(f"   cut at node {cut.index}: {cut.front_bytes:,} + {cut.back_bytes:,} weight bytes, {len(cut.crossing)} crossing")
    return {"index": cut.index, "front_bytes": cut.front_bytes, "back_bytes": cut.back_bytes, "crossing": len(cut.crossing)}


def aligned_distance(ma: np.ndarray, mb: np.ndarray, loud: np.ndarray, band: int = 4) -> float:
    """Mean |Δ log10 mel| over the loud frames of `ma`, along a dynamic time warping path kept within `band`
    frames of the diagonal (with equal lengths and no shift this is the plain frame-by-frame mean)."""
    n, m = len(ma), len(mb)
    if n == m:
        return float(np.abs(ma[loud] - mb[loud]).mean())
    cost = np.abs(ma[:, None, :] - mb[None, :, :]).mean(axis=2)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(m, i + band)
        for j in range(lo, hi + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    i, j, used = n, m, {}
    while i > 0 and j > 0:
        used.setdefault(i - 1, cost[i - 1, j - 1])
        step = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
    rows = [k for k in range(n) if loud[k] and k in used]
    return float(np.mean([used[k] for k in rows]))


def build(voice: dict[str, Any], spec: dict[str, Any], out: Path, local: dict[str, str], work: Path) -> dict[str, Any]:
    import onnxruntime as ort

    print(f"== {voice['id']}", flush=True)
    src = get_syspin(voice, spec["syspin_release"], local, work) if voice["source"] == "syspin" else get_piper(voice, local, work)
    dest = out / voice["id"]
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    original = src / "model.onnx"
    if voice["kind"] == "coqui":
        meta = ort.InferenceSession(str(original), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
        cfg = coqui_config(src, meta)
        if voice["source"] == "syspin":
            cfg["inference"] = syspin_inference(voice["id"])
        ids = coqui_ids(cfg, voice["sample"])
        shutil.copy2(src / "LICENSE", dest / "LICENSE")
        (dest / "ATTRIBUTION").write_text((src / "ATTRIBUTION").read_text(encoding="utf-8") + SYSPIN_NOTE, encoding="utf-8")
    else:
        cfg = piper_config(src)
        ids = piper_ids(cfg, voice["sample_ipa"])
        shutil.copy2(src / "MODEL_CARD", dest / "MODEL_CARD")
        revision = voice["base"].split("/resolve/")[1].split("/")[0]
        name, quality = voice["id"].rsplit("-", 2)[-2:]
        (dest / "ATTRIBUTION").write_text(
            PIPER_ATTRIBUTION.format(
                id=voice["id"], name=name, quality=quality, language=cfg["phoneme_language"],
                url=voice["base"] + voice["files"]["model.onnx"]["from"], revision=revision,
            ),
            encoding="utf-8",
        )  # fmt: skip
    counts = shrink(str(original), str(dest / "model.onnx"), KEEP[voice["kind"]])
    result = check(original, dest / "model.onnx", ids, cfg["sample_rate"])
    cfg = {"id": voice["id"], **cfg, "weights": {"int8": counts["int8"], "fp16": counts["fp16"]}}
    (dest / "voice.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    halves = cut_in_two(dest / "model.onnx", dest, ids) if voice.get("halves") else None
    files = sorted(p.name for p in dest.iterdir())
    entry = {
        "id": voice["id"],
        "files": {name: {"bytes": (dest / name).stat().st_size, "sha256": sha256(dest / name)} for name in files},
        "original_bytes": original.stat().st_size,
        "check": result,
    }
    if halves:
        entry["halves"] = halves
    # A download is either the whole graph or its two halves, never both: the size counts the whole one.
    entry["bytes"] = sum(f["bytes"] for n, f in entry["files"].items() if n not in ("front.onnx", "back.onnx"))
    if voice["kind"] == "piper":
        # Piper voices read phonemes; uccharan makes them in the page and needs its English data.
        data = UCCHARAN / "data" / "en-us.ucd"
        entry["shared"] = {"uccharan/en-us.ucd": {"bytes": data.stat().st_size, "sha256": sha256(data)}}
        entry["bytes"] += data.stat().st_size
    print(f"   {entry['original_bytes']:,} → {entry['bytes']:,} bytes, log-mel distance {result['logmel_distance']}")
    if src.is_relative_to(work):
        shutil.rmtree(src, ignore_errors=True)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory to write (served as ./voices/ next to the demo)")
    parser.add_argument("--spec", default=str(HERE / "voices.json"))
    parser.add_argument("--only", default="", help="comma-separated voice ids")
    parser.add_argument("--local", action="append", default=[], metavar="ID=DIR", help="use a local directory for this voice")
    parser.add_argument("--release", help="base URL of the SYSPIN archives (default: syspin_release in the spec)")
    parser.add_argument("--skip-missing", action="store_true", help="leave out voices whose source answers 404")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if args.release:
        spec["syspin_release"] = args.release.rstrip("/") + "/"
    local = dict(item.split("=", 1) for item in args.local)
    only = {v.strip() for v in args.only.split(",") if v.strip()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entries, missing = [], []
    with tempfile.TemporaryDirectory(prefix="bolo-voices-") as tmp:
        for voice in spec["voices"]:
            if only and voice["id"] not in only:
                continue
            try:
                entries.append(build(voice, spec, out, local, Path(tmp)))
            except urllib.error.HTTPError as err:
                if err.code == 404 and args.skip_missing:
                    print(f"   not published yet ({err.url}): left out")
                    missing.append(voice["id"])
                    continue
                raise
    # uccharan's language data (Piper and Kokoro voices, the text playground), next to the voices.
    shared = out / "uccharan"
    shared.mkdir(exist_ok=True)
    for data in sorted((UCCHARAN / "data").glob("*.ucd")):
        shutil.copy2(data, shared / data.name)
    for name in ("LICENSE", "NOTICE", "DATA_LICENSES.md"):
        shutil.copy2(UCCHARAN / name, shared / name)
    for name, digest in XLIT.items():
        dest = shared / name
        if not (dest.is_file() and sha256(dest) == digest):
            try:
                fetch(XLIT_RELEASE + name, dest)
            except OSError as err:  # optional: the playground falls back to uccharan's rules
                print(f"   {name}: not fetched ({err})")
                continue
            if sha256(dest) != digest:
                dest.unlink()
                raise SystemExit(f"{name} does not match the pinned SHA-256")
    index_path = out / "index.json"
    if only and index_path.is_file():  # a partial run keeps the other voices' entries
        previous = json.loads(index_path.read_text(encoding="utf-8"))
        done = {e["id"] for e in entries} | set(missing)
        entries = [e for e in previous.get("voices", []) if e["id"] not in done] + entries
        missing = [m for m in previous.get("missing", []) if m not in done] + missing
    order = [v["id"] for v in spec["voices"]]
    entries.sort(key=lambda e: order.index(e["id"]) if e["id"] in order else len(order))
    index = {"format": 1, "voices": entries, "missing": missing}
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    total = sum(e["bytes"] for e in entries)
    print(f"{len(entries)} voices, {total / 1e6:.1f} MB in {out}" + (f"; missing: {', '.join(missing)}" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
