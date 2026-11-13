#!/usr/bin/env python3
"""Switch on the catalog entries of voices that have been published as release assets.

``export_syspin.py`` / ``export_mms.py`` / ``convert_whisper.py`` (run by the "Voices" workflow)
upload one archive per model plus an index, ``voices.json``, to a GitHub release, and
``publish_hf.py`` uploads the same files to one Hugging Face repository per model, recording the
commit in the index. The catalog already describes those models with ``status: pending`` and
without checksums. This script reads the index and, for every published model, points the entry
at the Hugging Face repository pinned to the uploaded commit, with the release archive (size and
sha256) as the fallback source (archive only when the model is not on the Hub), and removes
``status: pending``:

    python tools/voices/update_catalog.py                       # the index of the voices-v1 release
    python tools/voices/update_catalog.py --index dist/voices.json
    python tools/voices/update_catalog.py --index dist/voices.json --base-url http://127.0.0.1:8123/   # local test

Entries whose voice is not in the index stay pending. Run ``python server/tools/gen_docs.py``
afterwards to refresh the tables in the documentation.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = ROOT / "server" / "src" / "bolo" / "catalog"
DEFAULT_CATALOGS = [CATALOG_DIR / n for n in ("tts.yaml", "tts-mms.yaml", "stt-faster-whisper.yaml")]
RELEASE_URL = "https://github.com/cb-Sigma/bolo/releases/download/{tag}/"


def update_file(path: Path, published: dict[str, Any], base: str) -> tuple[list[str], list[str]]:
    text = path.read_text(encoding="utf-8")
    header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
    entries = yaml.safe_load(text) or []
    done, pending = [], []
    for entry in entries:
        voice = published.get(entry["id"])
        if voice is None:
            if entry.get("status") == "pending":
                pending.append(entry["id"])
            continue
        archive = {
            "path": voice["file"], "url": base + voice["file"], "size": voice["size"], "sha256": voice["sha256"],
            "extract": True,
        }  # fmt: skip
        hub = voice.get("hf")
        if hub:
            entry["source"] = {"repo": hub["repo"], "revision": hub["revision"], "include": list(hub["files"]),
                               "fallback": [archive]}  # fmt: skip
        else:
            entry["source"] = {"urls": [archive]}
        entry["size_bytes"] = int(voice.get("unpacked_size") or voice["size"])
        if voice.get("sample_rate"):
            entry["sample_rate"] = int(voice["sample_rate"])
        entry.pop("status", None)
        done.append(entry["id"])
    if done:
        body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=110)
        path.write_text(header.rstrip() + "\n\n" + body, encoding="utf-8")
    return done, pending


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="voices-v1")
    parser.add_argument("--index", help="voices.json: a file or a URL (default: the one attached to the release)")
    parser.add_argument("--base-url", help="where the archives are served from (default: the release)")
    parser.add_argument("--catalog", action="append", help="catalog YAML file(s) to update (default: the voice files)")
    args = parser.parse_args()
    base = args.base_url or RELEASE_URL.format(tag=args.tag)
    if not base.endswith("/"):
        base += "/"
    source = args.index or base + "voices.json"
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response:
            index: dict[str, Any] = json.load(response)
    else:
        index = json.loads(Path(source).read_text(encoding="utf-8"))
    published = {voice["id"]: voice for voice in index.get("voices", [])}
    done, pending = [], []
    for path in [Path(p) for p in (args.catalog or DEFAULT_CATALOGS)]:
        d, p = update_file(path, published, base)
        done += d
        pending += p
    print(f"published: {', '.join(done) or '-'}")
    print(f"still pending: {', '.join(pending) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
