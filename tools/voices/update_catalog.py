#!/usr/bin/env python3
"""Switch on the catalog entries of voices that have been published as release assets.

``export_syspin.py`` (run by the "Voices" workflow) uploads one archive per voice plus an index,
``voices.json``, to a GitHub release. The catalog (server/src/bolo/catalog/tts.yaml) already
describes those voices with ``status: pending`` and without checksums. This script reads the
index, writes size and sha256 into the matching entries and removes ``status: pending``:

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
DEFAULT_CATALOG = ROOT / "server" / "src" / "bolo" / "catalog" / "tts.yaml"
RELEASE_URL = "https://github.com/cb-Sigma/bolo/releases/download/{tag}/"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="voices-v1")
    parser.add_argument("--index", help="voices.json: a file or a URL (default: the one attached to the release)")
    parser.add_argument("--base-url", help="where the archives are served from (default: the release)")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
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
    path = Path(args.catalog)
    text = path.read_text(encoding="utf-8")
    header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
    entries = yaml.safe_load(text)
    done, pending = [], []
    for entry in entries:
        voice = published.get(entry["id"])
        if voice is None:
            if entry.get("status") == "pending":
                pending.append(entry["id"])
            continue
        entry["source"] = {
            "urls": [
                {
                    "path": voice["file"],
                    "url": base + voice["file"],
                    "size": voice["size"],
                    "sha256": voice["sha256"],
                    "extract": True,
                }
            ]
        }
        entry["size_bytes"] = int(voice.get("unpacked_size") or voice["size"])
        entry["sample_rate"] = int(voice.get("sample_rate") or entry.get("sample_rate", 22050))
        entry.pop("status", None)
        done.append(entry["id"])
    if done:
        body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=110)
        path.write_text(header.rstrip() + "\n\n" + body, encoding="utf-8")
    print(f"published: {', '.join(done) or '-'}")
    print(f"still pending: {', '.join(pending) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
