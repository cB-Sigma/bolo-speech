#!/usr/bin/env python3
"""Upload converted models to Hugging Face, one public model repository each, with a model card.

Run by the "Voices" workflow after the release assets are published, when the repository secret
``HF_TOKEN`` exists (without it the step is skipped). For every entry of ``<dist>/voices.json`` it
unpacks the archive, writes ``README.md`` (licence, attribution, provenance, the pinned source
revision and the round-trip numbers measured by the maintainers, when there are any), creates
``<org>/<repo>`` if it does not exist and uploads the folder. The commit of each upload is written
back into ``voices.json`` as ``"hf": {"repo", "revision"}``; ``update_catalog.py`` then makes the
catalog pull from the Hub (pinned to that commit) with the release archive as the fallback.

    HF_TOKEN=... python tools/voices/publish_hf.py --dist dist [--org codebucketsolutions] [--only id,id]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_ORG = "codebucketsolutions"
#: licence names in voices.json -> Hugging Face licence identifiers
HF_LICENSES = {"MIT": "mit", "CC-BY-NC-4.0": "cc-by-nc-4.0", "CC-BY-4.0": "cc-by-4.0", "Apache-2.0": "apache-2.0"}


def repo_name(entry: dict[str, Any]) -> str:
    """``syspin-mai-female`` -> ``bolo-syspin-maithili-female``; ``mms-tts-pan`` -> ``bolo-mms-tts-punjabi``."""
    if entry.get("hf_name"):
        return str(entry["hf_name"])
    family = "syspin" if entry["id"].startswith("syspin-") else "-".join(entry["id"].split("-")[:2])
    parts = ["bolo", family, entry["language_name"].lower()]
    if entry.get("gender"):
        parts.append(entry["gender"])
    return re.sub(r"[^a-z0-9-]+", "-", "-".join(parts)).strip("-")


def weights_license(entry: dict[str, Any]) -> str:
    lic = entry.get("license") or {}
    name = str(lic.get("weights") if isinstance(lic, dict) else lic)
    return HF_LICENSES.get(name, "other")


def model_card(entry: dict[str, Any], attribution: str, measured: dict[str, Any] | None) -> str:
    lic = entry.get("license") or {}
    task = entry.get("task", "tts")
    tags = [
        "bolo",
        "sherpa-onnx",
        "onnx",
        *(["text-to-speech", "tts", "vits"] if task == "tts" else ["speech-recognition", "whisper", "ctranslate2"]),
    ]
    front = [
        "---",
        f"license: {weights_license(entry)}",
        f"language:\n- {entry['language']}",
        f"pipeline_tag: {'text-to-speech' if task == 'tts' else 'automatic-speech-recognition'}",
        "tags:",
        *[f"- {t}" for t in tags],
        "---",
    ]
    src = entry["source"]
    lines = [
        *front,
        "",
        f"# {entry['id']}",
        "",
        f"{entry['language_name']} {'voice' if task == 'tts' else 'speech recogniser'} converted for "
        "[bolo](https://github.com/cb-Sigma/bolo), a speech server that runs without PyTorch "
        f"({'sherpa-onnx' if task == 'tts' else 'faster-whisper / CTranslate2'}).",
        "",
        "## Use it",
        "",
        "```bash",
        f"bolo pull {entry['id']}",
        "```",
        "",
        "## Licence",
        "",
        f"Weights: **{lic.get('weights', lic) if isinstance(lic, dict) else lic}**"
        + (" - non-commercial use only." if isinstance(lic, dict) and lic.get("commercial_use") is False else "."),
        *([f"Training data: {lic['data']}."] if isinstance(lic, dict) and lic.get("data") else []),
        "The full licence text is in `LICENSE`.",
        "",
        "## Attribution and provenance",
        "",
        "```",
        attribution.strip(),
        "```",
        "",
        f"Source: https://huggingface.co/{src['repo']}"
        + (f"/tree/{src['revision']}/{src['path']}" if src.get("path") else f" (revision `{src['revision']}`)"),
        "",
        "This repository holds a format conversion only; the weights were not retrained.",
    ]
    if measured:
        lines += [
            "",
            "## Measured by the bolo maintainers",
            "",
            "| | |",
            "| --- | --- |",
            *[f"| {k} | {v} |" for k, v in measured.items()],
        ]
    return "\n".join(lines) + "\n"


def publish(
    dist: Path, org: str, api: Any, only: list[str] | None = None, measured_file: Path | None = None
) -> dict[str, Any]:
    index_path = dist / "voices.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    measured_all: dict[str, Any] = {}
    if measured_file and measured_file.is_file():
        measured_all = json.loads(measured_file.read_text(encoding="utf-8"))
    for entry in index.get("voices", []):
        if only and entry["id"] not in only:
            continue
        archive = dist / entry["file"]
        if not archive.is_file():
            print(f"  {entry['id']}: {archive.name} is not in {dist}; skipped", file=sys.stderr)
            continue
        repo_id = f"{org}/{repo_name(entry)}"
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(archive) as tar:
                tar.extractall(tmp, filter="data")
            folder = Path(tmp) / entry["id"]
            files = sorted(str(f.relative_to(folder)) for f in folder.rglob("*") if f.is_file())
            attribution = (
                (folder / "ATTRIBUTION").read_text(encoding="utf-8") if (folder / "ATTRIBUTION").is_file() else ""
            )
            (folder / "README.md").write_text(
                model_card(entry, attribution, measured_all.get(entry["id"])), encoding="utf-8"
            )
            api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
            commit = api.upload_folder(
                repo_id=repo_id, repo_type="model", folder_path=str(folder),
                commit_message=f"{entry['id']} from bolo voices release {index.get('release', '')}".strip(),
            )  # fmt: skip
        revision = getattr(commit, "oid", None) or str(commit)
        entry["hf"] = {"repo": repo_id, "revision": revision, "files": files}
        print(f"  {entry['id']} -> https://huggingface.co/{repo_id} @ {revision}")
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dist", required=True)
    parser.add_argument("--org", default=DEFAULT_ORG)
    parser.add_argument("--only", default="")
    parser.add_argument("--measured", default=str(HERE / "measured.json"))
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set: nothing uploaded to Hugging Face")
        return 0
    from huggingface_hub import HfApi

    only = [x.strip() for x in args.only.split(",") if x.strip()] or None
    publish(Path(args.dist), args.org, HfApi(token=token), only, Path(args.measured))
    shutil.rmtree(Path(args.dist) / "work", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
