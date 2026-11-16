#!/usr/bin/env python3
"""Copy a validation report (from ``validate_models.py``) into the catalog YAML.

Sets ``validated: cpu`` and the ``measured`` block for every model in the
report that ran successfully, and leaves everything else alone.

    python tools/apply_validation.py report.json [more-reports.json ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

CATALOG = Path(__file__).resolve().parent.parent / "src" / "bolo" / "catalog"
KEEP = (
    "rtf",
    "der",
    "der_collar_250ms",
    "wer",
    "round_trip_wer",
    "round_trip_stt",
    "ttfa_ms",
    "audio_s",
    "load_s",
    "memory_mb",
    "threads",
    "engine",
    "hardware",
    "date",
)


def main(paths: list[str]) -> int:
    report: dict[str, Any] = {}
    for p in paths:
        report.update(json.loads(Path(p).read_text(encoding="utf-8")))
    done = 0
    for path in sorted(CATALOG.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
        entries = yaml.safe_load(text)
        changed = False
        for entry in entries:
            result = report.get(entry["id"])
            if not result or not result.get("ok"):
                continue
            measured = {k: result[k] for k in KEEP if result.get(k) is not None}
            if result.get("source"):
                measured["test_audio"] = result["source"]
            if entry.get("validated") != "cpu" or entry.get("measured") != measured:
                rebuilt: dict[str, Any] = {}
                for key, value in entry.items():  # keep the field order: measured goes right after validated
                    if key == "measured":
                        continue
                    rebuilt[key] = "cpu" if key == "validated" else value
                    if key == "validated":
                        rebuilt["measured"] = measured
                entry.clear()
                entry.update(rebuilt)
                changed = True
                done += 1
        if changed:
            body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=110)
            path.write_text(header.rstrip() + "\n\n" + body, encoding="utf-8")
    failed = sorted(k for k, v in report.items() if not v.get("ok"))
    print(f"updated {done} entries; {len(failed)} models in the report did not validate: {', '.join(failed) or '-'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1:]))
