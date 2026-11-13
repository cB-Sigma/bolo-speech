#!/usr/bin/env python3
"""Regenerate the parts of the documentation that are derived from code and data.

* ``docs/openapi.json`` - the OpenAPI schema of the running application
* the route table in ``docs/api.md``
* the model tables in ``docs/models.md`` and ``README.md``

Generated blocks sit between ``<!-- BEGIN GENERATED: name -->`` and
``<!-- END GENERATED: name -->`` markers; everything outside them is
hand-written and left untouched. Run from anywhere: ``python tools/gen_docs.py``.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "server" / "src"))

from bolo.api import create_app  # noqa: E402
from bolo.catalog import Catalog, ModelSpec  # noqa: E402
from bolo.config import Settings  # noqa: E402

TASK_TITLES = {
    "stt": "Speech to text",
    "tts": "Text to speech",
    "vad": "Voice-activity detection",
    "diarization": "Speaker diarization",
    "turn": "End-of-turn detection",
}


def replace_block(path: Path, name: str, content: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"(<!-- BEGIN GENERATED: {re.escape(name)} -->\n).*?(<!-- END GENERATED: {re.escape(name)} -->)", re.S
    )
    if not pattern.search(text):
        raise SystemExit(f"{path}: no generated block named {name!r}")
    path.write_text(pattern.sub(lambda m: m.group(1) + content.rstrip() + "\n" + m.group(2), text), encoding="utf-8")


def size(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def langs(spec: ModelSpec, limit: int = 6) -> str:
    if not spec.languages:
        return "-"
    if len(spec.languages) > limit:
        return ", ".join(spec.languages[:limit]) + f" +{len(spec.languages) - limit}"
    return ", ".join(spec.languages)


def measured(spec: ModelSpec) -> str:
    if spec.status == "pending":
        return "not yet published"
    if spec.validated != "cpu":
        return "untested"
    m = spec.measured
    bits = [f"RTF {m['rtf']:.3f}"]
    if "ttfa_ms" in m:
        bits.append(f"first audio {m['ttfa_ms']} ms")
    if "der" in m:
        bits.append(f"DER {100 * m['der']:.1f}% on a two-voice test clip")
    if "wer" in m:
        bits.append(f"WER {100 * m['wer']:.1f}%")
    if "cer" in m:
        bits.append(f"CER {100 * m['cer']:.1f}%")
    if "round_trip_wer" in m:
        bits.append(f"round-trip WER {100 * m['round_trip_wer']:.1f}%")
    if "round_trip_cer" in m:
        bits.append(f"round-trip CER {100 * m['round_trip_cer']:.1f}%")
    return "**CPU** · " + " · ".join(bits)


def licence(spec: ModelSpec) -> str:
    text = spec.license.replace("|", "/")
    if len(text) > 60:
        text = text[:57] + "…"
    extra = ""
    if spec.needs_phonemes:
        from bolo.text.phonemize import uccharan_language

        extra = " · phonemes: uccharan" if uccharan_language(*spec.phoneme_target()) else " · needs espeak-ng"
    if spec.plugin:
        extra += f" · needs plugin {spec.plugin}"
    if spec.gated:
        extra += " · gated"
    return text + ("" if spec.commercial_use else " ⚠") + extra


def models_table(specs: list[ModelSpec]) -> str:
    out: list[str] = []
    by_task: dict[str, dict[str, list[ModelSpec]]] = defaultdict(lambda: defaultdict(list))
    for s in specs:
        by_task[s.task][s.family].append(s)
    pending = sum(s.status == "pending" for s in specs)
    total = len(specs) - pending
    valid = sum(s.validated == "cpu" for s in specs)
    out.append(
        f"{total} models: {valid} validated on CPU by the maintainers, {total - valid} untested."
        + (
            f' {pending} more are described but not yet published (see "Voices converted by this project").'
            if pending
            else ""
        )
        + "\n"
    )
    for task in ("stt", "tts", "vad", "diarization", "turn"):
        out.append(f"### {TASK_TITLES[task]}\n")
        for family, items in by_task[task].items():
            out.append(f"#### {family}\n")
            out.append("| id | languages | tier | streaming | download | licence | validated |")
            out.append("| --- | --- | --- | --- | --- | --- | --- |")
            for s in items:
                out.append(
                    f"| `{s.id}` | {langs(s)} | {s.tier} | {'native' if s.streaming else ('-' if task != 'stt' else 'pseudo')} "
                    f"| {size(s.size_bytes)} | {licence(s)} | {measured(s)} |"
                )
            out.append("")
    out.append("⚠ = the licence restricts commercial use. RTF = processing time / audio duration (lower is faster).")
    return "\n".join(out)


def summary_table(specs: list[ModelSpec]) -> str:
    specs = [s for s in specs if s.status != "pending"]
    rows: dict[tuple[str, str], list[ModelSpec]] = defaultdict(list)
    for s in specs:
        rows[(s.task, s.family)].append(s)
    out = [
        "| Task | Family | Models | Languages | Validated on CPU | Best measured RTF |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for (task, family), items in rows.items():
        all_langs = sorted({lang for s in items for lang in s.languages})
        lang_text = (
            ", ".join(all_langs[:8]) + (f" +{len(all_langs) - 8}" if len(all_langs) > 8 else "") if all_langs else "-"
        )
        valid = [s for s in items if s.validated == "cpu"]
        best = min((s.measured["rtf"] for s in valid), default=None)
        best_id = next((s.id for s in valid if s.measured["rtf"] == best), "")
        out.append(
            f"| {task.upper()} | {family} | {len(items)} | {lang_text} | {len(valid)} | "
            + (f"{best:.3f} (`{best_id}`)" if best is not None else "-")
            + " |"
        )
    total, valid_n = len(specs), sum(s.validated == "cpu" for s in specs)
    out.append(
        f"\n**{total} models in the catalog, {valid_n} validated on CPU, {total - valid_n} untested.** Full list: [docs/models.md](docs/models.md)."
    )
    return "\n".join(out)


def routes_table(schema: dict[str, Any]) -> str:
    groups: dict[str, list[str]] = defaultdict(list)
    for path, methods in schema["paths"].items():
        for method, op in methods.items():
            tag = (op.get("tags") or ["other"])[0]
            groups[tag].append(f"| `{method.upper()}` | `{path}` | {op.get('summary', '')} |")
    out: list[str] = []
    for tag in ("openai", "speaches", "native", "operations"):
        if tag not in groups:
            continue
        out += [f"**{tag}**\n", "| Method | Path | Summary |", "| --- | --- | --- |", *groups[tag], ""]
    out.append("WebSockets (not part of OpenAPI): `WS /api/v1/stt/stream`, `WS /api/v1/tts/stream`.")
    return "\n".join(out)


def main() -> None:
    specs = Catalog.load(None).all()
    app = create_app(Settings(data_dir=Path("/nonexistent"), ui=False))
    schema = app.openapi()
    (ROOT / "docs" / "openapi.json").write_text(
        json.dumps(schema, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    replace_block(ROOT / "docs" / "api.md", "routes", routes_table(schema))
    replace_block(ROOT / "docs" / "models.md", "models", models_table(specs))
    replace_block(ROOT / "README.md", "models-summary", summary_table(specs))
    print(f"wrote docs/openapi.json ({len(schema['paths'])} paths) and refreshed tables for {len(specs)} models")


if __name__ == "__main__":
    main()
