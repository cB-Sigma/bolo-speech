#!/usr/bin/env python3
"""Regenerate the parts of the documentation that are derived from code and data.

* ``docs/openapi.json`` - the OpenAPI schema of the running application
* the route table in ``docs/api.md``
* the model tables in ``docs/models.md`` and ``README.md``
* the example machines in ``docs/recommendations.md`` (what the recommender picks on each)

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
from bolo.hardware import Cpu, Gpu, Hardware, Memory  # noqa: E402
from bolo.providers import key_env  # noqa: E402
from bolo.recommend import (  # noqa: E402
    LEVEL_LABEL,
    Candidate,
    Query,
    Siblings,
    assess,
    heuristic_speed,
    rank,
    select,
)

TASK_TITLES = {
    "stt": "Speech to text",
    "tts": "Text to speech",
    "vad": "Voice-activity detection",
    "diarization": "Speaker diarization",
    "turn": "End-of-turn detection",
    "separation": "Overlapped-speech separation",
    "denoise": "Denoising",
    "lid": "Language identification",
    "speech-to-speech": "Speech to speech (Realtime speech-to-speech pipeline)",
    "audio-llm": "Audio-in language models (Realtime audio-llm pipeline)",
    "speaker-embedding": "Speaker embedding",
    "audio-tagging": "Audio tagging and sound events",
    "emotion": "Speech emotion",
    "decision": "Decisions (typed questions about a recording or a text)",
    "voice-conversion": "Voice conversion",
    "align": "Forced alignment (word timestamps)",
    "echo-cancellation": "Echo cancellation (voice agents)",
    "wake-word": "Wake words and keyword spotting",
    "watermark": "Watermarking",
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
    if not spec.runnable:
        return "not runnable yet"
    if spec.validated != "cpu":
        return "ran once here (shared CPU), not measured" if spec.measured else "untested"
    m = spec.measured
    bits = [f"RTF {m['rtf']:.3f}"]
    if "ttfa_ms" in m:
        bits.append(f"first audio {m['ttfa_ms']} ms")
    if "der" in m:
        where = "AMI meeting excerpts" if "AMI" in str(m.get("test_audio", "")) else "a two-voice test clip"
        bits.append(f"DER {100 * m['der']:.1f}% on {where}")
    if "wer" in m:
        bits.append(f"WER {100 * m['wer']:.1f}%")
    if "cer" in m:
        bits.append(f"CER {100 * m['cer']:.1f}%")
    if "round_trip_wer" in m:
        bits.append(f"round-trip WER {100 * m['round_trip_wer']:.1f}%")
    if "round_trip_cer" in m:
        bits.append(f"round-trip CER {100 * m['round_trip_cer']:.1f}%")
    if "speaker_similarity" in m:
        bits.append(f"cloned voice: speaker similarity {m['speaker_similarity']:.2f}")
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
    if spec.engine == "tts-server":
        gpu = (spec.options.get("hardware") or {}).get("gpu")
        extra += f" · runs on {spec.options.get('runtime')}" + (f" · GPU {gpu}" if gpu and gpu != "optional" else "")
    if spec.cloud:
        name = str(spec.options.get("provider") or "")
        extra += f" · [terms]({spec.license_url}) · needs provider `{name}` (key: `{key_env(name)}`)"
    if spec.gated:
        extra += " · gated"
    return text + ("" if spec.commercial_use else " ⚠") + extra


def group(spec: ModelSpec) -> str:
    """The heading a model is listed under: its family, or one heading per external TTS server runtime."""
    if spec.kind == "orpheus-llamacpp":
        return "Orpheus family on llama.cpp (SNAC in onnxruntime)"
    if spec.engine == "tts-server" and spec.task == "tts":
        return f"External TTS server: {spec.options.get('runtime')}"
    return spec.family


def models_table(specs: list[ModelSpec]) -> str:
    out: list[str] = []
    by_task: dict[str, dict[str, list[ModelSpec]]] = defaultdict(lambda: defaultdict(list))
    for s in specs:
        by_task[s.task][group(s)].append(s)
    pending = sum(s.status == "pending" for s in specs)
    cloud = sum(s.cloud for s in specs)
    total = len(specs) - pending - cloud
    valid = sum(s.validated == "cpu" for s in specs)
    out.append(
        f"{total} models: {valid} validated on CPU by the maintainers, {total - valid} untested."
        f" {cloud} more are cloud models (engine `remote`: paid cloud APIs; audio leaves this server; none was run"
        " against a real account, see [providers.md](providers.md))."
        + (
            f" {pending} more {'is' if pending == 1 else 'are'} described but not yet published"
            f" ({', '.join(sorted(s.id for s in specs if s.status == 'pending'))})."
            if pending
            else ""
        )
        + "\n"
    )
    for task, title in TASK_TITLES.items():
        out.append(f"### {title}\n")
        for family, items in by_task[task].items():
            out.append(f"#### {family}\n")
            out.append("| id | languages | tier | streaming | download | licence | validated |")
            out.append("| --- | --- | --- | --- | --- | --- | --- |")
            for s in items:
                name = f"`{s.id}`" + (f" ({s.family})" if s.engine == "tts-server" else "")
                out.append(
                    f"| {name} | {langs(s)} | {s.tier} | {'native' if s.streaming else ('-' if task != 'stt' else 'pseudo')} "
                    f"| {'none (cloud API)' if s.cloud else size(s.size_bytes)} | {licence(s)} | {measured(s)} |"
                )
            out.append("")
    out.append("⚠ = the licence restricts commercial use. RTF = processing time / audio duration (lower is faster).")
    return "\n".join(out)


def summary_table(specs: list[ModelSpec]) -> str:
    specs = [s for s in specs if s.status != "pending"]
    rows: dict[tuple[str, str], list[ModelSpec]] = defaultdict(list)
    for s in specs:
        rows[(s.task, group(s))].append(s)
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
        best = min((s.measured["rtf"] for s in valid if "rtf" in s.measured), default=None)
        best_id = next((s.id for s in valid if s.measured["rtf"] == best), "")
        out.append(
            f"| {task.upper()} | {family} | {len(items)} | {lang_text} | {len(valid)} | "
            + (f"{best:.3f} (`{best_id}`)" if best is not None else "-")
            + " |"
        )
    cloud = sum(s.cloud for s in specs)
    total, valid_n = len(specs) - cloud, sum(s.validated == "cpu" for s in specs)
    out.append(
        f"\n**{total} models in the catalog, {valid_n} validated on CPU, {total - valid_n} untested**, plus {cloud} "
        "cloud models (paid cloud APIs; audio leaves this server; [docs/providers.md](docs/providers.md)). "
        "Full list: [docs/models.md](docs/models.md)."
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


GB = 1024**3

#: machines the documentation shows the recommender's picks for (docs/recommendations.md)
EXAMPLE_MACHINES: list[tuple[str, Hardware]] = [
    (
        "The reference machine: 4 x86-64 cores with AVX-512 and AMX, 16 GB, no GPU",
        Hardware(
            system="linux",
            cpu=Cpu("Xeon", "GenuineIntel", "x86_64", 4, 4, 4.0, ["avx2", "avx512", "avx512_vnni", "amx"]),
            memory=Memory(16 * GB, 12 * GB),
            threads=4,
        ),
    ),
    (
        "A small board: 4 ARM cores with NEON only, 4 GB (Raspberry Pi 4 class)",
        Hardware(
            system="linux",
            cpu=Cpu("Cortex-A72", "ARM", "arm64", 4, 4, 4.0, ["neon"]),
            memory=Memory(4 * GB, 3 * GB),
            threads=4,
        ),
    ),
    (
        "A container: 1.5 cores of an AVX2 server, 2 GB memory limit",
        Hardware(
            system="linux",
            cpu=Cpu("Xeon", "GenuineIntel", "x86_64", 32, 64, 1.5, ["avx", "avx2"]),
            memory=Memory(256 * GB, 200 * GB, 2 * GB),
            threads=4,
        ),
    ),
    (
        'A workstation: 16 AVX2 cores, 64 GB, an NVIDIA GPU with 24 GB, `device = "cuda"`',
        Hardware(
            system="linux",
            cpu=Cpu("Ryzen", "AuthenticAMD", "x86_64", 16, 32, 32.0, ["avx", "avx2"]),
            memory=Memory(64 * GB, 56 * GB),
            gpus=[Gpu("nvidia", "an NVIDIA GPU", 24 * GB, 24 * GB, backend="cuda")],
            device="cuda",
            threads=4,
        ),
    ),
]
EXAMPLE_QUERIES = [
    ("Speech to text, English, live", Query("stt", "en", realtime=True)),
    ("Speech to text, English, files", Query("stt", "en", realtime=False)),
    ("Speech to text, Hindi, files", Query("stt", "hi", realtime=False)),
    ("Text to speech, English, live", Query("tts", "en", realtime=True)),
    ("Text to speech, Hindi, voice-over", Query("tts", "hi", realtime=False)),
    ("Speaker diarization", Query("diarization", realtime=False)),
]


def example_machines(specs: list[ModelSpec]) -> str:
    siblings = Siblings(specs)
    out: list[str] = []
    for title, hw in EXAMPLE_MACHINES:
        speed = heuristic_speed(hw)
        levels: dict[str, int] = defaultdict(int)
        for spec in specs:
            levels[assess(spec, hw, speed, siblings).level] += 1
        counts = ", ".join(f"{levels[k]} {LEVEL_LABEL[k]}" for k in LEVEL_LABEL if levels[k])
        out += [
            f"**{title}** (speed factor {speed.factor:.2f}; of {len(specs)} entries: {counts})",
            "",
            "| For | Default pick | Fit | Expected RTF (range) | Basis |",
            "| --- | --- | --- | --- | --- |",
        ]
        for label, query in EXAMPLE_QUERIES:
            kept, _hidden = select(specs, query)
            candidates = [Candidate(s, assess(s, hw, speed, siblings, bool(query.realtime))) for s in kept]
            ordered, _recommended, default = rank(candidates, query)
            pick = next((c for c in ordered if c.spec.id == default), None)
            if pick is None:
                out.append(f"| {label} | none: nothing is expected to run here | | | |")
                continue
            fit = pick.fit.to_dict()
            rtf = "not measured" if fit["expected_rtf"] is None else (
                f"{fit['expected_rtf']:g} ({fit['rtf_range'][0]:g} to {fit['rtf_range'][1]:g})"
            )  # fmt: skip
            basis = {"measured": "measured, scaled", "estimated": "estimated from a sibling"}.get(
                fit["rtf_basis"], "a guess"
            )
            out.append(f"| {label} | `{pick.spec.id}` | {fit['label']} | {rtf} | {basis} |")
        out.append("")
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
    replace_block(ROOT / "docs" / "recommendations.md", "example-machines", example_machines(specs))
    print(f"wrote docs/openapi.json ({len(schema['paths'])} paths) and refreshed tables for {len(specs)} models")


if __name__ == "__main__":
    main()
