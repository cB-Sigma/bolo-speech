"""Command-line interface: ``bolo serve``, ``pull``, ``rm``, ``list``, ``ps``, ``show``,
``transcribe``, ``speak``, ``bench`` and ``doctor``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from bolo import __version__
from bolo.config import Settings, load_settings
from bolo.errors import BoloError

_SETTING_FLAGS: list[tuple[str, str, dict[str, Any]]] = [
    ("--data-dir", "data_dir", {"help": "where models are stored (default: ~/.cache/bolo)"}),
    ("--threads", "threads", {"type": int, "help": "inference threads per worker (default: from CPU count)"}),
    ("--workers", "workers", {"type": int, "help": "worker threads per model (default: 1)"}),
    ("--max-batch", "max_batch", {"type": int, "help": "largest micro-batch (default: 8)"}),
    ("--hf-token", "hf_token", {"help": "Hugging Face token for gated models"}),
    ("--log-level", "log_level", {"help": "debug, info, warning or error"}),
]


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="path to a TOML configuration file")
    for flag, dest, kw in _SETTING_FLAGS:
        parser.add_argument(flag, dest=dest, default=None, **kw)
    parser.add_argument("--fake-engine", dest="fake_engine", action="store_const", const=True, default=None,
                        help="enable the deterministic fake models (for development and tests)")  # fmt: skip


def _settings(args: argparse.Namespace, extra: tuple[str, ...] = ()) -> Settings:
    names = [dest for _, dest, _ in _SETTING_FLAGS] + ["fake_engine", *extra]
    overrides = {n: getattr(args, n) for n in names if getattr(args, n, None) is not None}
    return load_settings(args.config, overrides)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _human(n: float | None) -> str:
    if not n:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return "-"


def _table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
    lines += ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    return "\n".join(lines)


# -- serve ----------------------------------------------------------------
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from bolo.api import create_app

    settings = _settings(
        args,
        ("host", "port", "api_key", "preload", "cors_origins", "wyoming", "wyoming_host", "wyoming_port", "auto_pull",
         "ui", "default_stt", "default_tts", "default_realtime_stt", "memory_budget_mb", "batch_wait_ms", "max_queue",
         "llm_base_url", "llm_model", "llm_api_key"),
    )  # fmt: skip
    _setup_logging(settings.log_level)
    app = create_app(settings)
    where = f"http://{settings.host}:{settings.port}"
    print(f"bolo {__version__} listening on {where}  (console: {where}/  API docs: {where}/docs)", file=sys.stderr)
    if settings.host not in ("127.0.0.1", "localhost", "::1") and not settings.api_key:
        print("warning: listening on a non-loopback address without an API key", file=sys.stderr)
    uvicorn.run(
        app, host=settings.host, port=settings.port, log_level=settings.log_level.lower(), ws_max_size=16 * 1024 * 1024
    )
    return 0


# -- models ---------------------------------------------------------------
def _core(args: argparse.Namespace) -> Any:
    from bolo.core import Bolo

    settings = _settings(args)
    _setup_logging(settings.log_level if args.log_level else "warning")
    return Bolo(settings)


def cmd_pull(args: argparse.Namespace) -> int:
    core = _core(args)
    name = args.model
    if name.startswith("hf:"):
        from bolo.custom import register_hf_model

        spec = register_hf_model(core, name[3:], args.kind, args.task)
        print(f"registered {name} as '{spec.id}' ({spec.engine}, kind {spec.kind})")
    elif name.startswith("piper:"):
        from bolo.piper import register_piper_voice

        try:
            spec = register_piper_voice(core, name[6:])
        except BoloError as exc:
            print(f"error: {exc.message}", file=sys.stderr)
            return 1
        print(f"registered {name} as '{spec.id}' (converted for sherpa-onnx after the download)")
    elif name.startswith("file:") or (Path(name).is_dir() and core.catalog.resolve(name) is None):
        from bolo.custom import register_local_model

        options: dict[str, Any] = {}
        for item in args.option or []:
            key, sep, value = item.partition("=")
            if not sep or not key:
                print(f"error: --option expects key=value, got {item!r}", file=sys.stderr)
                return 1
            options[key] = value
        try:
            spec = register_local_model(
                core,
                name.removeprefix("file:"),
                args.kind,
                args.task,
                name=args.name,
                languages=[x for x in (args.language or "").split(",") if x],
                options=options,
            )
        except BoloError as exc:
            print(f"error: {exc.message}", file=sys.stderr)
            return 1
        print(f"registered {name} as '{spec.id}' ({spec.engine}, kind {spec.kind})")
    else:
        spec = core.manager.resolve(name)
    if core.manager.is_installed(spec):
        print(f"{spec.id} is already installed")
        return 0
    try:
        spec.require_published()
    except BoloError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    if spec.gated and not core.store.hub.token:
        print(
            f"error: {spec.id} is a gated model. Accept its terms at {spec.source_url}, then pass --hf-token "
            "or set BOLO_HF_TOKEN.",
            file=sys.stderr,
        )
        return 1
    last = 0.0

    def progress(done: int, total: int) -> None:
        nonlocal last
        now = time.monotonic()
        if now - last < 0.2 and done != total:
            return
        last = now
        pct = f"{100 * done / total:5.1f}%" if total else "  ?  "
        print(f"\rpulling {spec.id}  {pct}  {_human(done)} / {_human(total)}   ", end="", file=sys.stderr, flush=True)

    manifest = core.store.pull(spec.id, spec.source, progress=progress)
    print(file=sys.stderr)
    print(f"installed {spec.id} ({_human(manifest['size_bytes'])}, {len(manifest['files'])} files)")
    if not spec.commercial_use:
        print(f"note: licence {spec.license} restricts commercial use")
    return 0


def cmd_rm(args: argparse.Namespace) -> int:
    core = _core(args)
    code = 0
    for name in args.models:
        spec = core.catalog.resolve(name)
        if spec is None or not core.store.remove(spec.id):
            print(f"{name}: not installed", file=sys.stderr)
            code = 1
            continue
        if spec.custom:
            core.catalog.remove_custom(spec.id)
        print(f"removed {spec.id}")
    return code


def cmd_list(args: argparse.Namespace) -> int:
    core = _core(args)
    models = core.catalog_api(task=args.task, q=args.query, installed=None if args.all else True)
    if args.json:
        print(json.dumps({"models": models}, indent=2, ensure_ascii=False))
        return 0
    if not models:
        print("no models installed. `bolo list --all` shows the catalog; `bolo pull <id>` installs one.")
        return 0
    rows = [
        [
            m["id"], m["task"], m["tier"], m["engine"], ",".join(m["languages"][:4]) + ("…" if len(m["languages"]) > 4 else ""),
            _human(m["size_bytes"]), m["license"], "yes" if m["installed"] else "", "cpu" if m["validated"] == "cpu" else "",
        ]
        for m in models
    ]  # fmt: skip
    print(_table(rows, ["ID", "TASK", "TIER", "ENGINE", "LANG", "SIZE", "LICENCE", "INSTALLED", "VALIDATED"]))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    core = _core(args)
    spec = core.manager.resolve(args.model)
    entry = core.model_api(spec)
    entry["source"] = spec.source.to_dict()
    entry["kind"] = spec.kind
    if len(entry["voices"]) > 12 and not args.json:
        entry["voices"] = [v["id"] for v in entry["voices"]]
    print(json.dumps(entry, indent=2, ensure_ascii=False))
    return 0


def cmd_ps(args: argparse.Namespace) -> int:
    import httpx

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    try:
        stats = (
            httpx.get(f"{args.url.rstrip('/')}/api/v1/stats", headers=headers, timeout=5.0).raise_for_status().json()
        )
    except httpx.HTTPError as exc:
        print(f"could not reach a bolo server at {args.url}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(stats, indent=2))
        return 0
    rows = []
    for m in stats["models"]:
        if not m["loaded"]:
            continue
        lat = m["latency_ms"]["final"] or m["latency_ms"]["ttfa"] or {}
        rows.append([
            m["id"], m["task"], _human(m["memory_bytes"]), str(m["active"]), str(m["requests"]),
            f"{lat.get('p50', '-')}", f"{(m['rtf'] or {}).get('p50', '-')}", f"{m['idle_s']}s",
        ])  # fmt: skip
    print(f"uptime {stats['uptime_s']:.0f}s  streams {stats['active_streams']}  queue {stats['queue_depth']}  "
          f"cpu {stats['system']['cpu_percent']}%  rss {_human(stats['system']['rss_bytes'])}")  # fmt: skip
    print(
        _table(rows, ["MODEL", "TASK", "MEMORY", "ACTIVE", "REQUESTS", "P50 MS", "RTF P50", "IDLE"])
        if rows
        else "no models loaded"
    )
    return 0


# -- one-shot inference ---------------------------------------------------
def cmd_transcribe(args: argparse.Namespace) -> int:
    from bolo.audio import decode_audio
    from bolo.obs.trace import Trace
    from bolo.text.formats import to_srt, to_vtt

    core = _core(args)
    path = Path(args.file)
    samples, rate = decode_audio(path.read_bytes(), filename=path.name)

    async def run() -> int:
        await core.start()
        try:
            trace = Trace("stt", "cli", "")
            t = await core.transcribe(
                samples, rate, model=args.model, language=args.language, trace=trace,
                task="translate" if args.translate else "transcribe", word_timestamps=args.format == "json",
            )  # fmt: skip
            if args.format == "srt":
                print(to_srt(t.segments))
            elif args.format == "vtt":
                print(to_vtt(t.segments))
            elif args.format == "json":
                print(json.dumps({
                    "text": t.text, "language": t.language, "duration": round(t.duration, 3), "model": t.model,
                    "segments": [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text} for s in t.segments],
                    "words": [w.to_dict() for w in t.words],
                    "timing": {"total_ms": trace.total_ms, "rtf": trace.rtf, "phases": [p.to_dict() for p in trace.phases]},
                }, indent=2, ensure_ascii=False))  # fmt: skip
            else:
                print(t.text)
            if args.verbose:
                print(
                    f"[{t.model}] {t.duration:.1f}s audio in {trace.total_ms:.0f} ms (RTF {trace.rtf:.3f})",
                    file=sys.stderr,
                )
            return 0
        finally:
            await core.stop()

    return asyncio.run(run())


def cmd_speak(args: argparse.Namespace) -> int:
    from bolo.audio.encode import encode_audio
    from bolo.obs.trace import Trace

    core = _core(args)
    text = args.text if args.text != "-" else sys.stdin.read()
    out = Path(args.output)
    fmt = args.format or (out.suffix.lstrip(".").lower() or "wav")

    async def run() -> int:
        await core.start()
        try:
            trace = Trace("tts", "cli", "")
            parts = []
            rate = 0
            try:
                async for chunk in core.synthesize(
                    text, model=args.model, voice=args.voice, speed=args.speed, trace=trace
                ):
                    trace.mark_ttfa()
                    parts.append(chunk.samples)
                    rate = chunk.sample_rate
            finally:
                core.complete_tts(trace)
            import numpy as np

            audio = np.concatenate(parts)
            out.write_bytes(encode_audio(audio, rate, fmt, args.sample_rate))
            print(f"wrote {out} ({len(audio) / rate:.1f}s, {fmt})")
            if args.verbose:
                print(
                    f"[{trace.model}] first audio after {trace.ttfa_ms:.0f} ms, total {trace.total_ms:.0f} ms, RTF {trace.rtf:.3f}",
                    file=sys.stderr,
                )
            return 0
        finally:
            await core.stop()

    return asyncio.run(run())


def cmd_bench(args: argparse.Namespace) -> int:
    from bolo.bench import run_bench

    core = _core(args)
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]

    async def run() -> dict[str, Any]:
        await core.start()
        try:
            return await run_bench(
                core, kind=args.kind, model=args.model, concurrency=levels, duration_s=args.duration,
                sample=args.sample, mode=args.mode, voice=args.voice,
            )  # fmt: skip
        finally:
            await core.stop()

    result = asyncio.run(run())
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(
        f"{result['kind']} {result['model']}  sample={result['sample']}  mode={result['mode']}  {result['duration_s']}s per level"
    )

    def q(d: dict[str, float] | None, key: str) -> str:
        return f"{d[key]:.0f}" if d else "-"

    rows = []
    for p in result["points"]:
        rtf = p["rtf"]
        rows.append([
            str(p["concurrency"]), str(p["ok"]), str(p["errors"]),
            q(p["first_partial_ms"], "p50"), q(p["first_partial_ms"], "p95"), q(p["final_ms"], "p50"), q(p["final_ms"], "p95"),
            q(p["ttfa_ms"], "p50"), q(p["ttfa_ms"], "p95"), f"{rtf['p50']:.3f}" if rtf else "-",
            f"{p['throughput_x_realtime']:.1f}", str(p.get("mean_batch", "-")),
        ])  # fmt: skip
    print(
        _table(
            rows,
            [
                "CONC",
                "OK",
                "ERR",
                "PARTIAL P50",
                "P95",
                "FINAL P50",
                "P95",
                "TTFA P50",
                "P95",
                "RTF P50",
                "X REALTIME",
                "BATCH",
            ],
        )
    )
    return 0


def cmd_components(args: argparse.Namespace) -> int:
    core = _core(args)
    manager = core.components
    if args.action == "list":
        for c in manager.status():
            state = f"installed ({c['source']}, {c['version']})" if c["installed"] else "not installed"
            print(f"{c['id']:<12} {state:<40} {c['license']:<18} needed by {c['required_by_count']} catalog models")
            if not c["installed"]:
                print(
                    f"             install: bolo components install {c['id']} --accept-license   (or: {c['manual_command']})"
                )
        return 0
    if args.component != "espeak-ng":
        print(f"error: unknown component '{args.component}'", file=sys.stderr)
        return 1
    if manager.is_installed(args.component):
        print(f"{args.component} is already installed")
        return 0
    if not args.accept_license:
        print(
            "error: eSpeak NG is licensed GPL-3.0-or-later and is not part of bolo. Re-run with --accept-license "
            "to install it into this Python environment.",
            file=sys.stderr,
        )
        return 1
    error = manager.install_blocking()
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"installed {args.component}; voices that need it are now available")
    return 0


def _doctor_llm(settings: Settings) -> None:
    from bolo.llm import LLMClient

    check = asyncio.run(LLMClient(settings).check())
    if not check["configured"]:
        print(
            "language model: not configured (set llm_base_url for /v1/realtime conversations and /v1/chat/completions)"
        )
        return
    if not check["reachable"]:
        print(f"language model: {check['base_url']} is NOT reachable ({check['error']})")
        return
    model = check["model"] or "(no llm_model set: the client's model name is passed through)"
    found = {True: "found", False: "NOT in the endpoint's model list", None: "not checked"}[check["model_found"]]
    print(f"language model: {check['base_url']} reachable; model {model}: {found}")


def cmd_doctor(args: argparse.Namespace) -> int:
    import platform

    from bolo.audio import decode, encode
    from bolo.audio.resample import HAVE_SOXR
    from bolo.engines import EngineRegistry
    from bolo.text.phonemize import get_phonemizer

    settings = _settings(args)
    print(
        f"bolo {__version__}  python {platform.python_version()}  {platform.system().lower()}-{platform.machine()}  cpus {os.cpu_count()}"
    )
    print(f"data dir: {settings.data_dir}")
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(settings.data_dir)
        print(f"  free disk: {_human(usage.free)}")
    except OSError as exc:
        print(f"  not writable: {exc}")
    print("engines:")
    ok = False
    for st in EngineRegistry(include_fake=False).statuses():
        mark = "ok " if st.available else "-- "
        ok = ok or st.available
        print(
            f"  {mark}{st.id:<15} {st.version or ''} {'' if st.available else '(' + (st.reason or 'unavailable') + ')'}"
        )
    if not ok:
        print("  no inference engine is installed. Try: pip install 'bolo-speech[sherpa]'")
    print("audio:")
    print(f"  resampler: {'libsoxr' if HAVE_SOXR else 'built-in (install soxr for better quality)'}")
    print(
        f"  decode wav/flac/ogg/opus/mp3: libsndfile;  m4a/aac/webm: {'PyAV' if decode.have_av() else ('ffmpeg' if decode.have_ffmpeg() else 'unavailable (pip install bolo-speech[av])')}"
    )
    fmts = encode.available_formats()
    print(
        "  encode: "
        + ", ".join(f for f, v in fmts.items() if v)
        + ("" if fmts["aac"] else "  (aac needs bolo-speech[av] or ffmpeg)")
    )
    ph = get_phonemizer()
    if ph:
        print(f"espeak-ng: found ({ph.source}, version {ph.version() or 'unknown'}) - optional, GPL-3.0-or-later")
    else:
        print(
            "espeak-ng: not installed (optional, GPL-3.0-or-later; Piper-format, KittenTTS and non-English Kokoro "
            "voices need it: `bolo components install espeak-ng --accept-license`)"
        )
    _doctor_llm(settings)
    try:
        import wyoming

        print(f"wyoming: ok {getattr(wyoming, '__version__', '')}")
    except Exception:
        print("wyoming: not installed (pip install 'bolo-speech[wyoming]')")
    if args.network:
        import httpx

        try:
            r = httpx.get(f"{settings.hf_endpoint}/api/models/csukuangfj/vad", timeout=10.0)
            print(f"hugging face: HTTP {r.status_code}")
        except httpx.HTTPError as exc:
            print(f"hugging face: unreachable ({exc})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bolo", description="A lightweight speech server: STT, TTS and VAD.")
    p.add_argument("--version", action="version", version=f"bolo {__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    s = sub.add_parser("serve", help="run the server")
    _common(s)
    s.add_argument("--host", default=None, help="bind address (default 127.0.0.1)")
    s.add_argument("--port", type=int, default=None, help="port (default 8000)")
    s.add_argument("--api-key", dest="api_key", default=None, help="require this bearer token")
    s.add_argument("--preload", action="append", default=None, help="load this model at startup (repeatable)")
    s.add_argument(
        "--cors-origin",
        dest="cors_origins",
        action="append",
        default=None,
        help="allowed CORS origin (repeatable, or *)",
    )
    s.add_argument("--wyoming", action="store_const", const=True, default=None, help="also serve the Wyoming protocol")
    s.add_argument("--wyoming-host", dest="wyoming_host", default=None)
    s.add_argument("--wyoming-port", dest="wyoming_port", type=int, default=None, help="default 10300")
    s.add_argument("--no-auto-pull", dest="auto_pull", action="store_const", const=False, default=None,
                   help="never download a model on first use")  # fmt: skip
    s.add_argument(
        "--no-ui", dest="ui", action="store_const", const=False, default=None, help="do not serve the web console"
    )
    s.add_argument("--default-stt", dest="default_stt", default=None)
    s.add_argument("--default-realtime-stt", dest="default_realtime_stt", default=None)
    s.add_argument("--default-tts", dest="default_tts", default=None)
    s.add_argument("--memory-budget-mb", dest="memory_budget_mb", type=int, default=None)
    s.add_argument("--batch-wait-ms", dest="batch_wait_ms", type=float, default=None)
    s.add_argument("--max-queue", dest="max_queue", type=int, default=None)
    s.add_argument(
        "--llm-base-url",
        dest="llm_base_url",
        default=None,
        help="OpenAI-compatible chat endpoint for voice agents, e.g. http://127.0.0.1:11434/v1",
    )
    s.add_argument("--llm-model", dest="llm_model", default=None)
    s.add_argument("--llm-api-key", dest="llm_api_key", default=None)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser(
        "pull", help="download a model (catalog id, alias or hf:<repo>), or register a local directory (./dir)"
    )
    _common(s)
    s.add_argument(
        "model", help="catalog id or alias, hf:<owner>/<repo>[@revision], or a directory (./dir, file:/path)"
    )
    s.add_argument("--kind", help="loader kind for hf: and local models when it cannot be detected")
    s.add_argument("--task", choices=["stt", "tts", "vad"], help="task for hf: and local models")
    s.add_argument("--name", help="local models: the id to register the directory under (default: local-<dirname>)")
    s.add_argument("--language", help="local models: language code(s), comma separated")
    s.add_argument("--option", action="append", help="local models: loader option key=value (repeatable)")
    s.set_defaults(func=cmd_pull)

    s = sub.add_parser("rm", help="delete installed models")
    _common(s)
    s.add_argument("models", nargs="+")
    s.set_defaults(func=cmd_rm)

    s = sub.add_parser("list", aliases=["ls"], help="list installed models (--all: the whole catalog)")
    _common(s)
    s.add_argument("--all", "-a", action="store_true")
    s.add_argument("--task", choices=["stt", "tts", "vad"])
    s.add_argument("--query", "-q", help="filter by text")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="show one catalog entry")
    _common(s)
    s.add_argument("model")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("ps", help="show models loaded in a running server")
    s.add_argument("--url", default=os.environ.get("BOLO_URL", "http://127.0.0.1:8000"))
    s.add_argument("--api-key", dest="api_key", default=os.environ.get("BOLO_API_KEY"))
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_ps)

    s = sub.add_parser("transcribe", help="transcribe an audio file")
    _common(s)
    s.add_argument("file")
    s.add_argument("--model", "-m")
    s.add_argument("--language", "-l")
    s.add_argument("--translate", action="store_true", help="translate to English (multilingual Whisper models)")
    s.add_argument("--format", "-f", choices=["text", "json", "srt", "vtt"], default="text")
    s.add_argument("--verbose", "-v", action="store_true")
    s.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("speak", help="synthesise speech to a file")
    _common(s)
    s.add_argument("text", help="text to speak, or - to read standard input")
    s.add_argument("--output", "-o", required=True)
    s.add_argument("--model", "-m")
    s.add_argument("--voice")
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--format", "-f", help="default: from the output file extension")
    s.add_argument("--sample-rate", dest="sample_rate", type=int)
    s.add_argument("--verbose", "-v", action="store_true")
    s.set_defaults(func=cmd_speak)

    s = sub.add_parser("bench", help="concurrency sweep against a model (in process)")
    _common(s)
    s.add_argument("--kind", choices=["stt", "tts"], default="stt")
    s.add_argument("--model", "-m")
    s.add_argument("--concurrency", "-c", default="1,2,4,8")
    s.add_argument("--duration", "-d", type=float, default=10.0, help="seconds per concurrency level")
    s.add_argument("--sample", help="built-in sample id (see GET /api/v1/samples)")
    s.add_argument("--mode", choices=["batch", "realtime"], default="batch")
    s.add_argument("--voice")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("components", help="optional third-party components (espeak-ng): list, install")
    _common(s)
    s.add_argument("action", choices=["list", "install"])
    s.add_argument("component", nargs="?", default="espeak-ng")
    s.add_argument(
        "--accept-license",
        dest="accept_license",
        action="store_true",
        help="accept the component's licence (GPL-3.0-or-later)",
    )
    s.set_defaults(func=cmd_components)

    s = sub.add_parser("doctor", help="check engines, codecs, components, the language model and the data directory")
    _common(s)
    s.add_argument("--network", action="store_true", help="also check that Hugging Face is reachable")
    s.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except BoloError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
