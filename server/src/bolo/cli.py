"""Command-line interface: ``bolo serve``, ``pull``, ``rm``, ``cp``, ``list``, ``ps``, ``load``/``run``, ``stop``,
``show``, ``models``, ``logs``, ``transcribe``, ``speak``, ``bench``, ``voices``, ``recommend`` and ``doctor``."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
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
         "llm_base_url", "llm_model", "llm_api_key", "num_parallel", "gpus", "gpu_spread", "log_file"),
    )  # fmt: skip
    from bolo import logs, procs

    log_file = logs.configure(settings)
    procs.install_hooks()  # from the main thread: children are stopped at exit and on SIGHUP
    app = create_app(settings)
    where = f"http://{settings.host}:{settings.port}"
    print(f"bolo {__version__} listening on {where}  (console: {where}/  API docs: {where}/docs)", file=sys.stderr)
    if log_file is not None:
        print(f"log file: {log_file} (`bolo logs -f` follows it)", file=sys.stderr)
    if settings.host not in ("127.0.0.1", "localhost", "::1") and not settings.api_key:
        print("warning: listening on a non-loopback address without an API key", file=sys.stderr)
    config = uvicorn.Config(
        app, host=settings.host, port=settings.port, log_level=settings.log_level.lower(), ws_max_size=16 * 1024 * 1024
    )
    logs.attach_file_handler("uvicorn", "uvicorn.error", "uvicorn.access")  # uvicorn's loggers skip the root logger
    uvicorn.Server(config).run()
    return 0


# -- models ---------------------------------------------------------------
def _core(args: argparse.Namespace) -> Any:
    from bolo.core import Bolo

    settings = _settings(args)
    _setup_logging(settings.log_level if args.log_level else "warning")
    return Bolo(settings)


def _pull_pack(core: Any, args: argparse.Namespace) -> int | None:
    """``bolo pull uccharan-<pack>``: one of uccharan's language packs (None: the name is not a pack)."""
    from bolo.text import packs

    name = packs.pack_of(args.model)
    if name is None:
        return None
    pack = packs.info(name)
    assert pack is not None
    if core.components.is_installed(args.model):
        print(f"{args.model} is already installed")
        return 0
    if pack.share_alike and not args.accept_license:
        print(
            f"error: uccharan's pack '{name}' is licensed {pack.license} (it is built from share-alike data). "
            "Re-run with --accept-license to download it.",
            file=sys.stderr,
        )
        return 1
    print(f"downloading uccharan language pack '{name}' ({packs.human_size(pack.size)}, {pack.license})")
    error = core.components.install_blocking(component=args.model)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    where = packs.cache_dir() or "uccharan's cache"
    print(f"installed {args.model} (SHA-256 checked) into {where}")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    core = _core(args)
    name = args.model
    pulled = _pull_pack(core, args)
    if pulled is not None:
        return pulled
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
    for warning in core.advisor.warnings(spec, pulling=True):
        # advice only: the download goes ahead (`hardware_warnings = false` silences it)
        print(f"warning: {warning['message']}", file=sys.stderr)
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


def _server(args: argparse.Namespace, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    """A request to the running server at ``--url``, or None when nothing answers there."""
    import httpx

    headers = {"Authorization": f"Bearer {args.api_key}"} if getattr(args, "api_key", None) else {}
    url = getattr(args, "url", None)
    if not url:
        return None
    try:
        return httpx.request(method, f"{url.rstrip('/')}{path}", headers=headers, json=body, timeout=5.0)
    except httpx.HTTPError:
        return None


def _error_of(r: Any) -> str:
    try:
        return str((r.json().get("error") or {}).get("message") or r.text)
    except ValueError:
        return str(r.text)


def _same_dir(a: str | os.PathLike[str], b: str | os.PathLike[str]) -> bool:
    def norm(p: str | os.PathLike[str]) -> str:
        return os.path.normcase(os.path.realpath(Path(p).expanduser()))

    return norm(a) == norm(b)


def _server_for_store(args: argparse.Namespace, data_dir: Path) -> bool:
    """Whether the server at ``--url`` keeps its models in ``data_dir``, the data directory this command resolved
    (``--data-dir``, ``BOLO_DATA_DIR``, the config file). Only then do ``rm`` and ``cp`` go through it: a server
    on another data directory would change its own store, not this one. False when nothing answers; a server
    whose /api/v1/info has no ``data_dir`` (an old one) is taken to share it."""
    r = _server(args, "GET", "/api/v1/info")
    if r is None or r.status_code != 200:
        return False
    try:
        theirs = r.json().get("data_dir")
    except ValueError:
        return False
    if theirs is None or _same_dir(theirs, data_dir):
        return True
    print(
        f"note: the server at {args.url} uses the data directory {theirs}; this command changes {data_dir} "
        "(--data-dir, BOLO_DATA_DIR or the config file), without the server",
        file=sys.stderr,
    )
    return False


def cmd_rm(args: argparse.Namespace) -> int:
    """``bolo rm <model>...``: delete a model's files (a preset: only the preset). A model the running server has
    loaded is refused unless ``--force``, which unloads it first."""
    from urllib.parse import quote

    core = _core(args)
    code = 0
    ps = _server(args, "GET", "/api/v1/ps") if _server_for_store(args, core.settings.data_dir) else None
    loaded = {m.get("name") for m in ps.json().get("models", [])} if ps is not None and ps.status_code == 200 else None
    for name in args.models:
        spec = core.catalog.resolve(name)
        if spec is None or (not spec.preset_of and not core.store.is_installed(spec.id)):
            print(f"{name}: not installed", file=sys.stderr)
            code = 1
            continue
        if loaded is not None and spec.id in loaded:
            if not args.force:
                print(
                    f"{spec.id} is loaded in the server at {args.url}: `bolo stop {spec.id}` first, or `bolo rm --force`",
                    file=sys.stderr,
                )
                code = 1
                continue
            _server(args, "POST", f"/api/v1/models/{quote(spec.id, safe='')}/unload")
        if loaded is not None:  # the server removes it, so its catalog (presets, custom models) stays right
            r = _server(args, "DELETE", f"/api/v1/models/{quote(spec.id, safe='')}")
            if r is not None and r.status_code < 300:
                print(f"removed {spec.id}" + (" (a preset: the model's files stay)" if spec.preset_of else ""))
                continue
            if r is not None:
                print(f"error: {_error_of(r)}", file=sys.stderr)
                code = 1
                continue
        if not asyncio.run(core.remove(spec.id)):
            print(f"{name}: not installed", file=sys.stderr)
            code = 1
            continue
        print(f"removed {spec.id}" + (" (a preset: the model's files stay)" if spec.preset_of else ""))
    return code


def cmd_cp(args: argparse.Namespace) -> int:
    """``bolo cp <source> <destination>``: a preset, a new name with the source's files and its own settings."""
    r = None
    if _server_for_store(args, Path(_settings(args).data_dir)):
        r = _server(args, "POST", "/api/v1/models/copy", {"source": args.source, "destination": args.destination})
    if r is not None and r.status_code != 404:
        if r.status_code >= 400:
            print(f"error: {_error_of(r)}", file=sys.stderr)
            return 1
        doc = r.json()
    else:
        core = _core(args)
        spec = core.copy_model(args.source, args.destination)
        doc = core.model_api(spec)
    print(
        f"copied {doc.get('preset_of') or args.source} to {doc['id']}; change its settings with "
        f"`bolo models config {doc['id']} --set key=value`"
    )
    return 0


def _ago(iso: str | None) -> str:
    """``2026-11-14T10:00:00Z`` -> ``3 days ago`` (Ollama's MODIFIED)."""
    if not iso:
        return "-"
    from datetime import datetime, timezone

    try:
        then = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return "-"
    secs = max(0.0, (datetime.now(timezone.utc) - then).total_seconds())
    for unit, size in (("year", 365 * 86400), ("month", 30 * 86400), ("week", 7 * 86400), ("day", 86400),
                       ("hour", 3600), ("minute", 60)):  # fmt: skip
        if secs >= size:
            n = int(secs // size)
            return f"{n} {unit}{'s' if n > 1 else ''} ago"
    return "just now"


def cmd_list(args: argparse.Namespace) -> int:
    core = _core(args)
    models = core.catalog_api(task=args.task, q=args.query, installed=None if args.all else True)
    if args.json:
        print(json.dumps({"models": models}, indent=2, ensure_ascii=False))
        return 0
    if not models:
        print("no models installed. `bolo list --all` shows the catalog; `bolo pull <id>` installs one.")
        return 0
    if not args.all:
        # downloaded models, as `ollama list` shows them: name, size on disk, when they were downloaded
        rows = [
            [
                m["id"], m["task"], m["engine"], "-" if m["cloud"] else _human(m["size_bytes"]), _ago(m.get("modified_at")),
                f"preset of {m['preset_of']}" if m.get("preset_of") else ("cloud" if m["cloud"] else m["license"]),
            ]
            for m in models
        ]  # fmt: skip
        print(_table(rows, ["ID", "TASK", "ENGINE", "SIZE", "MODIFIED", "LICENCE"]))
        return 0
    rows = [
        [
            m["id"], m["task"], m["tier"], m["engine"], ",".join(m["languages"][:4]) + ("…" if len(m["languages"]) > 4 else ""),
            _human(m["size_bytes"]), (m.get("fit") or {}).get("label") or "", "yes" if m["installed"] else "",
            "cpu" if m["validated"] == "cpu" else "", m["license"],
        ]
        for m in models
    ]  # fmt: skip
    # FIT: how the model suits this machine (advice only; `bolo recommend` has the numbers and the reasons)
    print(_table(rows, ["ID", "TASK", "TIER", "ENGINE", "LANG", "SIZE", "FIT", "INSTALLED", "VALIDATED", "LICENCE"]))
    return 0


def _phonemes_line(doc: dict[str, Any]) -> str | None:
    """The phonemiser a voice that reads phonemes uses on this installation (what is installed, the language it
    speaks, a pinned choice), and the other choices; None for a model that reads characters or a lexicon."""
    options = doc.get("phonemizers") or []
    if not options:
        return None
    chosen = next((o for o in options if o.get("default")), None)
    if chosen is None:  # pinned to one that is not installed, while another is
        pinned = (doc.get("settings") or {}).get("phonemizer")
        return (
            f"{pinned} (pinned), which is not installed: install it, or `bolo models config {doc['id']} --unset "
            "phonemizer`"
        )
    if not chosen.get("available"):
        line = f"{chosen['id']}, which is not installed ({chosen.get('note')})"
    else:
        line = str(chosen.get("note") or chosen["id"])
    others = [str(o["id"]) for o in options if o is not chosen and o.get("available")]
    if others:
        line += f"; also available: {', '.join(others)} (`bolo models config {doc['id']} --phonemizer <name>`)"
    return line


def format_show(doc: dict[str, Any]) -> str:
    """``bolo show``: the parts of :meth:`bolo.core.Bolo.show_api` a person reads, in sections."""
    out: list[str] = []

    def section(title: str, rows: list[tuple[str, Any]]) -> None:
        rows = [(k, v) for k, v in rows if v not in (None, "", [], {})]
        if not rows:
            return
        out.append(f"  {title}")
        width = max(len(k) for k, _ in rows)
        out.extend(f"    {k.ljust(width)}  {v}" for k, v in rows)
        out.append("")

    state = "loaded" if doc.get("loaded") else ("installed" if doc.get("installed") else "not installed")
    out.append(f"{doc['id']}  ({doc.get('task')}, {state})")
    if doc.get("description"):
        out.append(f"  {doc['description']}")
    out.append("")
    section("Model", [
        ("engine", f"{doc.get('engine')} (kind {doc.get('kind')})"), ("family", doc.get("family")),
        ("tier", doc.get("tier")), ("parameters", f"{doc['params']:,}" if doc.get("params") else None),
        ("sample rate", f"{doc.get('sample_rate')} Hz"), ("streaming", "yes" if doc.get("streaming") else "no"),
        ("aliases", ", ".join(doc.get("aliases") or [])), ("preset of", doc.get("preset_of")),
        ("presets", ", ".join(doc.get("presets") or [])), ("phonemes", _phonemes_line(doc)),
    ])  # fmt: skip
    section("Licence", [
        ("licence", doc.get("license")), ("url", doc.get("license_url")),
        ("commercial use", "yes" if doc.get("commercial_use") else "NO (restricted by the licence)"),
        ("gated", "yes: accept the terms on Hugging Face and set hf_token" if doc.get("gated") else "no"),
        ("cloud", "yes: audio or text leaves this server" if doc.get("cloud") else None),
    ])  # fmt: skip
    langs = doc.get("languages") or []
    voices = doc.get("voices") or []
    section("Languages and voices", [
        ("languages", ", ".join(langs[:24]) + (f" (+{len(langs) - 24})" if len(langs) > 24 else "")),
        ("voices", ", ".join(v["id"] for v in voices[:16]) + (f" (+{len(voices) - 16})" if len(voices) > 16 else "")),
    ])  # fmt: skip
    files = doc.get("files") or []
    on_disk = bool(doc.get("path"))
    section("Files", [
        ("directory", doc.get("path")), ("size", _human(doc.get("size_bytes")) if on_disk else None),
        ("modified", _ago(doc.get("modified_at")) if on_disk else None),
        *[(f["path"], _human(f.get("size"))) for f in files[:20]],
        ("…", f"{len(files) - 20} more" if len(files) > 20 else None),
    ])  # fmt: skip
    eff = doc.get("effective") or {}
    from bolo.keepalive import format_keep_alive

    section("Settings", [
        ("device", eff.get("device")), ("keep alive", format_keep_alive(eff.get("keep_alive_s"))),
        ("num_parallel", eff.get("num_parallel")), ("max_queue", eff.get("max_queue")),
        *[(f"llama {k}", v) for k, v in (eff.get("llama") or {}).items()],
        *[(f"pinned {k}", json.dumps(v)) for k, v in (doc.get("settings") or {}).items()],
    ])  # fmt: skip
    speed = doc.get("speed") or {}
    server = speed.get("server") or {}
    catalog = speed.get("catalog") or {}
    section("Speed", [
        ("expected here", f"real-time factor about {speed['expected_rtf']:g} ({speed.get('rtf_basis')})"
         if speed.get("expected_rtf") is not None else None),
        ("measured (catalog)", ", ".join(f"{k} {v}" for k, v in catalog.items() if not isinstance(v, (dict, list)))),
        ("measured (server)", f"{server.get('requests')} requests, rtf p50 {(server.get('rtf') or {}).get('p50', '-')}"
         if server.get("requests") else None),
    ])  # fmt: skip
    return "\n".join(out).rstrip()


def cmd_show(args: argparse.Namespace) -> int:
    """``bolo show <model> [--license] [--json]``: licence, files, settings, languages, voices, engine and speed."""
    core = _core(args)
    spec = core.manager.resolve(args.model)
    doc = core.show_api(spec, license_text=args.license)
    if args.json:
        print(json.dumps(doc, indent=2, ensure_ascii=False))
        return 0
    if args.license:
        print(f"{spec.id}: {spec.license}" + (f"  {spec.license_url}" if spec.license_url else ""))
        print(f"commercial use: {'yes' if spec.commercial_use else 'no'}; gated: {'yes' if spec.gated else 'no'}")
        if doc.get("license_text"):
            print()
            print(doc["license_text"].rstrip())
        elif not doc.get("installed"):
            print("(the model's files are not downloaded; its licence file, if it has one, is shown after `bolo pull`)")
        else:
            print("(no licence file among the model's files; the link above has the terms)")
        return 0
    print(format_show(doc))
    return 0


_INT_SETTINGS = ("num_parallel", "max_queue", "context_size", "batch_size")
_LIST_SETTINGS = ("wordlists", "fallback")


def _model_setting(key: str, raw: str) -> Any:
    """A ``--set key=value`` of ``bolo models config``, as the type the setting takes."""
    from bolo.keepalive import parse_keep_alive
    from bolo.llama_args import check_tuning

    if key in _INT_SETTINGS:
        value: Any = int(raw)
        if value < (1 if key == "num_parallel" else 0):
            raise ValueError(f"{key}: expected a number of at least {1 if key == 'num_parallel' else 0}")
    elif key in _LIST_SETTINGS:
        value = [x.strip() for x in raw.split(",") if x.strip()]
    elif key == "keep_alive":
        parse_keep_alive(raw)
        value = raw
    else:
        value = raw.strip().lower() if key in ("device", "flash_attention", "kv_cache_type") else raw
    check_tuning({key: value})
    return value


def cmd_models(args: argparse.Namespace) -> int:
    """``bolo models config <id> [--set key=value ...] [--unset key ...] [--phonemizer X] [--clear]``: settings
    pinned for one model (the same as ``PUT /api/v1/models/{id}/settings``)."""
    from bolo.errors import BoloError

    core = _core(args)
    spec = core.manager.resolve(args.model)
    known = core.store.MODEL_SETTINGS
    try:
        values: dict[str, Any] = {}
        if args.clear:  # every pin goes (then --set and --phonemizer, if given, pin again)
            values.update(dict.fromkeys(core.store.model_settings(spec.id)))
        if args.phonemizer:
            core.components.validate_pin(spec, args.phonemizer)
            values["phonemizer"] = args.phonemizer
        for item in args.set or []:
            key, sep, raw = item.partition("=")
            key = key.strip().replace("-", "_")
            if not sep or key not in known:
                print(
                    f"error: --set expects key=value with key one of {', '.join(known)}; got {item!r}", file=sys.stderr
                )
                return 1
            if key == "phonemizer":
                core.components.validate_pin(spec, raw)
            values[key] = _model_setting(key, raw)
        for key in args.unset or []:
            key = key.strip().replace("-", "_")
            if key not in known:
                print(f"error: unknown model setting {key!r} (one of {', '.join(known)})", file=sys.stderr)
                return 1
            values[key] = None
        if values:
            core.store.set_model_settings(spec.id, values)
    except BoloError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    print(json.dumps({"model": spec.id, "settings": core.store.model_settings(spec.id)}, indent=2))
    return 0


def _client_get(args: argparse.Namespace, path: str) -> Any:
    import httpx

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    return httpx.get(f"{args.url.rstrip('/')}{path}", headers=headers, timeout=10.0)


def _gb_pair(used: int | None, total: int | None) -> str:
    if total is None:
        return "memory unknown"
    gb = 1024**3
    return f"{(used or 0) / gb:.1f}/{total / gb:.1f} GB"


def format_ps(doc: dict[str, Any], verbose: bool = False) -> str:
    """``bolo ps``: a header line for the machine, then Ollama's columns (MODEL, TASK, SIZE, PROCESSOR, DEVICE, GPU,
    UNTIL); ``verbose`` adds ACTIVE (requests and sessions using the model), QUEUED (requests waiting for it),
    PARALLEL (``num_parallel``), REQUESTS, latency and real-time factor."""
    system = doc.get("system") or {}
    head = [f"cpu {system.get('cpu_percent', 0)}%", f"rss {_human(system.get('rss_bytes'))}"]
    if system.get("ram_total_bytes"):
        total, avail = system["ram_total_bytes"], system.get("ram_available_bytes") or 0
        head.append(f"ram {_gb_pair(total - avail, total)}")
    gpus = system.get("gpus") or []
    for g in gpus:
        label = g.get("name") or g.get("vendor") or "GPU"
        if g.get("unified"):
            mem = f"unified memory {_gb_pair(g.get('used_bytes'), g.get('total_bytes'))}"
        else:
            mem = f"vram {_gb_pair(g.get('used_bytes'), g.get('total_bytes'))}"
            if g.get("process_bytes") is not None:
                mem += f" (bolo {_human(g['process_bytes'])})"
        util = f" {g['utilization']:.0f}% busy" if g.get("utilization") is not None else ""
        skipped = " (not used: gpus setting)" if g.get("allowed") is False else ""
        head.append(f"gpu{g.get('index', 0)} {label} {mem}{util}{skipped}")
    if not gpus:
        head.append("no GPU memory readings" if system.get("device", "cpu") != "cpu" else "device cpu")
    rows = []
    for m in doc.get("models") or []:
        gpus = m.get("gpus") or (m.get("placement") or {}).get("gpus") or []
        row = [
            m["name"], m.get("task", ""), _human(m.get("size")), m.get("processor", "100% CPU"),
            m.get("device", "cpu"), ",".join(str(g) for g in gpus) or "-", m.get("until", ""),
        ]  # fmt: skip
        if verbose:
            lat = (m.get("latency_ms") or {}).get("final") or (m.get("latency_ms") or {}).get("ttfa") or {}
            row += [
                str(m.get("active", 0)), str(m.get("queued", 0)), str(m.get("num_parallel", "-")),
                str(m.get("requests", 0)), f"{lat.get('p50', '-')}", f"{(m.get('rtf') or {}).get('p50', '-')}",
                f"{m.get('idle_s', 0)}s",
            ]  # fmt: skip
        rows.append(row)
    headers = ["MODEL", "TASK", "SIZE", "PROCESSOR", "DEVICE", "GPU", "UNTIL"]
    if verbose:
        headers += ["ACTIVE", "QUEUED", "PARALLEL", "REQUESTS", "P50 MS", "RTF P50", "IDLE"]
    lines = ["  ".join(head)]
    lines.append(_table(rows, headers) if rows else "no models loaded")
    notices = [
        f"{m['name']}: {m['placement']['notice']}"
        for m in doc.get("models") or []
        if (m.get("placement") or {}).get("notice")
    ]
    lines += [f"note: {n}" for n in notices] if verbose else []
    return "\n".join(lines)


def _ps_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """An older server has no /api/v1/ps: the same table from /api/v1/stats."""
    models = []
    for m in stats.get("models", []):
        if not m.get("loaded"):
            continue
        ttl = m.get("ttl_s")
        left = None if ttl is None or ttl < 0 else max(0.0, ttl - (m.get("idle_s") or 0))
        from bolo.keepalive import humanize_until

        models.append({
            "name": m["id"], "task": m["task"], "size": m["memory_bytes"], "processor": m.get("processor", "100% CPU"),
            "device": m.get("device", "cpu"), "until": humanize_until(left), "active": m["active"],
            "queued": (m.get("queue") or {}).get("depth", 0), "num_parallel": (m.get("queue") or {}).get("workers"),
            "requests": m["requests"], "latency_ms": m["latency_ms"], "rtf": m.get("rtf"), "idle_s": m.get("idle_s"),
        })  # fmt: skip
    return {"models": models, "system": stats.get("system") or {}}


def cmd_ps(args: argparse.Namespace) -> int:
    import httpx

    try:
        r = _client_get(args, "/api/v1/ps")
        if r.status_code == 404:
            doc = _ps_from_stats(_client_get(args, "/api/v1/stats").raise_for_status().json())
        else:
            doc = r.raise_for_status().json()
    except httpx.HTTPError as exc:
        print(f"could not reach a bolo server at {args.url}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(doc, indent=2))
        return 0
    print(format_ps(doc, verbose=args.verbose))
    return 0


def _post(args: argparse.Namespace, path: str, body: dict[str, Any] | None = None) -> Any:
    import httpx

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    return httpx.post(f"{args.url.rstrip('/')}{path}", headers=headers, json=body, timeout=None)


def cmd_load(args: argparse.Namespace) -> int:
    """``bolo load <model> [--keep-alive 10m]`` (also ``bolo run``): load into a running server now."""
    from urllib.parse import quote

    import httpx

    from bolo.keepalive import format_keep_alive, parse_keep_alive

    if args.keep_alive is not None:
        parse_keep_alive(args.keep_alive)  # a typo fails here, not after a long load
    body = {"keep_alive": args.keep_alive} if args.keep_alive is not None else None
    try:
        r = _post(args, f"/api/v1/models/{quote(args.model, safe='')}/load", body)
    except httpx.HTTPError as exc:
        print(f"could not reach a bolo server at {args.url}: {exc}", file=sys.stderr)
        return 1
    doc = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code >= 400:
        print(f"error: {(doc.get('error') or {}).get('message') or r.text}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(doc, indent=2))
        return 0
    if not doc.get("loaded"):
        print(f"{doc.get('id', args.model)} loaded and unloaded again (keep_alive 0)")
        return 0
    keep = format_keep_alive(doc.get("keep_alive_s"))
    print(
        f"{doc['id']}: {_human(doc.get('memory_bytes'))}, {doc.get('processor', '100% CPU')} on "
        f"{doc.get('device', 'cpu')}, loaded in {doc.get('load_ms', 0) / 1000:.1f} s, keep alive {keep}"
    )
    if doc.get("notice"):
        print(f"note: {doc['notice']}")
    for w in doc.get("warnings") or []:
        print(f"warning: {w.get('message', w) if isinstance(w, dict) else w}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """``bolo stop <model>``: unload now (a request still using it finishes first)."""
    from urllib.parse import quote

    import httpx

    status = 0
    for name in args.models:
        try:
            r = _post(args, f"/api/v1/models/{quote(name, safe='')}/unload")
        except httpx.HTTPError as exc:
            print(f"could not reach a bolo server at {args.url}: {exc}", file=sys.stderr)
            return 1
        doc = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code >= 400:
            print(f"error: {(doc.get('error') or {}).get('message') or r.text}", file=sys.stderr)
            status = 1
        elif doc.get("unloaded") is False:
            print(f"{doc.get('id', name)} was not loaded")
        else:
            print(f"unloaded {doc.get('id', name)}")
    return status


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
                    text, model=args.model, voice=args.voice, speed=args.speed, trace=trace,
                    romanised=args.romanised,
                ):  # fmt: skip
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
    sim = result.get("speaker_similarity")
    if isinstance(sim, dict):
        if sim.get("similarity") is not None:
            print(
                f"speaker similarity {sim['similarity']:.3f} ({sim.get('verdict')}) against {sim.get('reference')}, "
                f"by {sim.get('embedding_model')}"
            )
        else:
            print(f"speaker similarity: not measured ({sim.get('note')})")
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


def cmd_voices(args: argparse.Namespace) -> int:
    """``bolo voices list|add|show|rm|export|import|score``: the saved voices in the data directory (a running
    server sees changes at once: it reads the same folder)."""
    import getpass
    import socket

    from bolo.audio import decode_audio
    from bolo.voices import clean_reference, store_for

    settings = _settings(args)
    store = store_for(Path(settings.data_dir))
    who = {"auth": "cli", "name": getpass.getuser(), "client": socket.gethostname()}
    action = "list" if args.action == "ls" else args.action
    if action == "list":
        voices = store.all()
        if args.json:
            print(json.dumps([v.to_dict() for v in voices], indent=2, ensure_ascii=False))
            return 0
        if not voices:
            print("no saved voices (bolo voices add <name> <file> --consent self|permission)")
            return 0
        rows = [
            [
                v.id, v.name, v.language or "-", f"{v.meta.get('reference', {}).get('duration_s', 0):.1f}s",
                str((v.meta.get("consent") or {}).get("basis", "-")), v.default_model or "-",
                _ago(v.meta.get("created_at")),
            ]
            for v in voices
        ]  # fmt: skip
        print(_table(rows, ["ID", "NAME", "LANGUAGE", "REFERENCE", "CONSENT", "DEFAULT MODEL", "CREATED"]))
        return 0
    if action == "add":
        if not args.name or not args.files:
            print("usage: bolo voices add <name> <file> [<file> ...] --consent self|permission", file=sys.stderr)
            return 2
        if not args.consent:
            print(
                "error: creating a voice needs explicit consent: --consent self (this is my voice) or --consent "
                "permission (I have the speaker's permission)",
                file=sys.stderr,
            )
            return 2
        clips = [decode_audio(Path(f).read_bytes(), filename=Path(f).name) for f in args.files]
        cleaned = clean_reference(clips)
        default_model = None
        if args.default_model:
            from bolo.catalog import Catalog

            spec = Catalog.load(Path(settings.data_dir)).resolve(args.default_model)
            if spec is None or "voice-cloning" not in spec.features:
                print(f"error: '{args.default_model}' is not a model that clones voices", file=sys.stderr)
                return 1
            default_model = spec.id
        voice = store.create(
            name=args.name, cleaned=cleaned, consent=args.consent, created_by=who, voice_id=args.id,
            transcript=args.transcript, language=args.language, description=args.description,
            default_model=default_model, source="cli",
        )  # fmt: skip
        ref = voice.meta.get("reference", {})
        print(f"saved voice {voice.id} ({ref.get('duration_s')} s of reference, consent: {args.consent})")
        for w in voice.meta.get("warnings") or []:
            print(f"warning: {w}")
        print(f'use it: {{"model": "<a model that clones voices>", "voice": "{voice.id}", "input": "..."}}')
        return 0
    if action in ("show", "rm", "score") and not args.name:
        print(f"usage: bolo voices {action} <id>", file=sys.stderr)
        return 2
    if action == "show":
        print(json.dumps(store.get(args.name).to_dict(), indent=2, ensure_ascii=False))
        return 0
    if action == "rm":
        store.delete(args.name)
        print(f"removed voice {args.name}")
        return 0
    if action == "export":
        ids = ([args.name] if args.name else []) + list(args.files or [])
        out = Path(args.output or "bolo-voices.zip")
        out.write_bytes(store.export_zip(ids or None))
        print(f"wrote {out}")
        return 0
    if action == "import":
        if not args.name:
            print("usage: bolo voices import <file.zip> [--consent ...] [--replace]", file=sys.stderr)
            return 2
        added = store.import_zip(Path(args.name).read_bytes(), imported_by=who, consent=args.consent,
                                 replace=args.replace)  # fmt: skip
        print(f"imported {len(added)} voice(s): {', '.join(added) or '-'}")
        return 0
    # score: speak a test sentence with a model, in process, and compare speakers
    from bolo import similarity

    core = _core(args)

    async def run() -> dict[str, Any]:
        await core.start()
        try:
            return await similarity.score_voice(core, args.name, args.model, args.text)
        finally:
            await core.stop()

    result = asyncio.run(run())
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(
            f"{result['voice']} with {result['model']}: speaker similarity {result['similarity']:.3f} "
            f"({result['verdict']}; {result['embedding_model']}), RTF {result['rtf']}, first audio "
            f"{result['ttfa_ms']} ms"
        )
    store.record_score(args.name, result["model"], {k: v for k, v in result.items() if k not in ("voice", "text")})
    return 0


def cmd_components(args: argparse.Namespace) -> int:
    core = _core(args)
    manager = core.components
    if args.action == "list":
        for c in manager.status():
            state = f"installed ({c['source']}, {c['version']})" if c["installed"] else "not installed"
            print(f"{c['id']:<12} {state:<40} {c['license']:<18} needed by {c['required_by_count']} catalog models")
            if not c["installed"] and c.get("kind") in ("phonemizer", "language-pack"):
                flag = " --accept-license" if c["license"] not in ("Apache-2.0",) else ""
                print(f"             install: bolo components install {c['id']}{flag}   (or: {c['manual_command']})")
            elif not c["installed"]:
                print(f"             install: {c['manual_command']}")
        return 0
    listed = manager.pack_status()
    packs = tuple(c["id"] for c in listed if c["license"] != "Apache-2.0")
    installable = ("espeak-ng", "uccharan", "llama-server", "audio.cpp", *(c["id"] for c in listed))
    if args.component not in installable:
        print(f"error: unknown component '{args.component}' (installable: {', '.join(installable)})", file=sys.stderr)
        return 1
    if manager.is_installed(args.component):
        print(f"{args.component} is already installed")
        return 0
    if args.component == "espeak-ng" and not args.accept_license:
        print(
            "error: eSpeak NG is licensed GPL-3.0-or-later and is not part of bolo. Re-run with --accept-license "
            "to install it into this Python environment.",
            file=sys.stderr,
        )
        return 1
    if args.component in packs and not args.accept_license:
        print(
            "error: this uccharan pack is licensed CC BY-SA 4.0 (it is built from share-alike data). Re-run with "
            "--accept-license to download it.",
            file=sys.stderr,
        )
        return 1
    if args.component in ("llama-server", "audio.cpp"):
        blocked = manager.runtime_blocked_reason(args.component)
        if blocked:
            print(f"error: cannot install {args.component}: {blocked}", file=sys.stderr)
            return 1
    error = manager.install_blocking(component=args.component)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.component.startswith("uccharan-xlit-"):
        print(f"installed {args.component}; romanised text for this language can now be converted")
        return 0
    if args.component.startswith("uccharan-"):
        print(f"installed {args.component} (SHA-256 checked)")
        return 0
    print(f"installed {args.component}; voices that need it are now available")
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    """Cloud providers from the configuration and <data_dir>/providers.json. Keys are never printed."""
    registry = _core(args).providers
    if args.action == "list":
        providers = registry.all()
        if not providers:
            print("no cloud providers are configured (docs/providers.md); nothing leaves this server")
            return 0
        for p in providers:
            state = "configured" if p.configured else "no API key"
            print(f"{p.name:<16} {p.kind:<18} {state:<12} {p.source:<9} {p.base_url}")
        return 0
    if not args.provider:
        print("error: name the provider to test", file=sys.stderr)
        return 1
    if registry.get(args.provider) is None:
        print(f"error: unknown provider '{args.provider}'", file=sys.stderr)
        return 1
    result = registry.test(args.provider)
    what = {"credentials": "key accepted", "reachability": "address reachable (the key was not proven)"}
    if result["ok"]:
        print(f"{result['name']}: {what.get(str(result['checked']), 'ok')} in {result['latency_ms']:.0f} ms")
        print(f"  {result['message']}")
        return 0
    print(f"{result['name']}: failed ({result['code']}): {result['message']}", file=sys.stderr)
    return 1


def _print_report(report: dict[str, Any]) -> None:
    caller, latency = report["caller"], report.get("latency") or {}
    verdict = "PASS" if report["passed"] else "FAIL"
    print(f"{report['scenario']}: {verdict}  ({report['provider']}, agent {report['agent']}, {report['duration_s']} s, "
          f"outcome {report.get('outcome')})")  # fmt: skip
    heard = caller.get("response_ms")
    if heard:
        print(f"  caller heard the agent after: p50 {heard['p50']:.0f} ms, max {heard['max']:.0f} ms "
              "(end of the caller's speech to the first audio at the provider)")  # fmt: skip
    parts = [(n, latency[n]["p50"]) for n in ("endpoint_ms", "stt_ms", "llm_first_token_ms", "tts_first_audio_ms",
                                              "send_ms") if isinstance(latency.get(n), dict)]  # fmt: skip
    if parts:
        print("  server breakdown (p50): " + "  ".join(f"{n[:-3]} {v:.0f} ms" for n, v in parts))
    print(f"  agent audio heard {caller['agent_audio_s']} s; barge-in clears {caller['clears']} "
          f"({caller['cleared_s']} s dropped)")  # fmt: skip
    for check in report["checks"]:
        print(f"  [{'ok' if check['ok'] else 'FAILED'}] {check['type']}: {check['detail']}")
    for line in (report.get("record") or {}).get("transcript") or []:
        print(f"    {line['t_ms'] / 1000:6.1f}s {line['speaker']:<6} {line['text']}")


def cmd_telephony(args: argparse.Namespace) -> int:
    """Phone calls (docs/telephony.md): what is configured, and the call simulator."""
    if args.action == "simulate":
        from bolo.telephony.simulator import SCENARIOS, simulate_with_server

        settings = _settings(args)
        _setup_logging(settings.log_level if args.log_level else "warning")
        names = list(SCENARIOS) if args.all else (args.scenario or ["basic"])
        reports = simulate_with_server(
            settings, names, provider=args.provider, agent=args.agent, noise_db=args.noise_db, direct=args.direct
        )
        if args.json:
            print(json.dumps(reports, indent=2, ensure_ascii=False))
        else:
            for report in reports:
                _print_report(report)
        return 0 if all(r["passed"] for r in reports) else 1
    overview = _core(args).telephony.overview()
    if args.json:
        print(json.dumps(overview, indent=2, ensure_ascii=False))
        return 0
    print(f"public_base_url: {overview['public_base_url'] or 'not set'}   "
          f"signature validation: {'on' if overview['validate_signatures'] else 'OFF'}")  # fmt: skip
    for p in overview["providers"]:
        state = "ready" if p["exposed"] and p["configured"] else "closed" if not p["exposed"] else "not configured"
        print(f"{p['label']}: {state}" + ("" if p["exposed"] else f" ({'; '.join(p['problems'])})"))
        for hook in p["webhooks"]:
            print(f"  {hook['name']}: {hook['method']} {hook['url']}")
    agents = ", ".join(a["name"] for a in overview["agents"]) or "none"
    print(f"agents: {agents}   default: {overview['default_agent'] or 'none'}")
    for number in overview["numbers"]:
        print(f"  {number['number']} -> {number['agent']}")
    return 0


# -- hardware and recommendations ---------------------------------------------------------------------
_TASK_TITLES = {
    "stt": "Speech to text", "tts": "Text to speech", "diarization": "Speaker diarization",
    "vad": "Voice-activity detection", "turn": "End-of-turn detection", "separation": "Overlapped-speech separation",
    "speech-to-speech": "Speech to speech",
}  # fmt: skip


def _unknown(value: Any, unit: str = "") -> str:
    return "unknown" if value is None else f"{value}{unit}"


def _cores(value: float | None) -> str:
    return "unknown" if value is None else f"{value:g}"


def _hardware_lines(hw: dict[str, Any]) -> list[str]:
    """This machine, as lines of text. What could not be read is printed as "unknown"."""
    from bolo.hardware import gigabytes

    cpu, mem, disk, box = hw["cpu"], hw["memory"], hw["disk"], hw["container"]
    features = "unknown" if cpu["features"] is None else (", ".join(cpu["features"]) or "none that matter")
    lines = [
        f"cpu: {_unknown(cpu['model'])} ({_unknown(cpu['architecture'])}); cores: {_unknown(cpu['physical_cores'])} "
        f"physical, {_unknown(cpu['logical_cores'])} logical, {_cores(cpu['usable_cores'])} usable; "
        f"inference threads: {_unknown(hw['threads'])}",
        f"instruction sets: {features}",
        f"memory: {gigabytes(mem['total_bytes']) or 'unknown'} total, "
        f"{gigabytes(mem['available_bytes']) or 'unknown'} available"
        + (f", limited to {gigabytes(mem['limit_bytes'])} by the container" if mem["limit_bytes"] else ""),
        f"disk: {gigabytes(disk['free_bytes']) or 'unknown'} free in {_unknown(disk['path'])}",
    ]
    if hw["gpus"]:
        for g in hw["gpus"]:
            memory = (
                "memory shared with the system"
                if g["unified_memory"] and g["vendor"] != "amd"
                else f"{gigabytes(g['vram_total_bytes']) or 'unknown'} VRAM, "
                f"{gigabytes(g['vram_free_bytes']) or 'unknown'} free"
            )
            use = f"usable through {g['backend']}" if g["backend"] else "no runtime bolo starts uses it"
            if g.get("backend") not in (None, "dml") and g.get("dml_index") is not None:
                use += f"; DirectML adapter dml:{g['dml_index']}"
            if g.get("integrated"):
                use += "; integrated graphics"
            driver = f", driver {g['driver']}" if g["driver"] else ""
            lines.append(f"gpu: {_unknown(g['name'])} ({memory}{driver}; {use})")
    else:
        lines.append("gpu: none found" if hw["gpus_known"] else "gpu: unknown (could not be detected)")
    rt = hw["runtimes"]
    parts = []
    ort = rt.get("onnxruntime", {})
    if ort.get("installed"):
        providers = "unknown" if ort.get("providers") is None else ", ".join(ort["providers"])
        parts.append(f"onnxruntime {ort.get('version') or ''} ({providers})")
    else:
        parts.append("onnxruntime not installed")
    sherpa = rt.get("sherpa-onnx", {})
    parts.append(
        f"sherpa-onnx {sherpa.get('version') or ''} (GPU build: unknown)"
        if sherpa.get("installed")
        else "sherpa-onnx not installed"
    )
    ct2 = rt.get("ctranslate2", {})
    parts.append(
        f"CTranslate2 {ct2.get('version') or ''} (CUDA devices: {_unknown(ct2.get('cuda_devices'))})"
        if ct2.get("installed")
        else "CTranslate2 not installed"
    )
    torch = rt.get("plugin:torch", {})
    parts.append(
        f"torch plugin (device: {_unknown(torch.get('device'))})"
        if torch.get("installed")
        else "torch plugin not installed"
    )
    for name in ("llama-server", "audio.cpp"):
        entry = rt.get(name, {})
        if entry.get("installed"):
            gpu = {True: "GPU build", False: "CPU build", None: "GPU support unknown"}[entry.get("gpu")]
            parts.append(f"{name} {entry.get('version') or ''} ({gpu})".replace("  ", " "))
        else:
            parts.append(f"{name} not installed")
    lines.append("runtimes: " + "; ".join(parts))
    where = []
    if box["detected"] is None:
        where.append("container: unknown")
    elif box["detected"]:
        where.append(f"container: {box['kind']}")
    if box["virtual_machine"]:
        where.append("virtual machine" + (f" ({box['hypervisor']})" if box["hypervisor"] else ""))
    if box["cpu_quota"] is not None:
        where.append(f"CPU quota {box['cpu_quota']} cores")
    if box["memory_limit_bytes"]:
        where.append(f"memory limit {gigabytes(box['memory_limit_bytes'])}")
    if where:
        lines.append("environment: " + ", ".join(where))
    lines += [f"note: {note}" for note in hw["notes"]]
    return lines


def _speed_line(speed: dict[str, Any]) -> str:
    return f"speed against the reference machine: {speed['factor']:.2f}x, {speed['note']}"


def _fit_cells(r: dict[str, Any]) -> list[str]:
    from bolo.hardware import gigabytes

    fit = r["fit"]
    if fit["expected_rtf"] is None:
        rtf = "-"
    else:
        lo, hi = fit["rtf_range"]
        rtf = f"{fit['expected_rtf']:g} ({lo:g}-{hi:g})"
    basis = {"measured": "measured", "estimated": "estimated"}.get(fit["rtf_basis"], "guess")
    if fit["level"] in ("cloud", "unavailable"):
        basis = "-"
    mem = fit["memory"]
    memory = "-"
    if mem and mem["needed_bytes"] is not None:
        have = gigabytes(mem["available_bytes"]) or "unknown"
        memory = f"{gigabytes(mem['needed_bytes'])} of {have}" + (" VRAM" if mem["kind"] == "vram" else "")
    q = r["quality"]
    quality = f"{q['metric'].replace('round_trip_', 'rt-').upper()} {q['value']:.1%}" if q else r["tier"]
    licence = r["license"] + ("" if r["commercial_use"] else " (non-commercial)")
    mark = "*" if r["default"] else ("+" if r["recommended"] else "")
    return [str(r["rank"]), f"{r['id']} {mark}".rstrip(), fit["label"], rtf, basis, memory, quality, licence]


def _print_ranking(answer: dict[str, Any], limit: int | None, why: bool) -> None:
    q = answer["query"]
    title = _TASK_TITLES.get(q["task"], q["task"])
    asked = [
        x
        for x in (
            q["language"],
            "commercial use only" if q["commercial"] else None,
            "local only" if q["local_only"] else None,
            {True: "real time", False: "batch"}.get(q["realtime"]),
            f"at most {q['max_size_bytes'] // (1024 * 1024)} MB" if q["max_size_bytes"] else None,
        )
        if x
    ]
    hidden = sum(answer["filtered_out"].values())
    print(
        f"{title}{' (' + ', '.join(asked) + ')' if asked else ''}: {answer['total']} models"
        + (f"; {hidden} more do not pass the filters you asked for" if hidden else "")
    )
    results = answer["results"]
    shown = results if limit is None else results[:limit]
    if not shown:
        print("  no catalog model matches.")
        return
    headers = ["#", "MODEL", "FIT", "RTF HERE (RANGE)", "BASIS", "MEMORY", "QUALITY", "LICENCE"]
    print(_table([_fit_cells(r) for r in shown], headers))
    if len(shown) < len(results):
        print(
            f"... and {len(results) - len(shown)} more, the least suitable last (--all lists every one; "
            "nothing is left out because of hardware)."
        )
    print("* default pick   + also recommended for this machine")
    if answer["note"]:
        print(answer["note"])
    for r in shown:
        if why or r["default"] or r["recommended"]:
            notes = "".join(f" Note: {n}." for n in r["fit"]["notes"])
            print(f"  {r['id']}: {r['fit']['reason']}{notes}")


def _print_overview(core: Any, language: str | None, commercial: bool, local_only: bool) -> None:
    """The default pick per task: what `bolo recommend` prints without --task, and `bolo doctor` in short."""
    from bolo.recommend import TASKS, Query

    rows = []
    for task, live in [("stt", None), ("stt", True), ("tts", None), ("tts", True), *[(t, None) for t in TASKS[2:]]]:
        answer = core.advisor.recommend(Query(task, language, commercial, local_only, live))
        by_id = {r["id"]: r for r in answer["results"]}
        pick = by_id.get(answer["default"]) if answer["default"] else None
        name = _TASK_TITLES[task] + (", live" if live else "")
        if pick is None:
            rows.append([name, "(nothing suitable)", "-", "-", "-", ""])
            continue
        cells = _fit_cells(pick)
        also = ", ".join(i for i in answer["recommended"] if i != pick["id"])
        rows.append([name, pick["id"], cells[2], cells[3], cells[4], also])
    print(_table(rows, ["FOR", "DEFAULT PICK", "FIT", "RTF HERE (RANGE)", "BASIS", "ALSO RECOMMENDED"]))


def cmd_recommend(args: argparse.Namespace) -> int:
    """``bolo recommend``: this machine, and the catalog models that suit it. Advice only: nothing is restricted."""
    from bolo.recommend import TASKS, USES, Query, normalize_task

    core = _core(args)
    advisor = core.advisor
    advisor.hardware(refresh=True, deep=True)
    if args.benchmark:
        print("running the micro-benchmark (a few seconds)...", file=sys.stderr)
        result = advisor.benchmark()
        where = "not stored: " + result["not_saved"] if "not_saved" in result else f"stored in {core.settings.data_dir}"
        print(
            f"benchmark: {result['workload']} workload, {result['threads']} threads, {result['ms']} ms per run "
            f"(reference {result['reference_ms']} ms) -> {result['speed']}x the reference machine; {where}",
            file=sys.stderr,
        )
    realtime = True if args.realtime else (False if args.batch else None)
    max_size = int(args.max_size_mb * 1024 * 1024) if args.max_size_mb else None
    task = normalize_task(args.task) if args.task else None
    if task is not None and task not in TASKS:
        print(f"error: unknown task {args.task!r}; use one of: {', '.join(TASKS)}", file=sys.stderr)
        return 1
    if args.use and args.use not in USES:
        print(f"error: unknown use {args.use!r}; use one of: {', '.join(USES)}", file=sys.stderr)
        return 1
    limit = None if args.all else args.limit
    if args.json:
        doc: dict[str, Any] = {"hardware": advisor.hardware_api()}
        if args.use:
            doc["setup"] = advisor.setup(args.use, args.language, args.commercial, not args.cloud)
        elif task:
            query = Query(task, args.language, args.commercial, args.local_only, realtime, max_size)
            doc["recommendations"] = advisor.recommend(query, limit)
        else:
            doc["recommendations"] = {
                t: advisor.recommend(Query(t, args.language, args.commercial, args.local_only, realtime, max_size), 5)
                for t in TASKS
            }
        print(json.dumps(doc, indent=2, ensure_ascii=False))
        return 0
    hw = advisor.hardware_api()
    print("this machine:")
    for line in _hardware_lines(hw):
        print(f"  {line}")
    print(f"  {_speed_line(hw['speed'])}")
    if hw["speed"]["source"] != "benchmark" and "stored benchmark" not in hw["speed"]["note"]:
        print("  (no benchmark stored: real-time factors below are rougher; run `bolo recommend --benchmark`)")
    print()
    if args.use:
        setup = advisor.setup(args.use, args.language, args.commercial, not args.cloud)
        print(f"{setup['label']}{' in ' + args.language if args.language else ''}:")
        rows = []
        for pick in setup["picks"]:
            m = pick["model"]
            name = _TASK_TITLES[pick["task"]] + (" (optional)" if pick["optional"] else "")
            if m is None:
                rows.append([name, "(nothing suitable)", "-", "-", "-", "-"])
            else:
                cells = _fit_cells(m)
                rows.append(
                    [name, m["id"], cells[2], cells[3], cells[4], "yes" if m["installed"] else _human(m["size_bytes"])]
                )
        print(_table(rows, ["FOR", "MODEL", "FIT", "RTF HERE (RANGE)", "BASIS", "INSTALLED / DOWNLOAD"]))
        if setup["pull"]:
            print(f"to install them ({_human(setup['download_bytes'])}):")
            for model_id in setup["pull"]:
                print(f"  bolo pull {model_id}")
    elif task:
        query = Query(task, args.language, args.commercial, args.local_only, realtime, max_size)
        _print_ranking(advisor.recommend(query), limit, args.why)
    else:
        print("recommended for this machine (`bolo recommend --task stt` ranks every model of a task):")
        _print_overview(core, args.language, args.commercial, args.local_only)
    print()
    print("These are recommendations, not restrictions: every catalog model can still be pulled and loaded.")
    return 0


def _doctor_hardware(args: argparse.Namespace) -> None:
    """The machine and a short recommended setup. Never fails the doctor."""
    try:
        core = _core(args)
        hw = core.advisor.hardware_api(refresh=True)
        print("this machine:")
        for line in _hardware_lines(hw):
            print(f"  {line}")
        print(f"  {_speed_line(hw['speed'])}")
        print("recommended setup for this machine (advice only; `bolo recommend` has the details):")
        from bolo.recommend import Query

        for label, task, live in (
            ("speech to text", "stt", None), ("live speech to text", "stt", True), ("text to speech", "tts", None),
            ("live text to speech", "tts", True), ("diarization", "diarization", None),
        ):  # fmt: skip
            answer = core.advisor.recommend(Query(task, realtime=live), 0)
            pick = next((r for r in answer["results"] if r["id"] == answer["default"]), None)
            if pick is None:
                print(f"  {label}: nothing in the catalog is expected to run well here")
                continue
            fit = pick["fit"]
            rtf = f", real-time factor about {fit['expected_rtf']:g}" if fit["expected_rtf"] is not None else ""
            basis = {"measured": "measured, scaled", "estimated": "estimated"}.get(fit["rtf_basis"], "a guess")
            print(f"  {label}: {pick['id']} ({fit['label']}{rtf}; {basis})")
    except Exception as exc:  # pragma: no cover - advice must never break the doctor
        print(f"this machine: could not be described ({type(exc).__name__}: {exc})")


def _doctor_providers(settings: Settings) -> None:
    from bolo.providers import ProviderRegistry

    providers = ProviderRegistry(settings).all()
    if not providers:
        print("cloud providers: none configured (nothing leaves this server)")
        return
    print("cloud providers (paid cloud APIs; `bolo providers test <name>` checks one):")
    for p in providers:
        print(f"  {p.name}: {p.kind} at {p.base_url}, {'configured' if p.configured else 'no API key'}")


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


def _doctor_packs(settings: Any) -> None:
    """uccharan's language packs: each with its size, licence and whether it is on this machine."""
    from bolo.text import packs

    listed = packs.all_packs()
    if not listed:
        return
    automatic = "on" if settings.uccharan_auto_download and os.environ.get("UCCHARAN_OFFLINE") != "1" else "off"
    print(
        f"  language packs ({len(listed)}; cache {packs.cache_dir() or 'unknown'}; uccharan_auto_download {automatic}):"
    )
    for pack in listed:
        state = "installed" if packs.installed(pack.name) else "not installed"
        if pack.name in packs.BUILT_IN:
            state = "built in"
        note = "  (on request only)" if pack.share_alike else ""
        print(f"    {pack.component:<24} {packs.human_size(pack.size):>8}  {pack.license:<13} {state}{note}")


def _doctor_onnxruntime(settings: Settings) -> None:
    from bolo import ort_providers

    packages = ort_providers.installed_ort_packages()
    providers: list[str] | None = None
    if packages:
        try:
            import onnxruntime

            providers = list(onnxruntime.get_available_providers())
        except Exception as exc:
            print(f"onnxruntime: installed but cannot be loaded ({type(exc).__name__}: {exc})")
    if providers is not None:
        listed = ", ".join(f"{n} {v}" for n, v in sorted(packages.items()))
        print(f"onnxruntime: {listed}; providers {', '.join(p.removesuffix('ExecutionProvider') for p in providers)}")
    elif not packages:
        print("onnxruntime: not installed (Nemotron diarization, DiCoW, Orpheus-family voices and the end-of-turn "
              "model need it: pip install 'bolo-speech[turn]')")  # fmt: skip
    for problem in ort_providers.ort_problems(settings.device, packages, providers):
        print(f"  warning: {problem}")


def cmd_doctor(args: argparse.Namespace) -> int:
    import platform

    from bolo.audio import decode, encode
    from bolo.audio.resample import HAVE_SOXR
    from bolo.engines import EngineRegistry
    from bolo.text.phonemize import UCCHARAN_LANGUAGES, get_phonemizer, get_uccharan

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
    for st in EngineRegistry(include_fake=False, settings=settings).statuses():
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
    uc = get_uccharan()
    if uc:
        print(
            f"uccharan: found (version {uc.version() or 'unknown'}) - phonemiser and text normaliser for voices in "
            f"{len(UCCHARAN_LANGUAGES)} language tags"
        )
        _doctor_packs(settings)
    else:
        print(
            "uccharan: not installed (Piper, Kitten and Kokoro voices in its languages need it or espeak-ng: "
            "`pip install 'bolo-speech[uccharan]'`)"
        )
    print(f"phonemizer setting: {settings.phonemizer}")
    ph = get_phonemizer()
    if ph:
        print(f"espeak-ng: found ({ph.source}, version {ph.version() or 'unknown'}) - optional, GPL-3.0-or-later")
    else:
        print(
            "espeak-ng: not installed (optional, GPL-3.0-or-later; phoneme-based voices in languages uccharan does "
            "not cover need it: `bolo components install espeak-ng --accept-license`)"
        )
    print(f"logs: {_log_where(settings)}")
    _doctor_onnxruntime(settings)
    _doctor_llm(settings)
    _doctor_providers(settings)
    _doctor_hardware(args)
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


def _log_where(settings: Settings, system: str | None = None, environ: dict[str, str] | None = None) -> str:
    """Where the server's log goes, for ``bolo doctor`` and ``bolo logs``, in this platform's terms (``system``:
    a ``sys.platform`` value, for tests)."""
    from bolo import logs

    name = (system or sys.platform).lower()
    path = logs.resolve_log_file(settings.log_file, environ=environ, serving=False, system=name)
    default = logs.default_log_path(environ, name)
    if path is not None:
        follow = "; `bolo logs -f` follows it" if name.startswith("win") else ""
        return f"{path} (log_file; rotated at {settings.log_max_mb} MB, {settings.log_backups} old files kept){follow}"
    found = f"; a service's log file is at {default}" if default.is_file() else ""
    if name.startswith("win"):
        return (
            "standard error of `bolo serve` (the console window it runs in; a scheduled task or a service has none)"
            f'{found}. Set log_file = "auto" (BOLO_LOG_FILE=auto) to also write {default}, which `bolo logs -f` '
            "follows; deploy/README.md has Task Scheduler, NSSM and WinSW recipes that do"
        )
    if name == "darwin":
        return (
            "standard error of `bolo serve` (the terminal; under launchd or `brew services`, the job's "
            f"StandardErrorPath, and bolo also writes {default} there by itself){found}. "
            f'Set log_file = "auto" to always write {default}; `bolo logs -f` follows it'
        )
    return (
        "standard error of `bolo serve` (journalctl -u bolo under systemd, which also gets "
        f'{default}; `docker logs` in a container){found}. Set log_file = "auto" to also write {default}'
    )


def cmd_logs(args: argparse.Namespace) -> int:
    """``bolo logs [-f] [--lines N]``: the server's log file (the running server says which; else the settings,
    else the platform's default place)."""
    from bolo import logs

    path: Path | None = None
    r = _server(args, "GET", "/api/v1/info")
    told = r is not None and r.status_code == 200
    if told and r.json().get("log_file"):
        path = Path(r.json()["log_file"])
    if path is None:
        settings = _settings(args)
        path = logs.resolve_log_file(settings.log_file, serving=False)
        if path is None and logs.default_log_path().is_file():
            path = logs.default_log_path()
        if path is None:
            who = f"the server at {args.url} writes no log file" if told else "no log file is configured"
            print(f"{who}: the log goes to {_log_where(settings)}", file=sys.stderr)
            return 1
    if not path.is_file() and not args.follow:
        print(f"no log file at {path} yet", file=sys.stderr)
        return 1
    if path.is_file():
        for line in logs.tail(path, args.lines):
            print(line)
    if args.follow:
        sys.stdout.flush()
        try:
            for line in logs.follow(path):
                print(line, flush=True)
        except KeyboardInterrupt:
            return 0
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
    s.add_argument("--num-parallel", dest="num_parallel", type=int, default=None,
                   help="requests each model works on at once (default: workers)")  # fmt: skip
    s.add_argument("--gpus", dest="gpus", default=None, help="GPUs bolo may use, e.g. 0,2 (default: all visible)")
    s.add_argument("--gpu-spread", dest="gpu_spread", action="store_const", const=True, default=None,
                   help="spread llama.cpp models over every allowed GPU")  # fmt: skip
    s.add_argument("--log-file", dest="log_file", default=None,
                   help='also write the log to this file ("auto": the platform\'s log directory; "off")')  # fmt: skip
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
    s.add_argument(
        "--accept-license",
        dest="accept_license",
        action="store_true",
        help="uccharan's CC BY-SA 4.0 packs (uccharan-xlit-*, uccharan-wikipron-*): accept the licence",
    )
    s.set_defaults(func=cmd_pull)

    def _remote(s: argparse.ArgumentParser) -> None:
        s.add_argument("--url", default=os.environ.get("BOLO_URL", "http://127.0.0.1:8000"))
        s.add_argument("--api-key", dest="api_key", default=os.environ.get("BOLO_API_KEY"))

    s = sub.add_parser("rm", help="delete downloaded models (refused while loaded, unless --force)")
    _common(s)
    _remote(s)
    s.add_argument("models", nargs="+")
    s.add_argument("--force", "-f", action="store_true", help="unload the model from the running server first")
    s.set_defaults(func=cmd_rm)

    s = sub.add_parser("cp", help="copy a model's settings to a new name (a preset with settings of its own)")
    _common(s)
    _remote(s)
    s.add_argument("source")
    s.add_argument("destination")
    s.set_defaults(func=cmd_cp)

    s = sub.add_parser("list", aliases=["ls"], help="list installed models (--all: the whole catalog)")
    _common(s)
    s.add_argument("--all", "-a", action="store_true")
    s.add_argument("--task", choices=["stt", "tts", "vad"])
    s.add_argument("--query", "-q", help="filter by text")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="one model: licence, files, settings, languages, voices, engine, speed")
    _common(s)
    s.add_argument("model")
    s.add_argument("--license", "--licence", dest="license", action="store_true",
                   help="print the licence (and the licence file shipped with the model, when there is one)")  # fmt: skip
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("models", help="per-model settings: models config <id> --set num_parallel=2")
    _common(s)
    s.add_argument("action", choices=["config"])
    s.add_argument("model")
    s.add_argument("--phonemizer", help="auto | uccharan | uccharan:improved | espeak-ng")
    s.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="pin a setting (repeatable): device, keep_alive, num_parallel, max_queue, flash_attention, "
                   "kv_cache_type, context_size, batch_size, phonemizer, text_normalizer, wordlists, fallback, romanised")  # fmt: skip
    s.add_argument("--unset", action="append", metavar="KEY", help="remove one pinned setting (repeatable)")
    s.add_argument("--clear", action="store_true", help="remove every pinned setting")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("logs", help="print the server's log file (-f: follow it)")
    _common(s)
    _remote(s)
    s.add_argument("--follow", "-f", action="store_true", help="keep printing new lines as they are written")
    s.add_argument("--lines", "-n", type=int, default=100, help="lines to print first (default 100; 0: all)")
    s.set_defaults(func=cmd_logs)

    s = sub.add_parser("ps", help="show models loaded in a running server: size, GPU/CPU split, time left")
    _remote(s)
    s.add_argument("--json", action="store_true")
    s.add_argument("--verbose", "-v", action="store_true",
                   help="also requests active and queued per model, num_parallel, totals and latency")  # fmt: skip
    s.set_defaults(func=cmd_ps)

    for name, aliases, text in (
        ("load", ["run"], "load a model into a running server now (preload), optionally with a keep-alive"),
    ):
        s = sub.add_parser(name, aliases=aliases, help=text)
        _remote(s)
        s.add_argument("model")
        s.add_argument(
            "--keep-alive", dest="keep_alive",
            help="how long it stays loaded once idle: seconds or a duration (10m, 1h); -1 forever; 0 unload at once",
        )  # fmt: skip
        s.add_argument("--json", action="store_true")
        s.set_defaults(func=cmd_load)

    s = sub.add_parser("stop", help="unload models from a running server now")
    _remote(s)
    s.add_argument("models", nargs="+")
    s.set_defaults(func=cmd_stop)

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
    s.add_argument(
        "--romanised",
        "--romanized",
        choices=["off", "auto", "on"],
        help="write an Indian language typed in Latin letters in the voice's script first (default: the "
        "model's setting, else the `romanised` setting)",
    )
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

    s = sub.add_parser(
        "voices",
        help="saved voices for models that clone voices: list, add, show, rm, export, import, score",
        description="Saved voices live in <data_dir>/voices. Creating one needs --consent: self (this is my voice) "
        "or permission (I have the speaker's permission).",
    )
    _common(s)
    s.add_argument("action", choices=["list", "ls", "add", "show", "rm", "export", "import", "score"])
    s.add_argument("name", nargs="?", help="add: the voice's name; show/rm/score: its id; import: the .zip")
    s.add_argument("files", nargs="*", help="add: recordings of the person; export: more ids")
    s.add_argument("--consent", choices=["self", "permission"],
                   help="required for add: self = this is my voice; permission = I have the speaker's permission")  # fmt: skip
    s.add_argument("--transcript", help="what is said in the recording (some models use it)")
    s.add_argument("--language", help="language of the voice (a code such as en)")
    s.add_argument("--description")
    s.add_argument("--default-model", dest="default_model", help="model used when a request names no model")
    s.add_argument("--id", help="add: the id (default: from the name)")
    s.add_argument("--output", "-o", help="export: the .zip to write (default bolo-voices.zip)")
    s.add_argument("--replace", action="store_true", help="import: overwrite voices with the same id")
    s.add_argument("--model", "-m", help="score: the model that speaks (default: the voice's default model)")
    s.add_argument("--text", help="score: what to say (default: a test sentence in the voice's language)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_voices)

    s = sub.add_parser(
        "components",
        help="optional components (uccharan, espeak-ng, llama-server, audio.cpp, engine plugins): list, install",
    )
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

    s = sub.add_parser("providers", help="cloud providers (paid cloud APIs; docs/providers.md): list, test")
    _common(s)
    s.add_argument("action", choices=["list", "test"])
    s.add_argument("provider", nargs="?")
    s.set_defaults(func=cmd_providers)

    s = sub.add_parser("telephony", help="phone calls (docs/telephony.md): status, simulate")
    _common(s)
    s.add_argument("action", choices=["status", "simulate"])
    s.add_argument("--scenario", action="append", help="a built-in scenario or a JSON file (repeatable); default basic")
    s.add_argument("--all", action="store_true", help="run every built-in scenario")
    s.add_argument("--provider", default="twilio", choices=["twilio", "plivo", "telnyx", "exotel"],
                   help="whose protocol the fake provider speaks")  # fmt: skip
    s.add_argument("--agent", help="the agent to call (default: telephony.default_agent)")
    s.add_argument("--noise-db", dest="noise_db", type=float, help="background noise, as a signal-to-noise ratio in dB")
    s.add_argument("--direct", action="store_true", help="skip the webhook and WebSocket: straight into the session")
    s.add_argument("--json", action="store_true", help="print the full reports as JSON")
    s.set_defaults(func=cmd_telephony)

    s = sub.add_parser(
        "recommend",
        help="describe this machine and rank the catalog's models for it (advice only: nothing is restricted)",
    )
    _common(s)
    s.add_argument("--task", help="stt, tts, diarization, vad, turn, separation or speech-to-speech")
    s.add_argument("--language", help="only models that cover this language (a code such as en or hi)")
    s.add_argument("--commercial", action="store_true", help="only models whose licence allows commercial use")
    s.add_argument("--local-only", dest="local_only", action="store_true", help="leave cloud models out")
    s.add_argument("--realtime", action="store_true", help="rank for live speech (stricter speed levels)")
    s.add_argument("--batch", action="store_true", help="rank for files and voice-over: the most capable that keeps up")
    s.add_argument("--max-size-mb", dest="max_size_mb", type=float, help="only models with a download up to this size")
    s.add_argument("--use", help="a whole setup instead of one task: live, files or voiceover")
    s.add_argument("--cloud", action="store_true", help="with --use: let cloud models be proposed too")
    s.add_argument("--limit", type=int, default=15, help="rows to print for a task (default 15)")
    s.add_argument("--all", action="store_true", help="print every model of the task, the least suitable last")
    s.add_argument("--why", action="store_true", help="print the reason for every row, not only the recommended ones")
    s.add_argument("--benchmark", action="store_true",
                   help="first time a tiny fixed workload (a few seconds) and store the result in the data directory")  # fmt: skip
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_recommend)

    s = sub.add_parser("doctor", help="check engines, codecs, components, the language model and the data directory")
    _common(s)
    s.add_argument("--network", action="store_true", help="also check that Hugging Face is reachable")
    s.set_defaults(func=cmd_doctor)
    return p


def utf8_stdio() -> None:
    """Read and write UTF-8 on standard input, output and error on every platform. Python on Windows otherwise uses
    the legacy code page for a pipe or a file redirect (cp1252, cp437, ...), which cannot write Devanagari or other
    non-Latin text (``UnicodeEncodeError``). A Windows console is unaffected (Python writes it as UTF-16)."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None and str(getattr(stream, "encoding", "")).lower().replace("-", "") != "utf8":
            with contextlib.suppress(Exception):  # a replaced stream (pytest's capture, an IDE) keeps its own
                reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
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
