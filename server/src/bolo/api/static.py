"""Serve the built web console from ``bolo/_ui``.

The console is a single-page app: unknown paths fall back to ``index.html``.
Fingerprinted assets are cached for a year; ``index.html`` is always
revalidated. When no bundle is present (for example a source checkout that
has not built the UI) a small placeholder page explains what is going on.
"""

from __future__ import annotations

import html
import mimetypes
from importlib import resources
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from bolo import __version__

router = APIRouter()

_RESERVED = ("api/", "v1/", "metrics", "healthz", "readyz", "health", "docs", "openapi.json", "redoc")

PLACEHOLDER = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bolo</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 16px/1.6 system-ui, sans-serif; max-width: 42rem; margin: 12vh auto; padding: 0 1.5rem; }}
  code, pre {{ font: 14px ui-monospace, monospace; background: rgba(127,127,127,.15); border-radius: 6px; }}
  code {{ padding: .1rem .35rem; }} pre {{ padding: 1rem; overflow-x: auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: .2rem; }} .muted {{ opacity: .7; }}
</style>
</head>
<body>
<h1>bolo <span class="muted">{version}</span></h1>
<p>The speech server is running. The web console bundle is not included in this installation,
so there is nothing to show here, but every API is available.</p>
<pre>curl {base}/api/v1/info
curl {base}/api/v1/catalog
curl {base}/v1/audio/transcriptions -F file=@audio.wav -F model=whisper-1</pre>
<p>API reference: <a href="/docs">/docs</a> (interactive) or <code>docs/api.md</code> in the repository.
To get the console from a source checkout, build it: <code>cd ui &amp;&amp; npm ci &amp;&amp; npm run build</code>.</p>
</body>
</html>
"""


def ui_dir() -> Path:
    return Path(str(resources.files("bolo") / "_ui"))


def has_bundle() -> bool:
    return (ui_dir() / "index.html").is_file()


ISOLATION_HEADERS = {
    # Cross-origin isolation lets the console's in-browser recogniser use WebAssembly threads.
    # "credentialless" keeps its model and runtime downloads from other origins working.
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "credentialless",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def _isolate(request: Request, response: Response) -> Response:
    if request.app.state.core.settings.cross_origin_isolation:
        response.headers.update(ISOLATION_HEADERS)
    return response


def _file_response(path: Path, root: Path) -> Response:
    rel = path.relative_to(root).as_posix()
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if path.name == "index.html":
        cache = "no-cache"
    elif rel.startswith("assets/"):
        cache = "public, max-age=31536000, immutable"
    else:
        cache = "public, max-age=3600"
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": cache})


def _index(request: Request, path: Path, root: Path) -> Response:
    """``index.html``; on a console-only port, with a ``<meta name="bolo-api">`` that tells the page where the API is."""
    pointer = getattr(request.app.state, "console_api", None)
    if not pointer:
        return _file_response(path, root)
    text = path.read_text(encoding="utf-8")
    meta = f'<meta name="bolo-api" content="{html.escape(pointer, quote=True)}" />'
    text = text.replace("</head>", f"  {meta}\n  </head>", 1) if "</head>" in text else meta + text
    return HTMLResponse(text, headers={"Cache-Control": "no-cache"})


@router.get("/{path:path}", include_in_schema=False)
async def console(request: Request, path: str) -> Response:
    if path.startswith(_RESERVED):
        pointer = getattr(request.app.state, "console_api", None)
        where = f"; this port serves only the web console, the API is at {pointer}" if pointer else ""
        return JSONResponse(
            {
                "error": {
                    "message": f"Unknown path '/{path}'{where}",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "not_found",
                },
                "detail": "Not Found",
            },
            status_code=404,
        )
    root = ui_dir().resolve()
    if path:
        candidate = (root / path).resolve()
        if candidate.is_file() and root in candidate.parents:
            if candidate.name == "index.html":
                return _isolate(request, _index(request, candidate, root))
            return _isolate(request, _file_response(candidate, root))
        if "." in path.rsplit("/", 1)[-1]:
            return Response(status_code=404)  # a missing asset is a 404, not the SPA shell
    index = root / "index.html"
    if index.is_file():
        return _isolate(request, _index(request, index, root))
    base = str(request.base_url).rstrip("/")
    return _isolate(
        request, HTMLResponse(PLACEHOLDER.format(version=__version__, base=base), headers={"Cache-Control": "no-cache"})
    )
