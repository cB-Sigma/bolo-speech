#!/usr/bin/env python3
"""Write the copy of README.md that goes into the Python package.

PyPI shows the package's README on its own site, where relative links and images lead nowhere.
This rewrites them to absolute URLs: links to https://github.com/cB-Sigma/bolo/blob/main/...
(tree/main for directories) and images to https://raw.githubusercontent.com/cB-Sigma/bolo/main/...
Text inside code blocks and code spans is left alone.

    python tools/pypi_readme.py README.md server/README.md           # write
    python tools/pypi_readme.py --check README.md server/README.md   # fail when the copy is stale
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = "https://github.com/cB-Sigma/bolo"
RAW = "https://raw.githubusercontent.com/cB-Sigma/bolo/main"
IMAGE = (".png", ".gif", ".jpg", ".jpeg", ".svg", ".webp")


def absolute(target: str, image: bool = False) -> str:
    if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:|#|//", target) or not target:
        return target
    path, sep, anchor = target.partition("#")
    path = path.lstrip("/") if not path.startswith("../") else path
    while path.startswith("./"):
        path = path[2:]
    if path.startswith("../"):
        return target
    if image or path.lower().endswith(IMAGE):
        return f"{RAW}/{path}"
    kind = "tree" if path.endswith("/") or "." not in path.rsplit("/", 1)[-1] and path not in ("LICENSE", "NOTICE") else "blob"
    return f"{REPO}/{kind}/main/{path}{sep}{anchor}"


def rewrite_text(text: str) -> str:
    text = re.sub(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)", lambda m: f"{m[1]}[{m[2]}]({absolute(m[3], bool(m[1]))})", text)
    text = re.sub(r'\b(src|srcset)="([^"]+)"', lambda m: f'{m[1]}="{absolute(m[2], True)}"', text)
    return re.sub(r'\bhref="([^"]+)"', lambda m: f'href="{absolute(m[1])}"', text)


def rewrite(markdown: str) -> str:
    out: list[str] = []
    fenced = False
    for line in markdown.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            out.append(line)
        elif fenced:
            out.append(line)
        else:  # code spans stay as they are
            parts = re.split(r"(`[^`]*`)", line)
            out.append("".join(p if p.startswith("`") else rewrite_text(p) for p in parts))
    return "\n".join(out)


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--check"]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    source, target = Path(args[0]), Path(args[1])
    text = rewrite(source.read_text(encoding="utf-8"))
    if "--check" in sys.argv:
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            print(f"{target} is stale: run python tools/pypi_readme.py {source} {target}", file=sys.stderr)
            return 1
        return 0
    target.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
