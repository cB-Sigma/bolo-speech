"""``python -m bolo_torch``: serve the plugin protocol on stdin/stdout (what bolo starts)."""

from __future__ import annotations

from bolo_torch.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["serve"]))
