"""bolo-torch: the PyTorch engine plugin of the bolo speech server.

bolo never imports PyTorch. It starts this package as a separate process (``python -m bolo_torch``) and talks to
it over the plugin protocol (bolo's docs/plugins.md). This module is what bolo's entry-point discovery imports in
the *server* process, so it must stay free of heavy imports: no torch here.
"""

from __future__ import annotations

import sys

__version__ = "0.1.8"


def plugin() -> list[str]:
    """The command line bolo runs to start the plugin (entry point ``bolo.engines: torch``)."""
    return [sys.executable, "-m", "bolo_torch"]


__all__ = ["__version__", "plugin"]
