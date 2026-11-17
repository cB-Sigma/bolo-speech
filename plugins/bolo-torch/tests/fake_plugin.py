"""The real plugin server with fake model loaders, run as a subprocess by the protocol tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bolo_torch.engine import TorchEngine
from bolo_torch.server import serve
from fakes import load_fake_dicow, load_fake_parler, load_fake_pyannote, load_fake_sepformer, load_gated
from omni_fakes import load_fake_omni

if __name__ == "__main__":
    engine = TorchEngine(
        {
            "parler-tts": load_fake_parler,
            "sepformer": load_fake_sepformer,
            "gated": load_gated,
            "dicow": load_fake_dicow,
            "pyannote-diarization": load_fake_pyannote,
            "qwen-omni": load_fake_omni,
        }
    )
    raise SystemExit(serve(engine))
