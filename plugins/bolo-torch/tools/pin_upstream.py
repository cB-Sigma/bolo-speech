"""Write ``src/bolo_torch/upstream/<name>.json`` from local checkouts of the pinned commits (maintainers only).

    python tools/pin_upstream.py step-audio-2 --checkout stepfun-ai/Step-Audio2=/path/to/Step-Audio2

Each checkout must be at the commit pinned below; the sha256 of every file the bundle uses is read from it. Read the
code (and its licence) at that commit before changing a pin.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

#: name -> bundle: the parts (repository, commit, licence, where its files go, which files) and the import roots
BUNDLES: dict[str, dict[str, Any]] = {
    "step-audio-2": {
        "about": "Step-Audio 2 mini: audio feature extraction (utils.py), the token2wav decoder (CosyVoice 2 flow "
        "matching and HiFT vocoder) and the two prompt voices",
        "extra": "step-audio-2",
        "paths": [""],
        "parts": [
            {
                "repo": "stepfun-ai/Step-Audio2", "commit": "76e272b56c3917a8d7188f18bbb5a65dfc8a0845",
                "license": "Apache-2.0",
                "license_url": "https://github.com/stepfun-ai/Step-Audio2/blob/76e272b56c3917a8d7188f18bbb5a65dfc8a0845/LICENSE",
                "dest": "",
                "include": [
                    "LICENSE", "utils.py", "token2wav.py", "cosyvoice2/*.py", "cosyvoice2/*/*.py",
                    "flashcosyvoice/__init__.py", "flashcosyvoice/modules/__init__.py",
                    "flashcosyvoice/modules/hifigan.py", "flashcosyvoice/modules/hifigan_components/*.py",
                    "flashcosyvoice/utils/__init__.py", "flashcosyvoice/utils/audio.py",
                    "assets/default_female.wav", "assets/default_male.wav",
                ],
            },
        ],
    },  # fmt: skip
    "glm-4-voice": {
        "about": "GLM-4-Voice: the Whisper-based speech tokenizer, the flow-matching decoder with HiFi-GAN "
        "(CosyVoice code) and the Matcha-TTS modules it imports (a git submodule of GLM-4-Voice)",
        "extra": "glm-4-voice",
        "paths": ["", "third_party/Matcha-TTS"],
        "parts": [
            {
                "repo": "THUDM/GLM-4-Voice", "commit": "eb00ce9142e8d98b0ed7c57cd47e0d6d5dce9a1a",
                "license": "Apache-2.0 (code; the model weights have their own licence)",
                "license_url": "https://github.com/THUDM/GLM-4-Voice/blob/eb00ce9142e8d98b0ed7c57cd47e0d6d5dce9a1a/LICENSE",
                "dest": "",
                "include": ["LICENSE", "flow_inference.py", "speech_tokenizer/*.py", "cosyvoice/**.py"],
            },
            {
                "repo": "shivammehta25/Matcha-TTS", "commit": "dd9105b34bf2be2230f4aa1e4769fb586a3c824e",
                "license": "MIT",
                "license_url": "https://github.com/shivammehta25/Matcha-TTS/blob/dd9105b34bf2be2230f4aa1e4769fb586a3c824e/LICENSE",
                "dest": "third_party/Matcha-TTS",
                "include": ["LICENSE", "matcha/__init__.py", "matcha/models/**.py", "matcha/utils/**.py"],
            },
        ],
    },  # fmt: skip
    "spark-tts": {
        "about": "Spark-TTS: the BiCodec audio tokenizer and decoder (wav2vec 2.0 features, ECAPA speaker encoder, "
        "Vocos-style decoder) and its token maps; the language model is a plain Qwen2.5 checkpoint for transformers",
        "extra": "spark-tts",
        "paths": [""],
        "parts": [
            {
                "repo": "SparkAudio/Spark-TTS", "commit": "2f1ea9082400547242641f5271b6f941c9f439d1",
                "license": "Apache-2.0 (code; the Spark-TTS-0.5B weights are CC-BY-NC-SA-4.0)",
                "license_url": "https://github.com/SparkAudio/Spark-TTS/blob/2f1ea9082400547242641f5271b6f941c9f439d1/LICENSE",
                "dest": "",
                "include": ["LICENSE", "sparktts/**.py"],
            },
        ],
    },  # fmt: skip
    "maskgct": {
        "about": "MaskGCT (Amphion): the text-to-semantic and semantic-to-acoustic models, the semantic (RepCodec) and "
        "acoustic codecs, the inference pipeline (maskgct_utils.py), its configuration and wav2vec 2.0-BERT feature "
        "statistics, and the grapheme-to-phoneme front end with its lexicons and Chinese polyphone model",
        "extra": "maskgct",
        "paths": [""],
        "parts": [
            {
                "repo": "open-mmlab/Amphion", "commit": "26f6883110181f1dbfe95c70a7c7dbaf4de5f42a",
                "license": "MIT (code; the MaskGCT weights are CC-BY-NC-4.0)",
                "license_url": "https://github.com/open-mmlab/Amphion/blob/26f6883110181f1dbfe95c70a7c7dbaf4de5f42a/LICENSE",
                "dest": "",
                "include": [
                    "LICENSE", "models/__init__.py", "models/codec/__init__.py", "models/codec/amphion_codec/codec.py",
                    "models/codec/amphion_codec/vocos.py", "models/codec/amphion_codec/quantize/*.py",
                    "models/codec/kmeans/repcodec_model.py", "models/codec/kmeans/vocos.py",
                    "models/tts/maskgct/llama_nar.py", "models/tts/maskgct/maskgct_s2a.py",
                    "models/tts/maskgct/maskgct_t2s.py", "models/tts/maskgct/maskgct_utils.py",
                    "models/tts/maskgct/config/maskgct.json", "models/tts/maskgct/ckpt/wav2vec2bert_stats.pt",
                    "models/tts/maskgct/g2p/g2p_generation.py", "models/tts/maskgct/g2p/g2p/*.py",
                    "models/tts/maskgct/g2p/g2p/vocab.json", "models/tts/maskgct/g2p/utils/*.py",
                    "models/tts/maskgct/g2p/utils/mls_en.json", "models/tts/maskgct/g2p/sources/*",
                    "models/tts/maskgct/g2p/sources/g2p_chinese_model/*",
                    "utils/__init__.py", "utils/hparam.py", "utils/util.py",
                ],
            },
        ],
    },  # fmt: skip
    "kimi-audio": {
        "about": "Kimi-Audio: kimia_infer (prompt format, Whisper features, the streaming flow-matching detokenizer "
        "and BigVGAN vocoder) with the GLM-4-Voice speech tokenizer it imports (a git submodule of Kimi-Audio)",
        "extra": "kimi-audio",
        "paths": [""],
        "parts": [
            {
                "repo": "MoonshotAI/Kimi-Audio", "commit": "349251e1d8f4f98d58fda59246381faecd7392e0",
                "license": "MIT (code derived from Qwen2.5-7B: Apache-2.0); no LICENSE file, stated in README.md and "
                "pyproject.toml",
                "license_url": "https://github.com/MoonshotAI/Kimi-Audio/blob/349251e1d8f4f98d58fda59246381faecd7392e0/README.md#license",
                "dest": "",
                "include": ["README.md", "pyproject.toml", "kimia_infer/**"],
            },
            {
                "repo": "THUDM/GLM-4-Voice", "commit": "eb00ce9142e8d98b0ed7c57cd47e0d6d5dce9a1a",
                "license": "Apache-2.0",
                "license_url": "https://github.com/THUDM/GLM-4-Voice/blob/eb00ce9142e8d98b0ed7c57cd47e0d6d5dce9a1a/LICENSE",
                "dest": "kimia_infer/models/tokenizer/glm4",
                "include": ["LICENSE", "speech_tokenizer/*.py"],
            },
        ],
    },  # fmt: skip
}


def tracked(checkout: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(checkout), "ls-files"], capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def head(checkout: Path) -> str:
    out = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def matches(path: str, pattern: str) -> bool:
    if "**" in pattern:
        prefix, _, suffix = pattern.partition("**")
        return path.startswith(prefix) and path.endswith(suffix)
    return fnmatch.fnmatchcase(path, pattern) and path.count("/") == pattern.count("/")


def build(name: str, checkouts: dict[str, Path]) -> dict[str, Any]:
    spec = BUNDLES[name]
    parts = []
    for part in spec["parts"]:
        checkout = checkouts[part["repo"]]
        if head(checkout) != part["commit"]:
            raise SystemExit(f"{checkout} is at {head(checkout)}, not the pinned {part['commit']}")
        files = {}
        for rel in tracked(checkout):
            if (checkout / rel).is_file() and any(matches(rel, p) for p in part["include"]):
                files[rel] = hashlib.sha256((checkout / rel).read_bytes()).hexdigest()
        if not files:
            raise SystemExit(f"{part['repo']}: no file matched")
        parts.append({k: v for k, v in part.items() if k != "include"} | {"files": dict(sorted(files.items()))})
    return {"name": name, "about": spec["about"], "extra": spec["extra"], "paths": spec["paths"], "parts": parts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", choices=sorted(BUNDLES))
    parser.add_argument("--checkout", action="append", default=[], help="owner/repo=/path/to/checkout")
    args = parser.parse_args()
    checkouts = {k: Path(v) for k, _, v in (c.partition("=") for c in args.checkout)}
    data = build(args.name, checkouts)
    out = Path(__file__).resolve().parents[1] / "src" / "bolo_torch" / "upstream" / f"{args.name}.json"
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({sum(len(p['files']) for p in data['parts'])} files)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
