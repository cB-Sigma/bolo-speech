"""Upstream Piper voices (``<voice>.onnx`` + ``<voice>.onnx.json``) made loadable by sherpa-onnx.

sherpa-onnx reads a VITS voice from a model that carries a few metadata entries plus a
``tokens.txt`` table. The voices published by the Piper project have neither: the symbol table
lives in the JSON file next to the model. The conversion is small and needs no machine-learning
library: ``tokens.txt`` is written from the JSON, and the metadata is appended to the ONNX file
(an ONNX model is a protobuf message, and repeated fields such as ``metadata_props`` may simply be
appended to the serialised bytes). bolo does this once, right after the download.

``bolo pull piper:<voice>`` (for example ``piper:te_IN-venkatesh-medium``) fetches a voice from
https://huggingface.co/rhasspy/piper-voices and converts it; catalog entries with
``source.convert: piper`` do the same for any repository in that layout. Speaking still needs
phonemes from the optional eSpeak NG component.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bolo.errors import DownloadError, InvalidRequest

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.core import Bolo

PIPER_REPO = "rhasspy/piper-voices"
_VOICE_RE = re.compile(r"^([a-z]{2,3})_([A-Z]{2})-([A-Za-z0-9_]+)-(x_low|low|medium|high)$")


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def metadata_bytes(meta: dict[str, Any]) -> bytes:
    """``ModelProto.metadata_props`` entries (field 14, each a key/value message) in protobuf wire format."""
    out = bytearray()
    for key, value in meta.items():
        k, v = str(key).encode(), str(value).encode()
        entry = b"\x0a" + _varint(len(k)) + k + b"\x12" + _varint(len(v)) + v
        out += b"\x72" + _varint(len(entry)) + entry
    return bytes(out)


def convert_piper_dir(directory: Path) -> dict[str, Any]:
    """Turn a downloaded Piper voice in ``directory`` into sherpa-onnx layout, in place.

    Afterwards the directory holds ``model.onnx`` (the original graph plus metadata), ``tokens.txt``
    and the untouched ``*.onnx.json``. Returns what the JSON says about the voice.
    """
    configs = sorted(directory.glob("*.onnx.json"))
    if not configs:
        raise DownloadError(f"{directory.name}: no '<voice>.onnx.json' file, so this is not an upstream Piper voice")
    config_path = configs[0]
    model_path = config_path.with_name(config_path.name[: -len(".json")])
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        id_map: dict[str, Any] = config["phoneme_id_map"]
        espeak_voice = str((config.get("espeak") or {}).get("voice") or "")
        sample_rate = int(config["audio"]["sample_rate"])
        speakers = int(config.get("num_speakers") or 1)
    except (ValueError, KeyError, TypeError) as exc:
        raise DownloadError(f"{config_path.name} is not a Piper voice description: {exc}") from exc
    if not espeak_voice:
        raise DownloadError(
            f"{config_path.name} names no espeak-ng voice; only phoneme-based Piper voices are supported"
        )
    target = directory / "model.onnx"
    if model_path.is_file():
        lines = []
        for symbol, ids in id_map.items():
            if len(symbol) != 1 or symbol in "\n\r\t":
                continue  # sherpa-onnx reads one code point per line; longer symbols are never produced anyway
            lines.append(f"{symbol} {ids[0] if isinstance(ids, list) else ids}")
        (directory / "tokens.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        language = str((config.get("language") or {}).get("name_english") or espeak_voice)
        meta = {
            "model_type": "vits", "comment": "piper", "language": language, "voice": espeak_voice,
            "has_espeak": 1, "has_g2pw": 0, "n_speakers": speakers, "sample_rate": sample_rate,
        }  # fmt: skip
        tmp = directory / "model.onnx.tmp"
        with model_path.open("rb") as src, tmp.open("wb") as dst:  # a copy: the download may be a link to a shared blob
            shutil.copyfileobj(src, dst, 1 << 20)
            dst.write(metadata_bytes(meta))
        tmp.replace(target)
        model_path.unlink()
    elif not (target.is_file() and (directory / "tokens.txt").is_file()):
        raise DownloadError(f"{directory.name}: {model_path.name} is missing")
    speaker_names = config.get("speaker_id_map") or {}
    return {
        "espeak_voice": espeak_voice,
        "sample_rate": sample_rate,
        "num_speakers": speakers,
        "speakers": sorted(speaker_names, key=lambda name: speaker_names[name]) if speaker_names else [],
        "quality": str((config.get("audio") or {}).get("quality") or ""),
    }


def register_piper_voice(core: Bolo, voice: str) -> ModelSpec:
    """Create (or return) a custom catalog entry for an upstream Piper voice such as ``te_IN-venkatesh-medium``."""
    from bolo.catalog import ModelSpec
    from bolo.store import Source

    voice = voice.strip()
    revision = "main"
    if "@" in voice:
        voice, revision = voice.split("@", 1)
    m = _VOICE_RE.match(voice)
    if not m:
        raise InvalidRequest(
            "expected piper:<language>_<COUNTRY>-<name>-<quality>, for example piper:te_IN-venkatesh-medium", param="id"
        )
    lang, country, name, quality = m.groups()
    model_id = f"piper-{lang}-{country.lower()}-{name.lower()}-{quality}".replace("_", "-")
    for known in (model_id, f"piper-{voice}", f"{PIPER_REPO}/{voice}"):
        existing = core.catalog.resolve(known)
        if existing is not None:
            return existing
    base = f"https://huggingface.co/{PIPER_REPO}/resolve/{revision}/{lang}/{lang}_{country}/{name}/{quality}"
    spec = ModelSpec(
        id=model_id,
        task="tts",
        engine="sherpa-onnx",
        kind="vits",
        family="Piper",
        languages=[lang],
        tier="realtime",
        sample_rate=22050 if quality in ("medium", "high") else 16000,
        license="see the voice's MODEL_CARD (the licence of a Piper voice follows its training data)",
        description=f"Piper voice '{name}' ({lang}-{country}, {quality}) from {PIPER_REPO}, converted for sherpa-onnx "
        "at download time (not validated by the bolo maintainers).",
        source_url=f"https://huggingface.co/{PIPER_REPO}",
        validated="untested",
        source=Source(
            urls=[
                {"path": f"{voice}.onnx", "url": f"{base}/{voice}.onnx"},
                {"path": f"{voice}.onnx.json", "url": f"{base}/{voice}.onnx.json"},
                {"path": "MODEL_CARD", "url": f"{base}/MODEL_CARD"},
            ],
            convert="piper",
        ),
        options={"language": f"{lang}-{country}"},
        requires=["phonemizer"],
        custom=True,
    )
    spec.validate()
    core.catalog.add_custom(spec)
    return spec
