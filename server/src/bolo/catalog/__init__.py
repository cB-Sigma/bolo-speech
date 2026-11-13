"""The model catalog: curated, data-driven descriptions of every model bolo knows.

Entries live in YAML files next to this module (one file per group) and are
validated when loaded. Users can add their own entries: ``bolo pull hf:<repo>``
writes them to ``<data_dir>/custom.yaml``.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from bolo.store import Source

TASKS = ("stt", "tts", "vad", "diarization", "turn", "translation")
TIERS = ("realtime", "balanced", "accuracy")
VALIDATED = ("cpu", "untested")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class CatalogError(ValueError):
    pass


@dataclass
class ModelSpec:
    id: str
    task: str
    engine: str
    kind: str
    family: str
    languages: list[str]
    tier: str = "balanced"
    aliases: list[str] = field(default_factory=list)
    streaming: bool = False
    sample_rate: int = 16000
    size_bytes: int = 0
    params: int | None = None
    license: str = "unknown"
    license_url: str | None = None
    commercial_use: bool = True
    gated: bool = False
    voices: list[dict[str, Any]] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    description: str = ""
    source_url: str = ""
    validated: str = "untested"
    measured: dict[str, Any] = field(default_factory=dict)
    source: Source = field(default_factory=Source)
    options: dict[str, Any] = field(default_factory=dict)
    #: what the model needs besides its engine: "phonemizer" (a voice that reads phonemes; which component provides
    #: them depends on the language, see bolo.text.phonemize), or a component id ("espeak-ng", "uccharan")
    requires: list[str] = field(default_factory=list)
    custom: bool = False
    builtin: bool = False  # needs no files (fake engine)
    #: "pending": described here, but its files are not published yet; hidden from listings, cannot be pulled
    status: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelSpec:
        d = dict(d)
        missing = [k for k in ("id", "task", "engine", "kind", "family", "languages") if k not in d]
        if missing:
            raise CatalogError(f"catalog entry {d.get('id', '?')!r} is missing {', '.join(missing)}")
        src = d.pop("source", None) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "source"}
        unknown = set(d) - known
        if unknown:
            raise CatalogError(f"catalog entry {d['id']!r} has unknown fields: {', '.join(sorted(unknown))}")
        spec = cls(source=Source.from_dict(src), **d)
        spec.validate()
        return spec

    @property
    def needs_phonemes(self) -> bool:
        # "espeak-ng" is how entries written by bolo 0.1 (custom.yaml) say the same thing
        return "phonemizer" in self.requires or "espeak-ng" in self.requires

    def phoneme_target(self) -> tuple[str, str]:
        """``(profile, language)`` of a voice that reads phonemes, for choosing the phonemiser."""
        from bolo.text.phonemize import PROFILE_BY_KIND

        language = str(self.options.get("espeak_voice") or self.options.get("language") or "")
        if not language and self.voices:
            language = str(self.voices[0].get("language", ""))
        if not language and self.languages:
            language = self.languages[0]
        return PROFILE_BY_KIND.get(self.kind, "piper"), language

    @property
    def plugin(self) -> str | None:
        """The plugin name of an ``engine: plugin:<name>`` entry."""
        return self.engine[len("plugin:") :] if self.engine.startswith("plugin:") else None

    def require_published(self) -> None:
        if self.status == "pending":
            from bolo.errors import ModelNotPublished

            raise ModelNotPublished(
                f"model '{self.id}' is described in the catalog but its files are not published yet. "
                "If you have the files, register the directory with `bolo pull ./dir`."
            )

    def validate(self) -> None:
        if not _ID_RE.match(self.id):
            raise CatalogError(f"invalid model id {self.id!r} (lowercase letters, digits, '.', '_' and '-')")
        if self.task not in TASKS:
            raise CatalogError(f"{self.id}: unknown task {self.task!r}")
        if self.tier not in TIERS:
            raise CatalogError(f"{self.id}: unknown tier {self.tier!r}")
        if self.validated not in VALIDATED:
            raise CatalogError(f"{self.id}: validated must be one of {VALIDATED}")
        if not isinstance(self.languages, list) or not all(isinstance(x, str) for x in self.languages):
            raise CatalogError(f"{self.id}: languages must be a list of codes")
        if not self.license or self.license == "unknown":
            raise CatalogError(f"{self.id}: every catalog entry must state its licence")
        if not self.builtin and not (self.source.repo or self.source.urls or self.source.local):
            raise CatalogError(f"{self.id}: no download source")
        if self.validated == "cpu" and not self.builtin and not self.measured:
            raise CatalogError(f"{self.id}: validated entries must record their measurements")
        if self.status not in (None, "pending"):
            raise CatalogError(f"{self.id}: status must be 'pending' or absent")
        for u in self.source.urls:
            if u.get("extract") and not u.get("sha256") and self.status != "pending":
                raise CatalogError(f"{self.id}: an archive to extract must be pinned with sha256")
        for v in self.voices:
            if "id" not in v:
                raise CatalogError(f"{self.id}: voice without id")

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form (used for custom.yaml)."""
        out: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            if name == "source":
                out["source"] = self.source.to_dict()
            elif name not in ("custom", "builtin") and not (name == "status" and self.status is None):
                out[name] = getattr(self, name)
        return out

    def to_api(
        self,
        *,
        installed: bool = False,
        loaded: bool = False,
        pulling: dict[str, Any] | None = None,
        size_on_disk: int | None = None,
        available: bool = True,
        requires: list[str] | None = None,
        phonemizers: list[dict[str, Any]] | None = None,
        voice_phonemizers: dict[str, list[dict[str, Any]]] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        features = list(self.features)
        if self.task == "stt" and "partials" not in features:
            features.append("partials")
        return {
            "id": self.id,
            "aliases": self.aliases,
            "task": self.task,
            "engine": self.engine,
            "family": self.family,
            "languages": self.languages,
            "tier": self.tier,
            "streaming": self.streaming,
            "sample_rate": self.sample_rate,
            "size_bytes": size_on_disk or self.size_bytes,
            "params": self.params,
            "license": self.license,
            "license_url": self.license_url,
            "commercial_use": self.commercial_use,
            "gated": self.gated,
            "voices": [
                {
                    "id": v["id"],
                    "name": v.get("name", v["id"]),
                    "language": v.get("language", ""),
                    "gender": v.get("gender", ""),
                    **(
                        {"phonemizers": voice_phonemizers[v["id"]]}
                        if voice_phonemizers and v["id"] in voice_phonemizers
                        else {}
                    ),
                }
                for v in self.voices
            ],
            "features": features,
            "installed": installed,
            "loaded": loaded,
            "pulling": pulling,
            "description": self.description,
            "source_url": self.source_url,
            "validated": self.validated,
            "measured": self.measured or None,
            "custom": self.custom,
            "requires": list(self.requires) if requires is None else requires,
            "phonemizers": phonemizers or [],
            "settings": settings or {},
            "available": available,
            "status": self.status,
        }


FAKE_MODELS: list[dict[str, Any]] = [
    {
        "id": "fake-stt", "task": "stt", "engine": "fake", "kind": "fake-stt", "family": "Fake",
        "languages": ["en", "hi"], "tier": "balanced", "license": "Apache-2.0", "builtin": True,
        "features": ["timestamps", "telephony", "translate"],
        "description": "Deterministic test model: emits words in proportion to audio length. Not real recognition.",
        "options": {"base_delay_s": 0.05, "delay_per_audio_s": 0.02},
    },
    {
        "id": "fake-stt-streaming", "aliases": ["fake-realtime"], "task": "stt", "engine": "fake",
        "kind": "fake-stt-streaming", "family": "Fake", "languages": ["en"], "tier": "realtime", "streaming": True,
        "license": "Apache-2.0", "builtin": True, "features": ["timestamps", "telephony"],
        "description": "Deterministic streaming test model with incremental partials. Not real recognition.",
        "options": {"step_delay_s": 0.01},
    },
    {
        "id": "fake-tts", "task": "tts", "engine": "fake", "kind": "fake-tts", "family": "Fake",
        "languages": ["en", "hi"], "tier": "realtime", "sample_rate": 24000, "license": "Apache-2.0",
        "builtin": True, "features": ["telephony"],
        "voices": [
            {"id": "alpha", "name": "Alpha", "language": "en-US", "gender": "f"},
            {"id": "beta", "name": "Beta", "language": "en-GB", "gender": "m"},
            {"id": "gamma", "name": "Gamma", "language": "hi-IN", "gender": "f"},
        ],
        "description": "Deterministic test voice: produces tones in proportion to text length. Not real speech.",
    },
    {
        "id": "fake-diarization", "task": "diarization", "engine": "fake", "kind": "fake-diarization", "family": "Fake",
        "languages": [], "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "description": "Test diarizer: separates bursts of sound by their dominant pitch. Not real speaker recognition.",
    },
    {
        "id": "fake-turn", "task": "turn", "engine": "fake", "kind": "fake-turn", "family": "Fake", "languages": [],
        "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "description": "Test end-of-turn model: a turn is finished once it holds three seconds of sound.",
    },
    {
        "id": "fake-vad", "task": "vad", "engine": "fake", "kind": "fake-vad", "family": "Fake", "languages": [],
        "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "description": "Energy-threshold voice-activity detector (no model file).",
    },
]  # fmt: skip


def _load_yaml_entries(text: str, origin: str) -> list[dict[str, Any]]:
    doc = yaml.safe_load(text) or []
    if isinstance(doc, dict):
        doc = doc.get("models", [])
    if not isinstance(doc, list):
        raise CatalogError(f"{origin}: expected a list of models")
    return doc


def packaged_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    root = resources.files("bolo.catalog")
    for item in sorted(root.iterdir(), key=lambda p: p.name):
        if item.name.endswith(".yaml"):
            entries.extend(_load_yaml_entries(item.read_text(encoding="utf-8"), item.name))
    return entries


class Catalog:
    def __init__(self, specs: list[ModelSpec], custom_path: Path | None = None) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._aliases: dict[str, str] = {}
        self._custom_path = custom_path
        self._lock = threading.Lock()
        for spec in specs:
            self.add(spec)

    @classmethod
    def load(cls, data_dir: Path | None = None, include_fake: bool = False) -> Catalog:
        specs = [ModelSpec.from_dict(e) for e in packaged_entries()]
        if include_fake:
            specs.extend(ModelSpec.from_dict(e) for e in FAKE_MODELS)
        custom_path = data_dir / "custom.yaml" if data_dir else None
        cat = cls(specs, custom_path)
        if custom_path and custom_path.is_file():
            for e in _load_yaml_entries(custom_path.read_text(encoding="utf-8"), str(custom_path)):
                spec = ModelSpec.from_dict({**e, "custom": True})
                if spec.id not in cat._specs:
                    cat.add(spec)
        return cat

    def add(self, spec: ModelSpec) -> None:
        with self._lock:
            if spec.id in self._specs:
                raise CatalogError(f"duplicate model id {spec.id!r}")
            if spec.id.lower() in self._aliases:
                raise CatalogError(f"model id {spec.id!r} collides with an alias of {self._aliases[spec.id.lower()]!r}")
            self._specs[spec.id] = spec
            for alias in spec.aliases:
                key = alias.lower()
                if key in self._aliases or key in self._specs:
                    raise CatalogError(f"alias {alias!r} of {spec.id!r} is already taken")
                self._aliases[key] = spec.id

    def add_custom(self, spec: ModelSpec) -> None:
        spec.custom = True
        self.add(spec)
        self._save_custom()

    def remove_custom(self, model_id: str) -> None:
        with self._lock:
            spec = self._specs.get(model_id)
            if spec is None or not spec.custom:
                return
            del self._specs[model_id]
            self._aliases = {a: m for a, m in self._aliases.items() if m != model_id}
        self._save_custom()

    def _save_custom(self) -> None:
        if not self._custom_path:
            return
        entries = [s.to_dict() for s in self._specs.values() if s.custom]
        self._custom_path.parent.mkdir(parents=True, exist_ok=True)
        self._custom_path.write_text(yaml.safe_dump(entries, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def set_alias(self, alias: str, model_id: str) -> None:
        """Point a public alias (``whisper-1``, ``tts-1``) at a model at runtime."""
        if model_id in self._specs:
            self._aliases[alias.lower()] = model_id

    def resolve(self, name: str) -> ModelSpec | None:
        if name in self._specs:
            return self._specs[name]
        key = name.lower()
        if key in self._specs:
            return self._specs[key]
        target = self._aliases.get(key)
        return self._specs.get(target) if target else None

    def all(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def search(self, task: str | None = None, q: str | None = None, include_pending: bool = False) -> list[ModelSpec]:
        out = []
        needle = (q or "").strip().lower()
        for spec in self._specs.values():
            if task and spec.task != task:
                continue
            if spec.status == "pending" and not include_pending:
                continue
            if needle:
                hay = " ".join(
                    [spec.id, spec.family, spec.engine, spec.description, spec.license, *spec.aliases, *spec.languages]
                ).lower()
                if not all(part in hay for part in needle.split()):
                    continue
            out.append(spec)
        return out
