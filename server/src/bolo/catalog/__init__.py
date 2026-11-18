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

TASKS = (
    "stt", "tts", "vad", "diarization", "turn", "translation", "separation", "denoise", "lid", "speech-to-speech",
    "audio-llm", "speaker-embedding", "audio-tagging", "emotion", "voice-conversion", "align", "punctuation",
    "echo-cancellation",
)  # fmt: skip
TIERS = ("realtime", "balanced", "accuracy")
VALIDATED = ("cpu", "untested")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


CLOUD_NOTICE = "paid cloud API; audio leaves this server"


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
    #: "pending": described here, but its files are not published yet; cannot be pulled
    status: str | None = None
    #: a model behind a paid cloud API (engine remote): nothing to download, audio or text leaves the server
    cloud: bool = False
    #: False for entries no runtime bolo has can run yet (kept so the catalog says why); see not_runnable_reason
    runnable: bool = True
    not_runnable_reason: str | None = None
    #: a ``provider:model`` id that was asked for by name and is not a curated entry
    adhoc: bool = False
    #: a preset (``bolo cp``): this entry is a copy of that model, with the same files and its own settings
    preset_of: str | None = None

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

    def not_recommended(self, language: str | None) -> str | None:
        """Why this model should not be used for ``language`` (``options.not_recommended_for``: language ->
        reason, for a language it speaks but transcribes badly), or None."""
        marks = self.options.get("not_recommended_for")
        if not language or not isinstance(marks, dict):
            return None
        from bolo.text.iso639 import same_language

        return next((str(why) for code, why in marks.items() if same_language(str(code), language)), None)

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
        if self.cloud and not self.license_url:
            raise CatalogError(f"{self.id}: a cloud entry must link the provider's terms in license_url")
        if not self.runnable and not self.not_runnable_reason:
            raise CatalogError(f"{self.id}: an entry that is not runnable must say why (not_runnable_reason)")
        if not self.builtin and not self.cloud and not (self.source.repo or self.source.urls or self.source.local):
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
        from bolo.remote_code import check_pin

        problem = check_pin(self)
        if problem:
            raise CatalogError(f"{self.id}: {problem}")

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form (used for custom.yaml)."""
        out: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            if name == "source":
                out["source"] = self.source.to_dict()
            elif name in ("cloud", "adhoc", "not_runnable_reason", "preset_of") and not getattr(self, name):
                continue  # defaults stay out of custom.yaml
            elif name == "builtin" and self.builtin:
                out[name] = True  # a preset of a built-in model
            elif name == "runnable" and self.runnable:
                continue  # (so does this one)
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
        if self.task == "tts" and self.kind in ("kokoro", "kokoro-onnx") and "voice-blending" not in features:
            features.append("voice-blending")  # bolo.engines.blend.BLEND_KINDS
        if self.task == "tts" and self.kind == "parler-tts" and "instructions" not in features:
            features.append("instructions")  # Parler-TTS: the request's instructions are its voice description
        if self.task == "turn" and str(self.options.get("input", "audio")) == "text" and "reads-text" not in features:
            features.append("reads-text")  # an end-of-turn model that judges the transcript (Bolo.turn_reads_text)
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
            # machine-readable "no runtime can run this yet" (the description says the same in prose)
            "runnable": self.runnable,
            "not_runnable_reason": self.not_runnable_reason,
            # a paid cloud API: nothing is downloaded, and audio or text leaves this server
            "cloud": self.cloud,
            "notice": CLOUD_NOTICE if self.cloud else None,
            # a preset (`bolo cp`): the model whose files this entry uses, with settings of its own
            "preset_of": self.preset_of,
            # what the model needs to run well, when the catalog says (options.hardware), e.g.
            # {"gpu": "recommended", "vram_gb": 3.5, "note": "..."}; null for entries that run fine on a CPU
            "hardware": self.options.get("hardware"),
            # Python code from the model's own repository that loading it runs (options.remote_code): the pinned
            # commit and files, and whether the operator opted in (per-model setting allow_remote_code); else null
            "remote_code": self._remote_code_api(settings or {}),
            # a licence term to act on before use (for example a registration for commercial use); else null
            "license_note": self.options.get("license_note"),
            # languages it speaks but should not be used for, with the reason (options.not_recommended_for)
            "not_recommended_for": self.options.get("not_recommended_for") or {},
        }

    def _remote_code_api(self, settings: dict[str, Any]) -> dict[str, Any] | None:
        from bolo import remote_code

        p = remote_code.pin(self)
        if p is None:
            return None
        return {
            "revision": p.get("revision"),
            "files": sorted(str(n) for n in p.get("files", {})),
            "reviewed": p.get("reviewed"),
            "allowed": remote_code.allowed(settings),
            "notice": remote_code.notice(self, settings),
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
        "id": "fake-diarization-overlap", "task": "diarization", "engine": "fake", "kind": "fake-diarization",
        "family": "Fake", "languages": [], "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "options": {"multi_pitch": True},
        "description": "Test diarizer that hears several tones at once, so its speaker turns can overlap.",
    },
    {
        "id": "fake-separation", "task": "separation", "engine": "fake", "kind": "fake-separation", "family": "Fake",
        "languages": [], "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "description": "Test separator: keeps the enrolled voice's dominant tone. Not real speech separation.",
    },
    {
        "id": "fake-separation-blind", "task": "separation", "engine": "fake", "kind": "fake-separation",
        "family": "Fake", "languages": [], "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "options": {"mode": "blind"},
        "description": "Test blind separator: one track per strong tone. Not real speech separation.",
    },
    {
        "id": "fake-denoise", "task": "denoise", "engine": "fake", "kind": "fake-denoise", "family": "Fake",
        "languages": [], "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "description": "Test denoiser: removes frequency bins weaker than a few times the median. Not real enhancement.",
    },
    {
        "id": "fake-denoise-48k", "task": "denoise", "engine": "fake", "kind": "fake-denoise", "family": "Fake",
        "languages": [], "tier": "realtime", "license": "Apache-2.0", "builtin": True, "options": {"sample_rate": 48000},
        "description": "Test denoiser that works at 48 kHz, like full-band enhancement models.",
    },
    {
        "id": "fake-echo", "task": "echo-cancellation", "engine": "fake", "kind": "fake-echo", "family": "Fake",
        "languages": [], "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "description": "Test echo canceller: mutes the input while the reference plays. Not real echo cancellation.",
    },
    {
        "id": "fake-tagging", "task": "audio-tagging", "engine": "fake", "kind": "fake-tagging", "family": "Fake",
        "languages": [], "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "description": "Test audio tagger: silence, a low tone (music) or a high tone (speech). Not real tagging.",
    },
    {
        "id": "fake-emotion", "task": "emotion", "engine": "fake", "kind": "fake-emotion", "family": "Fake",
        "languages": [], "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "description": "Test emotion model: a low pitch is neutral, a high one happy. Not real emotion recognition.",
    },
    {
        "id": "fake-punctuation", "task": "punctuation", "engine": "fake", "kind": "fake-punctuation",
        "family": "Fake", "languages": ["en", "hi"], "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "features": ["punctuation"],
        "description": "Test punctuation: a capital first, a comma every four words, a full stop at the end.",
    },
    {
        "id": "fake-lid", "task": "lid", "engine": "fake", "kind": "fake-lid", "family": "Fake",
        "languages": ["en", "hi", "ta", "fr"], "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "description": "Test language identifier: the dominant pitch picks the language (220 Hz English, 440 Hz Hindi, "
        "660 Hz Tamil, 880 Hz French). Not real language identification.",
    },
    {
        "id": "fake-align", "task": "align", "engine": "fake", "kind": "fake-align", "family": "Fake",
        "languages": [], "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "description": "Test aligner: the words share out the audio that is not silent by their letters. Not real "
        "alignment.",
    },
    {
        "id": "fake-stt-ta", "task": "stt", "engine": "fake", "kind": "fake-stt", "family": "Fake",
        "languages": ["ta"], "tier": "balanced", "license": "Apache-2.0", "builtin": True,
        "description": "Deterministic one-language test recogniser (Tamil), for language routing. Not real recognition.",
        "options": {"base_delay_s": 0.01, "delay_per_audio_s": 0.0},
    },
    {
        "id": "fake-turn", "task": "turn", "engine": "fake", "kind": "fake-turn", "family": "Fake", "languages": [],
        "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "description": "Test end-of-turn model: a turn is finished once it holds three seconds of sound.",
    },
    {
        "id": "fake-turn-text", "task": "turn", "engine": "fake", "kind": "fake-turn-text", "family": "Fake",
        "languages": [], "tier": "realtime", "license": "Apache-2.0", "builtin": True, "options": {"input": "text"},
        "description": "Test end-of-turn model that reads text: a turn is finished once it holds eight words.",
    },
    {
        "id": "fake-s2s", "task": "speech-to-speech", "engine": "fake", "kind": "fake-s2s", "family": "Fake",
        "languages": ["en"], "tier": "realtime", "sample_rate": 24000, "license": "Apache-2.0", "builtin": True,
        "features": ["speech-to-speech", "tool-calls"],
        "description": "Test speech-to-speech model: answers with what it heard (a word and a tone at a time).",
    },
    {
        "id": "fake-speaker-embedding", "task": "speaker-embedding", "engine": "fake", "kind": "fake-speaker-embedding",
        "family": "Fake", "languages": [], "tier": "realtime", "license": "Apache-2.0", "builtin": True,
        "description": "Test speaker embedder: the dominant pitch as a direction in two dimensions (as fake-diarization).",
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

    def add_adhoc(self, spec: ModelSpec) -> ModelSpec:
        """Register a ``provider:model`` id that was asked for by name (kept in memory only)."""
        with self._lock:
            existing = self._specs.get(spec.id)
            if existing is not None:
                return existing
            self._specs[spec.id] = spec
            return spec

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
