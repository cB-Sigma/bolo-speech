"""Enrolled speakers: enrol named people from recordings, verify a recording against one, identify who speaks.

An enrolled speaker is a name and one or more recordings of that person (kept at 16 kHz, at most
:data:`MAX_CLIP_S` seconds each, under ``<data_dir>/speakers/<id>/``). Their voiceprint for a speaker embedder is the
mean of the unit-length embeddings of their recordings; it is computed the first time that embedder is used and
kept next to the recordings, so any embedder of the catalog can be used later without enrolling again.

Verification compares a recording with one enrolled speaker: the cosine similarity of its embedding and the
speaker's voiceprint, against the embedder's threshold. Identification compares it with every enrolled speaker (or
the ones named) and answers the closest one when it reaches the threshold, else nobody.

Every embedder has its own scale, so every embedder has its own threshold (:data:`THRESHOLDS`). They were measured,
not assumed: on LibriSpeech dev-clean (CC BY 4.0) with each speaker enrolled from some of their utterances and
tested with the others (docs/measurements/speaker-verification.json, ``tools/calibrate_speakers.py``), each
threshold is the score that 1 in 1000 different-speaker trials reached or passed. They are a rule of thumb for clean
read speech, not proof of identity: telephone audio, noise, short clips, other languages and synthesised or
imitated voices all move the scores, and a voiceprint is not a security credential.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import threading
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo import similarity
from bolo.audio.codecs import float_to_pcm16
from bolo.audio.encode import wav_header
from bolo.audio.resample import resample
from bolo.engines.base import FloatArray
from bolo.errors import BoloError, InvalidRequest

if TYPE_CHECKING:
    from bolo.core import Bolo

RATE = similarity.EMBEDDING_RATE
#: the longest recording kept per enrolment clip (seconds); longer ones keep their first part
MAX_CLIP_S = 30.0
#: the shortest recording accepted (seconds)
MIN_CLIP_S = 1.0
#: recordings per speaker at most
MAX_CLIPS = 20
META = "speaker.json"
#: ids that are routes of their own
RESERVED = ("verify", "identify")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

#: same-speaker threshold per embedder: the cosine at or above which a recording is taken to be the enrolled speaker.
#: Measured on LibriSpeech dev-clean (docs/measurements/speaker-verification.json): the score that 1 in 1000
#: different-speaker trials reached, rounded up to two decimals.
THRESHOLDS: dict[str, float] = {
    "eres2net-voxceleb": 0.52,
    "campplus-voxceleb": 0.51,
    "campplus-zh-en": 0.53,
    "wespeaker-campplus-voxceleb": 0.55,
    "wespeaker-resnet34-voxceleb": 0.86,
    "titanet-small": 0.51,
    "ecapa-tdnn-voxceleb": 0.46,
}
#: diarization pipelines whose embedder is the very file of a speaker-embedding entry
SAME_FILE = {"pyannote-3.0-eres2net": "eres2net-voxceleb", "pyannote-3.0-titanet-small": "titanet-small",
             "pyannote-3.0-wespeaker-resnet34": "wespeaker-resnet34-voxceleb"}  # fmt: skip
#: the threshold of an embedder that was not measured (a diarization model's own embedder, a custom model)
DEFAULT_THRESHOLD = 0.5
#: the measurement :data:`THRESHOLDS` come from
CALIBRATION_FILE = "docs/measurements/speaker-verification.json"


class SpeakerNotFound(BoloError):
    status = 404
    code = "speaker_not_found"
    type = "invalid_request_error"


class SpeakerExists(BoloError):
    status = 409
    code = "speaker_exists"
    type = "invalid_request_error"


def slug(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text).strip("-_")
    return text[:64].strip("-_")


def embedder_id(core: Bolo, name: str | None = None) -> str:
    """The speaker embedder to use: ``name`` or the default (bolo.similarity.embedder), with a diarization pipeline
    named by the embedding model whose file it uses (:data:`SAME_FILE`): the same voiceprints, and it needs only
    that file installed."""
    model = similarity.embedder(core, name).id
    return SAME_FILE.get(model, model)


def threshold_for(model: str) -> tuple[float, bool]:
    """(threshold, calibrated) for an embedder; the pyannote-3.0-* pipelines embed with the same files."""
    key = SAME_FILE.get(model, model)
    if key in THRESHOLDS:
        return THRESHOLDS[key], True
    return DEFAULT_THRESHOLD, False


def unit(v: FloatArray) -> FloatArray:
    n = float(np.linalg.norm(v))
    return np.asarray(v / n if n > 0 else v, dtype=np.float32)


@dataclass
class Speaker:
    id: str
    meta: dict[str, Any]
    path: Path

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or self.id)

    def clips(self) -> list[dict[str, Any]]:
        return list(self.meta.get("clips") or [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "speaker",
            "name": self.name,
            "created_at": self.meta.get("created_at"),
            "updated_at": self.meta.get("updated_at"),
            "metadata": self.meta.get("metadata") or {},
            "clips": [{"id": c["id"], "seconds": c["seconds"], "added_at": c.get("added_at")} for c in self.clips()],
            "seconds": round(sum(float(c["seconds"]) for c in self.clips()), 2),
            "embedding_models": sorted((self.meta.get("embeddings") or {}).keys()),
        }


class SpeakerStore:
    """Enrolled speakers on disk (``<data_dir>/speakers``)."""

    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir) / "speakers"
        self._lock = threading.RLock()

    def _dir(self, speaker_id: str) -> Path:
        return self.root / speaker_id

    def _write(self, speaker: Speaker) -> None:
        tmp = speaker.path / (META + ".tmp")
        tmp.write_text(json.dumps(speaker.meta, ensure_ascii=False), encoding="utf-8")
        tmp.replace(speaker.path / META)

    def all(self) -> list[Speaker]:
        if not self.root.is_dir():
            return []
        out = []
        for p in sorted(self.root.iterdir()):
            if (p / META).is_file():
                with contextlib.suppress(SpeakerNotFound, ValueError, OSError):
                    out.append(self.get(p.name))
        return sorted(out, key=lambda s: s.name.lower())

    def get(self, speaker_id: str) -> Speaker:
        sid = speaker_id.strip().lower()
        path = self._dir(sid)
        if not _ID_RE.match(sid) or not (path / META).is_file():
            raise SpeakerNotFound(f"no enrolled speaker '{speaker_id}'", param="speaker")
        return Speaker(sid, json.loads((path / META).read_text(encoding="utf-8")), path)

    def find(self, ref: str) -> Speaker:
        """A speaker by id, else by name (case-insensitive)."""
        ref = ref.strip()
        with contextlib.suppress(SpeakerNotFound):
            return self.get(ref)
        matches = [s for s in self.all() if s.name.lower() == ref.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise InvalidRequest(f"more than one speaker is named '{ref}': use the id", param="speaker")
        raise SpeakerNotFound(f"no enrolled speaker '{ref}'", param="speaker")

    def create(self, name: str, speaker_id: str | None, metadata: dict[str, Any] | None, now: str) -> Speaker:
        name = name.strip()
        if not name:
            raise InvalidRequest("a speaker needs a name", param="name")
        if len(name) > 200:
            raise InvalidRequest("the name is longer than 200 characters", param="name")
        with self._lock:
            if speaker_id:
                sid = speaker_id.strip().lower()
                if not _ID_RE.match(sid):
                    raise InvalidRequest(
                        f"invalid speaker id {speaker_id!r}: lowercase letters, digits, '-' and '_' (up to 64)",
                        param="id",
                    )
                if sid in RESERVED:
                    raise InvalidRequest(f"'{sid}' cannot be a speaker id", param="id")
                if (self._dir(sid) / META).exists():
                    raise SpeakerExists(f"a speaker with the id '{sid}' is already enrolled", param="id")
            else:
                base = slug(name) or "speaker"
                base = f"{base}-speaker" if base in RESERVED else base
                sid, n = base, 2
                while (self._dir(sid) / META).exists():
                    sid = f"{base[:60]}-{n}"
                    n += 1
            path = self._dir(sid)
            path.mkdir(parents=True, exist_ok=True)
            meta = {"name": name, "created_at": now, "updated_at": now, "metadata": metadata or {}, "clips": []}
            speaker = Speaker(sid, meta, path)
            self._write(speaker)
            return speaker

    def add_clip(self, speaker: Speaker, samples16: FloatArray, now: str) -> dict[str, Any]:
        with self._lock:
            speaker = self.get(speaker.id)
            if len(speaker.clips()) >= MAX_CLIPS:
                raise InvalidRequest(f"a speaker has at most {MAX_CLIPS} recordings", param="file")
            clip_id = "clip_" + uuid.uuid4().hex[:12]
            data = wav_header(RATE, len(samples16)) + float_to_pcm16(samples16).tobytes()
            (speaker.path / f"{clip_id}.wav").write_bytes(data)
            clip = {"id": clip_id, "seconds": round(len(samples16) / RATE, 2), "added_at": now}
            speaker.meta.setdefault("clips", []).append(clip)
            speaker.meta["embeddings"] = {}  # voiceprints are recomputed with the new recording
            speaker.meta["updated_at"] = now
            self._write(speaker)
            return clip

    def remove_clip(self, speaker_id: str, clip_id: str, now: str) -> Speaker:
        with self._lock:
            speaker = self.get(speaker_id)
            clips = speaker.clips()
            if not any(c["id"] == clip_id for c in clips):
                raise SpeakerNotFound(f"speaker '{speaker.id}' has no recording '{clip_id}'", param="clip_id")
            speaker.meta["clips"] = [c for c in clips if c["id"] != clip_id]
            speaker.meta["embeddings"] = {}
            speaker.meta["updated_at"] = now
            with contextlib.suppress(OSError):
                (speaker.path / f"{clip_id}.wav").unlink()
            self._write(speaker)
            return speaker

    def update(self, speaker_id: str, changes: dict[str, Any], now: str) -> Speaker:
        with self._lock:
            speaker = self.get(speaker_id)
            if "name" in changes:
                name = str(changes["name"] or "").strip()
                if not name:
                    raise InvalidRequest("a speaker needs a name", param="name")
                speaker.meta["name"] = name
            if "metadata" in changes:
                if not isinstance(changes["metadata"], dict):
                    raise InvalidRequest("metadata must be an object", param="metadata")
                speaker.meta["metadata"] = changes["metadata"]
            speaker.meta["updated_at"] = now
            self._write(speaker)
            return speaker

    def delete(self, speaker_id: str) -> None:
        with self._lock:
            speaker = self.get(speaker_id)
            shutil.rmtree(speaker.path, ignore_errors=True)

    def clip_audio(self, speaker: Speaker, clip_id: str) -> bytes:
        path = speaker.path / f"{clip_id}.wav"
        if not any(c["id"] == clip_id for c in speaker.clips()) or not path.is_file():
            raise SpeakerNotFound(f"speaker '{speaker.id}' has no recording '{clip_id}'", param="clip_id")
        return path.read_bytes()

    def save_voiceprint(self, speaker_id: str, model: str, vector: FloatArray, clips: list[str]) -> None:
        with self._lock:
            speaker = self.get(speaker_id)
            if [c["id"] for c in speaker.clips()] != clips:
                return  # the recordings changed meanwhile: do not keep a stale voiceprint
            speaker.meta.setdefault("embeddings", {})[model] = [round(float(v), 6) for v in vector]
            self._write(speaker)


def prepare_clip(samples: FloatArray, rate: int) -> FloatArray:
    """A recording for enrolment: 16 kHz, its first :data:`MAX_CLIP_S` seconds, at least :data:`MIN_CLIP_S`."""
    x = resample(np.asarray(samples, dtype=np.float32), rate, RATE)[: int(MAX_CLIP_S * RATE)]
    if len(x) < int(MIN_CLIP_S * RATE):
        raise InvalidRequest(f"an enrolment recording needs at least {MIN_CLIP_S:g} s of speech", param="file")
    if float(np.max(np.abs(x))) < 1e-4:
        raise InvalidRequest("the recording is silent", param="file")
    return x


async def voiceprint(core: Bolo, speaker: Speaker, model: str) -> FloatArray:
    """The speaker's voiceprint for ``model``: the mean of their recordings' unit embeddings, unit length."""
    from bolo.audio.decode import decode_audio

    cached = (speaker.meta.get("embeddings") or {}).get(model)
    if cached:
        return np.asarray(cached, dtype=np.float32)
    clips = speaker.clips()
    if not clips:
        raise InvalidRequest(f"speaker '{speaker.id}' has no recordings yet", param="speaker")
    vectors = []
    for clip in clips:
        samples, rate = decode_audio(core.speakers.clip_audio(speaker, clip["id"]))
        vectors.append(unit(await similarity.embed(core, samples, rate, model)))
    vector = unit(np.mean(vectors, axis=0))
    core.speakers.save_voiceprint(speaker.id, model, vector, [c["id"] for c in clips])
    return vector


async def verify(
    core: Bolo, samples: FloatArray, rate: int, speaker: Speaker, model: str, threshold: float | None
) -> dict[str, Any]:
    probe = unit(await similarity.embed(core, samples, rate, model))
    ref = await voiceprint(core, speaker, model)
    score = round(similarity.cosine(probe, ref), 4)
    limit, calibrated = threshold_for(model)
    used = limit if threshold is None else threshold
    return {
        "object": "speaker.verification",
        "speaker": {"id": speaker.id, "name": speaker.name},
        "similarity": score,
        "threshold": used,
        "match": score >= used,
        "embedding_model": model,
        "calibrated": calibrated,
    }


async def identify(
    core: Bolo,
    samples: FloatArray,
    rate: int,
    candidates: list[Speaker],
    model: str,
    threshold: float | None,
    top_k: int = 5,
) -> dict[str, Any]:
    if not candidates:
        raise InvalidRequest("no speakers are enrolled", param="speakers")
    probe = unit(await similarity.embed(core, samples, rate, model))
    scored = []
    for speaker in candidates:
        if not speaker.clips():
            continue
        ref = await voiceprint(core, speaker, model)
        scored.append((round(similarity.cosine(probe, ref), 4), speaker))
    scored.sort(key=lambda t: -t[0])
    limit, calibrated = threshold_for(model)
    used = limit if threshold is None else threshold
    best = scored[0] if scored and scored[0][0] >= used else None
    return {
        "object": "speaker.identification",
        "speaker": {"id": best[1].id, "name": best[1].name} if best else None,
        "similarity": best[0] if best else (scored[0][0] if scored else None),
        "threshold": used,
        "candidates": [{"id": s.id, "name": s.name, "similarity": v, "match": v >= used} for v, s in scored[:top_k]],
        "embedding_model": model,
        "calibrated": calibrated,
    }
