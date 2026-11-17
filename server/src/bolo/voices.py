"""Saved voices: a library of cleaned reference recordings for the models that clone voices.

A saved voice is a folder ``<data_dir>/voices/<id>/`` holding ``reference.wav`` (mono PCM16 at 24 kHz, silence
trimmed, loudness normalised, 2 to 60 s) and ``voice.json`` (name, transcript, language, description, the consent
statement with who gave it and when, the analysis of the recording, an optional default model and the speaker
similarity measured per model). Any request that names the voice (``voice="priya"`` or ``voice={"id": "priya"}``
in ``/v1/audio/speech``) with a model that clones voices is spoken from that reference.

A saved *blend* is another kind of entry: ``voice.json`` only, with ``"kind": "blend"``, the blend of a model's
own voices (``af_bella(0.7)+af_sky(0.3)``, :mod:`bolo.engines.blend`) and the model it is for. It imitates no
one, so it needs no recording and no consent; it is listed apart from the recordings (``blends``).

Creating a voice needs an explicit consent statement: ``self`` ("this is my voice") or ``permission`` ("I have
the speaker's permission"). The store refuses a voice without one, and an import whose metadata has none.

This module has no heavy imports; the HTTP routes (:mod:`bolo.api.voices`) and ``bolo voices`` use it.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import re
import shutil
import threading
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import FloatArray
from bolo.errors import BoloError, InvalidRequest

#: sample rate of a stored reference (what the cloning servers resample to anyway)
VOICE_RATE = 24000
#: less speech than this (seconds, after trimming) is refused: no model clones from it
MIN_SPEECH_S = 2.0
#: less than this is accepted with a warning (most models want 5 to 30 s)
SHORT_S = 5.0
#: a reference is cut to this length (at a pause where there is one)
MAX_S = 60.0
#: loudness the speech is normalised to (RMS of the voiced frames, dBFS); peaks stay below -1 dBFS
TARGET_DBFS = -20.0
#: a speech-to-background estimate below this (dB) earns a "noisy" warning: the 90th minus the 10th percentile of
#: 25 ms frame levels. On LibriSpeech clips it was 18 to 55 dB clean, about 24 with white noise 20 dB under the
#: speech and about 14.5 with noise 10 dB under it
NOISY_SNR_DB = 16.0
#: a 3 s window of a reference less similar than this to the rest earns a "more than one speaker" warning (cosine
#: similarity of speaker embeddings; on LibriSpeech one person's windows scored 0.475 or more, two people's
#: median 0.23: docs/measurements/speaker-similarity-calibration.json)
MULTI_SPEAKER_SIMILARITY = 0.40
MAX_VOICES = 1000
METADATA = "voice.json"
REFERENCE = "reference.wav"
EXPORT_FORMAT = "bolo-voices"
EXPORT_VERSION = 1
#: ``kind`` of a saved blend (a recording has no ``kind``)
BLEND = "blend"

#: the consent statements a voice can be created with
CONSENT = {
    "self": "This is my voice.",
    "permission": "I have the speaker's permission to clone this voice.",
}
_CONSENT_ALIASES = {
    "self": "self", "mine": "self", "my-voice": "self", "my_voice": "self", "own": "self",
    "permission": "permission", "speaker-permission": "permission", "speaker_permission": "permission",
    "permitted": "permission",
}  # fmt: skip
#: OpenAI's voice names: a stock client sends one for an ordinary voice, so a saved voice cannot take one
RESERVED = (
    "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse", "marin", "cedar",
)  # fmt: skip
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ConsentRequired(InvalidRequest):
    code = "consent_required"


class VoiceNotFound(BoloError):
    status = 404
    code = "voice_not_found"
    type = "invalid_request_error"


class VoiceExists(BoloError):
    status = 409
    code = "voice_exists"
    type = "invalid_request_error"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def slug(name: str) -> str:
    """A voice id from its name: ``"Priya S."`` -> ``priya-s``."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text).strip("-_")
    return text[:64].strip("-_")


def check_id(voice_id: str, param: str = "id") -> str:
    vid = voice_id.strip().lower()
    if not _ID_RE.match(vid):
        raise InvalidRequest(
            f"invalid voice id {voice_id!r}: lowercase letters, digits, '-' and '_' (up to 64)", param=param
        )
    if vid in RESERVED:
        raise InvalidRequest(
            f"'{vid}' is one of OpenAI's voice names, which stock clients send for ordinary voices; choose another "
            "name",
            param=param,
        )
    return vid


def parse_consent(value: Any) -> str:
    """The consent basis (``self`` or ``permission``) from a request field; refuses anything else."""
    text = str(value).strip().lower() if value is not None and value is not False else ""
    basis = _CONSENT_ALIASES.get(text)
    if basis is None:
        raise ConsentRequired(
            'creating a voice needs explicit consent: send consent=self ("this is my voice") or '
            'consent=permission ("I have the speaker\'s permission")',
            param="consent",
        )
    return basis


# -- cleaning a recording -----------------------------------------------------------------------------------------


@dataclass
class Cleaned:
    samples: FloatArray
    sample_rate: int
    analysis: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _frames_db(x: FloatArray, frame: int) -> FloatArray:
    n = len(x) // frame
    if n == 0:
        return np.full(1, -120.0, dtype=np.float32)
    blocks = x[: n * frame].reshape(n, frame).astype(np.float64)
    rms = np.sqrt(np.mean(blocks**2, axis=1) + 1e-12)
    return (20 * np.log10(rms + 1e-9)).astype(np.float32)


def _remove_dc(x: FloatArray, rate: int) -> FloatArray:
    """``x`` without its DC offset, taken as the 0.1 s moving average (a high-pass well below speech) rather than
    the clip's mean: in synthetic speech the offset can come and go with the voice, and subtracting one mean then
    turns digital silence into a steady "background" 27 dB under the speech, which the noise estimate reported as
    a noisy recording."""
    n = len(x)
    win = max(1, int(rate * 0.1))
    if n < 2 * win:
        return (x - float(np.mean(x))).astype(np.float32) if n else x
    half = win // 2
    padded = np.pad(np.asarray(x, dtype=np.float64), (half, win - 1 - half), mode="reflect")
    sums = np.concatenate([[0.0], np.cumsum(padded)])
    average = (sums[win:] - sums[:-win]) / win
    out: FloatArray = (x - average).astype(np.float32)
    return out


def _voiced(db: FloatArray) -> np.ndarray:
    """Frames that hold speech: within 35 dB of the loud frames and well above the quietest ones."""
    loud = float(np.percentile(db, 95))
    floor = float(np.percentile(db, 10))
    threshold = max(loud - 35.0, -60.0)
    if loud - floor > 12.0:  # there is a background to rise above (not one steady sound throughout)
        threshold = max(threshold, min(floor + 6.0, loud - 6.0))
    return np.asarray(db > threshold)


def clean_reference(clips: list[tuple[FloatArray, int]]) -> Cleaned:
    """Trim, normalise and check reference clips; join several with a short pause.

    Steps: mono float at 24 kHz, DC removed (:func:`_remove_dc`); leading and trailing silence trimmed (25 ms frames, energy relative
    to the clip's loud frames) keeping 0.15 s; pauses longer than 0.4 s shortened to 0.4 s; voiced frames brought
    to ``TARGET_DBFS`` with peaks under -1 dBFS; at most ``MAX_S`` seconds (cut at a pause). Raises when less than
    ``MIN_SPEECH_S`` seconds of speech remain. Warnings: short, noisy, clipped, cut to 60 s."""
    from bolo.audio.resample import resample

    if not clips:
        raise InvalidRequest("no reference recording was given", param="file")
    rate = VOICE_RATE
    frame = int(rate * 0.025)
    pieces: list[FloatArray] = []
    warnings: list[str] = []
    clipped = 0
    total_in = 0.0
    snrs: list[float] = []
    for samples, sr in clips:
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1).astype(np.float32)
        total_in += len(x) / sr if sr else 0.0
        clipped += int(np.sum(np.abs(x) >= 0.999))
        x = resample(x, sr, rate) if sr != rate else x
        x = _remove_dc(x, rate)
        db = _frames_db(x, frame)
        voiced = _voiced(db)
        if not voiced.any():
            continue
        if len(db) >= 40:  # a second or more: enough frames for the estimate
            snrs.append(float(np.percentile(db, 90) - np.percentile(db, 10)))
        idx = np.flatnonzero(voiced)
        pad = int(0.15 * rate / frame)
        first, last = max(0, int(idx[0]) - pad), min(len(voiced), int(idx[-1]) + 1 + pad)
        kept: list[FloatArray] = []
        gap = 0
        keep_gap = int(0.4 * rate / frame)
        for i in range(first, last):
            gap = 0 if voiced[i] else gap + 1
            if gap > keep_gap:
                continue  # the rest of a long pause
            kept.append(x[i * frame : (i + 1) * frame])
        if kept:
            pieces.append(np.concatenate(kept))
    if not pieces:
        raise InvalidRequest("no speech was found in the reference recording", param="file")
    pause = np.zeros(int(0.3 * rate), dtype=np.float32)
    joined: list[FloatArray] = []
    for i, p in enumerate(pieces):
        if i:
            joined.append(pause)
        joined.append(p)
    y = np.concatenate(joined).astype(np.float32)
    db = _frames_db(y, frame)
    voiced = _voiced(db)
    speech_s = float(voiced.sum()) * frame / rate
    if speech_s < MIN_SPEECH_S:
        raise InvalidRequest(
            f"the reference holds only {speech_s:.1f} s of speech; give at least {MIN_SPEECH_S:.0f} s "
            f"(5 to 30 s of one person speaking clearly works best)",
            param="file",
        )
    if len(y) > MAX_S * rate:
        cut = int(MAX_S * rate) // frame
        quiet = np.flatnonzero(~voiced[: cut + 1])
        end = int(quiet[-1]) if len(quiet) and quiet[-1] > cut * 0.8 else cut
        y = y[: end * frame]
        warnings.append(f"the reference was longer than {MAX_S:.0f} s; the first {len(y) / rate:.0f} s were kept")
        db = _frames_db(y, frame)
        voiced = _voiced(db)
    level = float(10 * np.log10(np.mean(10 ** (db[voiced] / 10)) + 1e-12)) if voiced.any() else -60.0
    gain = 10 ** ((TARGET_DBFS - level) / 20)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak * gain > 0.89:  # -1 dBFS
        gain = 0.89 / peak
    y = np.clip(y * gain, -1.0, 1.0).astype(np.float32)
    duration = len(y) / rate
    if duration < SHORT_S:
        warnings.append(f"the reference is short ({duration:.1f} s); most models clone better from 5 to 30 s of speech")
    snr = float(np.median(snrs)) if snrs else None
    if snr is not None and snr < NOISY_SNR_DB:
        warnings.append(
            f"the recording sounds noisy (speech is about {snr:.0f} dB above the background); a quieter room "
            "gives a cleaner clone"
        )
    if clipped > 0.001 * max(1.0, total_in * rate):
        warnings.append("the recording is clipped (too loud at the source); record a little quieter")
    analysis = {
        "duration_s": round(duration, 2),
        "speech_s": round(speech_s, 2),
        "input_s": round(total_in, 2),
        "clips": len(clips),
        "snr_db": round(snr, 1) if snr is not None else None,
        "level_dbfs": round(level, 1),
        "gain_db": round(float(20 * np.log10(gain)), 1) if gain > 0 else None,
    }
    return Cleaned(y, rate, analysis, warnings)


def speaker_windows(samples: FloatArray, rate: int, window_s: float = 3.0) -> list[FloatArray]:
    """16 kHz windows of a reference, for the "more than one speaker" check (at most ten)."""
    from bolo.audio.resample import resample

    x = resample(samples, rate, 16000)
    step = int(window_s * 16000)
    out = [x[i : i + step] for i in range(0, len(x) - step // 2, step)]
    return [w for w in out if len(w) >= step // 2][:10]


def consistency(embeddings: list[FloatArray]) -> float | None:
    """The lowest cosine similarity of a window's embedding to the mean of the others (None: too few)."""
    if len(embeddings) < 2:
        return None
    unit = [e / (np.linalg.norm(e) + 1e-9) for e in embeddings]
    lows = []
    for i, e in enumerate(unit):
        rest = np.mean([u for j, u in enumerate(unit) if j != i], axis=0)
        lows.append(float(np.dot(e, rest / (np.linalg.norm(rest) + 1e-9))))
    return min(lows)


# -- the store ------------------------------------------------------------------------------------------------------


@dataclass
class SavedVoice:
    id: str
    meta: dict[str, Any]
    path: Path

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or self.id)

    @property
    def transcript(self) -> str | None:
        return str(self.meta["transcript"]) if self.meta.get("transcript") else None

    @property
    def default_model(self) -> str | None:
        return str(self.meta["default_model"]) if self.meta.get("default_model") else None

    @property
    def language(self) -> str | None:
        return str(self.meta["language"]) if self.meta.get("language") else None

    @property
    def is_blend(self) -> bool:
        return self.meta.get("kind") == BLEND

    @property
    def blend(self) -> str:
        """A saved blend's voices, ``af_bella(0.7)+af_sky(0.3)`` (empty for a recording)."""
        return str(self.meta.get("blend") or "")

    def reference_bytes(self) -> bytes:
        if self.is_blend:
            raise InvalidRequest(f"'{self.id}' is a saved blend of {self.blend}: it has no recording", param="voice")
        return (self.path / REFERENCE).read_bytes()

    def to_dict(self) -> dict[str, Any]:
        out = {"id": self.id, "object": "voice", **self.meta}
        out["audio_url"] = None if self.is_blend else f"/api/v1/voices/{self.id}/audio"
        return out


class VoiceStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir) / "voices"
        self._lock = threading.RLock()

    def _dir(self, voice_id: str) -> Path:
        return self.root / voice_id

    @staticmethod
    def _complete(path: Path) -> bool:
        """A recording with its reference, or a saved blend (metadata only)."""
        if not (path / METADATA).is_file():
            return False
        if (path / REFERENCE).is_file():
            return True
        try:
            return bool(json.loads((path / METADATA).read_text(encoding="utf-8")).get("kind") == BLEND)
        except (ValueError, OSError, AttributeError):
            return False

    def ids(self) -> list[str]:
        """Every entry: recordings and saved blends."""
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if not p.name.startswith(".") and self._complete(p))

    def _entries(self) -> list[SavedVoice]:
        out = []
        for vid in self.ids():
            with contextlib.suppress(VoiceNotFound, ValueError, OSError):
                out.append(self.get(vid))
        return sorted(out, key=lambda v: str(v.meta.get("created_at") or ""), reverse=True)

    def all(self) -> list[SavedVoice]:
        """The saved recordings (voices to clone), newest first; saved blends are in :meth:`blends`."""
        return [v for v in self._entries() if not v.is_blend]

    def blends(self) -> list[SavedVoice]:
        """The saved blends, newest first."""
        return [v for v in self._entries() if v.is_blend]

    def find(self, name: str | None) -> SavedVoice | None:
        """The saved voice with this id (case-insensitive), or None."""
        if not name or not isinstance(name, str):
            return None
        vid = name.strip().lower()
        if not _ID_RE.match(vid):
            return None
        path = self._dir(vid)
        if not self._complete(path):
            return None
        try:
            return self.get(vid)
        except (VoiceNotFound, ValueError, OSError):
            return None

    def get(self, voice_id: str) -> SavedVoice:
        vid = voice_id.strip().lower()
        path = self._dir(vid)
        if not _ID_RE.match(vid) or not (path / METADATA).is_file():
            raise VoiceNotFound(f"no saved voice '{voice_id}'")
        meta = json.loads((path / METADATA).read_text(encoding="utf-8"))
        return SavedVoice(vid, meta, path)

    def _write_meta(self, path: Path, meta: dict[str, Any]) -> None:
        tmp = path / (METADATA + ".tmp")
        tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path / METADATA)

    def create(
        self,
        *,
        name: str,
        cleaned: Cleaned,
        consent: str,
        created_by: dict[str, Any],
        voice_id: str | None = None,
        transcript: str | None = None,
        language: str | None = None,
        description: str | None = None,
        default_model: str | None = None,
        source: str = "upload",
        extra: dict[str, Any] | None = None,
        consent_record: dict[str, Any] | None = None,
    ) -> SavedVoice:
        """``consent_record`` replaces the statement for a consent given another way (OpenAI's recorded consent,
        ``{"basis": "recording", "recording": "cons_..."}``); ``consent`` is still required."""
        from bolo.audio.encode import encode_audio

        basis = parse_consent(consent)
        name = (name or "").strip()
        if not name:
            raise InvalidRequest("a voice needs a name", param="name")
        if len(name) > 100:
            raise InvalidRequest("a voice name is limited to 100 characters", param="name")
        vid = check_id(voice_id or slug(name), "id" if voice_id else "name")
        if transcript is not None and len(transcript) > 5000:
            raise InvalidRequest("the transcript is limited to 5000 characters", param="transcript")
        with self._lock:
            if len(self.ids()) >= MAX_VOICES:
                raise InvalidRequest(f"the voice library is limited to {MAX_VOICES} voices", param="name")
            path = self._dir(vid)
            if path.exists():
                raise VoiceExists(f"a voice with the id '{vid}' exists; choose another name or delete it first")
            tmp = self.root / f".{vid}.partial"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            try:
                (tmp / REFERENCE).write_bytes(encode_audio(cleaned.samples, cleaned.sample_rate, "wav"))
                at = now_iso()
                meta: dict[str, Any] = {
                    "name": name,
                    "description": (description or "").strip() or None,
                    "language": (language or "").strip() or None,
                    "transcript": (transcript or "").strip() or None,
                    "default_model": (default_model or "").strip() or None,
                    "created_at": at,
                    "consent": {
                        "basis": basis,
                        "statement": CONSENT[basis],
                        **(consent_record or {}),
                        "by": created_by,
                        "at": at,
                    },
                    "source": source,
                    "reference": {"sample_rate": cleaned.sample_rate, **cleaned.analysis},
                    "warnings": list(cleaned.warnings),
                    "scores": {},
                    **(extra or {}),
                }
                self._write_meta(tmp, meta)
                tmp.rename(path)
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
        return self.get(vid)

    def create_blend(
        self,
        *,
        name: str,
        blend: str,
        voices: list[dict[str, Any]],
        model: str,
        created_by: dict[str, Any],
        voice_id: str | None = None,
        language: str | None = None,
        description: str | None = None,
    ) -> SavedVoice:
        """Save a blend of a model's voices under a name. ``blend`` is the canonical spelling and ``voices`` its
        parts (``[{"voice": "af_bella", "weight": 0.7}, ...]``), both checked by the caller against the model."""
        name = (name or "").strip()
        if not name:
            raise InvalidRequest("a blend needs a name", param="name")
        if len(name) > 100:
            raise InvalidRequest("a voice name is limited to 100 characters", param="name")
        vid = check_id(voice_id or slug(name), "id" if voice_id else "name")
        with self._lock:
            if len(self.ids()) >= MAX_VOICES:
                raise InvalidRequest(f"the voice library is limited to {MAX_VOICES} voices", param="name")
            path = self._dir(vid)
            if path.exists():
                raise VoiceExists(f"a voice with the id '{vid}' exists; choose another name or delete it first")
            tmp = self.root / f".{vid}.partial"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            try:
                meta: dict[str, Any] = {
                    "kind": BLEND,
                    "name": name,
                    "description": (description or "").strip() or None,
                    "language": (language or "").strip() or None,
                    "blend": blend,
                    "voices": voices,
                    "default_model": model,
                    "created_at": now_iso(),
                    "created_by": created_by,
                    "source": BLEND,
                }
                self._write_meta(tmp, meta)
                tmp.rename(path)
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
        return self.get(vid)

    def update(self, voice_id: str, changes: dict[str, Any]) -> SavedVoice:
        """Change a voice's details. A saved blend also takes ``blend`` and ``voices`` (checked by the caller)
        and has no transcript."""
        allowed = {"name", "description", "language", "transcript", "default_model"}
        with self._lock:
            voice = self.get(voice_id)
            if voice.is_blend:
                allowed = (allowed - {"transcript"}) | {"blend", "voices"}
            unknown = set(changes) - allowed
            if unknown:
                raise InvalidRequest(f"cannot change {', '.join(sorted(unknown))}", param=sorted(unknown)[0])
            if "voices" in changes:
                voice.meta["voices"] = changes.pop("voices")
            for key, value in changes.items():
                text = None if value is None else str(value).strip()
                if key == "name" and not text:
                    raise InvalidRequest("a voice needs a name", param="name")
                voice.meta[key] = text or None
            voice.meta["updated_at"] = now_iso()
            self._write_meta(voice.path, voice.meta)
        return voice

    def add_warnings(self, voice_id: str, warnings: list[str], **fields: Any) -> SavedVoice:
        with self._lock:
            voice = self.get(voice_id)
            for w in warnings:
                if w not in voice.meta.setdefault("warnings", []):
                    voice.meta["warnings"].append(w)
            voice.meta.setdefault("reference", {}).update(fields)
            self._write_meta(voice.path, voice.meta)
        return voice

    def record_score(self, voice_id: str, model: str, score: dict[str, Any]) -> SavedVoice:
        with self._lock:
            voice = self.get(voice_id)
            voice.meta.setdefault("scores", {})[model] = score
            self._write_meta(voice.path, voice.meta)
        return voice

    def delete(self, voice_id: str) -> None:
        with self._lock:
            voice = self.get(voice_id)
            shutil.rmtree(voice.path)

    # -- export / import ----------------------------------------------------------------------------------------
    def export_zip(self, ids: list[str] | None = None) -> bytes:
        """A .zip of ``manifest.json`` and ``<id>/voice.json`` + ``<id>/reference.wav`` per voice."""
        voices = [self.get(i) for i in ids] if ids else self._entries()
        if not voices:
            raise VoiceNotFound("there are no saved voices to export")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            manifest = {
                "format": EXPORT_FORMAT, "version": EXPORT_VERSION, "exported_at": now_iso(),
                "voices": [v.id for v in voices],
            }  # fmt: skip
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            for v in voices:
                zf.writestr(f"{v.id}/{METADATA}", json.dumps(v.meta, indent=2, ensure_ascii=False))
                if not v.is_blend:  # a blend is its metadata only
                    zf.write(v.path / REFERENCE, f"{v.id}/{REFERENCE}")
        return buf.getvalue()

    def import_zip(
        self, data: bytes, *, imported_by: dict[str, Any], consent: str | None = None, replace: bool = False
    ) -> list[str]:
        """Add the voices of an export. Each must carry a consent record, or ``consent`` is given for all of
        them (refused otherwise). An existing id is skipped unless ``replace``."""
        from bolo.audio import decode_audio

        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise InvalidRequest("not a .zip file", param="file") from exc
        with zf:
            try:
                manifest = json.loads(zf.read("manifest.json"))
            except (KeyError, ValueError) as exc:
                raise InvalidRequest("the archive has no bolo voice manifest (manifest.json)", param="file") from exc
            if manifest.get("format") != EXPORT_FORMAT:
                raise InvalidRequest("the archive is not a bolo voice export", param="file")
            basis = parse_consent(consent) if consent else None
            added: list[str] = []
            for raw_id in manifest.get("voices") or []:
                vid = check_id(str(raw_id))
                try:
                    meta = json.loads(zf.read(f"{vid}/{METADATA}"))
                    is_blend = isinstance(meta, dict) and meta.get("kind") == BLEND
                    audio = b"" if is_blend else zf.read(f"{vid}/{REFERENCE}")
                except (KeyError, ValueError) as exc:
                    raise InvalidRequest(f"voice '{vid}' is incomplete in the archive", param="file") from exc
                if is_blend:
                    if not str(meta.get("blend") or "").strip():
                        raise InvalidRequest(f"blend '{vid}' in the archive names no voices", param="file")
                    with self._lock:
                        if self._dir(vid).exists():
                            if not replace:
                                continue
                            shutil.rmtree(self._dir(vid))
                        meta["imported_at"] = now_iso()
                        meta["imported_by"] = imported_by
                        tmp = self.root / f".{vid}.partial"
                        shutil.rmtree(tmp, ignore_errors=True)
                        tmp.mkdir(parents=True)
                        self._write_meta(tmp, meta)
                        tmp.rename(self._dir(vid))
                    added.append(vid)
                    continue
                record = meta.get("consent") if isinstance(meta.get("consent"), dict) else None
                if basis is None and (not record or record.get("basis") not in CONSENT):
                    raise ConsentRequired(
                        f"voice '{vid}' in the archive has no consent record; import it with consent=self or "
                        "consent=permission",
                        param="consent",
                    )
                samples, rate = decode_audio(audio, filename=REFERENCE)
                if len(samples) < rate * MIN_SPEECH_S or len(samples) > rate * (MAX_S + 1):
                    raise InvalidRequest(f"the reference of voice '{vid}' has an unusable length", param="file")
                with self._lock:
                    if self._dir(vid).exists():
                        if not replace:
                            continue
                        shutil.rmtree(self._dir(vid))
                    at = now_iso()
                    if basis is not None:
                        meta["consent"] = {"basis": basis, "statement": CONSENT[basis], "by": imported_by, "at": at}
                    meta["imported_at"] = at
                    meta["imported_by"] = imported_by
                    tmp = self.root / f".{vid}.partial"
                    shutil.rmtree(tmp, ignore_errors=True)
                    tmp.mkdir(parents=True)
                    (tmp / REFERENCE).write_bytes(audio)
                    self._write_meta(tmp, meta)
                    tmp.rename(self._dir(vid))
                added.append(vid)
        return added


_stores: dict[Path, VoiceStore] = {}


def store_for(data_dir: Path) -> VoiceStore:
    key = Path(data_dir).resolve()
    if key not in _stores:
        _stores[key] = VoiceStore(key)
    return _stores[key]
