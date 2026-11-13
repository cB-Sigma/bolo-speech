"""Separating overlapped speech before transcription (opt-in: ``separate_overlaps``).

Where diarization finds two or more people talking at once, the recogniser hears a
mixture and writes one line, usually the louder voice. This module gives each of
those speakers a track of their own for that stretch, with a target-speaker
extraction model (VoxLock): it keeps one voice of a mixture given a few seconds
of that voice. The steps:

1. **Voice samples.** For every speaker, the longest stretches where they talk
   alone (no other speaker's turn, away from any overlap) are joined into a
   sample of up to ``sample_max_s``. A speaker with less than the model's
   minimum (3 s for VoxLock) gets no track, and the report says so.
2. **Windows.** Each overlap of at least ``min_overlap_s`` is widened by
   ``context_s`` on both sides, the edges are moved to the quietest point
   nearby, and windows that touch are joined.
3. **Extraction.** Each window is run through the model once per involved
   speaker that has a voice sample (one extractor per speaker, reused for all
   windows of the request).
4. **Chunks.** Inside a window, each separated speaker's speech is transcribed
   from their own track; the rest of the recording is transcribed from the
   mixture as before. Segments from a track are marked ``separated``.

Anything that cannot be separated (no voice sample, a timeout, a model error)
falls back to today's behaviour: the mixture, with ``overlap: true``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import itertools
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.diarize import Overlap
from bolo.engines.base import FloatArray, SeparationModel, SpeakerTurn
from bolo.errors import BoloError, InvalidRequest

if TYPE_CHECKING:
    from bolo.core import Bolo
    from bolo.obs.trace import Trace

log = logging.getLogger("bolo")

SR = 16000
#: pieces of a turn left over after cutting out a window are dropped below this length (seconds)
MIN_PIECE_S = 0.2
#: speech this close to an overlap is not used for a voice sample (seconds)
SAMPLE_MARGIN_S = 0.25
#: blind separators need a voice print per speaker to assign tracks; this much clean speech is enough (seconds)
BLIND_PRINT_MIN_S = 1.0
#: window edges move to the quietest 20 ms within this distance (seconds)
SNAP_RADIUS_S = 0.4


@dataclass
class SeparateOptions:
    """What a request asked for; ``None`` means the server's setting."""

    model: str | None = None
    context_s: float | None = None
    min_overlap_s: float | None = None
    #: separate audio recorded below 16 kHz (telephony) after resampling, instead of refusing the request
    allow_narrowband: bool = False
    #: /api/v1/diarize only: return the separated tracks as WAV data URLs
    return_audio: bool = False
    #: filled in by the pipeline
    result: SeparationResult | None = None


@dataclass
class Window:
    start: float
    end: float
    overlaps: list[Overlap]
    speakers: list[int]
    #: filled in: speakers that got a track, and why the others did not
    separated: list[int] = field(default_factory=list)
    skipped: dict[int, str] = field(default_factory=dict)


@dataclass
class Chunk:
    """A stretch to transcribe: from the mixture (``track`` None) or from one speaker's separated track."""

    start: float
    end: float
    speaker: int
    track: tuple[int, int] | None = None  # (window index, speaker)
    overlap: bool = False


@dataclass
class SeparationResult:
    model: str
    status: str = "skipped"  # applied | partial | skipped
    reason: str | None = None  # machine-readable code when skipped or partial
    detail: str | None = None
    context_s: float = 0.0
    windows: list[Window] = field(default_factory=list)
    samples: dict[int, float] = field(default_factory=dict)  # speaker -> seconds of voice sample used
    sample_skipped: dict[int, str] = field(default_factory=dict)  # speaker -> reason
    tracks: dict[tuple[int, int], FloatArray] = field(default_factory=dict)
    separated_audio_s: float = 0.0  # window seconds x speakers sent through the model
    time_ms: float = 0.0
    narrowband: bool = False

    def to_dict(self, label: Callable[[int], str], duration: float, *, audio: bool = False) -> dict[str, Any]:
        speakers = sorted(set(self.samples) | set(self.sample_skipped))
        out: dict[str, Any] = {
            "model": self.model,
            "status": self.status,
            "reason": self.reason,
            "detail": self.detail,
            "context_s": self.context_s,
            "speakers": [
                {
                    "speaker": label(s),
                    "sample_s": round(self.samples[s], 2) if s in self.samples else None,
                    **({"skipped": self.sample_skipped[s]} if s in self.sample_skipped else {}),
                }
                for s in speakers
            ],
            "windows": [
                {
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "overlaps": [o.to_dict(label) for o in w.overlaps],
                    "speakers": [label(s) for s in w.speakers],
                    "separated": [label(s) for s in w.separated],
                    "skipped": [{"speaker": label(s), "reason": r} for s, r in sorted(w.skipped.items())],
                    **({"tracks": {label(s): wav_data_url(self.tracks[(i, s)]) for s in w.separated}} if audio else {}),
                }
                for i, w in enumerate(self.windows)
            ],
            "separated_audio_s": round(self.separated_audio_s, 2),
            "time_ms": round(self.time_ms, 1),
            "rtf": round(self.time_ms / 1000.0 / duration, 4) if duration > 0 else None,
        }
        if self.narrowband:
            out["narrowband"] = True
        return out

    def header(self) -> str:
        """Short form for the ``x-bolo-separation`` response header."""
        return self.status if not self.reason else f"{self.status}; reason={self.reason}"


# ----------------------------------------------------------------------
# pure helpers
# ----------------------------------------------------------------------
def subtract(intervals: list[tuple[float, float]], holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """``intervals`` minus ``holes`` (all as (start, end) pairs)."""
    out: list[tuple[float, float]] = []
    for a, b in intervals:
        pieces = [(a, b)]
        for ha, hb in holes:
            nxt: list[tuple[float, float]] = []
            for pa, pb in pieces:
                if hb <= pa or ha >= pb:
                    nxt.append((pa, pb))
                    continue
                if ha > pa:
                    nxt.append((pa, ha))
                if hb < pb:
                    nxt.append((hb, pb))
            pieces = nxt
        out.extend(p for p in pieces if p[1] > p[0])
    return out


def clean_speech(turns: list[SpeakerTurn], margin_s: float = SAMPLE_MARGIN_S) -> dict[int, list[tuple[float, float]]]:
    """Per speaker: stretches where only they talk, at least ``margin_s`` away from anyone else's speech."""
    out: dict[int, list[tuple[float, float]]] = {}
    for speaker in sorted({t.speaker for t in turns}):
        mine = [(t.start, t.end) for t in turns if t.speaker == speaker]
        others = [(t.start - margin_s, t.end + margin_s) for t in turns if t.speaker != speaker]
        out[speaker] = sorted(subtract(mine, others))
    return out


def choose_sample(
    clean: list[tuple[float, float]], min_s: float, max_s: float, min_piece_s: float = 0.5
) -> list[tuple[float, float]]:
    """The voice sample: the longest clean stretches (each at least ``min_piece_s``) up to ``max_s`` in total.

    Returns no stretches when they add up to less than ``min_s``. A single long stretch is cut to ``max_s``.
    """
    picked: list[tuple[float, float]] = []
    total = 0.0
    for a, b in sorted(clean, key=lambda p: p[0] - p[1]):
        if b - a < min_piece_s or total >= max_s:
            break
        take = min(b - a, max_s - total)
        picked.append((a, a + take))
        total += take
    if total + 1e-6 < min_s:
        return []
    return sorted(picked)


def quiet_point(samples: FloatArray, t: float, radius_s: float = SNAP_RADIUS_S, sr: int = SR) -> float:
    """The start of the quietest 20 ms frame within ``radius_s`` of ``t`` (seconds), to cut between words."""
    hop = int(0.02 * sr)
    a = max(0, int((t - radius_s) * sr))
    b = min(len(samples), int((t + radius_s) * sr))
    if b - a < 2 * hop:
        return t
    span = samples[a:b]
    n = len(span) // hop
    frames = span[: n * hop].reshape(n, hop)
    energy = np.mean(frames.astype(np.float64) ** 2, axis=1)
    # prefer the frame nearest to t among (nearly) equally quiet ones
    centre = (t * sr - a) / hop
    score = energy + 1e-9 * np.abs(np.arange(n) - centre)
    return (a + int(np.argmin(score)) * hop) / sr


def plan_windows(
    overlaps: list[Overlap],
    duration: float,
    *,
    context_s: float,
    min_overlap_s: float,
    samples: FloatArray | None = None,
) -> list[Window]:
    """Widen each long-enough overlap by ``context_s``, snap the edges to quiet points, join windows that touch."""
    windows: list[Window] = []
    for o in sorted(overlaps, key=lambda o: o.start):
        if o.end - o.start < min_overlap_s:
            continue
        start, end = max(0.0, o.start - context_s), min(duration, o.end + context_s)
        if samples is not None:
            if start > 0:
                start = min(quiet_point(samples, start), o.start)
            if end < duration:
                end = max(quiet_point(samples, end), o.end)
        if windows and start <= windows[-1].end:
            last = windows[-1]
            last.end = max(last.end, end)
            last.overlaps.append(o)
            last.speakers = sorted(set(last.speakers) | set(o.speakers))
        else:
            windows.append(Window(start, end, [o], list(o.speakers)))
    return windows


def plan_chunks(
    merged: list[SpeakerTurn],
    raw: list[SpeakerTurn],
    windows: list[Window],
    overlaps: list[Overlap],
    *,
    min_piece_s: float = MIN_PIECE_S,
) -> list[Chunk]:
    """What to transcribe from where.

    ``merged`` are the turns that would be transcribed without separation (clipped where they overlap);
    ``raw`` the diarization's own turns. Inside a window, every speaker in ``window.separated`` is
    transcribed from their track, over the span of their raw turns there; their merged turns lose that
    part. Everyone else, and everything outside windows, comes from the mixture.
    """

    def overlapped(a: float, b: float) -> bool:
        return any(o.start < b and o.end > a for o in overlaps)

    chunks: list[Chunk] = []
    for t in merged:
        holes = [(w.start, w.end) for w in windows if t.speaker in w.separated]
        for a, b in subtract([(t.start, t.end)], holes):
            if b - a >= min_piece_s:
                chunks.append(Chunk(a, b, t.speaker, None, overlapped(a, b)))
    for index, w in enumerate(windows):
        for speaker in w.separated:
            spans = [(max(t.start, w.start), min(t.end, w.end)) for t in raw if t.speaker == speaker]
            spans = [(a, b) for a, b in spans if b > a]
            if not spans:
                continue
            a, b = min(s[0] for s in spans), max(s[1] for s in spans)
            if b - a >= min_piece_s:
                chunks.append(Chunk(a, b, speaker, (index, speaker), True))
    return sorted(chunks, key=lambda c: (c.start, c.speaker))


def split_long(chunks: list[Chunk], max_s: float) -> list[Chunk]:
    """Cut chunks longer than ``max_s`` into equal parts (the recogniser's longest input)."""
    out: list[Chunk] = []
    for c in chunks:
        parts = max(1, int(np.ceil((c.end - c.start) / max_s - 1e-9)))
        step = (c.end - c.start) / parts
        out.extend(
            Chunk(c.start + i * step, c.start + (i + 1) * step, c.speaker, c.track, c.overlap) for i in range(parts)
        )
    return out


def is_narrowband(samples: FloatArray, sr: int = SR) -> bool:
    """True when (almost) nothing is above 4 kHz: telephone audio, even if it was resampled to 16 kHz."""
    if sr < 16000:
        return True
    n = min(len(samples), 30 * sr)
    if n < sr // 2:
        return False
    spectrum = np.abs(np.fft.rfft(samples[:n].astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    speech = float(spectrum[(freqs >= 100) & (freqs < 8000)].sum())
    high = float(spectrum[(freqs >= 4200) & (freqs < 8000)].sum())
    return bool(speech > 0 and high / speech < 1e-4)


def wav_data_url(samples: FloatArray, sr: int = SR) -> str:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.clip(samples, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def check_request(sample_rate: int, opts: SeparateOptions) -> None:
    """Refuse what separation does not support (400), before any work is done."""
    if sample_rate < 16000 and not opts.allow_narrowband:
        raise InvalidRequest(
            f"separate_overlaps needs audio recorded at 16 kHz or more; this upload is {sample_rate} Hz. "
            "The separation model was trained on wideband speech and did not help on telephone audio in our "
            "tests (docs/models.md, 'Overlapping speech'). Send separation_allow_narrowband=true to try anyway.",
            param="separate_overlaps",
        )
    for name in ("context_s", "min_overlap_s"):
        value = getattr(opts, name)
        if value is not None and not 0.0 <= value <= 10.0:
            raise InvalidRequest(f"separation_{name} must be between 0 and 10", param=f"separation_{name}")


# ----------------------------------------------------------------------
# the pipeline step
# ----------------------------------------------------------------------
async def separate(
    core: Bolo,
    samples16: FloatArray,
    raw: list[SpeakerTurn],
    overlaps: list[Overlap],
    opts: SeparateOptions,
    trace: Trace,
    *,
    narrowband: bool = False,
    diarization_model: str | None = None,
) -> SeparationResult:
    """Steps 1-3 of the module docstring. Adds a ``separate`` phase to ``trace``; never raises for a
    separation failure after the model is loaded (the result says what was skipped and why)."""
    s = core.settings
    spec = core.resolve(opts.model, "separation")
    duration = len(samples16) / SR
    context = s.separation_context_s if opts.context_s is None else opts.context_s
    min_overlap = s.separation_min_overlap_s if opts.min_overlap_s is None else opts.min_overlap_s
    result = SeparationResult(spec.id, context_s=context, narrowband=narrowband)
    started = trace.now()
    t0 = time.perf_counter()
    try:
        result.windows = plan_windows(
            overlaps, duration, context_s=context, min_overlap_s=min_overlap, samples=samples16
        )
        if not result.windows:
            result.reason = "no_overlap"
            result.detail = (
                f"diarization found no overlap of {min_overlap:g} s or more"
                if overlaps
                else "diarization found no overlapping speech"
            )
            return result
        involved = sorted({sp for w in result.windows for sp in w.speakers})
        clean = clean_speech(raw)
        deadline = time.monotonic() + max(1.0, s.separation_timeout_s)
        async with core.manager.use(spec.id) as entry:
            model = entry.model
            assert isinstance(model, SeparationModel)
            blind = model.mode == "blind"
            need = BLIND_PRINT_MIN_S if blind else model.min_sample_s
            sample_max = max(s.separation_sample_max_s, need)
            samples_by_speaker: dict[int, FloatArray] = {}
            for speaker in involved:
                spans = choose_sample(clean.get(speaker, []), need, sample_max)
                if not spans:
                    have = sum(b - a for a, b in clean.get(speaker, []))
                    result.sample_skipped[speaker] = (
                        f"not_enough_clean_speech: {have:.1f} s without overlap, {need:g} s needed"
                    )
                    continue
                samples_by_speaker[speaker] = np.concatenate([samples16[int(a * SR) : int(b * SR)] for a, b in spans])
            try:
                if blind:
                    await _run_blind(
                        core, entry, result, samples16, raw, samples_by_speaker, diarization_model, deadline
                    )
                else:
                    await _run_target(entry, model, result, samples16, samples_by_speaker, deadline)
            except (TimeoutError, asyncio.TimeoutError):
                result.reason = "timeout"
                result.detail = f"separation stopped after separation_timeout_s = {s.separation_timeout_s:g} s"
            except BoloError:
                raise
            except Exception as exc:  # a model failure: fall back to the mixture for what is left
                log.warning("overlap separation failed: %s: %s", type(exc).__name__, exc)
                result.reason = "error"
                result.detail = f"{type(exc).__name__}: {exc}"
        done = sum(len(w.separated) for w in result.windows)
        wanted = sum(len(w.speakers) for w in result.windows)
        if done == 0:
            result.status = "skipped"
            if result.reason is None:
                result.reason = "no_voice_sample"
                result.detail = (
                    f"no speaker in an overlap has {need:g} s of speech without overlap to use as a voice sample"
                )
        elif done < wanted or result.reason is not None:
            result.status = "partial"
            if result.reason is None:
                result.reason = "no_voice_sample"
                result.detail = "some speakers in overlaps had too little speech without overlap for a voice sample"
        else:
            result.status = "applied"
        return result
    finally:
        result.time_ms = (time.perf_counter() - t0) * 1000.0
        trace.add_phase("separate", started, trace.now())
        trace.extra["separation"] = result.header()
        trace.extra["separation_model"] = spec.id
        trace.extra["separation_windows"] = sum(1 for w in result.windows if w.separated)
        trace.extra["separation_ms"] = round(result.time_ms, 1)
        core.note_separation(result)


async def _run_target(
    entry: Any,
    model: SeparationModel,
    result: SeparationResult,
    samples16: FloatArray,
    voice_samples: dict[int, FloatArray],
    deadline: float,
) -> None:
    """Target-speaker models: enrol every speaker once, then extract each of them from each window."""
    prefix = uuid.uuid4().hex[:12]
    enrolled: set[int] = set()
    try:
        for speaker, sample in voice_samples.items():
            await _with_deadline(entry.queue.submit("enrol", (f"{prefix}:{speaker}", sample)), deadline)
            enrolled.add(speaker)
            result.samples[speaker] = round(len(sample) / SR, 2)
        for index, w in enumerate(result.windows):
            mixture = np.ascontiguousarray(samples16[int(w.start * SR) : int(w.end * SR)])
            for speaker in w.speakers:
                if speaker not in enrolled:
                    w.skipped[speaker] = result.sample_skipped.get(speaker, "no_voice_sample").split(":")[0]
                    continue
                track, _ = await _with_deadline(
                    entry.queue.submit("extract", (f"{prefix}:{speaker}", mixture)), deadline
                )
                result.tracks[(index, speaker)] = _fit(track, len(mixture))
                result.separated_audio_s += len(mixture) / SR
                w.separated.append(speaker)
    finally:
        for speaker in enrolled:
            model.release(f"{prefix}:{speaker}")


async def _run_blind(
    core: Bolo,
    entry: Any,
    result: SeparationResult,
    samples16: FloatArray,
    raw: list[SpeakerTurn],
    voice_samples: dict[int, FloatArray],
    diarization_model: str | None,
    deadline: float,
) -> None:
    """Blind models: split each window into tracks, then give each track to the window's speaker whose voice
    print (from their speech outside overlaps) it matches best, one track per speaker."""
    diar = core.resolve(diarization_model, "diarization")
    async with core.manager.use(diar.id) as dentry:
        prints: dict[int, FloatArray] = {}
        for speaker, sample in voice_samples.items():
            prints[speaker], _ = await _with_deadline(dentry.queue.submit("embed", sample), deadline)
            result.samples[speaker] = round(len(sample) / SR, 2)
        for index, w in enumerate(result.windows):
            mixture = np.ascontiguousarray(samples16[int(w.start * SR) : int(w.end * SR)])
            tracks, _ = await _with_deadline(entry.queue.submit("separate", mixture), deadline)
            result.separated_audio_s += len(mixture) / SR
            # embed each track where the speakers overlap: that is where the tracks differ
            a = int((min(o.start for o in w.overlaps) - w.start) * SR)
            b = int((max(o.end for o in w.overlaps) - w.start) * SR)
            tracks = [_fit(t, len(mixture)) for t in tracks]
            vectors = []
            for track in tracks:
                vector, _ = await _with_deadline(dentry.queue.submit("embed", track[a:b]), deadline)
                vectors.append(vector)
            candidates = [sp for sp in w.speakers if sp in prints]
            similarity = {
                (t, sp): float(np.dot(vectors[t], prints[sp])) for t in range(len(tracks)) for sp in candidates
            }
            share = solo_energy_share(tracks, raw, w, candidates)
            for sp, t in assign_tracks(similarity, share, len(tracks), candidates).items():
                w.separated.append(sp)
                result.tracks[(index, sp)] = tracks[t]
            w.separated.sort()
            for sp in w.speakers:
                if sp not in w.separated:
                    w.skipped[sp] = (
                        result.sample_skipped.get(sp, "").split(":")[0] if sp not in prints else "no_matching_track"
                    )


def solo_energy_share(
    tracks: list[FloatArray], raw: list[SpeakerTurn], window: Window, speakers: list[int]
) -> dict[tuple[int, int], float]:
    """For each (track, speaker): the track's share of all tracks' energy where only that speaker talks inside
    the window (the context around the overlap). A track that carries a speaker's solo speech is that speaker's."""
    out: dict[tuple[int, int], float] = {}
    for sp in speakers:
        mine = [(max(t.start, window.start), min(t.end, window.end)) for t in raw if t.speaker == sp]
        others = [(t.start, t.end) for t in raw if t.speaker != sp]
        solo = [(a, b) for a, b in subtract([p for p in mine if p[1] > p[0]], others) if b - a >= 0.2]
        energy = []
        for track in tracks:
            e = 0.0
            for a, b in solo:
                piece = track[int((a - window.start) * SR) : int((b - window.start) * SR)]
                e += float(np.sum(piece.astype(np.float64) ** 2))
            energy.append(e)
        total = sum(energy)
        for t, e in enumerate(energy):
            out[(t, sp)] = e / total if total > 0 else 0.0
    return out


def assign_tracks(
    similarity: dict[tuple[int, int], float],
    share: dict[tuple[int, int], float],
    n_tracks: int,
    speakers: list[int],
) -> dict[int, int]:
    """Give each speaker at most one track: the one-to-one assignment with the highest total score, where a pair
    scores voice similarity (cosine, in the overlap) plus solo-energy share. Pairs whose voices are not similar at
    all (cosine <= 0) are never made. Returns ``{speaker: track}``."""
    best: tuple[float, dict[int, int]] = (-1.0, {})
    slots: list[int | None] = [*range(n_tracks), *([None] * len(speakers))]
    for perm in itertools.permutations(slots, len(speakers)):
        if any(t is not None and perm.count(t) > 1 for t in perm):
            continue
        chosen = {sp: t for sp, t in zip(speakers, perm) if t is not None and similarity.get((t, sp), -1.0) > 0}
        score = sum(similarity[(t, sp)] + share.get((t, sp), 0.0) + 1.0 for sp, t in chosen.items())
        if score > best[0]:
            best = (score, chosen)
    return best[1]


def _fit(track: Any, n: int) -> FloatArray:
    audio = np.asarray(track, dtype=np.float32)
    if len(audio) < n:
        audio = np.pad(audio, (0, n - len(audio)))
    return np.ascontiguousarray(audio[:n])


async def _with_deadline(job: Any, deadline: float) -> Any:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        if asyncio.iscoroutine(job):
            job.close()
        raise TimeoutError
    return await asyncio.wait_for(job, remaining)
