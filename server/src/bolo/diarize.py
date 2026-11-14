"""Speaker diarization helpers: labels, turn merging, online labelling and scoring.

Offline diarization (a whole recording) is done by a
:class:`~bolo.engines.base.DiarizationModel`. For transcription the recording
is cut at the speaker turns and every turn is transcribed on its own, so the
speaker of a piece of text is known by construction and does not depend on
the recogniser having word timestamps.

On a live stream there is no whole recording to cluster. There the
:class:`OnlineSpeakerLabeler` embeds each finished utterance and assigns it
to the nearest speaker seen so far, or to a new one. It assumes one speaker
per utterance, which is what a voice-activity detector gives you when people
take turns; overlapping speech is attributed to whoever dominates.

Overlap (offline): the segmentation model (pyannote segmentation-3.0, a
powerset model) can give two speakers at once, and sherpa-onnx then returns
turns of different speakers that overlap in time. :func:`find_overlaps` lists
those stretches, which are marked on transcript segments. :func:`merge_turns`
can keep an overlapping turn whole (``keep_overlap``) instead of clipping it at
the end of the previous one; that is off by default because the recogniser,
hearing the mixture, then wrote the louder voice's words into both texts.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from bolo.engines.base import FloatArray, SpeakerActivity, SpeakerTurn


def speaker_label(index: int) -> str:
    """0 -> "A", 1 -> "B", ... as in OpenAI's diarized transcripts; "S27" beyond "Z"."""
    return chr(ord("A") + index) if 0 <= index < 26 else f"S{index + 1}"


@dataclass
class KnownSpeaker:
    """A named speaker with a short reference recording (16 kHz mono)."""

    name: str
    samples: FloatArray


@dataclass
class DiarizeOptions:
    model: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    threshold: float | None = None
    #: fold spurious speakers into real ones after clustering (None: the server's setting)
    consolidate: bool | None = None
    #: speakers to recognise by voice; a cluster that matches one is given its name
    known: list[KnownSpeaker] = field(default_factory=list)
    #: minimum cosine similarity between a cluster and a reference for the name to be used
    identify_threshold: float | None = None
    #: filled in by the pipeline: cluster index -> label ("A", "B", … or a known speaker's name)
    names: dict[int, str] = field(default_factory=dict)
    #: filled in by the pipeline: known speaker name -> best cosine similarity found
    similarities: dict[str, float] = field(default_factory=dict)
    #: transcribe both speakers' turns in full where they overlap (otherwise the later turn is clipped). Off by
    #: default: in our test the recogniser then wrote the louder speaker's words into the other's text as well
    keep_overlap: bool = False
    #: filled in by the pipeline: stretches where two or more speakers talk at once
    overlaps: list[Overlap] = field(default_factory=list)
    #: filled in by the pipeline when the model gives them (Nemotron-3-Diarization): per-frame speaker
    #: probabilities, one column per speaker index of the turns (``label(k)`` names column ``k``). Not part
    #: of any API response; for components that condition recognition on the diarization.
    activity: SpeakerActivity | None = None

    def label(self, speaker: int) -> str:
        return self.names.get(speaker) or speaker_label(speaker)


def match_known(
    clusters: dict[int, FloatArray], references: list[tuple[str, FloatArray]], threshold: float
) -> tuple[dict[int, str], dict[str, float]]:
    """Give clusters the names of the reference voices they match (unit vectors, cosine similarity).

    Greedy one-to-one assignment, best pair first; pairs below ``threshold`` are left unmatched.
    Returns ``({cluster: name}, {name: best similarity seen})``.
    """
    pairs = sorted(
        ((float(np.dot(vec, ref)), cluster, name) for cluster, vec in clusters.items() for name, ref in references),
        reverse=True,
    )
    best: dict[str, float] = {}
    for sim, _cluster, name in pairs:
        best.setdefault(name, round(sim, 3))
    names: dict[int, str] = {}
    for sim, cluster, name in pairs:
        if sim < threshold:
            break
        if cluster in names or name in names.values():
            continue
        names[cluster] = name
    return names, best


def label_speakers(speakers: list[int], named: dict[int, str]) -> dict[int, str]:
    """Labels for every cluster: the matched names, and A, B, C, … (in order) for the rest."""
    labels: dict[int, str] = {}
    letter = 0
    for speaker in speakers:
        if speaker in named:
            labels[speaker] = named[speaker]
            continue
        while speaker_label(letter) in named.values():
            letter += 1
        labels[speaker] = speaker_label(letter)
        letter += 1
    return labels


def cluster_embeddings(embeddings: list[FloatArray], threshold: float, max_speakers: int | None = None) -> list[int]:
    """Agglomerative clustering (average linkage, cosine similarity) of unit vectors.

    Clusters are merged while the closest pair is at least ``threshold`` similar, or while there are more than
    ``max_speakers``. Returns one cluster index per embedding, numbered in order of first appearance.
    """
    groups: list[list[int]] = [[i] for i in range(len(embeddings))]
    if not groups:
        return []
    sims = np.stack(embeddings) @ np.stack(embeddings).T
    while len(groups) > 1:
        best, pair = -2.0, (0, 0)
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                sim = float(np.mean(sims[np.ix_(groups[i], groups[j])]))
                if sim > best:
                    best, pair = sim, (i, j)
        if best < threshold and not (max_speakers and len(groups) > max_speakers):
            break
        i, j = pair
        groups[i] += groups.pop(j)
    order = sorted(groups, key=min)
    labels = [0] * len(embeddings)
    for index, members in enumerate(order):
        for m in members:
            labels[m] = index
    return labels


def consolidate_speakers(
    turns: list[SpeakerTurn],
    voices: dict[int, FloatArray],
    *,
    min_speech_s: float = 3.0,
    merge_similarity: float = 0.6,
    min_speakers: int = 1,
) -> dict[int, int]:
    """Fold spurious speakers into real ones. Returns ``{speaker: speaker it becomes}``.

    The clustering inside the diarization model tends to find too many speakers: a few seconds of
    laughter, a cough or a change of microphone opens a cluster of its own. Two repairs, repeated
    until nothing changes or ``min_speakers`` is reached:

    * two speakers whose voice prints (``voices``: unit vectors) are at least ``merge_similarity``
      alike are one speaker;
    * a speaker with less than ``min_speech_s`` of speech in total joins the most similar other
      speaker (or, without a voice print, the speaker of the nearest turn).

    The price: somebody who really says less than ``min_speech_s`` in the whole recording is
    attributed to another speaker.
    """
    spoken: dict[int, float] = {}
    for t in turns:
        spoken[t.speaker] = spoken.get(t.speaker, 0.0) + (t.end - t.start)
    groups: dict[int, list[int]] = {s: [s] for s in spoken}
    # on a short recording a real speaker may say very little: never ask for more than a tenth of the speech
    min_speech_s = min(min_speech_s, 0.1 * sum(spoken.values()))

    def voice(group: int) -> FloatArray | None:
        parts = [voices[s] * spoken[s] for s in groups[group] if s in voices]
        if not parts:
            return None
        total = np.sum(parts, axis=0)
        return np.asarray(total / (np.linalg.norm(total) + 1e-9), dtype=np.float32)

    def similarity(a: int, b: int) -> float:
        va, vb = voice(a), voice(b)
        return float(np.dot(va, vb)) if va is not None and vb is not None else -1.0

    def length(group: int) -> float:
        return sum(spoken[s] for s in groups[group])

    def nearest_in_time(group: int) -> int:
        mine = [t for t in turns if t.speaker in groups[group]]
        others = [t for t in turns if t.speaker not in groups[group]]
        best = min(others, key=lambda o: min(abs(o.start - m.end) + abs(m.start - o.end) for m in mine))
        return next(g for g, members in groups.items() if best.speaker in members)

    while len(groups) > max(1, min_speakers):
        pairs = [(similarity(a, b), a, b) for a, b in itertools.combinations(groups, 2)]
        sim, a, b = max(pairs)
        if sim >= merge_similarity:
            groups[a] += groups.pop(b)
            continue
        small = [g for g in groups if length(g) < min_speech_s]
        if not small:
            break
        g = min(small, key=length)
        target = max((o for o in groups if o != g), key=lambda o: similarity(g, o))
        if similarity(g, target) < 0:
            target = nearest_in_time(g)
        groups[target] += groups.pop(g)
    return {member: members[0] for members in groups.values() for member in members}


def merge_same_speaker(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Join turns of one speaker that touch or overlap (after speakers were merged)."""
    out: list[SpeakerTurn] = []
    for t in sorted(turns, key=lambda t: t.start):
        if out and out[-1].speaker == t.speaker and t.start <= out[-1].end + 1e-3:
            out[-1].end = max(out[-1].end, t.end)
        else:
            out.append(SpeakerTurn(t.start, t.end, t.speaker))
    return out


def renumber(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Number speakers in order of first appearance."""
    order: dict[int, int] = {}
    return [
        SpeakerTurn(t.start, t.end, order.setdefault(t.speaker, len(order)))
        for t in sorted(turns, key=lambda t: t.start)
    ]


@dataclass
class Overlap:
    start: float
    end: float
    speakers: list[int]

    def to_dict(self, label: Callable[[int], str]) -> dict[str, object]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "speakers": [label(s) for s in self.speakers]}


def find_overlaps(turns: list[SpeakerTurn], min_s: float = 0.25) -> list[Overlap]:
    """Stretches where two or more speakers talk at once (at least ``min_s`` long), in time order."""
    edges = sorted({t.start for t in turns} | {t.end for t in turns})
    out: list[Overlap] = []
    for a, b in itertools.pairwise(edges):
        active = sorted({t.speaker for t in turns if t.start < b and t.end > a})
        if len(active) < 2:
            continue
        if out and abs(out[-1].end - a) < 1e-6 and out[-1].speakers == active:
            out[-1].end = b
        else:
            out.append(Overlap(a, b, active))
    return [o for o in out if o.end - o.start >= min_s]


def merge_turns(
    turns: list[SpeakerTurn],
    *,
    max_gap_s: float = 1.0,
    max_len_s: float = 30.0,
    min_len_s: float = 0.2,
    keep_overlap: bool = False,
    min_overlap_s: float = 0.25,
) -> list[SpeakerTurn]:
    """Join consecutive turns of the same speaker into chunks worth transcribing.

    A turn that starts before the previous one (of another speaker) ends is clipped to start where the
    previous one ends, unless ``keep_overlap`` is set and the overlap lasts ``min_overlap_s`` or more: then
    both turns keep their full extent and the overlapping audio is transcribed twice, once for each speaker.
    """
    out: list[SpeakerTurn] = []
    for t in sorted(turns, key=lambda t: t.start):
        if t.end - t.start < min_len_s and not (out and out[-1].speaker == t.speaker):
            continue
        last = out[-1] if out else None
        if last and last.speaker == t.speaker and t.start - last.end <= max_gap_s and t.end - last.start <= max_len_s:
            last.end = max(last.end, t.end)
            continue
        prev = out[-1] if out else None
        if prev and t.start < prev.end and not (keep_overlap and min(prev.end, t.end) - t.start >= min_overlap_s):
            out.append(SpeakerTurn(prev.end, t.end, t.speaker))
        else:
            out.append(SpeakerTurn(t.start, t.end, t.speaker))
    return sorted((t for t in out if t.end - t.start >= min_len_s), key=lambda t: t.start)


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    """``(start, end)`` frame ranges where ``active`` is true."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], active.astype(np.int8), [0]])))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def activity_to_turns(
    activity: SpeakerActivity,
    *,
    threshold: float = 0.5,
    offset: float | None = None,
    pad_onset: float = 0.0,
    pad_offset: float = 0.0,
    min_duration_on: float = 0.0,
    min_duration_off: float = 0.0,
    min_speaker_s: float = 0.0,
) -> tuple[list[SpeakerTurn], SpeakerActivity]:
    """Speaker turns from per-frame speaker probabilities (the binarisation NeMo applies to Sortformer output).

    A speaker starts talking where their probability rises above ``threshold`` and stops where it falls
    below ``offset`` (hysteresis; default: ``threshold``). Each turn is widened by ``pad_onset`` /
    ``pad_offset`` seconds, gaps shorter than ``min_duration_off`` within one speaker are closed, turns
    shorter than ``min_duration_on`` are dropped, and so is a speaker with less than ``min_speaker_s`` of
    speech in all. Turns of different speakers may overlap: that is overlapped speech.

    Returns the turns (speakers renumbered in order of first appearance, sorted by start) and the
    probabilities of the speakers kept, columns in the same order.
    """
    probs = np.asarray(activity.probs, dtype=np.float32)
    step = activity.frame_s
    duration = len(probs) * step
    low = threshold if offset is None else min(offset, threshold)
    per_speaker: dict[int, list[tuple[float, float]]] = {}
    for k in range(probs.shape[1] if probs.ndim == 2 else 0):
        p = probs[:, k]
        event = np.where(p > threshold, 1, np.where(p < low, -1, 0)).astype(np.int8)
        index = np.where(event != 0, np.arange(len(p)), 0)
        np.maximum.accumulate(index, out=index)
        state = event[index]
        state[(index == 0) & (event[0] == 0)] = -1
        spans: list[tuple[float, float]] = []
        for a, b in _runs(state == 1):
            start, end = max(0.0, a * step - pad_onset), min(duration, b * step + pad_offset)
            if spans and start - spans[-1][1] <= min_duration_off:
                spans[-1] = (spans[-1][0], max(spans[-1][1], end))
            else:
                spans.append((start, end))
        spans = [(a, b) for a, b in spans if b - a >= max(min_duration_on, step / 2)]
        if spans and sum(b - a for a, b in spans) >= min_speaker_s:
            per_speaker[k] = spans
    order = sorted(per_speaker, key=lambda k: per_speaker[k][0][0])
    turns = sorted(
        (SpeakerTurn(a, b, new) for new, k in enumerate(order) for a, b in per_speaker[k]), key=lambda t: t.start
    )
    kept = probs[:, order] if order else np.zeros((len(probs), 0), np.float32)
    return turns, SpeakerActivity(np.ascontiguousarray(kept), step)


def remap_activity(activity: SpeakerActivity, mapping: dict[int, int]) -> SpeakerActivity:
    """Columns merged as speakers were merged: ``mapping`` old speaker -> new speaker; a merged column is the
    frame-wise maximum of its parts. Speakers missing from ``mapping`` are dropped."""
    n = max(mapping.values(), default=-1) + 1
    out = np.zeros((len(activity.probs), n), np.float32)
    for old, new in mapping.items():
        if 0 <= old < activity.num_speakers:
            out[:, new] = np.maximum(out[:, new], activity.probs[:, old])
    return SpeakerActivity(out, activity.frame_s)


def turns_to_activity(
    turns: list[SpeakerTurn], duration: float, *, frame_s: float = 0.01, num_speakers: int | None = None
) -> SpeakerActivity:
    """0/1 speaker activity from speaker turns: column ``k`` is 1 in the frames where speaker ``k`` talks."""
    n_spk = num_speakers if num_speakers is not None else max((t.speaker for t in turns), default=-1) + 1
    rate = 1.0 / frame_s  # times x rate, as DiCoW's reference does (x / frame_s rounds differently at .5)
    probs = np.zeros((max(0, round(duration * rate)), n_spk), np.float32)
    for t in turns:
        if 0 <= t.speaker < n_spk:
            probs[max(0, round(t.start * rate)) : max(0, round(t.end * rate)), t.speaker] = 1.0
    return SpeakerActivity(probs, frame_s)


#: how a speaker-conditioned recogniser is told who talks when (the model's ``conditioning`` option)
CONDITIONING = ("turns", "probabilities")
#: frame step of activity made from turns for conditioned recognition
TURN_FRAME_S = 0.02


def conditioning_activity(
    turns: list[SpeakerTurn], activity: SpeakerActivity | None, duration: float, mode: str = "turns"
) -> tuple[SpeakerActivity, str]:
    """The speaker activity to condition recognition on, and which kind it is.

    ``"turns"``: 0/1 from the final speaker turns (any diarizer; what DiCoW's reference pipeline does), on a 20 ms
    grid: the 50 Hz of a Whisper encoder's input, so frame ``j`` is active exactly where the reference's
    ``round(start * 50) <= j < round(end * 50)`` is.
    ``"probabilities"``: the diarizer's per-frame probabilities when it gives them (Nemotron) and they have
    one column per speaker of the turns; otherwise the turns.
    """
    if mode not in CONDITIONING:
        raise ValueError(f"unknown conditioning {mode!r} (one of {', '.join(CONDITIONING)})")
    n_spk = max((t.speaker for t in turns), default=-1) + 1
    if mode == "probabilities" and activity is not None and activity.num_speakers == n_spk:
        return activity, "probabilities"
    return turns_to_activity(turns, duration, frame_s=TURN_FRAME_S, num_speakers=n_spk), "turns"


def limit_speakers(
    turns: list[SpeakerTurn], activity: SpeakerActivity, max_speakers: int
) -> tuple[list[SpeakerTurn], SpeakerActivity]:
    """Fold the speakers who talk least into the others until ``max_speakers`` are left.

    Each folded speaker joins the remaining speaker the model found most likely during the folded
    speaker's turns (its second guess for those frames).
    """
    spoken: dict[int, float] = {}
    for t in turns:
        spoken[t.speaker] = spoken.get(t.speaker, 0.0) + t.end - t.start
    keep = sorted(spoken, key=lambda s: -spoken[s])[: max(1, max_speakers)]
    target: dict[int, int] = {s: s for s in keep}
    for s in spoken:
        if s in target:
            continue
        frames = np.zeros(len(activity.probs), bool)
        for t in turns:
            if t.speaker == s:
                frames[activity.frame(t.start) : activity.frame(t.end)] = True
        scores = {k: float(activity.probs[frames, k].sum()) if k < activity.num_speakers else 0.0 for k in keep}
        target[s] = max(keep, key=lambda k: scores[k])
    merged = merge_same_speaker([SpeakerTurn(t.start, t.end, target[t.speaker]) for t in turns])
    first: dict[int, int] = {}
    for t in sorted(merged, key=lambda t: t.start):
        first.setdefault(t.speaker, len(first))
    mapping = {s: first[target[s]] for s in spoken}
    return renumber(merged), remap_activity(activity, mapping)


class OnlineSpeakerLabeler:
    """Incremental speaker assignment from per-utterance embeddings (cosine similarity to centroids)."""

    def __init__(self, threshold: float = 0.45, max_speakers: int | None = None) -> None:
        self.threshold = threshold
        self.max_speakers = max_speakers
        self._sums: list[FloatArray] = []
        self.last: int | None = None
        self.embeddings: list[FloatArray] = []  # one per labelled utterance, for the revision at the end
        self.assigned: list[int] = []

    @property
    def num_speakers(self) -> int:
        return len(self._sums)

    def assign(self, embedding: FloatArray) -> int:
        norm = float(np.linalg.norm(embedding))
        e = embedding / norm if norm > 0 else embedding
        best, best_sim = -1, -1.0
        for i, total in enumerate(self._sums):
            sim = float(np.dot(e, total / (np.linalg.norm(total) + 1e-9)))
            if sim > best_sim:
                best, best_sim = i, sim
        full = self.max_speakers is not None and len(self._sums) >= self.max_speakers
        if best >= 0 and (best_sim >= self.threshold or full):
            self._sums[best] = self._sums[best] + e
            self.last = best
        else:
            self._sums.append(e.copy())
            self.last = len(self._sums) - 1
        self.embeddings.append(e)
        self.assigned.append(self.last)
        return self.last

    def revise(self) -> list[int]:
        """Cluster every utterance heard so far again, now that all of them are known.

        The online pass has to decide on each utterance as it arrives, so an early mistake (most often: a
        speaker's first short utterance opening a cluster of its own) can never be undone. With the whole
        stream available the embeddings are clustered together. Returns the revised speaker of each
        utterance, numbered in order of first appearance.
        """
        return cluster_embeddings(self.embeddings, self.threshold, self.max_speakers)


def diarization_error_rate(
    reference: list[SpeakerTurn], hypothesis: list[SpeakerTurn], *, step_s: float = 0.01, collar_s: float = 0.0
) -> dict[str, float]:
    """Frame-based diarization error rate with the best one-to-one speaker mapping.

    DER = (missed speech + false alarm + speaker confusion) / reference speech.
    ``collar_s`` ignores that much time on each side of every reference boundary.
    """
    end = max([t.end for t in reference + hypothesis], default=0.0)
    n = int(end / step_s) + 1

    def frames(turns: list[SpeakerTurn]) -> np.ndarray:
        arr = np.full(n, -1, dtype=np.int32)
        for t in turns:
            arr[round(t.start / step_s) : round(t.end / step_s)] = t.speaker
        return arr

    ref, hyp = frames(reference), frames(hypothesis)
    scored = np.ones(n, dtype=bool)
    if collar_s > 0:
        for t in reference:
            for edge in (t.start, t.end):
                scored[max(0, int((edge - collar_s) / step_s)) : int((edge + collar_s) / step_s) + 1] = False
    ref_speech = (ref >= 0) & scored
    total = int(ref_speech.sum())
    if total == 0:
        return {"der": 0.0, "missed": 0.0, "false_alarm": 0.0, "confusion": 0.0, "reference_speech_s": 0.0}
    missed = int((ref_speech & (hyp < 0)).sum())
    false_alarm = int(((ref < 0) & (hyp >= 0) & scored).sum())
    ref_ids, hyp_ids = sorted(set(ref[ref >= 0])), sorted(set(hyp[hyp >= 0]))
    both = ref_speech & (hyp >= 0)
    best_correct = 0
    k = min(len(ref_ids), len(hyp_ids))
    if k and len(ref_ids) <= 8 and len(hyp_ids) <= 8:
        overlap = {(r, h): int((both & (ref == r) & (hyp == h)).sum()) for r in ref_ids for h in hyp_ids}
        for chosen in itertools.permutations(hyp_ids, k) if len(hyp_ids) >= len(ref_ids) else [None]:
            if chosen is None:
                for chosen_ref in itertools.permutations(ref_ids, k):
                    best_correct = max(best_correct, sum(overlap[(r, h)] for r, h in zip(chosen_ref, hyp_ids)))
            else:
                best_correct = max(best_correct, sum(overlap[(r, h)] for r, h in zip(ref_ids, chosen)))
    confusion = int(both.sum()) - best_correct
    return {
        "der": round((missed + false_alarm + confusion) / total, 4),
        "missed": round(missed / total, 4),
        "false_alarm": round(false_alarm / total, 4),
        "confusion": round(confusion / total, 4),
        "reference_speech_s": round(total * step_s, 2),
    }
