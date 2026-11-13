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
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from bolo.engines.base import FloatArray, SpeakerTurn


def speaker_label(index: int) -> str:
    """0 -> "A", 1 -> "B", ... as in OpenAI's diarized transcripts; "S27" beyond "Z"."""
    return chr(ord("A") + index) if 0 <= index < 26 else f"S{index + 1}"


@dataclass
class DiarizeOptions:
    model: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    threshold: float | None = None


def renumber(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Number speakers in order of first appearance."""
    order: dict[int, int] = {}
    return [
        SpeakerTurn(t.start, t.end, order.setdefault(t.speaker, len(order)))
        for t in sorted(turns, key=lambda t: t.start)
    ]


def merge_turns(
    turns: list[SpeakerTurn], *, max_gap_s: float = 1.0, max_len_s: float = 30.0, min_len_s: float = 0.2
) -> list[SpeakerTurn]:
    """Join consecutive turns of the same speaker into chunks worth transcribing."""
    out: list[SpeakerTurn] = []
    for t in sorted(turns, key=lambda t: t.start):
        if t.end - t.start < min_len_s and not (out and out[-1].speaker == t.speaker):
            continue
        last = out[-1] if out else None
        if last and last.speaker == t.speaker and t.start - last.end <= max_gap_s and t.end - last.start <= max_len_s:
            last.end = max(last.end, t.end)
        else:
            out.append(
                SpeakerTurn(max(t.start, last.end) if last and t.start < last.end else t.start, t.end, t.speaker)
            )
    return [t for t in out if t.end - t.start >= min_len_s]


class OnlineSpeakerLabeler:
    """Incremental speaker assignment from per-utterance embeddings (cosine similarity to centroids)."""

    def __init__(self, threshold: float = 0.45, max_speakers: int | None = None) -> None:
        self.threshold = threshold
        self.max_speakers = max_speakers
        self._sums: list[FloatArray] = []
        self.last: int | None = None

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
        return self.last


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
