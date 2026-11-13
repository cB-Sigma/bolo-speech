"""Voice-activity detection helpers.

``EnergyVad`` is a dependency-free detector (adaptive energy threshold with
hang-over). It is the fallback when no neural VAD model is installed and
the detector the fake engine uses, so the rest of the realtime pipeline can
be exercised without downloads. Neural detectors (Silero, TEN) come from
engine plugins and implement the same :class:`~bolo.engines.base.VadDetector`
interface.
"""

from __future__ import annotations

import math

import numpy as np

from bolo.engines.base import FloatArray, VadDetector, VadEvent, VadParams

FRAME = 320  # 20 ms at 16 kHz


class EnergyVad(VadDetector):
    def __init__(self, params: VadParams | None = None, sample_rate: int = 16000) -> None:
        self.params = params or VadParams()
        self.sample_rate = sample_rate
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos = 0  # samples consumed in whole frames
        self._noise_db = -60.0
        self._speech = False
        self._candidate_start: int | None = None
        self._onset_gap = 0
        self._speech_start = 0
        self._last_voiced_end = 0
        # threshold 0.5 -> 12 dB above the noise floor; higher thresholds demand more
        self._margin_db = 6.0 + 12.0 * self.params.threshold
        self._floor_db = -50.0

    @property
    def in_speech(self) -> bool:
        return self._speech

    def _frame_db(self, frame: FloatArray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))
        return 20.0 * math.log10(rms + 1e-9)

    def accept(self, samples: FloatArray) -> list[VadEvent]:
        events: list[VadEvent] = []
        self._buf = np.concatenate([self._buf, samples]) if len(self._buf) else samples
        n_frames = len(self._buf) // FRAME
        p = self.params
        for i in range(n_frames):
            frame = self._buf[i * FRAME : (i + 1) * FRAME]
            start = self._pos
            end = start + FRAME
            self._pos = end
            db = self._frame_db(frame)
            voiced = db > max(self._noise_db + self._margin_db, self._floor_db)
            if not voiced:
                # track the noise floor on unvoiced frames only
                self._noise_db = min(max(0.95 * self._noise_db + 0.05 * db, -90.0), -20.0)
            if not self._speech:
                if voiced:
                    if self._candidate_start is None:
                        self._candidate_start = start
                    if end - self._candidate_start >= p.min_speech_s * self.sample_rate:
                        self._speech = True
                        self._speech_start = max(0, self._candidate_start - int(p.speech_pad_s * self.sample_rate))
                        self._last_voiced_end = end
                        events.append(VadEvent("speech_start", self._speech_start))
                elif self._candidate_start is not None:
                    # tolerate short dips (stops, syllable gaps) while speech is building up
                    self._onset_gap += FRAME
                    if self._onset_gap > 0.12 * self.sample_rate:
                        self._candidate_start = None
                if voiced:
                    self._onset_gap = 0
            else:
                if voiced:
                    self._last_voiced_end = end
                silence = end - self._last_voiced_end
                too_long = end - self._speech_start >= p.max_speech_s * self.sample_rate
                if silence >= p.min_silence_s * self.sample_rate or too_long:
                    self._speech = False
                    self._candidate_start = None
                    cut = (
                        end
                        if too_long and silence == 0
                        else self._last_voiced_end + int(p.speech_pad_s * self.sample_rate)
                    )
                    events.append(VadEvent("speech_end", min(cut, end)))
        self._buf = self._buf[n_frames * FRAME :]
        return events

    def flush(self) -> list[VadEvent]:
        if self._speech:
            self._speech = False
            return [VadEvent("speech_end", self._pos + len(self._buf))]
        return []


class FallbackVad(VadDetector):
    """A neural detector with an energy detector as a safety net.

    Neural VADs are trained on wide-band audio and can stay silent on input
    they have not seen (very narrow-band or very quiet lines). If the energy
    detector hears ``patience_s`` of continuous speech while the primary
    detector has reported nothing, the utterance is taken from the energy
    detector instead, so a stream never ends with speech and no transcript.
    """

    def __init__(self, primary: VadDetector, params: VadParams | None = None, patience_s: float = 1.5) -> None:
        self.primary = primary
        self.backup = EnergyVad(params or VadParams())
        self.patience = int(patience_s * 16000)
        self._pos = 0
        self._backup_start: int | None = None  # the energy detector is in speech since this sample
        self._primary_seen = False  # the primary reported something during that span
        self._using_backup = False
        self._primary_open = False
        self.fallbacks = 0

    @property
    def in_speech(self) -> bool:
        return self._using_backup or self.primary.in_speech

    def accept(self, samples: FloatArray) -> list[VadEvent]:
        out: list[VadEvent] = []
        primary_events = self.primary.accept(samples)
        backup_events = self.backup.accept(samples)
        self._pos += len(samples)
        for ev in primary_events:
            self._primary_open = ev.kind == "speech_start"
        if primary_events or self._primary_open or self.primary.in_speech:
            self._primary_seen = True
        if not self._using_backup:
            out.extend(primary_events)
        for ev in backup_events:
            if ev.kind == "speech_start":
                self._backup_start = ev.sample
                self._primary_seen = bool(primary_events) or self._primary_open or self.primary.in_speech
            elif ev.kind == "speech_end":
                if self._using_backup:
                    out.append(ev)
                    self._using_backup = False
                elif (
                    self._backup_start is not None
                    and not self._primary_seen
                    and ev.sample - self._backup_start >= self.patience
                ):
                    # a whole utterance went by unnoticed by the primary detector
                    self.fallbacks += 1
                    out.extend([VadEvent("speech_start", self._backup_start), ev])
                self._backup_start = None
        if (
            not self._using_backup
            and self._backup_start is not None
            and not self._primary_seen
            and self._pos - self._backup_start >= self.patience
        ):
            self._using_backup = True
            self.fallbacks += 1
            out.append(VadEvent("speech_start", self._backup_start))
        return out

    def flush(self) -> list[VadEvent]:
        primary_events = self.primary.flush()
        backup_events = self.backup.flush()
        if self._using_backup:
            self._using_backup = False
            return backup_events or [VadEvent("speech_end", self._pos)]
        return primary_events


def detect_segments(
    detector: VadDetector,
    samples: FloatArray,
    *,
    chunk: int = 16000,
) -> list[tuple[int, int]]:
    """Run a detector over a whole recording; returns ``(start, end)`` sample ranges."""
    segments: list[tuple[int, int]] = []
    start: int | None = None
    events: list[VadEvent] = []
    for off in range(0, len(samples), chunk):
        events.extend(detector.accept(samples[off : off + chunk]))
    events.extend(detector.flush())
    for ev in events:
        if ev.kind == "speech_start":
            start = ev.sample
        elif ev.kind == "speech_end" and start is not None:
            end = min(ev.sample, len(samples))
            if end > start:
                segments.append((start, end))
            start = None
    return segments


def merge_segments(
    segments: list[tuple[int, int]],
    total: int,
    *,
    sample_rate: int = 16000,
    max_chunk_s: float = 30.0,
    target_s: float | None = None,
    pad_s: float = 0.15,
) -> list[tuple[int, int]]:
    """Group speech segments into decode chunks no longer than ``max_chunk_s``.

    Neighbouring segments are packed together up to ``target_s`` so short
    phrases share context, and over-long segments are split.
    """
    if not segments:
        return []
    max_len = int(max_chunk_s * sample_rate)
    target = int((target_s if target_s is not None else max_chunk_s) * sample_rate)
    pad = int(pad_s * sample_rate)
    chunks: list[tuple[int, int]] = []
    cur_start, cur_end = segments[0]
    for start, end in segments[1:]:
        if end - cur_start <= target:
            cur_end = end
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    chunks.append((cur_start, cur_end))
    out: list[tuple[int, int]] = []
    for start, end in chunks:
        start = max(0, start - pad)
        end = min(total, end + pad)
        while end - start > max_len:
            out.append((start, start + max_len))
            start += max_len
        if end > start:
            out.append((start, end))
    return out
