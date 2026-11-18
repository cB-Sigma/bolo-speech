"""Safeguards that end a live conversation turn the voice-activity detector keeps open.

A steady noise (a fan, an air conditioner, a car) can look like speech to a detector: it reports the start of
speech and never the end, so the user's turn is never committed and the agent never answers. :class:`TurnGuard`
watches the level of the input during a turn and ends it anyway:

* **noise plateau**: speech goes up and down many times a second (syllables, pauses); a fan does not. When the
  level of the last ``plateau_s`` seconds stays within ``plateau_db`` (the standard deviation of 20 ms frames, in
  dB), the turn ends where that steady stretch began.
* **maximum turn length**: after ``max_turn_s`` seconds the turn is committed whatever the detector says.

After such a forced end the detector is usually still "in speech" (the noise goes on) and would not report the next
utterance. Until it reports the end of that speech itself, the guard starts the next turn when the level rises
``rise_db`` above the level the forced end was taken at, for at least ``rise_s`` with dips of up to 150 ms between
syllables (an energy-relative onset).

The levels are plain RMS per 20 ms frame: cheap, and independent of which detector runs.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from bolo.engines.base import FloatArray

FRAME = 320  # 20 ms at 16 kHz


class TurnGuard:
    def __init__(
        self,
        max_turn_s: float = 30.0,
        plateau_s: float = 2.0,
        *,
        plateau_db: float = 1.5,
        rise_db: float = 6.0,
        rise_s: float = 0.25,
        sample_rate: int = 16000,
    ) -> None:
        self.max_turn = int(max(0.0, max_turn_s) * sample_rate)
        self.plateau = int(max(0.0, plateau_s) * sample_rate)
        self.plateau_db = plateau_db
        self.rise_db = rise_db
        self.rise = int(rise_s * sample_rate)
        self.sample_rate = sample_rate
        self._rest = np.zeros(0, dtype=np.float32)
        self._pos = 0  # samples in whole frames so far
        keep = max(self.plateau, 2 * sample_rate) // FRAME + 1
        self._frames: deque[tuple[int, float]] = deque(maxlen=keep)  # (end sample, level in dB)
        #: after a forced end while the detector stays in speech: the level the next turn must rise above
        self.noise_db: float | None = None
        self._rise_from: int | None = None
        self._rise_last = 0
        self.gap = int(0.15 * sample_rate)

    @property
    def enabled(self) -> bool:
        return bool(self.max_turn or self.plateau)

    @property
    def armed(self) -> bool:
        """A forced end happened and the detector has not reported the end of that speech yet."""
        return self.noise_db is not None

    def feed(self, samples: FloatArray) -> int | None:
        """Take input audio (16 kHz). While :attr:`armed`, returns the sample where a new utterance starts above the
        noise the last forced end was taken at, else None."""
        buf = np.concatenate([self._rest, samples]) if len(self._rest) else np.asarray(samples, dtype=np.float32)
        n = len(buf) // FRAME
        onset: int | None = None
        for i in range(n):
            frame = buf[i * FRAME : (i + 1) * FRAME]
            start = self._pos
            self._pos += FRAME
            rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))
            db = 20.0 * math.log10(rms + 1e-9)
            self._frames.append((self._pos, db))
            if self.noise_db is not None and onset is None:
                if db >= self.noise_db + self.rise_db:
                    if self._rise_from is None:
                        self._rise_from = start
                    self._rise_last = self._pos
                    if self._pos - self._rise_from >= self.rise:
                        onset = self._rise_from
                elif self._rise_from is not None and self._pos - self._rise_last > self.gap:
                    self._rise_from = None  # a dip between syllables is allowed, a longer one starts over
        self._rest = buf[n * FRAME :].copy()
        if onset is not None:
            self.noise_db, self._rise_from = None, None
        return onset

    def check(self, turn_start: int, pos: int, plateau: bool = True) -> tuple[str, int] | None:
        """Whether the turn that started at ``turn_start`` must end now: ``(reason, end sample)``, else None.
        ``reason`` is ``"noise_plateau"`` or ``"max_turn_seconds"``. ``plateau``: whether to look for a steady level at
        all (only while the detector says "speech": a pause it already heard as silence is the detector's to judge,
        and digital silence is as flat as a fan)."""
        age = pos - turn_start
        if plateau and self.plateau and age >= self.plateau + self.sample_rate // 2:
            window = [db for end, db in self._frames if end > pos - self.plateau and end - FRAME >= turn_start]
            if len(window) * FRAME >= self.plateau - FRAME and float(np.std(window)) < self.plateau_db:
                begin = max(turn_start, pos - self.plateau)
                self.noise_db = float(np.median(window))
                return "noise_plateau", begin
        if self.max_turn and age >= self.max_turn:
            levels = [db for end, db in self._frames if end > turn_start]
            # the next turn must rise above the quieter part of this one (the noise between words)
            self.noise_db = float(np.percentile(levels, 20)) if levels else None
            return "max_turn_seconds", pos
        return None

    def detector_ended(self) -> None:
        """The detector reported the end of the speech it was in: it hears turns by itself again."""
        self.noise_db, self._rise_from = None, None
