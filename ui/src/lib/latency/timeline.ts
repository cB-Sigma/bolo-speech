/**
 * Per-utterance latency timeline.
 *
 * All instants are milliseconds on one monotonic clock (performance.now() in
 * the browser). An utterance is the span between a `speech_start` and the
 * `final` that follows its `speech_end`.
 */

export interface Utterance {
  id: number
  /** Wall time at which the audio containing speech start / end left the client. */
  speechStartAt: number | null
  speechEndAt: number | null
  firstPartialAt: number | null
  finalAt: number | null
  /** Server-reported processing latency on the first partial / the final. */
  firstPartialServerMs: number | null
  finalServerMs: number | null
  /** Audio-time bounds reported with the final (seconds). */
  t0: number | null
  t1: number | null
  text: string
}

export function newUtterance(id: number): Utterance {
  return {
    id,
    speechStartAt: null,
    speechEndAt: null,
    firstPartialAt: null,
    finalAt: null,
    firstPartialServerMs: null,
    finalServerMs: null,
    t0: null,
    t1: null,
    text: '',
  }
}

export interface UtteranceLatency {
  /** speech start → first partial */
  firstPartialMs: number | null
  /** speech end → final: the number a turn-taking system waits on */
  finalMs: number | null
  /** speech end → first partial; negative when the partial preceded the end of speech */
  endToFirstPartialMs: number | null
  speechMs: number | null
}

const diff = (a: number | null, b: number | null): number | null => (a == null || b == null ? null : a - b)

export function utteranceLatency(u: Utterance): UtteranceLatency {
  const finalMs = diff(u.finalAt, u.speechEndAt)
  return {
    firstPartialMs: nonNeg(diff(u.firstPartialAt, u.speechStartAt)),
    finalMs: nonNeg(finalMs),
    endToFirstPartialMs: diff(u.firstPartialAt, u.speechEndAt),
    speechMs: nonNeg(diff(u.speechEndAt, u.speechStartAt)),
  }
}

function nonNeg(v: number | null): number | null {
  return v == null ? null : Math.max(0, v)
}

export type MarkerKind = 'speech_start' | 'first_partial' | 'speech_end' | 'final'

export interface Marker {
  kind: MarkerKind
  /** ms from the timeline origin */
  at: number
  /** 0..1 position along the axis */
  pos: number
}

export interface Gap {
  from: MarkerKind
  to: MarkerKind
  ms: number
  pos0: number
  pos1: number
}

export interface TimelineLayout {
  originAt: number
  spanMs: number
  markers: Marker[]
  gaps: Gap[]
  /** The speech span as 0..1 positions, if both ends are known. */
  speech: { pos0: number; pos1: number } | null
}

/**
 * Lay the utterance out on a shared axis. The tail after speech end is where
 * the interesting gaps are, so when speech is long it is compressed to at
 * most `maxSpeechShare` of the axis; positions inside the tail stay linear.
 */
export function layoutTimeline(u: Utterance, maxSpeechShare = 0.45, minTailMs = 100): TimelineLayout | null {
  const raw: { kind: MarkerKind; at: number }[] = []
  if (u.speechStartAt != null) raw.push({ kind: 'speech_start', at: u.speechStartAt })
  if (u.firstPartialAt != null) raw.push({ kind: 'first_partial', at: u.firstPartialAt })
  if (u.speechEndAt != null) raw.push({ kind: 'speech_end', at: u.speechEndAt })
  if (u.finalAt != null) raw.push({ kind: 'final', at: u.finalAt })
  if (raw.length < 2) return null
  raw.sort((a, b) => a.at - b.at)
  const origin = (raw[0] as { at: number }).at
  const end = (raw[raw.length - 1] as { at: number }).at
  const spanMs = Math.max(1, end - origin)

  // Piecewise-linear scale with a knee at speech end.
  const knee = u.speechEndAt != null ? Math.max(0, u.speechEndAt - origin) : 0
  const tail = Math.max(minTailMs, spanMs - knee)
  const speechShare = knee > 0 ? Math.min(maxSpeechShare, knee / (knee + tail)) : 0
  const scale = (at: number): number => {
    const d = at - origin
    if (knee > 0 && d <= knee) return (d / knee) * speechShare
    return Math.min(1, speechShare + ((d - knee) / tail) * (1 - speechShare))
  }

  const markers: Marker[] = raw.map((m) => ({ kind: m.kind, at: m.at - origin, pos: scale(m.at) }))
  const gaps: Gap[] = []
  for (let i = 1; i < markers.length; i++) {
    const a = markers[i - 1] as Marker
    const b = markers[i] as Marker
    gaps.push({ from: a.kind, to: b.kind, ms: b.at - a.at, pos0: a.pos, pos1: b.pos })
  }
  const speech =
    u.speechStartAt != null && u.speechEndAt != null
      ? { pos0: scale(u.speechStartAt), pos1: scale(u.speechEndAt) }
      : null
  return { originAt: origin, spanMs, markers, gaps, speech }
}

/**
 * Maps audio time (seconds since the stream began) to the wall-clock instant
 * that audio left the client. Frames are recorded as they are sent; lookups
 * interpolate inside a frame and extrapolate at real time past the last one.
 */
export class AudioClock {
  private marks: { audioEnd: number; sentAt: number }[] = []
  private audioSeconds = 0

  /** Record that `seconds` more audio was sent at wall time `sentAt` (ms). */
  mark(seconds: number, sentAt: number): void {
    this.audioSeconds += seconds
    this.marks.push({ audioEnd: this.audioSeconds, sentAt })
    if (this.marks.length > 4096) this.marks.splice(0, 1024)
  }

  get sentSeconds(): number {
    return this.audioSeconds
  }

  /** Wall time (ms) at which the audio at `t` seconds was sent. */
  wallTimeOf(t: number): number | null {
    const m = this.marks
    if (m.length === 0) return null
    const last = m[m.length - 1] as { audioEnd: number; sentAt: number }
    if (t >= last.audioEnd) return last.sentAt + (t - last.audioEnd) * 1000
    let lo = 0
    let hi = m.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if ((m[mid] as { audioEnd: number }).audioEnd < t) lo = mid + 1
      else hi = mid
    }
    const cur = m[lo] as { audioEnd: number; sentAt: number }
    // The sample at `t` sits inside frame `lo`, which was sent in one piece.
    return cur.sentAt
  }
}
