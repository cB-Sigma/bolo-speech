import { AudioClock, newUtterance, type Utterance } from '@/lib/latency/timeline'
import { Store } from '@/lib/store'
import type { LiveDenoise, LiveDenoiseMetrics, SttWord } from '@/lib/api/types'

export type TranscriberStatus =
  'idle' | 'loading' | 'connecting' | 'streaming' | 'reconnecting' | 'finishing' | 'done' | 'error'

export interface FinalSegment {
  text: string
  t0: number
  t1: number
  words: SttWord[] | null
  language: string | null
  latencyMs: number | null
}

export interface SpeechSpan {
  t0: number
  t1: number | null
}

export interface TranscriptState {
  status: TranscriberStatus
  error: string | null
  finals: FinalSegment[]
  /** The newest interim; `t0`: where its segment starts (audio seconds, the server's `t0`). */
  partial: { text: string; stableChars: number; t0?: number } | null
  utterances: Utterance[]
  speech: SpeechSpan[]
  /** Seconds of audio pushed so far. */
  audioSeconds: number
  rtf: number | null
  queueMs: number | null
  requestId: string | null
  /** Where inference runs, for display. */
  backend: string | null
  /** Median websocket ping round trip (server mode only). */
  rttMs: number | null
  /** Words the repetition guard removed from finals (in-browser recogniser). */
  repeatsRemoved?: number
  /** Server stream: what denoises it (`ready.denoise`; null: nothing), and the time it takes (`metrics.denoise`). */
  denoise?: LiveDenoise | null
  denoiseMetrics?: LiveDenoiseMetrics | null
}

export const emptyTranscript = (): TranscriptState => ({
  status: 'idle',
  error: null,
  finals: [],
  partial: null,
  utterances: [],
  speech: [],
  audioSeconds: 0,
  rtf: null,
  queueMs: null,
  requestId: null,
  backend: null,
  rttMs: null,
})

/** Seconds of tolerance when matching a segment's start to an utterance's (detector and recogniser disagree). */
const START_SLACK_S = 0.05

/** Something that turns a stream of 16 kHz mono blocks into a transcript. */
export interface Transcriber {
  readonly store: Store<TranscriptState>
  start(): Promise<void>
  push(block: Float32Array): void
  /** No more audio: flush, wait for the last final, then settle in `done`. */
  finish(): Promise<void>
  /** Abort immediately. */
  cancel(): void
}

/**
 * Shared bookkeeping that turns VAD / partial / final events into the
 * transcript state and the per-utterance latency timeline.
 */
export class TranscriptTracker {
  readonly store = new Store<TranscriptState>(emptyTranscript())
  readonly clock = new AudioClock()
  /**
   * Utterances that have started but not received their final, oldest first.
   * More than one can be open: after a short pause the next utterance often
   * starts before the previous final arrives.
   */
  private pending: Utterance[] = []
  private closed: Utterance[] = []
  private nextId = 1
  /** Where each open utterance starts in the audio (seconds, before `timeOffset`), by utterance id. */
  private startT = new Map<number, number>()
  /** Added to audio times, so a reconnect continues the same time axis. */
  timeOffset = 0

  private now(): number {
    return performance.now()
  }

  markSent(seconds: number): void {
    this.clock.mark(seconds, this.now())
    this.store.patch({ audioSeconds: this.clock.sentSeconds })
  }

  private wall(t: number): number {
    return this.clock.wallTimeOf(t + this.timeOffset) ?? this.now()
  }

  private add(): Utterance {
    const u = newUtterance(this.nextId++)
    this.pending.push(u)
    return u
  }

  /** The utterance currently being spoken (the newest one still open). */
  private latest(): Utterance {
    return this.pending[this.pending.length - 1] ?? this.add()
  }

  private publish(extra: Partial<TranscriptState> = {}): void {
    this.store.patch({ ...extra, utterances: [...this.closed, ...this.pending.map((u) => ({ ...u }))] })
  }

  speechStart(t: number): void {
    const last = this.pending[this.pending.length - 1]
    // Reuse an utterance opened by an early partial; otherwise this is a new one.
    const u = last && last.speechEndAt == null ? last : this.add()
    if (!this.startT.has(u.id)) this.startT.set(u.id, t)
    // The detector's timestamp beats one inferred from an early partial.
    u.speechStartAt = this.wall(t)
    if (u.firstPartialAt != null && u.firstPartialAt < u.speechStartAt) u.speechStartAt = u.firstPartialAt
    const s = this.store.get()
    this.publish({ speech: [...s.speech, { t0: t + this.timeOffset, t1: null }] })
  }

  speechEnd(t: number): void {
    const u = this.pending.find((x) => x.speechEndAt == null) ?? this.add()
    u.speechEndAt = this.wall(t)
    const speech = this.store.get().speech.slice()
    const i = speech.findIndex((sp) => sp.t1 == null)
    if (i >= 0) speech[i] = { ...(speech[i] as SpeechSpan), t1: t + this.timeOffset }
    this.publish({ speech })
  }

  /** The open utterance a segment starting at `t0` belongs to: the newest that started by then (a late interim of an
   * earlier segment must not be taken for the one being spoken now). */
  private owner(t0: number): Utterance {
    for (let i = this.pending.length - 1; i >= 0; i--) {
      const u = this.pending[i] as Utterance
      const start = this.startT.get(u.id)
      if (start == null || start <= t0 + START_SLACK_S) return u
    }
    return this.pending[0] ?? this.add()
  }

  partial(text: string, t0: number, serverMs: number | null, stableChars?: number): void {
    const u = this.pending.length ? this.owner(t0) : this.latest()
    if (!this.startT.has(u.id)) this.startT.set(u.id, t0)
    if (u.speechStartAt == null) u.speechStartAt = this.wall(t0)
    if (u.firstPartialAt == null && text.trim()) {
      u.firstPartialAt = this.now()
      u.firstPartialServerMs = serverMs
    }
    u.text = text
    const shown = this.store.get().partial
    // an interim of an older segment never replaces the one of the segment being spoken now
    if (shown?.t0 != null && t0 < shown.t0 - START_SLACK_S) {
      this.publish()
      return
    }
    this.publish({ partial: { text, stableChars: Math.max(0, Math.min(text.length, stableChars ?? 0)), t0 } })
  }

  final(seg: FinalSegment): void {
    // Finals arrive in order, so this one belongs to the oldest open utterance.
    const u = this.pending.shift() ?? newUtterance(this.nextId++)
    this.startT.delete(u.id)
    const at = this.now()
    if (u.speechStartAt == null) u.speechStartAt = this.wall(seg.t0)
    if (u.speechEndAt == null) u.speechEndAt = Math.min(at, this.wall(seg.t1))
    if (u.firstPartialAt == null) {
      u.firstPartialAt = at
      u.firstPartialServerMs = seg.latencyMs
    }
    u.finalAt = at
    u.finalServerMs = seg.latencyMs
    u.t0 = seg.t0 + this.timeOffset
    u.t1 = seg.t1 + this.timeOffset
    u.text = seg.text
    this.closed.push(u)
    const s = this.store.get()
    const t1 = seg.t1 + this.timeOffset
    const finals = seg.text.trim() ? [...s.finals, { ...seg, t0: seg.t0 + this.timeOffset, t1 }] : s.finals
    // Without VAD events the final is the only thing that closes a span; never close one that began later.
    const speech = s.speech.map((sp) => (sp.t1 == null && sp.t0 <= t1 ? { ...sp, t1 } : sp))
    // The visible interim stays only if it belongs to a later segment still open: a final that arrives late (after
    // the next segment started) replaces its own interim instead of being shown next to it.
    const later = s.partial?.t0 == null || s.partial.t0 >= seg.t1 - START_SLACK_S
    const partial = later && this.pending.length > 0 && this.pending[this.pending.length - 1]?.text ? s.partial : null
    this.publish({ finals, partial, speech })
  }

  /** Drop utterances that never got a final (the stream ended). */
  abandon(): void {
    if (this.pending.length === 0) return
    this.pending = []
    this.startT.clear()
    const end = this.clock.sentSeconds
    const speech = this.store.get().speech.map((sp) => (sp.t1 == null ? { ...sp, t1: end } : sp))
    this.publish({ partial: null, speech })
  }
}

export function transcriptText(s: TranscriptState): string {
  return s.finals
    .map((f) => f.text.trim())
    .filter(Boolean)
    .join(' ')
}
