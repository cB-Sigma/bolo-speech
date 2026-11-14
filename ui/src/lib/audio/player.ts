export interface PlayerStats {
  /** Seconds of audio handed to the player. */
  queuedSeconds: number
  underruns: number
  /** Total time playback had nothing to play while the stream was still open (ms). */
  starvedMs: number
  /** `performance.now()` at which the first sample became audible. */
  startedAt: number | null
}

export interface ScheduledChunk {
  /** Offset of this chunk on the audio timeline (seconds). */
  t0: number
  dur: number
  /** ms the playhead had already waited when this chunk arrived (0 if on time). */
  lateMs: number
}

/**
 * Gapless streaming playback: each chunk is scheduled on the Web Audio clock
 * exactly where the previous one ends. If a chunk arrives after the playhead
 * has already run past the end of the queue, that is an underrun: it is
 * counted and the chunk is scheduled a few milliseconds ahead.
 */
export class GaplessPlayer {
  readonly ctx: AudioContext
  readonly analyser: AnalyserNode
  private nextTime = 0
  private startCtxTime: number | null = null
  private sources = new Set<AudioBufferSourceNode>()
  private streamOpen = true
  private stats: PlayerStats = { queuedSeconds: 0, underruns: 0, starvedMs: 0, startedAt: null }
  onEnded: (() => void) | null = null

  constructor(ctx?: AudioContext) {
    this.ctx = ctx ?? new AudioContext({ latencyHint: 'interactive' })
    this.analyser = this.ctx.createAnalyser()
    this.analyser.fftSize = 1024
    this.analyser.smoothingTimeConstant = 0.6
    this.analyser.connect(this.ctx.destination)
    // A player is made in the tap that asks for audio, and audio often arrives much later (a voice has to
    // load). Start the clock now, while the tap still counts as a user gesture, so that the first buffer is
    // scheduled on a context that is already running and its opening is not lost to the start-up.
    this.wake()
  }

  /** Ask a context that is not running (suspended, or interrupted on iOS) to run. Never throws. */
  private wake(): void {
    const ctx = this.ctx
    if (ctx.state === 'running' || ctx.state === 'closed') return
    try {
      void ctx.resume().catch(() => undefined)
    } catch {
      // not allowed yet: the next enqueue asks again
    }
  }

  /** Seconds of headroom kept ahead of the playhead when (re)starting. */
  static readonly LEAD = 0.03

  enqueue(samples: Float32Array, sampleRate: number): ScheduledChunk {
    const ctx = this.ctx
    this.wake()
    const buffer = ctx.createBuffer(1, Math.max(1, samples.length), sampleRate)
    buffer.getChannelData(0).set(samples)
    const src = ctx.createBufferSource()
    src.buffer = buffer
    src.connect(this.analyser)
    const now = ctx.currentTime
    let lateMs = 0
    if (this.startCtxTime === null) {
      this.nextTime = now + GaplessPlayer.LEAD
      this.startCtxTime = this.nextTime
      this.stats.startedAt = performance.now() + GaplessPlayer.LEAD * 1000
    } else if (this.nextTime < now) {
      lateMs = (now - this.nextTime) * 1000
      this.stats.underruns += 1
      this.stats.starvedMs += lateMs
      this.nextTime = now + GaplessPlayer.LEAD
    }
    const t0 = this.stats.queuedSeconds
    src.start(this.nextTime)
    this.nextTime += buffer.duration
    this.stats.queuedSeconds += buffer.duration
    this.sources.add(src)
    src.onended = () => {
      this.sources.delete(src)
      if (!this.streamOpen && this.sources.size === 0) this.onEnded?.()
    }
    return { t0, dur: buffer.duration, lateMs }
  }

  /** Mark the stream complete; `onEnded` fires when the queue drains. */
  end(): void {
    this.streamOpen = false
    if (this.sources.size === 0) this.onEnded?.()
  }

  /** Seconds of audio played so far (ignoring time lost to underruns). */
  get position(): number {
    if (this.startCtxTime === null) return 0
    const played = this.ctx.currentTime - this.startCtxTime - this.stats.starvedMs / 1000
    return Math.max(0, Math.min(this.stats.queuedSeconds, played))
  }

  /** Seconds of audio buffered ahead of the playhead. */
  get bufferedAhead(): number {
    return Math.max(0, this.nextTime - this.ctx.currentTime)
  }

  get snapshot(): PlayerStats {
    return { ...this.stats }
  }

  stop(): void {
    this.streamOpen = false
    for (const s of this.sources) {
      s.onended = null
      try {
        s.stop()
      } catch {
        // already stopped
      }
    }
    this.sources.clear()
  }

  async close(): Promise<void> {
    this.stop()
    if (this.ctx.state !== 'closed') await this.ctx.close()
  }
}
