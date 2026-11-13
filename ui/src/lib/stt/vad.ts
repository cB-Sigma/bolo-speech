export interface VadEvent {
  event: 'speech_start' | 'speech_end'
  /** Audio time in seconds. */
  t: number
}

export interface EnergyVadOptions {
  sampleRate: number
  /** Analysis hop in ms. */
  frameMs?: number
  /** dB above the tracked noise floor that counts as speech. */
  marginDb?: number
  /** Absolute floor in dBFS below which nothing is speech. */
  minDb?: number
  /** Speech must persist this long to open a segment. */
  startMs?: number
  /** Silence must persist this long to close a segment (the endpointing delay). */
  endMs?: number
}

/**
 * Small energy-based voice activity detector with an adaptive noise floor.
 * It is deliberately simple: the in-browser mode uses it for endpointing so
 * that no second model has to be downloaded.
 */
export class EnergyVad {
  private readonly frame: number
  private readonly rate: number
  private readonly marginDb: number
  private readonly minDb: number
  private readonly startFrames: number
  private readonly endFrames: number
  private carry = new Float32Array(0)
  private frames = 0
  // Noise floor = a low percentile of frame levels over the last few seconds.
  private recent: Float32Array
  private recentFill = 0
  private run = 0
  private silence = 0
  private speaking = false
  private lastSpeechT = 0

  constructor(opts: EnergyVadOptions) {
    const frameMs = opts.frameMs ?? 20
    this.rate = opts.sampleRate
    this.frame = Math.round((opts.sampleRate * frameMs) / 1000)
    this.marginDb = opts.marginDb ?? 12
    this.minDb = opts.minDb ?? -48
    this.startFrames = Math.max(1, Math.round((opts.startMs ?? 100) / frameMs))
    this.endFrames = Math.max(1, Math.round((opts.endMs ?? 500) / frameMs))
    this.recent = new Float32Array(Math.round(4000 / frameMs))
  }

  get isSpeaking(): boolean {
    return this.speaking
  }

  /** Feed samples; returns any segment boundaries that were crossed. */
  push(block: Float32Array): VadEvent[] {
    const events: VadEvent[] = []
    let data = block
    if (this.carry.length) {
      data = new Float32Array(this.carry.length + block.length)
      data.set(this.carry)
      data.set(block, this.carry.length)
    }
    let o = 0
    for (; o + this.frame <= data.length; o += this.frame) {
      let sum = 0
      for (let i = o; i < o + this.frame; i++) sum += (data[i] as number) * (data[i] as number)
      const db = 10 * Math.log10(sum / this.frame + 1e-10)
      const t = (this.frames * this.frame) / this.rate
      this.frames += 1
      this.recent[(this.frames - 1) % this.recent.length] = db
      if (this.recentFill < this.recent.length) this.recentFill += 1
      // A low percentile rather than the minimum, so a few frames of digital silence do not sink the floor.
      const sorted = this.recent.slice(0, this.recentFill).sort()
      let floor = sorted[Math.floor(sorted.length * 0.12)] as number
      // Never let a loud opening (no silence seen yet) push the floor above typical room noise.
      floor = Math.max(-80, Math.min(-32, floor))
      const isSpeech = db > this.minDb && db > floor + this.marginDb
      if (this.speaking) {
        if (isSpeech) {
          this.silence = 0
          this.lastSpeechT = t + this.frame / this.rate
        } else if (++this.silence >= this.endFrames) {
          this.speaking = false
          this.run = 0
          events.push({ event: 'speech_end', t: this.lastSpeechT })
        }
      } else if (isSpeech) {
        if (++this.run >= this.startFrames) {
          this.speaking = true
          this.silence = 0
          this.lastSpeechT = t + this.frame / this.rate
          events.push({ event: 'speech_start', t: Math.max(0, t - ((this.startFrames - 1) * this.frame) / this.rate) })
        }
      } else {
        this.run = 0
      }
    }
    this.carry = data.slice(o)
    return events
  }

  /** Close an open segment at end of stream. */
  flush(): VadEvent[] {
    if (!this.speaking) return []
    this.speaking = false
    return [{ event: 'speech_end', t: this.lastSpeechT }]
  }
}
