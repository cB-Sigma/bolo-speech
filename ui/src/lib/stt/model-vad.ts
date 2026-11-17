import { STT_RATE } from '@/lib/audio/capture'
import type { AudioModelEngine } from '@/lib/browser-audio/engine'
import type { VadEvent } from './vad'

/** What the recogniser needs from a voice-activity detector: the energy detector, or a model's. */
export interface FrameVad {
  readonly isSpeaking: boolean
  push(block: Float32Array): VadEvent[] | Promise<VadEvent[]>
  flush(): VadEvent[]
}

export interface ModelVadOptions {
  /** Speech at or above this probability (Silero's default 0.5). */
  threshold?: number
  /** Speech must persist this long to open a segment. */
  startMs?: number
  /** Silence (below threshold − 0.15) must persist this long to close it. */
  endMs?: number
}

/**
 * A model's speech probabilities turned into speech start and end, as the server's segmenter does for detectors
 * that score frames: a threshold with hysteresis (speech at `threshold`, silence below `threshold − 0.15`), a
 * minimum speech run to open a segment and a minimum silence to close it.
 */
export class ModelVad implements FrameVad {
  private readonly engine: AudioModelEngine
  private readonly threshold: number
  private readonly negThreshold: number
  private readonly startMs: number
  private readonly endMs: number
  private samples = 0
  private speaking = false
  private run = 0
  private silence = 0
  private lastSpeechT = 0
  private first = true
  /** Probabilities seen, for the page's readout (the latest few seconds). */
  readonly recent: number[] = []

  constructor(engine: AudioModelEngine, opts: ModelVadOptions = {}) {
    this.engine = engine
    this.threshold = opts.threshold ?? 0.5
    this.negThreshold = Math.max(0.01, this.threshold - 0.15)
    this.startMs = opts.startMs ?? 100
    this.endMs = opts.endMs ?? 500
  }

  get isSpeaking(): boolean {
    return this.speaking
  }

  async push(block: Float32Array): Promise<VadEvent[]> {
    const { probs, frame } = await this.engine.vad(block, this.first)
    this.first = false
    const events: VadEvent[] = []
    const frameMs = (frame / STT_RATE) * 1000
    const startFrames = Math.max(1, Math.round(this.startMs / frameMs))
    const endFrames = Math.max(1, Math.round(this.endMs / frameMs))
    for (const p of probs) {
      const t = this.samples / STT_RATE
      this.samples += frame
      this.recent.push(p)
      if (this.recent.length > 200) this.recent.shift()
      if (this.speaking) {
        if (p >= this.negThreshold) {
          this.silence = 0
          if (p >= this.threshold) this.lastSpeechT = t + frame / STT_RATE
        } else if (++this.silence >= endFrames) {
          this.speaking = false
          this.run = 0
          events.push({ event: 'speech_end', t: this.lastSpeechT })
        }
      } else if (p >= this.threshold) {
        if (++this.run >= startFrames) {
          this.speaking = true
          this.silence = 0
          this.lastSpeechT = t + frame / STT_RATE
          events.push({ event: 'speech_start', t: Math.max(0, t - ((startFrames - 1) * frame) / STT_RATE) })
        }
      } else {
        this.run = 0
      }
    }
    return events
  }

  flush(): VadEvent[] {
    if (!this.speaking) return []
    this.speaking = false
    return [{ event: 'speech_end', t: this.lastSpeechT }]
  }
}

/** The end-of-turn check: at each pause, the model hears the utterance and says whether it is finished. */
export interface TurnDetector {
  engine: AudioModelEngine
  /** Finished at or above this probability (0.5, as the server). */
  threshold?: number
  /** A pause longer than this ends the turn whatever the model says (ms). */
  maxWaitMs?: number
}

/** One check of the end-of-turn model, for the page. */
export interface TurnCheck {
  probability: number
  ms: number
  /** `end`: the turn ended; `wait`: the speaker was heard as not finished; `timeout`: the pause ran too long. */
  decision: 'end' | 'wait' | 'timeout'
  at: number
}
