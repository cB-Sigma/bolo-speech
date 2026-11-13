import { STT_RATE } from '@/lib/audio/capture'
import { concatFloat } from '@/lib/audio/pcm'
import type { BrowserSttEngine } from '@/lib/browser-stt/engine'
import { recordBrowserTrace } from '@/lib/browser-stt/traces'
import { TranscriptTracker, type Transcriber } from './transcriber'
import { EnergyVad } from './vad'

export interface BrowserTranscriberOptions {
  /** Minimum new audio between partial decodes (ms). */
  partialEveryMs?: number
  /** Silence that ends an utterance (ms). */
  endpointMs?: number
  /** Language to decode in, or `auto` to let a multilingual model guess per utterance. */
  language?: string
}

/** Whisper sees 30 s at a time; cut a non-stop utterance before that. */
const MAX_UTTERANCE_S = 24
/** Above this time per decode, partial results are skipped. */
const SLOW_DECODE_MS = 1500
/** A language guess from less audio than this is not trusted for the rest of the utterance. */
const DETECT_MIN_S = 1.5

function commonPrefix(a: string, b: string): number {
  const n = Math.min(a.length, b.length)
  let i = 0
  while (i < n && a[i] === b[i]) i++
  return i
}

/**
 * Runs recognition in the page: an energy VAD cuts utterances, the growing
 * utterance is re-decoded for partials while the worker is idle, and the
 * whole utterance is decoded once more for the final.
 */
export class BrowserTranscriber implements Transcriber {
  private readonly tracker = new TranscriptTracker()
  readonly store = this.tracker.store
  private readonly vad: EnergyVad
  private readonly engine: BrowserSttEngine
  private readonly partialEvery: number
  private preroll: Float32Array[] = []
  private utterance: Float32Array[] = []
  private utteranceSamples = 0
  private utteranceT0 = 0
  private lastPartialSamples = 0
  private lastPartialText = ''
  private generation = 0
  private decoding = 0
  private processedMs = 0
  private processedAudio = 0
  private cancelled = false
  private idle: (() => void) | null = null
  /** Compute time of the first partial of the current utterance. */
  private firstPartialMs: number | null = null
  private lastDecodeMs = 0
  private readonly language: string
  /** The model's guess for the utterance being spoken (auto mode). */
  private guessed: string | null = null

  constructor(engine: BrowserSttEngine, opts: BrowserTranscriberOptions = {}) {
    this.engine = engine
    this.partialEvery = ((opts.partialEveryMs ?? 600) / 1000) * STT_RATE
    this.vad = new EnergyVad({ sampleRate: STT_RATE, endMs: opts.endpointMs ?? 500 })
    this.language = opts.language ?? 'auto'
  }

  async start(): Promise<void> {
    this.store.patch({ status: 'loading', error: null, backend: 'browser' })
    try {
      await this.engine.load()
    } catch (err) {
      this.store.patch({ status: 'error', error: err instanceof Error ? err.message : String(err) })
      throw err
    }
    if (this.cancelled) return
    const e = this.engine.store.get()
    this.store.patch({ status: 'streaming', backend: `browser · ${e.device === 'webgpu' ? 'WebGPU' : 'WASM'}` })
  }

  push(block: Float32Array): void {
    const status = this.store.get().status
    if (status !== 'streaming') return
    const copy = block.slice()
    this.tracker.markSent(copy.length / STT_RATE)
    const speaking = this.vad.isSpeaking
    if (speaking) {
      this.utterance.push(copy)
      this.utteranceSamples += copy.length
    } else {
      this.preroll.push(copy)
      if (this.preroll.length > 15) this.preroll.shift() // ~300 ms
    }
    for (const ev of this.vad.push(copy)) {
      if (ev.event === 'speech_start') this.onStart(ev.t)
      else this.onEnd(ev.t)
    }
    if (this.vad.isSpeaking && this.utteranceSamples >= MAX_UTTERANCE_S * STT_RATE) {
      // Still talking: close this stretch and carry on in a new one.
      const t = this.tracker.clock.sentSeconds
      this.onEnd(t)
      this.preroll = []
      this.onStart(t)
    }
    if (
      this.vad.isSpeaking &&
      this.decoding === 0 &&
      // A model that needs this long per pass cannot keep up with partials; every pass then goes to finals.
      this.lastDecodeMs <= SLOW_DECODE_MS &&
      // A slow model gets fewer partials, so the final is never stuck behind a queue of them.
      this.utteranceSamples - this.lastPartialSamples >=
        Math.max(this.partialEvery, (this.lastDecodeMs / 1000) * 1.5 * STT_RATE)
    ) {
      this.decodePartial()
    }
  }

  private onStart(t: number): void {
    const pre = this.preroll
    this.preroll = []
    this.utterance = pre
    this.utteranceSamples = pre.reduce((a, b) => a + b.length, 0)
    this.utteranceT0 = Math.max(0, this.tracker.clock.sentSeconds - this.utteranceSamples / STT_RATE)
    this.lastPartialSamples = 0
    this.lastPartialText = ''
    this.guessed = null
    this.generation += 1
    this.tracker.speechStart(t)
  }

  private onEnd(t: number): void {
    this.tracker.speechEnd(t)
    const audio = concatFloat(this.utterance)
    const t0 = this.utteranceT0
    // The detector decides `endMs` after the last speech frame; that wait is part of the latency.
    const decidedAt = performance.now()
    const speechEndAt = this.tracker.clock.wallTimeOf(t) ?? decidedAt
    const firstPartialMs = this.firstPartialMs
    this.firstPartialMs = null
    this.utterance = []
    this.utteranceSamples = 0
    this.generation += 1
    const language = this.guessed ?? this.language
    void this.decode(audio, language, (text, ms, spoken) => {
      this.tracker.final({ text, t0, t1: t, words: null, language: spoken, latencyMs: ms })
      const finalMs = performance.now() - speechEndAt
      const endpointMs = Math.max(0, decidedAt - speechEndAt)
      recordBrowserTrace({
        model: this.engine.store.get().model.id,
        startedAtPerf: speechEndAt,
        phases: [
          { name: 'endpoint', ms: endpointMs },
          { name: 'queue', ms: Math.max(0, finalMs - endpointMs - ms) },
          { name: 'decode', ms },
        ],
        audioSeconds: audio.length / STT_RATE,
        firstPartialMs,
        finalMs,
        rtf: audio.length ? ms / 1000 / (audio.length / STT_RATE) : null,
        text,
      })
    })
  }

  private decodePartial(): void {
    const gen = this.generation
    const audio = concatFloat(this.utterance)
    this.lastPartialSamples = this.utteranceSamples
    void this.decode(audio, this.guessed ?? this.language, (text, ms, spoken) => {
      if (gen !== this.generation) return // the utterance moved on
      if (this.language === 'auto' && spoken && audio.length >= DETECT_MIN_S * STT_RATE) this.guessed = spoken
      if (!text) return
      this.firstPartialMs ??= ms
      const stable = commonPrefix(this.lastPartialText, text)
      this.lastPartialText = text
      this.tracker.partial(text, this.utteranceT0, ms, stable)
    })
  }

  private async decode(
    audio: Float32Array,
    language: string,
    done: (text: string, ms: number, language: string | null) => void,
  ): Promise<void> {
    if (audio.length < STT_RATE * 0.1) {
      done('', 0, null)
      return
    }
    this.decoding += 1
    try {
      const { text, ms, language: spoken } = await this.engine.transcribe(audio, { language })
      if (this.cancelled) return
      this.processedMs += ms
      this.lastDecodeMs = ms
      this.processedAudio += audio.length / STT_RATE
      this.store.patch({ rtf: this.processedMs / 1000 / Math.max(0.001, this.tracker.clock.sentSeconds) })
      done(text, ms, spoken)
    } catch (err) {
      if (!this.cancelled)
        this.store.patch({ status: 'error', error: err instanceof Error ? err.message : String(err) })
    } finally {
      this.decoding -= 1
      if (this.decoding === 0) this.idle?.()
    }
  }

  async finish(): Promise<void> {
    const status = this.store.get().status
    if (status !== 'streaming') {
      if (status === 'loading') this.cancel()
      return
    }
    this.store.patch({ status: 'finishing' })
    for (const ev of this.vad.flush()) if (ev.event === 'speech_end') this.onEnd(ev.t)
    if (this.decoding > 0) await new Promise<void>((resolve) => (this.idle = resolve))
    this.tracker.abandon()
    if (this.store.get().status !== 'error') this.store.patch({ status: 'done' })
  }

  cancel(): void {
    this.cancelled = true
    this.tracker.abandon()
    const s = this.store.get().status
    if (s !== 'error') this.store.patch({ status: 'done' })
    this.idle?.()
  }
}
