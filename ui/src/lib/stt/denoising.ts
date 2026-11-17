import { STT_RATE } from '@/lib/audio/capture'
import { concatFloat } from '@/lib/audio/pcm'
import type { AudioDeviceChoice, AudioModelEngine } from '@/lib/browser-audio/engine'
import { diag } from '@/lib/diagnostics'
import { Store } from '@/lib/store'
import type { Transcriber } from './transcriber'

/** The server's live denoising: half a second at a time, each piece enhanced with the half second before it. */
const CHUNK = STT_RATE / 2
const CONTEXT = STT_RATE / 2
/** Audio kept for the before / after comparison. */
const KEEP_SECONDS = 60

export interface DenoiseStats {
  model: string | null
  strength: number
  /** Seconds of audio denoised, and the time it took (ms). */
  seconds: number
  ms: number
  device: string | null
  error: string | null
}

/**
 * Denoising before recognition (`denoise` on the server): the audio goes through a denoiser in the page before
 * the recogniser hears it. Blocks are collected into half-second pieces; each piece is denoised together with the
 * half second before it (so the model has heard what came first) and only the new part is passed on, mixed with
 * the original at `strength` as the server does. The original and the denoised audio of the take are kept for
 * playback, side by side.
 */
export class DenoisingTranscriber implements Transcriber {
  readonly stats = new Store<DenoiseStats>({ model: null, strength: 1, seconds: 0, ms: 0, device: null, error: null })
  private readonly inner: Transcriber
  private readonly engine: AudioModelEngine
  private readonly model: string
  private readonly strength: number
  private readonly device: AudioDeviceChoice
  private pending: Float32Array[] = []
  private pendingSamples = 0
  private context = new Float32Array(0)
  private queue: Promise<void> = Promise.resolve()
  private original: Float32Array[] = []
  private denoised: Float32Array[] = []
  private kept = 0
  private cancelled = false

  constructor(
    inner: Transcriber,
    engine: AudioModelEngine,
    opts: { model: string; strength: number; device?: AudioDeviceChoice },
  ) {
    this.inner = inner
    this.engine = engine
    this.model = opts.model
    this.strength = opts.strength
    this.device = opts.device ?? 'auto'
    this.stats.patch({ model: opts.model, strength: opts.strength })
  }

  get store() {
    return this.inner.store
  }

  async start(): Promise<void> {
    await Promise.all([this.inner.start(), this.engine.load(this.model, this.device)])
    const d = this.engine.store.get().device
    this.stats.patch({ device: d === 'webgpu' ? 'WebGPU' : 'WebAssembly' })
  }

  push(block: Float32Array): void {
    if (this.cancelled) return
    this.pending.push(block.slice())
    this.pendingSamples += block.length
    if (this.pendingSamples >= CHUNK) this.flushPiece()
  }

  private flushPiece(): void {
    const piece = concatFloat(this.pending)
    this.pending = []
    this.pendingSamples = 0
    if (piece.length === 0) return
    const context = this.context
    const input = new Float32Array(context.length + piece.length)
    input.set(context)
    input.set(piece, context.length)
    this.context = input.slice(Math.max(0, input.length - CONTEXT))
    this.queue = this.queue.then(async () => {
      if (this.cancelled) return
      let out = piece
      try {
        const r = await this.engine.denoise(input, STT_RATE, this.strength)
        out = r.audio.slice(context.length)
        const s = this.stats.get()
        this.stats.patch({ seconds: s.seconds + piece.length / STT_RATE, ms: s.ms + r.ms })
      } catch (err) {
        // The recogniser still hears the audio, undenoised, as the server does when a denoiser cannot run.
        const message = err instanceof Error ? err.message : String(err)
        if (!this.stats.get().error) diag.error('denoise', `${this.model}: ${message}; passing the audio on as it is`)
        this.stats.patch({ error: message })
      }
      if (this.cancelled) return
      if (this.kept < KEEP_SECONDS * STT_RATE) {
        this.original.push(piece)
        this.denoised.push(out)
        this.kept += piece.length
      }
      this.inner.push(out)
    })
  }

  async finish(): Promise<void> {
    if (this.pendingSamples) this.flushPiece()
    await this.queue
    await this.inner.finish()
  }

  cancel(): void {
    this.cancelled = true
    this.inner.cancel()
  }

  /** The take before and after denoising, as the recogniser heard it (16 kHz). */
  takes(): { original: Float32Array; denoised: Float32Array } {
    return { original: concatFloat(this.original), denoised: concatFloat(this.denoised) }
  }
}
