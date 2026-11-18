import { GaplessPlayer } from '@/lib/audio/player'
import { concatFloat } from '@/lib/audio/pcm'
import { Store } from '@/lib/store'
import { emptyTts, type TtsChunk, type TtsState } from '@/lib/tts/session'
import type { BrowserTtsEngine } from './engine'
import { mergeShortChunks, prepareText, type PreparedText, type VoiceConfig } from './text'
import type { BrowserVoice } from './voices'

/**
 * The first piece decides how long the listener waits, so a long first
 * sentence is cut at its first comma (when that leaves a reasonable piece).
 */
export function shortFirstChunk(chunks: string[], over = 90): string[] {
  const first = chunks[0]
  if (!first || first.length <= over) return chunks
  const comma = first.indexOf(', ', 25)
  if (comma < 0 || comma > over) return chunks
  return [first.slice(0, comma + 1), first.slice(comma + 2), ...chunks.slice(1)]
}

/**
 * The pieces a voice is given, one model call each. A character voice garbles
 * a one-word sentence by itself, so short sentences first join a neighbour;
 * then a long opening is cut. A Piper voice says a short sentence well (and
 * its phonemiser makes one model call per sentence whatever it is handed), so
 * its sentences stay apart and the first one is heard as early as possible.
 */
export function synthesisChunks(sentences: string[], kind: VoiceConfig['kind']): string[] {
  return shortFirstChunk(kind === 'coqui' ? mergeShortChunks(sentences) : sentences)
}

/** Extra measurements that only exist when the voice runs in the page. */
export interface BrowserTtsExtra {
  prepared: PreparedText | null
  /** ms spent waiting for the voice to download and compile before synthesis could start. */
  loadWaitMs: number | null
  /** Sum of time spent in the model for all chunks, ms. */
  computeMs: number
  /** Phonemes per chunk, for voices that read phonemes. */
  phonemes: string[]
  /** Phonemes the voice could not say. */
  missing: string[]
}

/**
 * One synthesis in the page: prepares the text, synthesises it sentence by
 * sentence in the worker (a very short sentence together with its neighbour),
 * and plays each piece as soon as it is ready, so the first one is heard
 * while the rest are still being made. The state has
 * the same shape as a server session's, so the same panels show it.
 *
 * Time to first audio is measured from the moment the voice is ready, so that
 * a first-time download does not hide the synthesis latency; the wait for the
 * download is reported separately (`loadWaitMs`).
 */
export class BrowserTtsSession {
  readonly store = new Store<TtsState>(emptyTts())
  readonly extra = new Store<BrowserTtsExtra>({
    prepared: null,
    loadWaitMs: null,
    computeMs: 0,
    phonemes: [],
    missing: [],
  })
  readonly player: GaplessPlayer | null
  private pcm: Float32Array[] = []
  private received = 0
  private sentAt = 0
  private stopped = false
  private readonly engine: BrowserTtsEngine
  readonly voice: BrowserVoice
  readonly text: string

  constructor(
    engine: BrowserTtsEngine,
    voice: BrowserVoice,
    text: string,
    opts: { silent?: boolean; ctx?: AudioContext } = {},
  ) {
    this.engine = engine
    this.voice = voice
    this.text = text
    this.player = opts.silent ? null : new GaplessPlayer(opts.ctx)
    if (this.player) {
      this.player.onEnded = () => {
        if (this.store.get().status === 'playing') this.store.patch({ status: 'done' })
      }
    }
  }

  audio(): Float32Array {
    return concatFloat(this.pcm)
  }

  start(): Promise<void> {
    return this.run()
  }

  private async run(): Promise<void> {
    const pressed = performance.now()
    this.store.patch({ status: 'connecting' })
    try {
      await this.engine.load()
    } catch (err) {
      this.fail(err instanceof Error ? err.message : String(err))
      return
    }
    if (this.stopped) return
    const cfg = this.engine.store.get().config
    if (!cfg) {
      this.fail('The voice is not loaded')
      return
    }
    const prepared = prepareText(this.text, cfg, this.voice.numbers)
    prepared.chunks = synthesisChunks(prepared.chunks, cfg.kind)
    this.sentAt = performance.now()
    this.extra.patch({ prepared, loadWaitMs: this.sentAt - pressed > 50 ? this.sentAt - pressed : 0 })
    if (prepared.chunks.length === 0) {
      this.fail('Nothing to say: the voice has no symbols for any character in this text.')
      return
    }
    this.store.patch({ sampleRate: cfg.sample_rate })
    for (const [seq, chunk] of prepared.chunks.entries()) {
      if (this.stopped) return
      let result
      try {
        result = await this.engine.synthesize(chunk)
      } catch (err) {
        if (!this.stopped) this.fail(err instanceof Error ? err.message : String(err))
        return
      }
      if (this.stopped) return
      const x = this.extra.get()
      this.extra.patch({
        computeMs: x.computeMs + result.ms,
        phonemes: result.phonemes ? [...x.phonemes, result.phonemes] : x.phonemes,
        missing: [...new Set([...x.missing, ...result.missing])],
      })
      this.onAudio(seq, chunk, result.audio, result.sampleRate)
    }
    this.complete()
  }

  private since(): number {
    return performance.now() - this.sentAt
  }

  private onAudio(seq: number, text: string, samples: Float32Array, rate: number): void {
    if (samples.length === 0) return
    const s = this.store.get()
    const patch: Partial<TtsState> = {}
    if (s.ttfaMs == null) {
      patch.ttfaMs = this.since()
      patch.status = 'synthesizing'
    }
    this.pcm.push(samples)
    const t0 = this.received
    this.received += samples.length / rate
    let lateMs = 0
    if (this.player) {
      const sch = this.player.enqueue(samples, rate)
      lateMs = sch.lateMs
      const snap = this.player.snapshot
      patch.underruns = snap.underruns
      patch.starvedMs = snap.starvedMs
      if (s.playStartMs == null && snap.startedAt != null) patch.playStartMs = snap.startedAt - this.sentAt
    }
    const playStart = patch.playStartMs ?? s.playStartMs
    const chunk: TtsChunk = {
      seq,
      text,
      t0,
      dur: samples.length / rate,
      arrivedMs: this.since(),
      playsAtMs: playStart != null ? playStart + t0 * 1000 + (this.player?.snapshot.starvedMs ?? 0) : null,
      lateMs,
    }
    patch.chunks = [...s.chunks, chunk]
    patch.audioSeconds = this.received
    patch.sampleRate = rate
    this.store.patch(patch)
  }

  private complete(): void {
    if (this.stopped) return
    const totalMs = this.since()
    const computeMs = this.extra.get().computeMs
    this.store.patch({
      totalMs,
      // Time spent in the model per second of audio: below 1 is faster than real time.
      rtf: this.received > 0 ? computeMs / 1000 / this.received : null,
      status: this.player && this.received > 0 ? 'playing' : 'done',
    })
    this.player?.end()
  }

  private fail(message: string): void {
    if (this.stopped) return
    this.store.patch({ status: 'error', error: message })
    this.player?.stop()
    this.stopped = true
  }

  stop(): void {
    this.stopped = true
    this.engine.cancelPending()
    this.player?.stop()
    const s = this.store.get().status
    if (s !== 'error' && s !== 'done') this.store.patch({ status: 'done' })
  }

  async dispose(): Promise<void> {
    this.stop()
    await this.player?.close()
  }
}
