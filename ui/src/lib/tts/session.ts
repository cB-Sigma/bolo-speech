import { api, ApiError } from '@/lib/api/client'
import { getTransport, wsUrl } from '@/lib/api/config'
import type { TtsServerMessage } from '@/lib/api/types'
import { GaplessPlayer } from '@/lib/audio/player'
import { bytesToInt16, concatFloat, int16ToFloat } from '@/lib/audio/pcm'
import { Store } from '@/lib/store'

/** Chunk start times are rounded on the wire; audio must clearly pass one before the chunk counts as begun. */
const EPSILON = 0.005

export type TtsStatus = 'idle' | 'connecting' | 'synthesizing' | 'playing' | 'done' | 'error'
export type TtsTransport = 'websocket' | 'http'

export interface TtsChunk {
  seq: number
  text: string
  /** Position on the audio timeline (seconds). */
  t0: number
  dur: number
  /** ms after the request was sent that this chunk's audio arrived. */
  arrivedMs: number
  /** ms after the request at which the chunk starts playing; null until scheduled. */
  playsAtMs: number | null
  /** ms playback stalled waiting for this chunk (0 when it was on time). */
  lateMs: number
}

export interface TtsState {
  status: TtsStatus
  transport: TtsTransport
  error: string | null
  /** Machine-readable code from the server, e.g. `component_missing`. */
  errorCode: string | null
  sampleRate: number | null
  chunks: TtsChunk[]
  /** Time to first audio measured in the browser / reported by the server. */
  ttfaMs: number | null
  serverTtfaMs: number | null
  /** Synthesis time ÷ audio duration, measured in the browser / reported by the server. */
  rtf: number | null
  serverRtf: number | null
  audioSeconds: number
  /** ms from request to the last audio byte. */
  totalMs: number | null
  underruns: number
  starvedMs: number
  /** ms after the request at which playback began. */
  playStartMs: number | null
  /** What the server reported in the response headers (chunked HTTP only; the WebSocket sends none). */
  reported: TtsReported | null
}

/** The `x-bolo-*` headers of a synthesis response. */
export interface TtsReported {
  model: string | null
  /** `x-bolo-blend`: the blend as spoken, normalised weights. */
  blend: string | null
  /** `x-bolo-voice`: the saved voice or saved blend used. */
  voice: string | null
  /** `x-bolo-language`, raw (see `parseLanguageHeader`). */
  language: string | null
  fallback: string | null
  /** `x-bolo-instructions`: `applied` or `ignored` when the request carried instructions. */
  instructions: string | null
}

export function reportedFrom(headers: Headers): TtsReported {
  return {
    model: headers.get('x-bolo-model'),
    blend: headers.get('x-bolo-blend'),
    voice: headers.get('x-bolo-voice'),
    language: headers.get('x-bolo-language'),
    fallback: headers.get('x-bolo-fallback'),
    instructions: headers.get('x-bolo-instructions'),
  }
}

export const emptyTts = (): TtsState => ({
  status: 'idle',
  transport: 'websocket',
  error: null,
  errorCode: null,
  sampleRate: null,
  chunks: [],
  ttfaMs: null,
  serverTtfaMs: null,
  rtf: null,
  serverRtf: null,
  audioSeconds: 0,
  totalMs: null,
  underruns: 0,
  starvedMs: 0,
  playStartMs: null,
  reported: null,
})

export interface TtsOptions {
  model: string
  voice?: string
  text: string
  transport?: TtsTransport
  /** Skip playback (used for previews that only need the audio). */
  silent?: boolean
  /** Language for multilingual models (and text normalisation). */
  language?: string
  /** `auto`, `uccharan`, `uccharan:improved` or `espeak-ng`; omitted means the voice's pin or the server default. */
  phonemizer?: string
  /** Voice cloning: a `data:audio/…;base64,` recording. Only chunked HTTP carries it. */
  referenceAudio?: string
  referenceText?: string
  /** `text` (default), `ssml` or `markdown`. Only chunked HTTP carries a format other than text. */
  inputFormat?: 'text' | 'ssml' | 'markdown'
  /** Word lists to apply, highest priority first. */
  wordlists?: string[]
  /** Romanised text: `off`, `auto` or `on`; omitted means the voice's setting. */
  romanised?: string
  /** False turns off guessing the language from the text (chunked HTTP; the server's default is on). */
  detectLanguage?: boolean
  /** Direction for the delivery in words (style, emotion, voice), for models with the `instructions` feature. */
  instructions?: string
}

/** Finds the PCM payload of a WAV stream once enough header bytes have arrived. */
export function parseWavHeader(bytes: Uint8Array): { sampleRate: number; channels: number; dataOffset: number } | null {
  if (bytes.length < 44) return null
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const tag = (o: number) => String.fromCharCode(bytes[o]!, bytes[o + 1]!, bytes[o + 2]!, bytes[o + 3]!)
  if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE') throw new Error('Expected a WAV stream')
  let o = 12
  let sampleRate = 0
  let channels = 1
  while (o + 8 <= bytes.length) {
    const id = tag(o)
    const size = view.getUint32(o + 4, true)
    if (id === 'fmt ') {
      if (o + 24 > bytes.length) return null
      channels = view.getUint16(o + 10, true)
      sampleRate = view.getUint32(o + 12, true)
    } else if (id === 'data') {
      return sampleRate ? { sampleRate, channels, dataOffset: o + 8 } : null
    }
    o += 8 + size + (size & 1)
  }
  return null
}

/**
 * One synthesis request: streams audio over the TTS websocket (or chunked
 * HTTP), schedules it gaplessly, and records when each piece arrived versus
 * when it was needed.
 */
export class TtsSession {
  readonly store = new Store<TtsState>(emptyTts())
  readonly player: GaplessPlayer | null
  private ws: WebSocket | null = null
  private abort: AbortController | null = null
  private sentAt = 0
  private pcm: Float32Array[] = []
  private pendingMeta: Extract<TtsServerMessage, { type: 'chunk' }>[] = []
  private received = 0 // seconds of audio received
  private closed = false
  private readonly opts: TtsOptions

  constructor(opts: TtsOptions, ctx?: AudioContext) {
    this.opts = opts
    this.player = opts.silent ? null : new GaplessPlayer(ctx)
    this.store.patch({ transport: TtsSession.transportFor(opts) })
    if (this.player) {
      this.player.onEnded = () => {
        if (this.store.get().status === 'playing') this.store.patch({ status: 'done' })
      }
    }
  }

  /** The WebSocket cannot carry a reference recording, so cloning always goes over HTTP. */
  static transportFor(opts: Pick<TtsOptions, 'transport' | 'referenceAudio' | 'inputFormat'>): TtsTransport {
    if (opts.referenceAudio || (opts.inputFormat && opts.inputFormat !== 'text')) return 'http'
    return opts.transport ?? 'websocket'
  }

  /** All audio received so far, mono float at `sampleRate`. */
  audio(): Float32Array {
    return concatFloat(this.pcm)
  }

  start(): void {
    this.sentAt = performance.now()
    this.store.patch({ status: 'connecting' })
    if (TtsSession.transportFor(this.opts) === 'http') void this.runHttp()
    else this.runWs()
  }

  private since(): number {
    return performance.now() - this.sentAt
  }

  private onAudio(samples: Float32Array, rate: number): void {
    if (this.closed || samples.length === 0) return
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
    patch.audioSeconds = this.received
    patch.sampleRate = rate
    this.store.patch(patch)
    this.attach(t0, lateMs)
  }

  /** Bind chunk metadata to the audio that has arrived for it. */
  private attach(_t0: number, lateMs: number): void {
    const s = this.store.get()
    let chunks = s.chunks
    const startsChunk =
      this.pendingMeta.length > 0 && (this.pendingMeta[0] as { t0: number }).t0 + EPSILON < this.received
    // A stall belongs to the chunk whose audio was late: a new chunk if one starts here, else the current one.
    if (lateMs > 0 && !startsChunk && chunks.length) {
      const last = chunks[chunks.length - 1] as TtsChunk
      chunks = [...chunks.slice(0, -1), { ...last, lateMs: last.lateMs + lateMs }]
    }
    let first = true
    while (this.pendingMeta.length && (this.pendingMeta[0] as { t0: number }).t0 + EPSILON < this.received) {
      const m = this.pendingMeta.shift() as Extract<TtsServerMessage, { type: 'chunk' }>
      chunks = this.append(chunks, this.makeChunk(m, first ? lateMs : 0))
      first = false
    }
    if (chunks !== s.chunks) this.store.patch({ chunks })
  }

  /** A chunk without text continues the previous sentence: extend that row instead of adding one. */
  private append(chunks: TtsChunk[], next: TtsChunk): TtsChunk[] {
    const last = chunks[chunks.length - 1]
    if (last && !next.text.trim()) {
      return [...chunks.slice(0, -1), { ...last, dur: next.t0 + next.dur - last.t0, lateMs: last.lateMs + next.lateMs }]
    }
    return [...chunks, { ...next, seq: chunks.length }]
  }

  private makeChunk(m: Extract<TtsServerMessage, { type: 'chunk' }>, lateMs: number): TtsChunk {
    const s = this.store.get()
    const starved = this.player?.snapshot.starvedMs ?? 0
    return {
      seq: m.seq,
      text: m.text,
      t0: m.t0,
      dur: m.dur,
      arrivedMs: this.since(),
      playsAtMs: s.playStartMs != null ? s.playStartMs + m.t0 * 1000 + starved : null,
      lateMs,
    }
  }

  private complete(): void {
    if (this.closed) return
    if (this.received === 0) {
      this.fail('The server returned no audio for this text.')
      return
    }
    const s = this.store.get()
    const totalMs = this.since()
    // Metadata without matching audio (zero-length chunks) is still listed.
    let chunks = s.chunks
    for (const m of this.pendingMeta.splice(0)) chunks = this.append(chunks, this.makeChunk(m, 0))
    this.store.patch({
      chunks,
      totalMs,
      rtf: this.received > 0 ? totalMs / 1000 / this.received : null,
      status: this.player && this.received > 0 ? 'playing' : 'done',
    })
    this.player?.end()
  }

  private fail(message: string, code: string | null = null): void {
    if (this.closed) return
    this.store.patch({ status: 'error', error: message, errorCode: code })
    this.player?.stop()
    this.closed = true
    this.ws?.close()
  }

  private runWs(): void {
    const { model, voice, text, language, phonemizer, wordlists, romanised, instructions } = this.opts
    let ws: WebSocket
    try {
      const WS = getTransport().WebSocket
      ws = new WS(
        wsUrl('api/v1/tts/stream', {
          model,
          voice,
          format: 'pcm_s16le',
          language,
          phonemizer,
          wordlists: wordlists?.length ? wordlists.join(',') : undefined,
          romanised,
          instructions,
        }),
      )
    } catch (err) {
      this.fail(err instanceof Error ? err.message : 'Could not open the stream')
      return
    }
    ws.binaryType = 'arraybuffer'
    this.ws = ws
    let rate = 24000
    let finished = false
    let sent = false
    const sendText = () => {
      if (sent || ws.readyState !== 1) return
      sent = true
      this.sentAt = performance.now()
      ws.send(JSON.stringify({ type: 'text', text }))
      // `stop` lets the server finish what it was given and then say `done`.
      ws.send(JSON.stringify({ type: 'flush' }))
      ws.send(JSON.stringify({ type: 'stop' }))
    }
    ws.onmessage = (e: MessageEvent) => {
      if (typeof e.data !== 'string') {
        this.onAudio(int16ToFloat(bytesToInt16(new Uint8Array(e.data as ArrayBuffer))), rate)
        return
      }
      let msg: TtsServerMessage
      try {
        msg = JSON.parse(e.data) as TtsServerMessage
      } catch {
        return
      }
      if (msg.type === 'ready') {
        if (msg.sample_rate) rate = msg.sample_rate
        this.store.patch({ sampleRate: rate })
        sendText()
      } else if (msg.type === 'chunk') {
        if (msg.ttfa_ms != null && this.store.get().serverTtfaMs == null)
          this.store.patch({ serverTtfaMs: msg.ttfa_ms })
        this.pendingMeta.push(msg)
        this.attach(msg.t0, 0)
      } else if (msg.type === 'metrics') {
        this.store.patch({ serverRtf: msg.rtf })
      } else if (msg.type === 'error') {
        this.fail(msg.message, msg.code ?? null)
      } else if (msg.type === 'done') {
        finished = true
        this.complete()
        ws.close(1000)
      }
    }
    ws.onclose = (e: CloseEvent) => {
      if (finished || this.closed) return
      if (this.received > 0) this.complete()
      else this.fail(e.reason || 'Could not connect to the server')
    }
  }

  private async runHttp(): Promise<void> {
    const {
      model,
      voice,
      text,
      language,
      phonemizer,
      referenceAudio,
      referenceText,
      inputFormat,
      wordlists,
      romanised,
      detectLanguage,
      instructions,
    } = this.opts
    this.abort = new AbortController()
    try {
      const res = await api.speech(
        {
          model,
          input: text,
          voice: referenceAudio ? undefined : voice,
          response_format: 'wav',
          language,
          phonemizer,
          reference_audio: referenceAudio,
          reference_text: referenceAudio && referenceText ? referenceText : undefined,
          input_format: inputFormat && inputFormat !== 'text' ? inputFormat : undefined,
          wordlists: wordlists?.length ? wordlists : undefined,
          romanised,
          detect_language: detectLanguage === false ? false : undefined,
          instructions,
        },
        this.abort.signal,
      )
      this.store.patch({ reported: reportedFrom(res.headers) })
      if (!res.body) throw new Error('The server sent no audio')
      const reader = res.body.getReader()
      let head: Uint8Array = new Uint8Array(0)
      let header: ReturnType<typeof parseWavHeader> = null
      let odd: Uint8Array = new Uint8Array(0)
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        let data: Uint8Array = value
        if (!header) {
          const joined = new Uint8Array(head.length + data.length)
          joined.set(head)
          joined.set(data, head.length)
          header = parseWavHeader(joined)
          if (!header) {
            head = joined
            continue
          }
          data = joined.subarray(header.dataOffset)
        }
        if (odd.length) {
          const joined = new Uint8Array(odd.length + data.length)
          joined.set(odd)
          joined.set(data, odd.length)
          data = joined
        }
        const frame = 2 * header.channels
        const usable = data.length - (data.length % frame)
        odd = data.slice(usable)
        const pcm = bytesToInt16(data.subarray(0, usable))
        let mono = int16ToFloat(pcm)
        if (header.channels > 1) {
          const m = new Float32Array(pcm.length / header.channels)
          for (let i = 0; i < m.length; i++) m[i] = mono[i * header.channels] as number
          mono = m
        }
        this.onAudio(mono, header.sampleRate)
      }
      // Chunked HTTP carries no per-chunk metadata: show the whole response as one span.
      this.pendingMeta.push({ type: 'chunk', seq: 0, t0: 0, dur: this.received, text })
      this.complete()
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      this.fail(err instanceof Error ? err.message : 'Synthesis failed', err instanceof ApiError ? err.code : null)
    }
  }

  stop(): void {
    this.closed = true
    this.abort?.abort()
    this.ws?.close(1000)
    this.player?.stop()
    const s = this.store.get().status
    if (s !== 'error' && s !== 'done') this.store.patch({ status: 'done' })
  }

  async dispose(): Promise<void> {
    this.stop()
    await this.player?.close()
  }
}
