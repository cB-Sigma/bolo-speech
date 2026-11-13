import type { SttServerMessage, SttWord, TtsServerMessage } from '@/lib/api/types'
import { decodeAlaw, decodeMulaw } from '@/lib/audio/g711'
import { bytesToInt16, floatToInt16, int16ToFloat } from '@/lib/audio/pcm'
import { EnergyVad } from '@/lib/stt/vad'
import { CORPUS, type HintSegment } from './catalog'
import { MOCK_KEY, profileOf, type MockServer } from './engine'
import { babble, splitChunks } from './speech'

type Handler<E> = ((ev: E) => void) | null

/**
 * A WebSocket look-alike backed by the mock server. Only the parts of the
 * interface the console uses are implemented.
 */
export function createMockWebSocket(server: MockServer): typeof WebSocket {
  class MockWebSocket {
    static readonly CONNECTING = 0
    static readonly OPEN = 1
    static readonly CLOSING = 2
    static readonly CLOSED = 3
    readonly CONNECTING = 0
    readonly OPEN = 1
    readonly CLOSING = 2
    readonly CLOSED = 3
    readonly url: string
    readyState = 0
    binaryType: BinaryType = 'arraybuffer'
    onopen: Handler<Event> = null
    onmessage: Handler<MessageEvent> = null
    onclose: Handler<CloseEvent> = null
    onerror: Handler<Event> = null
    private session: SttSession | TtsSession | null = null
    private timers = new Set<ReturnType<typeof setTimeout>>()

    constructor(url: string | URL) {
      this.url = String(url)
      const u = new URL(this.url)
      this.after(15 + Math.random() * 20, () => {
        if (server.flags.offline) return this.finish(1006, '')
        if (server.flags.auth && u.searchParams.get('api_key') !== MOCK_KEY)
          return this.finish(1008, 'Invalid or missing API key')
        this.readyState = 1
        this.onopen?.(new Event('open'))
        const ctx: Ctx = {
          server,
          params: u.searchParams,
          send: (m) => this.deliver(typeof m === 'object' && !(m instanceof ArrayBuffer) ? JSON.stringify(m) : m),
          after: (ms, fn) => this.after(ms, fn),
          close: (code, reason) => this.closeWhenDrained(code, reason),
        }
        this.session = u.pathname.endsWith('/tts/stream') ? new TtsSession(ctx) : new SttSession(ctx)
      })
    }

    after(ms: number, fn: () => void): void {
      const id = setTimeout(() => {
        this.timers.delete(id)
        if (this.readyState <= 1) fn()
      }, ms)
      this.timers.add(id)
    }

    private outbox: { at: number; data: string | ArrayBuffer }[] = []
    private draining = false

    /** A few ms of simulated transit. One drain loop keeps frames strictly in order. */
    private deliver(data: string | ArrayBuffer): void {
      const last = this.outbox[this.outbox.length - 1]?.at ?? 0
      this.outbox.push({ at: Math.max(last, performance.now() + 3 + Math.random() * 5), data })
      if (!this.draining) this.drain()
    }

    private drain(): void {
      const head = this.outbox[0]
      if (!head) {
        this.draining = false
        return
      }
      this.draining = true
      this.after(Math.max(0, head.at - performance.now()), () => {
        const now = performance.now()
        while (this.outbox.length && (this.outbox[0] as { at: number }).at <= now + 0.5) {
          const msg = this.outbox.shift() as { data: string | ArrayBuffer }
          this.onmessage?.(new MessageEvent('message', { data: msg.data }))
          if (this.readyState > 1) return
        }
        this.drain()
      })
    }

    send(data: string | ArrayBufferLike | ArrayBufferView | Blob): void {
      if (this.readyState !== 1 || !this.session) return
      if (typeof data === 'string') this.session.text(data)
      else if (data instanceof Blob) void data.arrayBuffer().then((b) => this.session?.binary(new Uint8Array(b)))
      else if (ArrayBuffer.isView(data))
        this.session.binary(new Uint8Array(data.buffer, data.byteOffset, data.byteLength))
      else this.session.binary(new Uint8Array(data as ArrayBuffer))
    }

    private closeWhenDrained(code: number, reason: string): void {
      if (this.outbox.length === 0 || this.readyState > 1) this.finish(code, reason)
      else this.after(5, () => this.closeWhenDrained(code, reason))
    }

    private finish(code: number, reason: string): void {
      if (this.readyState === 3) return
      this.readyState = 3
      this.session?.dispose()
      for (const t of this.timers) clearTimeout(t)
      this.timers.clear()
      const ev = new CloseEvent('close', { code, reason, wasClean: code === 1000 })
      setTimeout(() => this.onclose?.(ev), 0)
    }

    close(code = 1000, reason = ''): void {
      this.finish(code, reason)
    }

    addEventListener(): void {}
    removeEventListener(): void {}
  }
  return MockWebSocket as unknown as typeof WebSocket
}

interface Ctx {
  server: MockServer
  params: URLSearchParams
  send: (m: SttServerMessage | TtsServerMessage | ArrayBuffer) => void
  after: (ms: number, fn: () => void) => void
  close: (code: number, reason: string) => void
}

interface TimedWord {
  w: string
  t0: number
  t1: number
}

function wordsFromSegments(segments: HintSegment[]): TimedWord[] {
  const out: TimedWord[] = []
  for (const s of segments) {
    const ws = s.text.split(/\s+/).filter(Boolean)
    const weights = ws.map((w) => w.length + 2)
    const total = weights.reduce((a, b) => a + b, 0)
    let t = s.t0
    ws.forEach((w, i) => {
      const d = ((s.t1 - s.t0) * (weights[i] as number)) / total
      out.push({ w, t0: t, t1: t + d })
      t += d
    })
  }
  return out
}

class SttSession {
  private readonly ctx: Ctx
  private readonly rate: number
  private readonly encoding: string
  private readonly vad: EnergyVad
  private readonly profile: ReturnType<typeof profileOf>
  private readonly modelId: string
  private readonly requestId: string
  private readonly started = Date.now()
  private scripted: TimedWord[] | null = null
  private cursor = 0
  private corpusIdx: number
  private audioSeconds = 0
  private segStart: number | null = null
  private lastPartialAt = 0
  private lastText = ''
  private finals: string[] = []
  private pendingFinals = 0
  private stopped = false
  private firstPartialMs: number | null = null
  private lastFinalMs: number | null = null

  constructor(ctx: Ctx) {
    this.ctx = ctx
    const p = ctx.params
    this.rate = Number(p.get('sample_rate') ?? 16000)
    this.encoding = p.get('encoding') ?? 'pcm_s16le'
    const model = ctx.server.model(p.get('model') ?? '')
    this.modelId = model?.id ?? 'unknown'
    this.profile = profileOf(model ?? { id: 'x', task: 'stt', tier: 'realtime' })
    this.vad = new EnergyVad({ sampleRate: this.rate, endMs: 200 })
    this.requestId = `r_live${Math.floor(Math.random() * 1e6).toString(36)}`
    this.corpusIdx = Math.floor(Math.random() * CORPUS.length)
    const hint = ctx.server.hint
    if (hint && hint.expires > Date.now() && hint.segments.length) this.scripted = wordsFromSegments(hint.segments)
    ctx.server.hint = null
    ctx.server.liveStreams += 1
    if (!model || model.task !== 'stt') {
      ctx.send({ type: 'error', code: 'model_not_found', message: `Unknown STT model: ${p.get('model')}` })
      ctx.after(10, () => ctx.close(1008, 'model_not_found'))
      return
    }
    if (!model.installed) {
      ctx.send({ type: 'error', code: 'model_not_installed', message: `${model.id} is not installed. Pull it first.` })
      ctx.after(10, () => ctx.close(1008, 'model_not_installed'))
      return
    }
    ctx.after(model.loaded ? 20 : 600, () => {
      model.loaded = true
      ctx.send({ type: 'ready', model: model.id, sample_rate: this.rate, request_id: this.requestId })
    })
  }

  private decode(bytes: Uint8Array): Float32Array {
    switch (this.encoding) {
      case 'mulaw':
        return int16ToFloat(decodeMulaw(bytes))
      case 'alaw':
        return int16ToFloat(decodeAlaw(bytes))
      case 'pcm_f32le':
        return new Float32Array(bytes.slice().buffer)
      default:
        return int16ToFloat(bytesToInt16(bytes))
    }
  }

  /**
   * Words heard up to an audio time. With a script, nothing is skipped: each
   * final takes every scripted word not yet emitted, so slightly different
   * segment boundaries never drop a word.
   */
  private words(t0: number, t1: number, consume: boolean): TimedWord[] {
    if (this.scripted) {
      let end = this.cursor
      while (
        end < this.scripted.length &&
        ((this.scripted[end] as TimedWord).t0 + (this.scripted[end] as TimedWord).t1) / 2 <= t1 + 0.25
      )
        end++
      const out = this.scripted.slice(this.cursor, end)
      if (consume) this.cursor = end
      return out
    }
    const sentence = (CORPUS[this.corpusIdx % CORPUS.length] as string).split(' ')
    const n = Math.max(1, Math.min(sentence.length, Math.round((t1 - t0) * 2.9)))
    const per = (t1 - t0) / n
    return sentence.slice(0, n).map((w, i) => ({ w, t0: t0 + i * per, t1: t0 + (i + 1) * per }))
  }

  private jitter(ms: number): number {
    const load = this.ctx.server.load()
    return ms * (0.85 + Math.random() * 0.35) * (1 + Math.max(0, load - 0.7) * 1.5)
  }

  binary(bytes: Uint8Array): void {
    if (this.stopped) return
    const samples = this.decode(bytes)
    this.audioSeconds += samples.length / this.rate
    for (const ev of this.vad.push(samples)) this.onVad(ev.event, ev.t)
    if (this.segStart != null && this.vad.isSpeaking && this.audioSeconds - this.lastPartialAt >= 0.24) {
      this.lastPartialAt = this.audioSeconds
      const upTo = this.audioSeconds - 0.12 // the decoder trails the audio slightly
      const t0 = this.segStart
      const latency = this.jitter(this.profile.partial)
      this.ctx.after(latency, () => {
        if (this.segStart !== t0) return
        const ws = this.words(t0, upTo, false)
        if (ws.length === 0) return
        const text = ws
          .map((w) => w.w)
          .join(' ')
          .replace(/[.,]$/, '')
          .toLowerCase()
        if (text === this.lastText) return
        const stable = ws
          .slice(0, Math.max(0, ws.length - 2))
          .map((w) => w.w)
          .join(' ').length
        this.lastText = text
        this.firstPartialMs ??= Math.round(latency)
        this.ctx.send({
          type: 'partial',
          text,
          t0: round(t0),
          t1: round(upTo),
          latency_ms: Math.round(latency),
          stable_chars: stable,
        })
      })
    }
  }

  private onVad(event: 'speech_start' | 'speech_end', t: number): void {
    this.ctx.send({ type: 'vad', event, t: round(t) })
    if (event === 'speech_start') {
      this.segStart = t
      this.lastPartialAt = t
      this.lastText = ''
      return
    }
    const t0 = this.segStart ?? Math.max(0, t - 1)
    this.segStart = null
    this.pendingFinals += 1
    const latency = this.jitter(this.profile.final)
    this.ctx.after(latency, () => {
      const ws = this.words(t0, t, true)
      let text = ws.map((w) => w.w).join(' ')
      if (!this.scripted) {
        this.corpusIdx += 1
        if (text && !/[.?!]$/.test(text)) text += '.'
      }
      if (text && !this.scripted) text = text[0]?.toUpperCase() + text.slice(1)
      const words: SttWord[] = ws.map((w) => ({ w: w.w.replace(/[.,?!]$/, ''), t0: round(w.t0), t1: round(w.t1) }))
      this.finals.push(text)
      this.lastFinalMs = Math.round(latency)
      this.ctx.send({
        type: 'final',
        text,
        t0: round(t0),
        t1: round(t),
        latency_ms: Math.round(latency),
        words,
        language: 'en',
      })
      this.ctx.send({
        type: 'metrics',
        audio_seconds: round(this.audioSeconds),
        rtf: round3(this.profile.rtf * (0.9 + Math.random() * 0.2)),
        queue_ms: Math.round(this.jitter(2)),
      })
      this.pendingFinals -= 1
      if (this.stopped && this.pendingFinals === 0) this.done()
    })
  }

  text(data: string): void {
    let msg: { type?: string }
    try {
      msg = JSON.parse(data) as { type?: string }
    } catch {
      return
    }
    if (msg.type === 'ping') {
      this.ctx.send({ type: 'pong' })
      return
    }
    if (msg.type === 'flush' || msg.type === 'stop') {
      for (const ev of this.vad.flush()) this.onVad(ev.event, ev.t)
      if (msg.type === 'stop') {
        this.stopped = true
        if (this.pendingFinals === 0) this.ctx.after(15, () => this.done())
      }
    }
  }

  private done(): void {
    const model = this.ctx.server.model(this.modelId)
    if (model && this.audioSeconds > 0.2) {
      const trace = this.ctx.server.makeTrace(model, this.started, {
        audioSeconds: round(this.audioSeconds),
        text: this.finals.join(' ').slice(0, 60) || undefined,
        transport: 'websocket',
      })
      trace.id = this.requestId
      if (this.firstPartialMs != null) trace.first_partial_ms = this.firstPartialMs
      if (this.lastFinalMs != null) trace.final_ms = this.lastFinalMs
      this.ctx.server.record(trace)
    }
    this.ctx.send({ type: 'done', request_id: this.requestId })
    this.ctx.after(30, () => this.ctx.close(1000, ''))
  }

  dispose(): void {
    this.ctx.server.liveStreams = Math.max(0, this.ctx.server.liveStreams - 1)
  }
}

class TtsSession {
  private readonly ctx: Ctx
  private buffer = ''
  private seq = 0
  private audioT = 0
  private busy: Promise<void> = Promise.resolve()
  private readonly started = performance.now()
  private firstTextAt: number | null = null
  private ttfa: number | null = null
  private segments: HintSegment[] = []
  private readonly rate: number
  private readonly modelId: string
  private readonly profile: ReturnType<typeof profileOf>
  private ok = false

  constructor(ctx: Ctx) {
    this.ctx = ctx
    const model = ctx.server.model(ctx.params.get('model') ?? '')
    this.modelId = model?.id ?? 'unknown'
    this.profile = profileOf(model ?? { id: 'x', task: 'tts', tier: 'balanced' })
    this.rate = Number(ctx.params.get('sample_rate')) || model?.sample_rate || 24000
    if (!model || model.task !== 'tts' || !model.installed) {
      ctx.send({ type: 'error', code: 'model_not_found', message: `TTS model unavailable: ${ctx.params.get('model')}` })
      ctx.after(10, () => ctx.close(1008, 'model_not_found'))
      return
    }
    if (model.available === false) {
      ctx.send({
        type: 'error',
        code: 'component_missing',
        message: `${model.id} needs eSpeak NG, which is not installed.`,
      })
      ctx.after(10, () => ctx.close(1008, 'component_missing'))
      return
    }
    this.ok = true
    ctx.server.liveStreams += 1
    ctx.after(model.loaded ? 15 : 500, () => {
      model.loaded = true
      ctx.send({ type: 'ready', model: model.id, voice: ctx.params.get('voice') ?? undefined, sample_rate: this.rate })
    })
  }

  binary(): void {}

  text(data: string): void {
    if (!this.ok) return
    let msg: { type?: string; text?: string }
    try {
      msg = JSON.parse(data) as { type?: string; text?: string }
    } catch {
      return
    }
    if (msg.type === 'text' && msg.text) {
      this.firstTextAt ??= performance.now()
      this.buffer += msg.text
    } else if (msg.type === 'flush' || msg.type === 'stop') {
      const text = this.buffer
      this.buffer = ''
      for (const chunk of splitChunks(text)) this.synth(chunk)
      this.busy = this.busy.then(() => this.finish(text))
    }
  }

  private synth(text: string): void {
    this.busy = this.busy.then(
      () =>
        new Promise<void>((resolve) => {
          const samples = babble(text, this.rate, this.seq + 1)
          const dur = samples.length / this.rate
          const load = this.ctx.server.load()
          const pressure = 1 + Math.max(0, load - 0.6) * 1.2
          // The first chunk pays the fixed start-up cost; later ones only the per-second cost.
          const cost =
            (this.seq === 0 ? this.profile.partial : 0) +
            dur * 1000 * this.profile.rtf * pressure * (0.85 + Math.random() * 0.3)
          this.ctx.after(cost, () => {
            this.ttfa ??= Math.round(performance.now() - (this.firstTextAt ?? this.started))
            this.ctx.send({
              type: 'chunk',
              seq: this.seq++,
              t0: round(this.audioT),
              dur: round(dur),
              ttfa_ms: this.ttfa,
              text,
            })
            const pcm = floatToInt16(samples)
            const frame = Math.round(this.rate * 0.1)
            for (let o = 0; o < pcm.length; o += frame) {
              this.ctx.send(pcm.slice(o, Math.min(pcm.length, o + frame)).buffer as ArrayBuffer)
            }
            this.segments.push({ t0: this.audioT, t1: this.audioT + dur, text })
            this.audioT += dur
            resolve()
          })
        }),
    )
  }

  private finish(text: string): void {
    const model = this.ctx.server.model(this.modelId)
    const elapsed = performance.now() - (this.firstTextAt ?? this.started)
    const rtf = this.audioT > 0 ? elapsed / 1000 / this.audioT : 0
    this.ctx.send({ type: 'metrics', rtf: round3(rtf) })
    this.ctx.send({ type: 'done' })
    // A round trip through the mock recogniser should hear what was said.
    this.ctx.server.hint = { segments: this.segments, expires: Date.now() + 120_000 }
    if (model && text) {
      const trace = this.ctx.server.makeTrace(model, Date.now() - elapsed, {
        chars: text.length,
        text: text.slice(0, 48),
        transport: 'websocket',
      })
      if (this.ttfa != null) trace.ttfa_ms = this.ttfa
      this.ctx.server.record(trace)
    }
    this.ctx.after(40, () => this.ctx.close(1000, ''))
  }

  dispose(): void {
    if (this.ok) this.ctx.server.liveStreams = Math.max(0, this.ctx.server.liveStreams - 1)
  }
}

const round = (v: number) => Math.round(v * 100) / 100
const round3 = (v: number) => Math.round(v * 1000) / 1000
