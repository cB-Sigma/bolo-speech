import { getTransport, wsUrl } from '@/lib/api/config'
import type { SttEncoding, SttServerMessage } from '@/lib/api/types'
import { floatToInt16 } from '@/lib/audio/pcm'
import { TelephonyEncoder, TELEPHONY_RATE } from '@/lib/audio/telephony'
import { STT_RATE } from '@/lib/audio/capture'
import { median } from '@/lib/latency/stats'
import { TranscriptTracker, type Transcriber } from './transcriber'

export interface ServerTranscriberOptions {
  model: string
  language?: string
  /** Send 8 kHz μ-law instead of 16 kHz PCM. */
  telephony?: boolean
  /** Reconnect attempts if the socket drops mid-stream. */
  maxReconnects?: number
}

export function sttStreamUrl(opts: ServerTranscriberOptions): string {
  const encoding: SttEncoding = opts.telephony ? 'mulaw' : 'pcm_s16le'
  return wsUrl('api/v1/stt/stream', {
    model: opts.model,
    language: opts.language && opts.language !== 'auto' ? opts.language : undefined,
    sample_rate: opts.telephony ? TELEPHONY_RATE : STT_RATE,
    encoding,
    interim: true,
    vad: true,
  })
}

/** Streams audio to `WS /api/v1/stt/stream` and tracks the transcript and latency. */
export class ServerTranscriber implements Transcriber {
  private readonly tracker = new TranscriptTracker()
  readonly store = this.tracker.store
  private ws: WebSocket | null = null
  private ready = false
  private closing = false
  private reconnects = 0
  private queued: Uint8Array<ArrayBuffer>[] = []
  private telephony: TelephonyEncoder | null
  private doneResolve: (() => void) | null = null
  private pingSentAt: number | null = null
  private pingTimer: ReturnType<typeof setInterval> | undefined
  private rtts: number[] = []
  private readonly opts: ServerTranscriberOptions

  constructor(opts: ServerTranscriberOptions) {
    this.opts = opts
    this.telephony = opts.telephony ? new TelephonyEncoder(STT_RATE) : null
  }

  start(): Promise<void> {
    this.store.patch({ status: 'connecting', error: null, backend: 'server' })
    return this.connect()
  }

  private connect(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      let settled = false
      let ws: WebSocket
      try {
        const WS = getTransport().WebSocket
        ws = new WS(sttStreamUrl(this.opts))
      } catch (err) {
        this.fail(err instanceof Error ? err.message : 'Could not open the stream')
        reject(err instanceof Error ? err : new Error(String(err)))
        return
      }
      ws.binaryType = 'arraybuffer'
      this.ws = ws
      this.ready = false
      ws.onmessage = (e: MessageEvent) => {
        if (typeof e.data !== 'string') return
        let msg: SttServerMessage
        try {
          msg = JSON.parse(e.data) as SttServerMessage
        } catch {
          return
        }
        if (msg.type === 'ready' && !settled) {
          settled = true
          this.ready = true
          this.reconnects = 0
          this.store.patch({ status: 'streaming', requestId: msg.request_id ?? null })
          const mode =
            msg.mode === 'pseudo' ? 'server · re-decoding' : msg.mode === 'native' ? 'server · streaming' : 'server'
          this.store.patch({ backend: mode })
          for (const frame of this.queued) ws.send(frame)
          this.queued = []
          this.startPing()
          resolve()
        }
        this.handle(msg)
      }
      ws.onerror = () => {
        // onclose follows with the details
      }
      ws.onclose = (e: CloseEvent) => {
        this.ws = null
        clearInterval(this.pingTimer)
        const status = this.store.get().status
        if (!settled) {
          // Closed before `ready`: the connection attempt itself failed.
          settled = true
          const reason =
            this.store.get().error ??
            (e.reason || (e.code === 1008 || e.code === 4401 ? 'Unauthorised' : 'Could not connect to the server'))
          if (this.reconnects === 0 || this.closing) {
            this.fail(reason)
            reject(new Error(reason))
            return
          }
        } else if (this.closing || status === 'done' || status === 'error') {
          if (status === 'finishing') this.settleDone()
          return
        }
        this.retry(e.reason || 'Connection lost')
      }
    })
  }

  /** Application-level ping: the browser cannot send websocket ping frames itself. */
  private startPing(): void {
    clearInterval(this.pingTimer)
    const ping = () => {
      if (this.ws?.readyState !== 1 || this.pingSentAt != null) return
      this.pingSentAt = performance.now()
      this.ws.send(JSON.stringify({ type: 'ping' }))
    }
    ping()
    this.pingTimer = setInterval(ping, 1500)
  }

  private retry(reason: string): void {
    const max = this.opts.maxReconnects ?? 3
    if (this.reconnects >= max) {
      this.fail(`${reason} (gave up after ${max} attempts)`)
      return
    }
    this.reconnects += 1
    this.tracker.abandon()
    this.tracker.timeOffset = this.tracker.clock.sentSeconds
    this.store.patch({ status: 'reconnecting' })
    const delay = 300 * 2 ** (this.reconnects - 1)
    setTimeout(() => {
      if (this.closing || this.store.get().status !== 'reconnecting') return
      this.connect().catch(() => undefined)
    }, delay)
  }

  private handle(msg: SttServerMessage): void {
    switch (msg.type) {
      case 'vad':
        if (msg.event === 'speech_start') this.tracker.speechStart(msg.t)
        else this.tracker.speechEnd(msg.t)
        break
      case 'partial':
        this.tracker.partial(msg.text, msg.t0, msg.latency_ms ?? null, msg.stable_chars)
        break
      case 'final':
        this.tracker.final({
          text: msg.text,
          t0: msg.t0,
          t1: msg.t1,
          words: msg.words ?? null,
          language: msg.language ?? null,
          latencyMs: msg.latency_ms ?? null,
        })
        break
      case 'metrics':
        this.store.patch({ rtf: msg.rtf, queueMs: msg.queue_ms ?? null })
        break
      case 'error':
        this.fail(msg.message || msg.code)
        break
      case 'pong':
        if (this.pingSentAt != null) {
          this.rtts.push(performance.now() - this.pingSentAt)
          if (this.rtts.length > 9) this.rtts.shift()
          this.pingSentAt = null
          this.store.patch({ rttMs: median(this.rtts) })
        }
        break
      case 'done':
        this.settleDone()
        break
      default:
        break
    }
  }

  private settleDone(): void {
    this.tracker.abandon()
    if (this.store.get().status !== 'error') this.store.patch({ status: 'done' })
    this.closing = true
    this.ws?.close(1000)
    this.doneResolve?.()
    this.doneResolve = null
  }

  private fail(message: string): void {
    this.closing = true
    this.tracker.abandon()
    this.store.patch({ status: 'error', error: message })
    this.ws?.close()
    this.doneResolve?.()
    this.doneResolve = null
  }

  push(block: Float32Array): void {
    const status = this.store.get().status
    if (status === 'done' || status === 'error' || status === 'finishing') return
    const bytes = (
      this.telephony ? this.telephony.encode(block) : new Uint8Array(floatToInt16(block).buffer)
    ) as Uint8Array<ArrayBuffer>
    this.tracker.markSent(block.length / STT_RATE)
    if (this.ws && this.ready && this.ws.readyState === 1) this.ws.send(bytes)
    else if (status === 'connecting') this.queued.push(bytes)
    // while reconnecting, audio is dropped on purpose: stale audio is worse than a gap
  }

  finish(): Promise<void> {
    const status = this.store.get().status
    if (status === 'done' || status === 'error' || status === 'idle') return Promise.resolve()
    this.store.patch({ status: 'finishing' })
    return new Promise<void>((resolve) => {
      this.doneResolve = resolve
      if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify({ type: 'stop' }))
      else this.settleDone()
      // Never hang the UI on a server that does not answer `stop`.
      setTimeout(() => {
        if (this.doneResolve) this.settleDone()
      }, 8000)
    })
  }

  cancel(): void {
    this.closing = true
    this.tracker.abandon()
    const s = this.store.get().status
    if (s !== 'done' && s !== 'error') this.store.patch({ status: 'done' })
    this.ws?.close(1000)
    this.doneResolve?.()
    this.doneResolve = null
  }
}
