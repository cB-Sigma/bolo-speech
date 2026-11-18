import { MicrophoneError, openMicrophone, type AudioSource } from '@/lib/audio/capture'
import { getTransport, wsUrl } from '@/lib/api/config'
import { floatToInt16 } from '@/lib/audio/pcm'
import { Store } from '@/lib/store'

export interface WakeHit {
  keyword: string
  /** Seconds of audio since listening started. */
  time: number
  score?: number
  /** When it arrived (Date.now()), for the list. */
  at: number
}

export interface WakeState {
  status: 'idle' | 'starting' | 'listening' | 'error'
  model: string | null
  keywords: string[]
  hits: WakeHit[]
  error: string | null
}

export const idleWake: WakeState = { status: 'idle', model: null, keywords: [], hits: [], error: null }

/** Phrases typed one per line or separated by commas, as the server takes them (`keywords`). */
export function splitKeywords(text: string): string[] {
  return text
    .split(/[,;\n]/)
    .map((k) => k.trim())
    .filter(Boolean)
}

/** Folds one server message of `WS /api/v1/wake/stream` into the state. */
export function applyWakeMessage(s: WakeState, msg: Record<string, unknown>, now = Date.now()): WakeState {
  if (msg.type === 'ready') {
    return {
      ...s,
      status: 'listening',
      model: typeof msg.model === 'string' ? msg.model : s.model,
      keywords: Array.isArray(msg.keywords) ? msg.keywords.map(String) : s.keywords,
    }
  }
  if (msg.type === 'wake') {
    const hit: WakeHit = {
      keyword: String(msg.keyword ?? ''),
      time: Number(msg.time ?? 0),
      score: typeof msg.score === 'number' ? msg.score : undefined,
      at: now,
    }
    return { ...s, hits: [hit, ...s.hits].slice(0, 50) }
  }
  if (msg.type === 'error')
    return { ...s, status: 'error', error: String(msg.message ?? 'The server reported an error.') }
  return s
}

/** The microphone streamed to the server's wake-word detector. */
export class WakeListener {
  readonly store = new Store<WakeState>({ ...idleWake, status: 'starting' })
  private ws: WebSocket | null = null
  private mic: AudioSource | null = null
  private closed = false

  private readonly model: string
  private readonly keywords: string[]
  private readonly threshold: number | undefined

  constructor(model: string, keywords: string[], threshold?: number) {
    this.model = model
    this.keywords = keywords
    this.threshold = threshold
  }

  /** Call from the tap: the microphone prompt comes first. */
  start(): Promise<void> {
    const mic = openMicrophone((block) => this.send(block))
    const WS = getTransport().WebSocket
    const ws = new WS(
      wsUrl('api/v1/wake/stream', {
        model: this.model || undefined,
        keywords: this.keywords.length ? this.keywords.join(',') : undefined,
        threshold: this.threshold,
        sample_rate: 16000,
      }),
    )
    ws.binaryType = 'arraybuffer'
    this.ws = ws
    ws.onmessage = (e: MessageEvent) => {
      if (typeof e.data !== 'string') return
      try {
        this.store.set(applyWakeMessage(this.store.get(), JSON.parse(e.data) as Record<string, unknown>))
      } catch {
        // not JSON: ignore
      }
    }
    ws.onclose = (e: CloseEvent) => {
      if (this.closed || this.store.get().status === 'error') return
      this.store.patch({ status: 'error', error: e.reason || 'The connection closed.' })
    }
    return mic.then(
      (m) => {
        if (this.closed) void m.stop()
        else this.mic = m
      },
      (err: unknown) => {
        const message = err instanceof MicrophoneError || err instanceof Error ? err.message : 'No microphone.'
        this.store.patch({ status: 'error', error: message })
        void this.stop()
      },
    )
  }

  private send(block: Float32Array): void {
    if (this.ws?.readyState !== 1) return
    const pcm = floatToInt16(block)
    this.ws.send(new Int16Array(pcm))
  }

  async stop(): Promise<void> {
    this.closed = true
    const ws = this.ws
    this.ws = null
    if (ws && ws.readyState === 1) {
      ws.send(JSON.stringify({ type: 'stop' }))
      ws.close()
    }
    const mic = this.mic
    this.mic = null
    if (mic) await mic.stop()
    if (this.store.get().status !== 'error') this.store.patch({ status: 'idle' })
  }
}
