import { getTransport, wsUrl } from '@/lib/api/config'
import { openMicrophone, STT_RATE, type AudioSource } from '@/lib/audio/capture'
import { floatToInt16, int16ToFloat } from '@/lib/audio/pcm'
import { GaplessPlayer } from '@/lib/audio/player'
import { Resampler } from '@/lib/audio/resample'
import type { PipelineId } from '@/lib/capabilities'
import { diag } from '@/lib/diagnostics'
import type { MicFailure } from '@/lib/mic-help'
import { MicrophoneError } from '@/lib/audio/capture'
import { Store } from '@/lib/store'

/**
 * A voice conversation over bolo's OpenAI Realtime endpoint (`WS /v1/realtime`):
 * microphone in as 24 kHz PCM16, the agent's voice out, transcripts both ways.
 * The pipeline (`session.pipeline`, a bolo extension) is sent before any audio
 * and cannot change afterwards, so it is fixed for the life of a session.
 */

export const REALTIME_RATE = 24000

export type ConversationStatus = 'idle' | 'starting' | 'live' | 'ended' | 'error'

export interface Turn {
  /** Conversation item id (user) or response id (assistant). */
  id: string
  role: 'user' | 'assistant'
  text: string
  final: boolean
  /** Assistant turns: why a response ended other than completing. */
  status?: 'cancelled' | 'failed' | 'incomplete' | null
  /** ms from the end of the user's speech to the first audio of this answer. */
  firstAudioMs?: number | null
}

export interface ConversationState {
  status: ConversationStatus
  pipeline: PipelineId
  /** True once audio has gone to the server: the pipeline can no longer change. */
  locked: boolean
  /** What the server says the session uses (from `session.created` / `session.updated`). */
  serverPipeline: string | null
  turns: Turn[]
  userSpeaking: boolean
  agentSpeaking: boolean
  error: string | null
  errorCode: string | null
  mic: MicFailure | null
  /** Recent server event types, newest last (for the debug panel). */
  events: string[]
}

export const emptyConversation = (pipeline: PipelineId = 'cascade'): ConversationState => ({
  status: 'idle',
  pipeline,
  locked: false,
  serverPipeline: null,
  turns: [],
  userSpeaking: false,
  agentSpeaking: false,
  error: null,
  errorCode: null,
  mic: null,
  events: [],
})

/** The `session.update` the console sends first. */
export function sessionUpdate(pipeline: PipelineId, voice?: string): Record<string, unknown> {
  const output: Record<string, unknown> = { format: { type: 'audio/pcm', rate: REALTIME_RATE } }
  if (voice) output.voice = voice
  return {
    type: 'session.update',
    session: {
      type: 'realtime',
      pipeline,
      output_modalities: ['audio'],
      audio: {
        input: {
          format: { type: 'audio/pcm', rate: REALTIME_RATE },
          turn_detection: { type: 'server_vad' },
        },
        output,
      },
    },
  }
}

type Ev = Record<string, unknown> & { type: string }

function upsert(turns: Turn[], id: string, role: Turn['role'], fn: (t: Turn) => Turn): Turn[] {
  const i = turns.findIndex((t) => t.id === id && t.role === role)
  if (i < 0) return [...turns, fn({ id, role, text: '', final: false })]
  const next = turns.slice()
  next[i] = fn(turns[i] as Turn)
  return next
}

/** Folds one server event into the state. Pure, so it can be tested without a socket. */
export function applyEvent(s: ConversationState, ev: Ev): ConversationState {
  const events = [...s.events.slice(-39), ev.type]
  const str = (v: unknown) => (typeof v === 'string' ? v : '')
  switch (ev.type) {
    case 'session.created':
    case 'session.updated': {
      const session = (ev.session ?? {}) as { pipeline?: unknown }
      return {
        ...s,
        events,
        serverPipeline: typeof session.pipeline === 'string' ? session.pipeline : s.serverPipeline,
      }
    }
    case 'input_audio_buffer.speech_started':
      return { ...s, events, userSpeaking: true, agentSpeaking: false }
    case 'input_audio_buffer.speech_stopped':
      return { ...s, events, userSpeaking: false }
    case 'conversation.item.input_audio_transcription.delta': {
      const id = str(ev.item_id)
      return {
        ...s,
        events,
        turns: upsert(s.turns, id, 'user', (t) => ({ ...t, text: t.final ? t.text : t.text + str(ev.delta) })),
      }
    }
    case 'conversation.item.input_audio_transcription.completed': {
      const id = str(ev.item_id)
      return {
        ...s,
        events,
        turns: upsert(s.turns, id, 'user', (t) => ({ ...t, text: str(ev.transcript), final: true })),
      }
    }
    case 'response.output_audio_transcript.delta':
    case 'response.output_text.delta': {
      const id = str(ev.response_id)
      return { ...s, events, turns: upsert(s.turns, id, 'assistant', (t) => ({ ...t, text: t.text + str(ev.delta) })) }
    }
    case 'response.output_audio_transcript.done':
    case 'response.output_text.done': {
      const id = str(ev.response_id)
      const text = str(ev.transcript ?? ev.text)
      return {
        ...s,
        events,
        turns: upsert(s.turns, id, 'assistant', (t) => ({ ...t, text: text || t.text, final: true })),
      }
    }
    case 'response.done': {
      const r = (ev.response ?? {}) as {
        id?: string
        status?: string
        status_details?: { error?: { message?: string } }
      }
      const id = r.id ?? ''
      const status = r.status === 'completed' || !r.status ? null : (r.status as Turn['status'])
      const failed = r.status === 'failed'
      return {
        ...s,
        events,
        agentSpeaking: false,
        turns: s.turns.some((t) => t.id === id && t.role === 'assistant')
          ? upsert(s.turns, id, 'assistant', (t) => ({ ...t, final: true, status }))
          : s.turns,
        error: failed ? (r.status_details?.error?.message ?? 'The response failed.') : s.error,
      }
    }
    case 'error': {
      const e = (ev.error ?? {}) as { message?: string; code?: string }
      return { ...s, events, error: e.message ?? 'The server reported an error.', errorCode: e.code ?? null }
    }
    default:
      return { ...s, events }
  }
}

function toBase64(bytes: Uint8Array): string {
  let bin = ''
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(bin)
}

function fromBase64(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

/** 16 kHz float blocks from the microphone → base64 24 kHz PCM16 for `input_audio_buffer.append`. */
export class InputEncoder {
  private readonly resampler = new Resampler(STT_RATE, REALTIME_RATE)
  encode(block: Float32Array): string {
    const pcm = floatToInt16(this.resampler.process(block))
    return toBase64(new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength))
  }
}

export function decodeAudioDelta(b64: string): Float32Array {
  const bytes = fromBase64(b64)
  const even = bytes.length - (bytes.length % 2)
  return int16ToFloat(new Int16Array(bytes.buffer, bytes.byteOffset, even / 2))
}

export class ConversationSession {
  readonly store: Store<ConversationState>
  private ws: WebSocket | null = null
  private mic: AudioSource | null = null
  private ctx: AudioContext | null = null
  private player: GaplessPlayer | null = null
  private encoder = new InputEncoder()
  private pending: string[] = []
  private ready = false
  private closed = false
  private speechStoppedAt: number | null = null
  private answered = new Set<string>()

  constructor(pipeline: PipelineId) {
    this.store = new Store(emptyConversation(pipeline))
  }

  /** Choose the pipeline. Refused (returns false) once audio has gone out. */
  setPipeline(p: PipelineId): boolean {
    const s = this.store.get()
    if (s.locked) return false
    this.store.patch({ pipeline: p })
    return true
  }

  /**
   * Call directly from the tap: the microphone prompt and the audio output are
   * both opened before the first `await`, as iOS Safari requires.
   */
  start(opts: { voice?: string } = {}): Promise<void> {
    if (this.closed) return Promise.resolve()
    const { pipeline } = this.store.get()
    this.store.patch({ status: 'starting', error: null, errorCode: null, mic: null })
    let micPromise: Promise<AudioSource>
    try {
      micPromise = openMicrophone((block) => this.onBlock(block))
    } catch (err) {
      micPromise = Promise.reject(err instanceof Error ? err : new Error(String(err)))
    }
    try {
      this.ctx = new AudioContext({ latencyHint: 'interactive' })
      if (this.ctx.state === 'suspended') void this.ctx.resume().catch(() => undefined)
      this.player = new GaplessPlayer(this.ctx)
    } catch {
      this.ctx = null
    }
    diag.event('conversation', `start, pipeline=${pipeline}`)
    this.openSocket(pipeline, opts.voice)
    return micPromise.then(
      (m) => {
        if (this.closed) void m.stop()
        else this.mic = m
      },
      (err: unknown) => {
        const failure = err instanceof MicrophoneError ? err.failure : null
        this.fail(err instanceof Error ? err.message : 'Could not open the microphone.', null, failure)
      },
    )
  }

  private openSocket(pipeline: PipelineId, voice?: string): void {
    let ws: WebSocket
    try {
      const WS = getTransport().WebSocket
      ws = new WS(wsUrl('v1/realtime', { model: 'bolo' }), ['realtime'])
    } catch (err) {
      this.fail(err instanceof Error ? err.message : 'Could not open the conversation.')
      return
    }
    this.ws = ws
    ws.onopen = () => {
      ws.send(JSON.stringify(sessionUpdate(pipeline, voice)))
      this.ready = true
      for (const chunk of this.pending.splice(0)) this.sendAudio(chunk)
      this.store.patch({ status: 'live' })
    }
    ws.onmessage = (e: MessageEvent) => {
      if (typeof e.data !== 'string') return
      let ev: Ev
      try {
        ev = JSON.parse(e.data) as Ev
      } catch {
        return
      }
      this.onEvent(ev)
    }
    ws.onclose = (e: CloseEvent) => {
      if (this.closed) return
      if (e.code === 1008) this.fail('The server refused the connection (API key).', 'unauthorized')
      else if (this.store.get().status !== 'error') this.fail(e.reason || 'The connection closed.')
    }
  }

  private onEvent(ev: Ev): void {
    const before = this.store.get()
    if (ev.type === 'error') diag.error('conversation', JSON.stringify(ev.error ?? ev))
    if (ev.type === 'session.created' || ev.type === 'session.updated') {
      const p = (ev.session as { pipeline?: unknown } | undefined)?.pipeline
      if (typeof p === 'string' && p !== before.serverPipeline) diag.event('conversation', `server pipeline=${p}`)
    }
    if (ev.type === 'input_audio_buffer.speech_started') {
      // Barge-in: drop whatever of the answer is still queued.
      if (this.player && this.ctx) {
        this.player.stop()
        this.player = new GaplessPlayer(this.ctx)
      }
    }
    if (ev.type === 'input_audio_buffer.speech_stopped') this.speechStoppedAt = performance.now()
    let next = applyEvent(before, ev)
    if (ev.type === 'response.output_audio.delta' && typeof ev.delta === 'string') {
      this.player?.enqueue(decodeAudioDelta(ev.delta), REALTIME_RATE)
      const id = typeof ev.response_id === 'string' ? ev.response_id : ''
      if (!this.answered.has(id)) {
        this.answered.add(id)
        const ms = this.speechStoppedAt != null ? performance.now() - this.speechStoppedAt : null
        next = {
          ...next,
          agentSpeaking: true,
          turns: upsert(next.turns, id, 'assistant', (t) => ({ ...t, firstAudioMs: ms })),
        }
      }
    }
    this.store.set(next)
  }

  private onBlock(block: Float32Array): void {
    if (this.closed) return
    const chunk = this.encoder.encode(block)
    if (this.ready) this.sendAudio(chunk)
    else if (this.pending.length < 500) this.pending.push(chunk)
  }

  private sendAudio(chunk: string): void {
    if (this.ws?.readyState !== 1) return
    this.ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: chunk }))
    if (!this.store.get().locked) this.store.patch({ locked: true })
  }

  private fail(message: string, code: string | null = null, mic: MicFailure | null = null): void {
    if (this.closed) return
    diag.error('conversation', message)
    this.store.patch({ status: 'error', error: message, errorCode: code, mic })
    void this.teardown()
  }

  private async teardown(): Promise<void> {
    this.closed = true
    this.ready = false
    const ws = this.ws
    this.ws = null
    if (ws && ws.readyState <= 1) ws.close(1000)
    const mic = this.mic
    this.mic = null
    await mic?.stop().catch(() => undefined)
    const ctx = this.ctx
    this.ctx = null
    this.player?.stop()
    this.player = null
    if (ctx && ctx.state !== 'closed') await ctx.close().catch(() => undefined)
  }

  async stop(): Promise<void> {
    if (this.closed) return
    diag.event('conversation', 'stop')
    const s = this.store.get()
    if (s.status !== 'error') this.store.patch({ status: 'ended', userSpeaking: false, agentSpeaking: false })
    await this.teardown()
  }
}
