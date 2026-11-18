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
import { WebRtcLink, type LinkState, type RemoteAudio } from './webrtc'

/**
 * A voice conversation over bolo's OpenAI Realtime endpoint, on one of two transports:
 *
 * - `websocket` (`WS /v1/realtime`): microphone in as 24 kHz PCM16 in `input_audio_buffer.append`
 *   events, the agent's voice out as `response.output_audio.delta` events;
 * - `webrtc` (`POST /v1/realtime/calls`, OpenAI's browser flow): the microphone is a WebRTC audio
 *   track, the voice comes back on the answer's track, the events travel on a data channel (./webrtc.ts).
 *
 * Transcripts arrive both ways on either. The pipeline (`session.pipeline`, a bolo extension) is sent
 * before any audio and cannot change afterwards, so it is fixed for the life of a session.
 */

export const REALTIME_RATE = 24000

export type ConversationStatus = 'idle' | 'starting' | 'live' | 'ended' | 'error'

export type RealtimeTransport = 'websocket' | 'webrtc'

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
  transport: RealtimeTransport
  /** WebRTC: the peer connection, ICE and data channel states. */
  link: LinkState | null
  /** WebRTC: what has arrived on the agent's audio track. */
  remoteAudio: RemoteAudio | null
  /** True once audio has gone to the server: the pipeline can no longer change. */
  locked: boolean
  /** What the server says the session uses (from `session.created` / `session.updated`). */
  serverPipeline: string | null
  /** The denoiser the server says runs before transcription (`audio.input.denoise`); null: none reported. */
  serverDenoise: { enabled: boolean; model: string | null; strength: number } | null
  /** The turn detection the server says the session uses (`audio.input.turn_detection`), null until it says. */
  serverListening: { type: string; vad_model?: string | null; turn_model?: string | null } | null
  /** The server says the session talks to a full-duplex model (`session.full_duplex`, a bolo extension). */
  serverFullDuplex: boolean | null
  turns: Turn[]
  userSpeaking: boolean
  agentSpeaking: boolean
  error: string | null
  errorCode: string | null
  mic: MicFailure | null
  /** MCP servers the session opened (`mcp_list_tools` items), by server label. */
  mcpServers: Record<string, McpServerState>
  /** MCP tool calls the server made for the language model, and approval requests waiting for an answer. */
  toolCalls: ToolCall[]
  /** Recent server event types, newest last (for the debug panel). */
  events: string[]
}

export interface McpServerState {
  status: 'listing' | 'ready' | 'failed'
  tools: string[]
  error: string | null
}

export interface ToolCall {
  /** The item id (`mcp_call` or `mcp_approval_request`). */
  id: string
  server: string
  name: string
  arguments: string
  status: 'approval' | 'allowed' | 'denied' | 'running' | 'done' | 'failed'
  output: string | null
}

export const emptyConversation = (
  pipeline: PipelineId = 'cascade',
  transport: RealtimeTransport = 'websocket',
): ConversationState => ({
  status: 'idle',
  pipeline,
  transport,
  link: null,
  remoteAudio: null,
  locked: false,
  serverPipeline: null,
  serverDenoise: null,
  serverListening: null,
  serverFullDuplex: null,
  turns: [],
  userSpeaking: false,
  agentSpeaking: false,
  error: null,
  errorCode: null,
  mic: null,
  mcpServers: {},
  toolCalls: [],
  events: [],
})

/**
 * The `session.update` the console sends first. Over WebRTC the audio travels on tracks (Opus), so no
 * audio formats are named; the same session object goes with the SDP offer instead.
 */
export function sessionUpdate(
  pipeline: PipelineId,
  voice?: string,
  transport: RealtimeTransport = 'websocket',
  denoise?: RealtimeDenoise,
  listening?: Listening,
  extras: SessionExtras = {},
): Record<string, unknown> & { session: Record<string, unknown> } {
  const pcm = transport === 'websocket'
  const format = { type: 'audio/pcm', rate: REALTIME_RATE }
  const output: Record<string, unknown> = pcm ? { format } : {}
  if (voice) output.voice = voice
  const turn_detection: Record<string, unknown> = { type: listening?.type ?? 'server_vad' }
  // bolo's extensions: the session's voice-activity detector and end-of-turn model; left out, the server's defaults
  if (listening?.vadModel) turn_detection.vad_model = listening.vadModel
  if (listening?.type === 'semantic_vad' && listening.turnModel) turn_detection.turn_model = listening.turnModel
  const input: Record<string, unknown> = pcm ? { format, turn_detection } : { turn_detection }
  // bolo's extension; left out, the server's default (its `denoise` setting) applies
  if (denoise !== undefined) input.denoise = denoise
  const session: Record<string, unknown> = {
    type: 'realtime',
    pipeline,
    output_modalities: ['audio'],
    audio: { input, output },
  }
  if (extras.tools?.length) session.tools = extras.tools
  return { type: 'session.update', session }
}

/** More of the first `session.update`: tools (MCP servers, `{"type": "mcp", ...}`). */
export interface SessionExtras {
  tools?: Record<string, unknown>[]
}

/**
 * How the session hears the end of a turn: `server_vad` (a pause ends it) or `semantic_vad` (an end-of-turn model is
 * asked at each pause), with the voice-activity detector and the end-of-turn model by id ('' for the server's).
 */
export interface Listening {
  type: 'server_vad' | 'semantic_vad'
  vadModel?: string
  turnModel?: string
}

/** `audio.input.denoise`: off, or a model and strength (see `realtimeDenoise`). */
export type RealtimeDenoise = false | { model?: string; strength: number }

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
      const session = (ev.session ?? {}) as {
        pipeline?: unknown
        full_duplex?: unknown
        audio?: { input?: { denoise?: unknown; turn_detection?: unknown } }
      }
      const d = session.audio?.input?.denoise as ConversationState['serverDenoise'] | undefined
      const td = session.audio?.input?.turn_detection as ConversationState['serverListening'] | undefined
      return {
        ...s,
        events,
        serverPipeline: typeof session.pipeline === 'string' ? session.pipeline : s.serverPipeline,
        serverDenoise: d && typeof d === 'object' ? d : d === null ? null : s.serverDenoise,
        serverListening: td && typeof td === 'object' && typeof td.type === 'string' ? td : s.serverListening,
        serverFullDuplex: typeof session.full_duplex === 'boolean' ? session.full_duplex : s.serverFullDuplex,
      }
    }
    case 'input_audio_buffer.speech_started':
      return { ...s, events, userSpeaking: true, agentSpeaking: false }
    case 'input_audio_buffer.speech_stopped':
      return { ...s, events, userSpeaking: false }
    // WebRTC: what the agent's audio track is doing
    case 'output_audio_buffer.started':
      return { ...s, events, agentSpeaking: true }
    case 'output_audio_buffer.stopped':
    case 'output_audio_buffer.cleared':
      return { ...s, events, agentSpeaking: false }
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
    case 'conversation.item.added':
    case 'conversation.item.done':
      return { ...applyItem(s, ev.item as Item | undefined, ev.type === 'conversation.item.done'), events }
    case 'response.mcp_call.in_progress':
      return { ...s, events, toolCalls: patchCall(s.toolCalls, str(ev.item_id), { status: 'running' }) }
    case 'error': {
      const e = (ev.error ?? {}) as { message?: string; code?: string }
      return { ...s, events, error: e.message ?? 'The server reported an error.', errorCode: e.code ?? null }
    }
    default:
      return { ...s, events }
  }
}

type Item = Record<string, unknown> & { id?: string; type?: string }

function patchCall(calls: ToolCall[], id: string, patch: Partial<ToolCall>): ToolCall[] {
  return calls.map((c) => (c.id === id ? { ...c, ...patch } : c))
}

/** MCP items: a server's tool list (`mcp_list_tools`), a call (`mcp_call`) and an approval request. */
function applyItem(s: ConversationState, item: Item | undefined, done: boolean): ConversationState {
  if (!item || typeof item.id !== 'string') return s
  const str = (v: unknown) => (typeof v === 'string' ? v : '')
  if (item.type === 'mcp_list_tools') {
    const tools = Array.isArray(item.tools) ? item.tools.map((t) => str((t as { name?: unknown }).name)) : []
    const error = typeof item.error === 'string' ? item.error : null
    const status: McpServerState['status'] = !done ? 'listing' : error ? 'failed' : 'ready'
    return { ...s, mcpServers: { ...s.mcpServers, [str(item.server_label)]: { status, tools, error } } }
  }
  if (item.type === 'mcp_call' || item.type === 'mcp_approval_request') {
    const err = item.error as { message?: string } | null | undefined
    const done = item.type === 'mcp_call' && (item.output != null || err != null)
    const call: ToolCall = {
      id: item.id,
      server: str(item.server_label),
      name: str(item.name),
      arguments: str(item.arguments),
      status: item.type === 'mcp_approval_request' ? 'approval' : done ? (err ? 'failed' : 'done') : 'running',
      output: err ? (err.message ?? 'failed') : typeof item.output === 'string' ? item.output : null,
    }
    const old = s.toolCalls.find((c) => c.id === call.id)
    if (old && call.status === 'approval') return s // answered already, or the same request again (item.done)
    const toolCalls = old ? patchCall(s.toolCalls, call.id, call) : [...s.toolCalls.slice(-19), call]
    // an approved request becomes a call of its own: mark the request
    const req = str(item.approval_request_id)
    return { ...s, toolCalls: req ? patchCall(toolCalls, req, { status: 'allowed' }) : toolCalls }
  }
  return s
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
  private rtc: WebRtcLink | null = null
  private mic: AudioSource | null = null
  private ctx: AudioContext | null = null
  private player: GaplessPlayer | null = null
  private encoder = new InputEncoder()
  private pending: string[] = []
  private ready = false
  private closed = false
  private speechStoppedAt: number | null = null
  private answered = new Set<string>()

  private readonly denoise: RealtimeDenoise | undefined
  private readonly listening: Listening | undefined
  private readonly extras: SessionExtras
  /** A full-duplex model keeps talking when the user talks: the page does not cut its audio on speech_started. */
  private readonly fullDuplex: boolean

  constructor(
    pipeline: PipelineId,
    transport: RealtimeTransport = 'websocket',
    denoise?: RealtimeDenoise,
    opts: { listening?: Listening; fullDuplex?: boolean; tools?: Record<string, unknown>[] } = {},
  ) {
    this.store = new Store(emptyConversation(pipeline, transport))
    this.denoise = denoise
    this.listening = opts.listening
    this.extras = { tools: opts.tools }
    this.fullDuplex = Boolean(opts.fullDuplex)
  }

  /**
   * Stop the answer being spoken: `response.cancel` (and, over WebRTC, `output_audio_buffer.clear`), and drop what
   * is queued here. For a full-duplex model this is the way to stop it; talking over it does not.
   */
  stopAnswer(): void {
    if (this.closed) return
    diag.event('conversation', 'stop the answer')
    if (this.rtc) {
      this.rtc.send({ type: 'output_audio_buffer.clear' })
      this.rtc.send({ type: 'response.cancel' })
    } else if (this.ws?.readyState === 1) {
      this.ws.send(JSON.stringify({ type: 'response.cancel' }))
    }
    if (this.player && this.ctx) {
      this.player.stop()
      this.player = new GaplessPlayer(this.ctx)
    }
    this.store.patch({ agentSpeaking: false })
  }

  /** Answer an MCP approval request (`mcp_approval_response`): the server runs the call, or tells the model no. */
  answerApproval(id: string, approve: boolean): void {
    if (this.closed) return
    const ev = {
      type: 'conversation.item.create',
      item: { type: 'mcp_approval_response', approval_request_id: id, approve },
    }
    if (this.rtc) this.rtc.send(ev)
    else if (this.ws?.readyState === 1) this.ws.send(JSON.stringify(ev))
    else return
    const s = this.store.get()
    this.store.patch({ toolCalls: patchCall(s.toolCalls, id, { status: approve ? 'allowed' : 'denied' }) })
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
    const { pipeline, transport } = this.store.get()
    this.store.patch({ status: 'starting', error: null, errorCode: null, mic: null })
    if (transport === 'webrtc') return this.startWebRtc(pipeline, opts.voice)
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

  private startWebRtc(pipeline: PipelineId, voice?: string): Promise<void> {
    diag.event('conversation', `start, pipeline=${pipeline}, transport=webrtc`)
    const link = new WebRtcLink({
      onEvent: (ev) => this.onEvent(ev),
      onLink: (l) => this.store.patch({ link: l }),
      onAudio: (audio) => this.store.patch({ remoteAudio: audio }),
      onConnected: () => {
        diag.event('conversation', 'webrtc connected')
        // the configuration went with the offer: audio flows from here on and the pipeline is fixed
        if (this.store.get().status === 'starting') this.store.patch({ status: 'live', locked: true })
      },
      onFail: (message, code, mic) => this.fail(message, code ?? null, mic ?? null),
    })
    this.rtc = link
    return link.start(sessionUpdate(pipeline, voice, 'webrtc', this.denoise, this.listening, this.extras).session)
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
      ws.send(JSON.stringify(sessionUpdate(pipeline, voice, 'websocket', this.denoise, this.listening, this.extras)))
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
    if (ev.type === 'input_audio_buffer.speech_started' && !this.fullDuplex) {
      // Barge-in: drop whatever of the answer is still queued (a full-duplex model decides that itself).
      if (this.player && this.ctx) {
        this.player.stop()
        this.player = new GaplessPlayer(this.ctx)
      }
    }
    if (ev.type === 'input_audio_buffer.speech_stopped') this.speechStoppedAt = performance.now()
    let next = applyEvent(before, ev)
    if (ev.type === 'output_audio_buffer.started') {
      // WebRTC: the answer's first audio is on the track now
      const id = typeof ev.response_id === 'string' ? ev.response_id : ''
      if (id && !this.answered.has(id)) {
        this.answered.add(id)
        const ms = this.speechStoppedAt != null ? performance.now() - this.speechStoppedAt : null
        next = { ...next, turns: upsert(next.turns, id, 'assistant', (t) => ({ ...t, firstAudioMs: ms })) }
      }
    }
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
    const rtc = this.rtc
    this.rtc = null
    await rtc?.close()
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
