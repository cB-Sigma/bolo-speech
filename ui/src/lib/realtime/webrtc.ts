import { apiUrl, authHeaders, getTransport } from '@/lib/api/config'
import { MicrophoneError, microphoneError } from '@/lib/audio/capture'
import type { MicFailure } from '@/lib/mic-help'

/**
 * The WebRTC transport of a Realtime session (`POST /v1/realtime/calls`, OpenAI's browser flow):
 * the microphone is the audio track of an SDP offer, the Realtime events travel as JSON on the
 * `oai-events` data channel, and the agent's voice comes back as the answer's audio track, played by
 * an `<audio>` element. The session configuration rides with the offer (multipart `session` field), so
 * the pipeline is fixed before any audio reaches the server.
 */

export interface LinkState {
  /** `RTCPeerConnection.connectionState` */
  connection: string
  /** `RTCPeerConnection.iceConnectionState` */
  ice: string
  /** the data channel's `readyState` */
  channel: string
}

export interface RemoteAudio {
  packets: number
  /** seconds of audio received on the agent's track (silence included: the server sends it between answers) */
  seconds: number
  /** true once anything louder than silence has arrived */
  heard: boolean
}

export interface WebRtcHandlers {
  onEvent(ev: Record<string, unknown> & { type: string }): void
  onLink(link: LinkState): void
  onAudio(audio: RemoteAudio): void
  onConnected(): void
  onFail(message: string, code?: string | null, mic?: MicFailure | null): void
}

/** How long to wait for the browser's ICE candidates before sending the offer without the rest. */
const GATHER_TIMEOUT_MS = 3000

/** Waits until ICE gathering is complete (or the timeout): bolo takes no candidates trickled in later. */
export function iceGathered(pc: RTCPeerConnection, timeoutMs = GATHER_TIMEOUT_MS): Promise<void> {
  if (pc.iceGatheringState === 'complete') return Promise.resolve()
  return new Promise((resolve) => {
    const timer = setTimeout(done, timeoutMs)
    function done() {
      clearTimeout(timer)
      pc.removeEventListener('icegatheringstatechange', check)
      resolve()
    }
    function check() {
      if (pc.iceGatheringState === 'complete') done()
    }
    pc.addEventListener('icegatheringstatechange', check)
  })
}

/** A readable message (and code) for a refused `POST /v1/realtime/calls`. */
export async function callRefused(resp: Response): Promise<{ message: string; code: string | null }> {
  const text = await resp.text().catch(() => '')
  let message = text.trim() || resp.statusText
  let code: string | null = null
  try {
    const body = JSON.parse(text) as { error?: { message?: string; code?: string } }
    message = body.error?.message ?? message
    code = body.error?.code ?? null
  } catch {
    /* not JSON */
  }
  if (resp.status === 503 && /aiortc|webrtc/i.test(message)) code = 'webrtc_unavailable'
  if (resp.status === 401) code = 'unauthorized'
  return { message: `The server refused the call (${resp.status}): ${message}`, code }
}

export class WebRtcLink {
  private pc: RTCPeerConnection | null = null
  private mic: MediaStream | null = null
  private audio: HTMLAudioElement | null = null
  private channel: RTCDataChannel | null = null
  private callUrl: string | null = null
  private timer: ReturnType<typeof setInterval> | null = null
  private closed = false
  private connected = false

  private readonly h: WebRtcHandlers

  constructor(handlers: WebRtcHandlers) {
    this.h = handlers
  }

  /**
   * Call directly from the tap: the microphone prompt and the audio element are both started before
   * the first `await`, as iOS Safari requires.
   */
  start(session: Record<string, unknown>, model = 'bolo'): Promise<void> {
    if (typeof RTCPeerConnection === 'undefined') {
      this.h.onFail('This browser has no WebRTC (RTCPeerConnection). Use the WebSocket transport.')
      return Promise.resolve()
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      const err = new MicrophoneError('unsupported', 'This browser cannot capture audio here (getUserMedia).')
      this.h.onFail(err.message, null, err.failure)
      return Promise.resolve()
    }
    const asked = performance.now()
    let micPromise: Promise<MediaStream>
    try {
      micPromise = navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
    } catch (err) {
      micPromise = Promise.reject(err instanceof Error ? err : new Error(String(err)))
    }
    try {
      this.audio = new Audio()
      this.audio.autoplay = true
      void this.audio.play().catch(() => undefined) // unlocks playback inside the gesture
    } catch {
      this.audio = null
    }
    return this.connect(micPromise, asked, session, model)
  }

  private async connect(
    micPromise: Promise<MediaStream>,
    asked: number,
    session: Record<string, unknown>,
    model: string,
  ): Promise<void> {
    try {
      this.mic = await micPromise
    } catch (err) {
      const e = await microphoneError(err, asked)
      this.h.onFail(e.message, null, e.failure)
      return
    }
    if (this.closed) return void this.release()
    const pc = new RTCPeerConnection()
    this.pc = pc
    const link = (): LinkState => ({
      connection: pc.connectionState,
      ice: pc.iceConnectionState,
      channel: this.channel?.readyState ?? 'none',
    })
    pc.addEventListener('connectionstatechange', () => {
      this.h.onLink(link())
      if (pc.connectionState === 'connected' && !this.connected) {
        this.connected = true
        this.h.onConnected()
      }
      if (pc.connectionState === 'failed')
        this.h.onFail(
          'The WebRTC connection failed: the browser and the server could not reach each other. Same machine or ' +
            'network, or a server with ICE servers (webrtc_ice_servers) is needed.',
          'webrtc_failed',
        )
      if (pc.connectionState === 'disconnected' || pc.connectionState === 'closed') this.h.onLink(link())
    })
    pc.addEventListener('iceconnectionstatechange', () => this.h.onLink(link()))
    pc.addEventListener('track', (e: RTCTrackEvent) => {
      if (!this.audio) return
      this.audio.srcObject = e.streams[0] ?? new MediaStream([e.track])
      void this.audio.play().catch(() => undefined)
    })
    const track = this.mic.getAudioTracks()[0]
    if (track) pc.addTrack(track, this.mic)
    const channel = pc.createDataChannel('oai-events')
    this.channel = channel
    channel.addEventListener('open', () => this.h.onLink(link()))
    channel.addEventListener('close', () => this.h.onLink(link()))
    channel.addEventListener('message', (e: MessageEvent) => {
      if (typeof e.data !== 'string') return
      try {
        this.h.onEvent(JSON.parse(e.data) as Record<string, unknown> & { type: string })
      } catch {
        /* not JSON: ignored */
      }
    })
    this.h.onLink(link())

    let answer: string
    try {
      await pc.setLocalDescription(await pc.createOffer())
      await iceGathered(pc)
      if (this.closed) return
      const form = new FormData()
      form.append('sdp', pc.localDescription?.sdp ?? '')
      form.append('session', JSON.stringify(session))
      const url = apiUrl('v1/realtime/calls', { model })
      const resp = await getTransport().fetch(url, { method: 'POST', body: form, headers: authHeaders() })
      if (!resp.ok) {
        const { message, code } = await callRefused(resp)
        this.h.onFail(message, code)
        return
      }
      const location = resp.headers.get('Location')
      this.callUrl = location ? new URL(location, url).toString() : null
      answer = await resp.text()
    } catch (err) {
      this.h.onFail(`Could not start the call: ${err instanceof Error ? err.message : String(err)}`)
      return
    }
    if (this.closed) return void this.hangUp()
    try {
      await pc.setRemoteDescription({ type: 'answer', sdp: answer })
    } catch (err) {
      this.h.onFail(`The server's answer could not be used: ${err instanceof Error ? err.message : String(err)}`)
      return
    }
    this.timer = setInterval(() => void this.poll(), 500)
  }

  /** Sends a client event on the data channel (dropped while it is not open). */
  send(event: Record<string, unknown>): void {
    if (this.channel?.readyState === 'open') this.channel.send(JSON.stringify(event))
  }

  private async poll(): Promise<void> {
    const pc = this.pc
    if (!pc) return
    const stats = await pc.getStats().catch(() => null)
    stats?.forEach((s: Record<string, unknown>) => {
      if (s.type === 'inbound-rtp' && s.kind === 'audio') {
        const num = (v: unknown) => (typeof v === 'number' ? v : 0)
        this.h.onAudio({
          packets: num(s.packetsReceived),
          seconds: num(s.totalSamplesDuration),
          heard: num(s.totalAudioEnergy) > 1e-4,
        })
      }
    })
  }

  private hangUp(): void {
    const url = this.callUrl
    this.callUrl = null
    if (url)
      void getTransport()
        .fetch(`${url.replace(/\/$/, '')}/hangup`, { method: 'POST', headers: authHeaders() })
        .catch(() => undefined)
  }

  private release(): void {
    this.mic?.getTracks().forEach((t) => t.stop())
    this.mic = null
  }

  /** Ends the call: hang-up on the server, then the peer connection, the microphone and the player. */
  async close(): Promise<void> {
    if (this.closed) return
    this.closed = true
    if (this.timer) clearInterval(this.timer)
    this.timer = null
    await this.poll().catch(() => undefined)
    this.hangUp()
    this.channel = null
    this.pc?.close()
    this.pc = null
    this.release()
    if (this.audio) {
      this.audio.pause()
      this.audio.srcObject = null
      this.audio = null
    }
  }
}
