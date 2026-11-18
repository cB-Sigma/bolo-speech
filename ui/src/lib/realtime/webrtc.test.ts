import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { setServerBase, setTransport } from '@/lib/api/config'
import { applyEvent, ConversationSession, emptyConversation, sessionUpdate } from './session'
import { callRefused, iceGathered } from './webrtc'

/** Just enough of RTCPeerConnection for the console's call flow. */
class FakePeer extends EventTarget {
  static last: FakePeer | null = null
  connectionState = 'new'
  iceConnectionState = 'new'
  iceGatheringState = 'new'
  localDescription: { type: string; sdp: string } | null = null
  remote: { type: string; sdp: string } | null = null
  tracks: unknown[] = []
  channelLabel = ''
  closed = false
  channel = Object.assign(new EventTarget(), { readyState: 'connecting', send: vi.fn() })
  constructor() {
    super()
    FakePeer.last = this
  }
  addTrack(track: unknown) {
    this.tracks.push(track)
  }
  createDataChannel(label: string) {
    this.channelLabel = label
    return this.channel
  }
  async createOffer() {
    return { type: 'offer', sdp: 'v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n' }
  }
  async setLocalDescription(d: { type: string; sdp: string }) {
    this.localDescription = d
    setTimeout(() => {
      this.iceGatheringState = 'complete'
      this.dispatchEvent(new Event('icegatheringstatechange'))
    }, 5)
  }
  async setRemoteDescription(d: { type: string; sdp: string }) {
    this.remote = d
  }
  async getStats() {
    return new Map([
      [
        'a',
        { type: 'inbound-rtp', kind: 'audio', packetsReceived: 50, totalSamplesDuration: 1, totalAudioEnergy: 0.2 },
      ],
    ])
  }
  connect() {
    this.connectionState = 'connected'
    this.iceConnectionState = 'connected'
    this.dispatchEvent(new Event('iceconnectionstatechange'))
    this.dispatchEvent(new Event('connectionstatechange'))
    this.channel.readyState = 'open'
    this.channel.dispatchEvent(new Event('open'))
  }
  receive(event: Record<string, unknown>) {
    this.channel.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(event) }))
  }
  close() {
    this.closed = true
  }
}

const micTrack = { kind: 'audio', stop: vi.fn() }
let requests: { url: string; init?: RequestInit }[] = []

beforeEach(() => {
  requests = []
  FakePeer.last = null
  vi.stubGlobal('RTCPeerConnection', FakePeer)
  vi.stubGlobal('MediaStream', class {})
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined) // jsdom has no media playback
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockReturnValue(undefined)
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia: vi.fn(async () => ({ getAudioTracks: () => [micTrack], getTracks: () => [micTrack] })),
    },
  })
  setServerBase('http://bolo.test/')
  setTransport({
    fetch: async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      requests.push({ url, init })
      if (url.includes('/hangup')) return new Response(null, { status: 200 })
      return new Response('v=0\r\nanswer\r\n', {
        status: 201,
        headers: { 'Content-Type': 'application/sdp', Location: '/v1/realtime/calls/rtc_1' },
      })
    },
  })
})

afterEach(() => {
  setServerBase(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

const flush = () => new Promise((r) => setTimeout(r, 30))

describe('WebRTC conversations', () => {
  it('sends the offer with the session, plays the track, and hangs up', async () => {
    const s = new ConversationSession('cascade', 'webrtc')
    await s.start({ voice: 'amy' })
    const pc = FakePeer.last as FakePeer
    expect(pc.tracks).toEqual([micTrack])
    expect(pc.channelLabel).toBe('oai-events')
    expect(requests[0]?.url).toBe('http://bolo.test/v1/realtime/calls?model=bolo')
    const form = requests[0]?.init?.body as FormData
    expect(form.get('sdp')).toContain('m=audio')
    const session = JSON.parse(String(form.get('session'))) as {
      pipeline: string
      audio: { input: Record<string, unknown>; output: Record<string, unknown> }
    }
    expect(session.pipeline).toBe('cascade')
    expect(session.audio.input).toEqual({ turn_detection: { type: 'server_vad' } }) // no PCM formats over WebRTC
    expect(session.audio.output).toEqual({ voice: 'amy' })
    expect(pc.remote?.type).toBe('answer')
    expect(s.store.get().status).toBe('starting')

    pc.connect()
    expect(s.store.get()).toMatchObject({
      status: 'live',
      locked: true,
      link: { connection: 'connected', ice: 'connected', channel: 'open' },
    })
    pc.receive({ type: 'input_audio_buffer.speech_stopped' })
    pc.receive({ type: 'conversation.item.input_audio_transcription.completed', item_id: 'i1', transcript: 'Hi.' })
    pc.receive({ type: 'response.output_audio_transcript.delta', response_id: 'r1', delta: 'Hello' })
    pc.receive({ type: 'output_audio_buffer.started', response_id: 'r1' })
    const live = s.store.get()
    expect(live.turns.map((t) => t.text)).toEqual(['Hi.', 'Hello'])
    expect(live.agentSpeaking).toBe(true)
    expect(live.turns[1]?.firstAudioMs).toBeGreaterThanOrEqual(0)

    await s.stop()
    expect(requests.at(-1)?.url).toBe('http://bolo.test/v1/realtime/calls/rtc_1/hangup')
    expect(pc.closed).toBe(true)
    expect(micTrack.stop).toHaveBeenCalled()
    expect(s.store.get().remoteAudio).toEqual({ packets: 50, seconds: 1, heard: true })
  })

  it('reports a server without the webrtc extra', async () => {
    setTransport({
      fetch: async () =>
        new Response(
          JSON.stringify({ error: { message: 'the WebRTC transport needs aiortc', code: 'engine_unavailable' } }),
          { status: 503 },
        ),
    })
    const s = new ConversationSession('cascade', 'webrtc')
    await s.start()
    await flush()
    expect(s.store.get()).toMatchObject({ status: 'error', errorCode: 'webrtc_unavailable' })
    expect(s.store.get().error).toContain('aiortc')
  })

  it('reports a refused microphone as a microphone failure', async () => {
    const denied = Object.assign(new Error('Permission denied'), { name: 'NotAllowedError' })
    navigator.mediaDevices.getUserMedia = vi.fn(async () => Promise.reject(denied))
    const s = new ConversationSession('cascade', 'webrtc')
    await s.start()
    expect(s.store.get().status).toBe('error')
    expect(s.store.get().mic?.kind).toBe('denied')
    expect(FakePeer.last).toBeNull()
  })

  it('reports a connection that fails', async () => {
    const s = new ConversationSession('cascade', 'webrtc')
    await s.start()
    const pc = FakePeer.last as FakePeer
    pc.connectionState = 'failed'
    pc.dispatchEvent(new Event('connectionstatechange'))
    expect(s.store.get()).toMatchObject({ status: 'error', errorCode: 'webrtc_failed' })
  })
})

describe('WebRTC helpers', () => {
  it('does not wait for ever for ICE candidates', async () => {
    const pc = new FakePeer() as unknown as RTCPeerConnection
    const t0 = performance.now()
    await iceGathered(pc, 20)
    expect(performance.now() - t0).toBeGreaterThanOrEqual(15)
  })

  it('reads refusals', async () => {
    expect(await callRefused(new Response('{"error":{"message":"bad key","code":"x"}}', { status: 401 }))).toEqual({
      message: 'The server refused the call (401): bad key',
      code: 'unauthorized',
    })
    expect((await callRefused(new Response('plain', { status: 400 }))).message).toContain('plain')
  })

  it('folds the output_audio_buffer events and keeps formats for the WebSocket', () => {
    let s = applyEvent(emptyConversation(), { type: 'output_audio_buffer.started' })
    expect(s.events).toEqual(['output_audio_buffer.started']) // the indicator is ./speaking.ts's
    s = applyEvent(s, { type: 'output_audio_buffer.cleared' })
    expect(s.agentSpeaking).toBe(false)
    const ws = sessionUpdate('cascade').session as { audio: { input: { format?: unknown } } }
    expect(ws.audio.input.format).toEqual({ type: 'audio/pcm', rate: 24000 })
  })
})
