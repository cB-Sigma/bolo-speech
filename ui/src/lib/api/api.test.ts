import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError, NetworkError } from './client'
import { apiUrl, pageBase, setApiKey, setServerBase, setTransport, wsUrl } from './config'
import { openEventStream, SseParser } from './sse'
import type { ServerEvent } from './types'

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })

describe('URL building', () => {
  afterEach(() => {
    setServerBase(null)
    setApiKey(null)
  })
  it('derives the base from where the page is mounted', () => {
    expect(pageBase({ origin: 'http://h:8000', pathname: '/' })).toBe('http://h:8000/')
    expect(pageBase({ origin: 'http://h', pathname: '/bolo/' })).toBe('http://h/bolo/')
    expect(pageBase({ origin: 'http://h', pathname: '/bolo/index.html' })).toBe('http://h/bolo/')
  })
  it('resolves API paths under a sub-path and encodes the query', () => {
    setServerBase('https://example.test/speech')
    expect(apiUrl('api/v1/info')).toBe('https://example.test/speech/api/v1/info')
    expect(apiUrl('/api/v1/catalog', { task: 'stt', q: 'a b', installed: true, skip: undefined, empty: '' })).toBe(
      'https://example.test/speech/api/v1/catalog?task=stt&q=a+b&installed=true',
    )
  })
  it('builds websocket URLs with the right scheme and the API key', () => {
    setServerBase('https://example.test/')
    expect(wsUrl('api/v1/stt/stream', { model: 'm' })).toBe('wss://example.test/api/v1/stt/stream?model=m')
    setServerBase('http://127.0.0.1:8000/')
    setApiKey('secret')
    expect(wsUrl('api/v1/tts/stream', { model: 'm' })).toBe(
      'ws://127.0.0.1:8000/api/v1/tts/stream?model=m&api_key=secret',
    )
  })
})

describe('api client', () => {
  const fetchMock = vi.fn<typeof fetch>()
  beforeEach(() => {
    fetchMock.mockReset()
    setServerBase('http://bolo.test/')
    setTransport({ fetch: fetchMock })
  })
  afterEach(() => {
    setServerBase(null)
    setApiKey(null)
  })

  it('GETs JSON', async () => {
    fetchMock.mockResolvedValue(json({ name: 'bolo', version: '0.1.0' }))
    const info = await api.info()
    expect(info.version).toBe('0.1.0')
    expect(fetchMock.mock.calls[0]?.[0]).toBe('http://bolo.test/api/v1/info')
    expect((fetchMock.mock.calls[0]?.[1] as RequestInit).method).toBe('GET')
  })
  it('POSTs JSON bodies and encodes ids in paths', async () => {
    fetchMock.mockImplementation(async () => json({ job_id: 'j_1' }, 202))
    expect(await api.pull('moonshine-tiny-en')).toEqual({ job_id: 'j_1' })
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit
    expect(init.method).toBe('POST')
    expect(init.body).toBe('{"id":"moonshine-tiny-en"}')
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json')
    await api.load('hf:owner/repo')
    expect(fetchMock.mock.calls[1]?.[0]).toBe('http://bolo.test/api/v1/models/hf%3Aowner%2Frepo/load')
  })
  it('handles 204 and sends DELETE', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }))
    await expect(api.remove('m')).resolves.toBeUndefined()
    expect((fetchMock.mock.calls[0]?.[1] as RequestInit).method).toBe('DELETE')
  })
  it('sends the API key as a bearer token', async () => {
    setApiKey('k-123')
    fetchMock.mockResolvedValue(json({ models: [] }))
    await api.catalog({ task: 'tts' })
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer k-123')
    expect(fetchMock.mock.calls[0]?.[0]).toBe('http://bolo.test/api/v1/catalog?task=tts')
  })
  it('turns HTTP errors into ApiError with the server message', async () => {
    fetchMock.mockImplementation(async () => json({ detail: 'model is gated' }, 403))
    const err = await api.pull('x').catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(403)
    expect((err as ApiError).message).toBe('model is gated')
    expect((err as ApiError).unauthorized).toBe(true)
  })
  it('reads OpenAI-style error envelopes', async () => {
    fetchMock.mockResolvedValue(json({ error: { message: 'queue full', type: 'rate_limit' } }, 429))
    await expect(api.stats()).rejects.toThrow('queue full')
  })
  it('falls back to the status line for non-JSON errors', async () => {
    fetchMock.mockResolvedValue(new Response('<html>', { status: 502, statusText: 'Bad Gateway' }))
    await expect(api.stats()).rejects.toThrow('502 Bad Gateway')
  })
  it('reports an unreachable server as NetworkError', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'))
    await expect(api.info()).rejects.toBeInstanceOf(NetworkError)
  })
  it('lets aborts through untouched', async () => {
    fetchMock.mockRejectedValue(new DOMException('Aborted', 'AbortError'))
    await expect(api.info()).rejects.toMatchObject({ name: 'AbortError' })
  })
  it('measures a ping', async () => {
    fetchMock.mockResolvedValue(json({ status: 'ok' }))
    expect(await api.ping()).toBeGreaterThanOrEqual(0)
    expect(fetchMock.mock.calls[0]?.[0]).toBe('http://bolo.test/healthz')
  })
  it('uploads multipart for transcription', async () => {
    fetchMock.mockResolvedValue(json({ text: 'hello' }))
    expect((await api.transcribe(new Blob(['x']), 'whisper-base', 'en')).text).toBe('hello')
    const body = (fetchMock.mock.calls[0]?.[1] as RequestInit).body as FormData
    expect(body.get('model')).toBe('whisper-base')
    expect(body.get('language')).toBe('en')
    expect(body.get('file')).toBeInstanceOf(Blob)
  })
})

describe('SseParser', () => {
  it('assembles events across arbitrary chunk boundaries', () => {
    const p = new SseParser()
    expect(p.push('data: {"a"')).toEqual([])
    expect(p.push(':1}\n')).toEqual([])
    expect(p.push('\ndata: {"b":2}\n\n')).toEqual(['{"a":1}', '{"b":2}'])
  })
  it('ignores comments and other fields, joins multi-line data, accepts CRLF', () => {
    const p = new SseParser()
    expect(p.push(': keep-alive\n\nevent: x\nid: 7\ndata: one\ndata: two\n\n')).toEqual(['one\ntwo'])
    expect(p.push('data:tight\r\n\r\n')).toEqual(['tight'])
  })
})

describe('openEventStream', () => {
  const enc = new TextEncoder()
  function streamOf(chunks: string[]): Response {
    return new Response(
      new ReadableStream<Uint8Array>({
        start(c) {
          for (const ch of chunks) c.enqueue(enc.encode(ch))
          c.close()
        },
      }),
      { headers: { 'Content-Type': 'text/event-stream' } },
    )
  }
  beforeEach(() => setServerBase('http://bolo.test/'))
  afterEach(() => setServerBase(null))

  it('delivers parsed events, then reconnects when the stream ends', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockImplementation(async () =>
        streamOf(['data: {"type":"log","level":"info","message":"hi"}\n\n', 'data: not json\n\n']),
      )
    setTransport({ fetch: fetchMock })
    const events: ServerEvent[] = []
    const statuses: string[] = []
    const stop = openEventStream({
      onEvent: (e) => events.push(e),
      onStatus: (s) => statuses.push(s),
      minDelay: 5,
      maxDelay: 10,
    })
    await vi.waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(2))
    stop()
    expect(events[0]).toEqual({ type: 'log', level: 'info', message: 'hi' })
    expect(statuses).toContain('open')
    expect(statuses).toContain('reconnecting')
    expect(statuses[statuses.length - 1]).toBe('closed')
    expect((fetchMock.mock.calls[0]?.[1] as RequestInit).headers).toMatchObject({ Accept: 'text/event-stream' })
  })
  it('stops and reports when the key is rejected', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response('', { status: 401 }))
    setTransport({ fetch: fetchMock })
    const statuses: string[] = []
    const stop = openEventStream({ onEvent: () => undefined, onStatus: (s) => statuses.push(s), minDelay: 5 })
    await vi.waitFor(() => expect(statuses).toContain('unauthorized'))
    await new Promise((r) => setTimeout(r, 30))
    expect(fetchMock).toHaveBeenCalledTimes(1)
    stop()
  })
})
