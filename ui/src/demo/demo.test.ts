import { BROWSER_MODELS } from '@/lib/browser-stt/models'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import type { Catalog, RequestTrace, ServerInfo, Stats } from '@/lib/api/types'
import { clearBrowserTraces, recordBrowserTrace } from '@/lib/browser-stt/traces'
import { BOLO_VERSION } from '@/lib/version'
import { DemoBackend, isApiUrl } from './backend'

let backend: DemoBackend
const get = async <T>(path: string, init?: RequestInit) => {
  const res = await backend.handle(`https://cb-sigma.github.io/bolo-speech/${path}`, init)
  return { status: res.status, body: (res.status === 204 ? null : await res.json()) as T }
}

beforeEach(() => {
  clearBrowserTraces()
  backend = new DemoBackend()
})
afterEach(clearBrowserTraces)

describe('demo backend', () => {
  it('recognises API URLs under the Pages sub-path', () => {
    expect(isApiUrl('https://cb-sigma.github.io/bolo-speech/api/v1/info')).toBe(true)
    expect(isApiUrl('https://cb-sigma.github.io/bolo-speech/healthz')).toBe(true)
    expect(isApiUrl('https://cb-sigma.github.io/bolo-speech/assets/index.js')).toBe(false)
    expect(isApiUrl('https://huggingface.co/api/v1x')).toBe(false)
  })

  it('describes itself as a browser, not a server', async () => {
    const { body } = await get<ServerInfo>('api/v1/info')
    expect(body.device).toBe('browser')
    expect(body.version).toMatch(/demo$/)
    // The footer shows this: the version of the bolo release, from the server package, not a fixed string.
    expect(BOLO_VERSION).toMatch(/^\d+\.\d+\.\d+/)
    expect(body.version).toBe(`${BOLO_VERSION} demo`)
    expect(body.auth_required).toBe(false)
  })

  it('rates nothing for "this machine": the machine is the visitor\'s browser', async () => {
    const { body: info } = await get<ServerInfo>('api/v1/info')
    expect(info.capabilities?.recommendations).toBeUndefined()
    const { body } = await get<Catalog>('api/v1/catalog')
    expect(body.models.some((m) => m.fit != null)).toBe(false)
  })

  it('lists the browser model first and the server catalog as a read-only snapshot', async () => {
    const { body } = await get<Catalog>('api/v1/catalog')
    const browser = body.models.filter((m) => m.runtime === 'browser')
    const server = body.models.filter((m) => m.runtime === 'server')
    expect(browser.map((m) => m.id)).toEqual(BROWSER_MODELS.map((m) => m.id))
    expect(browser[0]).toMatchObject({ license: 'MIT', engine: 'onnxruntime-web', task: 'stt' })
    expect(server.length).toBeGreaterThan(50)
    expect(server.every((m) => !m.installed && !m.loaded && m.pulling === null)).toBe(true)
    const stt = await get<Catalog>('api/v1/catalog?task=tts')
    expect(stt.body.models.every((m) => m.task === 'tts')).toBe(true)
  })

  it('starts with no traces and no statistics: nothing is invented', async () => {
    const stats = (await get<Stats>('api/v1/stats')).body
    expect(stats.totals).toEqual({ requests: 0, errors: 0, audio_seconds_in: 0, audio_seconds_out: 0 })
    expect(stats.models).toEqual([])
    expect((await get<{ requests: RequestTrace[] }>('api/v1/requests')).body.requests).toEqual([])
  })

  it('serves exactly the traces the recogniser recorded', async () => {
    const t = recordBrowserTrace({
      model: 'moonshine-tiny',
      startedAtPerf: performance.now() - 900,
      phases: [
        { name: 'endpoint', ms: 500 },
        { name: 'queue', ms: 0 },
        { name: 'decode', ms: 320.26 },
      ],
      audioSeconds: 2.4,
      firstPartialMs: 210.7,
      finalMs: 820.3,
      rtf: 0.13345,
      text: 'hello there',
    })
    expect(t.phases).toEqual([
      { name: 'endpoint', start_ms: 0, dur_ms: 500 },
      { name: 'decode', start_ms: 500, dur_ms: 320.3 },
    ])
    expect(t).toMatchObject({
      transport: 'browser',
      first_partial_ms: 211,
      final_ms: 820,
      total_ms: 820,
      rtf: 0.133,
      text_preview: 'hello there',
    })
    const { body } = await get<{ requests: RequestTrace[] }>('api/v1/requests')
    expect(body.requests).toEqual([t])
    const stats = (await get<Stats>('api/v1/stats')).body
    expect(stats.totals.requests).toBe(1)
    expect(stats.totals.audio_seconds_in).toBeCloseTo(2.4)
  })

  it('refuses server-only operations with an explanation and the install command', async () => {
    for (const [path, init] of [
      ['api/v1/models/pull', { method: 'POST', body: JSON.stringify({ id: 'whisper-base' }) }],
      ['api/v1/models/whisper-base/load', { method: 'POST' }],
      ['api/v1/models/whisper-base', { method: 'DELETE' }],
      [
        'api/v1/bench',
        {
          method: 'POST',
          body: JSON.stringify({ kind: 'tts', model: 'kokoro', concurrency: [1], duration_s: 1, sample: 'builtin:x' }),
        },
      ],
      ['v1/audio/speech', { method: 'POST', body: '{}' }],
    ] as const) {
      const { status, body } = await get<{ detail: string }>(path, init)
      expect(status, path).toBe(501)
      expect(body.detail).toMatch(/needs a bolo server/i)
      expect(body.detail).toContain('uv tool install')
    }
  })

  it('offers real audio samples and streams events starting with stats', async () => {
    const { body } = await get<{ samples: { id: string; kind: string; url: string }[] }>('api/v1/samples')
    expect(body.samples.every((s) => s.kind === 'audio' && s.url.endsWith('.wav'))).toBe(true)
    const controller = new AbortController()
    const res = await backend.handle('https://x/bolo/api/v1/events', { signal: controller.signal })
    const first = new TextDecoder().decode((await res.body!.getReader().read()).value)
    expect(JSON.parse(first.replace(/^data: /, '')).type).toBe('stats')
    controller.abort()
  })
})
