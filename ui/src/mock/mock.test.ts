import { readFileSync } from 'node:fs'
import path from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { api, ApiError } from '@/lib/api/client'
import { setApiKey, setServerBase, setTransport } from '@/lib/api/config'
import { SseParser } from '@/lib/api/sse'
import type { ServerEvent, Stats } from '@/lib/api/types'
import { decodeWav } from '@/lib/audio/wav'
import { ServerTranscriber } from '@/lib/stt/server'
import { transcriptText } from '@/lib/stt/transcriber'
import { TtsSession } from '@/lib/tts/session'
import { SAMPLE_SEGMENTS } from './catalog'
import { MockServer, MOCK_KEY } from './engine'
import { handleRequest } from './http'
import { splitChunks } from './speech'
import { createMockWebSocket } from './ws'

let server: MockServer

function install(flags = {}) {
  server = new MockServer({ quiet: true, ...flags })
  setServerBase('http://bolo.test/')
  setTransport({ fetch: (input, init) => handleRequest(server, input, init), WebSocket: createMockWebSocket(server) })
}

beforeEach(() => install())
afterEach(() => {
  server.dispose()
  setServerBase(null)
  setApiKey(null)
})

describe('mock HTTP API', () => {
  it('serves info, catalog and samples in the contract shape', async () => {
    const info = await api.info()
    expect(info).toMatchObject({ name: 'bolo', auth_required: false })
    const { models } = await api.catalog()
    expect(models.length).toBeGreaterThan(10)
    for (const m of models) {
      expect(['stt', 'tts', 'vad']).toContain(m.task)
      expect(['realtime', 'balanced', 'accuracy']).toContain(m.tier)
      expect(['cpu', 'untested']).toContain(m.validated)
      expect(typeof m.size_bytes).toBe('number')
    }
    expect((await api.catalog({ task: 'tts' })).models.every((m) => m.task === 'tts')).toBe(true)
    expect((await api.catalog({ installed: true })).models.every((m) => m.installed)).toBe(true)
    expect((await api.catalog({ q: 'whisper' })).models.every((m) => m.id.includes('whisper'))).toBe(true)
    const { samples } = await api.samples()
    expect(samples.some((s) => s.kind === 'audio' && s.url)).toBe(true)
    expect(samples.some((s) => s.kind === 'text' && s.language === 'hi')).toBe(true)
  })

  it('reports stats and traces whose phases add up', async () => {
    const stats: Stats = await api.stats()
    expect(stats.models.length).toBeGreaterThan(0)
    expect(stats.models.every((m) => m.loaded)).toBe(true)
    const { requests } = await api.requests({ limit: 50 })
    expect(requests).toHaveLength(50)
    for (const r of requests) {
      const end = Math.max(...r.phases.map((p) => p.start_ms + p.dur_ms))
      expect(Math.abs(end - r.total_ms)).toBeLessThanOrEqual(2)
      if (r.kind === 'tts') expect(r.ttfa_ms).not.toBeNull()
      else expect(r.first_partial_ms).not.toBeNull()
    }
    expect((await api.requests({ kind: 'tts' })).requests.every((r) => r.kind === 'tts')).toBe(true)
  })

  it('pulls a model with progress, then loads, unloads and deletes it', async () => {
    const { job_id } = await api.pull('ten-vad')
    let job = await api.job(job_id)
    expect(job.kind).toBe('pull')
    for (let i = 0; i < 50 && job.state !== 'done'; i++) {
      await new Promise((r) => setTimeout(r, 60))
      job = await api.job(job_id)
    }
    expect(job).toMatchObject({ state: 'done', progress: 1 })
    const find = async () => (await api.catalog()).models.find((m) => m.id === 'ten-vad')
    expect(await find()).toMatchObject({ installed: true, loaded: false, pulling: null })
    await api.load('ten-vad')
    expect((await find())?.loaded).toBe(true)
    await api.unload('ten-vad')
    expect((await find())?.loaded).toBe(false)
    await api.remove('ten-vad')
    expect((await find())?.installed).toBe(false)
  })

  it('refuses gated and unknown models with useful messages', async () => {
    await expect(api.pull('indic-parler-tts')).rejects.toThrow(/gated/)
    const err = await api.load('whisper-small').catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(409)
    await expect(api.pull('nope')).rejects.toMatchObject({ status: 404 })
  })

  it('runs a benchmark job that fills in one point per level', async () => {
    const { job_id } = await api.bench({
      kind: 'stt',
      model: 'moonshine-tiny-en',
      concurrency: [4, 1, 2],
      duration_s: 5,
      sample: 'builtin:en-ask-not',
    })
    let job = await api.benchJob(job_id)
    for (let i = 0; i < 60 && job.state !== 'done'; i++) {
      await new Promise((r) => setTimeout(r, 100))
      job = await api.benchJob(job_id)
    }
    const pts = job.result?.points ?? []
    expect(pts.map((p) => p.concurrency)).toEqual([1, 2, 4])
    expect(pts[2]!.final_ms!.p95).toBeGreaterThan(pts[0]!.final_ms!.p95)
    expect(pts.every((p) => p.throughput_x_realtime > 0 && p.ttfa_ms == null)).toBe(true)
  }, 10_000)

  it('streams server-sent events starting with stats', async () => {
    const controller = new AbortController()
    const res = await handleRequest(server, 'http://bolo.test/api/v1/events', { signal: controller.signal })
    const reader = res.body!.getReader()
    const parser = new SseParser()
    const first = parser.push(new TextDecoder().decode((await reader.read()).value))
    expect((JSON.parse(first[0]!) as ServerEvent).type).toBe('stats')
    server.emit({ type: 'model', id: 'x', event: 'loaded' })
    const next = parser.push(new TextDecoder().decode((await reader.read()).value))
    expect(JSON.parse(next[0]!)).toEqual({ type: 'model', id: 'x', event: 'loaded' })
    controller.abort()
  })

  it('enforces the API key when auth is on', async () => {
    server.dispose()
    install({ auth: true })
    expect((await api.info()).auth_required).toBe(true)
    await expect(api.catalog()).rejects.toMatchObject({ status: 401 })
    setApiKey(MOCK_KEY)
    expect((await api.catalog()).models.length).toBeGreaterThan(0)
  })

  it('looks unreachable when offline', async () => {
    server.flags.offline = true
    await expect(api.info()).rejects.toThrow()
  })
})

describe('mock streaming', () => {
  const wav = decodeWav(readFileSync(path.resolve(import.meta.dirname, 'assets/en-ask-not.wav')))

  it('transcribes a sample over the STT websocket with VAD, partials, finals and a ping', async () => {
    server.hint = { segments: SAMPLE_SEGMENTS['en-ask-not']!, expires: Date.now() + 60_000 }
    const t = new ServerTranscriber({ model: 'moonshine-tiny-en' })
    await t.start()
    expect(t.store.get().status).toBe('streaming')
    let sawPartial = false
    const off = t.store.subscribe(() => {
      if (t.store.get().partial?.text) sawPartial = true
    })
    // About four times real time: fast, but slow enough for partials to be scheduled.
    for (let o = 0; o < wav.samples.length; o += 320) {
      t.push(wav.samples.subarray(o, o + 320))
      if ((o / 320) % 5 === 0) await new Promise((r) => setTimeout(r, 25))
    }
    await t.finish()
    off()
    const s = t.store.get()
    expect(s.status).toBe('done')
    expect(s.finals.length).toBeGreaterThanOrEqual(3)
    expect(transcriptText(s).toLowerCase()).toContain('ask what you can do for your country')
    expect(sawPartial).toBe(true)
    expect(s.speech.every((sp) => sp.t1 != null && sp.t1 > sp.t0)).toBe(true)
    expect(s.finals[0]!.words!.length).toBeGreaterThan(2)
    expect(s.utterances.every((u) => u.finalAt != null && u.speechEndAt != null)).toBe(true)
    expect(s.rtf).toBeGreaterThan(0)
    expect(s.rttMs).toBeGreaterThan(0)
    expect(s.audioSeconds).toBeCloseTo(11, 0)
    expect(server.traces[0]).toMatchObject({ kind: 'stt', transport: 'websocket', model: 'moonshine-tiny-en' })
  }, 20_000)

  it('accepts 8 kHz μ-law telephony audio', async () => {
    const t = new ServerTranscriber({ model: 'moonshine-tiny-en', telephony: true })
    await t.start()
    for (let o = 0; o < wav.samples.length / 3; o += 320) t.push(wav.samples.subarray(o, o + 320))
    await t.finish()
    expect(t.store.get().finals.length).toBeGreaterThanOrEqual(1)
  }, 20_000)

  it('fails cleanly for a model that does not exist', async () => {
    const t = new ServerTranscriber({ model: 'does-not-exist' })
    await expect(t.start()).rejects.toThrow()
    expect(t.store.get().status).toBe('error')
    expect(t.store.get().error).toMatch(/Unknown STT model/)
  })

  it('streams TTS chunks with metadata, timing and audio', async () => {
    const text = 'Thanks for calling. How can I help you today?'
    const s = new TtsSession({ model: 'piper-en-us-lessac', voice: 'lessac', text, silent: true })
    s.start()
    await new Promise<void>((resolve) => {
      const off = s.store.subscribe(() => {
        const st = s.store.get().status
        if (st === 'done' || st === 'error') {
          off()
          resolve()
        }
      })
    })
    const st = s.store.get()
    expect(st.error).toBeNull()
    expect(st.chunks.map((c) => c.text)).toEqual(splitChunks(text))
    expect(st.sampleRate).toBe(22050)
    expect(st.ttfaMs).toBeGreaterThan(0)
    expect(st.serverTtfaMs).toBeGreaterThan(0)
    expect(st.rtf).toBeGreaterThan(0)
    expect(st.serverRtf).toBeGreaterThan(0)
    expect(st.chunks[1]!.arrivedMs).toBeGreaterThan(st.chunks[0]!.arrivedMs)
    expect(st.chunks[1]!.t0).toBeCloseTo(st.chunks[0]!.dur, 1)
    expect(s.audio().length / 22050).toBeCloseTo(st.audioSeconds, 3)
    expect(st.audioSeconds).toBeCloseTo(
      st.chunks.reduce((a, c) => a + c.dur, 0),
      1,
    )
  }, 20_000)

  it('reports a TTS error for a missing model instead of hanging', async () => {
    const s = new TtsSession({ model: 'nope', text: 'hi', silent: true })
    s.start()
    await new Promise((r) => setTimeout(r, 200))
    expect(s.store.get().status).toBe('error')
  })
})
