import type { BenchRequest, CatalogModel, ProviderUpdate, SetupUse } from '@/lib/api/types'
import { encodeWav } from '@/lib/audio/wav'
import { SAMPLE_SEGMENTS, SAMPLES } from './catalog'
import { HttpError, MOCK_KEY, sleep, type MockServer } from './engine'
import { babble, splitChunks } from './speech'
import longUrl from './assets/en-ask-not.wav?url'
import shortUrl from './assets/en-fellow-americans.wav?url'

const SAMPLE_FILES: Record<string, string> = { 'en-ask-not': longUrl, 'en-fellow-americans': shortUrl }

const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })

function pathOf(url: string): { path: string; query: URLSearchParams } {
  const u = new URL(url, 'http://mock.local/')
  const i = u.pathname.search(/\/(api\/v1|v1|healthz|readyz|metrics)(\/|$)/)
  return { path: i >= 0 ? u.pathname.slice(i) : u.pathname, query: u.searchParams }
}

export function isApiUrl(url: string): boolean {
  return /\/(api\/v1|v1\/(models|audio)|healthz|readyz|metrics)(\/|\?|$)/.test(
    new URL(url, 'http://mock.local/').pathname,
  )
}

function sseStream(server: MockServer, signal?: AbortSignal | null): Response {
  const enc = new TextEncoder()
  let unsub: (() => void) | undefined
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const send = (ev: unknown) => {
        try {
          controller.enqueue(enc.encode(`data: ${JSON.stringify(ev)}\n\n`))
        } catch {
          unsub?.()
        }
      }
      send({ type: 'stats', ...server.stats() })
      unsub = server.subscribe(send)
      signal?.addEventListener('abort', () => {
        unsub?.()
        try {
          controller.close()
        } catch {
          // already closed
        }
      })
    },
    cancel() {
      unsub?.()
    },
  })
  return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } })
}

function speechResponse(server: MockServer, model: CatalogModel, input: string): Response {
  const rate = model.sample_rate
  const chunks = splitChunks(input)
  const pcm = chunks.map((c, i) => babble(c, rate, i + 1))
  const total = pcm.reduce((a, b) => a + b.length, 0)
  const all = new Float32Array(total)
  let o = 0
  for (const p of pcm) {
    all.set(p, o)
    o += p.length
  }
  const wav = encodeWav(all, rate)
  const trace = server.cloudTrace(
    model,
    server.makeTrace(model, Date.now(), { chars: input.length, text: input.slice(0, 48), transport: 'http' }),
  )
  server.record(trace)
  let offset = 0
  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (offset >= wav.length) {
        controller.close()
        return
      }
      await sleep(offset === 0 ? 120 : 60)
      const end = Math.min(wav.length, offset + 16384)
      controller.enqueue(wav.slice(offset, end))
      offset = end
    },
  })
  return new Response(body, {
    headers: { 'Content-Type': 'audio/wav', 'x-request-id': trace.id, 'x-bolo-model': model.id },
  })
}

/** Answers API requests the way the real server would. Returns null for non-API URLs. */
/** Seconds from a keep_alive value ("10m", 300, -1), null when it cannot be read (the server answers 400). */
export function mockKeepAlive(raw: unknown): number | null {
  if (typeof raw === 'number') return raw < 0 ? -1 : raw
  const text = String(raw).trim().toLowerCase()
  if (/^-?\d+(\.\d+)?$/.test(text)) return Number(text) < 0 ? -1 : Number(text)
  const m = /^(-?)(\d+(?:\.\d+)?)(ms|s|m|h)$/.exec(text)
  if (!m) return null
  const unit = { ms: 0.001, s: 1, m: 60, h: 3600 }[m[3] as 'ms' | 's' | 'm' | 'h']
  return m[1] ? -1 : Number(m[2]) * unit
}

export async function handleRequest(
  server: MockServer,
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
  const method = (init?.method ?? 'GET').toUpperCase()
  const { path, query } = pathOf(url)
  if (server.flags.offline) throw new TypeError('Failed to fetch')
  await sleep(12 + Math.random() * 25) // loopback-ish latency
  if (init?.signal?.aborted) throw new DOMException('Aborted', 'AbortError')

  const open = path === '/healthz' || path === '/readyz' || path === '/api/v1/info'
  if (server.flags.auth && !open) {
    const headers = new Headers(init?.headers)
    if (headers.get('Authorization') !== `Bearer ${MOCK_KEY}`)
      return json({ detail: 'Invalid or missing API key' }, 401)
  }

  try {
    if (path === '/healthz' || path === '/readyz') return json({ status: 'ok' })
    if (path === '/api/v1/info') return json(server.info())
    if (path === '/api/v1/events') return sseStream(server, init?.signal)
    if (path === '/api/v1/stats') return json(server.stats())
    if (path === '/api/v1/catalog') {
      const task = query.get('task')
      const q = query.get('q')?.toLowerCase()
      const installed = query.get('installed')
      const models = server.models.filter(
        (m) =>
          (!task || m.task === task) &&
          (!q || `${m.id} ${m.family} ${m.description}`.toLowerCase().includes(q)) &&
          (installed == null || String(m.installed) === installed),
      )
      return json({ models })
    }
    const oneTrace = /^\/api\/v1\/requests\/([\w-]+)$/.exec(path)
    if (oneTrace) {
      const t = server.traces.find((x) => x.id === oneTrace[1])
      return t ? json(t) : json({ detail: 'Unknown request' }, 404)
    }
    if (path.startsWith('/api/v1/telephony')) {
      const phone = server.telephony
      const body = () => JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>
      const rest = decodeURIComponent(path.slice('/api/v1/telephony'.length))
      const agent = /^\/agents\/([\w-]+)$/.exec(rest)
      const number = /^\/numbers\/(.+)$/.exec(rest)
      const call = /^\/calls\/([\w-]+)$/.exec(rest)
      try {
        if (rest === '') return json(phone.overview())
        if (agent && method === 'PUT') return json(phone.putAgent(agent[1] as string, body()))
        if (agent && method === 'DELETE') {
          phone.deleteAgent(agent[1] as string)
          return new Response(null, { status: 204 })
        }
        if (number && method === 'PUT') return json(phone.putNumber(number[1] as string, String(body().agent)))
        if (number && method === 'DELETE') {
          phone.deleteNumber(number[1] as string)
          return new Response(null, { status: 204 })
        }
        if (rest === '/calls') return json({ calls: phone.summaries() })
        if (call) {
          const found = phone.calls.find((c) => c.id === call[1])
          return found ? json(found) : json({ detail: 'Unknown call' }, 404)
        }
        if (rest === '/scenarios')
          return json({
            scenarios: [
              { name: 'basic', description: 'The caller says one thing and hangs up.' },
              { name: 'barge-in', description: "The caller interrupts the agent's answer." },
            ],
          })
        if (rest === '/simulate' && method === 'POST') {
          await sleep(200)
          return json(phone.simulate(body()))
        }
      } catch (err) {
        throw new HttpError(400, err instanceof Error ? err.message : String(err), 'invalid_request')
      }
    }
    if (path === '/api/v1/providers') return json(server.providers())
    const providerTest = /^\/api\/v1\/providers\/([\w-]+)\/test$/.exec(path)
    if (providerTest && method === 'POST') {
      await sleep(150)
      return json(server.testProvider(providerTest[1] as string))
    }
    const provider = /^\/api\/v1\/providers\/([\w-]+)$/.exec(path)
    if (provider && method === 'PUT') {
      return json(server.putProvider(provider[1] as string, JSON.parse(String(init?.body ?? '{}')) as ProviderUpdate))
    }
    if (provider && method === 'DELETE') {
      server.deleteProvider(provider[1] as string)
      return new Response(null, { status: 204 })
    }
    if (provider) {
      const p = server.providers().providers.find((x) => x.name === provider[1])
      return p ? json(p) : json({ detail: 'Unknown provider' }, 404)
    }
    if (path === '/api/v1/requests') {
      const limit = Number(query.get('limit') ?? 100)
      const kind = query.get('kind')
      const model = query.get('model')
      const requests = server.traces
        .filter((t) => (!kind || t.kind === kind) && (!model || t.model === model))
        .slice(0, limit)
      return json({ requests })
    }
    if (path === '/api/v1/components') return json({ components: server.components() })
    const install = /^\/api\/v1\/components\/([\w-]+)\/install$/.exec(path)
    if (install && method === 'POST') {
      const body = JSON.parse(String(init?.body ?? '{}')) as { accept_license?: boolean }
      return json({ job_id: server.installComponent(install[1] as string, body.accept_license === true).id }, 202)
    }
    const component = /^\/api\/v1\/components\/([\w.-]+)$/.exec(path)
    if (component && method === 'DELETE') {
      server.removeComponent(component[1] as string)
      return new Response(null, { status: 204 })
    }
    if (path === '/api/v1/samples') return json({ samples: SAMPLES })
    const sample = /^\/api\/v1\/samples\/([\w-]+)\.wav$/.exec(path)
    if (sample) {
      const id = sample[1] as string
      const file = SAMPLE_FILES[id]
      if (!file) return json({ detail: 'Unknown sample' }, 404)
      server.hint = { segments: SAMPLE_SEGMENTS[id] ?? [], expires: Date.now() + 60_000 }
      return globalThis.fetch(file)
    }
    if (path === '/api/v1/models/pull' && method === 'POST') {
      const body = JSON.parse(String(init?.body ?? '{}')) as { id: string }
      const job = server.pull(body.id)
      // a model that is unlikely to suit the machine is pulled all the same, with a notice
      const warnings = server.warnings(body.id)
      return json(warnings.length ? { job_id: job.id, warnings } : { job_id: job.id }, 202)
    }
    if (path === '/api/v1/hardware') return json(server.hardware)
    if (path === '/api/v1/hardware/benchmark' && method === 'POST') {
      await sleep(400)
      const hw = server.runBenchmark()
      return json({ benchmark: hw.speed.benchmark, speed: hw.speed })
    }
    if (path === '/api/v1/recommendations/setup') {
      const use = query.get('use') ?? 'live'
      if (!['live', 'files', 'voiceover'].includes(use)) return json({ detail: `unknown use '${use}'` }, 400)
      return json(server.setup(use as SetupUse, query.get('language') || null, query.get('commercial') === 'true'))
    }
    const job = /^\/api\/v1\/jobs\/([\w-]+)$/.exec(path)
    if (job) {
      const j = server.jobs.get(job[1] as string)
      return j ? json(j) : json({ detail: 'Unknown job' }, 404)
    }
    const action = /^\/api\/v1\/models\/([^/]+)\/(load|unload)$/.exec(path)
    if (action && method === 'POST') {
      const id = decodeURIComponent(action[1] as string)
      if (action[2] === 'load') {
        const raw = (JSON.parse(String(init?.body || '{}')) as { keep_alive?: unknown }).keep_alive
        const keepAlive = raw === undefined || raw === null ? undefined : mockKeepAlive(raw)
        if (keepAlive === null) return json({ error: { message: `bad keep_alive ${String(raw)}` } }, 400)
        if (keepAlive === 0) {
          server.unloadModel(id)
          return json({ id, loaded: false })
        }
        await server.loadModel(id, keepAlive)
      } else server.unloadModel(id)
      const warnings = action[2] === 'load' ? server.warnings(id) : []
      const row = action[2] === 'load' ? server.stats().models.find((m) => m.id === id) : undefined
      const placed = row ? { processor: row.processor ?? '100% CPU', device: row.device ?? 'cpu' } : {}
      return json({ id, loaded: action[2] === 'load', ...placed, ...(warnings.length ? { warnings } : {}) })
    }
    if (path === '/api/v1/ps') {
      const stats = server.stats()
      return json({
        models: stats.models.map((m) => ({
          name: m.id,
          model: m.id,
          task: m.task,
          size: m.memory_bytes,
          size_vram: 0,
          processor: m.processor,
          device: m.device,
          expires_in_s: m.expires_in_s,
          expires_at: m.expires_in_s == null ? null : new Date(Date.now() + m.expires_in_s * 1000).toISOString(),
          active: m.active,
          requests: m.requests,
        })),
        system: stats.system,
      })
    }
    const del = /^\/api\/v1\/models\/([^/]+)$/.exec(path)
    if (del && method === 'DELETE') {
      server.removeModel(decodeURIComponent(del[1] as string))
      return new Response(null, { status: 204 })
    }
    if (path === '/api/v1/bench' && method === 'POST') {
      return json({ job_id: server.bench(JSON.parse(String(init?.body ?? '{}')) as BenchRequest).id }, 202)
    }
    if (path === '/v1/models') {
      return json({
        object: 'list',
        data: server.models.filter((m) => m.installed).map((m) => ({ id: m.id, object: 'model', owned_by: 'bolo' })),
      })
    }
    if (path === '/v1/audio/speech' && method === 'POST') {
      const body = JSON.parse(String(init?.body ?? '{}')) as { model: string; input: string }
      const model = server.model(body.model)
      if (!model || model.task !== 'tts') return json({ detail: `Unknown TTS model: ${body.model}` }, 404)
      if (model.available === false)
        throw new HttpError(409, `${model.id} needs eSpeak NG, which is not installed.`, 'component_missing')
      if (model.cloud && !model.installed)
        throw new HttpError(
          503,
          `model '${model.id}' needs the cloud provider '${model.provider?.name}', which is not configured`,
          'provider_not_configured',
        )
      return speechResponse(server, model, body.input)
    }
    if (path === '/v1/audio/transcriptions' && method === 'POST') {
      const form = init?.body instanceof FormData ? init.body : null
      const model = server.model(String(form?.get('model') ?? '')) ?? server.models.find((m) => m.task === 'stt')
      if (model && !model.installed)
        throw new HttpError(
          model.cloud ? 503 : 404,
          model.cloud
            ? `model '${model.id}' needs the cloud provider '${model.provider?.name}', which is not configured`
            : `model '${model.id}' is not installed`,
          model.cloud ? 'provider_not_configured' : 'model_not_installed',
        )
      await sleep(model?.cloud ? 700 : 400)
      const text = server.hint?.segments.map((s) => s.text).join(' ') ?? 'This is a mock transcription.'
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (model) {
        const audioSeconds = server.hint?.segments.at(-1)?.t1 ?? 4
        const trace = server.cloudTrace(
          model,
          server.makeTrace(model, Date.now(), { audioSeconds, text: text.slice(0, 48), transport: 'http' }),
        )
        server.record(trace)
        headers['x-request-id'] = trace.id
        headers['x-bolo-model'] = model.id
        if (trace.cost) headers['x-bolo-estimated-cost'] = `${trace.cost.amount} ${trace.cost.currency}`
      }
      return new Response(JSON.stringify({ text }), { headers })
    }
    if (path === '/metrics') return new Response('# HELP bolo_requests_total Requests\nbolo_requests_total 18230\n')
    return json({ detail: 'Not found' }, 404)
  } catch (err) {
    if (err instanceof HttpError) {
      return json({ detail: err.message, error: { message: err.message, code: err.code } }, err.status)
    }
    throw err
  }
}
