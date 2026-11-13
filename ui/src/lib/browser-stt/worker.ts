/// <reference lib="webworker" />
import { env, pipeline, Tensor } from '@huggingface/transformers'
import type { FetchRecord, FileProgress, LoadStage, WorkerRequest, WorkerResponse } from './models'

type AsrOptions = Record<string, unknown>
interface Asr {
  (audio: Float32Array, options?: AsrOptions): Promise<{ text: string } | { text: string }[]>
  processor: (audio: Float32Array) => Promise<{ input_features: unknown }>
  model: {
    forward(inputs: Record<string, unknown>): Promise<{ logits: { data: Float32Array; dims: number[] } }>
    generation_config?: { lang_to_id?: Record<string, number>; decoder_start_token_id?: number }
    config?: { decoder_start_token_id?: number }
  }
}

const scope = self as unknown as DedicatedWorkerGlobalScope
const post = (msg: WorkerResponse) => scope.postMessage(msg)

let asr: Asr | null = null
let family: 'Moonshine' | 'Whisper' = 'Moonshine'
let chain: Promise<void> = Promise.resolve()

const stage = (s: LoadStage, loaded?: number, total?: number) => post({ type: 'stage', stage: s, loaded, total })

/**
 * transformers.js keeps the ONNX Runtime binary in Cache Storage and, the
 * first time, downloads it without reporting anything: on a slow connection
 * that is a long silent wait. This does that first download itself, with
 * progress, into the same cache entry. The library then finds the file there
 * and carries on exactly as it does on any later visit.
 *
 * Returns quietly when there is nothing to do or the cache cannot be used; the
 * library then fetches the file its own way.
 */
async function prefetchRuntime(url: string): Promise<void> {
  if (typeof caches === 'undefined') return
  const cacheName = (env as unknown as { cacheKey?: string }).cacheKey ?? 'transformers-cache'
  const cache = await caches.open(cacheName)
  if (await cache.match(url)) return
  const started = performance.now()
  const record: FetchRecord = { url, status: 0, bytes: null, ms: 0 }
  try {
    const res = await fetch(url)
    record.status = res.status
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const total = Number(res.headers.get('content-length')) || 0
    stage('runtime', 0, total)
    const reader = res.body?.getReader()
    if (!reader) return
    const chunks: Uint8Array[] = []
    let loaded = 0
    let last = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      loaded += value.byteLength
      const now = performance.now()
      if (now - last > 100) {
        last = now
        stage('runtime', loaded, total)
      }
    }
    record.bytes = loaded
    stage('runtime', loaded, total || loaded)
    await cache.put(
      url,
      new Response(new Blob(chunks as BlobPart[], { type: 'application/wasm' }), {
        status: 200,
        headers: { 'content-type': 'application/wasm', 'content-length': String(loaded) },
      }),
    )
  } catch (err) {
    record.error = err instanceof Error ? err.message : String(err)
  } finally {
    record.ms = Math.round(performance.now() - started)
    post({ type: 'fetch', record })
  }
}

/** Reports every model-file request (URL, status, size, time to headers) for the diagnostics panel. */
function instrumentFetch(): void {
  const e = env as unknown as { fetch?: (input: string | URL, init?: unknown) => Promise<Response> }
  const base = e.fetch ?? ((input: string | URL, init?: unknown) => fetch(input, init as RequestInit))
  e.fetch = async (input, init) => {
    const url = String(input)
    const started = performance.now()
    try {
      const res = await base(input, init)
      const length = Number(res.headers.get('content-length'))
      post({
        type: 'fetch',
        record: {
          url,
          status: res.status,
          bytes: Number.isFinite(length) && length > 0 ? length : null,
          ms: Math.round(performance.now() - started),
        },
      })
      return res
    } catch (err) {
      post({
        type: 'fetch',
        record: {
          url,
          status: 0,
          bytes: null,
          ms: Math.round(performance.now() - started),
          error: err instanceof Error ? err.message : String(err),
        },
      })
      throw err
    }
  }
}

async function load(req: Extract<WorkerRequest, { type: 'load' }>): Promise<void> {
  const started = performance.now()
  family = req.family
  env.allowLocalModels = false
  if (req.config.modelHost) env.remoteHost = req.config.modelHost.replace(/\/?$/, '/')
  instrumentFetch()
  const wasm = env.backends.onnx.wasm as
    { numThreads?: number; wasmPaths?: unknown; wasmBinary?: ArrayBuffer } | undefined
  if (wasm) {
    wasm.numThreads = req.threads
    const current = wasm.wasmPaths as { mjs: string; wasm: string } | string | undefined
    const cdnBase = typeof current === 'object' && current ? current.mjs.replace(/[^/]*$/, '') : undefined
    const base = req.config.ortWasmBase?.replace(/\/?$/, '/') ?? cdnBase
    if (base) {
      // The plain build is half the size of the asyncify one and is all the wasm backend needs.
      const variant = req.device === 'wasm' ? 'ort-wasm-simd-threaded' : 'ort-wasm-simd-threaded.asyncify'
      const paths = { mjs: `${base}${variant}.mjs`, wasm: `${base}${variant}.wasm` }
      wasm.wasmPaths = paths
      stage('runtime')
      // Best effort: on any failure the library's own loader takes over and reports the real error.
      if (!req.plainRuntimeLoad)
        await prefetchRuntime(new URL(paths.wasm, scope.location.href).href).catch(() => undefined)
    }
  }

  const files = new Map<string, FileProgress>()
  let lastPost = 0
  let compiling = false
  const progress = (info: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (!info.file || info.status === 'progress_total') return
    const prev = files.get(info.file)
    if (info.status === 'progress') {
      files.set(info.file, { file: info.file, loaded: info.loaded ?? 0, total: info.total ?? prev?.total ?? 0 })
    } else if (info.status === 'done' && prev) {
      files.set(info.file, { ...prev, loaded: prev.total })
    } else if (!prev) {
      files.set(info.file, { file: info.file, loaded: 0, total: 0 })
    }
    const now = performance.now()
    if (now - lastPost > 80 || info.status === 'done') {
      lastPost = now
      const list = [...files.values()]
      post({ type: 'progress', files: list })
      // Both graphs (encoder and decoder) fetched: what remains is building the sessions.
      const graphs = list.filter((f) => f.file.endsWith('.onnx'))
      const complete = list.every((f) => f.total > 0 && f.loaded >= f.total)
      if (!compiling && graphs.length >= 2 && complete) {
        compiling = true
        stage('compiling')
      } else if (!compiling) {
        stage('model')
      }
    }
  }

  stage('model')
  const create = pipeline as unknown as (task: string, model: string, options: Record<string, unknown>) => Promise<Asr>
  asr = await create('automatic-speech-recognition', req.repo, {
    device: req.device,
    dtype: req.dtype,
    progress_callback: progress,
  })
  // Warm up so the first real utterance does not pay for session initialisation.
  stage('warmup')
  await asr(
    new Float32Array(16000),
    family === 'Whisper' ? { language: 'en', task: 'transcribe', max_new_tokens: 4 } : {},
  )
  post({ type: 'ready', device: req.device, loadMs: performance.now() - started })
}

/**
 * Whisper's own language guess: one decoder step after the start token, read
 * off the language tokens. transformers.js does not do this itself.
 */
async function detectLanguage(audio: Float32Array): Promise<string | null> {
  if (!asr) return null
  const langs = asr.model.generation_config?.lang_to_id
  const sot = asr.model.generation_config?.decoder_start_token_id ?? asr.model.config?.decoder_start_token_id
  if (!langs || sot == null) return null
  const { input_features } = await asr.processor(audio)
  const out = await asr.model.forward({
    input_features,
    decoder_input_ids: new Tensor('int64', BigInt64Array.from([BigInt(sot)]), [1, 1]),
  })
  const vocab = out.logits.dims[out.logits.dims.length - 1] as number
  const row = out.logits.data.subarray(out.logits.data.length - vocab)
  let best: string | null = null
  let bestScore = -Infinity
  for (const [token, id] of Object.entries(langs)) {
    const score = row[id]
    if (score != null && score > bestScore) {
      bestScore = score
      best = token
    }
  }
  return best ? best.replace(/[<|>]/g, '') : null
}

async function transcribe(req: Extract<WorkerRequest, { type: 'transcribe' }>): Promise<void> {
  if (!asr) throw new Error('Model is not loaded')
  const started = performance.now()
  const seconds = req.audio.length / 16000
  let language: string | null
  let options: AsrOptions
  if (family === 'Whisper') {
    language = req.language && req.language !== 'auto' ? req.language : await detectLanguage(req.audio)
    // Indic scripts take several tokens per word; the cap only stops run-on repetition.
    options = {
      language: language ?? 'en',
      task: 'transcribe',
      max_new_tokens: Math.min(220, Math.ceil(seconds * 16) + 12),
    }
  } else {
    language = 'en'
    // Moonshine has no length penalty; cap tokens relative to the audio to avoid run-on hallucination.
    options = { max_new_tokens: Math.max(8, Math.ceil(seconds * 6.5)) }
  }
  const out = await asr(req.audio, options)
  const text = (Array.isArray(out) ? out.map((o) => o.text).join(' ') : out.text).trim()
  post({ type: 'result', id: req.id, text, ms: performance.now() - started, language })
}

scope.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const req = e.data
  chain = chain
    .then(() => (req.type === 'load' ? load(req) : transcribe(req)))
    .catch((err: unknown) => {
      post({
        type: 'error',
        id: req.type === 'transcribe' ? req.id : undefined,
        message: err instanceof Error ? err.message : String(err),
      })
    })
}
