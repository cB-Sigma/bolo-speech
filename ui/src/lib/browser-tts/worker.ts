/// <reference lib="webworker" />
import type * as Ort from 'onnxruntime-web/wasm'
import {
  cachedDownload,
  lowerMemoryMaximum,
  ortCdnBase,
  RUNTIME_CACHE,
  runtimeMemoryBytes,
} from '@/lib/browser-runtime'
import type { FetchRecord, FileProgress, LoadStage } from '@/lib/browser-stt/models'
import { VOICE_CACHE, type TtsWorkerRequest, type TtsWorkerResponse } from './protocol'
import type * as UccharanModule from '@/vendor/uccharan/index.js'
import { scaleSilence } from './silence'
import { coquiIds, piperConfigFromJson, piperIds, withLeadIn, type VoiceConfig } from './text'

/**
 * Runs one voice with ONNX Runtime Web (WebAssembly). The VITS graph is called
 * directly: no transformers.js pipeline, no phonemiser for character voices,
 * and uccharan (Apache-2.0) for Piper voices.
 */

const scope = self as unknown as DedicatedWorkerGlobalScope
const post = (msg: TtsWorkerResponse, transfer: Transferable[] = []) => scope.postMessage(msg, transfer)
const stage = (s: LoadStage, loaded?: number, total?: number) => post({ type: 'stage', stage: s, loaded, total })

/** One ONNX Runtime session, or two (front on WebGPU, back on the CPU) behind the same surface. */
interface Runner {
  readonly outputNames: readonly string[]
  run(feeds: Record<string, Ort.Tensor>): Promise<Record<string, Ort.OnnxValue>>
  release(): Promise<void>
}

/**
 * Two halves of one graph, cut at build time (the server's bolo/onnx_split.py): the front runs on WebGPU, the
 * back on the CPU, and the tensors that cross the cut come back to the CPU in between.
 */
function twoPart(front: Ort.InferenceSession, back: Ort.InferenceSession): Runner {
  return {
    outputNames: back.outputNames.length ? back.outputNames : front.outputNames,
    async run(feeds) {
      const frontFeeds = Object.fromEntries(front.inputNames.filter((n) => n in feeds).map((n) => [n, feeds[n]]))
      const mid = await front.run(frontFeeds as Record<string, Ort.Tensor>)
      const backFeeds: Record<string, Ort.OnnxValue> = {}
      for (const n of back.inputNames) backFeeds[n] = (mid[n] ?? feeds[n]) as Ort.OnnxValue
      // A graph output the front already makes is passed on as it is.
      return { ...mid, ...(await back.run(backFeeds)) }
    },
    async release() {
      await front.release()
      await back.release()
    },
  }
}

/** The ONNX Runtime build in use: the WebAssembly-only one, or the one with WebGPU (and WebAssembly). */
let ort: typeof Ort
let session: Runner | null = null
let config: VoiceConfig | null = null
let uccharan: typeof UccharanModule | null = null
let kokoroStyle: Float32Array | null = null
let chain: Promise<void> = Promise.resolve()

const record = (r: FetchRecord) => post({ type: 'fetch', record: r })

async function loadRuntime(req: Extract<TtsWorkerRequest, { type: 'load' }>): Promise<void> {
  const gpu = req.device === 'webgpu' || Boolean(req.halves)
  ort = gpu ? ((await import('onnxruntime-web/webgpu')) as unknown as typeof Ort) : await import('onnxruntime-web/wasm')
  const wasm = ort.env.wasm
  wasm.numThreads = req.threads
  wasm.proxy = false
  const base = req.ortWasmBase?.replace(/\/?$/, '/') ?? ortCdnBase(ort.env.versions.web ?? '')
  // The WebGPU build needs the asyncify variant of the WebAssembly runtime.
  const variant = gpu ? 'ort-wasm-simd-threaded.asyncify' : 'ort-wasm-simd-threaded'
  const paths = { mjs: `${base}${variant}.mjs`, wasm: `${base}${variant}.wasm` }
  wasm.wasmPaths = { ...paths }
  stage('runtime')
  // The loader asks for a shared memory of up to 4 GiB; a voice needs far less, and on iPhones a second
  // memory that size does not fit next to the recogniser's. Load it with a lower maximum.
  try {
    const glue = await cachedDownload(
      new URL(paths.mjs, scope.location.href).href,
      RUNTIME_CACHE,
      { record },
      {
        contentType: 'text/javascript',
      },
    )
    const lowered = lowerMemoryMaximum(new TextDecoder().decode(glue.data), req.memoryPages)
    if (lowered) {
      wasm.wasmPaths = { ...paths, mjs: URL.createObjectURL(new Blob([lowered], { type: 'text/javascript' })) }
      post({
        type: 'note',
        text: `runtime memory capped at ${req.memoryPages / 16} MiB (shared, ${req.threads} thread${req.threads === 1 ? '' : 's'})`,
      })
    } else {
      post({ type: 'note', text: 'runtime loader not recognised: memory cap not applied (4 GiB maximum)' })
    }
  } catch (err) {
    post({ type: 'note', text: `memory cap not applied: ${err instanceof Error ? err.message : String(err)}` })
  }
  if (req.plainRuntimeLoad) return
  try {
    // The same file and cache entry the recogniser uses: whichever runs first downloads it for both.
    const { data } = await cachedDownload(
      new URL(paths.wasm, scope.location.href).href,
      RUNTIME_CACHE,
      { progress: (loaded, total) => stage('runtime', loaded, total), record },
      { contentType: 'application/wasm' },
    )
    wasm.wasmBinary = data
  } catch {
    // ONNX Runtime fetches the file itself and reports the real error if this fails too.
  }
}

async function load(req: Extract<TtsWorkerRequest, { type: 'load' }>): Promise<void> {
  const started = performance.now()
  session?.release().catch(() => undefined)
  session = null
  await loadRuntime(req)

  const files = new Map<string, FileProgress>()
  let lastPost = 0
  const track = (file: string) => (loaded: number, total: number) => {
    files.set(file, { file, loaded, total })
    const now = performance.now()
    if (now - lastPost > 80 || (total > 0 && loaded >= total)) {
      lastPost = now
      post({ type: 'progress', files: [...files.values()] })
    }
  }
  stage('model')
  const get = (url: string, name: string, type?: string) =>
    cachedDownload(url, VOICE_CACHE, { progress: track(name), record }, { contentType: type })
  const short = (url: string) =>
    url
      .replace(/[?#].*$/, '')
      .split('/')
      .slice(-2)
      .join('/')
  const pending: Promise<unknown>[] = []
  const model = req.halves ? null : get(req.model, short(req.model))
  const halves = req.halves
    ? Promise.all([get(req.halves.front, short(req.halves.front)), get(req.halves.back, short(req.halves.back))])
    : null
  const settings = req.config ? get(req.config, short(req.config), 'application/json') : null
  const style = req.style ? get(req.style, short(req.style)) : null
  if (req.phonemizer === 'uccharan') {
    pending.push(
      (async () => {
        const mod = await import('@/vendor/uccharan/index.js')
        // Language data is fetched from next to the voices the first time a language is used.
        await mod.init({ dataBaseUrl: req.uccharanBase })
        const lang = req.phonemeLanguage ?? 'en-us'
        await mod.loadLanguage(lang)
        if (req.kind === 'kokoro' && lang.startsWith('en')) await mod.loadKokoro(lang)
        uccharan = mod
      })(),
    )
  }
  const [modelFile, cfgFile, styleFile, halfFiles] = await Promise.all([model, settings, style, halves, ...pending])
  if (req.kind === 'kokoro') {
    if (!styleFile) throw new Error('The voice style is missing')
    kokoroStyle = new Float32Array(styleFile.data)
    config = {
      id: req.voice,
      kind: 'kokoro',
      sample_rate: 24000,
      phoneme_language: req.phonemeLanguage ?? 'en-us',
      inference: { noise_scale: 0, length_scale: 1, noise_scale_w: 0 },
    }
  } else {
    if (!cfgFile) throw new Error('The voice settings are missing')
    const json = JSON.parse(new TextDecoder().decode(cfgFile.data)) as unknown
    config =
      req.configFormat === 'piper'
        ? piperConfigFromJson(req.voice, json as Parameters<typeof piperConfigFromJson>[1], req.phonemeLanguage)
        : (json as VoiceConfig)
    if (req.configFormat === 'piper' || config.kind === 'piper')
      config = {
        ...config,
        phoneme_language: req.phonemeLanguage ?? (config as { phoneme_language: string }).phoneme_language,
      } as VoiceConfig
  }
  stage('compiling')
  const options = (providers: string[]): Ort.InferenceSession.SessionOptions => ({
    executionProviders: providers,
    graphOptimizationLevel: 'all',
    // The 8-bit weights are stored with a DequantizeLinear in front of each. Without this the runtime keeps
    // those nodes (for quantised kernels this graph does not use) and expands the weights on every call;
    // with it they are folded into constants once, at load, and synthesis is as fast as with the fp32 file.
    extra: { session: { disable_quant_qdq: '1' } },
  })
  const create = async (data: ArrayBuffer, gpu: boolean): Promise<Ort.InferenceSession> => {
    if (!gpu) return ort.InferenceSession.create(new Uint8Array(data), options(['wasm']))
    try {
      // Operators WebGPU lacks run on the CPU within the same session.
      return await ort.InferenceSession.create(new Uint8Array(data), options(['webgpu', 'wasm']))
    } catch (err) {
      post({ type: 'note', text: `WebGPU failed (${err instanceof Error ? err.message : String(err)}); using the CPU` })
      placed = 'wasm'
      return ort.InferenceSession.create(new Uint8Array(data), options(['wasm']))
    }
  }
  let placed: 'webgpu' | 'wasm' | 'split' = req.halves ? 'split' : req.device
  if (halfFiles) {
    const front = await create(halfFiles[0].data, true)
    const back = await create(halfFiles[1].data, false)
    session = twoPart(front, back)
  } else if (modelFile) {
    session = await create(modelFile.data, req.device === 'webgpu')
  } else throw new Error('The voice model is missing')
  stage('warmup')
  // A short run so the first sentence does not pay for first-use allocations.
  if (config.kind === 'kokoro') await runKokoro([0, 50, 83, 0], 1)
  else await run(config.kind === 'coqui' ? coquiIds(config, 'a') : piperIds(config.phoneme_id_map, 'ə').ids, [0, 1, 0])
  post({
    type: 'ready',
    loadMs: performance.now() - started,
    memoryBytes: runtimeMemoryBytes(),
    config,
    threads: req.threads,
    cached: (modelFile ?? halfFiles?.[0])?.fromCache ?? false,
    device: placed,
    cacheKeys: [req.halves ? null : req.model, req.halves?.front, req.halves?.back, req.config, req.style].filter(
      (x): x is string => Boolean(x),
    ),
  })
}

async function run(ids: number[], scales: [number, number, number]): Promise<Float32Array> {
  if (!session) throw new Error('The voice is not loaded')
  const feeds = {
    input: new ort.Tensor(
      'int64',
      BigInt64Array.from(ids, (x) => BigInt(x)),
      [1, ids.length],
    ),
    input_lengths: new ort.Tensor('int64', BigInt64Array.from([BigInt(ids.length)]), [1]),
    scales: new ort.Tensor('float32', Float32Array.from(scales), [3]),
  }
  const out = await session.run(feeds)
  const tensor = out[session.outputNames[0] as string] as Ort.Tensor
  const audio = new Float32Array(tensor.data as Float32Array)
  tensor.dispose()
  return audio
}

/** Kokoro: token ids, the style row for this length (510 rows of 256), and the speed. */
async function runKokoro(ids: number[], speed: number): Promise<Float32Array> {
  if (!session || !kokoroStyle) throw new Error('The voice is not loaded')
  const rows = kokoroStyle.length / 256
  const row = Math.min(Math.max(ids.length - 2, 0), rows - 1)
  const feeds = {
    input_ids: new ort.Tensor(
      'int64',
      BigInt64Array.from(ids, (x) => BigInt(x)),
      [1, ids.length],
    ),
    style: new ort.Tensor('float32', kokoroStyle.slice(row * 256, row * 256 + 256), [1, 256]),
    speed: new ort.Tensor('float32', Float32Array.from([speed]), [1]),
  }
  const out = await session.run(feeds)
  const tensor = out[session.outputNames[0] as string] as Ort.Tensor
  const audio = new Float32Array(tensor.data as Float32Array)
  tensor.dispose()
  return audio
}

async function synth(req: Extract<TtsWorkerRequest, { type: 'synth' }>): Promise<void> {
  if (!config) throw new Error('The voice is not loaded')
  const started = performance.now()
  let ids: number[][]
  let phonemes: string | undefined
  const missing: string[] = []
  if (config.kind === 'kokoro') {
    if (!uccharan) throw new Error('The phonemiser is not loaded')
    const tokens = await uccharan.phonemize(req.text, { lang: config.phoneme_language, profile: 'kokoro' })
    // Kokoro reads at most 510 tokens per call.
    const all = uccharan.kokoroIds(tokens)
    const audio = await runKokoro(all.length > 512 ? [...all.slice(0, 511), 0] : all, 1 / (req.scales[1] || 1))
    post(
      {
        type: 'audio',
        id: req.id,
        audio,
        ms: performance.now() - started,
        tokens: all.length,
        phonemes: tokens,
        missing,
        memoryBytes: runtimeMemoryBytes(),
      },
      [audio.buffer],
    )
    return
  }
  if (config.kind === 'coqui') {
    ids = [coquiIds(config, withLeadIn(config, req.text))]
  } else {
    if (!uccharan) throw new Error('The phonemiser is not loaded')
    const sentences = await uccharan.phonemize(req.text, { lang: config.phoneme_language, profile: 'piper' })
    phonemes = sentences.join(' ')
    ids = sentences.map((s) => {
      const r = piperIds(config?.kind === 'piper' ? config.phoneme_id_map : {}, s)
      missing.push(...r.missing)
      return r.ids
    })
  }
  const parts: Float32Array[] = []
  // One model call per sentence, each followed by the server's pause shortening, as sherpa-onnx does.
  for (const seq of ids) if (seq.length > 2) parts.push(scaleSilence(await run(seq, req.scales), config.sample_rate))
  const total = parts.reduce((a, p) => a + p.length, 0)
  const audio = new Float32Array(total)
  let o = 0
  for (const p of parts) {
    audio.set(p, o)
    o += p.length
  }
  post(
    {
      type: 'audio',
      id: req.id,
      audio,
      ms: performance.now() - started,
      tokens: ids.reduce((a, s) => a + s.length, 0),
      phonemes,
      missing: [...new Set(missing)],
      memoryBytes: runtimeMemoryBytes(),
    },
    [audio.buffer],
  )
}

scope.onmessage = (e: MessageEvent<TtsWorkerRequest>) => {
  const req = e.data
  chain = chain
    .then(() => (req.type === 'load' ? load(req) : synth(req)))
    .catch((err: unknown) => {
      post({
        type: 'error',
        id: req.type === 'synth' ? req.id : undefined,
        message: err instanceof Error ? err.message : String(err),
      })
    })
}
