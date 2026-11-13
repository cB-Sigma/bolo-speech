/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/wasm'
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
import { coquiIds, piperIds, type VoiceConfig } from './text'

/**
 * Runs one voice with ONNX Runtime Web (WebAssembly). The VITS graph is called
 * directly: no transformers.js pipeline, no phonemiser for character voices,
 * and uccharan (Apache-2.0) for Piper voices.
 */

const scope = self as unknown as DedicatedWorkerGlobalScope
const post = (msg: TtsWorkerResponse, transfer: Transferable[] = []) => scope.postMessage(msg, transfer)
const stage = (s: LoadStage, loaded?: number, total?: number) => post({ type: 'stage', stage: s, loaded, total })

let session: ort.InferenceSession | null = null
let config: VoiceConfig | null = null
let uccharan: typeof UccharanModule | null = null
let chain: Promise<void> = Promise.resolve()

const record = (r: FetchRecord) => post({ type: 'fetch', record: r })

async function loadRuntime(req: Extract<TtsWorkerRequest, { type: 'load' }>): Promise<void> {
  const wasm = ort.env.wasm
  wasm.numThreads = req.threads
  wasm.proxy = false
  const base = req.ortWasmBase?.replace(/\/?$/, '/') ?? ortCdnBase(ort.env.versions.web ?? '')
  const paths = { mjs: `${base}ort-wasm-simd-threaded.mjs`, wasm: `${base}ort-wasm-simd-threaded.wasm` }
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
  const dir = `${req.base}${req.voice}/`
  const v = `?v=${encodeURIComponent(req.version)}`
  const get = (url: string, name: string, type?: string) =>
    cachedDownload(url + v, VOICE_CACHE, { progress: track(name), record }, { contentType: type })
  const pending: Promise<unknown>[] = []
  const settings = get(`${dir}voice.json`, `${req.voice}/voice.json`, 'application/json')
  const model = get(`${dir}model.onnx`, `${req.voice}/model.onnx`)
  if (req.phonemizer === 'uccharan') {
    pending.push(
      (async () => {
        const mod = await import('@/vendor/uccharan/index.js')
        const data = await get(`${req.base}uccharan/en-us.ucd`, 'uccharan/en-us.ucd')
        await mod.init()
        await mod.loadLanguage('en-us', data.data)
        uccharan = mod
      })(),
    )
  }
  const [cfgFile, modelFile] = await Promise.all([settings, model, ...pending])
  config = JSON.parse(new TextDecoder().decode(cfgFile.data)) as VoiceConfig
  stage('compiling')
  session = await ort.InferenceSession.create(new Uint8Array(modelFile.data), {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    // The 8-bit weights are stored with a DequantizeLinear in front of each. Without this the runtime keeps
    // those nodes (for quantised kernels this graph does not use) and expands the weights on every call;
    // with it they are folded into constants once, at load, and synthesis is as fast as with the fp32 file.
    extra: { session: { disable_quant_qdq: '1' } },
  })
  stage('warmup')
  // A short run so the first sentence does not pay for first-use allocations.
  await run(config.kind === 'coqui' ? coquiIds(config, 'a') : piperIds(config.phoneme_id_map, 'ə').ids, [0, 1, 0])
  post({
    type: 'ready',
    loadMs: performance.now() - started,
    memoryBytes: runtimeMemoryBytes(),
    config,
    threads: req.threads,
    cached: modelFile.fromCache,
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
  const tensor = out[session.outputNames[0] as string] as ort.Tensor
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
  if (config.kind === 'coqui') {
    ids = [coquiIds(config, req.text)]
  } else {
    if (!uccharan) throw new Error('The phonemiser is not loaded')
    const sentences = uccharan.phonemizeSync(req.text, { lang: config.phoneme_language, profile: 'piper' })
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
