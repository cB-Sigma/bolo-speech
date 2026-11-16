/// <reference lib="webworker" />
import { AutoFeatureExtractor, env } from '@huggingface/transformers'
import { cachedDownload, ortCdnBase } from '@/lib/browser-runtime'
import { SIMILARITY_MODEL, SIMILARITY_FEATURES } from './similarity'
import type { EmbedWorkerRequest, EmbedWorkerResponse } from './protocol'

/**
 * Speaker embeddings for the similarity score: 3D-Speaker CAM++ in the ONNX
 * form sherpa-onnx publishes (the one the Speakers page uses), fed WeSpeaker's
 * 80-band filterbank from transformers.js. 16 kHz mono in, one vector out.
 */

const scope = self as unknown as DedicatedWorkerGlobalScope
const post = (msg: EmbedWorkerResponse, transfer: Transferable[] = []) => scope.postMessage(msg, transfer)

let embed: ((a: Float32Array) => Promise<Float32Array>) | null = null
let chain: Promise<void> = Promise.resolve()

async function load(req: Extract<EmbedWorkerRequest, { type: 'load' }>): Promise<void> {
  const started = performance.now()
  env.allowLocalModels = false
  const host = (req.modelHost ?? 'https://huggingface.co/').replace(/\/?$/, '/')
  env.remoteHost = host
  const files = new Map<string, { file: string; loaded: number; total: number }>()
  const keys = new Set<string>()
  const report = () => post({ type: 'progress', files: [...files.values()] })
  const ort = await import('onnxruntime-web/wasm')
  ort.env.wasm.numThreads = req.threads
  const base = req.ortWasmBase?.replace(/\/?$/, '/') ?? ortCdnBase(ort.env.versions.web ?? '')
  ort.env.wasm.wasmPaths = { mjs: `${base}ort-wasm-simd-threaded.mjs`, wasm: `${base}ort-wasm-simd-threaded.wasm` }
  const m = SIMILARITY_MODEL
  const url = `${host}${m.repo}/resolve/${m.revision}/${m.file}`
  keys.add(url)
  const f = SIMILARITY_FEATURES
  for (const name of ['preprocessor_config.json']) keys.add(`${host}${f.repo}/resolve/${f.revision}/${name}`)
  const [features, data] = await Promise.all([
    AutoFeatureExtractor.from_pretrained(f.repo, {
      revision: f.revision,
      progress_callback: (info: { status: string; file?: string; loaded?: number; total?: number }) => {
        if (info.status === 'progress' && info.file) {
          files.set(info.file, { file: info.file, loaded: info.loaded ?? 0, total: info.total ?? 0 })
          report()
        }
      },
    } as never),
    cachedDownload(url, 'transformers-cache', {
      progress: (loaded, total) => {
        files.set(m.file, { file: m.file, loaded, total })
        report()
      },
    }),
  ])
  const session = await ort.InferenceSession.create(new Uint8Array(data.data), { executionProviders: ['wasm'] })
  embed = async (a) => {
    const { input_features } = (await (
      features as unknown as (x: Float32Array) => Promise<Record<string, { data: Float32Array; dims: number[] }>>
    )(a)) as { input_features: { data: Float32Array; dims: number[] } }
    const x = new ort.Tensor('float32', Float32Array.from(input_features.data), input_features.dims)
    const out = await session.run({ [session.inputNames[0] as string]: x })
    return Float32Array.from((out[session.outputNames[0] as string] as { data: Float32Array }).data)
  }
  await embed(Float32Array.from({ length: 16_000 }, (_, i) => Math.sin(i / 7) * 0.01))
  post({ type: 'ready', loadMs: performance.now() - started, cacheKeys: [...keys], bytes: data.data.byteLength })
}

scope.onmessage = (e: MessageEvent<EmbedWorkerRequest>) => {
  const req = e.data
  chain = chain
    .then(async () => {
      if (req.type === 'load') return load(req)
      if (!embed) throw new Error('The speaker embedding model is not loaded')
      const started = performance.now()
      const vector = await embed(req.audio)
      post({ type: 'embedding', id: req.id, vector, ms: performance.now() - started }, [vector.buffer])
    })
    .catch((err: unknown) =>
      post({
        type: 'error',
        id: req.type === 'embed' ? req.id : undefined,
        message: err instanceof Error ? err.message : String(err),
      }),
    )
}
