/// <reference lib="webworker" />
import {
  AutoFeatureExtractor,
  AutoModel,
  AutoModelForAudioFrameClassification,
  AutoProcessor,
  env,
} from '@huggingface/transformers'
import { cachedDownload, ortCdnBase } from '@/lib/browser-runtime'
import { diarize, RATE } from './pipeline'
import type { DiarizeModel, DiarizeWorkerRequest, DiarizeWorkerResponse } from './models'

/**
 * Runs speaker diarization in a worker: pyannote segmentation and a voice
 * embedding model through transformers.js (ONNX Runtime Web, WebAssembly),
 * or an embedding model in the form sherpa-onnx publishes through ONNX
 * Runtime Web directly. The pipeline itself is ./pipeline.ts.
 */

const scope = self as unknown as DedicatedWorkerGlobalScope
const post = (msg: DiarizeWorkerResponse, transfer: Transferable[] = []) => scope.postMessage(msg, transfer)

type Callable = (
  inputs: Record<string, unknown>,
) => Promise<Record<string, { data: Float32Array; dims: number[]; tolist(): number[][][] }>>
let segment: ((w: Float32Array) => Promise<number[][]>) | null = null
let embed: ((a: Float32Array) => Promise<Float32Array>) | null = null
let chain: Promise<void> = Promise.resolve()

async function load(req: Extract<DiarizeWorkerRequest, { type: 'load' }>): Promise<void> {
  const started = performance.now()
  env.allowLocalModels = false
  const host = (req.modelHost ?? 'https://huggingface.co/').replace(/\/?$/, '/')
  env.remoteHost = host
  const wasm = env.backends.onnx.wasm as { numThreads?: number; wasmPaths?: unknown } | undefined
  if (wasm) {
    wasm.numThreads = req.threads
    if (req.ortWasmBase) {
      const base = req.ortWasmBase.replace(/\/?$/, '/')
      wasm.wasmPaths = { mjs: `${base}ort-wasm-simd-threaded.mjs`, wasm: `${base}ort-wasm-simd-threaded.wasm` }
    }
  }
  const files = new Map<string, { file: string; loaded: number; total: number }>()
  const keys = new Set<string>()
  let last = 0
  const progress =
    (repo: string, revision: string) => (info: { status: string; file?: string; loaded?: number; total?: number }) => {
      if (!info.file || info.status === 'progress_total') return
      const name = `${repo.split('/').pop()}/${info.file}`
      keys.add(`${host.replace(/\/$/, '')}/${repo}/resolve/${revision}/${info.file}`)
      const prev = files.get(name)
      if (info.status === 'progress')
        files.set(name, { file: name, loaded: info.loaded ?? 0, total: info.total ?? prev?.total ?? 0 })
      else if (info.status === 'done' && prev) files.set(name, { ...prev, loaded: prev.total })
      else if (!prev) files.set(name, { file: name, loaded: 0, total: 0 })
      const now = performance.now()
      if (now - last > 100 || info.status === 'done') {
        last = now
        post({ type: 'progress', files: [...files.values()] })
      }
    }
  const opts = (m: DiarizeModel) => ({
    revision: m.revision,
    device: 'wasm',
    dtype: m.runner.kind === 'transformers' ? m.runner.dtype : 'fp32',
    progress_callback: progress(m.repo, m.revision),
  })

  const seg = req.segmentation
  const [segModel, segProcessor] = await Promise.all([
    AutoModelForAudioFrameClassification.from_pretrained(seg.repo, opts(seg) as never),
    AutoProcessor.from_pretrained(seg.repo, {
      revision: seg.revision,
      progress_callback: progress(seg.repo, seg.revision),
    } as never),
  ])
  segment = async (w) => {
    const inputs = await (segProcessor as unknown as (a: Float32Array) => Promise<Record<string, unknown>>)(w)
    const out = await (segModel as unknown as Callable)(inputs)
    return (out.logits as { tolist(): number[][][] }).tolist()[0] as number[][]
  }

  const e = req.embedding
  if (e.runner.kind === 'transformers') {
    const [model, processor] = await Promise.all([
      AutoModel.from_pretrained(e.repo, opts(e) as never),
      AutoProcessor.from_pretrained(e.repo, {
        revision: e.revision,
        progress_callback: progress(e.repo, e.revision),
      } as never),
    ])
    embed = async (a) => {
      const inputs = await (processor as unknown as (a: Float32Array) => Promise<Record<string, unknown>>)(a)
      const out = await (model as unknown as Callable)(inputs)
      const t = out.embeddings ?? Object.values(out)[0]
      return Float32Array.from(t?.data ?? [])
    }
  } else {
    // sherpa-onnx's form: 80-band log-mel filterbank (Kaldi style, mean removed) in, one embedding out. The same
    // features as WeSpeaker's, from transformers.js's extractor for that model.
    const file = e.runner.file
    const ort = await import('onnxruntime-web/wasm')
    ort.env.wasm.numThreads = req.threads
    const base = req.ortWasmBase?.replace(/\/?$/, '/') ?? ortCdnBase(ort.env.versions.web ?? '')
    ort.env.wasm.wasmPaths = { mjs: `${base}ort-wasm-simd-threaded.mjs`, wasm: `${base}ort-wasm-simd-threaded.wasm` }
    const url = `${host}${e.repo}/resolve/${e.revision}/${file}`
    keys.add(url)
    const name = `${e.repo.split('/').pop()}/${file}`
    const [features, data] = await Promise.all([
      AutoFeatureExtractor.from_pretrained('onnx-community/wespeaker-voxceleb-resnet34-LM', {
        revision: '6a61a1833ff2583aabeba044f5c8221f00b67ceb',
        progress_callback: progress(
          'onnx-community/wespeaker-voxceleb-resnet34-LM',
          '6a61a1833ff2583aabeba044f5c8221f00b67ceb',
        ),
      } as never),
      cachedDownload(url, 'transformers-cache', {
        progress: (loaded, total) => {
          files.set(name, { file: name, loaded, total })
          post({ type: 'progress', files: [...files.values()] })
        },
      }),
    ])
    const session = await ort.InferenceSession.create(new Uint8Array(data.data), { executionProviders: ['wasm'] })
    embed = async (a) => {
      const { input_features } = (await (
        features as unknown as (x: Float32Array) => Promise<Record<string, { data: Float32Array; dims: number[] }>>
      )(a)) as {
        input_features: { data: Float32Array; dims: number[] }
      }
      const x = new ort.Tensor('float32', Float32Array.from(input_features.data), input_features.dims)
      const out = await session.run({ [session.inputNames[0] as string]: x })
      return Float32Array.from((out[session.outputNames[0] as string] as { data: Float32Array }).data)
    }
  }
  // Warm up both on a second of silence-ish noise.
  const noise = Float32Array.from({ length: RATE }, (_, i) => Math.sin(i / 7) * 0.01)
  const w = new Float32Array(10 * RATE)
  w.set(noise)
  await segment(w)
  await embed(noise)
  const bytes = [...files.values()].reduce((a, f) => a + f.total, 0)
  post({ type: 'ready', loadMs: performance.now() - started, cacheKeys: [...keys], bytes })
}

async function run(req: Extract<DiarizeWorkerRequest, { type: 'run' }>): Promise<void> {
  if (!segment || !embed) throw new Error('The diarization models are not loaded')
  const started = performance.now()
  const segments = await diarize(
    req.audio,
    { segment, embed },
    {
      speakers: req.speakers,
      threshold: req.threshold,
      step: 2.5,
      progress: (done) => post({ type: 'step', id: req.id, done }),
    },
  )
  post({ type: 'result', id: req.id, segments, ms: performance.now() - started })
}

scope.onmessage = (e: MessageEvent<DiarizeWorkerRequest>) => {
  const req = e.data
  chain = chain
    .then(() => (req.type === 'load' ? load(req) : run(req)))
    .catch((err: unknown) =>
      post({
        type: 'error',
        id: req.type === 'run' ? req.id : undefined,
        message: err instanceof Error ? err.message : String(err),
      }),
    )
}
