/// <reference lib="webworker" />
import { env, pipeline } from '@huggingface/transformers'
import type { FileProgress, WorkerRequest, WorkerResponse } from './models'

type Asr = (audio: Float32Array, options?: Record<string, unknown>) => Promise<{ text: string } | { text: string }[]>

const scope = self as unknown as DedicatedWorkerGlobalScope
const post = (msg: WorkerResponse) => scope.postMessage(msg)

let asr: Asr | null = null
let chain: Promise<void> = Promise.resolve()

async function load(req: Extract<WorkerRequest, { type: 'load' }>): Promise<void> {
  const started = performance.now()
  env.allowLocalModels = false
  if (req.config.modelHost) env.remoteHost = req.config.modelHost.replace(/\/?$/, '/')
  const wasm = env.backends.onnx.wasm
  if (wasm) {
    wasm.numThreads = req.threads
    const current = wasm.wasmPaths as { mjs: string; wasm: string } | string | undefined
    const cdnBase = typeof current === 'object' && current ? current.mjs.replace(/[^/]*$/, '') : undefined
    const base = req.config.ortWasmBase?.replace(/\/?$/, '/') ?? cdnBase
    if (base) {
      // The plain build is half the size of the asyncify one and is all the wasm backend needs.
      const variant = req.device === 'wasm' ? 'ort-wasm-simd-threaded' : 'ort-wasm-simd-threaded.asyncify'
      wasm.wasmPaths = { mjs: `${base}${variant}.mjs`, wasm: `${base}${variant}.wasm` }
    }
  }

  const files = new Map<string, FileProgress>()
  let lastPost = 0
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
      post({ type: 'progress', files: [...files.values()] })
    }
  }

  const create = pipeline as unknown as (task: string, model: string, options: Record<string, unknown>) => Promise<Asr>
  asr = await create('automatic-speech-recognition', req.repo, {
    device: req.device,
    dtype: req.dtype,
    progress_callback: progress,
  })
  // Warm up so the first real utterance does not pay for session initialisation.
  await asr(new Float32Array(16000))
  post({ type: 'ready', device: req.device, loadMs: performance.now() - started })
}

async function transcribe(req: Extract<WorkerRequest, { type: 'transcribe' }>): Promise<void> {
  if (!asr) throw new Error('Model is not loaded')
  const started = performance.now()
  // Moonshine has no length penalty; cap tokens relative to the audio to avoid run-on hallucination.
  const maxTokens = Math.max(8, Math.ceil((req.audio.length / 16000) * 6.5))
  const out = await asr(req.audio, { max_new_tokens: maxTokens })
  const text = (Array.isArray(out) ? out.map((o) => o.text).join(' ') : out.text).trim()
  post({ type: 'result', id: req.id, text, ms: performance.now() - started })
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
