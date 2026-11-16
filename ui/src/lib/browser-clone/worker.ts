/// <reference lib="webworker" />
import type * as Ort from 'onnxruntime-web/wasm'
import { cachedDownload, lowerMemoryMaximum, ortCdnBase, RUNTIME_CACHE } from '@/lib/browser-runtime'
import type { FileProgress } from '@/lib/browser-stt/models'
import { chatterboxFiles, pocketFiles } from './models'
import type { MakeTensor, OrtSession } from './ort-types'
import { freeDimension } from './onnx-patch'
import { PocketTts, type PocketMeta, type PocketVoice } from './pocket'
import type { CloneWorkerRequest, CloneWorkerResponse, GenerateOptions } from './protocol'
import { chunkForChatterbox, chatterboxTokenBudget } from './text'
import { SpmTokenizer } from './spm'

/**
 * Runs one voice-cloning model in a worker. Pocket TTS: five ONNX Runtime Web
 * sessions driven by ./pocket.ts. Chatterbox: transformers.js's
 * ChatterboxModel (speech encoder, embeddings, language model, decoder).
 */

const scope = self as unknown as DedicatedWorkerGlobalScope
const post = (msg: CloneWorkerResponse, transfer: Transferable[] = []) => scope.postMessage(msg, transfer)

type Load = Extract<CloneWorkerRequest, { type: 'load' }>

interface Engine {
  voice(key: string, audio: Float32Array): Promise<void>
  generate(key: string, text: string, options: GenerateOptions, id: number): Promise<number>
}

let engine: Engine | null = null
let chain: Promise<void> = Promise.resolve()
const cancelled = new Set<number>()
let interrupt: (() => void) | null = null

function progressTracker() {
  const files = new Map<string, FileProgress>()
  let last = 0
  return {
    files,
    set(file: string, loaded: number, total: number) {
      files.set(file, { file, loaded, total })
      const now = performance.now()
      if (now - last > 100 || (total > 0 && loaded >= total)) {
        last = now
        post({ type: 'progress', files: [...files.values()] })
      }
    },
  }
}

async function loadOrt(req: Load): Promise<typeof Ort> {
  const gpu = req.device === 'webgpu'
  const ort = gpu
    ? ((await import('onnxruntime-web/webgpu')) as unknown as typeof Ort)
    : await import('onnxruntime-web/wasm')
  const wasm = ort.env.wasm
  wasm.numThreads = req.threads
  wasm.proxy = false
  const base = req.ortWasmBase?.replace(/\/?$/, '/') ?? ortCdnBase(ort.env.versions.web ?? '')
  const variant = gpu ? 'ort-wasm-simd-threaded.asyncify' : 'ort-wasm-simd-threaded'
  const paths = { mjs: `${base}${variant}.mjs`, wasm: `${base}${variant}.wasm` }
  wasm.wasmPaths = { ...paths }
  post({ type: 'stage', stage: 'runtime' })
  try {
    // A lower memory maximum than ONNX Runtime's 4 GiB, as the voices do: on iPhones a second memory that size
    // does not fit next to the recogniser's.
    const glue = await cachedDownload(
      new URL(paths.mjs, scope.location.href).href,
      RUNTIME_CACHE,
      {},
      { contentType: 'text/javascript' },
    )
    const lowered = lowerMemoryMaximum(new TextDecoder().decode(glue.data), req.memoryPages)
    if (lowered)
      wasm.wasmPaths = { ...paths, mjs: URL.createObjectURL(new Blob([lowered], { type: 'text/javascript' })) }
    const { data } = await cachedDownload(
      new URL(paths.wasm, scope.location.href).href,
      RUNTIME_CACHE,
      {},
      { contentType: 'application/wasm' },
    )
    wasm.wasmBinary = data
  } catch (err) {
    post({
      type: 'note',
      text: `runtime fetched by ONNX Runtime itself (${err instanceof Error ? err.message : String(err)})`,
    })
  }
  return ort
}

async function loadPocket(
  req: Load,
): Promise<{ engine: Engine; keys: string[]; bytes: number; device: 'webgpu' | 'wasm' }> {
  const m = req.model
  const ort = await loadOrt(req)
  const host = (req.modelHost ?? 'https://huggingface.co/').replace(/\/?$/, '/')
  const track = progressTracker()
  post({ type: 'stage', stage: 'model' })
  const files = pocketFiles(m)
  const keys: string[] = []
  const data = Object.fromEntries(
    await Promise.all(
      files.map(async (f) => {
        const url = `${host}${m.repo}/resolve/${m.revision}/${f.path}`
        keys.push(url)
        const short = f.path.split('/').pop() as string
        const r = await cachedDownload(url, RUNTIME_CACHE, { progress: (l, t) => track.set(short, l, t) })
        return [f.name, r.data] as const
      }),
    ),
  ) as Record<string, ArrayBuffer>
  const bytes = Object.values(data).reduce((a, d) => a + d.byteLength, 0)
  post({ type: 'stage', stage: 'compiling' })
  let device = req.device
  const create = async (buf: ArrayBuffer, gpu: boolean): Promise<OrtSession> => {
    const opts = (providers: string[]): Ort.InferenceSession.SessionOptions => ({
      executionProviders: providers,
      graphOptimizationLevel: 'all',
      // The patched cache length makes the runtime warn on every frame that outputs differ from the declared shape.
      logSeverityLevel: 3,
    })
    if (gpu)
      try {
        return (await ort.InferenceSession.create(
          new Uint8Array(buf),
          opts(['webgpu', 'wasm']),
        )) as unknown as OrtSession
      } catch (err) {
        post({
          type: 'note',
          text: `WebGPU failed (${err instanceof Error ? err.message : String(err)}); using the CPU`,
        })
        device = 'wasm'
      }
    return (await ort.InferenceSession.create(new Uint8Array(buf), opts(['wasm']))) as unknown as OrtSession
  }
  // The frame loop (language model and flow) gains from WebGPU; the streaming decoder's many small state
  // tensors and the one-off encoder stay on the CPU.
  const gpu = req.device === 'webgpu'
  // Declare the attention cache's length free (see onnx-patch.ts) so a short one can be passed.
  const freed = freeDimension(new Uint8Array(data.main as ArrayBuffer))
  post({ type: 'note', text: `cache length free in ${freed} state tensors` })
  const sessions = {
    encoder: await create(data.encoder as ArrayBuffer, false),
    textConditioner: await create(data.textConditioner as ArrayBuffer, false),
    main: await create(data.main as ArrayBuffer, gpu),
    flow: await create(data.flow as ArrayBuffer, gpu),
    decoder: await create(data.decoder as ArrayBuffer, false),
  }
  const meta = JSON.parse(new TextDecoder().decode(data.bundle)) as PocketMeta
  const tensor: MakeTensor = (type, d, dims) =>
    new ort.Tensor(type, d as Float32Array, dims as number[]) as unknown as ReturnType<MakeTensor>
  const tts = new PocketTts(
    meta,
    SpmTokenizer.fromModel(data.tokenizer as ArrayBuffer),
    sessions,
    readNpy(data.bos as ArrayBuffer),
    tensor,
    { dynamicCache: freed > 0 },
  )
  post({ type: 'stage', stage: 'warmup' })
  let current: string | null = null
  const engine: Engine = {
    async voice(key, audio) {
      const v: PocketVoice = await tts.encodeVoice(audio)
      tts.setVoice(v)
      current = key
    },
    async generate(key, text, options, id) {
      if (current !== key) throw new Error('No reference voice taken in yet')
      let samples = 0
      await tts.generate(
        text,
        {
          temperature: options.temperature,
          lsdSteps: options.lsdSteps,
          seed: options.seed,
          cancelled: () => cancelled.has(id),
        },
        (a) => {
          samples += a.length
          post({ type: 'audio', id, samples: a }, [a.buffer])
        },
      )
      return samples
    },
  }
  // Warm up on a second of near-silence as the voice and a few frames of one word.
  await engine.voice(
    '__warmup',
    new Float32Array(24_000).map((_, i) => Math.sin(i / 9) * 0.01),
  )
  let frames = 0
  await tts.generate('Hi.', { temperature: 0, lsdSteps: 1, cancelled: () => ++frames > 3 }, () => undefined)
  current = null
  return { engine, keys, bytes, device }
}

async function hasShaderF16(): Promise<boolean> {
  try {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<{ features: Set<string> } | null> } }).gpu
    const adapter = await gpu?.requestAdapter()
    return adapter?.features.has('shader-f16') ?? false
  } catch {
    return false
  }
}

/** A float32 .npy file (the BOS vector Pocket TTS puts before the voice). */
function readNpy(buf: ArrayBuffer): Float32Array | null {
  try {
    const view = new DataView(buf)
    const headerLen = view.getUint16(8, true)
    return new Float32Array(buf.slice(10 + headerLen))
  } catch {
    return null
  }
}

async function loadChatterbox(
  req: Load,
): Promise<{ engine: Engine; keys: string[]; bytes: number; device: 'webgpu' | 'wasm' }> {
  const m = req.model
  const tf = await import('@huggingface/transformers')
  const { env, ChatterboxModel, AutoTokenizer, Tensor, InterruptableStoppingCriteria } = tf
  env.allowLocalModels = false
  const host = (req.modelHost ?? 'https://huggingface.co/').replace(/\/?$/, '/')
  env.remoteHost = host
  const wasm = env.backends.onnx.wasm as { numThreads?: number; wasmPaths?: unknown } | undefined
  if (wasm) {
    wasm.numThreads = req.threads
    if (req.ortWasmBase) {
      const base = req.ortWasmBase.replace(/\/?$/, '/')
      const variant = req.device === 'webgpu' ? 'ort-wasm-simd-threaded.asyncify' : 'ort-wasm-simd-threaded'
      wasm.wasmPaths = { mjs: `${base}${variant}.mjs`, wasm: `${base}${variant}.wasm` }
    }
  }
  const track = progressTracker()
  const progress = (info: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (!info.file || info.status === 'progress_total') return
    const prev = track.files.get(info.file)
    if (info.status === 'progress') track.set(info.file, info.loaded ?? 0, info.total ?? prev?.total ?? 0)
    else if (info.status === 'done' && prev) track.set(info.file, prev.total, prev.total)
  }
  post({ type: 'stage', stage: 'model' })
  let device = req.device
  // 16-bit weights and activations need the GPU's shader-f16; without it the 4-bit file with 32-bit activations.
  const f16 = device === 'webgpu' && (await hasShaderF16())
  const dtype = (device === 'webgpu' && !f16 ? m.dtype?.wasm : m.dtype?.[device]) ?? 'q4'
  const open = (dev: 'webgpu' | 'wasm', dt: string): Promise<Opened> =>
    Promise.all([
      ChatterboxModel.from_pretrained(m.repo, {
        revision: m.revision,
        dtype: dt,
        device: dev,
        progress_callback: progress,
      } as never),
      AutoTokenizer.from_pretrained(m.repo, { revision: m.revision, progress_callback: progress } as never),
    ]) as unknown as Promise<Opened>
  let model: InstanceType<typeof ChatterboxModel>
  let tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>
  type Opened = [InstanceType<typeof ChatterboxModel>, typeof tokenizer]
  let usedDtype = dtype
  try {
    ;[model, tokenizer] = await open(device, dtype)
  } catch (err) {
    if (device !== 'webgpu') throw err
    post({ type: 'note', text: `WebGPU failed (${err instanceof Error ? err.message : String(err)}); using the CPU` })
    device = 'wasm'
    usedDtype = m.dtype?.wasm ?? 'q4'
    ;[model, tokenizer] = await open(device, usedDtype)
  }
  const keys = chatterboxFiles(m, usedDtype).map(
    (f) => `${host.replace(/\/$/, '')}/${m.repo}/resolve/${m.revision}/${f}`,
  )
  const bytes = [...track.files.values()].reduce((a, f) => a + f.total, 0)
  type Speech = Awaited<ReturnType<typeof model.encode_speech>>
  const voices = new Map<string, Speech>()
  const engine: Engine = {
    async voice(key, audio) {
      voices.clear()
      voices.set(key, await model.encode_speech(new Tensor('float32', audio, [1, audio.length])))
    },
    async generate(key, text, options, id) {
      const speech = voices.get(key)
      if (!speech) throw new Error('No reference voice taken in yet')
      let samples = 0
      for (const chunk of chunkForChatterbox(text)) {
        if (cancelled.has(id)) break
        const prompt = m.languageTag ? `[${options.language}]${chunk}` : chunk
        const enc = tokenizer(prompt) as unknown as { input_ids: unknown; attention_mask: unknown }
        const stop = new InterruptableStoppingCriteria()
        interrupt = () => stop.interrupt()
        const wav = (await model.generate({
          input_ids: enc.input_ids,
          attention_mask: enc.attention_mask,
          ...speech,
          ...(m.languageTag ? { exaggeration: options.exaggeration } : {}),
          max_new_tokens: chatterboxTokenBudget(chunk),
          do_sample: false,
          repetition_penalty: 1.2,
          stopping_criteria: [stop],
        } as never)) as unknown as { data: Float32Array }
        interrupt = null
        if (cancelled.has(id)) break
        const a = Float32Array.from(wav.data)
        samples += a.length
        post({ type: 'audio', id, samples: a }, [a.buffer])
      }
      return samples
    },
  }
  post({ type: 'stage', stage: 'warmup' })
  return { engine, keys, bytes, device }
}

async function load(req: Load): Promise<void> {
  const started = performance.now()
  engine = null
  const r = req.model.family === 'pocket' ? await loadPocket(req) : await loadChatterbox(req)
  engine = r.engine
  post({
    type: 'ready',
    loadMs: performance.now() - started,
    device: r.device,
    cacheKeys: r.keys,
    bytes: r.bytes || req.model.bytes,
    threads: req.threads,
  })
}

scope.onmessage = (e: MessageEvent<CloneWorkerRequest>) => {
  const req = e.data
  if (req.type === 'cancel') {
    cancelled.add(req.id)
    interrupt?.()
    return
  }
  chain = chain
    .then(async () => {
      if (req.type === 'load') return load(req)
      if (!engine) throw new Error('The cloning model is not loaded')
      const started = performance.now()
      if (req.type === 'voice') {
        await engine.voice(req.key, req.audio)
        post({ type: 'voice', id: req.id, ms: performance.now() - started })
      } else {
        const samples = await engine.generate(req.key, req.text, req.options, req.id)
        cancelled.delete(req.id)
        post({ type: 'done', id: req.id, ms: performance.now() - started, samples })
      }
    })
    .catch((err: unknown) =>
      post({
        type: 'error',
        id: req.type === 'load' ? undefined : req.id,
        message: err instanceof Error ? err.message : String(err),
      }),
    )
}
