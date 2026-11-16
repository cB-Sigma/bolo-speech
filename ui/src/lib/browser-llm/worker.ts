/// <reference lib="webworker" />
import {
  AutoModelForCausalLM,
  AutoTokenizer,
  env,
  InterruptableStoppingCriteria,
  TextStreamer,
} from '@huggingface/transformers'
import { setOrtGpuPower } from '@/lib/webgpu'
import type { LlmWorkerRequest, LlmWorkerResponse } from './models'

/** Runs one language model in a worker with transformers.js and streams the reply, token by token. */

const scope = self as unknown as DedicatedWorkerGlobalScope
const post = (msg: LlmWorkerResponse) => scope.postMessage(msg)

type Tok = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>
let tokenizer: Tok | null = null
let model: { generate(args: Record<string, unknown>): Promise<unknown> } | null = null
const stopping = new InterruptableStoppingCriteria()
let chain: Promise<void> = Promise.resolve()

async function load(req: Extract<LlmWorkerRequest, { type: 'load' }>): Promise<void> {
  const started = performance.now()
  env.allowLocalModels = false
  const host = (req.modelHost ?? 'https://huggingface.co/').replace(/\/?$/, '/')
  env.remoteHost = host
  setOrtGpuPower(env.backends.onnx as Parameters<typeof setOrtGpuPower>[0], req.power)
  const wasm = env.backends.onnx.wasm as { numThreads?: number } | undefined
  if (wasm) wasm.numThreads = req.threads
  const files = new Map<string, { file: string; loaded: number; total: number }>()
  const keys = new Set<string>()
  let last = 0
  const progress = (info: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (!info.file || info.status === 'progress_total') return
    keys.add(`${host.replace(/\/$/, '')}/${req.repo}/resolve/${req.revision}/${info.file}`)
    const prev = files.get(info.file)
    if (info.status === 'progress')
      files.set(info.file, { file: info.file, loaded: info.loaded ?? 0, total: info.total ?? prev?.total ?? 0 })
    else if (info.status === 'done' && prev) files.set(info.file, { ...prev, loaded: prev.total })
    else if (!prev) files.set(info.file, { file: info.file, loaded: 0, total: 0 })
    const now = performance.now()
    if (now - last > 150 || info.status === 'done') {
      last = now
      post({ type: 'progress', files: [...files.values()] })
    }
  }
  tokenizer = await AutoTokenizer.from_pretrained(req.repo, {
    revision: req.revision,
    progress_callback: progress,
  } as never)
  model = (await AutoModelForCausalLM.from_pretrained(req.repo, {
    revision: req.revision,
    dtype: req.dtype,
    device: req.device,
    progress_callback: progress,
  } as never)) as unknown as typeof model
  const bytes = [...files.values()].reduce((a, f) => a + f.total, 0)
  post({ type: 'ready', loadMs: performance.now() - started, cacheKeys: [...keys], bytes, device: req.device })
}

async function generate(req: Extract<LlmWorkerRequest, { type: 'generate' }>): Promise<void> {
  if (!tokenizer || !model) throw new Error('The language model is not loaded')
  const started = performance.now()
  stopping.reset()
  const inputs = (tokenizer as unknown as { apply_chat_template(m: unknown, o: unknown): unknown }).apply_chat_template(
    req.messages,
    { add_generation_prompt: true, return_dict: true },
  ) as Record<string, unknown>
  let text = ''
  let tokens = 0
  const streamer = new TextStreamer(tokenizer as never, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (t: string) => {
      text += t
      tokens += 1
      post({ type: 'token', id: req.id, text: t })
    },
  })
  await model.generate({
    ...inputs,
    max_new_tokens: req.maxTokens,
    do_sample: false,
    streamer,
    stopping_criteria: stopping,
  })
  post({ type: 'done', id: req.id, text, tokens, ms: performance.now() - started })
}

scope.onmessage = (e: MessageEvent<LlmWorkerRequest>) => {
  const req = e.data
  // Stop does not wait in the queue: it interrupts the reply being generated.
  if (req.type === 'stop') {
    stopping.interrupt()
    return
  }
  chain = chain
    .then(() => (req.type === 'load' ? load(req) : generate(req)))
    .catch((err: unknown) =>
      post({
        type: 'error',
        id: req.type === 'generate' ? req.id : undefined,
        message: err instanceof Error ? err.message : String(err),
      }),
    )
}
