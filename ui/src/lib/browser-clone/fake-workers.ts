import type { WorkerLike } from './engine'
import {
  CLONE_RATE,
  type CloneWorkerRequest,
  type CloneWorkerResponse,
  type EmbedWorkerRequest,
  type EmbedWorkerResponse,
} from './protocol'

/**
 * Scripted stand-ins for the cloning and embedding workers (tests only, localStorage `bolo.mock.clone`):
 * nothing is fetched. The "model" speaks a tone whose pitch follows the reference's loudness, in two pieces
 * (so streaming is exercised), and the "embedding" is a few statistics of the audio, so a clip scores close
 * to itself and far from silence. `bolo.mock.clone = "oom"` fails the first load with an out-of-memory error.
 * Failures on WebGPU, as seen with Chatterbox Turbo in Chrome on Windows (generation on the CPU still works):
 * `"gpu-fail"` answers a generation with ONNX Runtime's error, `"gpu-hang"` never answers it, `"gpu-crash"`
 * raises an error the worker cannot catch. `"hang"` never answers on any device and ignores Stop.
 */
export const FAKE_CLONE_KEY = 'bolo.mock.clone'

/** What ONNX Runtime Web said when Chatterbox's decoder failed on WebGPU. */
export const WEBGPU_RUN_ERROR =
  'failed to call OrtRun(). ERROR_CODE: 1, ERROR_MESSAGE: webgpu/buffer_manager.cc:643 BufferManager::Download map_async_result.status == wgpu::MapAsyncStatus::Success was false'

function emitter<R>(target: { onmessage: ((e: MessageEvent<R>) => void) | null }) {
  return (msg: R, after: number) => void setTimeout(() => target.onmessage?.({ data: msg } as MessageEvent<R>), after)
}

let failedOnce = false

export class FakeCloneWorker implements WorkerLike<CloneWorkerRequest, CloneWorkerResponse> {
  onmessage: ((e: MessageEvent<CloneWorkerResponse>) => void) | null = null
  onerror: ((e: ErrorEvent) => void) | null = null
  private readonly mode: string | null
  private level = 0.2
  private cancelled = new Set<number>()
  private device: 'webgpu' | 'wasm' = 'wasm'

  constructor(mode: string | null = null) {
    this.mode = mode
  }

  postMessage(req: CloneWorkerRequest): void {
    const emit = emitter<CloneWorkerResponse>(this)
    if (req.type === 'load') {
      if (this.mode === 'oom' && !failedOnce) {
        failedOnce = true
        emit(
          {
            type: 'error',
            message: 'failed to call OrtRun(). ERROR_CODE: 6, ERROR_MESSAGE: std::bad_alloc: Out of memory',
          },
          50,
        )
        return
      }
      this.device = req.device
      const total = req.model.bytes
      emit({ type: 'stage', stage: 'model' }, 10)
      emit({ type: 'progress', files: [{ file: 'model.onnx', loaded: total / 2, total }] }, 40)
      emit(
        {
          type: 'ready',
          loadMs: 120,
          device: req.device,
          cacheKeys: [`fake://${req.model.id}`],
          bytes: total,
          threads: req.threads,
        },
        120,
      )
    } else if (req.type === 'voice') {
      let sum = 0
      for (const v of req.audio) sum += Math.abs(v)
      this.level = req.audio.length ? sum / req.audio.length : 0
      emit({ type: 'voice', id: req.id, ms: 30 }, 30)
    } else if (req.type === 'cancel') {
      this.cancelled.add(req.id)
    } else if (this.mode === 'hang' || (this.mode === 'gpu-hang' && this.device === 'webgpu')) {
      // A run that never comes back: a few steps, then silence.
      emit({ type: 'tick', id: req.id }, 10)
    } else if (this.mode === 'gpu-fail' && this.device === 'webgpu') {
      emit({ type: 'tick', id: req.id }, 10)
      emit({ type: 'error', id: req.id, message: WEBGPU_RUN_ERROR }, 40)
    } else if (this.mode === 'gpu-crash' && this.device === 'webgpu') {
      setTimeout(() => this.onerror?.({ message: `Uncaught Error: ${WEBGPU_RUN_ERROR}` } as ErrorEvent), 40)
    } else {
      // The texts the "model" was given, for the tests to read.
      const seen = globalThis as { __boloCloneTexts?: string[] }
      ;(seen.__boloCloneTexts ??= []).push(req.text)
      const seconds = Math.min(4, 0.5 + req.text.length / 30)
      const n = Math.round(seconds * CLONE_RATE)
      const f = 120 + 400 * Math.min(1, this.level * 4)
      const tone = Float32Array.from({ length: n }, (_, i) => 0.3 * Math.sin((2 * Math.PI * f * i) / CLONE_RATE))
      const half = Math.floor(n / 2)
      emit({ type: 'audio', id: req.id, samples: tone.slice(0, half) }, 60)
      setTimeout(() => {
        if (!this.cancelled.has(req.id))
          this.onmessage?.({
            data: { type: 'audio', id: req.id, samples: tone.slice(half) },
          } as MessageEvent<CloneWorkerResponse>)
        const total = this.cancelled.has(req.id) ? half : n
        this.onmessage?.({
          data: { type: 'done', id: req.id, ms: 150, samples: total },
        } as MessageEvent<CloneWorkerResponse>)
      }, 150)
    }
  }

  terminate(): void {
    this.onmessage = null
  }
}

export class FakeEmbedWorker implements WorkerLike<EmbedWorkerRequest, EmbedWorkerResponse> {
  onmessage: ((e: MessageEvent<EmbedWorkerResponse>) => void) | null = null
  onerror: ((e: ErrorEvent) => void) | null = null

  postMessage(req: EmbedWorkerRequest): void {
    const emit = emitter<EmbedWorkerResponse>(this)
    if (req.type === 'load') {
      emit({ type: 'ready', loadMs: 50, cacheKeys: ['fake://campplus'], bytes: 29_597_000 }, 50)
      return
    }
    // Loudness, zero crossings and a constant: close for clips alike, far from silence.
    let abs = 0
    let zc = 0
    for (let i = 0; i < req.audio.length; i++) {
      abs += Math.abs(req.audio[i] as number)
      if (i && Math.sign(req.audio[i] as number) !== Math.sign(req.audio[i - 1] as number)) zc++
    }
    const n = Math.max(1, req.audio.length)
    emit({ type: 'embedding', id: req.id, vector: Float32Array.of(abs / n, zc / n, 0.05), ms: 20 }, 40)
  }

  terminate(): void {
    this.onmessage = null
  }
}
