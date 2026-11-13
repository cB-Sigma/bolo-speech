import type { SpeechWorker } from '@/lib/browser-stt/engine'
import type { WorkerRequest, WorkerResponse } from '@/lib/browser-stt/models'

/**
 * A scripted stand-in for the speech worker, for the mock build only. It walks
 * through the same load stages and messages as the real one without fetching
 * a model, so the loading, buffering and failure paths can be exercised in
 * tests. Enabled by the `bolo.mock.browserStt` local-storage key.
 */
export interface FakeSttOptions {
  /** Total time the pretend load takes (ms). */
  loadMs?: number
  /** Fail with this message instead of becoming ready (unless the load is the single-threaded retry). */
  fail?: string
  /** Stop making progress while "compiling" (unless the load is the single-threaded retry). */
  stall?: boolean
}

export const FAKE_STT_KEY = 'bolo.mock.browserStt'

export function readFakeSttOptions(): FakeSttOptions | null {
  try {
    const raw = globalThis.localStorage?.getItem(FAKE_STT_KEY)
    return raw ? (JSON.parse(raw) as FakeSttOptions) : null
  } catch {
    return null
  }
}

const TEXT: Record<string, string> = {
  en: 'this is a pretend transcript',
  hi: 'नमस्ते, यह एक नकली प्रतिलेख है',
}

export class FakeSpeechWorker implements SpeechWorker {
  onmessage: ((e: MessageEvent<WorkerResponse>) => void) | null = null
  onerror: ((e: ErrorEvent) => void) | null = null
  private timers: ReturnType<typeof setTimeout>[] = []
  private readonly opts: FakeSttOptions

  constructor(opts: FakeSttOptions) {
    this.opts = opts
  }

  private emit(msg: WorkerResponse, after: number): void {
    this.timers.push(setTimeout(() => this.onmessage?.({ data: msg } as MessageEvent<WorkerResponse>), after))
  }

  postMessage(req: WorkerRequest): void {
    if (req.type === 'transcribe') {
      const seconds = req.audio.length / 16000
      const language = req.language && req.language !== 'auto' ? req.language : 'en'
      this.emit(
        {
          type: 'result',
          id: req.id,
          text: `${TEXT[language] ?? TEXT.en} (${seconds.toFixed(1)} s)`,
          ms: 40,
          language,
        },
        40,
      )
      return
    }
    const total = this.opts.loadMs ?? 1200
    const retry = req.plainRuntimeLoad
    const runtimeBytes = 14_264_838
    const modelBytes = 28_180_947
    const step = total / 10
    const base = 'https://example.invalid/'
    this.emit({ type: 'stage', stage: 'runtime', loaded: 0, total: runtimeBytes }, 0)
    this.emit({ type: 'stage', stage: 'runtime', loaded: runtimeBytes / 2, total: runtimeBytes }, step)
    this.emit({ type: 'stage', stage: 'runtime', loaded: runtimeBytes, total: runtimeBytes }, step * 2)
    this.emit(
      {
        type: 'fetch',
        record: { url: `${base}ort-wasm-simd-threaded.wasm`, status: 200, bytes: runtimeBytes, ms: 12 },
      },
      step * 2,
    )
    this.emit({ type: 'stage', stage: 'model' }, step * 2 + 1)
    for (let i = 1; i <= 4; i++) {
      this.emit(
        {
          type: 'progress',
          files: [
            { file: 'onnx/encoder_model_quantized.onnx', loaded: Math.min(1, i / 2) * 7_937_661, total: 7_937_661 },
            { file: 'onnx/decoder_model_merged_quantized.onnx', loaded: (i / 4) * 20_243_286, total: 20_243_286 },
          ],
        },
        step * (2 + i),
      )
    }
    this.emit(
      {
        type: 'fetch',
        record: {
          url: `${base}${req.repo}/resolve/main/onnx/decoder_model_merged_quantized.onnx`,
          status: 200,
          bytes: modelBytes,
          ms: 30,
        },
      },
      step * 6,
    )
    this.emit({ type: 'stage', stage: 'compiling' }, step * 6 + 1)
    if (this.opts.stall && !retry) return
    if (this.opts.fail && !retry) {
      this.emit({ type: 'error', message: this.opts.fail }, step * 8)
      return
    }
    this.emit({ type: 'stage', stage: 'warmup' }, step * 9)
    this.emit({ type: 'ready', device: req.device, loadMs: total }, total)
  }

  terminate(): void {
    this.timers.forEach(clearTimeout)
    this.timers = []
    this.onmessage = null
  }
}
