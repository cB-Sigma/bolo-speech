import workletUrl from './capture.worklet.js?url'
import type { MicFailure, MicFailureKind } from '@/lib/mic-help'
import { microphonePermission } from '@/lib/support'
import { Resampler } from './resample'

export const STT_RATE = 16000

export interface AudioSource {
  /** Sample rate of the blocks passed to `onBlock`. */
  readonly sampleRate: number
  stop(): Promise<void>
}

export type BlockHandler = (block: Float32Array) => void

export class MicrophoneError extends Error {
  readonly kind: MicFailureKind
  readonly failure: MicFailure
  constructor(kind: MicFailureKind, message: string, extra: Partial<Omit<MicFailure, 'kind'>> = {}) {
    super(message)
    this.name = 'MicrophoneError'
    this.kind = kind
    this.failure = {
      kind,
      name: extra.name ?? '',
      detail: extra.detail ?? message,
      elapsedMs: extra.elapsedMs ?? 0,
      permission: extra.permission ?? 'unknown',
    }
  }
}

const SHORT_MESSAGE: Record<MicFailureKind, string> = {
  denied: 'The browser refused access to the microphone.',
  dismissed: 'The microphone prompt was closed before it was answered.',
  missing: 'No microphone was found.',
  busy: 'The microphone is in use by something else, or could not be started.',
  unsupported: 'This browser cannot capture audio (getUserMedia and AudioWorklet are required).',
  insecure: 'Browsers only allow microphone access on HTTPS or localhost.',
  failed: 'Could not open the microphone.',
}

function classify(name: string, detail: string): MicFailureKind {
  if (name === 'NotAllowedError' || name === 'SecurityError' || name === 'PermissionDeniedError')
    return /dismiss/i.test(detail) ? 'dismissed' : 'denied'
  if (name === 'NotFoundError' || name === 'OverconstrainedError' || name === 'DevicesNotFoundError') return 'missing'
  if (name === 'NotReadableError' || name === 'AbortError' || name === 'TrackStartError') return 'busy'
  return 'failed'
}

/**
 * Capture the microphone as 16 kHz mono blocks (~20 ms each) via an AudioWorklet.
 *
 * Call this directly from the click or tap handler. Everything that needs the
 * user's gesture happens before the first `await`: `getUserMedia` is called
 * first, so the permission prompt is the first thing that happens, and the
 * AudioContext is created and resumed in the same turn (iOS Safari keeps a
 * context suspended for good if it was created after the gesture has passed).
 */
export function openMicrophone(onBlock: BlockHandler): Promise<AudioSource> {
  if (!globalThis.isSecureContext && !navigator.mediaDevices?.getUserMedia) {
    return Promise.reject(new MicrophoneError('insecure', SHORT_MESSAGE.insecure))
  }
  if (!navigator.mediaDevices?.getUserMedia || typeof AudioWorkletNode === 'undefined') {
    return Promise.reject(new MicrophoneError('unsupported', SHORT_MESSAGE.unsupported))
  }
  const asked = performance.now()
  // 1. The permission prompt, synchronously inside the gesture.
  let streamPromise: Promise<MediaStream>
  try {
    streamPromise = navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    })
  } catch (err) {
    streamPromise = Promise.reject(err instanceof Error ? err : new Error(String(err)))
  }
  // 2. The audio graph, also inside the gesture.
  let ctx: AudioContext | null = null
  let ctxError: unknown = null
  try {
    ctx = new AudioContext({ latencyHint: 'interactive' })
    if (ctx.state === 'suspended') void ctx.resume().catch(() => undefined)
  } catch (err) {
    ctxError = err
  }
  return finishOpening(streamPromise, ctx, ctxError, asked, onBlock)
}

async function finishOpening(
  streamPromise: Promise<MediaStream>,
  ctx: AudioContext | null,
  ctxError: unknown,
  asked: number,
  onBlock: BlockHandler,
): Promise<AudioSource> {
  let stream: MediaStream
  try {
    stream = await streamPromise
  } catch (err) {
    void ctx?.close().catch(() => undefined)
    const name = err instanceof Error ? err.name : ''
    const detail = err instanceof Error ? err.message : String(err)
    const kind = classify(name, detail)
    throw new MicrophoneError(kind, SHORT_MESSAGE[kind], {
      name,
      detail,
      elapsedMs: performance.now() - asked,
      permission: await microphonePermission(),
    })
  }
  const fail = (err: unknown, fallback: string): MicrophoneError => {
    stream.getTracks().forEach((t) => t.stop())
    void ctx?.close().catch(() => undefined)
    const detail = err instanceof Error ? err.message : fallback
    return new MicrophoneError('failed', fallback, { name: err instanceof Error ? err.name : '', detail })
  }
  if (!ctx) throw fail(ctxError, 'Could not start audio processing in this browser.')
  const context = ctx
  try {
    await context.audioWorklet.addModule(workletUrl)
  } catch (err) {
    throw fail(err, 'Could not start the audio worklet.')
  }
  const source = context.createMediaStreamSource(stream)
  const node = new AudioWorkletNode(context, 'bolo-capture', { numberOfInputs: 1, numberOfOutputs: 0 })
  const resampler = new Resampler(context.sampleRate, STT_RATE)
  node.port.onmessage = (e: MessageEvent<Float32Array>) => onBlock(resampler.process(e.data))
  source.connect(node)
  if (context.state === 'suspended') await context.resume().catch(() => undefined)
  return {
    sampleRate: STT_RATE,
    async stop() {
      node.port.onmessage = null
      source.disconnect()
      node.disconnect()
      stream.getTracks().forEach((t) => t.stop())
      await context.close()
    },
  }
}

/**
 * Plays a decoded buffer through `onBlock` at real-time pace (20 ms blocks),
 * as if it were being spoken into a microphone. Timing is drift-corrected
 * against the wall clock. Resolves `done` once the last block has been sent.
 */
export function openBufferSource(
  samples: Float32Array,
  sampleRate: number,
  onBlock: BlockHandler,
  opts: { realtime?: boolean; onDone?: () => void } = {},
): AudioSource {
  const data = sampleRate === STT_RATE ? samples : new Resampler(sampleRate, STT_RATE).process(samples)
  const blockSize = STT_RATE / 50
  let offset = 0
  let stopped = false
  let timer: ReturnType<typeof setTimeout> | undefined
  const started = performance.now()
  const realtime = opts.realtime ?? true

  const tick = () => {
    if (stopped) return
    const due = realtime ? Math.floor(((performance.now() - started) / 1000) * STT_RATE) + blockSize : data.length
    while (offset < data.length && offset < due) {
      onBlock(data.subarray(offset, Math.min(data.length, offset + blockSize)))
      offset += blockSize
    }
    if (offset >= data.length) {
      stopped = true
      opts.onDone?.()
      return
    }
    timer = setTimeout(tick, 10)
  }
  timer = setTimeout(tick, 0)
  return {
    sampleRate: STT_RATE,
    async stop() {
      stopped = true
      clearTimeout(timer)
    },
  }
}
