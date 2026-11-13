import workletUrl from './capture.worklet.js?url'
import { Resampler } from './resample'

export const STT_RATE = 16000

export interface AudioSource {
  /** Sample rate of the blocks passed to `onBlock`. */
  readonly sampleRate: number
  stop(): Promise<void>
}

export type BlockHandler = (block: Float32Array) => void

export class MicrophoneError extends Error {
  readonly kind: 'denied' | 'missing' | 'unsupported' | 'failed'
  constructor(kind: MicrophoneError['kind'], message: string) {
    super(message)
    this.name = 'MicrophoneError'
    this.kind = kind
  }
}

/** Capture the microphone as 16 kHz mono blocks (~20 ms each) via an AudioWorklet. */
export async function openMicrophone(onBlock: BlockHandler): Promise<AudioSource> {
  if (!navigator.mediaDevices?.getUserMedia || typeof AudioWorkletNode === 'undefined') {
    throw new MicrophoneError('unsupported', 'This browser cannot capture audio (AudioWorklet is required).')
  }
  let stream: MediaStream
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    })
  } catch (err) {
    const name = err instanceof DOMException ? err.name : ''
    if (name === 'NotAllowedError' || name === 'SecurityError') {
      throw new MicrophoneError('denied', 'Microphone access was blocked. Allow it in the site settings and try again.')
    }
    if (name === 'NotFoundError' || name === 'OverconstrainedError') {
      throw new MicrophoneError('missing', 'No microphone was found.')
    }
    throw new MicrophoneError('failed', err instanceof Error ? err.message : 'Could not open the microphone.')
  }
  const ctx = new AudioContext({ latencyHint: 'interactive' })
  try {
    await ctx.audioWorklet.addModule(workletUrl)
  } catch (err) {
    stream.getTracks().forEach((t) => t.stop())
    void ctx.close()
    throw new MicrophoneError('failed', err instanceof Error ? err.message : 'Could not start the audio worklet.')
  }
  const source = ctx.createMediaStreamSource(stream)
  const node = new AudioWorkletNode(ctx, 'bolo-capture', { numberOfInputs: 1, numberOfOutputs: 0 })
  const resampler = new Resampler(ctx.sampleRate, STT_RATE)
  node.port.onmessage = (e: MessageEvent<Float32Array>) => onBlock(resampler.process(e.data))
  source.connect(node)
  if (ctx.state === 'suspended') await ctx.resume()
  return {
    sampleRate: STT_RATE,
    async stop() {
      node.port.onmessage = null
      source.disconnect()
      node.disconnect()
      stream.getTracks().forEach((t) => t.stop())
      await ctx.close()
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
