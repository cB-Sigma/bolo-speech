import { resample } from '@/lib/audio/resample'
import { CLONE_RATE } from './protocol'

/** Shortest and longest reference the page takes. */
export const MIN_SECONDS = 5
export const MAX_SECONDS = 30

/** One of the bundled reference voices (made by ui/scripts/clone/make_examples.py). */
export interface CloneExample {
  id: string
  title: string
  language: string
  file: string
  seconds: number
  transcript: string
  credit: string
  synthetic: boolean
}

const base = () => `${import.meta.env.BASE_URL ?? '/'}clone/`

export function exampleUrl(file: string): string {
  return `${base()}${file}`
}

export async function loadExamples(): Promise<CloneExample[]> {
  const res = await fetch(`${base()}index.json`)
  if (!res.ok) throw new Error(`HTTP ${res.status} for the example voices`)
  return (await res.json()) as CloneExample[]
}

/** Decode any audio the browser can read into 24 kHz mono. */
export async function decodeToCloneRate(data: ArrayBuffer): Promise<Float32Array> {
  const Ctx =
    globalThis.OfflineAudioContext ??
    (globalThis as unknown as { webkitOfflineAudioContext: typeof OfflineAudioContext }).webkitOfflineAudioContext
  const ctx = new Ctx(1, 1, 44100)
  const buf = await ctx.decodeAudioData(data.slice(0))
  const mono = new Float32Array(buf.length)
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c)
    for (let i = 0; i < ch.length; i++) mono[i] = (mono[i] as number) + (ch[i] as number) / buf.numberOfChannels
  }
  return resample(mono, buf.sampleRate, CLONE_RATE)
}

export interface PreparedReference {
  audio: Float32Array
  seconds: number
  /** Things worth telling the visitor (too short, cut, quiet, clipped). */
  notes: string[]
  /** Below the minimum: usable, but the clone will be poor. */
  tooShort: boolean
}

/**
 * Leading and trailing silence trimmed, cut to MAX_SECONDS, peak brought to
 * about −3 dBFS when it is quiet; with notes on what was done.
 */
export function prepareReference(input: Float32Array, rate = CLONE_RATE): PreparedReference {
  const notes: string[] = []
  let peak = 0
  for (const v of input) peak = Math.max(peak, Math.abs(v))
  const floor = Math.max(0.005, peak * 0.03)
  let start = 0
  let end = input.length
  while (start < end && Math.abs(input[start] as number) < floor) start++
  while (end > start && Math.abs(input[end - 1] as number) < floor) end--
  const pad = Math.round(0.1 * rate)
  start = Math.max(0, start - pad)
  end = Math.min(input.length, end + pad)
  let audio = input.slice(start, end)
  if (audio.length > MAX_SECONDS * rate) {
    audio = audio.slice(0, MAX_SECONDS * rate)
    notes.push(`Cut to the first ${MAX_SECONDS} s.`)
  }
  let clipped = 0
  for (const v of audio) if (Math.abs(v) > 0.995) clipped++
  if (clipped > rate * 0.01) notes.push('The recording is clipped (too loud); the copy may sound distorted.')
  if (peak > 0 && peak < 0.25) {
    const g = 0.7 / peak
    for (let i = 0; i < audio.length; i++) audio[i] = (audio[i] as number) * g
    notes.push('The recording was quiet; it has been made louder.')
  }
  if (peak < 0.01) notes.push('The recording is almost silent.')
  const seconds = audio.length / rate
  const tooShort = seconds < MIN_SECONDS
  if (tooShort)
    notes.push(`Only ${seconds.toFixed(1)} s: ${MIN_SECONDS} to ${MAX_SECONDS} s of speech gives a much better copy.`)
  return { audio, seconds, notes, tooShort }
}

export function toInt16(audio: Float32Array): Int16Array {
  const out = new Int16Array(audio.length)
  for (let i = 0; i < audio.length; i++)
    out[i] = Math.max(-32768, Math.min(32767, Math.round((audio[i] as number) * 32767)))
  return out
}

export function fromInt16(pcm: Int16Array): Float32Array {
  const out = new Float32Array(pcm.length)
  for (let i = 0; i < pcm.length; i++) out[i] = (pcm[i] as number) / 32768
  return out
}

export interface Recorder {
  stop(): Promise<Float32Array>
  cancel(): void
}

/**
 * Record from the microphone at its own rate (the recogniser's capture is 16 kHz; a voice copy wants the full
 * band) with MediaRecorder, decoded to 24 kHz when stopped. Echo cancellation and noise suppression are off:
 * they colour the voice the model would copy.
 */
export async function startRecorder(): Promise<Recorder> {
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined')
    throw new Error('This browser cannot record audio here (it needs a secure page and MediaRecorder).')
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true },
  })
  const rec = new MediaRecorder(stream)
  const chunks: Blob[] = []
  rec.ondataavailable = (e) => {
    if (e.data.size) chunks.push(e.data)
  }
  rec.start(250)
  const release = () => stream.getTracks().forEach((t) => t.stop())
  return {
    stop: () =>
      new Promise<Float32Array>((resolve, reject) => {
        rec.onstop = () => {
          release()
          void new Blob(chunks, { type: rec.mimeType }).arrayBuffer().then(decodeToCloneRate).then(resolve, reject)
        }
        rec.stop()
      }),
    cancel: () => {
      try {
        rec.stop()
      } catch {
        // already stopped
      }
      release()
    },
  }
}
