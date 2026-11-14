import { STT_RATE } from '@/lib/audio/capture'
import { resample } from '@/lib/audio/resample'
import { BROWSER_MODELS, browserModel } from '@/lib/browser-stt/models'
import type { Segment } from './pipeline'

/** One of the bundled conversations (made by ui/scripts/diarization/make_examples.py). */
export interface Example {
  id: string
  title: string
  language: string
  seconds: number
  speakers: string[]
  turns: { speaker: string; start: number; end: number; text: string }[]
  credits: string[]
}

export const EXAMPLE_IDS = ['two-speakers-en', 'three-speakers-overlap-en', 'two-speakers-hi'] as const

const base = () => `${import.meta.env.BASE_URL ?? '/'}diarization/`

export function exampleAudioUrl(id: string): string {
  return `${base()}${id}.mp3`
}

export async function loadExample(id: string): Promise<Example> {
  const res = await fetch(`${base()}${id}.json`)
  if (!res.ok) throw new Error(`HTTP ${res.status} for the ${id} example`)
  return (await res.json()) as Example
}

/** Decode any audio the browser can read into 16 kHz mono. */
export async function decodeTo16k(data: ArrayBuffer): Promise<Float32Array> {
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
  return resample(mono, buf.sampleRate, STT_RATE)
}

/** Speaker turns to transcribe: same-speaker segments closer than `gap` joined, long ones cut at `max` s. */
export function turnsToTranscribe(segments: Segment[], gap = 0.6, max = 25): Segment[] {
  const out: Segment[] = []
  for (const s of [...segments].sort((a, b) => a.start - b.start)) {
    const last = out.findLast((o) => o.speaker === s.speaker)
    if (last && s.start - last.end <= gap && out.at(-1) === last) last.end = Math.max(last.end, s.end)
    else out.push({ ...s })
  }
  return out.flatMap((s) => {
    const parts: Segment[] = []
    for (let t = s.start; t < s.end; t += max)
      parts.push({ speaker: s.speaker, start: t, end: Math.min(s.end, t + max) })
    return parts
  })
}

/** A recogniser for the language: the one given if it reads it, else the suggested one. */
export function recogniserFor(language: string, current: string): string {
  const cur = browserModel(current)
  if (cur.languages.includes(language)) return cur.id
  if (language === 'hi') return 'whisper-tiny-hindi'
  return BROWSER_MODELS.find((m) => !m.multilingual && m.languages.includes(language))?.id ?? 'whisper-base-q8'
}

/** "S1", "S2", … */
export const speakerName = (i: number) => `S${i + 1}`
