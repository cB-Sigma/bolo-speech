import type { RequestTrace } from './api/types'
import { fmtCost } from './cost'
import { fmtBytes } from './format'

export const STT_PHASES = ['queue', 'resample', 'vad', 'decode', 'finalize'] as const
export const TTS_PHASES = ['queue', 'normalize', 'synthesize_first', 'synthesize_rest', 'encode'] as const

const LABEL: Record<string, string> = {
  queue: 'Queue',
  resample: 'Resample',
  vad: 'VAD',
  decode: 'Decode',
  finalize: 'Finalize',
  normalize: 'Normalize',
  synthesize_first: 'First audio',
  synthesize_rest: 'Remaining audio',
  encode: 'Encode',
  diarize: 'Diarize',
  consolidate: 'Consolidate speakers',
  separate: 'Separate overlaps',
  // in-browser recogniser
  load: 'Model load',
  endpoint: 'Endpointing',
}

export function phaseLabel(name: string): string {
  return LABEL[name] ?? name
}

/**
 * Phase → categorical slot. Slots follow pipeline order, so neighbours in a
 * stacked bar are always neighbours in the validated palette order. Unknown
 * phase names fall back to the neutral "other" colour rather than a new hue.
 */
export function phaseColor(kind: string, name: string): string {
  const order: readonly string[] = kind === 'tts' ? TTS_PHASES : STT_PHASES
  // The in-browser recogniser's endpointing wait is the same pipeline step as server-side VAD.
  const i = order.indexOf(name === 'endpoint' ? 'vad' : name)
  return i >= 0 ? `var(--chart-${i + 1})` : 'var(--chart-mute)'
}

export function phasesFor(kind: string): readonly string[] {
  return kind === 'tts' ? TTS_PHASES : STT_PHASES
}

/** The latency a user feels for this request: first audio for TTS, the final for STT. */
export function headlineMs(t: RequestTrace): number | null {
  return t.kind === 'tts' ? t.ttfa_ms : (t.final_ms ?? t.total_ms)
}

/** Phases the in-browser recogniser reports, in pipeline order. */
export const BROWSER_STT_PHASES = ['load', 'endpoint', 'queue', 'decode'] as const

/** Extra trace fields worth showing (diarization, separation, voice agents), as label/value pairs. */
export function traceExtras(t: RequestTrace): [string, string][] {
  const out: [string, string][] = []
  const add = (label: string, v: unknown, unit = '') => {
    if (v == null || v === '') return
    out.push([label, typeof v === 'number' ? `${Math.round(v * 10) / 10}${unit}` : String(v)])
  }
  add('Pipeline', t.pipeline)
  if (t.fallback)
    add(
      'Fallback',
      `${t.fallback.requested} → ${t.fallback.answered} (${t.fallback.attempts[0]?.code ?? 'unavailable'})`,
    )
  if (t.provider) {
    add('Cloud provider', `${t.provider.name} · ${t.provider.model}`)
    add('Provider requests', t.provider.requests)
    add('Provider time', t.provider.latency_ms, ' ms')
    add('Sent / received', `${fmtBytes(t.provider.bytes_sent)} / ${fmtBytes(t.provider.bytes_received)}`)
  }
  if (t.cost) add('Estimated cost', fmtCost(t.cost))
  add('Diarizer', t.diarization_model)
  add('Speakers', t.speakers)
  add('Speakers before consolidation', t.speakers_before_consolidation)
  add('Overlapped speech', t.overlapped_speech_s, ' s')
  add('Separation', t.separation)
  add('Separation model', t.separation_model)
  add('Separation windows', t.separation_windows)
  add('Separation time', t.separation_ms, ' ms')
  add('First token', t.first_token_ms, ' ms')
  add('Normaliser', t.normalizer)
  add('Pronunciations applied', t.pronunciations_applied?.length || null)
  if (t.romanised) add('Romanised text', t.romanised.applied ? `converted (${t.romanised.mode})` : 'not converted')
  return out
}
