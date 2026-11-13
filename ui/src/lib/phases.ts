import type { RequestTrace } from './api/types'

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
