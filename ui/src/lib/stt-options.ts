/**
 * The transcription extensions the console exposes beyond speaker labels: Whisper decoding options and the
 * repetition guard (`POST /v1/audio/transcriptions`), and denoising before recognition (transcriptions and
 * Realtime sessions). Everything left empty is the server's default (the model's pin, then the catalog).
 */

/** Numeric decoding options, by form field. Each is sent only when a value is given. */
export const DECODING_FIELDS = [
  {
    key: 'repetition_penalty',
    label: 'Repetition penalty',
    hint: 'faster-whisper; 1 is off, 1.1 to 1.3 against loops',
    min: 0.01,
    step: 0.05,
  },
  {
    key: 'no_repeat_ngram_size',
    label: 'No repeated n-grams of',
    hint: 'faster-whisper; tokens, 0 is off (3 is usual). Also drops speech that really repeats',
    min: 0,
    max: 100,
    step: 1,
  },
  {
    key: 'compression_ratio_threshold',
    label: 'Compression ratio threshold',
    hint: 'faster-whisper; 2.4. Acts only with a temperature increment',
    min: 0,
    step: 0.1,
  },
  { key: 'log_prob_threshold', label: 'Log probability threshold', hint: 'Whisper; -1', step: 0.1 },
  { key: 'no_speech_threshold', label: 'No-speech threshold', hint: 'Whisper; 0.6', min: 0, max: 1, step: 0.05 },
  {
    key: 'temperature_increment_on_fallback',
    label: 'Temperature increment on fallback',
    hint: 'Whisper; 0 to 1 (0.2 retries failed windows)',
    min: 0,
    max: 1,
    step: 0.1,
  },
  {
    key: 'hallucination_silence_threshold',
    label: 'Hallucination silence threshold',
    hint: 'faster-whisper; seconds of silence skipped when a hallucination is suspected',
    min: 0,
    step: 0.5,
  },
  {
    key: 'chunk_length',
    label: 'Chunk length',
    hint: 'seconds, 1 to 30 (30); longer audio is cut at pauses into pieces of at most this',
    min: 1,
    max: 30,
    step: 1,
  },
  {
    key: 'max_repeats',
    label: 'Repetition guard: max repeats',
    hint: 'every engine; a phrase repeated more often than this is cut to one copy (4; 0 is off)',
    min: 0,
    step: 1,
  },
] as const

export const VAD_FIELDS = [
  { key: 'threshold', label: 'Speech threshold', hint: '0 to 1 (0.5)', min: 0, max: 1, step: 0.05 },
  { key: 'min_speech_duration_ms', label: 'Min speech', hint: 'ms', min: 0, step: 50 },
  { key: 'max_speech_duration_s', label: 'Max speech', hint: 's', min: 0, step: 1 },
  { key: 'min_silence_duration_ms', label: 'Min silence', hint: 'ms (300)', min: 0, step: 50 },
  { key: 'speech_pad_ms', label: 'Speech pad', hint: 'ms (150)', min: 0, step: 10 },
] as const

export type DecodingKey = (typeof DECODING_FIELDS)[number]['key']
export type VadKey = (typeof VAD_FIELDS)[number]['key']

/** `''` means "the server's default". */
export type TriState = '' | 'true' | 'false'

export interface DecodingOptions {
  conditionOnPreviousText: TriState
  vadFilter: TriState
  /** As typed; empty strings are not sent. */
  numbers: Partial<Record<DecodingKey, string>>
  vad: Partial<Record<VadKey, string>>
}

export const emptyDecoding = (): DecodingOptions => ({
  conditionOnPreviousText: '',
  vadFilter: '',
  numbers: {},
  vad: {},
})

/** How many options differ from the server's defaults (for the section's summary). */
export function decodingCount(o: DecodingOptions): number {
  const filled = (r: Record<string, string | undefined>) => Object.values(r).filter((v) => v != null && v.trim() !== '')
  return (
    Number(o.conditionOnPreviousText !== '') +
    Number(o.vadFilter !== '') +
    filled(o.numbers).length +
    filled(o.vad).length
  )
}

/** A number typed in a field, or null when it is empty or not a number. */
function num(v: string | undefined): number | null {
  if (v == null || v.trim() === '') return null
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

/** Adds the decoding options to a transcription form. `vad_parameters` goes as JSON, which the server reads. */
export function applyDecoding(f: FormData, o: DecodingOptions | undefined): void {
  if (!o) return
  if (o.conditionOnPreviousText) f.set('condition_on_previous_text', o.conditionOnPreviousText)
  if (o.vadFilter) f.set('vad_filter', o.vadFilter)
  for (const field of DECODING_FIELDS) {
    const n = num(o.numbers[field.key])
    if (n != null) f.set(field.key, String(n))
  }
  const vad: Record<string, number> = {}
  for (const field of VAD_FIELDS) {
    const n = num(o.vad[field.key])
    if (n != null) vad[field.key] = n
  }
  if (Object.keys(vad).length) f.set('vad_parameters', JSON.stringify(vad))
}

/** Denoising before recognition. `default`: the model's pin, then the server's setting (off unless set). */
export interface Denoise {
  mode: 'default' | 'off' | 'on'
  /** 0 to 1: the denoised audio mixed with the original at this weight. */
  strength: number
  /** A denoise model id; empty is the server's default denoiser. */
  model: string
}

export const defaultDenoise = (): Denoise => ({ mode: 'default', strength: 1, model: '' })

/** The `denoise` / `denoise_strength` form fields. */
export function applyDenoise(f: FormData, d: Denoise | undefined): void {
  if (!d || d.mode === 'default') return
  if (d.mode === 'off') {
    f.set('denoise', 'false')
    return
  }
  f.set('denoise', d.model || 'true')
  if (d.strength < 1) f.set('denoise_strength', String(Math.max(0.05, Math.round(d.strength * 100) / 100)))
}

/** `session.audio.input.denoise` for a Realtime session; undefined leaves the server's default. */
export function realtimeDenoise(d: Denoise | undefined): false | { model?: string; strength: number } | undefined {
  if (!d || d.mode === 'default') return undefined
  if (d.mode === 'off') return false
  return { ...(d.model ? { model: d.model } : {}), strength: Math.max(0.05, Math.round(d.strength * 100) / 100) }
}
