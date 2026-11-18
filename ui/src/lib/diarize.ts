/**
 * Speaker-attributed transcription over `POST /v1/audio/transcriptions`
 * (`diarized_json`), with the bolo extensions: diarizer choice, speaker
 * count hints and separation of overlapped speech.
 */

import { applyDecoding, applyDenoise, type DecodingOptions, type Denoise } from './stt-options'

export type SpeakerCount =
  { mode: 'auto' } | { mode: 'exact'; n: number } | { mode: 'range'; min?: number; max?: number }

export interface TranscribeOptions {
  model: string
  language?: string
  diarize: boolean
  /** Diarization model; omitted means the server default (or the model's own, for speaker-conditioned models). */
  diarizer?: string
  speakers?: SpeakerCount
  separate?: boolean
  separationModel?: string
  /** Separate audio recorded below 16 kHz anyway. */
  allowNarrowband?: boolean
  /** Whisper decoding options and the repetition guard. */
  decoding?: DecodingOptions
  /** Denoising before recognition. */
  denoise?: Denoise
  /** Names and terms to favour, comma separated (`hotwords`; models that cannot use them say so). */
  hotwords?: string
  /** Punctuation restoration afterwards: '' server default, 'off', 'auto', 'on' or a punctuation model id. */
  punctuation?: string
}

export interface DiarizedSegment {
  id?: string | number
  start: number
  end: number
  text: string
  speaker?: string | null
  overlap?: boolean
  separated?: boolean
}

export interface SeparationWindow {
  start: number
  end: number
  overlaps?: { start: number; end: number; speakers: string[] }[]
  speakers?: string[]
  separated?: string[]
  skipped?: { speaker: string; reason: string }[]
}

export interface SeparationReport {
  model: string
  status: 'applied' | 'partial' | 'skipped' | string
  reason: string | null
  detail: string | null
  context_s?: number
  speakers?: { speaker: string; sample_s: number | null; skipped?: string }[]
  windows: SeparationWindow[]
  separated_audio_s?: number
  time_ms?: number
  rtf?: number
  narrowband?: boolean
}

export interface DiarizedResult {
  text: string
  duration?: number
  language?: string
  /** How sure the model was of the language it detected (0 to 1; faster-whisper). */
  languageProbability?: number
  segments: DiarizedSegment[]
  separation?: SeparationReport
  /** `x-bolo-ignored-options`: options the model could not honour. */
  ignoredOptions?: string | null
  /** `x-bolo-denoise`: the denoiser that ran (or `skipped; ...`). */
  denoiseHeader?: string | null
  /** `x-bolo-punctuation`: the punctuation model that ran (or `skipped; ...`). */
  punctuationHeader?: string | null
  /** `x-bolo-separation` response header, when sent. */
  separationHeader?: string | null
  /** `x-request-id`, to find the trace. */
  requestId?: string | null
  /** `x-bolo-route`: where `model=auto` sent the recording, and why. */
  routeHeader?: string | null
  /** `x-bolo-language`: the language heard and whether the recogniser was told it. */
  languageHeader?: string | null
  /** `x-bolo-model`: the model that answered. */
  modelHeader?: string | null
}

/** `x-bolo-route` (`auto -> whisper-base; reason=recommended; language=hi; probability=0.98; lid=...`), read. */
export interface RouteDecision {
  requested: string
  answered: string
  reason: string | null
  language: string | null
  probability: number | null
  lid: string | null
  recommended: string | null
  /** The recommended recogniser was not installed and was downloaded for this request (`pulled=true`). */
  pulled: boolean
}

export function parseRoute(header: string | null | undefined): RouteDecision | null {
  if (!header) return null
  const [head = '', ...parts] = header.split(';').map((p) => p.trim())
  const [requested = '', answered = ''] = head.split('->').map((p) => p.trim())
  const kv: Record<string, string> = {}
  for (const p of parts) {
    const i = p.indexOf('=')
    if (i > 0) kv[p.slice(0, i).trim()] = p.slice(i + 1).trim()
  }
  const probability = kv.probability != null ? Number(kv.probability) : null
  return {
    requested,
    answered: answered || requested,
    reason: kv.reason ?? null,
    language: kv.language ?? null,
    probability: probability != null && Number.isFinite(probability) ? probability : null,
    lid: kv.lid ?? null,
    recommended: kv.recommended ?? null,
    pulled: kv.pulled === 'true',
  }
}

/** Builds the multipart form. Throws for combinations the server would refuse, with the reason. */
export function transcriptionForm(file: Blob, name: string, o: TranscribeOptions): FormData {
  if (o.separate && !o.diarize) throw new Error('Separating overlapping speech needs speaker labels (diarization).')
  const f = new FormData()
  f.set('file', file, name)
  f.set('model', o.model)
  if (o.language && o.language !== 'auto') f.set('language', o.language)
  f.set('response_format', o.diarize ? 'diarized_json' : 'verbose_json')
  applyDecoding(f, o.decoding)
  applyDenoise(f, o.denoise)
  if (o.hotwords?.trim()) f.set('hotwords', o.hotwords.trim())
  if (o.punctuation) f.set('punctuation', o.punctuation)
  if (!o.diarize) return f
  f.set('diarize', 'true')
  if (o.diarizer) f.set('diarization_model', o.diarizer)
  const s = o.speakers ?? { mode: 'auto' }
  if (s.mode === 'exact' && s.n > 0) f.set('num_speakers', String(Math.round(s.n)))
  if (s.mode === 'range') {
    if (s.min != null && s.min > 0) f.set('min_speakers', String(Math.round(s.min)))
    if (s.max != null && s.max > 0) f.set('max_speakers', String(Math.round(s.max)))
  }
  if (o.separate) {
    f.set('separate_overlaps', 'true')
    if (o.separationModel) f.set('separation_model', o.separationModel)
    if (o.allowNarrowband) f.set('separation_allow_narrowband', 'true')
  }
  return f
}

/** A speaker-count hint that makes sense, or the reason it does not. */
export function speakerCountProblem(s: SpeakerCount): string | null {
  if (s.mode === 'exact') return s.n >= 1 && s.n <= 32 ? null : 'Give a number of speakers between 1 and 32.'
  if (s.mode === 'range') {
    if (s.min != null && (s.min < 1 || s.min > 32)) return 'At least: between 1 and 32.'
    if (s.max != null && (s.max < 1 || s.max > 32)) return 'At most: between 1 and 32.'
    if (s.min != null && s.max != null && s.min > s.max) return 'At least cannot be more than at most.'
  }
  return null
}

/** Normalises verbose_json and diarized_json into one shape. */
export function parseTranscription(body: unknown, headers?: Headers): DiarizedResult {
  const b = (body ?? {}) as Record<string, unknown>
  const raw = Array.isArray(b.segments) ? (b.segments as Record<string, unknown>[]) : []
  const segments: DiarizedSegment[] = raw.map((s, i) => ({
    id: (s.id as string | number | undefined) ?? i,
    start: Number(s.start ?? 0),
    end: Number(s.end ?? 0),
    text: String(s.text ?? '').trim(),
    speaker: (s.speaker as string | null | undefined) ?? null,
    overlap: s.overlap === true,
    separated: s.separated === true,
  }))
  return {
    text: String(b.text ?? ''),
    duration: typeof b.duration === 'number' ? b.duration : undefined,
    language: typeof b.language === 'string' ? b.language : undefined,
    languageProbability: typeof b.language_probability === 'number' ? b.language_probability : undefined,
    segments,
    separation: (b.separation as SeparationReport | undefined) ?? undefined,
    separationHeader: headers?.get('x-bolo-separation') ?? null,
    ignoredOptions: headers?.get('x-bolo-ignored-options') ?? null,
    denoiseHeader: headers?.get('x-bolo-denoise') ?? null,
    punctuationHeader: headers?.get('x-bolo-punctuation') ?? null,
    requestId: headers?.get('x-request-id') ?? null,
    routeHeader: headers?.get('x-bolo-route') ?? null,
    languageHeader: headers?.get('x-bolo-language') ?? null,
    modelHeader: headers?.get('x-bolo-model') ?? null,
  }
}

/** Speakers in order of first appearance. */
export function speakersOf(segments: DiarizedSegment[]): string[] {
  const seen: string[] = []
  for (const s of [...segments].sort((a, b) => a.start - b.start)) {
    if (s.speaker && !seen.includes(s.speaker)) seen.push(s.speaker)
  }
  return seen
}

/** Seconds of speech per speaker. */
export function talkTime(segments: DiarizedSegment[]): Map<string, number> {
  const out = new Map<string, number>()
  for (const s of segments) {
    if (!s.speaker) continue
    out.set(s.speaker, (out.get(s.speaker) ?? 0) + Math.max(0, s.end - s.start))
  }
  return out
}

const SEPARATION_REASON: Record<string, string> = {
  no_overlap: 'No overlapping speech long enough to separate was found.',
  no_voice_sample: 'No speaker in an overlap had enough clean speech (3 s) to take a voice sample from.',
  timeout: 'Separation did not finish in time; the overlaps were transcribed from the mixture.',
  error: 'The separation model failed; the overlaps were transcribed from the mixture.',
}

/** One sentence for the note above the transcript (after "Separation applied/partial/skipped."). */
export function separationSummary(r: SeparationReport): string {
  const windows = r.windows.filter((w) => (w.separated?.length ?? 0) > 0).length
  if (r.status === 'applied') return `${r.model} separated the voices in ${windows} window${windows === 1 ? '' : 's'}.`
  const why = r.detail ?? (r.reason ? (SEPARATION_REASON[r.reason] ?? r.reason) : null)
  const sentence = why ? `${why.charAt(0).toUpperCase()}${why.slice(1)}${/[.!?]$/.test(why) ? '' : '.'}` : ''
  if (r.status === 'partial') return `${r.model} separated some of the voices. ${sentence}`.trim()
  return sentence || 'Nothing was separated.'
}

/** Sample rate from a WAV header, or null for anything else. */
export function wavSampleRate(bytes: ArrayBuffer): number | null {
  if (bytes.byteLength < 28) return null
  const v = new DataView(bytes)
  const tag = (o: number) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3))
  if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE') return null
  let o = 12
  while (o + 8 <= bytes.byteLength) {
    const size = v.getUint32(o + 4, true)
    if (tag(o) === 'fmt ' && o + 16 <= bytes.byteLength) return v.getUint32(o + 12, true)
    o += 8 + size + (size & 1)
  }
  return null
}
