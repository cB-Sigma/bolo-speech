import { expandNumbers, type NumberLanguage } from './numbers'

/**
 * Text front end for the in-browser voices: what happens to the text before
 * it becomes model input.
 */

/** What the browser needs to know about a character-input (Coqui VITS) voice; from its voice.json. */
export interface CoquiVoiceConfig {
  kind: 'coqui'
  sample_rate: number
  add_blank: boolean
  blank_id: number
  use_eos_bos: boolean
  bos_id: number
  eos_id: number
  cleaner: 'multilingual'
  inference: { noise_scale: number; length_scale: number; noise_scale_w: number }
  /** Character → id. */
  symbols: Record<string, number>
}

/** A Piper voice: phonemes come from uccharan; ids from the phoneme table. */
export interface PiperVoiceConfig {
  kind: 'piper'
  sample_rate: number
  phoneme_language: string
  phoneme_id_map: Record<string, number[]>
  inference: { noise_scale: number; length_scale: number; noise_scale_w: number }
}

export type VoiceConfig = (CoquiVoiceConfig | PiperVoiceConfig) & { id: string }

/**
 * Coqui's `multilingual_cleaners`, which ran on these voices' training text:
 * NFC, lower case, `;` and `:` to commas, hyphens to spaces, brackets and
 * double quotes removed, whitespace collapsed.
 */
export function multilingualClean(text: string): string {
  return text
    .normalize('NFC')
    .toLowerCase()
    .replaceAll(';', ',')
    .replaceAll('-', ' ')
    .replaceAll(':', ',')
    .replace(/[<>()[\]"]+/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

/**
 * Token ids the way the voice's original front end (Coqui's TTSTokenizer)
 * makes them: characters looked up one by one, unknown ones dropped, then a
 * blank between every pair and at both ends when the voice was trained so.
 * sherpa-onnx produces the same ids for cleaned text (checked against its
 * output; see browser-tts.test.ts).
 */
export function coquiIds(
  cfg: Pick<CoquiVoiceConfig, 'symbols' | 'add_blank' | 'blank_id' | 'use_eos_bos' | 'bos_id' | 'eos_id'>,
  cleaned: string,
): number[] {
  const ids: number[] = []
  for (const ch of cleaned) {
    const id = cfg.symbols[ch]
    if (id != null) ids.push(id)
  }
  let out = ids
  if (cfg.add_blank) {
    out = new Array<number>(ids.length * 2 + 1).fill(cfg.blank_id)
    ids.forEach((id, i) => (out[2 * i + 1] = id))
  }
  return cfg.use_eos_bos ? [cfg.bos_id, ...out, cfg.eos_id] : out
}

/** Piper ids for one phoneme string: `^ _ p1 _ p2 _ … $` (the same layout as Piper and uccharan's piperIds). */
export function piperIds(map: Record<string, number[]>, phonemes: string): { ids: number[]; missing: string[] } {
  const pad = map._ ?? [0]
  const ids: number[] = [...(map['^'] ?? []), ...pad]
  const missing: string[] = []
  for (const ch of phonemes.normalize('NFD')) {
    const v = map[ch]
    if (v) ids.push(...v, ...pad)
    else missing.push(ch)
  }
  ids.push(...(map.$ ?? []))
  return { ids, missing }
}

const SENTENCE_END = /([.!?।॥]+["'”’)]*)\s+/gu

/**
 * Split text into sentences (kept with their punctuation) so the first one can
 * play while the rest are synthesised. Very long sentences are split again at
 * commas, then at spaces, so no piece runs much past `max` characters.
 */
export function splitSentences(text: string, max = 220): string[] {
  const out: string[] = []
  const marked = text.replace(SENTENCE_END, '$1\u0000')
  for (const raw of marked.split('\u0000')) {
    const s = raw.trim()
    if (!s) continue
    if (s.length <= max) {
      out.push(s)
      continue
    }
    let rest = s
    while (rest.length > max) {
      const window = rest.slice(0, max)
      let cut = Math.max(window.lastIndexOf(', '), window.lastIndexOf('، '))
      if (cut < max / 3) cut = window.lastIndexOf(' ')
      if (cut < max / 3) cut = max - 1
      out.push(rest.slice(0, cut + 1).trim())
      rest = rest.slice(cut + 1).trim()
    }
    if (rest) out.push(rest)
  }
  return out
}

const PUNCTUATION_FALLBACK: [string, string][] = [
  ['।', '.'],
  ['॥', '.'],
  ['…', '.'],
  ['’', "'"],
  ['‘', "'"],
  ['—', ','],
  ['–', ','],
  ['!', '.'],
  ['?', '.'],
]

export interface PreparedText {
  /** What will be spoken, after numbers were spelt out and the voice's cleaning was applied. */
  spoken: string
  /** Pieces synthesised one after another (sentences). */
  chunks: string[]
  /** Characters the voice has no symbol for (they are skipped), each listed once. */
  dropped: string[]
  /** True when the spoken text differs from the input in more than case and spacing. */
  rewritten: boolean
}

/**
 * Text as the voice will read it. For character voices the result is exactly
 * the model input (unknown characters aside); Piper voices get their phonemes
 * from uccharan, which does its own normalisation, so only the sentence split
 * applies to them.
 */
export function prepareText(text: string, cfg: VoiceConfig, numbers: NumberLanguage | null): PreparedText {
  if (cfg.kind === 'piper') {
    const chunks = splitSentences(text.replace(/\s+/g, ' ').trim())
    return { spoken: chunks.join(' '), chunks, dropped: [], rewritten: false }
  }
  const expanded = numbers ? expandNumbers(text, numbers) : text
  let cleaned = multilingualClean(expanded)
  // Punctuation the voice never saw, swapped for the nearest symbol it has.
  for (const [from, to] of PUNCTUATION_FALLBACK) {
    if (cleaned.includes(from) && cfg.symbols[from] == null && cfg.symbols[to] != null)
      cleaned = cleaned.replaceAll(from, to)
  }
  const chunks = splitSentences(cleaned)
  const dropped = new Set<string>()
  for (const chunk of chunks) for (const ch of chunk) if (cfg.symbols[ch] == null) dropped.add(ch)
  const spoken = chunks.join(' ')
  return {
    spoken,
    chunks,
    dropped: [...dropped],
    rewritten: multilingualClean(text) !== spoken,
  }
}
