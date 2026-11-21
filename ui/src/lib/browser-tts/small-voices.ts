/**
 * Small voices that read Kokoro's phoneme table, and little-tts: where their files are (each repository pinned) and
 * the text handling around their ONNX graphs, as the server does it (server/src/bolo/engines/kokoro_vocab.py and
 * kale.py).
 *
 * - little-tts (TuringLogic): tiny-voca's graph inputs and phonemiser, other weights (see tiny-voca.ts).
 * - Paradee-8M (Sahil Mahendrakar): Kokoro's ids, a voice index and the speed.
 * - Kitten TTS 2 Pico (KittenML): Kokoro's ids with Kitten's end token, a style row per input length from the
 *   language's voice pack, and Kitten's own folds and trimming (kittenml's kittentts2_tiny, Apache-2.0).
 * - kale-tts-1-nano (Kale Computer): its own front end (src/vendor/kale, Apache-2.0) writes IPA, made Kokoro-native
 *   here as Kale's runtime does (js/kale-tts.mjs, knIPA), and a style table and a few extra inputs per voice.
 */

export const LITTLE_TTS_BASE =
  'https://huggingface.co/TuringLogic/little-tts/resolve/550ce03fe781da74272d3103cfac5d282d819704/'
export const PARADEE_BASE =
  'https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2/resolve/ed2934d4df8a4fd891e29584c5e1104649409765/'
export const KITTEN_PICO_BASE =
  'https://huggingface.co/KittenML/kitten-tts-2-pico/resolve/a5632deac7d92f6178db814d933eba746a97f2dc/'
export const KALE_BASE =
  'https://huggingface.co/KaleComputer/kale-tts-1-nano/resolve/b6ce1887268c3667994fc3b0b2c82887c069915c/'

// -- Kitten TTS 2 ------------------------------------------------------------------------------------------------
export const KITTEN_END = 10
export const KITTEN_GAP_S = 0.15

/** Symbols the training lists folded (phonemize_v1.FOLD): marks without a row, and the nearest symbol with one. */
const KITTEN_FOLD: [string, string][] = [
  ['̪', ''],
  ['̩', ''],
  ['̍', ''],
  ['̯', ''],
  ['ɭ', 'l'],
  ['ɫ', 'l'],
  ['ɵ', 'ø'],
  ['ε', 'ɛ'],
  ['{', ''],
  ['}', ''],
  ['[', ''],
  [']', ''],
  ['¿', ''],
  ['¡', ''],
  [' ', ' '],
]

/** Phonemes as Kitten's training lists had them (phonemizer v1: Pico). */
export function kittenFold(phones: string): string {
  let out = phones.trim()
  for (const [from, to] of KITTEN_FOLD) out = out.replaceAll(from, to)
  return out
}

/** Kokoro ids `[0, …, 0]` as Kitten takes them: its end token before the last pad. */
export function kittenIds(kokoroIds: number[]): number[] {
  const inner = kokoroIds[kokoroIds.length - 1] === 0 ? kokoroIds.slice(0, -1) : kokoroIds
  return [...inner, KITTEN_END, 0]
}

/**
 * Kitten's output: the end token's frames cut (and, after the first chunk, the start pause), 5 ms fades in and out
 * (kittentts2_tiny.model._chunk).
 */
export function kittenTrim(wave: Float32Array, durations: ArrayLike<number>, keepStart: boolean): Float32Array {
  let total = 0
  for (let i = 0; i < durations.length; i++) total += Number(durations[i])
  const perFrame = Math.max(1, Math.round(wave.length / Math.max(total, 1)))
  const first = Number(durations[0] ?? 0)
  const last = Number(durations[durations.length - 1] ?? 0)
  const lead = keepStart ? 0 : Math.max(0, first - 1) * perFrame
  const trail = Math.max(0, last - 1) * perFrame
  let w = wave.map((x) => Math.max(-1, Math.min(1, x)))
  if (lead + trail + perFrame < w.length) w = w.slice(lead, w.length - trail)
  const fade = Math.min(120, Math.floor(w.length / 8))
  for (let i = 0; i < fade; i++) {
    const k = fade === 1 ? 0 : i / (fade - 1)
    w[i] = (w[i] ?? 0) * k
    w[w.length - 1 - i] = (w[w.length - 1 - i] ?? 0) * k
  }
  return w
}

// -- kale-tts-1 --------------------------------------------------------------------------------------------------
const KN_PAIRS: Record<string, string> = { eɪ: 'A', aɪ: 'I', aʊ: 'W', ɔɪ: 'Y', tʃ: 'ʧ', dʒ: 'ʤ' }
const KN_VOWELS = new Set('aeiouɑɐɒæəɘɚɛɜɝɞɪɨʊʉʌɔøœɤɯʏyAIWOYᵊᵻ')
const KN_END = new Set(' .,!?;:')

/** IPA characters that have ids, Kokoro-native (Kale's knIPA): American rules, or the British convention. */
export function kokoroNative(chars: string[], british: boolean): string[] {
  const pairs: Record<string, string> = { ...KN_PAIRS, ...(british ? { əʊ: 'Q' } : { oʊ: 'O' }) }
  const out: string[] = []
  for (let i = 0; i < chars.length;) {
    const two = (chars[i] ?? '') + (chars[i + 1] ?? '')
    if (pairs[two]) {
      out.push(pairs[two])
      i += 2
      continue
    }
    const c = chars[i] ?? ''
    i += 1
    if (british) {
      out.push(c)
      continue
    }
    if (c === 'ː') continue
    if (c === 'ɚ') out.push('ə', 'ɹ')
    else if (c === 'ɝ') out.push('ɜ', 'ɹ')
    else if (c === 'ɾ') out.push('T')
    else out.push(c)
  }
  for (let k = 1; k < out.length - 1; k++) {
    const next = out[k + 1] ?? ''
    const after = out[k + 2]
    const before = out[k - 1] ?? ''
    if (
      out[k] === 'ə' &&
      'lnm'.includes(next) &&
      next !== '' &&
      (k + 2 === out.length || (after !== undefined && KN_END.has(after))) &&
      !KN_VOWELS.has(before) &&
      before !== 'ˈ' &&
      before !== 'ˌ' &&
      !/\s/.test(before)
    )
      out[k] = 'ᵊ'
  }
  return out
}

/** Kokoro ids of a sentence's IPA (without the pads), Kokoro-native when the model says so. */
export function kaleIds(ipa: string, vocab: Record<string, number>, british: boolean, native = true): number[] {
  const chars = [...ipa].filter((c) => c in vocab)
  return (native ? kokoroNative(chars, british) : chars).filter((c) => c in vocab).map((c) => vocab[c] as number)
}

/** Kale's technical terms (frontend_entry.mjs termsReplacer): whole tokens, longest first; camelCase split; again. */
export function termsReplacer(table: Map<string, string>): (text: string) => string {
  if (table.size === 0) return (t) => t
  const esc = (x: string) => x.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const forms = [...table.keys()]
    .sort((a, b) => b.length - a.length)
    .map((f) => (f.startsWith('.') ? esc(f) : `(?<![\\p{L}\\p{N}_])${esc(f)}`))
  const pattern = new RegExp(`(?:${forms.join('|')})(?![\\p{L}\\p{N}_])`, 'gu')
  const lookUp = (t: string) =>
    t.replace(pattern, (f: string, at: number, whole: string) => {
      const spoken = table.get(f) ?? f
      return f.startsWith('.') && at > 0 && /[\p{L}\p{N}]/u.test(whole[at - 1] ?? '') ? ` ${spoken}` : spoken
    })
  const camel = /(?<![\p{L}\p{N}_])[a-z]+(?:[A-Z][a-z0-9]*)+(?![\p{L}\p{N}_])/gu
  const split = (t: string) => t.replace(camel, (w) => w.replace(/(?<=[a-z0-9])(?=[A-Z])/g, ' '))
  return (t) => lookUp(split(lookUp(t)))
}
