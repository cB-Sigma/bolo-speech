import { cWords, type Ssentencepiece } from './tokenizer'

/**
 * The server's English punctuation and casing (`punct-en-cnn-bilstm`, Edge-Punct-Casing on sherpa-onnx) in
 * TypeScript: sherpa-onnx's OnlinePunctuationCNNBiLSTMImpl (chunks of 200 tokens in, a case and a mark per word
 * out) and what bolo does around it (bolo.engines.sherpa.SherpaPunctuation.punctuate and bolo.punctuation.apply).
 * Given the model's answers, the text is the server's to the character (src/lib/browser-punct/punct.test.ts).
 */

/** kMaxSeqLen: tokens per chunk, `<s>` and `</s>` included. */
export const MAX_SEQ = 200
const BOS = 1
const EOS = 2
/** Text with a longer "word" is returned as it was: one word longer than a chunk crashes sherpa-onnx 2.0.1. */
export const MAX_WORD_BYTES = 190

/** One chunk of the model's input, unpadded. */
export interface PunctChunk {
  ids: number[]
  /** 1 for `<s>`, the first piece of each word and `</s>`. */
  valid: number[]
}

/** The model's input for a batch of chunks, padded to 200 with zeros. */
export interface PunctInput {
  n: number
  tokenIds: Int32Array
  validIds: Int32Array
  labelLens: Int32Array
}

/** The model: argmax of the case and punctuation logits, one each per word. */
export type PunctModel = (input: PunctInput) => Promise<{ cases: ArrayLike<number>; marks: ArrayLike<number> }>

/** EncodeSentences: `<s>`, the words' pieces and `</s>`, a new chunk whenever the next word would not fit. */
export function encodeSentences(sp: Ssentencepiece, text: string): PunctChunk[] {
  const chunks: PunctChunk[] = []
  let ids = [BOS]
  let valid = [1]
  for (const word of cWords(text)) {
    const pieces = sp.encode(word)
    if (ids.length + pieces.length > MAX_SEQ - 1) {
      chunks.push({ ids: [...ids, EOS], valid: [...valid, 1] })
      ids = [BOS]
      valid = [1]
    }
    ids.push(...pieces)
    valid.push(1, ...new Array<number>(Math.max(0, pieces.length - 1)).fill(0))
  }
  chunks.push({ ids: [...ids, EOS], valid: [...valid, 1] })
  return chunks
}

export function batch(chunks: PunctChunk[]): PunctInput {
  const n = chunks.length
  const tokenIds = new Int32Array(n * MAX_SEQ)
  const validIds = new Int32Array(n * MAX_SEQ)
  const labelLens = new Int32Array(n)
  chunks.forEach((c, k) => {
    tokenIds.set(c.ids, k * MAX_SEQ)
    validIds.set(c.valid, k * MAX_SEQ)
    labelLens[k] = c.valid.reduce((a, b) => a + b, 0)
  })
  return { n, tokenIds, validIds, labelLens }
}

/** `std::toupper` in the "C" locale: ASCII letters only. */
const asciiUpper = (s: string) => s.replace(/[a-z]+/g, (m) => m.toUpperCase())

/** DecodeSentences: each word in its predicted case (1 all capitals, 2 first letter), then `,` `.` or `?`. */
export function decodeSentences(text: string, cases: ArrayLike<number>, marks: ArrayLike<number>): string {
  return cWords(text)
    .map((word, i) => {
      const c = cases[i] ?? 0
      let w = word
      if (c === 1) w = asciiUpper(word)
      else if (c === 2) w = asciiUpper(word.charAt(0)) + word.slice(1)
      const m = marks[i] ?? 0
      return w + (m === 1 ? ',' : m === 2 ? '.' : m === 3 ? '?' : '')
    })
    .join(' ')
}

/** What Python's `str.isspace()` calls white space (`str.strip()` removes it). */
const PY_SPACE = '\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000'
const PY_STRIP = new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, 'g')

export function pyStrip(s: string): string {
  return s.replace(PY_STRIP, '')
}

/** Python's `str.isupper()`: at least one cased letter, and none in lower case. */
export function pyIsUpper(s: string): boolean {
  let cased = false
  for (const ch of s) {
    const lower = ch.toLowerCase()
    const upper = ch.toUpperCase()
    if (lower === upper) continue
    if (ch === lower && ch !== upper) return false
    cased = true
  }
  return cased
}

/** A word of more than MAX_WORD_BYTES UTF-8 bytes (no real word comes near). */
export function tooLong(text: string): boolean {
  const enc = new TextEncoder()
  return cWords(text).some((w) => enc.encode(w).length > MAX_WORD_BYTES)
}

/**
 * Punctuation and capitals for one transcript, as the server writes them: all-capital text is read in lower case
 * first (the model writes the capitals itself; bolo.punctuation.apply), blank text comes back as it was, and the
 * answer is stripped. Text with a word longer than a chunk comes back as it was too (sherpa-onnx crashes on it).
 */
export async function punctuate(sp: Ssentencepiece, model: PunctModel, raw: string): Promise<string> {
  const text = pyIsUpper(raw) ? raw.toLowerCase() : raw
  if (!pyStrip(text) || tooLong(text)) return text
  const { cases, marks } = await model(batch(encodeSentences(sp, text)))
  return pyStrip(decodeSentences(text, cases, marks))
}

/** Index of the largest of each row of `width` values (the first on a tie, as std::max_element). */
export function argmaxRows(data: ArrayLike<number>, width: number): Int32Array {
  const rows = Math.floor(data.length / width)
  const out = new Int32Array(rows)
  for (let r = 0; r < rows; r++) {
    let best = 0
    for (let c = 1; c < width; c++) if ((data[r * width + c] as number) > (data[r * width + best] as number)) best = c
    out[r] = best
  }
  return out
}
