/**
 * The repetition guard of the server (bolo.stt_options: `trim_repeats` and `RepeatGuard`), for the recognisers that
 * run in the page: Whisper sometimes loops on silence or long audio ("the the the ...", "thank you for watching"
 * over and over). A run of more than `maxRepeats` back-to-back copies of one phrase (1 to 8 words; twice as many
 * characters for text written without spaces) is cut to its first copy. `0` turns it off; the default is the
 * server's, 4. src/lib/repeat-guard.test.ts checks it against the server's results (scripts/parity/).
 */

export const DEFAULT_MAX_REPEATS = 4
/** Longest phrase (in words; twice as many characters for text written without spaces) the guard looks for. */
export const MAX_NGRAM = 8

// Python's `[^\w']+`: anything but letters, numbers, underscore and the apostrophe.
const NORM = /[^\p{L}\p{N}_']+/gu

export function normToken(token: string): string {
  return token.toLowerCase().replace(NORM, '')
}

function sameGram(keys: string[], at: number, gram: string[]): boolean {
  if (at + gram.length > keys.length) return false
  for (let k = 0; k < gram.length; k++) if (keys[at + k] !== gram[k]) return false
  return true
}

/** The first run of more than `maxRepeats` back-to-back copies of one n-gram: [start, n, copies]. */
function longestRun(keys: string[], maxRepeats: number, maxN: number): [number, number, number] | null {
  const size = keys.length
  for (let i = 0; i < size; i++) {
    let best: [number, number, number] | null = null
    for (let n = 1; n <= maxN; n++) {
      if (i + n * (maxRepeats + 1) > size) break
      const gram = keys.slice(i, i + n)
      if (!gram.some(Boolean)) continue
      let copies = 1
      while (sameGram(keys, i + copies * n, gram)) copies++
      if (copies > maxRepeats && (best == null || copies * n > best[1] * best[2])) best = [i, n, copies]
    }
    if (best) return best
  }
  return null
}

/** Text written without spaces (Chinese, Japanese, Thai): phrases are counted in characters. */
export function byChars(text: string): boolean {
  return !text.trim().includes(' ') && [...text].some((c) => (c.codePointAt(0) ?? 0) >= 0x0e00)
}

/** Python's `str.split()`: runs of whitespace, no empty pieces. */
const splitWords = (text: string) => text.split(/\s+/).filter(Boolean)

function tokens(text: string, chars: boolean): string[] {
  return chars ? [...text.trim()] : splitWords(text)
}

/** Which keys stay. The first `sent` keys were sent already (earlier finals): looked at, and always kept. */
function keepMask(keys: string[], maxRepeats: number, chars: boolean, sent = 0): boolean[] {
  const keep = keys.map(() => true)
  let start = 0
  for (;;) {
    const run = longestRun(keys.slice(start), maxRepeats, chars ? 2 * MAX_NGRAM : MAX_NGRAM)
    if (!run) break
    const [rel, n, copies] = run
    const at = rel + start
    for (let k = Math.max(at + n, sent); k < at + n * copies; k++) keep[k] = false
    start = at + n * copies
  }
  return keep
}

/** `text` with every run of repeats cut to one copy, and the number of words (or characters) removed. */
export function trimRepeats(text: string, maxRepeats = DEFAULT_MAX_REPEATS): { text: string; removed: number } {
  if (maxRepeats <= 0 || !text) return { text, removed: 0 }
  const chars = byChars(text)
  const toks = tokens(text, chars)
  const keep = keepMask(
    toks.map((t) => normToken(t)),
    maxRepeats,
    chars,
  )
  const removed = keep.filter((k) => !k).length
  if (!removed) return { text, removed: 0 }
  return { text: toks.filter((_, i) => keep[i]).join(chars ? '' : ' '), removed }
}

/**
 * The guard of a live transcript: a loop can run on across utterances, so each final is checked together with
 * the end of the finals sent before it. Only the new final is ever changed; one that is nothing but such copies
 * comes back empty. Partials are not touched (the final replaces them).
 */
export class RepeatGuard {
  removed = 0
  private tail: string[] = []
  private tailChars = false

  readonly maxRepeats: number

  constructor(maxRepeats = DEFAULT_MAX_REPEATS) {
    this.maxRepeats = maxRepeats
  }

  apply(text: string): string {
    if (this.maxRepeats <= 0 || !text.trim()) return text
    const chars = byChars(text)
    if (chars !== this.tailChars) {
      this.tail = []
      this.tailChars = chars
    }
    const toks = tokens(text, chars)
    const keys = toks.map((t) => normToken(t))
    const keep = keepMask([...this.tail, ...keys], this.maxRepeats, chars, this.tail.length).slice(this.tail.length)
    const removed = keep.filter((k) => !k).length
    let out = text
    if (removed) {
      this.removed += removed
      out = toks.filter((_, i) => keep[i]).join(chars ? '' : ' ')
    }
    const limit = (chars ? 2 * MAX_NGRAM : MAX_NGRAM) * (this.maxRepeats + 1)
    this.tail = [...this.tail, ...keys.filter((_, i) => keep[i])].slice(-limit)
    return out
  }
}
