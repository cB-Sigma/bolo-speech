/** Text helpers for the cloning models. */

const SENTENCE_END = /(?<=[.!?।॥。！？])\s+/u

/**
 * Chatterbox makes a whole clip per call, so long text is spoken in pieces
 * of whole sentences up to `max` characters (each one plays as soon as it is
 * ready). A sentence longer than that is split at commas, then at spaces.
 */
export function chunkForChatterbox(text: string, max = 220): string[] {
  const clean = text.replace(/\s+/g, ' ').trim()
  if (!clean) return []
  const pieces: string[] = []
  for (const sentence of clean.split(SENTENCE_END)) {
    if (sentence.length <= max) {
      pieces.push(sentence)
      continue
    }
    let cur = ''
    for (const part of sentence.split(/(?<=[,;:،、，])\s+/u).flatMap((p) => (p.length <= max ? [p] : p.split(' ')))) {
      if (cur && cur.length + part.length + 1 > max) {
        pieces.push(cur)
        cur = part
      } else cur = cur ? `${cur} ${part}` : part
    }
    if (cur) pieces.push(cur)
  }
  const out: string[] = []
  for (const p of pieces) {
    const last = out[out.length - 1]
    if (last != null && last.length + p.length + 1 <= max) out[out.length - 1] = `${last} ${p}`
    else out.push(p)
  }
  return out
}

/** Speech tokens (25 a second) to allow for `text`: generous, so a slow voice is not cut off. */
export function chatterboxTokenBudget(text: string): number {
  return Math.min(1000, Math.ceil([...text].length * 3) + 50)
}

/** Words for a word error rate: lower case, no punctuation. */
export function words(text: string): string[] {
  return text
    .toLowerCase()
    .normalize('NFC')
    .replace(/[^\p{L}\p{N}\p{M}'\s]/gu, ' ')
    .split(/\s+/)
    .map((w) => w.replace(/^'+|'+$/g, ''))
    .filter(Boolean)
}

/** Word error rate of `hypothesis` against `reference` (edit distance over words / reference words). */
export function wordErrorRate(reference: string, hypothesis: string): number {
  const r = words(reference)
  const h = words(hypothesis)
  if (!r.length) return h.length ? 1 : 0
  let prev = Array.from({ length: h.length + 1 }, (_, j) => j)
  for (let i = 1; i <= r.length; i++) {
    const cur = [i]
    for (let j = 1; j <= h.length; j++)
      cur[j] = Math.min(
        (prev[j] as number) + 1,
        (cur[j - 1] as number) + 1,
        (prev[j - 1] as number) + (r[i - 1] === h[j - 1] ? 0 : 1),
      )
    prev = cur
  }
  return (prev[h.length] as number) / r.length
}
