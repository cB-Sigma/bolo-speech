/** "Same speaker" at or above this cosine with the default embedding model (pyannote-3.0-eres2net). */
export const SAME_SPEAKER = 0.5
/** A resemblance at or above this; different people reach it too. */
export const SIMILAR = 0.3
/** The embedding model the thresholds were measured with. */
export const CALIBRATED_MODEL = 'pyannote-3.0-eres2net'

/** Cosine of two vectors (the server's are unit length, but a client should not rely on it); null if unusable. */
export function cosine(a: number[], b: number[]): number | null {
  if (!a.length || a.length !== b.length) return null
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as number
    const y = b[i] as number
    dot += x * y
    na += x * x
    nb += y * y
  }
  if (!na || !nb) return null
  return dot / Math.sqrt(na * nb)
}

export type Verdict = 'same speaker' | 'similar' | 'different'

export function verdictFor(score: number, thresholds = { same: SAME_SPEAKER, similar: SIMILAR }): Verdict {
  if (score >= thresholds.same) return 'same speaker'
  if (score >= thresholds.similar) return 'similar'
  return 'different'
}
