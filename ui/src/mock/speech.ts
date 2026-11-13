import { mulberry32 } from './engine'

/** Splits text into sentence-ish chunks the way a streaming synthesiser would. */
export function splitChunks(text: string): string[] {
  const parts = text
    .replace(/\s+/g, ' ')
    .trim()
    .match(/[^.!?।,;:]+[.!?।,;:]*\s*/g)
  if (!parts) return []
  const out: string[] = []
  for (const p of parts) {
    const t = p.trim()
    if (!t) continue
    const prev = out[out.length - 1]
    // Merge very short fragments with their neighbour.
    if (prev !== undefined && (prev.length < 12 || t.length < 6)) out[out.length - 1] = `${prev} ${t}`
    else out.push(t)
  }
  return out
}

/**
 * Speech-shaped test signal: voiced syllables (a few harmonics under two
 * moving formant-like envelopes) separated by short gaps. It is not speech,
 * but it has the level, rhythm and spectrum to exercise meters and VAD.
 */
export function babble(text: string, sampleRate: number, seed = 1): Float32Array {
  const rand = mulberry32(seed + text.length * 7919)
  const words = text.split(/\s+/).filter(Boolean)
  const blocks: Float32Array[] = []
  let total = 0
  const f0base = 105 + rand() * 70
  for (const w of words) {
    const syllables = Math.max(1, Math.round(w.replace(/[^\p{L}\p{M}\p{N}]/gu, '').length / 2.6))
    for (let s = 0; s < syllables; s++) {
      const dur = 0.11 + rand() * 0.09
      const n = Math.round(dur * sampleRate)
      const b = new Float32Array(n)
      const f0 = f0base * (0.9 + rand() * 0.25)
      const f1 = 350 + rand() * 500
      const f2 = 1100 + rand() * 1300
      let phase = 0
      for (let i = 0; i < n; i++) {
        const t = i / n
        const env = Math.sin(Math.PI * t) ** 0.8
        phase += (2 * Math.PI * f0 * (1 + 0.04 * Math.sin(t * 5))) / sampleRate
        let v = 0
        for (let h = 1; h <= 14; h++) {
          const f = f0 * h
          const g = Math.exp(-(((f - f1) / 260) ** 2)) + 0.6 * Math.exp(-(((f - f2) / 380) ** 2)) + 0.05
          v += (g * Math.sin(phase * h)) / Math.sqrt(h)
        }
        b[i] = v * env * 0.22
      }
      blocks.push(b)
      total += n
    }
    const gap = Math.round((/[.!?।,;:]$/.test(w) ? 0.16 + rand() * 0.08 : 0.035 + rand() * 0.03) * sampleRate)
    blocks.push(new Float32Array(gap))
    total += gap
  }
  const out = new Float32Array(total)
  let o = 0
  for (const b of blocks) {
    out.set(b, o)
    o += b.length
  }
  return out
}
