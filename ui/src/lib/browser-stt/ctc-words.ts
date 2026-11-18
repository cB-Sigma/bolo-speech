import type { SttWord } from '@/lib/api/types'

/**
 * Word timestamps from a CTC model's frame scores (Wav2Vec2 in the browser), the way the server's forced aligner
 * reads them: the most likely token of each frame, repeats merged and blanks dropped, puts every letter on the
 * frames where it is heard; a word runs from its first letter's frame to the end of its last. Words are split at
 * the word delimiter (`|`, or a space). `p` is the mean probability of the word's letters on their frames.
 *
 * `logits` is (frames x vocabulary), row-major; `frameS` the seconds per frame (Wav2Vec2: 0.02).
 */
export function ctcWords(
  logits: Float32Array,
  frames: number,
  vocabSize: number,
  idToToken: (id: number) => string,
  blank: number,
  frameS: number,
): SttWord[] {
  const words: SttWord[] = []
  let current: { w: string; t0: number; t1: number; probs: number[] } | null = null
  let previous = -1
  const flush = () => {
    if (current && current.w) {
      const p = current.probs.reduce((a, b) => a + b, 0) / Math.max(1, current.probs.length)
      words.push({ w: current.w, t0: current.t0, t1: current.t1, p: Math.round(p * 10000) / 10000 })
    }
    current = null
  }
  for (let t = 0; t < frames; t++) {
    const row = logits.subarray(t * vocabSize, (t + 1) * vocabSize)
    let best = 0
    let max = -Infinity
    for (let v = 0; v < vocabSize; v++) {
      const x = row[v] ?? -Infinity
      if (x > max) {
        max = x
        best = v
      }
    }
    let sum = 0
    for (let v = 0; v < vocabSize; v++) sum += Math.exp((row[v] ?? -Infinity) - max)
    const prob = 1 / sum
    const repeat = best === previous
    previous = best
    if (best === blank) continue
    const token = idToToken(best)
    if (token === '|' || token === ' ' || token === '▁') {
      flush()
      continue
    }
    if (token.startsWith('<') && token.endsWith('>')) continue
    if (!current) current = { w: '', t0: t * frameS, t1: (t + 1) * frameS, probs: [] }
    if (!repeat) current.w += token
    current.t1 = (t + 1) * frameS
    current.probs.push(prob)
  }
  flush()
  return words
}

/** What `wav2vec2Words` needs of a transformers.js tokenizer. */
export interface CtcTokenizer {
  pad_token_id?: number | null
  get_vocab?: () => Map<string, number> | Record<string, number>
  model?: { vocab?: string[] | Map<string, number> | Record<string, number> }
}

/** The tokens of a CTC tokenizer by id. */
export function tokensById(tokenizer: CtcTokenizer | undefined): string[] {
  const vocab = tokenizer?.model?.vocab ?? tokenizer?.get_vocab?.()
  if (!vocab) return []
  if (Array.isArray(vocab)) return vocab
  const out: string[] = []
  const entries = vocab instanceof Map ? [...vocab.entries()] : Object.entries(vocab)
  for (const [token, id] of entries) out[id] = token
  return out
}

interface CtcPipeline {
  processor: (audio: Float32Array) => Promise<unknown>
  tokenizer?: CtcTokenizer
  model: unknown
}

/** A Wav2Vec2 pipeline's words with their times, from one forward pass over 16 kHz audio. */
export async function wav2vec2Words(asr: CtcPipeline, audio: Float32Array): Promise<SttWord[]> {
  const inputs = await asr.processor(audio)
  const run = asr.model as (inputs: unknown) => Promise<{ logits: { data: Float32Array; dims: number[] } }>
  const { logits } = await run(inputs)
  const [, frames = 0, vocabSize = 0] = logits.dims
  const tokens = tokensById(asr.tokenizer)
  const blank = asr.tokenizer?.pad_token_id ?? 0
  const frameS = frames > 0 ? audio.length / 16000 / frames : 0.02
  return ctcWords(logits.data, frames, vocabSize, (id) => tokens[id] ?? '', blank, frameS)
}
