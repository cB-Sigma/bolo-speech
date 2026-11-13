/** Lower-case, strip punctuation (keeping letters/marks/digits of any script), collapse spaces. */
export function normalizeText(text: string): string[] {
  return text
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[^\p{L}\p{M}\p{N}\s']/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .filter(Boolean)
}

export type AlignOp = 'equal' | 'substitute' | 'delete' | 'insert'

export interface AlignStep {
  op: AlignOp
  ref?: string
  hyp?: string
}

export interface WerResult {
  wer: number
  substitutions: number
  deletions: number
  insertions: number
  refWords: number
  alignment: AlignStep[]
}

/** Word error rate by Levenshtein alignment of normalised words. */
export function wordErrorRate(reference: string, hypothesis: string): WerResult {
  const ref = normalizeText(reference)
  const hyp = normalizeText(hypothesis)
  const n = ref.length
  const m = hyp.length
  const d: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0))
  for (let i = 0; i <= n; i++) (d[i] as number[])[0] = i
  for (let j = 0; j <= m; j++) (d[0] as number[])[j] = j
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = ref[i - 1] === hyp[j - 1] ? 0 : 1
      ;(d[i] as number[])[j] = Math.min(
        (d[i - 1] as number[])[j]! + 1,
        (d[i] as number[])[j - 1]! + 1,
        (d[i - 1] as number[])[j - 1]! + cost,
      )
    }
  }
  const alignment: AlignStep[] = []
  let i = n
  let j = m
  let s = 0
  let del = 0
  let ins = 0
  while (i > 0 || j > 0) {
    const here = (d[i] as number[])[j]!
    if (i > 0 && j > 0 && ref[i - 1] === hyp[j - 1] && here === (d[i - 1] as number[])[j - 1]!) {
      alignment.push({ op: 'equal', ref: ref[i - 1], hyp: hyp[j - 1] })
      i--
      j--
    } else if (i > 0 && j > 0 && here === (d[i - 1] as number[])[j - 1]! + 1) {
      alignment.push({ op: 'substitute', ref: ref[i - 1], hyp: hyp[j - 1] })
      s++
      i--
      j--
    } else if (i > 0 && here === (d[i - 1] as number[])[j]! + 1) {
      alignment.push({ op: 'delete', ref: ref[i - 1] })
      del++
      i--
    } else {
      alignment.push({ op: 'insert', hyp: hyp[j - 1] })
      ins++
      j--
    }
  }
  alignment.reverse()
  const errors = s + del + ins
  return {
    wer: n === 0 ? (m === 0 ? 0 : 1) : errors / n,
    substitutions: s,
    deletions: del,
    insertions: ins,
    refWords: n,
    alignment,
  }
}
