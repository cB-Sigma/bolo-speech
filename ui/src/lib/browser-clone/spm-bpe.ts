import { parseSpmModel, type SpmModel } from './spm'

/**
 * A SentencePiece BPE tokenizer read from a `tokenizer.model`, as MOSS-TTS-Nano ships one: the `nmt_nfkc`
 * normaliser (NFKC plus SentencePiece's whitespace and control-character rules), extra whitespace removed, a
 * dummy prefix, user-defined symbols kept whole, then the highest-scoring adjacent pair merged until none is
 * left, and byte fallback for what the vocabulary lacks. It follows SentencePiece's bpe_model.cc; the ids
 * match the reference implementation on the sentences the tests and ui/scripts/clone/moss_reference.py check.
 */

const SPACE = '▁'

/** Characters SentencePiece's NMT rules turn into a space, and the control characters it drops. */
const NMT_SPACE = new RegExp(
  `[${[0x09, 0x0a, 0x0c, 0x0d, 0x1680, 0x200b, 0x200c, 0x200d, 0x200e, 0x200f, 0x2028, 0x2029, 0x2581, 0xfeff, 0xfffd].map((c) => String.fromCharCode(c)).join('')}]`,
  'g',
)
// eslint-disable-next-line no-control-regex -- these are the control characters SentencePiece drops
const NMT_DROP = /[\u0001-\u0008\u000b\u000e-\u001f\u007f\u008f\u009f]/g

/** Text as SentencePiece's normaliser gives it to the model (before spaces become ▁). */
export function normalizeForSpm(text: string, model: Pick<SpmModel, 'normalizer' | 'removeExtraWhitespaces'>): string {
  let t = text
  if (model.normalizer.startsWith('nmt')) t = t.replace(NMT_DROP, '').replace(NMT_SPACE, ' ')
  if (model.normalizer.includes('nfkc')) t = t.normalize('NFKC')
  if (model.removeExtraWhitespaces) t = t.replace(/ +/g, ' ').replace(/^ | $/g, '')
  return t
}

export class SpmBpeTokenizer {
  readonly model: SpmModel
  /** Pieces a merge may produce (normal, user-defined and unused ones). */
  private readonly pieces = new Map<string, number>()
  private readonly userDefined: string[]
  private readonly bytes = new Map<number, number>()

  constructor(model: SpmModel) {
    this.model = model
    model.pieces.forEach((p, i) => {
      if (p.type === 1 || p.type === 4 || p.type === 5) this.pieces.set(p.piece, i)
      if (p.type === 6) {
        const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(p.piece)
        if (m) this.bytes.set(parseInt(m[1] as string, 16), i)
      }
    })
    // Longest first, so the longest user-defined symbol at a position wins.
    this.userDefined = model.pieces
      .filter((p) => p.type === 4)
      .map((p) => p.piece)
      .sort((a, b) => b.length - a.length)
  }

  static fromModel(data: ArrayBuffer | Uint8Array): SpmBpeTokenizer {
    const model = parseSpmModel(data, { bpe: true })
    if (model.modelType !== 2) throw new Error('Not a BPE SentencePiece model')
    return new SpmBpeTokenizer(model)
  }

  /** Ids for `text` (no BOS/EOS), as SentencePiece's Encode gives them. */
  encode(text: string): number[] {
    let norm = normalizeForSpm(text, this.model)
    if (!norm) return []
    norm = norm.replace(/ /g, SPACE)
    if (this.model.addDummyPrefix) norm = SPACE + norm
    // Symbols: user-defined pieces whole (frozen), every other character on its own.
    const sym: { piece: string; freeze: boolean }[] = []
    for (let i = 0; i < norm.length;) {
      const user = this.userDefined.find((u) => norm.startsWith(u, i))
      if (user) {
        sym.push({ piece: user, freeze: true })
        i += user.length
        continue
      }
      const ch = String.fromCodePoint(norm.codePointAt(i) as number)
      sym.push({ piece: ch, freeze: false })
      i += ch.length
    }
    // Merge the best pair (highest score, then leftmost) until no adjacent pair is a piece. Symbols keep the index
    // of their first character, as SentencePiece's agenda does.
    const score = (a: number, b: number): number | null => {
      const x = sym[a] as { piece: string; freeze: boolean }
      const y = sym[b] as { piece: string; freeze: boolean }
      if (x.freeze || y.freeze) return null
      const id = this.pieces.get(x.piece + y.piece)
      return id == null ? null : (this.model.pieces[id]?.score ?? 0)
    }
    for (;;) {
      let best = -1
      let bestScore = -Infinity
      for (let i = 0; i + 1 < sym.length; i++) {
        const s = score(i, i + 1)
        if (s != null && s > bestScore) {
          best = i
          bestScore = s
        }
      }
      if (best < 0) break
      const left = sym[best] as { piece: string; freeze: boolean }
      left.piece += (sym[best + 1] as { piece: string }).piece
      sym.splice(best + 1, 1)
    }
    const out: number[] = []
    for (const s of sym) {
      const id = this.pieces.get(s.piece)
      if (id != null) out.push(id)
      else if (this.model.byteFallback)
        for (const b of new TextEncoder().encode(s.piece)) out.push(this.bytes.get(b) ?? this.model.unkId)
      else if (out[out.length - 1] !== this.model.unkId) out.push(this.model.unkId)
    }
    return out
  }

  /** Number of ids `text` encodes to. */
  count(text: string): number {
    return this.encode(text).length
  }
}
