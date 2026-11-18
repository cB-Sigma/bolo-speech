/**
 * simple-sentencepiece 0.7 (`ssentencepiece`, Apache-2.0), the unigram tokenizer sherpa-onnx gives its punctuation
 * models, in TypeScript: the same cut of the same UTF-8 bytes, so the page feeds the model the ids the server does.
 *
 * Each word gets `▁` in front, then the bytes are cut into vocabulary pieces: the cut with the highest sum of the
 * pieces' scores, summed in float32 from the end, the shorter first piece on a tie, `<unk>` for a byte no piece
 * starts with. The vocabulary is `bpe.vocab` (one piece and its score per line).
 */

const MARK = new Uint8Array([0xe2, 0x96, 0x81]) // ▁
const encoder = new TextEncoder()

/** What `std::istream >> std::string` splits on in the "C" locale: ASCII white space only. */
export const C_SPACE = /[ \t\n\v\f\r]+/

/** The words of `text` as sherpa-onnx sees them. */
export function cWords(text: string): string[] {
  return text.split(C_SPACE).filter((w) => w.length > 0)
}

interface TrieNode {
  next: Map<number, TrieNode>
  id: number
}

export class Ssentencepiece {
  readonly size: number
  private readonly scores: Float32Array
  private readonly root: TrieNode = { next: new Map(), id: -1 }
  private readonly unk: number

  constructor(pieces: string[], scores: number[]) {
    this.size = pieces.length
    this.scores = Float32Array.from(scores)
    let unk = 0
    pieces.forEach((piece, id) => {
      if (piece === '<unk>') unk = id
      let node = this.root
      for (const b of encoder.encode(piece)) {
        let child = node.next.get(b)
        if (!child) {
          child = { next: new Map(), id: -1 }
          node.next.set(b, child)
        }
        node = child
      }
      if (node.id < 0) node.id = id
    })
    this.unk = unk
  }

  /** From the text of `bpe.vocab`. */
  static fromVocab(text: string): Ssentencepiece {
    const pieces: string[] = []
    const scores: number[] = []
    for (const line of text.split('\n')) {
      const parts = line.split(C_SPACE).filter(Boolean)
      if (parts.length === 0) continue
      if (parts.length < 2) throw new Error(`bpe.vocab: a line without a score: ${line}`)
      pieces.push(parts[0] as string)
      scores.push(Number(parts[1]))
    }
    return new Ssentencepiece(pieces, scores)
  }

  /** Piece ids of `text` (its words each with `▁` in front, joined). */
  encode(text: string): number[] {
    const words = cWords(text).map((w) => encoder.encode(w))
    const s = new Uint8Array(words.reduce((a, w) => a + MARK.length + w.length, 0))
    let at = 0
    for (const w of words) {
      s.set(MARK, at)
      s.set(w, at + MARK.length)
      at += MARK.length + w.length
    }
    const n = s.length
    // route[i]: best score from i to the end, where its first piece ends (-1: none starts here) and which piece
    const score = new Float32Array(n + 1)
    const end = new Int32Array(n + 1)
    const piece = new Int32Array(n + 1)
    for (let i = n - 1; i >= 0; i--) {
      let best = -Infinity
      let bestEnd = -1
      let bestId = 0
      let node: TrieNode | undefined = this.root
      for (let j = i; j < n && node; j++) {
        node = node.next.get(s[j] as number)
        if (!node || node.id < 0) continue
        const e = j + 1
        const sc = Math.fround((this.scores[node.id] as number) + (score[e] as number))
        if (sc > best || (sc === best && bestEnd >= e)) {
          best = sc
          bestEnd = e
          bestId = node.id
        }
      }
      score[i] = best === -Infinity ? 0 : best
      end[i] = bestEnd
      piece[i] = bestId
    }
    const ids: number[] = []
    for (let i = 0; i < n;) {
      if ((end[i] as number) < 0) {
        ids.push(this.unk)
        i += 1
      } else {
        ids.push(piece[i] as number)
        i = end[i] as number
      }
    }
    return ids
  }
}
