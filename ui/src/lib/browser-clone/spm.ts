/**
 * A SentencePiece unigram tokenizer read straight from a `tokenizer.model`
 * file (the protobuf SentencePiece writes), for the voice-cloning models
 * that ship one. It covers what those models use: the identity normaliser,
 * a dummy prefix, and byte fallback for characters the vocabulary lacks.
 * The ids match the reference implementation on the sentences the tests check.
 */

interface Piece {
  piece: string
  score: number
  /** 1 normal, 2 unknown, 3 control, 4 user defined, 5 unused, 6 byte. */
  type: number
}

export interface SpmModel {
  pieces: Piece[]
  unkId: number
  addDummyPrefix: boolean
  byteFallback: boolean
}

class Reader {
  pos = 0
  readonly buf: Uint8Array
  constructor(buf: Uint8Array) {
    this.buf = buf
  }
  varint(): number {
    let result = 0
    let shift = 0
    for (;;) {
      const b = this.buf[this.pos++] as number
      result += (b & 0x7f) * 2 ** shift
      if (b < 0x80) return result
      shift += 7
    }
  }
  /** Skip a field of this wire type. */
  skip(wire: number): void {
    if (wire === 0) this.varint()
    else if (wire === 1) this.pos += 8
    else if (wire === 2) this.pos += this.varint()
    else if (wire === 5) this.pos += 4
    else throw new Error(`Unsupported protobuf wire type ${wire}`)
  }
}

const utf8 = new TextDecoder()

function readPiece(buf: Uint8Array, start: number, end: number): Piece {
  const r = new Reader(buf)
  r.pos = start
  const p: Piece = { piece: '', score: 0, type: 1 }
  while (r.pos < end) {
    const tag = r.varint()
    const field = Math.floor(tag / 8)
    const wire = tag & 7
    if (field === 1 && wire === 2) {
      const len = r.varint()
      p.piece = utf8.decode(buf.subarray(r.pos, r.pos + len))
      r.pos += len
    } else if (field === 2 && wire === 5) {
      p.score = new DataView(buf.buffer, buf.byteOffset + r.pos, 4).getFloat32(0, true)
      r.pos += 4
    } else if (field === 3 && wire === 0) p.type = r.varint()
    else r.skip(wire)
  }
  return p
}

/** Fields of a sub-message as (field, wire, start, end) for length-delimited ones, (field, value) for varints. */
function scan(
  buf: Uint8Array,
  start: number,
  end: number,
  visit: (field: number, wire: number, value: number, at: number) => void,
) {
  const r = new Reader(buf)
  r.pos = start
  while (r.pos < end) {
    const tag = r.varint()
    const field = Math.floor(tag / 8)
    const wire = tag & 7
    if (wire === 2) {
      const len = r.varint()
      visit(field, wire, len, r.pos)
      r.pos += len
    } else if (wire === 0) visit(field, wire, r.varint(), r.pos)
    else r.skip(wire)
  }
}

/** Parse a SentencePiece `tokenizer.model`. */
export function parseSpmModel(data: ArrayBuffer | Uint8Array): SpmModel {
  const buf = data instanceof Uint8Array ? data : new Uint8Array(data)
  const model: SpmModel = { pieces: [], unkId: 0, addDummyPrefix: true, byteFallback: false }
  scan(buf, 0, buf.length, (field, wire, value, at) => {
    if (wire !== 2) return
    if (field === 1) model.pieces.push(readPiece(buf, at, at + value))
    else if (field === 2)
      // TrainerSpec: unk_id = 40, byte_fallback = 35, model_type = 3
      scan(buf, at, at + value, (f, w, v) => {
        if (w !== 0) return
        if (f === 40) model.unkId = v
        if (f === 35) model.byteFallback = v !== 0
        if (f === 3 && v !== 1) throw new Error('Only unigram SentencePiece models are supported')
      })
    else if (field === 3)
      // NormalizerSpec: add_dummy_prefix = 3
      scan(buf, at, at + value, (f, w, v) => {
        if (w === 0 && f === 3) model.addDummyPrefix = v !== 0
      })
  })
  if (!model.pieces.length) throw new Error('No pieces in the SentencePiece model')
  return model
}

const SPACE = '▁'

export class SpmTokenizer {
  private readonly vocab = new Map<string, number>()
  private readonly bytes = new Map<number, number>()
  private readonly maxLen: number
  private readonly unkScore: number

  readonly model: SpmModel

  constructor(model: SpmModel) {
    this.model = model
    let min = 0
    let maxLen = 1
    model.pieces.forEach((p, i) => {
      if (p.type === 1 || p.type === 4) {
        this.vocab.set(p.piece, i)
        maxLen = Math.max(maxLen, [...p.piece].length)
        min = Math.min(min, p.score)
      } else if (p.type === 6) {
        const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(p.piece)
        if (m) this.bytes.set(parseInt(m[1] as string, 16), i)
      }
    })
    this.maxLen = maxLen
    this.unkScore = min - 10
  }

  static fromModel(data: ArrayBuffer | Uint8Array): SpmTokenizer {
    return new SpmTokenizer(parseSpmModel(data))
  }

  /** Ids for `text` (no BOS/EOS), as SentencePiece's Encode gives them. */
  encode(text: string): number[] {
    if (!text) return []
    let norm = text.replace(/ /g, SPACE)
    if (this.model.addDummyPrefix) norm = SPACE + norm
    const chars = [...norm]
    const n = chars.length
    // Viterbi over character positions.
    const best = new Float64Array(n + 1).fill(-Infinity)
    const from = new Int32Array(n + 1).fill(-1)
    const ids = new Int32Array(n + 1).fill(-1)
    best[0] = 0
    for (let i = 0; i < n; i++) {
      if (best[i] === -Infinity) continue
      const base = best[i] as number
      let s = ''
      let matched = false
      for (let len = 1; len <= this.maxLen && i + len <= n; len++) {
        s += chars[i + len - 1]
        const id = this.vocab.get(s)
        if (id == null) continue
        if (len === 1) matched = true
        const score = base + (this.model.pieces[id] as Piece).score
        if (score > (best[i + len] as number)) {
          best[i + len] = score
          from[i + len] = i
          ids[i + len] = id
        }
      }
      if (!matched) {
        const score = base + this.unkScore
        if (score > (best[i + 1] as number)) {
          best[i + 1] = score
          from[i + 1] = i
          ids[i + 1] = -1
        }
      }
    }
    const out: number[][] = []
    for (let at = n; at > 0; at = from[at] as number) {
      const id = ids[at] as number
      if (id >= 0) out.push([id])
      else {
        const ch = chars[from[at] as number] as string
        if (this.model.byteFallback && this.bytes.size)
          out.push([...new TextEncoder().encode(ch)].map((b) => this.bytes.get(b) ?? this.model.unkId))
        else out.push([this.model.unkId])
      }
    }
    // SentencePiece merges consecutive unknowns into one <unk> when there is no byte fallback.
    const flat = out.reverse().flat()
    if (this.model.byteFallback) return flat
    return flat.filter((id, i) => !(id === this.model.unkId && flat[i - 1] === this.model.unkId))
  }

  decode(ids: number[]): string {
    const bytes: number[] = []
    let text = ''
    const flush = () => {
      if (bytes.length) text += utf8.decode(new Uint8Array(bytes.splice(0)))
    }
    for (const id of ids) {
      const p = this.model.pieces[id]
      if (!p || p.type === 3) continue
      if (p.type === 6) {
        const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(p.piece)
        if (m) bytes.push(parseInt(m[1] as string, 16))
        continue
      }
      flush()
      text += p.type === 2 ? ' ⁇ ' : p.piece
    }
    flush()
    const out = text.replace(new RegExp(SPACE, 'g'), ' ')
    return this.model.addDummyPrefix && out.startsWith(' ') ? out.slice(1) : out
  }
}
