import { describe, expect, it } from 'vitest'
import { parseSpmModel, SpmTokenizer } from './spm'

/** A tiny SentencePiece model, written as the protobuf SentencePiece itself writes. */
function model(pieces: [string, number, number][], opts: { byteFallback?: boolean; type?: number } = {}): Uint8Array {
  const out: number[] = []
  const varint = (v: number, to: number[]) => {
    while (v >= 0x80) {
      to.push((v & 0x7f) | 0x80)
      v = Math.floor(v / 128)
    }
    to.push(v)
  }
  const field = (n: number, wire: number, to: number[]) => varint(n * 8 + wire, to)
  const bytes = (n: number, body: number[], to: number[]) => {
    field(n, 2, to)
    varint(body.length, to)
    to.push(...body)
  }
  for (const [piece, score, type] of pieces) {
    const body: number[] = []
    bytes(1, [...new TextEncoder().encode(piece)], body)
    field(2, 5, body)
    body.push(...new Uint8Array(Float32Array.of(score).buffer))
    field(3, 0, body)
    varint(type, body)
    bytes(1, body, out)
  }
  const trainer: number[] = []
  field(3, 0, trainer)
  varint(opts.type ?? 1, trainer)
  field(35, 0, trainer)
  varint(opts.byteFallback ? 1 : 0, trainer)
  field(40, 0, trainer)
  varint(0, trainer)
  bytes(2, trainer, out)
  const normalizer: number[] = []
  field(3, 0, normalizer)
  varint(1, normalizer)
  bytes(3, normalizer, out)
  return Uint8Array.from(out)
}

const BASE: [string, number, number][] = [
  ['<unk>', 0, 2],
  ['<s>', 0, 3],
  ['</s>', 0, 3],
  ['▁', -2, 1],
  ['▁hello', -1, 1],
  ['▁he', -3, 1],
  ['llo', -3, 1],
  ['h', -5, 1],
  ['e', -5, 1],
  ['l', -5, 1],
  ['o', -5, 1],
  ['.', -2, 1],
]

describe('SentencePiece unigram tokenizer', () => {
  it('reads the pieces and settings', () => {
    const m = parseSpmModel(model(BASE, { byteFallback: true }))
    expect(m.pieces).toHaveLength(BASE.length)
    expect(m.pieces[4]).toMatchObject({ piece: '▁hello', type: 1 })
    expect(m.pieces[4]?.score).toBeCloseTo(-1)
    expect(m.byteFallback).toBe(true)
    expect(m.addDummyPrefix).toBe(true)
  })

  it('picks the best-scoring segmentation (Viterbi), with the dummy prefix', () => {
    const t = SpmTokenizer.fromModel(model(BASE))
    expect(t.encode('hello.')).toEqual([4, 11])
    expect(t.encode('hello hello')).toEqual([4, 4])
    expect(t.decode(t.encode('hello hello.'))).toBe('hello hello.')
    // A leading space still gets the dummy prefix, as SentencePiece does.
    expect(t.encode(' hello')).toEqual([3, 4])
  })

  it('falls back to bytes for characters it does not know, or to one <unk>', () => {
    const bytes: [string, number, number][] = Array.from({ length: 256 }, (_, b) => [
      `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`,
      0,
      6,
    ])
    const withBytes = SpmTokenizer.fromModel(model([...BASE, ...bytes], { byteFallback: true }))
    const ids = withBytes.encode('hé')
    // 'é' is two UTF-8 bytes: C3 A9
    expect(ids.slice(-2)).toEqual([BASE.length + 0xc3, BASE.length + 0xa9])
    expect(withBytes.decode(ids)).toBe('hé')
    const plain = SpmTokenizer.fromModel(model(BASE))
    expect(plain.encode('hxyz')).toEqual([3, 7, 0])
  })

  it('refuses models that are not unigram', () => {
    expect(() => parseSpmModel(model(BASE, { type: 2 }))).toThrow(/unigram/)
  })
})
