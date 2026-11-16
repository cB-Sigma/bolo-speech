import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { parseSpmModel } from './spm'
import { normalizeForSpm, SpmBpeTokenizer } from './spm-bpe'

/** A tiny SentencePiece BPE model, written as the protobuf SentencePiece itself writes. */
function model(
  pieces: [string, number, number][],
  opts: { normalizer?: string; byteFallback?: boolean; removeExtraWhitespaces?: boolean } = {},
): Uint8Array {
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
  varint(2, trainer)
  field(35, 0, trainer)
  varint(opts.byteFallback ? 1 : 0, trainer)
  bytes(2, trainer, out)
  const normalizer: number[] = []
  bytes(1, [...new TextEncoder().encode(opts.normalizer ?? 'nmt_nfkc')], normalizer)
  field(3, 0, normalizer)
  varint(1, normalizer)
  field(4, 0, normalizer)
  varint(opts.removeExtraWhitespaces === false ? 0 : 1, normalizer)
  bytes(3, normalizer, out)
  return Uint8Array.from(out)
}

// Scores rank the merges: the higher, the earlier it is applied.
const PIECES: [string, number, number][] = [
  ['<unk>', 0, 2],
  ['<s>', 0, 3],
  ['</s>', 0, 3],
  ['<|im_start|>', 0, 4],
  ['▁', -1, 1],
  ['a', -1, 1],
  ['b', -1, 1],
  ['c', -1, 1],
  ['ab', -2, 1],
  ['bc', -3, 1],
  ['▁ab', -4, 1],
  ['abc', -5, 1],
  ['fi', -6, 1],
  ['1', -7, 1],
  ['A', -8, 1],
]
const bytePieces: [string, number, number][] = Array.from({ length: 256 }, (_, b) => [
  `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`,
  0,
  6,
])
const id = (p: string) => PIECES.findIndex(([x]) => x === p)

describe('SentencePiece BPE tokenizer', () => {
  it('reads a BPE model only when asked to', () => {
    expect(() => parseSpmModel(model(PIECES))).toThrow(/unigram/)
    const m = parseSpmModel(model(PIECES), { bpe: true })
    expect(m.modelType).toBe(2)
    expect(m.normalizer).toBe('nmt_nfkc')
    expect(m.removeExtraWhitespaces).toBe(true)
  })

  it('merges the best-scoring pair first, then the leftmost of equal ones', () => {
    const t = SpmBpeTokenizer.fromModel(model(PIECES))
    // ▁ a b c: "ab" (-2) beats "bc" (-3); then "▁ab" (-4) beats "abc" (-5).
    expect(t.encode('abc')).toEqual([id('▁ab'), id('c')])
    expect(t.encode('bc')).toEqual([id('▁'), id('bc')])
    expect(t.count('abc abc')).toBe(4)
  })

  it('normalises as nmt_nfkc does: NFKC, one space between words, none at the ends', () => {
    const t = SpmBpeTokenizer.fromModel(model(PIECES))
    expect(normalizeForSpm('  ﬁ\t１  Ａ ', t.model)).toBe('fi 1 A')
    expect(t.encode('ﬁ')).toEqual([id('▁'), id('fi')])
    expect(t.encode('   ')).toEqual([])
    expect(normalizeForSpm(' a  b ', { normalizer: 'identity', removeExtraWhitespaces: false })).toBe(' a  b ')
  })

  it('keeps user-defined symbols whole', () => {
    const t = SpmBpeTokenizer.fromModel(model(PIECES))
    expect(t.encode('<|im_start|>ab')).toEqual([id('▁'), id('<|im_start|>'), id('ab')])
  })

  it('falls back to UTF-8 bytes for what the vocabulary lacks', () => {
    const t = SpmBpeTokenizer.fromModel(model([...PIECES, ...bytePieces], { byteFallback: true }))
    expect(t.encode('aé')).toEqual([id('▁'), id('a'), PIECES.length + 0xc3, PIECES.length + 0xa9])
  })

  // With BOLO_MOSS_TOKENIZER pointing at MOSS-TTS-Nano's tokenizer.model, compare with ids the reference
  // SentencePiece library gave (ui/scripts/clone/moss_reference.py --vectors writes the file next to it).
  const real = process.env.BOLO_MOSS_TOKENIZER
  const vectors = real && path.join(path.dirname(real), 'moss-tokenizer-vectors.json')
  it.runIf(real && vectors && existsSync(vectors))('matches SentencePiece on the real model', () => {
    const t = SpmBpeTokenizer.fromModel(readFileSync(real as string))
    const cases = JSON.parse(readFileSync(vectors as string, 'utf8')) as { text: string; ids: number[] }[]
    expect(cases.length).toBeGreaterThan(10)
    for (const c of cases) expect(t.encode(c.text), c.text).toEqual(c.ids)
  })
})
