import { describe, expect, it } from 'vitest'
import { ctcWords, tokensById } from './ctc-words'

const VOCAB = ['<pad>', '|', 'a', 'b', 'l']

/** Frame scores where each frame's token is `path[t]` (an index into VOCAB). */
function logits(path: number[]): Float32Array {
  const out = new Float32Array(path.length * VOCAB.length).fill(-5)
  path.forEach((id, t) => (out[t * VOCAB.length + id] = 5))
  return out
}

describe('word times from CTC frames', () => {
  it('merges repeats, keeps letters split by a blank and cuts words at the delimiter', () => {
    // b a l _ l | | a b b
    const path = [0, 3, 2, 4, 0, 4, 1, 1, 2, 3, 3, 0]
    const words = ctcWords(logits(path), path.length, VOCAB.length, (id) => VOCAB[id] ?? '', 0, 0.02)
    expect(words.map((w) => w.w)).toEqual(['ball', 'ab'])
    expect(words[0]?.t0).toBeCloseTo(0.02)
    expect(words[0]?.t1).toBeCloseTo(0.12)
    expect(words[1]?.t0).toBeCloseTo(0.16)
    expect(words[1]?.t1).toBeCloseTo(0.22)
    expect(words[0]?.p).toBeGreaterThan(0.9)
  })

  it('reads a tokenizer vocabulary in any of its forms', () => {
    expect(tokensById({ model: { vocab: ['<pad>', 'a'] } })).toEqual(['<pad>', 'a'])
    expect(
      tokensById({
        get_vocab: () =>
          new Map([
            ['a', 1],
            ['<pad>', 0],
          ]),
      }),
    ).toEqual(['<pad>', 'a'])
    expect(tokensById({ model: { vocab: { a: 1, '<pad>': 0 } } })).toEqual(['<pad>', 'a'])
    expect(tokensById(undefined)).toEqual([])
  })
})
