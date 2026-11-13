import { describe, expect, it } from 'vitest'
import { normalizeText, wordErrorRate } from './wer'

describe('normalizeText', () => {
  it('lower-cases and strips punctuation but keeps apostrophes', () => {
    expect(normalizeText("Hello, World! It's 9:45.")).toEqual(['hello', 'world', "it's", '9', '45'])
  })
  it('keeps Devanagari words and their vowel signs intact', () => {
    expect(normalizeText('नमस्ते, आप कैसे हैं?')).toEqual(['नमस्ते', 'आप', 'कैसे', 'हैं'])
  })
})

describe('wordErrorRate', () => {
  it('is zero for identical text regardless of case and punctuation', () => {
    expect(wordErrorRate('Ask not what your country can do.', 'ask not what your country can do').wer).toBe(0)
  })
  it('counts substitutions, deletions and insertions', () => {
    const r = wordErrorRate('the quick brown fox jumps', 'the quick brown dog jumps high')
    expect(r).toMatchObject({ substitutions: 1, deletions: 0, insertions: 1, refWords: 5 })
    expect(r.wer).toBeCloseTo(2 / 5)
    const d = wordErrorRate('one two three four', 'one four')
    expect(d).toMatchObject({ substitutions: 0, deletions: 2, insertions: 0 })
    expect(d.wer).toBe(0.5)
  })
  it('produces an alignment that replays both sides', () => {
    const r = wordErrorRate('a b c d', 'a x c e f')
    expect(r.alignment.filter((s) => s.ref).map((s) => s.ref)).toEqual(['a', 'b', 'c', 'd'])
    expect(r.alignment.filter((s) => s.hyp).map((s) => s.hyp)).toEqual(['a', 'x', 'c', 'e', 'f'])
  })
  it('handles empty sides', () => {
    expect(wordErrorRate('', '').wer).toBe(0)
    expect(wordErrorRate('', 'extra words').wer).toBe(1)
    expect(wordErrorRate('two words', '').wer).toBe(1)
  })
  it('can exceed 100% when the hypothesis is much longer', () => {
    expect(wordErrorRate('hi', 'well hello there you').wer).toBeGreaterThan(1)
  })
})
