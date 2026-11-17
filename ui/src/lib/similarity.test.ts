import { describe, expect, it } from 'vitest'
import { cosine, verdictFor } from './similarity'

describe('speaker similarity', () => {
  it('computes the cosine of two embeddings', () => {
    expect(cosine([1, 0], [1, 0])).toBeCloseTo(1)
    expect(cosine([1, 0], [0, 2])).toBeCloseTo(0)
    expect(cosine([3, 4], [6, 8])).toBeCloseTo(1)
    expect(cosine([1], [1, 2])).toBeNull()
    expect(cosine([0, 0], [1, 2])).toBeNull()
  })

  it('reads a score with the default model thresholds', () => {
    expect(verdictFor(0.81)).toBe('same speaker')
    expect(verdictFor(0.4)).toBe('similar')
    expect(verdictFor(0.09)).toBe('different')
    expect(verdictFor(0.4, { same: 0.35, similar: 0.2 })).toBe('same speaker')
  })
})
