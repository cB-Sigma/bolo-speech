import { describe, expect, it } from 'vitest'
import { finished, limitPeak, shorter, TINY_VOCA_BASE } from './tiny-voca'
import { browserVoice, BROWSER_VOICES } from './voices'

describe('tiny-voca in the browser', () => {
  it('adds a full stop where the reference does', () => {
    expect(finished('Hello there')).toBe('Hello there.')
    expect(finished('Hello there!')).toBe('Hello there!')
    expect(finished('She said "go!"')).toBe('She said "go!"')
    expect(finished('see you :)')).toBe('see you :).')
    expect(finished('wait,')).toBe('wait,')
    expect(finished('  ')).toBe('  ')
  })

  it('cuts a long utterance after commas, then between words', () => {
    expect(shorter('one, two; three')).toEqual(['one,', 'two;', 'three'])
    expect(shorter('a b c d e')).toEqual(['a b c', 'd e'])
    expect(shorter('word')).toEqual([])
  })

  it('scales the audio down only when it peaks above 0.99', () => {
    expect(limitPeak(Float32Array.from([0.5, -0.5]))).toEqual(Float32Array.from([0.5, -0.5]))
    const loud = limitPeak(Float32Array.from([1.1, -0.55]))
    expect(Math.max(...loud.map(Math.abs))).toBeCloseTo(0.99, 5)
  })

  it('offers F2 and M2, with pinned files and the licence', () => {
    const voices = BROWSER_VOICES.filter((v) => v.id.startsWith('tiny-voca-'))
    expect(voices.map((v) => v.id)).toEqual(['tiny-voca-f2', 'tiny-voca-m2'])
    const m2 = browserVoice('tiny-voca-m2')
    expect(m2.gender).toBe('male')
    expect(m2.license).toBe('Apache-2.0')
    expect(m2.phonemizer).toBe('dizione')
    expect(m2.files).toEqual({
      from: 'tiny-voca',
      model: `${TINY_VOCA_BASE}onnx/model.onnx`,
      config: `${TINY_VOCA_BASE}config.json`,
    })
    expect(TINY_VOCA_BASE).toMatch(/\/resolve\/[0-9a-f]{40}\/$/)
  })
})
