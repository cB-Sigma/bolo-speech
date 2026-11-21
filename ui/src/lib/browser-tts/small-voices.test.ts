import { describe, expect, it } from 'vitest'
import { kaleIds, kittenFold, kittenIds, kittenTrim, kokoroNative, termsReplacer } from './small-voices'
import { BROWSER_VOICES, browserVoice } from './voices'

describe('small voices in the browser', () => {
  it('offers little-tts, Paradee, Kitten Pico and kale with pinned files and their licences', () => {
    const little = browserVoice('little-tts-6097')
    expect(little).toMatchObject({ kind: 'tiny-voca', gender: 'male', phonemizer: 'dizione' })
    expect(little.files).toMatchObject({ from: 'tiny-voca', speaker: '6097' })
    expect(little.license).toMatch(/NC/)
    expect(browserVoice('paradee-bf-isabella')).toMatchObject({ kind: 'paradee', phonemeLanguage: 'en-gb' })
    const priya = browserVoice('kitten-pico-hi-priya')
    expect(priya).toMatchObject({ kind: 'kitten', language: 'hi', phonemeLanguage: 'hi' })
    expect(priya.approximate).toMatch(/Approximate/)
    expect(browserVoice('kitten-pico-en-otis').approximate).toBeUndefined()
    const jack = browserVoice('kale-jack')
    expect(jack).toMatchObject({ kind: 'kale', phonemizer: 'kale', phonemeLanguage: 'en-gb', gender: 'male' })
    expect(BROWSER_VOICES.filter((v) => v.kind === 'kale')).toHaveLength(10)
  })

  it('Kitten: folds, its end token and the trimming of its output', () => {
    expect(kittenFold(' ɫa̪¿ ')).toBe('la')
    expect(kittenIds([0, 5, 6, 0])).toEqual([0, 5, 6, 10, 0])
    const wave = new Float32Array(1000).fill(1)
    const first = kittenTrim(wave, [3, 4, 3], true) // 100 samples per frame: the end token's last 2 frames go
    const later = kittenTrim(wave, [3, 4, 3], false) // and the start pause's 2 frames
    expect(first.length).toBe(800)
    expect(later.length).toBe(600)
    expect(first[0]).toBe(0)
    expect(first[400]).toBe(1)
  })

  it('kale: IPA made Kokoro-native, ids, and its technical terms', () => {
    expect(kokoroNative([...'oʊ ɾɚ bʌtən.'], false).join('')).toBe('O Təɹ bʌtᵊn.')
    expect(kokoroNative([...'əʊ'], true).join('')).toBe('Q')
    expect(kaleIds('ab?', { a: 1, b: 2 }, false)).toEqual([1, 2])
    const terms = termsReplacer(new Map([['Node.js', 'Node dot J S']]))
    expect(terms('Use Node.js and myVariable')).toBe('Use Node dot J S and my Variable')
  })
})
