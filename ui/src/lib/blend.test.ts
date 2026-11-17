import { describe, expect, it } from 'vitest'
import {
  blendString,
  canBlend,
  languageSummary,
  languageWarning,
  normalised,
  parseBlend,
  parseLanguageHeader,
} from './blend'

describe('voice blends', () => {
  it('spells a blend as the server does, with normalised weights', () => {
    expect(
      blendString([
        { voice: 'af_bella', weight: 7 },
        { voice: 'af_sky', weight: 3 },
      ]),
    ).toBe('af_bella(0.7)+af_sky(0.3)')
    expect(
      blendString([
        { voice: 'a', weight: 1 },
        { voice: 'b', weight: 2 },
      ]),
    ).toBe('a(0.333)+b(0.667)')
    expect(blendString([{ voice: 'a', weight: 0 }])).toBe('')
    expect(
      normalised([
        { voice: 'a', weight: 1 },
        { voice: 'a', weight: 1 },
        { voice: 'b', weight: NaN },
      ]),
    ).toEqual([{ voice: 'a', weight: 1 }])
  })

  it('reads a blend back', () => {
    expect(parseBlend('af_bella(0.7)+af_sky(0.3)')).toEqual([
      { voice: 'af_bella', weight: 0.7 },
      { voice: 'af_sky', weight: 0.3 },
    ])
    expect(parseBlend('af_bella + af_sky')).toEqual([
      { voice: 'af_bella', weight: 1 },
      { voice: 'af_sky', weight: 1 },
    ])
    expect(parseBlend('a(x)+b')).toBeNull()
    expect(parseBlend('a(-1)')).toBeNull()
  })

  it('knows which models blend', () => {
    expect(canBlend({ features: ['voice-blending'] })).toBe(true)
    expect(canBlend({ features: [] })).toBe(false)
    expect(canBlend(null)).toBe(false)
  })
})

describe('x-bolo-language', () => {
  it('reads a detected language and the voice chosen', () => {
    const d = parseLanguageHeader('hi-IN; source=detected; detected=hi; script=Devanagari; voice=hf_alpha')!
    expect(d).toMatchObject({ language: 'hi-IN', source: 'detected', detected: 'hi', voice: 'hf_alpha' })
    expect(languageSummary(d)).toBe('hi-IN, detected from the text (Devanagari), voice hf_alpha')
    expect(languageWarning(d)).toBeNull()
  })

  it('warns when a named voice speaks another language', () => {
    const d = parseLanguageHeader('en-US; source=voice; detected=hi; script=Devanagari; mismatch=true')!
    expect(d.mismatch).toBe(true)
    expect(languageWarning(d)).toContain('The text looks like Hindi')
  })

  it('says when the guess names a language the model does not offer', () => {
    const d = parseLanguageHeader('en-US; source=default; detected=ta; script=Tamil; offered=false')!
    expect(d.offered).toBe(false)
    expect(languageSummary(d)).toContain('which this model does not offer')
    expect(parseLanguageHeader('')).toBeNull()
  })
})

describe('synthesis headers', () => {
  it('keeps what the server reported', async () => {
    const { reportedFrom } = await import('./tts/session')
    const h = new Headers({
      'x-bolo-model': 'kokoro-v1.0-int8',
      'x-bolo-blend': 'af_bella(0.7)+af_sky(0.3)',
      'x-bolo-voice': 'warm-mix',
      'x-bolo-language': 'en-US; source=voice',
    })
    expect(reportedFrom(h)).toEqual({
      model: 'kokoro-v1.0-int8',
      blend: 'af_bella(0.7)+af_sky(0.3)',
      voice: 'warm-mix',
      language: 'en-US; source=voice',
      fallback: null,
    })
  })
})
