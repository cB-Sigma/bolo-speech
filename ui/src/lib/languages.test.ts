import { describe, expect, it } from 'vitest'
import { diag, diagnostics, isDebug } from './diagnostics'
import { groupLanguages, isIndic, languageLabel, languageSearchText, supportsLanguage } from './languages'

describe('languages', () => {
  it('labels Indian languages with their own name first', () => {
    expect(languageLabel('hi')).toBe('हिन्दी · Hindi')
    expect(languageLabel('bn')).toBe('বাংলা · Bengali')
    expect(languageLabel('ta')).toBe('தமிழ் · Tamil')
    expect(languageLabel('ur')).toBe('اردو · Urdu')
    expect(languageLabel('or')).toBe('ଓଡ଼ିଆ · Odia')
    expect(languageLabel('hi-IN')).toMatch(/^हिन्दी · Hindi/)
    expect(languageLabel('fr')).toBe('French')
  })
  it('groups a model’s languages: Indian ones in a fixed order, then English, then the rest by name', () => {
    const g = groupLanguages(['fr', 'ta', 'en', 'multi', 'hi', 'de', 'as'])
    expect(g).toEqual({ indic: ['hi', 'ta', 'as'], english: ['en'], other: ['fr', 'de'] })
    expect(isIndic('mr')).toBe(true)
    expect(isIndic('en')).toBe(false)
  })
  it('knows what a model covers', () => {
    expect(supportsLanguage(['en'], 'hi')).toBe(false)
    expect(supportsLanguage(['multi'], 'hi')).toBe(true)
    expect(supportsLanguage(['hi', 'en'], 'hi-IN')).toBe(true)
    expect(supportsLanguage(['en'], 'auto')).toBe(true)
  })
  it('can be searched by code, English name or native name', () => {
    expect(languageSearchText('hi')).toContain('hindi')
    expect(languageSearchText('hi')).toContain('हिन्दी')
  })
})

describe('diagnostics', () => {
  it('reads ?debug=1 before or after the hash and remembers it for the tab', () => {
    sessionStorage.clear()
    expect(isDebug({ search: '', hash: '#/stt' })).toBe(false)
    expect(isDebug({ search: '?debug=1', hash: '#/stt' })).toBe(true)
    expect(isDebug({ search: '', hash: '#/models' })).toBe(true) // remembered
    expect(isDebug({ search: '?debug=0', hash: '' })).toBe(false)
    expect(isDebug({ search: '', hash: '#/stt?r=abc&debug=1' })).toBe(true)
    sessionStorage.clear()
    expect(isDebug({ search: '?nodebug=1', hash: '' })).toBe(false)
  })
  it('keeps a bounded record', () => {
    diagnostics.set({ fetches: [], errors: [], events: [] })
    for (let i = 0; i < 200; i++) diag.error('test', `e${i}`)
    expect(diagnostics.get().errors).toHaveLength(80)
    expect(diagnostics.get().errors[79]?.message).toBe('e199')
  })
})
