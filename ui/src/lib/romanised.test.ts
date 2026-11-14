import { describe, expect, it } from 'vitest'

import type { ServerInfo } from '@/lib/api/types'

import { isIndicLanguage, romanisedBlocked, romanisedOffered, romanisedSummary } from './romanised'

const info = (romanised?: object): ServerInfo => ({ capabilities: romanised ? { romanised } : {} }) as ServerInfo

describe('romanised text', () => {
  it('knows which voices it applies to', () => {
    expect(isIndicLanguage('hi')).toBe(true)
    expect(isIndicLanguage('hi-IN')).toBe(true)
    expect(isIndicLanguage('ta_IN')).toBe(true)
    expect(isIndicLanguage('en')).toBe(false)
    expect(isIndicLanguage(null)).toBe(false)
  })

  it('is offered only by a server that announces it', () => {
    expect(romanisedOffered(undefined)).toBeNull()
    expect(romanisedOffered(info())).toBeNull()
    expect(romanisedOffered(info({ available: true }))).toEqual({ available: true })
  })

  it('says why a mode would do nothing', () => {
    expect(romanisedBlocked(info(), 'hi')).toMatch(/no romanised-text option/)
    expect(romanisedBlocked(info({ available: false, reason: 'pip install -U uccharan' }), 'hi')).toBe(
      'pip install -U uccharan',
    )
    const cap = { available: true, languages: ['hi', 'mr'], installed: ['hi'] }
    expect(romanisedBlocked(info(cap), 'hi-IN')).toBeNull()
    expect(romanisedBlocked(info(cap), 'mr')).toMatch(/uccharan-xlit-mr/)
    expect(romanisedBlocked(info(cap), 'ta')).toMatch(/no romanised-text pack for ta/)
  })

  it('summarises a trace record', () => {
    expect(romanisedSummary(null)).toBeNull()
    expect(
      romanisedSummary({ mode: 'auto', source: 'request', language: 'hi', applied: true, words: 4, text: 'मुझे' }),
    ).toBe("Romanised text: 4 words written in the voice's script (automatic, asked for by the request)")
    expect(romanisedSummary({ mode: 'on', source: 'voice', language: 'hi', applied: true, words: 1 })).toBe(
      "Romanised text: 1 word written in the voice's script (always, the voice's setting)",
    )
    expect(
      romanisedSummary({ mode: 'auto', source: 'voice', language: 'hi', applied: false, notice: 'not converted: x' }),
    ).toBe('Romanised text (automatic): not converted: x')
    expect(romanisedSummary({ mode: 'on', source: 'voice', language: 'hi', applied: false })).toBe(
      'Romanised text (always): nothing to convert',
    )
  })
})
