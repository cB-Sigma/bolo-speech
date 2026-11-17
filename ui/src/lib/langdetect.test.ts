import { describe, expect, it } from 'vitest'
import cases from './fixtures/server-text-cases.json'
import { baseLanguage, detectLanguage } from './langdetect'

/** The server's own answers (scripts/parity/export_cases.py ran bolo.text.langdetect on each text). */
describe('language detection matches the server', () => {
  it('carries every case of the server tests, with the language and script they expect', () => {
    const fromTests = cases.langdetect.filter((c) => c.source === 'server-test')
    expect(fromTests.length).toBeGreaterThanOrEqual(40)
    for (const c of fromTests) {
      const expect_ = (c as { expect?: { language: string | null; script?: string } }).expect
      if (!expect_) continue
      expect(c.language, c.text).toBe(expect_.language)
      if (expect_.script) expect(c.script, c.text).toBe(expect_.script)
    }
  })

  for (const c of cases.langdetect) {
    it(`${c.source}: ${c.text.slice(0, 48)}`, () => {
      const d = detectLanguage(c.text)
      expect({
        language: d.language,
        script: d.script,
        confidence: Number(d.confidence.toFixed(6)),
        alternatives: d.alternatives,
        reason: d.reason,
      }).toEqual({
        language: c.language,
        script: c.script,
        confidence: c.confidence,
        alternatives: c.alternatives,
        reason: c.reason,
      })
    })
  }

  it('reads tags the way the server does', () => {
    expect(baseLanguage('hi-IN')).toBe('hi')
    expect(baseLanguage('en_US')).toBe('en')
    expect(baseLanguage('nb')).toBe('no')
    expect(baseLanguage(null)).toBe('')
  })
})
