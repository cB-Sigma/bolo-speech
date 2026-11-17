import { describe, expect, it } from 'vitest'
import cases from './fixtures/server-text-cases.json'
import { RepeatGuard, trimRepeats } from './repeat-guard'

/** The server's own answers (scripts/parity/export_cases.py ran bolo.stt_options on each case). */
describe('the repetition guard matches the server', () => {
  for (const c of cases.trim) {
    it(`trims ${JSON.stringify(c.text.slice(0, 40))} (max_repeats ${c.max_repeats})`, () => {
      expect(trimRepeats(c.text, c.max_repeats)).toEqual({ text: c.out, removed: c.removed })
    })
  }

  for (const c of cases.live) {
    it(`guards live finals ${JSON.stringify(c.finals[0])}… (max_repeats ${c.max_repeats})`, () => {
      const guard = new RepeatGuard(c.max_repeats)
      expect(c.finals.map((f) => guard.apply(f))).toEqual(c.out)
      expect(guard.removed).toBe(c.removed)
    })
  }

  it('leaves text alone when nothing repeats', () => {
    expect(trimRepeats('the quick brown fox')).toEqual({ text: 'the quick brown fox', removed: 0 })
  })
})
