// @vitest-environment node
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import * as uccharan from '@/vendor/uccharan/index.js'
import { CLONE_MODELS } from './models'
import {
  hasNumbers,
  NORMALISER_LANGUAGE,
  NOT_NORMALISED,
  numbersFact,
  spokenText,
  uccharanNormaliser,
  type Normaliser,
} from './spoken'

describe('spoken form for the cloning models (fake normaliser)', () => {
  const fake = vi.fn<Normaliser>(async (text, tag) => `${tag}: ${text.replace('10:30', 'ten  thirty')}`)

  it('writes the text out with the uccharan language of the page language', async () => {
    fake.mockClear()
    expect(await spokenText('At 10:30.', 'en', fake)).toEqual({ text: 'en-us: At ten thirty.', tag: 'en-us' })
    expect(await spokenText('Um 10:30.', 'no', fake)).toMatchObject({ tag: 'nb' })
    expect(fake).toHaveBeenCalledTimes(2)
  })

  it('leaves languages uccharan does not read as they are, without calling it', async () => {
    fake.mockClear()
    for (const l of NOT_NORMALISED)
      expect(await spokenText('10:30', l, fake)).toEqual({ text: '10:30', tag: null, skipped: 'language' })
    expect(fake).not.toHaveBeenCalled()
  })

  it('falls back to the text as written when the normaliser fails (its data could not be fetched)', async () => {
    const failing: Normaliser = () => Promise.reject(new Error('failed to fetch de.ucd: 404'))
    expect(await spokenText('Um 10:30.', 'de', failing)).toEqual({
      text: 'Um 10:30.',
      tag: null,
      skipped: 'failed',
      error: 'failed to fetch de.ucd: 404',
    })
  })

  it('keeps the text when the normaliser returns nothing', async () => {
    expect(await spokenText('25', 'en', async () => '  ')).toEqual({ text: '25', tag: 'en-us' })
  })

  it('says per model which languages get their numbers written out', () => {
    expect(numbersFact(['en'])).toMatch(/^written out in words by uccharan/)
    expect(numbersFact(['zh'])).toMatch(/as written/)
    const mixed = numbersFact(['en', 'de', 'zh', 'ja'])
    expect(mixed).toContain('in English, German;')
    expect(mixed).toContain('in Chinese, Japanese they go to the model as written')
    expect(numbersFact(['en', 'de'])).toContain('in all its languages')
  })

  it('spots digits, currency and other signs', () => {
    expect(hasNumbers('Hello there.')).toBe(false)
    for (const t of ['25', '10h30', '٣', '€', '50%', 'a/b']) expect(hasNumbers(t)).toBe(true)
  })
})

describe('language coverage', () => {
  it('every language of every cloning model is either written out or listed as left as written', () => {
    for (const m of CLONE_MODELS)
      for (const l of m.languages)
        expect(
          Boolean(NORMALISER_LANGUAGE[l]) !== (NOT_NORMALISED as readonly string[]).includes(l),
          `${m.id} ${l}`,
        ).toBe(true)
  })

  it('maps onto languages the vendored uccharan has, and the skipped ones are really missing', async () => {
    for (const tag of Object.values(NORMALISER_LANGUAGE)) expect(uccharan.LANGUAGES).toContain(tag)
    for (const l of NOT_NORMALISED)
      await expect(uccharan.normalize('25', { lang: l })).rejects.toThrow(/unknown language/)
  })
})

// The vendored uccharan in Node, on its own data files (the page fetches the published copies of them).
describe('spoken form with the vendored uccharan', () => {
  const real = uccharanNormaliser(uccharan)

  it('English: times, money, dates and ordinals in words, brand names kept whole', async () => {
    const s = await spokenText(
      'Your appointment on 3 May is at 10:30, costs $25.50 and is the 21st; pay with PayPal.',
      'en',
      real,
    )
    expect(s.text).toBe(
      'Your appointment on three May is at ten thirty, costs twenty five dollars and fifty cents and is the twenty first; pay with PayPal.',
    )
  })

  it('German: ordinals and dates (uccharan 0.6.3)', async () => {
    const s = await spokenText('Ihr Termin am 3. Mai um 10:30 Uhr kostet 25 € und ist der 21. Termin.', 'de', real)
    expect(s.text).toBe(
      'Ihr Termin am dritten Mai um zehn dreißig Uhr kostet fünfundzwanzig Euro und ist der einundzwanzigste Termin.',
    )
  })

  it('writes 25 € in words of the language, with no digits left, in every language it is used for', async () => {
    for (const [lang, tag] of Object.entries(NORMALISER_LANGUAGE)) {
      const s = await spokenText('25 €', lang, real)
      expect(s.tag, lang).toBe(tag)
      expect(s.text, lang).not.toMatch(/\p{N}/u)
      if (lang !== 'en') expect(s.text, lang).not.toMatch(/twenty/)
    }
  })

  it('leaves text without numbers alone', async () => {
    const text = 'Vielen Dank für Ihren Anruf. Ihr Termin am Dienstag um halb elf ist bestätigt.'
    expect((await spokenText(text, 'de', real)).text).toBe(text)
  })
})

describe('start-up', () => {
  it('the clone page and its helpers load uccharan only on demand', () => {
    const src = (f: string) => readFileSync(path.resolve(import.meta.dirname, f), 'utf8')
    for (const f of ['./spoken.ts', '../../routes/clone-browser.tsx', '../uccharan.ts'])
      expect(src(f), f).not.toMatch(/^import (?!type)[^\n]*vendor\/uccharan/m)
    expect(src('./spoken.ts')).not.toMatch(/^import [^\n]*@\/lib\/uccharan/m)
  })
})
