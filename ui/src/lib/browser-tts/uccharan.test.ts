// @vitest-environment node
import { describe, expect, it } from 'vitest'
import { LANGUAGES, normalize, phonemize } from '@/vendor/uccharan/index.js'

// The vendored uccharan, run in Node on its own data files (as the page runs it on the published copies).
describe('vendored uccharan', () => {
  it('reads numbers in German with German words, not English ones', async () => {
    const { text } = await normalize('Herr Nguyen zahlt 25 € für 3,5 kg Äpfel.', { lang: 'de' })
    expect(text).toContain('fünfundzwanzig Euro')
    expect(text).toContain('drei Komma fünf')
    expect(text).not.toMatch(/twenty|three|point/)
  })
  it('reads the Spanish decimal comma', async () => {
    expect((await normalize('Son 3,5 kg por 20 €.', { lang: 'es' })).text).toContain('tres coma cinco')
  })
  // Before 0.6.1 these asked for mai.ucd, bho.ucd and mag.ucd, which do not exist.
  it('reads Maithili, Bhojpuri and Magahi with the Hindi data', async () => {
    for (const lang of ['mai', 'bho', 'mag']) {
      expect((await normalize('काल्हि 25 तारीख', { lang })).text, lang).toContain('पच्चीस')
      expect(await phonemize('प्रणाम', { lang }), lang).toMatch(/ˈ/)
    }
  })
  it('knows Belgian and Swiss French', async () => {
    expect(LANGUAGES).toEqual(expect.arrayContaining(['fr-be', 'fr-ch']))
    expect((await normalize('Il a 70 ans.', { lang: 'fr-ch' })).text).toContain('septante')
  })
})
