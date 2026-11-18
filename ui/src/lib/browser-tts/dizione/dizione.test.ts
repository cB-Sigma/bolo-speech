/**
 * The TypeScript port of dizione against dizione itself: the token ids and stress Python's dizione gives for 306
 * texts (fixtures/cases.json, written by ui/scripts/parity/dizione_cases.py) must come out exactly the same. The
 * fixture carries the part of the phonemiser data the corpus reaches and the unknown-word model's answers; with
 * BOLO_DIZIONE_DATA pointing at the voice's phonemizer/ folder the model and the full data are checked too.
 */
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { Phonemizer, type DizioneData } from './engine'
import { G2P, readNpz, type G2PSettings } from './g2p'
import type { HomographData } from './homographs'
import type { AlignTable } from './align'
import { Normalizer, cardinal, numberWords } from './normalize'
import fixture from './fixtures/cases.json'

interface Case {
  text: string
  ids?: number[]
  stress?: number[]
  error?: string
}

const cases = fixture.cases as Case[]
const predictions = fixture.predictions as Record<string, string[]>

/** The unknown-word model as the fixture recorded its answers. */
class RecordedModel extends G2P {
  override predict(word: string): string[] {
    const found = predictions[word]
    if (!found) throw new Error(`the fixture has no model answer for '${word}'`)
    return [...found]
  }
}

function fixtureData(): DizioneData {
  return {
    lexicon: Object.entries(fixture.lexicon as Record<string, string>)
      .map(([word, phones]) => `${word}\t${phones}`)
      .join('\n'),
    align: fixture.align as AlignTable,
    homographs: fixture.homographs as unknown as HomographData,
    g2p: new RecordedModel(fixture.g2p as G2PSettings, new Map()),
  }
}

function check(phonemizer: Phonemizer): string[] {
  const wrong: string[] = []
  for (const c of cases) {
    try {
      const out = phonemizer.phonemize(c.text)
      if (c.error) wrong.push(`${c.text}: expected ${c.error}`)
      else if (
        JSON.stringify(out.ids) !== JSON.stringify(c.ids) ||
        JSON.stringify(out.stress) !== JSON.stringify(c.stress)
      )
        wrong.push(`${c.text}\n  got      ${out.symbols.join(' ')}\n  ids      ${out.ids}\n  expected ${c.ids}`)
    } catch (error) {
      if (!c.error) wrong.push(`${c.text}: ${String(error)}`)
    }
  }
  return wrong
}

describe('dizione in TypeScript', () => {
  it('gives the tokens and stress Python gives, for every text of the corpus', () => {
    expect(cases.length).toBeGreaterThan(300)
    expect(check(new Phonemizer(fixtureData()))).toEqual([])
  })

  it('says numbers as Python does, also past the safe integers', () => {
    expect(cardinal(1234)).toBe('one thousand two hundred thirty four')
    expect(numberWords('123456789012345678')).toBe(
      'one hundred twenty three quadrillion four hundred fifty six trillion seven hundred eighty nine billion twelve million three hundred forty five thousand six hundred seventy eight',
    )
    expect(numberWords('3.14')).toBe('three point one four')
    expect(numberWords('007')).toBe('zero zero seven')
  })

  it('reads the tables shipped with dizione', () => {
    const normalizer = new Normalizer()
    expect(normalizer.units.get('km')).toEqual(['kilometer', 'kilometers'])
    expect(normalizer.chat.size).toBeGreaterThan(50)
  })

  const directory = process.env.BOLO_DIZIONE_DATA
  const given = !!directory && existsSync(join(directory, 'g2p.npz'))
  it.skipIf(!given)('runs the unknown-word model and the full data as Python does (BOLO_DIZIONE_DATA)', () => {
    const read = (name: string): string => readFileSync(join(directory!, name), 'utf-8')
    const model = new G2P(
      JSON.parse(read('g2p.json')) as G2PSettings,
      readNpz(readFileSync(join(directory!, 'g2p.npz'))),
    )
    for (const [spelling, expected] of Object.entries(predictions))
      expect(model.predict(spelling), spelling).toEqual(expected)
    const phonemizer = new Phonemizer({
      lexicon: read('lexicon.tsv'),
      align: JSON.parse(read('align.json')) as AlignTable,
      homographs: JSON.parse(read('homographs.json')) as HomographData,
      g2p: model,
    })
    expect(check(phonemizer)).toEqual([])
  })
})
