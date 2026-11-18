/** Sentence rules: what a word's pronunciation owes to its neighbours (`dizione/sentence.py`). */

import { VOWEL_SET, type Phone, type Phones } from './symbols'

const SPLIT: Record<string, string> = { ɑːɹ: 'ɑː', ɔːɹ: 'ɔː', oːɹ: 'oː', ɪɹ: 'ɪ', ɛɹ: 'ɛ', ʊɹ: 'ʊ' }
const RHOTIC = new Set(['ɚ', 'ɜː', 'aɪɚ'])
const NO_LINK_BEFORE = new Set(['and', 'or', 'if'])
const FLAP_BEFORE_VOWEL = new Set(['it', 'at'])

export class Unit {
  /** Spelling in lower case; several words for a phrase. */
  word: string
  phones: Phones
  /** Stress class of function_words.txt; "c" = stress as given. */
  kind: string
  /** The form to use when the word is promoted. */
  final: Phones | null
  /** No word boundary before this unit (hyphenated compound). */
  attach: boolean
  source: string

  constructor(word: string, phones: Phones, kind = 'c', final: Phones | null = null, attach = false, source = '') {
    this.word = word
    this.phones = phones
    this.kind = kind
    this.final = final
    this.attach = attach
    this.source = source
  }
  get lastWord(): string {
    const at = this.word.lastIndexOf(' ')
    return at < 0 ? this.word : this.word.slice(at + 1)
  }
  get firstWord(): string {
    const at = this.word.indexOf(' ')
    return at < 0 ? this.word : this.word.slice(0, at)
  }
}

export const unstress = (phones: Phones): Phone[] => phones.map(([phone]) => [phone, 0] as const)

/** The word with a primary stress: on its marked vowel, else on the first vowel. */
export function promote(phones: Phones): Phone[] {
  const marks = phones.map(([, stress]) => stress)
  if (marks.includes(1)) return [...phones]
  let target: number | null
  if (marks.includes(2)) target = marks.lastIndexOf(2)
  else {
    const found = phones.findIndex(([phone]) => VOWEL_SET.has(phone))
    target = found < 0 ? null : found
  }
  return phones.map(([phone, stress], i) => [phone, i === target ? 1 : stress] as const)
}

/** The stress class of every unit of a clause, right to left. */
export function stressClause(units: Unit[]): void {
  let primaryFollows = false
  for (let index = units.length - 1; index >= 0; index--) {
    const unit = units[index]!
    const promoted = (): Phones => (unit.final !== null ? unit.final : promote(unit.phones))
    const last = index === units.length - 1
    let phones: Phones
    if (unit.kind === 'u') phones = unstress(unit.phones)
    else if (unit.kind === 'uf') phones = last ? promoted() : unstress(unit.phones)
    else if (unit.kind === 'ur') phones = primaryFollows ? unstress(unit.phones) : promoted()
    else if (unit.kind === 's') phones = primaryFollows ? unit.phones : promoted()
    else if (unit.kind === 'si')
      phones = !primaryFollows ? promoted() : index === 0 ? unit.phones : unstress(unit.phones)
    else if (unit.kind === 'pf') phones = last ? promoted() : unit.phones
    else phones = unit.phones
    unit.phones = [...phones]
    primaryFollows = primaryFollows || unit.phones.some(([, stress]) => stress === 1)
  }
  if (units.length && !units.some((unit) => unit.phones.some(([, stress]) => stress))) {
    for (const unit of [...units].reverse()) {
      if (unit.phones.some(([phone]) => VOWEL_SET.has(phone))) {
        unit.phones = promote(unit.phones)
        break
      }
    }
  }
}

/** The end of each word adjusted to the beginning of the next. */
export function joinClause(units: Unit[]): void {
  for (let k = 0; k + 1 < units.length; k++) {
    const unit = units[k]!
    const following = units[k + 1]!
    if (!unit.phones.length || !following.phones.length || !VOWEL_SET.has(following.phones[0]![0])) continue
    const [last, stress] = unit.phones[unit.phones.length - 1]!
    const head = unit.phones.slice(0, -1)
    if (unit.lastWord === 'the' && last === 'ə') unit.phones = [...head, ['ɪ', stress]]
    else if (unit.lastWord === 'to' && last === 'ə') unit.phones = [...head, ['ʊ', stress]]
    else if (NO_LINK_BEFORE.has(following.firstWord)) continue
    else if (last in SPLIT) unit.phones = [...head, [SPLIT[last]!, stress], ['ɹ', 0]]
    else if (RHOTIC.has(last)) unit.phones = [...unit.phones, ['ɹ', 0]]
    else if (
      last === 't' &&
      FLAP_BEFORE_VOWEL.has(unit.word) &&
      unit.phones.length > 1 &&
      VOWEL_SET.has(unit.phones[unit.phones.length - 2]![0])
    )
      unit.phones = [...head, ['ɾ', 0]]
  }
}

/** The units of a clause as spoken words: an attached unit continues the one before. */
export function groups(units: readonly Unit[]): Phone[][] {
  const out: Phone[][] = []
  for (const unit of units) {
    if (unit.attach && out.length) out[out.length - 1] = [...out[out.length - 1]!, ...unit.phones]
    else out.push([...unit.phones])
  }
  return out.filter((group) => group.length)
}
