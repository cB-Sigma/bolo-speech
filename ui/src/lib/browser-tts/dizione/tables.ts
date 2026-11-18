/** dizione's hand-written tables (`dizione/tables.py`), read from the text of `data/*.txt`. */

import { fromText, isArpabet } from './convert'
import { key as tupleKey, split } from './py'
import type { Phone, Phones } from './symbols'
import abbreviationsTable from '../../../../../server/src/bolo/vendor/dizione/data/abbreviations.txt?raw'
import chatTable from '../../../../../server/src/bolo/vendor/dizione/data/chat.txt?raw'
import endingsTable from '../../../../../server/src/bolo/vendor/dizione/data/endings.txt?raw'
import exceptionsTable from '../../../../../server/src/bolo/vendor/dizione/data/exceptions.txt?raw'
import functionWordsTable from '../../../../../server/src/bolo/vendor/dizione/data/function_words.txt?raw'
import homographRulesTable from '../../../../../server/src/bolo/vendor/dizione/data/homograph_rules.txt?raw'
import lettersTable from '../../../../../server/src/bolo/vendor/dizione/data/letters.txt?raw'
import phrasesTable from '../../../../../server/src/bolo/vendor/dizione/data/phrases.txt?raw'
import unitsTable from '../../../../../server/src/bolo/vendor/dizione/data/units.txt?raw'

/** The tables shipped with dizione (bolo's vendored copy, server/src/bolo/vendor/dizione/data). */
export const DATA: Record<string, string> = {
  'abbreviations.txt': abbreviationsTable,
  'chat.txt': chatTable,
  'endings.txt': endingsTable,
  'exceptions.txt': exceptionsTable,
  'function_words.txt': functionWordsTable,
  'homograph_rules.txt': homographRulesTable,
  'letters.txt': lettersTable,
  'phrases.txt': phrasesTable,
  'units.txt': unitsTable,
}

const COLUMNS = /\t+| {2,}/
const STRESS_CLASSES = new Set(['u', 'uf', 'ur', 's', 'ss', 'si', 'pf', 'p'])

/** Rows of a table (a file name in `data/`) as `[line number, columns]`. */
export function rows(name: string): [number, string[]][] {
  const text = DATA[name]
  if (text === undefined) throw new Error(`no table ${name}`)
  const out: [number, string[]][] = []
  text.split(/\r\n|\r|\n/).forEach((raw, index) => {
    const line = raw.trim()
    if (line && !line.startsWith('#')) out.push([index + 1, line.split(COLUMNS).map((column) => column.trim())])
  })
  return out
}

export type FunctionWord = [kind: string, form: Phones, final: Phones | null]

export function functionWords(): Map<string, FunctionWord> {
  const table = new Map<string, FunctionWord>()
  for (const [number, columns] of rows('function_words.txt')) {
    if ((columns.length !== 3 && columns.length !== 4) || !STRESS_CLASSES.has(columns[1]!))
      throw new Error(`function_words.txt line ${number}: expected word, class, form[, final form]`)
    table.set(columns[0]!, [columns[1]!, fromText(columns[2]!), columns.length === 4 ? fromText(columns[3]!) : null])
  }
  return table
}

export type Phrase = [kind: string, forms: Phones[], final: Phones | null]

/** Keyed by the words joined with `key`. */
export function phraseTable(): Map<string, Phrase> {
  const table = new Map<string, Phrase>()
  for (const [number, columns] of rows('phrases.txt')) {
    if ((columns.length !== 3 && columns.length !== 4) || !STRESS_CLASSES.has(columns[1]!))
      throw new Error(`phrases.txt line ${number}: expected words, class, form[, final form]`)
    const forms = columns[2]!.split('|').map((part) => fromText(part))
    if (forms.length > 1 && (columns[1] !== 'p' || forms.length !== split(columns[0]!).length))
      throw new Error(`phrases.txt line ${number}: a form with boundaries needs class p and one part per word`)
    table.set(tupleKey(split(columns[0]!)), [columns[1]!, forms, columns.length === 4 ? fromText(columns[3]!) : null])
  }
  return table
}

export type Pronunciation = readonly ['arpabet', string[]] | readonly ['phones', Phone[]]

function pronunciation(text: string): Pronunciation {
  const symbols = split(text)
  return isArpabet(symbols) ? ['arpabet', symbols] : ['phones', fromText(text)]
}

/** word -> its pronunciation, from the exceptions table (rows of one word). */
export function pronunciations(name: string): Map<string, Pronunciation> {
  const table = new Map<string, Pronunciation>()
  for (const [where, columns] of rows(name)) {
    if (columns.length !== 2) throw new Error(`${where}: expected a word and its pronunciation`)
    if (split(columns[0]!).length === 1) table.set(columns[0]!.toLowerCase(), pronunciation(columns[1]!))
  }
  return table
}

/** Names of several words with one pronunciation per word, keyed by the words joined with `key`. */
export function names(name: string): Map<string, Pronunciation[]> {
  const table = new Map<string, Pronunciation[]>()
  for (const [where, columns] of rows(name)) {
    const words = split(columns[0]!.toLowerCase())
    if (columns.length !== 2 || words.length < 2) continue
    const parts = columns[1]!.split('|').map((part) => part.trim())
    if (parts.length !== words.length) throw new Error(`${where}: one pronunciation per word, separated by |`)
    table.set(tupleKey(words), parts.map(pronunciation))
  }
  return table
}

/** A two-column table as a map (the second column may be empty). */
export function mapping(name: string): Map<string, string> {
  return new Map(rows(name).map(([, columns]) => [columns[0]!, columns.length > 1 ? columns[1]! : '']))
}
