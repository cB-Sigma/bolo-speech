/**
 * The phonemiser: text -> spoken words of phones, clause by clause (`dizione/engine.py`). For one word: inline
 * pronunciation, the exceptions, function words and phrases, the dictionary (CMUdict), regular endings and compounds of
 * dictionary words, names in "-leigh" and words in "-virus", then the unknown-word model.
 *
 * This folder is a TypeScript port of dizione (Copyright Muhammed Nazeem, Apache License 2.0), made from bolo's copy
 * in server/src/bolo/vendor/dizione (commit 8cd889e4 of github.com/n4ze3m/dizione, its LICENSE next to it).
 */

import { Aligner, type AlignTable } from './align'
import { convert, fromText, settleStress } from './convert'
import type { G2P } from './g2p'
import { Homographs, type HomographData } from './homographs'
import { compound, derive, plural } from './morphology'
import { Normalizer, type Token, type Word } from './normalize'
import { key as tupleKey, re, search, split, sub } from './py'
import { groups, joinClause, promote, stressClause, Unit } from './sentence'
import { NothingToSay, tokenize, type Phone, type Phonemized, type Phones, type Piece } from './symbols'
import { functionWords, mapping, names, phraseTable, pronunciations, type Pronunciation } from './tables'

const LEIGH = re('(?<=[a-z]{2})leigh(?=s?$)')
const VIRUS = ['viruses', 'virus']

/** The voice's phonemiser data (the model repository's `phonemizer/` folder), read. */
export interface DizioneData {
  /** lexicon.tsv: one `word<TAB>phones` per line. */
  lexicon: string
  align: AlignTable
  homographs: HomographData | null
  g2p: G2P | null
}

let letterNames: Map<string, Phone[]> | null = null

export class Phonemizer {
  readonly lexicon = new Map<string, string>()
  readonly aligner: Aligner
  readonly model: G2P | null
  readonly homographs: Homographs | null
  readonly functions = functionWords()
  readonly phrases = phraseTable()
  readonly phraseLength: number
  readonly exceptions = pronunciations('exceptions.txt')
  readonly names = names('exceptions.txt')
  readonly nameStarts: Set<string>
  readonly nameLength: number
  readonly normalizer: Normalizer
  private readonly cache = new Map<string, [Phones, string]>()

  constructor(data: DizioneData) {
    for (const line of data.lexicon.split(/\r\n|\r|\n/)) {
      if (!line) continue
      const tab = line.indexOf('\t')
      this.lexicon.set(line.slice(0, tab), line.slice(tab + 1))
    }
    this.aligner = new Aligner(data.align)
    this.model = data.g2p
    this.homographs = data.homographs ? new Homographs(data.homographs) : null
    this.phraseLength = Math.max(1, ...[...this.phrases.keys()].map((k) => k.split('\u0000').length))
    this.nameStarts = new Set([...this.names.keys()].map((k) => k.split('\u0000')[0]!))
    this.nameLength = Math.max(1, ...[...this.names.keys()].map((k) => k.split('\u0000').length))
    this.normalizer = new Normalizer((w) => this.isWord(w))
  }

  // ---- single words -------------------------------------------------

  private entry(word: string, entry: Pronunciation): Phones {
    if (entry[0] === 'phones') return entry[1]
    return convert(word, entry[1], this.aligner.align(word, entry[1]))
  }

  /** The pronunciation from the exceptions or the dictionary, or null. */
  known(word: string): Phones | null {
    const exception = this.exceptions.get(word)
    if (exception) return this.entry(word, exception)
    const text = this.lexicon.get(word)
    return text !== undefined ? fromText(text) : null
  }

  /** A dictionary word that may be half of a compound (also listed with a regular ending). */
  ordinary(word: string): Phones | null {
    if (word.length < 3 || !['s', 'es', 'ed', 'ing'].some((ending) => this.lexicon.has(word + ending))) return null
    return this.known(word)
  }

  /** The unknown-word model, then the conversion rules, on one spelling. */
  guess(word: string): Phones {
    if (this.model === null) return this.spell(word)
    const spelling = this.model.spellable(word)
    const symbols = this.model.predict(spelling)
    const stressed = (k: number): boolean => k >= 0 && k < symbols.length && '012'.includes(symbols[k]!.slice(-1))
    symbols.forEach((symbol, k) => {
      if (symbol === 'R' && !stressed(k - 1) && !stressed(k + 1)) symbols[k] = 'ER0'
    })
    if (!symbols.some((symbol) => '012'.includes(symbol.slice(-1)))) return this.spell(word)
    return promote(convert(spelling, symbols, this.aligner.align(spelling, symbols)))
  }

  /** Letter names, joined: the last letter carries the main stress. */
  spell(letters: string): Phones {
    letterNames ??= new Map([...mapping('letters.txt')].map(([k, v]) => [k, fromText(v)]))
    const possessive = letters.toLowerCase().endsWith("'s")
    const source = (possessive ? letters.slice(0, -2) : letters).toLowerCase()
    const found = [...source].filter((c) => letterNames!.has(c)).map((c) => letterNames!.get(c)!)
    const out: Phone[] = []
    found.forEach((name, index) => {
      for (const [phone, stress] of name) out.push([phone, stress && index < found.length - 1 ? 2 : stress])
    })
    return possessive && out.length ? plural(out) : out
  }

  /** True for a spelling the tables or the dictionary know. */
  isWord(word: string): boolean {
    return this.lexicon.has(word) || this.functions.has(word) || this.exceptions.has(word)
  }

  private fromParts(word: string): [Phones, string] | null {
    if (word.endsWith("'s") && word.length > 3) {
      const found = this.fromParts(word.slice(0, -2))
      return found && [plural(found[0]), found[1]]
    }
    if (search(LEIGH, word)) return this.word(sub(LEIGH, 'ley', word))
    for (const tail of VIRUS) {
      const second = word.endsWith(tail) && word.length - tail.length > 2 ? this.known(tail) : null
      if (second !== null) {
        const [first, source] = this.word(word.slice(0, -tail.length))
        const phones = promote(settleStress(compound(first, second)))
        return [phones, source === 'model' ? 'model' : 'ending']
      }
    }
    return null
  }

  /** `[phones, source]` for one spelling in lower case. */
  word(word: string): [Phones, string] {
    const cached = this.cache.get(word)
    if (cached) return cached
    let phones: Phones | null = this.known(word)
    let source = 'dictionary'
    if (phones === null) {
      phones = derive(
        word,
        (w) => this.known(w),
        (w) => this.ordinary(w),
      )
      source = 'ending'
      if (phones !== null) phones = settleStress(phones)
    }
    if (phones === null) {
      const found = this.fromParts(word)
      if (found) [phones, source] = found
    }
    if (phones === null && word.endsWith("'s") && word.length > 3) {
      phones = plural(this.guess(word.slice(0, -2)))
      source = 'model'
    }
    if (phones === null) {
      phones = this.guess(word)
      source = 'model'
    }
    const result: [Phones, string] = [phones, source]
    this.cache.set(word, result)
    return result
  }

  // ---- sentences ----------------------------------------------------

  private units(words: Word[], places: number[], written: string[]): Unit[] {
    const units: Unit[] = []
    let index = 0
    while (index < words.length) {
      const token = words[index]!
      if (token.phones !== null) {
        units.push(new Unit(token.text, token.phones, 'p', null, token.attach, 'inline'))
      } else if (token.spell) {
        units.push(new Unit(token.text, this.spell(token.text), 'p', null, token.attach, 'letters'))
      } else {
        let size = this.nameStarts.has(token.text) ? this.name(words, index) : 0
        if (size) {
          const k = words.slice(index, index + size).map((w) => w.text)
          const entries = this.names.get(tupleKey(k))!
          k.forEach((text, position) =>
            units.push(
              new Unit(text, this.entry(text, entries[position]!), 'p', null, token.attach && position === 0, 'name'),
            ),
          )
          index += size
          continue
        }
        size = this.phrase(words, index)
        if (size) {
          const k = words.slice(index, index + size).map((w) => w.text)
          const [kind, forms, final] = this.phrases.get(tupleKey(k))!
          if (forms.length === 1) units.push(new Unit(k.join(' '), forms[0]!, kind, final, token.attach, 'phrase'))
          else
            k.forEach((text, position) =>
              units.push(new Unit(text, forms[position]!, kind, null, token.attach && position === 0, 'phrase')),
            )
          index += size
          continue
        }
        if (
          this.homographs &&
          written.length &&
          this.homographs.covers(token.text) &&
          !this.exceptions.has(token.text)
        ) {
          const [text, ending] = this.homographs.pronounce(token.text, written, places[index]!, token.attach)
          const phones = ending ? plural(fromText(text)) : fromText(text)
          units.push(new Unit(token.text, phones, 'c', null, token.attach, 'homograph'))
        } else if (this.functions.has(token.text)) {
          const [kind, form, final] = this.functions.get(token.text)!
          units.push(new Unit(token.text, form, kind, final, token.attach, 'function'))
        } else {
          const [phones, source] = this.word(token.text)
          units.push(new Unit(token.text, phones, 'c', null, token.attach, source))
        }
      }
      index += 1
    }
    return units
  }

  private name(words: Word[], index: number): number {
    for (let size = Math.min(this.nameLength, words.length - index); size > 1; size--) {
      const span = words.slice(index, index + size)
      if (
        this.names.has(tupleKey(span.map((w) => w.text))) &&
        span.every((w) => w.phones === null && !w.spell) &&
        !span.slice(1).some((w) => w.attach)
      )
        return size
    }
    return 0
  }

  private phrase(words: Word[], index: number): number {
    for (let size = Math.min(this.phraseLength, words.length - index); size > 1; size--) {
      const span = words.slice(index, index + size)
      if (
        span.slice(1).every((w) => w.phones === null && !w.spell && !w.attach) &&
        span[0]!.phones === null &&
        !span[0]!.spell &&
        this.phrases.has(tupleKey(span.map((w) => w.text)))
      )
        return size
    }
    return 0
  }

  /** Clauses and marks in order. */
  analyse(text: string): (readonly ['punct', string] | readonly ['clause', Unit[]])[] {
    const stream: Token[] = this.normalizer.tokens(text)
    const written = stream.map((token) => (token.kind === 'punct' ? token.mark : token.raw || token.text || 'x'))
    const out: (readonly ['punct', string] | readonly ['clause', Unit[]])[] = []
    let words: Word[] = []
    let places: number[] = []
    const flush = (): void => {
      if (words.length) {
        const units = this.units(words, places, written)
        stressClause(units)
        joinClause(units)
        out.push(['clause', units])
        words = []
        places = []
      }
    }
    stream.forEach((token, place) => {
      if (token.kind === 'punct') {
        flush()
        out.push(['punct', token.mark])
      } else {
        words.push(token)
        places.push(place)
      }
    })
    flush()
    return out
  }

  /** `[true, mark]` for punctuation, `[false, words of phones]` for a run of words. */
  pieces(text: string): Piece[] {
    return this.analyse(text).map((item) => (item[0] === 'punct' ? [true, item[1]] : [false, groups(item[1])]))
  }

  /** Text -> symbols, ids and stress; throws on empty text or text with nothing to say. */
  phonemize(text: string): Phonemized {
    if (!split(text).length) throw new NothingToSay('text must not be empty')
    return tokenize(this.pieces(text))
  }

  /** `phonemize`, or null for text with nothing to say. */
  tryPhonemize(text: string): Phonemized | null {
    try {
      return this.phonemize(text)
    } catch (error) {
      if (error instanceof NothingToSay) return null
      throw error
    }
  }
}
