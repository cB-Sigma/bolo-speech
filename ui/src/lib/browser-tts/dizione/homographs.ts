/** Words spelled alike and pronounced differently: which reading is meant (`dizione/homographs.py`). */

import { codePointCompare, isalpha, isdigit, isupper, split } from './py'
import { rows } from './tables'

const CLASSES: [string, string][] = [
  ['TO', 'to'],
  ['DET', 'the a an this that these those every each no some any another such what which whose'],
  ['POSS', 'my your his her its our their'],
  ['SUBJ', 'i you he she it we they who'],
  ['OBJ', 'me him us them'],
  ['MODAL', "will would shall should can could may might must 'll 'd won't can't cannot couldn't shouldn't wouldn't let's please"],
  ['DO', "do does did don't doesn't didn't"],
  ['BE', "am is are was were be been being isn't aren't wasn't weren't i'm you're we're they're he's she's it's that's there's"],
  ['HAVE', "have has had having haven't hasn't hadn't i've you've we've they've"],
  ['NEG', 'not never'],
  ['PREP', 'of in on at by for from with into onto over under about after before between through during without within ' +
    'against among around across behind beside near upon toward towards as than like'],
  ['CONJ', 'and or but nor so yet if because although though while when where unless until since whether'],
  ['ADV', 'very too quite rather more most less least also just only even still always often sometimes usually again ' +
    'now then here there soon already almost'],
] // prettier-ignore
const TAG = new Map(CLASSES.flatMap(([name, words]) => split(words).map((word) => [word, name] as const)))
const SENTENCE_END = new Set(['.', '?', '!'])
const WINDOW = 6

/** The coarse grammatical class of a neighbouring token. */
export function tag(token: string): string {
  if (token === '<s>' || token === '</s>') return token
  const head = [...token][0]!
  if (!isalpha(head)) return isdigit(head) ? 'NUM' : 'PUNCT'
  const lower = token.toLowerCase()
  const found = TAG.get(lower)
  if (found) return found
  for (const [ending, name] of [
    ["'s", 'POS'],
    ['ly', 'LY'],
    ['ing', 'ING'],
    ['ed', 'ED'],
  ] as const) {
    if (lower.endsWith(ending) && lower.length > ending.length + 1) return name
  }
  if (lower.endsWith('s') && !lower.endsWith('ss') && lower.length > 3) return 'S'
  return isupper(head) ? 'CAP' : 'W'
}

/** The names of the features that are on for `tokens[index]` (one sentence). */
export function features(tokens: readonly string[], index: number): string[] {
  const at = (offset: number): string => {
    const position = index + offset
    return position >= 0 && position < tokens.length ? tokens[position]! : offset < 0 ? '<s>' : '</s>'
  }
  const [p2, p1, n1, n2] = [at(-2), at(-1), at(1), at(2)]
  const [tp2, tp1, tn1, tn2] = [tag(p2), tag(p1), tag(n1), tag(n2)]
  const out = [`tp1=${tp1}`, `tp2=${tp2}`, `tn1=${tn1}`, `tn2=${tn2}`, `tp2p1=${tp2}+${tp1}`, `tp1n1=${tp1}+${tn1}`,
    `tn1n2=${tn1}+${tn2}`] // prettier-ignore
  if (index === 0) out.push('first')
  else if (isupper([...tokens[index]!][0]!)) out.push('cap')
  out.push(`p1=${p1.toLowerCase()}`, `p2=${p2.toLowerCase()}`, `n1=${n1.toLowerCase()}`, `n2=${n2.toLowerCase()}`)
  const target = tokens[index]!.toLowerCase()
  const nearby = new Set(
    tokens
      .slice(Math.max(0, index - WINDOW), index + WINDOW + 1)
      .filter((t) => isalpha([...t][0]!))
      .map((t) => t.toLowerCase()),
  )
  nearby.delete(target)
  out.push(...[...nearby].sort(codePointCompare).map((word) => `w=${word}`))
  return out
}

const shared = (names: string[]): string[] =>
  names.filter((name) => name[0] === 't' || name === 'first' || name === 'cap')

/** The sentence around `tokens[index]` and the index inside it. */
export function sentenceOf(tokens: readonly string[], index: number): [string[], number] {
  let start = index
  while (start > 0 && !SENTENCE_END.has(tokens[start - 1]!)) start--
  let end = index
  while (end < tokens.length && !SENTENCE_END.has(tokens[end]!)) end++
  return [tokens.slice(start, end + 1), index - start]
}

const SKIP_FOR_BEFORE = new Set(['ADV', 'NEG', 'LY'])
const SKIP_FOR_AUX = new Set([...SKIP_FOR_BEFORE, 'SUBJ', 'OBJ', 'DET', 'POSS'])

function left(tokens: readonly string[], index: number, skip: Set<string>): string {
  for (let position = index - 1; position > Math.max(index - 6, -1); position--) {
    const token = tokens[position]!
    if (skip.has(tag(token))) continue
    const lower = token.toLowerCase()
    return lower.endsWith("'ll") ? 'will' : lower.endsWith("'ve") ? 'have' : lower
  }
  return '<s>'
}

type Condition = [name: string, words: Set<string>]
type Rules = Map<string, [sense: string, conditions: Condition[]][]>

export function readRules(): Rules {
  const rules: Rules = new Map()
  for (const [number, columns] of rows('homograph_rules.txt')) {
    if (columns.length !== 3) throw new Error(`homograph_rules.txt line ${number}: expected word, sense, conditions`)
    const conditions: Condition[] = []
    for (const part of columns[2]!.split(';')) {
      const text = part.trim()
      const colon = text.indexOf(':')
      const name = colon < 0 ? text : text.slice(0, colon)
      const words = colon < 0 ? '' : text.slice(colon + 1)
      if (!['aux', 'before', 'next', 'near', 'first', 'joined'].includes(name) || (name === 'first') === !!words.trim())
        throw new Error(`homograph_rules.txt line ${number}: bad condition '${text}'`)
      conditions.push([name, new Set(split(words))])
    }
    const list = rules.get(columns[0]!) ?? []
    list.push([columns[1]!, conditions])
    rules.set(columns[0]!, list)
  }
  return rules
}

function holds(
  condition: string,
  words: Set<string>,
  tokens: readonly string[],
  index: number,
  joined = false,
): boolean {
  if (condition === 'first') return index === 0
  if (condition === 'joined') return joined && index > 0 && words.has(tokens[index - 1]!.toLowerCase())
  if (condition === 'aux') return words.has(left(tokens, index, SKIP_FOR_AUX))
  if (condition === 'before') return words.has(left(tokens, index, SKIP_FOR_BEFORE))
  if (condition === 'next') return words.has(index + 1 < tokens.length ? tokens[index + 1]!.toLowerCase() : '</s>')
  return tokens.slice(Math.max(0, index - WINDOW), index + WINDOW + 1).some((token) => words.has(token.toLowerCase()))
}

interface Sense {
  id: string
  role: string
  phones: string
}
interface Entry {
  senses: Sense[]
  model:
    | { kind: 'noun-verb'; bias: number; weights: Record<string, number> }
    | { kind: string; bias: number[]; weights: Record<string, number>[] }
}
export interface HomographData {
  words: Record<string, Entry>
  pooled: { bias: number; weights: Record<string, number> }
  plural_bases?: string[]
}

const weight = (weights: Record<string, number>, name: string): number =>
  Object.hasOwn(weights, name) ? weights[name]! : 0.0

/** Python's `sum` of the weights: from 0, in order. */
function total(weights: Record<string, number>, names: string[]): number {
  let sum = 0
  for (const name of names) sum += weight(weights, name)
  return sum
}

export class Homographs {
  readonly words: Record<string, Entry>
  readonly pooled: HomographData['pooled']
  readonly pluralBases: Set<string>
  readonly rules: Rules

  constructor(data: HomographData) {
    this.words = data.words
    this.pooled = data.pooled
    this.pluralBases = new Set(data.plural_bases ?? [])
    this.rules = readRules()
    for (const [word, wordRules] of this.rules) {
      const known = new Set(this.has(word) ? this.words[word]!.senses.map((s) => s.id) : [])
      for (const [sense] of wordRules) {
        if (!known.has(sense)) throw new Error(`homograph rule for '${word}' names the unknown sense '${sense}'`)
      }
    }
  }

  has(word: string): boolean {
    return Object.hasOwn(this.words, word)
  }

  /** True for a homograph and for the regular -s form of a noun-or-verb one. */
  covers(word: string): boolean {
    return this.has(word) || (word.endsWith('s') && this.pluralBases.has(word.slice(0, -1)))
  }

  /** `[phones as text, needs the -s ending]` for `tokens[index]`. */
  pronounce(word: string, tokens: readonly string[], index: number, joined = false): [string, boolean] {
    if (this.has(word)) return [this.choose(word, tokens, index, joined).phones, false]
    const entry = this.words[word.slice(0, -1)]!
    const adjective = entry.senses.find((sense) => sense.role === 'n')
    if (adjective && adjective.id === 'adj') return [entry.senses.find((sense) => sense.role === 'v')!.phones, true]
    return [this.choose(word.slice(0, -1), tokens, index, joined).phones, true]
  }

  scores(word: string, tokens: readonly string[], index: number): number[] {
    const entry = this.words[word]!
    const names = features(...sentenceOf(tokens, index))
    const model = entry.model
    if (model.kind === 'noun-verb') {
      const m = model as { bias: number; weights: Record<string, number> }
      let verb = this.pooled.bias + total(this.pooled.weights, shared(names))
      verb += m.bias + total(m.weights, names)
      return entry.senses.map((sense) => (sense.role === 'v' ? verb : -verb))
    }
    const m = model as { bias: number[]; weights: Record<string, number>[] }
    return m.bias.map((bias, k) => bias + total(m.weights[k]!, names))
  }

  /** The sense meant by `tokens[index]`: hand rules first, then the classifier. */
  choose(word: string, tokens: readonly string[], index: number, joined = false): Sense {
    const entry = this.words[word]!
    const [sentence, position] = sentenceOf(tokens, index)
    for (const [sense, conditions] of this.rules.get(word) ?? []) {
      if (conditions.every(([name, words]) => holds(name, words, sentence, position, joined)))
        return entry.senses.find((s) => s.id === sense)!
    }
    const scores = this.scores(word, tokens, index)
    let best = 0
    scores.forEach((score, k) => {
      if (score > scores[best]!) best = k
    })
    return entry.senses[best]!
  }
}
