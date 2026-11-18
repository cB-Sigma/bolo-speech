/** dizione's phone inventory and the token sequence built from it (`dizione/symbols.py`). The order is fixed: the
 * model's token ids depend on it. */

export const PAD = '<pad>'
export const BOS = '<bos>'
export const EOS = '<eos>'
export const SP = '<sp>'
export const UNK = '<unk>'
export const SPECIALS = [PAD, BOS, EOS, SP, UNK]
export const PUNCTUATION = [',', '.', '?', '!', ';', ':', '"', "'", '(', ')', '—', '…']
export const CONSONANTS = ['p', 'b', 't', 'd', 'k', 'ɡ', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h', 'm', 'n', 'ŋ', 'l',
  'ɹ', 'j', 'w', 'tʃ', 'dʒ', 'ɾ', 'ʔ'] // prettier-ignore
export const VOWELS = ['ə', 'ɐ', 'ɚ', 'ᵻ', 'ɪ', 'i', 'iː', 'ɛ', 'æ', 'ʌ', 'ɑː', 'ɔ', 'ɔː', 'ʊ', 'uː', 'ɜː', 'eɪ', 'aɪ',
  'ɔɪ', 'oʊ', 'aʊ', 'iə', 'oː', 'ɑːɹ', 'ɔːɹ', 'oːɹ', 'ɪɹ', 'ɛɹ', 'ʊɹ', 'aɪɚ', 'aɪə', 'əl', 'n̩'] // prettier-ignore
export const PHONES = [...CONSONANTS, ...VOWELS]
export const SYMBOLS = [...SPECIALS, ...PUNCTUATION, ...PHONES]
export const SYMBOL_TO_ID = new Map(SYMBOLS.map((s, i) => [s, i]))
export const PHONE_SET = new Set(PHONES)
export const VOWEL_SET = new Set(VOWELS)

/** One phone and its stress: 0 none, 1 primary, 2 secondary. */
export type Phone = readonly [string, number]
export type Phones = readonly Phone[]

export interface Phonemized {
  symbols: string[]
  ids: number[]
  stress: number[]
  /** Phones outside the inventory (never produced by the built-in rules). */
  unknown: string[]
}

/** Text with nothing to say (Python's ValueError from `phonemize` and `tokenize`). */
export class NothingToSay extends Error {}

/** A piece of text: `[true, mark]` for punctuation, `[false, words]` for words (each a list of phones). */
export type Piece = readonly [true, string] | readonly [false, Phone[][]]

/** The token sequence: words parted by `<sp>`, repeated punctuation collapsed. Throws when nothing is said. */
export function tokenize(pieces: readonly Piece[]): Phonemized {
  const symbols = [BOS]
  const stress = [0]
  const unknown: string[] = []
  for (const piece of pieces) {
    if (piece[0]) {
      if (symbols[symbols.length - 1] !== piece[1]) {
        symbols.push(piece[1])
        stress.push(0)
      }
      continue
    }
    for (const word of piece[1]) {
      const last = symbols[symbols.length - 1]!
      if (PHONE_SET.has(last) || last === UNK) {
        symbols.push(SP)
        stress.push(0)
      }
      for (const [phone, level] of word) {
        if (PHONE_SET.has(phone)) {
          symbols.push(phone)
          stress.push(level)
        } else {
          unknown.push(phone)
          symbols.push(UNK)
          stress.push(0)
        }
      }
    }
  }
  symbols.push(EOS)
  stress.push(0)
  if (!symbols.some((s) => PHONE_SET.has(s) || s === UNK)) throw new NothingToSay('the text has nothing to pronounce')
  return { symbols, ids: symbols.map((s) => SYMBOL_TO_ID.get(s)!), stress, unknown }
}
