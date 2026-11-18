/**
 * Text -> the words to say and the punctuation marks between them (`dizione/normalize.py`). Numbers, money, dates,
 * times, telephone numbers, units, abbreviations, initialisms, symbols, chat spellings, versions, codes, addresses and
 * file names become words by rule; emoticons and emoji are dropped; `{HH AH0 L OW1}` gives a pronunciation inline.
 * Each pattern is dizione's own, translated by `py.ts` so that it means what it means in Python.
 */

import { convert, fromText, isArpabet } from './convert'
import {
  escape,
  finditer,
  findall,
  findallGroups,
  fullmatch,
  isalnum,
  isalpha,
  isascii,
  isdigit,
  islower,
  isspace,
  isupper,
  match as reMatch,
  re,
  rstrip,
  search,
  split,
  sub,
  type Match,
} from './py'
import { PUNCTUATION, type Phone } from './symbols'
import { mapping, rows } from './tables'

export interface Word {
  kind: 'word'
  /** Letters and apostrophes, lower case. */
  text: string
  /** Continues the previous word (hyphenated compound, parts of one token). */
  attach: boolean
  /** As written. */
  raw: string
  /** Read letter by letter. */
  spell: boolean
  /** A pronunciation given inline. */
  phones: Phone[] | null
}

export interface Punct {
  kind: 'punct'
  mark: string
}

export type Token = Word | Punct

const word = (text: string, more: Partial<Omit<Word, 'kind' | 'text'>> = {}): Word => ({
  kind: 'word',
  text,
  attach: more.attach ?? false,
  raw: more.raw ?? '',
  spell: more.spell ?? false,
  phones: more.phones ?? null,
})

const SPELL = ''
const END = ''
const INLINE = ''
const INLINE_BASE = 0xe100
const ONES = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen ' +
  'seventeen eighteen nineteen').split(' ') // prettier-ignore
const TENS = 'twenty thirty forty fifty sixty seventy eighty ninety'.split(' ')
const SCALES = ['thousand', 'million', 'billion', 'trillion', 'quadrillion', 'quintillion']
const ORDINAL: Record<string, string> = { one: 'first', two: 'second', three: 'third', five: 'fifth', eight: 'eighth',
  nine: 'ninth', twelve: 'twelfth' } // prettier-ignore
const MONTHS = 'January February March April May June July August September October November December'.split(' ')
const MONTH_NUMBER = new Map(MONTHS.map((name, index) => [name.toLowerCase().slice(0, 3), index + 1]))
const MONTH_PATTERN = MONTHS.join('|') + '|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec'
const DAY_NAMES: Record<string, string> = { mon: 'Monday', tue: 'Tuesday', wed: 'Wednesday', thu: 'Thursday',
  fri: 'Friday', sat: 'Saturday', sun: 'Sunday' } // prettier-ignore
const DAY_PATTERN =
  'Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Thurs|Thur|Tues|Mon|Tue|Wed|Thu|Fri|Sat|Sun'
type Currency = [string, string, string, string]
const CURRENCY: Record<string, Currency> = {
  $: ['dollar', 'dollars', 'cent', 'cents'], '£': ['pound', 'pounds', 'penny', 'pence'],
  '€': ['euro', 'euros', 'cent', 'cents'], '¥': ['yen', 'yen', '', ''], '₹': ['rupee', 'rupees', 'paisa', 'paise'],
} // prettier-ignore
Object.assign(CURRENCY, {
  USD: CURRENCY.$, EUR: CURRENCY['€'], GBP: CURRENCY['£'], JPY: CURRENCY['¥'], INR: CURRENCY['₹'],
  CHF: ['Swiss franc', 'Swiss francs', 'centime', 'centimes'], CAD: ['Canadian dollar', 'Canadian dollars', 'cent', 'cents'],
  AUD: ['Australian dollar', 'Australian dollars', 'cent', 'cents'], CNY: ['yuan', 'yuan', '', ''],
}) // prettier-ignore
const CURRENCY_CODES = 'USD|EUR|GBP|JPY|INR|CHF|CAD|AUD|CNY'
const SCALE_LETTER: Record<string, string> = { k: 'thousand', m: 'million', b: 'billion', bn: 'billion', t: 'trillion' }
const FRACTION: Record<number, [string, string]> = { 2: ['half', 'halves'], 4: ['quarter', 'quarters'] }
const SYMBOL_WORDS: Record<string, string> = {
  '&': 'and', '@': 'at', '+': 'plus', '=': 'equals', '%': 'percent', '×': 'times', '÷': 'divided by',
  '±': 'plus or minus', '©': 'copyright', '§': 'section', '°': 'degrees', '≥': 'greater than or equal to',
  '≤': 'less than or equal to', '≠': 'not equal to', '≈': 'approximately', '∞': 'infinity', '√': 'square root of', π: 'pi',
} // prettier-ignore
const COMPARISONS: Record<string, string> = { '>=': 'greater than or equal to', '<=': 'less than or equal to',
  '!=': 'not equal to' } // prettier-ignore
const INITIALISMS = new Set('US IT AI ID UN EU UK LA AD BC AM PM OS PE PA MA LED SAT RIP ER'.split(' '))
const SOUNDS = new Set('sh shh ss zz hm hmm mm tsk psst pfft hmph psh pssh'.split(' '))
const IS = ''
const COMPARED = re(String.raw`(?P<gap>[ \t]?)(?P<sign>>=|<=|!=|[<>≥≤≠])(?P=gap)`)
const VALUE_BEFORE = re(String.raw`(?:(?<![\w.,])\d[\d.,]*%?|(?<![\w'’.-])[A-Za-z]|[)\]])$`)
const VALUE_AFTER = re(String.raw`(?:[-+]?[$£€¥₹]?\d[\d.,]*%?(?!\w)|\.\d|[A-Za-z](?![\w'’])|[(\[])`)
const CLAUSE_OPENERS = new Set(('if elif while when whenever where whether that because since as and or but so then think ' +
  'thinks thought know knows say says said suppose assume means implies hence thus therefore although though unless ' +
  'until once else also clearly obviously indeed here now again first next finally similarly otherwise however ' +
  'moreover').split(' ')) // prettier-ignore
const SIGN = String.raw`[<>+*/^≤≥≠≈×÷±]`
const SPACED_SIGN = String.raw` [<>+*≤≥≠≈×÷±] `
const NO_PHRASE = String.raw`(?![\w'’-]|\s+(?!(?:and|or|then|if|when|where|while|for|is|are|was|but|so|in|to)\b)[A-Za-z])`
const LETTER_PLACES = [
  String.raw`(?<![\w'’.-])(?P<letter>[Aakur])(?=` + IS + String.raw`?\s?[<>!=]?=)`,
  String.raw`[<>!=]?=\s?(?P<letter>[Aakur])` + NO_PHRASE,
  String.raw`(?<![\w'’.-])(?P<letter>[Aakur])(?=` + IS + '?' + SIGN + String.raw`(?:[\w($]|\.\d))`,
  String.raw`(?<=[\w)` + IS + ']' + SIGN + String.raw`)(?<![Ww]/)(?P<letter>[Aakur])(?![\w'’-])`,
  String.raw`(?<![\w'’.-])(?P<letter>[Aakur])(?=` +
    IS +
    '?' +
    SPACED_SIGN +
    String.raw`(?:[A-Za-z]|\d[\d.,]*)(?![\w'’]))`,
  String.raw`(?<![\w'’.])(?:[A-Za-z]|\d[\d.,]*)` + IS + '?' + SPACED_SIGN + '(?P<letter>[Aakur])' + NO_PHRASE,
  String.raw`(?<=\()(?P<letter>[Aakur])(?=\)|, )`,
  String.raw`(?<=, )(?P<letter>[Aakur])(?=\))`,
  String.raw`(?:(?<=\s)|^)(?P<letter>[akur])(?=\)\s)`,
].map((p) => re(p))
const ANGLES = re(
  String.raw`(?<![\w)` +
    END +
    IS +
    String.raw`])<[A-Za-z0-9/!?][^<>(){}+*;|\n]*>` +
    String.raw`|(?<=[\w` +
    END +
    IS +
    String.raw`])<[^<>(){}+*;|\n]{1,80}>(?![\w$])` +
    String.raw`|</?[A-Za-z][\w-]+/?>`,
)
const ANGLE_MARK = ''

function withoutAngles(textIn: string): string {
  let text = textIn
  while (text.includes('<') && text.includes('>')) {
    const marked = sub(ANGLES, (m) => ANGLE_MARK + m.group(0)!.slice(1, -1) + ANGLE_MARK, text)
    if (marked === text) break
    text = marked
  }
  return text
}

const JOINED_COMPARISON = re(
  String.raw`(?<=[\w)` + END + IS + String.raw`])[<>](?=[\w(` + SPELL + String.raw`]|\.\d| minus \d)`,
)
const FIGURE_BEFORE = re(String.raw`(?<![\w.,])\d[\d.,]*%?$`)
const FIGURE_AFTER = re(String.raw`(?:\d[\d.,]*%?(?!\w)|\.\d| minus \d)`)
const LETTER_BEFORE = re(String.raw`(?:(?<![\w'’.])[A-Za-z]|[)` + END + '])$')
const LETTER_AFTER = re(String.raw`(?:[A-Za-z](?![\w'’])|[(` + SPELL + '])')
const SENTENCE_STARTERS = new Set(('the he she it i we they this that there but and in on at a an if when what why how who ' +
  'my our you then so now after before however').split(' ')) // prettier-ignore
const YEAR_CONTEXT = new Set(('in since by from until till before after during around circa of year early late mid ' +
  'summer winter spring fall autumn class').split(' ')) // prettier-ignore
const CARDINAL_AFTER = new Set(('war chapter part book volume act scene section phase type class level stage title ' +
  'article appendix world').split(' ')) // prettier-ignore
const LABELLED = new Set(('plan vitamin grade type option class section part team group exhibit appendix point model ' +
  'series hepatitis level').split(' ')) // prettier-ignore
const ROMAN: Record<string, number> = { II: 2, III: 3, IV: 4, VI: 6, VII: 7, VIII: 8, IX: 9, XI: 11, XII: 12, XIII: 13,
  XIV: 14, XV: 15, XVI: 16, XVIII: 18, XX: 20 } // prettier-ignore
const LEADING_APOSTROPHE = new Set(["'em", "'tis", "'twas", "'til", "'cause", "'bout", "'round", "'n"])
// emoji, their variation selector and joiner, as dizione lists them (one by one, not as sequences)
// eslint-disable-next-line no-misleading-character-class
const EMOJI = new RegExp('[\u{1F000}-\u{1FAFF}☀-➿\u{1F1E6}-\u{1F1FF}←-⇿⬀-⯿️‍⃣]+', 'gud')
const EMOTICON = re(
  String.raw`(?<![A-Za-z0-9])(?:[:;=]['^-]?[DPpOo3|/\\*\]\[}{><]+|[)(\]\[][-^']?[:;=]|<3+|</3|\^_?\^|-_-|[oO]_[oO]|>_<|T_T|\\o/|:'[()])(?![A-Za-z0-9])` +
    String.raw`|[:;]['^-]?[)(]+` +
    String.raw`|(?<![A-Za-z0-9)])=['^-]?[)(]+|=['^-]?[)(]+(?![A-Za-z0-9(.$-])` +
    String.raw`|(?<=[A-Za-z]{2})<3+(?![.,]?\d)`,
)

const PHONE_WORDS = new Set('call dial ring phone telephone tel mobile cell fax hotline text sms whatsapp'.split(' '))
const PHONE_LINKS = new Set('to on at or us me him her them back'.split(' '))
const PIN_WORDS = new Set('code pin extension ext passcode password zip postcode cvv cvc otp'.split(' '))
const STRONG_WORDS = new Set('number id reference ref serial account tracking booking confirmation'.split(' '))
const WEAK_WORDS = new Set(('order ticket invoice case claim policy receipt transaction parcel shipment container ' +
  'payment reservation membership customer batch lot item job issue bug request').split(' ')) // prettier-ignore
const LINK_WORDS = new Set('is was are were ends ending in with starts starting begins beginning'.split(' '))
const NOT_COUNTED = new Set(('is was has does goes ships arrives departs leaves shows needs blocks remains includes ' +
  'contains says reads starts ends closes opens runs takes comes this his its us as yes plus minus less').split(' ')) // prettier-ignore
const NUMBERED = new Set(('ref reference serial account order invoice tracking part model phone tel fax id room flight ' +
  'case ticket policy licence license booking customer vat tax reg registration').split(' ')) // prettier-ignore
const DATE_BEFORE = new Set(('on dated due until till by since from before after for shipped delivered arriving arrives ' +
  'arrive arrived expires expired ends ending starts starting begins beginning monday tuesday wednesday thursday ' +
  'friday saturday sunday mon tue tues wed thu thur thurs fri sat sun').split(' ')) // prettier-ignore
const CLASSIC_FRACTIONS = new Set([2, 3, 4, 5, 6, 8, 10, 16, 32, 64, 100])
const RATIO_WORDS = new Set('ratio ratios scale odds aspect mix split proportion'.split(' '))
const DURATION_WORDS = new Set(
  'lasted took duration runtime recording video clip track lap pace length long'.split(' '),
)
const CLOCK_WORDS = new Set(('window slot open opens opening hours shift session served service available booked ' +
  'appointment delivery arrive arrives arriving').split(' ')) // prettier-ignore
const CAPITAL_WORDS = new Set(('IN ON AT TO OF BY OR IS AS IF NO SO WE HE MY ME DO GO UP AN AM PM BE IT US THE AND FOR ' +
  'BUT NOT ALL').split(' ')) // prettier-ignore
const SHORT_WORDS = new Set('my to in on of go up no do me we us it is at by be so or an as if he hi ok'.split(' '))
const ZONES = 'ET|CT|MT|PT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|GMT|UTC|BST|CET|CEST|IST|AEST|AEDT|JST'
const SIZES = 'XXXL|XXL|XXS|XL|XS|S|M|L'
const LENGTHS = 'mm|cm|km|m|ft|in|inch|inches|feet|foot|metres?|meters?|yards?|yd|px|pixels?'
const VERSIONED = ('version|ver|release|firmware|patch|kernel|build|python|node|nodejs|ios|ipados|macos|watchos|tvos|' +
  'android|chrome|firefox|safari|java|php|ruby|rust|swift|kotlin|golang|django|rails|react|angular|vue|docker|' +
  'kubernetes|ubuntu|debian|fedora|windows|linux|postgres|postgresql|mysql|redis|numpy|pytorch|tensorflow|cuda|' +
  'xcode|gcc|clang|unity|minecraft|typescript|bluetooth|usb|hdmi|opengl|vulkan|directx|' +
  'upgrade[ds]?|update[ds]?|downgrade[ds]?') // prettier-ignore
const SEPARATORS: Record<string, string> = { '.': 'dot', '/': 'slash', '@': 'at', '-': 'dash', _: 'underscore',
  ':': 'colon', '=': 'equals', '&': 'and', '+': 'plus' } // prettier-ignore
const HALF_DAY = String.raw`(?:\s?([AaPp])\.[Mm]\.?|\s?([AaPp])[Mm]\b)`
const AMOUNT = String.raw`(?P<whole>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<fraction>\d+))?`
const MONEY = re(
  String.raw`(?:(?<![\w.])(?P<minus>-)\s?)?(?P<symbol>[$£€¥₹]|(?<![A-Za-z])(?:` +
    CURRENCY_CODES +
    String.raw`)(?![A-Za-z]))\s?` +
    '(?P<inner>-)?' +
    AMOUNT +
    String.raw`(?:\s?(?P<scale>thousand|million|billion|trillion|bn|[kKmMbB])(?![A-Za-z]))?`,
)
const SIGN_AFTER = String.raw`[$£€¥₹](?![\w$£€¥₹])|(?:` + CURRENCY_CODES + ')(?![A-Za-z])'
const MONEY_AFTER = re(
  String.raw`(?:(?<![\w.])(?P<minus>-)\s?)?(?<![\w.,^])` +
    String.raw`(?P<whole>\d{1,3}(?:,\d{3})+|[1-9]\d{0,2}(?: \d{3})+|\d+)(?:\.(?P<fraction>\d+))?` +
    String.raw`\s?(?P<symbol>` +
    SIGN_AFTER +
    ')',
)
const RANGE_BEFORE_SIGN = re(
  String.raw`(?<![\w.,-])(\d[\d,]*(?:\.\d+)?)-(?=\d[\d,]*(?:\.\d+)?\s?(?:` + SIGN_AFTER + '))',
)
const CONTINENTAL = re(String.raw`(?<![\w.,])([1-9]\d{0,2}(?:\.\d{3})+),(\d{1,2})(?!\d|[.,]\d)`)
const COMMA_AFTER_SIGN = re(
  String.raw`((?:[$£€¥₹]|(?<![A-Za-z])(?:` + CURRENCY_CODES + String.raw`))\s?-?\d+),(\d{1,2})(?!\d|[.,]\d)`,
)
const COMMA_BEFORE_SIGN = re(String.raw`(?<![\w.,])(\d+),(\d{1,2})(?=\s?(?:%|` + SIGN_AFTER + '))')
const FORMULA_SIGN = ''
const FORMULA_END = re(
  String.raw`(?<![\w$])(?<!\d\s)\$(?:(?=[^\s\d$.,;:!?)])[^$\n]{0,80}` +
    String.raw`|(?=\d)[^$\n]{0,80}[=^_\\{}+*/<>][^$\n]{0,80}` +
    String.raw`|(?=\s)[^$\n]{0,80}[=^_\\{}+*<>][^$\n]{0,80})\d\s?\$(?![\w$])`,
)
const VULGAR = re(String.raw`(?<![\d.,/])(?:(\d+)\s?)?([½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅐⅛⅜⅝⅞⅑⅒])`)
const ADDRESS = re(
  String.raw`(?P<url>(?:https?://|www\.)[^\s<>\"']+)` +
    String.raw`|(?P<mail>[\w.+-]+@[\w-]+(?:\.[\w-]+)+)` +
    String.raw`|(?P<file>(?<![\w@./-])(?P<folder>/?(?:[\w~-]+/)+)?(?P<stem>[\w-]+(?:\.[\w-]+)*)\.(?P<ending>[A-Za-z][A-Za-z0-9]{0,9})` +
    String.raw`(?![\w-])(?P<path>/[^\s<>\"']*)?)`,
)
const DAY_RANGE = re(
  String.raw`\b(` +
    DAY_PATTERN +
    '|' +
    MONTH_PATTERN +
    String.raw`)\b(\.?)\s?[-–]\s?(` +
    DAY_PATTERN +
    '|' +
    MONTH_PATTERN +
    String.raw`)\b(\.?)`,
)
const RATIO = re(String.raw`(?<![\d:.,])(\d{1,3}):(\d{1,3}(?:,\d{3})+|\d+)(?![\d:]|[.,]\d)`)
const CODE = re(String.raw`(?<![\w.,/-])(#)?([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)(?![\w-]|[.,]\d)`)
const SLASHED = re(String.raw`(?<![\w.,/-])(\d+(?:/\d+){2,})(?![\w/]|[.,]\d)`)
// Python's "(?(lead)...)" (a trail only after a lead) as two alternatives
const GROUPED_CORE = String.raw`[A-Z0-9]*\d[A-Z0-9]*(?: [A-Z0-9]*\d[A-Z0-9]*)+`
const GROUPED = re(
  String.raw`(?<![\w.,:/#+-])(?P<plus>\+)?` +
    String.raw`(?:(?P<lead>[A-Z]{2}) (?P<core>` +
    GROUPED_CORE +
    String.raw`)(?: (?P<trail>[A-Z]{2})(?![\w]))?` +
    String.raw`|(?P<core2>` +
    GROUPED_CORE +
    '))' +
    String.raw`(?![\w]|[.,:/-]\d|\s?[%°])`,
)
const GLUED = re(String.raw`(?<![\w'.])(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+(?![\w']|\.\d)`)

const ACCENT_DECIDES: Record<string, string> = { 'résumé': 'R EH1 Z AH0 M EY2', 'resumé': 'R EH1 Z AH0 M EY2',
  'exposé': 'EH2 K S P OW0 Z EY1', 'rosé': 'R OW0 Z EY1', 'pâté': 'P AA0 T EY1', 'maté': 'M AA1 T EY2',
  'saké': 'S AA1 K EY2', 'lamé': 'L AA0 M EY1' } // prettier-ignore
const ACCENTED = re(String.raw`\b(` + Object.keys(ACCENT_DECIDES).join('|') + String.raw`)(s?)\b`, 'i')
const LETTER_FORMS: Record<string, string> = { ß: 'ss', æ: 'ae', Æ: 'Ae', œ: 'oe', Œ: 'Oe', ø: 'o', Ø: 'O', đ: 'd',
  ł: 'l', Ł: 'L' } // prettier-ignore

function plainLetter(char: string): string {
  if (isascii(char) || !isalpha(char)) return char
  if (char in LETTER_FORMS) return LETTER_FORMS[char]!
  const plain = [...char.normalize('NFKD')].filter((c) => !/\p{M}/u.test(c)).join('')
  return isascii(plain) && isalpha(plain) ? plain : char
}

// ---- Python's int() and the numbers as words --------------------------------

/** `int(text)` for a run of decimal digits of any script. */
export function pyInt(text: string): bigint {
  let out = 0n
  for (const c of text) {
    let cp = c.codePointAt(0)!
    if (cp >= 0x30 && cp <= 0x39) {
      out = out * 10n + BigInt(cp - 0x30)
      continue
    }
    let zero = cp
    while (/\p{Nd}/u.test(String.fromCodePoint(zero - 1)) && cp - zero < 9) zero--
    cp -= zero
    out = out * 10n + BigInt(cp)
  }
  return out
}

const int = (text: string): number => Number(pyInt(text))

export function cardinal(n: number | bigint): string {
  const number = BigInt(n)
  if (number < 0n) return 'minus ' + cardinal(-number)
  if (number < 20n) return ONES[Number(number)]!
  if (number < 100n) {
    const tens = Number(number / 10n)
    const ones = Number(number % 10n)
    return TENS[tens - 2]! + (ones ? ' ' + ONES[ones]! : '')
  }
  if (number < 1000n) {
    const hundreds = Number(number / 100n)
    const rest = number % 100n
    return ONES[hundreds]! + ' hundred' + (rest ? ' ' + cardinal(rest) : '')
  }
  if (number >= 1000n ** BigInt(SCALES.length + 1)) return digits(number.toString())
  const parts: string[] = []
  let left = number
  for (let power = SCALES.length; power > 0; power--) {
    const unit = 1000n ** BigInt(power)
    const count = left / unit
    left = left % unit
    if (count) parts.push(cardinal(count) + ' ' + SCALES[power - 1]!)
  }
  if (left) parts.push(cardinal(left))
  return parts.join(' ')
}

export function ordinal(number: number | bigint): string {
  const words = cardinal(number).split(' ')
  const last = words[words.length - 1]!
  words[words.length - 1] = ORDINAL[last] ?? (last.endsWith('y') ? last.slice(0, -1) + 'ieth' : last + 'th')
  return words.join(' ')
}

export function digits(text: string, zero = 'zero'): string {
  return [...text]
    .filter((c) => isdigit(c))
    .map((c) => (c === '0' ? zero : ONES[int(c)]!))
    .join(' ')
}

export function year(number: number): string {
  if (!(number >= 1000 && number <= 2999)) return cardinal(number)
  const high = Math.floor(number / 100)
  const low = number % 100
  if (low === 0) return high % 10 === 0 ? cardinal(number) : cardinal(high) + ' hundred'
  if (high === 20 && low < 10) return 'two thousand ' + ONES[low]!
  return cardinal(high) + (low < 10 ? ' oh ' + ONES[low]! : ' ' + cardinal(low))
}

export function pairs(number: number): string {
  const high = Math.floor(number / 100)
  const low = number % 100
  if (!(number >= 100 && number <= 9999) || (low === 0 && (high < 10 || high % 10 === 0))) return cardinal(number)
  if (low === 0) return cardinal(high) + ' hundred'
  return cardinal(high) + (low < 10 ? ' oh ' + ONES[low]! : ' ' + cardinal(low))
}

export function numberWords(text: string): string {
  const plain = text.replaceAll(',', '')
  const dot = plain.indexOf('.')
  const whole = dot < 0 ? plain : plain.slice(0, dot)
  const fraction = dot < 0 ? '' : plain.slice(dot + 1)
  if (fraction) return (whole ? cardinal(pyInt(whole)) : '') + ' point ' + digits(fraction)
  if ((whole.length > 1 && whole.startsWith('0')) || whole.length > 18) return digits(whole)
  return cardinal(pyInt(whole))
}

export function duration(hours: number, minutes: number, seconds: number): string {
  const parts = (
    [
      [hours, 'hour'],
      [minutes, 'minute'],
      [seconds, 'second'],
    ] as const
  )
    .filter(([count]) => count)
    .map(([count, name]) => `${cardinal(count)} ${count === 1 ? name : name + 's'}`)
  if (parts.length < 2) return parts.length ? parts[0]! : 'zero seconds'
  return parts.slice(0, -1).join(', ') + ' and ' + parts[parts.length - 1]!
}

function pluralOf(words: string): string {
  const at = words.lastIndexOf(' ')
  const head = at < 0 ? '' : words.slice(0, at)
  let last = at < 0 ? words : words.slice(at + 1)
  last = last.endsWith('y') ? last.slice(0, -1) + 'ies' : last + 's'
  return (head ? head + ' ' : '') + last
}

function fractionWords(top: number, bottom: number, article = false): string {
  const [one, many] = FRACTION[bottom] ?? [ordinal(bottom), ordinal(bottom) + 's']
  if (top !== 1) return `${cardinal(top)} ${many}`
  return (article ? ('aeiou'.includes(one[0]!) ? 'an ' : 'a ') : 'one ') + one
}

const versionPart = (part: string): string => (part.length > 2 ? digits(part) : numberWords(part))
const minor = (part: string): string => (part.startsWith('0') ? digits(part, 'oh') : cardinal(pyInt(part)))

function shortNumber(text: string): boolean {
  if (text.startsWith('0')) return false
  const rest = new Set(text.slice(1))
  return text.length <= 2 || (rest.size === 1 && rest.has('0'))
}

const spelled = (letters: string): string => `${SPELL}${letters}${END}`

const CLAUSE_BREAK = /[.!?;…]\s|[()"]/u
function before(text: string, position: number, count = 4): string[] {
  const pieces = text.slice(Math.max(0, position - 80), position).split(CLAUSE_BREAK)
  const clause = pieces[pieces.length - 1]!
  return (clause.match(/[A-Za-z]+/g) ?? []).map((w) => w.toLowerCase()).slice(-count)
}

function identifier(text: string, position: number): string | null {
  const words = before(text, position)
  if (!words.length) return null
  const last = words[words.length - 1]!
  if (PHONE_WORDS.has(last) || (PHONE_LINKS.has(last) && words.slice(-3, -1).some((w) => PHONE_WORDS.has(w))))
    return 'phone'
  let index = words.length - 1
  while (index > 0 && LINK_WORDS.has(words[index]!)) index--
  for (const [kind, group] of [
    ['phone', PHONE_WORDS],
    ['pin', PIN_WORDS],
    ['strong', STRONG_WORDS],
    ['weak', WEAK_WORDS],
  ] as const) {
    if (group.has(words[index]!)) return kind
  }
  return last === 'enter' ? 'enter' : null
}

const DECIMAL_COMMA = re(String.raw`\d+,\d{1,2}`)
const decimalComma = (text: string): string => (fullmatch(DECIMAL_COMMA, text) ? text.replaceAll(',', '.') : text)

function addressDigits(text: string): string {
  if (text.length <= 2 && !(text.length === 2 && text[0] === '0')) return cardinal(pyInt(text))
  if (text.length === 4 && int(text) >= 1900 && int(text) <= 2099) return year(int(text))
  return digits(text)
}

/** Python's `x in "abc"` for a one-character (or empty) string. */
const among = (x: string | undefined, chars: string): boolean => x !== undefined && chars.includes(x)

const WORD_AFTER = re(String.raw`\s*([A-Za-z]+)`)
const LETTERS_AT_START = re(String.raw`[A-Za-z']+`)

/** A case-insensitive spelling of `name` for a pattern without the flag (Python's `(?-i:...)` inside `re.I`). */
function folded(name: string): string {
  return [...name]
    .map((c) => {
      const l = c.toLowerCase()
      const u = c.toUpperCase()
      return l !== u && [...l].length === 1 && [...u].length === 1 ? `[${l}${u}]` : escape(c)
    })
    .join('')
}

// ---- rewriting -------------------------------------------------------------

export class Normalizer {
  readonly abbreviations = new Map<string, [string, string]>()
  readonly units = new Map<string, [string, string]>()
  readonly exactUnits = new Map<string, [string, string]>()
  readonly chat = new Map<string, string>()
  readonly replies = new Set<string>()
  readonly endings: Map<string, string>
  readonly unitNames: string
  private readonly unit: RegExp
  private readonly area = re(String.raw`(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(km|cm|mm|m|ft|mi|yd|in)([²³])`)
  private readonly areaPlain = re(String.raw`(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(km|cm|mm|m|ft)([23])(?![\w])`)
  inline: [string, Phone[]][] = []
  dayFirst = false
  dated = false

  readonly isWord: (word: string) => boolean

  constructor(isWord: (word: string) => boolean = () => true) {
    this.isWord = isWord
    for (const [, columns] of rows('abbreviations.txt')) this.abbreviations.set(columns[0]!, [columns[1]!, columns[2]!])
    for (const [, columns] of rows('units.txt')) {
      if (columns.length > 3 && columns[3] === 'exact') this.exactUnits.set(columns[0]!, [columns[1]!, columns[2]!])
      else this.units.set(columns[0]!.toLowerCase(), [columns[1]!, columns[2]!])
    }
    for (const [number, columns] of rows('chat.txt')) {
      if (columns.length > 2 && !(columns.length === 3 && columns[2] === 'alone'))
        throw new Error(`chat.txt line ${number}: after the words to say there can only be "alone"`)
      this.chat.set(columns[0]!, columns.length > 1 ? columns[1]! : '-')
      if (columns.length > 2) this.replies.add(columns[0]!)
    }
    this.endings = mapping('endings.txt')
    // Python escapes none of these names (no special characters), so its order is that of the names themselves
    const names = [...new Set([...this.units.keys(), ...this.exactUnits.keys()])].sort(
      (a, b) => [...b].length - [...a].length || (a < b ? -1 : a > b ? 1 : 0),
    )
    this.unitNames = names.map(escape).join('|')
    this.unit = re(
      String.raw`(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:[-–](\d[\d,]*(?:\.\d+)?))?\s?(` +
        names.map(folded).join('|') +
        String.raw`)(?![A-Za-z])(\.(?=\s+[a-z]))?`,
    )
  }

  // -- pieces of the text that are not words --

  private amount(symbol: string, whole: string, fraction: string | undefined, scaleIn: string | undefined): string {
    const [one, many, cent, cents] = CURRENCY[symbol]!
    if (scaleIn) {
      const scale = SCALE_LETTER[scaleIn.toLowerCase()] ?? scaleIn.toLowerCase()
      return `${numberWords(whole + (fraction ? '.' + fraction : ''))} ${scale} ${many}`
    }
    if (fraction && (fraction.length > 2 || !cent)) return `${numberWords(whole + '.' + fraction)} ${many}`
    const amount = pyInt(whole)
    const change = int((fraction || '0').padEnd(2, '0'))
    const parts: string[] = []
    if (amount || !change) parts.push(`${cardinal(amount)} ${amount === 1n ? one : many}`)
    if (change && cent) parts.push(`${cardinal(change)} ${change === 1 ? cent : cents}`)
    return parts.join(' and ')
  }

  private static continental(m: Match): string {
    const groups = m.group(1)!.split('.')
    if (groups.length === 4 && groups.every((group) => int(group) <= 255)) return m.group(0)!
    return groups.join(',') + '.' + m.group(2)!
  }

  private money(m: Match): string {
    const whole = m.group('whole')!.replaceAll(',', '').replaceAll(' ', '')
    const words = this.amount(m.group('symbol')!, whole, m.group('fraction'), m.group('scale'))
    return (m.group('minus') || m.group('inner') ? ' minus ' : ' ') + words + ' '
  }

  private static halfDay(m: Match, dotted: string | undefined, plain: string | undefined): string {
    if (!(dotted || plain)) return ''
    const following = m.string.slice(m.end()).trimStart()
    const first = reMatch(LETTERS_AT_START, following)
    const closes =
      !!dotted &&
      m.group(0)!.endsWith('.') &&
      (!following || (isupper(following[0]!) && first !== null && SENTENCE_STARTERS.has(first.group(0)!.toLowerCase())))
    return `${SPELL}${(dotted || plain)!.toUpperCase()}M${END}${closes ? ' .' : ''} `
  }

  private time(m: Match): string {
    const hour = int(m.group(1)!)
    const minute = int(m.group(2)!)
    const second = m.group(3)
    const half = m.group(4) || m.group(5)
    const words3 = before(m.string, m.start(), 3)
    if (second !== undefined && !half && !(words3.length && words3[words3.length - 1] === 'at'))
      return ` ${duration(hour, minute, int(second))} `
    if (hour > 24 && second === undefined && !half) {
      const after = reMatch(WORD_AFTER, m.string.slice(m.end()))
      if (words3.some((w) => DURATION_WORDS.has(w)) || (after && DURATION_WORDS.has(after.group(1)!.toLowerCase())))
        return ` ${duration(0, hour, minute)} `
      return ` ${cardinal(hour)} ${minute < 10 ? 'oh ' : ''}${cardinal(minute)} `
    }
    if (hour > 24) return m.group(0)!
    let words = cardinal(hour)
    if (minute === 0) {
      if (half) {
        // the half of the day follows
      } else if (hour === 0) words = 'midnight'
      else if (hour > 12) words += ' hundred'
      else words += " o'clock"
    } else {
      words += (minute < 10 ? ' oh ' : ' ') + cardinal(minute)
    }
    if (second) words += ' and ' + cardinal(int(second)) + (int(second) === 1 ? ' second' : ' seconds')
    return ` ${words} ` + Normalizer.halfDay(m, m.group(4), m.group(5))
  }

  private dotTimes(m: Match): string {
    const following = m.string.slice(m.end())
    if (
      int(m.group(1)!) > 23 ||
      int(m.group(4)!) > 23 ||
      !before(m.string, m.start()).some((w) => CLOCK_WORDS.has(w)) ||
      reMatch(
        re(String.raw`\s?(?:%|percent|million|billion|` + this.unitNames + String.raw`)(?![A-Za-z])`, 'i'),
        following,
      )
    )
      return m.group(0)!
    return `${m.group(1)}:${m.group(2)}${m.group(3)}${m.group(4)}:${m.group(5)}`
  }

  private ratio(m: Match): string {
    const first = m.group(1)!
    const second = m.group(2)!
    if (first.length <= 2 && second.length === 2 && '012345'.includes(second[0]!)) {
      const after = reMatch(WORD_AFTER, m.string.slice(m.end()))
      if (!(
        before(m.string, m.start(), 3).some((w) => RATIO_WORDS.has(w)) ||
        (after && RATIO_WORDS.has(after.group(1)!.toLowerCase()))
      ))
        return m.group(0)!
    }
    return ` ${numberWords(first)} to ${numberWords(second)} `
  }

  private date(month: number, day: number, yearText?: string, the = false): string | null {
    if (!(month >= 1 && month <= 12) || !(day >= 1 && day <= 31)) return null
    let words = the ? `the ${ordinal(day)} of ${MONTHS[month - 1]}` : `${MONTHS[month - 1]} ${ordinal(day)}`
    if (yearText)
      words +=
        ' ' +
        (yearText.length === 4 ? year(int(yearText)) : (int(yearText) < 10 ? 'oh ' : '') + cardinal(pyInt(yearText)))
    return ` ${words} `
  }

  private slashDate(m: Match): string {
    let first = int(m.group(1)!)
    let second = int(m.group(2)!)
    if ((first > 12 && 12 >= second) || (this.dayFirst && second <= 12)) [first, second] = [second, first]
    return this.date(first, second, m.group(3)) ?? m.group(0)!
  }

  private dotDate(m: Match): string {
    let day = int(m.group(1)!)
    let month = int(m.group(2)!)
    if (month > 12 && 12 >= day) [day, month] = [month, day]
    return this.date(month, day, m.group(3)) ?? m.group(0)!
  }

  private dayMonth(m: Match): string {
    const month = m.group(2)!
    const spoken = this.date(MONTH_NUMBER.get(month.toLowerCase().slice(0, 3))!, int(m.group(1)!), m.group(4), true)
    if (spoken === null) return m.group(0)!
    return spoken + (m.group(3) && MONTHS.includes(month) ? '. ' : '')
  }

  private daysOfMonth(m: Match): string {
    const monthFirst = isalpha(m.group(1)![0]!)
    const [month, firstText, secondText] = monthFirst
      ? [m.group(1)!, m.group(2)!, m.group(3)!]
      : [m.group(3)!, m.group(1)!, m.group(2)!]
    const first = int(firstText)
    const second = int(secondText)
    if (!(first >= 1 && first <= 31 && second >= 1 && second <= 31)) return m.group(0)!
    const name = MONTHS[MONTH_NUMBER.get(month.toLowerCase().slice(0, 3))! - 1]!
    const words = monthFirst
      ? `${name} ${ordinal(first)} to ${ordinal(second)}`
      : `the ${ordinal(first)} to the ${ordinal(second)} of ${name}`
    return ` ${words}${m.group(4) ? ' ' + year(int(m.group(4)!)) : ''} `
  }

  private dayRange(m: Match): string {
    const names: string[] = []
    for (const name of [m.group(1)!, m.group(3)!]) {
      const k = name.toLowerCase()
      names.push(this.abbreviations.has(k) ? this.abbreviations.get(k)![1] : (DAY_NAMES[k] ?? name))
    }
    const following = m.string.slice(m.end()).trimStart()
    const first = reMatch(LETTERS_AT_START, following)
    const ends =
      !!m.group(4) &&
      (!following || (isupper(following[0]!) && first !== null && SENTENCE_STARTERS.has(first.group(0)!.toLowerCase())))
    return `${names[0]} to ${names[1]}${ends ? ' .' : ''}`
  }

  private shortDate(m: Match): string | null {
    const firstText = m.group(1)!
    const secondText = m.group(2)!
    if (firstText.length > 2 || secondText.length > 2) return null
    let first = int(firstText)
    let second = int(secondText)
    const words = before(m.string, m.start())
    if (first >= 1 && first <= 12 && second >= 20 && second <= 49 && words.includes('card'))
      return ` ${MONTHS[first - 1]} ${year(2000 + second)} `
    const last = words.length ? words[words.length - 1]! : ''
    const classic = first < second && CLASSIC_FRACTIONS.has(second)
    const zero = (firstText.length === 2 && firstText[0] === '0') || (secondText.length === 2 && secondText[0] === '0')
    if (!(
      zero ||
      (first > 12 && first <= 31 && second <= 12 && !(first === 24 && second === 7)) ||
      (DATE_BEFORE.has(last) && (!classic || last === 'on')) ||
      (this.dated && !classic && first !== second)
    ))
      return null
    if (first > 12 || (this.dayFirst && second <= 12)) [first, second] = [second, first]
    return this.date(first, second)
  }

  private fraction(m: Match): string {
    const spoken = this.shortDate(m)
    if (spoken) return spoken
    const top = pyInt(m.group(1)!)
    const bottom = pyInt(m.group(2)!)
    if ((top === 24n && bottom === 7n) || top === bottom) return ` ${cardinal(top)} ${cardinal(bottom)} `
    if (!(top > 0n && top < 100n) || !(bottom > 1n && bottom <= 100n))
      return ` ${cardinal(top)} over ${cardinal(bottom)} `
    return ` ${fractionWords(Number(top), Number(bottom))} `
  }

  private mixedFraction(m: Match): string {
    const top = int(m.group(2)!)
    const bottom = int(m.group(3)!)
    if (!(top > 0 && top < bottom)) return m.group(0)!
    return ` ${numberWords(m.group(1)!)} and ${fractionWords(top, bottom, true)} `
  }

  private vulgar(m: Match): string {
    const [top, bottom] = m.group(2)!.normalize('NFKC').split('⁄').map(int) as [number, number]
    if (m.group(1) === undefined) return ` ${top}/${bottom} `
    return ` ${cardinal(pyInt(m.group(1)!))} and ${fractionWords(top, bottom, true)} `
  }

  private static phone(m: Match): string {
    const groups = m.groups().filter((g): g is string => !!g)
    return ' ' + groups.map((g) => (g === '800' ? 'eight hundred' : digits(g, 'oh'))).join(', ') + ' '
  }

  private measure(m: Match): string {
    const unit = m.group(3)!
    const names = this.exactUnits.get(unit) ?? this.units.get(unit.toLowerCase())
    if (
      names === undefined ||
      ([...unit].length === 1 && ['-', "'", '/'].includes(m.string.slice(m.end(3), m.end(3) + 1)))
    )
      return m.group(0)!
    const [one, many] = names
    const value = decimalComma(m.group(1)!)
    const upper = m.group(2) && decimalComma(m.group(2)!)
    if (upper) return ` ${numberWords(value)} to ${numberWords(upper)} ${many} `
    return ` ${numberWords(value)} ${value === '1' || value === '1.0' ? one : many} `
  }

  private square(m: Match): string {
    const unit = m.group(2)!
    const [one, many] = (unit === 'in' ? ['inch', 'inches'] : (this.exactUnits.get(unit) ?? this.units.get(unit)))!
    const kind = '²2'.includes(m.group(3)!) ? 'square' : 'cubic'
    const value = decimalComma(m.group(1)!)
    return ` ${numberWords(value)} ${kind} ${value === '1' || value === '1.0' ? one : many} `
  }

  private counted(text: string, end: number): boolean {
    const after = reMatch(WORD_AFTER, text.slice(end))
    if (!after) return false
    const w = after.group(1)!.toLowerCase()
    if (['people', 'men', 'women', 'children', 'feet', 'miles'].includes(w)) return true
    return w.length > 3 && w.endsWith('s') && !w.endsWith('ss') && !NOT_COUNTED.has(w) && this.isWord(w.slice(0, -1))
  }

  private number(m: Match, text: string): string {
    const value = m.group(0)!
    const plain = value.replaceAll(',', '')
    if (!plain.includes('.') && !value.includes(',') && !plain.startsWith('0') && plain.length > 2) {
      const kind = identifier(text, m.start())
      const limit = ({ phone: 3, pin: 3, strong: 4, enter: 4, weak: 5 } as Record<string, number>)[kind ?? ''] ?? 7
      if (
        plain.length >= limit &&
        !(kind === null && plain.endsWith('000')) &&
        !(kind === 'enter' && plain.length === 4 && int(plain) >= 1900 && int(plain) <= 2099) &&
        !((kind === null || kind === 'phone' || kind === 'weak') && this.counted(text, m.end()))
      )
        return ` ${digits(plain, kind === 'phone' ? 'oh' : 'zero')} `
    }
    if (!plain.includes('.') && !value.includes(',') && plain.length === 4 && !plain.startsWith('0')) {
      const number = int(plain)
      const words =
        text
          .slice(0, m.start())
          .slice(-24)
          .match(/[A-Za-z]+/g) ?? []
      const after = reMatch(WORD_AFTER, text.slice(m.end()))
      const counted = !!after && /(?:[^s]s|people|men|women|children|feet|miles)$/.test(after.group(1)!.toLowerCase())
      const lastWord = words.length ? words[words.length - 1]!.toLowerCase() : ''
      const dated = words.length > 0 && (YEAR_CONTEXT.has(lastWord) || MONTH_NUMBER.has(lastWord.slice(0, 3)))
      const model = search(re(String.raw`(?<![A-Za-z])([A-Z]{2,5}) $`), text.slice(0, m.start()))
      if (model && !CAPITAL_WORDS.has(model.group(1)!) && !counted) return ` ${pairs(number)} `
      if ((number >= 1100 && number <= 1999) || (number >= 2000 && number <= 2099 && (dated || !counted)))
        return ` ${year(number)} `
    }
    return ` ${numberWords(value)} `
  }

  // -- codes and reference numbers --

  private readsAsWord(letters: string): boolean {
    const low = letters.toLowerCase()
    if (low.length < 3 || !this.isWord(low)) return false
    return low.length > 3 || ['s', 'es', 'ed', 'ing'].some((ending) => this.isWord(low + ending))
  }

  private static alnum(group: string, zero = 'zero'): string {
    return findall(re('[A-Za-z]+|\\d+'), group)
      .map((run) => (isdigit(run) ? digits(run, zero) : spelled(run)))
      .join(' ')
  }

  private static digitGroups(groups: string[], zero = 'zero'): string {
    if (groups.every((group) => shortNumber(group) && group.length <= 2))
      return groups.map((group) => cardinal(pyInt(group))).join(' , ')
    return groups.map((group) => digits(group, zero)).join(' , ')
  }

  private codeWords(groups: string[], m: Match): string | null {
    const token = groups.join('-')
    if (![...token].some((c) => isdigit(c))) return null
    const kinds = groups.map((g) => (isdigit(g) ? 'd' : [...g].some((c) => isdigit(c)) ? 'm' : isupper(g) ? 'u' : 'l'))
    if (kinds.every((kind) => kind === 'd')) {
      if (groups.length === 2) {
        const years = groups.every((g) => g.length === 4 && int(g) >= 1000 && int(g) <= 2100)
        if (!(
          m.group(1) ||
          groups.some((g) => g.length > 1 && g[0] === '0') ||
          (Math.min(...groups.map((g) => g.length)) >= 4 && !years) ||
          ['strong', 'pin'].includes(identifier(m.string, m.start()) ?? '')
        ))
          return null
      }
      return Normalizer.digitGroups(groups)
    }
    if (groups.length === 2 && kinds[1] === 'd' && 'ul'.includes(kinds[0]!)) {
      const [letters, number] = groups as [string, string]
      if ((kinds[0] === 'l' && letters.length > 1) || this.readsAsWord(letters)) return null
      if (shortNumber(number)) return `${spelled(letters)} ${cardinal(pyInt(number))}`
      return `${spelled(letters)} , ${digits(number)}`
    }
    if (
      groups.length === 2 &&
      kinds[0] === 'd' &&
      (kinds[1] === 'l' || [...groups[1]!].length === 1 || fullmatch(re(String.raw`\d+[A-Za-z]+`), groups[1]!))
    )
      return null
    if (
      kinds.some(
        (kind, k) =>
          (kind === 'l' && groups[k]!.length > 2) ||
          (kind === 'm' && fullmatch(re(String.raw`\d+(?:s|st|nd|rd|th)`), groups[k]!)),
      )
    )
      return null
    if (![...token].some((c) => isupper(c)) && (kinds.includes('l') || kinds.includes('m'))) return null
    const parts: string[] = []
    groups.forEach((group, k) => {
      const kind = kinds[k]
      if (kind === 'd') parts.push(digits(group))
      else if (kind === 'm') parts.push(Normalizer.alnum(group))
      else if (group.length > 3 && this.isWord(group.toLowerCase())) parts.push(group)
      else parts.push(group.length <= 5 ? spelled(group) : group)
    })
    return parts.join(' , ')
  }

  private code(m: Match): string {
    const words = this.codeWords(m.group(2)!.split('-'), m)
    if (words === null) return m.group(0)!
    return (m.group(1) ? ' number ' : ' ') + words + ' '
  }

  private grouped(m: Match): string {
    const text = m.string
    const plus = m.group('plus')
    let lead = m.group('lead')
    let trail = m.group('trail')
    const core = (m.group('core') ?? m.group('core2'))!.split(' ')
    let head = ''
    let tail = ''
    if (lead !== undefined && CAPITAL_WORDS.has(lead)) {
      head = lead + ' '
      tail = trail ? ' ' + trail : ''
      lead = undefined
      trail = undefined
    }
    const kind = identifier(text, m.start())
    const pure = core.every((part) => isdigit(part))
    if (pure && !plus && !lead) {
      const following = text.slice(m.end())
      const amount =
        this.counted(text, m.end()) ||
        reMatch(re(String.raw`\s?(?:` + this.unitNames + String.raw`)(?![A-Za-z])`, 'i'), following) !== null
      if (
        kind === null &&
        core[0]![0] !== '0' &&
        core[0]!.length <= 3 &&
        core.slice(1).every((part) => part.length === 3) &&
        (core[core.length - 1] === '000' || (amount && core.length > 2))
      )
        return head + core.join(',') + tail
      if (!(
        kind ||
        (core[0]!.length > 1 && core[0]![0] === '0') ||
        (core.length > 2 && core.some((part) => part.length > 2))
      ))
        return m.group(0)!
    } else if (!pure && !lead && !plus && kind === null) {
      return m.group(0)!
    }
    const zero =
      plus || kind === 'phone' || (pure && core[0]![0] === '0' && core.reduce((n, part) => n + part.length, 0) >= 10)
        ? 'oh'
        : 'zero'
    let spoken = pure
      ? Normalizer.digitGroups(core, zero)
      : core.map((part) => (isdigit(part) ? digits(part, zero) : Normalizer.alnum(part, zero))).join(' , ')
    if (lead) spoken = `${spelled(lead)} , ${spoken}`
    if (trail) spoken = `${spoken} , ${spelled(trail)}`
    return head + (plus ? ' plus ' : ' ') + spoken + ' ' + tail
  }

  private glued(m: Match): string {
    const out: string[] = []
    for (const run of findall(re('[A-Za-z]+|\\d+'), m.group(0)!)) {
      if (isdigit(run)) out.push(shortNumber(run) ? cardinal(pyInt(run)) : digits(run))
      else if (run.length === 1 || ((isupper(run) || run.length <= 3) && !this.readsAsWord(run))) out.push(spelled(run))
      else out.push(run)
    }
    return ' ' + out.join(' ') + ' '
  }

  // -- web addresses and file names --

  private addressPart(part: string): string {
    const low = part.toLowerCase()
    if (part.length > 1 && this.isWord(low) && (part.length > 2 || SHORT_WORDS.has(low))) return part
    return part.length <= 3 ? spelled(part) : part
  }

  private spokenAddress(core: string): string {
    const pieces = findall(re('[A-Za-z]+|\\d+|.'), core)
    const mail = core.includes('@')
    let host = false
    const out: string[] = []
    pieces.forEach((piece, index) => {
      const following = index + 1 < pieces.length ? pieces[index + 1]! : ''
      if (piece === '@') host = true
      if (isdigit(piece)) out.push(addressDigits(piece))
      else if (!isalpha(piece)) out.push(SEPARATORS[piece] ?? '')
      else if (
        index &&
        pieces[index - 1] === '.' &&
        (host || !mail) &&
        (['', '/', ':', '?', '#'].includes(following) ||
          (piece.length <= 3 && (this.endings.has(piece.toLowerCase()) || !this.isWord(piece.toLowerCase()))))
      ) {
        out.push(this.endings.get(piece.toLowerCase()) === 'word' ? piece.toLowerCase() : spelled(piece))
      } else {
        let letters = ''
        for (const part of findall(re('[A-Z]?[a-z]+|[A-Z]+(?![a-z])'), piece)) {
          const spoken = this.addressPart(part)
          if (spoken.startsWith(SPELL)) {
            letters += part
            continue
          }
          if (letters) {
            out.push(spelled(letters))
            letters = ''
          }
          out.push(spoken)
        }
        if (letters) out.push(spelled(letters))
      }
    })
    return out.filter(Boolean).join(' ')
  }

  private address(m: Match): string {
    const whole = m.group(0)!
    const text = rstrip(whole, '.,;:!?)')
    const tail = whole.slice(text.length)
    if (m.group('file')) {
      const ending = m.group('ending')!
      const known = this.endings.has(ending.toLowerCase())
      const marked = !!(m.group('folder') || m.group('path')) || m.group('stem')!.includes('_')
      if (!(
        known ||
        (marked && ending.length <= 5 && (islower(ending) || isupper(ending)) && !this.isWord(ending.toLowerCase()))
      ))
        return whole
    }
    const core = rstrip(text.replace(/^(?:https?:\/\/)?(?:www\.)?/, ''), '/')
    return ` ${this.spokenAddress(core)} ${tail}`
  }

  private chatForm(m: Match): string {
    const said = this.chat.get(m.group(0)!.toLowerCase())
    if (said === undefined) return m.group(0)!
    if (said === '-') return ' '
    return (
      ' ' +
      split(said)
        .map((part) => (part.startsWith('*') ? `${SPELL}${part.slice(1)}${END}` : part))
        .join(' ') +
      ' '
    )
  }

  private inlined(m: Match): string {
    const symbols = split(m.group(2)!)
    if (!symbols.length || symbols.every((symbol) => [...symbol].length === 1 && isascii(symbol))) return m.group(0)!
    let phones: Phone[]
    try {
      phones = isArpabet(symbols) ? convert('', symbols) : fromText(symbols.join(' '))
    } catch {
      return m.group(0)!
    }
    this.inline.push([m.group(1) || '', phones])
    return ` ${INLINE}${String.fromCodePoint(INLINE_BASE + this.inline.length - 1)}${INLINE} `
  }

  private static dotted(m: Match, text: string): string {
    const letters = m.group(0)!.replaceAll('.', '')
    const k = letters.toLowerCase()
    const following = text.slice(m.end()).trimStart()
    const words = k === 'eg' ? 'for example' : k === 'ie' ? 'that is' : `${SPELL}${letters.toUpperCase()}${END}`
    const first = reMatch(LETTERS_AT_START, following)
    const endsSentence =
      !following || (isupper(following[0]!) && !!first && SENTENCE_STARTERS.has(first.group(0)!.toLowerCase()))
    return ` ${words}${endsSentence && m.group(0)!.endsWith('.') ? ' .' : ''} `
  }

  private abbreviation(m: Match, text: string): string {
    const token = m.group(1)!
    const stop = m.group(2)!
    const k = token.toLowerCase()
    const following = text.slice(m.end())
    const nextWord = reMatch(re(String.raw`\s*([A-Za-z]+|\d)`), following)
    const capitalNext = !!nextWord && isupper(nextWord.group(1)![0]!)
    const capitalBefore = search(re(String.raw`[A-Z][a-z]+\s+$`), text.slice(0, m.start())) !== null
    if ((k === 'st' || k === 'dr') && isupper(token[0]!)) {
      if (capitalNext && (stop || k === 'dr')) return k === 'st' ? ' saint ' : ' doctor '
      if (capitalBefore && stop) return (k === 'st' ? ' street' : ' drive') + (!following.trim() ? ' .' : ' ')
      return m.group(0)!
    }
    if (!this.abbreviations.has(k)) return m.group(0)!
    const [when, found] = this.abbreviations.get(k)!
    let words = found
    const numberNext = !!nextWord && isdigit(nextWord.group(1)!)
    if (when === 'title') {
      if (!(isupper(token[0]!) && (capitalNext || stop))) return m.group(0)!
    } else if (when === 'number') {
      const previous = search(re(String.raw`([A-Za-z]+)\.?\s+$`), text.slice(0, m.start()))
      if (!(stop && (numberNext || (k === 'no' && previous && NUMBERED.has(previous.group(1)!.toLowerCase())))))
        return m.group(0)!
    } else if (!stop || (when === 'small' && isupper(token))) {
      return m.group(0)!
    } else if (
      ['mar', 'sun', 'sat', 'jan', 'dec', 'max', 'min', 'est', 'no'].includes(k) &&
      !isupper(token[0]!) &&
      !numberNext
    ) {
      if (!(['max', 'min', 'est'].includes(k) && nextWord && islower(nextWord.group(1)![0]!))) return m.group(0)!
    }
    if (k === 'est' && !numberNext) words = 'estimated'
    const endsSentence =
      when !== 'title' &&
      (!following.trim() || (capitalNext && SENTENCE_STARTERS.has(nextWord!.group(1)!.toLowerCase())))
    return ` ${words}${endsSentence ? ' .' : ''} `
  }

  /** The text with every non-word written out, markers included. */
  rewrite(textIn: string): string {
    let text = sub(VULGAR, (m) => this.vulgar(m), textIn)
    text = sub(this.area, (m) => this.square(m), text)
    text = text.replaceAll('²', ' squared ').replaceAll('³', ' cubed ')
    text = text.normalize('NFKC').replaceAll('⁄', '/').replaceAll('−', ' minus ')
    if (!isascii(text)) {
      text = sub(
        ACCENTED,
        (m) => '{' + ACCENT_DECIDES[m.group(1)!.toLowerCase()]! + (m.group(2) ? ' Z' : '') + '}',
        text,
      )
      text = [...text].map(plainLetter).join('')
    }
    text = sub(re('(?<=\\d)–(?=\\d)'), '-', text)
    text = sub(DAY_RANGE, (m) => this.dayRange(m), text)
    for (const [old, replacement] of [
      ['‘', "'"],
      ['’', "'"],
      ['‚', "'"],
      ['‛', "'"],
      ['`', "'"],
      ['´', "'"],
      ['“', '"'],
      ['”', '"'],
      ['„', '"'],
      ['‟', '"'],
      ['«', '"'],
      ['»', '"'],
      ['–', '—'],
      ['―', '—'],
      ['[', '('],
      [']', ')'],
    ] as const)
      // prettier-ignore
      text = text.replaceAll(old, replacement)
    this.inline = []
    text = sub(re(String.raw`([A-Za-z']+)?\{([^{}]*)\}`), (m) => this.inlined(m), text)
    text = text.replaceAll('{', '(').replaceAll('}', ')')
    text = sub(ADDRESS, (m) => this.address(m), text)
    text = text.replaceAll('→', ' to ')
    text = sub(EMOJI, ' ', text)
    text = sub(EMOTICON, ' ', text)
    text = sub(re(String.raw`\.{3,}`), '…', text)
    text = sub(re(String.raw`(?<=\d)\s-\s(?=\d[\d.,]*\s?=)`), ' minus ', text)
    text = sub(re(String.raw`(?<![\d.,/-])(\d[\d.,]*)-(?=\d[\d.,]*=)`), '\\1 minus ', text)
    text = sub(re(String.raw`-{2,}|\s-\s`), '—', text)
    text = sub(re(String.raw`\bw/o\b`, 'i'), ' without ', text)
    text = sub(re(String.raw`\bw/(?=\s|$)`, 'i'), ' with ', text)
    text = sub(re(String.raw`\bn/a\b`, 'i'), ' not applicable ', text)
    text = text.replaceAll("'n'", " 'n ")
    if (search(re('[<>≥≤≠]|!='), text)) text = Normalizer.markComparisons(text)
    if (search(re('\\d'), text)) text = this.figures(text)
    return this.signs(text)
  }

  private figures(textIn: string): string {
    let text = sub(FORMULA_END, (m) => m.group(0)!.slice(0, -1) + FORMULA_SIGN, textIn)
    text = sub(CONTINENTAL, (m) => Normalizer.continental(m), text)
    text = sub(COMMA_AFTER_SIGN, '\\1.\\2', text)
    text = sub(COMMA_BEFORE_SIGN, '\\1.\\2', text)
    text = sub(MONEY, (m) => this.money(m), text)
    text = sub(RANGE_BEFORE_SIGN, '\\1 to ', text)
    text = sub(MONEY_AFTER, (m) => this.money(m), text)
    text = sub(
      re(String.raw`(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?¢`),
      (m) => ` ${numberWords(m.group(1)!)} ${m.group(1) === '1' ? 'cent' : 'cents'} `,
      text,
    )

    // times of day, lengths of time, ratios
    const clock = String.raw`\d{1,2}:[0-5]\d(?:\s?[AaPp]\.?[Mm]\.?)?|\b\d{1,2}\s?[AaPp]\.?[Mm]\.?`
    text = sub(
      re('(' + clock + String.raw`)\s(` + ZONES + String.raw`)\b`),
      (m) => `${m.group(1)} ${spelled(m.group(2)!)}`,
      text,
    )
    text = sub(re(String.raw`(?<![\d.,:])(\d{1,2})\.([0-5]\d)(?=\s?[AaPp]\.?[Mm]\b)`), '\\1:\\2', text)
    text = sub(
      re(
        String.raw`(?<![\d.,:])(\d{1,2})\.(00|15|30|45)(\s?(?:to|until|till|and|[-—])\s?)(\d{1,2})\.(00|15|30|45)(?!\d|\.\d)`,
      ),
      (m) => this.dotTimes(m),
      text,
    )
    text = sub(re('(' + clock + String.raw`)\s?[-—]\s?(?=\d{1,2}(?::[0-5]\d|\s?[AaPp]\.?[Mm]\b))`), '\\1 to ', text)
    text = sub(RATIO, (m) => this.ratio(m), text)
    text = sub(
      re(String.raw`(?<![\d:])(\d{1,2}):([0-5]\d)(?::([0-5]\d))?` + HALF_DAY + String.raw`?(?![\d:])`),
      (m) => this.time(m),
      text,
    )
    text = sub(
      re(String.raw`(?<![\d.:,])(\d{1,2})` + HALF_DAY + String.raw`(?![A-Za-z])`),
      (m) => ` ${cardinal(pyInt(m.group(1)!))} ` + Normalizer.halfDay(m, m.group(2), m.group(3)),
      text,
    )

    // dates
    const seen = findallGroups(re(String.raw`(?<![\d/.])(\d{1,2})/(\d{1,2})(?:/(?:\d{4}|\d{2}))?(?![\d/])`), text)
      .map(([a, b]) => [int(a!), int(b!)] as const)
      .filter(([a, b]) => !(a === 24 && b === 7))
    this.dayFirst = seen.some(([a, b]) => a > 12 && 12 >= b) && !seen.some(([a, b]) => b > 12 && 12 >= a)
    this.dated = search(re(String.raw`(?<![\d/])\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})(?![\d/])`), text) !== null
    text = sub(
      re(String.raw`(?<![\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])`),
      (m) => this.date(int(m.group(2)!), int(m.group(3)!), m.group(1)) ?? m.group(0)!,
      text,
    )
    text = sub(
      re(String.raw`(?<![\d/])(\d{4})/(\d{1,2})/(\d{1,2})(?![\d/])`),
      (m) => this.date(int(m.group(2)!), int(m.group(3)!), m.group(1)) ?? m.group(0)!,
      text,
    )
    text = sub(re(String.raw`(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d|\.\d)`), (m) => this.dotDate(m), text)
    text = sub(re(String.raw`(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\d/])`), (m) => this.slashDate(m), text)
    text = sub(
      re(
        String.raw`\b(` +
          MONTH_PATTERN +
          String.raw`)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s?[-—]\s?(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?(?![\d:])`,
      ),
      (m) => this.daysOfMonth(m),
      text,
    )
    text = sub(
      re(
        String.raw`(?<![\d.,/:-])(\d{1,2})(?:st|nd|rd|th)?\s?[-—]\s?(\d{1,2})(?:st|nd|rd|th)?\s+(` +
          MONTH_PATTERN +
          String.raw`)\b(?:,?\s+(\d{4}))?`,
      ),
      (m) => this.daysOfMonth(m),
      text,
    )
    text = sub(
      re(String.raw`\b(` + MONTH_PATTERN + String.raw`)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?(?![\d:])`),
      (m) =>
        this.date(MONTH_NUMBER.get(m.group(1)!.toLowerCase().slice(0, 3))!, int(m.group(2)!), m.group(3)) ??
        m.group(0)!,
      text,
    )
    text = sub(
      re(
        String.raw`(?<![\d.,/:-])(\d{1,2})(?:st|nd|rd|th)?\s+(` +
          MONTH_PATTERN +
          String.raw`)\b(\.(?=\s+[a-z\d]))?(?:,?\s+(\d{4}))?`,
      ),
      (m) => this.dayMonth(m),
      text,
    )

    // telephone numbers, versions, codes
    text = sub(
      re(String.raw`(?<![\d-])(?:\+?(1)[-.\s])?\(?(\d{3})\)?[-.\s](\d{3})[-.\s](\d{4})(?![\d-])`),
      Normalizer.phone,
      text,
    )
    text = sub(re(String.raw`(?<![\w.,-])(\d{3})-(\d{4})(?![\w-])`), Normalizer.phone, text)
    text = sub(
      re(String.raw`(?<![\w.])([vV])?(\d+(?:\.\d+){2,})(?:-(\d+)(?![\w-]|\.\d))?(?![\w])`),
      (m) =>
        ' ' +
        (m.group(1) ? spelled(m.group(1)!) + ' ' : '') +
        m.group(2)!.split('.').map(versionPart).join(' point ') +
        (m.group(3) ? ' dash ' + numberWords(m.group(3)!) : '') +
        ' ',
      text,
    )
    text = sub(
      re(String.raw`(?<![\w.])([vV])(\d+)\.(\d{2,3})(?!\d|\.\d|\w)`),
      (m) => ` ${spelled(m.group(1)!)} ${cardinal(pyInt(m.group(2)!))} point ${minor(m.group(3)!)} `,
      text,
    )
    text = sub(
      re(
        String.raw`\b(` +
          VERSIONED +
          String.raw`)(\.?\s(?:(?:from|to)\s)?)(\d+)\.(\d{2,3})(?:(\s(?:to|or|and)\s)(\d+)\.(\d{2,3}))?(?!\d|\.\d|\s?%|\s(?:percent|million|billion)\b)`,
        'i',
      ),
      (m) =>
        `${m.group(1)}${m.group(2)}${cardinal(pyInt(m.group(3)!))} point ${minor(m.group(4)!)}` +
        (m.group(5) ? `${m.group(5)}${cardinal(pyInt(m.group(6)!))} point ${minor(m.group(7)!)}` : '') +
        ' ',
      text,
    )
    text = sub(CODE, (m) => this.code(m), text)
    text = sub(SLASHED, (m) => ' ' + Normalizer.digitGroups(m.group(1)!.split('/')) + ' ', text)
    text = sub(GROUPED, (m) => this.grouped(m), text)

    text = sub(re(String.raw`(?<![\w.])-(?=\d)`), ' minus ', text)
    text = sub(re(String.raw`(?:(?<=\d)\s?[xX×]|×)\s?(?=\d[\d.,]*\s?(?:` + LENGTHS + String.raw`)\b)`), ' by ', text)
    text = sub(re(String.raw`(?<=\d)\s[xX×]\s(?=\d)`), ' times ', text)
    text = sub(re(String.raw`(?<![\w/'])[A-Za-z0-9/]*[0-9/][A-Za-z0-9/]*(?![\w/'])`), (m) => this.chatForm(m), text)
    text = sub(
      re(String.raw`(?<![\w.])'?(\d{2}|\d{4})'?s\b`),
      (m) => ' ' + pluralOf(m.group(1)!.length === 4 ? year(int(m.group(1)!)) : cardinal(pyInt(m.group(1)!))) + ' ',
      text,
    )
    text = sub(re(String.raw`(?<![\w.])(\d+)(?:st|nd|rd|th)\b`), (m) => ` ${ordinal(pyInt(m.group(1)!))} `, text)
    text = sub(
      re(String.raw`(?<![\w.])(144|240|360|480|540|576|720|900|1080|1440|2160|4320)([pi])\b`),
      (m) => ` ${pairs(int(m.group(1)!))} ${spelled(m.group(2)!)} `,
      text,
    )
    text = sub(re(String.raw`(\d[\d,]*(?:\.\d+)?)\s?%`), (m) => ` ${numberWords(m.group(1)!)} percent `, text)
    text = sub(this.areaPlain, (m) => this.square(m), text)
    text = sub(this.unit, (m) => this.measure(m), text)
    text = sub(
      re(String.raw`(?<![\d/.,-])(\d+) (\d)/(2|3|4|5|6|8|10|16)(?![\d/]|\.\d)`),
      (m) => this.mixedFraction(m),
      text,
    )
    text = sub(re(String.raw`(?<![\d/.])(\d+)/(\d+)(?![\d/]|\.\d)`), (m) => this.fraction(m), text)
    text = sub(
      re(String.raw`(?<![\w.,-])(\d{4})[-–](\d{4})(?![\w.,-])`),
      (m) => ` ${year(int(m.group(1)!))} to ${year(int(m.group(2)!))} `,
      text,
    )
    text = sub(
      re(String.raw`(?<![\w.,-])(\d+)[-–](\d+)(?![\w-]|[.,]\d)`),
      (m) => ` ${numberWords(m.group(1)!)} to ${numberWords(m.group(2)!)} `,
      text,
    )
    text = sub(re(String.raw`(?<![\w.])(\d+)x\b`), (m) => ` ${numberWords(m.group(1)!)} times `, text)
    text = sub(re(String.raw`(?<=\d)\^(?=\d)`), ' to the power of ', text)
    return text
  }

  private static namedLetters(text: string): string {
    if (![...'=<>+*/^()≤≥≠≈×÷±'].some((sign) => text.includes(sign))) return text
    const plain = withoutAngles(text)
    const places = new Set<number>()
    for (const pattern of LETTER_PLACES) for (const m of finditer(pattern, plain)) places.add(m.start('letter'))
    if (!places.size) return text
    let out = ''
    for (let index = 0; index < text.length; index++)
      out += places.has(index) ? `${SPELL}${text[index]!.toUpperCase()}${END}` : text[index]!
    return out
  }

  private static markComparisons(text: string): string {
    const plain = withoutAngles(text)
    const mark = (m: Match): string => {
      const beforeText = plain.slice(0, m.start())
      const after = plain.slice(m.end())
      const value = search(VALUE_BEFORE, beforeText)
      if (plain[m.start('sign')] !== m.group('sign')![0] || value === null || !reMatch(VALUE_AFTER, after))
        return m.group(0)!
      let start = value.start()
      if (')]'.includes(value.group(0)!) && value.group(0)!.length === 1) {
        let depth = 0
        while (start >= 0) {
          depth += (among(beforeText[start], ')]') ? 1 : 0) - (among(beforeText[start], '([') ? 1 : 0)
          if (depth === 0) break
          start -= 1
        }
        while (start > 0 && (isalnum(beforeText[start - 1]!) || '_.'.includes(beforeText[start - 1]!))) start -= 1
      }
      const w = search(re(String.raw`[A-Za-z']+$`), beforeText.slice(0, Math.max(start, 0)).trimEnd())
      if (w && !CLAUSE_OPENERS.has(w.group(0)!.toLowerCase())) return m.group(0)!
      return IS + m.group(0)!
    }
    return sub(COMPARED, mark, text)
  }

  private static compared(m: Match, words: string): string {
    const marked = m.group(0)!.startsWith(IS) || rstrip(m.string.slice(0, m.start()), ' ').endsWith(IS)
    return ` ${marked ? 'is ' : ''}${words} `
  }

  private joinedComparisons(text: string): string {
    const plain = withoutAngles(text)
    const read = (m: Match): string => {
      const beforeText = rstrip(text.slice(0, m.start()), IS)
      const after = text.slice(m.end())
      if (plain[m.start()] !== m.group(0)) return m.group(0)!
      if (!(
        search(FIGURE_BEFORE, beforeText) ||
        reMatch(FIGURE_AFTER, after) ||
        (search(LETTER_BEFORE, beforeText) && reMatch(LETTER_AFTER, after))
      ))
        return m.group(0)!
      return Normalizer.compared(m, m.group(0) === '<' ? 'less than' : 'greater than')
    }
    return sub(JOINED_COMPARISON, read, text)
  }

  private signs(textIn: string): string {
    let text = Normalizer.namedLetters(textIn)
    if (text.includes('<') || text.includes('>')) text = this.joinedComparisons(text)
    text = sub(
      re(String.raw`(?<![\w-])(` + SIZES + ')-(' + SIZES + String.raw`)(?![\w-])`),
      (m) => ` ${spelled(m.group(1)!)} to ${spelled(m.group(2)!)} `,
      text,
    )
    text = sub(re(String.raw`#(?=\d)`), ' number ', text)
    text = sub(re(String.raw`#(?=[A-Z]{1,4}-?\d)`), ' number ', text)
    text = sub(re(String.raw`#(?=[A-Za-z])`), ' hashtag ', text)
    text = sub(re(String.raw`~(?=\s?\d)`), ' about ', text)
    text = sub(
      re(String.raw`(?<=[\w\s` + END + IS + String.raw`])(>=|<=|!=)(?=[\w\s` + SPELL + '])'),
      (m) => Normalizer.compared(m, COMPARISONS[m.group(1)!]!),
      text,
    )
    text = sub(re(String.raw`(?<=\d) Ti\b`), ` ${spelled('TI')}`, text)
    text = sub(re(String.raw`\b([Rr])ef\.\s?[Nn]o\.(?=\s)`), '\\1eference number', text)
    if (text.includes('.')) {
      const current = text
      text = sub(re(String.raw`\b((?:[A-Za-z]\.){2,})`), (m) => Normalizer.dotted(m, current), current)
    }
    {
      const current = text
      text = sub(re(String.raw`\b([A-Za-z]{1,6})\b(\.?)(?![A-Za-z'’])`), (m) => this.abbreviation(m, current), current)
    }
    if (search(re('\\d'), text)) {
      text = sub(GLUED, (m) => this.glued(m), text)
      const current = text
      text = sub(
        re(String.raw`(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|(?<![\w.])\.\d+`),
        (m) => this.number(m, current),
        current,
      )
    }
    text = sub(re('(?<=[A-Za-z])/(?=[A-Za-z])'), ' ', text)
    text = sub(
      re(String.raw`(?<=\d)` + IS + String.raw`?\s?[<>]\s?(?=\d)|(?<=\s)[<>](?=\s)`),
      (m) => Normalizer.compared(m, m.group(0)!.includes('<') ? 'less than' : 'greater than'),
      text,
    )
    const symbols = Object.keys(SYMBOL_WORDS)
    text = sub(
      new RegExp('[' + symbols.map((s) => s.replace(/[\\\]^[-]/g, '\\$&')).join('') + ']', 'gud'),
      (m) =>
        '≥≤≠'.includes(m.group(0)!)
          ? Normalizer.compared(m, SYMBOL_WORDS[m.group(0)!]!)
          : ` ${SYMBOL_WORDS[m.group(0)!]} `,
      text,
    )
    return text.replaceAll(IS, '')
  }

  // -- words --

  private static unstretched(w: string): string[] {
    if (!search(re(String.raw`(.)\1\1`), w)) return []
    const cut = (doubleInside: boolean, doubleFinal: boolean): string =>
      sub(
        re(String.raw`(.)\1{2,}`),
        (m) => m.group(1)!.repeat((m.end() === w.length ? doubleFinal : doubleInside) ? 2 : 1),
        w,
      )
    const liked =
      !!search(re(String.raw`([lfsz])\1\1+$`), w) ||
      (new Set(w).size === 1 && w[0] === 'm') ||
      sub(re(String.raw`(.)\1{2,}$`), '\\1', w) === 'to'
    const ordered = [cut(true, liked), cut(true, !liked), cut(false, liked), cut(false, !liked)]
    return [...new Set(ordered)]
  }

  private static chatWords(chat: string, raw: string): Word[] {
    if (chat === '-') return []
    return split(chat).map((part) =>
      part.startsWith('*')
        ? word(part.slice(1).toLowerCase(), { raw, spell: true })
        : word(part.toLowerCase(), { raw: part }),
    )
  }

  private static reply(out: Token[], place: number): boolean {
    const marks: string[] = []
    for (const step of [-1, 1]) {
      let index = place + step
      while (
        index >= 0 &&
        index < out.length &&
        out[index]!.kind === 'punct' &&
        '"\'()'.includes((out[index] as Punct).mark)
      )
        index += step
      if (!(index >= 0 && index < out.length)) {
        marks.push('')
      } else if (out[index]!.kind === 'word') {
        return false
      } else {
        const mark = (out[index] as Punct).mark
        marks.push(mark)
        const beyond = index + step >= 0 && index + step < out.length ? out[index + step]! : null
        if (mark === ',' && beyond?.kind === 'word' && [...beyond.text].length === 1 && beyond.text !== 'i')
          return false
      }
    }
    return !(marks.length === 2 && marks[0] === ',' && marks[1] === ',')
  }

  private wordTokens(raw: string, allCaps: boolean, previous: string, following = true, fresh = false): Word[] {
    const low = raw.toLowerCase()
    const chat = this.chat.get(low)
    if (chat !== undefined && ([...raw].length > 1 || islower(raw)) && !this.replies.has(low))
      return Normalizer.chatWords(chat, raw)
    if (Object.hasOwn(ROMAN, raw) && previous && isupper(previous[0]!)) {
      const number = ROMAN[raw]!
      const words = CARDINAL_AFTER.has(previous.toLowerCase()) ? cardinal(number) : 'the ' + ordinal(number)
      return words.split(' ').map((w) => word(w, { raw }))
    }
    const letters = raw.replaceAll("'", '')
    const possessive = low.endsWith("'s")
    const stem = possessive ? raw.slice(0, -2) : raw
    if ([...stem].length === 1 && isalpha(stem)) {
      const label = stem === 'A' && !!previous && (!following || (!fresh && LABELLED.has(previous.toLowerCase())))
      if (['a', 'A', 'I', 'i'].includes(stem) && !possessive && !label) return [word(stem.toLowerCase(), { raw })]
      return [word(low, { raw, spell: true })]
    }
    if (stem.length > 2 && 'SsZz'.includes(stem[0]!) && new Set(low).size === 1 && (allCaps || !isupper(stem)))
      return [word(low, { raw, phones: [[low[0]!, 0]] })]
    if (
      isupper(stem) &&
      isalpha(stem) &&
      stem.length >= 2 &&
      stem.length <= 5 &&
      (!allCaps || !this.isWord(stem.toLowerCase()))
    ) {
      if (INITIALISMS.has(stem) || !this.isWord(stem.toLowerCase())) return [word(low, { raw, spell: true })]
    }
    if (!isalpha(letters)) return []
    const capitals = fullmatch(re(String.raw`[a-z]+(?:[A-Z][a-z]+)*[A-Z]{2,}(?:'s)?`), raw)
    if (capitals || !this.isWord(low)) {
      for (const candidate of capitals ? [] : Normalizer.unstretched(low)) {
        if (this.isWord(candidate)) return [word(candidate, { raw })]
      }
      const parts = findall(re(String.raw`[A-Z]?[a-z']+|[A-Z]+(?![a-z])`), raw)
      if (parts.length > 1 && parts.join('') === raw) {
        const out: Word[] = []
        for (const part of parts) {
          const named = capitals && isupper(part) ? INITIALISMS.has(part) : false
          const isSpelled = part.length === 1 || named || (isupper(part) && !this.isWord(part.toLowerCase()))
          if (capitals && out.length && out[out.length - 1]!.spell && (isSpelled || part === "'s")) {
            const last = out[out.length - 1]!
            out[out.length - 1] = { ...last, text: last.text + part.toLowerCase(), raw: last.raw + part }
          } else {
            out.push(word(part.toLowerCase(), { attach: out.length > 0, raw: part, spell: isSpelled }))
          }
        }
        return out
      }
    }
    return [word(low, { raw })]
  }

  /** Words and punctuation marks of `text`. */
  tokens(textIn: string): Token[] {
    const text = this.rewrite(textIn)
    const runs = findall(re('[A-Za-z]{2,}'), text)
    const allCaps = runs.length >= 3 && runs.filter((run) => isupper(run)).length >= 0.6 * runs.length
    const out: Token[] = []
    const pattern = re(
      SPELL +
        '([A-Za-z ]+)' +
        END +
        '|' +
        INLINE +
        '(.)' +
        INLINE +
        String.raw`|('?[A-Za-z]+(?:'[A-Za-z]+)*)|(\d+)|(-)|(\S)`,
    )
    let attach = false
    let previous = ''
    let fresh = true
    const short: number[] = []
    const replies: number[] = []
    const START_LETTER = re(String.raw`\s*[A-Za-z]`)
    const STANDS_APART = re(String.raw`[.!?…,;:—"')]*(?:\s|$)`)
    const REPLY_END = re(String.raw`(?:[.?…,;—]*["']?(?:\s|$)|!+["']?\s*$)`)
    for (const m of finditer(pattern, text)) {
      const [isSpelled, inline, found, number, hyphen, other] = m.groups()
      const beforeChar = m.start() ? text[m.start() - 1]! : ' '
      if (hyphen) {
        attach =
          out.length > 0 &&
          out[out.length - 1]!.kind === 'word' &&
          !isspace(beforeChar) &&
          m.end() < text.length &&
          isalnum(text[m.end()]!)
        continue
      }
      let fresh2: Word[] = []
      if (isSpelled !== undefined) {
        fresh2 = [word(isSpelled.replaceAll(' ', '').toLowerCase(), { raw: isSpelled, spell: true })]
      } else if (inline !== undefined) {
        const [written, phones] = this.inline[inline.codePointAt(0)! - INLINE_BASE]!
        fresh2 = [word(written.toLowerCase(), { raw: written, phones })]
      } else if (found !== undefined) {
        let w = found
        if (w.startsWith("'") && !LEADING_APOSTROPHE.has(w.toLowerCase())) {
          out.push({ kind: 'punct', mark: "'" })
          w = w.slice(1)
        } else if (w.startsWith("'") && !this.isWord(w.toLowerCase())) {
          w = w.slice(1)
        }
        fresh2 = this.wordTokens(w, allCaps, previous, reMatch(START_LETTER, text.slice(m.end())) !== null, fresh)
        previous = w
        if (
          ['Ss', 'ss', 'Zz', 'zz'].includes(w) &&
          fresh2.length === 1 &&
          (isspace(beforeChar) || '"\'('.includes(beforeChar)) &&
          reMatch(STANDS_APART, text.slice(m.end()))
        )
          short.push(out.length)
        if (
          this.replies.has(w.toLowerCase()) &&
          ([...w].length > 1 || islower(w)) &&
          fresh2.length === 1 &&
          (isspace(beforeChar) || '"\''.includes(beforeChar)) &&
          reMatch(REPLY_END, text.slice(m.end()))
        )
          replies.push(out.length)
      } else if (number !== undefined) {
        fresh2 = split(numberWords(number)).map((part) => word(part, { raw: number }))
      } else if (other !== undefined && PUNCTUATION.includes(other)) {
        if (!(other === "'" && isalnum(beforeChar) && m.end() < text.length && isalnum(text[m.end()]!)))
          out.push({ kind: 'punct', mark: other })
        if ('.!?;…'.includes(other)) fresh = true
        attach = false
        continue
      }
      if (fresh2.length && attach) fresh2[0] = { ...fresh2[0]!, attach: true }
      out.push(...fresh2)
      attach = false
      fresh = fresh && !fresh2.length
    }
    for (const place of short) {
      let first = place
      let last = place + 1
      const quiet = (token: Token): boolean => token.kind === 'word' || '"\'()'.includes(token.mark)
      while (first && quiet(out[first - 1]!)) first -= 1
      while (last < out.length && quiet(out[last]!)) last += 1
      const words = out.slice(first, last).filter((token): token is Word => token.kind === 'word')
      if (words.every((token) => SOUNDS.has(token.text) || (token.phones !== null && new Set(token.text).size === 1))) {
        const target = out[place] as Word
        out[place] = { ...target, phones: [[target.text[0]!, 0]] }
      }
    }
    for (const place of [...replies].reverse()) {
      if (Normalizer.reply(out, place)) {
        const raw = (out[place] as Word).raw
        out.splice(place, 1, ...Normalizer.chatWords(this.chat.get(raw.toLowerCase())!, raw))
      }
    }
    return out
  }
}
