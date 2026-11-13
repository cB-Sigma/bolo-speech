/**
 * Numbers to words for the in-browser voices. The voices read characters, not
 * digits (most have no digit symbols at all, and the ones that do were not
 * trained to read them), so numbers are spelt out before synthesis.
 *
 * Written for this project from the standard number names of each language.
 * Maithili and Bhojpuri use the Hindi names: close, but not how every speaker
 * would say them. Nobody who speaks these languages has reviewed the output.
 */

export type NumberLanguage = 'en' | 'hi' | 'mr' | 'kn'

const EN_ONES =
  'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(
    ' ',
  )
const EN_TENS = ' ten twenty thirty forty fifty sixty seventy eighty ninety'.split(' ')

function en(n: number): string {
  if (n < 20) return EN_ONES[n] as string
  if (n < 100) return `${EN_TENS[Math.floor(n / 10)]}${n % 10 ? ` ${EN_ONES[n % 10]}` : ''}`
  if (n < 1000) return `${EN_ONES[Math.floor(n / 100)]} hundred${n % 100 ? ` and ${en(n % 100)}` : ''}`
  for (const [size, name] of [
    [1e9, 'billion'],
    [1e6, 'million'],
    [1e3, 'thousand'],
  ] as const) {
    if (n >= size) {
      const rest = n % size
      return `${en(Math.floor(n / size))} ${name}${rest ? (rest < 100 ? ` and ${en(rest)}` : ` ${en(rest)}`) : ''}`
    }
  }
  return String(n)
}

// 0..99. Hindi number names are not regular, so the whole table is written out.
const HI = [
  'शून्य', 'एक', 'दो', 'तीन', 'चार', 'पाँच', 'छह', 'सात', 'आठ', 'नौ',
  'दस', 'ग्यारह', 'बारह', 'तेरह', 'चौदह', 'पंद्रह', 'सोलह', 'सत्रह', 'अठारह', 'उन्नीस',
  'बीस', 'इक्कीस', 'बाईस', 'तेईस', 'चौबीस', 'पच्चीस', 'छब्बीस', 'सत्ताईस', 'अट्ठाईस', 'उनतीस',
  'तीस', 'इकतीस', 'बत्तीस', 'तैंतीस', 'चौंतीस', 'पैंतीस', 'छत्तीस', 'सैंतीस', 'अड़तीस', 'उनतालीस',
  'चालीस', 'इकतालीस', 'बयालीस', 'तैंतालीस', 'चवालीस', 'पैंतालीस', 'छियालीस', 'सैंतालीस', 'अड़तालीस', 'उनचास',
  'पचास', 'इक्यावन', 'बावन', 'तिरपन', 'चौवन', 'पचपन', 'छप्पन', 'सत्तावन', 'अट्ठावन', 'उनसठ',
  'साठ', 'इकसठ', 'बासठ', 'तिरसठ', 'चौंसठ', 'पैंसठ', 'छियासठ', 'सड़सठ', 'अड़सठ', 'उनहत्तर',
  'सत्तर', 'इकहत्तर', 'बहत्तर', 'तिहत्तर', 'चौहत्तर', 'पचहत्तर', 'छिहत्तर', 'सतहत्तर', 'अठहत्तर', 'उनासी',
  'अस्सी', 'इक्यासी', 'बयासी', 'तिरासी', 'चौरासी', 'पचासी', 'छियासी', 'सत्तासी', 'अट्ठासी', 'नवासी',
  'नब्बे', 'इक्यानबे', 'बानबे', 'तिरानबे', 'चौरानबे', 'पंचानबे', 'छियानबे', 'सत्तानबे', 'अट्ठानबे', 'निन्यानबे',
] // prettier-ignore

function hi(n: number): string {
  if (n < 100) return HI[n] as string
  const parts: string[] = []
  for (const [size, name] of [
    [1e7, 'करोड़'],
    [1e5, 'लाख'],
    [1e3, 'हज़ार'],
    [100, 'सौ'],
  ] as const) {
    if (n >= size) {
      parts.push(`${hi(Math.floor(n / size))} ${name}`)
      n %= size
    }
  }
  if (n) parts.push(HI[n] as string)
  return parts.join(' ')
}

const MR = [
  'शून्य', 'एक', 'दोन', 'तीन', 'चार', 'पाच', 'सहा', 'सात', 'आठ', 'नऊ',
  'दहा', 'अकरा', 'बारा', 'तेरा', 'चौदा', 'पंधरा', 'सोळा', 'सतरा', 'अठरा', 'एकोणीस',
  'वीस', 'एकवीस', 'बावीस', 'तेवीस', 'चोवीस', 'पंचवीस', 'सव्वीस', 'सत्तावीस', 'अठ्ठावीस', 'एकोणतीस',
  'तीस', 'एकतीस', 'बत्तीस', 'तेहेतीस', 'चौतीस', 'पस्तीस', 'छत्तीस', 'सदतीस', 'अडतीस', 'एकोणचाळीस',
  'चाळीस', 'एक्केचाळीस', 'बेचाळीस', 'त्रेचाळीस', 'चव्वेचाळीस', 'पंचेचाळीस', 'सेहेचाळीस', 'सत्तेचाळीस', 'अठ्ठेचाळीस', 'एकोणपन्नास',
  'पन्नास', 'एक्कावन्न', 'बावन्न', 'त्रेपन्न', 'चोपन्न', 'पंचावन्न', 'छप्पन्न', 'सत्तावन्न', 'अठ्ठावन्न', 'एकोणसाठ',
  'साठ', 'एकसष्ट', 'बासष्ट', 'त्रेसष्ट', 'चौसष्ट', 'पासष्ट', 'सहासष्ट', 'सदुसष्ट', 'अडुसष्ट', 'एकोणसत्तर',
  'सत्तर', 'एक्काहत्तर', 'बाहत्तर', 'त्र्याहत्तर', 'चौऱ्याहत्तर', 'पंच्याहत्तर', 'शहात्तर', 'सत्याहत्तर', 'अठ्ठ्याहत्तर', 'एकोणऐंशी',
  'ऐंशी', 'एक्क्याऐंशी', 'ब्याऐंशी', 'त्र्याऐंशी', 'चौऱ्याऐंशी', 'पंच्याऐंशी', 'शहाऐंशी', 'सत्त्याऐंशी', 'अठ्ठ्याऐंशी', 'एकोणनव्वद',
  'नव्वद', 'एक्क्याण्णव', 'ब्याण्णव', 'त्र्याण्णव', 'चौऱ्याण्णव', 'पंच्याण्णव', 'शहाण्णव', 'सत्त्याण्णव', 'अठ्ठ्याण्णव', 'नव्व्याण्णव',
] // prettier-ignore

function mr(n: number): string {
  if (n < 100) return MR[n] as string
  const parts: string[] = []
  for (const [size, name] of [
    [1e7, 'कोटी'],
    [1e5, 'लाख'],
    [1e3, 'हजार'],
  ] as const) {
    if (n >= size) {
      parts.push(`${mr(Math.floor(n / size))} ${name}`)
      n %= size
    }
  }
  if (n >= 100) {
    const h = Math.floor(n / 100)
    // 100 on its own is शंभर; otherwise "<n>शे" (एकशे पाच, दोनशे, पाचशे).
    parts.push(h === 1 && n % 100 === 0 ? 'शंभर' : `${MR[h]}शे`)
    n %= 100
  }
  if (n) parts.push(MR[n] as string)
  return parts.join(' ')
}

const KN_ONES = ['ಸೊನ್ನೆ', 'ಒಂದು', 'ಎರಡು', 'ಮೂರು', 'ನಾಲ್ಕು', 'ಐದು', 'ಆರು', 'ಏಳು', 'ಎಂಟು', 'ಒಂಬತ್ತು']
const KN_TEENS = [
  'ಹತ್ತು', 'ಹನ್ನೊಂದು', 'ಹನ್ನೆರಡು', 'ಹದಿಮೂರು', 'ಹದಿನಾಲ್ಕು', 'ಹದಿನೈದು', 'ಹದಿನಾರು', 'ಹದಿನೇಳು', 'ಹದಿನೆಂಟು', 'ಹತ್ತೊಂಬತ್ತು',
] // prettier-ignore
const KN_TENS = ['', '', 'ಇಪ್ಪತ್ತು', 'ಮೂವತ್ತು', 'ನಲವತ್ತು', 'ಐವತ್ತು', 'ಅರವತ್ತು', 'ಎಪ್ಪತ್ತು', 'ಎಂಬತ್ತು', 'ತೊಂಬತ್ತು']
const KN_HUNDREDS = ['', 'ನೂರು', 'ಇನ್ನೂರು', 'ಮುನ್ನೂರು', 'ನಾನೂರು', 'ಐನೂರು', 'ಆರುನೂರು', 'ಏಳುನೂರು', 'ಎಂಟುನೂರು', 'ಒಂಬೈನೂರು']
/** Independent vowel → the vowel sign that replaces the inherent "a" when two words join (ಇಪ್ಪತ್ತ + ಒಂದು → ಇಪ್ಪತ್ತೊಂದು). */
const KN_SIGN: Record<string, string> = { ಅ: '', ಆ: 'ಾ', ಇ: 'ಿ', ಈ: 'ೀ', ಉ: 'ು', ಎ: 'ೆ', ಏ: 'ೇ', ಐ: 'ೈ', ಒ: 'ೊ', ಓ: 'ೋ' }
const KN_U = 'ು'

/** Join a word ending in "-ು" to the next: vowel-initial words merge, others follow the stem's inherent "a". */
function knJoin(left: string, right: string): string {
  const stem = left.endsWith(KN_U) ? left.slice(0, -1) : left
  const sign = KN_SIGN[right[0] as string]
  return sign != null ? stem + sign + right.slice(1) : stem + right
}

function kn(n: number): string {
  if (n < 10) return KN_ONES[n] as string
  if (n < 20) return KN_TEENS[n - 10] as string
  if (n < 100)
    return n % 10
      ? knJoin(KN_TENS[Math.floor(n / 10)] as string, KN_ONES[n % 10] as string)
      : (KN_TENS[n / 10] as string)
  if (n < 1000) {
    const h = KN_HUNDREDS[Math.floor(n / 100)] as string
    const rest = n % 100
    return rest ? `${h.slice(0, -1)} ${kn(rest)}` : h // "ನೂರ ಇಪ್ಪತ್ತು", as it is said
  }
  for (const [size, name, joined] of [
    [1e7, 'ಕೋಟಿ', 'ಕೋಟಿ'],
    [1e5, 'ಲಕ್ಷ', 'ಲಕ್ಷದ'],
    [1e3, 'ಸಾವಿರ', 'ಸಾವಿರದ'],
  ] as const) {
    if (n >= size) {
      const count = Math.floor(n / size)
      const rest = n % size
      const head = count === 1 && size === 1e3 ? '' : `${kn(count)} `
      return `${head}${rest ? joined : name}${rest ? ` ${kn(rest)}` : ''}`
    }
  }
  return String(n)
}

const WORDS: Record<NumberLanguage, (n: number) => string> = { en, hi, mr, kn }
const DIGITS: Record<NumberLanguage, string[]> = {
  en: EN_ONES.slice(0, 10),
  hi: HI.slice(0, 10),
  mr: MR.slice(0, 10),
  kn: KN_ONES,
}
const POINT: Record<NumberLanguage, string> = { en: 'point', hi: 'दशमलव', mr: 'पूर्णांक', kn: 'ಪಾಯಿಂಟ್' }
const PERCENT: Record<NumberLanguage, string> = { en: 'percent', hi: 'प्रतिशत', mr: 'टक्के', kn: 'ಶೇಕಡಾ' }

/** Spell out one integer (0 ≤ n < 10^12); larger numbers are read digit by digit. */
export function numberToWords(n: number, lang: NumberLanguage): string {
  if (!Number.isInteger(n) || n < 0) throw new Error(`not a whole number: ${n}`)
  if (n >= 1e12) return [...String(n)].map((d) => DIGITS[lang][Number(d)]).join(' ')
  return WORDS[lang](n)
}

/** Native digits (Devanagari, Kannada) → ASCII. */
export function asciiDigits(text: string): string {
  return text.replace(/[०-९೦-೯]/g, (d) => {
    const c = d.charCodeAt(0)
    return String(c >= 0x0ce6 ? c - 0x0ce6 : c - 0x0966)
  })
}

/**
 * Replace every number in the text with words: whole numbers, Indian or
 * Western digit grouping, decimals, and a trailing per cent sign. Numbers that
 * start with 0 (phone numbers, codes) and long digit strings are read digit by
 * digit.
 */
export function expandNumbers(text: string, lang: NumberLanguage): string {
  const digitWise = (s: string) => [...s].map((d) => DIGITS[lang][Number(d)]).join(' ')
  return asciiDigits(text).replace(
    /(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d+))?(\s?%)?/g,
    (_m, whole: string, frac?: string, pct?: string) => {
      const plain = whole.replace(/,/g, '')
      let words =
        (plain.length > 1 && plain.startsWith('0')) || plain.length > 12
          ? digitWise(plain)
          : numberToWords(Number(plain), lang)
      if (frac) words += ` ${POINT[lang]} ${digitWise(frac)}`
      if (pct) words = lang === 'kn' ? `${PERCENT.kn} ${words}` : `${words} ${PERCENT[lang]}`
      return words
    },
  )
}
