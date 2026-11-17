/**
 * The language of a text to be spoken, guessed from its script and, for Latin script, its common words: the
 * server's bolo.text.langdetect in TypeScript, for the voices that run in the page. It must give the same answers
 * as the server; src/lib/langdetect.test.ts checks it against the server's own test cases and a corpus run through
 * the Python code (scripts/parity/export_cases.py).
 *
 * No dependency and no model: the script of the letters settles most languages (Devanagari, Tamil, Bengali,
 * Gurmukhi, Hangul, Thai, ...). Where several languages share a script, a few frequent words or letters choose
 * among them. Latin-script text is scored against short lists of the most frequent words of twenty languages; a
 * short or ambiguous text is reported as unsure (`language` null) rather than guessed.
 */

/** [first code point, last code point, script] */
const RANGES: [number, number, string][] = [
  [0x0041, 0x005a, 'Latin'],
  [0x0061, 0x007a, 'Latin'],
  [0x00c0, 0x024f, 'Latin'],
  [0x1e00, 0x1eff, 'Latin'],
  [0x0370, 0x03ff, 'Greek'],
  [0x1f00, 0x1fff, 'Greek'],
  [0x0400, 0x052f, 'Cyrillic'],
  [0x0530, 0x058f, 'Armenian'],
  [0x0590, 0x05ff, 'Hebrew'],
  [0x0600, 0x06ff, 'Arabic'],
  [0x0750, 0x077f, 'Arabic'],
  [0xfb50, 0xfdff, 'Arabic'],
  [0xfe70, 0xfeff, 'Arabic'],
  [0x0900, 0x097f, 'Devanagari'],
  [0xa8e0, 0xa8ff, 'Devanagari'],
  [0x0980, 0x09ff, 'Bengali'],
  [0x0a00, 0x0a7f, 'Gurmukhi'],
  [0x0a80, 0x0aff, 'Gujarati'],
  [0x0b00, 0x0b7f, 'Oriya'],
  [0x0b80, 0x0bff, 'Tamil'],
  [0x0c00, 0x0c7f, 'Telugu'],
  [0x0c80, 0x0cff, 'Kannada'],
  [0x0d00, 0x0d7f, 'Malayalam'],
  [0x0d80, 0x0dff, 'Sinhala'],
  [0x0e00, 0x0e7f, 'Thai'],
  [0x0e80, 0x0eff, 'Lao'],
  [0x0f00, 0x0fff, 'Tibetan'],
  [0x1000, 0x109f, 'Myanmar'],
  [0x10a0, 0x10ff, 'Georgian'],
  [0x1100, 0x11ff, 'Hangul'],
  [0x3130, 0x318f, 'Hangul'],
  [0xac00, 0xd7af, 'Hangul'],
  [0x1200, 0x139f, 'Ethiopic'],
  [0x1780, 0x17ff, 'Khmer'],
  [0x1c50, 0x1c7f, 'Ol Chiki'],
  [0xabc0, 0xabff, 'Meetei Mayek'],
  [0x3040, 0x309f, 'Kana'],
  [0x30a0, 0x30ff, 'Kana'],
  [0x31f0, 0x31ff, 'Kana'],
  [0x3400, 0x4dbf, 'Han'],
  [0x4e00, 0x9fff, 'Han'],
  [0xf900, 0xfaff, 'Han'],
]

/** The languages written in a script, most widely spoken first; the first is the answer when nothing narrows it. */
export const SCRIPT_LANGUAGES: Record<string, string[]> = {
  Devanagari: ['hi', 'mr', 'ne', 'mai', 'bho', 'sa', 'kok', 'doi', 'brx', 'mag', 'hne'],
  Bengali: ['bn', 'as', 'mni'],
  Gurmukhi: ['pa'],
  Gujarati: ['gu'],
  Oriya: ['or'],
  Tamil: ['ta'],
  Telugu: ['te'],
  Kannada: ['kn'],
  Malayalam: ['ml'],
  Sinhala: ['si'],
  'Ol Chiki': ['sat'],
  'Meetei Mayek': ['mni'],
  Arabic: ['ar', 'ur', 'fa', 'ps', 'sd', 'ks'],
  Hebrew: ['he', 'yi'],
  Greek: ['el'],
  Cyrillic: ['ru', 'uk', 'bg', 'sr', 'be', 'mk', 'kk'],
  Armenian: ['hy'],
  Georgian: ['ka'],
  Thai: ['th'],
  Lao: ['lo'],
  Khmer: ['km'],
  Myanmar: ['my'],
  Tibetan: ['bo'],
  Ethiopic: ['am', 'ti'],
  Hangul: ['ko'],
  Kana: ['ja'],
  Han: ['zh', 'yue', 'ja'],
}

const set = (words: string) => new Set(words.split(/\s+/).filter(Boolean))

/** The most frequent words of Latin-script languages (lower case), in the server's order (it breaks ties). */
const LATIN_WORDS: [string, Set<string>][] = [
  [
    'en',
    set(
      'the and is are was were to of in that it you he she we they this with for not have has be on at what there will would can do does my your from but',
    ),
  ],
  [
    'es',
    set(
      'el la los las de que y en un una es por con para no se lo como más pero sus le ya o fue este está muy también hay yo',
    ),
  ],
  [
    'fr',
    set(
      'le la les de des et est un une que qui en du dans pour pas sur ce il elle nous vous sont avec au aux mais ne je',
    ),
  ],
  [
    'de',
    set(
      'der die das und ist nicht ein eine zu den mit von sich des auf für im dem auch es ich sie wir aber wie sind war',
    ),
  ],
  ['it', set('il lo la gli le di che e è un una per non sono del della con si ma come anche ho questo nel alla mi io')],
  [
    'pt',
    set('o a os as de que e do da em um uma para com não é se na no por mais dos das como mas foi ao ele ela você'),
  ],
  ['nl', set('de het een en van is dat niet ik je op te zijn met voor er maar die wat hij ze ook als bij naar heb')],
  ['sv', set('och att det är en som på för av med jag inte till har den de om ett var men så vi han hon kan')],
  ['da', set('og at det er en som på for af med jeg ikke til har den de om et var men så vi han hun kan skal')],
  ['no', set('og at det er en som på for av med jeg ikke til har den de om et var men så vi han hun kan skal ikkje')],
  ['fi', set('ja on ei se että hän oli ovat kuin mutta tai minä sinä me te he tämä joka mitä myös kanssa olen')],
  ['pl', set('i w nie na się z że do to jest jak co ale o po tak jego jej czy przez dla od był są')],
  ['cs', set('a v se na je že s z do to jak ale o by jsem jsou pro jeho jako už tak není bylo')],
  ['ro', set('și în de la a care cu nu un o pe este din să se mai ca pentru sunt dar sau fost')],
  ['tr', set('ve bir bu da de için ile ne çok o ben sen var değil gibi daha ama olarak en kadar mi')],
  ['hu', set('a az és hogy nem is egy van de meg ez csak már volt mint el ki vagy még kell')],
  ['id', set('yang dan di ini itu dengan untuk tidak dari dalam akan pada ada saya kami mereka juga atau bisa sudah')],
  ['vi', set('và của là có không một những được trong cho này người với các đã để tôi anh em')],
  ['ca', set('el la els les de que i a en un una és per amb no es del al com però més hi')],
  ['sw', set('na ya wa kwa ni za la katika hii kuwa yake hiyo lakini pia sana mimi wewe sisi')],
]

/** Letters (lower case) that belong to only a few Latin-script languages, with the weight they add. */
const LATIN_LETTERS: [string, [string, number][]][] = [
  ['ñ', [['es', 2]]],
  ['¿', [['es', 3]]],
  ['¡', [['es', 3]]],
  ['ã', [['pt', 2]]],
  ['õ', [['pt', 2]]],
  ['ß', [['de', 3]]],
  ['œ', [['fr', 3]]],
  [
    'ê',
    [
      ['fr', 0.5],
      ['pt', 0.5],
      ['vi', 0.5],
    ],
  ],
  [
    'è',
    [
      ['fr', 1],
      ['it', 0.5],
      ['ca', 0.5],
    ],
  ],
  ['ł', [['pl', 3]]],
  ['ą', [['pl', 3]]],
  ['ę', [['pl', 3]]],
  ['ś', [['pl', 2]]],
  ['ź', [['pl', 2]]],
  ['ż', [['pl', 2]]],
  ['ń', [['pl', 2]]],
  ['ř', [['cs', 3]]],
  ['ě', [['cs', 3]]],
  ['ů', [['cs', 3]]],
  ['ș', [['ro', 3]]],
  [
    'ş',
    [
      ['ro', 1],
      ['tr', 2],
    ],
  ],
  ['ț', [['ro', 3]]],
  ['ţ', [['ro', 2]]],
  [
    'ă',
    [
      ['ro', 2],
      ['vi', 1],
    ],
  ],
  ['ğ', [['tr', 3]]],
  ['ı', [['tr', 3]]],
  ['ő', [['hu', 3]]],
  ['ű', [['hu', 3]]],
  [
    'å',
    [
      ['sv', 1],
      ['da', 1],
      ['no', 1],
    ],
  ],
  [
    'ø',
    [
      ['da', 1.5],
      ['no', 1.5],
    ],
  ],
  [
    'æ',
    [
      ['da', 1.5],
      ['no', 1.5],
    ],
  ],
  [
    'ä',
    [
      ['de', 0.5],
      ['sv', 0.5],
      ['fi', 0.7],
    ],
  ],
  [
    'ö',
    [
      ['de', 0.5],
      ['sv', 0.5],
      ['fi', 0.5],
      ['tr', 0.5],
      ['hu', 0.5],
    ],
  ],
  ['đ', [['vi', 3]]],
  ['ơ', [['vi', 3]]],
  ['ư', [['vi', 3]]],
  ['ạ', [['vi', 3]]],
  ['ả', [['vi', 3]]],
  ['ấ', [['vi', 3]]],
  ['ộ', [['vi', 3]]],
  ['ệ', [['vi', 3]]],
  ['ế', [['vi', 3]]],
  ['ờ', [['vi', 3]]],
  ['ủ', [['vi', 3]]],
]

/** Split on single spaces, as the server's `_words_of` does. */
const wordsOf = (text: string) => new Set(text.split(' '))

/** Frequent words that tell languages of one non-Latin script apart. */
const SCRIPT_WORDS: Record<string, [string, Set<string>][]> = {
  Devanagari: [
    ['hi', wordsOf('है हैं और का की के में नहीं से को यह था थी थे हम आप मैं कि भी पर क्या')],
    ['mr', wordsOf('आहे आहेत आणि नाही हे ते मी तुम्ही आम्ही काय होते होता त्या या पण की')],
    ['ne', wordsOf('छ छन् हो र को मा गर्न भएको पनि त्यो यो म तिमी हुन्छ थियो छैन')],
  ],
  Cyrillic: [
    ['ru', wordsOf('и в не на что я с он как это по но все она так его только мне было')],
    ['uk', wordsOf('і в не на що я з він як це по але все вона так його тільки мені було')],
    ['bg', wordsOf('и в не на че аз с той как това по но всички тя така е са ще')],
  ],
}

/** Letters that only some languages of a script use. */
const SCRIPT_LETTERS: Record<string, [string, string][]> = {
  Bengali: [['ৰৱ', 'as']],
  Arabic: [
    ['ٹڈڑںےھۓ', 'ur'],
    ['پچژگی', 'fa'],
  ],
  Cyrillic: [
    ['іїєґ', 'uk'],
    ['ў', 'be'],
    ['ђјљњћџ', 'sr'],
    ['ѓќѕ', 'mk'],
    ['әғқңөұүһ', 'kk'],
  ],
  Kana: [],
}

// Letters as Python's `[^\W\d_]` sees them (letters and the numbers that are not decimal digits), with the vowel
// signs and other combining marks of the Indian scripts, which are not word characters to `re`.
const LETTER = String.raw`[\p{L}\p{Nl}\p{No}]`
const WORD_RE = new RegExp(String.raw`(?:${LETTER}|[̀-ͯऀ-෿‌‍])+(?:['’]${LETTER}+)?`, 'gu')
const TAG_RE = /<[^>]{0,200}>|\[([^\]]*)\]\(\/[^)]*\/\)/g
const ALPHA = /\p{L}/u

export interface Detection {
  /** ISO 639-1 where there is one; null when unsure. */
  language: string | null
  script: string | null
  confidence: number
  /** The other languages written in `script` (closest first), for a voice that does not speak `language`. */
  alternatives: string[]
  reason: string
  scores: Record<string, number>
}

const detection = (
  language: string | null,
  script: string | null,
  confidence: number,
  alternatives: string[] = [],
  reason = '',
  scores: Map<string, number> = new Map(),
): Detection => ({ language, script, confidence, alternatives, reason, scores: Object.fromEntries(scores) })

export function scriptOf(ch: string): string | null {
  const cp = ch.codePointAt(0) ?? 0
  for (const [lo, hi, name] of RANGES) if (lo <= cp && cp <= hi) return name
  return null
}

/** Letters per script, in order of first appearance (the server's dict order, which breaks ties). */
export function scriptCounts(text: string): Map<string, number> {
  const counts = new Map<string, number>()
  for (const ch of text) {
    if (!ALPHA.test(ch)) continue
    const s = scriptOf(ch)
    if (s != null) counts.set(s, (counts.get(s) ?? 0) + 1)
  }
  return counts
}

/** The first key with the largest value, as Python's `max(d, key=d.get)`. */
function argmax(m: Map<string, number>): string {
  let best: string | null = null
  let top = -Infinity
  for (const [k, v] of m)
    if (v > top) {
      best = k
      top = v
    }
  return best as string
}

function dominantScript(counts: Map<string, number>): string | null {
  if (counts.size === 0) return null
  let total = 0
  for (const v of counts.values()) total += v
  const kana = counts.get('Kana') ?? 0
  // Japanese mixes kana and kanji: any real share of kana makes it Japanese.
  if (kana >= Math.max(2, 0.1 * (kana + (counts.get('Han') ?? 0)))) return 'Kana'
  // Indian and other texts carry English words in Latin letters: a script of its own with a third of the
  // letters wins over the Latin ones.
  const others = new Map([...counts].filter(([s]) => s !== 'Latin'))
  if (others.size) {
    const best = argmax(others)
    if ((others.get(best) as number) >= total / 3) return best
  }
  return argmax(counts)
}

function words(text: string): string[] {
  return (text.match(WORD_RE) ?? []).map((w) => w.toLowerCase().replaceAll('’', "'"))
}

/** Occurrences of a single character (code point) in `text`. */
function count(text: string, ch: string): number {
  let n = 0
  for (let i = text.indexOf(ch); i !== -1; i = text.indexOf(ch, i + ch.length)) n++
  return n
}

/** Sorted by score, highest first; equal scores keep their order (Python's sort is stable, and so is ours). */
const ranked = (scores: Map<string, number>) => [...scores].sort((a, b) => b[1] - a[1])

function latin(text: string): Detection {
  const ws = words(text)
  if (ws.length === 0) return detection(null, 'Latin', 0, [], 'no words')
  const scores = new Map<string, number>()
  for (const w of ws)
    for (const [lang, common] of LATIN_WORDS) if (common.has(w)) scores.set(lang, (scores.get(lang) ?? 0) + 1)
  const lowered = text.toLowerCase()
  for (const [ch, weights] of LATIN_LETTERS) {
    const n = count(lowered, ch)
    if (n) for (const [lang, weight] of weights) scores.set(lang, (scores.get(lang) ?? 0) + weight * Math.min(n, 3))
  }
  const order = ranked(scores)
  if (order.length === 0) return detection(null, 'Latin', 0, [], 'no common words of a known language', scores)
  const [best, top] = order[0] as [string, number]
  const second = order.length > 1 ? (order[1] as [string, number])[1] : 0
  const confidence = top ? (top - second) / top : 0
  if (top < 2 || (ws.length < 3 && top < 3))
    return detection(null, 'Latin', confidence, [], 'too few words to tell', scores)
  if (second && top < 1.5 * second) {
    const other = (order[1] as [string, number])[0]
    return detection(null, 'Latin', confidence, [best, other], `${best} or ${other}`, scores)
  }
  return detection(best, 'Latin', Math.min(1, 0.5 + 0.5 * confidence), [], '', scores)
}

function withinScript(script: string, text: string): Detection {
  const langs = SCRIPT_LANGUAGES[script] ?? []
  if (langs.length === 0) return detection(null, script, 0, [], 'no language known for the script')
  let choice = langs[0] as string
  let confidence = langs.length === 1 ? 0.9 : 0.75
  let reason = ''
  for (const [letters, lang] of SCRIPT_LETTERS[script] ?? []) {
    if ([...letters].some((ch) => text.includes(ch))) {
      choice = lang
      reason = 'letters'
      break
    }
  }
  const table = SCRIPT_WORDS[script]
  if (table && !reason) {
    const tokens = words(text)
    const scores = new Map<string, number>()
    for (const [lang, common] of table) scores.set(lang, tokens.filter((t) => common.has(t)).length)
    const order = ranked(scores)
    const first = order[0]
    if (first && first[1] >= 1 && (order.length < 2 || first[1] > (order[1] as [string, number])[1])) {
      choice = first[0]
      reason = 'words'
      confidence = 0.85
    }
  }
  if (script === 'Han') confidence = 0.8 // Chinese (Cantonese and Japanese kanji read the same letters)
  return detection(
    choice,
    script,
    confidence,
    langs.filter((l) => l !== choice),
    reason,
  )
}

/** Guess the language of `text` (markup such as SSML tags is ignored). */
export function detectLanguage(text: string | null | undefined): Detection {
  const clean = (text ?? '').replace(TAG_RE, (_m, label: string | undefined) => label || ' ')
  const script = dominantScript(scriptCounts(clean))
  if (script == null) return detection(null, null, 0, [], 'no letters')
  if (script === 'Latin') return latin(clean)
  return withinScript(script, clean)
}

/** `"hi-IN"` -> `"hi"`; `"nb"` counts as Norwegian (`"no"`). */
export function baseLanguage(tag: string | null | undefined): string {
  const code = (tag ?? '').replaceAll('_', '-').split('-')[0]?.toLowerCase() ?? ''
  return ({ nb: 'no', nn: 'no', fil: 'tl', zho: 'zh', cmn: 'zh' } as Record<string, string>)[code] ?? code
}

/** Short description of a detection for people ("Hindi · Devanagari", "unsure: fr or it"). */
export function describeDetection(d: Detection, name: (code: string) => string = (c) => c): string {
  if (d.language) return `${name(d.language)}${d.script ? ` · ${d.script}` : ''}`
  return d.reason ? `unsure: ${d.reason}` : 'unsure'
}
