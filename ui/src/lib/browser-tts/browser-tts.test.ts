import { describe, expect, it } from 'vitest'
import spec from '../../../scripts/voices/voices.json'
import fixtures from './fixtures/sherpa-tokens.json'
import { asciiDigits, expandNumbers, numberToWords } from './numbers'
import { shortFirstChunk, synthesisChunks } from './session'
import { scaleSilence } from './silence'
import {
  coquiIds,
  mergeShortChunks,
  multilingualClean,
  piperIds,
  prepareText,
  splitSentences,
  withLeadIn,
  type CoquiVoiceConfig,
  type VoiceConfig,
} from './text'
import { LANGUAGES as UCCHARAN_LANGUAGES } from '@/vendor/uccharan/index.js'
import { baseVoiceId, browserVoice, BROWSER_VOICES, languageLabel, voicesByLanguage } from './voices'

const coqui = (symbols: Record<string, number>, blank = 99): VoiceConfig => ({
  id: 'test',
  kind: 'coqui',
  sample_rate: 22050,
  add_blank: true,
  blank_id: blank,
  use_eos_bos: false,
  bos_id: 100,
  eos_id: 100,
  cleaner: 'multilingual',
  inference: { noise_scale: 0.667, length_scale: 1, noise_scale_w: 0.8 },
  symbols,
})

describe('text cleaning (Coqui multilingual_cleaners)', () => {
  it('lower-cases, swaps ; and : for commas, hyphens for spaces, drops brackets and quotes, collapses spaces', () => {
    expect(multilingualClean('  Hello;  "World" (again) - yes: NO  ')).toBe('hello, world again yes, no')
    expect(multilingualClean('[a] <b>')).toBe('a b')
  })
  it('composes to NFC, as the training text was', () => {
    // क + nukta stays two code points under NFC (क़ is a composition exclusion); अ + ा does not compose.
    expect(multilingualClean('क़')).toBe('क़')
    expect(multilingualClean('क़')).toBe('क़')
    expect(multilingualClean('é')).toBe('é')
  })
})

describe('token ids', () => {
  it.each(fixtures.cases.map((c) => [c.voice, c.text, c] as const))(
    '%s: "%s" gives the ids that reproduce sherpa-onnx',
    (_voice, text, c) => {
      const cfg = coqui(c.symbols as unknown as Record<string, number>, c.blank_id) as CoquiVoiceConfig
      expect(coquiIds(cfg, multilingualClean(text))).toEqual(c.ids)
    },
  )
  it('has a fixture for every Indian language voice family', () => {
    expect(new Set(fixtures.cases.map((c) => c.voice))).toEqual(
      new Set([
        'syspin-hi-female',
        'syspin-mai-female',
        'syspin-bho-female',
        'syspin-mr-female',
        'syspin-kn-female',
        'syspin-en-female',
      ]),
    )
  })
  it('skips unknown characters and intersperses the blank', () => {
    expect(coquiIds(coqui({ a: 1, b: 2 }, 0) as CoquiVoiceConfig, 'axb')).toEqual([0, 1, 0, 2, 0])
    expect(
      coquiIds({ ...(coqui({ a: 1 }, 0) as CoquiVoiceConfig), add_blank: false, use_eos_bos: true }, 'aa'),
    ).toEqual([100, 1, 1, 100])
  })
  it('builds Piper ids as ^ _ p _ … $ (the layout uccharan and Piper use)', () => {
    const map = { _: [0], '^': [1], $: [2], h: [20], a: [14], ɪ: [74], ˈ: [120], '.': [10] }
    expect(piperIds(map, 'hˈaɪ.')).toEqual({ ids: [1, 0, 20, 0, 120, 0, 14, 0, 74, 0, 10, 0, 2], missing: [] })
    expect(piperIds(map, 'hx').missing).toEqual(['x'])
  })
})

describe('numbers', () => {
  it('English', () => {
    expect(numberToWords(0, 'en')).toBe('zero')
    expect(numberToWords(42, 'en')).toBe('forty two')
    expect(numberToWords(105, 'en')).toBe('one hundred and five')
    expect(numberToWords(2026, 'en')).toBe('two thousand and twenty six')
    expect(numberToWords(1_250_000, 'en')).toBe('one million two hundred and fifty thousand')
  })
  it('Hindi, with lakh and crore', () => {
    expect(numberToWords(10, 'hi')).toBe('दस')
    expect(numberToWords(45, 'hi')).toBe('पैंतालीस')
    expect(numberToWords(99, 'hi')).toBe('निन्यानबे')
    expect(numberToWords(125, 'hi')).toBe('एक सौ पच्चीस')
    expect(numberToWords(2026, 'hi')).toBe('दो हज़ार छब्बीस')
    expect(numberToWords(150_000, 'hi')).toBe('एक लाख पचास हज़ार')
    expect(numberToWords(30_000_000, 'hi')).toBe('तीन करोड़')
  })
  it('Marathi, with शंभर and -शे hundreds', () => {
    expect(numberToWords(100, 'mr')).toBe('शंभर')
    expect(numberToWords(105, 'mr')).toBe('एकशे पाच')
    expect(numberToWords(500, 'mr')).toBe('पाचशे')
    expect(numberToWords(2026, 'mr')).toBe('दोन हजार सव्वीस')
    expect(numberToWords(75, 'mr')).toBe('पंच्याहत्तर')
  })
  it('Kannada, joining vowel-initial units (ಇಪ್ಪತ್ತೊಂದು)', () => {
    expect(numberToWords(21, 'kn')).toBe('ಇಪ್ಪತ್ತೊಂದು')
    expect(numberToWords(23, 'kn')).toBe('ಇಪ್ಪತ್ತಮೂರು')
    expect(numberToWords(26, 'kn')).toBe('ಇಪ್ಪತ್ತಾರು')
    expect(numberToWords(99, 'kn')).toBe('ತೊಂಬತ್ತೊಂಬತ್ತು')
    expect(numberToWords(300, 'kn')).toBe('ಮುನ್ನೂರು')
    expect(numberToWords(1000, 'kn')).toBe('ಸಾವಿರ')
    expect(numberToWords(2026, 'kn')).toBe('ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರು')
  })
  it('every table has a word for 0..99', () => {
    for (const lang of ['en', 'hi', 'mr', 'kn'] as const) {
      for (let n = 0; n < 100; n++) expect(numberToWords(n, lang), `${lang} ${n}`).toMatch(/\S/)
    }
  })
  it('expands numbers in running text: native digits, grouping, decimals, per cent, codes', () => {
    expect(asciiDigits('१२३ ೪೫')).toBe('123 45')
    expect(expandNumbers('कल १० बजे', 'hi')).toBe('कल दस बजे')
    expect(expandNumbers('1,50,000 रुपये', 'hi')).toBe('एक लाख पचास हज़ार रुपये')
    expect(expandNumbers('It is 3.5 km', 'en')).toBe('It is three point five km')
    expect(expandNumbers('50% off', 'en')).toBe('fifty percent off')
    expect(expandNumbers('ಶೇ 50%', 'kn')).toBe('ಶೇ ಶೇಕಡಾ ಐವತ್ತು')
    expect(expandNumbers('call 0612', 'en')).toBe('call zero six one two')
  })
})

describe('sentences', () => {
  it('splits after sentence punctuation in Latin and Indian scripts, keeping it', () => {
    expect(splitSentences('नमस्ते! आप कैसे हैं? आज मौसम अच्छा है। ठीक है')).toEqual([
      'नमस्ते!',
      'आप कैसे हैं?',
      'आज मौसम अच्छा है।',
      'ठीक है',
    ])
    expect(splitSentences('One. Two')).toEqual(['One.', 'Two'])
  })
  it('splits an over-long sentence at a comma, then at a space', () => {
    const long = `${'a'.repeat(100)}, ${'b'.repeat(100)}, ${'c'.repeat(100)}`
    const parts = splitSentences(long, 150)
    expect(parts.every((p) => p.length <= 150)).toBe(true)
    expect(parts.join(' ').replace(/[ ,]/g, '')).toBe(long.replace(/[ ,]/g, ''))
  })
  it('cuts a long first sentence at its first comma so the first audio comes sooner', () => {
    const first = `${'x'.repeat(40)}, ${'y'.repeat(80)}.`
    expect(shortFirstChunk([first, 'Next.'])).toEqual([`${'x'.repeat(40)},`, `${'y'.repeat(80)}.`, 'Next.'])
    expect(shortFirstChunk(['Short, fine.'])).toEqual(['Short, fine.'])
  })
  it('never synthesises a one-word sentence by itself: it joins the next one', () => {
    expect(mergeShortChunks(splitSentences('नमस्ते! आप कैसे हैं? आज मौसम बहुत अच्छा है।'))).toEqual([
      'नमस्ते! आप कैसे हैं?',
      'आज मौसम बहुत अच्छा है।',
    ])
    // Every sample sentence of the demo opens with a one-word greeting; no character voice is given it alone.
    for (const v of BROWSER_VOICES) {
      const chunks = synthesisChunks(splitSentences(v.sample), v.kind)
      expect(chunks.length, v.id).toBeGreaterThan(1)
      expect(chunks.join(' ')).toBe(v.sample)
      if (v.kind === 'coqui') expect(chunks.every((c) => c.length >= 12)).toBe(true)
    }
  })
  it('keeps the sentences of a Piper voice apart: it says a short one well, and the first is heard sooner', () => {
    const sentences = ['Hello!', 'This voice runs entirely in your browser.']
    expect(synthesisChunks(sentences, 'piper')).toEqual(sentences)
    expect(synthesisChunks(sentences, 'coqui')).toEqual(['Hello! This voice runs entirely in your browser.'])
  })
  it('joins a short last sentence to the one before it, and several short ones in a row', () => {
    expect(mergeShortChunks(['The weather is lovely.', 'Thanks!'])).toEqual(['The weather is lovely. Thanks!'])
    expect(mergeShortChunks(['Yes.', 'No.', 'Maybe.', 'A sentence of ordinary length.'])).toEqual([
      'Yes. No. Maybe.',
      'A sentence of ordinary length.',
    ])
    expect(mergeShortChunks(['Hi.'])).toEqual(['Hi.'])
    expect(mergeShortChunks([])).toEqual([])
  })
  it('leaves sentences of ordinary length alone and never joins past the size limit', () => {
    const two = ['आप कैसे हैं? ठीक हूँ।', 'आज मौसम बहुत अच्छा है।']
    expect(mergeShortChunks(two)).toEqual(two)
    const long = 'x'.repeat(215)
    expect(mergeShortChunks(['Hello!', long])).toEqual(['Hello!', long])
    expect(mergeShortChunks([long, 'Goodbye!'])).toEqual([long, 'Goodbye!'])
  })
  it('a merged opening is still cut at a comma when it is long', () => {
    const next = `${'x'.repeat(40)}, ${'y'.repeat(80)}.`
    expect(shortFirstChunk(mergeShortChunks(['Hello!', next, 'Next sentence here.']))).toEqual([
      `Hello! ${'x'.repeat(40)},`,
      `${'y'.repeat(80)}.`,
      'Next sentence here.',
    ])
  })
})

describe('lead-in for character voices', () => {
  it('puts one space in front of the model input, when the voice has a space', () => {
    const cfg = { symbols: { a: 1, ' ': 2 } }
    expect(withLeadIn(cfg, 'a a')).toBe(' a a')
    expect(withLeadIn(cfg, ' a')).toBe(' a')
    expect(withLeadIn(cfg, '')).toBe('')
    expect(withLeadIn({ symbols: { a: 1 } }, 'a')).toBe('a')
  })
  it('becomes a space id between blanks at the start of the ids', () => {
    const cfg = coqui({ a: 1, ' ': 2 })
    if (cfg.kind !== 'coqui') throw new Error('expected a character voice')
    expect(coquiIds(cfg, withLeadIn(cfg, 'a'))).toEqual([99, 2, 99, 1, 99])
  })
})

describe('prepareText', () => {
  const cfg = coqui({ अ: 1, ब: 2, ' ': 3, '.': 4, ए: 5, क: 6, द: 7, स: 8, '्': 9, 'ो': 10 })
  it('spells numbers out, lists characters the voice cannot say, and falls back to . for a missing danda', () => {
    const p = prepareText('अब 2। Q', cfg, 'hi')
    expect(p.spoken).toBe('अब दो. q')
    expect(p.chunks).toEqual(['अब दो.', 'q'])
    expect(p.dropped).toEqual(['q'])
    expect(p.rewritten).toBe(true)
  })
  it('leaves Piper text to the phonemiser apart from the sentence split', () => {
    const piper: VoiceConfig = {
      id: 'p',
      kind: 'piper',
      sample_rate: 16000,
      phoneme_language: 'en-us',
      phoneme_id_map: {},
      inference: { noise_scale: 0.667, length_scale: 1, noise_scale_w: 0.8 },
    }
    expect(prepareText('It costs $5.  Really?', piper, null)).toEqual({
      spoken: 'It costs $5. Really?',
      chunks: ['It costs $5.', 'Really?'],
      dropped: [],
      rewritten: false,
    })
  })
})

describe('pause shortening (sherpa-onnx ScaleSilence)', () => {
  it('keeps a fifth of every quiet run longer than 0.2 s and leaves shorter ones alone', () => {
    const sr = 1000
    const loud = new Float32Array(100).fill(0.5)
    const x = new Float32Array([...loud, ...new Float32Array(300), ...loud, ...new Float32Array(150), ...loud])
    const y = scaleSilence(x, sr)
    expect(y.length).toBe(100 + 60 + 100 + 150 + 100)
    expect(scaleSilence(x, sr, 1)).toBe(x)
  })
})

describe('voice catalogue', () => {
  const published = BROWSER_VOICES.filter((v) => v.files.from === 'published')
  it('the voices published with the demo match the build list one to one', () => {
    expect(published.map((v) => v.id).sort()).toEqual(spec.voices.map((v) => v.id).sort())
  })
  it('says who made every voice and under which licence, with a sample, size and honest notes', () => {
    for (const v of BROWSER_VOICES) {
      expect(v.credit, v.id).toMatch(/\S/)
      expect(v.license, v.id).toMatch(/MIT|Apache-2\.0/)
      expect(v.dataLicense, v.id).toMatch(/\S/)
      expect(v.source, v.id).toMatch(/^https:\/\/huggingface\.co\//)
      expect(v.sample.length, v.id).toBeLessThan(110)
      expect(v.notes.length, v.id).toBeGreaterThan(0)
      expect(v.approxBytes, v.id).toBeGreaterThan(5e6)
      expect(v.peakBytes, v.id).toBeGreaterThan(v.approxBytes)
      if (v.files.from !== 'published')
        expect('model' in v.files ? v.files.model : v.files.base, v.id).toMatch(
          /^https:\/\/huggingface\.co\/.+\/resolve\/[0-9a-f]{40}\//,
        )
    }
    // The build list's sample for each voice is the console's sample.
    for (const s of spec.voices) {
      const v = BROWSER_VOICES.find((x) => x.id === s.id)
      if ('sample' in s && v?.kind === 'coqui') expect(v.sample, s.id).toBe(s.sample)
    }
  })
  it('every SYSPIN voice is offered 8-bit and as the original 32-bit export', () => {
    const syspin = BROWSER_VOICES.filter((v) => v.id.startsWith('syspin-'))
    expect(syspin.filter((v) => v.precision === '8-bit')).toHaveLength(22)
    expect(syspin.filter((v) => v.precision === '32-bit')).toHaveLength(22)
    expect(browserVoice('syspin-hi-female-fp32').files).toMatchObject({
      from: 'remote',
      config: 'syspin-hi-female/voice.json',
    })
  })
  it('Piper and Kokoro voices use only languages uccharan reads, and say when the phonemes are approximate', () => {
    for (const v of BROWSER_VOICES.filter((x) => x.kind === 'piper' || x.kind === 'kokoro')) {
      expect(v.phonemizer, v.id).toBe('uccharan')
      expect(
        UCCHARAN_LANGUAGES.some((l) => v.phonemeLanguage?.toLowerCase().startsWith(l)),
        v.id,
      ).toBe(true)
    }
    expect(browserVoice('kokoro-hf-alpha').approximate).toMatch(/Approximate/)
    expect(browserVoice('kokoro-af-heart').approximate).toBeUndefined()
    expect(browserVoice('piper-es-mx-ald-medium').approximate).toMatch(/Castilian/)
  })
  it('Kokoro voices come in three precisions sharing one download per precision', () => {
    expect(browserVoice('kokoro-af-heart').downloadId).toBe('kokoro-82m@q8')
    expect(browserVoice('kokoro-bf-emma@fp32')).toMatchObject({
      id: 'kokoro-bf-emma@fp32',
      downloadId: 'kokoro-82m@fp32',
    })
    expect(browserVoice('kokoro-bf-emma@fp16').approxBytes).toBeGreaterThan(160e6)
    expect(baseVoiceId('kokoro-bf-emma@fp16')).toBe('kokoro-bf-emma')
  })
  it('groups voices by language with native names, Indian languages first', () => {
    const labels = voicesByLanguage().map((g) => languageLabel(g))
    expect(labels.slice(0, 6)).toEqual([
      'हिन्दी · Hindi',
      'मैथिली · Maithili',
      'भोजपुरी · Bhojpuri',
      'मगही · Magahi',
      'छत्तीसगढ़ी · Chhattisgarhi',
      'मराठी · Marathi',
    ])
    for (const l of ['বাংলা · Bengali', 'తెలుగు · Telugu', 'ગુજરાતી · Gujarati', 'English (US)', 'Deutsch · German'])
      expect(labels).toContain(l)
    const hindi = voicesByLanguage()[0]?.voices.map((v) => v.kind)
    expect(new Set(hindi)).toEqual(new Set(['coqui', 'piper', 'kokoro', 'kitten']))
  })
})
