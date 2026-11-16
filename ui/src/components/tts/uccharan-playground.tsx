import { Loader2, Wand2 } from 'lucide-react'
import { useState } from 'react'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { voicesBase } from '@/lib/browser-tts/voices'
import type * as Uccharan from '@/vendor/uccharan/index.js'

/**
 * Every language uccharan reads, with its code. Romanised input is turned into
 * native script where a transliteration pack exists (`xlit` names the pack:
 * Maithili, Bhojpuri and Magahi use the Hindi one).
 */
const LANGS: { code: string; label: string; xlit: string | null }[] = [
  { code: 'hi', label: 'हिन्दी · Hindi', xlit: 'hi' },
  { code: 'mai', label: 'मैथिली · Maithili', xlit: 'hi' },
  { code: 'bho', label: 'भोजपुरी · Bhojpuri', xlit: 'hi' },
  { code: 'mag', label: 'मगही · Magahi', xlit: 'hi' },
  { code: 'mr', label: 'मराठी · Marathi', xlit: 'mr' },
  { code: 'ne', label: 'नेपाली · Nepali', xlit: null },
  { code: 'kok', label: 'कोंकणी · Konkani', xlit: null },
  { code: 'sa', label: 'संस्कृतम् · Sanskrit', xlit: null },
  { code: 'bn', label: 'বাংলা · Bengali', xlit: 'bn' },
  { code: 'as', label: 'অসমীয়া · Assamese', xlit: null },
  { code: 'or', label: 'ଓଡ଼ିଆ · Odia', xlit: null },
  { code: 'gu', label: 'ગુજરાતી · Gujarati', xlit: 'gu' },
  { code: 'pa', label: 'ਪੰਜਾਬੀ · Punjabi', xlit: null },
  { code: 'te', label: 'తెలుగు · Telugu', xlit: 'te' },
  { code: 'kn', label: 'ಕನ್ನಡ · Kannada', xlit: null },
  { code: 'ta', label: 'தமிழ் · Tamil', xlit: null },
  { code: 'ml', label: 'മലയാളം · Malayalam', xlit: null },
  { code: 'si', label: 'සිංහල · Sinhala', xlit: null },
  { code: 'ur', label: 'اردو · Urdu', xlit: null },
  { code: 'sd', label: 'سنڌي · Sindhi', xlit: null },
  { code: 'en-us', label: 'English (US)', xlit: null },
  { code: 'en-gb', label: 'English (UK)', xlit: null },
  { code: 'de', label: 'Deutsch · German', xlit: null },
  { code: 'nl', label: 'Nederlands · Dutch', xlit: null },
  { code: 'fr', label: 'Français · French', xlit: null },
  { code: 'fr-be', label: 'Français (Belgique) · French (Belgium)', xlit: null },
  { code: 'fr-ch', label: 'Français (Suisse) · French (Switzerland)', xlit: null },
  { code: 'es', label: 'Español · Spanish', xlit: null },
  { code: 'ca', label: 'Català · Catalan', xlit: null },
  { code: 'it', label: 'Italiano · Italian', xlit: null },
  { code: 'pt', label: 'Português (Portugal) · Portuguese (Portugal)', xlit: null },
  { code: 'pt-br', label: 'Português (Brasil) · Portuguese (Brazil)', xlit: null },
  { code: 'ro', label: 'Română · Romanian', xlit: null },
  { code: 'pl', label: 'Polski · Polish', xlit: null },
  { code: 'cs', label: 'Čeština · Czech', xlit: null },
  { code: 'hu', label: 'Magyar · Hungarian', xlit: null },
  { code: 'tr', label: 'Türkçe · Turkish', xlit: null },
  { code: 'fi', label: 'Suomi · Finnish', xlit: null },
  { code: 'sv', label: 'Svenska · Swedish', xlit: null },
  { code: 'nb', label: 'Norsk bokmål · Norwegian', xlit: null },
  { code: 'da', label: 'Dansk · Danish', xlit: null },
]

const SAMPLE: Record<string, string> = {
  hi: 'aap kaise ho? मेरे पास ₹25 हैं, कल 10:30 बजे मिलते हैं।',
  mai: 'प्रणाम! अहाँ केहन छी? काल्हि 10 बजे बैसक अछि।',
  'en-us': 'Dr. Rao paid $25 on 3/4/2025 at 10:30.',
  de: 'Herr Nguyen zahlt 25 € für 3,5 kg Äpfel.',
  es: 'Son 3,5 kg por 20 €.',
}

interface Result {
  romanised: string | null
  normalised: string
  phonemes: string
  ms: number
}

let loaded: Promise<typeof Uccharan> | null = null
const packs = new Set<string>()

async function lib(): Promise<typeof Uccharan> {
  loaded ??= import('@/vendor/uccharan/index.js').then(async (m) => {
    await m.init({ dataBaseUrl: `${voicesBase()}uccharan/` })
    return m
  })
  return loaded
}

/**
 * The text front end the Piper and Kokoro voices use, on its own: romanised
 * Indian-language text into native script, numbers and symbols read out, and
 * the phonemes a voice is given. Runs in the page; nothing is sent anywhere.
 */
export function UccharanPlayground() {
  const [lang, setLang] = useState('hi')
  const [text, setText] = useState(SAMPLE.hi as string)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<Result | null>(null)
  const [error, setError] = useState<string | null>(null)
  const meta = LANGS.find((l) => l.code === lang)

  const run = async () => {
    setBusy(true)
    setError(null)
    const started = performance.now()
    try {
      const m = await lib()
      let input = text
      let romanised: string | null = null
      const pack = meta?.xlit
      if (pack && /[A-Za-z]/.test(text)) {
        if (!packs.has(pack)) {
          try {
            const res = await fetch(`${voicesBase()}uccharan/xlit-${pack}.ucd`)
            if (res.ok) await m.loadTranslit(pack, new Uint8Array(await res.arrayBuffer()))
          } catch {
            // without the pack uccharan's rules answer
          }
          packs.add(pack)
        }
        romanised = (await m.transliterate(text, { lang: pack })).text
        input = romanised
      }
      const normalised = (await m.normalize(input, { lang })).text
      const phonemes = await m.phonemize(input, { lang })
      setResult({ romanised, normalised, phonemes, ms: performance.now() - started })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel
      title="Text front end"
      description="uccharan, in this page: native script, spoken form, phonemes"
      data-testid="uccharan-playground"
      bodyClassName="grid gap-3 text-[13px] leading-5"
    >
      <div className="flex flex-wrap items-center gap-2">
        <Select
          value={lang}
          onValueChange={(l) => {
            setLang(l)
            if (Object.values(SAMPLE).includes(text)) setText(SAMPLE[l] ?? text)
            setResult(null)
          }}
        >
          <SelectTrigger
            className="h-8 w-72 max-w-full"
            aria-label="Playground language"
            data-testid="playground-language"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent position="popper" align="start">
            {LANGS.map((l) => (
              <SelectItem key={l.code} value={l.code}>
                {l.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button size="sm" onClick={() => void run()} disabled={busy || !text.trim()} data-testid="playground-run">
          {busy ? <Loader2 className="animate-spin" /> : <Wand2 />} Show
        </Button>
      </div>
      <Textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        aria-label="Text for the front end"
        className="min-h-16 text-[14px] leading-6"
        data-testid="playground-text"
      />
      {error && (
        <p className="text-destructive" role="alert">
          {error}
        </p>
      )}
      {result && (
        <dl className="grid grid-cols-[7rem_minmax(0,1fr)] gap-x-3 gap-y-1.5" data-testid="playground-result">
          {result.romanised != null && (
            <>
              <dt className="text-muted-foreground">Native script</dt>
              <dd lang={lang} className="break-words" data-testid="playground-native">
                {result.romanised}
              </dd>
            </>
          )}
          <dt className="text-muted-foreground">Spoken form</dt>
          <dd lang={lang} className="break-words" data-testid="playground-normalised">
            {result.normalised}
          </dd>
          <dt className="text-muted-foreground">Phonemes</dt>
          <dd className="num break-words" data-testid="playground-phonemes">
            {result.phonemes}
          </dd>
          <dt className="text-muted-foreground">Time</dt>
          <dd className="num">{Math.round(result.ms)} ms</dd>
        </dl>
      )}
      <p className="text-xs text-muted-foreground">
        uccharan 0.6.3, Apache-2.0: {LANGS.length} languages and regional variants. Romanised text is turned into native
        script in {LANGS.filter((l) => l.xlit).length} of them with its transliteration packs (CC BY-SA 4.0, built from
        Dakshina, Aksharantar and other corpora; Maithili, Bhojpuri and Magahi use the Hindi pack); for the other
        languages and without a pack it falls back to its own rules.
      </p>
    </Panel>
  )
}
