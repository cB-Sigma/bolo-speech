import { Loader2, Wand2 } from 'lucide-react'
import { useState } from 'react'
import { Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { voicesBase } from '@/lib/browser-tts/voices'
import type * as Uccharan from '@/vendor/uccharan/index.js'

/** Languages offered, with the uccharan code. Romanised input is turned into native script where a pack exists. */
const LANGS: { code: string; label: string; xlit: boolean }[] = [
  { code: 'hi', label: 'हिन्दी · Hindi', xlit: true },
  { code: 'mr', label: 'मराठी · Marathi', xlit: true },
  { code: 'bn', label: 'বাংলা · Bengali', xlit: true },
  { code: 'te', label: 'తెలుగు · Telugu', xlit: true },
  { code: 'gu', label: 'ગુજરાતી · Gujarati', xlit: true },
  { code: 'kn', label: 'ಕನ್ನಡ · Kannada', xlit: false },
  { code: 'ta', label: 'தமிழ் · Tamil', xlit: false },
  { code: 'ml', label: 'മലയാളം · Malayalam', xlit: false },
  { code: 'pa', label: 'ਪੰਜਾਬੀ · Punjabi', xlit: false },
  { code: 'ne', label: 'नेपाली · Nepali', xlit: false },
  { code: 'ur', label: 'اردو · Urdu', xlit: false },
  { code: 'en-us', label: 'English (US)', xlit: false },
  { code: 'en-gb', label: 'English (UK)', xlit: false },
  { code: 'de', label: 'German', xlit: false },
  { code: 'fr', label: 'French', xlit: false },
  { code: 'es', label: 'Spanish', xlit: false },
  { code: 'it', label: 'Italian', xlit: false },
  { code: 'pt-br', label: 'Portuguese (Brazil)', xlit: false },
]

const SAMPLE: Record<string, string> = {
  hi: 'aap kaise ho? मेरे पास ₹25 हैं, कल 10:30 बजे मिलते हैं।',
  'en-us': 'Dr. Rao paid $25 on 3/4/2025 at 10:30.',
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
      if (meta?.xlit && /[A-Za-z]/.test(text)) {
        if (!packs.has(lang)) {
          try {
            const res = await fetch(`${voicesBase()}uccharan/xlit-${lang}.ucd`)
            if (res.ok) await m.loadTranslit(lang, new Uint8Array(await res.arrayBuffer()))
          } catch {
            // without the pack uccharan's rules answer
          }
          packs.add(lang)
        }
        romanised = (await m.transliterate(text, { lang })).text
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
          <SelectTrigger className="h-8 w-48" aria-label="Playground language" data-testid="playground-language">
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
        uccharan 0.4.0, Apache-2.0. Romanised text is turned into {LANGS.filter((l) => l.xlit).length} scripts with its
        transliteration packs (CC BY-SA 4.0, built from Dakshina, Aksharantar and other corpora); for the other
        languages and without a pack it falls back to its own rules.
      </p>
    </Panel>
  )
}
