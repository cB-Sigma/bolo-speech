import { baseLanguage, detectLanguage, type Detection } from '@/lib/langdetect'
import type { BrowserVoice } from './voices'

/**
 * Which voice reads a text, decided as the server decides it for a request with `detect_language` (bolo's text
 * language detector, ported in src/lib/langdetect.ts): the detected language, else the closest language of its
 * script that a voice speaks. The answer is spelled as the server's `x-bolo-language` header, so the console's
 * readers of that header (src/lib/blend.ts) describe it.
 */
export interface VoiceChoice {
  voice: BrowserVoice
  detection: Detection
  /** `hi-IN; source=detected; detected=hi; script=Devanagari; voice=…`. */
  header: string
  switched: boolean
}

const speaks = (v: BrowserVoice, lang: string) => baseLanguage(v.language) === lang

/**
 * The voice for `text`. With `follow` (the page's "pick the voice from the text"), a voice of the text's language
 * replaces `current` when `current` speaks another one: of the same kind and gender if there is one, else one
 * already downloaded, else the first. Without it, `current` reads the text and a mismatch is reported.
 */
export function voiceForText(
  text: string,
  current: BrowserVoice,
  voices: BrowserVoice[],
  opts: { follow: boolean; downloaded?: (v: BrowserVoice) => boolean; blend?: string | null },
): VoiceChoice {
  const d = detectLanguage(text)
  const script = d.script ? `; script=${d.script}` : ''
  const detected = d.language ?? 'unsure'
  const keep = (source: string, extra = ''): VoiceChoice => ({
    voice: current,
    detection: d,
    header: `${current.language}; source=${source}; detected=${detected}${script}${extra}; voice=${opts.blend ?? current.id}`,
    switched: false,
  })
  if (!d.language) return keep('voice')
  if (speaks(current, d.language)) return keep(opts.follow ? 'detected' : 'voice')
  if (!opts.follow || opts.blend) return keep('voice', '; mismatch=true')
  // Only the script was recognised (no letters or common words of one language decided): Maithili, Bhojpuri or
  // Marathi text reads as the script's first language, Hindi, so a voice of the same script that was picked stays
  if (!d.reason && d.alternatives.some((lang) => speaks(current, lang))) return keep('voice', '; same-script=true')
  for (const lang of [d.language, ...d.alternatives]) {
    const fit = voices.filter((v) => speaks(v, lang))
    if (!fit.length) continue
    const choice =
      fit.find((v) => v.kind === current.kind && v.gender === current.gender && opts.downloaded?.(v)) ??
      fit.find((v) => opts.downloaded?.(v)) ??
      fit.find((v) => v.kind === current.kind && v.gender === current.gender) ??
      (fit[0] as BrowserVoice)
    const matched = lang !== d.language ? `; matched=${lang}` : ''
    return {
      voice: choice,
      detection: d,
      header: `${choice.language}; source=detected; detected=${d.language}${script}${matched}; voice=${choice.id}`,
      switched: choice.id !== current.id,
    }
  }
  return keep('voice', '; offered=false')
}
