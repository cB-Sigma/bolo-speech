import { audioEngine, type AudioDeviceChoice } from '@/lib/browser-audio/engine'
import type { BrowserModel } from '@/lib/browser-stt/models'
import { diag } from '@/lib/diagnostics'
import type { Punctuator } from '@/lib/stt/browser'

/** The server's default punctuation model, the one the page runs. */
export const PUNCT_MODEL = 'punct-en-cnn-bilstm'

/** Languages it knows (the catalog entry's `languages`). */
export const PUNCT_LANGUAGES = ['en']

/** Whether the punctuation model speaks `language` (no language known: it is tried, as the server does). */
export function speaks(language: string | null): boolean {
  if (!language || language === 'auto') return true
  return PUNCT_LANGUAGES.includes(language.toLowerCase().split(/[-_]/)[0] as string)
}

/**
 * The server's `punctuation=auto` for the page's recogniser: each final of a recogniser that writes no punctuation
 * gets it restored, in the languages the model knows, while the visitor leaves it on (both are read at each final,
 * so a change applies at once). The model is loaded on the first final that needs it. When it cannot run, the text
 * stays as it was (as the server's `x-bolo-punctuation: skipped`).
 */
export function browserPunctuator(
  recogniser: () => Pick<BrowserModel, 'noPunctuation'>,
  enabled: () => boolean,
  device: () => AudioDeviceChoice = () => 'auto',
): Punctuator {
  return async (text, language) => {
    if (!enabled() || !recogniser().noPunctuation || !speaks(language)) return { text, ms: null }
    const engine = audioEngine('punct')
    try {
      await engine.load(PUNCT_MODEL, device())
      return await engine.punctuate(text)
    } catch (err) {
      diag.error('punctuation', `${PUNCT_MODEL} could not run, the text is kept as it was: ${String(err)}`)
      return { text, ms: null }
    }
  }
}
