import { isDemo } from './api/config'
import { Store } from './store'

export interface Prefs {
  sttModel: string | null
  ttsModel: string | null
  ttsVoice: string | null
  language: string
  /** Model used by the in-browser recogniser. */
  browserModel: string
  /** The visitor chose that model themselves; until then it is the default (Moonshine), whatever was stored. */
  browserModelPicked: boolean
  /** Language for the in-browser recogniser (`auto` or a code); only multilingual models use it. */
  browserLanguage: string
  /** End-to-end latency target in ms used by budget bars and target lines. */
  targetMs: number
  modelsView: 'cards' | 'table'
  /** Order of the Models screen: the catalog's own, or best fit for the server's machine first. */
  modelsSort: 'default' | 'fit'
  /** The first-run "Pick for me" prompt was dismissed. */
  pickDismissed: boolean
  /** In-browser recogniser: the repetition guard's `max_repeats` (0: off), as on the server. */
  browserMaxRepeats: number
}

const KEY = 'bolo.prefs'
const defaults: Prefs = {
  sttModel: null,
  ttsModel: null,
  ttsVoice: null,
  language: 'auto',
  browserModel: 'moonshine-tiny',
  browserModelPicked: false,
  browserLanguage: 'auto',
  // A tab's recogniser waits half a second of silence before it answers, so the demo starts with a looser target.
  targetMs: isDemo ? 800 : 300,
  modelsView: 'cards',
  modelsSort: 'default',
  pickDismissed: false,
  browserMaxRepeats: 4,
}

export function readPrefs(): Prefs {
  try {
    const stored = { ...defaults, ...(JSON.parse(localStorage.getItem(KEY) ?? '{}') as Partial<Prefs>) }
    // Earlier versions stored a recogniser the visitor never chose (the speech-synthesis round trip switched
    // it to the Hindi model and kept it). Only a deliberate choice outlives the visit.
    if (!stored.browserModelPicked) stored.browserModel = defaults.browserModel
    return stored
  } catch {
    return defaults
  }
}

export const prefs = new Store<Prefs>(readPrefs())

prefs.subscribe(() => {
  try {
    localStorage.setItem(KEY, JSON.stringify(prefs.get()))
  } catch {
    // ignore
  }
})

/** Remember the in-browser recogniser the visitor chose. */
export function pickBrowserModel(id: string): void {
  prefs.patch({ browserModel: id, browserModelPicked: true })
}

export function setPref<K extends keyof Prefs>(key: K, value: Prefs[K]): void {
  prefs.patch({ [key]: value } as Pick<Prefs, K>)
}
