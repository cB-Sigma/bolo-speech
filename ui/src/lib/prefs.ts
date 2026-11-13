import { isDemo } from './api/config'
import { Store } from './store'

export interface Prefs {
  sttModel: string | null
  ttsModel: string | null
  ttsVoice: string | null
  language: string
  /** End-to-end latency target in ms used by budget bars and target lines. */
  targetMs: number
  modelsView: 'cards' | 'table'
}

const KEY = 'bolo.prefs'
const defaults: Prefs = {
  sttModel: null,
  ttsModel: null,
  ttsVoice: null,
  language: 'auto',
  // A tab's recogniser waits half a second of silence before it answers, so the demo starts with a looser target.
  targetMs: isDemo ? 800 : 300,
  modelsView: 'cards',
}

function read(): Prefs {
  try {
    return { ...defaults, ...(JSON.parse(localStorage.getItem(KEY) ?? '{}') as Partial<Prefs>) }
  } catch {
    return defaults
  }
}

export const prefs = new Store<Prefs>(read())

prefs.subscribe(() => {
  try {
    localStorage.setItem(KEY, JSON.stringify(prefs.get()))
  } catch {
    // ignore
  }
})

export function setPref<K extends keyof Prefs>(key: K, value: Prefs[K]): void {
  prefs.patch({ [key]: value } as Pick<Prefs, K>)
}
