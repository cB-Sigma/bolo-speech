import { voicesBase } from '@/lib/browser-tts/voices'
import type * as Uccharan from '@/vendor/uccharan/index.js'

let loaded: Promise<typeof Uccharan> | null = null

/**
 * The vendored uccharan in the page, loaded on first use (never at start-up): the
 * WebAssembly module comes with the bundle, and each language's data is fetched from
 * next to the voices (`voices/uccharan/<lang>.ucd`) the first time that language is used.
 */
export function loadUccharan(): Promise<typeof Uccharan> {
  loaded ??= import('@/vendor/uccharan/index.js')
    .then(async (m) => {
      await m.init({ dataBaseUrl: `${voicesBase()}uccharan/` })
      return m
    })
    .catch((err: unknown) => {
      loaded = null
      throw err
    })
  return loaded
}
