import { afterEach, describe, expect, it } from 'vitest'
import { DEFAULT_BROWSER_MODEL, suggestedModelFor } from './browser-stt/models'
import { pickBrowserModel, prefs, readPrefs } from './prefs'

const KEY = 'bolo.prefs'

describe('the in-browser recogniser that is selected', () => {
  afterEach(() => {
    localStorage.removeItem(KEY)
    prefs.set(readPrefs())
    localStorage.removeItem(KEY)
  })

  it('is Moonshine for a new visitor', () => {
    expect(DEFAULT_BROWSER_MODEL.id).toBe('moonshine-tiny')
    expect(DEFAULT_BROWSER_MODEL.languages).toEqual(['en'])
    expect(readPrefs().browserModel).toBe('moonshine-tiny')
    expect(readPrefs().browserModelPicked).toBe(false)
  })

  it('is Moonshine again when an earlier version stored a model the visitor never chose', () => {
    localStorage.setItem(KEY, JSON.stringify({ browserModel: 'whisper-tiny-hindi', targetMs: 500 }))
    const p = readPrefs()
    expect(p.browserModel).toBe('moonshine-tiny')
    expect(p.targetMs).toBe(500)
  })

  it('stays what the visitor chose', () => {
    pickBrowserModel('whisper-tiny-hindi')
    expect(prefs.get().browserModel).toBe('whisper-tiny-hindi')
    expect(readPrefs().browserModel).toBe('whisper-tiny-hindi')
    expect(readPrefs().browserModelPicked).toBe(true)
  })

  it('Hindi is still offered its own model, and English Moonshine', () => {
    expect(suggestedModelFor('hi')?.id).toBe('whisper-tiny-hindi')
    expect(suggestedModelFor('en')?.id).toBe('moonshine-tiny')
  })
})
