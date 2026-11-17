import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Component } from '@/lib/api/types'
import {
  defaultFilter,
  filterPacks,
  needText,
  packName,
  pageOf,
  readPacksOpen,
  sortPacks,
  summarize,
  summaryText,
  writePacksOpen,
} from '@/lib/language-packs'

const pack = (name: string, extra: Partial<Component> = {}): Component => ({
  id: `uccharan-${name}`,
  name: `uccharan language pack '${name}'`,
  kind: 'language-pack',
  installed: false,
  source: null,
  version: null,
  license: 'Apache-2.0',
  license_url: null,
  required_by_count: 0,
  required_by_installed_count: 0,
  installable: true,
  install_blocked_reason: null,
  manual_command: '',
  description: '',
  languages: [name.replace(/^(kokoro|wikipron|xlit)-/, '')],
  ...extra,
})

const LIST = [
  pack('de', { required_by_count: 3, required_by_installed_count: 1 }),
  pack('fr', { required_by_count: 5, installed: true }),
  pack('it', { required_by_count: 4, required_by_installed_count: 2, installed: true }),
  pack('en-us', { built_in: true, used_by_count: 19 }),
  pack('hi', { built_in: true, used_by_count: 5 }),
  pack('xlit-hi', { license: 'CC BY-SA 4.0' }),
  pack('ca', { required_by_count: 1 }),
]

afterEach(() => {
  vi.unstubAllGlobals()
  localStorage.clear()
})

describe('language packs', () => {
  it('names packs by language and kind', () => {
    expect(packName(pack('de'))).toBe('German')
    expect(packName(pack('kokoro-en-us'))).toBe('American English · Kokoro')
    expect(packName(pack('xlit-hi'))).toBe('Hindi · romanised')
  })

  it('says a built-in pack is not needed instead of a bare zero', () => {
    expect(needText(LIST[3]!)).toBe('built in, not needed · used by 19 models')
    expect(needText(LIST[0]!)).toBe('needed by 3 models (1 installed)')
    expect(needText(LIST[5]!)).toBe('not needed by any model')
  })

  it('puts packs needed by installed models first, missing ones before installed ones', () => {
    expect(sortPacks(LIST).map((c) => c.id.replace('uccharan-', ''))).toEqual([
      'de',
      'it',
      'fr',
      'ca',
      'xlit-hi',
      'en-us',
      'hi',
    ])
  })

  it('summarises and filters', () => {
    const s = summarize(LIST)
    expect(s).toEqual({ total: 7, installed: 2, needed: 2, missing: 1, mine: true })
    expect(summaryText(s)).toBe('2 of 7 installed · 2 needed by your models, 1 missing')
    expect(defaultFilter(LIST)).toBe('needed')
    expect(defaultFilter([LIST[1]!])).toBe('all')
    expect(filterPacks(LIST, 'needed', '').map((c) => c.id)).toEqual(['uccharan-de', 'uccharan-it'])
    expect(filterPacks(LIST, 'installed', '').map((c) => c.id)).toEqual(['uccharan-it', 'uccharan-fr'])
    expect(filterPacks(LIST, 'all', 'hindi').map((c) => c.id)).toEqual(['uccharan-xlit-hi', 'uccharan-hi'])
    expect(filterPacks(LIST, 'all', 'CATAL').map((c) => c.id)).toEqual(['uccharan-ca'])
  })

  it('counts every catalog model on servers that do not report installed ones', () => {
    const old = LIST.map(({ required_by_installed_count: _, ...c }) => c as Component)
    expect(summaryText(summarize(old))).toBe('2 of 7 installed · 4 needed by models, 2 missing')
  })

  it('pages eight at a time', () => {
    const items = Array.from({ length: 20 }, (_, i) => i)
    expect(pageOf(items, 0)).toMatchObject({ items: [0, 1, 2, 3, 4, 5, 6, 7], from: 1, to: 8, total: 20, pages: 3 })
    expect(pageOf(items, 2)).toMatchObject({ items: [16, 17, 18, 19], from: 17, to: 20 })
    expect(pageOf(items, 9).page).toBe(2)
    expect(pageOf([], 0)).toMatchObject({ items: [], from: 0, to: 0, pages: 1 })
  })

  it('remembers whether the list is open, and survives storage that throws', () => {
    expect(readPacksOpen()).toBe(false)
    writePacksOpen(true)
    expect(readPacksOpen()).toBe(true)
    vi.stubGlobal('localStorage', {
      getItem: () => {
        throw new Error('blocked')
      },
      setItem: () => {
        throw new Error('blocked')
      },
    })
    expect(readPacksOpen()).toBe(false)
    expect(() => writePacksOpen(true)).not.toThrow()
  })
})
