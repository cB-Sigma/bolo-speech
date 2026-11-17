import type { Component } from '@/lib/api/types'
import { languageName } from '@/lib/format'

/** uccharan's language packs, as one summary and a compact list (Connect, Models, Components). */

export const PACKS_PAGE_SIZE = 8
export type PackFilter = 'needed' | 'installed' | 'all'

export function isLanguagePack(c: Pick<Component, 'id' | 'kind'>): boolean {
  return c.kind === 'language-pack'
}

/** The pack's own name: `uccharan-kokoro-en-us` -> `kokoro-en-us`. */
export function packCode(c: Pick<Component, 'id'>): string {
  return c.id.replace(/^uccharan-/, '')
}

const KIND_SUFFIX: [string, string][] = [
  ['kokoro-', 'Kokoro'],
  ['wikipron-', 'WikiPron'],
  ['xlit-', 'romanised'],
]

/** A short, readable name: `German`, `American English · Kokoro`, `Hindi · romanised`. */
export function packName(c: Pick<Component, 'id' | 'languages'>): string {
  const code = packCode(c)
  const found = KIND_SUFFIX.find(([prefix]) => code.startsWith(prefix))
  const language = c.languages?.[0] ?? (found ? code.slice(found[0].length) : code)
  const name = languageName(language)
  return found ? `${name} · ${found[1]}` : name
}

/** Models installed here that need the pack (older servers do not say: null). */
export function neededByMine(c: Component): number | null {
  return c.required_by_installed_count ?? null
}

/** Whether the pack is needed by a model installed here (or, on servers that do not say, by any catalog model). */
export function isNeeded(c: Component): boolean {
  if (c.built_in) return false
  const mine = neededByMine(c)
  return (mine ?? c.required_by_count) > 0
}

/** Needed, not installed, and the server can install it without asking for a licence. */
export function isMissing(c: Component): boolean {
  return isNeeded(c) && !c.installed
}

/** What the row says about who needs the pack. */
export function needText(c: Component): string {
  if (c.built_in) {
    const used = c.used_by_count ? ` · used by ${c.used_by_count} model${c.used_by_count === 1 ? '' : 's'}` : ''
    return `built in, not needed${used}`
  }
  const n = c.required_by_count
  if (n === 0) return 'not needed by any model'
  const mine = neededByMine(c)
  const yours = mine ? ` (${mine} installed)` : ''
  return `needed by ${n} model${n === 1 ? '' : 's'}${yours}`
}

function rank(c: Component): number {
  const mine = (neededByMine(c) ?? 0) > 0
  if (mine && !c.installed) return 0
  if (mine) return 1
  if (c.installed) return 2
  if (c.built_in) return 4
  return 3
}

/** Packs needed by installed models first (missing ones before installed ones), then installed, then by name. */
export function sortPacks(list: Component[]): Component[] {
  return [...list].sort((a, b) => rank(a) - rank(b) || packName(a).localeCompare(packName(b)))
}

export function defaultFilter(list: Component[]): PackFilter {
  return list.some(isNeeded) ? 'needed' : 'all'
}

export function filterPacks(list: Component[], filter: PackFilter, query: string): Component[] {
  const q = query.trim().toLowerCase()
  return sortPacks(list).filter((c) => {
    if (filter === 'needed' && !isNeeded(c)) return false
    if (filter === 'installed' && !c.installed) return false
    if (!q) return true
    return `${packName(c)} ${packCode(c)} ${c.languages?.join(' ') ?? ''}`.toLowerCase().includes(q)
  })
}

export interface PackSummary {
  total: number
  installed: number
  needed: number
  missing: number
  /** Whether `needed` counts the models installed here (newer servers) or every catalog model. */
  mine: boolean
}

export function summarize(list: Component[]): PackSummary {
  return {
    total: list.length,
    installed: list.filter((c) => c.installed).length,
    needed: list.filter(isNeeded).length,
    missing: list.filter(isMissing).length,
    mine: list.some((c) => c.required_by_installed_count != null),
  }
}

export function summaryText(s: PackSummary): string {
  const parts = [`${s.installed} of ${s.total} installed`]
  if (s.needed) {
    const who = s.mine ? 'your models' : 'models'
    parts.push(`${s.needed} needed by ${who}${s.missing ? `, ${s.missing} missing` : ''}`)
  }
  return parts.join(' · ')
}

/** One page of a list: the items and the 1-based range shown. */
export function pageOf<T>(list: T[], page: number, size = PACKS_PAGE_SIZE) {
  const pages = Math.max(1, Math.ceil(list.length / size))
  const current = Math.min(Math.max(0, page), pages - 1)
  const start = current * size
  const items = list.slice(start, start + size)
  return {
    items,
    page: current,
    pages,
    from: items.length ? start + 1 : 0,
    to: start + items.length,
    total: list.length,
  }
}

const OPEN_KEY = 'bolo.languagePacks.open'

/** Whether the list was left open in this browser (closed by default; storage can be unavailable). */
export function readPacksOpen(): boolean {
  try {
    return localStorage.getItem(OPEN_KEY) === '1'
  } catch {
    return false
  }
}

export function writePacksOpen(open: boolean): void {
  try {
    localStorage.setItem(OPEN_KEY, open ? '1' : '0')
  } catch {
    // private window or blocked storage: the choice lasts for this page only
  }
}
