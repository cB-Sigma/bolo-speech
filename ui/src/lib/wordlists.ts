import type { WordEntry } from './api/types'

/**
 * Word-list rows as the console edits them, and the TSV the server stores
 * (uccharan's format: `word, lang, ipa, say, match, note`, tab separated).
 */

export const COLUMNS = ['word', 'lang', 'ipa', 'say', 'match', 'note'] as const
export const MATCHES = ['word', 'case', 'exact', 'regex'] as const

export interface Row {
  /** Stable key for React; not sent. */
  key: string
  word: string
  lang: string
  ipa: string
  say: string
  match: string
  note: string
}

let seq = 0
export const newKey = (): string => `r${++seq}`

export function emptyRow(lang = '*'): Row {
  return { key: newKey(), word: '', lang, ipa: '', say: '', match: 'word', note: '' }
}

export function toRows(entries: WordEntry[]): Row[] {
  return entries.map((e) => ({
    key: newKey(),
    word: e.word ?? '',
    lang: e.lang || '*',
    ipa: e.ipa ?? '',
    say: e.say ?? '',
    match: e.match || 'word',
    note: e.note ?? '',
  }))
}

const clean = (v: string) => v.replace(/[\t\r\n]+/g, ' ').trim()

/** Rows that say nothing at all (a blank line being typed) are left out. */
export function isBlank(r: Row): boolean {
  return !r.word.trim() && !r.ipa.trim() && !r.say.trim() && !r.note.trim()
}

/** The TSV for a PUT. The header is line 1, so the n-th sent row is on line n + 2 (see `rowForLine`). */
export function toTsv(rows: Row[]): { tsv: string; sent: Row[] } {
  const sent = rows.filter((r) => !isBlank(r))
  const lines = [COLUMNS.join('\t')]
  for (const r of sent) {
    lines.push(
      [clean(r.word), clean(r.lang) || '*', clean(r.ipa), clean(r.say), clean(r.match) || 'word', clean(r.note)].join(
        '\t',
      ),
    )
  }
  return { tsv: `${lines.join('\n')}\n`, sent }
}

/** Which sent row a server line number refers to. */
export function rowForLine(sent: Row[], line: number): Row | null {
  return sent[line - 2] ?? null
}

/** Problems the server would report anyway, caught before saving. */
export function rowProblem(r: Row): string | null {
  if (isBlank(r)) return null
  if (!r.word.trim()) return 'A word is needed.'
  if (!r.ipa.trim() && !r.say.trim()) return 'Give an IPA pronunciation, a spoken form, or both.'
  if (r.match === 'regex') {
    try {
      new RegExp(r.word)
    } catch {
      return 'This is not a valid regular expression.'
    }
  }
  return null
}

/** Line number in a server error message ("line 7: …"), if any. */
export function lineOf(message: string): number | null {
  const m = /line (\d+)/i.exec(message)
  return m ? Number(m[1]) : null
}

export type Piece = { kind: 'text'; text: string } | { kind: 'override'; word: string; ipa: string }

/** Splits text with Kokoro-style `[word](/ipa/)` overrides into plain text and override chips. */
export function splitOverrides(text: string): Piece[] {
  const out: Piece[] = []
  const re = /\[([^\]]+)\]\(\/([^)]*)\/\)/g
  let last = 0
  for (let m = re.exec(text); m; m = re.exec(text)) {
    if (m.index > last) out.push({ kind: 'text', text: text.slice(last, m.index) })
    out.push({ kind: 'override', word: m[1] as string, ipa: m[2] as string })
    last = m.index + m[0].length
  }
  if (last < text.length) out.push({ kind: 'text', text: text.slice(last) })
  return out
}

export const NAME_RE = /^[a-z0-9_-]{1,64}$/

/** Content type for an imported file, from its name. */
export function importType(fileName: string): string {
  if (/\.pls$|\.xml$/i.test(fileName)) return 'application/pls+xml'
  if (/\.json$/i.test(fileName)) return 'application/json'
  return 'text/tab-separated-values'
}
