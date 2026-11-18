/**
 * The parts of Python the port of dizione relies on: its regular expressions (Unicode `\w`, `\d` and `\b`, named
 * groups, `re.sub` with a function, `re.match` and `re.fullmatch`) and its string predicates (`str.isalpha` and the
 * like), with Python's meaning, so that the TypeScript reads the same text the same way.
 */

const WORD = '\\p{L}\\p{N}_'
const DIGIT = '\\p{Nd}'
const BOUNDARY = `(?:(?<=[${WORD}])(?![${WORD}])|(?<![${WORD}])(?=[${WORD}]))`
const SYNTAX = new Set('^$\\.*+?()[]{}|/-')

/** A Python pattern as a JavaScript one (flag `u`): `\w`, `\d` and `\b` are Unicode-aware, as in Python. */
export function translate(pattern: string): string {
  let out = ''
  let inClass = false
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i]!
    if (c === '\\') {
      const n = pattern[++i] ?? ''
      if ((n === 'W' || n === 'D') && inClass) throw new Error(`\\${n} in a character class: ${pattern}`)
      if (n === 'w') out += inClass ? WORD : `[${WORD}]`
      else if (n === 'W') out += `[^${WORD}]`
      else if (n === 'd') out += DIGIT
      else if (n === 'D') out += `[^${DIGIT}]`
      else if (n === 'b' && !inClass) out += BOUNDARY
      else if ('sSntrfv0'.includes(n) || /[1-9]/.test(n)) out += `\\${n}`
      else if (n === 'u' || n === 'x') out += `\\${n}`
      else if (SYNTAX.has(n)) out += n === '-' && !inClass ? '-' : `\\${n}`
      else out += n
      continue
    }
    if (inClass) {
      if (c === ']') inClass = false
      // after a class escape ("[\\w-.]") a hyphen is itself, as in Python
      if (c === '-' && pattern[i - 2] === '\\' && 'wdsS'.includes(pattern[i - 1]!) && pattern[i + 1] !== ']')
        out += '\\-'
      else out += c
      continue
    }
    if (c === '[') {
      inClass = true
      out += c
      if (pattern[i + 1] === '^') out += pattern[++i]
      if (pattern[i + 1] === ']') out += `\\${pattern[++i]}` // "[]...]": a literal bracket first
      continue
    }
    if (c === '(' && pattern.startsWith('(?P<', i)) {
      out += '(?<'
      i += 3
      continue
    }
    if (c === '(' && pattern.startsWith('(?P=', i)) {
      const end = pattern.indexOf(')', i)
      out += `\\k<${pattern.slice(i + 4, end)}>`
      i = end
      continue
    }
    out += c
  }
  return out
}

const cache = new Map<string, RegExp>()

/** A compiled Python pattern; `flags` is `''` or `'i'` (re.IGNORECASE). */
export function re(pattern: string, flags = ''): RegExp {
  const key = `${flags}/${pattern}`
  let found = cache.get(key)
  if (!found) {
    found = new RegExp(translate(pattern), `gud${flags}`)
    cache.set(key, found)
  }
  return found
}

/** A match with the questions Python's match object answers. */
export class Match {
  private readonly m: RegExpExecArray
  readonly string: string
  constructor(m: RegExpExecArray, string: string) {
    this.m = m
    this.string = string
  }
  group(which: number | string = 0): string | undefined {
    return typeof which === 'number' ? this.m[which] : this.m.groups?.[which]
  }
  groups(): (string | undefined)[] {
    return this.m.slice(1)
  }
  private span(which: number | string): [number, number] {
    const indices = this.m.indices!
    const found = typeof which === 'number' ? indices[which] : indices.groups?.[which]
    return found ?? [-1, -1]
  }
  start(which: number | string = 0): number {
    return this.span(which)[0]
  }
  end(which: number | string = 0): number {
    return this.span(which)[1]
  }
}

/** Every match, left to right, as `re.finditer`. */
export function* finditer(pattern: RegExp, text: string, from = 0): Generator<Match> {
  pattern.lastIndex = from
  let previousEnd = -1
  for (;;) {
    const m = pattern.exec(text)
    if (!m) return
    const start = m.index
    const end = start + m[0].length
    if (start === end) {
      // Python skips an empty match right after the previous match ends
      if (start === previousEnd) {
        if (start >= text.length) return
        pattern.lastIndex = start + (text.codePointAt(start)! > 0xffff ? 2 : 1)
        continue
      }
      pattern.lastIndex = start + (start < text.length && text.codePointAt(start)! > 0xffff ? 2 : 1)
    }
    previousEnd = end
    yield new Match(m, text)
    if (start === end && start >= text.length) return
  }
}

/** `re.sub`: `replace` is a function of the match, or a template with `\1` and `\g<name>`. */
export function sub(pattern: RegExp, replace: string | ((m: Match) => string), text: string): string {
  let out = ''
  let last = 0
  for (const m of finditer(pattern, text)) {
    out += text.slice(last, m.start())
    out += typeof replace === 'function' ? replace(m) : expand(replace, m)
    last = m.end()
  }
  return out + text.slice(last)
}

function expand(template: string, m: Match): string {
  return template.replace(/\\(\d+)|\\g<(\w+)>/g, (_, n: string | undefined, name: string | undefined) =>
    n !== undefined ? (m.group(Number(n)) ?? '') : (m.group(/^\d+$/.test(name!) ? Number(name) : name!) ?? ''),
  )
}

/** `re.search`: the first match anywhere, or null. */
export function search(pattern: RegExp, text: string): Match | null {
  for (const m of finditer(pattern, text)) return m
  return null
}

const anchored = new Map<RegExp, RegExp>()

/** `re.match`: a match at the start of `text`, or null. */
export function match(pattern: RegExp, text: string): Match | null {
  let sticky = anchored.get(pattern)
  if (!sticky) {
    sticky = new RegExp(pattern.source, pattern.flags.replace('g', '') + 'y')
    anchored.set(pattern, sticky)
  }
  sticky.lastIndex = 0
  const m = sticky.exec(text)
  return m ? new Match(m, text) : null
}

const whole = new Map<RegExp, RegExp>()

/** `re.fullmatch`: true when the pattern spells all of `text`. */
export function fullmatch(pattern: RegExp, text: string): boolean {
  let full = whole.get(pattern)
  if (!full) {
    full = new RegExp(`^(?:${pattern.source})$`, pattern.flags.replace('g', ''))
    whole.set(pattern, full)
  }
  return full.test(text)
}

/** `re.findall`: the whole matches, the one group, or the tuples of groups (as arrays). */
export function findall(pattern: RegExp, text: string): string[] {
  return [...finditer(pattern, text)].map((m) => {
    const groups = m.groups()
    return groups.length === 0 ? m.group(0)! : (groups[0] ?? '')
  })
}

/** `re.findall` for a pattern of several groups. */
export function findallGroups(pattern: RegExp, text: string): string[][] {
  return [...finditer(pattern, text)].map((m) => m.groups().map((g) => g ?? ''))
}

/** `re.escape`, for use outside a character class. */
export function escape(text: string): string {
  return text.replace(/[\\^$.*+?()[\]{}|/]/g, '\\$&')
}

// ---- str predicates, with Python's meaning ---------------------------------

const ALPHA = /^[\p{L}]+$/u
const ALNUM = /^[\p{L}\p{N}]+$/u
const DIGITS = /^[\p{Nd}\u00b2\u00b3\u00b9\u2070\u2074-\u2079\u2080-\u2089]+$/u
const CASED_UPPER = /[\p{Lu}\p{Lt}]/u
const CASED_LOWER = /\p{Ll}/u

export const isalpha = (s: string): boolean => ALPHA.test(s)
export const isalnum = (s: string): boolean => ALNUM.test(s)
/** `str.isdigit`: decimal digits and the other digits ("²"). */
export const isdigit = (s: string): boolean => DIGITS.test(s)
// Python's whitespace includes the separators \x1c to \x1f
// eslint-disable-next-line no-control-regex
export const isspace = (s: string): boolean => s.length > 0 && /^[\s\x1c-\x1f\x85]+$/u.test(s) && !s.includes('﻿')
/** `str.isupper`: at least one cased character and no lower-case one. */
export const isupper = (s: string): boolean => CASED_UPPER.test(s) && !CASED_LOWER.test(s)
export const islower = (s: string): boolean => CASED_LOWER.test(s) && !CASED_UPPER.test(s)
// eslint-disable-next-line no-control-regex
export const isascii = (s: string): boolean => /^[\x00-\x7f]*$/.test(s)

/** `str.split()` without arguments: runs of whitespace, no empty pieces. */
export function split(s: string): string[] {
  // eslint-disable-next-line no-control-regex
  return s.split(/[\s\x1c-\x1f\x85]+/u).filter((piece) => piece && piece !== '﻿')
}

/** `str.strip(chars)`. */
export function strip(s: string, chars: string): string {
  let a = 0
  let b = s.length
  while (a < b && chars.includes(s[a]!)) a++
  while (b > a && chars.includes(s[b - 1]!)) b--
  return s.slice(a, b)
}

export const rstrip = (s: string, chars: string): string => {
  let b = s.length
  while (b > 0 && chars.includes(s[b - 1]!)) b--
  return s.slice(0, b)
}

/** `str.lower` (JavaScript's lower case differs from Python's only for a few characters outside English). */
export const lower = (s: string): string => s.toLowerCase()
export const upper = (s: string): string => s.toUpperCase()

/** Python's `x in "abc"` for a string `x`: true for the empty string too. */
export const inStr = (x: string, s: string): boolean => s.includes(x)

/** A key for maps keyed by tuples of strings. */
export const key = (parts: readonly string[]): string => parts.join('\u0000')

/** Compare by code points, as Python sorts strings. */
export function codePointCompare(a: string, b: string): number {
  const x = [...a]
  const y = [...b]
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const d = x[i]!.codePointAt(0)! - y[i]!.codePointAt(0)!
    if (d) return d
  }
  return x.length - y.length
}
