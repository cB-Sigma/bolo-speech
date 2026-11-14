import { describe, expect, it } from 'vitest'
import { emptyRow, importType, lineOf, rowForLine, rowProblem, splitOverrides, toRows, toTsv } from './wordlists'

describe('word-list rows', () => {
  it('round-trips entries to TSV and maps server lines back to rows', () => {
    const rows = toRows([
      { word: 'bolo', lang: 'en-us', ipa: 'ˈboʊloʊ', say: null, match: 'word', note: 'product' },
      { word: 'MH12', lang: '*', ipa: null, say: 'em aitch\ttwelve', match: 'case', note: null },
    ])
    const { tsv, sent } = toTsv([...rows, emptyRow()])
    expect(tsv.split('\n')).toEqual([
      'word\tlang\tipa\tsay\tmatch\tnote',
      'bolo\ten-us\tˈboʊloʊ\t\tword\tproduct',
      'MH12\t*\t\tem aitch twelve\tcase\t',
      '',
    ])
    expect(sent).toHaveLength(2)
    expect(rowForLine(sent, 3)?.word).toBe('MH12')
    expect(rowForLine(sent, 1)).toBeNull()
  })
  it('flags rows the server would warn about', () => {
    expect(rowProblem({ ...emptyRow(), word: 'x' })).toMatch(/IPA/)
    expect(rowProblem({ ...emptyRow(), say: 'x' })).toMatch(/word/)
    expect(rowProblem({ ...emptyRow(), word: '(', say: 'x', match: 'regex' })).toMatch(/regular expression/)
    expect(rowProblem(emptyRow())).toBeNull()
  })
  it('reads line numbers and import types', () => {
    expect(lineOf('invalid word list: line 7: unknown match rule')).toBe(7)
    expect(lineOf('nope')).toBeNull()
    expect(importType('a.pls')).toBe('application/pls+xml')
    expect(importType('a.json')).toBe('application/json')
    expect(importType('a.tsv')).toBe('text/tab-separated-values')
  })
  it('splits [word](/ipa/) overrides out of prepared text', () => {
    expect(splitOverrides('Call [bolo](/ˈboʊloʊ/) now')).toEqual([
      { kind: 'text', text: 'Call ' },
      { kind: 'override', word: 'bolo', ipa: 'ˈboʊloʊ' },
      { kind: 'text', text: ' now' },
    ])
  })
})
