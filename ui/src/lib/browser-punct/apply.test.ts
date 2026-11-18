import { describe, expect, it } from 'vitest'
import { rewriteWords } from '@/lib/stt/browser'
import { browserPunctuator, speaks } from './apply'

describe('punctuation in the page', () => {
  it('runs only for English, or when the language is not known', () => {
    expect(speaks('en')).toBe(true)
    expect(speaks('en-IN')).toBe(true)
    expect(speaks(null)).toBe(true)
    expect(speaks('hi')).toBe(false)
  })

  it('leaves recognisers that write their own alone, and does nothing when turned off', async () => {
    const own = browserPunctuator(
      () => ({ noPunctuation: undefined }),
      () => true,
    )
    expect(await own('hello there', 'en')).toEqual({ text: 'hello there', ms: null })
    const off = browserPunctuator(
      () => ({ noPunctuation: true }),
      () => false,
    )
    expect(await off('hello there', 'en')).toEqual({ text: 'hello there', ms: null })
    const hindi = browserPunctuator(
      () => ({ noPunctuation: true }),
      () => true,
    )
    expect(await hindi('नमस्ते', 'hi')).toEqual({ text: 'नमस्ते', ms: null })
  })

  it('gives each timed word its punctuated spelling while the words still line up', () => {
    const words = [
      { w: 'hello', t0: 0, t1: 0.4 },
      { w: 'there', t0: 0.5, t1: 0.9 },
    ]
    expect(rewriteWords(words, 'Hello, there.').map((w) => w.w)).toEqual(['Hello,', 'there.'])
    expect(rewriteWords(words, 'Hello there friend')).toBe(words)
  })
})
