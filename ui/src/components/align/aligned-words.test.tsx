import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { activeWord, AlignedWords, wordsToVtt } from './aligned-words'

const words = [
  { word: 'Good', start: 0.1, end: 0.3, score: 0.9 },
  { word: 'morning.', start: 0.4, end: 0.8, score: 0.3 },
  { word: 'Bye', start: 2.0, end: 2.2, score: null },
]

describe('aligned words', () => {
  it('finds the word being spoken', () => {
    expect(activeWord(words, null)).toBe(-1)
    expect(activeWord(words, 0.05)).toBe(-1)
    expect(activeWord(words, 0.2)).toBe(0)
    expect(activeWord(words, 0.35)).toBe(0) // between two words close together: the last one stays lit
    expect(activeWord(words, 1.5)).toBe(-1) // a pause
    expect(activeWord(words, 2.1)).toBe(2)
  })

  it('highlights the current word and marks doubtful ones', () => {
    render(
      <TooltipProvider>
        <AlignedWords words={words} time={0.5} />
      </TooltipProvider>,
    )
    const shown = screen.getAllByTestId('aligned-word')
    expect(shown.map((w) => w.textContent)).toEqual(['Good', 'morning.', 'Bye'])
    expect(shown[1]?.getAttribute('aria-current')).toBe('true')
    expect(shown[1]?.className).toContain('decoration-dotted')
    expect(shown[0]?.className).not.toContain('decoration-dotted')
  })

  it('writes WebVTT', () => {
    expect(wordsToVtt(words.slice(0, 1))).toBe('WEBVTT\n\n00:00:00.100 --> 00:00:00.300\nGood\n')
  })
})
