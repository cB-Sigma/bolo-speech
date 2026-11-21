import { describe, expect, it } from 'vitest'
import { answerText, buildQuestions } from './decisions'

describe('decision questions', () => {
  it('builds the request from the typed questions', () => {
    expect(
      buildQuestions([
        { type: 'yes_no', question: ' Polite? ', answers: 'ignored' },
        { type: 'choice', question: 'Topic?', answers: 'delivery\n\nbilling\n' },
        { type: 'score', question: 'Urgency?', answers: 'low\nhigh' },
      ]),
    ).toEqual({
      q1: { type: 'yes_no', question: 'Polite?' },
      q2: { type: 'choice', question: 'Topic?', options: ['delivery', 'billing'] },
      q3: { type: 'score', question: 'Urgency?', levels: ['low', 'high'] },
    })
  })

  it('says what is missing', () => {
    expect(buildQuestions([{ type: 'yes_no', question: ' ', answers: '' }])).toBe('Question 1 is empty')
    expect(buildQuestions([{ type: 'choice', question: 'x', answers: 'one' }])).toMatch(/two options/)
    const eleven = Array.from({ length: 11 }, (_, i) => String(i)).join('\n')
    expect(buildQuestions([{ type: 'score', question: 'x', answers: eleven }])).toMatch(/at most 10/)
  })

  it('words each answer type', () => {
    expect(answerText({ type: 'yes_no', answer: false, probability: 0.24 })).toBe('no (probability of yes 0.24)')
    expect(
      answerText({ type: 'choice', answer: 'delivery', confidence: 0.98, probabilities: { delivery: 0.99 } }),
    ).toBe('delivery (0.99)')
    expect(
      answerText({
        type: 'score',
        score: 1.51,
        level: 2,
        label: 'Right now',
        confidence: 0.3,
        probabilities: [0.1, 0.3, 0.6],
      }),
    ).toBe('Right now (level 1.51 of 0–2)')
  })
})
