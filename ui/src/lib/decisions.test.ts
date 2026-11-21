import { describe, expect, it } from 'vitest'
import { answerText, buildQuestions, hookConfig, hookForm } from '@/lib/decisions'

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

  it('builds the live-decision hook and reads it back', () => {
    const form = {
      enabled: true,
      model: 'fake-decision',
      questions: [
        { type: 'yes_no' as const, question: 'Escalate?', answers: '' },
        { type: 'choice' as const, question: 'Department?', answers: 'billing\ntechnical' },
      ],
      rules: [
        { question: 0, answer: 'yes', minProbability: '0.8', action: 'transfer' as const, value: '' },
        { question: 1, answer: 'billing', minProbability: '', action: 'tag' as const, value: 'billing' },
      ],
    }
    const hook = hookConfig(form)
    expect(hook).toEqual({
      model: 'fake-decision',
      questions: {
        q1: { type: 'yes_no', question: 'Escalate?' },
        q2: { type: 'choice', question: 'Department?', options: ['billing', 'technical'] },
      },
      actions: [
        { question: 'q1', do: 'transfer', answer: true, min_probability: 0.8 },
        { question: 'q2', do: 'tag', answer: 'billing', value: 'billing' },
      ],
    })
    expect(hookForm(hook as Parameters<typeof hookForm>[0])).toEqual(form)
    expect(hookConfig({ ...form, enabled: false })).toBeNull()
    expect(hookConfig({ ...form, rules: [{ ...form.rules[1]!, value: '' }] })).toMatch(/needs the tag/)
    expect(hookConfig({ ...form, rules: [{ ...form.rules[0]!, minProbability: '2' }] })).toMatch(/between 0 and 1/)
  })
})
