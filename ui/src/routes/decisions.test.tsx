import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { AudioDecision } from '@/lib/api/types'

const answer: AudioDecision = {
  object: 'audio.decision',
  model: 'd1-omni-600m-llamacpp',
  answers: {
    q1: { type: 'yes_no', answer: true, probability: 0.93 },
    q2: { type: 'choice', answer: 'billing', confidence: 0.9, probabilities: { billing: 0.95, sales: 0.05 } },
  },
}
const audioDecisions = vi.fn(async () => answer)

vi.mock('@/hooks/use-api', () => ({
  useInfo: () => ({ data: { capabilities: { decision: { default_model: 'd1-omni-600m-llamacpp' } } } }),
  useModels: () => ({ models: [] }),
  describeError: (e: unknown) => String(e),
}))
vi.mock('@/lib/api/client', () => ({ api: { audioDecisions } }))
vi.mock('@/components/catalog-select', () => ({ CatalogSelect: () => <div /> }))

const { default: DecisionsPage } = await import('./decisions')

describe('Decisions page', () => {
  it('asks the typed questions about a text and shows the answers', async () => {
    const user = userEvent.setup()
    render(<DecisionsPage />)
    await user.type(screen.getByLabelText('Text'), 'I was charged twice')
    await user.click(screen.getByRole('button', { name: /Question/ }))
    await user.type(screen.getByLabelText('Question 2'), 'Which team?')
    await user.click(screen.getByLabelText('Answer type of question 2'))
    await user.click(await screen.findByRole('option', { name: 'One of these' }))
    await user.type(screen.getByLabelText('Options of question 2'), 'billing{enter}sales')
    await user.click(screen.getByTestId('decision-ask'))
    expect(audioDecisions).toHaveBeenCalledWith(
      {
        q1: { type: 'yes_no', question: 'Is the speaker asking for help?' },
        q2: { type: 'choice', question: 'Which team?', options: ['billing', 'sales'] },
      },
      { file: undefined, name: undefined, text: 'I was charged twice', model: undefined },
    )
    const answers = await screen.findByTestId('decision-answers')
    expect(answers).toHaveTextContent('yes (probability of yes 0.93)')
    expect(answers).toHaveTextContent('billing (0.95)')
    expect(answers).toHaveTextContent('d1-omni-600m-llamacpp')
  })

  it('says what is missing before asking', async () => {
    const user = userEvent.setup()
    audioDecisions.mockClear()
    render(<DecisionsPage />)
    await user.click(screen.getByTestId('decision-ask'))
    expect(screen.getByTestId('decision-error')).toHaveTextContent('Give a recording, a text, or both')
    expect(audioDecisions).not.toHaveBeenCalled()
  })
})
