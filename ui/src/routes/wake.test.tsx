import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { CatalogModel } from '@/lib/api/types'

const models = [
  { id: 'kws', task: 'wake-word', custom_keywords: true, wake_phrases: ['HEY JARVIS'], installed: true },
  { id: 'oww', task: 'wake-word', custom_keywords: false, wake_phrases: ['alexa'], license_note: 'CC BY-NC-SA 4.0' },
] as unknown as CatalogModel[]

vi.mock('@/hooks/use-api', () => ({
  useInfo: () => ({ data: { capabilities: { wake: { default_model: 'kws' } } } }),
  useModels: () => ({ models }),
}))
vi.mock('@/components/catalog-select', () => ({
  CatalogSelect: ({ onChange, testId }: { onChange: (id: string) => void; testId: string }) => (
    <button data-testid={testId} onClick={() => onChange('oww')}>
      pick
    </button>
  ),
}))
vi.mock('@/components/layout/app-shell', () => ({
  Page: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

describe('Wake word page', () => {
  it('takes phrases for a keyword spotter, and names the fixed phrase and licence of the others', async () => {
    const { default: WakePage } = await import('./wake')
    render(<WakePage />)
    expect(screen.getByTestId('wake-keywords')).toBeInTheDocument()
    expect(screen.getByTestId('wake-phrases')).toHaveTextContent('hey jarvis')
    await userEvent.click(screen.getByTestId('wake-model'))
    expect(screen.queryByTestId('wake-keywords')).not.toBeInTheDocument()
    expect(screen.getByTestId('wake-phrases')).toHaveTextContent('Hears “alexa” only.')
    expect(screen.getByTestId('wake-phrases')).toHaveTextContent('CC BY-NC-SA 4.0')
    expect(screen.getByTestId('wake-start')).toBeEnabled()
  })
})
