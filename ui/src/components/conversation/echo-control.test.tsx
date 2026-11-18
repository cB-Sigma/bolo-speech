import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { ServerInfo } from '@/lib/api/types'
import { sessionUpdate } from '@/lib/realtime/session'
import { defaultEcho, EchoControl, realtimeEcho } from './echo-control'

vi.mock('@/hooks/use-api', () => ({ useModels: () => ({ models: [] }) }))

const info = {
  capabilities: { realtime: { echo_cancellation: { default_model: 'speex-aec', install: null } } },
} as unknown as ServerInfo

describe('echo cancellation', () => {
  it('maps the choice to audio.input.echo_cancellation', () => {
    expect(realtimeEcho(defaultEcho)).toBeUndefined()
    expect(realtimeEcho({ mode: 'off', model: '' })).toBe(false)
    expect(realtimeEcho({ mode: 'on', model: '' })).toEqual({})
    expect(realtimeEcho({ mode: 'on', model: 'dtln-aec-128' })).toEqual({ model: 'dtln-aec-128' })
    const u = sessionUpdate('cascade', undefined, 'websocket', undefined, undefined, { echo: { model: 'x' } }) as {
      session: { audio: { input: Record<string, unknown> } }
    }
    expect(u.session.audio.input.echo_cancellation).toEqual({ model: 'x' })
    const none = sessionUpdate('cascade') as { session: { audio: { input: Record<string, unknown> } } }
    expect('echo_cancellation' in none.session.audio.input).toBe(false)
  })

  it('turns on, and says what the server uses', async () => {
    const onChange = vi.fn()
    const { rerender } = render(<EchoControl info={info} value={defaultEcho} onChange={onChange} />)
    await userEvent.click(screen.getByTestId('echo-on'))
    expect(onChange).toHaveBeenCalledWith({ mode: 'on', model: '' })
    rerender(
      <EchoControl
        info={info}
        value={{ mode: 'on', model: '' }}
        onChange={onChange}
        live
        reported={{ model: 'speex-aec' }}
      />,
    )
    expect(screen.getByTestId('server-echo')).toHaveTextContent('cancelling echo with speex-aec')
  })

  it('is absent on servers without it', () => {
    const { container } = render(<EchoControl info={undefined} value={defaultEcho} onChange={() => {}} />)
    expect(container).toBeEmptyDOMElement()
  })
})
