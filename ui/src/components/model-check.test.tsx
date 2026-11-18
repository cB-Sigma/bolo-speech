import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import { ModelCheckButton } from '@/components/model-check'
import { setServerBase, setTransport } from '@/lib/api/config'

afterEach(() => {
  cleanup()
  setServerBase(null)
})

describe('ModelCheckButton', () => {
  it('runs bolo check on the server and lists its steps, with tiny when asked', async () => {
    const bodies: unknown[] = []
    setServerBase('http://bolo.test/')
    setTransport({
      fetch: async (input, init) => {
        expect(String(input)).toBe('http://bolo.test/api/v1/models/moshiko-pytorch-q8/check')
        bodies.push(JSON.parse(String(init?.body ?? '{}')))
        const steps = [
          { name: 'weights', status: 'ok', detail: '4 files there at their pinned revision (8.39 GB)' },
          {
            name: 'placement',
            status: 'fail',
            detail: 'refused here: moshiko-pytorch-q8 needs about 12.0 GB of GPU memory; RTX has 6.0 GB',
          },
        ]
        return new Response(JSON.stringify({ model: 'moshiko-pytorch-q8', ok: false, tiny: false, steps }), {
          headers: { 'content-type': 'application/json' },
        })
      },
      WebSocket: undefined as never,
    })
    render(<ModelCheckButton id="moshiko-pytorch-q8" />)
    await userEvent.click(screen.getByTestId('check-moshiko-pytorch-q8'))
    const dialog = screen.getByTestId('check-dialog')
    await userEvent.click(within(dialog).getByRole('switch', { name: 'Build it tiny' }))
    await userEvent.click(within(dialog).getByTestId('check-run'))
    const report = await screen.findByTestId('check-report')
    expect(within(report).getByText('Something would fail:')).toBeTruthy()
    expect(within(report).getByText(/needs about 12\.0 GB of GPU memory/)).toBeTruthy()
    await waitFor(() => expect(bodies).toEqual([{ tiny: true }]))
  })
})
