import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { audioConstraints, BROWSER_PROCESSING } from '@/lib/audio/capture'
import { sessionUpdate } from '@/lib/realtime/session'
import { CaptureControl } from './capture-control'

describe('browser microphone processing in a conversation', () => {
  it('builds the getUserMedia constraints, all on by default', () => {
    expect(audioConstraints()).toEqual({
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    })
    expect(audioConstraints({ ...BROWSER_PROCESSING, noiseSuppression: false }).noiseSuppression).toBe(false)
  })

  it('switches each one, and warns when nothing cancels the echo', async () => {
    const onChange = vi.fn()
    const { rerender } = render(<CaptureControl value={BROWSER_PROCESSING} onChange={onChange} serverEcho={false} />)
    expect(screen.queryByTestId('capture-no-aec')).toBeNull()
    await userEvent.click(screen.getByTestId('capture-ns'))
    expect(onChange).toHaveBeenCalledWith({ ...BROWSER_PROCESSING, noiseSuppression: false })
    const off = { ...BROWSER_PROCESSING, echoCancellation: false }
    rerender(<CaptureControl value={off} onChange={onChange} serverEcho={false} />)
    expect(screen.getByTestId('capture-no-aec').textContent).toContain('headphones')
    rerender(<CaptureControl value={off} onChange={onChange} serverEcho />) // the server's canceller takes over
    expect(screen.queryByTestId('capture-no-aec')).toBeNull()
  })

  it('can turn the server’s closer listening while the agent speaks off', () => {
    const td = (bargeIn?: boolean) =>
      (
        sessionUpdate('cascade', undefined, 'websocket', undefined, { type: 'server_vad', bargeIn }).session.audio as {
          input: { turn_detection: Record<string, unknown> }
        }
      ).input.turn_detection
    expect(td()).toEqual({ type: 'server_vad' })
    expect(td(false)).toEqual({ type: 'server_vad', barge_in_threshold: 0 })
  })
})
