import { act, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { activeLoads, DONE_LINGER_MS, markLoading, markSettled } from '@/lib/downloads'
import { Store } from '@/lib/store'
import { DownloadTray } from './download-tray'

afterEach(() => {
  markSettled()
  activeLoads.set([])
  vi.useRealTimers()
})

function engine() {
  return new Store<{ status: string; progress: number }>({ status: 'loading', progress: 0 })
}

describe('downloads tray', () => {
  it('shows every model a page is fetching, in one place, until it is ready', () => {
    vi.useFakeTimers()
    const stt = engine()
    const vad = engine()
    render(<DownloadTray />)
    expect(screen.queryByTestId('download-tray')).toBeNull()

    // two features load at once (a conversation's recogniser and its voice-activity model): both are listed
    act(() => markLoading('stt', 'moonshine-tiny', { label: 'Speech to text: Moonshine', store: stt }))
    act(() => markLoading('audio', 'silero', { label: 'Voice activity: Silero', store: vad }))
    expect(screen.getByTestId('download-tray')).toHaveTextContent('Downloading 2 models into this browser')
    act(() => stt.patch({ progress: 0.42 }))
    expect(screen.getByTestId('download-moonshine-tiny')).toHaveTextContent('42 %')
    act(() => stt.patch({ progress: 0.1 })) // a later stage starting low: the figure holds
    expect(screen.getByTestId('download-moonshine-tiny')).toHaveTextContent('42 %')

    // ready: shown done for a moment, then gone; a failed load leaves at once (its feature says why)
    act(() => stt.patch({ status: 'ready', progress: 1 }))
    expect(screen.getByTestId('download-moonshine-tiny')).toHaveTextContent('ready')
    act(() => vad.patch({ status: 'error' }))
    expect(screen.queryByTestId('download-silero')).toBeNull()
    act(() => vi.advanceTimersByTime(DONE_LINGER_MS + 10))
    expect(screen.queryByTestId('download-tray')).toBeNull()
  })

  it('a model loaded again replaces its earlier row instead of adding one', () => {
    const a = engine()
    const b = engine()
    markLoading('tts', 'voice-1', { label: 'Voice 1', store: a })
    markLoading('tts', 'voice-1', { label: 'Voice 1', store: b })
    expect(activeLoads.get()).toHaveLength(1)
    a.patch({ progress: 0.9 }) // the first watcher was dropped
    expect(activeLoads.get()[0]!.progress).toBe(0)
  })
})
