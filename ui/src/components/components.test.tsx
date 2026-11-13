import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { BudgetBar } from '@/components/charts/budget-bar'
import { LatencyTimeline, UtteranceList } from '@/components/charts/latency-timeline'
import { SegmentedBar } from '@/components/charts/segmented-bar'
import { StatTile } from '@/components/charts/stat-tile'
import { RequestDrawer, Waterfall } from '@/components/latency/waterfall'
import { Transcript } from '@/components/stt/transcript'
import { ChunkTimeline } from '@/components/tts/chunk-timeline'
import { TooltipProvider } from '@/components/ui/tooltip'
import type { RequestTrace } from '@/lib/api/types'
import { newUtterance } from '@/lib/latency/timeline'
import { emptyTranscript } from '@/lib/stt/transcriber'

const wrap = (ui: ReactNode) => render(<TooltipProvider>{ui}</TooltipProvider>)

describe('BudgetBar', () => {
  it('reports state in words and as a meter, not by colour alone', () => {
    wrap(<BudgetBar measuredMs={250} targetMs={300} />)
    expect(screen.getByText('Within budget')).toBeInTheDocument()
    const meter = screen.getByRole('meter')
    expect(meter).toHaveAttribute('aria-valuenow', '250')
    expect(meter).toHaveAttribute('aria-valuetext', '250 ms, target 300 ms')
    expect(screen.getByTestId('budget-bar')).toHaveAttribute('data-state', 'good')
  })
  it('turns to warning, then bad, and shows the overshoot', () => {
    const { rerender } = wrap(<BudgetBar measuredMs={400} targetMs={300} />)
    expect(screen.getByText('Over budget')).toBeInTheDocument()
    expect(screen.getByText(/\+\s*100 ms/)).toBeInTheDocument()
    rerender(
      <TooltipProvider>
        <BudgetBar measuredMs={900} targetMs={300} />
      </TooltipProvider>,
    )
    expect(screen.getByText('Well over budget')).toBeInTheDocument()
    expect(screen.getByTestId('budget-bar')).toHaveAttribute('data-state', 'bad')
  })
  it('has an empty state', () => {
    wrap(<BudgetBar measuredMs={null} targetMs={300} />)
    expect(screen.getByText('No measurement yet')).toBeInTheDocument()
    expect(screen.getByRole('meter')).toHaveAttribute('aria-valuetext', 'no measurement')
  })
})

describe('LatencyTimeline', () => {
  it('labels the gaps and describes itself for assistive tech', () => {
    const u = {
      ...newUtterance(1),
      speechStartAt: 0,
      firstPartialAt: 420,
      speechEndAt: 2000,
      finalAt: 2260,
      finalServerMs: 180,
      text: 'hi',
    }
    wrap(<LatencyTimeline utterance={u} targetMs={300} />)
    const img = screen.getByRole('img')
    expect(img).toHaveAccessibleName(/first partial 420 ms after speech started; final 260 ms after speech ended/)
    expect(screen.getAllByText('260 ms').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('420 ms')).toBeInTheDocument()
    expect(screen.getByText('180 ms')).toBeInTheDocument()
    expect(screen.getByLabelText(/Final at/)).toBeInTheDocument()
  })
  it('shows guidance before anything was said', () => {
    wrap(<LatencyTimeline utterance={null} />)
    expect(screen.getByText(/Speak, or play a sample/)).toBeInTheDocument()
  })
  it('lists finished utterances newest first', () => {
    const done = (id: number, final: number, text: string) => ({
      ...newUtterance(id),
      speechStartAt: 0,
      speechEndAt: 1000,
      finalAt: 1000 + final,
      text,
    })
    wrap(
      <UtteranceList
        utterances={[done(1, 200, 'first one'), done(2, 450, 'second one'), newUtterance(3)]}
        targetMs={300}
      />,
    )
    const rows = screen.getAllByRole('listitem')
    expect(rows).toHaveLength(2)
    expect(within(rows[0]!).getByText('second one')).toBeInTheDocument()
    expect(within(rows[0]!).getByRole('img')).toHaveAccessibleName(/450 ms.*over target/)
    expect(within(rows[1]!).getByText('200')).toBeInTheDocument()
  })
})

describe('Transcript', () => {
  it('separates final text, the stable part of a partial and the part that may change', () => {
    const state = {
      ...emptyTranscript(),
      status: 'streaming' as const,
      finals: [
        {
          text: 'Hello there.',
          t0: 0,
          t1: 1,
          words: [
            { w: 'Hello', t0: 0.1, t1: 0.4 },
            { w: 'there', t0: 0.5, t1: 0.9 },
          ],
          language: 'en',
          latencyMs: 100,
        },
      ],
      partial: { text: 'how are yo', stableChars: 7 },
    }
    wrap(<Transcript state={state} placeholder="nothing yet" />)
    expect(screen.getByTestId('final-text')).toHaveTextContent('Hello there.')
    const partial = screen.getByTestId('partial-text')
    expect(partial.children[0]).toHaveTextContent('how are')
    expect(partial.children[1]).toHaveTextContent('yo')
    expect(partial.children[1]).toHaveClass('italic')
    expect(screen.getByRole('log')).toBeInTheDocument()
  })
  it('reveals word timestamps on focus', async () => {
    const state = {
      ...emptyTranscript(),
      finals: [{ text: 'Hi.', t0: 0, t1: 1, words: [{ w: 'Hi', t0: 0.12, t1: 0.4 }], language: 'en', latencyMs: 1 }],
    }
    wrap(<Transcript state={state} placeholder="x" />)
    await userEvent.tab()
    expect((await screen.findAllByText('0.12–0.40 s')).length).toBeGreaterThan(0)
  })
  it('shows the placeholder, or Listening while live', () => {
    const { rerender } = wrap(<Transcript state={null} placeholder="nothing yet" />)
    expect(screen.getByText('nothing yet')).toBeInTheDocument()
    rerender(
      <TooltipProvider>
        <Transcript state={{ ...emptyTranscript(), status: 'streaming' }} placeholder="nothing yet" />
      </TooltipProvider>,
    )
    expect(screen.getByText('Listening…')).toBeInTheDocument()
  })
})

describe('SegmentedBar and StatTile', () => {
  it('lists every part with its value and share', () => {
    wrap(
      <SegmentedBar
        ariaLabel="Latency split"
        segments={[
          { key: 'a', label: 'Model', value: 300, color: 'red', display: '300 ms' },
          { key: 'b', label: 'Network', value: 100, color: 'blue', display: '100 ms' },
        ]}
      />,
    )
    expect(screen.getByRole('img')).toHaveAccessibleName('Latency split: Model 300 ms (75%), Network 100 ms (25%)')
    expect(screen.getByText('75%')).toBeInTheDocument()
    expect(screen.getByText('100 ms')).toBeInTheDocument()
  })
  it('renders a stat tile with unit, context and a status word for screen readers', () => {
    render(
      <StatTile
        label="First partial p95"
        value="212"
        unit="ms"
        status="good"
        context="target 300 ms"
        trend={[1, 2, 3]}
      />,
    )
    expect(screen.getByText('212')).toBeInTheDocument()
    expect(screen.getByText('within target:')).toHaveClass('sr-only')
    expect(screen.getByText('target 300 ms')).toBeInTheDocument()
  })
})

const trace: RequestTrace = {
  id: 'r_42',
  kind: 'stt',
  transport: 'websocket',
  model: 'moonshine-tiny-en',
  started_at: '2026-11-13T10:00:00.000Z',
  status: 'ok',
  error: null,
  audio_seconds: 4.2,
  chars: null,
  language: 'en',
  phases: [
    { name: 'queue', start_ms: 0, dur_ms: 3 },
    { name: 'vad', start_ms: 3, dur_ms: 12 },
    { name: 'decode', start_ms: 15, dur_ms: 210 },
    { name: 'finalize', start_ms: 225, dur_ms: 20 },
  ],
  first_partial_ms: 180,
  final_ms: 245,
  ttfa_ms: null,
  total_ms: 245,
  rtf: 0.058,
  batch_size: 2,
  text_preview: 'hello and welcome',
}

describe('Waterfall', () => {
  it('renders a keyboard-reachable row per request and reports clicks', async () => {
    const onSelect = vi.fn()
    wrap(
      <Waterfall
        traces={[trace, { ...trace, id: 'r_43', status: 'error', error: 'boom' }]}
        onSelect={onSelect}
        selectedId={null}
        kind="stt"
      />,
    )
    const rows = screen.getAllByRole('button')
    expect(rows).toHaveLength(2)
    expect(rows[0]).toHaveAccessibleName(/Request r_42, started\s*10:00:00.*moonshine-tiny-en.*245/)
    expect(rows[1]).toHaveAccessibleName(/failed/)
    expect(screen.getByText('Decode')).toBeInTheDocument()
    await userEvent.click(rows[0]!)
    expect(onSelect).toHaveBeenCalledWith(trace)
  })
  it('has an empty state', () => {
    wrap(<Waterfall traces={[]} onSelect={() => undefined} selectedId={null} kind="stt" />)
    expect(screen.getByText('No requests in this window')).toBeInTheDocument()
  })
  it('breaks a request down by phase in the drawer', () => {
    wrap(<RequestDrawer trace={trace} onClose={() => undefined} />)
    const drawer = screen.getByTestId('request-drawer')
    expect(within(drawer).getByText('r_42')).toBeInTheDocument()
    const rows = within(drawer).getAllByRole('row')
    expect(rows.some((r) => /Decode.*15 ms.*210 ms.*86%/.test(r.textContent ?? ''))).toBe(true)
    expect(within(drawer).getByText('hello and welcome')).toBeInTheDocument()
  })
})

describe('ChunkTimeline', () => {
  it('shows headroom per chunk and flags a late one', () => {
    wrap(
      <ChunkTimeline
        chunks={[
          { seq: 0, text: 'Hello there.', t0: 0, dur: 1, arrivedMs: 120, playsAtMs: 150, lateMs: 0 },
          { seq: 1, text: 'Second sentence.', t0: 1, dur: 1.5, arrivedMs: 1400, playsAtMs: 1150, lateMs: 250 },
        ]}
      />,
    )
    expect(screen.getByText('+30')).toBeInTheDocument()
    expect(screen.getByText(/−250/)).toBeInTheDocument()
    expect(screen.getByLabelText(/Chunk 2: .*playback waited 250 ms/)).toBeInTheDocument()
  })
})
