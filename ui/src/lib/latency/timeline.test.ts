import { describe, expect, it } from 'vitest'
import { AudioClock, layoutTimeline, newUtterance, utteranceLatency, type Utterance } from './timeline'

function utt(p: Partial<Utterance>): Utterance {
  return { ...newUtterance(1), ...p }
}

describe('utteranceLatency', () => {
  it('measures first partial from speech start and final from speech end', () => {
    const l = utteranceLatency(utt({ speechStartAt: 1000, firstPartialAt: 1400, speechEndAt: 3000, finalAt: 3250 }))
    expect(l).toEqual({ firstPartialMs: 400, finalMs: 250, endToFirstPartialMs: -1600, speechMs: 2000 })
  })
  it('returns nulls for what is not known yet', () => {
    const l = utteranceLatency(utt({ speechStartAt: 1000 }))
    expect(l.finalMs).toBeNull()
    expect(l.firstPartialMs).toBeNull()
  })
  it('never reports a negative latency', () => {
    expect(utteranceLatency(utt({ speechEndAt: 3000, finalAt: 2990 })).finalMs).toBe(0)
  })
})

describe('layoutTimeline', () => {
  const u = utt({ speechStartAt: 0, firstPartialAt: 500, speechEndAt: 4000, finalAt: 4300 })
  it('orders markers in time and labels the gaps in ms', () => {
    const l = layoutTimeline(u)
    expect(l?.markers.map((m) => m.kind)).toEqual(['speech_start', 'first_partial', 'speech_end', 'final'])
    expect(l?.gaps.map((g) => g.ms)).toEqual([500, 3500, 300])
    expect(l?.spanMs).toBe(4300)
  })
  it('compresses long speech so the tail keeps most of the axis', () => {
    const l = layoutTimeline(u, 0.45)
    const end = l?.markers.find((m) => m.kind === 'speech_end')
    const fin = l?.markers.find((m) => m.kind === 'final')
    expect(end?.pos).toBeCloseTo(0.45)
    expect(fin?.pos).toBeCloseTo(1)
    expect(l?.speech).toEqual({ pos0: 0, pos1: end?.pos })
  })
  it('keeps positions monotonic and inside 0..1', () => {
    const l = layoutTimeline(utt({ speechStartAt: 10, firstPartialAt: 2500, speechEndAt: 2000, finalAt: 2600 }))
    const pos = l?.markers.map((m) => m.pos) ?? []
    expect([...pos].sort((a, b) => a - b)).toEqual(pos)
    expect(Math.min(...pos)).toBeGreaterThanOrEqual(0)
    expect(Math.max(...pos)).toBeLessThanOrEqual(1)
    expect(l?.markers.map((m) => m.kind)).toEqual(['speech_start', 'speech_end', 'first_partial', 'final'])
  })
  it('does not stretch a tiny tail across the whole axis', () => {
    const l = layoutTimeline(utt({ speechStartAt: 0, speechEndAt: 1000, finalAt: 1010 }), 0.45, 100)
    const fin = l?.markers.find((m) => m.kind === 'final')
    expect(fin?.pos).toBeLessThan(0.6)
  })
  it('needs at least two events', () => {
    expect(layoutTimeline(utt({ speechStartAt: 5 }))).toBeNull()
  })
})

describe('AudioClock', () => {
  it('maps audio time to the moment that audio was sent', () => {
    const c = new AudioClock()
    c.mark(0.02, 1000) // audio 0.00–0.02 sent at t=1000
    c.mark(0.02, 1020)
    c.mark(0.02, 1041)
    expect(c.sentSeconds).toBeCloseTo(0.06)
    expect(c.wallTimeOf(0.01)).toBe(1000)
    expect(c.wallTimeOf(0.03)).toBe(1020)
    expect(c.wallTimeOf(0.06)).toBe(1041)
  })
  it('extrapolates in real time past the last frame and is null when empty', () => {
    const c = new AudioClock()
    expect(c.wallTimeOf(1)).toBeNull()
    c.mark(1, 5000)
    expect(c.wallTimeOf(1.25)).toBe(5250)
  })
  it('follows bursts: audio sent faster than real time keeps its send time', () => {
    const c = new AudioClock()
    for (let i = 0; i < 100; i++) c.mark(0.02, 2000 + i) // 2 s of audio in 100 ms
    expect(c.wallTimeOf(1.0)).toBe(2049)
  })
})
