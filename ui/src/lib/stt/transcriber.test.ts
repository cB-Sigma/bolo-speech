import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { utteranceLatency } from '@/lib/latency/timeline'
import { TranscriptTracker, transcriptText } from './transcriber'

describe('TranscriptTracker', () => {
  let now = 0
  beforeEach(() => {
    now = 10_000
    vi.spyOn(performance, 'now').mockImplementation(() => now)
  })
  afterEach(() => vi.restoreAllMocks())

  /** Send `seconds` of audio in 20 ms frames in real time. */
  function send(t: TranscriptTracker, seconds: number) {
    for (let i = 0; i < Math.round(seconds / 0.02); i++) {
      t.markSent(0.02)
      now += 20
    }
  }

  it('builds one utterance with the right latencies from a normal exchange', () => {
    const t = new TranscriptTracker()
    send(t, 0.5)
    t.speechStart(0.3) // audio at 0.3 s left at 10_280
    send(t, 0.4)
    now += 15
    t.partial('hello', 0.3, 120, 0)
    send(t, 1.0)
    t.partial('hello wor', 0.3, 110, 6)
    expect(t.store.get().partial).toEqual({ text: 'hello wor', stableChars: 6 })
    send(t, 0.5)
    t.speechEnd(1.9) // left at 11_880
    now += 240
    t.final({ text: 'Hello world.', t0: 0.3, t1: 1.9, words: null, language: 'en', latencyMs: 210 })

    const s = t.store.get()
    expect(s.partial).toBeNull()
    expect(s.finals).toHaveLength(1)
    expect(s.utterances).toHaveLength(1)
    expect(s.speech).toEqual([{ t0: 0.3, t1: 1.9 }])
    const u = s.utterances[0]!
    const l = utteranceLatency(u)
    expect(u.firstPartialServerMs).toBe(120)
    expect(u.finalServerMs).toBe(210)
    expect(l.firstPartialMs).toBeCloseTo(10_915 - 10_280, 0)
    expect(l.finalMs).toBeGreaterThan(240)
    expect(l.finalMs).toBeLessThan(900)
    expect(transcriptText(s)).toBe('Hello world.')
  })

  it('copes with servers that send no VAD events', () => {
    const t = new TranscriptTracker()
    send(t, 2)
    now += 50
    t.final({ text: 'no vad here', t0: 0.2, t1: 1.5, words: null, language: null, latencyMs: 80 })
    const u = t.store.get().utterances[0]!
    expect(u.speechStartAt).not.toBeNull()
    expect(u.speechEndAt).not.toBeNull()
    expect(u.firstPartialAt).toBe(u.finalAt)
    expect(utteranceLatency(u).finalMs).toBeGreaterThanOrEqual(0)
  })

  it('prefers the detector timestamp when a partial arrives before speech_start', () => {
    const t = new TranscriptTracker()
    send(t, 0.6)
    t.partial('the', 0, 15, 0)
    const guessed = t.store.get().utterances[0]!.speechStartAt
    t.speechStart(0.32)
    const u = t.store.get().utterances[0]!
    expect(u.speechStartAt).not.toBe(guessed)
    expect(u.speechStartAt).toBeLessThanOrEqual(u.firstPartialAt as number)
  })

  it('starts a new utterance for each segment and drops unfinished ones', () => {
    const t = new TranscriptTracker()
    send(t, 1)
    t.speechStart(0.1)
    t.speechEnd(0.8)
    t.final({ text: 'one', t0: 0.1, t1: 0.8, words: null, language: null, latencyMs: 50 })
    send(t, 1)
    t.speechStart(1.2)
    t.partial('tw', 1.2, 40, 0)
    expect(t.store.get().utterances).toHaveLength(2)
    t.abandon()
    const s = t.store.get()
    expect(s.utterances).toHaveLength(1)
    expect(s.partial).toBeNull()
    expect(s.utterances[0]?.id).toBe(1)
  })

  it('matches a late final to its own utterance when the next one has already started', () => {
    const t = new TranscriptTracker()
    send(t, 1)
    t.speechStart(0.1)
    t.speechEnd(0.9) // sent at 10_880
    send(t, 0.4)
    t.speechStart(1.2) // the speaker carries on before the first final is back
    t.partial('second', 1.2, 30, 0)
    now += 100
    t.final({ text: 'first', t0: 0.1, t1: 0.9, words: null, language: null, latencyMs: 90 })
    let s = t.store.get()
    expect(s.utterances.map((u) => [u.text, u.finalAt != null])).toEqual([
      ['first', true],
      ['second', false],
    ])
    expect(utteranceLatency(s.utterances[0]!).finalMs).toBeCloseTo(11_500 - 10_880, 0)
    expect(s.partial?.text).toBe('second')
    expect(s.speech).toEqual([
      { t0: 0.1, t1: 0.9 },
      { t0: 1.2, t1: null },
    ])
    send(t, 1)
    t.speechEnd(2.3)
    t.final({ text: 'second one', t0: 1.2, t1: 2.3, words: null, language: null, latencyMs: 80 })
    s = t.store.get()
    expect(s.utterances.every((u) => u.finalAt != null)).toBe(true)
    expect(s.speech[1]).toEqual({ t0: 1.2, t1: 2.3 })
    expect(s.partial).toBeNull()
  })

  it('ignores empty finals in the transcript but still closes the utterance', () => {
    const t = new TranscriptTracker()
    send(t, 1)
    t.speechStart(0.1)
    t.speechEnd(0.5)
    t.final({ text: '  ', t0: 0.1, t1: 0.5, words: null, language: null, latencyMs: 30 })
    expect(t.store.get().finals).toHaveLength(0)
    expect(t.store.get().utterances[0]?.finalAt).not.toBeNull()
  })

  it('continues the time axis after a reconnect', () => {
    const t = new TranscriptTracker()
    send(t, 3)
    t.timeOffset = t.clock.sentSeconds
    send(t, 1)
    t.speechStart(0.2)
    expect(t.store.get().speech[0]?.t0).toBeCloseTo(3.2)
  })
})
