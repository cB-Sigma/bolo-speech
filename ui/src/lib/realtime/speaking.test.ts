import { afterEach, describe, expect, it, vi } from 'vitest'
import { GaplessPlayer } from '@/lib/audio/player'
import { nextSpeaking, type SpeakingInput } from './speaking'
import type { RealtimeTransport } from './session'

const ev = (type: string): SpeakingInput => ({ kind: 'event', type })

/** Run inputs through the state machine; the indicator after each. */
function run(transport: RealtimeTransport, inputs: SpeakingInput[], fullDuplex = false): boolean[] {
  let on = false
  return inputs.map((i) => (on = nextSpeaking(on, i, transport, fullDuplex)))
}

describe('agent speaking indicator', () => {
  it('over WebSocket follows the page’s own playback, not response.done', () => {
    // the answer is generated in 1 s and played for 4 s: response.done arrives while it is still being heard
    expect(
      run('websocket', [
        ev('input_audio_buffer.speech_stopped'),
        ev('response.created'),
        ev('response.output_audio.delta'), // queued, not yet heard
        { kind: 'played' },
        ev('response.output_audio.done'),
        ev('response.done'),
        { kind: 'drained' },
      ]),
    ).toEqual([false, false, false, true, true, true, false])
  })

  it('over WebSocket goes off on barge-in and on Stop the answer, and on again with the next answer', () => {
    expect(
      run('websocket', [
        { kind: 'played' },
        ev('input_audio_buffer.speech_started'),
        ev('response.done'), // the cancelled answer's response.done
        { kind: 'played' }, // the next answer
        { kind: 'interrupted' }, // Stop the answer
        { kind: 'drained' },
      ]),
    ).toEqual([true, false, false, true, false, false])
  })

  it('over WebSocket ignores output_audio_buffer events (the server cannot know the page’s playback)', () => {
    expect(
      run('websocket', [ev('output_audio_buffer.started'), { kind: 'played' }, ev('output_audio_buffer.stopped')]),
    ).toEqual([false, true, true])
  })

  it('over WebRTC follows the server’s audio track', () => {
    expect(
      run('webrtc', [
        ev('response.output_audio_transcript.delta'),
        ev('output_audio_buffer.started'),
        ev('response.done'), // generation over; the track still plays
        ev('output_audio_buffer.stopped'),
        ev('output_audio_buffer.started'),
        ev('input_audio_buffer.speech_started'),
        ev('output_audio_buffer.cleared'),
        { kind: 'played' }, // the page plays nothing itself over WebRTC: no effect
      ]),
    ).toEqual([false, true, true, false, true, false, false, false])
  })

  it('stays on when the user talks over a full-duplex model', () => {
    expect(run('websocket', [{ kind: 'played' }, ev('input_audio_buffer.speech_started')], true)).toEqual([true, true])
    expect(run('webrtc', [ev('output_audio_buffer.started'), ev('input_audio_buffer.speech_started')], true)).toEqual([
      true,
      true,
    ])
  })
})

/** Just enough Web Audio for the player: a clock the test moves, and sources that end when it passes them. */
class FakeContext {
  currentTime = 0
  state = 'running'
  destination = {}
  sources: { at: number; dur: number; onended: (() => void) | null; done: boolean }[] = []
  createAnalyser() {
    return { fftSize: 0, smoothingTimeConstant: 0, connect: () => undefined }
  }
  createBuffer(_channels: number, length: number, rate: number) {
    return { duration: length / rate, getChannelData: () => new Float32Array(length) }
  }
  createBufferSource() {
    const src = {
      buffer: null as { duration: number } | null,
      onended: null as (() => void) | null,
      connect: () => undefined,
      start: (at: number) => this.sources.push(Object.assign(src, { at, dur: src.buffer?.duration ?? 0, done: false })),
      stop: () => undefined,
    }
    return src
  }
  resume() {
    return Promise.resolve()
  }
  /** Move the clock: sources that have finished end. */
  advance(seconds: number) {
    this.currentTime += seconds
    for (const s of this.sources as (typeof this.sources)[number][]) {
      if (!s.done && s.at + s.dur <= this.currentTime) {
        s.done = true
        s.onended?.()
      }
    }
  }
}

describe('playback activity (what drives the indicator over WebSocket)', () => {
  afterEach(() => vi.useRealTimers())

  it('reports when queued audio starts being heard and when the queue has run dry', () => {
    vi.useFakeTimers()
    const ctx = new FakeContext()
    const player = new GaplessPlayer(ctx as unknown as AudioContext)
    const log: string[] = []
    player.onActive = () => log.push('active')
    player.onIdle = () => log.push('idle')
    player.enqueue(new Float32Array(2400), 24000) // 0.1 s
    player.enqueue(new Float32Array(2400), 24000)
    expect(log).toEqual([]) // queued, not heard yet
    vi.advanceTimersByTime(GaplessPlayer.LEAD * 1000 + 1)
    expect(log).toEqual(['active'])
    ctx.advance(0.15)
    expect(log).toEqual(['active']) // one chunk still playing
    ctx.advance(0.1)
    expect(log).toEqual(['active', 'idle'])
    player.enqueue(new Float32Array(2400), 24000) // the next sentence
    vi.advanceTimersByTime(GaplessPlayer.LEAD * 1000 + 1)
    expect(log).toEqual(['active', 'idle', 'active'])
    player.stop() // barge-in: the page stops it, nothing more is reported
    ctx.advance(1)
    vi.advanceTimersByTime(1000)
    expect(log).toEqual(['active', 'idle', 'active'])
  })
})
