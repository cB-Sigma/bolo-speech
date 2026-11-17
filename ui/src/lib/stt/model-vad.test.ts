import { describe, expect, it } from 'vitest'
import type { AudioModelEngine } from '@/lib/browser-audio/engine'
import type { BrowserSttEngine } from '@/lib/browser-stt/engine'
import { Store } from '@/lib/store'
import { BrowserTranscriber } from './browser'
import { ModelVad, type TurnCheck } from './model-vad'

const RATE = 16000
const FRAME = 512

/** A detector whose speech probability is 0.9 where the audio is loud and 0.05 where it is quiet. */
function fakeVadEngine(): AudioModelEngine {
  let carry = new Float32Array(0)
  return {
    vad: async (audio: Float32Array) => {
      const data = new Float32Array(carry.length + audio.length)
      data.set(carry)
      data.set(audio, carry.length)
      const probs: number[] = []
      let o = 0
      for (; o + FRAME <= data.length; o += FRAME) probs.push(Math.abs(data[o + 10] as number) > 0.1 ? 0.9 : 0.05)
      carry = data.slice(o)
      return { probs: Float32Array.from(probs), frame: FRAME, ms: 1 }
    },
  } as unknown as AudioModelEngine
}

function fakeStt(): BrowserSttEngine {
  return {
    store: new Store({ device: 'wasm', model: { id: 'fake' } }),
    load: async () => undefined,
    transcribe: async (audio: Float32Array) => ({
      text: `${(audio.length / RATE).toFixed(1)} s`,
      ms: 5,
      language: 'en',
    }),
  } as unknown as BrowserSttEngine
}

/** `seconds` of loud (speech) or quiet audio. */
const audio = (seconds: number, loud: boolean) => Float32Array.from({ length: seconds * RATE }, () => (loud ? 0.5 : 0))

async function feed(t: BrowserTranscriber, parts: [number, boolean][]) {
  for (const [seconds, loud] of parts) {
    const a = audio(seconds, loud)
    for (let o = 0; o < a.length; o += 320) t.push(a.subarray(o, o + 320))
  }
  await t.finish()
}

describe('a model voice-activity detector', () => {
  it('opens and closes segments with hysteresis and minimum durations', async () => {
    const vad = new ModelVad(fakeVadEngine(), { startMs: 64, endMs: 300 })
    const events = [
      ...(await vad.push(audio(0.5, false))),
      ...(await vad.push(audio(1, true))),
      ...(await vad.push(audio(0.5, false))),
    ]
    expect(events.map((e) => e.event)).toEqual(['speech_start', 'speech_end'])
    expect(events[0]?.t).toBeCloseTo(0.512, 1)
    expect(events[1]?.t).toBeCloseTo(1.5, 1)
  })

  it('drives the recogniser: one final per utterance', async () => {
    const t = new BrowserTranscriber(fakeStt(), { vad: new ModelVad(fakeVadEngine(), { endMs: 300 }) })
    await t.start()
    await feed(t, [
      [0.3, false],
      [1, true],
      [0.6, false],
      [1, true],
      [0.6, false],
    ])
    expect(t.store.get().finals).toHaveLength(2)
  })
})

describe('the end-of-turn model', () => {
  async function run(probability: number, parts: [number, boolean][]) {
    const checks: TurnCheck[] = []
    const turn = { turn: async () => ({ value: probability, ms: 3 }) } as unknown as AudioModelEngine
    const t = new BrowserTranscriber(fakeStt(), {
      vad: new ModelVad(fakeVadEngine(), { endMs: 200 }),
      turn: { engine: turn, maxWaitMs: 1000 },
      onTurnCheck: (c) => checks.push(c),
    })
    await t.start()
    await feed(t, parts)
    return { finals: t.store.get().finals, checks }
  }

  it('ends the turn at a pause the model hears as finished', async () => {
    const { finals, checks } = await run(0.9, [
      [0.3, false],
      [1, true],
      [0.5, false],
      [1, true],
      [0.5, false],
    ])
    expect(finals).toHaveLength(2)
    expect(checks.map((c) => c.decision)).toEqual(['end', 'end'])
  })

  it('holds a pause the model hears as unfinished, so the speaker can go on in the same turn', async () => {
    const { finals, checks } = await run(0.1, [
      [0.3, false],
      [1, true],
      [0.5, false],
      [1, true],
      [1.6, false],
    ])
    // One turn: the first pause was held, the second ran past the longest wait.
    expect(finals).toHaveLength(1)
    expect(checks.map((c) => c.decision)).toEqual(['wait', 'wait', 'timeout'])
  })
})
