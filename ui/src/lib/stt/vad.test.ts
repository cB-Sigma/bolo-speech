import { readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { decodeWav } from '@/lib/audio/wav'
import { EnergyVad, type VadEvent } from './vad'

function run(samples: Float32Array, rate: number, block = 320): VadEvent[] {
  const vad = new EnergyVad({ sampleRate: rate })
  const events: VadEvent[] = []
  for (let o = 0; o < samples.length; o += block) events.push(...vad.push(samples.subarray(o, o + block)))
  events.push(...vad.flush())
  return events
}

function tone(seconds: number, rate: number, amp: number): Float32Array {
  const out = new Float32Array(Math.round(seconds * rate))
  for (let i = 0; i < out.length; i++)
    out[i] = amp * Math.sin((2 * Math.PI * 220 * i) / rate) + (Math.random() - 0.5) * 0.002
  return out
}

function concat(...parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((a, p) => a + p.length, 0))
  let o = 0
  for (const p of parts) {
    out.set(p, o)
    o += p.length
  }
  return out
}

describe('EnergyVad', () => {
  it('stays quiet on silence and low noise', () => {
    expect(run(tone(2, 16000, 0), 16000)).toEqual([])
  })

  it('finds a burst of speech-level sound between silences', () => {
    const audio = concat(tone(1, 16000, 0), tone(1.2, 16000, 0.3), tone(1.5, 16000, 0))
    const ev = run(audio, 16000)
    expect(ev.map((e) => e.event)).toEqual(['speech_start', 'speech_end'])
    expect(ev[0]?.t).toBeGreaterThan(0.9)
    expect(ev[0]?.t).toBeLessThan(1.1)
    expect(ev[1]?.t).toBeGreaterThan(2.1)
    expect(ev[1]?.t).toBeLessThan(2.3)
  })

  it('closes an open segment on flush', () => {
    const ev = run(concat(tone(0.5, 16000, 0), tone(1, 16000, 0.3)), 16000)
    expect(ev.map((e) => e.event)).toEqual(['speech_start', 'speech_end'])
  })

  it('gives the same result regardless of block size', () => {
    const audio = concat(tone(0.7, 16000, 0), tone(0.9, 16000, 0.25), tone(1, 16000, 0))
    expect(run(audio, 16000, 123)).toEqual(run(audio, 16000, 4096))
  })

  it('segments real speech with pauses and a noisy background', () => {
    const wav = decodeWav(readFileSync(path.resolve(import.meta.dirname, '../../mock/assets/en-ask-not.wav')))
    const ev = run(wav.samples, wav.sampleRate)
    const starts = ev.filter((e) => e.event === 'speech_start')
    const ends = ev.filter((e) => e.event === 'speech_end')
    expect(starts.length).toBe(ends.length)
    // The recording has four phrases; the two short pauses may or may not split.
    expect(starts.length).toBeGreaterThanOrEqual(2)
    expect(starts.length).toBeLessThanOrEqual(5)
    expect(starts[0]?.t).toBeLessThan(0.6)
  })
})
