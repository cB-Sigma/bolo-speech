import { describe, expect, it } from 'vitest'
import { cluster, der, diarize, FRAME_STEP, localActivity, RATE, type Models } from './pipeline'

/**
 * A fake conversation: speaker 0 from 0–6 s, speaker 1 from 5–12 s (1 s overlap), speaker 0 again 13–18 s.
 * The fake segmentation reads the truth off the sample values; the fake embedding is a one-hot per speaker.
 */
const TRUTH = [
  { speaker: 'A', start: 0, end: 6 },
  { speaker: 'B', start: 5, end: 12 },
  { speaker: 'A', start: 13, end: 18 },
]
const AUDIO_S = 20
function fakeAudio(): Float32Array {
  // Encode who talks in the sample: bit 0 = A, bit 1 = B (the models below decode it).
  const a = new Float32Array(AUDIO_S * RATE)
  for (const t of TRUTH)
    for (let i = t.start * RATE; i < t.end * RATE; i++) a[i] = (a[i] as number) + (t.speaker === 'A' ? 1 : 2)
  return a
}
const logit = (cls: number) => [0, 1, 2, 3, 4, 5, 6].map((c) => (c === cls ? 10 : -10))
const models: Models = {
  async segment(w) {
    const frames = Math.floor((w.length - 990) / FRAME_STEP) + 1
    return Array.from({ length: frames }, (_, f) => {
      const v = w[f * FRAME_STEP + 495] ?? 0
      // Local speaker 0 = A, local speaker 1 = B, whatever the window (clustering must join them up).
      return logit(v === 3 ? 4 : v === 1 ? 1 : v === 2 ? 2 : 0)
    })
  },
  async embed(audio) {
    let a = 0
    let b = 0
    for (const x of audio) {
      if (x === 1) a++
      else if (x === 2) b++
    }
    return Float32Array.from(a >= b ? [1, 0.05, 0] : [0.05, 1, 0])
  },
}

describe('in-browser diarization pipeline', () => {
  it('reads powerset logits as per-speaker activity', () => {
    const act = localActivity([logit(4), logit(0), logit(6)])
    expect(act[0]?.map((x) => Math.round(x))).toEqual([1, 1, 0])
    expect(act[1]?.map((x) => Math.round(x))).toEqual([0, 0, 0])
    expect(act[2]?.map((x) => Math.round(x))).toEqual([0, 1, 1])
  })

  it('clusters by threshold or to a count, never joining two speakers of one window', () => {
    const e = [Float32Array.from([1, 0]), Float32Array.from([0.99, 0.1]), Float32Array.from([0, 1])]
    expect(cluster(e, { k: null, threshold: 0.3 })).toEqual([0, 0, 1])
    expect(cluster(e, { k: 1, threshold: 0.3 })).toEqual([0, 0, 0])
    expect(cluster(e, { k: 1, threshold: 0.3, cannotLink: (i, j) => i + j === 1 })).toEqual([0, 0, 0])
    expect(cluster(e, { k: 2, threshold: 0.3, cannotLink: (i, j) => i + j === 1 })).toEqual([0, 1, 1])
  })

  it('finds who spoke when, overlaps included, and scores well against the truth', async () => {
    const segs = await diarize(fakeAudio(), models, { speakers: null, threshold: 0.5, step: 2.5 })
    expect(new Set(segs.map((s) => s.speaker))).toEqual(new Set([0, 1]))
    const hyp = segs.map((s) => ({ speaker: String(s.speaker), start: s.start, end: s.end }))
    const score = der(TRUTH, hyp)
    expect(score.der).toBeLessThan(0.05)
    expect(score.mapping).toEqual({ '0': 'A', '1': 'B' })
    // Asking for one speaker merges them: every second of B becomes confusion or a miss.
    const one = await diarize(fakeAudio(), models, { speakers: 1, threshold: 0.5, step: 2.5 })
    expect(new Set(one.map((s) => s.speaker))).toEqual(new Set([0]))
  })

  it('scores DER: missed, false alarm and confusion', () => {
    const ref = [{ speaker: 'A', start: 0, end: 10 }]
    expect(der(ref, [{ speaker: 'x', start: 0, end: 10 }]).der).toBeCloseTo(0)
    expect(der(ref, [{ speaker: 'x', start: 0, end: 5 }]).der).toBeCloseTo(0.5)
    expect(
      der(ref, [
        { speaker: 'x', start: 0, end: 10 },
        { speaker: 'y', start: 0, end: 2 },
      ]).der,
    ).toBeCloseTo(0.2)
    const two = der([...ref, { speaker: 'B', start: 10, end: 20 }], [{ speaker: 'x', start: 0, end: 20 }])
    expect(two.confusion).toBeCloseTo(10)
    expect(two.der).toBeCloseTo(0.5)
  })
})
