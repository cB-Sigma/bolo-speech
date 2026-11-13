/**
 * Shorten long pauses the way sherpa-onnx (and so the bolo server) does after
 * every VITS call (GeneratedAudio::ScaleSilence): a run of samples with
 * |x| <= 0.01 that lasts longer than 0.2 s keeps `scale` of its length (its
 * first part). With the server's default of 0.2 the in-browser output is the
 * same, sample for sample, as the server's for the same input and noise.
 */
export const SILENCE_SCALE = 0.2

export function scaleSilence(samples: Float32Array, sampleRate: number, scale = SILENCE_SCALE): Float32Array {
  if (scale === 1) return samples
  const threshold = Math.trunc(sampleRate * 0.2)
  const intervals: [number, number][] = []
  const n = samples.length
  let last = -1
  for (let i = 0; i < n; i++) {
    if (Math.abs(samples[i] as number) <= 0.01) {
      if (last === -1) last = i
      continue
    }
    if (last !== -1 && i - last < threshold) {
      last = -1
      continue
    }
    if (last !== -1) {
      intervals.push([last, i])
      last = -1
    }
  }
  if (last !== -1 && n - last > threshold) intervals.push([last, n])
  if (intervals.length === 0) return samples
  const kept = intervals.reduce((a, [s, e]) => a + (e - s) - Math.trunc((e - s) * scale), 0)
  const out = new Float32Array(n - kept)
  let o = 0
  let i = 0
  for (const [s, e] of intervals) {
    out.set(samples.subarray(i, s), o)
    o += s - i
    const keep = Math.trunc((e - s) * scale)
    out.set(samples.subarray(s, s + keep), o)
    o += keep
    i = e
  }
  out.set(samples.subarray(i), o)
  return out
}
