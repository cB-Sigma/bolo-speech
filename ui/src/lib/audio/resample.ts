/** Windowed-sinc low-pass FIR. `cutoff` is a fraction of the sample rate (0..0.5). */
export function designLowpass(cutoff: number, taps: number): Float32Array {
  const n = taps | 1 // odd length for a symmetric, linear-phase filter
  const h = new Float32Array(n)
  const mid = (n - 1) / 2
  let sum = 0
  for (let i = 0; i < n; i++) {
    const x = i - mid
    const sinc = x === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x)
    const window = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (n - 1))
    h[i] = sinc * window
    sum += h[i] as number
  }
  for (let i = 0; i < n; i++) h[i] = (h[i] as number) / sum
  return h
}

/**
 * Streaming sample-rate converter: an anti-alias FIR (only when
 * downsampling) followed by linear interpolation. State is carried across
 * calls so audio can be fed block by block without clicks at the seams.
 */
export class Resampler {
  readonly inRate: number
  readonly outRate: number
  private readonly fir: Float32Array | null
  private history: Float32Array
  private readonly step: number
  private pos = 0 // fractional read position relative to the start of the next block
  private last = 0 // last filtered sample of the previous block

  constructor(inRate: number, outRate: number, taps = 63) {
    if (!(inRate > 0) || !(outRate > 0)) throw new RangeError('sample rates must be positive')
    this.inRate = inRate
    this.outRate = outRate
    this.step = inRate / outRate
    this.fir = outRate < inRate ? designLowpass((0.5 * outRate * 0.92) / inRate, taps) : null
    this.history = new Float32Array(this.fir ? this.fir.length - 1 : 0)
  }

  private filter(input: Float32Array): Float32Array {
    const fir = this.fir
    if (!fir) return input
    const k = fir.length
    const joined = new Float32Array(this.history.length + input.length)
    joined.set(this.history)
    joined.set(input, this.history.length)
    const out = new Float32Array(input.length)
    for (let i = 0; i < input.length; i++) {
      let acc = 0
      for (let j = 0; j < k; j++) acc += (joined[i + j] as number) * (fir[j] as number)
      out[i] = acc
    }
    this.history = joined.slice(joined.length - (k - 1))
    return out
  }

  process(input: Float32Array): Float32Array {
    if (input.length === 0) return new Float32Array(0)
    if (this.inRate === this.outRate) return input
    const x = this.filter(input)
    const n = x.length
    const out = new Float32Array(Math.max(0, Math.ceil((n - this.pos) / this.step)))
    let o = 0
    let pos = this.pos
    while (pos < n) {
      const i = Math.floor(pos)
      const frac = pos - i
      const a = i - 1 < 0 ? this.last : (x[i - 1] as number)
      const b = x[i] as number
      // Position p lies between sample p-1 and p of the virtual stream.
      out[o++] = a + (b - a) * frac
      pos += this.step
    }
    this.pos = pos - n
    this.last = x[n - 1] as number
    return o === out.length ? out : out.subarray(0, o)
  }
}

/** One-shot conversion of a whole buffer. */
export function resample(input: Float32Array, inRate: number, outRate: number): Float32Array {
  if (inRate === outRate) return input
  return new Resampler(inRate, outRate).process(input)
}
