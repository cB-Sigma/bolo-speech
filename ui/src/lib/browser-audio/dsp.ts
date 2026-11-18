/**
 * Signal processing for the small audio models the page runs (denoisers, voice-activity and end-of-turn detectors,
 * language identifiers): an FFT of any size, short-time Fourier transforms and the feature front ends the server
 * computes in NumPy (bolo.audio.features, bolo.engines.lid_onnx), written to give the same numbers.
 */

/** In-place complex FFT of a power-of-two size (re and im of length n). */
function fftPow2(re: Float64Array, im: Float64Array): void {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      let t = re[i] as number
      re[i] = re[j] as number
      re[j] = t
      t = im[i] as number
      im[i] = im[j] as number
      im[j] = t
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len
    const wr = Math.cos(ang)
    const wi = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let cr = 1
      let ci = 0
      for (let k = 0; k < len / 2; k++) {
        const a = i + k
        const b = a + len / 2
        const xr = (re[b] as number) * cr - (im[b] as number) * ci
        const xi = (re[b] as number) * ci + (im[b] as number) * cr
        re[b] = (re[a] as number) - xr
        im[b] = (im[a] as number) - xi
        re[a] = (re[a] as number) + xr
        im[a] = (im[a] as number) + xi
        const nr = cr * wr - ci * wi
        ci = cr * wi + ci * wr
        cr = nr
      }
    }
  }
}

const isPow2 = (n: number) => n > 0 && (n & (n - 1)) === 0

/**
 * A real FFT of one size, planned once: power-of-two sizes directly, others (400 for the 25 ms windows of the
 * speech front ends) through Bluestein's algorithm on a power-of-two FFT.
 */
export class RealFft {
  readonly n: number
  readonly bins: number
  private readonly m: number
  private readonly chirpRe: Float64Array | null = null
  private readonly chirpIm: Float64Array | null = null
  private readonly kernelRe: Float64Array | null = null
  private readonly kernelIm: Float64Array | null = null
  private readonly re: Float64Array
  private readonly im: Float64Array

  constructor(n: number) {
    this.n = n
    this.bins = Math.floor(n / 2) + 1
    if (isPow2(n)) {
      this.m = n
    } else {
      let m = 1
      while (m < 2 * n - 1) m <<= 1
      this.m = m
      const cr = new Float64Array(n)
      const ci = new Float64Array(n)
      for (let k = 0; k < n; k++) {
        // exp(-i pi k^2 / n), with k^2 reduced mod 2n to keep the angle small
        const a = (Math.PI * ((k * k) % (2 * n))) / n
        cr[k] = Math.cos(a)
        ci[k] = -Math.sin(a)
      }
      const kr = new Float64Array(m)
      const ki = new Float64Array(m)
      kr[0] = cr[0] as number
      ki[0] = -(ci[0] as number)
      for (let k = 1; k < n; k++) {
        kr[k] = kr[m - k] = cr[k] as number
        ki[k] = ki[m - k] = -(ci[k] as number)
      }
      fftPow2(kr, ki)
      this.chirpRe = cr
      this.chirpIm = ci
      this.kernelRe = kr
      this.kernelIm = ki
    }
    this.re = new Float64Array(this.m)
    this.im = new Float64Array(this.m)
  }

  /** In-place DFT of a complex signal of length n held in this plan's buffers (`re`, `im`, first n values). */
  private dft(): void {
    const { n, re, im } = this
    if (!this.chirpRe) {
      fftPow2(re, im)
      return
    }
    const cr = this.chirpRe
    const ci = this.chirpIm as Float64Array
    for (let k = 0; k < n; k++) {
      const a = re[k] as number
      const b = im[k] as number
      re[k] = a * (cr[k] as number) - b * (ci[k] as number)
      im[k] = a * (ci[k] as number) + b * (cr[k] as number)
    }
    re.fill(0, n)
    im.fill(0, n)
    fftPow2(re, im)
    const kr = this.kernelRe as Float64Array
    const ki = this.kernelIm as Float64Array
    for (let k = 0; k < this.m; k++) {
      const a = re[k] as number
      const b = im[k] as number
      re[k] = a * (kr[k] as number) - b * (ki[k] as number)
      // conjugated, for the inverse FFT that follows
      im[k] = -(a * (ki[k] as number) + b * (kr[k] as number))
    }
    fftPow2(re, im)
    const s = 1 / this.m
    for (let k = 0; k < n; k++) {
      const a = (re[k] as number) * s
      const b = -(im[k] as number) * s
      re[k] = a * (cr[k] as number) - b * (ci[k] as number)
      im[k] = a * (ci[k] as number) + b * (cr[k] as number)
    }
  }

  /** Spectrum of `frame` (length n): `outRe`/`outIm` get the n/2+1 bins. */
  forward(frame: ArrayLike<number>, outRe: Float64Array, outIm: Float64Array): void {
    const { n, re, im } = this
    for (let i = 0; i < n; i++) {
      re[i] = frame[i] as number
      im[i] = 0
    }
    this.dft()
    for (let k = 0; k < this.bins; k++) {
      outRe[k] = re[k] as number
      outIm[k] = im[k] as number
    }
  }

  /** Real signal of length n from its n/2+1 bins (the inverse of `forward`). */
  inverse(binsRe: ArrayLike<number>, binsIm: ArrayLike<number>, out: Float64Array): void {
    const { n, re, im } = this
    // The full conjugate-symmetric spectrum, conjugated: DFT(conj(X)) = n * conj(x), and x is real.
    for (let k = 0; k < this.bins; k++) {
      re[k] = binsRe[k] as number
      im[k] = -(binsIm[k] as number)
    }
    for (let k = this.bins; k < n; k++) {
      re[k] = binsRe[n - k] as number
      im[k] = binsIm[n - k] as number
    }
    this.dft()
    for (let i = 0; i < n; i++) out[i] = (re[i] as number) / n
  }
}

/** numpy.pad(x, pad, mode='reflect'). */
export function reflectPad(x: ArrayLike<number>, pad: number): Float64Array {
  const n = x.length
  const out = new Float64Array(n + 2 * pad)
  for (let i = 0; i < out.length; i++) {
    let j = i - pad
    if (n > 1) {
      const period = 2 * (n - 1)
      j = ((j % period) + period) % period
      if (j >= n) j = period - j
    } else j = 0
    out[i] = n ? (x[j] as number) : 0
  }
  return out
}

export const periodicHann = (n: number) =>
  Float64Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n))

// ---------------------------------------------------------------------------------------------------------------
// Whisper's log-mel features (bolo.audio.features.whisper_log_mel), used by Smart Turn
// ---------------------------------------------------------------------------------------------------------------

const hzToMel = (f: number) => (f >= 1000 ? 15 + Math.log(f / 1000) * (27 / Math.log(6.4)) : (3 * f) / 200)
const melToHz = (m: number) => (m >= 15 ? 1000 * Math.exp((Math.log(6.4) / 27) * (m - 15)) : (200 * m) / 3)

/** Slaney-style triangular filters with area normalisation, [bin][mel] flattened (bins × nMels). */
export function slaneyFilterbank(nMels: number, nFft: number, rate = 16000): Float64Array {
  const bins = Math.floor(nFft / 2) + 1
  const lo = hzToMel(0)
  const hi = hzToMel(rate / 2)
  const edges = Array.from({ length: nMels + 2 }, (_, i) => melToHz(lo + ((hi - lo) * i) / (nMels + 1)))
  const out = new Float64Array(bins * nMels)
  for (let b = 0; b < bins; b++) {
    const f = (rate / 2) * (b / (bins - 1))
    for (let m = 0; m < nMels; m++) {
      const e0 = edges[m] as number
      const e1 = edges[m + 1] as number
      const e2 = edges[m + 2] as number
      const down = (f - e0) / (e1 - e0)
      const up = (e2 - f) / (e2 - e1)
      out[b * nMels + m] = Math.max(0, Math.min(down, up)) * (2 / (e2 - e0))
    }
  }
  return out
}

let whisperBank: Float64Array | null = null
let fft400: RealFft | null = null

/**
 * Whisper's features, shape [80][frames] flattened row by row (mel-major), frames = len / 160: 25 ms periodic Hann
 * windows every 10 ms on reflect-padded audio, power spectrum, 80 Slaney mel bands, log10, clamped to 8 below the
 * peak, (x + 4) / 4. `normalize` first gives the audio zero mean and unit variance (Smart Turn's input).
 */
export function whisperLogMel(audio: Float32Array, normalize = false): { data: Float32Array; frames: number } {
  const N = 400
  const HOP = 160
  const MELS = 80
  let x: ArrayLike<number> = audio
  if (normalize && audio.length) {
    let mean = 0
    for (const v of audio) mean += v
    mean /= audio.length
    let v2 = 0
    for (const v of audio) v2 += (v - mean) * (v - mean)
    const sd = Math.sqrt(v2 / audio.length + 1e-7)
    x = Float64Array.from(audio, (v) => (v - mean) / sd)
  }
  whisperBank ??= slaneyFilterbank(MELS, N)
  fft400 ??= new RealFft(N)
  const bank = whisperBank
  const fft = fft400
  const window = periodicHann(N)
  const padded = reflectPad(x, N / 2)
  const frames = Math.max(0, 1 + Math.floor((padded.length - N) / HOP) - 1)
  const log = new Float64Array(frames * MELS)
  const frame = new Float64Array(N)
  const re = new Float64Array(fft.bins)
  const im = new Float64Array(fft.bins)
  let peak = -Infinity
  for (let t = 0; t < frames; t++) {
    for (let i = 0; i < N; i++) frame[i] = (padded[t * HOP + i] as number) * (window[i] as number)
    fft.forward(frame, re, im)
    for (let m = 0; m < MELS; m++) {
      let s = 0
      for (let b = 0; b < fft.bins; b++) {
        const w = bank[b * MELS + m] as number
        if (w) s += w * ((re[b] as number) ** 2 + (im[b] as number) ** 2)
      }
      const v = Math.log10(Math.max(s, 1e-10))
      log[t * MELS + m] = v
      if (v > peak) peak = v
    }
  }
  const data = new Float32Array(MELS * frames)
  for (let t = 0; t < frames; t++)
    for (let m = 0; m < MELS; m++) data[m * frames + t] = (Math.max(log[t * MELS + m] as number, peak - 8) + 4) / 4
  return { data, frames }
}

// ---------------------------------------------------------------------------------------------------------------
// SpeechBrain's Fbank(n_mels=60) (bolo.engines.lid_onnx.fbank60), used by the VoxLingua107 language identifier
// ---------------------------------------------------------------------------------------------------------------

let fbankMatrix: Float64Array | null = null

function htkFilterbank(nMels: number, nFft: number, rate: number): Float64Array {
  const toMel = (f: number) => 2595 * Math.log10(1 + f / 700)
  const toHz = (m: number) => 700 * (10 ** (m / 2595) - 1)
  const lo = toMel(0)
  const hi = toMel(rate / 2)
  const points = Array.from({ length: nMels + 2 }, (_, i) => toHz(lo + ((hi - lo) * i) / (nMels + 1)))
  const bins = Math.floor(nFft / 2) + 1
  const out = new Float64Array(bins * nMels)
  for (let b = 0; b < bins; b++) {
    const f = (rate / 2) * (b / (bins - 1))
    for (let m = 0; m < nMels; m++) {
      const band = (points[m + 1] as number) - (points[m] as number)
      const slope = (f - (points[m + 1] as number)) / band
      out[b * nMels + m] = Math.max(0, Math.min(slope + 1, 1 - slope))
    }
  }
  return out
}

/**
 * SpeechBrain `Fbank(n_mels=60)`: [frames][60] flattened: centred 25 ms periodic Hamming frames every 10 ms (zero
 * padded), power spectrum, 60 HTK mel bands, 10 log10 floored 80 dB under the loudest value.
 */
export function fbank60(audio: Float32Array): { data: Float32Array; frames: number } {
  const N = 400
  const HOP = 160
  const MELS = 60
  fbankMatrix ??= htkFilterbank(MELS, N, 16000)
  fft400 ??= new RealFft(N)
  const bank = fbankMatrix
  const fft = fft400
  const window = Float64Array.from({ length: N }, (_, i) => 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / N))
  const padded = new Float64Array(audio.length + N)
  padded.set(audio, N / 2)
  const frames = Math.max(0, 1 + Math.floor((padded.length - N) / HOP))
  const db = new Float64Array(frames * MELS)
  const frame = new Float64Array(N)
  const re = new Float64Array(fft.bins)
  const im = new Float64Array(fft.bins)
  let peak = -Infinity
  for (let t = 0; t < frames; t++) {
    for (let i = 0; i < N; i++) frame[i] = (padded[t * HOP + i] as number) * (window[i] as number)
    fft.forward(frame, re, im)
    for (let m = 0; m < MELS; m++) {
      let s = 0
      for (let b = 0; b < fft.bins; b++) {
        const w = bank[b * MELS + m] as number
        if (w) s += w * ((re[b] as number) ** 2 + (im[b] as number) ** 2)
      }
      const v = 10 * Math.log10(Math.max(s, 1e-10))
      db[t * MELS + m] = v
      if (v > peak) peak = v
    }
  }
  const data = new Float32Array(frames * MELS)
  for (let i = 0; i < data.length; i++) data[i] = Math.max(db[i] as number, peak - 80)
  return { data, frames }
}

// ---------------------------------------------------------------------------------------------------------------
// CED's features (sherpa-onnx's OfflineStream for CED: Kaldi fbank, 64 bands, 10 log10, 120 dB range)
// ---------------------------------------------------------------------------------------------------------------

let cedBank: Float64Array | null = null
let fft512: RealFft | null = null

/** Kaldi's mel filterbank: `nMels` triangles in Kaldi's mel scale between `low` and `high`, [mel][bin] flattened. */
function kaldiFilterbank(nMels: number, nFft: number, rate: number, low: number, high: number): Float64Array {
  const mel = (f: number) => 1127 * Math.log(1 + f / 700)
  const bins = nFft / 2 + 1
  const lo = mel(low)
  const delta = (mel(high) - lo) / (nMels + 1)
  const out = new Float64Array(nMels * bins)
  for (let m = 0; m < nMels; m++) {
    const left = lo + m * delta
    const center = left + delta
    const right = center + delta
    for (let b = 0; b < nFft / 2; b++) {
      const f = mel((rate * b) / nFft)
      if (f > left && f < right) out[m * bins + b] = f <= center ? (f - left) / delta : (right - f) / delta
    }
  }
  return out
}

/**
 * CED's input, shape [64][frames] flattened row by row (band-major), frames = round(len / 160): Kaldi's fbank
 * without snipped edges (32 ms symmetric Hann windows every 10 ms centred on each shift, reflected at the ends, no
 * dither, pre-emphasis or DC removal), power spectrum, 64 Kaldi mel bands from 0 to 8 kHz, then 10 log10 floored
 * 120 dB under the loudest value. A NumPy copy of this gave tag probabilities within 0.02 of sherpa-onnx's own on
 * its test clips (the server's tagger).
 */
export function cedFeatures(audio: Float32Array): { data: Float32Array; frames: number } {
  const N = 512
  const HOP = 160
  const MELS = 64
  cedBank ??= kaldiFilterbank(MELS, N, 16000, 0, 8000)
  fft512 ??= new RealFft(N)
  const bank = cedBank
  const fft = fft512
  const n = audio.length
  const frames = Math.floor((n + HOP / 2) / HOP)
  const window = Float64Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1)))
  const frame = new Float64Array(N)
  const re = new Float64Array(fft.bins)
  const im = new Float64Array(fft.bins)
  const power = new Float64Array(fft.bins)
  const db = new Float64Array(MELS * frames)
  let peak = -Infinity
  for (let t = 0; t < frames; t++) {
    const start = t * HOP + HOP / 2 - N / 2
    for (let i = 0; i < N; i++) {
      let k = start + i
      if (k < 0) k = -k - 1
      else if (k >= n) k = 2 * n - 1 - k
      frame[i] = ((audio[Math.min(Math.max(k, 0), n - 1)] as number) || 0) * (window[i] as number)
    }
    fft.forward(frame, re, im)
    for (let b = 0; b < fft.bins; b++) power[b] = (re[b] as number) ** 2 + (im[b] as number) ** 2
    for (let m = 0; m < MELS; m++) {
      let s = 0
      const row = m * fft.bins
      for (let b = 0; b < fft.bins; b++) {
        const w = bank[row + b] as number
        if (w) s += w * (power[b] as number)
      }
      const v = 10 * Math.log10(Math.max(s, 1e-10))
      db[m * frames + t] = v
      if (v > peak) peak = v
    }
  }
  const data = new Float32Array(MELS * frames)
  for (let i = 0; i < data.length; i++) data[i] = Math.max(db[i] as number, peak - 120)
  return { data, frames }
}

/** Softmax of scores, in double precision. */
export function softmax(scores: ArrayLike<number>): Float64Array {
  let max = -Infinity
  for (let i = 0; i < scores.length; i++) max = Math.max(max, scores[i] as number)
  const out = Float64Array.from({ length: scores.length }, (_, i) => Math.exp((scores[i] as number) - max))
  let sum = 0
  for (const v of out) sum += v
  for (let i = 0; i < out.length; i++) out[i] = (out[i] as number) / sum
  return out
}

/** The server's `mix`: `strength` of the enhanced audio plus the rest of the original (same rate and length). */
export function mixWithOriginal(original: Float32Array, enhanced: Float32Array, strength: number): Float32Array {
  if (strength >= 1) return enhanced
  const out = new Float32Array(original.length)
  for (let i = 0; i < out.length; i++)
    out[i] = strength * ((enhanced[i] as number) ?? 0) + (1 - strength) * (original[i] as number)
  return out
}

/** A symmetric windowed-sinc low-pass (Blackman), unit gain at DC; `cutoff` as a fraction of the rate. */
function lowpass(cutoff: number, half: number): Float64Array {
  const n = 2 * half + 1
  const h = new Float64Array(n)
  let sum = 0
  for (let i = 0; i < n; i++) {
    const x = i - half
    const sinc = x === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x)
    const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (n - 1))
    h[i] = sinc * w
    sum += h[i] as number
  }
  for (let i = 0; i < n; i++) h[i] = (h[i] as number) / sum
  return h
}

/**
 * Resampling by a whole factor with a linear-phase filter whose delay is taken out, so the output lines up with
 * the input sample for sample (16 kHz to the 48 kHz denoisers and back). Other ratios are not needed here.
 */
export function resampleAligned(x: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return x
  const up = to > from
  const factor = up ? to / from : from / to
  if (!Number.isInteger(factor)) throw new Error(`resampleAligned: ${from} to ${to} is not a whole factor`)
  const half = 16 * factor
  const h = lowpass(0.5 / factor, half)
  if (up) {
    const out = new Float32Array(x.length * factor)
    // y[m] = factor * sum_k x[k] h[m - k*factor + half]
    for (let m = 0; m < out.length; m++) {
      let acc = 0
      const kMin = Math.max(0, Math.ceil((m - half) / factor))
      const kMax = Math.min(x.length - 1, Math.floor((m + half) / factor))
      for (let k = kMin; k <= kMax; k++) acc += (x[k] as number) * (h[m - k * factor + half] as number)
      out[m] = acc * factor
    }
    return out
  }
  const out = new Float32Array(Math.floor(x.length / factor))
  for (let j = 0; j < out.length; j++) {
    const c = j * factor
    let acc = 0
    const lo = Math.max(0, c - half)
    const hi = Math.min(x.length - 1, c + half)
    for (let i = lo; i <= hi; i++) acc += (x[i] as number) * (h[i - c + half] as number)
    out[j] = acc
  }
  return out
}
