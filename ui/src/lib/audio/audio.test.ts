import { describe, expect, it } from 'vitest'
import {
  alawToLinear,
  decodeAlaw,
  decodeMulaw,
  encodeAlaw,
  encodeMulaw,
  linearToAlaw,
  linearToMulaw,
  mulawToLinear,
} from './g711'
import {
  bytesToInt16,
  concatFloat,
  floatToInt16,
  Framer,
  int16ToBytes,
  int16ToFloat,
  mixToMono,
  peak,
  rms,
} from './pcm'
import { designLowpass, resample, Resampler } from './resample'
import { decodeTelephony, simulateTelephony, TelephonyEncoder } from './telephony'
import { decodeWav, encodeWav } from './wav'

function sine(freq: number, rate: number, seconds: number, amp = 0.5): Float32Array {
  const out = new Float32Array(Math.round(rate * seconds))
  for (let i = 0; i < out.length; i++) out[i] = amp * Math.sin((2 * Math.PI * freq * i) / rate)
  return out
}

/** Amplitude of one frequency via a single-bin DFT, ignoring the edges. */
function toneAmplitude(x: Float32Array, freq: number, rate: number): number {
  const a = Math.floor(x.length * 0.1)
  const b = Math.floor(x.length * 0.9)
  let re = 0
  let im = 0
  for (let i = a; i < b; i++) {
    re += (x[i] as number) * Math.cos((2 * Math.PI * freq * i) / rate)
    im += (x[i] as number) * Math.sin((2 * Math.PI * freq * i) / rate)
  }
  return (2 * Math.hypot(re, im)) / (b - a)
}

describe('G.711 μ-law', () => {
  it('matches reference code points', () => {
    expect(linearToMulaw(0)).toBe(0xff)
    expect(linearToMulaw(-1)).toBe(0x7f)
    expect(linearToMulaw(32767)).toBe(0x80)
    expect(linearToMulaw(-32768)).toBe(0x00)
    expect(mulawToLinear(0xff)).toBe(0)
    expect(mulawToLinear(0x80)).toBe(32124)
    expect(mulawToLinear(0x00)).toBe(-32124)
  })
  it('round-trips within the quantisation step of each segment', () => {
    for (let s = -32768; s <= 32767; s += 37) {
      const back = mulawToLinear(linearToMulaw(s))
      const tolerance = Math.max(8, Math.abs(s) * 0.04) + 900 * (Math.abs(s) > 32000 ? 1 : 0)
      expect(Math.abs(back - s)).toBeLessThanOrEqual(tolerance)
    }
  })
  it('is idempotent after one pass (decode∘encode∘decode = decode)', () => {
    for (let b = 0; b < 256; b++) {
      const lin = mulawToLinear(b)
      expect(mulawToLinear(linearToMulaw(lin))).toBe(lin)
    }
  })
  it('is odd-symmetric and monotonic', () => {
    let prev = -Infinity
    for (let s = -32000; s <= 32000; s += 500) {
      const v = mulawToLinear(linearToMulaw(s))
      expect(v).toBeGreaterThanOrEqual(prev)
      prev = v
      expect(mulawToLinear(linearToMulaw(-s))).toBe(-mulawToLinear(linearToMulaw(s)) || 0)
    }
  })
  it('encodes and decodes arrays', () => {
    const pcm = floatToInt16(sine(440, 8000, 0.05))
    const bytes = encodeMulaw(pcm)
    expect(bytes).toHaveLength(pcm.length)
    const back = decodeMulaw(bytes)
    let err = 0
    for (let i = 0; i < pcm.length; i++) err = Math.max(err, Math.abs((back[i] as number) - (pcm[i] as number)))
    expect(err).toBeLessThan(700)
  })
})

describe('G.711 A-law', () => {
  it('matches reference code points', () => {
    expect(linearToAlaw(0)).toBe(0xd5)
    expect(linearToAlaw(-1)).toBe(0x55)
    expect(linearToAlaw(32767)).toBe(0xaa)
    expect(linearToAlaw(-32768)).toBe(0x2a)
    expect(alawToLinear(0xd5)).toBe(8)
    expect(alawToLinear(0x55)).toBe(-8)
  })
  it('round-trips within the quantisation step and is idempotent', () => {
    for (let s = -32768; s <= 32767; s += 41) {
      const back = alawToLinear(linearToAlaw(s))
      expect(Math.abs(back - s)).toBeLessThanOrEqual(
        Math.max(16, Math.abs(s) * 0.04) + 600 * (Math.abs(s) > 32000 ? 1 : 0),
      )
    }
    for (let b = 0; b < 256; b++) {
      const lin = alawToLinear(b)
      expect(alawToLinear(linearToAlaw(lin))).toBe(lin)
    }
  })
  it('encodes and decodes arrays', () => {
    const pcm = floatToInt16(sine(300, 8000, 0.05, 0.8))
    const back = decodeAlaw(encodeAlaw(pcm))
    let err = 0
    for (let i = 0; i < pcm.length; i++) err = Math.max(err, Math.abs((back[i] as number) - (pcm[i] as number)))
    expect(err).toBeLessThan(1100)
  })
})

describe('PCM helpers', () => {
  it('converts float to int16 with clipping and back', () => {
    const i16 = floatToInt16(new Float32Array([0, 1, -1, 2, -2, 0.5]))
    expect([...i16]).toEqual([0, 32767, -32768, 32767, -32768, 16384])
    const f = int16ToFloat(i16)
    expect(f[1]).toBeCloseTo(1, 3)
    expect(f[2]).toBe(-1)
    expect(f[5]).toBeCloseTo(0.5, 3)
  })
  it('serialises little-endian and tolerates odd byte offsets', () => {
    const bytes = int16ToBytes(new Int16Array([1, -2, 0x1234]))
    expect([...bytes]).toEqual([1, 0, 0xfe, 0xff, 0x34, 0x12])
    const shifted = new Uint8Array(7)
    shifted.set(bytes, 1)
    expect([...bytesToInt16(shifted.subarray(1))]).toEqual([1, -2, 0x1234])
    expect(bytesToInt16(new Uint8Array([1, 0, 9]))).toHaveLength(1)
  })
  it('frames arbitrary blocks into fixed sizes without losing or reordering samples', () => {
    const framer = new Framer(320)
    const input = Float32Array.from({ length: 1000 }, (_, i) => i)
    const frames = [
      ...framer.push(input.subarray(0, 100)),
      ...framer.push(input.subarray(100, 777)),
      ...framer.push(input.subarray(777)),
    ]
    expect(frames).toHaveLength(3)
    expect(frames.every((f) => f.length === 320)).toBe(true)
    const rest = framer.flush()
    expect(rest).toHaveLength(40)
    expect([...concatFloat([...frames, rest as Float32Array])]).toEqual([...input])
    expect(framer.flush()).toBeNull()
    expect(() => new Framer(0)).toThrow(RangeError)
  })
  it('measures level and mixes channels', () => {
    expect(rms(sine(100, 8000, 1, 1))).toBeCloseTo(Math.SQRT1_2, 2)
    expect(rms(new Float32Array(0))).toBe(0)
    expect(peak(new Float32Array([0.1, -0.7, 0.3]))).toBeCloseTo(0.7)
    expect([...mixToMono([new Float32Array([1, 0]), new Float32Array([0, 1])])]).toEqual([0.5, 0.5])
  })
})

describe('Resampler', () => {
  it('produces the expected number of samples', () => {
    expect(resample(new Float32Array(48000), 48000, 16000)).toHaveLength(16000)
    expect(Math.abs(resample(new Float32Array(44100), 44100, 16000).length - 16000)).toBeLessThanOrEqual(1)
    expect(resample(new Float32Array(8000), 8000, 16000)).toHaveLength(16000)
  })
  it('returns the input untouched at equal rates', () => {
    const x = sine(100, 16000, 0.1)
    expect(resample(x, 16000, 16000)).toBe(x)
  })
  it('preserves an in-band tone when downsampling', () => {
    const y = resample(sine(1000, 48000, 0.5, 0.5), 48000, 16000)
    expect(toneAmplitude(y, 1000, 16000)).toBeGreaterThan(0.45)
    expect(toneAmplitude(y, 1000, 16000)).toBeLessThan(0.55)
  })
  it('suppresses tones above the new Nyquist instead of aliasing them', () => {
    // 11 kHz at 48 kHz would alias to 5 kHz at 16 kHz without the filter.
    const y = resample(sine(11000, 48000, 0.5, 0.5), 48000, 16000)
    expect(toneAmplitude(y, 5000, 16000)).toBeLessThan(0.01)
  })
  it('gives the same output streamed in blocks as in one call', () => {
    const x = sine(700, 44100, 0.3)
    const whole = resample(x, 44100, 16000)
    const r = new Resampler(44100, 16000)
    const parts: Float32Array[] = []
    for (let o = 0; o < x.length; o += 441) parts.push(r.process(x.subarray(o, o + 441)).slice())
    const streamed = concatFloat(parts)
    expect(streamed).toHaveLength(whole.length)
    let err = 0
    for (let i = 0; i < whole.length; i++) err = Math.max(err, Math.abs((whole[i] as number) - (streamed[i] as number)))
    expect(err).toBeLessThan(1e-5)
  })
  it('designs a unity-gain low-pass', () => {
    const h = designLowpass(0.15, 63)
    expect(h.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5)
    expect(h[0]).toBeCloseTo(h[h.length - 1] as number, 8)
    expect(() => new Resampler(0, 16000)).toThrow(RangeError)
  })
})

describe('WAV', () => {
  it('round-trips 16-bit mono', () => {
    const x = sine(440, 16000, 0.1)
    const wav = encodeWav(x, 16000)
    expect(wav).toHaveLength(44 + x.length * 2)
    expect(String.fromCharCode(...wav.subarray(0, 4))).toBe('RIFF')
    const back = decodeWav(wav)
    expect(back.sampleRate).toBe(16000)
    expect(back.samples).toHaveLength(x.length)
    expect(Math.abs((back.samples[100] as number) - (x[100] as number))).toBeLessThan(1 / 16000)
  })
  it('decodes stereo, μ-law and skips unknown chunks', () => {
    // Hand-built: RIFF, a LIST chunk to skip, fmt (μ-law, 2ch, 8 kHz), data
    const data = [0xff, 0x80, 0x00, 0xff]
    const bytes = new Uint8Array(12 + 12 + 26 + 8 + data.length)
    const v = new DataView(bytes.buffer)
    const put = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)))
    put(0, 'RIFF')
    v.setUint32(4, bytes.length - 8, true)
    put(8, 'WAVE')
    put(12, 'LIST')
    v.setUint32(16, 4, true)
    put(20, 'INFO')
    put(24, 'fmt ')
    v.setUint32(28, 18, true)
    v.setUint16(32, 7, true)
    v.setUint16(34, 2, true)
    v.setUint32(36, 8000, true)
    v.setUint32(40, 16000, true)
    v.setUint16(44, 2, true)
    v.setUint16(46, 8, true)
    put(50, 'data')
    v.setUint32(54, data.length, true)
    bytes.set(data, 58)
    const out = decodeWav(bytes)
    expect(out.sampleRate).toBe(8000)
    expect(out.samples).toHaveLength(2)
    expect(out.samples[0]).toBeCloseTo((0 + 32124) / 2 / 32768, 4)
    expect(out.samples[1]).toBeCloseTo((-32124 + 0) / 2 / 32768, 4)
  })
  it('rejects non-WAV data', () => {
    expect(() => decodeWav(new Uint8Array(64))).toThrow(/WAV/)
  })
})

describe('telephony simulation', () => {
  it('halves the rate, band-limits and stays recognisable', () => {
    const x = sine(1000, 16000, 0.5, 0.5)
    const y = simulateTelephony(x, 16000)
    expect(y).toHaveLength(4000)
    expect(toneAmplitude(y, 1000, 8000)).toBeGreaterThan(0.42)
  })
  it('removes content a phone line cannot carry', () => {
    const y = simulateTelephony(sine(6000, 16000, 0.5, 0.5), 16000)
    expect(rms(y)).toBeLessThan(0.02)
  })
  it('emits one byte per 8 kHz sample when streamed', () => {
    const enc = new TelephonyEncoder(16000)
    let n = 0
    for (let i = 0; i < 50; i++) n += enc.encode(new Float32Array(320)).length
    expect(n).toBe(8000)
    expect(decodeTelephony(new Uint8Array([0xff, 0xff]))).toEqual(new Float32Array([0, 0]))
  })
})
