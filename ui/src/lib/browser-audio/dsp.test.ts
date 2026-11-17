import { describe, expect, it } from 'vitest'
import ref from '../fixtures/server-features.json'
import { fbank60, RealFft, reflectPad, resampleAligned, whisperLogMel } from './dsp'
import { onnxMetadata } from './onnx-meta'

/** The same test signal as scripts/parity/export_cases.py. */
function signal(n = 16000): Float32Array {
  return Float32Array.from(
    { length: n },
    (_, i) => Math.sin(i / 7) * 0.3 + 0.1 * Math.sin(i / 3.1) + 0.05 * Math.sin(i / 0.9),
  )
}

function naive(x: number[]): [number[], number[]] {
  const n = x.length
  const re: number[] = []
  const im: number[] = []
  for (let k = 0; k <= n / 2; k++) {
    let a = 0
    let b = 0
    for (let t = 0; t < n; t++) {
      a += (x[t] as number) * Math.cos((2 * Math.PI * k * t) / n)
      b -= (x[t] as number) * Math.sin((2 * Math.PI * k * t) / n)
    }
    re.push(a)
    im.push(b)
  }
  return [re, im]
}

describe('signal processing', () => {
  for (const n of [8, 320, 400, 512]) {
    it(`a ${n}-point FFT agrees with the DFT`, () => {
      const x = Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) + (i % 5) * 0.1)
      const fft = new RealFft(n)
      const re = new Float64Array(fft.bins)
      const im = new Float64Array(fft.bins)
      fft.forward(x, re, im)
      const [r, m] = naive(x)
      for (let k = 0; k < fft.bins; k++) {
        expect(re[k]).toBeCloseTo(r[k] as number, 6)
        expect(im[k]).toBeCloseTo(m[k] as number, 6)
      }
      {
        const back = new Float64Array(n)
        fft.inverse(re, im, back)
        for (let i = 0; i < n; i++) expect(back[i]).toBeCloseTo(x[i] as number, 9)
      }
    })
  }

  it('pads by reflection as numpy does', () => {
    expect([...reflectPad([1, 2, 3, 4], 2)]).toEqual([3, 2, 1, 2, 3, 4, 3, 2])
  })

  it('computes Whisper features as the server does (Smart Turn)', () => {
    const { data, frames } = whisperLogMel(signal(), true)
    expect([80, frames]).toEqual(ref.whisper.shape)
    const col = (t: number) => Array.from({ length: 80 }, (_, m) => data[m * frames + t] as number)
    col(0).forEach((v, m) => expect(v).toBeCloseTo(ref.whisper.col0[m] as number, 3))
    col(50).forEach((v, m) => expect(v).toBeCloseTo(ref.whisper.col50[m] as number, 3))
  })

  it('computes SpeechBrain’s 60-band filterbank as the server does (VoxLingua107)', () => {
    const { data, frames } = fbank60(signal())
    expect([frames, 60]).toEqual(ref.fbank60.shape)
    for (let m = 0; m < 60; m++) {
      expect(data[m]).toBeCloseTo(ref.fbank60.row0[m] as number, 2)
      expect(data[30 * 60 + m]).toBeCloseTo(ref.fbank60.row30[m] as number, 2)
    }
  })

  it('reads metadata_props from an ONNX model', () => {
    // ModelProto { ir_version: 8 (field 1), metadata_props (field 14): { key: "n_fft", value: "512" } }
    const enc = new TextEncoder()
    const k = enc.encode('n_fft')
    const v = enc.encode('512')
    const entry = [0x0a, k.length, ...k, 0x12, v.length, ...v]
    const bytes = new Uint8Array([0x08, 0x08, 0x72, entry.length, ...entry])
    expect(onnxMetadata(bytes)).toEqual({ n_fft: '512' })
  })
})

describe('aligned resampling', () => {
  it('goes from 16 to 48 kHz and back without moving the signal', () => {
    const x = Float32Array.from({ length: 16000 }, (_, i) => Math.sin(i / 7) * 0.3 + Math.sin(i / 2.3) * 0.1)
    const back = resampleAligned(resampleAligned(x, 16000, 48000), 48000, 16000)
    expect(back.length).toBe(x.length)
    let num = 0
    let den = 0
    for (let i = 200; i < 15800; i++) {
      num += (x[i] as number) ** 2
      den += ((back[i] as number) - (x[i] as number)) ** 2
    }
    expect(10 * Math.log10(num / den)).toBeGreaterThan(40)
  })
})
