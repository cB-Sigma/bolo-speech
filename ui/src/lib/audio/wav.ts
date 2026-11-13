import { alawToLinear, mulawToLinear } from './g711'
import { floatToInt16, mixToMono } from './pcm'

export interface DecodedAudio {
  sampleRate: number
  /** Mono samples in [-1, 1]. */
  samples: Float32Array
}

/** Encode mono float samples as a 16-bit PCM WAV file. */
export function encodeWav(samples: Float32Array, sampleRate: number): Uint8Array {
  const pcm = floatToInt16(samples)
  const out = new Uint8Array(44 + pcm.length * 2)
  const view = new DataView(out.buffer)
  const ascii = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i))
  }
  ascii(0, 'RIFF')
  view.setUint32(4, 36 + pcm.length * 2, true)
  ascii(8, 'WAVE')
  ascii(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  ascii(36, 'data')
  view.setUint32(40, pcm.length * 2, true)
  for (let i = 0; i < pcm.length; i++) view.setInt16(44 + i * 2, pcm[i] as number, true)
  return out
}

/**
 * Decode a RIFF/WAVE file (PCM 8/16/24/32-bit, IEEE float, G.711) to mono
 * float. Throws on anything that is not a WAV this parser understands, so
 * callers can fall back to `AudioContext.decodeAudioData` for other formats.
 */
export function decodeWav(data: ArrayBuffer | Uint8Array): DecodedAudio {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data)
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const tag = (o: number) => String.fromCharCode(bytes[o]!, bytes[o + 1]!, bytes[o + 2]!, bytes[o + 3]!)
  if (bytes.length < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') throw new Error('Not a WAV file')
  let format = 0
  let channels = 1
  let sampleRate = 0
  let bits = 16
  let dataOffset = -1
  let dataLength = 0
  let o = 12
  while (o + 8 <= bytes.length) {
    const id = tag(o)
    const size = view.getUint32(o + 4, true)
    if (id === 'fmt ') {
      format = view.getUint16(o + 8, true)
      channels = view.getUint16(o + 10, true)
      sampleRate = view.getUint32(o + 12, true)
      bits = view.getUint16(o + 22, true)
      if (format === 0xfffe && size >= 26) format = view.getUint16(o + 32, true) // extensible
    } else if (id === 'data') {
      dataOffset = o + 8
      dataLength = Math.min(size, bytes.length - dataOffset)
      break
    }
    o += 8 + size + (size & 1)
  }
  if (dataOffset < 0 || !sampleRate || channels < 1) throw new Error('Malformed WAV file')
  const bytesPer = bits / 8
  const frames = Math.floor(dataLength / (bytesPer * channels))
  const chans = Array.from({ length: channels }, () => new Float32Array(frames))
  for (let i = 0; i < frames; i++) {
    for (let c = 0; c < channels; c++) {
      const p = dataOffset + (i * channels + c) * bytesPer
      let v: number
      if (format === 1) {
        if (bits === 8) v = (view.getUint8(p) - 128) / 128
        else if (bits === 16) v = view.getInt16(p, true) / 32768
        else if (bits === 24)
          v = ((view.getInt8(p + 2) << 16) | (view.getUint8(p + 1) << 8) | view.getUint8(p)) / 8388608
        else if (bits === 32) v = view.getInt32(p, true) / 2147483648
        else throw new Error(`Unsupported PCM bit depth: ${bits}`)
      } else if (format === 3) {
        v = bits === 64 ? view.getFloat64(p, true) : view.getFloat32(p, true)
      } else if (format === 7) {
        v = mulawToLinear(view.getUint8(p)) / 32768
      } else if (format === 6) {
        v = alawToLinear(view.getUint8(p)) / 32768
      } else {
        throw new Error(`Unsupported WAV format tag: ${format}`)
      }
      ;(chans[c] as Float32Array)[i] = v
    }
  }
  return { sampleRate, samples: mixToMono(chans) }
}

/** Decode any audio the browser understands; WAV is parsed directly to keep its sample rate. */
export async function decodeAudio(data: ArrayBuffer): Promise<DecodedAudio> {
  try {
    return decodeWav(data)
  } catch {
    const Ctx =
      globalThis.AudioContext ??
      (globalThis as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
    const ctx = new Ctx()
    try {
      const buf = await ctx.decodeAudioData(data.slice(0))
      const chans = Array.from({ length: buf.numberOfChannels }, (_, i) => buf.getChannelData(i))
      return { sampleRate: buf.sampleRate, samples: mixToMono(chans) }
    } finally {
      void ctx.close()
    }
  }
}
