/**
 * ITU-T G.711 companding (μ-law and A-law), 16-bit linear PCM <-> 8-bit.
 * Table-free reference implementations following the classic Sun g711.c.
 */

const BIAS = 0x84
const CLIP = 32635
const SEG_END = [0xff, 0x1ff, 0x3ff, 0x7ff, 0xfff, 0x1fff, 0x3fff, 0x7fff]

function segment(value: number): number {
  for (let i = 0; i < SEG_END.length; i++) {
    if (value <= (SEG_END[i] as number)) return i
  }
  return SEG_END.length
}

export function linearToMulaw(sample: number): number {
  let pcm = Math.max(-32768, Math.min(32767, sample | 0))
  const sign = pcm < 0 ? 0x80 : 0
  if (pcm < 0) pcm = -pcm
  if (pcm > CLIP) pcm = CLIP
  pcm += BIAS
  let exponent = 7
  for (let mask = 0x4000; (pcm & mask) === 0 && exponent > 0; mask >>= 1) exponent--
  const mantissa = (pcm >> (exponent + 3)) & 0x0f
  return ~(sign | (exponent << 4) | mantissa) & 0xff
}

export function mulawToLinear(byte: number): number {
  const u = ~byte & 0xff
  const sign = u & 0x80
  const exponent = (u >> 4) & 0x07
  const mantissa = u & 0x0f
  const magnitude = (((mantissa << 3) + BIAS) << exponent) - BIAS
  return (sign ? -magnitude : magnitude) | 0
}

export function linearToAlaw(sample: number): number {
  let pcm = Math.max(-32768, Math.min(32767, sample | 0)) >> 3 // 13-bit
  let mask: number
  if (pcm >= 0) {
    mask = 0xd5
  } else {
    mask = 0x55
    pcm = -pcm - 1
  }
  const seg = segment(pcm << 3)
  if (seg >= 8) return 0x7f ^ mask
  let aval = seg << 4
  aval |= seg < 2 ? (pcm >> 1) & 0x0f : (pcm >> seg) & 0x0f
  return (aval ^ mask) & 0xff
}

export function alawToLinear(byte: number): number {
  const a = (byte ^ 0x55) & 0xff
  let t = (a & 0x0f) << 4
  const seg = (a & 0x70) >> 4
  if (seg === 0) t += 8
  else if (seg === 1) t += 0x108
  else {
    t += 0x108
    t <<= seg - 1
  }
  return a & 0x80 ? t : -t
}

function mapEncode(pcm: Int16Array, fn: (s: number) => number): Uint8Array {
  const out = new Uint8Array(pcm.length)
  for (let i = 0; i < pcm.length; i++) out[i] = fn(pcm[i] as number)
  return out
}

function mapDecode(bytes: Uint8Array, fn: (b: number) => number): Int16Array {
  const out = new Int16Array(bytes.length)
  for (let i = 0; i < bytes.length; i++) out[i] = fn(bytes[i] as number)
  return out
}

export const encodeMulaw = (pcm: Int16Array): Uint8Array => mapEncode(pcm, linearToMulaw)
export const decodeMulaw = (bytes: Uint8Array): Int16Array => mapDecode(bytes, mulawToLinear)
export const encodeAlaw = (pcm: Int16Array): Uint8Array => mapEncode(pcm, linearToAlaw)
export const decodeAlaw = (bytes: Uint8Array): Int16Array => mapDecode(bytes, alawToLinear)
