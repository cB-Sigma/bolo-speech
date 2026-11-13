/** Float [-1, 1] → signed 16-bit, with clipping. */
export function floatToInt16(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length)
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i] as number))
    out[i] = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767)
  }
  return out
}

export function int16ToFloat(input: Int16Array): Float32Array {
  const out = new Float32Array(input.length)
  for (let i = 0; i < input.length; i++) out[i] = (input[i] as number) / 32768
  return out
}

/** Interpret little-endian 16-bit PCM bytes (copying, so odd offsets are fine). */
export function bytesToInt16(bytes: Uint8Array): Int16Array {
  const n = bytes.byteLength >> 1
  const view = new DataView(bytes.buffer, bytes.byteOffset, n * 2)
  const out = new Int16Array(n)
  for (let i = 0; i < n; i++) out[i] = view.getInt16(i * 2, true)
  return out
}

export function int16ToBytes(pcm: Int16Array): Uint8Array {
  const out = new Uint8Array(pcm.length * 2)
  const view = new DataView(out.buffer)
  for (let i = 0; i < pcm.length; i++) view.setInt16(i * 2, pcm[i] as number, true)
  return out
}

export function mixToMono(channels: Float32Array[]): Float32Array {
  if (channels.length === 0) return new Float32Array(0)
  if (channels.length === 1) return channels[0] as Float32Array
  const n = (channels[0] as Float32Array).length
  const out = new Float32Array(n)
  for (const ch of channels) for (let i = 0; i < n; i++) out[i] = (out[i] as number) + (ch[i] as number)
  for (let i = 0; i < n; i++) out[i] = (out[i] as number) / channels.length
  return out
}

/**
 * Re-chunks an arbitrary sequence of sample blocks into fixed-size frames.
 * `flush()` returns the remainder (shorter than a frame), or null.
 */
export class Framer {
  private pending: Float32Array
  private fill = 0
  readonly frameSize: number

  constructor(frameSize: number) {
    if (!(frameSize > 0)) throw new RangeError('frameSize must be positive')
    this.frameSize = frameSize
    this.pending = new Float32Array(frameSize)
  }

  push(block: Float32Array): Float32Array[] {
    const frames: Float32Array[] = []
    let offset = 0
    while (offset < block.length) {
      const take = Math.min(this.frameSize - this.fill, block.length - offset)
      this.pending.set(block.subarray(offset, offset + take), this.fill)
      this.fill += take
      offset += take
      if (this.fill === this.frameSize) {
        frames.push(this.pending)
        this.pending = new Float32Array(this.frameSize)
        this.fill = 0
      }
    }
    return frames
  }

  flush(): Float32Array | null {
    if (this.fill === 0) return null
    const rest = this.pending.slice(0, this.fill)
    this.fill = 0
    return rest
  }
}

/** Root-mean-square level of a block, 0..1. */
export function rms(block: Float32Array): number {
  if (block.length === 0) return 0
  let sum = 0
  for (let i = 0; i < block.length; i++) sum += (block[i] as number) * (block[i] as number)
  return Math.sqrt(sum / block.length)
}

export function peak(block: Float32Array): number {
  let p = 0
  for (let i = 0; i < block.length; i++) {
    const a = Math.abs(block[i] as number)
    if (a > p) p = a
  }
  return p
}

export function concatFloat(blocks: Float32Array[]): Float32Array {
  const n = blocks.reduce((a, b) => a + b.length, 0)
  const out = new Float32Array(n)
  let o = 0
  for (const b of blocks) {
    out.set(b, o)
    o += b.length
  }
  return out
}
