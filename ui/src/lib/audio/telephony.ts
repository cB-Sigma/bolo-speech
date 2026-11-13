import { decodeMulaw, encodeMulaw } from './g711'
import { floatToInt16, int16ToFloat } from './pcm'
import { Resampler } from './resample'

export const TELEPHONY_RATE = 8000

/**
 * Simulates a narrow-band phone line: band-limit and decimate to 8 kHz, then
 * G.711 μ-law. Feed it blocks at `inRate`; each call returns the 8-bit
 * payload exactly as a telephony gateway would deliver it.
 */
export class TelephonyEncoder {
  private readonly down: Resampler
  constructor(inRate: number) {
    this.down = new Resampler(inRate, TELEPHONY_RATE, 95)
  }
  encode(block: Float32Array): Uint8Array {
    return encodeMulaw(floatToInt16(this.down.process(block)))
  }
}

/** What the far end hears: μ-law bytes back to float at 8 kHz. */
export function decodeTelephony(bytes: Uint8Array): Float32Array {
  return int16ToFloat(decodeMulaw(bytes))
}

/** Convenience for whole buffers: returns the degraded audio at 8 kHz. */
export function simulateTelephony(samples: Float32Array, inRate: number): Float32Array {
  return decodeTelephony(new TelephonyEncoder(inRate).encode(samples))
}
