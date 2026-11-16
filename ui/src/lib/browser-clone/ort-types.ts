/**
 * The small part of the ONNX Runtime Web API the cloning pipelines use, so
 * they can run against the real runtime in the worker and against fakes in
 * the unit tests.
 */

export type TensorData = Float32Array | BigInt64Array | Uint8Array | Int32Array | Uint16Array

export interface OrtTensor {
  readonly type: string
  readonly data: TensorData
  readonly dims: readonly number[]
  /** `gpu-buffer` for a tensor left on the GPU (WebGPU), which is then freed with `dispose`. */
  readonly location?: string
  dispose?: () => void
}

export interface OrtSession {
  readonly inputNames: readonly string[]
  readonly outputNames: readonly string[]
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>
  release?: () => Promise<void>
}

export type MakeTensor = (
  type: 'float32' | 'int64' | 'int32' | 'bool',
  data: TensorData,
  dims: readonly number[],
) => OrtTensor

/** A seeded random source (mulberry32) with normal draws, so a run can be repeated. */
export class Random {
  private s: number
  private spare: number | null = null

  constructor(seed = Date.now()) {
    this.s = seed >>> 0
  }

  uniform(): number {
    let t = (this.s += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }

  normal(): number {
    if (this.spare != null) {
      const v = this.spare
      this.spare = null
      return v
    }
    let u = 0
    while (u === 0) u = this.uniform()
    const v = this.uniform()
    const r = Math.sqrt(-2 * Math.log(u))
    this.spare = r * Math.sin(2 * Math.PI * v)
    return r * Math.cos(2 * Math.PI * v)
  }
}
