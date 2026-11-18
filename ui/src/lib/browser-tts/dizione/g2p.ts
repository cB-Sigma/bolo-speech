/** The unknown-word model: letters -> CMUdict phonemes (`dizione/g2p.py`): letter embeddings, residual convolutions,
 * a bidirectional GRU and a CTC output read greedily. One forward pass per word. */

import { strip } from './py'

export interface G2PSettings {
  letters: string
  symbols: string[]
  slots: number
  convs: number
  layers: number
}

export interface Tensor {
  data: Float32Array
  shape: number[]
}

/** The arrays of an uncompressed `.npz` (a zip of `.npy` files, float32). */
export function readNpz(bytes: Uint8Array): Map<string, Tensor> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const out = new Map<string, Tensor>()
  const decoder = new TextDecoder()
  let at = 0
  while (at + 30 <= bytes.length && view.getUint32(at, true) === 0x04034b50) {
    const method = view.getUint16(at + 8, true)
    const flags = view.getUint16(at + 6, true)
    let size = view.getUint32(at + 18, true)
    const nameLength = view.getUint16(at + 26, true)
    const extraLength = view.getUint16(at + 28, true)
    const name = decoder.decode(bytes.subarray(at + 30, at + 30 + nameLength))
    const start = at + 30 + nameLength + extraLength
    if (method !== 0) throw new Error(`${name}: compressed arrays are not supported`)
    if (size === 0xffffffff || flags & 8) {
      // zip64 or a size given after the data: read it from the extra field
      const extra = new DataView(bytes.buffer, bytes.byteOffset + at + 30 + nameLength, extraLength)
      for (let e = 0; e + 4 <= extraLength;) {
        const id = extra.getUint16(e, true)
        const length = extra.getUint16(e + 2, true)
        if (id === 1) size = Number(extra.getBigUint64(e + 4, true))
        e += 4 + length
      }
    }
    out.set(name.replace(/\.npy$/, ''), readNpy(bytes.subarray(start, start + size)))
    at = start + size
  }
  return out
}

function readNpy(bytes: Uint8Array): Tensor {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const major = bytes[6]!
  const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true)
  const offset = major === 1 ? 10 : 12
  const header = new TextDecoder().decode(bytes.subarray(offset, offset + headerLength))
  if (!header.includes("'<f4'")) throw new Error(`only float32 arrays are supported: ${header}`)
  if (header.includes("'fortran_order': True")) throw new Error('Fortran-ordered arrays are not supported')
  const shape = (/'shape': \(([^)]*)\)/.exec(header)?.[1] ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number)
  const count = shape.reduce((a, b) => a * b, 1)
  const start = offset + headerLength
  const data = new Float32Array(count)
  const values = new DataView(bytes.buffer, bytes.byteOffset + start, count * 4)
  for (let i = 0; i < count; i++) data[i] = values.getFloat32(i * 4, true)
  return { data, shape }
}

type Matrix = { data: Float64Array; rows: number; cols: number }

const sigmoid = (x: number): number => 1.0 / (1.0 + Math.exp(-x))

/** x (rows x in) @ weight.T (weight: out x in) + bias. */
function linear(x: Matrix, weight: Tensor, bias: Tensor | null): Matrix {
  const [outSize, inSize] = weight.shape as [number, number]
  const out = new Float64Array(x.rows * outSize)
  for (let r = 0; r < x.rows; r++) {
    for (let o = 0; o < outSize; o++) {
      let sum = 0
      const w = o * inSize
      const xr = r * inSize
      for (let k = 0; k < inSize; k++) sum += x.data[xr + k]! * weight.data[w + k]!
      out[r * outSize + o] = sum + (bias ? bias.data[o]! : 0)
    }
  }
  return { data: out, rows: x.rows, cols: outSize }
}

function layerNorm(x: Matrix, weight: Tensor, bias: Tensor): Matrix {
  const out = new Float64Array(x.data.length)
  for (let r = 0; r < x.rows; r++) {
    let mean = 0
    for (let c = 0; c < x.cols; c++) mean += x.data[r * x.cols + c]!
    mean /= x.cols
    let variance = 0
    for (let c = 0; c < x.cols; c++) variance += (x.data[r * x.cols + c]! - mean) ** 2
    variance /= x.cols
    const scale = Math.sqrt(variance + 1e-5)
    for (let c = 0; c < x.cols; c++)
      out[r * x.cols + c] = ((x.data[r * x.cols + c]! - mean) / scale) * weight.data[c]! + bias.data[c]!
  }
  return { data: out, rows: x.rows, cols: x.cols }
}

/** A convolution over frames with zero padding; weight: (out, in, kernel). */
function conv(x: Matrix, weight: Tensor, bias: Tensor): Matrix {
  const [outSize, inSize, kernel] = weight.shape as [number, number, number]
  const pad = Math.floor(kernel / 2)
  const out = new Float64Array(x.rows * outSize)
  for (let t = 0; t < x.rows; t++) {
    for (let o = 0; o < outSize; o++) {
      let sum = bias.data[o]!
      for (let k = 0; k < kernel; k++) {
        const source = t + k - pad
        if (source < 0 || source >= x.rows) continue
        for (let c = 0; c < inSize; c++)
          sum += x.data[source * inSize + c]! * weight.data[(o * inSize + c) * kernel + k]!
      }
      out[t * outSize + o] = sum
    }
  }
  return { data: out, rows: x.rows, cols: outSize }
}

function gru(x: Matrix, wIh: Tensor, wHh: Tensor, bIh: Tensor, bHh: Tensor, reverse: boolean): Matrix {
  const size = wHh.shape[1]!
  const inputs = linear(x, wIh, bIh)
  let state = new Float64Array(size)
  const out = new Float64Array(x.rows * size)
  const order = [...Array(x.rows).keys()]
  if (reverse) order.reverse()
  for (const t of order) {
    const recurrent = linear({ data: state, rows: 1, cols: size }, wHh, bHh).data
    const row = t * 3 * size
    const next = new Float64Array(size)
    for (let j = 0; j < size; j++) {
      const reset = sigmoid(inputs.data[row + j]! + recurrent[j]!)
      const update = sigmoid(inputs.data[row + size + j]! + recurrent[size + j]!)
      const candidate = Math.tanh(inputs.data[row + 2 * size + j]! + reset * recurrent[2 * size + j]!)
      next[j] = (1.0 - update) * candidate + update * state[j]!
    }
    state = next
    out.set(state, t * size)
  }
  return { data: out, rows: x.rows, cols: size }
}

export class G2P {
  readonly letters: Map<string, number>
  readonly settings: G2PSettings
  readonly weights: Map<string, Tensor>

  constructor(settings: G2PSettings, weights: Map<string, Tensor>) {
    this.settings = settings
    this.weights = weights
    this.letters = new Map([...settings.letters].map((letter, index) => [letter, index + 1]))
  }

  private w(name: string): Tensor {
    const found = this.weights.get(name)
    if (!found) throw new Error(`the unknown-word model has no array ${name}`)
    return found
  }

  /** The word reduced to the letters the model knows (accents removed). */
  spellable(word: string): string {
    const plain = word.toLowerCase().normalize('NFKD')
    return strip([...plain].filter((c) => this.letters.has(c)).join(''), "'")
  }

  /** The best label of every frame. */
  labels(word: string): number[] {
    const slots = this.settings.slots
    const embed = this.w('embed__weight')
    const slot = this.w('slot')
    const width = embed.shape[1]!
    const ids = [...word].map((c) => this.letters.get(c)!)
    let x: Matrix = { data: new Float64Array(ids.length * slots * width), rows: ids.length * slots, cols: width }
    ids.forEach((id, n) => {
      for (let s = 0; s < slots; s++)
        for (let c = 0; c < width; c++)
          x.data[(n * slots + s) * width + c] = embed.data[id * width + c]! + slot.data[s * width + c]!
    })
    for (let block = 0; block < this.settings.convs; block++) {
      const normed = layerNorm(x, this.w(`norms__${block}__weight`), this.w(`norms__${block}__bias`))
      const y = conv(normed, this.w(`convs__${block}__weight`), this.w(`convs__${block}__bias`))
      const sum = new Float64Array(x.data.length)
      for (let i = 0; i < sum.length; i++) sum[i] = x.data[i]! + Math.max(y.data[i]!, 0.0)
      x = { data: sum, rows: x.rows, cols: x.cols }
    }
    for (let layer = 0; layer < this.settings.layers; layer++) {
      const parts = (['', '_reverse'] as const).map((suffix) =>
        gru(
          x,
          this.w(`gru__weight_ih_l${layer}${suffix}`),
          this.w(`gru__weight_hh_l${layer}${suffix}`),
          this.w(`gru__bias_ih_l${layer}${suffix}`),
          this.w(`gru__bias_hh_l${layer}${suffix}`),
          suffix === '_reverse',
        ),
      )
      const cols = parts[0]!.cols + parts[1]!.cols
      const joined = new Float64Array(x.rows * cols)
      for (let r = 0; r < x.rows; r++) {
        joined.set(parts[0]!.data.subarray(r * parts[0]!.cols, (r + 1) * parts[0]!.cols), r * cols)
        joined.set(parts[1]!.data.subarray(r * parts[1]!.cols, (r + 1) * parts[1]!.cols), r * cols + parts[0]!.cols)
      }
      x = { data: joined, rows: x.rows, cols }
    }
    const logits = linear(x, this.w('out__weight'), this.w('out__bias'))
    const out: number[] = []
    for (let r = 0; r < logits.rows; r++) {
      let best = 0
      for (let c = 1; c < logits.cols; c++)
        if (logits.data[r * logits.cols + c]! > logits.data[r * logits.cols + best]!) best = c
      out.push(best)
    }
    return out
  }

  /** CMUdict symbols with stress digits for `word`. */
  predict(wordIn: string): string[] {
    const word = this.spellable(wordIn)
    if (!word) return []
    const out: string[] = []
    let previous = 0
    for (const label of this.labels(word)) {
      if (label !== previous && label !== 0) out.push(this.settings.symbols[label - 1]!)
      previous = label
    }
    return out
  }
}
