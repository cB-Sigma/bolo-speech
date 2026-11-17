/**
 * The custom metadata of an ONNX model (`metadata_props`: DPDFNet keeps its initial state there, sherpa-onnx's
 * exports their STFT settings), read straight from the protobuf: ONNX Runtime Web does not expose it. Only the
 * top level of the ModelProto is walked; the graph and its weights are skipped by their length.
 */

function varint(b: Uint8Array, at: number): [number, number] {
  let value = 0
  let shift = 0
  let i = at
  for (;;) {
    const byte = b[i++] as number
    value += (byte & 0x7f) * 2 ** shift
    if (byte < 0x80) return [value, i]
    shift += 7
    if (shift > 63) throw new Error('bad varint')
  }
}

/** Skip a field's value of `wire` type at `at`; returns where the next field starts. */
function skip(b: Uint8Array, at: number, wire: number): number {
  if (wire === 0) return varint(b, at)[1]
  if (wire === 1) return at + 8
  if (wire === 5) return at + 4
  if (wire === 2) {
    const [len, i] = varint(b, at)
    return i + len
  }
  throw new Error(`unsupported wire type ${wire}`)
}

function entry(b: Uint8Array, start: number, end: number): [string, string] {
  const text = new TextDecoder()
  let key = ''
  let value = ''
  let i = start
  while (i < end) {
    const [tag, j] = varint(b, i)
    const field = Math.floor(tag / 8)
    const wire = tag & 7
    if (wire === 2 && (field === 1 || field === 2)) {
      const [len, k] = varint(b, j)
      const s = text.decode(b.subarray(k, k + len))
      if (field === 1) key = s
      else value = s
      i = k + len
    } else i = skip(b, j, wire)
  }
  return [key, value]
}

export function onnxMetadata(model: ArrayBuffer | Uint8Array): Record<string, string> {
  const b = model instanceof Uint8Array ? model : new Uint8Array(model)
  const out: Record<string, string> = {}
  let i = 0
  while (i < b.length) {
    const [tag, j] = varint(b, i)
    const field = Math.floor(tag / 8)
    const wire = tag & 7
    if (field === 14 && wire === 2) {
      const [len, k] = varint(b, j)
      const [key, value] = entry(b, k, k + len)
      out[key] = value
      i = k + len
    } else i = skip(b, j, wire)
  }
  return out
}
