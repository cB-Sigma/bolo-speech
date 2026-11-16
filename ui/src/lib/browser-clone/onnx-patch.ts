/**
 * Pocket TTS's language model keeps its attention cache in fixed-size state
 * tensors of 1000 frames, which are copied into and out of the runtime on
 * every 80 ms frame. The graph itself works with any length (only the
 * declared input shapes say 1000; checked to give identical outputs), so the
 * page declares that dimension free and passes a cache just long enough for
 * the voice and one chunk of text: a third to a quarter of the copying.
 *
 * The change is made in place on the model bytes: a dimension written as
 * `dim_value: 1000` (3 bytes) becomes `dim_param: "L"` (also 3 bytes), so no
 * length prefix around it changes.
 */

interface Field {
  field: number
  wire: number
  /** Where the field's value starts (for length-delimited ones, after the length). */
  start: number
  end: number
}

function varint(buf: Uint8Array, pos: number): [number, number] {
  let result = 0
  let shift = 0
  for (;;) {
    const b = buf[pos++] as number
    result += (b & 0x7f) * 2 ** shift
    if (b < 0x80) return [result, pos]
    shift += 7
  }
}

function* fields(buf: Uint8Array, start: number, end: number): Generator<Field> {
  let pos = start
  while (pos < end) {
    const [tag, p] = varint(buf, pos)
    pos = p
    const field = Math.floor(tag / 8)
    const wire = tag & 7
    if (wire === 0) {
      const [, q] = varint(buf, pos)
      yield { field, wire, start: pos, end: q }
      pos = q
    } else if (wire === 2) {
      const [len, q] = varint(buf, pos)
      yield { field, wire, start: q, end: q + len }
      pos = q + len
    } else if (wire === 1) {
      yield { field, wire, start: pos, end: pos + 8 }
      pos += 8
    } else if (wire === 5) {
      yield { field, wire, start: pos, end: pos + 4 }
      pos += 4
    } else throw new Error(`Unsupported protobuf wire type ${wire}`)
  }
}

const sub = (buf: Uint8Array, f: Field, n: number) =>
  [...fields(buf, f.start, f.end)].filter((x) => x.field === n && x.wire === 2)

/**
 * Make every graph input and output dimension declared as `size` at `axis` of a `rank`-dimensional tensor free.
 * Returns how many dimensions were changed.
 */
export function freeDimension(model: Uint8Array, size = 1000, axis = 2, rank = 5): number {
  let changed = 0
  for (const graph of [...fields(model, 0, model.length)].filter((f) => f.field === 7 && f.wire === 2)) {
    for (const info of [...sub(model, graph, 11), ...sub(model, graph, 12)]) {
      for (const type of sub(model, info, 2))
        for (const tensor of sub(model, type, 1))
          for (const shape of sub(model, tensor, 2)) {
            const dims = sub(model, shape, 1)
            if (dims.length !== rank) continue
            const d = dims[axis] as Field
            const inner = [...fields(model, d.start, d.end)]
            const value = inner.find((x) => x.field === 1 && x.wire === 0)
            if (!value || inner.length !== 1 || varint(model, value.start)[0] !== size) continue
            // tag (1 byte) + value; the replacement must be the same length.
            const from = value.start - 1
            const len = value.end - from
            if (len !== 3) continue
            model[from] = 0x12 // field 2 (dim_param), length-delimited
            model[from + 1] = 0x01
            model[from + 2] = 0x4c // "L"
            changed++
          }
    }
  }
  return changed
}
