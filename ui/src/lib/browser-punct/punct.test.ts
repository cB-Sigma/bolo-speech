// @vitest-environment node
import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { gunzipSync } from 'node:zlib'
import { describe, expect, it } from 'vitest'
import {
  argmaxRows,
  batch,
  decodeSentences,
  encodeSentences,
  MAX_SEQ,
  punctuate,
  pyIsUpper,
  pyStrip,
  type PunctModel,
} from './punct'
import { Ssentencepiece } from './tokenizer'

/**
 * The page's port against the server: fixtures/server-punct-cases.json holds, for 285 transcripts (sentences of
 * the project's documentation without punctuation or capitals, and edge cases), what the server's
 * punct-en-cnn-bilstm returns and the token ids, case and marks in between (ui/scripts/punctuation/export_cases.py).
 * With the model's answers from the fixture, the TypeScript steps must give the same ids and the same text.
 */

interface Case {
  text: string
  expected: string
  chunks?: { ids: number[]; valid: string }[]
  case?: string
  punct?: string
}

const here = import.meta.dirname
const { cases } = JSON.parse(readFileSync(path.join(here, 'fixtures/server-punct-cases.json'), 'utf8')) as {
  cases: Case[]
}
const vocab = gunzipSync(readFileSync(path.join(here, 'fixtures/bpe.vocab.gz'))).toString('utf8')
const sp = Ssentencepiece.fromVocab(vocab)
const digits = (s: string | undefined) => Array.from(s ?? '', Number)

/** The model as the fixture recorded it, after checking that it was given the same input. */
function recorded(c: Case): PunctModel {
  return (input) => {
    const want = c.chunks ?? []
    expect(input.n).toBe(want.length)
    want.forEach((chunk, k) => {
      expect(Array.from(input.tokenIds.subarray(k * MAX_SEQ, k * MAX_SEQ + chunk.ids.length))).toEqual(chunk.ids)
      expect(input.tokenIds.subarray(k * MAX_SEQ + chunk.ids.length, (k + 1) * MAX_SEQ).every((x) => x === 0)).toBe(
        true,
      )
      expect(Array.from(input.validIds.subarray(k * MAX_SEQ, k * MAX_SEQ + chunk.valid.length))).toEqual(
        digits(chunk.valid),
      )
    })
    return Promise.resolve({ cases: digits(c.case), marks: digits(c.punct) })
  }
}

describe('punctuation restoration, as the server', () => {
  it('reads the whole vocabulary', () => {
    expect(sp.size).toBe(5000)
    expect(cases.length).toBeGreaterThan(250)
  })

  it('gives the server’s token ids and text for every transcript', async () => {
    const wrong: string[] = []
    for (const c of cases) {
      let called = false
      const model: PunctModel = (input) => {
        called = true
        return recorded(c)(input)
      }
      const out = await punctuate(sp, model, c.text)
      if (out !== c.expected) wrong.push(`${JSON.stringify(c.text)}: ${JSON.stringify(out)}`)
      expect(called).toBe(c.chunks != null)
    }
    expect(wrong).toEqual([])
  })

  it('cuts long text into chunks of 200 tokens, never splitting a word', () => {
    const long = cases.find((c) => (c.chunks?.length ?? 0) > 2)
    expect(long).toBeDefined()
    const chunks = encodeSentences(sp, long?.text ?? '')
    for (const c of chunks) {
      expect(c.ids.length).toBeLessThanOrEqual(MAX_SEQ)
      expect(c.ids[0]).toBe(1)
      expect(c.ids.at(-1)).toBe(2)
    }
    const input = batch(chunks)
    expect(Array.from(input.labelLens)).toEqual(chunks.map((c) => c.valid.filter(Boolean).length))
  })

  it('changes the case of ASCII letters only, as the C locale', () => {
    expect(decodeSentences('café émile ok', [1, 2, 2], [0, 1, 3])).toBe('CAFé émile, Ok?')
    expect(decodeSentences('a b', [2], [2])).toBe('A b.')
  })

  it('follows Python’s strip and isupper', () => {
    expect(pyStrip('　 x  \x1c')).toBe('x')
    expect(pyStrip('﻿x')).toBe('﻿x')
    expect(pyIsUpper('HELLO 42')).toBe(true)
    expect(pyIsUpper('HELLO ß')).toBe(false)
    expect(pyIsUpper('42')).toBe(false)
  })

  it('returns text with a word longer than a chunk as it was', async () => {
    const never: PunctModel = () => Promise.reject(new Error('not for this text'))
    const long = `it says ${'a'.repeat(250)} and more`
    expect(await punctuate(sp, never, long)).toBe(long)
    expect(await punctuate(sp, never, '  ')).toBe('  ')
  })

  it('takes the first of equal logits, as std::max_element', () => {
    expect(Array.from(argmaxRows([1, 3, 3, 0, 5, 5, 5, 5], 4))).toEqual([1, 0])
  })
})

/**
 * The real model in ONNX Runtime Web (WebAssembly) in Node, against the server's text. Runs only with
 * BOLO_PUNCT_DIR, the unpacked sherpa-onnx-online-punct-en-2024-08-06 (model.int8.onnx).
 */
const DIR = process.env.BOLO_PUNCT_DIR
describe.runIf(DIR && existsSync(path.join(DIR, 'model.int8.onnx')))('punctuation model in ONNX Runtime Web', () => {
  it('writes the server’s text', async () => {
    const ort = await import('onnxruntime-web')
    const session = await ort.InferenceSession.create(
      new Uint8Array(readFileSync(path.join(DIR as string, 'model.int8.onnx'))),
      {
        executionProviders: ['wasm'],
      },
    )
    const model: PunctModel = async (input) => {
      const out = await session.run({
        token_ids: new ort.Tensor('int32', input.tokenIds, [input.n, MAX_SEQ]),
        valid_ids: new ort.Tensor('int32', input.validIds, [input.n, MAX_SEQ]),
        label_lens: new ort.Tensor('int32', input.labelLens, [input.n]),
      })
      return {
        cases: argmaxRows(out.active_case_logits?.data as Float32Array, 4),
        marks: argmaxRows(out.active_punct_logits?.data as Float32Array, 4),
      }
    }
    const wrong: string[] = []
    for (const c of cases) {
      const out = await punctuate(sp, model, c.text)
      if (out !== c.expected) wrong.push(c.text)
    }
    // ONNX Runtime Web 1.31 and sherpa-onnx's ONNX Runtime round the 8-bit graph a little differently: on 230
    // times "x" (no real text) one word's logits are close enough to tie and come out the other way.
    expect(wrong.filter((t) => !/^(x )+$/.test(t))).toEqual([])
  }, 120_000)
})
