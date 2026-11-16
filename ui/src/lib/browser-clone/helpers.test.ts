import { describe, expect, it } from 'vitest'
import { MAX_SECONDS, prepareReference, fromInt16, toInt16 } from './audio'
import { CLONE_MODELS, chatterboxFiles, cloneModel, pocketFiles } from './models'
import { freeDimension } from './onnx-patch'
import { memoryBackend, SavedVoices, type SavedVoice, type VoiceBackend } from './saved-voices'
import { cosine, DIFFERENT_SPEAKER, forEmbedding, SAME_SPEAKER, verdictFor } from './similarity'
import { chatterboxTokenBudget, chunkForChatterbox, wordErrorRate } from './text'

describe('voice-cloning catalogue', () => {
  it('pins every model to a revision and states size, licence and languages', () => {
    for (const m of CLONE_MODELS) {
      expect(m.revision).toMatch(/^[0-9a-f]{40}$/)
      expect(m.bytes).toBeGreaterThan(1e8)
      expect(m.peakBytes).toBeGreaterThan(m.bytes)
      expect(m.license).toMatch(/^(CC-BY-4.0|MIT|Apache-2.0)$/)
      expect(m.languages.length).toBeGreaterThan(0)
      expect(m.needsTranscript).toBe(false)
    }
    expect(new Set(CLONE_MODELS.map((m) => m.id)).size).toBe(CLONE_MODELS.length)
    expect(cloneModel('nope').id).toBe('pocket-en')
  })

  it('says that Chatterbox output carries no watermark here', () => {
    for (const m of CLONE_MODELS.filter((x) => x.family === 'chatterbox')) expect(m.watermark).toMatch(/no watermark/)
  })

  it('lists the files each family fetches', () => {
    const p = pocketFiles(cloneModel('pocket-de'))
    expect(p.map((f) => f.path)).toContain('onnx/german/flow_lm_main_int8.onnx')
    const turbo = chatterboxFiles(cloneModel('chatterbox-turbo'), 'q4')
    expect(turbo).toContain('onnx/language_model_q4.onnx_data')
    const multi = chatterboxFiles(cloneModel('chatterbox-multilingual'), 'fp32')
    expect(multi).toContain('onnx/language_model.onnx')
    expect(multi.some((f) => f.endsWith('_data'))).toBe(false)
  })
})

describe('reference preparation', () => {
  it('trims silence, cuts to the maximum and lifts a quiet recording', () => {
    const rate = 24_000
    const x = new Float32Array(rate * (MAX_SECONDS + 5))
    for (let i = rate; i < x.length - rate; i++) x[i] = 0.1 * Math.sin(i / 10)
    const p = prepareReference(x, rate)
    expect(p.seconds).toBeCloseTo(MAX_SECONDS, 3)
    expect(p.notes.join(' ')).toMatch(/Cut to/)
    expect(p.notes.join(' ')).toMatch(/quiet/)
    expect(Math.max(...p.audio.slice(rate * 2, rate * 3))).toBeGreaterThan(0.5)
  })

  it('flags a short reference', () => {
    const p = prepareReference(Float32Array.from({ length: 24_000 * 2 }, (_, i) => 0.5 * Math.sin(i / 5)))
    expect(p.tooShort).toBe(true)
  })

  it('round-trips 16-bit PCM', () => {
    const a = Float32Array.of(0, 0.5, -0.5, 1, -1)
    const b = fromInt16(toInt16(a))
    for (let i = 0; i < a.length; i++) expect(b[i]).toBeCloseTo(a[i] as number, 3)
  })
})

describe('speaker similarity', () => {
  it('is the cosine of the embeddings, with plain thresholds', () => {
    expect(cosine(Float32Array.of(1, 0), Float32Array.of(1, 0))).toBeCloseTo(1)
    expect(cosine(Float32Array.of(1, 0), Float32Array.of(0, 1))).toBeCloseTo(0)
    expect(cosine(Float32Array.of(0, 0), Float32Array.of(0, 1))).toBe(0)
    expect(verdictFor(SAME_SPEAKER + 0.01)).toBe('same')
    expect(verdictFor((SAME_SPEAKER + DIFFERENT_SPEAKER) / 2)).toBe('unclear')
    expect(verdictFor(DIFFERENT_SPEAKER - 0.01)).toBe('different')
  })

  it('feeds the embedding model 16 kHz audio from the middle of long clips', () => {
    const x = new Float32Array(24_000 * 40)
    const y = forEmbedding(x, 24_000, 20)
    expect(y.length).toBe(16_000 * 20)
  })
})

describe('text', () => {
  it('splits text for Chatterbox at sentence ends, packing short ones', () => {
    expect(chunkForChatterbox('One. Two! Three?', 40)).toEqual(['One. Two! Three?'])
    const long = 'This is a sentence that goes on. '.repeat(10)
    const chunks = chunkForChatterbox(long, 80)
    expect(chunks.length).toBeGreaterThan(3)
    for (const c of chunks) expect(c.length).toBeLessThanOrEqual(80)
    expect(chunkForChatterbox('नमस्ते। आप कैसे हैं?', 12)).toEqual(['नमस्ते।', 'आप कैसे हैं?'])
    expect(chunkForChatterbox('   ')).toEqual([])
  })

  it('allows enough speech tokens, capped', () => {
    expect(chatterboxTokenBudget('hello')).toBe(65)
    expect(chatterboxTokenBudget('x'.repeat(5000))).toBe(1000)
  })

  it('scores word error rate without case or punctuation', () => {
    expect(wordErrorRate('Hello, world.', 'hello world')).toBe(0)
    expect(wordErrorRate('half past ten', '"\'Half-past ten.')).toBe(0)
    expect(wordErrorRate('the cat sat', 'the cat sat down')).toBeCloseTo(1 / 3)
    expect(wordErrorRate('a b c d', 'a x c')).toBeCloseTo(0.5)
    expect(wordErrorRate('', '')).toBe(0)
  })
})

describe('ONNX cache-length patch', () => {
  /** A ModelProto with one graph input and output of shape [2, 1, 1000, 16, 64] and one [1, 1000]. */
  function model(): Uint8Array {
    const enc = (field: number, body: number[]) => {
      const len: number[] = []
      let n = body.length
      while (n >= 0x80) {
        len.push((n & 0x7f) | 0x80)
        n >>= 7
      }
      len.push(n)
      return [field * 8 + 2, ...len, ...body]
    }
    const dim = (v: number) => enc(1, v >= 128 ? [0x08, (v & 0x7f) | 0x80, v >> 7] : [0x08, v])
    const info = (name: string, dims: number[]) =>
      enc(1, [...new TextEncoder().encode(name)]).concat(enc(2, enc(1, [0x08, 0x01, ...enc(2, dims.flatMap(dim))])))
    const graph = [
      ...enc(11, info('state', [2, 1, 1000, 16, 64])),
      ...enc(11, info('other', [1, 1000])),
      ...enc(12, info('out_state', [2, 1, 1000, 16, 64])),
    ]
    return Uint8Array.from([0x08, 0x07, ...enc(7, graph)])
  }

  it('frees the cache dimension of 5-D inputs and outputs only, without changing the size', () => {
    const m = model()
    const before = m.length
    expect(freeDimension(m)).toBe(2)
    expect(m.length).toBe(before)
    let found = 0
    for (let i = 0; i + 2 < m.length; i++) if (m[i] === 0x12 && m[i + 1] === 0x01 && m[i + 2] === 0x4c) found++
    expect(found).toBe(2)
    // A second pass finds nothing left to change.
    expect(freeDimension(m)).toBe(0)
  })
})

describe('saved voices', () => {
  const voice = (id: string, createdAt: number): SavedVoice => ({
    id,
    name: id,
    createdAt,
    sampleRate: 24_000,
    pcm: Int16Array.of(1, 2, 3),
    transcript: 'hi',
    language: 'en',
    source: 'mic',
  })

  it('saves, lists newest first and deletes', async () => {
    const v = new SavedVoices(memoryBackend())
    await v.load()
    await v.save(voice('a', 1))
    await v.save(voice('b', 2))
    expect(v.store.get().voices.map((x) => x.id)).toEqual(['b', 'a'])
    expect(v.get('a')?.pcm).toEqual(Int16Array.of(1, 2, 3))
    await v.remove('b')
    expect(v.store.get().voices.map((x) => x.id)).toEqual(['a'])
    expect(v.store.get().persistent).toBe(true)
  })

  it('falls back to memory (this visit only) when storage fails', async () => {
    const broken: VoiceBackend = {
      list: () => Promise.reject(new Error('blocked')),
      put: () => Promise.reject(new Error('quota')),
      delete: () => Promise.reject(new Error('blocked')),
    }
    const v = new SavedVoices(broken)
    await v.load()
    expect(v.store.get().persistent).toBe(false)
    await v.save(voice('a', 1))
    expect(v.store.get().voices).toHaveLength(1)
  })

  it('reports no storage when IndexedDB is missing', () => {
    const v = new SavedVoices(null)
    expect(v.store.get().persistent).toBe(false)
  })
})
