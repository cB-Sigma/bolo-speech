import { describe, expect, it } from 'vitest'
import { cloneModel } from './models'
import {
  buildRequestRows,
  decodeBudget,
  frameUniforms,
  MAX_REFERENCE_SECONDS,
  MIN_DECODE_FRAMES,
  MossTts,
  upsample2x,
  type MossCodecMeta,
  type MossManifest,
  type MossTtsMeta,
} from './moss'
import type { MakeTensor, OrtSession, OrtTensor } from './ort-types'
import { CLONE_RATE } from './protocol'

const N_VQ = 3
const BOOK = 8
const HIDDEN = 4
const SAMPLES_PER_FRAME = 8 // at 48 kHz in the fake codec

const manifest: MossManifest = {
  tts_config: {
    n_vq: N_VQ,
    audio_pad_token_id: BOOK,
    audio_start_token_id: 6,
    audio_end_token_id: 7,
    audio_user_slot_token_id: 8,
    audio_assistant_slot_token_id: 9,
    audio_codebook_sizes: [BOOK, BOOK, BOOK],
  },
  prompt_templates: {
    user_prompt_prefix_token_ids: [100, 101],
    user_prompt_after_reference_token_ids: [102],
    assistant_prompt_prefix_token_ids: [103, 104],
  },
  generation_defaults: { max_new_frames: 20 },
}

const meta: MossTtsMeta = {
  onnx: {
    prefill_output_names: ['global_hidden', 'present_key_0'],
    decode_input_names: ['input_ids', 'past_valid_lengths', 'past_key_0'],
    decode_output_names: ['global_hidden', 'present_key_0'],
  },
}

const codec: MossCodecMeta = {
  codec_config: { sample_rate: 48_000, channels: 2, num_quantizers: N_VQ },
  streaming_decode: {
    transformer_offsets: [{ input_name: 'offset_0', output_name: 'offset_out_0', shape: [1] }],
    attention_caches: [],
  },
}

const tensor: MakeTensor = (type, data, dims) => ({ type, data, dims })
const session = (run: OrtSession['run']): OrtSession => ({ inputNames: [], outputNames: [], run })

/** Words as tokens. */
const tokenizer = {
  encode: (t: string) =>
    t
      .split(/\s+/)
      .filter(Boolean)
      .map((w) => w.length),
  count: (t: string) => t.split(/\s+/).filter(Boolean).length,
}

/**
 * Fake graphs: the transformer's "hidden state" is its cache length, the frame graph stops after `frames`
 * frames and returns codes that count up, the codec gives SAMPLES_PER_FRAME stereo samples per frame.
 */
function fakes(frames: number, opts: { gpu?: boolean } = {}) {
  const calls = {
    prefillRows: [] as number[],
    pastLengths: [] as number[],
    rows: [] as number[][],
    masks: [] as number[][],
    uniforms: [] as number[][],
    codec: [] as number[],
    offsets: [] as number[],
    encoderLength: 0,
    disposed: 0,
  }
  let made = 0
  const cache = (n: number): OrtTensor => ({
    type: 'float32',
    data: new Float32Array(n),
    dims: [1, n, 1, 1],
    ...(opts.gpu ? { location: 'gpu-buffer', dispose: () => void calls.disposed++ } : {}),
  })
  const sessions = {
    encoder: session(async (f) => {
      const n = (f.waveform as OrtTensor).dims[2] as number
      calls.encoderLength = n
      const count = Math.ceil(n / 3840)
      return {
        audio_codes: tensor('int32', new Int32Array(count * N_VQ).fill(5), [1, count, N_VQ]),
        audio_code_lengths: tensor('int32', Int32Array.of(count), [1]),
      }
    }),
    prefill: session(async (f) => {
      const n = (f.input_ids as OrtTensor).dims[1] as number
      calls.prefillRows.push(n)
      made = 0
      return {
        global_hidden: tensor('float32', new Float32Array(n * HIDDEN).fill(n), [1, n, HIDDEN]),
        present_key_0: cache(n),
      }
    }),
    decode: session(async (f) => {
      const past = (f.past_key_0 as OrtTensor).dims[1] as number
      calls.pastLengths.push(Number((f.past_valid_lengths as OrtTensor).data[0]))
      calls.rows.push([...((f.input_ids as OrtTensor).data as Int32Array)])
      return {
        global_hidden: tensor('float32', new Float32Array(HIDDEN).fill(past + 1), [1, 1, HIDDEN]),
        present_key_0: cache(past + 1),
      }
    }),
    frame: session(async (f) => {
      const mask = (f.repetition_seen_mask as OrtTensor).data as Int32Array
      calls.masks.push([...mask])
      calls.uniforms.push([
        ...((f.assistant_random_u as OrtTensor).data as Float32Array),
        ...((f.audio_random_u as OrtTensor).data as Float32Array),
      ])
      made++
      return {
        should_continue: tensor('int32', Int32Array.of(made <= frames ? 1 : 0), [1, 1]),
        frame_token_ids: tensor(
          'int32',
          Int32Array.from({ length: N_VQ }, (_, q) => (made + q) % BOOK),
          [1, N_VQ],
        ),
      }
    }),
    codecDecoder: session(async (f) => {
      const n = (f.audio_codes as OrtTensor).dims[1] as number
      calls.codec.push(n)
      calls.offsets.push(Number((f.offset_0 as OrtTensor).data[0]))
      const len = n * SAMPLES_PER_FRAME
      return {
        audio: tensor('float32', new Float32Array(2 * len).fill(0.5), [1, 2, len]),
        audio_lengths: tensor('int32', Int32Array.of(len), [1]),
        offset_out_0: tensor('int32', Int32Array.of(Number((f.offset_0 as OrtTensor).data[0]) + n), [1]),
      }
    }),
  }
  return { sessions, calls }
}

const make = (frames: number, opts: { gpu?: boolean } = {}) => {
  const f = fakes(frames, opts)
  const tts = new MossTts(manifest, meta, codec, tokenizer, f.sessions, tensor)
  return { tts, ...f }
}

describe('MOSS-TTS-Nano prompt', () => {
  it('lays out the voice-cloning request as the reference runtime does', () => {
    const codes = [Int32Array.of(1, 2, 3), Int32Array.of(4, 5, 6)]
    const rows = buildRequestRows(manifest, codes, [50, 51])
    const width = N_VQ + 1
    const asRows = Array.from({ length: rows.length / width }, (_, r) => [
      ...rows.subarray(r * width, r * width + width),
    ])
    const P = BOOK
    expect(asRows).toEqual([
      [100, P, P, P],
      [101, P, P, P],
      [6, P, P, P], // <|audio_start|>
      [8, 1, 2, 3], // user audio slot + the reference's codes
      [8, 4, 5, 6],
      [7, P, P, P], // <|audio_end|>
      [102, P, P, P],
      [50, P, P, P], // the text
      [51, P, P, P],
      [103, P, P, P],
      [104, P, P, P],
      [6, P, P, P],
    ])
  })
})

describe('MOSS-TTS-Nano sampling inputs', () => {
  it('draws 1 + 16 uniforms per frame, clamped below 1 as the reference does', () => {
    const values = [1, -0.5, 0.25, 0.999999999]
    let i = 0
    const u = frameUniforms(() => values[i++] as number, 3)
    expect(u.assistant[0]).toBeLessThan(1)
    expect([...u.audio]).toEqual([0, 0.25, Math.fround(0.99999994)])
  })

  it('repeats a run for a seed, and passes the codes seen so far to the repetition penalty', async () => {
    const a = make(3)
    a.tts.setVoice([Int32Array.of(1, 1, 1)])
    await a.tts.generate('one two', { seed: 42 }, () => undefined)
    const b = make(3)
    b.tts.setVoice([Int32Array.of(1, 1, 1)])
    await b.tts.generate('one two', { seed: 42 }, () => undefined)
    expect(a.calls.uniforms).toEqual(b.calls.uniforms)
    expect(a.calls.uniforms[0]).toHaveLength(1 + N_VQ)
    // Before the first frame nothing has been seen; afterwards the codes of frame 1 (1, 2, 3) are marked.
    expect(a.calls.masks[0]?.some((v) => v !== 0)).toBe(false)
    const seen = a.calls.masks[1] as number[]
    expect([0, 1, 2].map((q) => seen[q * BOOK + 1 + q])).toEqual([1, 1, 1])
    expect(seen.reduce((x, y) => x + y, 0)).toBe(N_VQ)
  })
})

describe('MOSS-TTS-Nano generation', () => {
  it('prefills once, then feeds each frame back with the assistant slot and a growing cache', async () => {
    const { tts, calls } = make(4)
    tts.setVoice([Int32Array.of(1, 1, 1), Int32Array.of(2, 2, 2)])
    const frames: number[][] = []
    const r = await tts.generate('a few words', { seed: 1, onFrame: (c) => frames.push([...c]) }, () => undefined)
    expect(r.frames).toBe(4)
    const prompt = 3 + 2 + 2 + 3 + 3 // template, voice, end + after, text, assistant prefix + start
    expect(calls.prefillRows).toEqual([prompt])
    expect(calls.pastLengths).toEqual([prompt, prompt + 1, prompt + 2, prompt + 3])
    expect(calls.rows[0]).toEqual([9, ...(frames[0] as number[])])
  })

  it('stops at the frame limit', async () => {
    const { tts } = make(100)
    tts.setVoice([Int32Array.of(1, 1, 1)])
    expect((await tts.generate('words', { maxFrames: 5 }, () => undefined)).frames).toBe(5)
  })

  it('streams: the first frame alone, then at least four frames per codec call; codec state carried over', async () => {
    const { tts, calls } = make(10)
    tts.setVoice([Int32Array.of(1, 1, 1)])
    let pieces = 0
    let samples = 0
    await tts.generate('just a sentence', {}, (a) => {
      pieces++
      samples += a.length
    })
    expect(calls.codec[0]).toBe(1)
    expect(calls.codec.slice(1, -1).every((n) => n >= MIN_DECODE_FRAMES)).toBe(true)
    expect(calls.codec.reduce((a, b) => a + b, 0)).toBe(10)
    expect(calls.offsets).toEqual(calls.codec.map((_, i) => calls.codec.slice(0, i).reduce((a, b) => a + b, 0)))
    expect(pieces).toBe(calls.codec.length)
    // 48 kHz stereo to 24 kHz mono: half the samples, give or take the resampler's edge.
    expect(Math.abs(samples - (10 * SAMPLES_PER_FRAME) / 2)).toBeLessThanOrEqual(1)
  })

  it('speaks long text chunk by chunk with a pause between, the codec restarting for each', async () => {
    const { tts, calls } = make(2)
    tts.setVoice([Int32Array.of(1, 1, 1)])
    const sentence = 'This sentence has exactly eight words in it.'
    const text = Array.from({ length: 12 }, () => sentence).join(' ')
    const chunks = tts.prepare(text)
    expect(chunks.length).toBe(2)
    expect(chunks[0]?.pause).toBe(0.24)
    expect(chunks[1]?.pause).toBe(0)
    let samples = 0
    const r = await tts.generate(text, {}, (a) => (samples += a.length))
    expect(r.frames).toBe(4)
    expect(calls.prefillRows).toHaveLength(2)
    expect(calls.offsets.filter((o) => o === 0)).toHaveLength(2)
    expect(samples).toBeGreaterThan(0.24 * CLONE_RATE)
  })

  it('stops between frames when cancelled', async () => {
    const { tts } = make(50)
    tts.setVoice([Int32Array.of(1, 1, 1)])
    let n = 0
    const r = await tts.generate('words here', { cancelled: () => n >= 3, onFrame: () => n++ }, () => undefined)
    expect(r.frames).toBe(3)
  })

  it('frees the cache it keeps on the GPU as it goes', async () => {
    const { tts, calls } = make(3, { gpu: true })
    tts.setVoice([Int32Array.of(1, 1, 1)])
    await tts.generate('words', {}, () => undefined)
    // One cache from the prefill and one per decode step, every one released.
    expect(calls.disposed).toBe(1 + 3)
  })
})

describe('MOSS-TTS-Nano voice', () => {
  it('feeds the codec 48 kHz stereo and takes only the first seconds of a long reference', async () => {
    const { tts, calls } = make(1)
    const codes = await tts.encodeVoice(new Float32Array(2 * CLONE_RATE))
    expect(calls.encoderLength).toBe(4 * CLONE_RATE)
    expect(codes).toHaveLength(Math.ceil((4 * CLONE_RATE) / 3840))
    await tts.encodeVoice(new Float32Array(40 * CLONE_RATE))
    expect(calls.encoderLength).toBe(2 * MAX_REFERENCE_SECONDS * CLONE_RATE)
    expect(cloneModel('moss-tts-nano').maxReferenceSeconds).toBe(MAX_REFERENCE_SECONDS)
  })

  it('upsamples 2x without changing a low tone', () => {
    const f = 440
    const x = Float32Array.from({ length: 2400 }, (_, i) => Math.sin((2 * Math.PI * f * i) / 24_000))
    const y = upsample2x(x)
    expect(y.length).toBe(4800)
    let err = 0
    for (let i = 400; i < 4400; i++)
      err = Math.max(err, Math.abs((y[i] as number) - Math.sin((2 * Math.PI * f * i) / 48_000)))
    expect(err).toBeLessThan(0.01)
  })
})

describe('MOSS-TTS-Nano streaming budget', () => {
  it('decodes more frames at once the further the audio is ahead of playback', () => {
    expect(decodeBudget(0, 48_000, null, 0)).toBe(1)
    expect(decodeBudget(48_000 * 0.1, 48_000, 0, 0)).toBe(1)
    expect(decodeBudget(48_000 * 0.5, 48_000, 0, 0)).toBe(2)
    expect(decodeBudget(48_000, 48_000, 0, 0)).toBe(4)
    expect(decodeBudget(48_000 * 2, 48_000, 0, 0)).toBe(8)
    expect(decodeBudget(48_000 * 2, 48_000, 0, 1900)).toBe(1)
  })
})
