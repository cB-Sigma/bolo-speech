import { describe, expect, it } from 'vitest'
import type { MakeTensor, OrtSession, OrtTensor } from './ort-types'
import { PocketTts, type PocketMeta } from './pocket'
import type { SpmTokenizer } from './spm'

const L = 4
const C = 6
const SPF = 10

const tensor: MakeTensor = (type, data, dims) => ({ type, data, dims })

/** Words as tokens: each word one id (its length), punctuation its own token. */
const tokenizer = {
  encode: (t: string) =>
    t
      .replace(/([.!?,;:])/g, ' $1 ')
      .split(/\s+/)
      .filter(Boolean)
      .map((w) => (w === '.' ? 99 : w === '!' ? 98 : w === '?' ? 97 : /[,;:]/.test(w) ? 96 : w.length)),
  decode: (ids: number[]) => ids.map((i) => (i === 99 ? '.' : 'w'.repeat(i))).join(' '),
} as unknown as SpmTokenizer

function session(inputNames: string[], outputNames: string[], run: OrtSession['run']): OrtSession {
  return { inputNames, outputNames, run }
}

function fakes(eosAfter: number) {
  const calls = { main: 0, flow: 0, decoder: [] as number[], textLens: [] as number[] }
  let frame = 0
  const sessions = {
    encoder: session(['audio'], ['latents'], async (f) => {
      const n = Math.ceil((f.audio as OrtTensor).dims[2]! / 1920)
      return { latents: tensor('float32', new Float32Array(n * C).fill(0.5), [1, n, C]) }
    }),
    textConditioner: session(['token_ids'], ['embeddings'], async (f) => {
      const n = (f.token_ids as OrtTensor).dims[1]!
      calls.textLens.push(n)
      return { embeddings: tensor('float32', new Float32Array(n * C), [1, n, C]) }
    }),
    main: session(
      ['sequence', 'text_embeddings', 'state_0'],
      ['conditioning', 'eos_logit', 'out_state_0'],
      async (f) => {
        calls.main++
        const seq = (f.sequence as OrtTensor).dims[1]
        const step = Number((f.state_0 as OrtTensor).data[0])
        if (seq === 1) frame++
        const eos = seq === 1 && frame > eosAfter ? 0 : -10
        return {
          conditioning: tensor('float32', new Float32Array(C).fill(1), [1, C]),
          eos_logit: tensor('float32', Float32Array.of(eos), [1, 1]),
          out_state_0: tensor('int64', BigInt64Array.of(BigInt(step + 1)), [1]),
        }
      },
    ),
    flow: session(['c', 's', 't', 'x'], ['flow_dir'], async () => {
      calls.flow++
      return { flow_dir: tensor('float32', new Float32Array(L).fill(1), [1, L]) }
    }),
    decoder: session(['latent', 'state_0'], ['audio_frame', 'out_state_0'], async (f) => {
      const n = (f.latent as OrtTensor).dims[1]!
      calls.decoder.push(n)
      return {
        audio_frame: tensor('float32', new Float32Array(n * SPF).fill(0.1), [1, 1, n * SPF]),
        out_state_0: f.state_0 as OrtTensor,
      }
    }),
  }
  return { sessions, calls, reset: () => (frame = 0) }
}

const meta: PocketMeta = {
  sample_rate: 24000,
  samples_per_frame: SPF,
  frame_rate: 12.5,
  latent_dim: L,
  conditioning_dim: C,
  insert_bos_before_voice: true,
  max_token_per_chunk: 6,
  flow_lm_state_manifest: [
    { input_name: 'state_0', output_name: 'out_state_0', dtype: 'int64', fill: 'zeros', shape: [1] },
  ],
  mimi_state_manifest: [{ input_name: 'state_0', output_name: 'out_state_0', dtype: 'bool', fill: 'ones', shape: [1] }],
}

describe('Pocket TTS pipeline', () => {
  it('takes in a voice: encoder frames with the BOS vector in front', async () => {
    const f = fakes(3)
    const tts = new PocketTts(meta, tokenizer, f.sessions, new Float32Array(C).fill(9), tensor)
    const voice = await tts.encodeVoice(new Float32Array(1920 * 5))
    expect(voice).toMatchObject({ frames: 5, dim: C })
    let seen: OrtTensor | null = null
    const main = f.sessions.main.run
    f.sessions.main.run = async (feeds) => {
      seen = feeds.text_embeddings as OrtTensor
      return main(feeds)
    }
    const state = await tts.voiceState(voice)
    expect(seen!.dims).toEqual([1, 6, C])
    expect(Array.from(seen!.data as Float32Array).slice(0, C)).toEqual(Array(C).fill(9))
    expect(Number(state.state_0!.data[0])).toBe(1)
  })

  it('prepares text as the export does', () => {
    const tts = new PocketTts(meta, tokenizer, fakes(1).sessions, null, tensor)
    expect(tts.prepareText('hello there')).toEqual({ text: 'Hello there.', framesAfterEos: 5 })
    expect(tts.prepareText('one two three four five six').framesAfterEos).toBe(3)
    expect(tts.prepareText('Done!').text).toBe('Done!')
    expect(() => tts.prepareText('   ')).toThrow()
  })

  it('splits long text at sentence ends into chunks the model can take', () => {
    const tts = new PocketTts(meta, tokenizer, fakes(1).sessions, null, tensor)
    const chunks = tts.splitText('aa bb cc. dd ee ff. gg.')
    expect(chunks.length).toBeGreaterThan(1)
    for (const c of chunks) expect(tokenizer.encode(c).length).toBeLessThanOrEqual(6)
  })

  it('generates frames until the end signal and a few after, streaming two frames first', async () => {
    const f = fakes(4)
    const tts = new PocketTts(meta, tokenizer, f.sessions, null, tensor)
    tts.setVoice(await tts.encodeVoice(new Float32Array(1920 * 3)))
    const chunks: Float32Array[] = []
    const r = await tts.generate('one two three four five', { temperature: 0.7, lsdSteps: 2, seed: 1 }, (a) =>
      chunks.push(a),
    )
    // end signal at frame 5 (index 4); 3 more frames after it for a five-word sentence (1 + 2)
    expect(r.frames).toBe(7)
    expect(f.calls.flow).toBe(7 * 2)
    expect(f.calls.decoder[0]).toBe(2)
    expect(f.calls.decoder.reduce((a, b) => a + b, 0)).toBe(7)
    expect(chunks.reduce((a, c) => a + c.length, 0)).toBe(7 * SPF)
  })

  it('stops between frames when cancelled', async () => {
    const f = fakes(100)
    const tts = new PocketTts(meta, tokenizer, f.sessions, null, tensor)
    tts.setVoice(await tts.encodeVoice(new Float32Array(1920)))
    let n = 0
    const r = await tts.generate('hello', { temperature: 0, lsdSteps: 1, cancelled: () => ++n > 4 }, () => undefined)
    expect(r.frames).toBeLessThan(5)
  })
})
