import { Random, type MakeTensor, type OrtSession, type OrtTensor } from './ort-types'
import type { SpmTokenizer } from './spm'

/**
 * Kyutai's Pocket TTS (100M parameters, CC BY 4.0) in the ONNX form
 * KevinAHM/pocket-tts-onnx publishes: five graphs run with ONNX Runtime Web.
 *
 * - The reference voice goes through Mimi's encoder (`mimi_encoder`): 24 kHz audio in, one 1024-wide
 *   vector per 80 ms frame out. That sequence *is* the voice: no transcript is needed. It is fed once
 *   to the language model as a prefix (after a learned BOS vector) and the model's state after it is
 *   kept, so every sentence starts from the same voice state.
 * - Text: SentencePiece ids → `text_conditioner` → embeddings, fed to `flow_lm_main` as more prefix.
 * - Then one 80 ms frame at a time: `flow_lm_main` gives a conditioning vector and an end-of-speech
 *   logit, `flow_lm_flow` turns noise into the frame's 32-dim latent in `lsdSteps` Euler steps, and
 *   the streaming Mimi decoder (`mimi_decoder`) turns latents into audio, a few frames at a time.
 *
 * Ported from the export's own Python wrapper (pocket_tts_onnx.py at the pinned revision).
 */

export interface StateEntry {
  input_name: string
  output_name: string
  dtype: 'float32' | 'int64' | 'bool' | 'float16'
  fill: 'nan' | 'zeros' | 'ones' | 'empty'
  shape: number[]
}

export interface PocketMeta {
  sample_rate: number
  samples_per_frame: number
  frame_rate: number
  latent_dim: number
  conditioning_dim: number
  insert_bos_before_voice?: boolean
  pad_with_spaces_for_short_inputs?: boolean
  remove_semicolons?: boolean
  model_recommended_frames_after_eos?: number | null
  max_token_per_chunk?: number
  flow_lm_state_manifest: StateEntry[]
  mimi_state_manifest: StateEntry[]
}

export interface PocketSessions {
  encoder: OrtSession
  textConditioner: OrtSession
  main: OrtSession
  flow: OrtSession
  decoder: OrtSession
}

/** The voice: Mimi encoder frames of the reference, [frames × 1024]. */
export interface PocketVoice {
  frames: number
  dim: number
  data: Float32Array
}

export interface PocketOptions {
  temperature: number
  lsdSteps: number
  seed?: number
  /** Stop when this returns true (checked between frames). */
  cancelled?: () => boolean
}

type State = Record<string, OrtTensor>

const EOS_THRESHOLD = -4
const TOKENS_PER_SECOND = 3
const PADDING_SECONDS = 2

export class PocketTts {
  readonly meta: PocketMeta
  private readonly tok: SpmTokenizer
  private readonly s: PocketSessions
  private readonly bos: Float32Array | null
  private readonly tensor: MakeTensor
  /** The cache length the model declares (1000 frames). */
  private readonly cacheMax: number
  /** The language model takes a shorter cache (its graph was patched; see onnx-patch.ts). */
  private readonly dynamicCache: boolean

  constructor(
    meta: PocketMeta,
    tokenizer: SpmTokenizer,
    sessions: PocketSessions,
    bos: Float32Array | null,
    tensor: MakeTensor,
    opts: { dynamicCache?: boolean } = {},
  ) {
    this.meta = meta
    this.tok = tokenizer
    this.s = sessions
    this.bos = bos
    this.tensor = tensor
    this.dynamicCache = opts.dynamicCache ?? false
    this.cacheMax = Math.max(
      0,
      ...meta.flow_lm_state_manifest.filter((e) => e.shape.length === 5).map((e) => e.shape[2] as number),
    )
  }

  /** Frames of cache the state tensors get for a voice of `prefix` frames and `need` more. */
  cacheFor(prefix: number, need: number): number {
    if (!this.dynamicCache || !this.cacheMax) return this.cacheMax
    return Math.min(this.cacheMax, Math.ceil((prefix + need + 8) / 128) * 128)
  }

  get sampleRate(): number {
    return this.meta.sample_rate
  }

  private initState(manifest: StateEntry[], cache?: number): State {
    const out: State = {}
    for (const entry of manifest) {
      const e =
        cache != null && entry.shape.length === 5 && entry.shape[2] === this.cacheMax
          ? { ...entry, shape: entry.shape.map((d, i) => (i === 2 ? cache : d)) }
          : entry
      const n = e.shape.reduce((a, b) => a * b, 1)
      if (e.dtype === 'int64')
        out[e.input_name] = this.tensor('int64', new BigInt64Array(n).fill(e.fill === 'ones' ? 1n : 0n), e.shape)
      else if (e.dtype === 'bool')
        out[e.input_name] = this.tensor('bool', new Uint8Array(n).fill(e.fill === 'ones' ? 1 : 0), e.shape)
      else
        out[e.input_name] = this.tensor(
          'float32',
          new Float32Array(n).fill(e.fill === 'nan' ? NaN : e.fill === 'ones' ? 1 : 0),
          e.shape,
        )
    }
    return out
  }

  private static next(out: Record<string, OrtTensor>, manifest: StateEntry[]): State {
    const s: State = {}
    for (const e of manifest) {
      const t = out[e.output_name]
      if (!t) throw new Error(`Pocket TTS: missing output ${e.output_name}`)
      s[e.input_name] = t
    }
    return s
  }

  /** Encode 24 kHz mono reference audio into the voice. */
  async encodeVoice(audio: Float32Array): Promise<PocketVoice> {
    const out = await this.s.encoder.run({ audio: this.tensor('float32', audio, [1, 1, audio.length]) })
    const t = out[this.s.encoder.outputNames[0] as string] as OrtTensor
    const dim = t.dims[t.dims.length - 1] as number
    const data = Float32Array.from(t.data as Float32Array)
    return { frames: data.length / dim, dim, data }
  }

  /** Frames the voice takes in the model's context (with the BOS vector). */
  prefixFrames(voice: PocketVoice): number {
    return voice.frames + (this.meta.insert_bos_before_voice && this.bos ? 1 : 0)
  }

  /** The language model's state after reading the voice prefix, with a cache of `cache` frames. */
  async voiceState(voice: PocketVoice, cache = this.cacheMax): Promise<State> {
    const bos = this.meta.insert_bos_before_voice && this.bos ? this.bos : null
    const frames = voice.frames + (bos ? 1 : 0)
    const data = new Float32Array(frames * voice.dim)
    if (bos) data.set(bos.subarray(0, voice.dim))
    data.set(voice.data, bos ? voice.dim : 0)
    const state = this.initState(this.meta.flow_lm_state_manifest, cache)
    const out = await this.s.main.run({
      sequence: this.tensor('float32', new Float32Array(0), [1, 0, this.meta.latent_dim]),
      text_embeddings: this.tensor('float32', data, [1, frames, voice.dim]),
      ...state,
    })
    return PocketTts.next(out, this.meta.flow_lm_state_manifest)
  }

  /** Text as the model expects it, and how many frames to let run after the end-of-speech signal. */
  prepareText(text: string): { text: string; framesAfterEos: number } {
    let t = text
      .trim()
      .replace(/[\r\n]/g, ' ')
      .replace(/ {2}/g, ' ')
    if (!t) throw new Error('Text cannot be empty')
    if (this.meta.remove_semicolons) t = t.replace(/;/g, ',')
    const words = t.split(/\s+/).length
    const guess = words <= 4 ? 3 : 1
    t = t[0]!.toUpperCase() + t.slice(1)
    if (/[\p{L}\p{N}]$/u.test(t)) t += '.'
    if (this.meta.pad_with_spaces_for_short_inputs && t.split(/\s+/).length < 5) t = ' '.repeat(8) + t
    return { text: t, framesAfterEos: this.meta.model_recommended_frames_after_eos ?? guess + 2 }
  }

  /** Split into chunks of at most `max_token_per_chunk` tokens at sentence ends (then commas). */
  splitText(text: string): string[] {
    const max = this.meta.max_token_per_chunk ?? 50
    const prepared = this.prepareText(text).text.trim()
    const tokens = this.tok.encode(prepared)
    const segments = (ids: number[], boundary: Set<number>) => {
      const idx = [0]
      let prev = false
      ids.forEach((id, i) => {
        if (boundary.has(id)) prev = true
        else {
          if (prev) idx.push(i)
          prev = false
        }
      })
      idx.push(ids.length)
      const out: [number, string][] = []
      for (let i = 0; i < idx.length - 1; i++)
        out.push([(idx[i + 1] as number) - (idx[i] as number), this.tok.decode(ids.slice(idx[i], idx[i + 1]))])
      return out
    }
    const eos = new Set(this.tok.encode('.!...?').slice(1))
    const comma = new Set(this.tok.encode(',;:').slice(1))
    const refined: [number, string][] = []
    for (const [count, seg] of segments(tokens, eos)) {
      if (count <= max) {
        refined.push([count, seg])
        continue
      }
      const sub = segments(this.tok.encode(seg.trim()), comma)
      if (sub.length > 1) refined.push(...sub)
      else refined.push([count, seg])
    }
    const chunks: string[] = []
    let cur = ''
    let n = 0
    for (const [count, seg] of refined) {
      if (!cur) {
        cur = seg
        n = count
      } else if (n + count > max) {
        chunks.push(cur.trim())
        cur = seg
        n = count
      } else {
        cur += ' ' + seg
        n += count
      }
    }
    if (cur.trim()) chunks.push(cur.trim())
    return chunks
  }

  private voice: PocketVoice | null = null
  private states = new Map<number, State>()

  /** Use this voice from now on (its states are made per cache length when first needed). */
  setVoice(voice: PocketVoice): void {
    this.voice = voice
    this.states.clear()
  }

  private async stateFor(cache: number): Promise<State> {
    if (!this.voice) throw new Error('No voice set')
    let s = this.states.get(cache)
    if (!s) {
      s = await this.voiceState(this.voice, cache)
      this.states.set(cache, s)
    }
    return s
  }

  /**
   * Speak `text` in the voice given to `setVoice`. Audio is handed to `onAudio` as it is decoded: two
   * frames (160 ms) first, then larger pieces.
   */
  async generate(
    text: string,
    opts: PocketOptions,
    onAudio: (samples: Float32Array) => void,
  ): Promise<{ frames: number }> {
    const rng = new Random(opts.seed)
    const { latent_dim: L, conditioning_dim: C } = this.meta
    const steps = Math.max(1, Math.round(opts.lsdSteps))
    const dt = 1 / steps
    const st = Array.from({ length: steps }, (_, j) => [
      this.tensor('float32', Float32Array.of(j / steps), [1, 1]),
      this.tensor('float32', Float32Array.of(j / steps + dt), [1, 1]),
    ])
    const emptyText = this.tensor('float32', new Float32Array(0), [1, 0, C])
    let frames = 0
    for (const chunk of this.splitText(text)) {
      if (opts.cancelled?.()) break
      const { text: prepared, framesAfterEos } = this.prepareText(chunk)
      const ids = this.tok.encode(prepared)
      const textOut = await this.s.textConditioner.run({
        token_ids: this.tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
      })
      const emb = textOut[this.s.textConditioner.outputNames[0] as string] as OrtTensor
      const limit = Math.ceil((ids.length / TOKENS_PER_SECOND + PADDING_SECONDS) * this.meta.frame_rate)
      const prefix = this.prefixFrames(this.voice as PocketVoice)
      const cache = this.cacheFor(prefix, ids.length + limit)
      // The model's context ends at the cache length: never run past it.
      const frameLimit = this.cacheMax ? Math.max(0, Math.min(limit, cache - prefix - ids.length - 2)) : limit
      let state = await this.stateFor(cache)
      let out = await this.s.main.run({
        sequence: this.tensor('float32', new Float32Array(0), [1, 0, L]),
        text_embeddings: this.tensor('float32', emb.data, [1, ids.length, C]),
        ...state,
      })
      state = PocketTts.next(out, this.meta.flow_lm_state_manifest)
      let curr = new Float32Array(L).fill(NaN)
      let eosStep: number | null = null
      let mimi = this.initState(this.meta.mimi_state_manifest)
      const pending: Float32Array[] = []
      let decoded = 0
      const decode = async (count: number) => {
        const lat = new Float32Array(count * L)
        for (let i = 0; i < count; i++) lat.set(pending[i] as Float32Array, i * L)
        pending.splice(0, count)
        const r = await this.s.decoder.run({ latent: this.tensor('float32', lat, [1, count, L]), ...mimi })
        mimi = PocketTts.next(r, this.meta.mimi_state_manifest)
        decoded += 1
        onAudio(Float32Array.from((r[this.s.decoder.outputNames[0] as string] as OrtTensor).data as Float32Array))
      }
      for (let step = 0; step < frameLimit; step++) {
        if (opts.cancelled?.()) break
        out = await this.s.main.run({
          sequence: this.tensor('float32', curr, [1, 1, L]),
          text_embeddings: emptyText,
          ...state,
        })
        const cond = out.conditioning as OrtTensor
        const eos = (out.eos_logit as OrtTensor).data[0] as number
        state = PocketTts.next(out, this.meta.flow_lm_state_manifest)
        if (eos > EOS_THRESHOLD && eosStep == null) eosStep = step
        if (eosStep != null && step >= eosStep + framesAfterEos) break
        const std = Math.sqrt(Math.max(0, opts.temperature))
        const x = new Float32Array(L)
        if (std > 0) for (let i = 0; i < L; i++) x[i] = rng.normal() * std
        for (const [s, t] of st) {
          const f = await this.s.flow.run({
            c: cond,
            s: s as OrtTensor,
            t: t as OrtTensor,
            x: this.tensor('float32', x, [1, L]),
          })
          const dir = f[this.s.flow.outputNames[0] as string]!.data as Float32Array
          for (let i = 0; i < L; i++) x[i]! += dir[i]! * dt
        }
        pending.push(x)
        curr = x
        frames += 1
        // First audio after two frames, then in growing pieces (fewer decoder calls, still ahead of playback).
        const want = decoded === 0 ? 2 : Math.min(12, 2 * 2 ** decoded)
        if (pending.length >= want) await decode(pending.length)
      }
      if (pending.length) await decode(pending.length)
    }
    return { frames }
  }
}
