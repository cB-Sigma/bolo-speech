import { designLowpass, Resampler } from '@/lib/audio/resample'
import { interChunkPauseSeconds, normalizeTtsText, splitVoiceCloneText } from './moss-text'
import { Random, type MakeTensor, type OrtSession, type OrtTensor } from './ort-types'
import { CLONE_RATE } from './protocol'

/**
 * MOSS-TTS-Nano 100M (OpenMOSS, Apache-2.0) in the browser, on the official ONNX exports
 * (OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX and MOSS-Audio-Tokenizer-Nano-ONNX) run by ONNX Runtime Web.
 * Written from the model's own Apache-2.0 reference runtime (github.com/OpenMOSS/MOSS-TTS-Nano at 8b7bcc93:
 * ort_cpu_runtime.py and onnx_tts_runtime.py, "voice_clone" mode with the default "fixed" sampling).
 *
 * - The voice: the reference recording, at 48 kHz in two channels, goes through the codec's encoder: 16 codes
 *   per 80 ms frame (12.5 frames a second). Those codes *are* the voice prompt: no transcript is needed.
 * - The prompt: one row of 17 ids per position (a text id, then 16 audio codes, padded with 1024). Text rows
 *   for the fixed chat template, one row per reference frame (the user audio slot id and its codes), more
 *   template, the text's SentencePiece ids, the assistant prefix and `<|audio_start|>`.
 * - The global transformer reads the prompt once (`prefill`), then one row per generated frame
 *   (`decode_step`, with its key/value cache; on WebGPU the cache stays on the GPU between steps).
 * - Each frame: `local_fixed_sampled_frame` turns the global hidden state into the next 16 codes, sampling
 *   inside the graph (audio temperature 0.8, top-p 0.95, top-k 25, repetition penalty 1.2 over the codes seen
 *   so far; whether to go on, from the text head) from 17 uniform random numbers the caller gives it.
 * - Audio: the codec's streaming decoder turns codes into 48 kHz stereo, a few frames at a time; the page
 *   takes the mean of the two channels at 24 kHz.
 */

export interface MossManifest {
  tts_config: {
    n_vq: number
    audio_pad_token_id: number
    audio_start_token_id: number
    audio_end_token_id: number
    audio_user_slot_token_id: number
    audio_assistant_slot_token_id: number
    audio_codebook_sizes: number[]
  }
  prompt_templates: {
    user_prompt_prefix_token_ids: number[]
    user_prompt_after_reference_token_ids: number[]
    assistant_prompt_prefix_token_ids: number[]
  }
  generation_defaults: { max_new_frames: number }
}

export interface MossTtsMeta {
  onnx: { prefill_output_names: string[]; decode_input_names: string[]; decode_output_names: string[] }
}

interface CodecState {
  input_name: string
  output_name: string
  shape: number[]
}

interface CodecCache {
  offset_input_name: string
  offset_output_name: string
  cached_keys_input_name: string
  cached_keys_output_name: string
  cached_values_input_name: string
  cached_values_output_name: string
  cached_positions_input_name: string
  cached_positions_output_name: string
  offset_shape: number[]
  cache_shape: number[]
  positions_shape: number[]
}

export interface MossCodecMeta {
  codec_config: { sample_rate: number; channels: number; num_quantizers: number }
  streaming_decode: { transformer_offsets: CodecState[]; attention_caches: CodecCache[] }
}

export interface MossSessions {
  prefill: OrtSession
  decode: OrtSession
  frame: OrtSession
  encoder: OrtSession
  codecDecoder: OrtSession
}

export interface MossTokenizer {
  encode(text: string): number[]
  count(text: string): number
}

export interface MossOptions {
  seed?: number
  /** Stop when this returns true (checked between frames). */
  cancelled?: () => boolean
  /** Frames per chunk at most (default: the manifest's 375, 30 s). */
  maxFrames?: number
  /** Uniform numbers in [0, 1) for the sampler instead of the seeded generator (tests, comparisons). */
  random?: () => number
  /** Called with each frame's 16 codes. */
  onFrame?: (codes: Int32Array) => void
}

type Feeds = Record<string, OrtTensor>

/** Frames per codec call once audio has started (see `generate`). */
export const MIN_DECODE_FRAMES = 4

/**
 * Longest reference the codec's encoder is given. Its attention spans the whole recording, so its memory grows
 * with the square of the length: about 0.25 GB for 5 s, 0.75 GB for 10 s, 1.4 GB for 15 s and 5 GB for 30 s
 * (measured with ONNX Runtime), and WebAssembly memory never shrinks. The model's own preset voices are 3 to 17 s.
 */
export const MAX_REFERENCE_SECONDS = 12

/** The largest float32 below 1, as the reference clamps its uniforms. */
const BELOW_ONE = 0.99999994

/**
 * Frames to decode at once while streaming, from how far the audio already made is ahead of playback
 * (the reference's `_resolve_stream_decode_frame_budget`): one at first, more as the lead grows.
 */
export function decodeBudget(emittedSamples: number, rate: number, firstAudioAt: number | null, now: number): number {
  if (firstAudioAt == null || rate <= 0) return 1
  const lead = emittedSamples / rate - Math.max(0, now - firstAudioAt) / 1000
  if (lead < 0.2) return 1
  if (lead < 0.55) return 2
  if (lead < 1.1) return 4
  return 8
}

/** 2x upsampling with a windowed-sinc low-pass at the old Nyquist frequency (24 kHz to 48 kHz). */
export function upsample2x(input: Float32Array, taps = 127): Float32Array {
  const h = designLowpass(0.25, taps)
  const half = (h.length - 1) / 2
  const out = new Float32Array(input.length * 2)
  for (let n = 0; n < out.length; n++) {
    // y[n] = 2 * sum_k h[k] * z[n - k + half], z the zero-stuffed input (z[2m] = x[m]).
    let acc = 0
    const kStart = (n + half) & 1
    for (let k = kStart; k < h.length; k += 2) {
      const z = n + half - k
      if (z < 0) break
      const m = z >> 1
      if (m < input.length) acc += (h[k] as number) * (input[m] as number)
    }
    out[n] = 2 * acc
  }
  return out
}

/** The prompt rows of a voice-cloning request: the reference codes and the text ids in the chat template. */
export function buildRequestRows(manifest: MossManifest, codes: Int32Array[], textIds: number[]): Int32Array {
  const c = manifest.tts_config
  const t = manifest.prompt_templates
  const width = c.n_vq + 1
  const before = [...t.user_prompt_prefix_token_ids, c.audio_start_token_id]
  const after = [
    c.audio_end_token_id,
    ...t.user_prompt_after_reference_token_ids,
    ...textIds,
    ...t.assistant_prompt_prefix_token_ids,
    c.audio_start_token_id,
  ]
  const rows = new Int32Array((before.length + codes.length + after.length) * width).fill(c.audio_pad_token_id)
  let r = 0
  for (const id of before) rows[r++ * width] = id
  for (const frame of codes) {
    rows[r * width] = c.audio_user_slot_token_id
    for (let q = 0; q < Math.min(frame.length, c.n_vq); q++) rows[r * width + 1 + q] = frame[q] as number
    r++
  }
  for (const id of after) rows[r++ * width] = id
  return rows
}

/** The 17 uniforms one frame takes: whether to go on, then one per codebook (clamped as the reference does). */
export function frameUniforms(random: () => number, nVq: number): { assistant: Float32Array; audio: Float32Array } {
  const u = () => Math.min(BELOW_ONE, Math.max(0, random()))
  const assistant = Float32Array.of(u())
  const audio = new Float32Array(nVq)
  for (let i = 0; i < nVq; i++) audio[i] = u()
  return { assistant, audio }
}

export interface PreparedChunk {
  text: string
  ids: number[]
  /** Seconds of silence after it (none after the last). */
  pause: number
}

export class MossTts {
  readonly manifest: MossManifest
  private readonly meta: MossTtsMeta
  private readonly codec: MossCodecMeta
  private readonly tok: MossTokenizer
  private readonly s: MossSessions
  private readonly tensor: MakeTensor
  private voice: Int32Array[] | null = null

  constructor(
    manifest: MossManifest,
    meta: MossTtsMeta,
    codec: MossCodecMeta,
    tokenizer: MossTokenizer,
    sessions: MossSessions,
    tensor: MakeTensor,
  ) {
    this.manifest = manifest
    this.meta = meta
    this.codec = codec
    this.tok = tokenizer
    this.s = sessions
    this.tensor = tensor
  }

  /** Output sample rate of the codec (48 kHz). */
  get codecRate(): number {
    return this.codec.codec_config.sample_rate
  }

  /** Codes of 24 kHz mono reference audio (its first MAX_REFERENCE_SECONDS): 16 per 80 ms frame. */
  async encodeVoice(input: Float32Array): Promise<Int32Array[]> {
    const audio =
      input.length > MAX_REFERENCE_SECONDS * CLONE_RATE ? input.subarray(0, MAX_REFERENCE_SECONDS * CLONE_RATE) : input
    const channels = this.codec.codec_config.channels
    const wave =
      this.codecRate === CLONE_RATE * 2 ? upsample2x(audio) : new Resampler(CLONE_RATE, this.codecRate).process(audio)
    const planar = new Float32Array(wave.length * channels)
    for (let c = 0; c < channels; c++) planar.set(wave, c * wave.length)
    const out = await this.s.encoder.run({
      waveform: this.tensor('float32', planar, [1, channels, wave.length]),
      input_lengths: this.tensor('int32', Int32Array.of(wave.length), [1]),
    })
    const codes = out.audio_codes as OrtTensor
    const n = Number((out.audio_code_lengths as OrtTensor).data[0])
    const q = codes.dims[2] as number
    const data = codes.data as Int32Array
    return Array.from({ length: n }, (_, f) => Int32Array.from(data.subarray(f * q, f * q + q)))
  }

  setVoice(codes: Int32Array[]): void {
    this.voice = codes
  }

  /** The text cleaned up and cut into chunks, with their token ids. */
  prepare(text: string): PreparedChunk[] {
    const normalized = normalizeTtsText(text)
    if (!normalized) throw new Error('Text cannot be empty')
    const chunks = splitVoiceCloneText(normalized, (t) => this.tok.count(t))
    return chunks.map((c, i) => ({
      text: c,
      ids: this.tok.encode(c),
      pause: i < chunks.length - 1 ? interChunkPauseSeconds(c) : 0,
    }))
  }

  /** Codes of the frames for one chunk, sampled with `random`; `onFrame` sees each as it comes. */
  async generateFrames(
    ids: number[],
    opts: MossOptions,
    random: () => number,
    onFrame: (codes: Int32Array) => Promise<void> | void,
  ): Promise<number> {
    if (!this.voice) throw new Error('No voice set')
    const c = this.manifest.tts_config
    const width = c.n_vq + 1
    const book = c.audio_codebook_sizes[0] ?? 1024
    const rows = buildRequestRows(this.manifest, this.voice, ids)
    const n = rows.length / width
    let out = await this.s.prefill.run({
      input_ids: this.tensor('int32', rows, [1, n, width]),
      attention_mask: this.tensor('int32', new Int32Array(n).fill(1), [1, n]),
    })
    let past = this.cacheFrom(out, this.meta.onnx.prefill_output_names)
    let hidden = lastHidden(out.global_hidden as OrtTensor)
    let length = n
    const seen = new Int32Array(c.n_vq * book)
    const max = opts.maxFrames ?? this.manifest.generation_defaults.max_new_frames
    let frames = 0
    for (let step = 0; step < max; step++) {
      if (opts.cancelled?.()) break
      const u = frameUniforms(random, c.n_vq)
      const f = await this.s.frame.run({
        global_hidden: this.tensor('float32', hidden, [1, hidden.length]),
        repetition_seen_mask: this.tensor('int32', seen, [1, c.n_vq, book]),
        assistant_random_u: this.tensor('float32', u.assistant, [1]),
        audio_random_u: this.tensor('float32', u.audio, [1, c.n_vq]),
      })
      if (Number((f.should_continue as OrtTensor).data[0]) === 0) break
      const codes = Int32Array.from((f.frame_token_ids as OrtTensor).data as Int32Array)
      for (let q = 0; q < c.n_vq; q++) {
        const tok = codes[q] as number
        if (tok >= 0 && tok < book) seen[q * book + tok] = 1
      }
      frames++
      opts.onFrame?.(codes)
      const row = new Int32Array(width).fill(c.audio_pad_token_id)
      row[0] = c.audio_assistant_slot_token_id
      row.set(codes.subarray(0, c.n_vq), 1)
      const feeds: Feeds = {
        input_ids: this.tensor('int32', row, [1, 1, width]),
        past_valid_lengths: this.tensor('int32', Int32Array.of(length), [1]),
        ...past,
      }
      out = await this.s.decode.run(feeds)
      hidden = lastHidden(out.global_hidden as OrtTensor)
      release(past)
      past = this.cacheFrom(out, this.meta.onnx.decode_output_names)
      length++
      await onFrame(codes)
    }
    release(past)
    return frames
  }

  private cacheFrom(out: Record<string, OrtTensor>, names: string[]): Feeds {
    const feeds: Feeds = {}
    for (const name of names.slice(1)) {
      const t = out[name]
      if (!t) throw new Error(`MOSS-TTS-Nano: missing output ${name}`)
      feeds[name.replace('present_', 'past_')] = t
    }
    return feeds
  }

  private initialCodecState(): Feeds {
    const s: Feeds = {}
    const int = (shape: number[], fill = 0) =>
      this.tensor('int32', new Int32Array(shape.reduce((a, b) => a * b, 1)).fill(fill), shape)
    for (const t of this.codec.streaming_decode.transformer_offsets) s[t.input_name] = int(t.shape)
    for (const a of this.codec.streaming_decode.attention_caches) {
      const n = a.cache_shape.reduce((x, y) => x * y, 1)
      s[a.offset_input_name] = int(a.offset_shape)
      s[a.cached_keys_input_name] = this.tensor('float32', new Float32Array(n), a.cache_shape)
      s[a.cached_values_input_name] = this.tensor('float32', new Float32Array(n), a.cache_shape)
      s[a.cached_positions_input_name] = int(a.positions_shape, -1)
    }
    return s
  }

  /**
   * Speak `text` in the voice given to `setVoice`. Audio (24 kHz mono) goes to `onAudio` as the codec decodes
   * it: after the first frame, then in pieces that grow while the audio made stays ahead of playback.
   */
  async generate(
    text: string,
    opts: MossOptions,
    onAudio: (samples: Float32Array) => void,
  ): Promise<{ frames: number; chunks: PreparedChunk[] }> {
    const chunks = this.prepare(text)
    // One random sequence for the whole text, as the reference keeps one generator.
    const rng = new Random(opts.seed)
    const random = opts.random ?? (() => rng.uniform())
    const rate = this.codecRate
    const channels = this.codec.codec_config.channels
    const q = this.codec.codec_config.num_quantizers
    const down = new Resampler(rate, CLONE_RATE)
    const emit = (wave: Float32Array) => {
      const a = down.process(wave)
      if (a.length) onAudio(Float32Array.from(a))
    }
    let emitted = 0
    let firstAt: number | null = null
    let frames = 0
    const now = () => performance.now()
    for (const chunk of chunks) {
      if (opts.cancelled?.()) break
      let state = this.initialCodecState()
      const pending: Int32Array[] = []
      const decode = async (count: number) => {
        const take = pending.splice(0, count)
        const codes = new Int32Array(take.length * q)
        take.forEach((f, i) => codes.set(f.subarray(0, q), i * q))
        const out = await this.s.codecDecoder.run({
          audio_codes: this.tensor('int32', codes, [1, take.length, q]),
          audio_code_lengths: this.tensor('int32', Int32Array.of(take.length), [1]),
          ...state,
        })
        const next: Feeds = {}
        for (const t of this.codec.streaming_decode.transformer_offsets)
          next[t.input_name] = out[t.output_name] as OrtTensor
        for (const a of this.codec.streaming_decode.attention_caches) {
          next[a.offset_input_name] = out[a.offset_output_name] as OrtTensor
          next[a.cached_keys_input_name] = out[a.cached_keys_output_name] as OrtTensor
          next[a.cached_values_input_name] = out[a.cached_values_output_name] as OrtTensor
          next[a.cached_positions_input_name] = out[a.cached_positions_output_name] as OrtTensor
        }
        state = next
        const audio = out.audio as OrtTensor
        const len = Number((out.audio_lengths as OrtTensor).data[0])
        if (len <= 0) return
        const total = audio.dims[2] as number
        const data = audio.data as Float32Array
        const mono = new Float32Array(len)
        for (let ch = 0; ch < channels; ch++)
          for (let i = 0; i < len; i++) mono[i] = (mono[i] as number) + (data[ch * total + i] as number) / channels
        firstAt ??= now()
        emitted += len
        emit(mono)
      }
      frames += await this.generateFrames(chunk.ids, opts, random, async (codes) => {
        pending.push(codes)
        // The reference decodes one frame at a time while playback is ahead of it. Here a codec call costs about
        // as much for one frame as for four (its 25 MB of state goes in and out of WebAssembly memory each time),
        // so once the first audio is out it takes at least four.
        const budget = Math.max(decodeBudget(emitted, rate, firstAt, now()), firstAt == null ? 1 : MIN_DECODE_FRAMES)
        if (pending.length >= budget) await decode(Math.min(pending.length, budget))
      })
      if (pending.length) await decode(pending.length)
      if (chunk.pause > 0 && !opts.cancelled?.()) {
        const silence = new Float32Array(Math.round(rate * chunk.pause))
        emitted += silence.length
        emit(silence)
      }
    }
    return { frames, chunks }
  }
}

/** Free a key/value cache kept on the GPU (on the CPU it is ordinary memory). */
function release(feeds: Feeds): void {
  for (const t of Object.values(feeds)) if (t.location === 'gpu-buffer') t.dispose?.()
}

function lastHidden(t: OrtTensor): Float32Array {
  const dim = t.dims[t.dims.length - 1] as number
  const data = t.data as Float32Array
  return Float32Array.from(data.subarray(data.length - dim))
}
