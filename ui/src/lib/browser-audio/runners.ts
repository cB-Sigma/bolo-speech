import type * as Ort from 'onnxruntime-web/wasm'
import { AutoFeatureExtractor, env } from '@huggingface/transformers'
import { fbank60, periodicHann, RealFft, reflectPad, softmax, whisperLogMel } from './dsp'
import type { AudioModel } from './models'
import { onnxMetadata } from './onnx-meta'

/**
 * What runs each kind of model inside the audio worker. Every denoiser here follows the server's code (bolo's
 * engines and sherpa-onnx, which it uses): the same STFT, the same model state from frame to frame, the same delay
 * cut off so the output lines up with the input.
 */

export interface Runner {
  rate: number
  /** Samples per voice-activity frame. */
  frame?: number
  denoise?: (audio: Float32Array) => Promise<Float32Array>
  vad?: (audio: Float32Array) => Promise<Float32Array>
  reset?: () => void
  turn?: (audio: Float32Array) => Promise<number>
  embed?: (audio: Float32Array) => Promise<Float32Array>
  lid?: (audio: Float32Array) => Promise<[string, number][]>
  warmup?: () => Promise<void>
  release: () => Promise<void>
}

export interface RunnerContext {
  ort: typeof Ort | null
  session: (data: ArrayBuffer) => Promise<Ort.InferenceSession>
  files: Record<string, ArrayBuffer>
  /** Where each file came from (a module loaded from a blob still needs its own address). */
  urls: Record<string, string>
  model: AudioModel
  features?: { host: string; repo: string; revision: string }
}

type Tensor = Ort.Tensor

function file(ctx: RunnerContext, key: string): ArrayBuffer {
  const f = ctx.files[key]
  if (!f) throw new Error(`${ctx.model.label}: ${key} is missing`)
  return f
}

function f32(ctx: RunnerContext, data: Float32Array, dims: readonly number[]): Tensor {
  return new (ctx.ort as typeof Ort).Tensor('float32', data, dims)
}

/**
 * Import a cached JavaScript module (an Emscripten build) from a blob. Its `import.meta.url` is replaced by the
 * address it was downloaded from, which it uses to find its other files.
 */
async function importCached<T>(ctx: RunnerContext, key: string): Promise<{ mod: T; url: string }> {
  const source = new TextDecoder()
    .decode(file(ctx, key))
    .replaceAll('import.meta.url', JSON.stringify(ctx.urls[key] ?? ''))
  const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }))
  return { mod: (await import(/* @vite-ignore */ url)) as T, url }
}

const release = (s: Ort.InferenceSession | null) => async () => {
  await s?.release().catch(() => undefined)
}

/** A test tone, to compile the graph before the first real request. */
const tone = (n: number) => Float32Array.from({ length: n }, (_, i) => Math.sin(i / 7) * 0.05)

// -------------------------------------------------------------------------------------------------------------
// Denoisers
// -------------------------------------------------------------------------------------------------------------

/**
 * A spectral denoiser that sees one STFT frame at a time and carries its state: centred frames with reflect
 * padding, the model on each frame, then overlap-add with the same window normalised by its squared sum (what
 * torch.istft, librosa.istft and kaldi-native-fbank's IStft do).
 */
async function stftDenoise(
  audio: Float32Array,
  nFft: number,
  window: Float64Array,
  step: (re: Float64Array, im: Float64Array) => Promise<[Float32Array, Float32Array]>,
): Promise<Float32Array> {
  const hop = nFft / 2
  const fft = new RealFft(nFft)
  const padded = reflectPad(audio, nFft / 2)
  const frames = 1 + Math.floor((padded.length - nFft) / hop)
  const out = new Float64Array((frames - 1) * hop + nFft)
  const norm = new Float64Array(out.length)
  const frame = new Float64Array(nFft)
  const re = new Float64Array(fft.bins)
  const im = new Float64Array(fft.bins)
  const time = new Float64Array(nFft)
  for (let t = 0; t < frames; t++) {
    for (let i = 0; i < nFft; i++) frame[i] = (padded[t * hop + i] as number) * (window[i] as number)
    fft.forward(frame, re, im)
    const [er, ei] = await step(re, im)
    fft.inverse(er, ei, time)
    for (let i = 0; i < nFft; i++) {
      const w = window[i] as number
      out[t * hop + i] = (out[t * hop + i] as number) + (time[i] as number) * w
      norm[t * hop + i] = (norm[t * hop + i] as number) + w * w
    }
  }
  const y = new Float32Array(audio.length)
  for (let i = 0; i < y.length; i++) {
    const j = i + nFft / 2
    const n = norm[j] ?? 0
    y[i] = j < out.length && n > 1e-8 ? (out[j] as number) / n : 0
  }
  return y
}

/** Interleaved [bin][re, im] float32 from separate arrays, and back. */
function interleave(re: Float64Array, im: Float64Array): Float32Array {
  const out = new Float32Array(re.length * 2)
  for (let k = 0; k < re.length; k++) {
    out[2 * k] = re[k] as number
    out[2 * k + 1] = im[k] as number
  }
  return out
}
function split(data: ArrayLike<number>, bins: number): [Float32Array, Float32Array] {
  const re = new Float32Array(bins)
  const im = new Float32Array(bins)
  for (let k = 0; k < bins; k++) {
    re[k] = data[2 * k] as number
    im[k] = data[2 * k + 1] as number
  }
  return [re, im]
}

/** GTCRN in sherpa-onnx's streaming export: 512-point STFT, hop 256, square-root Hann window, three caches. */
async function gtcrn(ctx: RunnerContext): Promise<Runner> {
  const meta = onnxMetadata(file(ctx, 'model'))
  const session = await ctx.session(file(ctx, 'model'))
  const shape = (key: string, fallback: number[]) =>
    (meta[key] ?? '')
      .split(',')
      .map(Number)
      .filter((x) => x > 0).length
      ? (meta[key] as string).split(',').map(Number)
      : fallback
  const shapes = {
    conv_cache: shape('conv_cache_shape', [2, 1, 16, 16, 33]),
    tra_cache: shape('tra_cache_shape', [2, 3, 1, 1, 16]),
    inter_cache: shape('inter_cache_shape', [2, 1, 33, 16]),
  }
  const nFft = Number(meta.n_fft ?? 512)
  const window = Float64Array.from(periodicHann(nFft), Math.sqrt)
  const zeros = (dims: number[]) => f32(ctx, new Float32Array(dims.reduce((a, b) => a * b, 1)), dims)
  const denoise = async (audio: Float32Array) => {
    let caches: Record<string, Tensor> = {
      conv_cache: zeros(shapes.conv_cache),
      tra_cache: zeros(shapes.tra_cache),
      inter_cache: zeros(shapes.inter_cache),
    }
    return stftDenoise(audio, nFft, window, async (re, im) => {
      const out = await session.run({ mix: f32(ctx, interleave(re, im), [1, re.length, 1, 2]), ...caches })
      caches = {
        conv_cache: out.conv_cache_out as Tensor,
        tra_cache: out.tra_cache_out as Tensor,
        inter_cache: out.inter_cache_out as Tensor,
      }
      return split((out.enh as Tensor).data as Float32Array, re.length)
    })
  }
  return {
    rate: Number(meta.sample_rate ?? 16000),
    denoise,
    warmup: async () => void (await denoise(tone(4096))),
    release: release(session),
  }
}

/** Vorbis window (DPDFNet's), which overlap-adds to one at half overlap. */
const vorbis = (n: number) =>
  Float64Array.from({ length: n }, (_, i) =>
    Math.sin(0.5 * Math.PI * Math.sin((0.5 * Math.PI * (i + 0.5)) / (n / 2)) ** 2),
  )

/**
 * DPDFNet's streaming ONNX export, as Ceva's own onnx_model/infer_dpdfnet_onnx.py runs it: centred STFT with the
 * Vorbis window, one frame per call with the state the metadata describes, inverse STFT, and the output advanced
 * by two window lengths (the model's delay).
 */
async function dpdfnet(ctx: RunnerContext): Promise<Runner> {
  const meta = onnxMetadata(file(ctx, 'model'))
  const session = await ctx.session(file(ctx, 'model'))
  const nFft = Number(meta.n_fft ?? 320)
  const stateSize = Number(meta.state_size)
  if (!stateSize) throw new Error('DPDFNet: the model carries no state size in its metadata')
  const erb = (meta.erb_norm_init ?? '').split(',').map(Number)
  const spec = (meta.spec_norm_init ?? '').split(',').map(Number)
  const initial = new Float32Array(stateSize)
  initial.set(erb, 0)
  initial.set(spec, Number(meta.erb_norm_state_size ?? erb.length))
  const window = vorbis(nFft)
  const [specIn, stateIn] = session.inputNames as [string, string]
  const [specOut, stateOut] = session.outputNames as [string, string]
  const denoise = async (audio: Float32Array) => {
    let state = f32(ctx, initial.slice(), [stateSize])
    const y = await stftDenoise(audio, nFft, window, async (re, im) => {
      const out = await session.run({ [specIn]: f32(ctx, interleave(re, im), [1, 1, re.length, 2]), [stateIn]: state })
      state = out[stateOut] as Tensor
      return split((out[specOut] as Tensor).data as Float32Array, re.length)
    })
    const shift = 2 * nFft
    const aligned = new Float32Array(y.length)
    aligned.set(y.subarray(Math.min(shift, y.length)))
    return aligned
  }
  return {
    rate: Number(meta.sample_rate ?? 16000),
    denoise,
    warmup: async () => void (await denoise(tone(4096))),
    release: release(session),
  }
}

/** Frames of `size` samples with room for the model's delay to come out; and the input length. */
function framed(audio: Float32Array, size: number, delay: number, scale = 1): Float32Array {
  const n = audio.length
  const pad = (((-(n + delay) % size) + size) % size) + delay
  const out = new Float32Array(n + pad)
  for (let i = 0; i < n; i++) out[i] = (audio[i] as number) * scale
  return out
}

/** DeepFilterNet 3 as torchDF's single streaming graph: 480 samples at 48 kHz per call, delay 1440 samples. */
async function deepFilterNet(ctx: RunnerContext): Promise<Runner> {
  const session = await ctx.session(file(ctx, 'model'))
  const FRAME = 480
  const DELAY = 3 * FRAME
  const size = 45_304
  const limit = f32(ctx, new Float32Array([0]), [])
  const denoise = async (audio: Float32Array) => {
    const x = framed(audio, FRAME, DELAY)
    const out = new Float32Array(x.length)
    let states = f32(ctx, new Float32Array(size), [size])
    for (let i = 0; i < x.length; i += FRAME) {
      const r = await session.run({
        input_frame: f32(ctx, x.slice(i, i + FRAME), [FRAME]),
        states,
        atten_lim_db: limit,
      })
      states = r.new_states as Tensor
      out.set((r.enhanced_audio_frame as Tensor).data as Float32Array, i)
    }
    return out.slice(DELAY, DELAY + audio.length)
  }
  return {
    rate: 48000,
    denoise,
    warmup: async () => void (await denoise(tone(4800))),
    release: release(session),
  }
}

interface RnnoiseModule {
  Rnnoise: {
    load(): Promise<{
      frameSize: number
      createDenoiseState(): { processFrame(frame: Float32Array): number; destroy(): void }
    }>
  }
}

/** RNNoise's own WebAssembly build (@shiguredo/rnnoise-wasm): 480 samples at 48 kHz, 16-bit scale, delay 960. */
async function rnnoise(ctx: RunnerContext): Promise<Runner> {
  const { mod, url } = await importCached<RnnoiseModule>(ctx, 'module')
  const lib = await mod.Rnnoise.load()
  const FRAME = lib.frameSize
  const DELAY = 2 * FRAME
  const denoise = async (audio: Float32Array) => {
    const x = framed(audio, FRAME, DELAY, 32768)
    const state = lib.createDenoiseState()
    try {
      for (let i = 0; i < x.length; i += FRAME) {
        const frame = x.subarray(i, i + FRAME)
        state.processFrame(frame)
      }
    } finally {
      state.destroy()
    }
    const out = x.slice(DELAY, DELAY + audio.length)
    for (let i = 0; i < out.length; i++) out[i] = (out[i] as number) / 32768
    return out
  }
  return { rate: 48000, denoise, release: async () => URL.revokeObjectURL(url) }
}

export async function makeDenoiser(ctx: RunnerContext): Promise<Runner> {
  switch (ctx.model.runner) {
    case 'gtcrn':
      return gtcrn(ctx)
    case 'dpdfnet':
      return dpdfnet(ctx)
    case 'deepfilternet3':
      return deepFilterNet(ctx)
    case 'rnnoise':
      return rnnoise(ctx)
    default:
      throw new Error(`${ctx.model.id} is not a denoiser`)
  }
}

// -------------------------------------------------------------------------------------------------------------
// Voice-activity detectors
// -------------------------------------------------------------------------------------------------------------

/** Cuts a stream into frames of `size` samples, keeping what is left over for the next call. */
class Framer {
  private carry = new Float32Array(0)
  readonly size: number
  constructor(size: number) {
    this.size = size
  }
  push(audio: Float32Array): Float32Array[] {
    const data = new Float32Array(this.carry.length + audio.length)
    data.set(this.carry)
    data.set(audio, this.carry.length)
    const frames: Float32Array[] = []
    let o = 0
    for (; o + this.size <= data.length; o += this.size) frames.push(data.slice(o, o + this.size))
    this.carry = data.slice(o)
    return frames
  }
  reset(): void {
    this.carry = new Float32Array(0)
  }
}

/** Silero VAD v4 (inputs `input`, `sr`, `h`, `c`) or v5 / v6 (`input` with 64 samples of context, `sr`, `state`). */
async function silero(ctx: RunnerContext, v5: boolean): Promise<Runner> {
  const ort = ctx.ort as typeof Ort
  const session = await ctx.session(file(ctx, 'model'))
  const FRAME = 512
  const CONTEXT = v5 ? 64 : 0
  const framer = new Framer(FRAME)
  const sr = new ort.Tensor('int64', BigInt64Array.from([16000n]), [])
  let h: Tensor
  let c: Tensor
  let state: Tensor
  let context = new Float32Array(CONTEXT)
  const reset = () => {
    framer.reset()
    h = f32(ctx, new Float32Array(128), [2, 1, 64])
    c = f32(ctx, new Float32Array(128), [2, 1, 64])
    state = f32(ctx, new Float32Array(256), [2, 1, 128])
    context = new Float32Array(CONTEXT)
  }
  reset()
  const vad = async (audio: Float32Array) => {
    const frames = framer.push(audio)
    const probs = new Float32Array(frames.length)
    for (let i = 0; i < frames.length; i++) {
      const frame = frames[i] as Float32Array
      if (v5) {
        const input = new Float32Array(CONTEXT + FRAME)
        input.set(context)
        input.set(frame, CONTEXT)
        context = frame.slice(FRAME - CONTEXT)
        const out = await session.run({ input: f32(ctx, input, [1, input.length]), state, sr })
        state = out.stateN as Tensor
        probs[i] = ((out.output as Tensor).data as Float32Array)[0] as number
      } else {
        const out = await session.run({ input: f32(ctx, frame, [1, FRAME]), sr, h, c })
        h = out.hn as Tensor
        c = out.cn as Tensor
        probs[i] = ((out.output as Tensor).data as Float32Array)[0] as number
      }
    }
    return probs
  }
  return {
    rate: 16000,
    frame: FRAME,
    vad,
    reset,
    warmup: async () => {
      await vad(tone(FRAME * 2))
      reset()
    },
    release: release(session),
  }
}

interface TenVadModule {
  _ten_vad_create(handlePtr: number, hop: number, threshold: number): number
  _ten_vad_process(handle: number, audioPtr: number, length: number, probPtr: number, flagPtr: number): number
  _ten_vad_destroy(handlePtr: number): number
  _malloc(size: number): number
  _free(ptr: number): void
  HEAP16: Int16Array
  HEAPF32: Float32Array
  HEAP32: Int32Array
}

/** TEN VAD's own WebAssembly build: 256-sample hops at 16 kHz, 16-bit samples in, a probability out. */
async function tenVad(ctx: RunnerContext): Promise<Runner> {
  const { mod, url } = await importCached<{ default: (o: object) => Promise<TenVadModule> }>(ctx, 'module')
  const m = await mod.default({ wasmBinary: file(ctx, 'wasm') })
  const HOP = 256
  const framer = new Framer(HOP)
  const handlePtr = m._malloc(4)
  const audioPtr = m._malloc(HOP * 2)
  const probPtr = m._malloc(4)
  const flagPtr = m._malloc(4)
  let handle = 0
  const create = () => {
    if (handle) m._ten_vad_destroy(handlePtr)
    if (m._ten_vad_create(handlePtr, HOP, 0.5) !== 0) throw new Error('TEN VAD could not start')
    handle = m.HEAP32[handlePtr >> 2] as number
  }
  create()
  const vad = async (audio: Float32Array) => {
    const frames = framer.push(audio)
    const probs = new Float32Array(frames.length)
    for (let i = 0; i < frames.length; i++) {
      const f = frames[i] as Float32Array
      const base = audioPtr >> 1
      for (let k = 0; k < HOP; k++)
        m.HEAP16[base + k] = Math.max(-32768, Math.min(32767, Math.round((f[k] as number) * 32767)))
      if (m._ten_vad_process(handle, audioPtr, HOP, probPtr, flagPtr) !== 0) throw new Error('TEN VAD failed')
      probs[i] = m.HEAPF32[probPtr >> 2] as number
    }
    return probs
  }
  return {
    rate: 16000,
    frame: HOP,
    vad,
    reset: () => {
      framer.reset()
      create()
    },
    release: async () => {
      if (handle) m._ten_vad_destroy(handlePtr)
      for (const p of [handlePtr, audioPtr, probPtr, flagPtr]) m._free(p)
      URL.revokeObjectURL(url)
    },
  }
}

export async function makeVad(ctx: RunnerContext): Promise<Runner> {
  switch (ctx.model.runner) {
    case 'silero-v4':
      return silero(ctx, false)
    case 'silero-v5':
      return silero(ctx, true)
    case 'ten-vad':
      return tenVad(ctx)
    default:
      throw new Error(`${ctx.model.id} is not a voice-activity detector`)
  }
}

// -------------------------------------------------------------------------------------------------------------
// End of turn, speaker embeddings, language identification
// -------------------------------------------------------------------------------------------------------------

/** Smart Turn v3: the last 8 s (zeros in front when shorter), Whisper features of the normalised audio, one probability. */
export async function makeTurn(ctx: RunnerContext): Promise<Runner> {
  const session = await ctx.session(file(ctx, 'model'))
  const WANT = 8 * 16000
  const input = session.inputNames[0] as string
  const turn = async (audio: Float32Array) => {
    const x = new Float32Array(WANT)
    const tail = audio.subarray(Math.max(0, audio.length - WANT))
    x.set(tail, WANT - tail.length)
    const { data, frames } = whisperLogMel(x, true)
    const out = await session.run({ [input]: f32(ctx, data, [1, 80, frames]) })
    return ((out[session.outputNames[0] as string] as Tensor).data as Float32Array)[0] as number
  }
  return { rate: 16000, turn, warmup: async () => void (await turn(tone(16000))), release: release(session) }
}

/** 3D-Speaker embedders in sherpa-onnx's form, fed WeSpeaker's 80-band filterbank (as the cloning page does). */
export async function makeEmbedder(ctx: RunnerContext): Promise<Runner> {
  const session = await ctx.session(file(ctx, 'model'))
  const input = session.inputNames[0] as string
  const output = session.outputNames[0] as string
  if (input === 'waveform') {
    // The filterbank is inside the graph (SpeechBrain's ECAPA-TDNN export): 16 kHz samples in.
    const embed = async (audio: Float32Array) => {
      const out = await session.run({ [input]: f32(ctx, Float32Array.from(audio), [1, audio.length]) })
      return Float32Array.from((out[output] as Tensor).data as Float32Array)
    }
    return { rate: 16000, embed, warmup: async () => void (await embed(tone(16000))), release: release(session) }
  }
  const f = ctx.features
  if (!f) throw new Error('The filterbank settings are missing')
  env.allowLocalModels = false
  env.remoteHost = f.host
  const extractor = await AutoFeatureExtractor.from_pretrained(f.repo, { revision: f.revision } as never)
  const embed = async (audio: Float32Array) => {
    const { input_features } = (await (
      extractor as unknown as (x: Float32Array) => Promise<{ input_features: { data: Float32Array; dims: number[] } }>
    )(audio)) as { input_features: { data: Float32Array; dims: number[] } }
    const x = f32(ctx, Float32Array.from(input_features.data), input_features.dims)
    const out = await session.run({ [input]: x })
    return Float32Array.from((out[output] as Tensor).data as Float32Array)
  }
  return { rate: 16000, embed, warmup: async () => void (await embed(tone(16000))), release: release(session) }
}

/** The language codes of a labels file: `ab: Abkhazian` per line, or one code per line. */
export function readLabels(text: string): string[] {
  return text
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .map((l) => (l.split(':')[0] ?? '').trim().toLowerCase().replaceAll('_', '-').split('-')[0] ?? '')
}

/** Labels with their probability, most likely first; labels that name one code are added up (as the server). */
export function rankLanguages(labels: string[], probs: ArrayLike<number>): [string, number][] {
  const total = new Map<string, number>()
  for (let i = 0; i < Math.min(labels.length, probs.length); i++) {
    const l = labels[i] as string
    total.set(l, (total.get(l) ?? 0) + (probs[i] as number))
  }
  return [...total].sort((a, b) => b[1] - a[1])
}

/** VoxLingua107 ECAPA (8-bit): SpeechBrain's 60-band filterbank of at most 30 s, 107 language scores. */
export async function makeLid(ctx: RunnerContext): Promise<Runner> {
  const session = await ctx.session(file(ctx, 'model'))
  const labels = readLabels(new TextDecoder().decode(file(ctx, 'labels')))
  const MAX = 30 * 16000
  const lid = async (audio: Float32Array) => {
    let x = audio.subarray(0, MAX)
    if (x.length < 1600) {
      const padded = new Float32Array(1600)
      padded.set(x)
      x = padded
    }
    const { data, frames } = fbank60(x)
    const out = await session.run({ [session.inputNames[0] as string]: f32(ctx, data, [1, frames, 60]) })
    const scores = (out[session.outputNames[0] as string] as Tensor).data as Float32Array
    return rankLanguages(labels, softmax(scores.subarray(0, labels.length)))
  }
  return { rate: 16000, lid, warmup: async () => void (await lid(tone(16000))), release: release(session) }
}
