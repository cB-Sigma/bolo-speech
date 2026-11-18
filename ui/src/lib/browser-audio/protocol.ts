import type { FetchRecord } from '@/lib/browser-stt/models'
import type { GpuPower } from '@/lib/webgpu'
import type { AudioModel } from './models'

/** Cache Storage bucket for the small audio models. */
export const AUDIO_CACHE = 'bolo-audio-models'

export type AudioWorkerRequest =
  | {
      type: 'load'
      model: AudioModel
      /** Absolute URL of each of the model's files, by role (`model`, `labels`, `module`, `wasm`). */
      urls: Record<string, string>
      device: 'wasm' | 'webgpu'
      threads: number
      ortWasmBase?: string
      power?: GpuPower
      /** Speaker embedders: where WeSpeaker's filterbank settings are (Hub host, repository, revision). */
      features?: { host: string; repo: string; revision: string }
    }
  | { type: 'denoise'; id: number; audio: Float32Array; rate: number; strength: number }
  | { type: 'vad'; id: number; audio: Float32Array; reset?: boolean }
  | { type: 'turn'; id: number; audio: Float32Array }
  | { type: 'embed'; id: number; audio: Float32Array }
  | { type: 'lid'; id: number; audio: Float32Array; top?: number }
  | { type: 'tag'; id: number; audio: Float32Array; top?: number; window?: number }
  /** Punctuation and capitals for a transcript. */
  | { type: 'punct'; id: number; text: string }

export type AudioWorkerResponse =
  | { type: 'progress'; files: { file: string; loaded: number; total: number }[] }
  | { type: 'fetch'; record: FetchRecord }
  | { type: 'ready'; loadMs: number; device: 'wasm' | 'webgpu'; bytes: number; cacheKeys: string[] }
  | { type: 'audio'; id: number; audio: Float32Array; ms: number }
  /** Speech probability per frame of `frame` samples. */
  | { type: 'probs'; id: number; probs: Float32Array; frame: number; ms: number }
  | { type: 'value'; id: number; value: number; ms: number }
  | { type: 'vector'; id: number; vector: Float32Array; ms: number }
  | { type: 'langs'; id: number; ranked: [string, number][]; ms: number }
  /** Sound events: the whole clip, and each window when one was asked for. */
  | {
      type: 'tags'
      id: number
      ranked: [string, number][]
      windows: { start: number; end: number; ranked: [string, number][] }[]
      ms: number
    }
  | { type: 'text'; id: number; text: string; ms: number }
  | { type: 'error'; id?: number; message: string }
