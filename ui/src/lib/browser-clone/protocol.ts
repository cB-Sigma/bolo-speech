import type { FileProgress } from '@/lib/browser-stt/models'
import type { GpuPower } from '@/lib/webgpu'
import type { CloneModel } from './models'

/** Sample rate of every reference and every output on the page. */
export const CLONE_RATE = 24_000

export interface GenerateOptions {
  /** Language of the text (multilingual models). */
  language: string
  /** Pocket TTS: noise scale of the flow sampler (0 = always the same). */
  temperature: number
  /** Pocket TTS: Euler steps of the flow sampler. */
  lsdSteps: number
  /** Chatterbox Multilingual: how expressive (0.25 to 2). */
  exaggeration: number
  seed?: number
}

export type CloneWorkerRequest =
  | {
      type: 'load'
      model: CloneModel
      device: 'webgpu' | 'wasm'
      threads: number
      /** Origin serving model files in Hub layout (tests); defaults to huggingface.co. */
      modelHost?: string
      ortWasmBase?: string
      memoryPages: number
      /** Which GPU WebGPU asks for (the device panel's choice); high performance when absent. */
      power?: GpuPower
    }
  /** Take in a reference (24 kHz mono) under `key`; generate calls name the key. */
  | { type: 'voice'; id: number; key: string; audio: Float32Array }
  | { type: 'generate'; id: number; key: string; text: string; options: GenerateOptions }
  | { type: 'cancel'; id: number }

export type CloneWorkerResponse =
  | { type: 'progress'; files: FileProgress[] }
  | { type: 'stage'; stage: 'runtime' | 'model' | 'compiling' | 'warmup' }
  | { type: 'note'; text: string }
  | {
      type: 'ready'
      loadMs: number
      device: 'webgpu' | 'wasm'
      cacheKeys: string[]
      bytes: number
      threads: number
    }
  | { type: 'voice'; id: number; ms: number }
  | { type: 'audio'; id: number; samples: Float32Array }
  /** Still working on request `id` (a language-model step, at most about once a second): the stall watchdog's pulse. */
  | { type: 'tick'; id: number }
  | { type: 'done'; id: number; ms: number; samples: number }
  | { type: 'error'; id?: number; message: string }

export type EmbedWorkerRequest =
  | { type: 'load'; threads: number; modelHost?: string; ortWasmBase?: string }
  | { type: 'embed'; id: number; audio: Float32Array }

export type EmbedWorkerResponse =
  | { type: 'progress'; files: FileProgress[] }
  | { type: 'ready'; loadMs: number; cacheKeys: string[]; bytes: number }
  | { type: 'embedding'; id: number; vector: Float32Array; ms: number }
  | { type: 'error'; id?: number; message: string }
