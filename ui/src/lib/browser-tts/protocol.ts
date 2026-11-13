import type { FetchRecord, FileProgress, LoadStage } from '@/lib/browser-stt/models'
import type { VoiceConfig } from './text'

export type TtsWorkerRequest =
  | {
      type: 'load'
      /** Absolute URL of the voices directory (ends in a slash). */
      base: string
      voice: string
      /** Changes when the published files change, so a stale cached copy is not used. */
      version: string
      phonemizer: 'uccharan' | null
      threads: number
      /** Directory with ort-wasm-simd-threaded.{wasm,mjs}; defaults to the jsDelivr copy of the bundled version. */
      ortWasmBase?: string
      /** Leave fetching the runtime binary to ONNX Runtime (no progress, no shared cache). */
      plainRuntimeLoad: boolean
    }
  | {
      type: 'synth'
      id: number
      /** One chunk, already prepared by prepareText (cleaned for character voices). */
      text: string
      scales: [number, number, number]
    }

export type TtsWorkerResponse =
  | { type: 'progress'; files: FileProgress[] }
  | { type: 'stage'; stage: LoadStage; loaded?: number; total?: number }
  | { type: 'fetch'; record: FetchRecord }
  | { type: 'ready'; loadMs: number; config: VoiceConfig; threads: number; cached: boolean }
  | {
      type: 'audio'
      id: number
      audio: Float32Array
      /** Time spent in the model (and the phonemiser), ms. */
      ms: number
      /** Number of model input ids. */
      tokens: number
      /** Phonemes, for voices that read them. */
      phonemes?: string
      /** Phonemes the voice has no id for (skipped). */
      missing: string[]
    }
  | { type: 'error'; id?: number; message: string }

/** Cache Storage bucket for voice files (models, settings, phonemiser data). */
export const VOICE_CACHE = 'bolo-voices'
