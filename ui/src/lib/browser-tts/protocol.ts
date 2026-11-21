import type { FetchRecord, FileProgress, LoadStage } from '@/lib/browser-stt/models'
import type { GpuPower } from '@/lib/webgpu'
import type { VoiceConfig } from './text'

export type TtsWorkerRequest =
  | {
      type: 'load'
      /** Absolute URL of the voices directory (ends in a slash). */
      base: string
      voice: string
      kind: 'coqui' | 'piper' | 'kokoro' | 'tiny-voca' | 'paradee' | 'kitten' | 'kale'
      /** Absolute URLs of the files (published ones carry a version query, so a stale cached copy is not used). */
      model: string
      config: string | null
      configFormat: 'bolo' | 'piper' | null
      /** Kokoro's style vectors for the voice. */
      style?: string
      phonemizer: 'uccharan' | 'dizione' | 'kale' | null
      /** The voice's name in its model (tiny-voca, little-tts, Paradee, Kitten, kale). */
      speaker?: string
      /** Further files by role: Kitten's voice pack (`voices`); kale's front end and voice files. */
      assets?: Record<string, string>
      /** uccharan's language for the phonemes. */
      phonemeLanguage?: string
      /** Directory with uccharan's language data (ends in a slash). */
      uccharanBase: string
      threads: number
      /** Run the graph on WebGPU (operators it lacks fall back to the CPU) or on the CPU only. */
      device: 'webgpu' | 'wasm'
      /** The graph cut in two at build time: front on WebGPU, back on the CPU (instead of `model`). */
      halves?: { front: string; back: string }
      /** Directory with ort-wasm-simd-threaded.{wasm,mjs}; defaults to the jsDelivr copy of the bundled version. */
      ortWasmBase?: string
      /** Leave fetching the runtime binary to ONNX Runtime (no progress, no shared cache). */
      plainRuntimeLoad: boolean
      /** Maximum of the runtime's WebAssembly memory, in 64 KiB pages. */
      memoryPages: number
      /** Which GPU WebGPU asks for (the device panel's choice); high performance when absent. */
      power?: GpuPower
    }
  | {
      /**
       * Kokoro: speak with a blend of voices' style vectors, added up with these weights (normalised to sum to 1),
       * as the server blends `af_bella(0.7)+af_sky(0.3)`. An empty list goes back to the voice's own style.
       */
      type: 'style'
      id: number
      styles: { url: string; weight: number }[]
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
  | {
      type: 'ready'
      loadMs: number
      config: VoiceConfig
      threads: number
      cached: boolean
      /** Where it ended up (WebGPU falls back to the CPU when it fails). */
      device: 'webgpu' | 'wasm' | 'split'
      /** Size of the runtime's WebAssembly memory after loading, when known. */
      memoryBytes: number | null
      /** Cache Storage keys of this voice's own files (for deleting it later without listing the cache). */
      cacheKeys?: string[]
    }
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
      memoryBytes: number | null
    }
  | { type: 'styled'; id: number; ms: number }
  | { type: 'note'; text: string }
  | { type: 'error'; id?: number; message: string }

/** Cache Storage bucket for voice files (models, settings, phonemiser data). */
export const VOICE_CACHE = 'bolo-voices'
