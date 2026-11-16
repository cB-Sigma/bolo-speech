import { resample } from '@/lib/audio/resample'
import { CLONE_RATE } from './protocol'

/**
 * How close a generated voice is to the reference: the cosine between two
 * speaker embeddings from 3D-Speaker CAM++ (VoxCeleb), the model the Speakers
 * page offers, in the same ONNX form.
 *
 * Thresholds, calibrated with ui/scripts/clone/calibrate_similarity.mjs on clips from six of this demo's own
 * voices (SYSPIN Indian English and Hindi, female and male; Kokoro am_michael and af_heart; two sentences
 * each, made by make_examples.py --calibration): the same voice saying different sentences scored 0.80 to
 * 0.86; different voices −0.11 to 0.51 (the highest: SYSPIN's English and Hindi male voices, recorded alike).
 * Real voices recorded on different days score lower than these clean pairs, so "likely the same" starts at
 * 0.6 and everything from 0.4 to 0.6 is called unclear.
 */
export const SAME_SPEAKER = 0.6
export const DIFFERENT_SPEAKER = 0.4

export const SIMILARITY_MODEL = {
  id: 'campplus-voxceleb',
  label: '3D-Speaker CAM++ (VoxCeleb)',
  repo: 'csukuangfj/speaker-embedding-models',
  revision: '0743f301363dec56491a490f6d6cbc9d67f9a3bf',
  file: '3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx',
  bytes: 29_597_000,
  peakBytes: 200e6,
  license: 'Apache-2.0',
}

/** WeSpeaker's filterbank settings (from its preprocessor config), which CAM++ was trained with too. */
export const SIMILARITY_FEATURES = {
  repo: 'onnx-community/wespeaker-voxceleb-resnet34-LM',
  revision: '6a61a1833ff2583aabeba044f5c8221f00b67ceb',
}

export const SIMILARITY_DOWNLOAD_ID = 'clone-similarity-campplus'

export function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0
  let na = 0
  let nb = 0
  const n = Math.min(a.length, b.length)
  for (let i = 0; i < n; i++) {
    const x = a[i] as number
    const y = b[i] as number
    dot += x * y
    na += x * x
    nb += y * y
  }
  return na && nb ? dot / Math.sqrt(na * nb) : 0
}

export type SimilarityVerdict = 'same' | 'unclear' | 'different'

export function verdictFor(score: number): SimilarityVerdict {
  if (score >= SAME_SPEAKER) return 'same'
  if (score < DIFFERENT_SPEAKER) return 'different'
  return 'unclear'
}

export const VERDICT_WORDS: Record<SimilarityVerdict, string> = {
  same: 'Likely the same speaker',
  unclear: 'Unclear',
  different: 'Likely a different speaker',
}

/** 24 kHz clip → 16 kHz for the embedding model, at most `maxSeconds` from the middle (long clips add little). */
export function forEmbedding(audio: Float32Array, rate = CLONE_RATE, maxSeconds = 20): Float32Array {
  const max = Math.floor(maxSeconds * rate)
  const start = audio.length > max ? Math.floor((audio.length - max) / 2) : 0
  const clip = audio.subarray(start, start + Math.min(max, audio.length))
  return rate === 16_000 ? Float32Array.from(clip) : resample(clip, rate, 16_000)
}
