import type { CatalogModel, Component, Hardware, PhonemizerChoice, ServerInfo, Voice } from './api/types'

/**
 * Reading what a server can do (`GET /api/v1/info` capabilities, catalog
 * fields) into the shapes the screens need. Everything tolerates older servers
 * and the browser-only demo, which report none of it: the answer is then
 * "not available", never an exception.
 */

// --- Realtime pipelines ---------------------------------------------------------------------------------------

export type PipelineId = 'cascade' | 'audio-llm' | 'speech-to-speech'

export interface PipelineOption {
  id: PipelineId
  label: string
  summary: string
  available: boolean
  /** Why it cannot be used (or what it still needs), when not available. */
  reason: string | null
  /** What it would use: the language model, the audio model or the speech-to-speech model. */
  detail: string | null
  /** Licence caveat shown next to the option. */
  note: string | null
}

export const PIPELINE_IDS: PipelineId[] = ['cascade', 'audio-llm', 'speech-to-speech']

const PIPELINE_TEXT: Record<PipelineId, { label: string; summary: string }> = {
  cascade: {
    label: 'Cascade',
    summary: "bolo transcribes you, your language model answers in text, and bolo's voice speaks the answer.",
  },
  'audio-llm': {
    label: 'Audio LLM',
    summary: "An audio-understanding model hears your turn directly; bolo's voice speaks its answer.",
  },
  'speech-to-speech': {
    label: 'Speech to speech',
    summary:
      'A speech-to-speech model hears you and answers with its own voice. bolo keeps turn taking and transcripts.',
  },
}

/** LFM3-Audio, the first speech-to-speech backend, is under the LFM Open License. */
export const S2S_LICENCE_NOTE = 'LFM Open License: commercial use only below USD 10M revenue'

/** Whether the server knows about pipelines at all (newer than the voice-agent release). */
export function hasPipelines(info: ServerInfo | undefined): boolean {
  return Boolean(info?.capabilities?.realtime?.pipelines)
}

export function pipelineOptions(info: ServerInfo | undefined, components: Component[] = []): PipelineOption[] {
  const rt = info?.capabilities?.realtime
  const p = rt?.pipelines ?? {}
  const llm = info?.capabilities?.llm
  return PIPELINE_IDS.map((id) => {
    const text = PIPELINE_TEXT[id]
    const entry = p[id]
    let available = false
    let reason: string | null = null
    let detail: string | null
    if (id === 'cascade') {
      available = Boolean(entry?.available ?? llm?.configured)
      detail = llm?.model ? `language model ${llm.model}` : null
      if (!available) reason = 'No language model is configured (setting llm_base_url).'
    } else if (id === 'audio-llm') {
      available = Boolean(entry?.configured ?? entry?.available)
      detail = entry?.model ? `audio model ${entry.model}` : null
      if (!available)
        reason = 'No audio-understanding model is configured (settings audio_llm_base_url, audio_llm_model).'
    } else {
      // The server reports what it would use, not whether it can: a running server it was pointed at, or a
      // llama-server model it starts itself, which needs llama-server installed.
      detail = entry?.model ? `model ${entry.model}` : null
      if (entry?.url) {
        available = true
        detail = `${detail ? `${detail} at ` : ''}${entry.url}`
      } else if (entry?.model) {
        const llama = components.find((c) => c.id === 'llama-server')
        available = llama ? llama.installed : true
        if (!available)
          reason = 'Needs llama-server (install it on the Components screen) or the setting s2s_server_url.'
      } else if (entry) {
        reason = 'No speech-to-speech model is configured (settings s2s_model or s2s_server_url).'
      } else {
        reason = 'This server has no speech-to-speech pipeline.'
      }
    }
    if (!entry && id !== 'cascade') {
      available = false
      reason ??= 'This server does not offer this pipeline.'
    }
    return {
      id,
      ...text,
      available,
      reason: available ? null : reason,
      detail,
      note: id === 'speech-to-speech' ? S2S_LICENCE_NOTE : null,
    }
  })
}

/** The pipeline to preselect: the server default when usable, else the first usable one, else the default. */
export function defaultPipeline(info: ServerInfo | undefined, options: PipelineOption[]): PipelineId {
  const def = info?.capabilities?.realtime?.pipeline as PipelineId | undefined
  const usable = options.filter((o) => o.available)
  if (def && usable.some((o) => o.id === def)) return def
  return usable[0]?.id ?? def ?? 'cascade'
}

// --- Diarization ------------------------------------------------------------------------------------------------

export interface DiarizerOption {
  id: string
  label: string
  note: string | null
  isDefault: boolean
  installed: boolean
  usable: boolean
}

const DIARIZER_NOTES: Record<string, string> = {
  'nemotron-3-diarization': 'Better with overlapping or phone audio, misses more quiet speech',
  'pyannote-3.0-eres2net': 'Default: segmentation and clustering, small and fast',
}

/** Diarization models a request can name, the server default first. */
export function diarizerOptions(info: ServerInfo | undefined, catalog: CatalogModel[]): DiarizerOption[] {
  const def = info?.capabilities?.diarization?.default_model ?? 'pyannote-3.0-eres2net'
  const list = catalog
    .filter((m) => m.task === 'diarization' && m.status !== 'pending' && !isNotImplemented(m))
    .map((m) => ({
      id: m.id,
      label: m.id === def ? `${m.id} (default)` : m.id,
      note: DIARIZER_NOTES[m.id] ?? null,
      isDefault: m.id === def,
      installed: m.installed,
      usable: m.available !== false,
    }))
  return list.sort((a, b) => Number(b.isDefault) - Number(a.isDefault) || rankDiarizer(a.id) - rankDiarizer(b.id))
}

function rankDiarizer(id: string): number {
  if (id.startsWith('pyannote')) return 0
  if (id.startsWith('nemotron')) return 1
  if (id.startsWith('fake')) return 3
  return 2
}

/** Speaker-conditioned recognisers (DiCoW) the server lists, with their catalog entries when present. */
export function speakerConditionedModels(info: ServerInfo | undefined, catalog: CatalogModel[]): CatalogModel[] {
  const ids = new Set(info?.capabilities?.diarization?.speaker_conditioned_models ?? [])
  return catalog.filter(
    (m) => m.task === 'stt' && (ids.has(m.id) || m.features.includes('speaker-conditioned')) && m.status !== 'pending',
  )
}

export function separationModels(catalog: CatalogModel[]): CatalogModel[] {
  return catalog.filter((m) => m.task === 'separation' && m.status !== 'pending' && !isNotImplemented(m))
}

// --- Catalog entries ----------------------------------------------------------------------------------------------

export function isPending(m: Pick<CatalogModel, 'status'>): boolean {
  return m.status === 'pending'
}

/**
 * Entries the catalog describes but no runtime can run yet. The server says so only in the description ("Not
 * runnable yet: …"), or in a future `status`.
 */
export function isNotImplemented(m: Pick<CatalogModel, 'status' | 'description'>): boolean {
  return m.status === 'not_implemented' || m.status === 'not-implemented' || /not runnable yet/i.test(m.description)
}

/** True when a model is fine without a GPU. */
export function isCpuFriendly(m: Pick<CatalogModel, 'hardware'>): boolean {
  return !m.hardware || m.hardware.gpu === 'optional'
}

export function hardwareLabel(h: Hardware | null | undefined): string | null {
  if (!h) return null
  const gpu =
    h.gpu === 'required'
      ? 'GPU required'
      : h.gpu === 'recommended'
        ? 'GPU recommended'
        : h.gpu === 'optional'
          ? 'GPU optional'
          : `GPU ${h.gpu}`
  return h.vram_gb != null ? `${gpu}, ~${formatGb(h.vram_gb)}` : gpu
}

function formatGb(gb: number): string {
  return `${gb >= 10 ? Math.round(gb) : Math.round(gb * 10) / 10} GB`
}

/** Short name of the runtime behind a model, for voice families that do not run in bolo itself. */
export function runtimeLabel(m: Pick<CatalogModel, 'engine' | 'requires' | 'id'>): string | null {
  if (m.engine.startsWith('plugin:')) return `plugin ${m.engine.slice(7)}`
  if (m.engine === 'tts-server') {
    if (m.requires?.includes('audio.cpp') || m.id.endsWith('-audiocpp')) return 'audio.cpp'
    return 'llama.cpp'
  }
  return null
}

const ORPHEUS = /^(veena|svara|orpheus|snortts)/i

/** Voice family group for the text-to-speech picker. */
export function voiceFamily(m: Pick<CatalogModel, 'engine' | 'id' | 'family'>): string {
  if (m.engine === 'tts-server' && ORPHEUS.test(m.id)) return 'Orpheus family (llama.cpp)'
  if (m.engine === 'tts-server') return m.id.endsWith('-audiocpp') ? 'audio.cpp voices' : 'llama.cpp voices'
  if (m.engine.startsWith('plugin:')) return 'Engine plugin voices'
  return 'Built-in voices'
}

export function canClone(m: Pick<CatalogModel, 'features'> | null | undefined): boolean {
  return Boolean(m?.features.includes('voice-cloning'))
}

// --- Phonemisers --------------------------------------------------------------------------------------------------

export const PHONEMIZER_LABEL: Record<string, string> = {
  auto: 'Automatic',
  uccharan: 'uccharan',
  'uccharan:compatible': 'uccharan',
  'uccharan:improved': 'uccharan, improved',
  'espeak-ng': 'eSpeak NG',
}

/** The choices for the current voice: per voice (Kokoro) or per model, empty when the voice reads characters. */
export function phonemizerChoices(m: CatalogModel | null | undefined, voice: Voice | null | undefined) {
  return (voice?.phonemizers ?? m?.phonemizers ?? []) as PhonemizerChoice[]
}
