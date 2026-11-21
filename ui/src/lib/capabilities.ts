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
      available = Boolean(entry?.available ?? entry?.configured)
      detail = entry?.model ? `audio model ${entry.model}` : null
      if (!available)
        reason = 'No audio-understanding model is configured (settings audio_llm_base_url, audio_llm_model).'
    } else {
      detail = entry?.model ? `model ${entry.model}` : null
      if (typeof entry?.available === 'boolean') {
        // Newer servers say whether a session can use it now, and why not.
        available = entry.available
        if (entry.url) detail = `${detail ? `${detail} at ` : ''}${entry.url}`
        if (!available)
          reason =
            sentence(entry.reason) ?? 'No speech-to-speech model is configured (settings s2s_model or s2s_server_url).'
      } else if (entry?.url) {
        // Older servers report what they would use, not whether they can: a running server they were pointed
        // at, or a llama-server model they start themselves, which needs llama-server installed.
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
      note:
        id === 'speech-to-speech'
          ? isCloudS2s(entry)
            ? S2S_CLOUD_NOTE
            : entry?.license
              ? `Licence: ${entry.license}${entry.full_duplex ? ' · full duplex' : ''}`
              : !entry?.model || /^lfm/i.test(entry.model)
                ? S2S_LICENCE_NOTE
                : null
          : null,
    }
  })
}

/** A speech-to-speech backend at a cloud provider (`s2s_model = "<provider>:<model>"`). */
function isCloudS2s(entry: { cloud?: boolean; model?: string | null } | undefined): boolean {
  return Boolean(entry?.cloud) || /^[a-z0-9_-]+:.+/.test(entry?.model ?? '')
}

export const S2S_CLOUD_NOTE = "Paid cloud API: the conversation's audio leaves this server"

function sentence(text: string | null | undefined): string | null {
  if (!text) return null
  const t = text.trim()
  return `${t.charAt(0).toUpperCase()}${t.slice(1)}${/[.!?]$/.test(t) ? '' : '.'}`
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
  license: string
  /** The weights are gated on Hugging Face: accept the conditions there and give the server a token. */
  gated: boolean
  commercialUse: boolean | null
  /** Runs on an engine plugin (`plugin torch`), else null. */
  runtime: string | null
}

const DIARIZER_NOTES: Record<string, string> = {
  'nemotron-3-diarization':
    'Better with overlapping speech and similar voices; can merge speakers on clean phone-quality turn-taking',
  'pyannote-3.0-eres2net': 'Default: segmentation and clustering, small and fast',
  'pyannote-speaker-diarization-3.1':
    "pyannote's own PyTorch pipeline, for pyannote's exact results; needs the torch plugin with pyannote, and is gated: accept its conditions on Hugging Face and set BOLO_HF_TOKEN",
  'pyannote-speaker-diarization-community-1':
    "pyannote 4's community-1 pipeline (VBx clustering) in PyTorch; needs the torch plugin with pyannote, and is gated: accept its conditions on Hugging Face and set BOLO_HF_TOKEN",
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
      license: m.license,
      gated: m.gated,
      commercialUse: m.commercial_use ?? null,
      runtime: runtimeLabel(m),
    }))
  return list.sort((a, b) => Number(b.isDefault) - Number(a.isDefault) || rankDiarizer(a.id) - rankDiarizer(b.id))
}

function rankDiarizer(id: string): number {
  if (id.startsWith('pyannote-speaker-diarization')) return 2
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
 * Entries the catalog describes but no runtime can run yet. Servers say so with `runnable: false`; older ones
 * only in the description ("Not runnable yet: …").
 */
export function isNotImplemented(m: Pick<CatalogModel, 'status' | 'description' | 'runnable'>): boolean {
  if (typeof m.runnable === 'boolean') return !m.runnable
  return m.status === 'not_implemented' || m.status === 'not-implemented' || /not runnable yet/i.test(m.description)
}

/** Why an entry cannot run, in the server's words (newer servers), else null. */
export function notRunnableReason(m: Pick<CatalogModel, 'not_runnable_reason'>): string | null {
  return sentence(m.not_runnable_reason)
}

// --- Cloud models ---------------------------------------------------------------------------------------------------

export const CLOUD_NOTICE = 'Paid cloud API; audio leaves this server'

/** A model behind a paid cloud API (engine `remote`). */
export function isCloud(m: Pick<CatalogModel, 'cloud' | 'engine'> | null | undefined): boolean {
  return Boolean(m && (m.cloud ?? m.engine === 'remote'))
}

/** The notice to show with a cloud model: the server's wording when it sends one. */
export function cloudNotice(m: Pick<CatalogModel, 'notice'> | null | undefined): string {
  const n = m?.notice?.trim()
  return n ? `${n.charAt(0).toUpperCase()}${n.slice(1)}` : CLOUD_NOTICE
}

// --- Server behaviour ---------------------------------------------------------------------------------------------

/** Whether the server downloads a model on first use. Null when it does not say (older servers). */
export function autoPull(info: ServerInfo | undefined): boolean | null {
  return typeof info?.auto_pull === 'boolean' ? info.auto_pull : null
}

/** What to tell someone who picked a model that is not installed. */
export function notInstalledHint(info: ServerInfo | undefined, m: Pick<CatalogModel, 'cloud' | 'engine'>): string {
  if (isCloud(m)) return 'Its cloud provider is not configured yet. Add the provider on the Providers screen.'
  const auto = autoPull(info)
  if (auto === true) return 'The server downloads it on first use, so the first request takes longer.'
  if (auto === false)
    return 'This server does not download models on demand (auto_pull is off). Pull it from the Models screen.'
  return 'Pull it from the Models screen, or pick an installed model.'
}

/**
 * Whether the server has word lists: true or false when it announces them (`capabilities.wordlists`), null
 * when it does not say, in which case the caller probes the route.
 */
export function wordListsAnnounced(info: ServerInfo | undefined): boolean | null {
  const w = info?.capabilities?.wordlists
  if (w) return true
  // A server new enough to announce cloud providers announces word lists too: absent means it has none.
  return info?.capabilities && 'providers' in info.capabilities ? false : null
}

/** Whether the server can connect phone calls to voice agents (the Phone screen). */
export function telephonyOffered(info: ServerInfo | undefined): boolean {
  return Boolean(info?.capabilities?.telephony)
}

/** Whether the server describes its hardware and rates models for it (the fit line, the hardware card). */
export function recommendationsOffered(info: ServerInfo | undefined): boolean {
  return Boolean(info?.capabilities?.recommendations)
}

/** Whether the server has cloud providers (the Providers screen and cost estimates). */
/** Whether the server spots wake words (`/api/v1/wake`, the Wake word screen). */
/** Whether the server answers typed questions about a recording or a text (the Decisions screen). */
export function decisionsOffered(info: ServerInfo | undefined): boolean {
  return Boolean(info?.capabilities?.decision)
}

export function wakeOffered(info: ServerInfo | undefined): boolean {
  return Boolean(info?.capabilities?.wake)
}

/** Whether the server keeps API keys with scopes, rate limits and quotas (the Keys screen). */
export function keysOffered(info: ServerInfo | undefined): boolean {
  return Boolean(info?.capabilities?.api_keys)
}

export function providersOffered(info: ServerInfo | undefined): boolean {
  return Boolean(info?.capabilities?.providers)
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
  if (m.engine === 'remote') return 'cloud API'
  if (m.engine === 'whisper.cpp') return 'whisper.cpp'
  if (m.engine === 'duplex') return 'full-duplex server'
  if (m.engine === 'tts-server') {
    if (m.requires?.includes('audio.cpp') || m.id.endsWith('-audiocpp')) return 'audio.cpp'
    return 'llama.cpp'
  }
  return null
}

// --- Licences, gates and remote code ---------------------------------------------------------------------------

type Labelled = Pick<CatalogModel, 'gated' | 'commercial_use' | 'license_note' | 'remote_code'>

/**
 * The short warnings a model carries wherever it is offered: gated, not for commercial use, a licence term to act
 * on, and Python code from its own repository (with whether the operator opted in).
 */
export function licenceTags(m: Partial<Labelled>): string[] {
  const tags: string[] = []
  if (m.gated) tags.push('gated')
  if (m.commercial_use === false) tags.push('non-commercial')
  if (m.license_note) tags.push('licence note')
  if (m.remote_code) tags.push(m.remote_code.allowed ? 'remote code, allowed' : 'remote code, not allowed')
  return tags
}

/** The per-model setting that lets a model's remote code run, and the command that sets it. */
export const REMOTE_CODE_SETTING = 'allow_remote_code'
export function remoteCodeCommand(id: string): string {
  return `bolo models config ${id} --set ${REMOTE_CODE_SETTING}=true`
}

/** A speech-to-speech model that hears and speaks at once (Moshi, Hibiki, PersonaPlex, ...). */
export function isFullDuplex(m: Pick<CatalogModel, 'features'> | null | undefined): boolean {
  return Boolean(m?.features.includes('full-duplex'))
}

/** End-of-turn models that read the transcript rather than hear the audio (catalog option `input: text`). */
const TEXT_TURN_IDS = ['namo-turn-en', 'namo-turn-multilingual', 'ten-turn-detection', 'fake-turn-text']
export function readsText(m: Pick<CatalogModel, 'id' | 'task' | 'features'>): boolean {
  if (m.task !== 'turn') return false
  // Newer servers mark them; older ones are known by id.
  return m.features.includes('reads-text') || TEXT_TURN_IDS.includes(m.id)
}

/**
 * Why the server cannot run a model at all yet: its engine is not installed (whisper.cpp without pywhispercpp,
 * the full-duplex bridge without PyAV, ...), from `GET /api/v1/info` engines. Null when it can, or when unknown.
 */
export function engineMissing(
  info: ServerInfo | undefined,
  m: Pick<CatalogModel, 'engine' | 'cloud'>,
): { engine: string; reason: string; install: string | null } | null {
  const e = info?.engines?.find((x) => x.id === m.engine)
  if (!e || e.available) return null
  const reason = e.reason ?? `${e.id} is not installed`
  const quoted = /pip install\s+(['"])([^'"]+)\1/.exec(reason)
  return { engine: e.id, reason, install: quoted ? `pip install '${quoted[2]}'` : null }
}

const ORPHEUS = /^(veena|svara|orpheus|snortts)/i

/** Voice family group for the text-to-speech picker. */
export function voiceFamily(m: Pick<CatalogModel, 'engine' | 'id' | 'family'>): string {
  if (m.engine === 'tts-server' && ORPHEUS.test(m.id)) return 'Orpheus family (llama.cpp)'
  if (m.engine === 'tts-server') return m.id.endsWith('-audiocpp') ? 'audio.cpp voices' : 'llama.cpp voices'
  if (m.engine.startsWith('plugin:')) return 'Engine plugin voices'
  if (m.engine === 'remote') return 'Cloud voices (paid APIs)'
  return 'Built-in voices'
}

export function canClone(m: Pick<CatalogModel, 'features'> | null | undefined): boolean {
  return Boolean(m?.features.includes('voice-cloning'))
}

/** The model follows a request's `instructions` (style, emotion or the voice itself, described in words). */
export function followsInstructions(m: Pick<CatalogModel, 'features'> | null | undefined): boolean {
  return Boolean(m?.features.includes('instructions'))
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
