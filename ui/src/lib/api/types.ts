/** Types for the bolo native API (`/api/v1`). Field names mirror the wire format. */

export type Task = 'stt' | 'tts' | 'vad' | 'diarization' | 'translation' | 'separation' | 'turn'
export type Tier = 'realtime' | 'balanced' | 'accuracy'
export type Validated = 'cpu' | 'untested'

export interface EngineInfo {
  id: string
  version: string | null
  available: boolean
  reason: string | null
}

export interface ServerInfo {
  name: string
  version: string
  engines: EngineInfo[]
  device: string
  cpu_count: number
  platform: string
  auth_required: boolean
  data_dir: string
  /** Whether a model that is not installed is downloaded on first use. Absent on older servers. */
  auto_pull?: boolean
  /** Cloud providers the server knows (never a key). */
  providers?: { name: string; kind: string; configured: boolean }[]
  /** Newer servers only. */
  defaults?: Partial<Record<'stt' | 'realtime_stt' | 'tts' | 'realtime_tts' | 'vad', string>>
  components?: Component[]
  capabilities?: Capabilities
}

/** `info.capabilities`: what the server can do beyond plain STT/TTS. Every part is optional (older servers). */
export interface Capabilities {
  diarization?: {
    available?: boolean
    installed?: boolean
    default_model?: string | null
    speaker_conditioned_models?: string[]
    [k: string]: unknown
  }
  separation?: {
    available?: boolean
    installed?: boolean
    default_model?: string | null
    languages?: string[]
    min_sample_rate?: number
    context_s?: number
    min_overlap_s?: number
    [k: string]: unknown
  }
  realtime?: {
    endpoint?: string
    transcription_sessions?: boolean
    conversation_sessions?: boolean
    pipeline?: string
    pipelines?: Record<string, RealtimePipelineInfo | undefined>
    [k: string]: unknown
  }
  llm?: { configured?: boolean; base_url?: string | null; model?: string | null }
  /** Pronunciation word lists: announced by newer servers (older ones are probed with a request). */
  wordlists?: { available?: boolean; endpoint?: string; install?: string | null; [k: string]: unknown }
  /** Cloud providers behind `engine: remote` models. */
  providers?: {
    endpoint?: string
    kinds?: string[]
    editable?: boolean
    edit_blocked_reason?: string | null
    configured?: string[]
    cost_estimates?: boolean
    [k: string]: unknown
  }
  [k: string]: unknown
}

export interface RealtimePipelineInfo {
  /** A session can use this pipeline now. Newer servers report it for every pipeline. */
  available?: boolean
  /** Its settings are in place (it may still lack something, see `reason`). */
  configured?: boolean
  /** Why it is not available, in the server's words. */
  reason?: string | null
  /** A cloud speech-to-speech backend: the conversation's audio leaves the server. */
  cloud?: boolean
  provider?: string
  base_url?: string | null
  model?: string | null
  url?: string | null
  backend?: string
  managed?: boolean
  process?: unknown
}

export interface Voice {
  id: string
  name: string
  language: string
  gender?: string | null
  /** Kokoro-style multi-language models carry the phonemiser choices per voice. */
  phonemizers?: PhonemizerChoice[]
}

/** One way of turning text into phonemes for a voice (`uccharan`, `uccharan:improved`, `espeak-ng`). */
export interface PhonemizerChoice {
  id: string
  available: boolean
  default: boolean
  note?: string | null
}

/** GPU advice for a model; null means it runs fine on a CPU. */
export interface Hardware {
  gpu: 'optional' | 'recommended' | 'required' | string
  vram_gb?: number | null
  note?: string | null
}

export interface PullState {
  job_id: string
  progress: number
}

export interface CatalogModel {
  id: string
  aliases: string[]
  task: Task
  engine: string
  family: string
  languages: string[]
  tier: Tier
  streaming: boolean
  sample_rate: number
  size_bytes: number
  params: number | null
  license: string
  gated: boolean
  voices: Voice[]
  features: string[]
  installed: boolean
  loaded: boolean
  pulling: PullState | null
  description: string
  source_url: string
  validated: Validated
  /** Optional extras newer servers add. */
  license_url?: string | null
  commercial_use?: boolean | null
  measured?: Record<string, unknown> | null
  custom?: boolean
  /** Optional components (by id) this model needs, e.g. `espeak-ng`. */
  requires?: string[]
  /** False when a required component is not installed; absent on older servers (treated as true). */
  available?: boolean
  /** Phonemiser choices for voices that read phonemes (`[]` otherwise). */
  phonemizers?: PhonemizerChoice[]
  /** Settings the user pinned for this model, e.g. `{"phonemizer": "espeak-ng"}`. */
  settings?: Record<string, unknown>
  /** `pending`: described but its files are not published yet. */
  status?: 'pending' | string | null
  hardware?: Hardware | null
  /** False when no runtime the server has can run the model yet; `not_runnable_reason` says why. */
  runnable?: boolean
  not_runnable_reason?: string | null
  /** A paid cloud API (engine `remote`): nothing is downloaded, and audio or text leaves the server. */
  cloud?: boolean
  notice?: string | null
  provider?: ModelProvider | null
  /** Models that answer instead when this one cannot. */
  fallback?: string[]
  /** A `<provider>:<model>` id that was used by name, not a curated entry. */
  adhoc?: boolean
  /** Live demo only: where this entry can run. */
  runtime?: 'browser' | 'server'
}

/** The cloud provider behind a catalog entry. */
export interface ModelProvider {
  name: string
  kind: string
  model: string
  configured: boolean
  price?: Price | null
}

/** One row of a provider's price table: what the operator pays, for cost estimates. */
export interface Price {
  per_minute?: number
  per_hour?: number
  per_1k_chars?: number
  per_1m_chars?: number
  per_request?: number
  currency?: string
}

export interface ProviderStats {
  requests: number
  errors: number
  latency_ms_mean: number | null
  latency_ms_total?: number
  bytes_sent: number
  bytes_received: number
  audio_seconds: number
  chars: number
  estimated_cost: Record<string, number>
  last_error: string | null
}

/** A cloud provider as the server shows it. The API key is never part of it. */
export interface Provider {
  name: string
  kind: string
  base_url: string
  configured: boolean
  api_key_set: boolean
  options: Record<string, unknown>
  prices: Record<string, Price>
  /** `settings`: from the configuration (read-only here); `stored`: added through the API. */
  source: 'settings' | 'stored' | string
  editable: boolean
  terms_url: string | null
  stats?: Record<string, ProviderStats>
}

export interface ProviderKind {
  kind: string
  default_base_url: string
  terms_url: string
}

export interface ProviderList {
  providers: Provider[]
  kinds: ProviderKind[]
  editable: boolean
  edit_blocked_reason: string | null
  notice: string
}

/** Body of `PUT /api/v1/providers/{name}`. `api_key` is write-only: leave it out to keep the stored key. */
export interface ProviderUpdate {
  kind?: string
  base_url?: string
  api_key?: string
  clear_api_key?: boolean
  options?: Record<string, unknown>
  prices?: Record<string, Price>
}

export interface ProviderTest {
  name: string
  ok: boolean
  message: string
  code: string | null
  /** `credentials`: the key was accepted; `reachability`: only that the address answers. */
  checked: 'credentials' | 'reachability' | string | null
  latency_ms: number
}

/** An estimated cost from the operator's price table (never a bill). */
export interface Cost {
  amount: number
  currency: string
  estimated?: boolean
}

export interface Catalog {
  models: CatalogModel[]
}

export type JobState = 'queued' | 'running' | 'done' | 'error'

export interface Job<R = unknown> {
  id: string
  kind: string
  model: string
  state: JobState
  progress: number
  bytes_done?: number | null
  bytes_total?: number | null
  error: string | null
  result?: R | null
}

export interface Percentiles {
  p50: number
  p95: number
  p99?: number
}

export interface ModelLatency {
  first_partial: Percentiles | null
  final: Percentiles | null
  ttfa: Percentiles | null
}

export interface ModelStats {
  id: string
  task: Task
  loaded: boolean
  memory_bytes: number
  idle_s: number
  ttl_s: number | null
  active: number
  requests: number
  latency_ms: ModelLatency
  rtf: { p50: number; p95: number } | null
}

export interface Stats {
  uptime_s: number
  active_streams: number
  queue_depth: number
  system: { cpu_percent: number; rss_bytes: number; threads: number }
  totals: { requests: number; errors: number; audio_seconds_in: number; audio_seconds_out: number }
  models: ModelStats[]
}

export interface Phase {
  name: string
  start_ms: number
  dur_ms: number
}

export interface RequestTrace {
  /** Extra fields some traces carry (diarization, separation, voice agents, …). */
  diarization_model?: string
  speakers?: number
  separation?: string
  separation_model?: string
  separation_windows?: number
  separation_ms?: number
  pipeline?: string
  /** A request served by a cloud model: what it took at the provider, and the estimated cost. */
  provider?: {
    name: string
    kind?: string
    model: string
    requests: number
    latency_ms: number
    first_byte_ms?: number | null
    bytes_sent: number
    bytes_received: number
  }
  cost?: Cost
  /** Another model answered instead of the one asked for. */
  fallback?: { requested: string; answered: string; attempts: { model: string; code: string; message: string }[] }
  first_token_ms?: number
  overlapped_speech_s?: number
  /** TTS: which normaliser ran, the text the voice received (first 500 characters), and word-list matches. */
  normalizer?: string
  normalized_text?: string
  pronunciations_applied?: AppliedPronunciation[]
  speakers_before_consolidation?: number
  conditioning?: unknown
  id: string
  kind: 'stt' | 'tts' | string
  transport: string
  model: string
  started_at: string
  status: 'ok' | 'error' | string
  error: string | null
  audio_seconds: number | null
  chars: number | null
  language: string | null
  phases: Phase[]
  first_partial_ms: number | null
  final_ms: number | null
  ttfa_ms: number | null
  total_ms: number
  rtf: number | null
  batch_size: number | null
  text_preview: string | null
}

export interface Sample {
  id: string
  kind: 'audio' | 'text'
  language: string
  seconds?: number | null
  url?: string | null
  text: string
}

export interface BenchRequest {
  kind: 'stt' | 'tts'
  model: string
  concurrency: number[]
  duration_s: number
  sample: string
}

export interface BenchPoint {
  concurrency: number
  ok: number
  errors: number
  first_partial_ms?: Percentiles | null
  final_ms?: Percentiles | null
  ttfa_ms?: Percentiles | null
  rtf: { p50: number; p95: number } | null
  throughput_x_realtime: number
}

export interface BenchResult {
  points: BenchPoint[]
}

/** An optional piece of software that some models need and bolo does not ship. */
export interface Component {
  id: string
  name: string
  installed: boolean
  source: 'system-executable' | 'python-package' | 'configured' | string | null
  version: string | null
  license: string
  license_url: string | null
  required_by_count: number
  installable: boolean
  install_blocked_reason: string | null
  manual_command: string
  description: string
  /** Newer servers: `phonemizer`, `tts-server` or `engine-plugin`. */
  kind?: 'phonemizer' | 'tts-server' | 'engine-plugin' | string
  /** uccharan: the languages it covers. */
  languages?: string[]
  /** tts-server components: binary in use, configured server URLs, pinned upstream release. */
  path?: string | null
  servers?: string[]
  release?: { version: string; url: string; sha256: string } | null
}

/** Server-sent events on `/api/v1/events`. */
export type ServerEvent =
  | ({ type: 'stats' } & Stats)
  | ({ type: 'request' } & RequestTrace)
  | { type: 'request_progress'; id: string; phase?: string; [k: string]: unknown }
  | ({ type: 'job' } & Job)
  | { type: 'model'; id: string; event: 'loaded' | 'unloaded' | 'installed' | 'removed' | 'settings' }
  | { type: 'component'; id: string; event: 'installed' | 'removed' | string }
  | { type: 'log'; level: string; message: string }
  | { type: 'wordlist'; name: string; event: 'updated' | 'deleted' | string }
  | { type: 'provider'; name: string; event: 'saved' | 'removed' | string }

/** Messages on the STT websocket (server → client). */
export interface SttWord {
  w: string
  t0: number
  t1: number
}
export type SttServerMessage =
  | {
      type: 'ready'
      model: string
      sample_rate: number
      request_id: string
      mode?: 'native' | 'pseudo'
      vad?: string
      encoding?: string
    }
  | { type: 'vad'; event: 'speech_start' | 'speech_end'; t: number }
  | { type: 'partial'; text: string; t0: number; t1: number; latency_ms: number; stable_chars?: number }
  | {
      type: 'final'
      text: string
      t0: number
      t1: number
      latency_ms: number
      words?: SttWord[] | null
      language?: string | null
    }
  | { type: 'metrics'; audio_seconds: number; rtf: number; queue_ms?: number }
  | { type: 'error'; code: string; message: string }
  | { type: 'done'; request_id?: string }
  | { type: 'pong' }

export type SttEncoding = 'pcm_s16le' | 'pcm_f32le' | 'mulaw' | 'alaw'

export interface SttStreamParams {
  model: string
  language?: string
  sample_rate: number
  encoding: SttEncoding
  interim?: boolean
  vad?: boolean
}

/** Messages on the TTS websocket (server → client JSON frames). */
export type TtsServerMessage =
  | { type: 'ready'; model?: string; voice?: string; sample_rate?: number; request_id?: string }
  | { type: 'chunk'; seq: number; t0: number; dur: number; ttfa_ms?: number | null; text: string }
  | { type: 'metrics'; rtf: number }
  | { type: 'error'; code?: string; message: string }
  | { type: 'done' }

export interface TtsStreamParams {
  model: string
  voice?: string
  format?: 'pcm_s16le'
  sample_rate?: number
}

/** Pronunciation word lists (`/api/v1/wordlists`). */
export interface WordListSummary {
  name: string
  entries: number
  languages: string[]
  bytes: number
  updated: string
}

export type WordMatch = 'word' | 'case' | 'exact' | 'regex'

export interface WordEntry {
  word: string
  lang: string
  ipa: string | null
  say: string | null
  match: WordMatch | string
  note: string | null
  line?: number
}

export interface WordList extends Omit<WordListSummary, 'entries'> {
  entries: WordEntry[]
}

export interface WordListPutResult extends WordListSummary {
  warnings: { line: number; message: string }[]
}

/** How a voice takes text (`GET /api/v1/text-input/{model}`). */
export interface TextInput {
  reads: 'phonemes' | 'text' | 'characters' | string
  normalizer: string
  markup: string
  latin: string
  respell: string
  digits: string
  wordlists: string[]
  pronunciation: 'phonemes' | 'markup' | 'respelling' | string
}

/** One word-list match in a preview or a trace. */
export interface AppliedPronunciation {
  text: string
  word?: string
  spoken?: string
  ipa?: string
}

export interface WordListPreview {
  model: string
  voice: string | null
  language: string
  text_input: TextInput
  normalizer: string
  text: string
  applied: AppliedPronunciation[]
  phonemes?: string | null
}
