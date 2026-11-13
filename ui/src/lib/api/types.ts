/** Types for the bolo native API (`/api/v1`). Field names mirror the wire format. */

export type Task = 'stt' | 'tts' | 'vad' | 'diarization' | 'translation'
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
}

export interface Voice {
  id: string
  name: string
  language: string
  gender?: string | null
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
  /** Live demo only: where this entry can run. */
  runtime?: 'browser' | 'server'
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
  license_url: string
  required_by_count: number
  installable: boolean
  install_blocked_reason: string | null
  manual_command: string
  description: string
}

/** Server-sent events on `/api/v1/events`. */
export type ServerEvent =
  | ({ type: 'stats' } & Stats)
  | ({ type: 'request' } & RequestTrace)
  | { type: 'request_progress'; id: string; phase?: string; [k: string]: unknown }
  | ({ type: 'job' } & Job)
  | { type: 'model'; id: string; event: 'loaded' | 'unloaded' | 'installed' | 'removed' }
  | { type: 'component'; id: string; event: 'installed' | 'removed' | string }
  | { type: 'log'; level: string; message: string }

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
