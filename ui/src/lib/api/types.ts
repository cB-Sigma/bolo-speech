/** Types for the bolo native API (`/api/v1`). Field names mirror the wire format. */

export type Task =
  | 'stt'
  | 'tts'
  | 'vad'
  | 'diarization'
  | 'translation'
  | 'separation'
  | 'turn'
  | 'denoise'
  | 'lid'
  | 'speaker-embedding'
  | 'speech-to-speech'
  | 'audio-llm'
  | 'audio-tagging'
  | 'emotion'
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
  /** API keys with scopes, rate limits and quotas (`/api/v1/keys`). */
  api_keys?: { endpoint?: string; auth?: boolean; root_key?: boolean; scopes?: string[]; quota_periods?: string[] }
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
    transports?: string[]
    /** `POST /v1/realtime/calls`; `available` is false without the server's `webrtc` extra. */
    webrtc?: { endpoint?: string; available?: boolean; install?: string | null; ice_servers?: string[] }
    /** The end-of-turn model `semantic_vad` asks by default; `input` says whether it hears audio or reads text. */
    semantic_vad?: { model?: string | null; available?: boolean; installed?: boolean; input?: 'audio' | 'text' | null }
    [k: string]: unknown
  }
  llm?: { configured?: boolean; base_url?: string | null; model?: string | null }
  /** Spoken-language identification and routing by language (`model=auto`), newer servers. */
  lid?: {
    available?: boolean
    installed?: boolean
    default_model?: string | null
    endpoint?: string
    routing?: {
      enabled?: boolean
      model_names?: string[]
      routes?: Record<string, string>
      min_probability?: number
      header?: string
    }
    [k: string]: unknown
  }
  /** Background noise removal (`POST /v1/audio/denoise` and the denoise option of transcriptions). */
  denoise?: { available?: boolean; installed?: boolean; default_model?: string | null; [k: string]: unknown }
  /** Sound events (`POST /v1/audio/tags`), newer servers. */
  audio_tagging?: { endpoint?: string; default_model?: string | null; windows?: boolean; [k: string]: unknown }
  /** Speech emotion (`POST /v1/audio/emotion`); `transcription_tags`: recognisers that tag emotions as they go. */
  emotion?: {
    endpoint?: string
    default_model?: string | null
    windows?: boolean
    transcription_tags?: string[]
    [k: string]: unknown
  }
  /** Enrolled speakers: verification and identification (`/v1/audio/speakers`), newer servers. */
  speakers?: {
    endpoint?: string
    embedding_model?: string
    threshold?: number
    enrolled?: number
    [k: string]: unknown
  }
  /** Pronunciation word lists: announced by newer servers (older ones are probed with a request). */
  wordlists?: { available?: boolean; endpoint?: string; install?: string | null; [k: string]: unknown }
  /** Romanised text for text-to-speech: what the server's transliterator can do. */
  romanised?: {
    available?: boolean
    reason?: string | null
    modes?: string[]
    /** Languages the installed uccharan lists a pack for, and those whose pack is on the server. */
    languages?: string[]
    installed?: string[]
    default?: string
    [k: string]: unknown
  }
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
  /** Saved voices for the models that clone voices (`/api/v1/voices`). */
  voices?: {
    endpoint?: string
    consent?: string[]
    cloning_models?: string[]
    similarity_model?: string
    [k: string]: unknown
  }
  /** Phone calls: telephony providers connected to voice agents. */
  telephony?: {
    endpoint?: string
    providers?: string[]
    configured?: string[]
    exposed?: string[]
    validate_signatures?: boolean
    agents?: number
    sip?: { enabled?: boolean; listening?: boolean; port?: number }
    [k: string]: unknown
  }
  /** OpenTelemetry export of the request traces (newer servers). Read-only: set by the `otel` setting and `OTEL_*`. */
  opentelemetry?: {
    /** Spans are being exported. */
    enabled?: boolean
    /** Switched on (setting or an OTLP endpoint in the environment), whether or not it could start. */
    switched_on?: boolean
    installed?: boolean
    endpoint?: string | null
    protocol?: string | null
    service_name?: string | null
    exported?: number
    install?: string | null
    [k: string]: unknown
  }
  /** Hardware detection and model recommendations. They advise; `restricts` is always false. */
  recommendations?: {
    hardware?: string
    endpoint?: string
    setup?: string
    benchmark?: string
    tasks?: string[]
    uses?: string[]
    fit_levels?: string[]
    warnings?: boolean
    restricts?: boolean
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
  /** Speech to speech: a full-duplex model (it hears and speaks at once; talking over it does not stop it). */
  full_duplex?: boolean
  features?: string[]
  license?: string
  max_sessions?: number
  active_sessions?: number
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

/** How well a model suits the server's machine. `unlikely` models are never hidden or refused: this is advice. */
export type FitLevel = 'comfortable' | 'usable' | 'slow' | 'unlikely' | 'cloud' | 'unavailable'

export interface ModelFit {
  level: FitLevel | string
  label?: string
  /** Expected real-time factor on this machine; null when nothing was measured or estimated. */
  expected_rtf: number | null
  rtf_range?: [number, number] | null
  /** `measured` (run by the maintainers, scaled to this machine), `estimated` (from a sibling model) or `unknown`. */
  rtf_basis?: 'measured' | 'estimated' | 'unknown' | string
  /** What scaled the catalog's figure: the stored micro-benchmark or a rougher estimate from the CPU. */
  scaled_by?: 'benchmark' | 'heuristic' | string | null
  device?: 'cpu' | 'gpu' | 'cloud' | string | null
  memory?: {
    needed_bytes: number | null
    available_bytes: number | null
    kind: 'ram' | 'vram' | string | null
    basis: string | null
  } | null
  engine_available?: boolean
  reason: string
  notes?: string[]
  /** Among the few recommended for its task on this machine, whatever the language. */
  recommended?: boolean
  /** Languages (base codes; `*` is "any") this model is among the recommended ones for. */
  recommended_for?: string[]
  default_for?: string[]
}

/** A notice attached to a pull or load. It never means the action was refused. */
export interface FitWarning {
  code: 'hardware_fit' | 'disk_space' | string
  model: string
  level: string
  message: string
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
  /** How well the model suits the server's machine (newer servers; absent in the live demo). */
  fit?: ModelFit | null
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
  /**
   * Python code from the model's own repository that loading it runs (Hugging Face remote code), pinned by commit
   * and sha256; it runs only once the operator opts in (`allow_remote_code`). Null for every other model.
   */
  remote_code?: RemoteCode | null
  /** A licence term to act on before use (for example a registration for commercial use). */
  license_note?: string | null
}

export interface RemoteCode {
  revision: string | null
  files: string[]
  /** What the review of the code found. */
  reviewed: string | null
  /** The operator opted in for this model (per-model setting `allow_remote_code`). */
  allowed: boolean
  /** The server's warning: what runs, and how to opt in. */
  notice: string | null
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

/** `GET /api/v1/hardware`: what the server detected. `null` means "could not be read", never "none". */
export interface HardwareInfo {
  system: string | null
  cpu: {
    model: string | null
    vendor: string | null
    architecture: string | null
    physical_cores: number | null
    logical_cores: number | null
    usable_cores: number | null
    features: string[] | null
  }
  memory: {
    total_bytes: number | null
    available_bytes: number | null
    limit_bytes: number | null
    usable_bytes: number | null
  }
  disk: { path: string | null; total_bytes: number | null; free_bytes: number | null }
  gpus: {
    vendor: string
    name: string | null
    vram_total_bytes: number | null
    vram_free_bytes: number | null
    driver: string | null
    unified_memory: boolean
    backend: string | null
  }[]
  gpus_known: boolean
  runtimes: Record<string, Record<string, unknown>>
  container: {
    detected: boolean | null
    kind: string | null
    virtual_machine: boolean | null
    hypervisor: string | null
    cpu_quota: number | null
    memory_limit_bytes: number | null
  }
  device: string
  threads: number | null
  notes: string[]
  detected_at: string
  summary: string
  speed: SpeedInfo
}

/** This machine against the one the catalog was measured on. */
export interface SpeedInfo {
  factor: number
  source: 'benchmark' | 'heuristic' | string
  spread?: [number, number]
  note: string
  benchmark?: Record<string, unknown> | null
  reference?: string
}

/** One row of `GET /api/v1/recommendations`. */
export interface Recommendation {
  rank: number
  id: string
  task: Task
  family: string
  engine: string
  tier: Tier
  languages: string[]
  size_bytes: number
  installed: boolean
  license: string
  commercial_use: boolean
  cloud: boolean
  validated: Validated
  quality: { metric: string; label: string; value: number; note: string } | null
  recommended: boolean
  default: boolean
  fit: ModelFit
}

export type SetupUse = 'live' | 'files' | 'voiceover'

/** `GET /api/v1/recommendations/setup`: one model per task for a use and language. */
export interface SetupPlan {
  use: SetupUse | string
  label: string
  language: string | null
  commercial: boolean
  local_only: boolean
  hardware: string
  speed: SpeedInfo
  picks: {
    task: Task | string
    optional: boolean
    realtime: boolean
    model: Recommendation | null
    alternatives: Recommendation[]
    note: string | null
  }[]
  /** Ids of the chosen models that are not installed yet. */
  pull: string[]
  download_bytes: number
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
  /** Notices about the job (a model that is unlikely to fit this machine). They never stop it. */
  warnings?: FitWarning[]
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
  /* Placement and keep-alive (servers from bolo 0.5; absent before, and in the browser demo). */
  /** Of `memory_bytes`, the part in accelerator memory. */
  size_vram_bytes?: number
  ram_bytes?: number
  vram_measured?: boolean
  /** Where `size_vram_bytes` comes from: "process", "dxgi", "device", "plugin", "unified", or "plan" (it could not
   * be measured, as on Windows under WDDM: the figure is the plan and the model is still on the GPU). */
  vram_source?: string | null
  /** Ollama's PROCESSOR column: "100% GPU", "100% CPU", "35%/65% CPU/GPU". */
  processor?: string
  /** Where it runs: "cpu", "cuda:0", "coreml", "remote", "cloud". */
  device?: string
  placement?: ModelPlacement
  /** Seconds it stays loaded once idle; -1: until unloaded. */
  keep_alive_s?: number | null
  /** Seconds until it is unloaded for being idle (counted from now while in use); null: never. */
  expires_in_s?: number | null
  expires_at?: string | null
  stopping?: boolean
  /** `memory_bytes` is an estimate, not a measurement (the browser demo: a page cannot measure one model). */
  memory_estimated?: boolean
  /** On a GPU that was lost: why (it is being moved to the CPU, or its requests fail). */
  degraded?: string | null
}

/** Is the GPU still there (`system.gpu_health`; absent from older servers). */
export interface GpuHealth {
  state: 'unchecked' | 'ok' | 'lost' | 'none' | 'unknown'
  lost: boolean
  reason: string | null
  /** "GPU LOST (...): reboot to recover; bolo is using the CPU" while lost, else null. */
  notice: string | null
  moved_to_cpu?: string[]
  degraded?: string[]
}

export interface ModelPlacement {
  device: string
  requested: string
  split: 'whole' | 'layers' | 'graphs' | 'device_map' | 'onnx_split'
  gpu_bytes: number
  cpu_bytes: number
  processor: string
  notice: string | null
  gpu_layers?: number | null
  total_layers?: number | null
  graphs?: Record<string, string>
  /** CUDA GPUs the weights are on: several for a llama.cpp model spread with --tensor-split. */
  gpus?: number[]
}

export interface GpuMemory {
  index: number
  vendor: string
  name: string | null
  backend: string
  total_bytes: number | null
  free_bytes: number | null
  used_bytes: number | null
  /** The GPU shares the machine's memory (Apple Silicon). */
  unified: boolean
  /** Percent busy over the last sample, when the driver says. */
  utilization: number | null
  /** Accelerator memory held by bolo and the servers it started; null when it cannot be told. */
  process_bytes: number | null
  source: string
  /** Windows: "WDDM", "TCC" or "MCDM" (under WDDM `process_bytes` is null: the driver does not report it). */
  driver_model?: string | null
  /** Windows: what bolo's own process uses on this GPU, as DXGI says. */
  own_bytes?: number | null
  /** NVIDIA: the `gpus` setting lets bolo use it (absent: older servers, Apple). */
  allowed?: boolean
  /** What bolo's loaded models hold on this GPU. */
  models_vram_bytes?: number
}

export interface Stats {
  uptime_s: number
  active_streams: number
  queue_depth: number
  system: {
    cpu_percent: number
    rss_bytes: number
    threads: number
    ram_total_bytes?: number
    ram_available_bytes?: number
    gpus?: GpuMemory[]
    accelerator_source?: string | null
    models_vram_bytes?: number
    models_ram_bytes?: number
    gpu_health?: GpuHealth
  }
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
  /** TTS: what happened to romanised text (absent when the option did not come into play). */
  romanised?: RomanisedRecord
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
  /** language-pack components (newer servers): part of uccharan itself, so never needed. */
  built_in?: boolean
  /** language-pack components: how many models installed here need the pack. */
  required_by_installed_count?: number
  /** built-in language packs: how many catalog models read with that data. */
  used_by_count?: number | null
  /** language-pack components: a downloaded copy the server can delete. */
  removable?: boolean
  size_bytes?: number | null
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
  | { type: 'call'; event: 'started' | 'ended' | 'summary' | 'dialing' | string; call: CallSummary }
  | { type: 'telephony'; event: string; name?: string; number?: string }

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
      /** What denoises the stream (newer servers): null for none. */
      denoise?: LiveDenoise | null
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
  | { type: 'metrics'; audio_seconds: number; rtf: number; queue_ms?: number; denoise?: LiveDenoiseMetrics }
  | { type: 'error'; code: string; message: string }
  | { type: 'done'; request_id?: string }
  | { type: 'pong' }

/** `ready.denoise` of the live speech-to-text stream. */
export interface LiveDenoise {
  model: string
  strength: number
  status: 'applied' | 'skipped' | string
  sample_rate?: number
  stateful?: boolean
  detail?: string | null
}

/** `metrics.denoise`: time spent denoising each chunk and the audio held back, for the stream so far. */
export interface LiveDenoiseMetrics {
  chunks?: number
  mean_ms?: number
  p95_ms?: number
  max_ms?: number
  held_ms?: number
  stateful?: boolean
}

/** `POST /v1/audio/language`. */
export interface LanguageId {
  model: string
  language: string
  name?: string
  probability: number
  languages: { language: string; name?: string; probability: number }[]
  duration?: number
  heard?: number
  recogniser?: {
    model: string | null
    reason: string
    language: string | null
    recommended?: string | null
    detail?: string | null
  } | null
}

/** One class and its probability (`POST /v1/audio/tags`, `POST /v1/audio/emotion`). */
export interface LabelScore {
  label: string
  score: number
}

/** `POST /v1/audio/tags`: sound events heard in a recording, and per window when `window` was sent. */
export interface AudioTags {
  object: 'audio.tags'
  model: string
  duration: number
  tags: LabelScore[]
  segments?: { start: number; end: number; tags: LabelScore[] }[]
}

/** `POST /v1/audio/emotion`: how the speaker sounds. */
export interface AudioEmotion {
  object: 'audio.emotion'
  model: string
  duration: number
  emotion: string | null
  score: number
  emotions: LabelScore[]
  segments?: { start: number; end: number; emotion: string | null; score: number; emotions: LabelScore[] }[]
}

/** An enrolled speaker (`/v1/audio/speakers`). */
export interface EnrolledSpeaker {
  id: string
  object: 'speaker'
  name: string
  created_at?: string
  updated_at?: string
  metadata: Record<string, unknown>
  clips: { id: string; seconds: number; added_at?: string }[]
  seconds: number
  embedding_models: string[]
}

export interface SpeakerList {
  object: 'list'
  data: EnrolledSpeaker[]
  embedding_model: string
  threshold: number
  calibrated: boolean
  thresholds: Record<string, number>
}

export interface SpeakerVerification {
  object: 'speaker.verification'
  speaker: { id: string; name: string }
  similarity: number
  threshold: number
  match: boolean
  embedding_model: string
  calibrated: boolean
}

export interface SpeakerIdentification {
  object: 'speaker.identification'
  speaker: { id: string; name: string } | null
  similarity: number | null
  threshold: number
  candidates: { id: string; name: string; similarity: number; match: boolean }[]
  embedding_model: string
  calibrated: boolean
}

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
  /** Romanised text: the voice's effective mode (`off`, `auto`, `on`); absent on older servers. */
  romanised?: string
}

/** What happened to romanised text in one request (`romanised` in a TTS trace and in the preview). */
export interface RomanisedRecord {
  mode: string
  /** `request`, or `voice` for the model's pin or the server's default. */
  source: string
  language: string | null
  applied: boolean
  latin_share?: number
  pack?: string
  /** The converted text (first 500 characters). */
  text?: string
  /** Words written in the script, when the transliterator reports them. */
  words?: number
  /** Why nothing was converted. */
  notice?: string
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
  /** What happened to romanised text for this voice (null when the option did not come into play). */
  romanised?: RomanisedRecord | null
}

// --- phone calls (docs/telephony.md) ---------------------------------------------------------------------------

/** Where to paste one of bolo's addresses in a telephony provider's dashboard. */
export interface TelephonyWebhook {
  name: string
  where: string
  url: string
  method: string
}

export interface TelephonyProvider {
  name: string
  label: string
  /** Some credentials are in the server's configuration. */
  configured: boolean
  /** The provider's webhook and stream endpoints answer. */
  exposed: boolean
  /** Why they do not (missing settings), when `exposed` is false. */
  problems: string[]
  /** `signature`: requests are checked against the provider's signature; `basic`: against credentials it sends. */
  protection: string
  can_validate: boolean
  can_place_calls: boolean
  machine_detection: boolean
  ends_by: string
  /** Settings by name; secrets read `***` when set, never their value. */
  settings: Record<string, string | null>
  docs_url: string
  webhooks: TelephonyWebhook[]
  stream_url: string
  notes: string[]
}

export interface AgentTool {
  name: string
  description?: string
  parameters?: Record<string, unknown>
  url?: string
  timeout_s?: number
}

/** A voice agent that answers and places calls. Unset fields fall back to the server's settings. */
export interface PhoneAgent {
  name: string
  instructions: string | null
  greeting: string | null
  voice: string | null
  language: string | null
  stt_model: string | null
  tts_model: string | null
  llm_model: string | null
  pipeline: string | null
  tools: AgentTool[]
  max_duration_s: number
  max_duration_message: string | null
  silence_timeout_s: number
  silence_prompt: string
  silence_prompts: number
  goodbye: string | null
  transfer_number: string | null
  transfer_message: string | null
  recording_consent: string | null
  voicemail: 'hangup' | 'leave_message' | 'ignore' | string
  voicemail_message: string | null
  dtmf: 'message' | 'ignore' | string
  summary: boolean
  turn_detection: Record<string, unknown> | null
  source: 'settings' | 'stored' | string
  editable: boolean
}

/** Body of `PUT /api/v1/telephony/agents/{name}`: the agent without its name and where it came from. */
export type PhoneAgentUpdate = Partial<Omit<PhoneAgent, 'name' | 'source' | 'editable'>>

export interface PhoneNumber {
  number: string
  agent: string
  source: string
  editable: boolean
}

export interface TelephonyOverview {
  public_base_url: string | null
  validate_signatures: boolean
  default_agent: string | null
  providers: TelephonyProvider[]
  agents: PhoneAgent[]
  numbers: PhoneNumber[]
  editable: boolean
  edit_blocked_reason: string | null
  llm_configured: boolean
  stats: {
    active_calls: number
    calls_total: number
    by_outcome: { provider: string; outcome: string; calls: number }[]
  }
  compliance_notice: string
}

export interface LatencyQuantiles {
  p50: number
  max: number
  mean: number
}

/** Per-call latency: each part of a turn, from the end of the caller's speech to the first audio sent back. */
export interface CallLatency {
  turns: number
  endpoint_ms?: LatencyQuantiles
  stt_ms?: LatencyQuantiles
  llm_first_token_ms?: LatencyQuantiles
  tts_first_audio_ms?: LatencyQuantiles
  send_ms?: LatencyQuantiles
  total_ms?: LatencyQuantiles
  response_ms?: LatencyQuantiles
}

export interface CallSummary {
  id: string
  provider: string
  direction: 'inbound' | 'outbound' | string
  agent: string | null
  from: string | null
  to: string | null
  provider_call_id: string | null
  created_at: string
  started_at: string | null
  ended_at: string | null
  status: 'ringing' | 'in_progress' | 'ended' | string
  outcome: string | null
  error: string | null
  answered_by: string | null
  duration_s: number | null
  pipeline: string | null
  turns: number
  interruptions: number
  latency: CallLatency
  summary: string | null
}

export interface CallTurn {
  index: number
  kind: 'speech' | 'dtmf' | 'tool' | 'prompt' | string
  started_ms: number
  user_text: string | null
  assistant_text: string | null
  interrupted: boolean
  status: string
  tool_calls: string[]
  latency: {
    endpoint_ms: number | null
    stt_ms: number | null
    llm_first_token_ms: number | null
    tts_first_audio_ms: number | null
    send_ms: number | null
    total_ms: number | null
    response_ms: number | null
  }
}

export interface CallTranscriptLine {
  t_ms: number
  speaker: 'caller' | 'agent' | string
  text: string
  kind?: string
  interrupted?: boolean
  full_text?: string
}

export interface CallEvent {
  t_ms: number
  type: string
  [k: string]: unknown
}

export interface CallRecord extends Omit<CallSummary, 'turns'> {
  stream_id: string | null
  transfer_to: string | null
  dtmf: string
  summary_error: string | null
  metadata: Record<string, unknown>
  timeline: CallEvent[]
  turns: CallTurn[]
  transcript: CallTranscriptLine[]
}

export interface SimulationCheck {
  type: string
  ok: boolean
  detail: string
}

/** The call simulator's report (`POST /api/v1/telephony/simulate`). */
export interface SimulationReport {
  scenario: string
  description?: string | null
  provider: string
  agent: string | null
  call_id: string | null
  passed: boolean
  checks: SimulationCheck[]
  duration_s: number
  noise_db: number | null
  caller: {
    utterances: { label: string; end_s: number; seconds: number; response_ms: number | null }[]
    response_ms: LatencyQuantiles | null
    agent_audio_s: number
    clears: number
    cleared_s: number
  }
  outcome: string | null
  latency: CallLatency | null
  record: CallRecord | null
}

/** How a saved voice's maker stated consent: `self` = "this is my voice", `permission` = the speaker's permission. */
export type ConsentBasis = 'self' | 'permission' | 'recording'

/** Speaker similarity of one model's clone of a saved voice (`GET /api/v1/voices/{id}/score`). */
export interface VoiceScore {
  voice?: string
  model: string
  /** Cosine similarity of speaker embeddings, -1 to 1. */
  similarity?: number
  verdict?: 'same speaker' | 'similar' | 'different' | null
  embedding_model?: string
  /** Whether the verdict's thresholds were measured with this embedding model. */
  calibrated?: boolean
  text?: string
  audio_s?: number
  rtf?: number | null
  ttfa_ms?: number | null
  at?: string
  /** `data:audio/wav;base64,…` when asked for (`audio=true`). */
  audio?: string
  /** Set instead of a score when scoring failed (creation still succeeded). */
  error?: string
}

/** A saved voice: a cleaned reference recording plus what is known about it. */
export interface SavedVoice {
  id: string
  object: 'voice'
  name: string
  description: string | null
  language: string | null
  transcript: string | null
  transcript_source?: 'auto'
  default_model: string | null
  created_at: string
  consent: {
    basis: ConsentBasis
    statement: string
    by: { auth?: string; name?: string | null; client?: string | null }
    at: string
    recording?: string
  }
  reference: {
    sample_rate: number
    duration_s: number
    speech_s?: number
    input_s?: number
    clips?: number
    snr_db?: number | null
    level_dbfs?: number
    gain_db?: number | null
    speaker_consistency?: number
  }
  warnings: string[]
  scores: Record<string, VoiceScore>
  audio_url: string
  /** Only in the answer to `POST /api/v1/voices` with a score model. */
  score?: VoiceScore
}

/** A saved blend of a model's own voices (`"kind": "blend"`): no recording, no consent, no score. */
export interface SavedBlend {
  id: string
  object: 'voice'
  kind: 'blend'
  name: string
  description: string | null
  language: string | null
  /** `af_bella(0.7)+af_sky(0.3)`, normalised weights. */
  blend: string
  voices: { voice: string; weight: number }[]
  /** The model the blend is for. */
  default_model: string | null
  created_at: string
  audio_url: null
}

/** `POST /v1/audio/speech/embedding`: OpenAI's embeddings shape. */
export interface SpeakerEmbedding {
  object: 'list'
  data: { object: 'embedding'; embedding: number[]; index: number }[]
  model: string
  usage?: { prompt_tokens: number; total_tokens: number }
}

export interface VoiceList {
  voices: SavedVoice[]
  /** Saved blends, apart from the recordings (newer servers). */
  blends?: SavedBlend[]
  blending_models?: string[]
  cloning_models: string[]
  consent: Record<string, string>
  similarity: { embedding_model: string; same_speaker: number; similar: number; calibration?: Record<string, unknown> }
}

/** A stored API key as `/api/v1/keys` returns it: never the key itself, except `key` once, on creation. */
export interface ApiKeyInfo {
  id: string
  name: string
  hint: string
  scopes: string[]
  created_at: string
  expires_at: string | null
  revoked_at: string | null
  active: boolean
  rate_limit: number | null
  burst: number | null
  quota: number | null
  quota_period: 'day' | 'month' | 'total'
  usage: {
    requests: number
    period: string
    period_requests: number
    quota_remaining: number | null
    by_scope: Record<string, number>
    rejected: number
    last_used_at: string | null
  }
  key?: string
}

export interface ApiKeyList {
  auth: boolean
  root_key: boolean
  you: string | null
  scopes: Record<string, string>
  periods: string[]
  data: ApiKeyInfo[]
}

export interface ApiKeyCreate {
  name: string
  scopes?: string[]
  rate_limit?: number | null
  burst?: number | null
  quota?: number | null
  quota_period?: string
  expires_in_days?: number | null
}

export type ApiKeyUpdate = Partial<Omit<ApiKeyCreate, 'expires_in_days'>> & { reset_usage?: boolean }
