import { apiUrl, authHeaders, getTransport } from './config'
import { parseTranscription, transcriptionForm, type DiarizedResult, type TranscribeOptions } from '@/lib/diarize'
import type { DialogueBody } from '@/lib/dialogue'

/** The answer of `POST /v1/audio/watermark/detect`. */
export interface WatermarkDetection {
  model: string
  marked: boolean
  score: number
  payload: string | null
  payload_is_this_servers: boolean
  segments: { start: number; end: number }[]
  duration: number
  results?: WatermarkDetection[]
}
import type {
  ApiKeyCreate,
  ApiKeyInfo,
  ApiKeyList,
  ApiKeyUpdate,
  AudioEmotion,
  AudioTags,
  BenchRequest,
  BenchResult,
  CallRecord,
  CallSummary,
  Catalog,
  Component,
  EnrolledSpeaker,
  FitWarning,
  HardwareInfo,
  Job,
  PhoneAgent,
  PhoneAgentUpdate,
  PhoneNumber,
  Provider,
  ProviderList,
  ProviderTest,
  ProviderUpdate,
  RequestTrace,
  Sample,
  SavedBlend,
  SavedVoice,
  ServerInfo,
  SetupPlan,
  SetupUse,
  SimulationReport,
  LanguageId,
  Alignment,
  SpeakerEmbedding,
  SpeakerIdentification,
  SpeakerList,
  SpeakerVerification,
  SpeedInfo,
  Stats,
  Task,
  TelephonyOverview,
  TextInput,
  VoiceList,
  VoiceScore,
  WordList,
  WordListPreview,
  WordListPutResult,
  WordListSummary,
} from './types'

export class ApiError extends Error {
  readonly status: number
  readonly body: unknown
  constructor(status: number, message: string, body?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
  /** Machine-readable error code, when the server sent one (`error.code` or `code`). */
  get code(): string | null {
    const b = this.body as { code?: unknown; error?: { code?: unknown } } | undefined
    const c = b?.error?.code ?? b?.code
    return typeof c === 'string' ? c : null
  }
  get unauthorized(): boolean {
    return this.status === 401 || this.status === 403
  }
}

/** Raised when the server cannot be reached at all (as opposed to an HTTP error). */
export class NetworkError extends Error {
  constructor(message = 'Server unreachable') {
    super(message)
    this.name = 'NetworkError'
  }
}

function errorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === 'object') {
    const b = body as Record<string, unknown>
    const detail = b.detail ?? b.message ?? b.error
    if (typeof detail === 'string') return detail
    if (detail && typeof detail === 'object' && typeof (detail as Record<string, unknown>).message === 'string') {
      return (detail as Record<string, string>).message as string
    }
  }
  return fallback
}

export interface RequestOptions {
  method?: string
  query?: Record<string, string | number | boolean | undefined | null>
  json?: unknown
  body?: BodyInit
  signal?: AbortSignal
  headers?: Record<string, string>
}

export async function request(path: string, opts: RequestOptions = {}): Promise<Response> {
  const headers: Record<string, string> = { ...authHeaders(), ...opts.headers }
  let body = opts.body
  if (opts.json !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(opts.json)
  }
  let res: Response
  try {
    res = await getTransport().fetch(apiUrl(path, opts.query), {
      method: opts.method ?? 'GET',
      headers,
      body,
      signal: opts.signal,
    })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err
    throw new NetworkError(err instanceof Error ? err.message : undefined)
  }
  if (!res.ok) {
    let parsed: unknown
    try {
      parsed = await res.clone().json()
    } catch {
      parsed = undefined
    }
    throw new ApiError(res.status, errorMessage(parsed, `${res.status} ${res.statusText}`.trim()), parsed)
  }
  return res
}

async function json<T>(path: string, opts?: RequestOptions): Promise<T> {
  const res = await request(path, opts)
  if (res.status === 204) return undefined as T
  const text = await res.text()
  return (text ? JSON.parse(text) : undefined) as T
}

export interface CatalogQuery {
  task?: Task | ''
  q?: string
  installed?: boolean
}

export const api = {
  info: (signal?: AbortSignal) => json<ServerInfo>('api/v1/info', { signal }),
  catalog: (q: CatalogQuery = {}, signal?: AbortSignal) =>
    json<Catalog>('api/v1/catalog', { query: { task: q.task, q: q.q, installed: q.installed }, signal }),
  pull: (id: string) =>
    json<{ job_id: string; warnings?: FitWarning[] }>('api/v1/models/pull', { method: 'POST', json: { id } }),
  job: <R = unknown>(jobId: string, signal?: AbortSignal) =>
    json<Job<R>>(`api/v1/jobs/${encodeURIComponent(jobId)}`, { signal }),
  /** `keepAlive`: seconds or a duration ("10m"); -1 keeps the model loaded until it is unloaded. */
  load: (id: string, keepAlive?: number | string) =>
    json<{ warnings?: FitWarning[]; processor?: string; notice?: string } | undefined>(
      `api/v1/models/${encodeURIComponent(id)}/load`,
      { method: 'POST', ...(keepAlive === undefined ? {} : { json: { keep_alive: keepAlive } }) },
    ),
  /** This machine as the server sees it; `refresh` looks again and runs the slower probes too. */
  hardware: (refresh = false, signal?: AbortSignal) =>
    json<HardwareInfo>('api/v1/hardware', { query: { refresh: refresh || undefined }, signal }),
  /** A few seconds of CPU on the server: times a tiny workload so real-time factors are scaled, not guessed. */
  benchmarkHardware: () =>
    json<{ benchmark: Record<string, unknown>; speed: SpeedInfo }>('api/v1/hardware/benchmark', { method: 'POST' }),
  /** Models for one use and language, picked for the server's machine. Advice: nothing is pulled by this call. */
  recommendSetup: (
    q: { use: SetupUse; language?: string; commercial?: boolean; local_only?: boolean },
    signal?: AbortSignal,
  ) => json<SetupPlan>('api/v1/recommendations/setup', { query: q, signal }),
  unload: (id: string) => json<unknown>(`api/v1/models/${encodeURIComponent(id)}/unload`, { method: 'POST' }),
  remove: (id: string) => json<void>(`api/v1/models/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  stats: (signal?: AbortSignal) => json<Stats>('api/v1/stats', { signal }),
  requests: (q: { limit?: number; kind?: string; model?: string } = {}, signal?: AbortSignal) =>
    json<{ requests: RequestTrace[] }>('api/v1/requests', { query: q, signal }),
  requestTrace: (id: string, signal?: AbortSignal) =>
    json<RequestTrace>(`api/v1/requests/${encodeURIComponent(id)}`, { signal }),
  /** Cloud providers. No call returns an API key: the server only says whether one is stored. */
  providers: (signal?: AbortSignal) => json<ProviderList>('api/v1/providers', { signal }),
  putProvider: (name: string, body: ProviderUpdate) =>
    json<Provider>(`api/v1/providers/${encodeURIComponent(name)}`, { method: 'PUT', json: body }),
  deleteProvider: (name: string) => json<void>(`api/v1/providers/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  testProvider: (name: string) =>
    json<ProviderTest>(`api/v1/providers/${encodeURIComponent(name)}/test`, { method: 'POST' }),
  /** API keys. The key itself comes back once, from `createKey`. */
  keys: (signal?: AbortSignal) => json<ApiKeyList>('api/v1/keys', { signal }),
  createKey: (body: ApiKeyCreate) => json<ApiKeyInfo>('api/v1/keys', { method: 'POST', json: body }),
  updateKey: (id: string, body: ApiKeyUpdate) =>
    json<ApiKeyInfo>(`api/v1/keys/${encodeURIComponent(id)}`, { method: 'PATCH', json: body }),
  revokeKey: (id: string, remove = false) =>
    json<void>(`api/v1/keys/${encodeURIComponent(id)}`, { method: 'DELETE', query: { delete: remove || undefined } }),
  /** Phone calls: providers and the addresses to give them, agents, numbers. No call returns a credential. */
  telephony: (signal?: AbortSignal) => json<TelephonyOverview>('api/v1/telephony', { signal }),
  putAgent: (name: string, body: PhoneAgentUpdate) =>
    json<PhoneAgent>(`api/v1/telephony/agents/${encodeURIComponent(name)}`, { method: 'PUT', json: body }),
  deleteAgent: (name: string) =>
    json<void>(`api/v1/telephony/agents/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  putNumber: (number: string, agent: string) =>
    json<{ numbers: PhoneNumber[] }>(`api/v1/telephony/numbers/${encodeURIComponent(number)}`, {
      method: 'PUT',
      json: { agent },
    }),
  deleteNumber: (number: string) =>
    json<void>(`api/v1/telephony/numbers/${encodeURIComponent(number)}`, { method: 'DELETE' }),
  calls: (q: { limit?: number; provider?: string; agent?: string } = {}, signal?: AbortSignal) =>
    json<{ calls: CallSummary[] }>('api/v1/telephony/calls', { query: q, signal }),
  call: (id: string, signal?: AbortSignal) =>
    json<CallRecord>(`api/v1/telephony/calls/${encodeURIComponent(id)}`, { signal }),
  scenarios: (signal?: AbortSignal) =>
    json<{ scenarios: { name: string; description: string }[] }>('api/v1/telephony/scenarios', { signal }),
  simulate: (body: { agent?: string; provider?: string; scenario?: string; noise_db?: number }) =>
    json<SimulationReport>('api/v1/telephony/simulate', { method: 'POST', json: body }),
  /** Optional components. Older servers have no such route: that reads as an empty list. */
  components: async (signal?: AbortSignal): Promise<{ components: Component[] }> => {
    try {
      return await json<{ components: Component[] }>('api/v1/components', { signal })
    } catch (err) {
      if (err instanceof ApiError && (err.status === 404 || err.status === 501)) return { components: [] }
      throw err
    }
  },
  /** Starts a server-side install. Only eSpeak NG asks for licence acceptance; the flag is harmless elsewhere. */
  installComponent: (id: string) =>
    json<{ job_id: string }>(`api/v1/components/${encodeURIComponent(id)}/install`, {
      method: 'POST',
      json: { accept_license: true },
    }),
  /** Deletes a downloaded language pack (the only components the server removes). */
  removeComponent: (id: string) => json<void>(`api/v1/components/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  modelSettings: (id: string, signal?: AbortSignal) =>
    json<{ model: string; settings: Record<string, unknown> }>(`api/v1/models/${encodeURIComponent(id)}/settings`, {
      signal,
    }),
  /** Pins (null removes the pin) the phonemiser, the text normaliser or the word lists of a voice. */
  setModelSettings: (
    id: string,
    settings: {
      phonemizer?: string | null
      text_normalizer?: string | null
      wordlists?: string[] | null
      romanised?: string | null
      fallback?: string[] | null
    },
  ) =>
    json<{ model: string; settings: Record<string, unknown> }>(`api/v1/models/${encodeURIComponent(id)}/settings`, {
      method: 'PUT',
      json: settings,
    }),
  /** Word lists. Servers without them answer 404, which `useWordLists` reads as "not offered". */
  wordLists: (signal?: AbortSignal) =>
    json<{ wordlists: WordListSummary[]; available: boolean }>('api/v1/wordlists', { signal }),
  wordList: (name: string, signal?: AbortSignal) =>
    json<WordList>(`api/v1/wordlists/${encodeURIComponent(name)}`, { signal }),
  /** Replaces a list. `body` is TSV, PLS or JSON text, sent as is. */
  putWordList: (name: string, body: string, contentType = 'text/tab-separated-values') =>
    json<WordListPutResult>(`api/v1/wordlists/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body,
      headers: { 'Content-Type': contentType },
    }),
  deleteWordList: (name: string) =>
    json<{ deleted: string }>(`api/v1/wordlists/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  /** The list as a file: TSV, or PLS for one language. */
  async exportWordList(name: string, format: 'tsv' | 'pls', lang?: string): Promise<Blob> {
    const res = await request(`api/v1/wordlists/${encodeURIComponent(name)}`, { query: { format, lang } })
    return res.blob()
  },
  previewWordList: (
    name: string,
    body: { text: string; models: string[]; language?: string; voice?: string; input_format?: string },
    signal?: AbortSignal,
  ) =>
    json<{ previews: WordListPreview[] }>(`api/v1/wordlists/${encodeURIComponent(name)}/preview`, {
      method: 'POST',
      json: body,
      signal,
    }),
  textInput: (model: string, voice?: string, signal?: AbortSignal) =>
    json<{ model: string; voice: string | null; text_input: TextInput }>(
      `api/v1/text-input/${encodeURIComponent(model)}`,
      { query: { voice }, signal },
    ),
  samples: (signal?: AbortSignal) => json<{ samples: Sample[] }>('api/v1/samples', { signal }),
  /** Saved voices. Servers without them answer 404, which `useVoices` reads as "not offered". */
  voices: (signal?: AbortSignal) => json<VoiceList>('api/v1/voices', { signal }),
  /** Creates a saved voice (multipart). `consent` is required by the server; see `NewVoice`. */
  createVoice: (voice: NewVoice, signal?: AbortSignal) =>
    json<SavedVoice>('api/v1/voices', { method: 'POST', body: newVoiceForm(voice), signal }),
  updateVoice: (
    id: string,
    changes: Partial<Pick<SavedVoice, 'name' | 'description' | 'language' | 'transcript' | 'default_model'>>,
  ) => json<SavedVoice>(`api/v1/voices/${encodeURIComponent(id)}`, { method: 'PATCH', json: changes }),
  /** Saves a blend of a model's voices (`af_bella(0.7)+af_sky(0.3)`) under a name. No recording, no consent. */
  createBlend: (b: { name: string; blend: string; model?: string; description?: string }) =>
    json<SavedBlend>('api/v1/voices/blends', { method: 'POST', json: b }),
  /** Deletes a saved voice or a saved blend. */
  deleteVoice: (id: string) =>
    json<{ id: string; deleted: boolean }>(`api/v1/voices/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  /** The cleaned reference recording (WAV). */
  async voiceAudio(id: string, signal?: AbortSignal): Promise<Blob> {
    const res = await request(`api/v1/voices/${encodeURIComponent(id)}/audio`, { signal })
    return res.blob()
  },
  /** Speaks a test sentence in the voice with `model` and compares it with the reference. */
  voiceScore: (id: string, model: string, opts: { text?: string; audio?: boolean } = {}, signal?: AbortSignal) =>
    json<VoiceScore>(`api/v1/voices/${encodeURIComponent(id)}/score`, {
      query: { model, text: opts.text, audio: opts.audio || undefined },
      signal,
    }),
  async exportVoices(ids?: string[]): Promise<Blob> {
    const res = await request('api/v1/voices/export', { query: { ids: ids?.length ? ids.join(',') : undefined } })
    return res.blob()
  },
  /** The speaker embedding of a recording (unit length: the cosine of two is their dot product). */
  async speakerEmbedding(file: Blob, name: string, model?: string, signal?: AbortSignal): Promise<SpeakerEmbedding> {
    const form = new FormData()
    form.set('file', file, name)
    if (model) form.set('model', model)
    const res = await request('v1/audio/speech/embedding', { method: 'POST', body: form, signal })
    return res.json() as Promise<SpeakerEmbedding>
  },
  /** Which language is spoken in a recording (`POST /v1/audio/language`), with the `topK` most likely ones. */
  async identifyLanguage(
    file: Blob,
    name: string,
    opts: { model?: string; topK?: number } = {},
    signal?: AbortSignal,
  ): Promise<LanguageId> {
    const form = new FormData()
    form.set('file', file, name)
    if (opts.model) form.set('model', opts.model)
    if (opts.topK) form.set('top_k', String(opts.topK))
    const res = await request('v1/audio/language', { method: 'POST', body: form, signal })
    return res.json() as Promise<LanguageId>
  },
  /** Background noise removed (`POST /v1/audio/denoise`): the cleaned recording as WAV, and the model that ran. */
  async denoise(
    file: Blob,
    name: string,
    opts: { model?: string; strength?: number } = {},
    signal?: AbortSignal,
  ): Promise<{ audio: Blob; model: string | null }> {
    const form = new FormData()
    form.set('file', file, name)
    form.set('response_format', 'wav')
    if (opts.model) form.set('model', opts.model)
    if (opts.strength != null) form.set('strength', String(opts.strength))
    const res = await request('v1/audio/denoise', { method: 'POST', body: form, signal })
    return { audio: await res.blob(), model: res.headers.get('x-bolo-model') }
  },
  /** Sound events in a recording (`POST /v1/audio/tags`), also window by window when `window` (seconds) is set. */
  async audioTags(
    file: Blob,
    name: string,
    opts: { model?: string; topK?: number; threshold?: number; window?: number } = {},
    signal?: AbortSignal,
  ): Promise<AudioTags> {
    const form = new FormData()
    form.set('file', file, name)
    if (opts.model) form.set('model', opts.model)
    if (opts.topK) form.set('top_k', String(opts.topK))
    if (opts.threshold != null) form.set('threshold', String(opts.threshold))
    if (opts.window) form.set('window', String(opts.window))
    const res = await request('v1/audio/tags', { method: 'POST', body: form, signal })
    return res.json() as Promise<AudioTags>
  },
  /** How the speaker sounds (`POST /v1/audio/emotion`), also window by window when `window` (seconds) is set. */
  async audioEmotion(
    file: Blob,
    name: string,
    opts: { model?: string; window?: number } = {},
    signal?: AbortSignal,
  ): Promise<AudioEmotion> {
    const form = new FormData()
    form.set('file', file, name)
    if (opts.model) form.set('model', opts.model)
    if (opts.window) form.set('window', String(opts.window))
    const res = await request('v1/audio/emotion', { method: 'POST', body: form, signal })
    return res.json() as Promise<AudioEmotion>
  },
  /** A question about a recording, answered by an audio language model that hears it (`/v1/chat/completions`). */
  async askAudio(file: Blob, format: string, question: string, model: string, signal?: AbortSignal): Promise<string> {
    const bytes = new Uint8Array(await file.arrayBuffer())
    let binary = ''
    for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
    const body = {
      model,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: question },
            { type: 'input_audio', input_audio: { data: btoa(binary), format } },
          ],
        },
      ],
    }
    const out = await json<{ choices: { message: { content: string | null } }[] }>('v1/chat/completions', {
      method: 'POST',
      json: body,
      signal,
    })
    return out.choices[0]?.message.content ?? ''
  },
  speakers: () => json<SpeakerList>('v1/audio/speakers'),
  /** Enrol a speaker from one or more recordings of them alone. */
  async enrolSpeaker(name: string, clips: { blob: Blob; name: string }[], model?: string): Promise<EnrolledSpeaker> {
    const form = new FormData()
    form.set('name', name)
    for (const c of clips) form.append('file', c.blob, c.name)
    if (model) form.set('model', model)
    const res = await request('v1/audio/speakers', { method: 'POST', body: form })
    return res.json() as Promise<EnrolledSpeaker>
  },
  /** Add recordings to an enrolled speaker, or rename them. */
  async changeSpeaker(id: string, changes: { name?: string; clips?: { blob: Blob; name: string }[] }) {
    const form = new FormData()
    if (changes.name) form.set('name', changes.name)
    for (const c of changes.clips ?? []) form.append('file', c.blob, c.name)
    const res = await request(`v1/audio/speakers/${encodeURIComponent(id)}`, { method: 'POST', body: form })
    return res.json() as Promise<EnrolledSpeaker>
  },
  deleteSpeaker: (id: string) =>
    json<{ deleted: boolean }>(`v1/audio/speakers/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  deleteSpeakerClip: (id: string, clip: string) =>
    json<EnrolledSpeaker>(`v1/audio/speakers/${encodeURIComponent(id)}/clips/${encodeURIComponent(clip)}`, {
      method: 'DELETE',
    }),
  /** Is this recording the enrolled speaker? */
  async verifySpeaker(
    file: Blob,
    name: string,
    speaker: string,
    opts: { model?: string; threshold?: number } = {},
  ): Promise<SpeakerVerification> {
    const form = new FormData()
    form.set('file', file, name)
    form.set('speaker', speaker)
    if (opts.model) form.set('model', opts.model)
    if (opts.threshold != null) form.set('threshold', String(opts.threshold))
    const res = await request('v1/audio/speakers/verify', { method: 'POST', body: form })
    return res.json() as Promise<SpeakerVerification>
  },
  /** Which enrolled speaker is this, if any? */
  async identifySpeaker(
    file: Blob,
    name: string,
    opts: { model?: string; threshold?: number } = {},
  ): Promise<SpeakerIdentification> {
    const form = new FormData()
    form.set('file', file, name)
    if (opts.model) form.set('model', opts.model)
    if (opts.threshold != null) form.set('threshold', String(opts.threshold))
    const res = await request('v1/audio/speakers/identify', { method: 'POST', body: form })
    return res.json() as Promise<SpeakerIdentification>
  },
  /** Each word of `text` with when it is spoken in the recording (`POST /v1/audio/alignments`). */
  async align(
    file: Blob,
    name: string,
    text: string,
    opts: { model?: string; language?: string } = {},
    signal?: AbortSignal,
  ): Promise<Alignment> {
    const form = new FormData()
    form.set('file', file, name)
    form.set('text', text)
    if (opts.model) form.set('model', opts.model)
    if (opts.language) form.set('language', opts.language)
    const res = await request('v1/audio/alignments', { method: 'POST', body: form, signal })
    return res.json() as Promise<Alignment>
  },
  bench: (req: BenchRequest) => json<{ job_id: string }>('api/v1/bench', { method: 'POST', json: req }),
  benchJob: (jobId: string, signal?: AbortSignal) => api.job<BenchResult>(jobId, signal),

  /** Round-trip time of a tiny request, used as the network share of latency. */
  async ping(signal?: AbortSignal): Promise<number> {
    const t0 = performance.now()
    await request('healthz', { signal })
    return performance.now() - t0
  },

  /** Fetch a sample (or any server-relative URL) as bytes. */
  async bytes(url: string, signal?: AbortSignal): Promise<ArrayBuffer> {
    const res = await request(url, { signal })
    return res.arrayBuffer()
  },

  /** One-shot synthesis through the OpenAI-compatible route. */
  async speech(body: SpeechBody, signal?: AbortSignal): Promise<Response> {
    return request('v1/audio/speech', { method: 'POST', json: body, signal })
  },

  /** A multi-speaker dialogue as one recording (`POST /v1/audio/dialogue`). */
  async dialogue(body: DialogueBody, signal?: AbortSignal): Promise<Response> {
    return request('v1/audio/dialogue', { method: 'POST', json: body, signal })
  },

  /** The same words in another voice (`POST /v1/audio/voice-conversion`). */
  async convertVoice(
    source: Blob,
    opts: { model?: string; voice?: string; reference?: Blob },
    signal?: AbortSignal,
  ): Promise<Response> {
    const form = new FormData()
    form.set('file', source, 'source.wav')
    if (opts.model) form.set('model', opts.model)
    if (opts.voice) form.set('voice', opts.voice)
    if (opts.reference) form.set('reference_audio', opts.reference, 'reference.wav')
    form.set('response_format', 'wav')
    return request('v1/audio/voice-conversion', { method: 'POST', body: form, signal })
  },

  /** Is this recording watermarked? (`POST /v1/audio/watermark/detect`) */
  async detectWatermark(file: Blob, model?: string, signal?: AbortSignal): Promise<WatermarkDetection> {
    const form = new FormData()
    form.set('file', file, 'clip')
    if (model) form.set('model', model)
    const res = await request('v1/audio/watermark/detect', { method: 'POST', body: form, signal })
    return (await res.json()) as WatermarkDetection
  },

  /** One-shot transcription through the OpenAI-compatible route. */
  async transcribe(file: Blob, model: string, language?: string, signal?: AbortSignal): Promise<{ text: string }> {
    const form = new FormData()
    form.set('file', file, 'audio.wav')
    form.set('model', model)
    form.set('response_format', 'json')
    if (language) form.set('language', language)
    const res = await request('v1/audio/transcriptions', { method: 'POST', body: form, signal })
    return res.json() as Promise<{ text: string }>
  },

  /** One-shot transcription that also returns the response headers (request id, model that answered, cost). */
  async transcribeWithHeaders(
    file: Blob,
    model: string,
    language?: string,
    signal?: AbortSignal,
  ): Promise<{ text: string; headers: Headers }> {
    const form = new FormData()
    form.set('file', file, 'audio.wav')
    form.set('model', model)
    form.set('response_format', 'json')
    if (language) form.set('language', language)
    const res = await request('v1/audio/transcriptions', { method: 'POST', body: form, signal })
    const body = (await res.json()) as { text: string }
    return { text: body.text, headers: res.headers }
  },

  /** Transcription with segments, optionally speaker-attributed (diarization, separation of overlaps). */
  async transcribeSegments(
    file: Blob,
    name: string,
    opts: TranscribeOptions,
    signal?: AbortSignal,
  ): Promise<DiarizedResult> {
    const form = transcriptionForm(file, name, opts)
    const res = await request('v1/audio/transcriptions', { method: 'POST', body: form, signal })
    return parseTranscription(await res.json(), res.headers)
  },
}

/** What `POST /api/v1/voices` takes. */
export interface NewVoice {
  name: string
  /** One or more recordings of the same person. */
  files: { blob: Blob; name: string }[]
  /** Required: `self` ("this is my voice") or `permission` ("I have the speaker's permission"). */
  consent: 'self' | 'permission'
  transcript?: string
  language?: string
  description?: string
  defaultModel?: string
  /** Speak a test sentence with this model and return the speaker similarity. */
  scoreModel?: string
  scoreAudio?: boolean
}

export function newVoiceForm(v: NewVoice): FormData {
  const form = new FormData()
  form.set('name', v.name)
  form.set('consent', v.consent)
  for (const f of v.files) form.append('file', f.blob, f.name)
  if (v.transcript) form.set('transcript', v.transcript)
  if (v.language) form.set('language', v.language)
  if (v.description) form.set('description', v.description)
  if (v.defaultModel) form.set('default_model', v.defaultModel)
  if (v.scoreModel) form.set('score_model', v.scoreModel)
  if (v.scoreAudio) form.set('score_audio', 'true')
  return form
}

/** Body of `POST /v1/audio/speech`, with the bolo extensions the console uses. */
export interface SpeechBody {
  model: string
  input: string
  voice?: string
  response_format?: string
  language?: string
  phonemizer?: string
  /** Base64 or `data:audio/…;base64,` recording of the voice to imitate (models with `voice-cloning`). */
  reference_audio?: string
  /** Model-specific options (Tontaube: `style`, `rate`, `pause`, `pitch_range`, `pitch_level`, `legato`). */
  options?: Record<string, string | number>
  reference_text?: string
  /** `text` (default), `ssml` or `markdown`. */
  input_format?: string
  /** Word lists to apply, highest priority first. */
  wordlists?: string[]
  normalize?: boolean | string
  /** Romanised text: `off`, `auto` or `on` (omitted: the model's setting, else the server's). */
  romanised?: string
  /** Guess the language from the text when none is given (the server's default is true). */
  detect_language?: boolean
  pronunciations?: { word: string; ipa?: string; say?: string; lang?: string; match?: string }[]
  /** Direction for the delivery in words, for models whose `features` include `instructions`. */
  instructions?: string
  /** `on`, `off` or a watermark model; unset: the server's policy. */
  watermark?: string
}
