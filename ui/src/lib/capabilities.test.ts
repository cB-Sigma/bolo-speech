import { describe, expect, it } from 'vitest'
import type { CatalogModel, Component, ServerInfo } from './api/types'
import {
  autoPull,
  CLOUD_NOTICE,
  cloudNotice,
  defaultPipeline,
  diarizerOptions,
  hardwareLabel,
  isCloud,
  isCpuFriendly,
  isNotImplemented,
  isPending,
  notInstalledHint,
  notRunnableReason,
  providersOffered,
  S2S_CLOUD_NOTE,
  wordListsAnnounced,
  phonemizerChoices,
  pipelineOptions,
  runtimeLabel,
  S2S_LICENCE_NOTE,
  speakerConditionedModels,
  voiceFamily,
} from './capabilities'

const info = (realtime: Record<string, unknown>, extra: Record<string, unknown> = {}): ServerInfo =>
  ({
    name: 'bolo',
    version: '0.2.0',
    engines: [],
    device: 'cpu',
    cpu_count: 4,
    platform: 'linux',
    auth_required: false,
    data_dir: '/x',
    capabilities: { realtime, ...extra },
  }) as ServerInfo

const model = (over: Partial<CatalogModel>): CatalogModel =>
  ({
    id: 'm',
    aliases: [],
    task: 'tts',
    engine: 'sherpa-onnx',
    family: 'F',
    languages: ['en'],
    tier: 'realtime',
    streaming: false,
    sample_rate: 16000,
    size_bytes: 1,
    params: null,
    license: 'MIT',
    gated: false,
    voices: [],
    features: [],
    installed: true,
    loaded: false,
    pulling: null,
    description: '',
    source_url: '',
    validated: 'cpu',
    ...over,
  }) as CatalogModel

describe('pipelineOptions', () => {
  it('reads availability from info and keeps unavailable pipelines with a reason', () => {
    const opts = pipelineOptions(
      info({
        pipeline: 'cascade',
        pipelines: {
          cascade: { available: true },
          'audio-llm': { configured: false, base_url: null, model: null },
          'speech-to-speech': { model: 'lfm3-audio-2b-llamacpp', url: 'http://127.0.0.1:9/' },
        },
      }),
    )
    expect(opts.map((o) => [o.id, o.available])).toEqual([
      ['cascade', true],
      ['audio-llm', false],
      ['speech-to-speech', true],
    ])
    expect(opts[1]?.reason).toMatch(/audio_llm/)
    expect(opts[2]?.note).toBe(S2S_LICENCE_NOTE)
    expect(opts[2]?.detail).toContain('http://127.0.0.1:9/')
  })

  it('needs llama-server for a managed speech-to-speech model', () => {
    const llama = { id: 'llama-server', installed: false } as Component
    const opts = pipelineOptions(
      info({ pipelines: { 'speech-to-speech': { model: 'lfm3-audio-2b-llamacpp', url: null } } }),
      [llama],
    )
    expect(opts[2]).toMatchObject({ available: false })
    expect(opts[2]?.reason).toMatch(/llama-server/)
    expect(
      pipelineOptions(info({ pipelines: { 'speech-to-speech': { model: 'x' } } }), [{ ...llama, installed: true }])[2]
        ?.available,
    ).toBe(true)
  })

  it('takes availability and the reason from the server when it reports them', () => {
    const llama = { id: 'llama-server', installed: true } as Component
    const off = pipelineOptions(
      info({
        pipelines: {
          cascade: { configured: true, available: true },
          'audio-llm': { configured: true, available: true, model: 'gemma' },
          'speech-to-speech': {
            configured: true,
            available: false,
            reason: 'llama-server is not installed (bolo components install llama-server)',
            model: 'lfm3-audio-2b-llamacpp',
          },
        },
      }),
      [llama], // the component list no longer decides: the server's flag does
    )
    expect(off.map((o) => o.available)).toEqual([true, true, false])
    expect(off[2]?.reason).toBe('Llama-server is not installed (bolo components install llama-server).')
    const cloud = pipelineOptions(
      info({
        pipelines: {
          'speech-to-speech': { configured: true, available: true, model: 'openai:gpt-realtime-2.1', cloud: true },
        },
      }),
    )
    expect(cloud[2]).toMatchObject({ available: true, note: S2S_CLOUD_NOTE })
  })

  it('offers nothing on servers (or the demo) without capabilities', () => {
    const opts = pipelineOptions(undefined)
    expect(opts.every((o) => !o.available)).toBe(true)
    expect(defaultPipeline(undefined, opts)).toBe('cascade')
  })

  it('prefers the server default only when it is usable', () => {
    const i = info({
      pipeline: 'audio-llm',
      pipelines: { cascade: { available: true }, 'audio-llm': { configured: false } },
    })
    expect(defaultPipeline(i, pipelineOptions(i))).toBe('cascade')
  })
})

describe('diarization helpers', () => {
  const catalog = [
    model({ id: 'nemotron-3-diarization', task: 'diarization', installed: false }),
    model({ id: 'pyannote-3.0-eres2net', task: 'diarization' }),
    model({ id: 'diarizen-x', task: 'diarization', description: 'Not runnable yet: no runtime.' }),
    model({ id: 'dicow-v3.3', task: 'stt', features: ['speaker-conditioned'], available: false }),
  ]
  it('puts the default first, labels Nemotron and drops models that cannot run', () => {
    const d = diarizerOptions(info({}, { diarization: { default_model: 'pyannote-3.0-eres2net' } }), catalog)
    expect(d.map((x) => x.id)).toEqual(['pyannote-3.0-eres2net', 'nemotron-3-diarization'])
    expect(d[0]?.label).toContain('(default)')
    expect(d[1]?.note).toBe(
      'Better with overlapping speech and similar voices; can merge speakers on clean phone-quality turn-taking',
    )
  })
  it('finds speaker-conditioned models', () => {
    expect(speakerConditionedModels(undefined, catalog).map((m) => m.id)).toEqual(['dicow-v3.3'])
  })
})

describe('catalog entry helpers', () => {
  it('marks pending and not-runnable entries', () => {
    expect(isPending(model({ status: 'pending' }))).toBe(true)
    expect(isNotImplemented(model({ description: 'Big model. Not runnable yet: no runtime has it.' }))).toBe(true)
    expect(isNotImplemented(model({}))).toBe(false)
  })
  it('reads the runnable flag before the description', () => {
    const flagged = model({ runnable: false, not_runnable_reason: 'no runtime bolo can use runs this model yet' })
    expect(isNotImplemented(flagged)).toBe(true)
    expect(notRunnableReason(flagged)).toBe('No runtime bolo can use runs this model yet.')
    // a server that says "runnable" is believed, whatever the description says
    expect(isNotImplemented(model({ runnable: true, description: 'Not runnable yet: old text.' }))).toBe(false)
    expect(notRunnableReason(model({}))).toBeNull()
  })
  it('recognises cloud models and their notice', () => {
    expect(isCloud(model({ cloud: true, engine: 'remote' }))).toBe(true)
    expect(isCloud(model({ engine: 'remote' }))).toBe(true)
    expect(isCloud(model({}))).toBe(false)
    expect(isCloud(null)).toBe(false)
    expect(cloudNotice(model({ notice: 'paid cloud API; audio leaves this server' }))).toBe(CLOUD_NOTICE)
    expect(cloudNotice(model({}))).toBe(CLOUD_NOTICE)
  })
  it('describes hardware needs', () => {
    expect(hardwareLabel(null)).toBeNull()
    expect(hardwareLabel({ gpu: 'recommended', vram_gb: 3.24 })).toBe('GPU recommended, ~3.2 GB')
    expect(isCpuFriendly(model({ hardware: { gpu: 'optional', vram_gb: 1 } }))).toBe(true)
    expect(isCpuFriendly(model({ hardware: { gpu: 'required' } }))).toBe(false)
  })
  it('groups voice families', () => {
    expect(voiceFamily(model({ id: 'veena-llamacpp', engine: 'tts-server' }))).toMatch(/Orpheus/)
    expect(voiceFamily(model({ id: 'voxcpm2-audiocpp', engine: 'tts-server' }))).toBe('audio.cpp voices')
    expect(voiceFamily(model({ id: 'veena', engine: 'plugin:torch' }))).toBe('Engine plugin voices')
    expect(voiceFamily(model({}))).toBe('Built-in voices')
    expect(voiceFamily(model({ id: 'openai-tts-1', engine: 'remote' }))).toBe('Cloud voices (paid APIs)')
    expect(runtimeLabel(model({ id: 'openai-tts-1', engine: 'remote' }))).toBe('cloud API')
    expect(runtimeLabel(model({ id: 'x-audiocpp', engine: 'tts-server', requires: [] }))).toBe('audio.cpp')
  })
  it('takes phonemiser choices per voice when the voice has them', () => {
    const m = model({ phonemizers: [{ id: 'espeak-ng', available: true, default: true }] })
    expect(phonemizerChoices(m, null).map((c) => c.id)).toEqual(['espeak-ng'])
    const v = { id: 'af', name: 'A', language: 'en', phonemizers: [{ id: 'uccharan', available: true, default: true }] }
    expect(phonemizerChoices(m, v).map((c) => c.id)).toEqual(['uccharan'])
  })
})

describe('server behaviour flags', () => {
  it('reads auto_pull and words the not-installed hint from it', () => {
    const on = { ...info({}), auto_pull: true } as ServerInfo
    const off = { ...info({}), auto_pull: false } as ServerInfo
    expect(autoPull(on)).toBe(true)
    expect(autoPull(off)).toBe(false)
    expect(autoPull(info({}))).toBeNull()
    expect(notInstalledHint(on, model({}))).toMatch(/downloads it on first use/)
    expect(notInstalledHint(off, model({}))).toMatch(/auto_pull is off/)
    expect(notInstalledHint(info({}), model({}))).toMatch(/Pull it from the Models screen/)
    expect(notInstalledHint(on, model({ cloud: true }))).toMatch(/Providers screen/)
  })
  it('knows from capabilities whether word lists and providers exist', () => {
    expect(wordListsAnnounced(undefined)).toBeNull()
    expect(wordListsAnnounced(info({}))).toBeNull() // an older server: probe the route
    expect(wordListsAnnounced(info({}, { wordlists: { available: false, endpoint: '/api/v1/wordlists' } }))).toBe(true)
    expect(wordListsAnnounced(info({}, { providers: { endpoint: '/api/v1/providers' } }))).toBe(false)
    expect(providersOffered(info({}, { providers: {} }))).toBe(true)
    expect(providersOffered(info({}))).toBe(false)
  })
})
