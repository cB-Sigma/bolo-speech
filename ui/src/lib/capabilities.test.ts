import { describe, expect, it } from 'vitest'
import type { CatalogModel, Component, ServerInfo } from './api/types'
import {
  defaultPipeline,
  diarizerOptions,
  hardwareLabel,
  isCpuFriendly,
  isNotImplemented,
  isPending,
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
    expect(d[1]?.note).toBe('Better with overlapping or phone audio, misses more quiet speech')
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
    expect(runtimeLabel(model({ id: 'x-audiocpp', engine: 'tts-server', requires: [] }))).toBe('audio.cpp')
  })
  it('takes phonemiser choices per voice when the voice has them', () => {
    const m = model({ phonemizers: [{ id: 'espeak-ng', available: true, default: true }] })
    expect(phonemizerChoices(m, null).map((c) => c.id)).toEqual(['espeak-ng'])
    const v = { id: 'af', name: 'A', language: 'en', phonemizers: [{ id: 'uccharan', available: true, default: true }] }
    expect(phonemizerChoices(m, v).map((c) => c.id)).toEqual(['uccharan'])
  })
})
