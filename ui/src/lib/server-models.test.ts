import { describe, expect, it } from 'vitest'
import type { CatalogModel, ServerInfo } from './api/types'
import { engineMissing, isFullDuplex, licenceTags, pipelineOptions, readsText, runtimeLabel } from './capabilities'
import { parseRoute } from './diarize'
import { sessionUpdate } from './realtime/session'
import { denoiseParams } from './stt/server'

const model = (over: Partial<CatalogModel>): CatalogModel =>
  ({
    id: 'm',
    aliases: [],
    task: 'stt',
    engine: 'sherpa-onnx',
    family: 'f',
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
    installed: false,
    loaded: false,
    pulling: null,
    description: '',
    source_url: '',
    validated: 'untested',
    ...over,
  }) as CatalogModel

describe('x-bolo-route', () => {
  it('reads the decision of model=auto', () => {
    expect(
      parseRoute(
        'auto -> indicconformer-ta-community; reason=recommended; language=ta; probability=0.987; lid=voxlingua107-ecapa',
      ),
    ).toEqual({
      requested: 'auto',
      answered: 'indicconformer-ta-community',
      reason: 'recommended',
      language: 'ta',
      probability: 0.987,
      lid: 'voxlingua107-ecapa',
      recommended: null,
    })
    expect(parseRoute('auto -> whisper-base; reason=default; language=or; recommended=x')?.recommended).toBe('x')
    expect(parseRoute(null)).toBeNull()
  })
})

describe('live denoising', () => {
  it('becomes the stream’s query parameters', () => {
    expect(denoiseParams(undefined)).toEqual({})
    expect(denoiseParams({ mode: 'default', strength: 1, model: '' })).toEqual({})
    expect(denoiseParams({ mode: 'off', strength: 1, model: '' })).toEqual({ denoise: 'false' })
    expect(denoiseParams({ mode: 'on', strength: 1, model: '' })).toEqual({ denoise: 'true' })
    expect(denoiseParams({ mode: 'on', strength: 0.5, model: 'dpdfnet2' })).toEqual({
      denoise: 'dpdfnet2',
      denoise_strength: 0.5,
    })
  })
})

describe('turn detection of a session', () => {
  const td = (u: ReturnType<typeof sessionUpdate>) =>
    (u.session.audio as { input: { turn_detection: Record<string, unknown> } }).input.turn_detection
  it('names the detector and the end-of-turn model only when chosen', () => {
    expect(td(sessionUpdate('cascade'))).toEqual({ type: 'server_vad' })
    expect(
      td(sessionUpdate('cascade', undefined, 'websocket', undefined, { type: 'server_vad', vadModel: 'ten-vad' })),
    ).toEqual({
      type: 'server_vad',
      vad_model: 'ten-vad',
    })
    // an end-of-turn model goes only with semantic_vad
    expect(
      td(sessionUpdate('cascade', undefined, 'webrtc', undefined, { type: 'server_vad', turnModel: 'namo-turn-en' })),
    ).toEqual({ type: 'server_vad' })
    expect(
      td(sessionUpdate('cascade', undefined, 'webrtc', undefined, { type: 'semantic_vad', turnModel: 'namo-turn-en' })),
    ).toEqual({ type: 'semantic_vad', turn_model: 'namo-turn-en' })
  })
})

describe('model labels', () => {
  it('lists gates, licences and remote code', () => {
    expect(licenceTags(model({}))).toEqual([])
    expect(
      licenceTags(
        model({
          gated: true,
          commercial_use: false,
          license_note: 'register first',
          remote_code: { revision: 'a', files: ['x.py'], reviewed: null, allowed: false, notice: 'n' },
        }),
      ),
    ).toEqual(['gated', 'non-commercial', 'licence note', 'remote code, not allowed'])
    expect(
      licenceTags(model({ remote_code: { revision: 'a', files: [], reviewed: null, allowed: true, notice: null } })),
    ).toEqual(['remote code, allowed'])
  })
  it('knows text end-of-turn models, full-duplex models and whisper.cpp', () => {
    expect(readsText(model({ task: 'turn', id: 'x', features: ['reads-text'] }))).toBe(true)
    expect(readsText(model({ task: 'turn', id: 'namo-turn-en' }))).toBe(true)
    expect(readsText(model({ task: 'turn', id: 'smart-turn-v3' }))).toBe(false)
    expect(isFullDuplex(model({ features: ['speech-to-speech', 'full-duplex'] }))).toBe(true)
    expect(runtimeLabel(model({ engine: 'whisper.cpp' }))).toBe('whisper.cpp')
  })
  it('says when the engine is missing and how to install it', () => {
    const info = {
      engines: [
        {
          id: 'whisper.cpp',
          version: null,
          available: false,
          reason:
            "ModuleNotFoundError: No module named 'pywhispercpp'. Optional: pip install 'bolo-speech[whispercpp]'",
        },
        { id: 'sherpa-onnx', version: '2', available: true, reason: null },
      ],
    } as ServerInfo
    expect(engineMissing(info, model({ engine: 'whisper.cpp' }))?.install).toBe("pip install 'bolo-speech[whispercpp]'")
    expect(engineMissing(info, model({}))).toBeNull()
  })
  it('gives a full-duplex model’s own licence on the speech-to-speech pipeline', () => {
    const info = {
      capabilities: {
        realtime: {
          pipelines: {
            'speech-to-speech': {
              configured: true,
              available: false,
              model: 'moshiko-pytorch-q8',
              full_duplex: true,
              license: 'CC-BY-4.0',
            },
          },
        },
      },
    } as unknown as ServerInfo
    expect(pipelineOptions(info)[2]?.note).toBe('Licence: CC-BY-4.0 · full duplex')
  })
})
