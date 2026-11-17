import { describe, expect, it } from 'vitest'
import { parseTranscription, transcriptionForm } from './diarize'
import { applyEvent, emptyConversation, sessionUpdate } from './realtime/session'
import {
  applyDecoding,
  applyDenoise,
  decodingCount,
  defaultDenoise,
  emptyDecoding,
  realtimeDenoise,
} from './stt-options'

describe('decoding options', () => {
  it('sends only what was set, with vad_parameters as JSON', () => {
    const f = new FormData()
    applyDecoding(f, emptyDecoding())
    expect([...f.keys()]).toEqual([])
    const o = {
      conditionOnPreviousText: 'true' as const,
      vadFilter: '' as const,
      numbers: { repetition_penalty: '1.2', no_repeat_ngram_size: '3', chunk_length: ' ', max_repeats: 'x' },
      vad: { threshold: '0.6', speech_pad_ms: '200' },
    }
    expect(decodingCount(o)).toBe(6)
    applyDecoding(f, o)
    expect(f.get('condition_on_previous_text')).toBe('true')
    expect(f.get('repetition_penalty')).toBe('1.2')
    expect(f.get('no_repeat_ngram_size')).toBe('3')
    expect(f.has('chunk_length')).toBe(false)
    expect(f.has('max_repeats')).toBe(false)
    expect(f.has('vad_filter')).toBe(false)
    expect(JSON.parse(String(f.get('vad_parameters')))).toEqual({ threshold: 0.6, speech_pad_ms: 200 })
  })

  it('goes into the transcription form with or without speaker labels', () => {
    const decoding = { ...emptyDecoding(), numbers: { max_repeats: '0' } }
    const f = transcriptionForm(new Blob(['x']), 'a.wav', { model: 'whisper-base', diarize: false, decoding })
    expect(f.get('max_repeats')).toBe('0')
    expect(f.get('response_format')).toBe('verbose_json')
  })

  it('reads language_probability and the option headers', () => {
    const r = parseTranscription(
      { text: 'hi', language: 'en', language_probability: 0.31, segments: [] },
      new Headers({ 'x-bolo-ignored-options': 'repetition_penalty', 'x-bolo-denoise': 'gtcrn-simple; strength=0.5' }),
    )
    expect(r.languageProbability).toBe(0.31)
    expect(r.ignoredOptions).toBe('repetition_penalty')
    expect(r.denoiseHeader).toBe('gtcrn-simple; strength=0.5')
  })
})

describe('denoise', () => {
  it('maps to the transcription form', () => {
    const form = (d: Parameters<typeof applyDenoise>[1]) => {
      const f = new FormData()
      applyDenoise(f, d)
      return Object.fromEntries(f.entries())
    }
    expect(form(defaultDenoise())).toEqual({})
    expect(form({ mode: 'off', strength: 1, model: '' })).toEqual({ denoise: 'false' })
    expect(form({ mode: 'on', strength: 1, model: '' })).toEqual({ denoise: 'true' })
    expect(form({ mode: 'on', strength: 0.5, model: 'gtcrn-simple' })).toEqual({
      denoise: 'gtcrn-simple',
      denoise_strength: '0.5',
    })
  })

  it('maps to a Realtime session and reads what the server reports', () => {
    expect(realtimeDenoise(defaultDenoise())).toBeUndefined()
    expect(realtimeDenoise({ mode: 'off', strength: 1, model: '' })).toBe(false)
    expect(realtimeDenoise({ mode: 'on', strength: 0.7, model: '' })).toEqual({ strength: 0.7 })
    const u = sessionUpdate('cascade', undefined, 'websocket', { model: 'gtcrn-simple', strength: 1 })
    expect((u.session.audio as { input: { denoise: unknown } }).input.denoise).toEqual({
      model: 'gtcrn-simple',
      strength: 1,
    })
    expect('denoise' in (sessionUpdate('cascade').session.audio as { input: object }).input).toBe(false)
    const s = applyEvent(emptyConversation(), {
      type: 'session.updated',
      session: { audio: { input: { denoise: { enabled: true, model: 'gtcrn-simple', strength: 0.5 } } } },
    })
    expect(s.serverDenoise).toEqual({ enabled: true, model: 'gtcrn-simple', strength: 0.5 })
  })
})
