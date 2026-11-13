import { describe, expect, it } from 'vitest'
import { fmtBytes, fmtDuration, fmtMs, fmtParams, fmtPercent, fmtRtf, fmtSpeed, languageName } from './format'
import { phaseColor, phaseLabel } from './phases'
import { buildSnippets } from './snippets'
import { detectSupport } from './support'
import { parseWavHeader } from './tts/session'
import { encodeWav } from './audio/wav'

describe('formatting', () => {
  it('formats milliseconds, switching to seconds when long', () => {
    expect(fmtMs(249.6)).toBe('250 ms')
    expect(fmtMs(1234)).toBe('1,234 ms')
    expect(fmtMs(12345)).toBe('12.3 s')
    expect(fmtMs(null)).toBe('–')
    expect(fmtMs(80, false)).toBe('80')
  })
  it('formats sizes, parameters, rtf and speed', () => {
    expect(fmtBytes(123_456_789)).toBe('123 MB')
    expect(fmtBytes(2_410_000_000)).toBe('2.4 GB')
    expect(fmtBytes(900)).toBe('900 B')
    expect(fmtParams(27_000_000)).toBe('27M')
    expect(fmtParams(1_500_000_000)).toBe('1.5B')
    expect(fmtParams(null)).toBe('–')
    expect(fmtRtf(0.058)).toBe('0.058')
    expect(fmtRtf(0.42)).toBe('0.42')
    expect(fmtSpeed(0.05)).toBe('20×')
    expect(fmtSpeed(0.4)).toBe('2.5×')
    expect(fmtSpeed(0)).toBe('–')
    expect(fmtPercent(0.163)).toBe('16%')
    expect(fmtDuration(3725)).toBe('1h 2m')
  })
  it('names languages, including Indian ones, and falls back to the code', () => {
    expect(languageName('hi')).toBe('Hindi')
    expect(languageName('ta')).toBe('Tamil')
    expect(languageName('multi')).toBe('Multilingual')
    expect(languageName('')).toBe('Any')
    expect(languageName('zz-not-a-language')).toBeTruthy()
  })
})

describe('phases', () => {
  it('assigns slots in pipeline order per task and never invents a hue', () => {
    expect(phaseColor('stt', 'queue')).toBe('var(--chart-1)')
    expect(phaseColor('stt', 'decode')).toBe('var(--chart-4)')
    expect(phaseColor('tts', 'synthesize_first')).toBe('var(--chart-3)')
    expect(phaseColor('stt', 'something_new')).toBe('var(--chart-mute)')
    expect(phaseLabel('synthesize_rest')).toBe('Remaining audio')
    expect(phaseLabel('custom')).toBe('custom')
  })
})

describe('snippets', () => {
  const ctx = {
    baseUrl: 'http://10.0.0.5:8000',
    sttModel: 'moonshine-tiny-en',
    ttsModel: 'kokoro-82m',
    voice: 'af_heart',
    auth: false,
  }
  it('fills in the server address and models everywhere', () => {
    const all = buildSnippets(ctx)
    expect(all.map((s) => s.id)).toEqual(['curl', 'python', 'javascript', 'websocket', 'home-assistant'])
    const text = all.flatMap((s) => s.blocks.map((b) => b.code)).join('\n')
    expect(text).toContain('http://10.0.0.5:8000/v1/audio/transcriptions')
    expect(text).toContain('base_url="http://10.0.0.5:8000/v1"')
    expect(text).toContain('ws://10.0.0.5:8000/api/v1/stt/stream?model=moonshine-tiny-en')
    expect(text).toContain('"voice": "af_heart"')
    expect(text).toContain('Host: 10.0.0.5')
    expect(text).not.toContain('Authorization')
  })
  it('adds credentials only when the server needs them', () => {
    const text = buildSnippets({ ...ctx, auth: true, baseUrl: 'https://speech.example' })
      .flatMap((s) => s.blocks.map((b) => b.code))
      .join('\n')
    expect(text).toContain('Authorization: Bearer $BOLO_API_KEY')
    expect(text).toContain('wss://speech.example/api/v1/stt/stream')
    expect(text).toContain('api_key=$BOLO_API_KEY')
  })
})

describe('support detection', () => {
  it('reports capabilities without throwing in a bare environment', () => {
    const s = detectSupport()
    expect(typeof s.wasm).toBe('boolean')
    expect(s.wasmThreads).toBe(false) // jsdom is never cross-origin isolated
    expect(s.hardwareConcurrency).toBeGreaterThanOrEqual(1)
  })
})

describe('parseWavHeader', () => {
  it('waits for enough bytes, then finds the data offset', () => {
    const wav = encodeWav(new Float32Array(100), 22050)
    expect(parseWavHeader(wav.subarray(0, 20))).toBeNull()
    expect(parseWavHeader(wav)).toEqual({ sampleRate: 22050, channels: 1, dataOffset: 44 })
    expect(() => parseWavHeader(new Uint8Array(64))).toThrow()
  })
})
