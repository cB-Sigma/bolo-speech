import { describe, expect, it } from 'vitest'
import { detectBrowser, detectPlatform, verdictFor, type DeviceInfo } from './device'

const desktop: DeviceInfo = {
  platform: 'desktop',
  browser: 'Chrome',
  memoryGb: 8,
  cores: 8,
  webgpu: { available: true, f16: true, maxBufferSize: 4e9, maxStorageBufferBindingSize: 2e9, adapter: 'test' },
  threads: true,
  storage: { usage: 0, quota: 50e9, persisted: false },
}
const iphone: DeviceInfo = {
  ...desktop,
  platform: 'ios',
  browser: 'Safari',
  memoryGb: null,
  webgpu: { available: false, f16: false, maxBufferSize: null, maxStorageBufferBindingSize: null, adapter: null },
}

describe('device probe', () => {
  it('tells iPhone, iPad, Android and computers apart, and names the browser', () => {
    const nav = (userAgent: string, maxTouchPoints = 0) => detectPlatform({ userAgent, maxTouchPoints })
    expect(nav('Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15')).toBe('ios')
    expect(nav('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15', 5)).toBe(
      'ipados',
    )
    expect(nav('Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/141.0 Mobile Safari/537.36')).toBe(
      'android',
    )
    expect(nav('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/141.0 Safari/537.36')).toBe('desktop')
    expect(detectBrowser('Mozilla/5.0 (iPhone) AppleWebKit/605.1.15 CriOS/141.0 Mobile/15E148 Safari/604.1')).toBe(
      'Chrome (iOS)',
    )
    expect(detectBrowser('Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0')).toBe('Firefox')
  })
})

describe('verdicts warn, they never block', () => {
  it('a small model fits anywhere', () => {
    expect(verdictFor({ bytes: 40e6, peakBytes: 200e6 }, iphone).level).toBe('ok')
  })
  it('a WebGPU-only model is unlikely to work without WebGPU or without 16-bit shaders', () => {
    const needs = { bytes: 567e6, peakBytes: 1e9, requiresWebGpu: true, needsF16: true }
    expect(verdictFor(needs, iphone)).toMatchObject({ level: 'unlikely' })
    expect(verdictFor(needs, iphone).reasons.join()).toMatch(/needs WebGPU/)
    const noF16 = { ...desktop, webgpu: { ...desktop.webgpu!, f16: false } }
    expect(verdictFor(needs, noF16).reasons.join()).toMatch(/shader-f16/)
    expect(verdictFor(needs, desktop).level).toBe('ok')
  })
  it('a model meant for WebGPU is heavy on the CPU', () => {
    const v = verdictFor({ bytes: 253e6, peakBytes: 680e6, prefersWebGpu: true }, { ...desktop, webgpu: iphone.webgpu })
    expect(v.level).toBe('heavy')
    expect(v.reasons.join()).toMatch(/runs on the CPU/)
  })
  it('memory: about 1.4 GB on an iPhone is too much, with the reason in plain words', () => {
    const v = verdictFor({ bytes: 763e6, peakBytes: 1.65e9 }, iphone)
    expect(v.level).toBe('heavy')
    expect(v.reasons.join()).toMatch(/about 1\.6 GB in memory; iPhones and iPads usually close a tab/)
    expect(verdictFor({ bytes: 1e9, peakBytes: 3e9 }, iphone).level).toBe('unlikely')
    const low: DeviceInfo = { ...desktop, platform: 'android', memoryGb: 3 }
    expect(verdictFor({ bytes: 300e6, peakBytes: 900e6 }, low).reasons.join()).toMatch(/reports 3 GB/)
  })
  it('storage: a download larger than what is left says so, unless it is already downloaded', () => {
    const small: DeviceInfo = { ...desktop, storage: { usage: 300e6, quota: 1.2e9, persisted: false } }
    const v = verdictFor({ bytes: 1.1e9, peakBytes: 2e9 }, small)
    expect(v.reasons.join()).toMatch(/needs 1\.1 GB, 900 MB free/)
    expect(verdictFor({ bytes: 1.1e9, peakBytes: 2e9, downloaded: true }, small).reasons.join()).not.toMatch(/free/)
  })
})
