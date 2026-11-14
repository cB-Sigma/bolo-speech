import { beforeEach, describe, expect, it } from 'vitest'
import { browserModel, placeableStt } from '@/lib/browser-stt/models'
import { quickDevice, type DeviceInfo } from '@/lib/device'
import {
  bytesByModule,
  gpuShare,
  placementOptions,
  rememberModuleBytes,
  rememberSpeed,
  reloadPlacements,
  resolvePlacement,
} from '@/lib/placement'

const gpu = (over: Partial<NonNullable<DeviceInfo['webgpu']>> = {}): DeviceInfo => ({
  ...quickDevice(),
  platform: 'desktop',
  memoryGb: 8,
  webgpu: {
    available: true,
    f16: true,
    maxBufferSize: 2 ** 31,
    maxStorageBufferBindingSize: 2 ** 30,
    adapter: 'nvidia ampere',
    ...over,
  },
})
const noGpu: DeviceInfo = {
  ...quickDevice(),
  webgpu: { available: false, f16: false, maxBufferSize: null, maxStorageBufferBindingSize: null, adapter: null },
}

describe('GPU/CPU placement in the browser', () => {
  beforeEach(() => {
    localStorage.clear()
    reloadPlacements()
  })

  it('splits an encoder-decoder by graph and offers split only where there is more than one graph', () => {
    const whisper = placeableStt(browserModel('whisper-base-q8'))
    expect(whisper.modules).toEqual(['encoder_model', 'decoder_model_merged'])
    expect(placementOptions(whisper)).toContain('split')
    expect(resolvePlacement(whisper, 'split', gpu()).devices).toEqual({
      encoder_model: 'webgpu',
      decoder_model_merged: 'wasm',
    })
    const single = { ...whisper, modules: ['model'] }
    expect(placementOptions(single)).not.toContain('split')
  })

  it('Auto: CPU without WebGPU, WebGPU with it, split when a graph is larger than the largest buffer', () => {
    const whisper = placeableStt(browserModel('whisper-base-q8'))
    expect(resolvePlacement(whisper, 'auto', noGpu).placement).toBe('wasm')
    expect(resolvePlacement(whisper, 'auto', gpu()).placement).toBe('webgpu')
    rememberModuleBytes(whisper.id, { encoder_model: 30e6, decoder_model_merged: 160e6 })
    const small = resolvePlacement(whisper, 'auto', gpu({ maxStorageBufferBindingSize: 128 * 2 ** 20 }))
    expect(small.placement).toBe('split')
    expect(small.why).toMatch(/decoder_model_merged .* larger than this GPU's largest buffer/)
  })

  it('Auto follows the speeds measured on this device', () => {
    const whisper = placeableStt(browserModel('whisper-base-q8'))
    rememberSpeed(whisper.id, 'webgpu', 0.4)
    rememberSpeed(whisper.id, 'split', 0.2)
    const r = resolvePlacement(whisper, 'auto', gpu())
    expect(r.placement).toBe('split')
    expect(r.why).toMatch(/fastest measured/)
  })

  it('Auto keeps a heavy model on the CPU of a phone, whose memory the GPU shares', () => {
    const big = { ...placeableStt(browserModel('whisper-small')), peakBytes: 1e9 }
    const phone = { ...gpu({ adapter: 'apple' }), platform: 'ios' as const }
    expect(resolvePlacement(big, 'auto', phone).placement).toBe('wasm')
  })

  it('counts bytes per graph from the downloaded files, and the GPU share from them', () => {
    const files = [
      { file: 'onnx/encoder_model_quantized.onnx', total: 25e6 },
      { file: 'onnx/decoder_model_merged_quantized.onnx', total: 75e6 },
      { file: 'tokenizer.json', total: 2e6 },
    ]
    const bytes = bytesByModule(files, ['encoder_model', 'decoder_model_merged'])
    expect(bytes).toEqual({ encoder_model: 25e6, decoder_model_merged: 75e6 })
    expect(gpuShare({ encoder_model: 'webgpu', decoder_model_merged: 'wasm' }, bytes)).toBeCloseTo(0.25)
    expect(gpuShare({ model: 'webgpu' }, undefined)).toBe(1)
  })
})
