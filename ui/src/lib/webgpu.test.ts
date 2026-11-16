import { afterEach, describe, expect, it, vi } from 'vitest'
import { probeDevice } from './device'
import {
  adapterDetails,
  adapterLabel,
  GPU_POWER_KEY,
  readGpuPower,
  requestGpuAdapter,
  setOrtGpuPower,
  writeGpuPower,
} from './webgpu'

const nav = (adapter: unknown) => {
  const requestAdapter = vi.fn(async () => adapter)
  return {
    nav: {
      userAgent: 'Mozilla/5.0 (Windows NT 10.0) Chrome/157.0 Safari/537.36',
      maxTouchPoints: 0,
      hardwareConcurrency: 16,
      gpu: { requestAdapter },
    },
    requestAdapter,
  }
}

const radeon = {
  features: new Set(['shader-f16']),
  limits: { maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 30 },
  info: { vendor: 'amd', architecture: 'gcn-5', description: 'AMD Radeon(TM) Graphics', type: 'integrated GPU' },
}
const rtx = {
  ...radeon,
  info: {
    vendor: 'nvidia',
    architecture: 'ampere',
    description: 'NVIDIA GeForce RTX 3060 Laptop GPU',
    type: 'discrete GPU',
  },
}

afterEach(() => localStorage.removeItem(GPU_POWER_KEY))

describe('one way to ask for a WebGPU adapter', () => {
  it('asks for the high-performance adapter unless the visitor chose power saving', async () => {
    const { nav: n, requestAdapter } = nav(rtx)
    expect(readGpuPower()).toBe('high-performance')
    await requestGpuAdapter(n.gpu)
    expect(requestAdapter).toHaveBeenLastCalledWith({ powerPreference: 'high-performance' })
    writeGpuPower('low-power')
    expect(localStorage.getItem(GPU_POWER_KEY)).toBe('low-power')
    await requestGpuAdapter(n.gpu)
    expect(requestAdapter).toHaveBeenLastCalledWith({ powerPreference: 'low-power' })
    writeGpuPower('high-performance')
    expect(localStorage.getItem(GPU_POWER_KEY)).toBeNull()
    expect(await requestGpuAdapter(undefined)).toBeNull()
  })

  it('sets ONNX Runtime’s power preference (onnxruntime-web and transformers.js)', () => {
    const env: { webgpu?: { powerPreference?: string } } = { webgpu: {} }
    setOrtGpuPower(env as Parameters<typeof setOrtGpuPower>[0])
    expect(env.webgpu?.powerPreference).toBe('high-performance')
    setOrtGpuPower(env as Parameters<typeof setOrtGpuPower>[0], 'low-power')
    expect(env.webgpu?.powerPreference).toBe('low-power')
    expect(() => setOrtGpuPower(undefined)).not.toThrow()
  })

  it('the device probe passes the option and keeps what the browser says about the adapter', async () => {
    const { nav: n, requestAdapter } = nav(rtx)
    const d = await probeDevice(n)
    expect(requestAdapter).toHaveBeenCalledWith({ powerPreference: 'high-performance' })
    expect(d.webgpu?.adapter).toBe('nvidia · ampere · NVIDIA GeForce RTX 3060 Laptop GPU')
    expect(d.webgpu?.details).toMatchObject({ vendor: 'nvidia', integrated: false, type: 'discrete GPU' })
    expect(d.webgpu?.power).toBe('high-performance')
    const low = await probeDevice(nav(radeon).nav, 'low-power')
    expect(low.webgpu?.details?.integrated).toBe(true)
    expect(low.webgpu?.power).toBe('low-power')
  })
})

describe('integrated or not', () => {
  it('trusts the adapter type where the browser gives one', () => {
    expect(adapterDetails(radeon.info).integrated).toBe(true)
    expect(adapterDetails(rtx.info).integrated).toBe(false)
  })
  it('falls back on the description and Intel architectures', () => {
    expect(adapterDetails({ vendor: 'amd', description: 'AMD Radeon(TM) Graphics' }).integrated).toBe(true)
    expect(adapterDetails({ vendor: 'amd', description: 'AMD Radeon 780M Graphics' }).integrated).toBe(true)
    expect(adapterDetails({ vendor: 'intel', architecture: 'gen-12lp' }).integrated).toBe(true)
    expect(adapterDetails({ vendor: 'intel', architecture: 'xe-hpg' }).integrated).toBe(false) // Arc A-series card
    expect(adapterDetails({ vendor: 'amd', description: 'AMD Radeon RX 6800M' }).integrated).toBe(false)
    expect(adapterDetails({ vendor: 'nvidia', architecture: 'ampere' }).integrated).toBe(false)
    expect(adapterDetails(undefined)).toMatchObject({ vendor: null, integrated: false })
  })
  it('labels the adapter with what is known', () => {
    expect(adapterLabel(adapterDetails({ vendor: 'amd', architecture: 'gcn-5', description: '' }))).toBe('amd · gcn-5')
    expect(adapterLabel(adapterDetails({}))).toBeNull()
  })
})
