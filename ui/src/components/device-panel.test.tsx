import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { DevicePanel } from '@/components/model-fit'
import { device, reprobeDevice } from '@/lib/device'
import { GPU_POWER_KEY } from '@/lib/webgpu'

// A Windows laptop with an integrated Radeon and an NVIDIA card: what the browser hands back depends on the
// power preference and, in the end, on Windows' per-application setting.
const adapter = (info: Record<string, string>) => ({
  features: new Set(['shader-f16']),
  limits: { maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 30 },
  info,
})
const RADEON = { vendor: 'amd', architecture: 'gcn-5', description: 'AMD Radeon(TM) Graphics', type: 'integrated GPU' }
const RTX = { vendor: 'nvidia', architecture: 'ampere', description: 'NVIDIA GeForce RTX 3060 Laptop GPU' }

function fakeGpu(pick: (power: string | undefined) => Record<string, string>) {
  const requestAdapter = vi.fn(async (opts?: { powerPreference?: string }) => adapter(pick(opts?.powerPreference)))
  Object.defineProperty(navigator, 'gpu', { value: { requestAdapter }, configurable: true })
  return requestAdapter
}

afterEach(() => {
  vi.restoreAllMocks()
  Reflect.deleteProperty(navigator, 'gpu')
  localStorage.removeItem(GPU_POWER_KEY)
})

describe('Your device: which GPU', () => {
  it('shows the adapter the browser gave', async () => {
    const requestAdapter = fakeGpu((p) => (p === 'high-performance' ? RTX : RADEON))
    await reprobeDevice()
    render(<DevicePanel />)
    const panel = screen.getByTestId('device-panel')
    expect(panel).toHaveTextContent('nvidia · ampere · NVIDIA GeForce RTX 3060 Laptop GPU')
    expect(panel).not.toHaveTextContent('integrated graphics')
    expect(screen.queryByTestId('gpu-integrated-warning')).toBeNull()
    expect(requestAdapter).toHaveBeenCalledWith({ powerPreference: 'high-performance' })
  })

  it('says so when the browser still gave the integrated GPU, with the Windows fix', async () => {
    fakeGpu(() => RADEON) // the browser ignored the hint
    vi.spyOn(navigator, 'userAgent', 'get').mockReturnValue(
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/157.0 Safari/537.36 Edg/152.0',
    )
    await reprobeDevice()
    render(<DevicePanel />)
    expect(screen.getByTestId('device-panel')).toHaveTextContent('AMD Radeon(TM) Graphics (integrated graphics)')
    expect(device.get().webgpu?.details?.integrated).toBe(true)
    const warning = screen.getByTestId('gpu-integrated-warning')
    expect(warning).toHaveTextContent('WebGPU got the integrated GPU')
    expect(warning).toHaveTextContent(
      'Windows picks the GPU for the browser: Settings > System > Display > Graphics > (browser) > High performance',
    )
    expect(warning).toHaveTextContent('NVIDIA Control Panel')
    // The page does not claim the request worked: on Windows the browser ignores it.
    expect(screen.getByTestId('gpu-choice-note')).toHaveTextContent('Chrome and Edge on Windows ignore the hint')
  })

  it('what Chrome on Windows really reports (no description): a Ryzen gcn-5 adapter reads as integrated', async () => {
    fakeGpu(() => ({ vendor: 'amd', architecture: 'gcn-5', description: '' }))
    vi.spyOn(navigator, 'userAgent', 'get').mockReturnValue(
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/157.0 Safari/537.36',
    )
    await reprobeDevice()
    render(<DevicePanel />)
    expect(screen.getByTestId('device-panel')).toHaveTextContent('GPU in use')
    expect(screen.getByTestId('device-panel')).toHaveTextContent('amd · gcn-5 (most likely integrated graphics)')
    expect(screen.getByTestId('gpu-integrated-warning')).toHaveTextContent('Windows picks the GPU for the browser')
  })

  it('lets the visitor choose power saving, kept per browser', async () => {
    const requestAdapter = fakeGpu((p) => (p === 'high-performance' ? RTX : RADEON))
    await reprobeDevice()
    render(<DevicePanel />)
    fireEvent.click(screen.getByTestId('gpu-low-power'))
    expect(localStorage.getItem(GPU_POWER_KEY)).toBe('low-power')
    await waitFor(() => expect(requestAdapter).toHaveBeenLastCalledWith({ powerPreference: 'low-power' }))
    await waitFor(() => expect(screen.getByTestId('device-panel')).toHaveTextContent('AMD Radeon(TM) Graphics'))
    // integrated on purpose: no warning
    expect(screen.queryByTestId('gpu-integrated-warning')).toBeNull()
  })
})

describe('Your device: "GPU in use" is always there', () => {
  it('says checking before the probe answers, then the adapter, under stable test ids', async () => {
    device.patch({ webgpu: null })
    render(<DevicePanel />)
    expect(screen.getByTestId('device-panel')).toHaveTextContent('Your device')
    expect(screen.getByTestId('gpu-in-use')).toHaveTextContent('GPU in use')
    expect(screen.getByTestId('gpu-in-use')).toHaveTextContent('checking…')
    fakeGpu(() => RTX)
    await reprobeDevice()
    await waitFor(() =>
      expect(screen.getByTestId('gpu-in-use')).toHaveTextContent('NVIDIA GeForce RTX 3060 Laptop GPU'),
    )
  })

  it('says none when the browser has no WebGPU adapter', async () => {
    await reprobeDevice()
    render(<DevicePanel />)
    expect(screen.getByTestId('gpu-in-use')).toHaveTextContent('GPU in usenone (no WebGPU adapter in this browser)')
  })
})

describe('Your device: a WebGPU probe that never answered', () => {
  it('says so instead of "checking…"', () => {
    device.patch({
      webgpu: {
        available: false,
        f16: false,
        maxBufferSize: null,
        maxStorageBufferBindingSize: null,
        adapter: null,
        noAnswer: true,
      },
    })
    render(<DevicePanel />)
    expect(screen.getByTestId('gpu-in-use')).toHaveTextContent('the browser did not answer within 5 s')
  })
})
