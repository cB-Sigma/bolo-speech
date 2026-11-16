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
    expect(warning).toHaveTextContent('integrated GPU although the page asked for the high-performance one')
    expect(warning).toHaveTextContent('Settings > System > Display > Graphics, choose this browser, set it to High')
    expect(warning).toHaveTextContent('NVIDIA Control Panel')
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
