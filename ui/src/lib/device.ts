import { isPhoneLike, isWebKit } from '@/lib/browser-runtime'
import { storageUse } from '@/lib/downloads'
import { Store } from '@/lib/store'
import { detectSupport } from '@/lib/support'

/**
 * What this device and browser offer the in-page models, and a verdict per
 * model: fine, heavy (may be slow, may close the tab), or not expected to
 * work here. Verdicts only warn: every model stays selectable.
 */

export type Platform = 'ios' | 'ipados' | 'android' | 'desktop'

export interface DeviceInfo {
  platform: Platform
  browser: string
  /** navigator.deviceMemory in GB (Chromium only; capped at 8 by the browser). */
  memoryGb: number | null
  cores: number
  webgpu: {
    available: boolean
    f16: boolean
    maxBufferSize: number | null
    maxStorageBufferBindingSize: number | null
    adapter: string | null
  } | null
  /** SharedArrayBuffer, i.e. WebAssembly threads. */
  threads: boolean
  storage: { usage: number | null; quota: number | null; persisted: boolean | null }
}

type Nav = Pick<Navigator, 'userAgent' | 'maxTouchPoints' | 'hardwareConcurrency'> & { deviceMemory?: number }

export function detectPlatform(nav: Pick<Navigator, 'userAgent' | 'maxTouchPoints'>): Platform {
  const ua = nav.userAgent
  if (/iPhone|iPod/.test(ua)) return 'ios'
  if (/iPad/.test(ua) || (/Macintosh/.test(ua) && nav.maxTouchPoints > 1)) return 'ipados'
  if (/Android/.test(ua)) return 'android'
  return 'desktop'
}

export function detectBrowser(ua: string): string {
  if (/EdgiOS|Edg\//.test(ua)) return 'Edge'
  if (/CriOS/.test(ua)) return 'Chrome (iOS)'
  if (/FxiOS/.test(ua)) return 'Firefox (iOS)'
  if (/SamsungBrowser/.test(ua)) return 'Samsung Internet'
  if (/OPR\//.test(ua)) return 'Opera'
  if (/Firefox\//.test(ua)) return 'Firefox'
  if (/Chrome\//.test(ua)) return 'Chrome'
  if (/Safari\//.test(ua)) return 'Safari'
  return 'Unknown browser'
}

/** Synchronous part of the probe (no WebGPU adapter, no storage figures yet). */
export function quickDevice(nav: Nav | undefined = globalThis.navigator): DeviceInfo {
  const support = detectSupport()
  return {
    platform: nav ? detectPlatform(nav) : 'desktop',
    browser: nav ? detectBrowser(nav.userAgent) : 'Unknown browser',
    memoryGb: typeof nav?.deviceMemory === 'number' ? nav.deviceMemory : null,
    cores: nav?.hardwareConcurrency ?? 1,
    webgpu: null,
    threads: support.wasmThreads,
    storage: { usage: null, quota: null, persisted: null },
  }
}

interface GpuAdapterLike {
  features: { has(name: string): boolean }
  limits: { maxBufferSize?: number; maxStorageBufferBindingSize?: number }
  info?: { vendor?: string; architecture?: string; description?: string }
}

/** The whole probe: asks for a WebGPU adapter and the storage figures. */
export async function probeDevice(nav: Nav | undefined = globalThis.navigator): Promise<DeviceInfo> {
  const info = quickDevice(nav)
  try {
    const gpu = (nav as (Nav & { gpu?: { requestAdapter(): Promise<GpuAdapterLike | null> } }) | undefined)?.gpu
    if (gpu) {
      const adapter = await gpu.requestAdapter()
      info.webgpu = adapter
        ? {
            available: true,
            f16: adapter.features.has('shader-f16'),
            maxBufferSize: adapter.limits.maxBufferSize ?? null,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize ?? null,
            adapter:
              [adapter.info?.vendor, adapter.info?.architecture || adapter.info?.description]
                .filter(Boolean)
                .join(' ') || null,
          }
        : { available: false, f16: false, maxBufferSize: null, maxStorageBufferBindingSize: null, adapter: null }
    } else {
      info.webgpu = {
        available: false,
        f16: false,
        maxBufferSize: null,
        maxStorageBufferBindingSize: null,
        adapter: null,
      }
    }
  } catch {
    info.webgpu = {
      available: false,
      f16: false,
      maxBufferSize: null,
      maxStorageBufferBindingSize: null,
      adapter: null,
    }
  }
  info.storage = await storageUse()
  return info
}

/** Test and debugging override, read from local storage (`bolo.fakeDevice`): any DeviceInfo fields. */
function fakeDevice(): Partial<DeviceInfo> | null {
  try {
    const raw = globalThis.localStorage?.getItem('bolo.fakeDevice')
    return raw ? (JSON.parse(raw) as Partial<DeviceInfo>) : null
  } catch {
    return null
  }
}

export const device = new Store<DeviceInfo>({ ...quickDevice(), ...fakeDevice() })
let probing: Promise<void> | null = null

/** Run the full probe once (WebGPU adapter, storage); the store updates when it is done. */
export function ensureDeviceProbe(): Promise<void> {
  probing ??= probeDevice()
    .then((d) => device.set({ ...d, ...fakeDevice() }))
    .catch(() => undefined)
  return probing
}

/** Refresh the storage figures (after downloads change). */
export async function refreshDeviceStorage(): Promise<void> {
  const fake = fakeDevice()
  device.patch({ storage: fake?.storage ?? (await storageUse()) })
}

/** What a model needs, for the verdict. */
export interface ModelNeeds {
  /** Bytes downloaded the first time. */
  bytes: number
  /** Estimated peak memory while it runs, bytes. */
  peakBytes: number
  /** Runs only with WebGPU (no WebAssembly path). */
  requiresWebGpu?: boolean
  /** Runs on WebAssembly but is meant for WebGPU: on the CPU it is this many times slower than real time, or worse. */
  prefersWebGpu?: boolean
  /** WebGPU path needs 16-bit float shaders. */
  needsF16?: boolean
  /** Already in this browser's storage. */
  downloaded?: boolean
}

export type VerdictLevel = 'ok' | 'heavy' | 'unlikely'

export interface Verdict {
  level: VerdictLevel
  /** Short reasons, plain language. */
  reasons: string[]
}

const GB = 1e9
const fmt = (b: number) => (b >= GB ? `${(b / GB).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`)

/** Rough memory a single tab may use on this device before the browser closes it. */
export function tabMemoryBudget(d: DeviceInfo): { bytes: number; why: string } {
  if (d.platform === 'ios' || d.platform === 'ipados')
    return { bytes: 1.4 * GB, why: 'iPhones and iPads usually close a tab above about 1 to 1.5 GB' }
  if (d.platform === 'android') {
    const gb = d.memoryGb ?? 4
    return {
      bytes: Math.min(gb * 0.35, 2.5) * GB,
      why: `phones close a tab well before their memory is full${d.memoryGb ? ` (this one reports ${gb} GB)` : ''}`,
    }
  }
  if (d.memoryGb != null && d.memoryGb <= 4)
    return { bytes: d.memoryGb * 0.45 * GB, why: `this device reports ${d.memoryGb} GB of memory` }
  return { bytes: 3.5 * GB, why: 'a tab has about 4 GB to work with' }
}

/** How a model is expected to fare here. Never a reason to refuse: the visitor decides. */
export function verdictFor(needs: ModelNeeds, d: DeviceInfo): Verdict {
  const reasons: string[] = []
  let level: VerdictLevel = 'ok'
  const raise = (to: VerdictLevel) => {
    if (to === 'unlikely' || (to === 'heavy' && level === 'ok')) level = to
  }
  const gpu = d.webgpu
  if (needs.requiresWebGpu) {
    if (gpu && !gpu.available) {
      raise('unlikely')
      reasons.push('needs WebGPU, which this browser does not offer')
    } else if (gpu && needs.needsF16 && !gpu.f16) {
      raise('unlikely')
      reasons.push('needs 16-bit WebGPU shaders (shader-f16), which this GPU does not offer')
    }
  } else if (needs.prefersWebGpu && gpu && !gpu.available) {
    raise('heavy')
    reasons.push('no WebGPU here: it runs on the CPU, expect it to be several times slower than real time')
  }
  const budget = tabMemoryBudget(d)
  if (needs.peakBytes > budget.bytes) {
    raise(needs.peakBytes > budget.bytes * 1.6 ? 'unlikely' : 'heavy')
    reasons.push(`about ${fmt(needs.peakBytes)} in memory; ${budget.why}`)
  } else if (needs.peakBytes > budget.bytes * 0.6) {
    raise('heavy')
    reasons.push(`about ${fmt(needs.peakBytes)} in memory, near this device's limit (${budget.why})`)
  }
  if (gpu?.available && needs.requiresWebGpu && gpu.maxBufferSize != null && needs.bytes > gpu.maxBufferSize * 4) {
    raise('heavy')
    reasons.push(`large for this GPU (largest buffer ${fmt(gpu.maxBufferSize)})`)
  }
  const { usage, quota } = d.storage
  if (!needs.downloaded && quota != null) {
    const free = Math.max(0, quota - (usage ?? 0))
    if (needs.bytes > free) {
      raise('heavy')
      reasons.push(`needs ${fmt(needs.bytes)}, ${fmt(free)} free in this browser's storage`)
    }
  }
  return { level, reasons }
}

export const VERDICT_TEXT: Record<VerdictLevel, string> = {
  ok: 'Fits this device',
  heavy: 'Heavy for this device',
  unlikely: 'Unlikely to work here',
}

/** Measured after a model loaded on this device: refines later warnings. */
export interface Measured {
  loadMs: number
  /** Page memory after loading (precise where the browser offers it, else the JS heap), bytes. */
  memoryBytes: number | null
  at: number
}

const MEASURED_KEY = 'bolo.measured'

export function readMeasured(): Record<string, Measured> {
  try {
    return JSON.parse(globalThis.localStorage?.getItem(MEASURED_KEY) ?? '{}') as Record<string, Measured>
  } catch {
    return {}
  }
}

export function rememberMeasured(id: string, m: Omit<Measured, 'at'>): void {
  try {
    const all = readMeasured()
    all[id] = { ...m, at: Date.now() }
    globalThis.localStorage?.setItem(MEASURED_KEY, JSON.stringify(all))
  } catch {
    // ignore
  }
}

/** The page's memory now: precise where offered (Chromium, isolated), else the JS heap; null when unknown. */
export async function pageMemoryBytes(): Promise<number | null> {
  const perf = globalThis.performance as Performance & {
    measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }>
    memory?: { usedJSHeapSize: number }
  }
  try {
    if (perf.measureUserAgentSpecificMemory && globalThis.crossOriginIsolated)
      return (await perf.measureUserAgentSpecificMemory()).bytes
  } catch {
    // not offered
  }
  return perf.memory?.usedJSHeapSize ?? null
}

/** Peak estimate for a model: a measurement from this device when there is one, else the catalogue's. */
export function peakFor(id: string, estimate: number): number {
  const m = readMeasured()[id]
  return m?.memoryBytes != null && m.memoryBytes > estimate * 0.3 ? Math.max(m.memoryBytes, estimate * 0.5) : estimate
}

export function platformLabel(p: Platform): string {
  return { ios: 'iPhone (iOS)', ipados: 'iPad (iPadOS)', android: 'Android', desktop: 'Computer' }[p]
}

export { isPhoneLike, isWebKit }
