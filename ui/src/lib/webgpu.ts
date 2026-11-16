/**
 * Which GPU the in-page models run on. Every WebGPU adapter the page asks for, and every WebGPU session ONNX
 * Runtime (directly or through transformers.js) creates, goes through here, so they all ask for the same one.
 *
 * Without options `navigator.gpu.requestAdapter()` gives the default adapter, which on a laptop with two GPUs
 * is the power-saving, integrated one: Chrome and Edge on a Windows laptop with a Radeon or Intel GPU in the
 * processor and an NVIDIA card would run every model on the integrated GPU. bolo asks for
 * `powerPreference: 'high-performance'` unless the visitor chose power saving. Browsers may ignore the hint
 * (Windows then decides per application), so the device panel says which adapter came back.
 */

export type GpuPower = 'high-performance' | 'low-power'

export const GPU_POWER_KEY = 'bolo.gpuPower'
export const DEFAULT_GPU_POWER: GpuPower = 'high-performance'

/** This browser's choice (stored per browser); high performance unless the visitor chose power saving. */
export function readGpuPower(): GpuPower {
  try {
    return globalThis.localStorage?.getItem(GPU_POWER_KEY) === 'low-power' ? 'low-power' : DEFAULT_GPU_POWER
  } catch {
    return DEFAULT_GPU_POWER
  }
}

export function writeGpuPower(power: GpuPower): void {
  try {
    if (power === DEFAULT_GPU_POWER) globalThis.localStorage?.removeItem(GPU_POWER_KEY)
    else globalThis.localStorage?.setItem(GPU_POWER_KEY, power)
  } catch {
    // storage blocked: the choice lasts for this page only
  }
}

/** The options for `navigator.gpu.requestAdapter()`. */
export function adapterOptions(power: GpuPower = readGpuPower()): { powerPreference: GpuPower } {
  return { powerPreference: power }
}

interface GpuLike<A> {
  requestAdapter(options?: { powerPreference?: GpuPower }): Promise<A | null>
}

/** The `navigator.gpu` of this page or worker, or undefined without WebGPU. */
export function navigatorGpu<A = unknown>(nav: unknown = globalThis.navigator): GpuLike<A> | undefined {
  return (nav as { gpu?: GpuLike<A> } | undefined)?.gpu
}

/** A WebGPU adapter of the chosen kind (null when the browser has none). */
export async function requestGpuAdapter<A = unknown>(
  gpu: GpuLike<A> | undefined = navigatorGpu<A>(),
  power: GpuPower = readGpuPower(),
): Promise<A | null> {
  if (!gpu) return null
  return gpu.requestAdapter(adapterOptions(power))
}

/**
 * Tell ONNX Runtime which adapter to ask for. Pass `ort.env` (onnxruntime-web) or `env.backends.onnx`
 * (transformers.js); it takes effect when the first WebGPU session of that runtime is created.
 */
export function setOrtGpuPower(
  ortEnv: { webgpu?: { powerPreference?: GpuPower } } | undefined,
  power: GpuPower = readGpuPower(),
): void {
  if (ortEnv?.webgpu) ortEnv.webgpu.powerPreference = power
}

/** What the browser says about an adapter (GPUAdapterInfo; `type` only where Chromium exposes it). */
export interface AdapterInfoLike {
  vendor?: string
  architecture?: string
  device?: string
  description?: string
  type?: string
  isFallbackAdapter?: boolean
}

export interface AdapterDetails {
  vendor: string | null
  architecture: string | null
  description: string | null
  /** Chromium's adapter type ("discrete GPU", "integrated GPU", ...), where exposed. */
  type: string | null
  /** The browser's software fallback (no GPU). */
  fallback: boolean
  /** It looks like integrated graphics (the adapter type or description says so, or the name is an iGPU's). */
  integrated: boolean
}

const INTEGRATED_NAME =
  /\bintegrated\b|\b(uhd|iris|hd) graphics\b|\bradeon(\(tm\))? graphics\b|\bradeon( vega)? \d{1,3}m? graphics\b/i

export function adapterDetails(info: AdapterInfoLike | undefined, fallback = false): AdapterDetails {
  const clean = (s: string | undefined) => (s && s.trim() ? s.trim() : null)
  const vendor = clean(info?.vendor)
  const architecture = clean(info?.architecture)
  const description = clean(info?.description)
  const type = clean(info?.type)
  const integrated = type
    ? /integrated/i.test(type)
    : INTEGRATED_NAME.test(description ?? '') ||
      // Intel GPUs other than the discrete Arc cards (Xe-HPG) are part of the processor
      (vendor === 'intel' && architecture != null && !/hpg/i.test(architecture))
  return { vendor, architecture, description, type, fallback, integrated }
}

/** "nvidia · ampere · NVIDIA GeForce RTX 3060 Laptop GPU", or null when the browser says nothing. */
export function adapterLabel(d: Pick<AdapterDetails, 'vendor' | 'architecture' | 'description'>): string | null {
  return [d.vendor, d.architecture, d.description].filter(Boolean).join(' · ') || null
}
