/** Browser capability detection for the features the console depends on. */

export interface BrowserSupport {
  secureContext: boolean
  getUserMedia: boolean
  audioWorklet: boolean
  webAudio: boolean
  webSocket: boolean
  wasm: boolean
  wasmSimd: boolean
  wasmThreads: boolean
  crossOriginIsolated: boolean
  webgpu: boolean
  hardwareConcurrency: number
}

// (module (func (result v128) i32.const 0 i8x16.splat)) — validates only with SIMD.
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
])

function safely(fn: () => boolean): boolean {
  try {
    return fn()
  } catch {
    return false
  }
}

export function detectSupport(): BrowserSupport {
  const g = globalThis as typeof globalThis & { crossOriginIsolated?: boolean }
  const nav = typeof navigator === 'undefined' ? undefined : navigator
  const wasm = typeof WebAssembly === 'object' && typeof WebAssembly.validate === 'function'
  const wasmSimd = wasm && safely(() => WebAssembly.validate(SIMD_PROBE))
  const isolated = Boolean(g.crossOriginIsolated)
  return {
    secureContext: Boolean(g.isSecureContext),
    getUserMedia: Boolean(nav?.mediaDevices?.getUserMedia),
    audioWorklet: typeof AudioWorkletNode !== 'undefined',
    webAudio: typeof AudioContext !== 'undefined',
    webSocket: typeof WebSocket !== 'undefined',
    wasm,
    wasmSimd,
    // Wasm threads need SharedArrayBuffer, which browsers only expose to cross-origin-isolated pages.
    wasmThreads: wasm && isolated && typeof SharedArrayBuffer !== 'undefined',
    crossOriginIsolated: isolated,
    webgpu: Boolean(nav && 'gpu' in nav),
    hardwareConcurrency: nav?.hardwareConcurrency ?? 1,
  }
}

/** True when a WebGPU adapter can actually be obtained (the API existing is not enough). */
export async function probeWebGpu(): Promise<boolean> {
  try {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu
    if (!gpu) return false
    return (await gpu.requestAdapter()) != null
  } catch {
    return false
  }
}

export type MicPermission = 'granted' | 'denied' | 'prompt' | 'unknown'

export async function microphonePermission(): Promise<MicPermission> {
  try {
    const status = await navigator.permissions.query({ name: 'microphone' as PermissionName })
    return status.state
  } catch {
    return 'unknown'
  }
}
