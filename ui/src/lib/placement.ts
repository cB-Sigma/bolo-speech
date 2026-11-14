import type { DeviceInfo } from '@/lib/device'
import { tabMemoryBudget } from '@/lib/device'
import { Store } from '@/lib/store'

/**
 * Where an in-page model runs: all on WebGPU, all on the CPU (WebAssembly),
 * or split, with some graphs on WebGPU and the rest on the CPU. "Auto" picks
 * one from the device probe (WebGPU present, its buffer limits, memory) and
 * from speeds measured on this device.
 *
 * On phones, tablets and Apple-silicon Macs the GPU uses the same memory as
 * everything else: a split there helps with WebGPU's limits and with speed,
 * never with total memory. The UI says so.
 */

export type PlacementChoice = 'auto' | 'webgpu' | 'wasm' | 'split'
export type Placement = 'webgpu' | 'wasm' | 'split'
export type Device = 'webgpu' | 'wasm'

export const PLACEMENT_TEXT: Record<PlacementChoice, string> = {
  auto: 'Auto',
  webgpu: 'All WebGPU',
  wasm: 'All CPU',
  split: 'Split',
}

/** What the placement logic needs to know about a model. */
export interface PlaceableModel {
  id: string
  /** Graphs it is made of, in the order they run (e.g. encoder_model, decoder_model_merged). */
  modules: string[]
  /** Bytes downloaded. */
  bytes: number
  peakBytes: number
  /** Only runs on WebGPU (16-bit weights). */
  requiresWebGpu?: boolean
  /** Faster on WebGPU as a rule (Auto starts there when nothing was measured). */
  gpuFirst: boolean
  /** One graph cut in two at build time: front on WebGPU, back on the CPU. */
  cutHalves?: boolean
  /** How the split is placed, in words. */
  splitText?: string
}

export interface Resolved {
  placement: Placement
  /** Device per module. */
  devices: Record<string, Device>
  /** Why Auto chose this (empty for an explicit choice). */
  why: string
}

interface Saved {
  choices: Record<string, PlacementChoice>
  /** Real-time factor measured here per model and placement (lower is faster). */
  speed: Record<string, Partial<Record<Placement, number>>>
  /** Bytes of each module, seen when the model was loaded. */
  moduleBytes: Record<string, Record<string, number>>
}

const KEY = 'bolo.placement'

function read(): Saved {
  try {
    const raw = globalThis.localStorage?.getItem(KEY)
    if (raw) {
      const v = JSON.parse(raw) as Partial<Saved>
      return { choices: v.choices ?? {}, speed: v.speed ?? {}, moduleBytes: v.moduleBytes ?? {} }
    }
  } catch {
    // unreadable
  }
  return { choices: {}, speed: {}, moduleBytes: {} }
}

export const placements = new Store<Saved>(read())

placements.subscribe(() => {
  try {
    globalThis.localStorage?.setItem(KEY, JSON.stringify(placements.get()))
  } catch {
    // blocked: this visit only
  }
})

export function reloadPlacements(): void {
  placements.set(read())
}

export function placementChoice(id: string): PlacementChoice {
  return placements.get().choices[id] ?? 'auto'
}

export function setPlacementChoice(id: string, choice: PlacementChoice): void {
  const s = placements.get()
  placements.patch({ choices: { ...s.choices, [id]: choice } })
}

/** Remember a speed (real-time factor) for a model on a placement: a running average. */
export function rememberSpeed(id: string, placement: Placement, rtf: number): void {
  if (!Number.isFinite(rtf) || rtf <= 0) return
  const s = placements.get()
  const prev = s.speed[id]?.[placement]
  const next = prev == null ? rtf : prev * 0.7 + rtf * 0.3
  placements.patch({ speed: { ...s.speed, [id]: { ...s.speed[id], [placement]: Math.round(next * 1000) / 1000 } } })
}

export function rememberModuleBytes(id: string, bytes: Record<string, number>): void {
  if (Object.keys(bytes).length === 0) return
  const s = placements.get()
  placements.patch({ moduleBytes: { ...s.moduleBytes, [id]: bytes } })
}

/** Which module a downloaded file belongs to (by name), or null. */
export function moduleOfFile(file: string, modules: string[]): string | null {
  const name = file.split('/').pop() ?? file
  // Longest name first: "decoder_model_merged" before "decoder_model".
  for (const m of [...modules].sort((a, b) => b.length - a.length)) if (name.startsWith(m)) return m
  return null
}

/** Bytes per module from the files a load fetched. */
export function bytesByModule(files: { file: string; total: number }[], modules: string[]): Record<string, number> {
  const out: Record<string, number> = {}
  for (const f of files) {
    if (!/\.onnx(_data)?$/.test(f.file) && !/\.onnx_data_\d+$/.test(f.file)) continue
    const m = moduleOfFile(f.file, modules) ?? (modules.length === 1 ? modules[0] : null)
    if (m) out[m] = (out[m] ?? 0) + f.total
  }
  return out
}

/** Share of the model's weights on WebGPU, 0..1, by bytes. */
export function gpuShare(devices: Record<string, Device>, moduleBytes: Record<string, number> | undefined): number {
  const mods = Object.keys(devices)
  if (mods.length === 0) return 0
  const known = moduleBytes && mods.every((m) => moduleBytes[m] != null)
  const size = (m: string) => (known ? (moduleBytes?.[m] ?? 0) : 1)
  const total = mods.reduce((a, m) => a + size(m), 0)
  if (total <= 0) return 0
  return mods.filter((m) => devices[m] === 'webgpu').reduce((a, m) => a + size(m), 0) / total
}

/** Placements this model can take. */
export function placementOptions(m: PlaceableModel): PlacementChoice[] {
  const out: PlacementChoice[] = ['auto', 'webgpu', 'wasm']
  if (m.modules.length > 1 || m.cutHalves) out.push('split')
  return out
}

function devicesFor(m: PlaceableModel, p: Placement): Record<string, Device> {
  if (p !== 'split') return Object.fromEntries(m.modules.map((x) => [x, p]))
  // The first graph (the encoder, or the front half) on WebGPU, the rest on the CPU.
  return Object.fromEntries(m.modules.map((x, i) => [x, i === 0 ? 'webgpu' : 'wasm']))
}

/** Largest single graph, from what was seen at load or a rough share of the download. */
function largestModule(m: PlaceableModel): { module: string; bytes: number } {
  const seen = placements.get().moduleBytes[m.id]
  if (seen && Object.keys(seen).length) {
    const [module, bytes] = Object.entries(seen).sort((a, b) => b[1] - a[1])[0] as [string, number]
    return { module, bytes }
  }
  // An encoder-decoder's merged decoder is usually the larger graph.
  return m.modules.length > 1
    ? { module: m.modules.at(-1) as string, bytes: m.bytes * 0.65 }
    : { module: m.modules[0] as string, bytes: m.bytes }
}

const MB = (b: number) => `${Math.round(b / 1e6)} MB`

/** The placement to use for a model with a choice, on a device. */
export function resolvePlacement(m: PlaceableModel, choice: PlacementChoice, d: DeviceInfo): Resolved {
  const gpu = d.webgpu
  const canSplit = m.modules.length > 1 || Boolean(m.cutHalves)
  if (choice !== 'auto') {
    const p: Placement = choice === 'split' && !canSplit ? 'webgpu' : choice
    return { placement: p, devices: devicesFor(m, p), why: '' }
  }
  const pick = (p: Placement, why: string): Resolved => ({ placement: p, devices: devicesFor(m, p), why })
  if (m.requiresWebGpu) return pick('webgpu', '16-bit weights run on WebGPU only')
  // Still probing (null) counts as "not known yet": fall back to the CPU, which always works.
  if (!gpu?.available) return pick('wasm', gpu ? 'no WebGPU in this browser' : 'WebGPU not checked yet')
  const speed = placements.get().speed[m.id] ?? {}
  const measured = (Object.entries(speed) as [Placement, number][]).filter(([p]) => p !== 'split' || canSplit)
  if (measured.length >= 2) {
    const [best] = measured.sort((a, b) => a[1] - b[1])[0] as [Placement, number]
    return pick(best, `fastest measured on this device (${PLACEMENT_TEXT[best].toLowerCase()})`)
  }
  const limit = Math.min(
    gpu.maxStorageBufferBindingSize ?? Number.POSITIVE_INFINITY,
    gpu.maxBufferSize ?? Number.POSITIVE_INFINITY,
  )
  const largest = largestModule(m)
  if (largest.bytes > limit) {
    return canSplit
      ? pick(
          'split',
          `${largest.module} (${MB(largest.bytes)}) is larger than this GPU's largest buffer (${MB(limit)})`,
        )
      : pick('wasm', `the model (${MB(largest.bytes)}) is larger than this GPU's largest buffer (${MB(limit)})`)
  }
  // Uploading to the GPU briefly holds a second copy of the weights; where memory is short and shared, stay on the CPU.
  const small = d.platform !== 'desktop' || (d.memoryGb != null && d.memoryGb <= 4)
  if (small && m.peakBytes > tabMemoryBudget(d).bytes * 0.5)
    return pick('wasm', 'little memory, shared with the GPU: the CPU avoids a second copy while loading')
  return m.gpuFirst ? pick('webgpu', 'WebGPU is available') : pick('wasm', 'small enough to be fastest on the CPU')
}

/** "encoder_model on WebGPU, decoder_model_merged on the CPU" */
export function describeDevices(devices: Record<string, Device>): string {
  const names = Object.keys(devices)
  if (names.length <= 1) return ''
  return names.map((n) => `${n} on ${devices[n] === 'webgpu' ? 'WebGPU' : 'the CPU'}`).join(', ')
}
