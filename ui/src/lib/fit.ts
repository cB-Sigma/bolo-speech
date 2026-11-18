import type { CatalogModel, FitWarning, HardwareInfo, ModelFit } from './api/types'
import { fmtRtf, fmtSpeed } from './format'

/**
 * How well a model suits the server's machine, as the server computed it (`fit` on catalog entries).
 *
 * The console's rule is the server's: recommend, never restrict. A model that is unlikely to fit sorts last and
 * says why; it is only hidden when the user switches the clearly labelled "Hide unlikely to fit" toggle on.
 */

/**
 * Memory, VRAM and disk the way the server words them in its reasons ("needs about 0.3 GB of the 15.7 GB"):
 * binary gigabytes, one decimal.
 */
export function fmtGb(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return 'unknown'
  const gb = bytes / 1024 ** 3
  return `${gb >= 100 ? Math.round(gb) : gb.toFixed(1)} GB`
}

/** Best first. `cloud` sits between what runs well here and what does not; `unavailable` is not about hardware. */
export const FIT_ORDER = ['comfortable', 'usable', 'cloud', 'slow', 'unlikely', 'unavailable'] as const

export const FIT_LABEL: Record<string, string> = {
  comfortable: 'Comfortable',
  usable: 'Usable',
  slow: 'Slow',
  unlikely: 'Unlikely to fit',
  cloud: 'Cloud',
  unavailable: 'Not available yet',
}

/** The fit's label: "Won't fit this GPU" for a model that cannot be split and needs more GPU memory than there is. */
export function fitLabel(fit: ModelFit): string {
  if (fit.gpu_too_small) return "Won't fit this GPU"
  return FIT_LABEL[fit.level] ?? fit.label ?? fit.level
}

export type FitTone = 'good' | 'neutral' | 'warning'

export function fitTone(level: string | undefined): FitTone {
  if (level === 'comfortable' || level === 'usable') return 'good'
  if (level === 'slow' || level === 'unlikely') return 'warning'
  return 'neutral'
}

export function fitRank(fit: ModelFit | null | undefined): number {
  const i = (FIT_ORDER as readonly string[]).indexOf(fit?.level ?? '')
  return i < 0 ? FIT_ORDER.length : i
}

/** The language to judge "recommended" by: the base code of a filter such as `hi` or `en-US`, else any (`*`). */
export function fitLanguage(lang: string | null | undefined): string {
  if (!lang || lang === 'auto' || lang === 'multi') return '*'
  return lang.toLowerCase().split(/[-_]/)[0] || '*'
}

/** Whether the server marked the model as one of the few recommended for its task (and this language). */
export function isRecommended(m: Pick<CatalogModel, 'fit'>, lang?: string | null): boolean {
  const fit = m.fit
  if (!fit) return false
  const key = fitLanguage(lang)
  if (key === '*') return Boolean(fit.recommended ?? fit.recommended_for?.includes('*'))
  return Boolean(fit.recommended_for?.includes(key))
}

/** Whether the model is the default pick for its task (and this language). */
export function isDefaultPick(m: Pick<CatalogModel, 'fit'>, lang?: string | null): boolean {
  return Boolean(m.fit?.default_for?.includes(fitLanguage(lang)))
}

/** True for the models the optional "Hide unlikely to fit" toggle removes. Nothing else ever hides a model. */
export function isUnlikely(m: Pick<CatalogModel, 'fit'>): boolean {
  return m.fit?.level === 'unlikely'
}

/** "Best for this machine": the default pick, then the recommended, then by fit level; ties keep their order. */
export function sortByFit<T extends Pick<CatalogModel, 'fit' | 'id'>>(models: T[], lang?: string | null): T[] {
  const key = (m: T) =>
    (isDefaultPick(m, lang) ? 0 : isRecommended(m, lang) ? 1 : 2) * 10 +
    fitRank(m.fit) +
    // a measured figure before an estimate, before a guess
    ({ measured: 0, estimated: 0.3 }[m.fit?.rtf_basis ?? ''] ?? 0.6)
  return models
    .map((m, i) => ({ m, i, k: key(m) }))
    .sort((a, b) => a.k - b.k || a.i - b.i)
    .map((x) => x.m)
}

/** How the expected speed was arrived at, in a few words. */
export function fitBasis(fit: ModelFit): string {
  if (fit.expected_rtf == null) return fit.level === 'cloud' || fit.level === 'unavailable' ? '' : 'not measured'
  const scaled = fit.scaled_by === 'benchmark' ? 'scaled by the benchmark' : 'roughly scaled'
  return fit.rtf_basis === 'measured' ? `measured, ${scaled}` : `estimated, ${scaled}`
}

/** One line for a card: "Comfortable here · 9.1× real time (measured, roughly scaled)". */
export function fitLine(fit: ModelFit): string {
  const label = fitLabel(fit)
  if (fit.level === 'cloud') return 'Cloud: this machine does not matter'
  if (fit.level === 'unavailable' || fit.gpu_too_small) return label
  const here = `${label} here`
  if (fit.expected_rtf == null) return `${here} · speed not measured`
  return `${here} · ${fmtSpeed(fit.expected_rtf)} real time (${fitBasis(fit)})`
}

/** "RTF 0.11 (0.06–0.23)". */
export function fitRtf(fit: ModelFit): string | null {
  if (fit.expected_rtf == null) return null
  const range = fit.rtf_range ? ` (${fmtRtf(fit.rtf_range[0])}–${fmtRtf(fit.rtf_range[1])})` : ''
  return `${fmtRtf(fit.expected_rtf)}${range}`
}

/** "0.3 GB of 16 GB memory" / "8.9 GB of 24 GB GPU memory". */
export function fitMemory(fit: ModelFit): string | null {
  const mem = fit.memory
  if (!mem || mem.needed_bytes == null) return null
  const kind = mem.kind === 'vram' ? 'GPU memory' : 'memory'
  const have = mem.available_bytes == null ? 'an unknown amount' : fmtGb(mem.available_bytes)
  return `about ${fmtGb(mem.needed_bytes)} of ${have} ${kind}`
}

/** The text of the non-blocking notice shown when a pull or load came back with warnings. */
export function warningText(warnings: FitWarning[] | undefined | null): string | null {
  if (!warnings?.length) return null
  return warnings.map((w) => w.message).join(' ')
}

const unknown = (v: unknown, unit = ''): string => (v == null ? 'unknown' : `${String(v)}${unit}`)

/** The hardware card's rows. A value the server could not read is shown as "unknown", never guessed. */
export function hardwareRows(hw: HardwareInfo): { label: string; value: string }[] {
  const cpu = hw.cpu
  const isa = cpu.features == null ? 'unknown' : cpu.features.filter((f) => !['fma', 'f16c', 'sse4.2'].includes(f))
  const rows = [
    {
      label: 'CPU',
      value: `${unknown(cpu.model)} · ${unknown(cpu.usable_cores ?? cpu.logical_cores)} usable cores (${unknown(cpu.physical_cores)} physical, ${unknown(cpu.logical_cores)} logical)`,
    },
    {
      label: 'Instruction sets',
      value: typeof isa === 'string' ? isa : isa.length ? isa.join(', ') : 'none that matter',
    },
    {
      label: 'Memory',
      value:
        hw.memory.total_bytes == null
          ? 'unknown'
          : `${fmtGb(hw.memory.total_bytes)} total, ${hw.memory.available_bytes == null ? 'unknown' : fmtGb(hw.memory.available_bytes)} available` +
            (hw.memory.limit_bytes ? ` · limited to ${fmtGb(hw.memory.limit_bytes)} by the container` : ''),
    },
    {
      label: 'Disk',
      value: hw.disk.free_bytes == null ? 'unknown' : `${fmtGb(hw.disk.free_bytes)} free in the data directory`,
    },
    {
      label: 'GPU',
      value: hw.gpus.length
        ? hw.gpus
            .map((g) => {
              const mem = g.unified_memory
                ? 'memory shared with the system'
                : g.vram_total_bytes == null
                  ? 'memory unknown'
                  : `${fmtGb(g.vram_total_bytes)}${g.vram_free_bytes == null ? '' : `, ${fmtGb(g.vram_free_bytes)} free`}`
              return `${g.name ?? g.vendor} (${mem}${g.backend ? '' : '; not used by any runtime here'})`
            })
            .join('; ')
        : hw.gpus_known
          ? 'none found'
          : 'unknown',
    },
  ]
  const box = hw.container
  const env = [
    box.detected ? `container (${box.kind ?? 'unknown kind'})` : null,
    box.virtual_machine ? 'virtual machine' : null,
    box.cpu_quota != null ? `CPU quota ${box.cpu_quota} cores` : null,
  ].filter(Boolean)
  if (env.length) rows.push({ label: 'Environment', value: env.join(', ') })
  return rows
}

/** "1.2× the reference machine, measured" / "0.4× the reference machine, a rough estimate". */
export function speedLine(hw: Pick<HardwareInfo, 'speed'>): string {
  const s = hw.speed
  const how = s.source === 'benchmark' ? 'measured by the benchmark' : 'a rough estimate from the CPU'
  return `${s.factor.toFixed(2)}× the reference machine (${how})`
}
