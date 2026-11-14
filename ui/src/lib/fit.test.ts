import { describe, expect, it } from 'vitest'
import type { CatalogModel, HardwareInfo, ModelFit } from './api/types'
import {
  fitLanguage,
  fmtGb,
  fitLine,
  fitMemory,
  fitRank,
  fitRtf,
  fitTone,
  hardwareRows,
  isDefaultPick,
  isRecommended,
  isUnlikely,
  sortByFit,
  speedLine,
  warningText,
} from './fit'
import { mockHardware } from '@/mock/hardware'

const GIB = 1024 ** 3
const fit = (level: string, extra: Partial<ModelFit> = {}): ModelFit => ({
  level,
  expected_rtf: null,
  reason: `${level}.`,
  ...extra,
})
const model = (id: string, f: ModelFit | null): Pick<CatalogModel, 'id' | 'fit'> => ({ id, fit: f })

describe('fit', () => {
  it('orders levels from what runs well to what does not, with unknown levels last', () => {
    const levels = ['unavailable', 'slow', 'comfortable', 'unlikely', 'cloud', 'usable']
    expect([...levels].sort((a, b) => fitRank(fit(a)) - fitRank(fit(b)))).toEqual([
      'comfortable',
      'usable',
      'cloud',
      'slow',
      'unlikely',
      'unavailable',
    ])
    expect(fitRank(null)).toBeGreaterThan(fitRank(fit('unavailable')))
    expect(fitRank(fit('something-new'))).toBeGreaterThan(fitRank(fit('unavailable')))
    expect([fitTone('comfortable'), fitTone('slow'), fitTone('unlikely'), fitTone('cloud')]).toEqual([
      'good',
      'warning',
      'warning',
      'neutral',
    ])
  })

  it('sorts best for this machine first and never drops a model', () => {
    const models = [
      model('unlikely', fit('unlikely')),
      model('old-server', null),
      model('estimated', fit('comfortable', { rtf_basis: 'estimated', expected_rtf: 0.1 })),
      model('measured', fit('comfortable', { rtf_basis: 'measured', expected_rtf: 0.2 })),
      model('pick', fit('usable', { recommended: true, recommended_for: ['*'], default_for: ['*'] })),
      model('also', fit('comfortable', { recommended: true, recommended_for: ['*', 'hi'], default_for: ['hi'] })),
      model('cloud', fit('cloud')),
      model('slow', fit('slow')),
    ]
    const sorted = sortByFit(models).map((m) => m.id)
    expect(sorted).toEqual(['pick', 'also', 'measured', 'estimated', 'cloud', 'slow', 'unlikely', 'old-server'])
    expect(sorted).toHaveLength(models.length)
    // for one language, its own default pick leads
    expect(sortByFit(models, 'hi-IN').map((m) => m.id)[0]).toBe('also')
    expect(models.map((m) => m.id)[0]).toBe('unlikely') // the input is left alone
    expect(models.filter(isUnlikely).map((m) => m.id)).toEqual(['unlikely'])
  })

  it('judges "recommended" for the language filter, or for any language without one', () => {
    const m = model('m', fit('comfortable', { recommended: false, recommended_for: ['hi'], default_for: ['hi'] }))
    expect(isRecommended(m)).toBe(false)
    expect(isRecommended(m, 'hi')).toBe(true)
    expect(isRecommended(m, 'hi-IN')).toBe(true)
    expect(isRecommended(m, 'ta')).toBe(false)
    expect(isDefaultPick(m, 'hi')).toBe(true)
    expect(isDefaultPick(m)).toBe(false)
    expect(isRecommended(model('none', null), 'hi')).toBe(false)
    expect([fitLanguage(null), fitLanguage('auto'), fitLanguage('multi'), fitLanguage('en-US')]).toEqual([
      '*',
      '*',
      '*',
      'en',
    ])
  })

  it('writes the card line and says how the number was arrived at', () => {
    const measured = fit('comfortable', { expected_rtf: 0.11, rtf_basis: 'measured', scaled_by: 'heuristic' })
    expect(fitLine(measured)).toBe('Comfortable here · 9.1× real time (measured, roughly scaled)')
    const benchmarked = fit('slow', { expected_rtf: 2, rtf_basis: 'estimated', scaled_by: 'benchmark' })
    expect(fitLine(benchmarked)).toBe('Slow here · 0.5× real time (estimated, scaled by the benchmark)')
    expect(fitLine(fit('unlikely'))).toBe('Unlikely to fit here · speed not measured')
    expect(fitLine(fit('cloud'))).toBe('Cloud: this machine does not matter')
    expect(fitLine(fit('unavailable'))).toBe('Not available yet')
    expect(fitRtf(fit('usable', { expected_rtf: 0.5, rtf_range: [0.25, 1] }))).toBe('0.50 (0.25–1.00)')
    expect(fitRtf(fit('usable'))).toBeNull()
  })

  it('compares the memory a model needs with what there is', () => {
    const ram = fit('usable', {
      memory: { needed_bytes: 1.2 * GIB, available_bytes: 16 * GIB, kind: 'ram', basis: 'estimated' },
    })
    expect(fitMemory(ram)).toBe('about 1.2 GB of 16.0 GB memory')
    const vram = fit('usable', {
      memory: { needed_bytes: 8.9 * GIB, available_bytes: null, kind: 'vram', basis: 'catalog' },
    })
    expect(fitMemory(vram)).toBe('about 8.9 GB of an unknown amount GPU memory')
    expect(fitMemory(fit('cloud'))).toBeNull()
  })

  it('joins warnings into one notice and has none for a clean pull', () => {
    expect(warningText([])).toBeNull()
    expect(warningText(undefined)).toBeNull()
    expect(
      warningText([
        { code: 'hardware_fit', model: 'm', level: 'unlikely', message: 'Too big.' },
        { code: 'disk_space', model: 'm', level: 'unlikely', message: 'No room.' },
      ]),
    ).toBe('Too big. No room.')
  })

  it('describes the hardware, and prints what could not be read as unknown', () => {
    const hw = mockHardware()
    const rows = Object.fromEntries(hardwareRows(hw).map((r) => [r.label, r.value]))
    expect(rows.CPU).toContain('Intel(R) Xeon(R) Processor @ 2.10GHz · 4 usable cores (4 physical, 4 logical)')
    expect(rows['Instruction sets']).toBe('avx, avx2, avx512, avx512_vnni, amx')
    expect(rows.Memory).toBe('16.0 GB total, 11.2 GB available')
    expect(rows.GPU).toBe('none found')
    expect(rows.Environment).toBe('virtual machine')
    expect(fmtGb(null)).toBe('unknown')
    expect(fmtGb(256 * GIB)).toBe('256 GB')
    expect(speedLine(hw)).toBe('1.00× the reference machine (a rough estimate from the CPU)')
    expect(speedLine(mockHardware(true))).toBe('0.94× the reference machine (measured by the benchmark)')

    const blank: HardwareInfo = {
      ...hw,
      cpu: {
        model: null,
        vendor: null,
        architecture: null,
        physical_cores: null,
        logical_cores: null,
        usable_cores: null,
        features: null,
      },
      memory: { total_bytes: null, available_bytes: null, limit_bytes: null, usable_bytes: null },
      disk: { path: null, total_bytes: null, free_bytes: null },
      gpus_known: false,
      container: { ...hw.container, virtual_machine: null },
    }
    const unknown = Object.fromEntries(hardwareRows(blank).map((r) => [r.label, r.value]))
    expect(unknown.CPU).toBe('unknown · unknown usable cores (unknown physical, unknown logical)')
    expect(unknown['Instruction sets']).toBe('unknown')
    expect([unknown.Memory, unknown.Disk, unknown.GPU]).toEqual(['unknown', 'unknown', 'unknown'])
    expect(unknown.Environment).toBeUndefined()

    const gpu: HardwareInfo = {
      ...hw,
      gpus: [
        {
          vendor: 'nvidia',
          name: 'RTX 4090',
          vram_total_bytes: 24 * GIB,
          vram_free_bytes: 20 * GIB,
          driver: '550',
          unified_memory: false,
          backend: 'cuda',
        },
      ],
      memory: { ...hw.memory, limit_bytes: 4 * GIB },
      container: { ...hw.container, detected: true, kind: 'docker', cpu_quota: 2 },
    }
    const boxed = Object.fromEntries(hardwareRows(gpu).map((r) => [r.label, r.value]))
    expect(boxed.GPU).toBe('RTX 4090 (24.0 GB, 20.0 GB free)')
    expect(boxed.Memory).toContain('limited to 4.0 GB by the container')
    expect(boxed.Environment).toBe('container (docker), virtual machine, CPU quota 2 cores')
  })
})
