import { readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import {
  claimRuntime,
  isOutOfMemory,
  isWebKit,
  lowerMemoryMaximum,
  oneRuntimeAtATime,
  releaseOtherRuntimes,
  releaseRuntime,
  requireOneRuntime,
} from './browser-runtime'

const ortDist = path.resolve(import.meta.dirname, '../../node_modules/onnxruntime-web/dist')

describe('ONNX Runtime memory maximum', () => {
  it('the bundled runtime asks for a shared memory of up to 4 GiB, even on one thread', () => {
    const glue = readFileSync(path.join(ortDist, 'ort-wasm-simd-threaded.mjs'), 'utf8')
    expect(glue).toContain('maximum:65536,shared:!0')
  })
  it('is lowered in the loader of the version in use, and the memory is kept for reporting', () => {
    const glue = readFileSync(path.join(ortDist, 'ort-wasm-simd-threaded.mjs'), 'utf8')
    const lowered = lowerMemoryMaximum(glue, 16_384)
    expect(lowered).not.toBeNull()
    expect(lowered).not.toContain('maximum:65536,shared:!0')
    expect(lowered).toContain(
      'globalThis.__boloOrtMemory=new WebAssembly.Memory({initial:256,maximum:16384,shared:!0})',
    )
  })
  it('leaves an unrecognised loader alone', () => {
    expect(lowerMemoryMaximum('export default 1', 16_384)).toBeNull()
  })
})

describe('out-of-memory detection', () => {
  it('recognises what WebKit, V8 and ONNX Runtime say', () => {
    expect(isOutOfMemory('no available backend found. ERR: [wasm] RangeError: Out of memory')).toBe(true)
    expect(isOutOfMemory('WebAssembly.Memory(): could not allocate memory')).toBe(true)
    expect(isOutOfMemory('Array buffer allocation failed')).toBe(true)
    expect(isOutOfMemory('HTTP 404')).toBe(false)
    expect(isOutOfMemory(null)).toBe(false)
  })
})

describe('WebKit detection', () => {
  const ua = (userAgent: string, maxTouchPoints = 0) => isWebKit({ userAgent, maxTouchPoints })
  it('every browser on iPhone and iPad, and desktop Safari', () => {
    expect(
      ua(
        'Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 EdgiOS/141.0 Mobile/15E148 Safari/605.1.15',
      ),
    ).toBe(true)
    expect(
      ua(
        'Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/141.0 Mobile/15E148 Safari/604.1',
      ),
    ).toBe(true)
    expect(
      ua(
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15',
        5,
      ),
    ).toBe(true)
    expect(
      ua(
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15',
      ),
    ).toBe(true)
  })
  it('not Chromium or Firefox elsewhere', () => {
    expect(
      ua('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36'),
    ).toBe(false)
    expect(
      ua(
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36 Edg/141.0',
      ),
    ).toBe(false)
    expect(ua('Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0')).toBe(false)
  })
})

describe('one runtime at a time', () => {
  it('releases the other holder when only one may be alive', () => {
    requireOneRuntime()
    expect(oneRuntimeAtATime()).toBe(true)
    const stt = vi.fn()
    const tts = vi.fn()
    claimRuntime('stt', stt)
    claimRuntime('tts', tts)
    expect(stt).toHaveBeenCalledTimes(1)
    expect(tts).not.toHaveBeenCalled()
    // Claiming again for the same holder does not release it.
    claimRuntime('tts', tts)
    expect(tts).not.toHaveBeenCalled()
    expect(releaseOtherRuntimes('stt')).toEqual(['tts'])
    expect(tts).toHaveBeenCalledTimes(1)
    releaseRuntime('stt')
  })
})
