import { describe, expect, it } from 'vitest'
import { decodeShare, encodeShare, shareUrl, type SharedResult } from './share'

const result: SharedResult = {
  m: 'Moonshine Tiny',
  d: 'WASM, 4 threads',
  n: 4,
  fp: 412.4,
  f50: 828.6,
  f95: 984,
  rtf: 0.4123,
}

describe('share links', () => {
  it('round-trips a result through a URL-safe token', () => {
    const token = encodeShare(result)
    expect(token).toMatch(/^[A-Za-z0-9_-]+$/)
    expect(decodeShare(token)).toEqual({ ...result, fp: 412, f50: 829, f95: 984, rtf: 0.412 })
  })
  it('keeps non-ASCII device names intact and missing values null', () => {
    const r = decodeShare(encodeShare({ ...result, d: 'WebGPU · फ़ोन', fp: null, rtf: null }))
    expect(r?.d).toBe('WebGPU · फ़ोन')
    expect(r?.fp).toBeNull()
    expect(r?.rtf).toBeNull()
  })
  it('builds a link into the playground under whatever path the demo is mounted at', () => {
    const url = shareUrl(result, { origin: 'https://cb-sigma.github.io', pathname: '/bolo/' })
    expect(url.startsWith('https://cb-sigma.github.io/bolo/#/stt?r=')).toBe(true)
    expect(decodeShare(url.split('?r=')[1])?.n).toBe(4)
  })
  it('rejects anything that is not a well-formed result', () => {
    const bad = (v: unknown) => btoa(JSON.stringify(v)).replace(/=+$/, '')
    expect(decodeShare(null)).toBeNull()
    expect(decodeShare('')).toBeNull()
    expect(decodeShare('not base64 !!')).toBeNull()
    expect(decodeShare(bad({ a: 1 }))).toBeNull()
    expect(decodeShare(bad([2, 'm', 'd', 1, 1, 1, 1, 1]))).toBeNull() // unknown version
    expect(decodeShare(bad([1, 'm', 'd', 0, 1, 1, 1, 1]))).toBeNull() // no utterances
    expect(decodeShare(bad([1, 'm', 'd', 3, 1, -5, 1, 1]))).toBeNull() // negative latency
    expect(decodeShare(bad([1, 'm', 'd', 3, 1, '<img>', 1, 1]))).toBeNull()
    expect(decodeShare('A'.repeat(500))).toBeNull()
  })
  it('truncates over-long labels instead of trusting them', () => {
    const r = decodeShare(btoa(JSON.stringify([1, 'x'.repeat(200), 'd', 2, null, 100, 200, null])).replace(/=+$/, ''))
    expect(r?.m).toHaveLength(40)
  })
})
