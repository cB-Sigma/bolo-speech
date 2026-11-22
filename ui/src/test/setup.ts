import '@testing-library/jest-dom/vitest'
import { cleanup, configure } from '@testing-library/react'
import { afterEach, inject } from 'vitest'
import { setMockLatency } from '@/mock/engine'

declare module 'vitest' {
  export interface ProvidedContext {
    waitTimeoutMs: number
  }
}

afterEach(cleanup)

// findBy* and waitFor wait as long as vitest.config.ts says (scaled by BOLO_TEST_TIMEOUT_SCALE), not 1 s.
configure({ asyncUtilTimeout: inject('waitTimeoutMs') })
// The mock server answers at once: its simulated latency (up to 0.7 s a request, 2.5 s to load a model) is for the
// demo, and in a test it was waiting against the time limits.
setMockLatency(0)

// jsdom lacks a few browser APIs that Radix primitives expect.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver
if (typeof Element !== 'undefined') {
  Element.prototype.scrollIntoView ??= () => undefined
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.releasePointerCapture ??= () => undefined
}
