import path from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'
import { FS_ALLOW, boloVersion } from './vite.config'

// Clock times in the tests are written in UTC, as CI runs; a machine in another zone (IST on a developer's Mac)
// would render them hours apart. Set before the test workers start, which inherit it.
process.env.TZ = 'UTC'

// Time limits, in one place. A page test types, clicks and re-renders a whole page in jsdom: 1-4 s on a quiet
// machine, several times that on a busy one, so vitest's 5 s per test and Testing Library's 1 s for findBy/waitFor
// left no margin. BOLO_TEST_TIMEOUT_SCALE multiplies both (2 on a slow or shared runner). Tests do not wait on the
// mock server's simulated latency (src/test/setup.ts turns it off), so a longer limit only costs time when a test
// fails.
const scale = Number(process.env.BOLO_TEST_TIMEOUT_SCALE) > 0 ? Number(process.env.BOLO_TEST_TIMEOUT_SCALE) : 1
const TEST_TIMEOUT_MS = 15_000 * scale
const WAIT_TIMEOUT_MS = 5_000 * scale

export default defineConfig({
  plugins: [react()],
  define: { 'import.meta.env.VITE_BOLO_VERSION': JSON.stringify(boloVersion()) },
  server: { fs: { allow: FS_ALLOW } },
  resolve: { alias: { '@': path.resolve(import.meta.dirname, './src') } },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: false,
    testTimeout: TEST_TIMEOUT_MS,
    hookTimeout: TEST_TIMEOUT_MS,
    provide: { waitTimeoutMs: WAIT_TIMEOUT_MS },
  },
})
