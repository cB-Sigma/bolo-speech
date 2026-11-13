import path from 'node:path'
import { defineConfig } from '@playwright/test'

const PORT = 4174
const mirror = process.env.BOLO_E2E_MODEL_DIR
const micAudio = path.resolve(import.meta.dirname, 'src/mock/assets/en-ask-not.wav')

/**
 * Tests for the live demo build (`npm run build:demo`), served the way GitHub
 * Pages serves it: under /bolo/, with no special headers (scripts/serve-pages.mjs).
 *
 * The tests that run the recogniser need the model. Point BOLO_E2E_MODEL_DIR at
 * a local mirror in Hugging Face layout; it is then served from a second origin
 * with the same CORS header the real CDNs send, so cross-origin loading under
 * the service worker's isolation is exercised for real. Without it they skip.
 */
export default defineConfig({
  testDir: './e2e-demo',
  timeout: 180_000,
  expect: { timeout: 15_000 },
  workers: 2,
  reporter: [['list']],
  outputDir: 'test-results',
  use: {
    baseURL: `http://127.0.0.1:${PORT}/bolo/`,
    trace: 'off',
    screenshot: 'only-on-failure',
    permissions: ['microphone', 'clipboard-read', 'clipboard-write'],
    launchOptions: {
      args: [
        '--use-fake-ui-for-media-stream',
        '--use-fake-device-for-media-stream',
        `--use-file-for-fake-audio-capture=${micAudio}`,
        '--autoplay-policy=no-user-gesture-required',
      ],
    },
  },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
  webServer: {
    command: `npm run build:demo && node scripts/serve-pages.mjs --port ${PORT}${mirror ? ` --mirror "${mirror}"` : ''}`,
    url: `http://127.0.0.1:${PORT}/bolo/`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
})
