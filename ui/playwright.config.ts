import path from 'node:path'
import { defineConfig } from '@playwright/test'

const PORT = 4173
const micAudio = path.resolve(import.meta.dirname, 'src/mock/assets/en-ask-not.wav')

/**
 * End-to-end tests run against a production build of the console wired to
 * the in-process mock API (`npm run build:mock`), served by `vite preview`.
 * Chromium gets a fake microphone that plays a speech sample on a loop.
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: true,
  workers: 3,
  retries: 0,
  reporter: [['list']],
  outputDir: 'test-results',
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'off',
    video: 'off',
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
    command: `npm run build:mock && npm run preview:mock`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
