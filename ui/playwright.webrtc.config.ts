import fs from 'node:fs'
import path from 'node:path'
import { defineConfig } from '@playwright/test'

/**
 * WebRTC in a real browser against a real server (`npm run e2e:webrtc`). Opt-in: it downloads small real models
 * (about 250 MB: moonshine-tiny-en, piper-en-us-amy-low, silero-vad) on first run and needs Python with bolo and its
 * `webrtc` extra. Three servers are started:
 *
 * - the deterministic OpenAI-compatible language model of the server tests (server/tests/fake_llm.py), so no
 *   language model has to be downloaded: it answers "You said: <what it heard>. ...";
 * - `bolo serve` from this checkout, with that model connected;
 * - the console's dev server, which proxies the API to that bolo (the WebRTC media goes straight to bolo).
 *
 * Chromium's fake microphone plays bolo's en-short sample followed by 8 s of silence, on a loop: a turn, then a
 * pause long enough for the answer to be spoken before the sample comes round again.
 *
 * Environment: BOLO_PYTHON (default `python`), BOLO_DATA_DIR (where the models go; default bolo's own),
 * BOLO_WEBRTC_MEASUREMENT (write the measurement JSON there).
 */

const here = import.meta.dirname
const root = path.resolve(here, '..')
const python = process.env.BOLO_PYTHON || 'python'
export const BOLO_PORT = Number(process.env.BOLO_WEBRTC_PORT || 8766)
const LLM_PORT = BOLO_PORT + 100
const UI_PORT = BOLO_PORT + 200
export const SAMPLE = path.join(root, 'server/src/bolo/samples/en-short.wav')
export const MODELS = { stt: 'moonshine-tiny-en', tts: 'piper-en-us-amy-low', vad: 'silero-vad' }

/** The sample with `seconds` of silence after it (PCM WAV: the data chunk grows, the header says so). */
function padded(source: string, seconds: number, dest: string): string {
  const wav = fs.readFileSync(source)
  let offset = 12
  let rate = 16000
  let blockAlign = 2
  while (offset + 8 <= wav.length) {
    const id = wav.toString('ascii', offset, offset + 4)
    const size = wav.readUInt32LE(offset + 4)
    if (id === 'fmt ') {
      rate = wav.readUInt32LE(offset + 12)
      blockAlign = wav.readUInt16LE(offset + 20)
    }
    if (id === 'data') {
      const silence = Buffer.alloc(rate * blockAlign * seconds)
      const out = Buffer.concat([wav.subarray(0, offset + 8 + size), silence])
      out.writeUInt32LE(out.length - 8, 4)
      out.writeUInt32LE(size + silence.length, offset + 4)
      fs.mkdirSync(path.dirname(dest), { recursive: true })
      fs.writeFileSync(dest, out)
      return dest
    }
    offset += 8 + size + (size % 2)
  }
  throw new Error(`no data chunk in ${source}`)
}

const micAudio = padded(SAMPLE, 8, path.join(here, 'test-results-webrtc', 'mic.wav'))
const pythonEnv = { ...process.env, PYTHONPATH: path.join(root, 'server/src') } as Record<string, string>

export default defineConfig({
  testDir: './e2e-webrtc',
  timeout: 180_000,
  expect: { timeout: 60_000 },
  fullyParallel: false,
  workers: 1, // one fake microphone, one server
  retries: 0,
  reporter: [['list']],
  outputDir: 'test-results-webrtc',
  use: {
    baseURL: `http://127.0.0.1:${UI_PORT}`,
    trace: 'off',
    video: 'off',
    screenshot: 'only-on-failure',
    permissions: ['microphone'],
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
  webServer: [
    {
      command: `${python} ${path.join(root, 'server/tests/fake_llm.py')} --port ${LLM_PORT}`,
      url: `http://127.0.0.1:${LLM_PORT}/v1/models`,
      reuseExistingServer: false,
      timeout: 30_000,
    },
    {
      command: [
        `${python} -m bolo serve --port ${BOLO_PORT} --threads 2`,
        `--llm-base-url http://127.0.0.1:${LLM_PORT}/v1 --llm-model fake-llm`,
        `--default-realtime-stt ${MODELS.stt}`,
        ...Object.values(MODELS).map((m) => `--preload ${m}`),
        process.env.BOLO_DATA_DIR ? `--data-dir ${process.env.BOLO_DATA_DIR}` : '',
      ].join(' '),
      url: `http://127.0.0.1:${BOLO_PORT}/readyz`, // ready once the preloaded models are loaded
      env: pythonEnv,
      reuseExistingServer: false,
      timeout: 900_000, // the first run downloads the models
      stdout: 'ignore',
      stderr: 'pipe',
    },
    {
      command: `npx vite --port ${UI_PORT} --strictPort`,
      url: `http://127.0.0.1:${UI_PORT}`,
      env: { ...process.env, BOLO_SERVER: `http://127.0.0.1:${BOLO_PORT}` } as Record<string, string>,
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
})
