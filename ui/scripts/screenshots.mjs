// Takes the README screenshots from a running console.
//
//   node scripts/screenshots.mjs --base http://127.0.0.1:8000 --out ../docs/assets --speech path/to/speech.wav
//
// --base     where the console is served (a real `bolo serve`, or `npm run preview:mock`)
// --speech   a speech recording to transcribe in the playground (WAV)
// --only     comma-separated subset: playground,latency,models,tts,benchmark,light,timeline,phone,hindi,compare,
//            animation,demo,components
// --demo     where the live demo build is served (for `demo`), e.g. http://127.0.0.1:4174/bolo
// --browser-stt  JSON for the in-browser recogniser's local mirror (for `compare` and `demo`)
// --stt/--tts  model ids to select (default: whatever the console picks)
// --hindi-model  an installed model that supports Hindi (for `hindi`), e.g. faster-whisper-tiny-hindi
// --components   where a server WITHOUT eSpeak NG is served (for `components`), e.g. one started with BOLO_ESPEAK=off
// --components-tts  a voice that needs eSpeak NG (default piper-en-us-amy-low)
//
// Nothing is faked: each image is the page as it looked after really doing the thing.
import { execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { parseArgs } from 'node:util'
import { chromium } from '@playwright/test'

const { values: args } = parseArgs({
  options: {
    base: { type: 'string', default: 'http://127.0.0.1:8000' },
    out: { type: 'string', default: '../docs/assets' },
    speech: { type: 'string', default: 'src/mock/assets/en-ask-not.wav' },
    only: { type: 'string' },
    stt: { type: 'string' },
    tts: { type: 'string' },
    voice: { type: 'string' },
    target: { type: 'string', default: '300' },
    'playground-target': { type: 'string' },
    'bench-seconds': { type: 'string', default: '5' },
    demo: { type: 'string' },
    'browser-stt': { type: 'string' },
    sample: { type: 'string', default: 'en-long' },
    'hindi-model': { type: 'string' },
    components: { type: 'string' },
    'components-tts': { type: 'string', default: 'piper-en-us-amy-low' },
  },
})
const only = args.only ? new Set(args.only.split(',')) : null
const out = path.resolve(args.out)
const speech = path.resolve(args.speech)

const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] })

async function shot(
  name,
  { route, theme = 'dark', width = 1440, height = 900, scale = 2, prefs = {}, act, base = args.base, clip },
) {
  if (only && !only.has(name.replace(/^console-/, ''))) return
  const context = await browser.newContext({
    viewport: { width, height },
    deviceScaleFactor: scale,
    colorScheme: theme,
  })
  const page = await context.newPage()
  // The live demo is shown exactly as a first-time visitor gets it: no stored preferences.
  const stored =
    base === args.base
      ? {
          targetMs: Number(args.target),
          ...(args.stt && { sttModel: args.stt }),
          ...(args.tts && { ttsModel: args.tts }),
          ...(args.voice && { ttsVoice: args.voice }),
          ...prefs,
        }
      : {}
  await page.addInitScript(
    ([t, p, b]) => {
      localStorage.setItem('bolo.theme', t)
      localStorage.setItem('bolo.prefs', JSON.stringify(p))
      if (b) localStorage.setItem('bolo.browserStt', b)
    },
    [theme, stored, args['browser-stt'] ?? null],
  )
  await page.goto(`${base}/#${route}`)
  await page.locator('[data-testid=connection][data-state=online]').waitFor({ timeout: 20_000 })
  await act?.(page)
  // Let toasts leave and tooltips close before the picture.
  await page.mouse.move(700, 5)
  await page
    .locator('[data-sonner-toast]')
    .first()
    .waitFor({ state: 'detached', timeout: 15_000 })
    .catch(() => {})
  const file = path.join(out, `${name}.png`)
  const target = clip ? page.locator(clip).first() : page
  await target.screenshot({ path: file })
  console.log('wrote', file)
  await context.close()
}

const transcribe = async (page) => {
  await page.locator('input[type=file]').setInputFiles(speech)
  await page.getByTestId('stop').waitFor()
  await page.getByTestId('mic').waitFor({ timeout: 60_000 })
  await page.waitForTimeout(500)
}
const playgroundPrefs = args['playground-target'] ? { targetMs: Number(args['playground-target']) } : {}

await shot('console-playground', { route: '/stt', height: 1080, prefs: playgroundPrefs, act: transcribe })
await shot('console-light', { route: '/stt', theme: 'light', height: 1080, prefs: playgroundPrefs, act: transcribe })
await shot('console-latency', {
  route: '/latency',
  height: 1240,
  act: async (page) => {
    await page.getByTestId('waterfall').waitFor({ timeout: 30_000 })
    await page.waitForTimeout(20_000) // let the live sparklines fill in
  },
})
await shot('console-models', {
  route: '/models',
  height: 1000,
  act: async (page) => {
    await page.getByTestId('model-card').first().waitFor()
    await page.waitForTimeout(1500)
  },
})
await shot('console-tts', {
  route: '/tts',
  height: 1100,
  act: async (page) => {
    await page.getByTestId('round-trip').click()
    await page.getByTestId('wer').waitFor({ timeout: 120_000 })
    await page.getByTestId('speak').waitFor({ timeout: 120_000 })
    await page.waitForTimeout(500)
  },
})
await shot('console-benchmark', {
  route: '/benchmark',
  height: 960,
  act: async (page) => {
    await page.locator('#bench-duration').fill(args['bench-seconds'])
    await page.getByTestId('bench-run').click()
    await page.getByText('Benchmark finished').waitFor({ timeout: 300_000 })
  },
})

// Close-up of the latency timeline panel.
await shot('console-timeline', {
  route: '/stt',
  height: 1080,
  prefs: playgroundPrefs,
  act: transcribe,
  clip: 'section:has([data-testid=latency-timeline])',
})

// Phone width.
await shot('console-phone', {
  route: '/stt',
  width: 390,
  height: 844,
  scale: 3,
  prefs: playgroundPrefs,
  act: transcribe,
})

// Hindi on the server: the language select, a Hindi model and the built-in Hindi sample.
if (args['hindi-model']) {
  await shot('console-hindi', {
    route: '/stt',
    height: 1080,
    prefs: { ...playgroundPrefs, sttModel: args['hindi-model'], language: 'hi' },
    act: async (page) => {
      await page.getByTestId('sample-hi-short').click()
      await page.getByTestId('stop').waitFor()
      await page.getByTestId('mic').waitFor({ timeout: 60_000 })
      await page.waitForTimeout(500)
    },
  })
}

// A voice that needs eSpeak NG on a server that does not have it: the notice, then the confirmation dialog.
if (args.components && (!only || only.has('components'))) {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    deviceScaleFactor: 2,
    colorScheme: 'dark',
  })
  const page = await context.newPage()
  await page.addInitScript((tts) => {
    localStorage.setItem('bolo.theme', 'dark')
    localStorage.setItem('bolo.prefs', JSON.stringify({ ttsModel: tts }))
  }, args['components-tts'])
  await page.goto(`${args.components}/#/tts`)
  const notice = page.getByTestId('component-notice')
  await notice.waitFor({ timeout: 20_000 })
  await page.mouse.move(700, 5)
  await page.screenshot({ path: path.join(out, 'console-components.png') })
  console.log('wrote', path.join(out, 'console-components.png'))
  await notice.getByTestId('install-component').click()
  await page.getByTestId('install-dialog').waitFor()
  await page.waitForTimeout(400)
  // The dialog is photographed, not confirmed: nothing is installed by taking the picture.
  await page.screenshot({ path: path.join(out, 'console-components-dialog.png') })
  console.log('wrote', path.join(out, 'console-components-dialog.png'))
  await context.close()
}

// Browser vs server on the same audio (needs --browser-stt or internet access for the model).
await shot('console-compare', {
  route: '/stt',
  height: 1180,
  prefs: playgroundPrefs,
  act: async (page) => {
    await page.getByTestId('mode-compare').click()
    await page.locator('input[type=file]').setInputFiles(speech)
    await page.locator('[data-testid=engine-status][data-status=ready]').waitFor({ timeout: 180_000 })
    await page.getByTestId('mic').waitFor({ timeout: 120_000 })
    await page.waitForTimeout(500)
  },
})

// The live demo build: everything in the browser, after a real run.
if (args.demo) {
  await shot('console-demo', {
    base: args.demo,
    route: '/stt',
    height: 1180,
    act: async (page) => {
      await page.getByTestId('hero-sample').click()
      await page.locator('[data-testid=engine-status][data-status=ready]').waitFor({ timeout: 180_000 })
      await page.getByTestId('mic').waitFor({ timeout: 120_000 })
      await page.waitForTimeout(500)
    },
  })
}

// Animated capture of a live transcription: frames from the real page, assembled with ffmpeg.
if (!only || only.has('animation')) {
  const frames = mkdtempSync(path.join(os.tmpdir(), 'bolo-frames-'))
  const context = await browser.newContext({
    viewport: { width: 1320, height: 700 },
    deviceScaleFactor: 1,
    colorScheme: 'dark',
  })
  const page = await context.newPage()
  await page.addInitScript(
    (p) => {
      localStorage.setItem('bolo.theme', 'dark')
      localStorage.setItem('bolo.prefs', JSON.stringify(p))
    },
    { targetMs: Number(args.target), ...(args.stt && { sttModel: args.stt }), ...playgroundPrefs },
  )
  await page.goto(`${args.base}/#/stt`)
  await page.locator('[data-testid=connection][data-state=online]').waitFor({ timeout: 20_000 })
  await page.getByTestId(`sample-${args.sample}`).waitFor()
  await page.waitForTimeout(800)
  const FPS = 10
  let n = 0
  let running = true
  const grab = async () => {
    const t0 = Date.now()
    // Leave the sidebar out: the frame is the playground itself.
    await page.screenshot({
      path: path.join(frames, `f${String(n++).padStart(4, '0')}.png`),
      clip: { x: 224, y: 0, width: 1096, height: 700 },
    })
    return Date.now() - t0
  }
  const loop = (async () => {
    while (running) {
      const spent = await grab()
      await page.waitForTimeout(Math.max(0, 1000 / FPS - spent))
    }
  })()
  await page.waitForTimeout(600)
  await page.getByTestId(`sample-${args.sample}`).click()
  await page.getByTestId('stop').waitFor()
  await page.getByTestId('mic').waitFor({ timeout: 60_000 })
  await page.waitForTimeout(2500) // hold on the result
  running = false
  await loop
  await context.close()
  mkdirSync(out, { recursive: true })
  const gif = path.join(out, 'console-playground.gif')
  const palette = path.join(frames, 'palette.png')
  const scale = 'scale=960:-1:flags=lanczos'
  execFileSync('ffmpeg', [
    '-loglevel',
    'error',
    '-y',
    '-framerate',
    String(FPS),
    '-i',
    path.join(frames, 'f%04d.png'),
    '-vf',
    `${scale},palettegen=max_colors=96:stats_mode=diff`,
    palette,
  ])
  execFileSync('ffmpeg', [
    '-loglevel',
    'error',
    '-y',
    '-framerate',
    String(FPS),
    '-i',
    path.join(frames, 'f%04d.png'),
    '-i',
    palette,
    '-lavfi',
    `${scale}[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle`,
    '-loop',
    '0',
    gif,
  ])
  rmSync(frames, { recursive: true, force: true })
  console.log('wrote', gif, `(${n} frames at ${FPS} fps)`)
}

await browser.close()
