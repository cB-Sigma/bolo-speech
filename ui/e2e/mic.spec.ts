import { expect, test, type Page } from '@playwright/test'
import { expectNoA11yViolations, expectNoHorizontalScroll, open, watchErrors } from './helpers'

/**
 * The microphone and first-use experience: permission failures explained in
 * place, the microphone requested inside the click, speech buffered while the
 * in-browser model loads, staged progress, the stall watchdog and `?debug=1`.
 *
 * The in-browser model is replaced by a scripted worker (src/mock/fake-stt-worker.ts)
 * so that slow, stuck and failing loads can be staged.
 */

interface MicStub {
  name: string
  message: string
  delayMs: number
  permission: 'denied' | 'prompt' | 'granted'
}

/** Make getUserMedia fail the way a browser does, and the Permissions API answer accordingly. */
async function stubMicFailure(page: Page, stub: MicStub) {
  await page.addInitScript((s) => {
    navigator.mediaDevices.getUserMedia = () =>
      new Promise((_resolve, reject) => setTimeout(() => reject(new DOMException(s.message, s.name)), s.delayMs))
    const query = navigator.permissions.query.bind(navigator.permissions)
    navigator.permissions.query = (d: PermissionDescriptor) =>
      (d.name as string) === 'microphone'
        ? Promise.resolve({ state: s.permission, onchange: null } as unknown as PermissionStatus)
        : query(d)
  }, stub)
}

const fakeModel = (opts: Record<string, unknown>, extra: Record<string, unknown> = {}) => ({
  'bolo.mock.browserStt': JSON.stringify(opts),
  ...(Object.keys(extra).length ? { 'bolo.browserStt': JSON.stringify(extra) } : {}),
})

test.describe('microphone permission help', () => {
  for (const width of [1440, 390]) {
    test(`a site-level block is explained in place, with the alternatives · ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 })
      await stubMicFailure(page, {
        name: 'NotAllowedError',
        message: 'Permission denied',
        delayMs: 5,
        permission: 'denied',
      })
      await open(page, '/stt')
      await page.getByTestId('mic').click()
      const help = page.getByTestId('mic-help')
      await expect(help).toBeVisible()
      await expect(help).toHaveAttribute('data-kind', 'denied')
      await expect(help.getByRole('heading')).toHaveText('The microphone is blocked for this site')
      await expect(help).toContainText('refused without asking')
      // Exact steps for this browser (desktop Chromium), and what the browser itself said.
      await expect(help).toContainText('Click the icon at the left of the address bar')
      await expect(help).toContainText('NotAllowedError: Permission denied')
      await expect(help).toContainText('saved permission: denied')
      await expect(help.getByTestId('mic-retry')).toBeVisible()
      await expect(help.getByTestId('mic-help-upload')).toBeVisible()
      // No one-line toast any more.
      await expect(page.getByText('Microphone blocked', { exact: true })).toHaveCount(0)
      await expectNoA11yViolations(page)
      await expectNoHorizontalScroll(page)

      // "Play a sample" right there works without a microphone.
      await help.getByTestId('mic-help-sample').click()
      await expect(page.getByTestId('final-text').first()).toBeVisible({ timeout: 20_000 })
      await expect(page.getByTestId('mic-help')).toHaveCount(0)
    })
  }

  test('a dismissed prompt is told apart from a block, and Try again asks again', async ({ page }) => {
    await page.addInitScript(() => {
      let calls = 0
      const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices)
      navigator.mediaDevices.getUserMedia = (c) => {
        calls += 1
        ;(window as unknown as { __gum: number }).__gum = calls
        return calls === 1
          ? new Promise((_r, reject) =>
              setTimeout(() => reject(new DOMException('Permission dismissed', 'NotAllowedError')), 900),
            )
          : real(c)
      }
    })
    await open(page, '/stt')
    await page.getByTestId('mic').click()
    // While the prompt is up the page says what is happening.
    await expect(page.getByTestId('starting-note')).toContainText('asking for the microphone')
    const help = page.getByTestId('mic-help')
    await expect(help.getByRole('heading')).toHaveText('The microphone prompt was closed')
    await expect(help).toHaveAttribute('data-kind', 'dismissed')
    await help.getByTestId('mic-retry').click()
    await expect(page.getByTestId('stop')).toBeVisible()
    await expect(page.getByTestId('mic-help')).toHaveCount(0)
    expect(await page.evaluate(() => (window as unknown as { __gum: number }).__gum)).toBe(2)
    await expect(page.getByTestId('final-text').first()).toBeVisible({ timeout: 20_000 })
    await page.getByTestId('stop').click()
  })

  test('an instant refusal with no saved block points at the operating system', async ({ page }) => {
    await stubMicFailure(page, {
      name: 'NotAllowedError',
      message: 'Permission denied',
      delayMs: 5,
      permission: 'prompt',
    })
    await open(page, '/stt')
    await page.getByTestId('mic').click()
    const help = page.getByTestId('mic-help')
    await expect(help.getByRole('heading')).toContainText('is not allowed to use the microphone')
    await expect(help).toContainText('before any prompt could appear')
  })

  test('no microphone at all is reported as such', async ({ page }) => {
    await stubMicFailure(page, {
      name: 'NotFoundError',
      message: 'Requested device not found',
      delayMs: 5,
      permission: 'granted',
    })
    await open(page, '/stt')
    await page.getByTestId('mic').click()
    await expect(page.getByTestId('mic-help').getByRole('heading')).toHaveText('No microphone was found')
  })
})

test.describe('in-app browsers', () => {
  test.use({
    userAgent:
      'Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/22F76 Instagram 385.0.0.26.82 (iPhone15,2; iOS 18_5; en_US; en; scale=3.00; 1179x2556; 742961871)',
    viewport: { width: 390, height: 844 },
  })

  test('an in-app browser is named, never blocked up front, and offered a link to copy', async ({ page }) => {
    await stubMicFailure(page, {
      name: 'NotAllowedError',
      message: 'The request is not allowed by the user agent or the platform in the current context.',
      delayMs: 5,
      permission: 'prompt',
    })
    await open(page, '/stt')
    // Detection never disables the attempt.
    await expect(page.getByTestId('mic')).toBeEnabled()
    await page.getByTestId('mic').click()
    const help = page.getByTestId('mic-help')
    await expect(help.getByRole('heading')).toHaveText('This looks like a browser inside another app')
    await expect(help).toContainText('inside Instagram')
    await expect(help).toContainText('Open in Safari')
    await expect(help.getByTestId('mic-help-url')).toContainText('/#/stt')
    await help.getByTestId('mic-copy-link').click()
    await expect(help.getByTestId('mic-copy-link')).toContainText('Link copied')
    expect(await page.evaluate(() => navigator.clipboard.readText())).toContain('/#/stt')
    await expectNoA11yViolations(page)
    await expectNoHorizontalScroll(page)
  })
})

test.describe('granted microphone', () => {
  test.use({ viewport: { width: 1440, height: 900 } })

  test('the permission request and the audio context both happen inside the click', async ({ page }) => {
    await page.addInitScript(() => {
      const log: Record<string, unknown> = {}
      ;(window as unknown as { __mic: typeof log }).__mic = log
      // True only while the click event is being dispatched: anything called then was not awaited first.
      let dispatching = false
      document.addEventListener('click', () => (dispatching = true), true)
      window.addEventListener('click', () => (dispatching = false))
      const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices)
      navigator.mediaDevices.getUserMedia = (c) => {
        log.gumInClick = dispatching
        log.gumActive = navigator.userActivation.isActive
        log.order = [...((log.order as string[]) ?? []), 'getUserMedia']
        return real(c)
      }
      const RealContext = window.AudioContext
      window.AudioContext = class extends RealContext {
        constructor(o?: AudioContextOptions) {
          super(o)
          log.ctxInClick = dispatching
          log.ctxActive = navigator.userActivation.isActive
          log.order = [...((log.order as string[]) ?? []), 'AudioContext']
        }
      }
    })
    // A slow model must not delay the prompt.
    await open(page, '/stt', { storage: fakeModel({ loadMs: 4000 }) })
    await page.getByTestId('mode-browser').click()
    await page.getByTestId('mic').click()
    await expect(page.getByTestId('starting-note')).toHaveAttribute('data-mic', 'live')
    const log = await page.evaluate(() => (window as unknown as { __mic: Record<string, boolean | string[]> }).__mic)
    expect(log.order).toEqual(['getUserMedia', 'AudioContext'])
    expect(log.gumActive).toBe(true)
    expect(log.ctxActive).toBe(true)
    // Both ran while the click was still being dispatched: nothing was awaited first.
    expect(log.gumInClick).toBe(true)
    expect(log.ctxInClick).toBe(true)
    await page.getByTestId('stop').click()
  })
})

test.describe('in-browser model: loading', () => {
  test.use({ viewport: { width: 1440, height: 900 } })

  test('speech is buffered while the model loads and transcribed once it is ready, with staged progress', async ({
    page,
  }) => {
    const errors = watchErrors(page)
    await open(page, '/stt', { storage: fakeModel({ loadMs: 6000 }) })
    await page.getByTestId('mode-browser').click()
    const chip = page.getByTestId('model-chip')
    await expect(chip).toContainText('Moonshine Tiny')
    await expect(chip).toContainText('Not downloaded · 32.1 MB')

    await page.getByTestId('mic').click()
    // The microphone is live long before the model is.
    const note = page.getByTestId('starting-note')
    await expect(note).toHaveAttribute('data-mic', 'live')
    await expect(note).toContainText('The microphone is on')
    await expect(page.getByTestId('stop')).toBeVisible()

    // Real stages with bytes, in the downloads tray: the one place a download shows; the page says only "Loading".
    const row = page
      .getByTestId('download-tray')
      .getByTestId(/^download-/)
      .first()
    await expect(row).toContainText(/Downloading the runtime · [\d.]+ MB of 14\.3 MB/)
    await expect(row).toContainText(/Downloading the model · [\d.]+ MB of [\d.]+ MB/)
    await expect(row).toHaveAttribute('data-stage', 'compiling')
    await expect(row).toContainText('Compiling the model')
    await expect(chip).toContainText('Getting the model ready')
    await expect(page.getByTestId('load-progress')).toHaveCount(0)
    await expect(row.getByRole('listitem')).toHaveText([
      /Runtime.*done/,
      /Model.*done/,
      /Compile.*in progress/,
      /Warm up.*waiting/,
    ])

    // Audio is being held meanwhile.
    await expect
      .poll(async () => Number.parseFloat(await page.getByTestId('buffered-seconds').innerText()), { timeout: 10_000 })
      .toBeGreaterThan(2)
    const held = Number.parseFloat(await page.getByTestId('buffered-seconds').innerText())

    // Ready: the held audio is handed over at once, so the take is already longer than what was held.
    await expect(chip).toContainText('Ready', { timeout: 15_000 })
    await expect(note).toHaveCount(0)
    const status = await page.getByText(/Streaming · microphone · [\d.]+ s/).innerText()
    expect(Number.parseFloat(/microphone · ([\d.]+) s/.exec(status)?.[1] ?? '0')).toBeGreaterThanOrEqual(held)
    await expect(page.getByTestId('final-text').first()).toContainText('pretend transcript', { timeout: 20_000 })
    await page.getByTestId('stop').click()
    await expect(page.getByTestId('mic')).toBeVisible({ timeout: 10_000 })
    expect(errors).toEqual([])
  })

  test('a download shows in the tray alone, with its stage and bytes; the page keeps one chip', async ({ page }) => {
    const errors = watchErrors(page)
    await open(page, '/stt', { storage: fakeModel({ loadMs: 6000 }) })
    await page.getByTestId('mode-browser').click()
    const chip = page.getByTestId('model-chip')
    await expect(chip).toHaveCount(1)
    await chip.click()
    const row = page
      .getByTestId('download-tray')
      .getByTestId(/^download-/)
      .first()
    await expect(row).toContainText('Speech to text: Moonshine Tiny')
    await expect(row).toContainText(/Downloading the (runtime|model) · [\d.]+ MB of [\d.]+ MB/)
    await expect(row).toHaveAttribute('data-stage', 'compiling')
    await expect(row.getByRole('listitem')).toHaveText([
      /Runtime.*done/,
      /Model.*done/,
      /Compile.*in progress/,
      /Warm up.*waiting/,
    ])
    await expect(chip).toContainText('Getting the model ready')
    await expect(chip).not.toContainText('MB')
    await expect(page.getByTestId('load-progress')).toHaveCount(0)
    await expect(chip).toContainText('Ready', { timeout: 15_000 })
    await expect(page.getByTestId('download-tray')).toHaveCount(0, { timeout: 5_000 })
    expect(errors).toEqual([])
  })

  test('stopping while the model is still loading abandons the take cleanly', async ({ page }) => {
    await open(page, '/stt', { storage: fakeModel({ loadMs: 8000 }) })
    await page.getByTestId('mode-browser').click()
    await page.getByTestId('mic').click()
    await expect(page.getByTestId('starting-note')).toHaveAttribute('data-mic', 'live')
    await page.getByTestId('stop').click()
    await expect(page.getByTestId('mic')).toBeVisible()
    await expect(page.getByTestId('starting-note')).toHaveCount(0)
    // The model keeps loading in the background for next time.
    await expect(page.getByTestId('model-chip')).toContainText('Ready', { timeout: 15_000 })
  })

  test('a stuck load is called out with the stage, and a single-threaded retry gets through', async ({ page }) => {
    await open(page, '/stt', { storage: fakeModel({ loadMs: 1500, stall: true }, { stallSeconds: 2 }) })
    await page.getByTestId('mode-browser').click()
    await page.getByTestId('sample-en-fellow-americans').click()
    const trouble = page.getByTestId('load-trouble').first()
    await expect(trouble).toContainText(/Still compiling the model after \d+ s/, { timeout: 15_000 })
    await expect(trouble.getByTestId('trouble-sample')).toBeVisible()
    await expectNoA11yViolations(page)
    await trouble.getByTestId('retry-safe').click()
    await expect(page.getByTestId('model-chip')).toContainText('Ready · WebAssembly · 1 thread', { timeout: 15_000 })
    await expect(page.getByTestId('load-trouble')).toHaveCount(0)
  })

  test('a failed load shows the real error with ways forward', async ({ page }) => {
    await open(page, '/stt', {
      storage: fakeModel({ loadMs: 800, fail: 'no available backend found. ERR: [wasm] CompileError: bad magic' }),
    })
    await page.getByTestId('mode-browser').click()
    await page.getByTestId('sample-en-fellow-americans').click()
    const trouble = page.getByTestId('load-trouble').first()
    await expect(trouble).toContainText('The model could not be loaded')
    await expect(trouble.getByTestId('load-error')).toHaveText(
      'no available backend found. ERR: [wasm] CompileError: bad magic',
    )
    await expect(page.getByTestId('model-chip')).toContainText('Could not load')
    await trouble.getByTestId('retry-safe').click()
    await expect(page.getByTestId('model-chip')).toContainText('Ready', { timeout: 15_000 })
    // And a sample then runs.
    await page.getByTestId('sample-en-fellow-americans').click()
    await expect(page.getByTestId('final-text').first()).toContainText('pretend transcript', { timeout: 20_000 })
  })

  test('an out-of-memory load is retried once on one thread by itself', async ({ page }) => {
    await open(page, '/stt', {
      storage: fakeModel({ loadMs: 800, fail: 'no available backend found. ERR: [wasm] RangeError: Out of memory' }),
    })
    await page.getByTestId('mode-browser').click()
    await page.getByTestId('sample-en-fellow-americans').click()
    await expect(page.getByTestId('model-chip')).toContainText('Ready · WebAssembly · 1 thread', { timeout: 15_000 })
    await expect(page.getByTestId('final-text').first()).toContainText('pretend transcript', { timeout: 20_000 })
  })

  test('pressing down on the microphone button starts fetching the model before the click lands', async ({ page }) => {
    await open(page, '/stt', { storage: fakeModel({ loadMs: 3000 }) })
    await page.getByTestId('mode-browser').click()
    await page.getByTestId('mic').dispatchEvent('pointerdown')
    await expect(page.getByTestId('model-chip')).toHaveAttribute('data-status', 'loading')
    await expect(page.getByTestId('model-chip')).toContainText('Ready', { timeout: 15_000 })
    // Nothing was recorded: it was only a preload.
    await expect(page.getByText('Nothing recorded yet')).toBeVisible()
  })
})

test.describe('?debug=1', () => {
  test.use({ viewport: { width: 1440, height: 900 } })

  test('shows a diagnostics panel that can be copied', async ({ page }) => {
    await page.addInitScript(
      (storage) => {
        for (const [k, v] of Object.entries(storage)) localStorage.setItem(k, v)
        localStorage.setItem('bolo.theme', 'dark')
        localStorage.setItem('bolo.mock', JSON.stringify({ espeak: true }))
      },
      fakeModel({ loadMs: 600 }),
    )
    await page.goto('/?debug=1#/stt')
    const panel = page.getByTestId('diagnostics')
    await expect(panel).toBeVisible()
    const env = panel.getByTestId('diagnostics-env')
    for (const label of [
      'User agent',
      'Secure context',
      'crossOriginIsolated',
      'SharedArrayBuffer',
      'Logical cores',
      'COEP mode',
      'Service worker',
      'Microphone permission',
    ]) {
      await expect(env).toContainText(label)
    }
    await expect(env).toContainText('HeadlessChrome')
    await expect(panel.getByTestId('diagnostics-fetches')).toContainText('None yet')

    await page.getByTestId('mode-browser').click()
    await page.getByTestId('model-chip').click()
    await expect(page.getByTestId('model-chip')).toContainText('Ready', { timeout: 15_000 })
    const fetches = panel.getByTestId('diagnostics-fetches')
    await expect(fetches).toContainText('ort-wasm-simd-threaded.wasm')
    await expect(fetches).toContainText(/200\s*14\.3 MB\s*\d+ ms/)
    await expect(fetches).toContainText('decoder_model_merged_quantized.onnx')
    await expect(panel.getByTestId('diagnostics-engine')).toContainText('ready')
    await expect(panel.getByTestId('diagnostics-events')).toContainText('compiling the model')
    await expect(panel.getByTestId('diagnostics-errors')).toContainText('None')

    await panel.getByTestId('diagnostics-copy').click()
    await expect(panel.getByTestId('diagnostics-copy')).toContainText('Copied')
    const text = await page.evaluate(() => navigator.clipboard.readText())
    expect(text).toContain('bolo diagnostics')
    expect(text).toMatch(/crossOriginIsolated: (yes|no)/)
    expect(text).toContain('ort-wasm-simd-threaded.wasm')
    expect(text).toMatch(/Microphone permission: \w+/)
    await expectNoA11yViolations(page)

    // It stays on while moving between screens, and is absent without the flag.
    await page.goto('/?debug=1#/models')
    await expect(page.getByTestId('diagnostics')).toBeVisible()
  })

  test('errors are recorded: a refused microphone and a failed load', async ({ page }) => {
    await stubMicFailure(page, {
      name: 'NotAllowedError',
      message: 'Permission denied',
      delayMs: 5,
      permission: 'denied',
    })
    await page.addInitScript(
      (storage) => {
        for (const [k, v] of Object.entries(storage)) localStorage.setItem(k, v)
        localStorage.setItem('bolo.mock', JSON.stringify({ espeak: true }))
      },
      fakeModel({ loadMs: 500, fail: 'boom while compiling' }),
    )
    await page.goto('/?debug=1#/stt')
    await page.getByTestId('mode-browser').click()
    await page.getByTestId('mic').click()
    const errors = page.getByTestId('diagnostics-errors')
    await expect(errors).toContainText('microphone: denied: NotAllowedError: Permission denied')
    await page.getByTestId('model-chip').click()
    await expect(errors).toContainText('boom while compiling')
  })

  test('is not there without the flag', async ({ page }) => {
    await open(page, '/stt')
    await expect(page.getByTestId('diagnostics')).toHaveCount(0)
  })
})
