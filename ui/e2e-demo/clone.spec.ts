import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import AxeBuilder from '@axe-core/playwright'
import { expect, test, type Page } from '@playwright/test'

/**
 * The "Clone a voice" page of the live demo. Scripted workers stand in for the models
 * (`bolo.mock.clone`; see src/lib/browser-clone/fake-workers.ts) and a scripted recogniser fills in the
 * transcript (`bolo.mock.browserStt`), so these run without downloads. With BOLO_E2E_MODEL_DIR pointing at a
 * mirror holding Pocket TTS (English), MOSS-TTS-Nano with its codec, CAM++ and Moonshine Tiny, two tests run the
 * real models.
 */

const MIRROR = Boolean(process.env.BOLO_E2E_MODEL_DIR)
const CDN = 'http://127.0.0.1:4175'

const UCCHARAN_DATA = path.resolve(import.meta.dirname, '../src/vendor/uccharan/data')

/**
 * uccharan's language data, as the Pages build publishes it under voices/uccharan/ (CI has no voices). On the
 * context: the page's service worker (cross-origin isolation) makes the fetches.
 */
async function serveUccharanData(page: Page) {
  await page.context().route('**/voices/uccharan/*.ucd', (route) => {
    const file = path.join(UCCHARAN_DATA, path.basename(new URL(route.request().url()).pathname))
    return existsSync(file)
      ? route.fulfill({ body: readFileSync(file), contentType: 'application/octet-stream' })
      : route.fulfill({ status: 404, body: 'not found' })
  })
}

async function open(page: Page, opts: { fake?: string | null; theme?: 'dark' | 'light' } = {}) {
  await serveUccharanData(page)
  await page.addInitScript(
    ([theme, cdn, fake]) => {
      localStorage.setItem('bolo.theme', theme as string)
      localStorage.setItem('bolo.browserStt', JSON.stringify({ modelHost: cdn, ortWasmBase: `${cdn}/ort/` }))
      if (fake) {
        localStorage.setItem('bolo.mock.clone', fake)
        localStorage.setItem('bolo.mock.browserStt', '1')
      }
    },
    [opts.theme ?? 'dark', CDN, opts.fake === undefined ? '1' : opts.fake],
  )
  await page.goto('./#/clone')
  await expect(page.getByTestId('demo-banner')).toBeVisible()
  await expect.poll(() => page.evaluate(() => crossOriginIsolated), { timeout: 20_000 }).toBe(true)
  await expect(page.getByTestId('clone-examples').getByRole('button').first()).toBeVisible()
}

async function axe(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'])
    .analyze()
  expect(results.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(', ')}`)).toEqual([])
}

async function noSideScroll(page: Page) {
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  )
  expect(overflow).toBeLessThanOrEqual(1)
}

test.describe('clone a voice in the page', () => {
  test('example voice → generate → streamed audio, latency, similarity, WAV; listed as downloaded and loaded', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 900 })
    await open(page)
    await expect(page.getByTestId('clone-privacy')).toContainText('never uploaded')
    await expect(page.getByTestId('clone-examples')).toContainText('Synthetic')
    await page.getByTestId('clone-example-syspin-en-female').click()
    await expect(page.getByTestId('clone-chosen')).toContainText('Indian English')
    await expect(page.getByTestId('clone-transcript')).toHaveValue(/Good morning/)
    await expect(page.getByTestId('clone-model-facts')).toContainText('CC-BY-4.0')
    await page.getByTestId('clone-generate').click()
    await expect(page.getByTestId('clone-output')).toBeVisible()
    await expect(page.getByTestId('clone-rtf')).toHaveAttribute('data-rtf', /\d/)
    await expect(page.getByTestId('clone-ttfa')).toContainText(' s')
    await expect(page.getByTestId('similarity')).toHaveAttribute('data-verdict', /same|unclear|different/)
    const download = page.waitForEvent('download')
    await page.getByTestId('clone-download').click()
    expect((await download).suggestedFilename()).toBe('bolo-clone-pocket-en.wav')
    await page.getByTestId('clone-check').click()
    await expect(page.getByTestId('clone-roundtrip')).toHaveAttribute('data-wer', /\d/)
    await noSideScroll(page)
    await axe(page)
    await expect(page.getByTestId('downloads-panel')).toContainText('Voice cloning')
    await page.goto('./#/models')
    await expect(page.getByTestId('loaded-models').getByTestId('loaded-model').first()).toBeVisible()
    await expect(page.getByTestId('loaded-models')).toContainText('clone-pocket-en')
  })

  test('own recording needs consent; upload, transcript filled in and editable; save, reuse and delete', async ({
    page,
  }) => {
    await open(page)
    await expect(page.getByTestId('clone-record')).toBeDisabled()
    await expect(page.getByTestId('clone-upload')).toBeDisabled()
    await page.getByTestId('clone-consent').click()
    await expect(page.getByTestId('clone-record')).toBeEnabled()
    await page
      .getByTestId('clone-upload')
      .setInputFiles(path.resolve(import.meta.dirname, '../src/mock/assets/en-ask-not.wav'))
    await expect(page.getByTestId('clone-ref-seconds')).toContainText(/1\d\.\d s/)
    await expect(page.getByTestId('clone-transcript')).not.toHaveValue('', { timeout: 20_000 })
    await page.getByTestId('clone-transcript').fill('Ask not what your country can do for you.')
    await page.getByTestId('clone-voice-name').fill('Test voice')
    await page.getByTestId('clone-save').click()
    await expect(page.getByTestId('clone-saved')).toContainText('Test voice')
    // Kept across a reload (IndexedDB), with its transcript.
    await page.reload()
    await expect(page.getByTestId('clone-saved')).toContainText('Test voice')
    await page.getByTestId('clone-use-saved').click()
    await expect(page.getByTestId('clone-transcript')).toHaveValue('Ask not what your country can do for you.')
    await page.getByTestId('clone-generate').click()
    await expect(page.getByTestId('clone-output')).toBeVisible()
    await page.getByTestId('clone-delete-saved').click()
    await expect(page.getByTestId('clone-saved')).toHaveCount(0)
  })

  test('records from the microphone after consent', async ({ page }) => {
    await open(page)
    await page.getByTestId('clone-consent').click()
    await page.getByTestId('clone-record').click()
    await expect(page.getByTestId('clone-record')).toContainText('Stop')
    await page.waitForTimeout(6_500)
    await page.getByTestId('clone-record').click()
    await expect(page.getByTestId('clone-chosen')).toContainText('My recording')
    await expect(page.getByTestId('clone-ref-seconds')).toContainText(' s')
  })

  test('model picker: grouped by size, verdicts from the device probe, switching unloads', async ({ page }) => {
    await page.addInitScript(() =>
      localStorage.setItem(
        'bolo.fakeDevice',
        JSON.stringify({
          platform: 'android',
          memoryGb: 3,
          webgpu: {
            available: false,
            f16: false,
            maxBufferSize: null,
            maxStorageBufferBindingSize: null,
            adapter: null,
          },
        }),
      ),
    )
    await open(page)
    await page.getByTestId('clone-model-select').click()
    await expect(page.getByRole('listbox')).toContainText('Small: phones too')
    await expect(page.getByRole('listbox')).toContainText('Large: a desktop, WebGPU advised')
    await expect(page.getByTestId('clone-model-pocket-en').getByTestId('verdict-badge')).toHaveCount(0)
    await expect(page.getByTestId('clone-model-chatterbox-multilingual').getByTestId('verdict-badge')).toHaveAttribute(
      'data-level',
      'unlikely',
    )
    await page.getByTestId('clone-model-chatterbox-turbo').click()
    await expect(page.getByTestId('clone-model-facts')).toContainText('no watermark')
    await expect(page.getByTestId('verdict-line')).toBeVisible()
    // Not blocked: the risk is explained and the visitor decides.
    await page.getByTestId('clone-example-kokoro-am-michael').click()
    await page.getByTestId('clone-generate').click()
    await expect(page.getByTestId('risk-dialog')).toBeVisible()
    await page.getByTestId('risk-confirm').click()
    await expect(page.getByTestId('clone-output')).toBeVisible()
    await page.getByTestId('clone-model-select').click()
    await page.getByTestId('clone-model-pocket-en').click()
    await expect(page.getByTestId('clone-status')).toContainText('downloads on first use')
  })

  test('MOSS-TTS-Nano: picker entry and facts, 20 languages, long reference cut, streamed clip, WAV; listed', async ({
    page,
  }) => {
    await open(page)
    await expect(page.getByTestId('clone-not-offered')).not.toContainText('MOSS')
    await page.getByTestId('clone-model-select').click()
    const entry = page.getByTestId('clone-model-moss-tts-nano')
    await expect(entry).toContainText('Apache-2.0')
    await expect(entry).toContainText('20 languages')
    await entry.click()
    const facts = page.getByTestId('clone-model-facts')
    await expect(facts).toContainText('OpenMOSS')
    await expect(facts).toContainText('Persian')
    await expect(facts).toContainText('No transcript needed')
    await expect(page.getByTestId('clone-status')).toContainText('downloads on first use')
    // A 15 s example: the model takes its first 12 s, and says so.
    await page.getByTestId('clone-example-syspin-hi-male').click()
    await expect(page.getByTestId('clone-ref-cut')).toContainText('first 12 s')
    await page.getByTestId('clone-language').click()
    await page.getByRole('option', { name: 'Chinese' }).click()
    await page.getByTestId('clone-text').fill('欢迎关注模思智能。')
    await page.getByTestId('clone-generate').click()
    await expect(page.getByTestId('clone-output')).toBeVisible()
    await expect(page.getByTestId('clone-output')).toContainText('Played as it was made')
    await expect(page.getByTestId('clone-rtf')).toHaveAttribute('data-rtf', /\d/)
    const download = page.waitForEvent('download')
    await page.getByTestId('clone-download').click()
    expect((await download).suggestedFilename()).toBe('bolo-clone-moss-tts-nano.wav')
    await expect(page.getByTestId('downloads-panel')).toContainText('MOSS-TTS-Nano')
    await page.goto('./#/models')
    await expect(page.getByTestId('loaded-models')).toContainText('clone-moss-tts-nano')
  })

  test('choosing an example voice fills in its sentence only when the text box is empty or untouched', async ({
    page,
  }) => {
    await open(page)
    await page.getByTestId('clone-model-select').click()
    await page.getByTestId('clone-model-chatterbox-multilingual').click()
    const box = page.getByTestId('clone-text')
    await expect(box).toHaveValue(/^Thanks for calling/)
    // The English default is untouched: the Hindi example puts its own sentence in.
    await page.getByTestId('clone-example-syspin-hi-male').click()
    await expect(page.getByTestId('clone-chosen')).toBeVisible()
    await expect(box).toHaveValue(/^नमस्ते/)
    // Typed text stays, whichever example comes next.
    await box.fill('My own words, please keep them.')
    await page.getByTestId('clone-example-syspin-en-female').click()
    await expect(page.getByTestId('clone-chosen')).toContainText('Indian English')
    await expect(box).toHaveValue('My own words, please keep them.')
    await page.getByTestId('clone-example-syspin-hi-male').click()
    await expect(page.getByTestId('clone-ref-seconds')).toContainText('Hindi')
    await expect(box).toHaveValue('My own words, please keep them.')
    // An emptied box gets the example's sentence again.
    await box.fill('')
    await page.getByTestId('clone-example-syspin-en-female').click()
    await expect(box).toHaveValue(/^Thanks for calling/)
  })

  test('numbers are written out in words before the model reads them, and shown; not where uccharan has no normaliser', async ({
    page,
  }) => {
    // The service worker precaches the whole bundle (uccharan's module included) in the background; the page
    // itself asks for uccharan, and the service worker for its language data, only when a clip is generated.
    const uccharanRequests: string[] = []
    page.on('request', (r) => {
      if (/uccharan/i.test(r.url())) uccharanRequests.push(r.url())
    })
    page.context().on('request', (r) => {
      if (r.url().endsWith('.ucd')) uccharanRequests.push(r.url())
    })
    await open(page)
    await expect(page.getByTestId('clone-numbers')).toContainText('written out in words by uccharan')
    await page.getByTestId('clone-example-kokoro-am-michael').click()
    await expect(page.getByTestId('clone-chosen')).toBeVisible()
    expect(uccharanRequests).toEqual([])
    await page.getByTestId('clone-text').fill('Your table for 4 is booked at 10:30 on 3 May; the deposit is $25.')
    await page.getByTestId('clone-generate').click()
    await expect(page.getByTestId('clone-output')).toBeVisible()
    const spoken = 'Your table for four is booked at ten thirty on three May; the deposit is twenty five dollars.'
    await expect(page.getByTestId('clone-spoken')).toContainText(`Spoken as: “${spoken}”`)
    expect(await page.evaluate(() => (globalThis as { __boloCloneTexts?: string[] }).__boloCloneTexts)).toEqual([
      spoken,
    ])
    expect(uccharanRequests.some((u) => u.endsWith('/voices/uccharan/en-us.ucd'))).toBe(true)
    expect(uccharanRequests.some((u) => /\/voices\/uccharan\/(?!en-us)/.test(u))).toBe(false)
    // German: its own word table (and the English one, for English words in German text).
    await page.getByTestId('clone-model-select').click()
    await page.getByTestId('clone-model-moss-tts-nano').click()
    await expect(page.getByTestId('clone-numbers')).toContainText('Chinese they go to the model as written')
    await page.getByTestId('clone-language').click()
    await page.getByRole('option', { name: 'German' }).click()
    await page.getByTestId('clone-text').fill('Ihr Termin am 3. Mai um 10:30 Uhr kostet 25 €.')
    await page.getByTestId('clone-generate').click()
    await expect(page.getByTestId('clone-spoken')).toHaveAttribute(
      'data-spoken',
      'Ihr Termin am dritten Mai um zehn dreißig Uhr kostet fünfundzwanzig Euro.',
    )
    expect(uccharanRequests.some((u) => u.endsWith('/voices/uccharan/de.ucd'))).toBe(true)
    // Chinese: no normaliser, the digits go as written and the page says so.
    await page.getByTestId('clone-language').click()
    await page.getByRole('option', { name: 'Chinese' }).click()
    await page.getByTestId('clone-text').fill('会议在10点30分开始。')
    await page.getByTestId('clone-generate').click()
    await expect(page.getByTestId('clone-spoken')).toHaveAttribute('data-skipped', 'language')
    await expect(page.getByTestId('clone-spoken')).toContainText('no normaliser for Chinese')
    expect(await page.evaluate(() => (globalThis as { __boloCloneTexts?: string[] }).__boloCloneTexts?.at(-1))).toBe(
      '会议在10点30分开始。',
    )
    // Heard back: scored in words on both sides.
    await page.getByTestId('clone-check').click()
    await expect(page.getByTestId('clone-roundtrip')).toHaveAttribute('data-wer', /\d/)
  })

  test('if the language data cannot be fetched, the text goes as written and the page says so', async ({ page }) => {
    await open(page)
    await page.context().route('**/voices/uccharan/*.ucd', (route) => route.fulfill({ status: 404, body: 'not found' }))
    await page.getByTestId('clone-example-kokoro-am-michael').click()
    await page.getByTestId('clone-text').fill('See you at 10:30.')
    await page.getByTestId('clone-generate').click()
    await expect(page.getByTestId('clone-spoken')).toHaveAttribute('data-skipped', 'failed')
    await expect(page.getByTestId('clone-spoken')).toContainText('went to the model as written')
    expect(await page.evaluate(() => (globalThis as { __boloCloneTexts?: string[] }).__boloCloneTexts?.at(-1))).toBe(
      'See you at 10:30.',
    )
  })

  test('out of memory: unloads the rest and retries on one CPU thread', async ({ page }) => {
    await open(page, { fake: 'oom' })
    await page.getByTestId('clone-example-syspin-en-female').click()
    await page.getByTestId('clone-generate').click()
    await expect(page.getByTestId('clone-output')).toBeVisible()
    await expect(page.getByTestId('clone-oom')).toHaveCount(0)
    // From now on one model runtime at a time: scoring the similarity unloads the cloning model.
    await expect(page.getByTestId('similarity')).toHaveAttribute('data-verdict', /./)
    await expect(page.getByTestId('clone-status')).toContainText('downloads on first use')
  })

  test('Chatterbox Turbo: a failed WebGPU run is shown and the clip is made on the CPU instead', async ({ page }) => {
    // A desktop with WebGPU, so Auto puts Chatterbox there; the scripted worker fails every WebGPU generation the
    // way Chrome on Windows did (ONNX Runtime's BufferManager::Download error from the decoder).
    await page.addInitScript(() =>
      localStorage.setItem(
        'bolo.fakeDevice',
        JSON.stringify({
          platform: 'desktop',
          memoryGb: 16,
          webgpu: {
            available: true,
            f16: true,
            maxBufferSize: 2 ** 31,
            maxStorageBufferBindingSize: 2 ** 30,
            adapter: 'amd · gcn-5',
          },
        }),
      ),
    )
    await open(page, { fake: 'gpu-fail' })
    await page.getByTestId('clone-model-select').click()
    await page.getByTestId('clone-model-chatterbox-turbo').click()
    // Both downloads are stated before anything is fetched: the CPU's files are not the WebGPU ones.
    await expect(page.getByTestId('clone-size')).toContainText('725 MB download on the CPU, 565 MB on WebGPU')
    await expect(page.getByTestId('clone-status')).toContainText('565 MB for WebGPU')
    // Your device is on this page too, with the GPU the browser gave.
    await expect(page.getByTestId('device-panel')).toBeVisible()
    await expect(page.getByTestId('gpu-in-use')).toContainText('GPU in use')
    await expect(page.getByTestId('gpu-in-use')).toContainText('amd · gcn-5')
    await page.getByTestId('clone-example-kokoro-am-michael').click()
    await page.getByTestId('clone-generate').click()
    await expect(page.getByTestId('clone-output')).toBeVisible()
    await expect(page.getByTestId('clone-notice')).toContainText('Generation failed on WebGPU: failed to call OrtRun()')
    await expect(page.getByTestId('clone-notice')).toContainText('Made on the CPU instead')
    await expect(page.getByTestId('clone-status')).toContainText('ready on the CPU')
    await expect(page.getByTestId('clone-error')).toHaveCount(0)
    await noSideScroll(page)
    await axe(page)
  })

  test('light theme at phone width: no side scroll, accessible', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 900 })
    await open(page, { theme: 'light' })
    await page.getByTestId('clone-consent').click()
    await page.getByTestId('clone-example-syspin-hi-male').click()
    await expect(page.getByTestId('clone-chosen')).toBeVisible()
    await noSideScroll(page)
    await axe(page)
  })

  test.describe('with the models available', () => {
    test.skip(
      !MIRROR,
      'set BOLO_E2E_MODEL_DIR to a mirror with Pocket TTS, MOSS-TTS-Nano, CAM++ and Moonshine Tiny to run',
    )
    test('real Pocket TTS: intelligible, and close to the example voice', async ({ page }) => {
      test.setTimeout(600_000)
      await open(page, { fake: null })
      await page.getByTestId('clone-placement').click()
      await page.getByRole('option', { name: 'CPU' }).click()
      await page.getByTestId('clone-example-kokoro-am-michael').click()
      await page.getByTestId('clone-generate').click()
      await expect(page.getByTestId('clone-output')).toBeVisible({ timeout: 300_000 })
      await expect(page.getByTestId('similarity')).toHaveAttribute('data-score', /\d/, { timeout: 120_000 })
      const score = Number(await page.getByTestId('similarity').getAttribute('data-score'))
      await page.getByTestId('clone-check').click()
      await expect(page.getByTestId('clone-roundtrip')).toHaveAttribute('data-wer', /\d/, { timeout: 300_000 })
      const wer = Number(await page.getByTestId('clone-roundtrip').getAttribute('data-wer'))
      console.log(`[clone] pocket-en: similarity ${score.toFixed(3)}, WER ${(100 * wer).toFixed(1)}%`)
      expect(score).toBeGreaterThan(0.5)
      expect(wer).toBeLessThan(0.2)
    })

    test('real MOSS-TTS-Nano: intelligible, and close to the example voice', async ({ page }) => {
      test.setTimeout(900_000)
      await open(page, { fake: null })
      await page.getByTestId('clone-model-select').click()
      await page.getByTestId('clone-model-moss-tts-nano').click()
      await page.getByTestId('clone-placement').click()
      await page.getByRole('option', { name: 'CPU' }).click()
      await page.getByTestId('clone-example-kokoro-am-michael').click()
      await page.getByTestId('clone-generate').click()
      await expect(page.getByTestId('clone-output')).toBeVisible({ timeout: 600_000 })
      await expect(page.getByTestId('similarity')).toHaveAttribute('data-score', /\d/, { timeout: 120_000 })
      const score = Number(await page.getByTestId('similarity').getAttribute('data-score'))
      const rtf = Number(await page.getByTestId('clone-rtf').getAttribute('data-rtf'))
      await page.getByTestId('clone-check').click()
      await expect(page.getByTestId('clone-roundtrip')).toHaveAttribute('data-wer', /\d/, { timeout: 300_000 })
      const wer = Number(await page.getByTestId('clone-roundtrip').getAttribute('data-wer'))
      console.log(`[clone] moss-tts-nano: similarity ${score.toFixed(3)}, WER ${(100 * wer).toFixed(1)}%, RTF ${rtf}`)
      expect(score).toBeGreaterThan(0.5)
      // Moonshine Tiny writes "ten" as "10": one word of 22.
      expect(wer).toBeLessThan(0.2)
    })
  })
})
