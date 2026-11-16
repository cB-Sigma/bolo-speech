import path from 'node:path'
import AxeBuilder from '@axe-core/playwright'
import { expect, test, type Page } from '@playwright/test'

/**
 * The "Clone a voice" page of the live demo. Scripted workers stand in for the models
 * (`bolo.mock.clone`; see src/lib/browser-clone/fake-workers.ts) and a scripted recogniser fills in the
 * transcript (`bolo.mock.browserStt`), so these run without downloads. With BOLO_E2E_MODEL_DIR pointing at a
 * mirror holding Pocket TTS (English), CAM++ and Moonshine Tiny, one test runs the real models.
 */

const MIRROR = Boolean(process.env.BOLO_E2E_MODEL_DIR)
const CDN = 'http://127.0.0.1:4175'

async function open(page: Page, opts: { fake?: string | null; theme?: 'dark' | 'light' } = {}) {
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
    test.skip(!MIRROR, 'set BOLO_E2E_MODEL_DIR to a mirror with Pocket TTS, CAM++ and Moonshine Tiny to run')
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
  })
})
