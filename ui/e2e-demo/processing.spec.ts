import path from 'node:path'
import { expect, test, type Page } from '@playwright/test'
import { MIRROR, openDemo, overflow } from './open'

const EN = path.resolve(import.meta.dirname, '../src/mock/assets/en-ask-not.wav')
const HI = path.resolve(import.meta.dirname, '../src/demo/assets/hi-short.wav')

/** Pick `option` in the select labelled `name` inside `scope`. */
async function choose(page: Page, scope: string, name: string, option: string | RegExp) {
  await page.getByTestId(scope).getByRole('combobox', { name }).click()
  await page.getByRole('option', { name: option }).click()
}

test('the repetition guard is set like the server’s max_repeats and kept', async ({ page }) => {
  await openDemo(page, 'stt')
  const guard = page.getByTestId('repeat-guard')
  await expect(guard).toContainText('max_repeats')
  const select = guard.getByRole('combobox', { name: 'Repetition guard' })
  await expect(select).toContainText('4 (default)')
  await select.click()
  await page.getByRole('option', { name: 'Off' }).click()
  await expect(select).toContainText('Off')
  await page.reload()
  await expect(page.getByTestId('repeat-guard').getByRole('combobox')).toContainText('Off')
  await page.setViewportSize({ width: 390, height: 900 })
  expect(await overflow(page)).toBeLessThanOrEqual(1)
})

test.describe('with the recogniser available', () => {
  test.skip(!MIRROR, 'set BOLO_E2E_MODEL_DIR to a local model mirror to run')

  test('a sample goes through the guard and comes out unchanged when nothing repeats', async ({ page }) => {
    await openDemo(page, 'stt')
    await page.getByTestId('hero-sample').click()
    await expect(page.getByTestId('mic')).toBeVisible({ timeout: 120_000 })
    await expect(page.getByTestId('transcript')).toContainText(/my fellow americans/i)
    await expect(page.getByTestId('repeats-removed')).toHaveCount(0)
  })

  test('denoise first: GTCRN runs before the recogniser, with the take before and after', async ({ page }) => {
    const errors: string[] = []
    page.on('pageerror', (e) => errors.push(e.message))
    await openDemo(page, 'stt')
    await choose(page, 'denoise-option', 'Denoiser', /^GTCRN/)
    await expect(page.getByTestId('denoise-strength')).toBeVisible()
    await page.getByTestId('hero-sample').click()
    await expect(page.getByTestId('mic')).toBeVisible({ timeout: 120_000 })
    await expect(page.getByTestId('transcript')).toContainText(/fellow americans/i)
    const stats = page.getByTestId('denoise-stats')
    await expect(stats).toContainText('gtcrn-simple')
    await expect(stats).toContainText('WebAssembly')
    console.log(`[denoise] ${await stats.innerText()}`)
    await page.getByTestId('play-denoised').click()
    await page.getByTestId('play-original').click()
    expect(errors).toEqual([])
  })

  test('language identification sends a clip to a recogniser that speaks it, or suggests one', async ({ page }) => {
    await openDemo(page, 'stt')
    await choose(page, 'lid-option', 'Language identifier', /^VoxLingua107/)
    await page.getByTestId('hero-sample').click()
    const result = page.getByTestId('lid-result')
    await expect(result).toContainText('English', { timeout: 120_000 })
    await expect(page.getByTestId('lid-route')).toContainText('auto -> moonshine-tiny; reason=kept; language=en')
    await expect(page.getByTestId('mic')).toBeVisible({ timeout: 120_000 })
    // Hindi, with only an English recogniser downloaded: a Hindi one is suggested, not fetched.
    await page.getByLabel('Audio file').setInputFiles(HI)
    await expect(page.getByTestId('lid-route')).toContainText('language=hi', { timeout: 60_000 })
    await expect(page.getByTestId('lid-route')).toContainText('recommended=whisper-tiny-hindi')
    await page.getByTestId('lid-suggest').click()
    await expect(page.getByRole('dialog')).toContainText(/Hindi/)
  })

  test('compare two voices with the server’s embedder, in the page', async ({ page }) => {
    await openDemo(page, 'speakers')
    const panel = page.getByTestId('voice-compare')
    await expect(panel).toContainText('Speaker embeddings computed in this page')
    await panel.getByTestId('compare-a').getByLabel('First voice: recording').setInputFiles(EN)
    await panel.getByTestId('compare-b').getByLabel('Second voice: recording').setInputFiles(HI)
    await panel.getByTestId('compare-run').click()
    const result = panel.getByTestId('compare-result')
    await expect(result).toContainText('eres2net-voxceleb', { timeout: 120_000 })
    await expect(result).toContainText(/different|similar/)
    console.log(`[compare] ${(await result.innerText()).replace(/\s+/g, ' ')}`)
    // The same file twice is the same speaker.
    await panel.getByTestId('compare-b').getByLabel('Second voice: recording').setInputFiles(EN)
    await panel.getByTestId('compare-run').click()
    await expect(result).toContainText('same speaker')
    await page.setViewportSize({ width: 390, height: 900 })
    expect(await overflow(page)).toBeLessThanOrEqual(1)
  })

  test('the conversation listens with Silero VAD v5 and asks Smart Turn at each pause', async ({ page }) => {
    await page.addInitScript(() => {
      // Scripted recogniser, language model and voice; the detector and the end-of-turn model are real.
      localStorage.setItem('bolo.mock.browserStt', JSON.stringify({ loadMs: 200 }))
      localStorage.setItem('bolo.mock.llm', '1')
      localStorage.setItem('bolo.mock.tts', '1')
    })
    await openDemo(page, 'conversation')
    await choose(page, 'vad-option', 'Voice-activity detector', /^Silero VAD v5/)
    await choose(page, 'turn-option', 'End-of-turn model', /^Smart Turn v3\.2(?! \()/)
    await page.getByTestId('conversation-start').click()
    const turns = page.getByTestId('conversation-turns')
    await expect(turns.locator('[data-role="user"]').first()).toContainText('pretend transcript', { timeout: 90_000 })
    const checks = page.getByTestId('turn-checks')
    await expect(checks).toContainText(/finished|waiting|pause ran past/)
    console.log(`[turn] ${(await checks.innerText()).replace(/\n/g, ' | ')}`)
    await page.getByTestId('conversation-stop').click()
    await page.goto('./#/models')
    const loaded = page.getByTestId('loaded-models')
    await expect(loaded.locator('[data-model="audio-silero-vad-v5"]')).toContainText('vad')
    await expect(loaded.locator('[data-model="audio-smart-turn-v3"]')).toContainText('turn')
  })
})
