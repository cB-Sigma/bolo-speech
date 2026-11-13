/**
 * Registers the demo's service worker and, the first time, reloads the page
 * once so that it comes back cross-origin isolated (see sw.js). Returns true
 * when a reload is under way and the app should not bother rendering.
 *
 * Without a service worker (private windows in some browsers, plain http)
 * the demo still works, on a single WebAssembly thread.
 */
const FLAG = 'bolo.coi.reloaded'

export async function ensureIsolation(): Promise<boolean> {
  if (!('serviceWorker' in navigator) || !globalThis.isSecureContext) return false
  try {
    const chromium = 'userAgentData' in navigator || 'chrome' in globalThis
    const registration = await navigator.serviceWorker.register(
      `./sw.js?coep=${chromium ? 'credentialless' : 'require-corp'}`,
      { scope: './' },
    )
    if (globalThis.crossOriginIsolated) {
      sessionStorage.removeItem(FLAG)
      return false
    }
    // One reload per tab at most: if isolation still fails afterwards, carry on without it.
    if (sessionStorage.getItem(FLAG)) return false
    await Promise.race([
      new Promise<void>((resolve) => {
        if (navigator.serviceWorker.controller) resolve()
        else navigator.serviceWorker.addEventListener('controllerchange', () => resolve(), { once: true })
        void registration.update().catch(() => undefined)
      }),
      new Promise<void>((resolve) => setTimeout(resolve, 4000)),
    ])
    if (!navigator.serviceWorker.controller) return false
    sessionStorage.setItem(FLAG, '1')
    globalThis.location.reload()
    return true
  } catch {
    return false
  }
}
