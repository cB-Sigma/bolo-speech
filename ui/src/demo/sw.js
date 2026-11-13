/*
 * Service worker for the bolo live demo on static hosting.
 *
 * 1. Cross-origin isolation. Static hosts such as GitHub Pages cannot send
 *    the COOP/COEP headers that browsers require before they expose
 *    SharedArrayBuffer (and with it WebAssembly threads). This worker adds
 *    them to every response it relays. The technique is the one popularised
 *    by coi-serviceworker (Guido Zuidhof, MIT licence); this is an
 *    independent, smaller implementation.
 * 2. Offline. Same-origin files and the pinned ONNX Runtime files from
 *    jsDelivr are cached as they are fetched, so a second visit works
 *    without a network. Model weights are cached separately by the
 *    recogniser itself (Cache Storage, "transformers-cache").
 */
const CACHE = 'bolo-demo-v1'
// `credentialless` needs no cooperation from other origins but is Chromium-only;
// elsewhere `require-corp` works because the CDNs used send CORS headers.
const COEP = new URL(self.location.href).searchParams.get('coep') || 'credentialless'

// Replaced at build time with the list of files in the bundle, so every screen works offline
// after the first visit, not only the ones already opened.
const PRECACHE = /* __PRECACHE__ */ []

self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE)
      // Best effort: a failed precache must not stop the worker from providing isolation.
      await Promise.allSettled(PRECACHE.map((file) => cache.add(new Request(file, { cache: 'reload' }))))
      await self.skipWaiting()
    })(),
  )
})
self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      for (const key of await caches.keys()) {
        if (key.startsWith('bolo-demo-') && key !== CACHE) await caches.delete(key)
      }
      await self.clients.claim()
    })(),
  )
})

function isolate(response) {
  // Opaque responses cannot be re-wrapped; they pass through untouched.
  if (response.status === 0) return response
  const headers = new Headers(response.headers)
  headers.set('Cross-Origin-Embedder-Policy', COEP)
  headers.set('Cross-Origin-Opener-Policy', 'same-origin')
  headers.set('Cross-Origin-Resource-Policy', 'cross-origin')
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers })
}

async function networkFirst(request) {
  const cache = await caches.open(CACHE)
  try {
    const response = await fetch(request)
    if (response.ok) await cache.put(request, response.clone())
    return response
  } catch (err) {
    const hit = await cache.match(request, { ignoreSearch: request.mode === 'navigate' })
    if (hit) return hit
    throw err
  }
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE)
  const hit = await cache.match(request)
  if (hit) return hit
  const response = await fetch(request)
  if (response.ok) await cache.put(request, response.clone())
  return response
}

self.addEventListener('fetch', (event) => {
  const request = event.request
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return
  const url = new URL(request.url)
  const sameOrigin = url.origin === self.location.origin
  const cacheable = request.method === 'GET' && !request.headers.has('range')
  let work
  if (sameOrigin && cacheable) {
    // Hashed build assets never change; everything else (the page itself) prefers the network.
    work = /\/assets\//.test(url.pathname) ? cacheFirst(request) : networkFirst(request)
  } else if (url.hostname === 'cdn.jsdelivr.net' && cacheable) {
    work = cacheFirst(request)
  } else if (!sameOrigin && request.mode === 'no-cors') {
    work = fetch(new Request(request, { credentials: 'omit' }))
  } else {
    work = fetch(request)
  }
  event.respondWith(work.then(isolate))
})
