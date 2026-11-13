import { readFileSync } from 'node:fs'
import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv, type Plugin } from 'vite'

// Cross-origin isolation lets the in-browser recogniser use WebAssembly threads.
// The bolo server should send the same two headers for the console (see docs/ui-server-notes.md).
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
}

/**
 * onnxruntime-web references its ~27 MB WebAssembly binary with
 * `new URL(..., import.meta.url)`, which would copy it into the bundle. The
 * in-browser recogniser always loads the runtime from a configured URL
 * instead (see src/lib/browser-stt/worker.ts), so the copy is dead weight.
 */
function dropOrtWasm(): Plugin {
  return {
    name: 'bolo:drop-ort-wasm',
    generateBundle(_options, bundle) {
      for (const name of Object.keys(bundle)) {
        if (/ort-wasm[^/]*\.wasm$/.test(name)) delete bundle[name]
      }
    },
  }
}

/** Where the live demo is published; used for absolute social-preview URLs. */
const DEMO_URL = 'https://cb-sigma.github.io/bolo/'

/**
 * Extras for the static live demo (`vite build --mode demo`): the service
 * worker that provides cross-origin isolation and offline use, and absolute
 * Open Graph URLs (link previews do not resolve relative ones).
 */
function demoSite(): Plugin {
  return {
    name: 'bolo:demo-site',
    generateBundle(_options, bundle) {
      // Everything except the page itself is precached by the service worker (about 2 MB).
      const files = Object.keys(bundle)
        // Fonts for other scripts are fetched (and then cached) only by pages that show them.
        .filter((f) => f !== 'index.html' && !/noto-(sans|naskh)-/.test(f))
        .concat('favicon.svg')
        .sort()
      const source = readFileSync(path.resolve(import.meta.dirname, 'src/demo/sw.js'), 'utf8').replace(
        '/* __PRECACHE__ */ []',
        JSON.stringify(files.map((f) => `./${f}`)),
      )
      this.emitFile({ type: 'asset', fileName: 'sw.js', source })
      // Tell GitHub Pages not to run the output through Jekyll.
      this.emitFile({ type: 'asset', fileName: '.nojekyll', source: '' })
    },
    transformIndexHtml(html) {
      return html
        .replace('content="./og.png"', `content="${DEMO_URL}og.png"`)
        .replace(
          '<meta name="twitter:card"',
          `<meta property="og:url" content="${DEMO_URL}" />\n    <meta name="twitter:image" content="${DEMO_URL}og.png" />\n    <meta name="twitter:card"`,
        )
        .replace('<title>bolo</title>', '<title>bolo · live demo</title>')
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, import.meta.dirname, ['BOLO_', 'VITE_'])
  const mock = mode === 'mock'
  const demo = mode === 'demo'
  const target = env.BOLO_SERVER || 'http://127.0.0.1:8000'
  const proxy = Object.fromEntries(
    ['/api', '/v1', '/healthz', '/readyz', '/metrics'].map((p) => [p, { target, ws: true, changeOrigin: true }]),
  )
  return {
    // Relative asset URLs: the bundle works at `/` and under any sub-path.
    base: './',
    plugins: [react(), tailwindcss(), dropOrtWasm(), ...(demo ? [demoSite()] : [])],
    resolve: { alias: { '@': path.resolve(import.meta.dirname, './src') } },
    define: mock
      ? { 'import.meta.env.VITE_BOLO_MOCK': JSON.stringify('1') }
      : demo
        ? { 'import.meta.env.VITE_BOLO_DEMO': JSON.stringify('1') }
        : {},
    worker: { format: 'es' as const, plugins: () => [dropOrtWasm()] },
    server: { host: '127.0.0.1', headers: isolation, proxy: mock || demo ? undefined : proxy },
    preview: { host: '127.0.0.1', headers: isolation },
    build: {
      outDir: mock ? 'dist-mock' : demo ? 'dist-demo' : path.resolve(import.meta.dirname, '../server/src/bolo/_ui'),
      emptyOutDir: true,
      sourcemap: false,
      chunkSizeWarningLimit: 900,
    },
  }
})
