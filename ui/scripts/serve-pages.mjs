// A static server that behaves like GitHub Pages for local checks of the live demo:
// it serves `dist-demo/` under /bolo/ with no special response headers at all.
//
//   npm run build:demo && node scripts/serve-pages.mjs            # http://127.0.0.1:4174/bolo/
//
// With --mirror <dir> it also starts a second origin (port + 1) that serves that
// directory with the same CORS header the model and runtime CDNs send, so the
// in-browser recogniser can be tested without the internet:
//   <dir>/onnx-community/moonshine-tiny-ONNX/resolve/main/...   model files (Hub layout)
//   /ort/...                                                    ONNX Runtime files from node_modules
//
// With --voices <dir> the voices for in-browser speech synthesis (the output of
// scripts/voices/build.py) are served same-origin under <base>voices/, as the
// Pages workflow publishes them.
import { createReadStream, existsSync, statSync } from 'node:fs'
import http from 'node:http'
import path from 'node:path'
import { parseArgs } from 'node:util'

const { values } = parseArgs({
  options: {
    port: { type: 'string', default: '4174' },
    dir: { type: 'string', default: 'dist-demo' },
    base: { type: 'string', default: '/bolo/' },
    mirror: { type: 'string' },
    voices: { type: 'string' },
  },
})
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.wav': 'audio/wav',
  '.woff2': 'font/woff2',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
}

function send(res, file, extra = {}) {
  if (!existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404, { 'content-type': 'text/plain', ...extra }).end('Not found')
    return
  }
  res.writeHead(200, {
    'content-type': TYPES[path.extname(file)] ?? 'application/octet-stream',
    'content-length': statSync(file).size,
    ...extra,
  })
  createReadStream(file).pipe(res)
}

const root = path.resolve(values.dir)
const port = Number(values.port)
http
  .createServer((req, res) => {
    const url = new URL(req.url, 'http://x')
    if (!url.pathname.startsWith(values.base)) {
      res.writeHead(url.pathname === values.base.slice(0, -1) ? 301 : 404, { location: values.base }).end()
      return
    }
    let rel = decodeURIComponent(url.pathname.slice(values.base.length)) || 'index.html'
    if (values.voices && rel.startsWith('voices/')) {
      const dir = path.resolve(values.voices)
      const file = path.join(dir, rel.slice('voices/'.length))
      if (!file.startsWith(dir)) return void res.writeHead(403).end()
      return void send(res, file)
    }
    if (rel.endsWith('/')) rel += 'index.html'
    const file = path.join(root, rel)
    if (!file.startsWith(root)) return void res.writeHead(403).end()
    send(res, file)
  })
  .listen(port, '127.0.0.1', () => console.log(`pages-like server: http://127.0.0.1:${port}${values.base}`))

if (values.mirror) {
  const mirror = path.resolve(values.mirror)
  const ort = path.resolve(import.meta.dirname, '../node_modules/onnxruntime-web/dist')
  const cors = { 'access-control-allow-origin': '*' }
  http
    .createServer((req, res) => {
      const pathname = decodeURIComponent(new URL(req.url, 'http://x').pathname)
      const file = pathname.startsWith('/ort/') ? path.join(ort, path.basename(pathname)) : path.join(mirror, pathname)
      if (!file.startsWith(mirror) && !file.startsWith(ort)) return void res.writeHead(403, cors).end()
      send(res, file, cors)
    })
    .listen(port + 1, '127.0.0.1', () => console.log(`cdn-like mirror:    http://127.0.0.1:${port + 1}/`))
}
