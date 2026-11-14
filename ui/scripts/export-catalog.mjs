// Writes a read-only snapshot of a server's model catalog for the live demo build.
//
//   node scripts/export-catalog.mjs --base http://127.0.0.1:8000
//
// The demo has no server, so its Models screen lists this snapshot as
// "available when you run the server". Machine-specific state is stripped:
// install and load state, and `fit` (how well a model suits the exporting
// server's hardware), which would say nothing about a visitor's browser.
import { writeFileSync } from 'node:fs'
import path from 'node:path'
import { parseArgs } from 'node:util'

const { values } = parseArgs({ options: { base: { type: 'string', default: 'http://127.0.0.1:8000' } } })
const [info, catalog] = await Promise.all([
  fetch(`${values.base}/api/v1/info`).then((r) => r.json()),
  fetch(`${values.base}/api/v1/catalog`).then((r) => r.json()),
])
const models = catalog.models
  .filter((m) => !m.custom && m.engine !== 'fake')
  .map((m) => ({
    id: m.id,
    aliases: m.aliases,
    task: m.task,
    engine: m.engine,
    family: m.family,
    languages: m.languages,
    tier: m.tier,
    streaming: m.streaming,
    sample_rate: m.sample_rate,
    size_bytes: m.size_bytes,
    params: m.params,
    license: m.license,
    commercial_use: m.commercial_use ?? null,
    gated: m.gated,
    voices: m.voices.length > 12 ? m.voices.slice(0, 12) : m.voices,
    features: m.features,
    description: m.description,
    source_url: m.source_url,
    validated: m.validated,
    measured: m.measured ?? null,
    requires: m.requires ?? [],
  }))
const out = path.resolve(import.meta.dirname, '../src/demo/catalog-snapshot.json')
writeFileSync(
  out,
  JSON.stringify({ server_version: info.version, exported: new Date().toISOString().slice(0, 10), models }) + '\n',
)
console.log(`wrote ${models.length} models to ${out}`)
