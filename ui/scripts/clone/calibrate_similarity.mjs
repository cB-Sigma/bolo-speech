#!/usr/bin/env node
// Calibrate the "Clone a voice" page's speaker-similarity thresholds.
//
// Embeds every WAV in a directory with the model the page uses (3D-Speaker CAM++ in sherpa-onnx's ONNX form,
// fed WeSpeaker's 80-band filterbank from transformers.js, at the revisions in src/lib/browser-clone/similarity.ts)
// and prints the cosine of every pair: the same voice saying different sentences, and different voices. Clip
// names are `<voice>-<n>.wav` (as make_examples.py --calibration writes them); extra WAVs named
// `clone-<voice>-<anything>.wav` are scored against that voice's clips (a cloned output against its reference).
//
//   node ui/scripts/clone/calibrate_similarity.mjs <clips dir> [--host https://huggingface.co]

import { readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { parseArgs } from 'node:util'
import { AutoFeatureExtractor, env } from '@huggingface/transformers'
import * as ort from 'onnxruntime-web'

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: { host: { type: 'string', default: 'https://huggingface.co' } },
})
const dir = positionals[0]
if (!dir) throw new Error('usage: calibrate_similarity.mjs <clips dir>')

const MODEL = {
  repo: 'csukuangfj/speaker-embedding-models',
  revision: '0743f301363dec56491a490f6d6cbc9d67f9a3bf',
  file: '3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx',
}
const FEATURES = {
  repo: 'onnx-community/wespeaker-voxceleb-resnet34-LM',
  revision: '6a61a1833ff2583aabeba044f5c8221f00b67ceb',
}

function readWav(file) {
  const b = readFileSync(file)
  let o = 12
  let rate = 0
  let pcm = null
  while (o < b.length) {
    const id = b.toString('ascii', o, o + 4)
    const size = b.readUInt32LE(o + 4)
    if (id === 'fmt ') rate = b.readUInt32LE(o + 12)
    if (id === 'data') pcm = new Int16Array(b.buffer.slice(b.byteOffset + o + 8, b.byteOffset + o + 8 + size))
    o += 8 + size + (size & 1)
  }
  const x = Float32Array.from(pcm, (v) => v / 32768)
  if (rate === 16000) return x
  // Linear interpolation is enough for an embedding.
  const n = Math.round((x.length * 16000) / rate)
  return Float32Array.from({ length: n }, (_, i) => {
    const t = (i * (x.length - 1)) / Math.max(1, n - 1)
    const j = Math.floor(t)
    return x[j] + (x[Math.min(x.length - 1, j + 1)] - x[j]) * (t - j)
  })
}

env.remoteHost = values.host.replace(/\/?$/, '/')
const features = await AutoFeatureExtractor.from_pretrained(FEATURES.repo, { revision: FEATURES.revision })
const res = await fetch(`${env.remoteHost}${MODEL.repo}/resolve/${MODEL.revision}/${MODEL.file}`)
if (!res.ok) throw new Error(`HTTP ${res.status} for the CAM++ model`)
const session = await ort.InferenceSession.create(new Uint8Array(await res.arrayBuffer()))

async function embed(x) {
  const { input_features } = await features(x)
  const t = new ort.Tensor('float32', Float32Array.from(input_features.data), input_features.dims)
  const out = await session.run({ [session.inputNames[0]]: t })
  return out[session.outputNames[0]].data
}

const cos = (a, b) => {
  let d = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  return d / Math.sqrt(na * nb)
}

const clips = []
for (const f of readdirSync(dir)
  .filter((f) => f.endsWith('.wav'))
  .sort()) {
  const clone = f.startsWith('clone-')
  const voice = clone ? f.replace(/^clone-/, '').replace(/--.*$/, '') : f.replace(/-\d+\.wav$/, '')
  clips.push({ f, voice, clone, e: await embed(readWav(path.join(dir, f))) })
}

const same = []
const different = []
const cloned = []
for (let i = 0; i < clips.length; i++)
  for (let j = i + 1; j < clips.length; j++) {
    const a = clips[i]
    const b = clips[j]
    const s = cos(a.e, b.e)
    if (a.clone || b.clone) {
      const [c, r] = a.clone ? [a, b] : [b, a]
      if (!r.clone && r.voice === c.voice) cloned.push([c.f, r.f, s])
    } else (a.voice === b.voice ? same : different).push([a.f, b.f, s])
  }
const show = (name, rows) => {
  if (!rows.length) return
  const v = rows.map((r) => r[2])
  console.log(`${name}: ${rows.length} pairs, min ${Math.min(...v).toFixed(3)}, max ${Math.max(...v).toFixed(3)}`)
  for (const [a, b, s] of rows.sort((x, y) => y[2] - x[2])) console.log(`  ${s.toFixed(3)}  ${a} · ${b}`)
}
show('same voice', same)
show('different voices', different)
show('cloned output vs its reference voice', cloned)
