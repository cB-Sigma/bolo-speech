/// <reference lib="webworker" />
import type * as Ort from 'onnxruntime-web/wasm'
import {
  cachedDownload,
  lowerMemoryMaximum,
  ortCdnBase,
  RUNTIME_CACHE,
  runtimeMemoryBytes,
} from '@/lib/browser-runtime'
import type { FetchRecord, FileProgress, LoadStage } from '@/lib/browser-stt/models'
import { setOrtGpuPower } from '@/lib/webgpu'
import { VOICE_CACHE, type TtsWorkerRequest, type TtsWorkerResponse } from './protocol'
import type * as UccharanModule from '@/vendor/uccharan/index.js'
import { scaleSilence } from './silence'
import { coquiIds, piperConfigFromJson, piperIds, withLeadIn, type VoiceConfig } from './text'
import { DIZIONE_DATA_FILES, finished, limitPeak, shorter, TINY_VOCA_BASE, TINY_VOCA_TOKEN_LIMIT } from './tiny-voca'
import { KITTEN_GAP_S, kaleIds, kittenFold, kittenIds, kittenTrim, termsReplacer } from './small-voices'
import type { AlignTable } from './dizione/align'
import type { Phonemizer as Dizione } from './dizione/engine'
import type { G2PSettings } from './dizione/g2p'
import type { HomographData } from './dizione/homographs'

/**
 * Runs one voice with ONNX Runtime Web (WebAssembly). The VITS graph is called
 * directly: no transformers.js pipeline, no phonemiser for character voices,
 * and uccharan (Apache-2.0) for Piper voices.
 */

const scope = self as unknown as DedicatedWorkerGlobalScope
const post = (msg: TtsWorkerResponse, transfer: Transferable[] = []) => scope.postMessage(msg, transfer)
const stage = (s: LoadStage, loaded?: number, total?: number) => post({ type: 'stage', stage: s, loaded, total })

/** One ONNX Runtime session, or two (front on WebGPU, back on the CPU) behind the same surface. */
interface Runner {
  readonly outputNames: readonly string[]
  run(feeds: Record<string, Ort.Tensor>): Promise<Record<string, Ort.OnnxValue>>
  release(): Promise<void>
}

/**
 * Two halves of one graph, cut at build time (the server's bolo/onnx_split.py): the front runs on WebGPU, the
 * back on the CPU, and the tensors that cross the cut come back to the CPU in between.
 */
function twoPart(front: Ort.InferenceSession, back: Ort.InferenceSession): Runner {
  return {
    outputNames: back.outputNames.length ? back.outputNames : front.outputNames,
    async run(feeds) {
      const frontFeeds = Object.fromEntries(front.inputNames.filter((n) => n in feeds).map((n) => [n, feeds[n]]))
      const mid = await front.run(frontFeeds as Record<string, Ort.Tensor>)
      const backFeeds: Record<string, Ort.OnnxValue> = {}
      for (const n of back.inputNames) backFeeds[n] = (mid[n] ?? feeds[n]) as Ort.OnnxValue
      // A graph output the front already makes is passed on as it is.
      return { ...mid, ...(await back.run(backFeeds)) }
    },
    async release() {
      await front.release()
      await back.release()
    },
  }
}

/** The ONNX Runtime build in use: the WebAssembly-only one, or the one with WebGPU (and WebAssembly). */
let ort: typeof Ort
let session: Runner | null = null
let config: VoiceConfig | null = null
let uccharan: typeof UccharanModule | null = null
let kokoroStyle: Float32Array | null = null
/** tiny-voca's phonemiser (dizione, ported to TypeScript), loaded once per worker. */
let dizione: Dizione | null = null
/** Kitten TTS 2: the voice's style rows (510 x 256), from the language's voice pack. */
let kittenStyle: Float32Array | null = null
/** kale-tts-1: its front end, its id table, the voice's style table and its other inputs. */
let kale: {
  sentences: (text: string) => string[]
  ipa: (sentence: string) => string
  vocab: Record<string, number>
  native: boolean
  british: boolean
  style: Float32Array
  feeds: { name: string; data: Float32Array; dims: number[] }[]
} | null = null
/** The loaded voice's own style, to go back to after a blend. */
let ownStyle: Float32Array | null = null
let chain: Promise<void> = Promise.resolve()

const record = (r: FetchRecord) => post({ type: 'fetch', record: r })

async function loadRuntime(req: Extract<TtsWorkerRequest, { type: 'load' }>): Promise<void> {
  const gpu = req.device === 'webgpu' || Boolean(req.halves)
  ort = gpu ? ((await import('onnxruntime-web/webgpu')) as unknown as typeof Ort) : await import('onnxruntime-web/wasm')
  const wasm = ort.env.wasm
  wasm.numThreads = req.threads
  wasm.proxy = false
  if (gpu) setOrtGpuPower(ort.env as Parameters<typeof setOrtGpuPower>[0], req.power)
  const base = req.ortWasmBase?.replace(/\/?$/, '/') ?? ortCdnBase(ort.env.versions.web ?? '')
  // The WebGPU build needs the asyncify variant of the WebAssembly runtime.
  const variant = gpu ? 'ort-wasm-simd-threaded.asyncify' : 'ort-wasm-simd-threaded'
  const paths = { mjs: `${base}${variant}.mjs`, wasm: `${base}${variant}.wasm` }
  wasm.wasmPaths = { ...paths }
  stage('runtime')
  // The loader asks for a shared memory of up to 4 GiB; a voice needs far less, and on iPhones a second
  // memory that size does not fit next to the recogniser's. Load it with a lower maximum.
  try {
    const glue = await cachedDownload(
      new URL(paths.mjs, scope.location.href).href,
      RUNTIME_CACHE,
      { record },
      {
        contentType: 'text/javascript',
      },
    )
    const lowered = lowerMemoryMaximum(new TextDecoder().decode(glue.data), req.memoryPages)
    if (lowered) {
      wasm.wasmPaths = { ...paths, mjs: URL.createObjectURL(new Blob([lowered], { type: 'text/javascript' })) }
      post({
        type: 'note',
        text: `runtime memory capped at ${req.memoryPages / 16} MiB (shared, ${req.threads} thread${req.threads === 1 ? '' : 's'})`,
      })
    } else {
      post({ type: 'note', text: 'runtime loader not recognised: memory cap not applied (4 GiB maximum)' })
    }
  } catch (err) {
    post({ type: 'note', text: `memory cap not applied: ${err instanceof Error ? err.message : String(err)}` })
  }
  if (req.plainRuntimeLoad) return
  try {
    // The same file and cache entry the recogniser uses: whichever runs first downloads it for both.
    const { data } = await cachedDownload(
      new URL(paths.wasm, scope.location.href).href,
      RUNTIME_CACHE,
      { progress: (loaded, total) => stage('runtime', loaded, total), record },
      { contentType: 'application/wasm' },
    )
    wasm.wasmBinary = data
  } catch {
    // ONNX Runtime fetches the file itself and reports the real error if this fails too.
  }
}

async function load(req: Extract<TtsWorkerRequest, { type: 'load' }>): Promise<void> {
  const started = performance.now()
  session?.release().catch(() => undefined)
  session = null
  await loadRuntime(req)

  const files = new Map<string, FileProgress>()
  let lastPost = 0
  const track = (file: string) => (loaded: number, total: number) => {
    files.set(file, { file, loaded, total })
    const now = performance.now()
    if (now - lastPost > 80 || (total > 0 && loaded >= total)) {
      lastPost = now
      post({ type: 'progress', files: [...files.values()] })
    }
  }
  stage('model')
  const get = (url: string, name: string, type?: string) =>
    cachedDownload(url, VOICE_CACHE, { progress: track(name), record }, { contentType: type })
  const short = (url: string) =>
    url
      .replace(/[?#].*$/, '')
      .split('/')
      .slice(-2)
      .join('/')
  const pending: Promise<unknown>[] = []
  const model = req.halves ? null : get(req.model, short(req.model))
  const halves = req.halves
    ? Promise.all([get(req.halves.front, short(req.halves.front)), get(req.halves.back, short(req.halves.back))])
    : null
  const settings = req.config ? get(req.config, short(req.config), 'application/json') : null
  const style = req.style ? get(req.style, short(req.style)) : null
  if (req.phonemizer === 'dizione' && !dizione) {
    pending.push(
      (async () => {
        // the port of dizione and the voice's phonemiser data
        const [{ Phonemizer }, g2p, data] = await Promise.all([
          import('./dizione/engine'),
          import('./dizione/g2p'),
          Promise.all(DIZIONE_DATA_FILES.map((f) => get(`${TINY_VOCA_BASE}phonemizer/${f}`, `phonemizer/${f}`))),
        ])
        const file = (name: (typeof DIZIONE_DATA_FILES)[number]) => data[DIZIONE_DATA_FILES.indexOf(name)]!.data
        const text = (name: (typeof DIZIONE_DATA_FILES)[number]) => new TextDecoder().decode(file(name))
        dizione = new Phonemizer({
          lexicon: text('lexicon.tsv'),
          align: JSON.parse(text('align.json')) as AlignTable,
          homographs: JSON.parse(text('homographs.json')) as HomographData,
          g2p: new g2p.G2P(JSON.parse(text('g2p.json')) as G2PSettings, g2p.readNpz(new Uint8Array(file('g2p.npz')))),
        })
      })(),
    )
  }
  if (req.phonemizer === 'uccharan') {
    pending.push(
      (async () => {
        const mod = await import('@/vendor/uccharan/index.js')
        // Language data is fetched from next to the voices the first time a language is used.
        await mod.init({ dataBaseUrl: req.uccharanBase })
        const lang = req.phonemeLanguage ?? 'en-us'
        await mod.loadLanguage(lang)
        if ((req.kind === 'kokoro' || req.kind === 'paradee' || req.kind === 'kitten') && lang.startsWith('en'))
          await mod.loadKokoro(lang)
        uccharan = mod
      })(),
    )
  }
  const assets = Promise.all(
    Object.entries(req.assets ?? {}).map(async ([role, url]) => [role, (await get(url, short(url))).data] as const),
  ).then((pairs) => new Map(pairs))
  const [modelFile, cfgFile, styleFile, halfFiles, assetFiles] = await Promise.all([
    model,
    settings,
    style,
    halves,
    assets,
    ...pending,
  ])
  const asset = (role: string): ArrayBuffer => {
    const data = assetFiles.get(role)
    if (!data) throw new Error(`The voice file "${role}" is missing`)
    return data
  }
  if (req.kind === 'tiny-voca') {
    if (!cfgFile) throw new Error('The voice settings are missing')
    const json = JSON.parse(new TextDecoder().decode(cfgFile.data)) as { voices?: string[]; sample_rate?: number }
    const name = req.speaker ?? (req.voice.endsWith('m2') ? 'M2' : 'F2')
    const speaker = (json.voices ?? ['F2', 'M2']).indexOf(name)
    config = {
      id: req.voice,
      kind: 'tiny-voca',
      sample_rate: json.sample_rate ?? 24000,
      speaker: Math.max(0, speaker),
      inference: { noise_scale: 0, length_scale: 1, noise_scale_w: 0 },
    }
  } else if (req.kind === 'paradee') {
    if (!cfgFile) throw new Error('The voice settings are missing')
    const json = JSON.parse(new TextDecoder().decode(cfgFile.data)) as { voices: string[]; sample_rate?: number }
    config = {
      id: req.voice,
      kind: 'paradee',
      sample_rate: json.sample_rate ?? 24000,
      phoneme_language: req.phonemeLanguage ?? 'en-us',
      speaker: Math.max(0, json.voices.indexOf(req.speaker ?? '')),
      inference: { noise_scale: 0, length_scale: 1, noise_scale_w: 0 },
    }
  } else if (req.kind === 'kitten') {
    const { readNpz } = await import('./dizione/g2p')
    const pack = readNpz(new Uint8Array(asset('voices'))).get(req.speaker ?? '')
    if (!pack) throw new Error(`The voice pack has no voice "${req.speaker ?? ''}"`)
    kittenStyle = pack.data
    config = {
      id: req.voice,
      kind: 'kitten',
      sample_rate: 24000,
      phoneme_language: req.phonemeLanguage ?? 'en-us',
      speaker: 0,
      inference: { noise_scale: 0, length_scale: 1, noise_scale_w: 0 },
    }
  } else if (req.kind === 'kale') {
    if (!cfgFile) throw new Error('The voice settings are missing')
    const meta = JSON.parse(new TextDecoder().decode(cfgFile.data)) as {
      phonemes?: string
      voice_inputs: { name: string; dims: number[] }[]
    }
    const text = (role: string) => new TextDecoder().decode(asset(role))
    const vocab = (JSON.parse(text('tokenizer')) as { model: { vocab: Record<string, number> } }).model.vocab
    const british = req.phonemeLanguage === 'en-gb'
    // Kale's own front end (src/vendor/kale, Apache-2.0), as its release loader builds it
    const words = await import('@/vendor/kale/words/kaleTts.js')
    const gb = await import('@/vendor/kale/words/enGb.js')
    const lex = words.readTable(text('lexicon'))
    words.readTable(text('override'), lex)
    const model = new words.G2pModel(asset('g2p'))
    const g2p = british ? new gb.BritishG2p(lex, model) : new words.G2p(lex, model)
    const rules = british ? gb.britishNormalizer(lex) : words.normalizer(lex)
    const terms = termsReplacer(words.readTable(text('terms')))
    const split = (words as unknown as { sentences: (t: string, n: (t: string) => string) => string[] }).sentences
    const extra = new Float32Array(asset('extra'))
    let at = 0
    const feeds = meta.voice_inputs.map((v) => {
      const size = v.dims.reduce((a, b) => a * b, 1)
      const data = extra.slice(at, at + size)
      at += size
      return { name: v.name, data, dims: v.dims }
    })
    kale = {
      sentences: (t) => split(t, (x) => rules(terms(x))),
      ipa: (sentence) => g2p.phonemes(sentence),
      vocab,
      native: meta.phonemes === 'kokoro-native',
      british,
      style: new Float32Array(asset('style')),
      feeds,
    }
    config = {
      id: req.voice,
      kind: 'kale',
      sample_rate: 24000,
      phoneme_language: req.phonemeLanguage ?? 'en-us',
      speaker: 0,
      inference: { noise_scale: 0, length_scale: 1, noise_scale_w: 0 },
    }
  } else if (req.kind === 'kokoro') {
    if (!styleFile) throw new Error('The voice style is missing')
    kokoroStyle = new Float32Array(styleFile.data)
    ownStyle = kokoroStyle
    config = {
      id: req.voice,
      kind: 'kokoro',
      sample_rate: 24000,
      phoneme_language: req.phonemeLanguage ?? 'en-us',
      inference: { noise_scale: 0, length_scale: 1, noise_scale_w: 0 },
    }
  } else {
    if (!cfgFile) throw new Error('The voice settings are missing')
    const json = JSON.parse(new TextDecoder().decode(cfgFile.data)) as unknown
    config =
      req.configFormat === 'piper'
        ? piperConfigFromJson(req.voice, json as Parameters<typeof piperConfigFromJson>[1], req.phonemeLanguage)
        : (json as VoiceConfig)
    if (req.configFormat === 'piper' || config.kind === 'piper')
      config = {
        ...config,
        phoneme_language: req.phonemeLanguage ?? (config as { phoneme_language: string }).phoneme_language,
      } as VoiceConfig
  }
  stage('compiling')
  const options = (providers: string[]): Ort.InferenceSession.SessionOptions => ({
    executionProviders: providers,
    graphOptimizationLevel: 'all',
    // The 8-bit weights are stored with a DequantizeLinear in front of each. Without this the runtime keeps
    // those nodes (for quantised kernels this graph does not use) and expands the weights on every call;
    // with it they are folded into constants once, at load, and synthesis is as fast as with the fp32 file.
    extra: { session: { disable_quant_qdq: '1' } },
  })
  const create = async (data: ArrayBuffer, gpu: boolean): Promise<Ort.InferenceSession> => {
    if (!gpu) return ort.InferenceSession.create(new Uint8Array(data), options(['wasm']))
    try {
      // Operators WebGPU lacks run on the CPU within the same session.
      return await ort.InferenceSession.create(new Uint8Array(data), options(['webgpu', 'wasm']))
    } catch (err) {
      post({ type: 'note', text: `WebGPU failed (${err instanceof Error ? err.message : String(err)}); using the CPU` })
      placed = 'wasm'
      return ort.InferenceSession.create(new Uint8Array(data), options(['wasm']))
    }
  }
  let placed: 'webgpu' | 'wasm' | 'split' = req.halves ? 'split' : req.device
  if (halfFiles) {
    const front = await create(halfFiles[0].data, true)
    const back = await create(halfFiles[1].data, false)
    session = twoPart(front, back)
  } else if (modelFile) {
    session = await create(modelFile.data, req.device === 'webgpu')
  } else throw new Error('The voice model is missing')
  stage('warmup')
  // A short run so the first sentence does not pay for first-use allocations.
  if (config.kind === 'kokoro') await runKokoro([0, 50, 83, 0], 1)
  else if (config.kind === 'paradee') await runParadee([0, 50, 83, 0], 1)
  else if (config.kind === 'kitten') await runKitten([0, 50, 83, 10, 0], 1)
  else if (config.kind === 'kale') await runKale([50, 83], 1)
  else if (config.kind === 'tiny-voca') await runTinyVoca({ ids: [1, 31, 43, 6, 2], stress: [0, 0, 1, 0, 0] }, 1)
  else if (config.kind === 'coqui') await run(coquiIds(config, 'a'), [0, 1, 0])
  else if (config.kind === 'piper') await run(piperIds(config.phoneme_id_map, 'ə').ids, [0, 1, 0])
  post({
    type: 'ready',
    loadMs: performance.now() - started,
    memoryBytes: runtimeMemoryBytes(),
    config,
    threads: req.threads,
    cached: (modelFile ?? halfFiles?.[0])?.fromCache ?? false,
    device: placed,
    cacheKeys: [req.halves ? null : req.model, req.halves?.front, req.halves?.back, req.config, req.style].filter(
      (x): x is string => Boolean(x),
    ),
  })
}

async function run(ids: number[], scales: [number, number, number]): Promise<Float32Array> {
  if (!session) throw new Error('The voice is not loaded')
  const feeds = {
    input: new ort.Tensor(
      'int64',
      BigInt64Array.from(ids, (x) => BigInt(x)),
      [1, ids.length],
    ),
    input_lengths: new ort.Tensor('int64', BigInt64Array.from([BigInt(ids.length)]), [1]),
    scales: new ort.Tensor('float32', Float32Array.from(scales), [3]),
  }
  const out = await session.run(feeds)
  const tensor = out[session.outputNames[0] as string] as Ort.Tensor
  const audio = new Float32Array(tensor.data as Float32Array)
  tensor.dispose()
  return audio
}

/** Kokoro: token ids, the style row for this length (510 rows of 256), and the speed. */
async function runKokoro(ids: number[], speed: number): Promise<Float32Array> {
  if (!session || !kokoroStyle) throw new Error('The voice is not loaded')
  const rows = kokoroStyle.length / 256
  const row = Math.min(Math.max(ids.length - 2, 0), rows - 1)
  const feeds = {
    input_ids: new ort.Tensor(
      'int64',
      BigInt64Array.from(ids, (x) => BigInt(x)),
      [1, ids.length],
    ),
    style: new ort.Tensor('float32', kokoroStyle.slice(row * 256, row * 256 + 256), [1, 256]),
    speed: new ort.Tensor('float32', Float32Array.from([speed]), [1]),
  }
  const out = await session.run(feeds)
  const tensor = out[session.outputNames[0] as string] as Ort.Tensor
  const audio = new Float32Array(tensor.data as Float32Array)
  tensor.dispose()
  return audio
}

const int64 = (xs: number[]) =>
  new ort.Tensor(
    'int64',
    BigInt64Array.from(xs, (x) => BigInt(x)),
    [1, xs.length],
  )
const scalar = (x: number) => new ort.Tensor('float32', Float32Array.from([x]), [1])

async function firstOutput(feeds: Record<string, Ort.Tensor>): Promise<Float32Array> {
  if (!session) throw new Error('The voice is not loaded')
  const out = await session.run(feeds)
  const tensor = out[session.outputNames[0] as string] as Ort.Tensor
  const audio = new Float32Array(tensor.data as Float32Array)
  tensor.dispose()
  return audio
}

/** Paradee-8M: Kokoro ids with their pads, the speed and the voice's index. */
async function runParadee(ids: number[], speed: number): Promise<Float32Array> {
  if (config?.kind !== 'paradee') throw new Error('The voice is not loaded')
  return firstOutput({
    input_ids: int64(ids),
    speed: scalar(speed),
    voice: new ort.Tensor('int64', BigInt64Array.from([BigInt(config.speaker)]), [1]),
  })
}

/** Kitten TTS 2: its ids, the style row for this length and the speed; the waveform and the frames per token. */
async function runKitten(ids: number[], speed: number, keepStart = true): Promise<Float32Array> {
  if (!session || !kittenStyle) throw new Error('The voice is not loaded')
  const rows = kittenStyle.length / 256
  const row = Math.min(ids.length - 1, rows - 1)
  const out = await session.run({
    input_ids: int64(ids),
    style: new ort.Tensor('float32', kittenStyle.slice(row * 256, row * 256 + 256), [1, 256]),
    speed: scalar(speed),
  })
  const wave = out.waveform as Ort.Tensor
  const durations = out.durations as Ort.Tensor
  const audio = kittenTrim(
    new Float32Array(wave.data as Float32Array),
    Array.from(durations.data as BigInt64Array, Number),
    keepStart,
  )
  wave.dispose()
  durations.dispose()
  return audio
}

/** kale-tts-1: one sentence's ids (without pads), the style row for its length, the speed and the voice's inputs. */
async function runKale(ids: number[], speed: number): Promise<Float32Array> {
  if (!kale) throw new Error('The voice is not loaded')
  const kept = ids.slice(0, 510)
  const rows = kale.style.length / 256
  const row = Math.min(kept.length, rows) - 1
  const feeds: Record<string, Ort.Tensor> = {
    input_ids: int64([0, ...kept, 0]),
    style: new ort.Tensor('float32', kale.style.slice(row * 256, row * 256 + 256), [1, 256]),
    speed: scalar(speed),
  }
  for (const f of kale.feeds) feeds[f.name] = new ort.Tensor('float32', f.data, f.dims)
  const audio = await firstOutput(feeds)
  return audio.map((x) => Math.max(-1, Math.min(1, x)))
}

/** Pieces joined, with a pause of `gapS` seconds between them. */
function joinWithGaps(parts: Float32Array[], rate: number, gapS: number): Float32Array {
  const gap = Math.round(gapS * rate)
  const total = parts.reduce((a, x) => a + x.length, 0) + gap * Math.max(0, parts.length - 1)
  const out = new Float32Array(total)
  let at = 0
  parts.forEach((x, i) => {
    if (i > 0) at += gap
    out.set(x, at)
    at += x.length
  })
  return out
}

/** tiny-voca: phoneme ids, stress, the voice, speed, pitch shift and range; the waveform at 24 kHz. */
async function runTinyVoca(p: { ids: number[]; stress: number[] }, speed: number): Promise<Float32Array> {
  if (!session || config?.kind !== 'tiny-voca') throw new Error('The voice is not loaded')
  const ints = (xs: number[]) => BigInt64Array.from(xs, (x) => BigInt(x))
  const feeds = {
    tokens: new ort.Tensor('int64', ints(p.ids), [1, p.ids.length]),
    stress: new ort.Tensor('int64', ints(p.stress), [1, p.stress.length]),
    voice: new ort.Tensor('int64', ints([config.speaker]), [1]),
    speed: new ort.Tensor('float32', Float32Array.from([speed]), [1]),
    pitch_shift: new ort.Tensor('float32', Float32Array.from([0]), [1]),
    pitch_range: new ort.Tensor('float32', Float32Array.from([1]), [1]),
  }
  const out = await session.run(feeds)
  const tensor = out.audio as Ort.Tensor
  const audio = new Float32Array(tensor.data as Float32Array)
  tensor.dispose()
  return audio
}

/** One utterance of tiny-voca, cut after commas, then between words, when it is longer than the model takes. */
async function speakTinyVoca(text: string, speed: number, spoken: { ids: number }): Promise<Float32Array[]> {
  if (!dizione) throw new Error('The phonemiser is not loaded')
  const p = dizione.tryPhonemize(text)
  if (!p) return [] // nothing to pronounce: another script, signs only
  if (p.ids.length <= TINY_VOCA_TOKEN_LIMIT) {
    spoken.ids += p.ids.length
    return [await runTinyVoca(p, speed)]
  }
  const pieces = shorter(text)
  if (pieces.length === 0) throw new Error(`One word is too long to speak: ${text.slice(0, 40)}`)
  const out: Float32Array[] = []
  for (const piece of pieces) out.push(...(await speakTinyVoca(piece, speed, spoken)))
  return out
}

async function synth(req: Extract<TtsWorkerRequest, { type: 'synth' }>): Promise<void> {
  if (!config) throw new Error('The voice is not loaded')
  const started = performance.now()
  if (config.kind === 'tiny-voca') {
    const spoken = { ids: 0 }
    const parts = await speakTinyVoca(finished(req.text.trim()), 1 / (req.scales[1] || 1), spoken)
    const joined = new Float32Array(parts.reduce((a, x) => a + x.length, 0))
    let at = 0
    for (const x of parts) {
      joined.set(x, at)
      at += x.length
    }
    const audio = limitPeak(joined)
    post(
      {
        type: 'audio',
        id: req.id,
        audio,
        ms: performance.now() - started,
        tokens: spoken.ids,
        missing: [],
        memoryBytes: runtimeMemoryBytes(),
      },
      [audio.buffer],
    )
    return
  }
  let ids: number[][]
  let phonemes: string | undefined
  const missing: string[] = []
  if (config.kind === 'paradee' || config.kind === 'kitten' || config.kind === 'kale') {
    const speed = 1 / (req.scales[1] || 1)
    const parts: Float32Array[] = []
    let tokens = 0
    let gapS: number
    if (config.kind === 'kale') {
      if (!kale) throw new Error('The voice is not loaded')
      const spoken: string[] = []
      for (const sentence of kale.sentences(req.text)) {
        const ipa = kale.ipa(sentence)
        const sentenceIds = kaleIds(ipa, kale.vocab, kale.british, kale.native)
        if (!sentenceIds.length) continue
        spoken.push(ipa)
        tokens += sentenceIds.length + 2
        parts.push(await runKale(sentenceIds, speed))
      }
      phonemes = spoken.join(' ')
      gapS = 0.12
    } else {
      if (!uccharan) throw new Error('The phonemiser is not loaded')
      phonemes = await uccharan.phonemize(req.text, { lang: config.phoneme_language, profile: 'kokoro' })
      if (config.kind === 'kitten') phonemes = kittenFold(phonemes)
      const all = uccharan.kokoroIds(phonemes)
      const cut = all.length > 512 ? [...all.slice(0, 511), 0] : all
      tokens = cut.length
      parts.push(config.kind === 'kitten' ? await runKitten(kittenIds(cut), speed) : await runParadee(cut, speed))
      gapS = config.kind === 'kitten' ? KITTEN_GAP_S : 0
    }
    const audio = limitPeak(joinWithGaps(parts, config.sample_rate, gapS))
    post(
      {
        type: 'audio',
        id: req.id,
        audio,
        ms: performance.now() - started,
        tokens,
        phonemes,
        missing,
        memoryBytes: runtimeMemoryBytes(),
      },
      [audio.buffer],
    )
    return
  }
  if (config.kind === 'kokoro') {
    if (!uccharan) throw new Error('The phonemiser is not loaded')
    const tokens = await uccharan.phonemize(req.text, { lang: config.phoneme_language, profile: 'kokoro' })
    // Kokoro reads at most 510 tokens per call.
    const all = uccharan.kokoroIds(tokens)
    const audio = await runKokoro(all.length > 512 ? [...all.slice(0, 511), 0] : all, 1 / (req.scales[1] || 1))
    post(
      {
        type: 'audio',
        id: req.id,
        audio,
        ms: performance.now() - started,
        tokens: all.length,
        phonemes: tokens,
        missing,
        memoryBytes: runtimeMemoryBytes(),
      },
      [audio.buffer],
    )
    return
  }
  if (config.kind === 'coqui') {
    ids = [coquiIds(config, withLeadIn(config, req.text))]
  } else {
    if (!uccharan) throw new Error('The phonemiser is not loaded')
    const sentences = await uccharan.phonemize(req.text, { lang: config.phoneme_language, profile: 'piper' })
    phonemes = sentences.join(' ')
    ids = sentences.map((s) => {
      const r = piperIds(config?.kind === 'piper' ? config.phoneme_id_map : {}, s)
      missing.push(...r.missing)
      return r.ids
    })
  }
  const parts: Float32Array[] = []
  // One model call per sentence, each followed by the server's pause shortening, as sherpa-onnx does.
  for (const seq of ids) if (seq.length > 2) parts.push(scaleSilence(await run(seq, req.scales), config.sample_rate))
  const total = parts.reduce((a, p) => a + p.length, 0)
  const audio = new Float32Array(total)
  let o = 0
  for (const p of parts) {
    audio.set(p, o)
    o += p.length
  }
  post(
    {
      type: 'audio',
      id: req.id,
      audio,
      ms: performance.now() - started,
      tokens: ids.reduce((a, s) => a + s.length, 0),
      phonemes,
      missing: [...new Set(missing)],
      memoryBytes: runtimeMemoryBytes(),
    },
    [audio.buffer],
  )
}

/** A blend of Kokoro voices: their style vectors (one per input length) added up with normalised weights. */
async function blend(req: Extract<TtsWorkerRequest, { type: 'style' }>): Promise<void> {
  const started = performance.now()
  if (!ownStyle) throw new Error('Blends need a Kokoro voice')
  if (req.styles.length === 0) {
    kokoroStyle = ownStyle
  } else {
    const files = await Promise.all(req.styles.map((s) => cachedDownload(s.url, VOICE_CACHE, { record })))
    const total = req.styles.reduce((a, s) => a + s.weight, 0)
    if (!(total > 0)) throw new Error('A blend needs a weight above 0')
    const mixed = new Float32Array(ownStyle.length)
    files.forEach((f, i) => {
      const v = new Float32Array(f.data)
      if (v.length !== mixed.length) throw new Error('These voices’ styles do not have the same shape')
      const w = (req.styles[i] as { weight: number }).weight / total
      for (let k = 0; k < mixed.length; k++) mixed[k] = (mixed[k] as number) + w * (v[k] as number)
    })
    kokoroStyle = mixed
  }
  post({ type: 'styled', id: req.id, ms: performance.now() - started })
}

scope.onmessage = (e: MessageEvent<TtsWorkerRequest>) => {
  const req = e.data
  chain = chain
    .then(() => (req.type === 'load' ? load(req) : req.type === 'style' ? blend(req) : synth(req)))
    .catch((err: unknown) => {
      post({
        type: 'error',
        id: req.type === 'synth' || req.type === 'style' ? req.id : undefined,
        message: err instanceof Error ? err.message : String(err),
      })
    })
}
