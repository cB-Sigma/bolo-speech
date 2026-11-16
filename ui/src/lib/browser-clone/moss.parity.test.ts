// @vitest-environment node
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { buildRequestRows, MossTts, upsample2x, type MossManifest } from './moss'
import { Random, type MakeTensor, type OrtSession } from './ort-types'
import { CLONE_RATE } from './protocol'
import { SpmBpeTokenizer } from './spm-bpe'

/**
 * The page's MOSS-TTS-Nano driver on the real model, with ONNX Runtime Web (WebAssembly) in Node, for comparison
 * with the model's reference runtime (ui/scripts/clone/moss_reference.py describes the steps). Runs only with
 * BOLO_MOSS_DIR (the two Hub repositories side by side) and BOLO_MOSS_OUT (the example voices as 24 kHz float32
 * files, written by `moss_reference.py refs`; the runs are written there too).
 */

const DIR = process.env.BOLO_MOSS_DIR
const OUT = process.env.BOLO_MOSS_OUT
const TEXTS = [
  'Thanks for calling. Your appointment is confirmed for Tuesday at half past ten, and we will send a reminder the day before.',
  'Vielen Dank für Ihren Anruf. Ihr Termin am Dienstag um halb elf ist bestätigt.',
  '欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。',
  // Long enough for three chunks: the random sequence and the pauses carry across them.
  'Every year the city holds a festival by the river. People come from far away to see the boats, taste the food, and listen to music until late at night. Last summer it rained all weekend, but nobody seemed to mind. The children danced in the puddles, and the bands kept playing under their umbrellas. When the sun finally came out on Sunday evening, the whole crowd cheered as if their team had just won a match, and the music went on until well after midnight.',
]
const SEEDS = [1, 7]

describe.runIf(DIR && OUT && existsSync(DIR))('MOSS-TTS-Nano driver on the real model', () => {
  it('writes runs for the reference comparison', async () => {
    const ort = await import('onnxruntime-web')
    ort.env.wasm.numThreads = 4
    const tts = path.join(DIR as string, 'MOSS-TTS-Nano-100M-ONNX')
    const codec = path.join(DIR as string, 'MOSS-Audio-Tokenizer-Nano-ONNX')
    const file = (dir: string, name: string) => new Uint8Array(readFileSync(path.join(dir, name)))
    const json = (dir: string, name: string) => JSON.parse(readFileSync(path.join(dir, name), 'utf8')) as never
    const open = (dir: string, graph: string, data: string) =>
      ort.InferenceSession.create(file(dir, graph), {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        externalData: [{ path: data, data: file(dir, data) }],
      }) as unknown as Promise<OrtSession>
    const sessions = {
      prefill: await open(tts, 'moss_tts_prefill.onnx', 'moss_tts_global_shared.data'),
      decode: await open(tts, 'moss_tts_decode_step.onnx', 'moss_tts_global_shared.data'),
      frame: await open(tts, 'moss_tts_local_fixed_sampled_frame.onnx', 'moss_tts_local_shared.data'),
      encoder: await open(codec, 'moss_audio_tokenizer_encode.onnx', 'moss_audio_tokenizer_encode.data'),
      codecDecoder: await open(
        codec,
        'moss_audio_tokenizer_decode_step.onnx',
        'moss_audio_tokenizer_decode_shared.data',
      ),
    }
    const manifest = json(tts, 'browser_poc_manifest.json') as MossManifest
    const tok = SpmBpeTokenizer.fromModel(readFileSync(path.join(tts, 'tokenizer.model')))
    const tensor: MakeTensor = (type, d, dims) => new ort.Tensor(type, d as Float32Array, dims as number[]) as never
    const moss = new MossTts(
      manifest,
      json(tts, 'tts_browser_onnx_meta.json'),
      json(codec, 'codec_browser_onnx_meta.json'),
      { encode: (t) => tok.encode(t), count: (t) => tok.count(t) },
      sessions,
      tensor,
    )
    const voices = ['kokoro-am-michael', 'syspin-en-female', 'syspin-hi-male'].filter((v) =>
      existsSync(path.join(OUT as string, `${v}.f32`)),
    )
    expect(voices.length).toBeGreaterThan(0)
    for (const voice of voices) {
      const pcm = new Float32Array(readFileSync(path.join(OUT as string, `${voice}.f32`)).buffer.slice(0))
      // What the codec is given: the page's 2x upsampling, both channels the same.
      const wave = upsample2x(pcm.subarray(0, 12 * CLONE_RATE))
      const planar = new Float32Array(wave.length * 2)
      planar.set(wave)
      planar.set(wave, wave.length)
      writeFileSync(path.join(OUT as string, `${voice}.48.f32`), planar)
      const codes = await moss.encodeVoice(pcm)
      moss.setVoice(codes)
      for (const [t, text] of TEXTS.entries())
        for (const seed of SEEDS) {
          const rng = new Random(seed)
          const uniforms: number[] = []
          const frames: number[][] = []
          const audio: number[] = []
          const started = performance.now()
          const r = await moss.generate(
            text,
            {
              random: () => {
                const u = rng.uniform()
                uniforms.push(u)
                return u
              },
              onFrame: (c) => frames.push([...c]),
            },
            (a) => audio.push(...a),
          )
          const ms = performance.now() - started
          const tag = `${voice}.${t}.${seed}`
          writeFileSync(path.join(OUT as string, `${tag}.js.f32`), Float32Array.from(audio))
          writeFileSync(
            path.join(OUT as string, `${tag}.js.json`),
            JSON.stringify({
              voice,
              text,
              chunks: r.chunks,
              codes: codes.map((c) => [...c]),
              rows: r.chunks.map((c) => [...buildRequestRows(manifest, codes, c.ids)]),
              uniforms,
              frames,
              ms,
              rtf: ms / 1000 / (audio.length / CLONE_RATE),
            }),
          )
          expect(frames.length).toBe(r.frames)
          expect(audio.length).toBeGreaterThan(CLONE_RATE)
        }
    }
  }, 3_600_000)
})
