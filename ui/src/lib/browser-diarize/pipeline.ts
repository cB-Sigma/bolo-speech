/**
 * Speaker diarization in the page, the way pyannote's 3.x pipeline works,
 * with the models as plug-ins so this file stays free of any runtime:
 *
 * 1. a segmentation model (pyannote segmentation-3.0) reads 10 s windows,
 *    sliding by `step` seconds, and says per frame which of up to three
 *    local speakers talk (its 7 "powerset" classes: none, 1, 2, 3, 1+2,
 *    1+3, 2+3);
 * 2. each local speaker of each window gets a voice embedding from the
 *    audio where it talks (alone, where there is enough of that);
 * 3. the embeddings are clustered (average linkage on cosine distance, two
 *    speakers of one window are never merged): down to a distance
 *    threshold when the number of speakers is automatic, or to the number
 *    asked for;
 * 4. per frame, the windows' activities are averaged per cluster, and the
 *    most active clusters are kept, as many as the windows say talk at once.
 */

export const RATE = 16000
export const WINDOW_S = 10
/** pyannote segmentation-3.0: one output frame every 270 samples, each reading 990. */
export const FRAME_STEP = 270
export const FRAME_SPAN = 990
/** Powerset classes of segmentation-3.0, as sets of local speakers. */
export const POWERSET: number[][] = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]]

export interface Segment {
  speaker: number
  start: number
  end: number
}

export interface DiarizeOptions {
  /** Exact number of speakers; null: found from `threshold`. */
  speakers: number | null
  /** Cosine distance under which two voices count as one (auto mode). */
  threshold: number
  /** Seconds between window starts. */
  step: number
  /** Called after each window and each embedding, 0..1. */
  progress?: (done: number) => void
}

export interface Models {
  /** Logits [frames][7] for one 10 s window (zero-padded at the end). */
  segment(window: Float32Array): Promise<number[][]>
  /** A voice embedding for some 16 kHz audio. */
  embed(audio: Float32Array): Promise<Float32Array>
}

export const DEFAULTS: DiarizeOptions = { speakers: null, threshold: 0.6, step: 2.5 }

function softmax(row: number[]): number[] {
  const m = Math.max(...row)
  const e = row.map((x) => Math.exp(x - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((x) => x / s)
}

/** Per-frame activity of the three local speakers, 0..1, from powerset logits. */
export function localActivity(logits: number[][]): number[][] {
  return logits.map((row) => {
    const p = softmax(row)
    const out = [0, 0, 0]
    POWERSET.forEach((set, c) => {
      for (const k of set) out[k] = (out[k] ?? 0) + (p[c] ?? 0)
    })
    return out
  })
}

export function cosineDistance(a: Float32Array, b: Float32Array): number {
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += (a[i] as number) * (b[i] as number)
    na += (a[i] as number) ** 2
    nb += (b[i] as number) ** 2
  }
  return 1 - dot / (Math.sqrt(na * nb) || 1)
}

/**
 * Average-linkage clustering. `cannotLink[i][j]` keeps i and j apart. Stops at `k` clusters, or when the
 * closest pair is farther than `threshold` (k null). Returns a cluster index per item, numbered by first use.
 */
export function cluster(
  embeddings: Float32Array[],
  opts: { k: number | null; threshold: number; cannotLink?: (i: number, j: number) => boolean },
): number[] {
  const n = embeddings.length
  if (n === 0) return []
  const dist = embeddings.map((a) => embeddings.map((b) => cosineDistance(a, b)))
  let groups: number[][] = embeddings.map((_, i) => [i])
  let relaxed = false
  const apart = (g: number[], h: number[]) => !relaxed && g.some((i) => h.some((j) => opts.cannotLink?.(i, j)))
  const link = (g: number[], h: number[]) => {
    let s = 0
    for (const i of g) for (const j of h) s += (dist[i] as number[])[j] as number
    return s / (g.length * h.length)
  }
  const target = opts.k == null ? 1 : Math.max(1, opts.k)
  while (groups.length > target) {
    let best: [number, number, number] | null = null
    for (let a = 0; a < groups.length; a++)
      for (let b = a + 1; b < groups.length; b++) {
        const ga = groups[a] as number[]
        const gb = groups[b] as number[]
        if (apart(ga, gb)) continue
        const d = link(ga, gb)
        if (!best || d < best[0]) best = [d, a, b]
      }
    if (!best && opts.k != null && relaxed === false) {
      // A count was asked for and only joining speakers of one window reaches it: the visitor's count wins.
      relaxed = true
      continue
    }
    if (!best || (opts.k == null && best[0] > opts.threshold)) break
    const [, a, b] = best
    groups = groups
      .filter((_, i) => i !== a && i !== b)
      .concat([[...(groups[a] as number[]), ...(groups[b] as number[])]])
  }
  // Number clusters by where they first appear.
  groups.sort((g, h) => Math.min(...g) - Math.min(...h))
  const label = new Array<number>(n).fill(0)
  groups.forEach((g, c) => g.forEach((i) => (label[i] = c)))
  return label
}

/** Turn per-frame speaker sets into segments, closing gaps under `gap` s and dropping pieces under `min` s. */
export function toSegments(active: boolean[][], frameSeconds: number, gap = 0.25, min = 0.25): Segment[] {
  const speakers = active[0]?.length ?? 0
  const out: Segment[] = []
  for (let s = 0; s < speakers; s++) {
    let open: Segment | null = null
    for (let f = 0; f < active.length; f++) {
      const on = (active[f] as boolean[])[s]
      const t = f * frameSeconds
      if (on) {
        if (open && t - open.end <= gap) open.end = t + frameSeconds
        else {
          if (open) out.push(open)
          open = { speaker: s, start: t, end: t + frameSeconds }
        }
      }
    }
    if (open) out.push(open)
  }
  return out
    .filter((x) => x.end - x.start >= min)
    .sort((a, b) => a.start - b.start || a.speaker - b.speaker)
    .map((x) => ({ ...x, start: Math.round(x.start * 1000) / 1000, end: Math.round(x.end * 1000) / 1000 }))
}

/** Who spoke when in 16 kHz mono audio. Speakers are numbered 0.. by first appearance. */
export async function diarize(
  audio: Float32Array,
  models: Models,
  opts: DiarizeOptions = DEFAULTS,
): Promise<Segment[]> {
  const win = WINDOW_S * RATE
  const hop = Math.max(1, Math.round(opts.step * RATE))
  const starts: number[] = []
  for (let s = 0; ; s += hop) {
    starts.push(s)
    if (s + win >= audio.length) break
  }
  const frameSeconds = FRAME_STEP / RATE
  const totalFrames = Math.ceil(audio.length / FRAME_STEP)
  const windows: { start: number; act: number[][] }[] = []
  const work = starts.length * 2
  let done = 0
  for (const start of starts) {
    const chunk = new Float32Array(win)
    chunk.set(audio.subarray(start, Math.min(audio.length, start + win)))
    windows.push({ start, act: localActivity(await models.segment(chunk)) })
    opts.progress?.(++done / work)
  }

  // One embedding per local speaker that talks long enough in its window.
  const items: { w: number; k: number; emb: Float32Array }[] = []
  for (const [w, { start, act }] of windows.entries()) {
    for (let k = 0; k < 3; k++) {
      const alone: number[] = []
      const any: number[] = []
      act.forEach((p, f) => {
        if (start + f * FRAME_STEP >= audio.length) return
        if ((p[k] as number) > 0.5) {
          any.push(f)
          if (p.every((q, j) => j === k || q < 0.5)) alone.push(f)
        }
      })
      if (any.length * frameSeconds < 0.4) continue
      const frames = alone.length * frameSeconds >= 0.4 ? alone : any
      const pieces = frames.map((f) => audio.subarray(start + f * FRAME_STEP, start + f * FRAME_STEP + FRAME_STEP))
      const joined = new Float32Array(pieces.reduce((a, p) => a + p.length, 0))
      let o = 0
      for (const p of pieces) {
        joined.set(p, o)
        o += p.length
      }
      items.push({ w, k, emb: await models.embed(joined) })
    }
    opts.progress?.(Math.min(1, (++done + 0) / work))
  }
  if (items.length === 0) return []
  const labels = cluster(
    items.map((i) => i.emb),
    {
      k: opts.speakers,
      threshold: opts.threshold,
      cannotLink: (a, b) => items[a]?.w === items[b]?.w,
    },
  )
  const speakers = Math.max(...labels) + 1
  const local = new Map(items.map((it, i) => [`${it.w}:${it.k}`, labels[i] as number]))

  // Average each speaker's activity over the windows that cover a frame, and how many talk at once.
  const score = Array.from({ length: totalFrames }, () => new Array<number>(speakers).fill(0))
  const count = new Array<number>(totalFrames).fill(0)
  const cover = new Array<number>(totalFrames).fill(0)
  for (const [w, { start, act }] of windows.entries()) {
    const offset = Math.round(start / FRAME_STEP)
    act.forEach((p, f) => {
      const g = offset + f
      if (g >= totalFrames) return
      cover[g] = (cover[g] as number) + 1
      count[g] = (count[g] as number) + p.filter((q) => q > 0.5).length
      for (let k = 0; k < 3; k++) {
        const c = local.get(`${w}:${k}`)
        if (c != null) (score[g] as number[])[c] = ((score[g] as number[])[c] as number) + (p[k] as number)
      }
    })
  }
  const active = score.map((row, g) => {
    const n = Math.round((count[g] as number) / Math.max(1, cover[g] as number))
    const order = row.map((v, c) => [v, c] as const).sort((a, b) => b[0] - a[0])
    const on = new Array<boolean>(speakers).fill(false)
    for (const [v, c] of order.slice(0, n)) if (v > 0) on[c] = true
    return on
  })
  // Renumber by first appearance in time.
  const segs = toSegments(active, frameSeconds)
  const order: number[] = []
  for (const s of segs) if (!order.includes(s.speaker)) order.push(s.speaker)
  return segs.map((s) => ({ ...s, speaker: order.indexOf(s.speaker) }))
}

export interface DerResult {
  der: number
  missed: number
  falseAlarm: number
  confusion: number
  /** Reference speech, seconds (overlaps counted per speaker). */
  total: number
  /** Hypothesis speaker → reference speaker that maximises agreement. */
  mapping: Record<string, string>
}

function permutations<T>(xs: T[], k: number): T[][] {
  if (k === 0) return [[]]
  const out: T[][] = []
  xs.forEach((x, i) => {
    for (const rest of permutations([...xs.slice(0, i), ...xs.slice(i + 1)], k - 1)) out.push([x, ...rest])
  })
  return out
}

/**
 * Diarization error rate, scored on 10 ms frames without a collar: (missed + false alarm + confusion) / total
 * reference speech, overlaps counted per speaker, with the speaker mapping that agrees most.
 */
export function der(
  reference: { speaker: string; start: number; end: number }[],
  hypothesis: { speaker: string; start: number; end: number }[],
  resolution = 0.01,
): DerResult {
  const end = Math.max(0, ...reference.map((r) => r.end), ...hypothesis.map((h) => h.end))
  const n = Math.ceil(end / resolution)
  const refSpk = [...new Set(reference.map((r) => r.speaker))]
  const hypSpk = [...new Set(hypothesis.map((h) => h.speaker))]
  const grid = (segs: { speaker: string; start: number; end: number }[], names: string[]) => {
    const g = Array.from({ length: n }, () => new Set<number>())
    for (const s of segs) {
      const i = names.indexOf(s.speaker)
      for (let f = Math.round(s.start / resolution); f < Math.min(n, Math.round(s.end / resolution)); f++)
        (g[f] as Set<number>).add(i)
    }
    return g
  }
  const R = grid(reference, refSpk)
  const H = grid(hypothesis, hypSpk)
  // Agreement between each hypothesis and reference speaker, in frames.
  const agree = hypSpk.map((_, h) =>
    refSpk.map((_, r) => R.filter((set, f) => set.has(r) && (H[f] as Set<number>).has(h)).length),
  )
  // Best one-to-one mapping (few speakers: try them all). best[h] = reference index, or -1.
  let best: number[] = new Array<number>(hypSpk.length).fill(-1)
  let bestScore = -1
  const hypIdx = hypSpk.map((_, i) => i)
  const refIdx = refSpk.map((_, i) => i)
  const consider = (pairs: [number, number][]) => {
    const sc = pairs.reduce((a, [h, r]) => a + ((agree[h] as number[])[r] as number), 0)
    if (sc > bestScore) {
      bestScore = sc
      best = new Array<number>(hypSpk.length).fill(-1)
      for (const [h, r] of pairs) best[h] = r
    }
  }
  if (hypSpk.length <= refSpk.length)
    for (const rs of permutations(refIdx, hypSpk.length)) consider(hypIdx.map((h, i) => [h, rs[i] as number]))
  else for (const hs of permutations(hypIdx, refSpk.length)) consider(refIdx.map((r, i) => [hs[i] as number, r]))
  let missed = 0
  let falseAlarm = 0
  let confusion = 0
  let total = 0
  for (let f = 0; f < n; f++) {
    const r = R[f] as Set<number>
    const h = H[f] as Set<number>
    total += r.size
    const correct = [...h].filter((x) => r.has(best[x] as number)).length
    missed += Math.max(0, r.size - h.size)
    falseAlarm += Math.max(0, h.size - r.size)
    confusion += Math.min(r.size, h.size) - correct
  }
  const mapping: Record<string, string> = {}
  best.forEach((r, h) => {
    if (r >= 0) mapping[hypSpk[h] as string] = refSpk[r] as string
  })
  const s = (x: number) => x * resolution
  return {
    der: total ? (missed + falseAlarm + confusion) / total : 0,
    missed: s(missed),
    falseAlarm: s(falseAlarm),
    confusion: s(confusion),
    total: s(total),
    mapping,
  }
}
