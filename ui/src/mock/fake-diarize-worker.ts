import type { DiarizeWorker } from '@/lib/browser-diarize/engine'
import type { DiarizeWorkerRequest, DiarizeWorkerResponse } from '@/lib/browser-diarize/models'

/**
 * A scripted stand-in for the diarization worker (tests only, `bolo.mock.diarize`): it "loads" without
 * fetching anything and answers with two speakers taking turns every 4 s, so the page's flow (timelines, DER,
 * a transcript per speaker) can be tested without the models.
 */
export const FAKE_DIARIZE_KEY = 'bolo.mock.diarize'

export class FakeDiarizeWorker implements DiarizeWorker {
  onmessage: ((e: MessageEvent<DiarizeWorkerResponse>) => void) | null = null
  onerror: ((e: ErrorEvent) => void) | null = null
  private emit(msg: DiarizeWorkerResponse, after: number) {
    setTimeout(() => this.onmessage?.({ data: msg } as MessageEvent<DiarizeWorkerResponse>), after)
  }
  postMessage(req: DiarizeWorkerRequest): void {
    if (req.type === 'load') {
      const total = req.segmentation.bytes + req.embedding.bytes
      this.emit({ type: 'progress', files: [{ file: 'model.onnx', loaded: total / 2, total }] }, 50)
      this.emit({ type: 'ready', loadMs: 200, cacheKeys: [], bytes: total }, 200)
      return
    }
    const seconds = req.audio.length / 16000
    const segments = []
    for (let t = 0, i = 0; t < seconds; t += 4, i++)
      segments.push({ speaker: req.speakers === 1 ? 0 : i % 2, start: t, end: Math.min(seconds, t + 3.6) })
    this.emit({ type: 'step', id: req.id, done: 0.5 }, 50)
    this.emit({ type: 'result', id: req.id, segments, ms: 300 }, 300)
  }
  terminate(): void {
    this.onmessage = null
  }
}
