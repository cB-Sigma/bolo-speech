import type { LlmWorker } from '@/lib/browser-llm/engine'
import type { LlmWorkerRequest, LlmWorkerResponse } from '@/lib/browser-llm/models'

/** A scripted language model (tests only, `bolo.mock.llm`): a fixed two-sentence reply, word by word. */
export class FakeLlmWorker implements LlmWorker {
  onmessage: ((e: MessageEvent<LlmWorkerResponse>) => void) | null = null
  onerror: ((e: ErrorEvent) => void) | null = null
  private timers: ReturnType<typeof setTimeout>[] = []
  private emit(msg: LlmWorkerResponse, after: number) {
    this.timers.push(setTimeout(() => this.onmessage?.({ data: msg } as MessageEvent<LlmWorkerResponse>), after))
  }
  postMessage(req: LlmWorkerRequest): void {
    if (req.type === 'load')
      this.emit({ type: 'ready', loadMs: 100, cacheKeys: [], bytes: 1e6, device: req.device }, 100)
    else if (req.type === 'stop') {
      this.timers.forEach(clearTimeout)
      this.timers = []
    } else {
      const words =
        'I heard you. This is a pretend reply from a scripted model. It goes on for a while so you can interrupt it.'.split(
          /(?<= )/,
        )
      words.forEach((w, i) => this.emit({ type: 'token', id: req.id, text: w }, 60 * (i + 1)))
      this.emit(
        { type: 'done', id: req.id, text: words.join(''), tokens: words.length, ms: 60 * words.length },
        60 * (words.length + 1),
      )
    }
  }
  terminate(): void {
    this.timers.forEach(clearTimeout)
    this.onmessage = null
  }
}
