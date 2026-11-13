import { setTransport } from '@/lib/api/config'
import { DemoBackend, isApiUrl, NEEDS_SERVER } from './backend'

/** A WebSocket that refuses to connect: there is no server to stream to. */
class NoServerSocket {
  readyState = 3
  binaryType: BinaryType = 'arraybuffer'
  onopen: ((e: Event) => void) | null = null
  onmessage: ((e: MessageEvent) => void) | null = null
  onerror: ((e: Event) => void) | null = null
  onclose: ((e: CloseEvent) => void) | null = null
  constructor() {
    setTimeout(() => this.onclose?.(new CloseEvent('close', { code: 1008, reason: NEEDS_SERVER })), 0)
  }
  send(): void {}
  close(): void {}
}

/** Route the API client to the in-page backend. Used only by the live demo build. */
export function installDemo(): DemoBackend {
  const backend = new DemoBackend()
  setTransport({
    fetch: (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      return isApiUrl(url) ? backend.handle(input, init) : globalThis.fetch(input, init)
    },
    WebSocket: NoServerSocket as unknown as typeof WebSocket,
  })
  return backend
}
