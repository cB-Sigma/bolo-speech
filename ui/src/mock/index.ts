import { setTransport } from '@/lib/api/config'
import { MockServer, type MockFlags } from './engine'
import { handleRequest, isApiUrl } from './http'
import { createMockWebSocket } from './ws'

export { MOCK_KEY } from './engine'

function readFlags(): MockFlags {
  try {
    return JSON.parse(globalThis.localStorage?.getItem('bolo.mock') ?? '{}') as MockFlags
  } catch {
    return {}
  }
}

let server: MockServer | null = null

/**
 * Route the API client through an in-memory server that implements the same
 * contract. Used by `npm run dev:mock`, the end-to-end tests and demos.
 */
export function installMock(flags: MockFlags = readFlags()): MockServer {
  server?.dispose()
  const s = new MockServer(flags)
  server = s
  setTransport({
    fetch: (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      return isApiUrl(url) ? handleRequest(s, input, init) : globalThis.fetch(input, init)
    },
    WebSocket: createMockWebSocket(s),
  })
  return s
}
