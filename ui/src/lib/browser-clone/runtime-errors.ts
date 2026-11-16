/**
 * Run `work`, failing as soon as transformers.js reports that an ONNX Runtime run failed. It logs the error with
 * `console.error` ("An error occurred during model execution: ...", then the inputs) before rethrowing; on WebGPU
 * the rethrow did not always come back (a failed GPU buffer download in Chatterbox's decoder left generation
 * waiting for minutes), so the logged message is enough to fail the request.
 */
export async function failOnRuntimeError<T>(work: () => Promise<T>, log: Pick<Console, 'error'> = console): Promise<T> {
  const original = log.error
  let fail: (err: Error) => void = () => undefined
  const failed = new Promise<never>((_, reject) => {
    fail = reject
  })
  // Whichever settles second is nobody's business.
  failed.catch(() => undefined)
  log.error = (...args: unknown[]) => {
    original.apply(log, args)
    const first = typeof args[0] === 'string' ? args[0] : ''
    const m = /An error occurred during model execution: "([\s\S]*)"\.?$/.exec(first)
    if (m) fail(new Error(m[1] || first))
  }
  try {
    return await Promise.race([work(), failed])
  } finally {
    log.error = original
  }
}
