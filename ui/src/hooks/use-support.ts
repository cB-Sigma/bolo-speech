import { useEffect, useState } from 'react'
import {
  detectSupport,
  microphonePermission,
  probeWebGpu,
  type BrowserSupport,
  type MicPermission,
} from '@/lib/support'

export interface SupportState extends BrowserSupport {
  /** null while probing */
  webgpuAdapter: boolean | null
  micPermission: MicPermission
}

export function useSupport(): SupportState {
  const [state, setState] = useState<SupportState>(() => ({
    ...detectSupport(),
    webgpuAdapter: null,
    micPermission: 'unknown',
  }))
  useEffect(() => {
    let alive = true
    void Promise.all([probeWebGpu(), microphonePermission()]).then(([webgpuAdapter, micPermission]) => {
      if (alive) setState((s) => ({ ...s, webgpuAdapter, micPermission }))
    })
    return () => {
      alive = false
    }
  }, [])
  return state
}
