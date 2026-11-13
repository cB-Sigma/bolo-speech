import { useEffect, useState } from 'react'
import { api } from '@/lib/api/client'
import { median } from '@/lib/latency/stats'

/** Median round-trip time of a tiny request to the server, sampled while `active`. */
export function usePing(active: boolean, intervalMs = 2500): number | null {
  const [rtt, setRtt] = useState<number | null>(null)
  useEffect(() => {
    if (!active) return
    let alive = true
    const samples: number[] = []
    const controller = new AbortController()
    const tick = async () => {
      try {
        const ms = await api.ping(controller.signal)
        samples.push(ms)
        if (samples.length > 9) samples.shift()
        if (alive) setRtt(median(samples))
      } catch {
        // offline: keep the last value
      }
    }
    void tick()
    const id = setInterval(() => void tick(), intervalMs)
    return () => {
      alive = false
      controller.abort()
      clearInterval(id)
    }
  }, [active, intervalMs])
  return rtt
}
