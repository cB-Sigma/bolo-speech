import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useInfo } from '@/hooks/use-api'
import { api, type NewVoice } from '@/lib/api/client'
import { isDemo } from '@/lib/api/config'
import type { SavedVoice, ServerInfo, VoiceScore } from '@/lib/api/types'

/**
 * Whether the server keeps saved voices: true when it announces them (`capabilities.voices`), false when it is new
 * enough to announce other capabilities but not this one, null when it does not say (the route is then probed).
 */
export function voicesAnnounced(info: ServerInfo | undefined): boolean | null {
  const caps = info?.capabilities
  if (caps?.voices) return true
  return caps && 'providers' in caps ? false : null
}

/** The saved voices (`GET /api/v1/voices`); never asked for in the live demo, which has no server. */
export function useVoices() {
  const announced = voicesAnnounced(useInfo().data)
  const q = useQuery({
    queryKey: ['voices'],
    queryFn: ({ signal }) => api.voices(signal),
    staleTime: 15_000,
    retry: false,
    enabled: announced !== false && !isDemo,
  })
  return { ...q, offered: !isDemo && (announced ?? q.isSuccess) }
}

export function useVoiceMutations() {
  const qc = useQueryClient()
  const settle = () => void qc.invalidateQueries({ queryKey: ['voices'] })
  const create = useMutation({ mutationFn: (v: NewVoice) => api.createVoice(v), onSettled: settle })
  const remove = useMutation({ mutationFn: (id: string) => api.deleteVoice(id), onSettled: settle })
  const score = useMutation({
    mutationFn: ({ id, model }: { id: string; model: string }) => api.voiceScore(id, model),
    onSettled: settle,
  })
  return { create, remove, score }
}

export const CONSENT_LABEL: Record<string, string> = {
  self: 'This is my voice',
  permission: "I have the speaker's permission",
  recording: 'Recorded consent from the speaker',
}

/** "0.66 · same speaker" for a score, or what went wrong. */
export function scoreText(s: VoiceScore | undefined | null): string {
  if (!s) return 'not scored'
  if (s.error) return s.error
  if (s.similarity == null) return 'not scored'
  return `${s.similarity.toFixed(2)}${s.verdict ? ` · ${s.verdict}` : ''}`
}

export function scoreTone(s: VoiceScore | undefined | null): 'good' | 'warning' | 'bad' | 'none' {
  if (!s || s.similarity == null) return 'none'
  if (s.verdict === 'same speaker') return 'good'
  if (s.verdict === 'similar') return 'warning'
  return 'bad'
}

/** The voice's score for one model, if it was scored with it. */
export function scoreFor(v: SavedVoice, model: string | null | undefined): VoiceScore | undefined {
  return model ? v.scores?.[model] : undefined
}
