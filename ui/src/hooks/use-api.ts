import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo } from 'react'
import { toast } from 'sonner'
import { api, ApiError, NetworkError } from '@/lib/api/client'
import { isDemo } from '@/lib/api/config'
import type { Catalog, CatalogModel, Component, Task } from '@/lib/api/types'
import {
  keysOffered,
  providersOffered,
  recommendationsOffered,
  telephonyOffered,
  wordListsAnnounced,
} from '@/lib/capabilities'
import { warningText } from '@/lib/fit'
import { live } from '@/lib/live'
import type { NavFeatures } from '@/lib/nav'
import { prefs, setPref } from '@/lib/prefs'
import { useStore } from './use-store'

export function useInfo() {
  return useQuery({
    queryKey: ['info'],
    queryFn: ({ signal }) => api.info(signal),
    staleTime: 60_000,
    retry: (count, err) => err instanceof NetworkError && count < 2,
    refetchInterval: (q) => (q.state.status === 'error' ? 4000 : false),
  })
}

export function useCatalog() {
  return useQuery({
    queryKey: ['catalog'],
    queryFn: ({ signal }) => api.catalog({}, signal),
    staleTime: 15_000,
    // While something is downloading, poll so progress shows even without SSE job events.
    refetchInterval: (q) => (q.state.data?.models.some((m) => m.pulling) ? 1000 : false),
  })
}

export function useComponents() {
  return useQuery({ queryKey: ['components'], queryFn: ({ signal }) => api.components(signal), staleTime: 30_000 })
}

const COMPONENT_NAMES: Record<string, string> = {
  'espeak-ng': 'eSpeak NG',
  uccharan: 'uccharan',
  'llama-server': 'llama-server',
  'audio.cpp': 'audio.cpp',
  'plugin:torch': 'the torch engine plugin',
}

/** The first required component of a model that is not installed, if the model is unavailable because of it. */
export function missingComponent(
  model: CatalogModel | null | undefined,
  components: Component[] | undefined,
): Component | null {
  if (!model || model.available !== false) return null
  const need = model.requires ?? []
  return (
    components?.find((c) => need.includes(c.id) && !c.installed) ??
    // The catalog says a requirement is unmet but the component list is not in yet (or is older): still name it.
    (need[0]
      ? {
          id: need[0],
          name: COMPONENT_NAMES[need[0]] ?? need[0],
          installed: false,
          source: null,
          version: null,
          license: '',
          license_url: '',
          required_by_count: 0,
          installable: false,
          install_blocked_reason: null,
          manual_command: '',
          description: '',
        }
      : null)
  )
}

/**
 * Word lists, when the server has them. Newer servers announce them (`capabilities.wordlists`); for older ones
 * the route is probed, and a 404 (or the demo) reads as "not offered".
 */
export function useWordLists() {
  const announced = wordListsAnnounced(useInfo().data)
  const q = useQuery({
    queryKey: ['wordlists'],
    queryFn: ({ signal }) => api.wordLists(signal),
    staleTime: 30_000,
    retry: false,
    enabled: announced !== false,
  })
  return { ...q, offered: announced ?? q.isSuccess }
}

/** Cloud providers (`GET /api/v1/providers`); only asked for when the server announces them. */
export function useProviders() {
  const offered = providersOffered(useInfo().data)
  const q = useQuery({
    queryKey: ['providers'],
    queryFn: ({ signal }) => api.providers(signal),
    staleTime: 15_000,
    retry: false,
    enabled: offered && !isDemo,
  })
  return { ...q, offered }
}

/** The server's machine (`GET /api/v1/hardware`); never asked for in the live demo, whose machine is the browser. */
export function useHardware() {
  const offered = recommendationsOffered(useInfo().data) && !isDemo
  const q = useQuery({
    queryKey: ['hardware'],
    queryFn: ({ signal }) => api.hardware(false, signal),
    staleTime: 60_000,
    retry: false,
    enabled: offered,
  })
  return { ...q, offered }
}

/** Which optional screens this server has. */
export function useNavFeatures(): NavFeatures {
  const info = useInfo().data
  return {
    wordLists: useWordLists().offered,
    providers: providersOffered(info),
    telephony: telephonyOffered(info) && !isDemo,
    speakers: Boolean(info?.capabilities?.speakers),
    keys: keysOffered(info) && !isDemo,
  }
}

/** Phone calls: providers, agents, numbers (`GET /api/v1/telephony`); only asked for when the server has them. */
export function useTelephony() {
  const offered = telephonyOffered(useInfo().data)
  const q = useQuery({
    queryKey: ['telephony'],
    queryFn: ({ signal }) => api.telephony(signal),
    staleTime: 10_000,
    retry: false,
    enabled: offered && !isDemo,
  })
  return { ...q, offered }
}

/** Recent calls, refreshed while the screen is open (calls in progress change by the second). */
export function useCalls(enabled = true) {
  return useQuery({
    queryKey: ['calls'],
    queryFn: ({ signal }) => api.calls({ limit: 50 }, signal),
    refetchInterval: 3000,
    retry: false,
    enabled: enabled && !isDemo,
  })
}

export function useCall(id: string | null) {
  return useQuery({
    queryKey: ['call', id],
    queryFn: ({ signal }) => api.call(id as string, signal),
    enabled: id != null,
    retry: false,
    refetchInterval: (q) =>
      q.state.data && q.state.data.status === 'ended' && q.state.data.summary !== null ? false : 2000,
  })
}

export function useSamples() {
  return useQuery({ queryKey: ['samples'], queryFn: ({ signal }) => api.samples(signal), staleTime: 300_000 })
}

export function useModels(task: Task): { models: CatalogModel[]; isLoading: boolean; error: Error | null } {
  const q = useCatalog()
  // In the live demo only models that run in the page can be used; the rest of the catalog is a read-only listing.
  const models = useMemo(
    () => (q.data?.models ?? []).filter((m) => m.task === task && (!isDemo || m.runtime === 'browser')),
    [q.data, task],
  )
  return { models, isLoading: q.isLoading, error: q.error }
}

const TIER_ORDER = { realtime: 0, balanced: 1, accuracy: 2 } as const

/** The model to use for a task: the stored choice if still valid, else the best installed one. */
export function pickDefault(models: CatalogModel[], stored: string | null): CatalogModel | null {
  const chosen = models.find((m) => m.id === stored)
  if (chosen) return chosen
  const ranked = [...models].sort(
    (a, b) =>
      Number(b.loaded) - Number(a.loaded) ||
      Number(b.installed) - Number(a.installed) ||
      TIER_ORDER[a.tier] - TIER_ORDER[b.tier],
  )
  return ranked[0] ?? null
}

export function useSelectedModel(task: 'stt' | 'tts') {
  const { models, isLoading, error } = useModels(task)
  const p = useStore(prefs)
  const stored = task === 'stt' ? p.sttModel : p.ttsModel
  const model = useMemo(() => pickDefault(models, stored), [models, stored])
  const select = (id: string) => setPref(task === 'stt' ? 'sttModel' : 'ttsModel', id)
  return { models, model, select, isLoading, error }
}

export function describeError(err: unknown): string {
  if (err instanceof NetworkError) return 'The server is unreachable.'
  if (err instanceof ApiError) return err.message
  return err instanceof Error ? err.message : 'Something went wrong.'
}

type Patch = (m: CatalogModel) => CatalogModel

export function useModelActions() {
  const qc = useQueryClient()
  const patch = (id: string, fn: Patch) =>
    qc.setQueryData<Catalog>(['catalog'], (old) =>
      old ? { models: old.models.map((m) => (m.id === id ? fn(m) : m)) } : old,
    )
  const refresh = () => qc.invalidateQueries({ queryKey: ['catalog'] })

  const pull = useMutation({
    mutationFn: (id: string) => api.pull(id),
    onSuccess: (res, id) => {
      patch(id, (m) => ({ ...m, pulling: { job_id: res.job_id, progress: 0 } }))
      const warning = warningText(res.warnings)
      // a notice, not a refusal: the download has started either way
      if (warning) toast.warning(`Pulling ${id} anyway`, { description: warning, duration: 12_000 })
      else toast.info(`Pulling ${id}`, { description: 'You can keep working; progress shows on the model.' })
    },
    onError: (err, id) => toast.error(`Could not pull ${id}`, { description: describeError(err) }),
  })
  const load = useMutation({
    mutationFn: (id: string) => api.load(id),
    onSuccess: (res, id) => {
      patch(id, (m) => ({ ...m, loaded: true }))
      const warning = warningText(res?.warnings)
      if (warning) toast.warning(`${id} loaded`, { description: warning, duration: 12_000 })
      else toast.success(`${id} loaded`)
    },
    onError: (err, id) => toast.error(`Could not load ${id}`, { description: describeError(err) }),
    onSettled: refresh,
  })
  const unload = useMutation({
    mutationFn: (id: string) => api.unload(id),
    onSuccess: (_r, id) => {
      patch(id, (m) => ({ ...m, loaded: false }))
      toast.success(`${id} unloaded`)
    },
    onError: (err, id) => toast.error(`Could not unload ${id}`, { description: describeError(err) }),
    onSettled: refresh,
  })
  const remove = useMutation({
    mutationFn: (id: string) => api.remove(id),
    onSuccess: (_r, id) => {
      patch(id, (m) => ({ ...m, loaded: false, installed: false }))
      toast.success(`${id} deleted`)
    },
    onError: (err, id) => toast.error(`Could not delete ${id}`, { description: describeError(err) }),
    onSettled: refresh,
  })
  return { pull, load, unload, remove }
}

export function useLive() {
  return useStore(live.store)
}
