import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo } from 'react'
import { toast } from 'sonner'
import { api, ApiError, NetworkError } from '@/lib/api/client'
import { isDemo } from '@/lib/api/config'
import type { Catalog, CatalogModel, Component, Task } from '@/lib/api/types'
import { live } from '@/lib/live'
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

/** Word lists, when the server has them: a 404 (older server, or the demo) reads as "not offered". */
export function useWordLists() {
  const q = useQuery({
    queryKey: ['wordlists'],
    queryFn: ({ signal }) => api.wordLists(signal),
    staleTime: 30_000,
    retry: false,
  })
  return { ...q, offered: q.isSuccess }
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
      toast.info(`Pulling ${id}`, { description: 'You can keep working; progress shows on the model.' })
    },
    onError: (err, id) => toast.error(`Could not pull ${id}`, { description: describeError(err) }),
  })
  const load = useMutation({
    mutationFn: (id: string) => api.load(id),
    onSuccess: (_r, id) => {
      patch(id, (m) => ({ ...m, loaded: true }))
      toast.success(`${id} loaded`)
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
