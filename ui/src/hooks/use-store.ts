import { useSyncExternalStore } from 'react'
import type { Store } from '@/lib/store'

export function useStore<T>(store: Pick<Store<T>, 'get' | 'subscribe'>): T {
  return useSyncExternalStore(store.subscribe, store.get, store.get)
}
