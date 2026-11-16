import { Store } from '@/lib/store'

/**
 * Voices saved in this browser for cloning: the reference recording (16-bit,
 * 24 kHz), its transcript and where it came from. Kept in IndexedDB; nothing
 * is uploaded. Where IndexedDB is missing or refuses (some private windows),
 * voices live for this visit only and the page says so.
 */

export interface SavedVoice {
  id: string
  name: string
  createdAt: number
  sampleRate: number
  /** 16-bit PCM samples. */
  pcm: Int16Array
  transcript: string
  language: string
  source: 'mic' | 'upload' | 'example'
}

export interface SavedVoicesState {
  voices: Omit<SavedVoice, 'pcm'>[]
  /** False when saving falls back to memory (this visit only). */
  persistent: boolean
  loaded: boolean
}

const DB = 'bolo-clone'
const STORE = 'voices'

export interface VoiceBackend {
  list(): Promise<SavedVoice[]>
  put(v: SavedVoice): Promise<void>
  delete(id: string): Promise<void>
}

function request<T>(r: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result)
    r.onerror = () => reject(r.error ?? new Error('IndexedDB request failed'))
  })
}

export function indexedDbBackend(factory: IDBFactory): VoiceBackend {
  let db: Promise<IDBDatabase> | null = null
  const open = () =>
    (db ??= new Promise<IDBDatabase>((resolve, reject) => {
      const r = factory.open(DB, 1)
      r.onupgradeneeded = () => {
        if (!r.result.objectStoreNames.contains(STORE)) r.result.createObjectStore(STORE, { keyPath: 'id' })
      }
      r.onsuccess = () => resolve(r.result)
      r.onerror = () => reject(r.error ?? new Error('IndexedDB is not available'))
      r.onblocked = () => reject(new Error('IndexedDB is blocked'))
    }))
  const tx = async (mode: IDBTransactionMode) => (await open()).transaction(STORE, mode).objectStore(STORE)
  return {
    async list() {
      return (await request((await tx('readonly')).getAll())) as SavedVoice[]
    },
    async put(v) {
      await request((await tx('readwrite')).put(v))
    },
    async delete(id) {
      await request((await tx('readwrite')).delete(id))
    },
  }
}

export function memoryBackend(): VoiceBackend {
  const m = new Map<string, SavedVoice>()
  return {
    list: async () => [...m.values()],
    put: async (v) => void m.set(v.id, v),
    delete: async (id) => void m.delete(id),
  }
}

const strip = ({ pcm: _pcm, ...rest }: SavedVoice): Omit<SavedVoice, 'pcm'> => rest

export class SavedVoices {
  readonly store = new Store<SavedVoicesState>({ voices: [], persistent: true, loaded: false })
  private backend: VoiceBackend
  private readonly memory = memoryBackend()
  private cache = new Map<string, SavedVoice>()

  constructor(backend?: VoiceBackend | null) {
    let b = backend ?? null
    if (b === null && backend === undefined) {
      try {
        b = globalThis.indexedDB ? indexedDbBackend(globalThis.indexedDB) : null
      } catch {
        b = null
      }
    }
    this.backend = b ?? this.memory
    if (!b) this.store.patch({ persistent: false })
  }

  /** Read the saved voices (once per visit). Falls back to memory when IndexedDB fails. */
  async load(): Promise<void> {
    try {
      const all = await this.backend.list()
      this.cache = new Map(all.map((v) => [v.id, v]))
    } catch {
      this.backend = this.memory
      this.cache = new Map()
      this.store.patch({ persistent: false })
    }
    this.publish()
  }

  private publish(): void {
    const voices = [...this.cache.values()].sort((a, b) => b.createdAt - a.createdAt).map(strip)
    this.store.patch({ voices, loaded: true })
  }

  get(id: string): SavedVoice | null {
    return this.cache.get(id) ?? null
  }

  async save(v: SavedVoice): Promise<void> {
    try {
      await this.backend.put(v)
    } catch {
      // Quota or a private window: keep it for this visit.
      this.backend = this.memory
      this.store.patch({ persistent: false })
      await this.memory.put(v)
    }
    this.cache.set(v.id, v)
    this.publish()
  }

  async remove(id: string): Promise<void> {
    try {
      await this.backend.delete(id)
    } catch {
      // already gone, or storage unavailable
    }
    this.cache.delete(id)
    this.publish()
  }
}

let shared: SavedVoices | null = null
export function savedVoices(): SavedVoices {
  shared ??= new SavedVoices()
  return shared
}
