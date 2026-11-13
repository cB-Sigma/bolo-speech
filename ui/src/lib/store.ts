/** Minimal external store for use with React's `useSyncExternalStore`. */
export class Store<T> {
  private state: T
  private listeners = new Set<() => void>()

  constructor(initial: T) {
    this.state = initial
  }

  get = (): T => this.state

  set(next: T | ((prev: T) => T)): void {
    const value = typeof next === 'function' ? (next as (prev: T) => T)(this.state) : next
    if (Object.is(value, this.state)) return
    this.state = value
    for (const l of this.listeners) l()
  }

  patch(partial: Partial<T>): void {
    this.set({ ...this.state, ...partial })
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }
}
