/**
 * bolo's version, taken at build time from the server package (server/src/bolo/__init__.py; see
 * `boloVersion()` in vite.config.ts), or null when the console was built without the server sources.
 */
export const BOLO_VERSION: string | null = (import.meta.env.VITE_BOLO_VERSION as string | null | undefined) ?? null
