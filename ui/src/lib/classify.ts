/** Helpers of the Classify screen: labels typed by hand, and example clips dropped as folders. */

const AUDIO_EXT = /\.(wav|flac|mp3|ogg|oga|opus|m4a|aac|webm|aiff?|mka)$/i

/** Labels from the text box: one per line, or comma-separated on a line; blanks and repeats dropped. */
export function parseLabels(text: string): string[] {
  const out: string[] = []
  for (const line of text.split('\n')) {
    for (const part of line.split(',')) {
      const label = part.trim()
      if (label && !out.includes(label)) out.push(label)
    }
  }
  return out
}

/** The audio files among `files` (by type, else by extension); hidden files such as `.DS_Store` are skipped. */
export function audioFiles(files: Iterable<File>): File[] {
  return [...files].filter((f) => !f.name.startsWith('.') && (f.type.startsWith('audio/') || AUDIO_EXT.test(f.name)))
}

/** The parts of the File and Directory Entries API a drop gives (`DataTransferItem.webkitGetAsEntry()`). */
export interface DropEntry {
  isFile: boolean
  isDirectory: boolean
  name: string
  file?: (ok: (f: File) => void, fail?: (e: unknown) => void) => void
  createReader?: () => { readEntries: (ok: (entries: DropEntry[]) => void, fail?: (e: unknown) => void) => void }
}

async function walk(entry: DropEntry, out: File[], depth = 0): Promise<void> {
  if (entry.isFile && entry.file) {
    out.push(await new Promise<File>((ok, fail) => entry.file!(ok, fail)))
    return
  }
  if (!entry.isDirectory || !entry.createReader || depth > 8) return
  const reader = entry.createReader()
  // readEntries gives at most 100 entries at a time: call it until it gives none
  for (;;) {
    const batch = await new Promise<DropEntry[]>((ok, fail) => reader.readEntries(ok, fail))
    if (!batch.length) break
    for (const child of batch) await walk(child, out, depth + 1)
  }
}

/** Every audio file in what was dropped: files, and the files of folders (searched through). */
export async function filesFromDrop(dt: DataTransfer): Promise<File[]> {
  const entries: DropEntry[] = []
  for (const item of Array.from(dt.items ?? [])) {
    const entry = (item as DataTransferItem & { webkitGetAsEntry?: () => DropEntry | null }).webkitGetAsEntry?.()
    if (entry) entries.push(entry)
  }
  if (!entries.length) return audioFiles(Array.from(dt.files ?? []))
  const out: File[] = []
  for (const entry of entries) await walk(entry, out)
  return audioFiles(out)
}

/** "Top folder name" of a dropped folder's files, as a class label suggestion (`webkitRelativePath`). */
export function folderName(files: File[]): string | null {
  const path = (files[0] as (File & { webkitRelativePath?: string }) | undefined)?.webkitRelativePath
  return path?.includes('/') ? (path.split('/')[0] ?? null) : null
}
