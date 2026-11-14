import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Download, FileUp, Plus, Save, Search, Trash2, Undo2, Wand2, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { ErrorState, Panel } from '@/components/panel'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { describeError } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import { download } from '@/lib/bench'
import { emptyRow, importType, lineOf, MATCHES, rowForLine, rowProblem, toRows, toTsv, type Row } from '@/lib/wordlists'
import { cn } from '@/lib/utils'

/** Server messages are kept by word and language, so they survive the reload after a save. */
const noteKey = (r: Row) => `${r.word.trim()}\t${r.lang.trim() || '*'}`

/** Rows drawn at once; long lists are searched or expanded. */
const PAGE = 150

const FIELDS: { key: keyof Omit<Row, 'key' | 'match'>; label: string; placeholder: string; mono?: boolean }[] = [
  { key: 'word', label: 'Word', placeholder: 'word' },
  { key: 'lang', label: 'Language', placeholder: '*', mono: true },
  { key: 'ipa', label: 'IPA', placeholder: 'IPA', mono: true },
  { key: 'say', label: 'Say', placeholder: 'spoken form' },
  { key: 'note', label: 'Note', placeholder: 'note' },
]

const GRID =
  'grid grid-cols-2 gap-1.5 sm:grid-cols-[minmax(7rem,1.2fr)_4.5rem_minmax(7rem,1.2fr)_minmax(7rem,1.2fr)_6.5rem_minmax(5rem,1fr)_auto]'

/**
 * One word list as editable rows: search, add and delete rows, save (as TSV),
 * import a file, export TSV or PLS, delete the list; an IPA helper per row.
 * Warnings and errors from the server are shown next to the rows they name.
 */
export function WordListEditor({
  name,
  helperModels,
  onDeleted,
  className,
}: {
  name: string
  /** Voices the IPA helper asks (a phoneme voice gives IPA, a text voice a respelling). */
  helperModels: string[]
  onDeleted: () => void
  className?: string
}) {
  const qc = useQueryClient()
  const q = useQuery({ queryKey: ['wordlists', name], queryFn: ({ signal }) => api.wordList(name, signal) })
  const [rows, setRows] = useState<Row[]>([])
  const [dirty, setDirty] = useState(false)
  const [filter, setFilter] = useState('')
  const [limit, setLimit] = useState(PAGE)
  const [saving, setSaving] = useState(false)
  const [notes, setNotes] = useState<Record<string, string>>({})
  const [hints, setHints] = useState<Record<string, string>>({})
  const [helping, setHelping] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  // Fresh data replaces the rows unless the user has unsaved edits (an SSE refresh must not eat them).
  useEffect(() => {
    if (q.data && !dirty) {
      setRows(toRows(q.data.entries))
    }
  }, [q.data, dirty])

  const shown = useMemo(() => {
    const needle = filter.trim().toLowerCase()
    return needle
      ? rows.filter((r) => `${r.word} ${r.ipa} ${r.say} ${r.note} ${r.lang}`.toLowerCase().includes(needle))
      : rows
  }, [rows, filter])

  const edit = (key: string, patch: Partial<Row>) => {
    setRows((rs) => rs.map((r) => (r.key === key ? { ...r, ...patch } : r)))
    setDirty(true)
  }
  const remove = (key: string) => {
    setRows((rs) => rs.filter((r) => r.key !== key))
    setDirty(true)
  }
  const add = () => {
    const lang = q.data?.languages.find((l) => l !== '*') ?? '*'
    setRows((rs) => [emptyRow(lang), ...rs])
    setFilter('')
    setDirty(true)
  }

  const problems = rows.map(rowProblem)
  const firstProblem = problems.find((p) => p != null) ?? null

  const save = async () => {
    const { tsv, sent } = toTsv(rows)
    setSaving(true)
    setError(null)
    try {
      const res = await api.putWordList(name, tsv)
      const byRow: Record<string, string> = {}
      for (const w of res.warnings) {
        const r = rowForLine(sent, w.line)
        if (r) byRow[noteKey(r)] = w.message
      }
      setNotes(byRow)
      setDirty(false)
      await qc.invalidateQueries({ queryKey: ['wordlists'] })
      toast.success(`Saved ${name}`, {
        description: `${res.entries} entr${res.entries === 1 ? 'y' : 'ies'}${res.warnings.length ? `, ${res.warnings.length} warning(s)` : ''}`,
      })
    } catch (err) {
      const message = describeError(err)
      const line = lineOf(message)
      const r = line != null ? rowForLine(sent, line) : null
      if (r) setNotes({ [noteKey(r)]: message })
      setError(r ? `${message} (the row is marked below)` : message)
    } finally {
      setSaving(false)
    }
  }

  const importFile = async (f: File) => {
    if (dirty && !globalThis.confirm('Importing replaces this list and drops your unsaved edits. Continue?')) return
    try {
      const res = await api.putWordList(name, await f.text(), importType(f.name))
      setDirty(false)
      await qc.invalidateQueries({ queryKey: ['wordlists'] })
      toast.success(`Imported ${f.name}`, {
        description: `${res.entries} entries${res.warnings.length ? `, ${res.warnings.length} warning(s)` : ''}`,
      })
    } catch (err) {
      setError(describeError(err))
    }
  }

  const exportAs = async (format: 'tsv' | 'pls', lang?: string) => {
    try {
      const blob = await api.exportWordList(name, format, lang)
      download(`${name}${lang ? `-${lang}` : ''}.${format}`, new Uint8Array(await blob.arrayBuffer()), blob.type)
    } catch (err) {
      toast.error('Could not export', { description: describeError(err) })
    }
  }

  /** Ask the voices how they would say this word without any list: IPA from a phoneme voice, a respelling from a text voice. */
  const suggest = async (r: Row) => {
    if (!r.word.trim() || helperModels.length === 0) return
    setHelping(r.key)
    try {
      const res = await api.previewWordList('-', {
        text: r.word,
        models: helperModels.slice(0, 4),
        language: r.lang && r.lang !== '*' ? r.lang : undefined,
      })
      const phon = res.previews.find((p) => p.phonemes)
      const text = res.previews.find((p) => p.text_input.reads !== 'phonemes')
      if (phon?.phonemes && !r.ipa.trim()) edit(r.key, { ipa: phon.phonemes.split('\n')[0]?.trim() ?? '' })
      const parts = [
        phon?.phonemes ? `${phon.model} reads /${phon.phonemes.trim()}/` : null,
        text ? `${text.model} gets “${text.text}”` : null,
      ].filter(Boolean)
      setHints((h) => ({
        ...h,
        [r.key]: parts.length ? `Without the list: ${parts.join('; ')}.` : 'No voice gave a suggestion.',
      }))
    } catch (err) {
      setHints((h) => ({ ...h, [r.key]: describeError(err) }))
    } finally {
      setHelping(null)
    }
  }

  const removeList = async () => {
    try {
      await api.deleteWordList(name)
      await qc.invalidateQueries({ queryKey: ['wordlists'] })
      toast.success(`Deleted ${name}`)
      onDeleted()
    } catch (err) {
      toast.error('Could not delete the list', { description: describeError(err) })
    }
  }

  const languages = (q.data?.languages ?? []).filter((l) => l !== '*')

  return (
    <Panel
      title={<span className="num">{name}</span>}
      description={
        q.data
          ? `${rows.length} entr${rows.length === 1 ? 'y' : 'ies'}${dirty ? ' · unsaved changes' : ''}`
          : 'Loading…'
      }
      className={className}
      data-testid="wordlist-editor"
      actions={
        <>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button size="sm" variant="ghost" disabled={!q.data}>
                <Download /> Export
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onSelect={() => void exportAs('tsv')}>TSV (all languages)</DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuLabel className="text-xs text-muted-foreground">PLS lexicon for</DropdownMenuLabel>
              {(languages.length ? languages : ['en']).map((l) => (
                <DropdownMenuItem key={l} onSelect={() => void exportAs('pls', l)}>
                  PLS · <span className="num">{l}</span>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <Button size="sm" variant="ghost" onClick={() => fileInput.current?.click()} disabled={!q.data}>
            <FileUp /> Import
          </Button>
          <input
            ref={fileInput}
            type="file"
            accept=".tsv,.txt,.pls,.xml,.json,text/tab-separated-values,application/pls+xml,application/json"
            className="sr-only"
            tabIndex={-1}
            aria-label={`Import a file into ${name}`}
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) void importFile(f)
              e.target.value = ''
            }}
          />
          <Button
            size="icon-sm"
            variant="ghost"
            onClick={() => setConfirmDelete(true)}
            aria-label={`Delete the list ${name}`}
          >
            <Trash2 />
          </Button>
        </>
      }
    >
      {q.isLoading ? (
        <Skeleton className="h-40 w-full" />
      ) : q.isError ? (
        <ErrorState title="Could not load the list" error={describeError(q.error)} onRetry={() => void q.refetch()} />
      ) : (
        <div className="grid gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative min-w-40 flex-1 sm:max-w-64">
              <Search
                className="pointer-events-none absolute top-2 left-2.5 size-4 text-muted-foreground"
                aria-hidden
              />
              <Input
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Search entries"
                aria-label="Search entries"
                type="search"
                className="h-8 pl-8"
              />
            </div>
            <Button size="sm" variant="outline" onClick={add} data-testid="wordlist-add">
              <Plus /> Add entry
            </Button>
            <div className="ml-auto flex items-center gap-2">
              {dirty && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setDirty(false)
                    setError(null)
                  }}
                >
                  <Undo2 /> Discard
                </Button>
              )}
              <Button
                size="sm"
                onClick={() => void save()}
                disabled={!dirty || saving || firstProblem != null}
                title={firstProblem ?? undefined}
                data-testid="wordlist-save"
              >
                <Save /> {saving ? 'Saving…' : 'Save'}
              </Button>
            </div>
          </div>
          {error && (
            <p role="alert" className="rounded-lg border border-critical/30 bg-critical/10 p-3 text-[13px] leading-5">
              {error}
            </p>
          )}

          <div className={cn(GRID, 'max-sm:hidden')} aria-hidden>
            {['Word', 'Lang', 'IPA', 'Say', 'Match', 'Note', ''].map((h, i) => (
              <span key={i} className="text-[11px] font-medium text-muted-foreground">
                {h}
              </span>
            ))}
          </div>
          {rows.length === 0 ? (
            <p className="rounded-lg border border-dashed border-border p-4 text-center text-[13px] text-muted-foreground">
              This list is empty. Add an entry, or import a TSV, PLS or JSON file.
            </p>
          ) : (
            <ul className="grid gap-2 sm:gap-1.5" data-testid="wordlist-rows">
              {shown.slice(0, limit).map((r) => {
                const n = rows.indexOf(r) + 1
                const problem = problems[n - 1]
                const note = notes[noteKey(r)]
                return (
                  <li
                    key={r.key}
                    className="rounded-lg border border-border p-2 sm:rounded-none sm:border-0 sm:p-0"
                    data-testid="wordlist-row"
                  >
                    <div className={GRID}>
                      {FIELDS.slice(0, 4).map((f) => (
                        <Input
                          key={f.key}
                          value={r[f.key]}
                          placeholder={f.placeholder}
                          onChange={(e) => edit(r.key, { [f.key]: e.target.value })}
                          aria-label={`${f.label}, entry ${n}`}
                          aria-invalid={f.key === 'word' && problem != null ? true : undefined}
                          className={cn('h-8', f.mono && 'num')}
                        />
                      ))}
                      <Select value={r.match} onValueChange={(v) => edit(r.key, { match: v })}>
                        <SelectTrigger className="h-8 w-full" aria-label={`Match rule, entry ${n}`}>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent position="popper" align="start">
                          {MATCHES.map((m) => (
                            <SelectItem key={m} value={m}>
                              {m}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Input
                        value={r.note}
                        onChange={(e) => edit(r.key, { note: e.target.value })}
                        aria-label={`Note, entry ${n}`}
                        placeholder="note"
                        className="h-8"
                      />
                      <span className="col-span-2 flex items-center justify-end gap-0.5 sm:col-span-1">
                        <Button
                          size="icon-sm"
                          variant="ghost"
                          onClick={() => void suggest(r)}
                          disabled={!r.word.trim() || helperModels.length === 0 || helping === r.key}
                          aria-label={`Suggest a pronunciation for entry ${n}`}
                          title={
                            helperModels.length
                              ? 'Ask the preview voices how they say this word'
                              : 'Choose voices in the preview first'
                          }
                        >
                          <Wand2 className={helping === r.key ? 'live-dot' : undefined} />
                        </Button>
                        <Button
                          size="icon-sm"
                          variant="ghost"
                          onClick={() => remove(r.key)}
                          aria-label={`Delete entry ${n}`}
                        >
                          <X />
                        </Button>
                      </span>
                    </div>
                    {(problem || note || hints[r.key]) && (
                      <div className="mt-1 grid gap-0.5 text-xs leading-4">
                        {problem && <p className="text-critical-text">{problem}</p>}
                        {note && (
                          <p className="text-warning-text" role="status">
                            {note}
                          </p>
                        )}
                        {hints[r.key] && <p className="text-muted-foreground">{hints[r.key]}</p>}
                      </div>
                    )}
                  </li>
                )
              })}
            </ul>
          )}
          {shown.length > limit && (
            <Button
              size="sm"
              variant="outline"
              className="justify-self-start"
              onClick={() => setLimit((l) => l + PAGE)}
            >
              Show {Math.min(PAGE, shown.length - limit)} more of {shown.length - limit}
            </Button>
          )}
          {filter && shown.length === 0 && <p className="text-xs text-muted-foreground">No entry matches.</p>}
        </div>
      )}

      <Dialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>
              Delete <span className="num">{name}</span>?
            </DialogTitle>
            <DialogDescription>
              Voices that use this list read its words their own way again. Export it first if you may want it back.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setConfirmDelete(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                setConfirmDelete(false)
                void removeList()
              }}
            >
              <Trash2 /> Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Panel>
  )
}
