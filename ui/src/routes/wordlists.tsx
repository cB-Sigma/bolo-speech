import { useQueryClient } from '@tanstack/react-query'
import { BookA, FileUp, Plus } from 'lucide-react'
import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { toast } from 'sonner'
import { ComponentInstallButton } from '@/components/component-install'
import { NeedsServer } from '@/components/demo'
import { Page } from '@/components/layout/app-shell'
import { Empty, ErrorState, Panel } from '@/components/panel'
import { WordListEditor } from '@/components/wordlists/editor'
import { MAX_PREVIEW_MODELS, WordListPreviewPane } from '@/components/wordlists/preview'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { describeError, useCatalog, useComponents, useWordLists } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import { isDemo } from '@/lib/api/config'
import { fmtBytes, fmtClock } from '@/lib/format'
import { importType, NAME_RE } from '@/lib/wordlists'
import { cn } from '@/lib/utils'

/** Pronunciation word lists: the lists, an editor for one, and a preview across voices. */
export default function WordListsPage() {
  const lists = useWordLists()
  const catalog = useCatalog()
  const qc = useQueryClient()
  const uccharan = useComponents().data?.components.find((c) => c.id === 'uccharan')
  const [params, setParams] = useSearchParams()
  const selected = params.get('list')
  const [newName, setNewName] = useState('')
  const [chosen, setChosen] = useState<string[] | null>(null)
  const id = useId()
  const importInput = useRef<HTMLInputElement>(null)

  const ttsModels = useMemo(
    () => (catalog.data?.models ?? []).filter((m) => m.task === 'tts' && m.installed && m.available !== false),
    [catalog.data],
  )
  // Default comparison: up to two installed voices, so both a phoneme voice and a text voice tend to show.
  const previewModels = chosen ?? ttsModels.slice(0, 2).map((m) => m.id)
  const all = lists.data?.wordlists ?? []
  const select = (name: string | null) => {
    if (name) params.set('list', name)
    else params.delete('list')
    setParams(params, { replace: true })
  }
  useEffect(() => {
    if (selected && lists.data && !all.some((l) => l.name === selected)) select(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lists.data])

  const nameProblem = newName && !NAME_RE.test(newName) ? 'Lower-case letters, digits, - and _ only.' : null
  const create = async (body = 'word\tlang\tipa\tsay\tmatch\tnote\n', type?: string, label?: string) => {
    const name = newName.trim()
    if (!NAME_RE.test(name)) return
    if (all.some((l) => l.name === name)) {
      toast.error(`A list named ${name} exists`, { description: 'Open it from the table, or choose another name.' })
      return
    }
    try {
      await api.putWordList(name, body, type)
      await qc.invalidateQueries({ queryKey: ['wordlists'] })
      setNewName('')
      select(name)
      if (label) toast.success(`Imported ${label} as ${name}`)
    } catch (err) {
      toast.error('Could not create the list', { description: describeError(err) })
    }
  }

  if (isDemo) {
    return (
      <Page title="Word lists">
        <NeedsServer title="Word lists live on a bolo server">
          They tell every voice of a server how to say names, brands and abbreviations. This demo runs in your browser
          only.
        </NeedsServer>
      </Page>
    )
  }

  return (
    <Page title="Word lists">
      <p className="mb-4 max-w-3xl text-[13px] leading-5 text-muted-foreground">
        Tell voices how to say names, brands and abbreviations. An entry gives the IPA (for voices that read phonemes)
        and or a spoken form (for voices that read text). Lists apply when a request names them, when they are pinned to
        a model, and always for the list named <span className="num text-foreground">default</span>.
      </p>
      {lists.data && !lists.data.available && (
        <Alert className="mb-4">
          <BookA />
          <AlertTitle>Word lists need the uccharan package</AlertTitle>
          <AlertDescription>
            <span>
              Lists can be listed and deleted, but not read, saved or applied until it is installed (
              <span className="num">pip install &apos;bolo-speech[uccharan]&apos;</span>).
            </span>
            {uccharan && !uccharan.installed && <ComponentInstallButton component={uccharan} className="mt-2" />}
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,4fr)_minmax(0,8fr)] xl:items-start">
        <div className="min-w-0 xl:col-start-1 xl:row-start-1">
          <Panel
            title="Lists"
            description={lists.data ? `${all.length} on this server` : undefined}
            bodyClassName="p-0"
          >
            {lists.isLoading ? (
              <Skeleton className="m-4 h-24" />
            ) : lists.isError ? (
              <ErrorState
                title="Word lists are not available"
                error={describeError(lists.error)}
                onRetry={() => void lists.refetch()}
              />
            ) : all.length === 0 ? (
              <Empty icon={BookA} title="No word lists yet" className="py-6">
                Create one below, or import a TSV, PLS or JSON file.
              </Empty>
            ) : (
              <ul className="divide-y divide-border" data-testid="wordlists">
                {all.map((l) => (
                  <li key={l.name}>
                    <button
                      type="button"
                      onClick={() => select(l.name)}
                      aria-current={selected === l.name ? 'true' : undefined}
                      className={cn(
                        'grid w-full grid-cols-[minmax(0,1fr)_auto] gap-x-3 px-4 py-2.5 text-left transition-colors duration-150 hover:bg-muted/50',
                        selected === l.name && 'bg-muted/60',
                      )}
                    >
                      <span className="truncate num text-[13px] font-medium">{l.name}</span>
                      <span className="num text-xs text-muted-foreground">
                        {l.entries} entr{l.entries === 1 ? 'y' : 'ies'}
                      </span>
                      <span className="truncate text-xs text-muted-foreground">
                        {l.languages.join(', ') || 'no language'} · {fmtBytes(l.bytes)}
                      </span>
                      <span className="num text-xs text-muted-foreground">{fmtClock(l.updated)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <form
              className="grid gap-1.5 border-t border-border p-4"
              onSubmit={(e) => {
                e.preventDefault()
                void create()
              }}
            >
              <Label htmlFor={`${id}-name`} className="text-xs">
                New list
              </Label>
              <div className="flex flex-wrap gap-2">
                <Input
                  id={`${id}-name`}
                  value={newName}
                  onChange={(e) => setNewName(e.target.value.toLowerCase())}
                  placeholder="brands"
                  className="h-8 min-w-32 flex-1 num"
                  aria-invalid={nameProblem ? true : undefined}
                  disabled={lists.isError}
                />
                <Button type="submit" size="sm" disabled={!newName || nameProblem != null || lists.isError}>
                  <Plus /> Create
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  disabled={!newName || nameProblem != null || lists.isError}
                  onClick={() => importInput.current?.click()}
                >
                  <FileUp /> Import file
                </Button>
                <input
                  ref={importInput}
                  type="file"
                  accept=".tsv,.txt,.pls,.xml,.json"
                  className="sr-only"
                  tabIndex={-1}
                  aria-label="Import a file as a new list"
                  onChange={(e) => {
                    const f = e.target.files?.[0]
                    if (f) void f.text().then((t) => create(t, importType(f.name), f.name))
                    e.target.value = ''
                  }}
                />
              </div>
              {nameProblem && <p className="text-xs text-critical-text">{nameProblem}</p>}
            </form>
          </Panel>
        </div>
        <div className="min-w-0 xl:col-start-2 xl:row-span-2 xl:row-start-1">
          {selected ? (
            <WordListEditor
              key={selected}
              name={selected}
              helperModels={previewModels}
              onDeleted={() => select(null)}
            />
          ) : (
            <div className="rounded-xl border border-dashed border-border bg-card">
              <Empty icon={BookA} title="Choose a list">
                Pick a list to edit its entries and hear how the voices read them.
              </Empty>
            </div>
          )}
        </div>
        <div className="min-w-0 xl:col-start-1 xl:row-start-2">
          {selected && (
            <WordListPreviewPane
              name={selected}
              models={ttsModels}
              chosen={previewModels}
              onChosenChange={(ids) => setChosen(ids.slice(0, MAX_PREVIEW_MODELS))}
            />
          )}
        </div>
      </div>
    </Page>
  )
}
