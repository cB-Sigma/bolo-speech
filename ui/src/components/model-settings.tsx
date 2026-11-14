import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Save } from 'lucide-react'
import { useId, useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { WordListSelect } from '@/components/wordlists/wordlist-select'
import { describeError, useWordLists } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import type { CatalogModel } from '@/lib/api/types'
import { PHONEMIZER_LABEL } from '@/lib/capabilities'
import { isIndicLanguage, ROMANISED_LABEL, ROMANISED_MODES } from '@/lib/romanised'

const SERVER = '__server'

export const NORMALIZERS: { id: string; label: string }[] = [
  { id: 'uccharan', label: 'uccharan (numbers, dates, abbreviations)' },
  { id: 'light', label: 'light (bolo’s small normaliser)' },
  { id: 'model', label: 'model (the voice’s own)' },
  { id: 'none', label: 'none (text as written)' },
]

const READS: Record<string, string> = {
  phonemes: 'phonemes from a phonemiser',
  text: 'text (the model reads it itself)',
  characters: 'characters from a fixed set',
}

/**
 * Settings pinned to one voice on the server (phonemiser, text normaliser, word lists), and its
 * text-input record, read-only, so it is clear why a voice gets IPA, markup or respellings.
 * Give it `key={m.id + settings}` so a change on the server resets the form.
 */
export function ModelSettingsSection({ m }: { m: CatalogModel }) {
  const id = useId()
  const qc = useQueryClient()
  const lists = useWordLists()
  const textInput = useQuery({
    queryKey: ['text-input', m.id],
    queryFn: ({ signal }) => api.textInput(m.id, undefined, signal),
    retry: false,
    staleTime: 30_000,
  })
  const pinned = m.settings ?? {}
  const str = (v: unknown) => (typeof v === 'string' ? v : '')
  const [phonemizer, setPhonemizer] = useState(str(pinned.phonemizer))
  const [normalizer, setNormalizer] = useState(str(pinned.text_normalizer))
  const [wordlists, setWordlists] = useState<string[]>(
    Array.isArray(pinned.wordlists) ? (pinned.wordlists as string[]) : [],
  )
  const [romanised, setRomanised] = useState(str(pinned.romanised))
  const [saving, setSaving] = useState(false)
  const choices = m.phonemizers ?? []
  const ti = textInput.data?.text_input
  // a server with the option reports the voice's mode; it applies to voices of Indian languages
  const romanisable = typeof ti?.romanised === 'string' && m.languages.some(isIndicLanguage)
  const save = async () => {
    setSaving(true)
    try {
      const body: Parameters<typeof api.setModelSettings>[1] = {}
      if (choices.length) body.phonemizer = phonemizer || null
      if (ti) body.text_normalizer = normalizer || null
      if (lists.offered) body.wordlists = wordlists.length ? wordlists : null
      if (romanisable) body.romanised = romanised || null
      await api.setModelSettings(m.id, body)
      await Promise.all([
        qc.invalidateQueries({ queryKey: ['catalog'] }),
        qc.invalidateQueries({ queryKey: ['text-input', m.id] }),
      ])
      toast.success(`Saved the settings of ${m.id}`)
    } catch (err) {
      toast.error('Could not save', { description: describeError(err) })
    } finally {
      setSaving(false)
    }
  }

  if (!choices.length && !ti && !lists.offered) return null
  return (
    <section data-testid="model-settings">
      <h3 className="mb-2 eyebrow">Pinned for this voice</h3>
      <div className="grid gap-3 rounded-lg border border-border p-3">
        {choices.length > 0 && (
          <div className="grid gap-1.5">
            <Label htmlFor={`${id}-ph`} className="text-xs">
              Phonemiser
            </Label>
            <Select value={phonemizer || SERVER} onValueChange={(v) => setPhonemizer(v === SERVER ? '' : v)}>
              <SelectTrigger id={`${id}-ph`} className="h-8 w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent position="popper" align="start">
                <SelectItem value={SERVER}>Server setting</SelectItem>
                <SelectItem value="auto">Automatic</SelectItem>
                {choices.map((c) => (
                  <SelectItem key={c.id} value={c.id} disabled={!c.available}>
                    {PHONEMIZER_LABEL[c.id] ?? c.id}
                    {c.available ? '' : ' (not installed)'}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
        {ti && (
          <div className="grid gap-1.5">
            <Label htmlFor={`${id}-norm`} className="text-xs">
              Text normaliser
            </Label>
            <Select value={normalizer || SERVER} onValueChange={(v) => setNormalizer(v === SERVER ? '' : v)}>
              <SelectTrigger id={`${id}-norm`} className="h-8 w-full" data-testid="normalizer-select">
                <SelectValue />
              </SelectTrigger>
              <SelectContent position="popper" align="start">
                <SelectItem value={SERVER}>Default for this voice ({ti.normalizer})</SelectItem>
                {NORMALIZERS.map((n) => (
                  <SelectItem key={n.id} value={n.id}>
                    {n.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
        {romanisable && (
          <div className="grid gap-1.5">
            <Label htmlFor={`${id}-rom`} className="text-xs">
              Romanised text (Latin letters written in the voice’s script)
            </Label>
            <Select value={romanised || SERVER} onValueChange={(v) => setRomanised(v === SERVER ? '' : v)}>
              <SelectTrigger id={`${id}-rom`} className="h-8 w-full" data-testid="romanised-select">
                <SelectValue />
              </SelectTrigger>
              <SelectContent position="popper" align="start">
                <SelectItem value={SERVER}>Server setting</SelectItem>
                {ROMANISED_MODES.map((mode) => (
                  <SelectItem key={mode} value={mode}>
                    {ROMANISED_LABEL[mode]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
        {lists.offered && (
          <div className="grid gap-1.5">
            <span className="text-xs font-medium">Word lists applied to every request</span>
            <WordListSelect
              names={(lists.data?.wordlists ?? []).map((l) => l.name)}
              value={wordlists}
              onChange={setWordlists}
              className="justify-self-start"
            />
          </div>
        )}
        <Button size="sm" className="justify-self-start" onClick={() => void save()} disabled={saving}>
          <Save /> {saving ? 'Saving…' : 'Save'}
        </Button>
        {ti && (
          <dl
            className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 border-t border-border pt-3 text-[13px]"
            data-testid="text-input-record"
          >
            <dt className="text-muted-foreground">Reads</dt>
            <dd className="text-right">{READS[ti.reads] ?? ti.reads}</dd>
            <dt className="text-muted-foreground">Word-list entries become</dt>
            <dd className="text-right">{ti.pronunciation}</dd>
            <dt className="text-muted-foreground">Markup</dt>
            <dd className="text-right num">{ti.markup}</dd>
            <dt className="text-muted-foreground">Latin words</dt>
            <dd className="text-right">{ti.latin}</dd>
            <dt className="text-muted-foreground">Respelling</dt>
            <dd className="text-right num">{ti.respell}</dd>
            <dt className="text-muted-foreground">Digits</dt>
            <dd className="text-right">{ti.digits}</dd>
          </dl>
        )}
      </div>
    </section>
  )
}
