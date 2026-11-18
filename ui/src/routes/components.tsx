import { Puzzle } from 'lucide-react'
import { componentKind, ComponentsCard, KIND_LABEL } from '@/components/component-install'
import { NeedsServer } from '@/components/demo'
import { Page } from '@/components/layout/app-shell'
import { Empty, ErrorState, Panel } from '@/components/panel'
import { Skeleton } from '@/components/ui/skeleton'
import { describeError, useComponents } from '@/hooks/use-api'
import { isDemo } from '@/lib/api/config'

const KIND_ORDER = ['phonemizer', 'language-pack', 'tts-server', 'engine-plugin', 'pytorch', 'other']

const KIND_NOTE: Record<string, string> = {
  phonemizer: 'Turn text into the phonemes many voices read. uccharan covers English and Hindi; eSpeak NG the rest.',
  'language-pack':
    "uccharan's word tables for the languages that are not part of it (downloaded the first time a voice needs one), Kokoro dictionaries, and share-alike data for improved mode and romanised text (CC BY-SA 4.0, installed only when you ask).",
  'tts-server':
    'Separate speech servers for the voices marked llama.cpp or audio.cpp. bolo starts them itself, or connects to one you run (setting tts_servers).',
  'engine-plugin': 'Out-of-process engines for models that need PyTorch. Installed as Python packages by the operator.',
  pytorch:
    "The PyTorch build each bolo-torch environment has. PyPI's torch is CPU only on Windows and macOS: install the GPU version here to run its models on the GPU.",
  other: 'Other optional software.',
}

/** Every optional component the server knows: install state, version, licence, and an install button. */
export default function ComponentsPage() {
  const q = useComponents()
  const list = q.data?.components ?? []
  const kinds = KIND_ORDER.filter((k) => list.some((c) => componentKind(c) === k))
  return (
    <Page title="Components">
      <p className="mb-4 max-w-3xl text-[13px] leading-5 text-muted-foreground">
        Software some models need that bolo does not ship. Installing one here runs on the server, shows its progress,
        and takes effect without a restart; the models that needed it become usable.
      </p>
      {isDemo && (
        <NeedsServer title="Installing components needs the server" className="mb-4">
          This demo runs in your browser. The list below is what a server reports; installing happens on the server.
        </NeedsServer>
      )}
      {q.isLoading ? (
        <div className="grid gap-4" aria-busy="true">
          <Skeleton className="h-32 rounded-xl" />
          <Skeleton className="h-32 rounded-xl" />
        </div>
      ) : q.isError ? (
        <div className="rounded-xl border border-border bg-card">
          <ErrorState
            title="Could not load the components"
            error={describeError(q.error)}
            onRetry={() => void q.refetch()}
          />
        </div>
      ) : list.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border bg-card">
          <Empty icon={Puzzle} title="No optional components">
            This server does not report any optional components.
          </Empty>
        </div>
      ) : (
        <div className="grid gap-4">
          {kinds.map((k) => (
            <Panel
              key={k}
              title={KIND_LABEL[k] ?? k}
              description={KIND_NOTE[k]}
              bodyClassName="py-1.5"
              data-testid={`components-${k}`}
            >
              <ComponentsCard bare detailed kind={k} />
            </Panel>
          ))}
        </div>
      )}
    </Page>
  )
}
