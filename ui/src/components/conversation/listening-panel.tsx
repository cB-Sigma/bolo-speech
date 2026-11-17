import { Ear, Hourglass } from 'lucide-react'
import { Panel } from '@/components/panel'
import { AudioModelSelect, DeviceOption, OptionRow } from '@/components/stt/browser-processing'
import { useStore } from '@/hooks/use-store'
import { audioModelsFor, NOT_IN_THE_PAGE } from '@/lib/browser-audio/models'
import { fmtMs } from '@/lib/format'
import { prefs, setPref } from '@/lib/prefs'
import type { TurnCheck } from '@/lib/stt/model-vad'

const DECISION_TEXT: Record<TurnCheck['decision'], string> = {
  end: 'finished: the turn ends',
  wait: 'not finished: waiting',
  timeout: 'pause ran past 2.5 s: the turn ends',
}

/**
 * How the page hears the end of what was said (the server's `turn_detection` with `vad_model` and `turn_model`):
 * a voice-activity detector cuts the speech, and an end-of-turn model may be asked at each pause whether the
 * speaker has finished.
 */
export function ListeningPanel({ disabled, checks }: { disabled?: boolean; checks: TurnCheck[] }) {
  const p = useStore(prefs)
  const leftOut = NOT_IN_THE_PAGE.filter((x) => x.task === 'vad' || x.task === 'turn')
  return (
    <Panel
      title="Listening"
      description="Voice activity and end of turn, as a Realtime session chooses them"
      data-testid="listening"
      bodyClassName="divide-y divide-border"
    >
      <OptionRow
        testId="vad-option"
        icon={<Ear />}
        title="Voice activity"
        hint={
          <>
            What counts as speech (<span className="font-mono">turn_detection.vad_model</span>). On the server’s noisy
            test stream: Silero v5 and v6 0.93 frame accuracy, v4 0.89, TEN VAD 0.78, the energy detector 0.74.
          </>
        }
        control={
          <AudioModelSelect
            task="vad"
            label="Voice-activity detector"
            value={p.browserVad}
            onChange={(v) => setPref('browserVad', v ?? 'energy-vad')}
            offLabel={null}
            disabled={disabled}
          />
        }
      />
      <OptionRow
        testId="turn-option"
        icon={<Hourglass />}
        title="End of turn"
        hint={
          <>
            At each pause, Smart Turn hears the last 8 s and says whether the speaker has finished (
            <span className="font-mono">semantic_vad</span>, <span className="font-mono">turn_model</span>); an
            unfinished turn waits up to 2.5 s. On the server’s English set: 0.63 for v3.2 (8-bit), 0.80 for its 32-bit
            build.
          </>
        }
        control={
          <AudioModelSelect
            task="turn"
            label="End-of-turn model"
            value={p.browserTurn}
            onChange={(v) => setPref('browserTurn', v)}
            offLabel="Silence only"
            disabled={disabled}
          />
        }
      >
        {checks.length > 0 && (
          <ol className="grid gap-0.5 pl-8.5 text-xs" data-testid="turn-checks">
            {checks.map((c, i) => (
              <li key={i} className="text-muted-foreground">
                {Number.isFinite(c.probability) && (
                  <span className="num text-foreground">{c.probability.toFixed(2)} · </span>
                )}
                {DECISION_TEXT[c.decision]}
                {c.ms > 0 && <span className="num"> · {fmtMs(c.ms)}</span>}
              </li>
            ))}
          </ol>
        )}
      </OptionRow>
      <DeviceOption disabled={disabled} />
      {leftOut.length > 0 && (
        <details className="pt-3 text-xs leading-4 text-muted-foreground">
          <summary className="cursor-pointer select-none">
            Not in the page ({audioModelsFor('vad').length + audioModelsFor('turn').length} models are)
          </summary>
          <ul className="mt-1 grid list-disc gap-0.5 pl-4">
            {leftOut.map((x) => (
              <li key={x.label}>
                <span className="text-foreground">{x.label}</span>: {x.why}
              </li>
            ))}
          </ul>
        </details>
      )}
    </Panel>
  )
}
