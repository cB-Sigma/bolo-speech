import { CatalogSelect } from '@/components/catalog-select'
import { Switch } from '@/components/ui/switch'
import type { CatalogModel } from '@/lib/api/types'
import { cn } from '@/lib/utils'

/**
 * The watermark a request asks for: null (none: the switch is off), '' (the server's default watermarker) or a
 * watermark model's id.
 */
export type WatermarkChoice = string | null

/** The request's `watermark` field: left out when off, `on` for the server's default, else the model. */
export function watermarkParam(choice: WatermarkChoice): string | undefined {
  if (choice === null) return undefined
  return choice === '' ? 'on' : choice
}

/**
 * An inaudible watermark on the speech: a switch (off by default) and, when it is on, which watermarker marks it.
 * `models` are the catalog's `watermark` models; new ones show up here by themselves.
 */
export function WatermarkControl({
  value,
  onChange,
  models,
  defaultModel,
  autoPull,
  disabled,
  testId = 'watermark',
  className,
}: {
  value: WatermarkChoice
  onChange: (v: WatermarkChoice) => void
  models: CatalogModel[]
  /** The server's `watermark_model` (named in the default item). */
  defaultModel?: string | null
  autoPull?: boolean | null
  disabled?: boolean
  testId?: string
  className?: string
}) {
  const on = value !== null
  return (
    <div className={cn('flex flex-wrap items-center gap-2', className)}>
      <label
        className="flex h-8 items-center gap-2 rounded-lg border border-input px-2.5 text-xs dark:bg-input/30"
        title="An inaudible mark that says the speech was made by a machine; bolo can detect it later"
      >
        Watermark
        <Switch
          checked={on}
          onCheckedChange={(v) => onChange(v ? '' : null)}
          size="sm"
          disabled={disabled}
          aria-label="Add an inaudible watermark"
          data-testid={testId}
        />
      </label>
      {on && (
        <CatalogSelect
          models={models}
          value={value}
          onChange={onChange}
          label="Watermark model"
          defaultLabel={`Default (${defaultModel || 'the server’s watermarker'})`}
          disabled={disabled}
          autoPull={autoPull}
          className="w-60"
          testId={`${testId}-model`}
        />
      )}
    </div>
  )
}
