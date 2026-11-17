import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { OtelStatus } from './connect'

describe('OpenTelemetry on the Connect screen', () => {
  it('says off, switched on but not running, or where the spans go', () => {
    const { rerender } = render(<OtelStatus otel={undefined} />)
    expect(screen.getByText('not reported by this server')).toBeInTheDocument()
    rerender(<OtelStatus otel={{ switched_on: false, enabled: false }} />)
    expect(screen.getByText('Off')).toBeInTheDocument()
    rerender(<OtelStatus otel={{ switched_on: true, enabled: false, install: 'pip install "bolo-speech[otel]"' }} />)
    expect(screen.getByText(/Switched on, not running/)).toHaveTextContent('bolo-speech[otel]')
    rerender(
      <OtelStatus
        otel={{
          switched_on: true,
          enabled: true,
          endpoint: 'http://collector:4318',
          protocol: 'http/protobuf',
          service_name: 'bolo-eu1',
          exported: 7,
        }}
      />,
    )
    expect(screen.getByText(/On · collector:4318/)).toHaveTextContent('http/protobuf · bolo-eu1 · 7 spans')
  })
})
