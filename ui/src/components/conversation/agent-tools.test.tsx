import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import type { ServerInfo } from '@/lib/api/types'
import { emptyConversation } from '@/lib/realtime/session'
import { AgentToolsPanel, mcpTools, noMcp, type McpChoice } from './agent-tools'

const info = (mcp: Record<string, unknown>) => ({ capabilities: { realtime: { mcp } } }) as unknown as ServerInfo

describe('mcpTools', () => {
  it('names configured servers by label and a URL server with its approval rule', () => {
    expect(mcpTools(noMcp)).toEqual([])
    expect(mcpTools({ configured: ['files'], url: ' https://example.com/mcp ', label: '', approve: true })).toEqual([
      { type: 'mcp', server_label: 'files' },
      { type: 'mcp', server_label: 'tools', server_url: 'https://example.com/mcp', require_approval: 'always' },
    ])
    expect(mcpTools({ ...noMcp, url: 'not a url' })).toEqual([])
  })
})

function Harness({ onChange, onApprove }: { onChange: (c: McpChoice) => void; onApprove: () => void }) {
  const [value, setValue] = useState<McpChoice>(noMcp)
  const state = {
    ...emptyConversation(),
    mcpServers: { files: { status: 'ready' as const, tools: ['read'], error: null } },
    toolCalls: [
      { id: 'r1', server: 'files', name: 'read', arguments: '{}', status: 'approval' as const, output: null },
    ],
  }
  return (
    <AgentToolsPanel
      info={info({ available: true, configured: ['files', 'search'], session_urls: true })}
      value={value}
      onChange={(c) => {
        setValue(c)
        onChange(c)
      }}
      state={state}
      onApprove={onApprove}
    />
  )
}

describe('AgentToolsPanel', () => {
  it('picks configured servers, takes a URL, shows activity and answers approvals', async () => {
    const onChange = vi.fn()
    const onApprove = vi.fn()
    render(<Harness onChange={onChange} onApprove={onApprove} />)
    await userEvent.click(screen.getByTestId('mcp-server-search'))
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ configured: ['search'] }))
    await userEvent.type(screen.getByTestId('mcp-url'), 'https://h/mcp')
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ url: 'https://h/mcp' }))
    expect(screen.getByTestId('mcp-status-files')).toHaveTextContent('1 tool (read)')
    await userEvent.click(screen.getByTestId('mcp-allow'))
    expect(onApprove).toHaveBeenCalledWith('r1', true)
  })

  it('says how to install the SDK, and hides itself on servers that do not report MCP', () => {
    const { container, rerender } = render(<AgentToolsPanel info={undefined} value={noMcp} onChange={() => {}} />)
    expect(container).toBeEmptyDOMElement()
    rerender(
      <AgentToolsPanel
        info={info({ available: false, install: "pip install 'bolo-speech[mcp]'" })}
        value={noMcp}
        onChange={() => {}}
      />,
    )
    expect(screen.getByTestId('mcp-unavailable')).toHaveTextContent('bolo-speech[mcp]')
  })
})
