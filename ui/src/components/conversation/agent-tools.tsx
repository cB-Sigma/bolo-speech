import { Plug, Wrench } from 'lucide-react'
import { Panel } from '@/components/panel'
import { OptionRow } from '@/components/stt/browser-processing'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import type { ServerInfo } from '@/lib/api/types'
import type { ConversationState, ToolCall } from '@/lib/realtime/session'

/** The MCP servers a conversation uses: configured ones by label, and one more by URL. */
export interface McpChoice {
  configured: string[]
  url: string
  label: string
  /** Ask before each call of the URL's server (`require_approval: "always"`); off: `"never"`. */
  approve: boolean
}

export const noMcp: McpChoice = { configured: [], url: '', label: '', approve: false }

/** `session.tools` entries of type `mcp` for a choice (an invalid URL is left out). */
export function mcpTools(choice: McpChoice): Record<string, unknown>[] {
  const tools: Record<string, unknown>[] = choice.configured.map((label) => ({ type: 'mcp', server_label: label }))
  const url = choice.url.trim()
  if (/^https?:\/\/\S+$/.test(url)) {
    tools.push({
      type: 'mcp',
      server_label: choice.label.trim() || 'tools',
      server_url: url,
      require_approval: choice.approve ? 'always' : 'never',
    })
  }
  return tools
}

const CALL_STATUS: Record<ToolCall['status'], string> = {
  approval: 'waiting for you',
  allowed: 'allowed',
  denied: 'declined',
  running: 'running',
  done: 'done',
  failed: 'failed',
}

/**
 * Tools for the language model from MCP servers (`{"type": "mcp"}` in `session.tools`): the server opens them when
 * the conversation starts, calls their tools when the model asks and hands the results back to the model.
 */
export function AgentToolsPanel({
  info,
  value,
  onChange,
  disabled,
  state,
  onApprove,
  className,
}: {
  info: ServerInfo | undefined
  value: McpChoice
  onChange: (c: McpChoice) => void
  disabled?: boolean
  state?: Pick<ConversationState, 'mcpServers' | 'toolCalls'>
  onApprove?: (id: string, approve: boolean) => void
  className?: string
}) {
  const mcp = info?.capabilities?.realtime?.mcp
  if (!mcp) return null
  const configured = mcp.configured ?? []
  const servers = Object.entries(state?.mcpServers ?? {})
  const calls = state?.toolCalls ?? []
  const toggle = (label: string, on: boolean) =>
    onChange({
      ...value,
      configured: on ? [...value.configured, label] : value.configured.filter((l) => l !== label),
    })
  return (
    <Panel
      title="Tools (MCP)"
      description="Model Context Protocol servers whose tools the language model may call"
      className={className}
      data-testid="agent-tools"
      bodyClassName="divide-y divide-border"
    >
      {!mcp.available ? (
        <p className="text-xs text-muted-foreground" data-testid="mcp-unavailable">
          The server needs the MCP SDK:{' '}
          <span className="font-mono">{mcp.install ?? "pip install 'bolo-speech[mcp]'"}</span>
        </p>
      ) : (
        <>
          <OptionRow
            testId="mcp-configured-option"
            icon={<Wrench />}
            title="Servers of this bolo"
            hint={
              configured.length ? (
                <>
                  Set up by the operator (<span className="font-mono">mcp_servers</span>); their approval rules apply.
                </>
              ) : (
                <>
                  None configured (<span className="font-mono">mcp_servers</span> in bolo.toml).
                </>
              )
            }
          >
            {configured.length > 0 && (
              <div className="grid gap-1.5 pl-8.5">
                {configured.map((label) => (
                  <label key={label} className="flex items-center gap-2 text-[13px]">
                    <Checkbox
                      checked={value.configured.includes(label)}
                      onCheckedChange={(c) => toggle(label, c === true)}
                      disabled={disabled}
                      aria-label={`Use ${label}`}
                      data-testid={`mcp-server-${label}`}
                    />
                    <span className="font-mono">{label}</span>
                  </label>
                ))}
              </div>
            )}
          </OptionRow>
          {mcp.session_urls !== false && (
            <OptionRow
              testId="mcp-url-option"
              icon={<Plug />}
              title="A server by URL"
              hint="Streamable HTTP (or SSE when the URL ends in /sse). The bolo server connects to it, not this browser."
            >
              <div className="grid gap-1.5 pl-8.5">
                <div className="flex flex-wrap gap-1.5">
                  <Input
                    value={value.url}
                    onChange={(e) => onChange({ ...value, url: e.target.value })}
                    placeholder="https://example.com/mcp"
                    aria-label="MCP server URL"
                    disabled={disabled}
                    className="h-8 min-w-0 flex-[3_1_12rem] text-xs"
                    data-testid="mcp-url"
                  />
                  <Input
                    value={value.label}
                    onChange={(e) => onChange({ ...value, label: e.target.value })}
                    placeholder="label"
                    aria-label="MCP server label"
                    disabled={disabled}
                    className="h-8 min-w-0 flex-[1_1_6rem] text-xs"
                    data-testid="mcp-label"
                  />
                </div>
                <label className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Checkbox
                    checked={value.approve}
                    onCheckedChange={(c) => onChange({ ...value, approve: c === true })}
                    disabled={disabled}
                    aria-label="Ask before each call"
                    data-testid="mcp-approve"
                  />
                  Ask me before each call
                </label>
              </div>
            </OptionRow>
          )}
        </>
      )}
      {(servers.length > 0 || calls.length > 0) && (
        <div className="grid gap-1.5 pt-2 text-xs" data-testid="mcp-activity">
          {servers.map(([label, s]) => (
            <p key={label} className="text-muted-foreground" data-testid={`mcp-status-${label}`}>
              <span className="font-mono text-foreground">{label}</span>:{' '}
              {s.status === 'listing'
                ? 'connecting…'
                : s.status === 'failed'
                  ? (s.error ?? 'could not connect')
                  : `${s.tools.length} tool${s.tools.length === 1 ? '' : 's'}${s.tools.length ? ` (${s.tools.join(', ')})` : ''}`}
            </p>
          ))}
          {calls.map((c) => (
            <div key={c.id} className="grid gap-1 rounded-md border border-border p-2" data-testid="mcp-call">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-mono">
                  {c.server}.{c.name}({c.arguments.length > 80 ? `${c.arguments.slice(0, 80)}…` : c.arguments})
                </span>
                <span className="text-muted-foreground">{CALL_STATUS[c.status]}</span>
              </div>
              {c.output && <p className="line-clamp-3 text-muted-foreground">{c.output}</p>}
              {c.status === 'approval' && onApprove && (
                <div className="flex gap-1.5">
                  <Button size="sm" onClick={() => onApprove(c.id, true)} data-testid="mcp-allow">
                    Allow
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => onApprove(c.id, false)} data-testid="mcp-deny">
                    Decline
                  </Button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </Panel>
  )
}
